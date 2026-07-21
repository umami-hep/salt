"""`MaskformerConfusionMatrix` — per-epoch MaskFormer object-class confusion matrix."""

from __future__ import annotations

from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from salt.callbacks.confusion_matrix import ConfusionMatrix
from salt.graph.bundle import Bundle


class MaskformerConfusionMatrix(Callback):
    """Log a per-epoch validation confusion matrix for the MaskFormer object classes.

    Consumes the matcher-permuted ``matched.objects.*`` keys
    `MaskFormerMatchedLoss` publishes (query-i is already aligned to
    truth-object-i, so the per-object class confusion is well-defined).
    Accumulates argmax predictions vs truth labels over the validation
    epoch and logs the matrix to Comet at epoch end via
    ``log_confusion_matrix`` (no seaborn/scikit-learn dependency).

    `fit_val_demand` declares the consumed ``matched.<input_stream>.{class_logits,
    object_class}`` keys as FIT/VAL plan sinks so the matched-loss products
    survive demand pruning even though no loss anchors them. Pruned from
    TEST/ONNX (the matched loss is FIT|VAL-only). The accumulated lists and
    computed counts matrix are stashed on the callback
    (``last_truth_labels`` / ``last_pred_labels`` / ``last_matrix``) for
    logger-free inspection.

    Parameters
    ----------
    log_every_n_epochs : int, optional
        Log only every N validation epochs, by default 1.
    class_names : list[str] | None, optional
        Display names for the object classes (null last), by default None
        (uses integer indices ``range(num_classes)``; the class count is
        derived from the matched class-logits last dim).
    input_stream : str, optional
        The matched object stream name, by default ``objects``.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        class_names: list[str] | None = None,
        input_stream: str = "objects",
    ) -> None:
        self.log_every_n_epochs = int(log_every_n_epochs)
        self.class_names = list(class_names) if class_names is not None else None
        self.input_stream = str(input_stream)
        self.requires: tuple[str, str] = (
            f"matched.{self.input_stream}.class_logits",
            f"matched.{self.input_stream}.object_class",
        )
        # per-epoch accumulators
        self.truth_labels: list[Tensor] = []
        self.pred_labels: list[Tensor] = []
        self._num_classes: int = 0
        # stashed at epoch end for logger-free comparison
        self.last_truth_labels: list[Tensor] = []
        self.last_pred_labels: list[Tensor] = []
        self.last_matrix: Tensor | None = None

    def fit_val_demand(self, model_modules: Any) -> tuple[str, ...]:
        """The ``matched.objects.*`` keys this callback reads each VAL epoch.

        Static/config-only — does not depend on `setup` having run.

        Returns
        -------
        tuple[str, ...]
            The ``matched.<input_stream>.{class_logits,object_class}`` keys.
        """
        del model_modules
        return self.requires

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Reset the per-epoch accumulators (fit only)."""
        del trainer, pl_module
        if stage != "fit":
            return
        self.truth_labels = []
        self.pred_labels = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate argmax matched-object predictions and truth classes."""
        del pl_module, batch, batch_idx, dataloader_idx
        if trainer.fast_dev_run:
            return
        bundle: Bundle = outputs["bundle"]
        logits_key, label_key = self.requires
        logits = bundle.get(logits_key).detach()
        self._num_classes = int(logits.shape[-1])  # null LAST (MaskFormerTargets contract)
        self.truth_labels.append(bundle.get(label_key).detach().cpu().reshape(-1))
        self.pred_labels.append(logits.argmax(-1).cpu().reshape(-1))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Every N epochs: stash the values, log the matrix to Comet, reset."""
        del pl_module
        if not self.truth_labels:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self.truth_labels = []
            self.pred_labels = []
            return
        n_classes = len(self.class_names) if self.class_names is not None else self._num_classes
        labels = self.class_names or [str(i) for i in range(n_classes)]
        self.last_truth_labels = self.truth_labels
        self.last_pred_labels = self.pred_labels
        self.last_matrix, _ = ConfusionMatrix.confusion_counts(
            self.truth_labels, self.pred_labels, n_classes
        )
        if isinstance(trainer.logger, CometLogger):
            trainer.logger.experiment.log_confusion_matrix(
                y_true=torch.cat(self.truth_labels).tolist(),
                y_predicted=torch.cat(self.pred_labels).tolist(),
                labels=labels,
                epoch=trainer.current_epoch,
            )
        self.truth_labels = []
        self.pred_labels = []
