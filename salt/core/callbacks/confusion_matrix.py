"""`ConfusionMatrix` — per-epoch validation confusion matrix for one classification task."""

from __future__ import annotations

from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError


class ConfusionMatrix(Callback):
    """Log a per-epoch validation confusion matrix for one classification task.

    Predictions are read from ``preds.<stream>.<task_name>`` (argmax) and
    truth from ``labels.<stream>.<label>``. Stream / label / class names are
    resolved from the named task module by duck-typing (the
    ``ClassificationTaskModule`` surface: ``stream``/``label``/
    ``class_names``), so user task modules participate too.

    At each validation epoch end the matrix is logged to Comet when a
    `CometLogger` is attached, and the accumulated lists and computed counts
    matrix are stashed on the callback (``last_truth_labels`` /
    ``last_pred_labels`` / ``last_matrix`` / ``last_ignored``) for
    logger-free inspection.

    Parameters
    ----------
    task_name : str
        Instance name of the classification task module (the
        ``model.modules`` dict key, e.g. ``jets_classification``).
    class_names_override : list[str] | dict[str, str] | None, optional
        Class names for logging: a full replacement list, or a mapping from
        existing to new names, by default None (uses the task's
        ``class_names``).
    """

    def __init__(
        self, task_name: str, class_names_override: list[str] | dict[str, str] | None = None
    ) -> None:
        self.task_name = task_name
        self.class_names_override = class_names_override
        # resolved at setup
        self.task_stream: str | None = None
        self.task_label_name: str | None = None
        self.task_class_names: list[str] = []
        self.requires: tuple[str, ...] = ()
        # per-epoch accumulators
        self.truth_labels: list[Tensor] = []
        self.pred_labels: list[Tensor] = []
        # stashed at epoch end for logger-free comparison
        self.last_truth_labels: list[Tensor] = []
        self.last_pred_labels: list[Tensor] = []
        self.last_matrix: Tensor | None = None
        self.last_ignored: int = 0

    def _resolve_task(self, modules: Any) -> tuple[str, str, list[str]]:
        """Resolve ``(stream, label, class_names)`` from the named task module.

        Returns
        -------
        tuple[str, str, list[str]]
            The task's stream, label name, and class names.

        Raises
        ------
        ConfigError
            When `modules` is not a graph-module dict, or `task_name` does
            not resolve to a classification-task module (candidates listed).
        """
        if not isinstance(modules, dict):
            raise ConfigError(
                f"ConfusionMatrix needs a SaltModule-style LightningModule with a graph-module "
                f"dict, got {type(modules).__name__} (design §3.4)"
            )
        module = modules.get(self.task_name)
        stream = getattr(module, "stream", None)
        label = getattr(module, "label", None)
        class_names = getattr(module, "class_names", None)
        if module is None or stream is None or label is None or class_names is None:
            surface = ("stream", "label", "class_names")
            candidates = sorted(
                name
                for name, mod in modules.items()
                if all(getattr(mod, attr, None) is not None for attr in surface)
            )
            raise ConfigError(
                f"ConfusionMatrix: task_name {self.task_name!r} does not name a classification "
                f"task module (needs stream/label/class_names — design §3.3). "
                f"Configured candidates: {candidates or '<none>'}"
            )
        return stream, label, list(class_names)

    def fit_val_demand(self, model_modules: Any) -> tuple[str, ...]:
        """The bundle keys this callback reads each VAL epoch.

        Does not depend on `setup` having run — re-resolves from
        `model_modules` directly.

        Returns
        -------
        tuple[str, ...]
            ``(preds.<stream>.<task>, labels.<stream>.<label>)``. Propagates
            the `_resolve_task` `ConfigError` when `task_name` does not name
            a classification task.
        """
        stream, label, _ = self._resolve_task(model_modules)
        return (f"preds.{stream}.{self.task_name}", f"labels.{stream}.{label}")

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve stream/label/class names from the named task module (fit only)."""
        del trainer
        if stage != "fit":
            return
        self.truth_labels = []
        self.pred_labels = []
        modules = getattr(pl_module, "_graph_modules", None)
        stream, label, class_names = self._resolve_task(modules)
        self.task_stream = stream
        self.task_label_name = label
        if isinstance(self.class_names_override, dict):
            self.task_class_names = [
                self.class_names_override.get(name, name) for name in class_names
            ]
        else:
            self.task_class_names = list(self.class_names_override or class_names)
        self.requires = (
            f"preds.{stream}.{self.task_name}",
            f"labels.{stream}.{label}",
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate argmax predictions and truth labels from the step bundle."""
        del trainer, pl_module, batch, batch_idx, dataloader_idx
        bundle: Bundle = outputs["bundle"]
        pred_key, label_key = self.requires
        pred_labels_batch = torch.argmax(bundle.get(pred_key), dim=-1)
        truth_labels_batch = bundle.get(label_key)
        # extend iterates dim 0: scalars for [B], rows for [B, T]
        self.truth_labels.extend(truth_labels_batch)
        self.pred_labels.extend(pred_labels_batch)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Stash the epoch's values, log to Comet when attached, reset."""
        del pl_module
        self.last_truth_labels = self.truth_labels
        self.last_pred_labels = self.pred_labels
        self.last_matrix, self.last_ignored = self.confusion_counts(
            self.truth_labels, self.pred_labels, len(self.task_class_names)
        )
        if isinstance(trainer.logger, CometLogger):
            trainer.logger.experiment.log_confusion_matrix(
                y_true=self.truth_labels,
                y_predicted=self.pred_labels,
                labels=self.task_class_names,
                epoch=trainer.current_epoch,
            )
        self.truth_labels = []
        self.pred_labels = []

    @staticmethod
    def confusion_counts(
        truth: list[Tensor] | Tensor,
        preds: list[Tensor] | Tensor,
        num_classes: int,
    ) -> tuple[Tensor, int]:
        """Compute an integer confusion-counts matrix from accumulated labels.

        Rows are truth classes, columns predicted classes. Entries whose
        truth value lies outside ``[0, num_classes)`` — e.g. the ``-1``
        padding of per-track labels — are dropped and counted in the second
        return value.

        Parameters
        ----------
        truth : list[Tensor] | Tensor
            Accumulated truth labels (scalars or per-token rows).
        preds : list[Tensor] | Tensor
            Accumulated argmax predictions, same layout as `truth`.
        num_classes : int
            The number of classes ``C``.

        Returns
        -------
        tuple[Tensor, int]
            The ``[C, C]`` int64 counts matrix and the number of dropped
            (out-of-range truth) entries.

        Raises
        ------
        ValueError
            When the flattened truth/pred shapes differ.
        """
        if len(truth) == 0:
            return torch.zeros(num_classes, num_classes, dtype=torch.int64), 0
        t = torch.cat([torch.as_tensor(x).flatten() for x in truth]).long()
        p = torch.cat([torch.as_tensor(x).flatten() for x in preds]).long()
        if t.shape != p.shape:
            raise ValueError(f"truth/pred shapes differ after flattening: {t.shape} vs {p.shape}")
        valid = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
        matrix = torch.bincount(
            t[valid] * num_classes + p[valid], minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)
        return matrix, int((~valid).sum())
