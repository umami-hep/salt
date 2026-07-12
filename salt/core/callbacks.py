"""Lightning callbacks: bundle-consuming metrics and run-dir artifacts.

Includes checkpointing, a progress bar, confusion-matrix / MaskFormer metric
callbacks, and graph/plan artifact writers.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers.comet import CometLogger

from salt.core.graph.errors import ConfigError, GraphError
from salt.core.graph.spec import Mode, TensorSpec
from salt.core.render import dot_source, plan_table

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lightning.pytorch.utilities.types import STEP_OUTPUT
    from torch import Tensor

    from salt.core.data.dataset import GraphDataset
    from salt.core.graph.bundle import Bundle
    from salt.core.graph.planner import Plan

__all__ = [
    "Checkpoint",
    "ConfusionMatrix",
    "GraphArtifacts",
    "MaskformerConfusionMatrix",
    "MaskformerMetrics",
    "ProgressBar",
]


class Checkpoint(ModelCheckpoint):
    """Save a checkpoint per epoch under ``ckpts/`` with the ``loss=`` filename stem.

    Filename and directory are a contract: ``salt2 test`` without
    ``--ckpt_path`` resolves the best epoch by globbing
    ``<config dir>/{ckpts,checkpoints}/*.ckpt`` and parsing the smallest
    ``loss=<value>`` out of each name (`salt.core.main._best_checkpoint`).
    Keep the filename stem and the glob in sync.

    Parameters
    ----------
    monitor_loss : str
        The metric key to monitor and embed in the filename, e.g.
        ``val/jets_classification_loss`` or ``val/loss``. Must exist in
        ``trainer.callback_metrics`` at checkpoint time.
    fname_string : str, optional
        The filename loss tag, by default ``"loss"`` — the ``loss=`` stem the
        best-epoch glob keys on.
    mode : str, optional
        ``min``/``max`` selection direction passed to `ModelCheckpoint`, by
        default ``"min"``.
    save_top_k : int, optional
        How many checkpoints to keep, by default ``-1`` (every epoch).
    dirname : str, optional
        The log-dir sub-directory checkpoints land in, by default ``"ckpts"``.
    """

    def __init__(
        self,
        monitor_loss: str = "val/loss",
        fname_string: str = "loss",
        mode: str = "min",
        save_top_k: int = -1,
        dirname: str = "ckpts",
    ) -> None:
        filename = "epoch={epoch:03d}-" + fname_string + "={" + monitor_loss + ":.5f}"
        super().__init__(
            monitor=monitor_loss,
            mode=mode,
            save_top_k=save_top_k,
            filename=filename,
            auto_insert_metric_name=False,
        )
        self.dirname = dirname

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Fix the checkpoint dir to ``<log_dir>/<dirname>`` on a real fit.

        Raises
        ------
        ConfigError
            When the trainer log dir is an ``s3://`` path (not yet supported).
        """
        if stage == "fit" and not trainer.fast_dev_run:
            log_dir = trainer.log_dir or trainer.default_root_dir
            if log_dir is not None and str(log_dir).startswith(("s3://", "s3:/")):
                raise ConfigError(
                    "salt.core.callbacks.Checkpoint does not support s3:// log dirs yet "
                    "(rides with the M6 Comet/run-dir wiring); use a local trainer.log_dir "
                    "(v1 checkpoint.py:34-43 s3 branch deferred)"
                )
            self.dirpath = str(Path(log_dir) / self.dirname)
        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)


class ProgressBar(TQDMProgressBar):
    """Progress bar callback — a thin, salt-owned subclass of `TQDMProgressBar`."""


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


class MaskformerMetrics(Callback):
    """Log per-epoch MaskFormer object metrics from the matched predictions.

    Consumes the step bundle's ``matched.objects.*`` keys — the
    matcher-permuted predictions + truth labels `MaskFormerMatchedLoss`
    publishes. Query-i is already aligned to truth-object-i, so the
    class/mask/regression metrics read straight from
    ``matched.objects.{class_logits,object_class,masks,target_masks,
    regression,target_regression}`` with no re-matching in the callback.

    Metrics: class exact-match + micro/macro accuracy, per-class + not-null
    efficiency/purity, mask reco efficiency/fake-rate per criterion, and
    per-target regression MAE (computed in the matched loss's own scaled
    space to stay task-decoupled). Logged to the attached logger and
    stashed on the callback (``last_metrics``) for logger-free inspection.

    `fit_val_demand` declares the consumed ``matched.objects.*`` keys as
    FIT/VAL plan sinks so the matched-loss products survive demand pruning
    even though no loss anchors them. Pruned from TEST/ONNX (the matched
    loss is FIT|VAL-only).

    Parameters
    ----------
    only_val : bool, optional
        Log only on validation batches, by default True.
    mask_criteria : Mapping[str, tuple[float, float]] | None, optional
        ``{name: (min_recall, min_purity)}`` mask-match criteria, by default
        None (uses ``perfect (1, 1)`` / ``loose (0.5, 0.5)``).
    input_stream : str, optional
        The matched object stream name, by default ``objects``.
    constituent_stream : str, optional
        The constituent stream whose pad mask suppresses padded tokens in
        the mask metrics, by default ``tracks``.
    """

    def __init__(
        self,
        only_val: bool = True,
        mask_criteria: Mapping[str, tuple[float, float]] | None = None,
        input_stream: str = "objects",
        constituent_stream: str = "tracks",
    ) -> None:
        self.only_val = only_val
        self.mask_criteria: dict[str, tuple[float, float]] = (
            dict(mask_criteria) if mask_criteria else {"perfect": (1.0, 1.0), "loose": (0.5, 0.5)}
        )
        self.input_stream = str(input_stream)
        self.constituent_stream = str(constituent_stream)
        # stashed at epoch end for logger-free comparison
        self.last_metrics: dict[str, float] = {}

    def _matched_key(self, leaf: str) -> str:
        """A ``matched.<input_stream>.<leaf>`` bundle key."""
        return f"matched.{self.input_stream}.{leaf}"

    def fit_val_demand(self, model_modules: Any) -> tuple[str, ...]:
        """The ``matched.objects.*`` keys this callback reads each VAL epoch.

        Static/config-only — does not depend on `setup` having run.

        Returns
        -------
        tuple[str, ...]
            The ``matched.<input_stream>.{class_logits,object_class,masks,
            target_masks}`` keys.
        """
        del model_modules
        return (
            self._matched_key("class_logits"),
            self._matched_key("object_class"),
            self._matched_key("masks"),
            self._matched_key("target_masks"),
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
        """Compute + log the object metrics from the step bundle's matched keys."""
        del batch, batch_idx, dataloader_idx
        if trainer.fast_dev_run:
            return
        metrics = self._compute(outputs["bundle"])
        self.last_metrics = {k: float(v) for k, v in metrics.items()}
        self._log(trainer, pl_module, metrics, "val")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Compute + log the object metrics on train batches when ``only_val`` is off."""
        del batch, batch_idx
        if self.only_val or trainer.fast_dev_run:
            return
        metrics = self._compute(outputs["bundle"])
        self._log(trainer, pl_module, metrics, "train")

    def _compute(self, bundle: Bundle) -> dict[str, Tensor]:
        """Compute the metric set from the matched object bundle keys.

        Returns
        -------
        dict[str, Tensor]
            ``{metric name: scalar tensor}``.
        """
        from salt.core.utils.mask_utils import mask_from_logits, reco_metrics  # noqa: PLC0415

        class_logits = bundle.get(self._matched_key("class_logits")).detach()
        object_class = bundle.get(self._matched_key("object_class")).detach()
        pred_masks = bundle.get(self._matched_key("masks")).detach()
        tgt_masks = bundle.get(self._matched_key("target_masks")).detach()

        n_classes = class_logits.shape[-1]
        null_index = n_classes - 1  # null is the LAST class (MaskFormerTargets contract)
        obj_class_pred = class_logits.argmax(-1)

        metrics: dict[str, Tensor] = {}
        pred_flat, tgt_flat = obj_class_pred.reshape(-1), object_class.reshape(-1)
        # class exact match per row (all M objects correct), micro/macro accuracy
        metrics["class_exact_match"] = (obj_class_pred == object_class).all(-1).float().mean()
        metrics["class_accuracy_micro"] = (pred_flat == tgt_flat).float().mean()
        per_class_acc = torch.stack([
            (pred_flat[tgt_flat == c] == c).float().mean()
            for c in range(n_classes)
            if (tgt_flat == c).any()
        ])
        metrics["class_accuracy_macro"] = per_class_acc.mean()

        # per-class + not-null efficiency (recall) / purity (precision)
        present_tgt = tgt_flat != null_index
        present_pred = pred_flat != null_index
        metrics["notnull_eff"] = _recall(present_pred, present_tgt)
        metrics["notnull_pur"] = _precision(present_pred, present_tgt)
        for c in range(null_index):  # every non-null class
            is_tgt, is_pred = tgt_flat == c, pred_flat == c
            metrics[f"class{c}_eff"] = _recall(is_pred, is_tgt)
            metrics[f"class{c}_pur"] = _precision(is_pred, is_tgt)

        # mask reco metrics: predicted masks suppressed on padded tokens + null objects
        pad_key = f"masks.{self.constituent_stream}"
        pad_mask = bundle.get(pad_key).detach() if pad_key in bundle else None
        recon = mask_from_logits(pred_masks, "sigmoid", pad_mask, obj_class_pred)
        for name, (recall, purity) in self.mask_criteria.items():
            eff, fake = reco_metrics(
                recon, tgt_masks, min_recall=recall, min_purity=purity, reduce=True
            )
            metrics[f"query_{name}_match_eff"] = eff
            metrics[f"query_{name}_match_fake"] = fake

        # per-target regression MAE, computed in the matched loss's own scaled space
        reg_key = self._matched_key("regression")
        if reg_key in bundle:
            reg_pred = bundle.get(reg_key).detach()
            reg_tgt = bundle.get(self._matched_key("target_regression")).detach()
            valid = object_class != null_index  # [B, M]
            if valid.any():
                metrics["query_regression_mae"] = torch.nn.functional.l1_loss(
                    reg_pred[valid], reg_tgt[valid]
                )
        return metrics

    def _log(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        metrics: Mapping[str, Tensor],
        stage: str,
    ) -> None:
        """Log each metric through the LightningModule (sync across devices)."""
        del trainer
        for name, value in metrics.items():
            pl_module.log(f"{stage}/{name}", value)


def _recall(pred: Tensor, tgt: Tensor) -> Tensor:
    """Binary recall TP / (TP + FN); 0.0 when there are no positives."""
    tp = (pred & tgt).sum().float()
    denom = tgt.sum().float()
    return tp / denom if denom > 0 else torch.zeros((), device=pred.device)


def _precision(pred: Tensor, tgt: Tensor) -> Tensor:
    """Binary precision TP / (TP + FP); 0.0 when there are no predictions."""
    tp = (pred & tgt).sum().float()
    denom = pred.sum().float()
    return tp / denom if denom > 0 else torch.zeros((), device=pred.device)


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


class GraphArtifacts(Callback):
    """Write run-dir artifacts (plan tables, resolved I/O, graph images) at fit/test start.

    Per stage start (after `SaltModule.setup` compiled the plans), writes at
    rank zero:

    - ``plan_<mode>.txt`` for every stage mode (``fit``+``val`` / ``test``):
      the ordered step table for both the dataset plan and the model plan.
      The TEST table appends the writer-sinks section (writer instance ->
      consumed ``preds.*`` keys).
    - ``resolved_io.yaml`` — machine-readable: per mode, the plan sources
      and every module's flattened requires/produces with resolved specs.
    - ``graph_<stage>.dot`` + ``graph_<stage>.<image_format>`` (and
      ``graph_<stage>_dataset.*`` when a `GraphDataModule` is attached):
      Graphviz DOT rasterised via the ``dot`` binary, kept alongside for
      manual re-rendering.

    Never fails a run: a LightningModule without compiled plans, or a
    missing/failing ``dot`` binary, degrade to a warning / DOT-only output.

    Parameters
    ----------
    output_dir : str | None, optional
        Artifact directory, by default None — the trainer log dir on the
        fit path, and the checkpoint's directory on the test path (so eval
        outputs land next to the checkpoint and the eval H5).
    image_format : str, optional
        Graph image suffix (``svg``/``png``/``pdf``), by default ``svg``.
    """

    def __init__(self, output_dir: str | None = None, image_format: str = "svg") -> None:
        self.output_dir = output_dir
        self.image_format = image_format.lstrip(".")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write fit+val plan tables and the FIT graph before training starts."""
        self._write(trainer, pl_module, stage="fit", mode_names=("fit", "val"))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write the test plan table and TEST graph before evaluation starts."""
        self._write(trainer, pl_module, stage="test", mode_names=("test",))

    # -- internals -----------------------------------------------------------

    def _write(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
        mode_names: tuple[str, ...],
    ) -> None:
        """Write all artifacts for one stage (rank zero, never raising)."""
        if not trainer.is_global_zero:
            return
        plans: Mapping[Mode, Plan] = getattr(pl_module, "plans", None) or {}
        if not plans:
            warnings.warn(
                f"GraphArtifacts: {type(pl_module).__name__} has no compiled plans — "
                "run-dir artifacts skipped (design §4.4 expects a SaltModule)",
                stacklevel=2,
            )
            return
        out_dir = Path(self.output_dir) if self.output_dir else self._default_dir(trainer, stage)
        out_dir.mkdir(parents=True, exist_ok=True)
        writer_sinks = self._writer_sinks(trainer, pl_module) if stage == "test" else None
        for mode_name in mode_names:
            plan = plans.get(Mode[mode_name.upper()])
            if plan is None:
                continue
            dataset = self._stage_dataset(trainer, mode_name)
            (out_dir / f"plan_{mode_name}.txt").write_text(
                self._plan_text(plan, dataset, writer_sinks if mode_name == "test" else None)
            )
        (out_dir / "resolved_io.yaml").write_text(self._resolved_io(trainer, plans, mode_names))
        # one graph per stage: FIT (VAL is identical) / TEST
        plan = plans[Mode[stage.upper()]]
        modules = getattr(pl_module, "_graph_modules", None) or {}
        pruned = sorted(set(modules) - set(plan.module_names))
        self._render(plan, modules, pruned, out_dir / f"graph_{stage}")
        dataset = self._stage_dataset(trainer, stage)
        if dataset is not None:
            self._render(
                dataset.plan,
                dataset.modules,
                sorted(set(dataset.modules) - set(dataset.plan.module_names)),
                out_dir / f"graph_{stage}_dataset",
            )
        print(f"wrote graph/plan artifacts to {out_dir} (design §4.4)")

    @staticmethod
    def _default_dir(trainer: Trainer, stage: str) -> Path:
        """The stage's default artifact directory.

        Returns
        -------
        Path
            Test: the checkpoint's directory (where the eval H5 goes) when
            ``trainer.ckpt_path`` is known; otherwise — and always on the
            fit path — the trainer log dir.
        """
        ckpt_path = getattr(trainer, "ckpt_path", None)
        if stage == "test" and ckpt_path:
            return Path(ckpt_path).parent
        return Path(trainer.log_dir or trainer.default_root_dir)

    @staticmethod
    def _writer_sinks(trainer: Trainer, pl_module: LightningModule) -> list[str] | None:
        """Per-writer consumed-keys lines for the TEST plan table.

        Returns
        -------
        list[str] | None
            ``"name (Class): key, key"`` lines from the attached writer
            callback's `per_writer_demand`, or None when no writer callback
            / reader / graph-module dict is attached.
        """
        callbacks = getattr(trainer, "callbacks", None) or []
        cb = next((c for c in callbacks if callable(getattr(c, "per_writer_demand", None))), None)
        reader = getattr(getattr(trainer, "datamodule", None), "reader", None)
        modules = getattr(pl_module, "_graph_modules", None)
        if cb is None or reader is None or not modules:
            return None
        try:
            per_writer = cb.per_writer_demand(modules, reader)
        except GraphError:
            return None  # a broken writers block fails the run elsewhere
        return [
            f"  {name} ({type(cb.writers[name]).__name__}): " + ", ".join(keys)
            for name, keys in per_writer.items()
        ]

    @staticmethod
    def _plan_text(
        plan: Plan, dataset: GraphDataset | None, writer_sinks: list[str] | None = None
    ) -> str:
        """Build one ``plan_<mode>.txt`` payload: dataset plan + model plan.

        Returns
        -------
        str
            The artifact text; `writer_sinks` lines (TEST only) append as
            the writer-sinks section.
        """
        sections: list[str] = []
        if dataset is not None:
            sections.append(f"# dataset plan (design §6.1)\n{plan_table(dataset.plan)}")
            columns = [
                f"  {stream}: " + ", ".join(f"{field} ({who})" for field, who in fields.items())
                for stream, fields in dataset.read_fields.items()
            ]
            if columns:
                sections.append(
                    "read columns (demand-narrowed, design §6.1):\n" + "\n".join(columns)
                )
        sections.append(f"# model plan (design §3.1)\n{plan_table(plan)}")
        if writer_sinks:
            sections.append("# writer sinks (design §8)\n" + "\n".join(writer_sinks))
        return "\n\n".join(sections) + "\n"

    def _resolved_io(
        self, trainer: Trainer, plans: Mapping[Mode, Plan], mode_names: tuple[str, ...]
    ) -> str:
        """Build the ``resolved_io.yaml`` payload.

        Returns
        -------
        str
            YAML: per mode, the plan ``sources`` and each module's flattened
            ``requires``/``produces`` with resolved specs — dataset-plan
            modules included when a `GraphDataModule` stage dataset exists.
        """
        payload: dict[str, Any] = {}
        for mode_name in mode_names:
            plan = plans.get(Mode[mode_name.upper()])
            if plan is None:
                continue
            model_io = _plan_io(plan)
            section: dict[str, Any] = {
                "plan_hash": plan.plan_hash,
                "sources": model_io["sources"],
                "modules": model_io["modules"],
            }
            dataset = self._stage_dataset(trainer, mode_name)
            if dataset is not None:
                section["dataset_modules"] = _plan_io(dataset.plan)["modules"]
            payload[mode_name] = section
        return yaml.safe_dump(payload, sort_keys=False)

    def _render(
        self,
        plan: Plan,
        modules: Mapping[str, Any],
        pruned: list[str],
        base: Path,
    ) -> None:
        """Write DOT + image for one plan via the `dot` binary; degrade to DOT-only.

        Best-effort: a missing/failing ``dot`` degrades to a DOT-only hint
        and never crashes a training run.
        """
        dot_path = base.with_suffix(".dot")
        dot_path.write_text(dot_source(plan, modules, pruned))
        img_path = base.with_suffix(f".{self.image_format}")
        hint = f"render manually with: dot -T{self.image_format} {dot_path} -o {img_path}"
        dot_bin = shutil.which("dot")
        if dot_bin is None:
            print(f"graphviz `dot` not on PATH — wrote {dot_path} only; {hint}")
            return
        result = subprocess.run(
            [dot_bin, f"-T{self.image_format}", str(dot_path), "-o", str(img_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"`dot` failed to render {img_path} (exit {result.returncode}): "
                f"{result.stderr.strip() or '(no stderr)'} — wrote {dot_path} only; {hint}"
            )

    @staticmethod
    def _stage_dataset(trainer: Trainer, mode_name: str) -> GraphDataset | None:
        """The stage's `GraphDataset` from the attached datamodule, if any.

        Returns
        -------
        GraphDataset | None
            The dataset whose plan matches `mode_name`, or None for
            non-Graph datamodules.
        """
        dm = getattr(trainer, "datamodule", None)
        attr = {"fit": "train_dset", "val": "val_dset", "test": "test_dset"}[mode_name]
        dataset = getattr(dm, attr, None)
        return dataset if hasattr(dataset, "plan") and hasattr(dataset, "read_fields") else None


def _spec_dict(spec: TensorSpec) -> dict[str, Any]:
    """A `TensorSpec` as plain YAML-able data (for ``resolved_io.yaml``).

    Returns
    -------
    dict[str, Any]
        ``shape``/``dtype``/``kind`` always; ``fields`` when declared.
    """
    out: dict[str, Any] = {
        "shape": list(spec.shape) if spec.shape is not None else None,
        "dtype": spec.dtype,
        "kind": spec.kind,
    }
    if spec.fields is not None:
        out["fields"] = list(spec.fields)
    return out


def _plan_io(plan: Plan) -> dict[str, Any]:
    """One plan's sources + per-module resolved requires/produces.

    Returns
    -------
    dict[str, Any]
        ``{"sources": {key: spec}, "modules": {name: {"class", "requires",
        "produces"}}}`` in plan (topological) order.
    """
    return {
        "sources": {key: _spec_dict(spec) for key, spec in plan.sources.items()},
        "modules": {
            step.name: {
                "class": type(step.module).__name__,
                "requires": {key: _spec_dict(spec) for key, spec in step.requires.items()},
                "produces": {key: _spec_dict(spec) for key, spec in step.produces.items()},
            }
            for step in plan.steps
        },
    }
