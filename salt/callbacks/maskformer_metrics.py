"""`MaskformerMetrics` — per-epoch MaskFormer object metrics from the matched predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from salt.core.graph.bundle import Bundle


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
        """Compute ``{metric name: scalar tensor}`` from the matched object bundle keys."""
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
