"""`SkipBatchesWithoutLabels` — skip training steps with no labelled object for a task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from lightning import Callback, LightningModule, Trainer

from salt.graph.errors import ConfigError
from salt.utils.logging import get_logger

_LOG = get_logger(__name__)


def _lookup(batch: Any, key: str) -> Any:
    """``labels.jets.x`` from a flat (dotted keys) or nested batch mapping, else None."""
    if isinstance(batch, Mapping) and key in batch:
        return batch[key]
    node = batch
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


class SkipBatchesWithoutLabels(Callback):
    """Skip a training step when the batch has no labelled object for one of `labels`.

    A label of ``-1`` (classification) or ``NaN`` (regression target) marks an object a
    task does not learn from (e.g. the other jet type's rows in a joint small-R +
    large-R training). If every entry of a listed label is ``-1`` or ``NaN``, that task
    has nothing to learn from in this batch, and under `LossGLS` its loss of 0 collapses
    the geometric mean and gives invalid gradients. Such a step
    is skipped: the batch still runs forward and backward, but every gradient is dropped
    before the optimizer step, so no weight changes. Each skip is logged as a warning,
    and the number of skipped steps per epoch is logged as ``train/skipped_batches``.

    With several processes (DDP) the decision is synchronised, so all ranks skip the
    same step. With gradient accumulation, an empty batch anywhere in the accumulation
    window skips that window's optimizer step. Validation is not affected.

    Parameters
    ----------
    labels : Sequence[str]
        Batch keys of the labels to check, e.g.
        ``["labels.jets.small_r_flavour", "labels.jets.large_r_flavour"]``.

    Raises
    ------
    ConfigError
        If `labels` is empty.
    """

    def __init__(self, labels: Sequence[str]) -> None:
        super().__init__()
        if isinstance(labels, str) or not labels:
            raise ConfigError("SkipBatchesWithoutLabels needs a non-empty list of label keys")
        self.labels = tuple(labels)
        self.skip_pending = False
        self.skipped_this_epoch = 0
        self.skipped_total = 0

    def empty_labels(self, batch: Any) -> list[str]:
        """The listed labels with no entry other than ``-1`` or ``NaN`` in `batch`."""
        empty = []
        for key in self.labels:
            values = _lookup(batch, key)
            if values is None:
                raise KeyError(f"SkipBatchesWithoutLabels: no {key!r} in the training batch")
            values = torch.as_tensor(values)
            labelled = values != -1
            if values.is_floating_point():
                labelled &= ~torch.isnan(values)
            if not bool(labelled.any()):
                empty.append(key)
        return empty

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        empty = self.empty_labels(batch)
        flag = torch.tensor(float(bool(empty)), device=pl_module.device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            self.skip_pending = True
            self.skipped_this_epoch += 1
            self.skipped_total += 1
            reason = ", ".join(empty) if empty else "a batch on another rank"
            _LOG.warning(
                f"skipping training step (epoch {trainer.current_epoch}, batch {batch_idx}): "
                f"no labelled objects for {reason}"
            )

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Any
    ) -> None:
        del trainer, optimizer
        if self.skip_pending:
            for param in pl_module.parameters():
                param.grad = None
        self.skip_pending = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.log("train/skipped_batches", float(self.skipped_this_epoch))
        if self.skipped_this_epoch:
            _LOG.warning(
                f"epoch {trainer.current_epoch}: skipped {self.skipped_this_epoch} training "
                f"step(s) with no labelled objects for one of {list(self.labels)}"
            )
        self.skipped_this_epoch = 0
