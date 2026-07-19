"""Callback driving multi-stage `training_schedule` transitions inside one fit.

At each stage boundary (`on_train_epoch_start` when the epoch crosses into a new
stage) it applies the new stage's freeze mask and rebuilds the optimizer + LR
scheduler via ``trainer.strategy.setup_optimizers`` (S1-spike-validated recipe);
stage 0 is built by Lightning's initial `configure_optimizers`, so the callback
only rebuilds for stages >= 1 (Gotcha #2). Auto-injected by `SaltCLI` when the
model's schedule is multi-stage or freezes anything — the user never registers it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

log = logging.getLogger(__name__)

__all__ = ["TrainingScheduleCallback"]


class TrainingScheduleCallback(Callback):
    """Applies stage-boundary freeze flips + optimizer/scheduler rebuilds for a
    multi-stage `training_schedule`. Stateless — all schedule state lives on the
    `SaltModule` (`_schedule`, `_current_stage_index`, `_frozen_module_names`).
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Before DDP wraps the model (this runs in ``_call_setup_hook``, ahead of
        ``strategy.setup``), enable ``find_unused_parameters`` on a DDP strategy
        when the schedule flips the freeze set across stages — the reducer's
        buckets are fixed at wrap time, so a mid-fit freeze/unfreeze otherwise
        errors ("parameters not used in producing the loss") or silently
        desyncs ranks (S2 spike). A no-op off DDP or for a static freeze set.
        """
        if stage != "fit":
            return
        schedule = getattr(pl_module, "_schedule", None)
        if schedule is None or not schedule.changes_freeze_across_stages():
            return
        strategy = trainer.strategy
        ddp_kwargs = getattr(strategy, "_ddp_kwargs", None)
        if ddp_kwargs is None:  # not a DDP-family strategy — nothing to configure
            return
        if not ddp_kwargs.get("find_unused_parameters", False):
            ddp_kwargs["find_unused_parameters"] = True
            log.warning(
                "training_schedule changes the frozen module set across stages under a DDP "
                "strategy — auto-enabling find_unused_parameters=True so the reducer tolerates "
                "the mid-fit freeze/unfreeze flips (plan 01 W3, S2 spike). This adds a small "
                "per-step overhead; it is required for staged freezing under DDP."
            )

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """At a stage boundary, apply the new stage's freeze mask and rebuild the
        optimizer + OneCycle scheduler over the now-trainable params. Stage 0 (and
        any epoch that stays within the current stage) is left to the initial
        `configure_optimizers`; only crossings into a *later* stage rebuild.
        """
        schedule = pl_module._schedule  # noqa: SLF001 - same-package schedule state
        new_index = schedule.stage_index_for_epoch(trainer.current_epoch, trainer.max_epochs)
        if trainer.current_epoch == 0 or new_index == pl_module._current_stage_index:  # noqa: SLF001
            return
        pl_module._current_stage_index = new_index  # noqa: SLF001
        # freeze mask BEFORE the rebuild so configure_optimizers sees the new
        # requires_grad set (frozen params are excluded from the optimizer).
        pl_module._apply_stage_freeze(schedule.stages[new_index])  # noqa: SLF001
        # rebuild: re-invokes configure_optimizers and refreshes every
        # trainer/strategy optimizer + scheduler reference (S1 spike, G3d).
        trainer.strategy.setup_optimizers(trainer)
