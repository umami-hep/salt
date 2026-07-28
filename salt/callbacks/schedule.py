"""Callback driving multi-stage `training_schedule` transitions inside one fit.

At each stage boundary (`on_train_epoch_start` when the epoch crosses into a new
stage) it applies the new stage's freeze mask and rebuilds the optimizer + LR
scheduler via ``trainer.strategy.setup_optimizers`` (S1-spike-validated recipe);
stage 0 is built by Lightning's initial `configure_optimizers`, so the callback
only rebuilds for stages >= 1 (Gotcha #2). Boundaries are epoch-arithmetic by
default; when any stage declares `early_stop` they become
data-dependent — `on_validation_end` folds the monitored metric into the active
stage's tracker (rank-synced) and either flags a pending advance (non-final) or
stops the fit (final). Auto-injected by `SaltCLI` when the model's schedule is
multi-stage or freezes anything — the user never registers it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

from salt.schedule import reducer_safe_freeze_required

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

log = logging.getLogger(__name__)

__all__ = ["StageScopedCallbacks", "TrainingScheduleCallback"]


class TrainingScheduleCallback(Callback):
    """Applies stage-boundary freeze flips + optimizer/scheduler rebuilds for a
    multi-stage `training_schedule`, and drives per-stage early stopping when
    declared. Stateless — all schedule state lives on the `SaltModule`
    (`_schedule`, `_current_stage_index`, `_frozen_module_names`, and the
    early-stop counters).
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
        strategy = trainer.strategy
        # Same predicate that puts SaltModule into reducer-safe freeze mode: a
        # DDP-family strategy + a freeze set that changes across stages. The
        # reducer-safe mode keeps managed params in the reducer across flips; a
        # frozen module that is also loss-disconnected in a stage still produces
        # no grad, so find_unused_parameters remains required.
        if not reducer_safe_freeze_required(strategy, schedule):
            return
        ddp_kwargs = getattr(strategy, "_ddp_kwargs", None)
        if ddp_kwargs is not None and not ddp_kwargs.get("find_unused_parameters", False):
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
        `configure_optimizers`; only crossings into a *later* stage rebuild. The
        target stage is epoch arithmetic by default, or — when any stage declares
        `early_stop` — the data-dependent `next_stage_index_early_stop` (advance by
        one on a pending early-stop trigger or the active stage's epoch cap).

        This is also the resume boundary handler: after a checkpoint restore,
        `pl_module._current_stage_index` holds the *saved* stage (set by
        `SaltModule.on_load_checkpoint` before the optimizer was rebuilt). If the
        resume epoch's implied stage is later (resume exactly at a boundary), this
        fires exactly once to rebuild fresh into that stage — identical to the
        uninterrupted run's boundary. A mid-stage resume sees the same stage and
        does not rebuild, so the restored optimizer moments are kept.
        """
        schedule = pl_module._schedule  # noqa: SLF001 - same-package schedule state
        if trainer.current_epoch == 0:
            return
        if schedule.has_early_stop:
            reason = "early_stop" if pl_module.pending_early_advance else "epochs"
            new_index = pl_module.next_stage_index_early_stop(trainer.current_epoch)
        else:
            new_index = schedule.stage_index_for_epoch(trainer.current_epoch, trainer.max_epochs)
            reason = "epochs"
        if new_index == pl_module._current_stage_index:  # noqa: SLF001
            return
        # advance_to_stage applies the freeze mask (delta) BEFORE the rebuild so
        # configure_optimizers sees the new trainable set, and (under early_stop)
        # records the boundary + resets the entered stage's counters.
        pl_module.advance_to_stage(new_index, trainer.global_step, trainer.current_epoch, reason)
        # rebuild: re-invokes configure_optimizers and refreshes every
        # trainer/strategy optimizer + scheduler reference (S1 spike, G3d).
        trainer.strategy.setup_optimizers(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Per-stage early-stop check. On the real validation pass
        (skips Lightning's sanity check), read the active stage's `monitor` from the
        rank-reduced `trainer.callback_metrics`, fold it into the stage's tracker,
        and rank-sync the decision via ``strategy.reduce_boolean_decision`` so every
        rank agrees on the boundary (no reducer/optimizer-rebuild desync). A trigger
        on a non-final stage flags a pending advance (executed at the next
        `on_train_epoch_start`); on the final stage it sets ``trainer.should_stop``.
        No-op unless the active stage declares `early_stop`.
        """
        schedule = getattr(pl_module, "_schedule", None)
        if schedule is None or not schedule.has_early_stop or trainer.sanity_checking:
            return
        stage = schedule.stages[pl_module._current_stage_index]  # noqa: SLF001
        if stage.early_stop is None:
            return
        monitored = _read_monitor(trainer, stage.early_stop.monitor)
        local_should_stop = pl_module.evaluate_early_stop(monitored)
        should_stop = trainer.strategy.reduce_boolean_decision(local_should_stop, all=False)
        if not should_stop:
            return
        is_final = pl_module._current_stage_index == len(schedule.stages) - 1  # noqa: SLF001
        if is_final:
            trainer.should_stop = True
        else:
            pl_module.mark_pending_early_advance()


def _read_monitor(trainer: Trainer, monitor: str) -> float | None:
    """The `monitor` metric from `trainer.callback_metrics` as a float, or ``None``
    when absent (a fail-fast misconfiguration surfaced by `evaluate_early_stop`).
    The value is already rank-reduced when logged with ``sync_dist`` — the synced
    monitor half of the rank-consistency contract.
    """
    value = trainer.callback_metrics.get(monitor)
    if value is None:
        return None
    return float(value.item() if hasattr(value, "item") else value)


# Lightning callback hooks the coordinator forwards to the active stage's scoped
# delegates. setup/teardown are handled explicitly (validation + build + teardown);
# on_fit_start/end + on_train_start/end are fit-level (not per-stage) and are NOT
# forwarded — a stage-scoped callback's lifecycle is its stage, bracketed by the
# manual setup()/teardown() the coordinator calls when it (re)builds the delegates.
_STAGE_SCOPED_HOOKS = (
    "on_train_epoch_start",
    "on_train_epoch_end",
    "on_validation_epoch_start",
    "on_validation_epoch_end",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_validation_batch_start",
    "on_validation_batch_end",
    "on_before_backward",
    "on_after_backward",
    "on_before_optimizer_step",
    "on_before_zero_grad",
    "on_validation_start",
    "on_validation_end",
)


class StageScopedCallbacks(Callback):
    """Coordinator for per-stage scoped `callbacks`. Registered
    ONCE by `SaltCLI` when the schedule declares any stage `callbacks` — Lightning
    fixes ``trainer.callbacks`` at fit start, so the always-propagated top-level
    (global) callbacks persist for the whole fit while THIS coordinator hosts the
    stage-scoped ones: at each stage entry it instantiates the entering stage's
    callbacks FRESH (fresh state), forwards Lightning's per-stage hooks to
    them ONLY while their stage is active, and tears them down at stage exit. The
    user's "combined set" per stage is therefore the persistent globals plus these
    freshly-instantiated stage-scoped delegates.

    Stateless at construction (empty delegate list) so it is picklable for DDP
    spawn; delegates are built lazily in the (spawned) worker at the first hook.
    """

    def __init__(self) -> None:
        self._active_delegates: list[Callback] = []
        self._active_stage_index: int | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """At fit setup, import+instantiate EVERY declared stage callback once so a
        bad ``class_path``/``init_args`` fails before training (not at a boundary),
        then discards them; the active stage's live delegates are built lazily at
        the first per-stage hook. No-op off ``fit`` or without stage callbacks.
        """
        del trainer  # validation only reads the model's schedule
        if stage != "fit":
            return
        schedule = getattr(pl_module, "_schedule", None)
        if schedule is None or not schedule.has_stage_callbacks:
            return
        for stage_cfg in schedule.stages:
            for spec in stage_cfg.callbacks or ():
                _instantiate_stage_callback(spec)  # validation only — discarded

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """At fit teardown, tear down the active stage's delegates and forget them."""
        if stage != "fit":
            return
        for delegate in self._active_delegates:
            delegate.teardown(trainer, pl_module, "fit")
        self._active_delegates = []
        self._active_stage_index = None

    def _sync_active_stage(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Rebuild the scoped delegates when the active stage changed since the last
        hook: tear down the leaving stage's delegates, instantiate the entering
        stage's fresh, and call their `setup` (Lightning already ran its own setup
        phase before this stage existed). Cheap no-op while the stage is unchanged.
        """
        schedule = getattr(pl_module, "_schedule", None)
        if schedule is None:
            return
        index = pl_module._current_stage_index  # noqa: SLF001 - same-package schedule state
        if index == self._active_stage_index:
            return
        for delegate in self._active_delegates:
            delegate.teardown(trainer, pl_module, "fit")
        specs = schedule.stages[index].callbacks or ()
        self._active_delegates = [_instantiate_stage_callback(spec) for spec in specs]
        self._active_stage_index = index
        for delegate in self._active_delegates:
            delegate.setup(trainer, pl_module, "fit")


def _instantiate_stage_callback(spec: Mapping[str, Any]) -> Callback:
    """Instantiate one stage-scoped callback from a ``{class_path[, init_args]}``
    spec (`init_args` passed as keyword arguments verbatim).

    Returns
    -------
    Callback
        The instantiated callback.
    """
    from salt.main import _resolve_class_path

    cls = _resolve_class_path(spec["class_path"])
    init_args = spec.get("init_args") or {}
    return cls(**init_args)


def _make_stage_hook_forwarder(hook_name: str):
    """Build the coordinator method for `hook_name`: sync the active stage, then
    forward the hook to each live stage-scoped delegate.
    """

    def _forward(
        self: StageScopedCallbacks,
        trainer: Trainer,
        pl_module: LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._sync_active_stage(trainer, pl_module)
        for delegate in self._active_delegates:
            getattr(delegate, hook_name)(trainer, pl_module, *args, **kwargs)

    _forward.__name__ = hook_name
    _forward.__qualname__ = f"StageScopedCallbacks.{hook_name}"
    return _forward


for _hook in _STAGE_SCOPED_HOOKS:
    setattr(StageScopedCallbacks, _hook, _make_stage_hook_forwarder(_hook))
del _hook
