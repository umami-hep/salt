"""Callback driving multi-stage `training_schedule` transitions inside one fit.

At each stage boundary it applies the new stage's freeze mask and rebuilds the
optimizer + LR scheduler via ``trainer.strategy.setup_optimizers`` (stage 0 is
built by Lightning's initial `configure_optimizers`). Boundaries are
epoch-arithmetic unless a stage declares `early_stop`, which makes them
data-dependent via the rank-synced monitored metric. A sample stage advances
mid-epoch on its per-rank step target; `stop_after` ends the fit at a boundary
and writes a ``stop_after=<stage>-...`` checkpoint. Auto-injected by
`SaltCLI`; the user never registers it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

from salt.schedule import reducer_safe_freeze_required
from salt.utils.logging import get_logger

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

    from salt.model.multistage_training import TrainingController
    from salt.schedule import StageConfig

log = get_logger(__name__)

__all__ = ["StageScopedCallbacks", "TrainingScheduleCallback"]


class TrainingScheduleCallback(Callback):
    """Stateless — all schedule state (stage index, freeze set, early-stop
    counters, sample-stage step targets, the `stop_after` record) lives on
    `pl_module.training_controller`, a `TrainingController`.
    """

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Before DDP wraps the model (this runs in ``_call_setup_hook``, ahead of
        ``strategy.setup``), enable ``find_unused_parameters`` on a DDP strategy
        when the schedule flips the freeze set across stages — the reducer's
        buckets are fixed at wrap time, so a mid-fit freeze/unfreeze otherwise
        errors ("parameters not used in producing the loss") or silently
        desyncs ranks. A no-op off DDP or for a static freeze set.
        """
        if stage != "fit":
            return
        schedule = pl_module.training_controller.schedule
        strategy = getattr(trainer, "strategy", None)  # a stub trainer may have none
        # Same predicate that puts SaltModule into reducer-safe freeze mode: a
        # DDP-family strategy + a freeze set that changes across stages. The
        # reducer-safe mode keeps managed params in the reducer across flips; a
        # frozen module that is also loss-disconnected in a stage still produces
        # no grad, so find_unused_parameters remains required.
        if not (reducer_safe_freeze_required(strategy, schedule) or schedule.has_skip_losses):
            return
        ddp_kwargs = getattr(strategy, "_ddp_kwargs", None)
        if ddp_kwargs is not None and not ddp_kwargs.get("find_unused_parameters", False):
            ddp_kwargs["find_unused_parameters"] = True
            log.warning(
                "training_schedule changes the frozen module set across stages or skips losses "
                "under a DDP strategy — auto-enabling find_unused_parameters=True so the reducer "
                "tolerates the mid-fit freeze/unfreeze flips. This adds a small per-step "
                "overhead; it is required for staged freezing under DDP."
            )

    @staticmethod
    def _target_stage_index(trainer: Trainer, controller: TrainingController, epoch: int) -> int:
        """The stage for `epoch`: `next_stage_index` under the sample/`early_stop`
        switches, plain epoch arithmetic otherwise.
        """
        schedule = controller.schedule
        if schedule.has_sample_stages or schedule.has_early_stop:
            return controller.next_stage_index(epoch)
        return schedule.stage_index_for_epoch(epoch, trainer.max_epochs)

    def _maybe_advance_sample_stage(
        self, trainer: Trainer, controller: TrainingController, *, sync: bool, allow_stop: bool
    ) -> bool:
        """Advance (or, under `allow_stop` at the `stop_after` stage, stop) once the
        active sample stage reaches its step target N; returns whether it acted.

        ``sync=True`` (every batch end) runs the collective on EVERY rank — a
        collective entered only by the rank that hit N would hang — plus an
        ``all=True`` tripwire against per-rank step divergence. ``sync=False``
        (epoch start) is the resume seam: a ``stop_after`` checkpoint resumes at
        the next epoch with ``global_step >= N``, so the boundary is re-applied
        once there, advancing rather than re-stopping.
        """
        schedule = controller.schedule
        idx = controller.current_stage_index
        stage = schedule.stages[idx]
        if not stage.is_sample_stage:
            return False
        controller.resolve_step_targets(trainer)
        n = controller.step_target(idx)
        local_hit = trainer.global_step - controller.stage_start_step >= n
        if sync:
            if not trainer.strategy.reduce_boolean_decision(local_hit, all=False):
                return False
            if not trainer.strategy.reduce_boolean_decision(local_hit, all=True):
                raise RuntimeError(
                    f"training_schedule stage {stage.name!r} (sample target N={n} optimizer "
                    f"steps) hit its step boundary on rank {getattr(trainer, 'global_rank', 0)} "
                    f"(global_step={trainer.global_step}, "
                    f"stage_start_step={controller.stage_start_step}, local_hit={local_hit}) "
                    "but not every rank agrees — refusing to advance ranks out of lockstep. "
                    "This is a per-rank optimizer-step divergence."
                )
        elif not local_hit:
            return False
        if allow_stop and schedule.stop_after_index == idx:
            rec = controller.record_stop_after(trainer.global_step, trainer.current_epoch)
            trainer.should_stop = True
            log.info(
                f"training_schedule: stop_after={stage.name!r} reached at global step "
                f"{rec['global_step']} (epoch {rec['epoch']}); the fit ends at this train-epoch "
                "end, where the stop_after checkpoint is written"
            )
            return True
        controller.advance_to_stage(idx + 1, trainer.global_step, trainer.current_epoch, "steps")
        trainer.strategy.setup_optimizers(trainer)
        log.info(
            f"training_schedule: sample-stage boundary {stage.name!r} -> "
            f"{schedule.stages[idx + 1].name!r} at global step {trainer.global_step} "
            f"(epoch {trainer.current_epoch})"
        )
        return True

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Sample stages advance by step count here; a no-op without sample stages."""
        del outputs, batch, batch_idx
        controller = pl_module.training_controller
        if not controller.schedule.has_sample_stages:
            return
        self._maybe_advance_sample_stage(trainer, controller, sync=True, allow_stop=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """At a stage boundary, apply the new stage's freeze mask and rebuild the
        optimizer + OneCycle scheduler over the now-trainable params. Stage 0 (and
        any epoch that stays within the current stage) is left to the initial
        `configure_optimizers`; only crossings into a *later* stage rebuild. The
        target stage is epoch arithmetic by default, or — when any stage is a sample
        stage or declares `early_stop` — the data-dependent `next_stage_index` (advance by
        one on a pending early-stop trigger or the active stage's epoch cap). FIRST
        it records the epoch's start step on the controller (`epoch_start_step`, which
        sizes a stage entered mid-epoch), then runs the sample-stage resume seam
        (`_maybe_advance_sample_stage`).

        This is also the resume boundary handler: after a checkpoint restore,
        `pl_module.training_controller.current_stage_index` holds the *saved* stage
        (set by `SaltModule.on_load_checkpoint` before the optimizer was rebuilt). If
        the resume epoch's implied stage is later (resume exactly at a boundary), this
        fires exactly once to rebuild fresh into that stage — identical to the
        uninterrupted run's boundary. A mid-stage resume sees the same stage and
        does not rebuild, so the restored optimizer moments are kept.
        """
        controller = pl_module.training_controller
        controller.epoch_start_step = trainer.global_step
        schedule = controller.schedule
        if schedule.has_sample_stages and self._maybe_advance_sample_stage(
            trainer, controller, sync=False, allow_stop=False
        ):
            return
        if trainer.current_epoch == 0:
            return
        new_index = self._target_stage_index(trainer, controller, trainer.current_epoch)
        if new_index == controller.current_stage_index:
            return
        reason = (
            "early_stop"
            if schedule.has_early_stop and controller.pending_early_advance
            else "epochs"
        )
        # advance_to_stage applies the freeze mask (delta) BEFORE the rebuild so
        # configure_optimizers sees the new trainable set, and (under early_stop)
        # records the boundary + resets the entered stage's counters.
        controller.advance_to_stage(new_index, trainer.global_step, trainer.current_epoch, reason)
        # rebuild: re-invokes configure_optimizers and refreshes every
        # trainer/strategy optimizer + scheduler reference.
        trainer.strategy.setup_optimizers(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """`stop_after` boundary action at train-epoch end (so mid-epoch validations
        cannot trigger it). An epoch-cap stop stage, at its last epoch, records the
        boundary, sets ``trainer.should_stop`` (a clean stop) and writes
        ``stop_after=<stage>-epoch=NNN.ckpt``; a sample stop stage writes its deferred
        ``stop_after=<stage>-step=NNNNNN.ckpt`` here, first recording the boundary and
        stopping when its `early_stop` fired before the sample target was reached.
        """
        controller = pl_module.training_controller
        schedule = controller.schedule
        stop_idx = schedule.stop_after_index
        if stop_idx is None or controller.current_stage_index != stop_idx:
            return
        stage = schedule.stages[stop_idx]
        if stage.is_sample_stage:
            # pending_early_advance is rank-synced, so every rank stops together
            if controller.stop_after_record is None and controller.pending_early_advance:
                rec = controller.record_stop_after(trainer.global_step, trainer.current_epoch)
                trainer.should_stop = True
                log.info(
                    f"training_schedule: stop_after={stage.name!r} reached by early_stop at "
                    f"global step {rec['global_step']} (epoch {rec['epoch']}), before its "
                    "sample target"
                )
            self._save_sample_stop_after_checkpoint(trainer, controller, stage)
            return
        if self._target_stage_index(trainer, controller, trainer.current_epoch + 1) <= stop_idx:
            return
        rec = controller.record_stop_after(trainer.global_step, trainer.current_epoch)
        trainer.should_stop = True
        path = _stop_after_dir(trainer) / (
            f"stop_after={stage.name}-epoch={trainer.current_epoch:03d}.ckpt"
        )
        trainer.save_checkpoint(path)  # strategy-aware: rank 0 writes; safe on every rank
        log.info(
            f"training_schedule: stop_after={stage.name!r} reached at epoch {rec['epoch']} "
            f"(global step {rec['global_step']}); ending the fit before stage "
            f"{schedule.stages[stop_idx + 1].name!r}. Continue with: salt fit --config "
            f"<run config.yaml> --ckpt_path {path} (drop training_schedule.stop_after)"
        )

    @staticmethod
    def _save_sample_stop_after_checkpoint(
        trainer: Trainer, controller: TrainingController, stage: StageConfig
    ) -> None:
        """Deferred write of a sample-stage `stop_after` checkpoint. A checkpoint
        saved inside `on_train_batch_end` resumes as Lightning's RESTARTED_MID_EPOCH,
        which skips the `on_train_epoch_start` resume seam and over-steps the stage's
        OneCycleLR; written here it resumes at the NEXT epoch (the rest of the stop
        epoch is discarded). `global_step` cannot move after `should_stop`, so the
        ``-step=`` name is the boundary step. No-op unless a stop record is pending.
        """
        rec = controller.stop_after_record
        if rec is None or not trainer.should_stop:
            return
        path = _stop_after_dir(trainer) / (
            f"stop_after={stage.name}-step={trainer.global_step:06d}.ckpt"
        )
        trainer.save_checkpoint(path)  # strategy-aware: rank 0 writes; safe on every rank
        log.info(
            f"training_schedule: stop_after={stage.name!r} checkpoint written at train-epoch "
            f"end (global step {rec['global_step']}, epoch {rec['epoch']}). Continue with: "
            f"salt fit --config <run config.yaml> --ckpt_path {path} "
            "(drop training_schedule.stop_after)"
        )

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
        controller = pl_module.training_controller
        schedule = controller.schedule
        if not schedule.has_early_stop or trainer.sanity_checking:
            return
        stage = schedule.stages[controller.current_stage_index]
        if stage.early_stop is None:
            return
        monitored = _read_monitor(trainer, stage.early_stop.monitor)
        local_should_stop = controller.evaluate_early_stop(monitored)
        should_stop = trainer.strategy.reduce_boolean_decision(local_should_stop, all=False)
        if not should_stop:
            return
        if controller.current_stage_index == len(schedule.stages) - 1:  # the final stage
            trainer.should_stop = True
        else:
            # consumed at the next train-epoch-start transition
            controller.pending_early_advance = True


def _read_monitor(trainer: Trainer, monitor: str) -> float | None:
    """The `monitor` metric from `trainer.callback_metrics` as a float, or ``None``
    when absent (a fail-fast misconfiguration surfaced by `evaluate_early_stop`).
    The value is already rank-reduced when logged with ``sync_dist`` — the synced
    monitor half of the rank-consistency contract.
    """
    if (value := trainer.callback_metrics.get(monitor)) is None:
        return None
    return float(value.item() if hasattr(value, "item") else value)


def _stop_after_dir(trainer: Trainer) -> Path:
    """The checkpoint callback's `dirpath`, else ``trainer.default_root_dir``."""
    dirpath = getattr(trainer.checkpoint_callback, "dirpath", None)
    return Path(dirpath or trainer.default_root_dir)


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
        schedule = pl_module.training_controller.schedule
        if not schedule.has_stage_callbacks:
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
        controller = pl_module.training_controller
        schedule = controller.schedule
        index = controller.current_stage_index
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
    return cls(**(spec.get("init_args") or {}))


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
