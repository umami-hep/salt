"""Runtime multistage-training policy extracted from `SaltModule`.

Holds the stage index, freeze set, early-stop counters, and checkpoint
sub-payload for one model's multistage training schedule; pure configuration
and epoch arithmetic stay in `salt.schedule`. Lightning ordering is owned by
`SaltModule`'s lifecycle entry points (`setup`, `on_load_checkpoint`, ...)
plus `TrainingScheduleCallback` (`salt/callbacks/schedule.py`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from salt.graph.errors import ConfigError
from salt.schedule import (
    EarlyStopTracker,
    StageConfig,
    TrainingSchedule,
    apply_stage_freeze,
    boundary_record,
    reducer_safe_freeze_required,
)

if TYPE_CHECKING:
    from torch import nn

__all__ = ["TrainingController"]


def _require(state: Mapping[str, Any], key: str, where: str) -> Any:
    """``state[key]`` for a checkpoint payload, raising `ConfigError` (not `KeyError`) when
    absent.
    """
    if key not in state:
        raise ConfigError(
            f"checkpoint {where} payload is missing {key!r} - it was not written by this "
            "version of SaltModule (hand-edited or foreign checkpoint); resume is not defined."
        )
    return state[key]


class TrainingController:
    """Runtime multistage-training policy for one `SaltModule`.

    Holds the stage index, freeze set, early-stop counters, and checkpoint
    sub-payload for one model's multistage training schedule. The controller
    holds a REFERENCE to the model's `nn.ModuleDict` (`net`) — it never
    registers it as a submodule, since this is a plain object, not an
    `nn.Module`. It never references the owning `SaltModule` itself: the
    Lightning `trainer` is passed explicitly to every method that needs it,
    rather than stored, because the moved code reads it through two
    accessors with different semantics — `self._trainer` (may be `None`)
    and `self.trainer` (Lightning's property, raises when unattached) — and
    tests swap `model._trainer` for a stand-in AFTER construction.

    Parameters
    ----------
    schedule : TrainingSchedule
        The parsed schedule (pure config, from `salt.schedule`).
    net : nn.ModuleDict
        The owning `SaltModule`'s module dict — held as a reference only.
    lrs : Mapping[str, float]
        The top-level `lrs` mapping (`SaltModule.lrs`, set once at init).
    optimizer : str
        The top-level optimizer name (`SaltModule.optimizer`, set once at
        init).
    """

    def __init__(
        self,
        schedule: TrainingSchedule,
        net: nn.ModuleDict,
        lrs: Mapping[str, float],
        optimizer: str,
    ) -> None:
        self.schedule = schedule
        self.net = net
        self.lrs = lrs
        self.optimizer = optimizer
        # advanced by TrainingScheduleCallback at each stage boundary
        self.current_stage_index = 0
        # set by _apply_stage_freeze; re-asserted every epoch via `SaltModule.train`
        self.frozen_module_names: set[str] = set()
        # per-stage early-stop runtime state — inert unless some stage declares
        # `early_stop`, so unscheduled runs write no new checkpoint state.
        self.stage_start_epoch = 0
        self.early_stop_tracker: EarlyStopTracker | None = None
        self.boundary_records: list[dict[str, Any]] = []
        self.pending_early_advance = False  # plain attribute — replaces the read-only property
        # when True, schedule-managed params keep requires_grad=True at DDP wrap
        # (unfreeze stays rank-synced); "frozen" is enforced by optimizer-exclusion
        # + eval + per-step grad clearing. False = requires_grad-based freeze.
        self.reducer_safe_freeze = False

    # -- fit setup ------------------------------------------------------------------

    def preflight(self, trainer: Any) -> None:
        """Validate the schedule against the attached trainer and run the
        early-stop preflight, before any freeze mask is applied.

        Split from `start_fit` ONLY so that `salt.optim.preflight_lr_schedulers()`
        (called by `SaltModule.setup`) runs between the two at exactly its old
        position — after this early-stop preflight, before `start_fit`'s
        reducer-safe decision — preserving which `ConfigError` fires first on a
        multiply-misconfigured schedule.

        A `ConfigError` propagates from the validators on an over-allocated epoch
        budget, a multi-stage schedule with no finite `trainer.max_epochs` (see
        `TrainingSchedule.validate_epochs`), or an `early_stop` stage under a
        trainer with validation disabled (see `_preflight_early_stop`).
        """
        max_epochs = getattr(trainer, "max_epochs", None)
        self.schedule.validate_epochs(max_epochs)
        self._preflight_early_stop(trainer)

    def start_fit(self, trainer: Any) -> None:
        """Decide the freeze mode and apply stage 0's freeze mask, seeding the
        live stage-transition and early-stop counters at fit setup (before the
        initial optimizer build).

        Split from `preflight` ONLY so that `salt.optim.preflight_lr_schedulers()`
        (called by `SaltModule.setup`) runs between the two at exactly its old
        position — after `preflight`'s early-stop check, before this method's
        reducer-safe decision — preserving which `ConfigError` fires first on a
        multiply-misconfigured schedule.
        """
        # Decide the freeze mode BEFORE applying the stage-0 mask: a managed param
        # must not enter DDP wrap already frozen (init-frozen-unfreeze desync).
        self.reducer_safe_freeze = reducer_safe_freeze_required(
            getattr(trainer, "strategy", None), self.schedule
        )
        self.current_stage_index = 0
        self.stage_start_epoch = 0
        self.boundary_records = []
        self.pending_early_advance = False
        self.early_stop_tracker = self._make_early_stop_tracker(self.schedule.stages[0])
        self._apply_stage_freeze(self.schedule.stages[0])

    def _preflight_early_stop(self, trainer: Any) -> None:
        """Fail fast at fit setup if the schedule declares `early_stop` but the
        trainer cannot ever evaluate it (validation disabled) — an early-stop that
        can never fire would silently wait for the epoch cap. Inert unless the
        schedule declares an `early_stop` (the `has_early_stop` master switch).

        Raises
        ------
        ConfigError
            A stage declares `early_stop` but ``trainer.limit_val_batches == 0``.
        """
        if not self.schedule.has_early_stop:
            return
        if getattr(trainer, "limit_val_batches", 1) == 0:
            raise ConfigError(
                "training_schedule declares an 'early_stop' stage but the trainer has validation "
                "disabled (limit_val_batches=0) — early stopping is evaluated on validation-epoch "
                "end and could never fire. Enable validation or remove early_stop."
            )

    def _make_early_stop_tracker(self, stage: StageConfig) -> EarlyStopTracker | None:
        """A fresh `EarlyStopTracker` for `stage` (reset counters), or ``None`` when
        the stage declares no `early_stop`.
        """
        return EarlyStopTracker(stage.early_stop) if stage.early_stop is not None else None

    def _apply_stage_freeze(self, stage: StageConfig) -> None:
        """Apply `stage`'s freeze mask as a DELTA against the currently-frozen
        set: modules entering the frozen set get ``eval()`` (stops
        dropout + running-stat updates during training); modules leaving it are
        restored to ``train()``. Modules outside both sets are left untouched, so
        the desugared no-freeze path never mutates a param (bitwise parity).

        The requires_grad handling depends on the freeze mode (see
        `apply_stage_freeze`): the default mode toggles ``requires_grad`` so the
        optimizer excludes frozen params by it; reducer-safe mode (under DDP
        with a freeze-flipping schedule) leaves ``requires_grad=True`` on every
        managed param so the reducer keeps managing it across the flip, excluding
        frozen params from the optimizer by membership instead. Records the frozen
        set so `SaltModule.train` re-asserts eval after Lightning's per-epoch
        ``model.train()`` and `configure_optimizers`/`on_after_backward` can key
        off it.
        """
        frozen = self.schedule.frozen_names(stage)
        apply_stage_freeze(
            self.net,
            frozen,
            self.frozen_module_names,
            reducer_safe=self.reducer_safe_freeze,
        )
        self.frozen_module_names = frozen

    # -- stage transitions (driven by TrainingScheduleCallback) ---------------------

    def advance_to_stage(self, new_index: int, global_step: int, epoch: int, reason: str) -> None:
        """Enter stage `new_index`: set the active index + apply its freeze mask
        (the delta off the previous stage). Under the `early_stop` master switch,
        also records the boundary, resets the stage's live counters, and marks the
        new stage's start epoch (so the per-stage epoch-cap count is measured from
        here). The optimizer/LR rebuild is done by the caller
        (`TrainingScheduleCallback`) via ``strategy.setup_optimizers`` right after.
        `reason` is ``"epochs"`` or ``"early_stop"`` (recorded only under the
        `early_stop` switch).
        """
        self.current_stage_index = new_index
        stage = self.schedule.stages[new_index]
        self._apply_stage_freeze(stage)
        if self.schedule.has_early_stop:
            self.stage_start_epoch = epoch
            self.pending_early_advance = False
            self.boundary_records.append(
                boundary_record(stage.name, new_index, global_step, epoch, reason)
            )
            self.early_stop_tracker = self._make_early_stop_tracker(stage)

    def next_stage_index_early_stop(self, current_epoch: int) -> int:
        """The stage index to run at `current_epoch` under the `early_stop` switch:
        advance by exactly one when the active stage's early-stop fired (a pending
        advance) OR it reached its epoch cap (``current_epoch - stage_start >=
        epochs``); the final stage never advances by cap (its early-stop ends the
        fit instead). Data-dependent boundaries make this replace the pure
        epoch-arithmetic `stage_index_for_epoch` whenever any stage can early-stop.
        """
        idx = self.current_stage_index
        if idx >= len(self.schedule.stages) - 1:
            return idx
        stage = self.schedule.stages[idx]
        cap_reached = stage.epochs is not None and (current_epoch - self.stage_start_epoch) >= (
            stage.epochs
        )
        return idx + 1 if (self.pending_early_advance or cap_reached) else idx

    def evaluate_early_stop(self, monitored: float | None) -> bool:
        """Fold the active stage's monitored validation value into its early-stop
        tracker and return the LOCAL early-stop decision (this rank). Called by the
        callback at validation-epoch end with the RANK-REDUCED monitor value (or
        ``None`` when the metric is absent — a fail-fast misconfiguration). The
        caller rank-syncs the returned boolean before acting on it; this method only
        advances the stage-local counters. Returns ``False`` (no stop) when the
        active stage declares no `early_stop`; raises `ConfigError` when the
        monitored metric is absent from ``trainer.callback_metrics``.
        """
        stage = self.schedule.stages[self.current_stage_index]
        if stage.early_stop is None:
            return False
        if monitored is None:
            raise ConfigError(
                f"training_schedule stage {stage.name!r} early_stop monitors "
                f"{stage.early_stop.monitor!r} but it is absent from trainer.callback_metrics — "
                "check the metric name (e.g. 'val/loss') or that validation logs it."
            )
        assert self.early_stop_tracker is not None
        return self.early_stop_tracker.check(monitored)

    def mark_pending_early_advance(self) -> None:
        """Flag that the active (non-final) stage's early-stop has fired — consumed
        at the next train-epoch-start transition (`next_stage_index_early_stop`).
        """
        self.pending_early_advance = True

    # -- optimizer --------------------------------------------------------------

    def active_optim_config(self) -> tuple[Mapping[str, float], str]:
        """The ``(lrs, optimizer_name)`` for the currently-active stage. The
        desugared single `fit` stage overrides neither, so this returns the
        top-level pair unchanged (the bitwise-parity path). A stage's `lrs`
        deep-overrides the top-level keys; its `optimizer` replaces the name.
        """
        stage = self.schedule.stages[self.current_stage_index]
        lrs = {**self.lrs, **stage.lrs} if stage.lrs is not None else self.lrs
        return lrs, stage.optimizer or self.optimizer

    def stage_total_steps(self, trainer: Any) -> int:
        """The `OneCycleLR.total_steps` for the active stage. A single-stage
        schedule uses the whole-run `estimated_stepping_batches` exactly (parity);
        a multi-stage schedule uses this stage's proportional per-stage allocation
        of that estimate (never the whole-run figure for a sub-stage).
        Under the `early_stop` switch the envelope is sized from the stage's epoch
        cap measured from its ACTUAL start epoch (see `_early_stop_stage_total_steps`)
        so a stage that starts early — because an earlier stage early-stopped — never
        over-steps its OneCycle.
        """
        total = trainer.estimated_stepping_batches
        if not self.schedule.is_multi_stage:
            return total
        if self.schedule.has_early_stop:
            return self._early_stop_stage_total_steps(total, trainer)
        allocations = self.schedule.stage_step_allocations(total, trainer.max_epochs)
        return allocations[self.current_stage_index]

    def _early_stop_stage_total_steps(self, total: int, trainer: Any) -> int:
        """OneCycle `total_steps` for the active stage under the `early_stop` switch:
        a per-epoch step estimate (``total / max_epochs``) times the stage's epoch
        budget — its `epochs` cap for a non-final stage, or ``max_epochs -
        stage_start_epoch`` for the final stage (whose real length depends on when
        earlier stages ended). Sized ``>=`` the steps a stage can actually take, so
        OneCycle never over-steps; equals the arithmetic split when no stage stops
        early. Truncated early, the stage simply under-runs its envelope (D-ES).
        """
        max_epochs = trainer.max_epochs
        steps_per_epoch = max(1, round(total / max_epochs))
        index = self.current_stage_index
        is_final = index == len(self.schedule.stages) - 1
        stage = self.schedule.stages[index]
        if is_final:
            budget_epochs = max_epochs - self.stage_start_epoch
        else:
            assert stage.epochs is not None  # non-final stages require epochs
            budget_epochs = stage.epochs
        return max(1, steps_per_epoch * budget_epochs)

    # -- checkpoints --------------------------------------------------------------

    def checkpoint_state(self) -> dict[str, Any]:
        """The `schedule` sub-payload for the checkpoint. Legacy/no-early-stop
        configs get exactly ``{stage_index, stage_name}`` (byte-identical to the
        legacy payload — a parity guard). Under the `early_stop` master switch it
        additionally carries the active stage's start epoch, the completed-boundary
        records, and the live early-stop counters, so a data-dependent resume
        reconstructs the exact stage position + patience state.
        """
        state: dict[str, Any] = {
            "stage_index": self.current_stage_index,
            "stage_name": self.schedule.stages[self.current_stage_index].name,
        }
        if self.schedule.has_early_stop:
            state["stage_start_epoch"] = self.stage_start_epoch
            state["boundaries"] = list(self.boundary_records)
            if self.early_stop_tracker is not None:
                state["early_stop_state"] = self.early_stop_tracker.state_dict()
        return state

    def restore_schedule_stage(
        self, schedule_state: Mapping[str, Any] | None, trainer: Any
    ) -> None:
        """On a fit resume, restore the saved stage index + freeze mask
        (multi-stage only; the single-stage path stays bitwise-untouched), then
        the early-stop counters (any-stage, when declared). No-op unless the
        trainer is fitting — `salt test --ckpt_path` never mutates this state.
        A saved stage out of range for, or named differently than, the current
        schedule raises `ConfigError` (schedule changed; resume is undefined).
        A payload missing any key this version's writer emits (`stage_index`,
        `stage_name`, and under `early_stop`: `stage_start_epoch`, `boundaries`,
        `early_stop_state`, `early_stop_state.fingerprint`) raises `ConfigError`
        — there is no legacy-payload path.
        """
        if schedule_state is None:
            return
        from lightning.pytorch.trainer.states import TrainerFn

        fn = getattr(getattr(trainer, "state", None), "fn", None)
        if fn is not None and fn != TrainerFn.FITTING:
            return
        if self.schedule.is_multi_stage:
            index = _require(schedule_state, "stage_index", "schedule")
            if not 0 <= index < len(self.schedule.stages):
                raise ConfigError(
                    f"checkpoint records training_schedule stage index {index}, out of range for "
                    f"the current {len(self.schedule.stages)}-stage schedule — the schedule "
                    "changed since the checkpoint was written; resume is not defined."
                )
            saved_name = _require(schedule_state, "stage_name", "schedule")
            current_name = self.schedule.stages[index].name
            if saved_name != current_name:
                raise ConfigError(
                    f"checkpoint records training_schedule stage {index} as {saved_name!r} but the "
                    f"current schedule names it {current_name!r} — the schedule changed since the "
                    "checkpoint was written; resume is not defined."
                )
            self.current_stage_index = index
            self._apply_stage_freeze(self.schedule.stages[index])
        else:
            index = 0
        if self.schedule.has_early_stop:
            self._restore_early_stop_state(index, schedule_state)

    def _restore_early_stop_state(self, index: int, schedule_state: Mapping[str, Any]) -> None:
        """Restore early-stop resume state (stage start epoch, boundary records,
        live counters) so a mid-stage resume continues patience exactly. A stage
        without an `early_stop` criterion restores no tracker (the writer emits
        no `early_stop_state` for it); a criterion-fingerprint mismatch with the
        current config raises `ConfigError` (resume undefined).
        """
        self.stage_start_epoch = _require(schedule_state, "stage_start_epoch", "schedule")
        self.boundary_records = list(_require(schedule_state, "boundaries", "schedule"))
        self.pending_early_advance = False
        stage = self.schedule.stages[index]
        if stage.early_stop is None:
            self.early_stop_tracker = None
            return
        es_state = _require(schedule_state, "early_stop_state", "schedule")
        saved_fp = _require(es_state, "fingerprint", "schedule.early_stop_state")
        current_fp = stage.early_stop.fingerprint()
        if saved_fp != current_fp:
            raise ConfigError(
                f"checkpoint records an early_stop criterion {saved_fp} for stage "
                f"{stage.name!r} but the current config declares {current_fp} — the criterion "
                "changed since the checkpoint; patience resume is not defined."
            )
        self.early_stop_tracker = EarlyStopTracker.from_state_dict(stage.early_stop, es_state)
