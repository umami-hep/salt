"""Training-schedule schema: named stages with per-stage freeze + optimizer/LR.

Parsed and validated fail-loud at `SaltModule.__init__`. It is the single
canonical home for optimizer/LR config: a plain (no
`training_schedule`) config desugars to a single `fit` stage (see
`desugar_legacy`), so `SaltModule.configure_optimizers` has exactly one code
path. This module also owns the per-stage epoch→step allocation math and the
stage-boundary lookup the `TrainingScheduleCallback` drives.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from salt.graph.errors import ConfigError

__all__ = [
    "EarlyStopConfig",
    "EarlyStopTracker",
    "LRSchedulerConfig",
    "StageConfig",
    "TrainingSchedule",
    "apply_stage_freeze",
    "boundary_record",
    "clear_frozen_grads",
    "reducer_safe_freeze_required",
    "trainable_named_params",
]

# OneCycle-only `lrs:` keys — meaningless (and rejected in a stage's OWN override)
# when the stage swaps in a custom `lr_scheduler`; `initial`/`weight_decay` remain
# the optimizer's base LR / weight decay and stay valid.
_ONECYCLE_ONLY_LRS = frozenset({"max", "end", "pct_start", "last_epoch"})


@dataclass(frozen=True)
class EarlyStopConfig:
    """A stage's early-stopping criterion. Mirrors Lightning
    `EarlyStopping` vocabulary: end the stage — advance to the next, or end the fit
    on the final stage — when `monitor` fails to improve by at least `min_delta`
    for `patience` consecutive validation checks. The stage's `epochs` remains the
    hard cap (a stage ends at whichever comes first). An early-stopped stage simply
    truncates its OneCycle envelope mid-curve; the next stage rebuilds cleanly.
    """

    monitor: str
    mode: str = "min"
    patience: int = 3
    min_delta: float = 0.0
    check_finite: bool = True

    def fingerprint(self) -> dict[str, Any]:
        """The criterion identity persisted alongside the live counters, so a
        mid-stage resume can reject a checkpoint whose `early_stop` config has since
        changed (patience resume is undefined across a criterion change).

        Returns
        -------
        dict[str, Any]
            The monitor/mode/min_delta/patience fingerprint.
        """
        return {key: getattr(self, key) for key in ("monitor", "mode", "min_delta", "patience")}


@dataclass(frozen=True)
class LRSchedulerConfig:
    """A stage's LR-scheduler class choice. `class_path`/`init_args`
    name a `torch.optim.lr_scheduler` class instantiated at the stage boundary over
    the freshly-rebuilt stage optimizer (never a user-supplied `optimizer`). The
    remaining fields are Lightning scheduler-config keys: `interval` (epoch|step),
    `frequency`, and `monitor` (REQUIRED for a metric-driven scheduler, e.g.
    `ReduceLROnPlateau`). Absent on a stage → the default per-stage OneCycleLR.
    """

    class_path: str
    init_args: Mapping[str, Any] | None = None
    interval: str = "epoch"
    frequency: int = 1
    monitor: str | None = None


@dataclass(frozen=True)
class StageConfig:
    """One training stage: its epoch budget, freeze spec, and optional
    optimizer/LR overrides. Exactly one of `frozen`/`trainable` may be set
    (default `frozen=()` — everything trainable). `epochs=None` means "take the
    remaining epochs" (only the final stage may omit it). `order` pins execution
    position; otherwise declaration order is used. `early_stop` optionally ends the
    stage before its epoch cap when a monitored metric stops improving. A non-final
    sample stage sets `num_samples` XOR `frac_samples` (not `epochs`), ending mid-epoch on N.
    `skip_losses` names task losses this stage does not train; modules feeding only those are
    not executed.
    """

    name: str
    epochs: int | None = None
    num_samples: int | None = None
    frac_samples: float | None = None
    frozen: tuple[str, ...] | None = None
    trainable: tuple[str, ...] | None = None
    optimizer: str | None = None
    lrs: Mapping[str, float] | None = None
    order: int | None = None
    early_stop: EarlyStopConfig | None = None
    callbacks: tuple[Mapping[str, Any], ...] | None = None
    lr_scheduler: LRSchedulerConfig | None = None
    skip_losses: tuple[str, ...] = ()

    @property
    def is_sample_stage(self) -> bool:
        """Whether the stage ends on a `num_samples`/`frac_samples` step target."""
        return self.num_samples is not None or self.frac_samples is not None


# recognised per-stage / `early_stop` / `lr_scheduler` keys: exactly the dataclass fields
# (a stage's `name` is its mapping key). Anything else is a config typo, rejected fail-loud.
_STAGE_FIELDS = frozenset(f.name for f in fields(StageConfig)) - {"name"}
_EARLY_STOP_FIELDS = frozenset(f.name for f in fields(EarlyStopConfig))
_LR_SCHEDULER_FIELDS = frozenset(f.name for f in fields(LRSchedulerConfig))


class TrainingSchedule:
    """Parsed, validated training schedule — an ordered list of `StageConfig`.

    Names in `frozen`/`trainable` must be `model.modules` keys (the `net.<name>.*`
    state-dict prefixes); a freeze spec resolves to the set of module names whose
    parameters are held fixed for that stage (see `frozen_names`).
    """

    def __init__(
        self, stages: list[StageConfig], module_names: Sequence[str], stop_after: str | None = None
    ) -> None:
        self.stages = stages
        self._module_names = tuple(module_names)
        self.stop_after: str | None = stop_after

    @property
    def is_multi_stage(self) -> bool:
        """Whether the schedule declares more than one stage."""
        return len(self.stages) > 1

    @property
    def stop_after_index(self) -> int | None:
        """Execution-order index of the `stop_after` stage, or ``None``."""
        names = [stage.name for stage in self.stages]
        return None if self.stop_after is None else names.index(self.stop_after)

    @property
    def module_names(self) -> tuple[str, ...]:
        """The model-module names this schedule's freeze specs range over."""
        return self._module_names

    @property
    def has_freezing(self) -> bool:
        """Whether any stage freezes at least one module."""
        return any(self.frozen_names(stage) for stage in self.stages)

    @property
    def has_early_stop(self) -> bool:
        """Whether any stage declares an `early_stop` criterion — the master switch
        that gates every early-stop code path. When ``False``, boundaries are
        pure epoch arithmetic and no early-stop checkpoint state is written.
        """
        return any(stage.early_stop is not None for stage in self.stages)

    @property
    def has_skip_losses(self) -> bool:
        """Whether any stage declares `skip_losses`."""
        return any(stage.skip_losses for stage in self.stages)

    @property
    def has_stage_callbacks(self) -> bool:
        """Whether any stage declares scoped `callbacks` — the switch that injects
        the `StageScopedCallbacks` coordinator. When ``False`` no
        coordinator is added and callback handling is left untouched.
        """
        return any(stage.callbacks is not None for stage in self.stages)

    @property
    def has_lr_scheduler(self) -> bool:
        """Whether any stage overrides the LR-scheduler class — the
        master switch for the per-stage `lr_scheduler`. When ``False`` every stage
        uses the default per-stage OneCycleLR.
        """
        return any(stage.lr_scheduler is not None for stage in self.stages)

    def changes_freeze_across_stages(self) -> bool:
        """Whether the frozen set differs between any two consecutive stages —
        the condition under which DDP needs ``find_unused_parameters=True``
        (freeze/unfreeze flips break the reducer's fixed bucketing otherwise).
        """
        masks = [frozenset(self.frozen_names(stage)) for stage in self.stages]
        return any(a != b for a, b in itertools.pairwise(masks))

    def frozen_names(self, stage: StageConfig) -> set[str]:
        """Resolve `stage`'s freeze spec to the set of frozen module names:
        `frozen` freezes those modules; `trainable` freezes their complement over
        `model.modules`; neither freezes nothing.
        """
        if stage.trainable is not None:
            return set(self._module_names) - set(stage.trainable)
        return set(stage.frozen or ())

    def _non_final_epoch_bounds(self) -> list[int]:
        """Cumulative epoch index at which each *non-final* stage ends (its
        explicit `epochs` summed left-to-right). The final stage owns everything
        from the last bound to `max_epochs`, so it is not represented here.
        """
        bounds, acc = [], 0
        for stage in self.stages[:-1]:
            if stage.is_sample_stage:  # 0 epochs: runs inside the next epoch stage
                continue
            assert stage.epochs is not None  # validated by validate_epochs
            acc += stage.epochs
            bounds.append(acc)
        return bounds

    def stage_index_for_epoch(self, epoch: int, max_epochs: int) -> int:
        """The index of the stage that owns `epoch` (0-based). Non-final stages
        own ``[bound_{i-1}, bound_i)``; the final stage owns everything from the
        last bound onward (so any epochs past the explicit budgets run there).
        """
        # `max_epochs` is accepted for symmetry with `stage_step_allocations`,
        # which the same caller invokes with the same value; the final stage
        # owning everything past the last bound makes it unnecessary here.
        del max_epochs
        bounds = self._non_final_epoch_bounds()
        return next((i for i, bound in enumerate(bounds) if epoch < bound), len(self.stages) - 1)

    @property
    def has_sample_stages(self) -> bool:
        """Whether any stage is a `num_samples`/`frac_samples` sample stage."""
        return any(stage.is_sample_stage for stage in self.stages)

    def sample_stage_indices(self) -> list[int]:
        """Execution-order indices of the sample stages."""
        return [i for i, stage in enumerate(self.stages) if stage.is_sample_stage]

    def stage_step_targets(
        self,
        batch_size: int,
        world_size: int,
        accumulate_grad_batches: int = 1,
        steps_per_epoch: int | None = None,
    ) -> dict[int, int]:
        """The single samples→steps conversion site: ``{stage_index: N}`` per-rank
        optimizer steps per sample stage, ``ceil(num_samples / (batch_size *
        world_size * accumulate_grad_batches))`` or ``ceil(frac_samples * steps_per_epoch)``.
        """
        checks: dict[str, Any] = {"batch_size": batch_size, "world_size": world_size}
        checks["accumulate_grad_batches"] = accumulate_grad_batches
        if steps_per_epoch is not None:
            checks["steps_per_epoch"] = steps_per_epoch
        for field, value in checks.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(
                    f"stage_step_targets '{field}' must be a positive integer (got {value!r})."
                )
        divisor = batch_size * world_size * accumulate_grad_batches
        targets: dict[int, int] = {}
        for i, stage in enumerate(self.stages):
            if stage.num_samples is not None:
                targets[i] = math.ceil(stage.num_samples / divisor)
            elif stage.frac_samples is not None:
                if steps_per_epoch is None:
                    raise ConfigError(
                        f"stage {stage.name!r}: frac_samples needs steps_per_epoch (loader length)."
                    )
                targets[i] = math.ceil(stage.frac_samples * steps_per_epoch)
        return targets

    def stage_step_allocations(
        self, total_steps: int, max_epochs: int, step_targets: Mapping[int, int] | None = None
    ) -> list[int]:
        """Split `total_steps` (Lightning's whole-run `estimated_stepping_batches`)
        across the stages so each stage's `OneCycleLR.total_steps` spans only that
        stage (a single stage returns ``[total_steps]``). Epoch ``e`` starts at step
        ``round(total_steps * e / max_epochs)``; a sample stage takes exactly its
        `step_targets` N, an epoch stage is measured from its actual start epoch
        (as in `TrainingController.next_stage_index`), the final stage takes the rest.
        Raises `ConfigError` if the sample targets leave the final stage no steps.
        """
        if not self.is_multi_stage:
            return [total_steps]
        targets: Mapping[int, int] = step_targets or {}
        if missing := [i for i in self.sample_stage_indices() if i not in targets]:
            raise ConfigError(
                "stage_step_allocations needs a step target for every sample stage (missing "
                f"index(es) {missing}) — pass `step_targets` from `stage_step_targets`."
            )

        def boundary(epoch: int) -> int:
            return round(total_steps * epoch / max_epochs)

        def epoch_containing(step: int) -> int:  # largest e with boundary(e) < step (step > 0)
            epoch = 0
            while boundary(epoch + 1) < step:
                epoch += 1
            return epoch

        allocs: list[int] = []
        cursor, cursor_epoch, prev_sample = 0, 0, False
        for i, stage in enumerate(self.stages[:-1]):
            if stage.is_sample_stage:
                cursor += (alloc := targets[i])
                prev_sample = True
            else:
                assert stage.epochs is not None  # validated by validate_epochs
                # after a sample stage, start in the epoch its boundary fired in
                start_epoch = epoch_containing(cursor) if prev_sample and cursor else cursor_epoch
                cursor_epoch = start_epoch + stage.epochs
                end = boundary(cursor_epoch)
                # a sample stage ending on this end edge leaves 0 steps; OneCycleLR needs >= 1
                alloc = max(1, end - cursor)
                cursor = max(cursor, end)
                prev_sample = False
            allocs.append(alloc)
        if (remainder := total_steps - cursor) < 1:  # final stage: exact remainder
            raise ConfigError(
                f"training_schedule sample-stage step targets {dict(targets)} consume {cursor} of "
                f"{total_steps} steps, leaving the final stage {self.stages[-1].name!r} none - "
                "lower num_samples/frac_samples or raise trainer.max_epochs."
            )
        allocs.append(remainder)
        return allocs

    def validate_epochs(self, max_epochs: int | None) -> None:
        """Validate the stage epoch allocation against `trainer.max_epochs`
        (available only once the trainer is attached, so this runs at setup).

        Non-final, non-sample stages must give a positive `epochs`; only the final
        stage may omit it (taking the remainder). The explicit epochs must sum to
        ``<= max_epochs``, and an omitted final stage needs a remainder of ``>= 1``
        — over-allocation or a zero-length remainder is a hard `ConfigError`.
        A non-positive/`None` `max_epochs` (infinite training) skips the check.

        Raises
        ------
        ConfigError
            A non-final stage omits `epochs`, the explicit epochs over-allocate,
            or an omitted final stage is left a zero/negative remainder.
        """
        if max_epochs is None or max_epochs < 0:
            if self.is_multi_stage:
                raise ConfigError(
                    "training_schedule declares multiple epoch-delimited stages but "
                    f"trainer.max_epochs is {max_epochs} — staged training needs a finite "
                    "positive max_epochs to allocate per-stage epochs/steps."
                )
            return
        *non_final, final = self.stages
        for stage in non_final:
            if stage.epochs is None and not stage.is_sample_stage:
                raise ConfigError(
                    f"training_schedule stage {stage.name!r} omits 'epochs' — only the final "
                    "stage may omit it (to take the remaining epochs); every earlier stage needs "
                    "an explicit positive 'epochs' or a 'num_samples'/'frac_samples' target."
                )
        explicit_sum = sum(s.epochs for s in self.stages if s.epochs is not None)
        if final.epochs is None:
            remainder = max_epochs - explicit_sum
            if remainder < 1:
                raise ConfigError(
                    f"training_schedule over-allocates epochs: the explicit stages sum to "
                    f"{explicit_sum} but trainer.max_epochs is {max_epochs}, leaving no epochs "
                    f"for the final stage {final.name!r} (needs >= 1). Reduce stage epochs or "
                    "raise trainer.max_epochs."
                )
        elif explicit_sum > max_epochs:
            raise ConfigError(
                f"training_schedule over-allocates epochs: the stages sum to {explicit_sum} "
                f"but trainer.max_epochs is {max_epochs}."
            )

    @classmethod
    def from_config(cls, raw: Mapping[str, Any], module_names: Sequence[str]) -> TrainingSchedule:
        """Parse + validate a ``training_schedule`` config block (fail-loud).

        Expects ``{"stages": {name: {epochs, frozen|trainable, optimizer, lrs,
        order}, ...}}``; a ``name: null`` stage is dropped (deep-merge deletion).
        Validates: non-empty schedule, known per-stage keys, `frozen` XOR
        `trainable`, freeze names ⊆ `module_names`, integer `epochs`/`order`, and
        no duplicate explicit `order`. Execution order is declaration order,
        stably overridden by explicit `order`. A sample stage (one budget key only)
        may not be final; the optional top-level `stop_after` names a non-final stage.

        Returns
        -------
        TrainingSchedule
            The parsed, validated schedule with stages in execution order.

        Raises
        ------
        ConfigError
            On any structural or naming violation above.
        """
        if not isinstance(raw, Mapping) or "stages" not in raw:
            raise ConfigError(
                "training_schedule must be a mapping with a 'stages' key "
                "(a dict of stage-name -> stage config)."
            )
        if extra := set(raw) - {"stages", "stop_after"}:
            raise ConfigError(
                f"training_schedule has unknown top-level key(s) {sorted(extra)} — only "
                "'stages' and 'stop_after' are supported."
            )
        raw_stages = raw["stages"]
        if not isinstance(raw_stages, Mapping):
            raise ConfigError("training_schedule.stages must be a mapping of stage-name -> config.")
        # a `stage: null` override deletes that stage (deep-merge deletion idiom).
        stage_items = [(name, cfg) for name, cfg in raw_stages.items() if cfg is not None]
        if not stage_items:
            raise ConfigError("training_schedule.stages is empty — declare at least one stage.")

        known = set(module_names)
        stages = [_parse_stage(name, cfg, known) for name, cfg in stage_items]
        _check_orders(stages)
        ordered = _order_stages(stages)
        if ordered[-1].is_sample_stage:
            raise ConfigError(
                f"training_schedule stage {ordered[-1].name!r} names a sample target on the "
                "final stage — the final stage runs to trainer.max_epochs; give it 'epochs'."
            )
        stop_after = raw.get("stop_after")
        if stop_after is not None:
            names, problem = [s.name for s in ordered], ""
            if not isinstance(stop_after, str) or not stop_after.strip():
                problem = "must be a non-empty string naming a stage"
            elif stop_after not in names:
                problem = f"names unknown stage — declared stages are {names}"
            elif stop_after == names[-1]:
                problem = "names the final stage — the fit ends there anyway; remove stop_after"
            if problem:
                raise ConfigError(f"training_schedule.stop_after {problem} (got {stop_after!r}).")
        return cls(ordered, module_names, stop_after=stop_after)

    @classmethod
    def desugar_legacy(cls, module_names: Sequence[str]) -> TrainingSchedule:
        """The single-`fit`-stage schedule a plain (no `training_schedule`) config
        desugars to: one stage, no freeze, no per-stage optimizer/LR override —
        so `configure_optimizers` falls back to the top-level
        ``lrs:``/``optimizer:``.
        """
        return cls([StageConfig(name="fit")], module_names)


def _parse_stage(name: str, cfg: Any, module_names: set[str]) -> StageConfig:
    """Parse + validate a single stage config into a `StageConfig`."""
    if not isinstance(cfg, Mapping):
        raise ConfigError(
            f"training_schedule stage {name!r} must be a mapping of stage fields "
            f"(got {type(cfg).__name__})."
        )
    if unknown := set(cfg) - _STAGE_FIELDS:
        raise ConfigError(
            f"training_schedule stage {name!r} has unknown key(s) {sorted(unknown)} — "
            f"valid keys are {sorted(_STAGE_FIELDS)}."
        )
    frozen, trainable = cfg.get("frozen"), cfg.get("trainable")
    if frozen is not None and trainable is not None:
        raise ConfigError(
            f"training_schedule stage {name!r} sets BOTH 'frozen' and 'trainable' — give at "
            "most one (the other is its complement over model.modules)."
        )
    frozen_t = _validate_names(name, "frozen", frozen, module_names)
    trainable_t = _validate_names(name, "trainable", trainable, module_names)
    epochs = _as_positive_int(name, "epochs", cfg.get("epochs"))
    num_samples = _as_positive_int(name, "num_samples", cfg.get("num_samples"))
    frac_samples = _as_fraction(name, "frac_samples", cfg.get("frac_samples"))
    budgets = {"epochs": epochs, "num_samples": num_samples, "frac_samples": frac_samples}
    if len(given := [repr(k) for k, v in budgets.items() if v is not None]) > 1:
        raise ConfigError(
            f"training_schedule stage {name!r} sets BOTH {' and '.join(given)} — give at most "
            "one stage budget (an epoch count or a single sample target)."
        )
    order = _as_int(name, "order", cfg.get("order"))
    lrs = cfg.get("lrs")
    if lrs is not None and not isinstance(lrs, Mapping):
        raise ConfigError(f"training_schedule stage {name!r} 'lrs' must be a mapping.")
    lr_scheduler = _parse_lr_scheduler(name, cfg)
    if lr_scheduler is not None and lrs is not None and (clash := _ONECYCLE_ONLY_LRS & set(lrs)):
        raise ConfigError(
            f"training_schedule stage {name!r} sets both 'lr_scheduler' and OneCycle-only "
            f"'lrs' key(s) {sorted(clash)} — those keys only apply to the default OneCycleLR. "
            "With a custom lr_scheduler keep only 'initial'/'weight_decay' in 'lrs'."
        )
    skip = cfg.get("skip_losses", ())
    bad = isinstance(skip, str) or not isinstance(skip, Sequence)
    if bad or not all(isinstance(k, str) for k in skip):
        raise ConfigError(
            f"training_schedule stage {name!r} 'skip_losses' must be a list of task names "
            f"(got {type(skip).__name__})."
        )
    return StageConfig(
        name=name,
        epochs=epochs,
        num_samples=num_samples,
        frac_samples=frac_samples,
        frozen=frozen_t,
        trainable=trainable_t,
        optimizer=cfg.get("optimizer"),
        lrs=lrs,
        order=order,
        early_stop=_parse_early_stop(name, cfg),
        callbacks=_parse_stage_callbacks(name, cfg),
        lr_scheduler=lr_scheduler,
        skip_losses=tuple(skip),
    )


def _parse_lr_scheduler(stage: str, cfg: Mapping[str, Any]) -> LRSchedulerConfig | None:
    """Parse + structurally validate a stage's optional `lr_scheduler` block into an
    `LRSchedulerConfig` (fail-loud, import-free); ``None`` when the stage declares
    none. `class_path` is a required non-empty string; `init_args` (if present) a
    mapping that must NOT set `optimizer` (injected at the boundary); `interval` is
    ``epoch``/``step``; `frequency` a positive int; `monitor` a non-empty string.
    Import + the metric-driven-⇒-monitor rule are enforced at fit start.
    """
    raw = cfg.get("lr_scheduler")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler' must be a mapping "
            f"(got {type(raw).__name__})."
        )
    if unknown := set(raw) - _LR_SCHEDULER_FIELDS:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler' has unknown key(s) "
            f"{sorted(unknown)} — valid keys are {sorted(_LR_SCHEDULER_FIELDS)}."
        )
    class_path = raw.get("class_path")
    if not isinstance(class_path, str) or not class_path.strip():
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler.class_path' is required and must "
            "be a non-empty string (a dotted torch.optim.lr_scheduler class)."
        )
    init_args = raw.get("init_args")
    if init_args is not None and not isinstance(init_args, Mapping):
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler.init_args' must be a mapping "
            f"(got {type(init_args).__name__})."
        )
    if isinstance(init_args, Mapping) and "optimizer" in init_args:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler.init_args' must not set 'optimizer' "
            "— the freshly-rebuilt stage optimizer is injected automatically."
        )
    interval = raw.get("interval", "epoch")
    if interval not in {"epoch", "step"}:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler.interval' must be 'epoch' or 'step' "
            f"(got {interval!r})."
        )
    frequency = raw.get("frequency", 1)
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 1:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler.frequency' must be a positive "
            f"integer (got {frequency!r})."
        )
    monitor = raw.get("monitor")
    if monitor is not None and (not isinstance(monitor, str) or not monitor.strip()):
        raise ConfigError(
            f"training_schedule stage {stage!r} 'lr_scheduler.monitor' must be a non-empty string."
        )
    return LRSchedulerConfig(
        class_path=class_path,
        init_args=dict(init_args) if isinstance(init_args, Mapping) else None,
        interval=interval,
        frequency=frequency,
        monitor=monitor,
    )


def _parse_stage_callbacks(
    stage: str, cfg: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...] | None:
    """Parse + structurally validate a stage's optional scoped `callbacks` list
    into a tuple of ``{class_path[, init_args]}`` specs (fail-loud); ``None`` when
    the stage declares none. Each spec must be a mapping with a string
    ``class_path`` and, if present, a mapping ``init_args``. Import/instantiation
    validation of the class is deferred to fit start (`StageScopedCallbacks.setup`),
    so a bad path fails before training rather than at the stage boundary.
    """
    raw = cfg.get("callbacks")
    if raw is None:
        return None
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
        raise ConfigError(
            f"training_schedule stage {stage!r} 'callbacks' must be a list of "
            f"class_path/init_args specs (got {type(raw).__name__})."
        )
    specs: list[Mapping[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ConfigError(
                f"training_schedule stage {stage!r} callbacks[{i}] must be a mapping with a "
                f"'class_path' (got {type(item).__name__})."
            )
        class_path = item.get("class_path")
        if not isinstance(class_path, str) or not class_path.strip():
            raise ConfigError(
                f"training_schedule stage {stage!r} callbacks[{i}] needs a non-empty string "
                "'class_path' (a dotted module.Class)."
            )
        if unknown := set(item) - {"class_path", "init_args"}:
            raise ConfigError(
                f"training_schedule stage {stage!r} callbacks[{i}] has unknown key(s) "
                f"{sorted(unknown)} — only 'class_path' and 'init_args' are allowed."
            )
        init_args = item.get("init_args")
        if init_args is not None and not isinstance(init_args, Mapping):
            raise ConfigError(
                f"training_schedule stage {stage!r} callbacks[{i}] 'init_args' must be a mapping "
                f"(got {type(init_args).__name__})."
            )
        specs.append(dict(item))
    return tuple(specs)


def _parse_early_stop(stage: str, cfg: Mapping[str, Any]) -> EarlyStopConfig | None:
    """Parse + validate a stage's optional `early_stop` block into an
    `EarlyStopConfig` (fail-loud); ``None`` when the stage declares none. `monitor`
    is required and non-empty; `mode` is `min`/`max`; `patience` a positive int;
    `min_delta` a non-negative number; `check_finite` a bool.
    """
    raw = cfg.get("early_stop")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"training_schedule stage {stage!r} 'early_stop' must be a mapping "
            f"(got {type(raw).__name__})."
        )
    if unknown := set(raw) - _EARLY_STOP_FIELDS:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'early_stop' has unknown key(s) {sorted(unknown)} "
            f"— valid keys are {sorted(_EARLY_STOP_FIELDS)}."
        )
    monitor = raw.get("monitor")
    if not isinstance(monitor, str) or not monitor.strip():
        raise ConfigError(
            f"training_schedule stage {stage!r} 'early_stop.monitor' is required and must be a "
            "non-empty string (a trainer.callback_metrics key, e.g. 'val/loss')."
        )
    mode = raw.get("mode", "min")
    if mode not in {"min", "max"}:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'early_stop.mode' must be 'min' or 'max' "
            f"(got {mode!r})."
        )
    patience = raw.get("patience", 3)
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'early_stop.patience' must be a positive integer "
            f"(got {patience!r})."
        )
    min_delta = raw.get("min_delta", 0.0)
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or min_delta < 0:
        raise ConfigError(
            f"training_schedule stage {stage!r} 'early_stop.min_delta' must be a non-negative "
            f"number (got {min_delta!r})."
        )
    check_finite = raw.get("check_finite", True)
    if not isinstance(check_finite, bool):
        raise ConfigError(
            f"training_schedule stage {stage!r} 'early_stop.check_finite' must be a boolean "
            f"(got {check_finite!r})."
        )
    return EarlyStopConfig(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=float(min_delta),
        check_finite=check_finite,
    )


def _validate_names(
    stage: str, field: str, value: Any, module_names: set[str]
) -> tuple[str, ...] | None:
    """Coerce a freeze name list to a tuple, rejecting non-list values and names
    that are not `model.modules` keys.
    """
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' must be a list of module names "
            f"(got {type(value).__name__})."
        )
    names = tuple(value)
    if unknown := [n for n in names if n not in module_names]:
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' names unknown module(s) {unknown} — "
            f"they must be keys of model.modules ({sorted(module_names)})."
        )
    return names


def _as_positive_int(stage: str, field: str, value: Any) -> int | None:
    """Validate an optional positive-integer field."""
    result = _as_int(stage, field, value)
    if result is not None and result < 1:
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' must be a positive integer "
            f"(got {result})."
        )
    return result


def _as_fraction(stage: str, field: str, value: Any) -> float | None:
    """Validate an optional fraction strictly inside ``(0, 1)``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' must be a number (got {value!r})."
        )
    if not 0 < value < 1:
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' must satisfy 0 < {field} < 1 "
            f"(got {value!r}); for a fraction >= 1 use 'epochs' instead."
        )
    return float(value)


def _as_int(stage: str, field: str, value: Any) -> int | None:
    """Validate an optional integer field (rejects bool and non-int)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' must be an integer (got {value!r})."
        )
    return value


def _check_orders(stages: list[StageConfig]) -> None:
    """Reject duplicate explicit `order` values."""
    explicit = [s.order for s in stages if s.order is not None]
    if len(set(explicit)) != len(explicit):
        raise ConfigError(
            "training_schedule has duplicate explicit 'order' values — each pinned stage order "
            f"must be unique (got {sorted(explicit)})."
        )


def _order_stages(stages: list[StageConfig]) -> list[StageConfig]:
    """Order stages by explicit `order`, falling back to declaration index; the
    stable sort keeps declaration order among unpinned stages.
    """
    keyed = sorted(enumerate(stages), key=lambda p: p[1].order if p[1].order is not None else p[0])
    return [stage for _, stage in keyed]


# Reducer-safe freeze: a module frozen via requires_grad=False BEFORE DDP wraps
# is permanently excluded from the reducer's managed set — a later unfreeze
# leaves its grads rank-local and ranks silently desync. Reducer-safe mode keeps
# requires_grad=True on every schedule-managed param; "frozen" is enforced by
# optimizer exclusion + eval() + per-step grad clearing (clear_frozen_grads).
# Cost: wasted backward compute/comms for frozen params — accepted. The
# predicate below is the SINGLE condition for both this mode and the
# find_unused_parameters auto-enable; keep them in lockstep. Off DDP / static
# freeze, the requires_grad freeze is kept (optimal, bitwise-parity).


def reducer_safe_freeze_required(strategy: Any, schedule: TrainingSchedule | None) -> bool:
    """Whether reducer-safe freeze semantics are needed for this ``(strategy,
    schedule)`` pair: a DDP-family strategy (whose reducer fixes its managed-param
    set at wrap) combined with a schedule that changes the frozen set across
    stages (so a wrap-time-frozen module may later be unfrozen). Off a DDP-family
    strategy, or for a static freeze set, returns ``False``. True iff
    ``strategy`` exposes ``_ddp_kwargs`` (DDP-family) and the schedule changes
    its frozen set across stages.
    """
    if schedule is None or not schedule.changes_freeze_across_stages():
        return False
    return getattr(strategy, "_ddp_kwargs", None) is not None


def apply_stage_freeze(
    net: Any, frozen: set[str], previously_frozen: set[str], *, reducer_safe: bool
) -> None:
    """Apply a stage's freeze mask to ``net`` (a name→module mapping) as a DELTA
    against ``previously_frozen``: modules entering the frozen set are put in
    ``eval()`` (stops dropout + running-stat updates); modules leaving it are
    restored to ``train()``.

    In the default (single-device / static-freeze) mode this ALSO toggles
    ``requires_grad`` (``False`` on freeze, ``True`` on unfreeze) — the optimizer
    then excludes frozen params by ``requires_grad``. In ``reducer_safe`` mode
    ``requires_grad`` is left untouched (every managed param stays ``True`` so the
    DDP reducer keeps managing it across flips); the freeze is enforced by the
    optimizer-membership filter + grad clearing instead.
    """
    # newly frozen first (train=False, i.e. eval()), then newly unfrozen
    for names, train in ((frozen - previously_frozen, False), (previously_frozen - frozen, True)):
        for name in names:
            if not reducer_safe:
                net[name].requires_grad_(train)
            net[name].train(train)


def trainable_named_params(
    named_params: Iterable[tuple[str, Any]], frozen: set[str]
) -> list[tuple[str, Any]]:
    """The ``(name, param)`` pairs the optimizer should own: those with
    ``requires_grad`` whose name is NOT under a frozen module's ``net.<name>.``
    prefix. Keying off the frozen SET (not just ``requires_grad``) is what excludes
    a reducer-safe frozen module — whose params keep ``requires_grad=True`` — from
    the optimizer (and hence from weight decay). In the default freeze mode frozen
    params already have ``requires_grad=False``, so the prefix filter selects the
    identical set (bitwise-parity preserved).

    Returns
    -------
    list[tuple[str, Any]]
        Optimizer-eligible ``(name, param)`` pairs, input order preserved.
    """
    prefixes = tuple(f"net.{name}." for name in frozen)
    return [
        (n, p)
        for n, p in named_params
        if p.requires_grad and (not prefixes or not n.startswith(prefixes))
    ]


def clear_frozen_grads(net: Any, frozen: set[str]) -> None:
    """Clear (`.grad = None`) every parameter of the frozen modules in ``net``.

    Reducer-safe mode only: frozen params keep ``requires_grad=True`` and so
    accumulate a gradient each backward (all-reduced but never stepped). Clearing
    them after the reducer has finished (``on_after_backward``) stops a frozen
    stage's gradients from surviving — and being consumed — at the first optimizer
    step after the module is unfrozen.
    """
    for name in frozen:
        for param in net[name].parameters():
            param.grad = None


# Per-stage early stopping: the tracker owns the monitor/patience/min_delta
# arithmetic; all schedule state lives on the SaltModule (callback stateless).
# Boundaries are data-dependent once a stage can early-stop, so boundary_record
# captures each transition for the checkpoint — resume reconstructs position
# from records + the persisted tracker, not epoch arithmetic.


_TRACKER_COUNTERS = ("best_score", "wait_count", "check_count")  # checkpointed, in this order


@dataclass
class EarlyStopTracker:
    """Live early-stop counters for the ACTIVE stage: checkpointed for exact
    mid-stage resume, reset at each stage entry. Held on the `TrainingController`
    (`salt.model.multistage_training`); the callback drives it but stays stateless.
    """

    config: EarlyStopConfig
    best_score: float | None = None
    wait_count: int = 0
    check_count: int = 0

    def check(self, value: float) -> bool:
        """Fold one validation-check `value` into the counters and report whether
        the stage should now early-stop: patience exhausted (no improvement of at
        least `min_delta` for `patience` consecutive checks), or a non-finite value
        under `check_finite`. The first check seeds `best_score` and never stops.

        Returns
        -------
        bool
            ``True`` iff the stage's early-stop criterion is now met.
        """
        import math

        self.check_count += 1
        if self.config.check_finite and not math.isfinite(value):
            return True
        if self.best_score is None or self._improved(value):
            self.best_score = value
            self.wait_count = 0
            return False
        self.wait_count += 1
        return self.wait_count >= self.config.patience

    def _improved(self, value: float) -> bool:
        """Whether `value` improves on `best_score` by at least `min_delta` under
        the configured `mode` (`min`: lower is better; `max`: higher is better).
        """
        assert self.best_score is not None
        if self.config.mode == "min":
            return value < self.best_score - self.config.min_delta
        return value > self.best_score + self.config.min_delta

    def state_dict(self) -> dict[str, Any]:
        """The checkpoint payload for mid-stage resume. The fingerprint lets
        `TrainingController.restore_schedule_stage` reject a changed `early_stop`.

        Returns
        -------
        dict[str, Any]
            ``{best_score, wait_count, check_count, fingerprint}``.
        """
        state = {key: getattr(self, key) for key in _TRACKER_COUNTERS}
        return {**state, "fingerprint": self.config.fingerprint()}

    @classmethod
    def from_state_dict(cls, config: EarlyStopConfig, state: Mapping[str, Any]) -> EarlyStopTracker:
        """Rebuild a tracker from a checkpointed `state_dict` under `config`.

        Returns
        -------
        EarlyStopTracker
            A tracker with the restored counters.
        """
        return cls(config, **{key: state[key] for key in _TRACKER_COUNTERS})


def boundary_record(
    stage_name: str, stage_index: int, global_step: int, epoch: int, reason: str
) -> dict[str, Any]:
    """A completed stage-transition record appended to the checkpoint at each
    boundary. `reason` is ``"epochs"`` (epoch cap), ``"early_stop"`` (criterion),
    ``"steps"`` (sample-stage step target) or ``"stop_after"`` (prefix run ended); the
    records let a resume reconstruct the stage position rather than epoch arithmetic.

    Returns
    -------
    dict[str, Any]
        The serialisable boundary record.
    """
    return {
        "stage_name": stage_name,
        "stage_index": stage_index,
        "global_step": global_step,
        "epoch": epoch,
        "reason": reason,
    }
