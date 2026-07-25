"""Training-schedule schema: named stages with per-stage freeze + optimizer/LR.

Parsed and validated fail-loud at `SaltModule.__init__`. It is the single
canonical home for optimizer/LR config (plan D1/D2): a plain (no
`training_schedule`) config desugars to a single `fit` stage (see
`desugar_legacy`), so `SaltModule.configure_optimizers` has exactly one code
path. This module also owns the per-stage epoch→step allocation math and the
stage-boundary lookup the `TrainingScheduleCallback` drives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from salt.graph.errors import ConfigError

__all__ = [
    "EarlyStopConfig",
    "EarlyStopTracker",
    "StageConfig",
    "TrainingSchedule",
    "apply_stage_freeze",
    "boundary_record",
    "clear_frozen_grads",
    "reducer_safe_freeze_required",
    "trainable_named_params",
]

# recognised per-stage keys — anything else is a config typo, rejected fail-loud.
_STAGE_FIELDS = frozenset(
    {"epochs", "frozen", "trainable", "optimizer", "lrs", "order", "early_stop", "callbacks"}
)
# recognised `early_stop` sub-keys (Lightning EarlyStopping vocabulary; plan 12 W7).
_EARLY_STOP_FIELDS = frozenset({"monitor", "mode", "patience", "min_delta", "check_finite"})


@dataclass(frozen=True)
class EarlyStopConfig:
    """A stage's early-stopping criterion (plan 12 W7 / D-ES). Mirrors Lightning
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
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "patience": self.patience,
        }


@dataclass(frozen=True)
class StageConfig:
    """One training stage: its epoch budget, freeze spec, and optional
    optimizer/LR overrides. Exactly one of `frozen`/`trainable` may be set
    (default `frozen=()` — everything trainable). `epochs=None` means "take the
    remaining epochs" (only the final stage may omit it). `order` pins execution
    position; otherwise declaration order is used. `early_stop` optionally ends the
    stage before its epoch cap when a monitored metric stops improving (plan 12 W7).
    """

    name: str
    epochs: int | None = None
    frozen: tuple[str, ...] | None = None
    trainable: tuple[str, ...] | None = None
    optimizer: str | None = None
    lrs: Mapping[str, float] | None = None
    order: int | None = None
    early_stop: EarlyStopConfig | None = None
    callbacks: tuple[Mapping[str, Any], ...] | None = None


class TrainingSchedule:
    """Parsed, validated training schedule — an ordered list of `StageConfig`.

    Names in `frozen`/`trainable` must be `model.modules` keys (the `net.<name>.*`
    state-dict prefixes); a freeze spec resolves to the set of module names whose
    parameters are held fixed for that stage (see `frozen_names`).
    """

    def __init__(self, stages: list[StageConfig], module_names: Sequence[str]) -> None:
        self.stages = stages
        self._module_names = tuple(module_names)

    @property
    def initial_stage(self) -> StageConfig:
        """The first stage in execution order."""
        return self.stages[0]

    @property
    def is_multi_stage(self) -> bool:
        """Whether the schedule declares more than one stage."""
        return len(self.stages) > 1

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
        that gates every W7 early-stop code path. When ``False``, boundaries are
        pure epoch arithmetic and no early-stop checkpoint state is written, so
        behaviour is bitwise-identical to the pre-W7 tip (the G7a parity guard).
        """
        return any(stage.early_stop is not None for stage in self.stages)

    @property
    def has_stage_callbacks(self) -> bool:
        """Whether any stage declares scoped `callbacks` — the switch that injects
        the `StageScopedCallbacks` coordinator (plan 12 W7). When ``False`` no
        coordinator is added and callback handling is unchanged from the pre-W7 tip.
        """
        return any(stage.callbacks is not None for stage in self.stages)

    def changes_freeze_across_stages(self) -> bool:
        """Whether the frozen set differs between any two consecutive stages —
        the condition under which DDP needs ``find_unused_parameters=True``
        (freeze/unfreeze flips break the reducer's fixed bucketing otherwise).
        """  # noqa: DOC201
        masks = [frozenset(self.frozen_names(stage)) for stage in self.stages]
        return any(a != b for a, b in zip(masks, masks[1:], strict=False))

    def frozen_names(self, stage: StageConfig) -> set[str]:
        """Resolve `stage`'s freeze spec to the set of frozen module names:
        `frozen` freezes those modules; `trainable` freezes their complement over
        `model.modules`; neither freezes nothing.
        """  # noqa: DOC201
        if stage.trainable is not None:
            return set(self._module_names) - set(stage.trainable)
        return set(stage.frozen or ())

    def _non_final_epoch_bounds(self, max_epochs: int) -> list[int]:
        """Cumulative epoch index at which each *non-final* stage ends (its
        explicit `epochs` summed left-to-right). The final stage owns everything
        from the last bound to `max_epochs`, so it is not represented here.
        """
        bounds, acc = [], 0
        for stage in self.stages[:-1]:
            assert stage.epochs is not None  # validated by validate_epochs
            acc += stage.epochs
            bounds.append(acc)
        return bounds

    def stage_index_for_epoch(self, epoch: int, max_epochs: int) -> int:
        """The index of the stage that owns `epoch` (0-based). Non-final stages
        own ``[bound_{i-1}, bound_i)``; the final stage owns everything from the
        last bound onward (so any epochs past the explicit budgets run there).
        """  # noqa: DOC201
        for index, bound in enumerate(self._non_final_epoch_bounds(max_epochs)):
            if epoch < bound:
                return index
        return len(self.stages) - 1

    def stage_step_allocations(self, total_steps: int, max_epochs: int) -> list[int]:
        """Split `total_steps` (Lightning's whole-run `estimated_stepping_batches`)
        across the stages proportionally to their epoch budgets, so each stage's
        `OneCycleLR.total_steps` spans only that stage (Gotcha #1). Boundaries are
        rounded at each non-final stage end and the final stage takes the exact
        remainder, so the allocations always sum to `total_steps`. A single-stage
        schedule returns ``[total_steps]`` unchanged (bitwise parity path).
        """  # noqa: DOC201
        if not self.is_multi_stage:
            return [total_steps]
        allocations, prev = [], 0
        for bound in self._non_final_epoch_bounds(max_epochs):
            boundary_step = round(total_steps * bound / max_epochs)
            allocations.append(boundary_step - prev)
            prev = boundary_step
        allocations.append(total_steps - prev)  # final stage: exact remainder
        return allocations

    def validate_epochs(self, max_epochs: int | None) -> None:
        """Validate the stage epoch allocation against `trainer.max_epochs`
        (available only once the trainer is attached, so this runs at setup).

        Non-final stages must give an explicit positive `epochs`; only the final
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
                    "positive max_epochs to allocate per-stage epochs/steps (plan D1)."
                )
            return
        *non_final, final = self.stages
        for stage in non_final:
            if stage.epochs is None:
                raise ConfigError(
                    f"training_schedule stage {stage.name!r} omits 'epochs' — only the final "
                    "stage may omit it (to take the remaining epochs); every earlier stage needs "
                    "an explicit positive 'epochs' (plan D1)."
                )
        explicit_sum = sum(s.epochs for s in self.stages if s.epochs is not None)
        if final.epochs is None:
            remainder = max_epochs - explicit_sum
            if remainder < 1:
                raise ConfigError(
                    f"training_schedule over-allocates epochs: the explicit stages sum to "
                    f"{explicit_sum} but trainer.max_epochs is {max_epochs}, leaving no epochs "
                    f"for the final stage {final.name!r} (needs >= 1). Reduce stage epochs or "
                    "raise trainer.max_epochs (plan D1)."
                )
        elif explicit_sum > max_epochs:
            raise ConfigError(
                f"training_schedule over-allocates epochs: the stages sum to {explicit_sum} "
                f"but trainer.max_epochs is {max_epochs} (plan D1)."
            )

    @classmethod
    def from_config(
        cls, raw: Mapping[str, Any], module_names: Sequence[str]
    ) -> TrainingSchedule:
        """Parse + validate a ``training_schedule`` config block (fail-loud).

        Expects ``{"stages": {name: {epochs, frozen|trainable, optimizer, lrs,
        order}, ...}}``; a ``name: null`` stage is dropped (deep-merge deletion).
        Validates: non-empty schedule, known per-stage keys, `frozen` XOR
        `trainable`, freeze names ⊆ `module_names`, integer `epochs`/`order`, and
        no duplicate explicit `order`. Execution order is declaration order,
        stably overridden by explicit `order`.

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
                "(a dict of stage-name -> stage config; plan D1)."
            )
        if extra := set(raw) - {"stages"}:
            raise ConfigError(
                f"training_schedule has unknown top-level key(s) {sorted(extra)} — only "
                "'stages' is supported (plan D1)."
            )
        raw_stages = raw["stages"]
        if not isinstance(raw_stages, Mapping):
            raise ConfigError("training_schedule.stages must be a mapping of stage-name -> config.")
        # a `stage: null` override deletes that stage (deep-merge deletion idiom).
        stage_items = [(name, cfg) for name, cfg in raw_stages.items() if cfg is not None]
        if not stage_items:
            raise ConfigError("training_schedule.stages is empty — declare at least one stage.")

        known = set(module_names)
        stages: list[StageConfig] = []
        for name, cfg in stage_items:
            stages.append(_parse_stage(name, cfg, known))
        _check_orders(stages)
        ordered = _order_stages(stages)
        return cls(ordered, module_names)

    @classmethod
    def desugar_legacy(cls, module_names: Sequence[str]) -> TrainingSchedule:
        """The single-`fit`-stage schedule a plain (no `training_schedule`) config
        desugars to (plan D2): one stage, no freeze, no per-stage optimizer/LR
        override — so `configure_optimizers` falls back to the top-level
        ``lrs:``/``optimizer:`` and training is bitwise-identical to legacy code.
        """  # noqa: DOC201
        return cls([StageConfig(name="fit")], module_names)


def _parse_stage(name: str, cfg: Any, module_names: set[str]) -> StageConfig:
    """Parse + validate a single stage config into a `StageConfig`."""  # noqa: DOC201, DOC501
    if not isinstance(cfg, Mapping):
        raise ConfigError(
            f"training_schedule stage {name!r} must be a mapping of stage fields "
            f"(got {type(cfg).__name__})."
        )
    if unknown := set(cfg) - _STAGE_FIELDS:
        raise ConfigError(
            f"training_schedule stage {name!r} has unknown key(s) {sorted(unknown)} — "
            f"valid keys are {sorted(_STAGE_FIELDS)} (plan D1)."
        )
    frozen, trainable = cfg.get("frozen"), cfg.get("trainable")
    if frozen is not None and trainable is not None:
        raise ConfigError(
            f"training_schedule stage {name!r} sets BOTH 'frozen' and 'trainable' — give at "
            "most one (the other is its complement over model.modules; plan D1)."
        )
    frozen_t = _validate_names(name, "frozen", frozen, module_names)
    trainable_t = _validate_names(name, "trainable", trainable, module_names)
    epochs = _as_positive_int(name, "epochs", cfg.get("epochs"))
    order = _as_int(name, "order", cfg.get("order"))
    lrs = cfg.get("lrs")
    if lrs is not None and not isinstance(lrs, Mapping):
        raise ConfigError(f"training_schedule stage {name!r} 'lrs' must be a mapping.")
    return StageConfig(
        name=name,
        epochs=epochs,
        frozen=frozen_t,
        trainable=trainable_t,
        optimizer=cfg.get("optimizer"),
        lrs=lrs,
        order=order,
        early_stop=_parse_early_stop(name, cfg),
        callbacks=_parse_stage_callbacks(name, cfg),
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
    """  # noqa: DOC201, DOC501
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
    """  # noqa: DOC201, DOC501
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
            f"— valid keys are {sorted(_EARLY_STOP_FIELDS)} (plan 12 W7)."
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
    """  # noqa: DOC201, DOC501
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
            f"they must be keys of model.modules ({sorted(module_names)}; plan D1)."
        )
    return names


def _as_positive_int(stage: str, field: str, value: Any) -> int | None:
    """Validate an optional positive-integer field."""  # noqa: DOC201, DOC501
    result = _as_int(stage, field, value)
    if result is not None and result < 1:
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' must be a positive integer "
            f"(got {result})."
        )
    return result


def _as_int(stage: str, field: str, value: Any) -> int | None:
    """Validate an optional integer field (rejects bool and non-int)."""  # noqa: DOC201, DOC501
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"training_schedule stage {stage!r} '{field}' must be an integer (got {value!r})."
        )
    return value


def _check_orders(stages: list[StageConfig]) -> None:
    """Reject duplicate explicit `order` values."""  # noqa: DOC501
    explicit = [s.order for s in stages if s.order is not None]
    if len(set(explicit)) != len(explicit):
        raise ConfigError(
            "training_schedule has duplicate explicit 'order' values — each pinned stage order "
            f"must be unique (got {sorted(explicit)}; plan D1)."
        )


def _order_stages(stages: list[StageConfig]) -> list[StageConfig]:
    """Order stages by explicit `order`, falling back to declaration index; the
    stable sort keeps declaration order among unpinned stages.
    """  # noqa: DOC201
    keyed = sorted(
        enumerate(stages),
        key=lambda item: item[1].order if item[1].order is not None else item[0],
    )
    return [stage for _, stage in keyed]


# ---------------------------------------------------------------------------
# Reducer-safe freeze semantics (plan 06 / W6).
#
# A module frozen in the INITIAL stage via ``requires_grad=False`` applied BEFORE
# DDP wraps is permanently excluded from the reducer's fixed managed-parameter
# set; a later unfreeze never registers a reducer hook, so its grads stay
# rank-local and ranks silently desync (W5 exp-06 G5d defect). The reducer-safe
# freeze mode fixes this by NEVER dropping ``requires_grad`` on a schedule-managed
# param that the schedule may later unfreeze under a distributed strategy — every
# managed param stays in the reducer at wrap. "Frozen" is then enforced by
# (a) exclusion from the optimizer (see ``trainable_named_params``), (b) ``.eval()``
# on the frozen module, (c) no optimizer step → bitwise-immobile even though grads
# are computed/all-reduced, and (d) explicit grad clearing each step (see
# ``clear_frozen_grads``) so a frozen stage's accumulated gradients cannot survive
# to the first post-unfreeze optimizer step. Cost: wasted backward compute + comms
# for frozen params on multi-GPU multi-stage runs — accepted, documented.
#
# The predicate below is the SINGLE condition governing both this mode and the
# ``find_unused_parameters`` auto-enable (they must agree). Off DDP or for a
# static freeze set the requires_grad-based freeze is kept (optimal, bitwise-parity).
# ---------------------------------------------------------------------------


def reducer_safe_freeze_required(strategy: Any, schedule: TrainingSchedule | None) -> bool:
    """Whether reducer-safe freeze semantics are needed for this ``(strategy,
    schedule)`` pair: a DDP-family strategy (whose reducer fixes its managed-param
    set at wrap) combined with a schedule that changes the frozen set across
    stages (so a wrap-time-frozen module may later be unfrozen). Off a DDP-family
    strategy, or for a static freeze set, returns ``False`` (the requires_grad
    freeze is safe and optimal there). This is also the exact condition under
    which ``find_unused_parameters`` is auto-enabled — keep them in lockstep.

    Returns
    -------
    bool
        ``True`` iff ``strategy`` exposes ``_ddp_kwargs`` (DDP-family) and
        ``schedule`` changes its frozen set across stages.
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
    for name in frozen - previously_frozen:  # newly frozen
        module = net[name]
        if not reducer_safe:
            module.requires_grad_(False)
        module.eval()
    for name in previously_frozen - frozen:  # newly unfrozen
        module = net[name]
        if not reducer_safe:
            module.requires_grad_(True)
        module.train()


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


# ---------------------------------------------------------------------------
# Per-stage early stopping (plan 12 / W7).
#
# The tracker below owns the monitor/patience/min_delta arithmetic for the ACTIVE
# stage so it is unit-testable in isolation and the `TrainingScheduleCallback`
# stays stateless (all schedule state lives on the `SaltModule`). Boundaries become
# data-dependent once a stage can early-stop, so `boundary_record` captures each
# completed transition for the checkpoint — resume reconstructs the stage position
# from records + the persisted tracker instead of epoch arithmetic.
# ---------------------------------------------------------------------------


class EarlyStopTracker:
    """Live early-stop counters for the ACTIVE stage (plan 12 W7). Mutable runtime
    state, checkpointed for exact mid-stage resume and reset at each stage entry.
    Held on the `SaltModule`; the callback drives it but stays stateless.
    """

    def __init__(
        self,
        config: EarlyStopConfig,
        *,
        best_score: float | None = None,
        wait_count: int = 0,
        check_count: int = 0,
    ) -> None:
        self.config = config
        self.best_score = best_score
        self.wait_count = wait_count
        self.check_count = check_count

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
        import math  # noqa: PLC0415

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
        """  # noqa: DOC201
        assert self.best_score is not None
        if self.config.mode == "min":
            return value < self.best_score - self.config.min_delta
        return value > self.best_score + self.config.min_delta

    def state_dict(self) -> dict[str, Any]:
        """The checkpoint payload for mid-stage resume: the counters plus the
        criterion fingerprint (which `_restore_schedule_stage` matches against the
        current config to reject a changed `early_stop`).

        Returns
        -------
        dict[str, Any]
            ``{best_score, wait_count, check_count, fingerprint}``.
        """
        return {
            "best_score": self.best_score,
            "wait_count": self.wait_count,
            "check_count": self.check_count,
            "fingerprint": self.config.fingerprint(),
        }

    @classmethod
    def from_state_dict(
        cls, config: EarlyStopConfig, state: Mapping[str, Any]
    ) -> EarlyStopTracker:
        """Rebuild a tracker from a checkpointed `state_dict` under `config`.

        Returns
        -------
        EarlyStopTracker
            A tracker with the restored counters.
        """
        return cls(
            config,
            best_score=state["best_score"],
            wait_count=state["wait_count"],
            check_count=state["check_count"],
        )


def boundary_record(
    stage_name: str, stage_index: int, global_step: int, epoch: int, reason: str
) -> dict[str, Any]:
    """A completed stage-transition record appended to the checkpoint at each
    boundary (plan 12 W7). `reason` is ``"epochs"`` (the stage hit its epoch cap)
    or ``"early_stop"`` (its criterion triggered); the records let a resume
    reconstruct the data-dependent stage position rather than epoch arithmetic.

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
