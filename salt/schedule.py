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
    "StageConfig",
    "TrainingSchedule",
    "apply_stage_freeze",
    "clear_frozen_grads",
    "reducer_safe_freeze_required",
    "trainable_named_params",
]

# recognised per-stage keys — anything else is a config typo, rejected fail-loud.
_STAGE_FIELDS = frozenset({"epochs", "frozen", "trainable", "optimizer", "lrs", "order"})


@dataclass(frozen=True)
class StageConfig:
    """One training stage: its epoch budget, freeze spec, and optional
    optimizer/LR overrides. Exactly one of `frozen`/`trainable` may be set
    (default `frozen=()` — everything trainable). `epochs=None` means "take the
    remaining epochs" (only the final stage may omit it). `order` pins execution
    position; otherwise declaration order is used.
    """

    name: str
    epochs: int | None = None
    frozen: tuple[str, ...] | None = None
    trainable: tuple[str, ...] | None = None
    optimizer: str | None = None
    lrs: Mapping[str, float] | None = None
    order: int | None = None


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
