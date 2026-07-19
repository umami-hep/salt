"""Training-schedule schema: named stages with per-stage freeze + optimizer/LR.

Parsed and validated fail-loud at `SaltModule.__init__`. W2 delivers the schema
surface, freeze resolution and the epoch-allocation check; multi-stage execution
(stage transitions + mid-fit optimizer rebuild) is W3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from salt.graph.errors import ConfigError

__all__ = ["StageConfig", "TrainingSchedule"]

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

    def frozen_names(self, stage: StageConfig) -> set[str]:
        """Resolve `stage`'s freeze spec to the set of frozen module names:
        `frozen` freezes those modules; `trainable` freezes their complement over
        `model.modules`; neither freezes nothing.
        """  # noqa: DOC201
        if stage.trainable is not None:
            return set(self._module_names) - set(stage.trainable)
        return set(stage.frozen or ())

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
