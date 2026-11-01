"""Plan execution for the salt v2 graph kernel.

Walks a compiled `Plan`'s steps as ``produced = module(bundle, mode)`` and
merges returns under write-once + declaration checks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, NoReturn, cast

import torch

from salt.graph.bundle import Bundle, _flatten_produced
from salt.graph.errors import (
    ConfigError,
    DeclarationError,
    MutationError,
    NonFiniteError,
    UndeclaredAccessError,
)
from salt.graph.planner import Plan, PlanStep
from salt.graph.spec import KEY_SEP, GraphModule, Mode, SinkModule, flatten_spec

__all__ = [
    "STEP_SCOPE_PREFIX",
    "Executor",
    "ReadTrackedBundle",
    "canonical_produced",
    "declared_reads",
    "record_steps",
]

STEP_SCOPE_PREFIX = "salt.step/"
"""Prefix of the profiler scope name wrapping each plan step (`record_steps`)."""


class _StepRecording:
    """Process-global toggle for the per-step profiler scopes (off by default)."""

    enabled = False


@contextmanager
def record_steps() -> Iterator[None]:
    """Wrap every plan step's module call in a ``torch.profiler.record_function``.

    Names each scope ``salt.step/<step name>``, so a `torch.profiler` capture
    attributes CPU and device time to the configured module instances
    (``encoder``, ``track_origin``, ``norm``, ...) instead of only to aten ops.

    The scope sits OUTSIDE the module call, so it is unaffected by
    ``--compile`` (salt compiles each graph module in place): no graph break
    is introduced and the compiled region is identical to an unprofiled run.
    Backward work is not covered — it runs outside the forward scopes and is
    read off the autograd-engine events instead.

    Re-entrant and restores the previous state on exit; the check is a single
    class-attribute read per step when off.

    Yields
    ------
    None
        For the duration of the ``with`` block.
    """
    previous = _StepRecording.enabled
    _StepRecording.enabled = True
    try:
        yield
    finally:
        _StepRecording.enabled = previous


def _is_sink(module: GraphModule) -> bool:
    """Whether a plan-step module is a terminal SINK (excluded from the forward loop).

    Duck-typed on the `SinkModule` Protocol marker ``is_sink() -> True``. A
    sink stays IN ``plan.steps`` (so it renders its own card and anchors
    demand) but is never invoked as a tensor forward — it produces no tensor
    and need not be callable.
    """
    is_sink = getattr(module, "is_sink", None)
    return isinstance(module, SinkModule) and callable(is_sink) and bool(is_sink())


class Executor:
    """Runs a compiled `Plan` over a `Bundle`.

    The plan is frozen: steps execute in plan order, each module is called
    as ``module(bundle, mode)`` (see the module docstring for the full call
    convention), and the returned keys are merged write-once with the
    declared key set enforced on every merge.
    """

    def __init__(self, plan: Plan) -> None:
        """Bind a plan to the live module instances frozen into its steps.

        Raises `ConfigError` if a non-sink plan module isn't callable.
        """
        self.plan = plan
        self._modules: dict[str, GraphModule] = {}
        self._allowed: dict[str, frozenset[str]] = {}
        self._scanned: dict[str, tuple[str, ...]] = {}
        # sink steps stay in `plan.steps` (render + demand) but are partitioned
        # OUT of the per-batch forward loop: a sink produces no tensor and is
        # never called.
        self._forward_steps: list[PlanStep] = []
        for step in plan.steps:
            module = step.module
            self._modules[step.name] = module
            self._allowed[step.name] = declared_reads(module, step, plan.mode)
            self._scanned[step.name] = _scanned_keys(step)
            if _is_sink(module):
                # a terminal sink: no tensor forward, no callable requirement —
                # its consume/flush lifecycle is driven by the Lightning bridge
                continue
            if not callable(module):
                raise ConfigError(
                    f"module {step.name!r} ({type(module).__name__}) is not callable — the "
                    "executor invokes modules as module(bundle, mode)"
                )
            self._forward_steps.append(step)

    def run(self, bundle: Bundle, debug: bool = False) -> Bundle:
        """Execute the plan's steps in order over `bundle` and return it.

        The caller provides every non-optional plan source leaf in
        `bundle`. With ``debug=True`` each module gets a `ReadTrackedBundle`:
        an access outside its declared requires raises
        `UndeclaredAccessError`; in-place mutation of an existing tensor
        raises `MutationError` (version-counter snapshots; declared reads
        that are inference tensors are compared by value; non-tensor leaves
        aren't covered); and a newly produced floating ``data``/``loss`` leaf
        with ``TensorSpec.finite`` holding NaN/Inf raises `NonFiniteError`
        naming the module. The returned-keys-vs-declaration check is always
        on; write-once collisions propagate from `Bundle.merge` as
        `KeyCollisionError`.

        Raises `KeyError` if the input bundle is missing a non-optional
        source leaf, `DeclarationError` on a malformed/mismatched return.
        """
        mode = self.plan.mode
        missing = sorted(
            k for k, s in self.plan.sources.items() if not s.optional and k not in bundle
        )
        if missing:
            raise KeyError(
                f"input bundle is missing source leaves required by the {mode.name} "
                f"plan: {missing} — the caller provides every non-optional source"
            )
        for step in self._forward_steps:
            module = cast("Callable[[Any, Mode], Any]", self._modules[step.name])
            allowed = self._allowed[step.name]
            view = ReadTrackedBundle(bundle, step.name, mode, allowed) if debug else None
            snapshot = _mutation_snapshot(bundle, allowed) if debug else None
            scope = f"{STEP_SCOPE_PREFIX}{step.name}"
            with torch.profiler.record_function(scope) if _StepRecording.enabled else nullcontext():
                produced = module(bundle if view is None else view, mode)
            if not isinstance(produced, dict):
                raise DeclarationError(
                    f"module {step.name!r} returned {type(produced).__name__} — modules return "
                    "their declared produces as a dict of newly produced keys"
                )
            if snapshot is not None and (mutated := _first_mutated(bundle, snapshot)) is not None:
                raise MutationError(
                    f"[mode={mode.name}] module {step.name!r} mutated bundle key "
                    f"{mutated!r} in place during debug execution — bundle leaves are read-only "
                    "for modules; clone before mutating (e.g. b.get(...).clone()) and return "
                    "new keys"
                )
            expected = set(step.produces)
            nested = canonical_produced(produced, expected, step.name)
            bundle.merge(nested, who=step.name, expected=expected, overwrite=set(step.rewrites))
            if view is not None:
                _check_finite(bundle, view, self._scanned[step.name])
        return bundle


_SCANNED_KINDS = frozenset({"data", "loss"})
"""Spec kinds the debug non-finite scan covers (``label``/``meta``/``pad_mask`` never)."""


def _scanned_keys(step: PlanStep) -> tuple[str, ...]:
    """The step's produced keys the debug non-finite scan covers (kind data/loss, finite)."""
    return tuple(k for k, s in step.produces.items() if s.kind in _SCANNED_KINDS and s.finite)


def _is_non_finite(value: Any) -> bool:
    """Whether `value` is a floating tensor holding NaN/Inf (ints, bools, non-tensors: never)."""
    return (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and not bool(torch.isfinite(value).all())
    )


def _check_finite(bundle: Bundle, view: ReadTrackedBundle, keys: tuple[str, ...]) -> None:
    """Raise `NonFiniteError` for the first of `keys` (just produced by ``view``'s module)
    holding NaN/Inf, naming the counts and the already non-finite inputs it read.
    """
    for key in keys:
        value = bundle.get(key)
        if not _is_non_finite(value):
            continue
        counts = (torch.isnan(value), torch.isposinf(value), torch.isneginf(value))
        n_nan, n_pinf, n_ninf = (int(c.sum()) for c in counts)
        bad = [k for k in sorted(view.reads) if _is_non_finite(bundle.get(k))]
        inputs = f"already non-finite: {bad}" if bad else "all finite"
        raise NonFiniteError(
            f"[mode={view.mode.name}] module {view.who!r} produced non-finite values in "
            f"{key!r} (NaN {n_nan}, +Inf {n_pinf}, -Inf {n_ninf} of {value.numel()}); inputs "
            f"it read were {inputs}"
        )


def _mutation_snapshot(bundle: Bundle, allowed: frozenset[str]) -> dict[str, int | torch.Tensor]:
    """Snapshot every tensor leaf before a step (debug only).

    Normal tensors: ``torch.Tensor._version``, torch's autograd in-place
    version counter — private but stable, and the cheapest torch-native
    mutation detector (integer reads, no data copies). Inference tensors
    (Lightning runs validation/test under ``inference_mode``) carry no
    version counter, so the ones the step may read are cloned and compared
    by value instead — a module can only reach a tensor it read.
    """
    out: dict[str, int | torch.Tensor] = {}
    for key in bundle.keys():  # noqa: SIM118 - Bundle is not a Mapping
        value = bundle.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if not value.is_inference():
            out[key] = value._version  # noqa: SLF001 - torch's in-place version counter
        elif key in allowed:
            out[key] = value.clone()
    return out


def _first_mutated(bundle: Bundle, snapshot: dict[str, int | torch.Tensor]) -> str | None:
    """Find the first pre-step tensor leaf whose version counter bumped or value changed."""
    for key, before in snapshot.items():
        after = bundle.get(key)
        if isinstance(before, int):
            if after._version != before:  # noqa: SLF001
                return key
        elif not _same_values(before, after):
            return key
    return None


def _same_values(before: torch.Tensor, after: torch.Tensor) -> bool:
    """Bitwise-style equality of a value snapshot, NaN positions compared as equal."""
    if before.shape != after.shape or before.dtype != after.dtype:
        return False
    if not before.is_floating_point():
        return torch.equal(before, after)
    return bool(((before == after) | (before.isnan() & after.isnan())).all())


def declared_reads(module: GraphModule, step: PlanStep, mode: Mode) -> frozenset[str]:
    """Compute the keys `module` may read in `mode` — the `ReadTrackedBundle` allowlist.

    The union of the step's bound requires and every mode-active declared
    require — including optional ports the planner dropped for lack of a
    producer, so a module probing them with ``in`` sees a normal absence
    instead of an `UndeclaredAccessError`. Shared by the model executor and
    the dataset plan runner.

    Parameters
    ----------
    module : GraphModule
        The step's module (its ``declare_io(mode)`` is consulted).
    step : PlanStep
        The compiled plan step (its bound ``requires``).
    mode : Mode
        The plan's primary mode.

    Returns
    -------
    frozenset[str]
        The dotted keys the module may read or probe.
    """
    io = module.declare_io(mode)
    declared = {key for key, spec in flatten_spec(io.requires).items() if spec.active_in(mode)}
    return frozenset(declared | set(step.requires))


class ReadTrackedBundle:
    """Read-tracking bundle view handed to a module on a guarded (debug) run.

    Duck-types `Bundle`'s read surface (``get``/``subtree``/``keys``/``in``/
    ``len``). Access outside the module's declared requires raises
    `UndeclaredAccessError`; mutation and raw payload access are blocked —
    modules return their produces, only the executor merges. Every leaf read
    through ``get``/``subtree`` is recorded in `reads` (the `NonFiniteError`
    input report). Used by ``Executor.run(debug=True)`` and by the dataset
    plan runner's guarded batches.

    Parameters
    ----------
    bundle : Bundle
        The run bundle to wrap.
    who : str
        The module instance name (named in every error).
    mode : Mode
        The running mode (named in every error).
    allowed : frozenset[str]
        The declared reads (`declared_reads`).
    """

    __slots__ = ("_allowed", "_bundle", "_mode", "_who", "reads")

    def __init__(self, bundle: Bundle, who: str, mode: Mode, allowed: frozenset[str]) -> None:
        """Wrap `bundle` for module `who`, allowing reads of `allowed` keys only."""
        self._bundle = bundle
        self._who = who
        self._mode = mode
        self._allowed = allowed
        self.reads: set[str] = set()

    @property
    def who(self) -> str:
        """The wrapped module's instance name."""
        return self._who

    @property
    def mode(self) -> Mode:
        """The running mode."""
        return self._mode

    @property
    def data(self) -> NoReturn:
        """Blocked: raw payload access bypasses read tracking; always raises
        `UndeclaredAccessError`.
        """
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} accessed Bundle.data directly under "
            "debug execution — raw payload access bypasses read tracking; read declared keys "
            "via get()/subtree()"
        )

    def get(self, key: str) -> Any:
        """Return the leaf value at a declared dotted key (see `Bundle.get`);
        raises `UndeclaredAccessError` if `key` is undeclared, else behaves
        exactly like the plain bundle (including `KeyError` if absent).
        """
        self._require_declared(key)
        value = self._bundle.get(key)
        self.reads.add(key)
        return value

    def subtree(self, prefix: str) -> dict[str, Any]:
        """Return the nested dict under `prefix`, checking every leaf is declared.

        Raises `UndeclaredAccessError` if nothing is declared under
        `prefix` or a leaf in the subtree is undeclared; an unknown or
        leaf-valued `prefix` raises `KeyError` like the plain bundle.
        """
        head = prefix + KEY_SEP
        if not any(k.startswith(head) for k in self._allowed):
            self._require_declared(prefix)
        out = self._bundle.subtree(prefix)
        leaves = [key for key in self._bundle.keys() if key.startswith(head)]  # noqa: SIM118
        for key in leaves:
            self._require_declared(key)
        self.reads.update(leaves)
        return out

    def keys(self) -> list[str]:
        """Return the declared dotted leaf keys currently present in the bundle."""
        present = self._bundle.keys()
        return [key for key in present if key in self._allowed]

    def __contains__(self, key: str) -> bool:
        """Probe a declared key's presence (the optional-port idiom).

        Probing an undeclared key raises `UndeclaredAccessError` —
        branching on an undeclared key is itself an undeclared dependence.
        """
        self._require_declared(key, verb="probed")
        return key in self._bundle

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"ReadTrackedBundle(module={self._who!r}, keys={self.keys()})"

    def set(self, key: str, value: Any) -> NoReturn:  # noqa: ARG002 - blocked surface
        """Blocked: modules return produced keys; only the executor writes;
        always raises `UndeclaredAccessError`.
        """
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} called set({key!r}, ...) on its "
            "bundle view — modules RETURN their produced keys; the executor merges them"
        )

    def merge(self, produced: dict[str, Any], who: str = "", expected: Any = None) -> NoReturn:  # noqa: ARG002
        """Blocked: merging is executor-only; always raises `UndeclaredAccessError`."""
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} called merge() on its bundle view "
            "— merging is executor-only; modules RETURN their produced keys"
        )

    def _require_declared(self, key: str, verb: str = "read") -> None:
        """Raise `UndeclaredAccessError` if `key` is not a declared read, naming the
        module, the key, and the declaration to amend.
        """
        if key in self._allowed:
            return
        declared = ", ".join(sorted(self._allowed)) or "<none>"
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} {verb} bundle key {key!r} which is "
            f"not in its declared requires — declared: {declared}. fix: amend "
            f"{self._who!r}.declare_io to require {key!r} (optional=True if consumed-if-"
            "present), or drop the access"
        )


# ---------------------------------------------------------------------------
# returned-keys canonicalisation (nested and flat dotted spellings)
# ---------------------------------------------------------------------------


def canonical_produced(produced: dict[str, Any], expected: set[str], who: str) -> dict[str, Any]:
    """Canonicalise a module's returned dict to nested single-component form.

    Modules may return nested dicts, flat dotted keys, or any mixture;
    `Bundle.merge` expects nested dicts with single-component keys. A dict
    value whose dotted path is in `expected` is a declared dict-valued leaf
    and is kept whole; empty undeclared dicts are kept as leaves so they
    surface as unexpected keys. Public because the dataset-side runner
    (`salt.data.dataset`) merges module returns under the same
    convention.

    Raises `DeclarationError` on malformed/duplicate keys (e.g. both
    spellings of one key), or keys that are simultaneously a leaf and a
    subtree.
    """
    flat = _flatten_produced(produced, expected, who, dotted=True)
    nested: dict[str, Any] = {}
    leaves: set[str] = set()
    for key, value in flat.items():
        parts = key.split(KEY_SEP)
        node = nested
        for depth in range(len(parts) - 1):
            path = KEY_SEP.join(parts[: depth + 1])
            if path in leaves:
                raise DeclarationError(
                    f"module {who!r} returned overlapping keys: {key!r} descends through the "
                    f"returned leaf {path!r} (write-once)"
                )
            node = node.setdefault(parts[depth], {})
        if parts[-1] in node:
            raise DeclarationError(
                f"module {who!r} returned overlapping keys: {key!r} is both a leaf and a "
                "prefix of other returned keys (write-once)"
            )
        node[parts[-1]] = value
        leaves.add(key)
    return nested
