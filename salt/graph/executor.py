"""Plan execution for the salt v2 graph kernel.

Walks a compiled `Plan`'s steps as ``produced = module(bundle, mode)`` and
merges returns under write-once + declaration checks.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NoReturn, cast

import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import (
    ConfigError,
    DeclarationError,
    MutationError,
    UndeclaredAccessError,
)
from salt.graph.planner import Plan, PlanStep
from salt.graph.spec import KEY_SEP, GraphModule, Mode, SinkModule, flatten_spec, split_key

__all__ = ["STEP_SCOPE_PREFIX", "Executor", "canonical_produced", "record_steps"]

STEP_SCOPE_PREFIX = "salt.step/"
"""Prefix of the profiler scope name wrapping each plan step (`record_steps`)."""


class _StepRecording:
    """Process-global toggle for the per-step profiler scopes (off by default)."""

    enabled = False


@contextlib.contextmanager
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

    def __init__(self, plan: Plan, modules: Mapping[str, GraphModule] | None = None) -> None:
        """Bind a plan to the live module instances that will execute it.

        `modules` (e.g. the config's full instance dict) may be a superset
        of the plan's modules — pruned or mode-inactive entries are
        ignored; when omitted, the live instances frozen into the plan
        steps are used. Raises `ConfigError` if a plan module is missing
        from `modules`, doesn't implement `GraphModule`, isn't callable, or
        its declared name mismatches its plan-step name.
        """
        self.plan = plan
        self._modules: dict[str, GraphModule] = {}
        self._allowed: dict[str, frozenset[str]] = {}
        # sink steps stay IN plan.steps (render + demand) but are partitioned
        # OUT of the per-batch forward loop — the inverse of the setup-only
        # partition: a sink produces no tensor and is never called.
        self._forward_steps: list[PlanStep] = []
        for step in plan.steps:
            module = step.module if modules is None else modules.get(step.name)
            if module is None:
                raise ConfigError(
                    f"modules mapping has no entry {step.name!r}, required by the "
                    f"{plan.mode.name} plan (design §3.2)"
                )
            if not isinstance(module, GraphModule):
                raise ConfigError(
                    f"module {step.name!r} ({type(module).__name__}) does not implement the "
                    "GraphModule protocol (name + declare_io) (design §2.2)"
                )
            if module.name != step.name:
                raise ConfigError(
                    f"module supplied for step {step.name!r} declares name={module.name!r} — "
                    "instance names must match their plan-step names (design §2.2)"
                )
            self._modules[step.name] = module
            self._allowed[step.name] = _declared_reads(module, step, plan.mode)
            if _is_sink(module):
                # a terminal sink: no tensor forward, no callable requirement —
                # its consume/flush lifecycle is driven by the Lightning bridge
                continue
            if not callable(module):
                raise ConfigError(
                    f"module {step.name!r} ({type(module).__name__}) is not callable — the "
                    "executor invokes modules as module(bundle, mode) (design §3.2)"
                )
            self._forward_steps.append(step)

    def run(self, bundle: Bundle, debug: bool = False) -> Bundle:
        """Execute the plan's steps in order over `bundle` and return it.

        The caller provides every non-optional plan source leaf in
        `bundle`. With ``debug=True`` each module gets a read-tracking
        bundle view: an access outside its declared requires raises
        `UndeclaredAccessError`, and in-place mutation of an existing
        tensor (detected via ``torch.Tensor._version`` snapshots) raises
        `MutationError` (non-tensor leaves aren't covered). The
        returned-keys-vs-declaration check is always on; write-once
        collisions propagate from `Bundle.merge` as `KeyCollisionError`.

        Raises `KeyError` if the input bundle is missing a non-optional
        source leaf, `DeclarationError` on a malformed/mismatched return.
        """
        missing = sorted(
            key
            for key, spec in self.plan.sources.items()
            if not spec.optional and key not in bundle
        )
        if missing:
            raise KeyError(
                f"input bundle is missing source leaves required by the {self.plan.mode.name} "
                f"plan: {missing} — the caller provides every non-optional source (design §3.2)"
            )
        for step in self._forward_steps:
            module = cast("Callable[[Any, Mode], Any]", self._modules[step.name])
            view: Bundle | _ReadTrackedBundle = (
                _ReadTrackedBundle(bundle, step.name, self.plan.mode, self._allowed[step.name])
                if debug
                else bundle
            )
            versions = _tensor_versions(bundle) if debug else None
            if _StepRecording.enabled:
                with torch.profiler.record_function(f"{STEP_SCOPE_PREFIX}{step.name}"):
                    produced = module(view, self.plan.mode)
            else:
                produced = module(view, self.plan.mode)
            if not isinstance(produced, dict):
                raise DeclarationError(
                    f"module {step.name!r} returned {type(produced).__name__} — modules return "
                    "their declared produces as a dict of newly produced keys "
                    "(design §2.5, §3.2)"
                )
            if versions is not None and (mutated := _first_mutated(bundle, versions)) is not None:
                raise MutationError(
                    f"[mode={self.plan.mode.name}] module {step.name!r} mutated bundle key "
                    f"{mutated!r} in place during debug execution — bundle leaves are read-only "
                    "for modules; clone before mutating (e.g. b.get(...).clone()) and return "
                    "new keys (design §2.1, §3.2)"
                )
            expected = set(step.produces)
            bundle.merge(
                canonical_produced(produced, expected, step.name),
                who=step.name,
                expected=expected,
            )
        return bundle


def _tensor_versions(bundle: Bundle) -> dict[str, int]:
    """Snapshot the in-place version counter of every tensor leaf (debug only).

    ``torch.Tensor._version`` is torch's autograd in-place version counter —
    private but stable, and the cheapest torch-native mutation detector
    (integer reads, no data copies).
    """
    return {
        key: value._version  # noqa: SLF001 - torch's in-place version counter
        for key in bundle.keys()  # noqa: SIM118 - Bundle is not a Mapping
        if isinstance(value := bundle.get(key), torch.Tensor)
    }


def _first_mutated(bundle: Bundle, versions: dict[str, int]) -> str | None:
    """Find the first pre-step tensor leaf whose version counter bumped."""
    for key, version in versions.items():
        if bundle.get(key)._version != version:  # noqa: SLF001
            return key
    return None


def _declared_reads(module: GraphModule, step: PlanStep, mode: Mode) -> frozenset[str]:
    """Compute the keys `module` may read in `mode` (the debug-view allowlist).

    The union of the step's bound requires and every mode-active declared
    require — including optional ports the planner dropped for lack of a
    producer, so a module probing them with ``in`` sees a normal absence
    instead of an `UndeclaredAccessError`.
    """
    io = module.declare_io(mode)
    declared = {key for key, spec in flatten_spec(io.requires).items() if spec.active_in(mode)}
    return frozenset(declared | set(step.requires))


class _ReadTrackedBundle:
    """Read-tracking bundle view handed to modules under ``run(debug=True)``.

    Duck-types `Bundle`'s read surface (``get``/``subtree``/``keys``/``in``/
    ``len``). Access outside the module's declared requires raises
    `UndeclaredAccessError`; mutation and raw payload access are blocked —
    modules return their produces, only the executor merges.
    """

    __slots__ = ("_allowed", "_bundle", "_mode", "_who")

    def __init__(self, bundle: Bundle, who: str, mode: Mode, allowed: frozenset[str]) -> None:
        """Wrap `bundle` for module `who`, allowing reads of `allowed` keys only."""
        self._bundle = bundle
        self._who = who
        self._mode = mode
        self._allowed = allowed

    @property
    def data(self) -> NoReturn:
        """Blocked: raw payload access bypasses read tracking; always raises
        `UndeclaredAccessError`.
        """
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} accessed Bundle.data directly under "
            "debug execution — raw payload access bypasses read tracking; read declared keys "
            "via get()/subtree() (design §3.2)"
        )

    def get(self, key: str) -> Any:
        """Return the leaf value at a declared dotted key (see `Bundle.get`);
        raises `UndeclaredAccessError` if `key` is undeclared, else behaves
        exactly like the plain bundle (including `KeyError` if absent).
        """
        if key not in self._allowed:
            self._undeclared(key)
        return self._bundle.get(key)

    def subtree(self, prefix: str) -> dict[str, Any]:
        """Return the nested dict under `prefix`, checking every leaf is declared.

        Raises `UndeclaredAccessError` if nothing is declared under
        `prefix` or a leaf in the subtree is undeclared; an unknown or
        leaf-valued `prefix` raises `KeyError` like the plain bundle.
        """
        head = prefix + KEY_SEP
        if prefix not in self._allowed and not any(k.startswith(head) for k in self._allowed):
            self._undeclared(prefix)
        out = self._bundle.subtree(prefix)
        present = self._bundle.keys()
        for key in present:
            if key.startswith(head) and key not in self._allowed:
                self._undeclared(key)
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
        if key not in self._allowed:
            self._undeclared(key, verb="probed")
        return key in self._bundle

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"ReadTrackedBundle(module={self._who!r}, keys={self.keys()})"

    def set(self, key: str, value: Any) -> NoReturn:
        """Blocked: modules return produced keys; only the executor writes;
        always raises `UndeclaredAccessError`.
        """
        del value
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} called set({key!r}, ...) on its "
            "bundle view — modules RETURN their produced keys; the executor merges them "
            "(design §3.2)"
        )

    def merge(self, produced: dict[str, Any], who: str = "", expected: Any = None) -> NoReturn:
        """Blocked: merging is executor-only; always raises `UndeclaredAccessError`."""
        del produced, who, expected
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} called merge() on its bundle view "
            "— merging is executor-only; modules RETURN their produced keys (design §2.1, §3.2)"
        )

    def _undeclared(self, key: str, verb: str = "read") -> NoReturn:
        """Raise `UndeclaredAccessError` for `key`, naming the module, the
        key, and the declaration to amend.
        """
        declared = ", ".join(sorted(self._allowed)) or "<none>"
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} {verb} bundle key {key!r} which is "
            f"not in its declared requires — declared: {declared}. fix: amend "
            f"{self._who!r}.declare_io to require {key!r} (optional=True if consumed-if-"
            "present), or drop the access (design §3.2, §4.1)"
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
    flat: dict[str, Any] = {}
    _flatten_returned(produced, expected, who, "", flat)
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
                    f"returned leaf {path!r} (design §2.1 write-once)"
                )
            node = node.setdefault(parts[depth], {})
        if parts[-1] in node:
            raise DeclarationError(
                f"module {who!r} returned overlapping keys: {key!r} is both a leaf and a "
                "prefix of other returned keys (design §2.1 write-once)"
            )
        node[parts[-1]] = value
        leaves.add(key)
    return nested


def _flatten_returned(
    produced: dict[str, Any], expected: set[str], who: str, prefix: str, flat: dict[str, Any]
) -> None:
    """Flatten a returned dict to dotted leaves, guided by the declaration.

    Mirrors `Bundle.merge`'s flattening (dict values are structure unless
    their dotted path is declared), but additionally accepts dotted keys at
    any nesting level. Raises `DeclarationError` on non-string/malformed
    keys, or a key produced more than once.
    """
    for name, value in produced.items():
        if not isinstance(name, str):
            raise DeclarationError(
                f"module {who!r} returned a non-str produced key {name!r} (design §2.1)"
            )
        key = f"{prefix}{KEY_SEP}{name}" if prefix else name
        try:
            split_key(key)
        except (TypeError, ValueError) as err:
            raise DeclarationError(
                f"module {who!r} returned an invalid produced key {key!r}: {err}"
            ) from err
        if isinstance(value, dict) and value and key not in expected:
            _flatten_returned(value, expected, who, key, flat)
            continue
        if key in flat:
            raise DeclarationError(
                f"module {who!r} returned key {key!r} more than once (e.g. both nested and "
                "dotted spellings) (design §2.1 write-once)"
            )
        flat[key] = value
