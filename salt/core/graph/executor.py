"""Plan execution for the salt v2 graph kernel.

Design §3.2: the executor is the entire runtime — it walks a compiled
`Plan`'s steps in order, invokes each module, and merges the returned keys
into the run `Bundle` under write-once + declaration checks. No reflection,
no dispatch tables, no dict-order dependence; ONNX tracing sees a plain
sequence of module invocations.

Module call convention (design §2.5, §3.2)
------------------------------------------
Each step's module is invoked as ``produced = module(b, mode)``:

- ``b`` is the run `Bundle` itself (or, under ``run(debug=True)``, a
  read-tracking view of it — design §4.1). Modules read their declared
  requires via ``b.get(...)`` / ``b.subtree(...)``; for ``nn.Module``
  subclasses the call lands in ``forward(b, mode)``.
- ``mode`` is the plan's primary `Mode` — a plan property resolved before
  tracing, never a tensor input (design §2.5).
- The return value is the module's newly produced keys ONLY. Both spellings
  used in the design are accepted and canonicalised before the merge: nested
  dicts (``{"preds": {"x": y}}``), flat dotted keys (``{"preds.x": y}``), or
  any mixture. A dict value whose dotted path is itself a declared key (e.g.
  the ``seq.layout`` meta leaf) is kept whole as a dict-valued leaf.
- The executor merges via
  ``Bundle.merge(produced, who=step.name, expected=set(step.produces))`` —
  extra, missing, or colliding keys raise with the module name in the
  message, on every merge, independent of debug mode.

Optional ports (design §2.2): an optional require with no producer in the
plan's mode is dropped from `PlanStep.requires` at compile time; at runtime
the key is simply *absent* from the bundle — absent optional inputs are
omitted, not passed as ``None`` (the bundle-passing convention has no
argument to thread ``None`` through). Modules probe with ``key in b``; the
debug view permits this for every declared require, bound or not.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NoReturn, cast

import torch

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import (
    ConfigError,
    DeclarationError,
    MutationError,
    UndeclaredAccessError,
)
from salt.core.graph.planner import Plan, PlanStep
from salt.core.graph.spec import KEY_SEP, GraphModule, Mode, flatten_spec, split_key

__all__ = ["Executor"]


class Executor:
    """Runs a compiled `Plan` over a `Bundle` (design §3.2).

    The plan is frozen: steps execute in plan order, each module is called as
    ``module(bundle, mode)`` (see the module docstring for the full call
    convention), and the returned keys are merged write-once with the
    declared key set enforced on every merge.
    """

    def __init__(self, plan: Plan, modules: Mapping[str, GraphModule] | None = None) -> None:
        """Bind a plan to the live module instances that will execute it.

        `modules` (e.g. the config's full instance dict) may be a superset of
        the plan's modules — pruned or mode-inactive entries are ignored.
        When omitted, the live instances frozen into the plan steps are used.

        Raises
        ------
        ConfigError
            If a plan module is missing from `modules`, does not implement
            the `GraphModule` protocol, is not callable, or declares a name
            that differs from its plan-step name.
        """
        self.plan = plan
        self._modules: dict[str, GraphModule] = {}
        self._allowed: dict[str, frozenset[str]] = {}
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
            if not callable(module):
                raise ConfigError(
                    f"module {step.name!r} ({type(module).__name__}) is not callable — the "
                    "executor invokes modules as module(bundle, mode) (design §3.2)"
                )
            self._modules[step.name] = module
            self._allowed[step.name] = _declared_reads(module, step, plan.mode)

    def run(self, bundle: Bundle, debug: bool = False) -> Bundle:
        """Execute the plan's steps in order over `bundle` and return it (design §3.2).

        The caller provides every non-optional plan source leaf in `bundle`.
        With ``debug=True`` each module receives a read-tracking bundle view:
        any access outside its declared requires raises
        `UndeclaredAccessError` naming the module, the key, and the
        declaration to amend (design §4.1 quality bar), and in-place mutation
        of existing bundle tensors is detected via ``torch.Tensor._version``
        snapshots around each step and raises `MutationError` (design §2.1;
        non-tensor leaves such as numpy arrays are not covered). The
        returned-keys-vs-declaration check is always on, debug or not;
        write-once collisions propagate from `Bundle.merge` as
        `KeyCollisionError` (design §2.1).

        Returns
        -------
        Bundle
            The same bundle instance, with every step's produces merged in.

        Raises
        ------
        KeyError
            If the input bundle is missing a non-optional plan source leaf.
        DeclarationError
            If a module's return value is not a dict, contains malformed /
            duplicate / overlapping keys, or its key set does not match the
            declared produces.
        MutationError
            Under ``debug=True``, if a module mutated a bundle tensor in place.
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
        for step in self.plan.steps:
            module = cast("Callable[[Any, Mode], Any]", self._modules[step.name])
            view: Bundle | _ReadTrackedBundle = (
                _ReadTrackedBundle(bundle, step.name, self.plan.mode, self._allowed[step.name])
                if debug
                else bundle
            )
            versions = _tensor_versions(bundle) if debug else None
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
                _canonical_produced(produced, expected, step.name),
                who=step.name,
                expected=expected,
            )
        return bundle


def _tensor_versions(bundle: Bundle) -> dict[str, int]:
    """Snapshot the in-place version counter of every tensor leaf (debug only).

    ``torch.Tensor._version`` is torch's autograd in-place version counter —
    private but stable, and the cheapest torch-native mutation detector
    (integer reads, no data copies).

    Returns
    -------
    dict[str, int]
        ``{dotted_key: version}`` for tensor-valued leaves.
    """
    return {
        key: value._version  # noqa: SLF001 - torch's in-place version counter
        for key in bundle.keys()  # noqa: SIM118 - Bundle is not a Mapping
        if isinstance(value := bundle.get(key), torch.Tensor)
    }


def _first_mutated(bundle: Bundle, versions: dict[str, int]) -> str | None:
    """Find the first pre-step tensor leaf whose version counter bumped.

    Returns
    -------
    str | None
        The mutated dotted key, or None when nothing was mutated in place.
    """
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

    Returns
    -------
    frozenset[str]
        Dotted keys the module may access under ``run(debug=True)``.
    """
    io = module.declare_io(mode)
    declared = {key for key, spec in flatten_spec(io.requires).items() if spec.active_in(mode)}
    return frozenset(declared | set(step.requires))


class _ReadTrackedBundle:
    """Read-tracking bundle view handed to modules under ``run(debug=True)``.

    Design §3.2/§4.1: duck-types the read surface of `Bundle` (``get`` /
    ``subtree`` / ``keys`` / ``in`` / ``len``). Any access outside the
    module's declared requires raises `UndeclaredAccessError` naming the
    module, the key, and the declaration to amend; declared-but-absent
    optional keys behave exactly as on the plain bundle. Mutation and raw
    payload access are blocked — modules return their produces; only the
    executor merges.
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
        """Blocked: raw payload access bypasses read tracking.

        Raises
        ------
        UndeclaredAccessError
            Always.
        """
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} accessed Bundle.data directly under "
            "debug execution — raw payload access bypasses read tracking; read declared keys "
            "via get()/subtree() (design §3.2)"
        )

    def get(self, key: str) -> Any:
        """Return the leaf value at a declared dotted key (see `Bundle.get`).

        An undeclared `key` raises `UndeclaredAccessError`; a declared but
        absent key (e.g. an optional port not produced in this mode) raises
        `KeyError` exactly like the plain bundle.

        Returns
        -------
        Any
            The leaf value.
        """
        if key not in self._allowed:
            self._undeclared(key)
        return self._bundle.get(key)

    def subtree(self, prefix: str) -> dict[str, Any]:
        """Return the nested dict under `prefix`, checking every leaf is declared.

        Raises `UndeclaredAccessError` if the module declares nothing under
        `prefix` or the subtree contains a leaf outside the declared
        requires; an unknown or leaf-valued `prefix` raises `KeyError` like
        the plain bundle.

        Returns
        -------
        dict[str, Any]
            A fresh nested dict; leaf values are shared with the bundle.
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
        """Return the declared dotted leaf keys currently present in the bundle.

        Returns
        -------
        list[str]
            Declared-and-present keys, in bundle insertion order.
        """
        present = self._bundle.keys()
        return [key for key in present if key in self._allowed]

    def __contains__(self, key: str) -> bool:
        """Probe a declared key's presence (the optional-port idiom, design §2.2).

        Probing an undeclared key raises `UndeclaredAccessError` — branching
        on an undeclared key is itself an undeclared dependence.

        Returns
        -------
        bool
            True if the declared key is present in the bundle.
        """
        if key not in self._allowed:
            self._undeclared(key, verb="probed")
        return key in self._bundle

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"ReadTrackedBundle(module={self._who!r}, keys={self.keys()})"

    def set(self, key: str, value: Any) -> NoReturn:
        """Blocked: modules return produced keys; only the executor writes.

        Raises
        ------
        UndeclaredAccessError
            Always.
        """
        del value
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} called set({key!r}, ...) on its "
            "bundle view — modules RETURN their produced keys; the executor merges them "
            "(design §3.2)"
        )

    def merge(self, produced: dict[str, Any], who: str = "", expected: Any = None) -> NoReturn:
        """Blocked: merging is executor-only (design §2.1).

        Raises
        ------
        UndeclaredAccessError
            Always.
        """
        del produced, who, expected
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} called merge() on its bundle view "
            "— merging is executor-only; modules RETURN their produced keys (design §2.1, §3.2)"
        )

    def _undeclared(self, key: str, verb: str = "read") -> NoReturn:
        """Raise the undeclared-access error for `key` (design §4.1 quality bar).

        Raises
        ------
        UndeclaredAccessError
            Always — naming the module, the key, and the declaration to amend.
        """
        declared = ", ".join(sorted(self._allowed)) or "<none>"
        raise UndeclaredAccessError(
            f"[mode={self._mode.name}] module {self._who!r} {verb} bundle key {key!r} which is "
            f"not in its declared requires — declared: {declared}. fix: amend "
            f"{self._who!r}.declare_io to require {key!r} (optional=True if consumed-if-"
            "present), or drop the access (design §3.2, §4.1)"
        )


# ---------------------------------------------------------------------------
# returned-keys canonicalisation (nested and flat dotted spellings, §2.5/§3.3)
# ---------------------------------------------------------------------------


def _canonical_produced(produced: dict[str, Any], expected: set[str], who: str) -> dict[str, Any]:
    """Canonicalise a module's returned dict to nested single-component form.

    Modules may return nested dicts, flat dotted keys, or any mixture (the
    design writes both spellings, §2.5/§3.3); `Bundle.merge` expects nested
    dicts with single-component keys. A dict value whose dotted path is in
    `expected` is a declared dict-valued leaf and is kept whole; empty
    undeclared dicts are kept as leaves so they surface as unexpected keys.

    Returns
    -------
    dict[str, Any]
        The nested produced dict, iteration order preserved.

    Raises
    ------
    DeclarationError
        On malformed keys, duplicate keys (e.g. both spellings of one key),
        or keys that are simultaneously a leaf and a subtree.
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
    any nesting level.

    Raises
    ------
    DeclarationError
        On non-string / malformed keys, or a key produced more than once.
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
