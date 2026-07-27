"""Plan compilation for the salt v2 graph kernel.

Flattens `declare_io` to dotted ports, checks connectivity/kinds/shapes,
demand-prunes per mode against the sinks, and emits a frozen, hashed `Plan`.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Literal, NoReturn, TypeAlias

from salt.graph.errors import (
    _SUGGESTION_CUTOFF,
    AllModesDeadError,
    ConfigError,
    ConnectivityError,
    CycleError,
    GraphError,
    KindError,
    ShapeError,
)
from salt.graph.setup_spec import (
    SetupStage,
    SourceSpec,
    flatten_source_spec,
)
from salt.graph.spec import (
    KEY_SEP,
    PRIMARY_MODES,
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
    _has_wildcard,
    _pattern_matches,
    flatten_spec,
    iter_spec_leaves,
    split_key,
)

__all__ = [
    "SINKS",
    "SOURCES",
    "DeadOutput",
    "Edge",
    "GraphModule",
    "Plan",
    "PlanStep",
    "Sinks",
    "compile_plan",
    "compile_setup_plan",
    "deadcode",
]

SOURCES = "<sources>"
"""Edge producer sentinel: the key is provided a priori by the framework."""

SINKS = "<sinks>"
"""Edge consumer sentinel: the key is demanded by a sink (writer/export/loss boundary)."""

Sinks: TypeAlias = "Iterable[str] | Mapping[Mode, Iterable[str]] | None"
"""Sink keys: flat iterable (compiled mode only) or per-mode mapping."""

_MAX_SHOWN_KEYS = 12


# ---------------------------------------------------------------------------
# public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Edge:
    """One resolved graph edge: `producer` writes `key`, `consumer` reads it.

    `producer` is a module name or the `SOURCES` sentinel; `consumer` is a
    module name or the `SINKS` sentinel.
    """

    producer: str
    key: str
    consumer: str


@dataclass(frozen=True)
class PlanStep:
    """One executable step: a module plus its fully resolved flat IO.

    `requires`/`produces` are flattened ``{dotted_key: TensorSpec}`` mappings
    with wildcards narrowed and optional-but-absent requires dropped —
    `produces` is exactly the key set the executor passes to
    ``Bundle.merge(expected=...)``. `compile_plan` builds them as read-only
    `MappingProxyType` views, so the frozen plan is genuinely immutable
    (mutation raises TypeError); hashing uses `name` only.
    """

    name: str
    module: GraphModule = field(compare=False, repr=False)
    requires: Mapping[str, TensorSpec] = field(default_factory=dict, hash=False)
    produces: Mapping[str, TensorSpec] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class Plan:
    """A compiled, frozen execution plan for one mode.

    `steps` are in deterministic topological order; `edges` are sorted by
    ``(producer, key, consumer)``; `sources` are the mode-active framework
    boundary leaves (a read-only mapping); `plan_hash` is a sha256 over a
    canonical serialisation, stable across runs and machines.
    """

    mode: Mode
    steps: tuple[PlanStep, ...]
    edges: tuple[Edge, ...]
    sources: Mapping[str, TensorSpec] = field(hash=False)
    plan_hash: str

    @property
    def module_names(self) -> tuple[str, ...]:
        """The ordered module names of this plan."""
        return tuple(step.name for step in self.steps)

    def step(self, name: str) -> PlanStep:
        """Look up a step by module instance name.

        Raises
        ------
        KeyError
            If no step has that name.
        """
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"plan for mode {self.mode.name} has no step {name!r}")


@dataclass(frozen=True)
class DeadOutput:
    """One dead-output finding from `deadcode`.

    `key` is the dead produced key, or ``"*"`` when the whole module was
    demand-pruned; `module` may be the `SOURCES` sentinel for unconsumed
    source leaves. `severity` classifies the finding: an unconsumed
    ``preds.*`` port in TEST mode is an ``"error"`` by default (predictions
    silently vanishing from eval files); an unconsumed ``preds.*`` port in
    FIT/VAL is ``"info"`` (the normal case of no configured metric callback;
    never promoted by ``--strict``), as is a module pruned from the ONNX plan
    (the ONNX sinks are the writer-declared manifest ports, and an export
    surface narrower than eval is legitimate by design, so
    ``--strict --mode onnx`` stays usable on narrowed configs); everything
    else is a ``"warning"``. The CLI exits non-zero on any error-level
    finding.
    """

    module: str
    key: str
    reason: str
    severity: Literal["error", "warning", "info"] = "warning"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def compile_plan(
    modules: dict[str, GraphModule],
    mode: Mode,
    sources: NestedSpec,
    schema: Collection[str] | None = None,
    sinks: Sinks = None,
    sink_origins: Mapping[str, str] | None = None,
) -> Plan:
    """Compile the execution plan for one primary mode.

    `sources` is the a-priori framework boundary; `schema` bounds wildcard
    narrowing; `sinks` anchors demand pruning. Only non-optional requires
    create wildcard demand. `sink_origins` maps sink keys to a
    human-readable demander, used only to improve error messages.

    Raises `ConfigError`/`ConnectivityError`/`KindError`/`ShapeError`/
    `CycleError`/`AllModesDeadError` (all `GraphError` subclasses) per the
    respective validation failure.
    """
    _check_primary_mode(mode)
    if sinks is not None and not isinstance(sinks, Mapping):
        sinks = list(sinks)
    res = _resolve(modules, mode, sources, schema, _sinks_for(sinks, mode, mode), sink_origins)
    _check_all_modes_dead(modules, mode, sources, sinks, res)
    return _assemble_plan(res)


def compile_setup_plan(
    modules: dict[str, GraphModule],
    stage: SetupStage,
) -> Plan:
    """Compile the SETUP-time source plan for one stage.

    The setup-graph twin of `compile_plan`, sharing topo-sort, connectivity/
    kind/cycle checks and `plan_hash`. Differs in two ways: the IO getter is
    ``module.declare_setup_io(stage)`` (a `SetupIO` of `SourceSpec` leaves,
    not ``declare_io(mode)``); shape-unification is skipped (`SourceSpec`
    has no shape/dtype). No demand pruning, no sinks, no all-modes-dead
    check. `stage` is a Lightning-style stage, NOT a `Mode`; the returned
    `Plan.mode` is set to `Mode.ALL` as a placeholder — the setup executor
    never reads it.

    Raises `ConfigError`/`ConnectivityError`/`KindError`/`CycleError` per
    the respective validation failure.
    """
    _check_incompatibilities(modules)
    res = _resolve(modules, Mode.ALL, {}, None, None, None, _setup_face(stage))
    return _assemble_plan(res)


def _check_incompatibilities(modules: Mapping[str, GraphModule]) -> None:
    """Enforce declared mutual-exclusion between configured setup modules.

    For each module declaring class names in ``incompatible_with``, raise
    `ConfigError` if another configured module has that class name (matched
    by name, so the target need not exist yet — dormant until both are
    present).
    """
    for name, module in modules.items():
        forbidden = getattr(module, "incompatible_with", ())
        for other_name, other in modules.items():
            if other_name == name:
                continue
            if type(other).__name__ in forbidden:
                raise ConfigError(
                    f"module {name!r} ({type(module).__name__}) is incompatible with "
                    f"{other_name!r} ({type(other).__name__}): "
                    f"{type(module).__name__}.incompatible_with = {tuple(forbidden)!r} "
                    "(plan-25 Rev-2 — e.g. staging a VDS copies h5py pointers, not data); "
                    "drop one of them from data.modules"
                )


def _assemble_plan(res: _Resolution) -> Plan:
    """Topo-sort a resolved graph and freeze it into a hashed `Plan` (shared core).

    The tail shared by `compile_plan` and `compile_setup_plan`: deterministic
    Kahn order, immutable `PlanStep`/`Edge` views, and the structural
    `plan_hash`. Leaf-type-agnostic — `TensorSpec` and `SourceSpec` both
    serialise via `_spec_payload`.
    """
    order = _topo_order(res)
    steps = tuple(
        PlanStep(
            name=name,
            module=res.alive[name].module,
            requires=MappingProxyType(dict(res.alive[name].bound)),
            produces=MappingProxyType(res.alive[name].all_produces()),
        )
        for name in order
    )
    edges = tuple(sorted(res.edges))
    sources_out: Mapping[str, Any] = MappingProxyType(dict(res.sources))
    return Plan(
        mode=res.mode,
        steps=steps,
        edges=edges,
        sources=sources_out,
        plan_hash=_plan_hash(steps, edges, sources_out),
    )


def deadcode(
    modules: dict[str, GraphModule],
    mode: Mode,
    sources: NestedSpec,
    schema: Collection[str] | None = None,
    sinks: Sinks = None,
) -> list[DeadOutput]:
    """Report produced-but-never-consumed keys for one mode.

    Mode-gated absences are not reported. Findings cover: whole modules
    dropped by demand pruning (when `sinks` given), produced leaves of
    surviving modules with no consumer, and unconsumed source leaves.
    Unlike `compile_plan`, a module dead in every mode is reported, not
    raised — but connectivity/cycle errors still raise.

    Severity: unconsumed ``preds.*`` in TEST is ``"error"`` (a computed
    prediction with no writer); in FIT/VAL, or a module pruned in ONNX
    mode, it's ``"info"`` (expected — no metric callback, or an
    intentionally narrower export surface); everything else is
    ``"warning"``. The per-task ``expose:`` opt-out gates ports out of a
    mode before this runs, so an opted-out ``preds.*`` prunes its module
    (warning) rather than hitting the TEST error.

    Returns deterministically ordered findings; empty when everything is
    consumed.
    """
    _check_primary_mode(mode)
    if sinks is not None and not isinstance(sinks, Mapping):
        sinks = list(sinks)
    res = _resolve(modules, mode, sources, schema, _sinks_for(sinks, mode, mode))
    consumed: dict[str, set[str]] = {}
    for edge in res.edges:
        consumed.setdefault(edge.producer, set()).add(edge.key)
    # ONNX-mode pruning is the writer-manifest narrowing story: the export
    # surface is explicitly declared and may legitimately be narrower than
    # eval — info-level, never promoted by --strict. All other modes keep
    # the warning default.
    pruned_suffix = (
        " (narrowed out of the writer-declared export surface — legitimate, M4.5 amendment §4)"
        if mode == Mode.ONNX
        else ""
    )
    # A CONVERSION PRODUCER (a salt.outputs node producing an
    # ``outputs.*`` leaf — ClassProbs/SeqClassIndex/etc.)
    # that prunes in a non-ONNX mode is the by-design export-pruning story:
    # its OnnxExportSink (or an H5OutputSink) is inactive in that mode, so
    # the node has no sink and prunes legitimately. Demote it to info
    # (never --strict-promoted), exactly like the ONNX-narrowing case — a
    # config carrying ONNX export nodes must still pass
    # `salt graph validate --strict --mode test`.

    def _is_conversion_producer(name: str) -> bool:
        return isinstance(getattr(modules.get(name), "output_key", None), str)

    out: list[DeadOutput] = [
        DeadOutput(
            name,
            "*",
            f"module pruned in mode {mode.name}: {res.pruned[name]}{pruned_suffix}",
            severity=(
                "info" if (mode == Mode.ONNX or _is_conversion_producer(name)) else "warning"
            ),
        )
        for name in sorted(res.pruned)
    ]
    unconsumed_reason = f"produced but never consumed in mode {mode.name}"
    preds_in_test_reason = (
        f"produced but never consumed in mode {mode.name} — an unconsumed preds.* port in "
        "TEST means a computed prediction is never persisted; wire a writer or drop the "
        "port (design §4.2)"
    )
    preds_in_training_reason = (
        f"produced but never consumed in mode {mode.name} "
        "(normal: no configured metric callback consumes this prediction — design §3.3)"
    )
    for name, node in res.alive.items():
        used = consumed.get(name, set())
        for key in node.all_produces():
            if key in used:
                continue
            is_preds = key.partition(KEY_SEP)[0] == "preds"
            if is_preds and mode == Mode.TEST:
                out.append(DeadOutput(name, key, preds_in_test_reason, severity="error"))
            elif is_preds:
                out.append(DeadOutput(name, key, preds_in_training_reason, severity="info"))
            else:
                out.append(DeadOutput(name, key, unconsumed_reason))
    src_used = consumed.get(SOURCES, set())
    out.extend(
        DeadOutput(SOURCES, key, f"source never consumed in mode {mode.name}")
        for key in res.sources
        if key not in src_used
    )
    return out


# ---------------------------------------------------------------------------
# IO-face adapter (tensor vs setup graph share _compile_core)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _IOFace:
    """The per-graph IO contract `_resolve` reads through.

    Lets the tensor and setup planners share node/edge/topo/hash machinery,
    differing only in the IO getter and whether shape-unification runs: the
    tensor face reads ``declare_io(mode)`` and unifies; the setup face reads
    ``declare_setup_io(stage)`` and skips unification (`SourceSpec` has no
    shape/dtype). Both leaf types expose ``.kind``/``.optional``/
    ``.active_in(...)``, so kind-matching and gating stay leaf-agnostic.
    """

    getter: Callable[[GraphModule], Any]
    flatten: Callable[[Any], dict[str, Any]]
    unify: bool
    gate: Callable[[Any, Mode], bool]  # (spec, mode) -> active; mode ignored by setup face


def _tensor_face(mode: Mode) -> _IOFace:
    """The tensor IO face bound to `mode` (the per-batch graph)."""
    return _IOFace(
        getter=lambda module: module.declare_io(mode),
        flatten=flatten_spec,
        unify=True,
        gate=lambda spec, m: spec.active_in(m),
    )


def _setup_face(stage: SetupStage) -> _IOFace:
    """The setup IO face bound to `stage` (the setup graph)."""
    return _IOFace(
        getter=lambda module: module.declare_setup_io(stage),
        flatten=flatten_source_spec,
        unify=False,
        gate=lambda spec, _m: spec.active_in(stage),
    )


# ---------------------------------------------------------------------------
# resolution (shared by compile_plan / deadcode / the all-modes-dead probe)
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """Internal per-module resolution state for one mode.

    `rank` is the module's position in the config ``modules`` dict — the
    topological tie-break key (config declaration order).
    """

    name: str
    rank: int
    module: GraphModule
    requires: dict[str, TensorSpec]
    produces: dict[str, TensorSpec]
    patterns: dict[str, TensorSpec]
    narrowed: dict[str, TensorSpec] = field(default_factory=dict)
    bound: dict[str, TensorSpec] = field(default_factory=dict)

    def all_produces(self) -> dict[str, TensorSpec]:
        """Concrete plus narrowed produces, as a fresh flat dict."""
        return {**self.produces, **self.narrowed}


@dataclass
class _Resolution:
    """Internal result of resolving one mode's graph."""

    mode: Mode
    sources: dict[str, TensorSpec]
    nodes: dict[str, _Node]  # mode-active modules, pre-prune, in config order
    alive: dict[str, _Node]  # post demand-prune, in config order
    edges: list[Edge]  # among alive modules (+ SOURCES/SINKS), sorted
    pruned: dict[str, str]  # demand-pruned module -> reason
    inactive: tuple[str, ...]  # mode-inactive module names (expected absences)
    sink_keys: tuple[str, ...]


def _resolve(
    modules: dict[str, GraphModule],
    mode: Mode,
    sources: NestedSpec,
    schema: Collection[str] | None,
    sink_keys: list[str] | None,
    sink_origins: Mapping[str, str] | None = None,
    face: _IOFace | None = None,
) -> _Resolution:
    """Resolve one mode's graph: narrow wildcards, build edges, check, prune.

    `face` selects the IO contract: the default tensor face reads
    ``declare_io(mode)`` and unifies shapes; the setup face reads
    ``declare_setup_io(stage)`` and skips unification.
    """
    if face is None:
        face = _tensor_face(mode)
    src = _active_sources(sources, mode)
    nodes, inactive = _collect_nodes(modules, mode, face)
    producer_of = _concrete_producers(src, nodes, mode)
    sink_list = _checked_sink_keys(sink_keys)
    demand = _collect_demand(nodes, sink_list or [])
    _narrow_wildcards(nodes, producer_of, demand, schema, mode, sink_origins)
    edges = _build_edges(
        nodes, producer_of, src, sink_list or [], mode, modules, sources, sink_origins, face
    )
    _check_wildcard_self_feed(nodes, edges, mode)
    if sink_list is None:
        alive, pruned = dict(nodes), {}
    else:
        needed = _demand_closure(nodes, sink_list)
        reason = f"outputs reach no sink in mode {mode.name}"
        pruned = {name: reason for name in nodes if name not in needed}
        alive = {name: node for name, node in nodes.items() if name in needed}
        edges = [
            edge
            for edge in edges
            if (edge.producer == SOURCES or edge.producer in alive)
            and (edge.consumer == SINKS or edge.consumer in alive)
        ]
        _drop_unconsumed_narrowed(alive, edges)
    return _Resolution(
        mode=mode,
        sources=src,
        nodes=nodes,
        alive=alive,
        edges=sorted(edges),
        pruned=pruned,
        inactive=tuple(inactive),
        sink_keys=tuple(sink_list or []),
    )


def _active_sources(sources: NestedSpec, mode: Mode) -> dict[str, TensorSpec]:
    """Flatten the source boundary and keep mode-active leaves.

    Raises `ConfigError` if a source key contains a wildcard component.
    """
    src: dict[str, TensorSpec] = {}
    for key, spec in flatten_spec(sources).items():
        if _has_wildcard(key):
            raise ConfigError(
                f"source key {key!r} may not contain wildcards — sources are concrete (design §2.2)"
            )
        if spec.active_in(mode):
            src[key] = spec
    return src


def _collect_nodes(
    modules: dict[str, GraphModule], mode: Mode, face: _IOFace
) -> tuple[dict[str, _Node], list[str]]:
    """Build per-module nodes with mode-active flattened ports, in config
    declaration order (the topological tie-break, recorded as `_Node.rank`).

    Returns active nodes and mode-inactive module names. Raises
    `ConfigError` on protocol/name violations, reserved names, or wildcard
    misuse.
    """
    nodes: dict[str, _Node] = {}
    inactive: list[str] = []
    for rank, (name, module) in enumerate(modules.items()):
        if name in {SOURCES, SINKS}:
            raise ConfigError(
                f"module instance name {name!r} collides with a reserved planner sentinel "
                f"({SOURCES!r}/{SINKS!r}) — rename the module (design §3.1)"
            )
        if not isinstance(module, GraphModule):
            raise ConfigError(
                f"module {name!r} ({type(module).__name__}) does not implement the GraphModule "
                "protocol (name + declare_io) (design §2.2)"
            )
        if module.name != name:
            raise ConfigError(
                f"module mapped at key {name!r} declares name={module.name!r} — instance names "
                "must match their config keys (design §2.2)"
            )
        io = face.getter(module)
        # framework-internal seam for "only framework-shipped producers may
        # declare patterns": the attribute is not user API. TODO(M2): bind
        # the capability to shipped code (module-path check or a framework
        # registry) so user classes cannot grant it to themselves.
        allow_wildcards = bool(getattr(module, "allow_wildcards", False))
        requires: dict[str, Any] = {}
        produces: dict[str, Any] = {}
        patterns: dict[str, Any] = {}
        for key, spec in face.flatten(io.requires).items():
            if _has_wildcard(key):
                raise ConfigError(
                    f"module {name!r} declares wildcard require {key!r} — only framework "
                    "producers may declare patterns, and only in produces (design §2.2)"
                )
            if face.gate(spec, mode):
                requires[key] = spec
        for key, spec in face.flatten(io.produces).items():
            if not face.gate(spec, mode):
                continue
            if _has_wildcard(key):
                if not allow_wildcards:
                    raise ConfigError(
                        f"module {name!r} declares wildcard produces {key!r} but is not a "
                        "framework wildcard producer (allow_wildcards is not set) — user "
                        "modules may not declare wildcards (design §2.2)"
                    )
                patterns[key] = spec
            else:
                produces[key] = spec
        if not (requires or produces or patterns):
            inactive.append(name)
            continue
        nodes[name] = _Node(name, rank, module, requires, produces, patterns)
    return nodes, inactive


def _concrete_producers(
    src: dict[str, TensorSpec], nodes: dict[str, _Node], mode: Mode
) -> dict[str, str]:
    """Map each concrete produced key to its single producer (sources -> `SOURCES`).

    Raises `ConnectivityError` if any key has two producers.
    """
    producer_of: dict[str, str] = dict.fromkeys(src, SOURCES)
    for name, node in nodes.items():
        for key in node.produces:
            other = producer_of.get(key)
            if other is not None:
                raise ConnectivityError(
                    f"[mode={mode.name}] key {key!r} has two producers: {other!r} and {name!r} "
                    "— every bundle key must have exactly one producer (design §3.1)"
                )
            producer_of[key] = name
    return producer_of


def _checked_sink_keys(sink_keys: list[str] | None) -> list[str] | None:
    """Validate sink keys are well-formed concrete dotted keys.

    Raises `ConfigError` if a sink key contains a wildcard component.
    """
    if sink_keys is None:
        return None
    for key in sink_keys:
        split_key(key)
        if _has_wildcard(key):
            raise ConfigError(f"sink key {key!r} may not contain wildcards — sinks are concrete")
    return sink_keys


def _collect_demand(nodes: dict[str, _Node], sink_keys: list[str]) -> dict[str, list[str]]:
    """Collect concrete demand: non-optional requires plus sink keys.

    Returns ``{demanded_key: [consumer names]}`` (`SINKS` for sink demand).
    """
    demand: dict[str, list[str]] = {}
    for name, node in nodes.items():
        for key, spec in node.requires.items():
            if not spec.optional:
                demand.setdefault(key, []).append(name)
    for key in sink_keys:
        demand.setdefault(key, []).append(SINKS)
    return demand


def _describe_consumer(consumer: str, key: str, sink_origins: Mapping[str, str] | None) -> str:
    """Render one demand consumer for an error message: module repr, or the
    configured sink-origin description (never the raw ``'<sinks>'`` placeholder).
    """
    if consumer != SINKS:
        return repr(consumer)
    if sink_origins and key in sink_origins:
        return sink_origins[key]
    return "the configured sinks"


def _narrow_wildcards(
    nodes: dict[str, _Node],
    producer_of: dict[str, str],
    demand: dict[str, list[str]],
    schema: Collection[str] | None,
    mode: Mode,
    sink_origins: Mapping[str, str] | None = None,
) -> None:
    """Narrow wildcard patterns against concrete demand (design §2.2 rules (a)-(d)).

    Concrete producers beat wildcards (rule (a)); narrowed keys are
    validated against `schema` when given (rule (d)); the result is written
    into each node's `narrowed` dict and frozen into the plan (rule (c)).
    Raises `ConnectivityError` when two wildcard producers match one
    demanded key, or a narrowed key is not in the schema.
    """
    wildcard_owner: dict[str, str] = {}
    for name, node in nodes.items():
        for pattern in sorted(node.patterns):
            spec = node.patterns[pattern]
            for key in sorted(demand):
                if key in producer_of or not _pattern_matches(pattern, key):
                    continue
                owner = wildcard_owner.get(key)
                if owner == name:
                    continue  # already narrowed via another pattern of this producer
                if owner is not None:
                    raise ConnectivityError(
                        f"[mode={mode.name}] key {key!r} matches wildcard patterns of both "
                        f"{owner!r} and {name!r} — every bundle key must have exactly one "
                        "producer (design §3.1)"
                    )
                if schema is not None and key not in schema:
                    consumers = ", ".join(
                        _describe_consumer(c, key, sink_origins) for c in demand[key]
                    )
                    near = get_close_matches(key, sorted(schema), n=3, cutoff=_SUGGESTION_CUTOFF)
                    hint = f"\n  nearest schema keys: {', '.join(near)}" if near else ""
                    raise ConnectivityError(
                        f"[mode={mode.name}] key {key!r} (demanded by {consumers}) is not in "
                        f"the declared schema — {name!r} can only serve schema-backed keys "
                        f"(wildcard {pattern!r}, design §2.2 rule (d)).{hint}\n"
                        f"  fix: correct the key in the demanding module's config"
                    )
                wildcard_owner[key] = name
                node.narrowed[key] = spec
    for name, node in nodes.items():
        for key in node.narrowed:
            producer_of[key] = name


def _build_edges(
    nodes: dict[str, _Node],
    producer_of: dict[str, str],
    src: dict[str, TensorSpec],
    sink_keys: list[str],
    mode: Mode,
    modules: dict[str, GraphModule],
    sources: NestedSpec,
    sink_origins: Mapping[str, str] | None = None,
    face: _IOFace | None = None,
) -> list[Edge]:
    """Bind every require/sink to its producer; check kinds and unify shapes.

    Kind-matching runs for every face; shape unification (`_unify_edge`)
    runs only for the tensor face (`SourceSpec` has no shape/dtype). Missing
    producers raise `ConnectivityError` (via `_raise_missing_producer`);
    unification conflicts raise `ShapeError`; kind mismatches raise
    `KindError`. Returns the resolved edges (pre-prune); optional-but-absent
    requires bind no edge, and bound requires are recorded on each node.
    """
    unify = face.unify if face is not None else True
    dims = _DimTable(mode)
    edges: list[Edge] = []
    for name, node in nodes.items():
        for key, spec in node.requires.items():
            producer = producer_of.get(key)
            if producer is None:
                if spec.optional:
                    continue
                _raise_missing_producer(name, key, mode, producer_of, modules, sources)
            node.bound[key] = spec
            edges.append(Edge(producer, key, name))
            pspec = src[key] if producer == SOURCES else nodes[producer].all_produces()[key]
            if pspec.kind != spec.kind:
                raise KindError(
                    f"[mode={mode.name}] module {name!r} port {key!r} expects "
                    f"kind={spec.kind!r} but producer {producer!r} provides "
                    f"kind={pspec.kind!r} (design §2.2)"
                )
            if unify:
                _unify_edge(key, producer, name, pspec, spec, dims, mode)
    for key in sink_keys:
        producer = producer_of.get(key)
        if producer is None:
            _raise_missing_producer(SINKS, key, mode, producer_of, modules, sources, sink_origins)
        edges.append(Edge(producer, key, SINKS))
    return edges


def _drop_unconsumed_narrowed(alive: dict[str, _Node], edges: list[Edge]) -> None:
    """Drop narrowed wildcard keys whose only demand was removed by pruning.

    Narrowing runs against pre-prune demand: once a sole consumer is
    pruned, its key must not stay in the wildcard producer's plan
    (`Bundle.merge` enforces the frozen key set exactly). Keys with any
    surviving edge are kept.
    """
    consumed: dict[str, set[str]] = {}
    for edge in edges:
        consumed.setdefault(edge.producer, set()).add(edge.key)
    for name, node in alive.items():
        if not node.narrowed:
            continue
        used = consumed.get(name, set())
        for key in [key for key in node.narrowed if key not in used]:
            del node.narrowed[key]


def _is_terminal_consumer(module: GraphModule) -> bool:
    """Whether a no-current-produces module is a genuine terminal consumer.

    True only if `module` has no CONCRETE produced port active in ANY
    primary mode — covers true terminal consumers (writers) and
    demand-driven wildcard producers (a pattern port narrowing to nothing
    here). An `expose: [fit, val]` task that's merely inactive in this mode
    still has a concrete port elsewhere, so it's a prunable producer, not a
    sink — the opt-out mechanism relies on this.
    """
    for m in PRIMARY_MODES:
        for key, spec in flatten_spec(module.declare_io(m).produces).items():
            if not _has_wildcard(key) and spec.active_in(m):
                return False
    return True


def _demand_closure(nodes: dict[str, _Node], sink_keys: list[str]) -> set[str]:
    """Reverse demand walk: modules whose outputs (transitively) reach a sink.

    Terminal consumers (active requires, no produces) anchor demand
    alongside explicit sink keys. A module gated OUT of this mode only by
    the per-task ``expose:`` opt-out still has live produces elsewhere, so
    it's a prunable producer here, never a sink — only a module producing
    nothing in ANY primary mode is a genuine terminal. Returns the names of
    needed modules.
    """
    needed_keys = set(sink_keys)
    needed: set[str] = set()
    for name, node in nodes.items():
        if not node.all_produces() and _is_terminal_consumer(node.module):
            needed.add(name)
            needed_keys |= set(node.bound)
    changed = True
    while changed:
        changed = False
        for name, node in nodes.items():
            if name in needed:
                continue
            if needed_keys & set(node.all_produces()):
                needed.add(name)
                needed_keys |= set(node.bound)
                changed = True
    return needed


# ---------------------------------------------------------------------------
# cycle checks
# ---------------------------------------------------------------------------


def _module_adjacency(edges: list[Edge], nodes: dict[str, _Node]) -> dict[str, dict[str, str]]:
    """Module-level adjacency from edges: ``adj[producer][consumer] = key``
    (modules only; sentinels excluded; one binding key kept per pair).
    """
    adj: dict[str, dict[str, str]] = {}
    for edge in edges:
        if edge.producer in nodes and edge.consumer in nodes:
            adj.setdefault(edge.producer, {}).setdefault(edge.consumer, edge.key)
    return adj


def _check_wildcard_self_feed(nodes: dict[str, _Node], edges: list[Edge], mode: Mode) -> None:
    """Reject narrowed wildcard outputs that transitively feed the producer's own inputs.

    Checked explicitly so the error names the wildcard, not just the
    resulting cycle. Raises `CycleError` naming the producer and the
    key-level cycle path.
    """
    adj = _module_adjacency(edges, nodes)
    for name, node in nodes.items():
        if not node.narrowed:
            continue
        starts = [
            (edge.consumer, edge.key)
            for edge in edges
            if edge.producer == name and edge.key in node.narrowed and edge.consumer in nodes
        ]
        path = _path_back(adj, starts, name)
        if path is not None:
            chain = name + "".join(f" -({key})-> {module}" for key, module in path)
            raise CycleError(
                f"[mode={mode.name}] wildcard producer {name!r}: narrowed outputs transitively "
                f"feed its own inputs: {chain} (design §2.2 rule (b))"
            )


def _path_back(
    adj: dict[str, dict[str, str]], starts: list[tuple[str, str]], target: str
) -> tuple[tuple[str, str], ...] | None:
    """BFS from `starts` edges back to `target`; returns the path as
    ``((key, module), ...)`` ending at `target`, or None.
    """
    queue: deque[tuple[str, tuple[tuple[str, str], ...]]] = deque()
    seen: set[str] = set()
    for consumer, key in sorted(starts):
        step: tuple[tuple[str, str], ...] = ((key, consumer),)
        if consumer == target:
            return step
        if consumer not in seen:
            seen.add(consumer)
            queue.append((consumer, step))
    while queue:
        current, path = queue.popleft()
        for nxt, key in sorted(adj.get(current, {}).items()):
            if nxt == target:
                return (*path, (key, nxt))
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, (*path, (key, nxt))))
    return None


def _topo_order(res: _Resolution) -> list[str]:
    """Deterministic topological order: Kahn, tie-break = config declaration order.

    Ties between independent modules are broken by `_Node.rank`, so
    reordering independent modules in YAML is the supported way to nudge
    execution order (e.g. peak memory). Raises `CycleError`, naming it as a
    key-level chain, if the graph has a cycle.
    """
    alive = res.alive
    deps: dict[str, set[str]] = {name: set() for name in alive}
    rdeps: dict[str, set[str]] = {name: set() for name in alive}
    for edge in res.edges:
        if edge.producer in alive and edge.consumer in alive:
            deps[edge.consumer].add(edge.producer)
            rdeps[edge.producer].add(edge.consumer)
    indegree = {name: len(parents) for name, parents in deps.items()}
    heap = [(alive[name].rank, name) for name, count in indegree.items() if count == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, name = heapq.heappop(heap)
        order.append(name)
        for consumer in sorted(rdeps[name]):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(heap, (alive[consumer].rank, consumer))
    if len(order) != len(alive):
        residual = {name for name in alive if name not in set(order)}
        chain = _find_cycle(residual, _module_adjacency(res.edges, res.alive))
        raise CycleError(
            f"[mode={res.mode.name}] dependency cycle: {chain} — a module's outputs may not "
            "feed its own inputs; cycles are always a config error (design §3.1)"
        )
    return order


def _find_cycle(residual: set[str], adj: dict[str, dict[str, str]]) -> str:
    """Find one cycle in the residual graph; returns it as a key-level chain,
    e.g. ``"a -(k1)-> b -(k2)-> a"``.
    """
    state = dict.fromkeys(residual, 0)  # 0 unvisited, 1 on stack, 2 done
    for start in sorted(residual):
        if state[start]:
            continue
        stack: list[tuple[str, Any]] = [(start, iter(sorted(adj.get(start, {}).items())))]
        state[start] = 1
        while stack:
            current, edge_iter = stack[-1]
            advanced = next(edge_iter, None)
            if advanced is None:
                state[current] = 2
                stack.pop()
                continue
            nxt, key = advanced
            if nxt not in residual:
                continue
            if state[nxt] == 1:
                names = [name for name, _ in stack]
                cycle_nodes = names[names.index(nxt) :]
                chain = cycle_nodes[0]
                for a, b in pairwise(cycle_nodes):
                    chain += f" -({adj[a][b]})-> {b}"
                return chain + f" -({key})-> {nxt}"
            if state[nxt] == 0:
                state[nxt] = 1
                stack.append((nxt, iter(sorted(adj.get(nxt, {}).items()))))
    return "<no cycle found>"  # pragma: no cover - residual implies a cycle


# ---------------------------------------------------------------------------
# kind / shape / dtype checks
# ---------------------------------------------------------------------------


class _DimTable:
    """Union-find over symbolic dims with concrete bindings."""

    def __init__(self, mode: Mode) -> None:
        """Create an empty table for error messages tagged with `mode`."""
        self._mode = mode
        self._parent: dict[str, str] = {}
        self._size: dict[str, tuple[int, str]] = {}  # root -> (size, endpoint)

    def _find(self, dim: str) -> str:
        self._parent.setdefault(dim, dim)
        root = dim
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[dim] != root:  # path compression
            self._parent[dim], dim = root, self._parent[dim]
        return root

    def bind(self, dim: str, size: int, endpoint: str) -> None:
        """Bind a symbolic dim to a concrete size; raises `ShapeError` on
        conflict, naming both endpoints and the conflicting sizes.
        """
        root = self._find(dim)
        previous = self._size.get(root)
        if previous is not None and previous[0] != size:
            raise ShapeError(
                f"[mode={self._mode.name}] symbolic dim {dim!r} is {previous[0]} at "
                f"{previous[1]} but {size} at {endpoint} — conflicting sizes (design §2.2)"
            )
        if previous is None:
            self._size[root] = (size, endpoint)

    def union(self, a: str, b: str, endpoint: str) -> None:
        """Unify two symbolic dims; raises `ShapeError` on conflicting
        concrete bindings, naming both binding endpoints and the sizes.
        """
        root_a, root_b = self._find(a), self._find(b)
        if root_a == root_b:
            return
        size_a, size_b = self._size.get(root_a), self._size.get(root_b)
        if size_a is not None and size_b is not None and size_a[0] != size_b[0]:
            raise ShapeError(
                f"[mode={self._mode.name}] unifying {a!r} with {b!r} at {endpoint}: "
                f"{a!r} is {size_a[0]} (from {size_a[1]}) but {b!r} is {size_b[0]} "
                f"(from {size_b[1]}) — conflicting sizes (design §2.2)"
            )
        self._parent[root_b] = root_a
        if size_a is None and size_b is not None:
            self._size[root_a] = size_b


def _unify_edge(
    key: str,
    producer: str,
    consumer: str,
    pspec: TensorSpec,
    cspec: TensorSpec,
    dims: _DimTable,
    mode: Mode,
) -> None:
    """Unify one edge's producer/consumer specs (shape rank, dims, dtype);
    raises `ShapeError` on rank, size, symbolic-binding, or dtype conflict.
    """
    if pspec.dtype is not None and cspec.dtype is not None and pspec.dtype != cspec.dtype:
        raise ShapeError(
            f"[mode={mode.name}] dtype mismatch on {key!r}: producer {producer!r} declares "
            f"{pspec.dtype!r} but consumer {consumer!r} expects {cspec.dtype!r}"
        )
    if pspec.shape is None or cspec.shape is None:
        return
    if len(pspec.shape) != len(cspec.shape):
        raise ShapeError(
            f"[mode={mode.name}] rank mismatch on {key!r}: producer {producer!r} declares "
            f"shape {pspec.shape} but consumer {consumer!r} expects {cspec.shape}"
        )
    for i, (pdim, cdim) in enumerate(zip(pspec.shape, cspec.shape, strict=True)):
        endpoint = f"{key!r} ({producer} -> {consumer}, dim {i})"
        if isinstance(pdim, int) and isinstance(cdim, int):
            if pdim != cdim:
                raise ShapeError(
                    f"[mode={mode.name}] size mismatch at {endpoint}: producer has {pdim}, "
                    f"consumer expects {cdim}"
                )
        elif isinstance(pdim, str) and isinstance(cdim, int):
            dims.bind(pdim, cdim, endpoint)
        elif isinstance(pdim, int) and isinstance(cdim, str):
            dims.bind(cdim, pdim, endpoint)
        else:
            dims.union(str(pdim), str(cdim), endpoint)


# ---------------------------------------------------------------------------
# error-message helpers
# ---------------------------------------------------------------------------


def _raise_missing_producer(
    consumer: str,
    key: str,
    mode: Mode,
    producer_of: dict[str, str],
    modules: dict[str, GraphModule],
    sources: NestedSpec,
    sink_origins: Mapping[str, str] | None = None,
) -> NoReturn:
    """Raise `ConnectivityError` for a missing producer, naming the
    consumer (the demanding module behind a sink, when `sink_origins` knows
    it), the key, nearest-key suggestions, the available keys, modes in
    which the key would exist, and a concrete fix.
    """
    if consumer == SINKS:
        demander = (
            f" (demanded by {sink_origins[key]})" if sink_origins and key in sink_origins else ""
        )
        head = f"[mode={mode.name}] sink key {key!r}{demander} — no module or source produces it."
        fix = f"fix: correct the sink key, or add a module producing {key!r}"
    else:
        head = (
            f"[mode={mode.name}] module {consumer!r} requires {key!r} — no module or source "
            "produces it."
        )
        fix = f"fix: correct the require in module {consumer!r}, or add a module producing {key!r}"
    lines = [head]
    available = sorted(producer_of)
    near = get_close_matches(key, available, n=3, cutoff=_SUGGESTION_CUTOFF)
    if near:
        lines.append(f"  did you mean: {', '.join(repr(k) for k in near)}?")
    if available:
        shown = available[:_MAX_SHOWN_KEYS]
        more = f" (+{len(available) - len(shown)} more)" if len(available) > len(shown) else ""
        lines.append(f"  available keys: {', '.join(shown)}{more}")
    lines.extend(f"  {note}" for note in _other_mode_producers(modules, sources, key, mode))
    lines.append(f"  {fix}")
    raise ConnectivityError("\n".join(lines))


def _other_mode_producers(
    modules: dict[str, GraphModule], sources: NestedSpec, key: str, mode: Mode
) -> list[str]:
    """Find modules/sources that could produce `key` in another primary
    mode; returns human-readable notes, deterministically ordered.
    """
    candidates: dict[str, list[str]] = {}
    for other in PRIMARY_MODES:
        if other == mode:
            continue
        for name in sorted(modules):
            io = modules[name].declare_io(other)
            for pkey, spec in iter_spec_leaves(io.produces):
                if not spec.active_in(other):
                    continue
                if pkey == key or (_has_wildcard(pkey) and _pattern_matches(pkey, key)):
                    candidates.setdefault(name, []).append(other.name)
                    break
    notes = [
        f"{name!r} could produce {key!r} in mode(s) {'/'.join(mode_names)}"
        for name, mode_names in sorted(candidates.items())
    ]
    for skey, spec in iter_spec_leaves(sources):
        if skey == key and not spec.active_in(mode):
            active = [m.name for m in PRIMARY_MODES if spec.active_in(m)]
            if active:
                notes.append(f"sources provide {key!r} in mode(s) {'/'.join(active)}")
            break
    return notes


# ---------------------------------------------------------------------------
# mode/sink plumbing and the all-modes-dead check
# ---------------------------------------------------------------------------


def _check_primary_mode(mode: Mode) -> None:
    """Reject composite modes — one plan is compiled per primary mode.

    Raises `ConfigError` if `mode` is not one of `PRIMARY_MODES`.
    """
    if mode not in PRIMARY_MODES:
        names = "/".join(m.name for m in PRIMARY_MODES)
        raise ConfigError(
            f"plans are compiled per primary mode ({names}); got {mode!r} — compile one plan "
            "per mode (design §3.1)"
        )


def _sinks_for(sinks: Sinks, mode: Mode, compiled_mode: Mode) -> list[str] | None:
    """Resolve the sink keys applying to `mode`; None means demand is unknown.

    A flat iterable applies only to the compiled mode; a ``{Mode: keys}``
    mapping applies to every primary mode its (possibly composite) keys
    overlap. Modes with no overlapping mapping entry have unknown demand —
    they are never demand-pruned, keeping the all-modes-dead check lenient.
    Returns sorted sink keys for `mode`, or None when demand is unknown.
    """
    if sinks is None:
        return None
    if isinstance(sinks, Mapping):
        if not any(mode_key & mode for mode_key in sinks):
            return None
        return sorted({key for mode_key, keys in sinks.items() if mode_key & mode for key in keys})
    if mode != compiled_mode:
        return None
    return sorted(set(sinks))


def _check_all_modes_dead(
    modules: dict[str, GraphModule],
    mode: Mode,
    sources: NestedSpec,
    sinks: Sinks,
    res: _Resolution,
) -> None:
    """Raise `AllModesDeadError` if any configured module is dead in every
    primary mode.

    A module absent from the compiled plan (mode-inactive or demand-pruned)
    is probed in each other primary mode; modes that fail to resolve are
    leniently treated as alive — only provably-dead-everywhere modules raise.
    """
    for name in sorted(set(modules) - set(res.alive)):
        alive_somewhere = any(
            other != mode
            and _alive_probe(modules, name, other, sources, _sinks_for(sinks, other, mode))
            for other in PRIMARY_MODES
        )
        if not alive_somewhere:
            module = modules[name]
            names = "/".join(m.name for m in PRIMARY_MODES)
            raise AllModesDeadError(
                f"module {name!r} ({type(module).__name__}) is dead in every mode ({names}): "
                "no port is active, or its outputs reach no sink in any mode — a configured "
                "module must do something; remove it or wire a consumer "
                "(design §3.1, principle 10)"
            )


def _alive_probe(
    modules: dict[str, GraphModule],
    name: str,
    mode: Mode,
    sources: NestedSpec,
    sink_keys: list[str] | None,
) -> bool:
    """Check whether module `name` survives `mode`'s pruning (lenient).

    Returns True if the module is alive in `mode`, or if `mode` cannot be
    resolved at all (deadness must be provable).
    """
    try:
        probe = _resolve(modules, mode, sources, None, sink_keys)
    except GraphError:
        return True
    return name in probe.alive


# ---------------------------------------------------------------------------
# plan hashing
# ---------------------------------------------------------------------------


def _spec_payload(spec: TensorSpec | SourceSpec) -> dict[str, Any]:
    """Canonical JSON-able form of a leaf spec for hashing (tensor OR setup).

    Both leaf types serialise through the same helper: a `SourceSpec` has no
    ``shape``/``dtype``/``fields`` and gates on string `stages` rather than
    `Mode`, so those keys are emitted as None / the stage list. The setup
    hash is structural and stable, giving the setup graph a reproducible
    identity for ``salt graph`` exactly like the tensor graph.
    """
    if isinstance(spec, SourceSpec):
        return {
            "shape": None,
            "dtype": None,
            "kind": spec.kind,
            "stages": list(spec.stages),
            "optional": spec.optional,
            "fields": None,
        }
    return {
        "shape": list(spec.shape) if spec.shape is not None else None,
        "dtype": spec.dtype,
        "kind": spec.kind,
        "modes": [m.name for m in PRIMARY_MODES if spec.active_in(m)],
        "optional": spec.optional,
        "fields": list(spec.fields) if spec.fields is not None else None,
    }


def _plan_hash(
    steps: tuple[PlanStep, ...],
    edges: tuple[Edge, ...],
    sources: Mapping[str, TensorSpec],
) -> str:
    """sha256 over a canonical plan serialisation (structural, not mode-tagged).

    Step order is significant (execution order); dict keys are JSON-sorted
    so insertion order never leaks. Mode is deliberately excluded from the
    payload, so two modes with identical structure hash equal — making the
    FIT/VAL plan-identity check a cheap hash comparison.
    """
    payload = {
        "steps": [
            {
                "name": step.name,
                "requires": {k: _spec_payload(v) for k, v in step.requires.items()},
                "produces": {k: _spec_payload(v) for k, v in step.produces.items()},
            }
            for step in steps
        ],
        "edges": [[edge.producer, edge.key, edge.consumer] for edge in edges],
        "sources": {k: _spec_payload(v) for k, v in sources.items()},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
