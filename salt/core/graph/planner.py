"""Plan compilation for the salt v2 graph kernel.

Design §3.1: modules declare their interfaces (`declare_io`), the planner
flattens them to dotted ports, narrows wildcard producers against concrete
demand (design §2.2 rules (a)-(d)), builds producer→consumer edges, checks
connectivity / kinds / symbolic shape unification, prunes per mode, and emits
a frozen, hashed `Plan`.

Static only: compilation never touches data files, the network, or tensors
(design principle 8). Topological order is fully deterministic — Kahn's
algorithm with ties broken by *config declaration order* (the order module
names appear in the ``modules`` dict, which YAML preserves) — so the same
config yields the same plan (and `plan_hash`) on every machine, and
reordering independent modules in YAML is the supported way to nudge
execution order, e.g. for peak memory (design §3.1, documented loudly;
risk 3 mitigation). The plan hash makes any reorder detectable.

Sinks anchor demand pruning (design §3.1): in the full framework FIT/VAL
sinks are ``loss.total`` plus callback requires, TEST sinks are the writers,
ONNX sinks are the export ports. The kernel takes them as an explicit
``sinks`` argument — either a flat iterable of dotted keys for the compiled
mode, or a ``{Mode: keys}`` mapping enabling full per-mode demand analysis
for the all-modes-dead check (design §1 principle 10). Modules that declare
requires but no produces in ANY mode (writers) are terminal consumers and
anchor demand themselves; a module producing nothing only in THIS mode because
its ports are mode-gated out (the §4.2 ``expose:`` opt-out) is a prunable
producer, not a terminal.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import deque
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Literal, NoReturn, TypeAlias

from salt.core.graph.errors import (
    AllModesDeadError,
    ConfigError,
    ConnectivityError,
    CycleError,
    GraphError,
    KindError,
    ShapeError,
)
from salt.core.graph.spec import (
    KEY_SEP,
    PRIMARY_MODES,
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
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
    "deadcode",
]

SOURCES = "<sources>"
"""Edge producer sentinel: the key is provided a priori by the framework (design §3.1)."""

SINKS = "<sinks>"
"""Edge consumer sentinel: the key is demanded by a sink (writer/export/loss boundary)."""

Sinks: TypeAlias = "Iterable[str] | Mapping[Mode, Iterable[str]] | None"
"""Sink keys: flat iterable (compiled mode only) or per-mode mapping (design §3.1)."""

_WILDCARD_PARTS = frozenset({"*", "**"})
_SUGGESTION_CUTOFF = 0.5
_MAX_SHOWN_KEYS = 12


# ---------------------------------------------------------------------------
# public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Edge:
    """One resolved graph edge: `producer` writes `key`, `consumer` reads it (design §3.1).

    `producer` is a module name or the `SOURCES` sentinel; `consumer` is a
    module name or the `SINKS` sentinel.
    """

    producer: str
    key: str
    consumer: str


@dataclass(frozen=True)
class PlanStep:
    """One executable step: a module plus its fully resolved flat IO (design §3.1).

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
    """A compiled, frozen execution plan for one mode (design §3.1).

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
        """The ordered module names of this plan.

        Returns
        -------
        tuple[str, ...]
            Module instance names in execution order.
        """
        return tuple(step.name for step in self.steps)

    def step(self, name: str) -> PlanStep:
        """Look up a step by module instance name.

        Returns
        -------
        PlanStep
            The step for `name`.

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
    """One dead-output finding from `deadcode` (design §4.2).

    `key` is the dead produced key, or ``"*"`` when the whole module was
    demand-pruned; `module` may be the `SOURCES` sentinel for unconsumed
    source leaves. `severity` classifies the finding: an unconsumed
    ``preds.*`` port in TEST mode is an ``"error"`` by default (design §4.2 —
    predictions silently vanishing from eval files); an unconsumed
    ``preds.*`` port in FIT/VAL is ``"info"`` (design §3.3 — the normal case
    of no configured metric callback; never promoted by ``--strict``), as is
    a module pruned from the ONNX plan (M4.5: the ONNX sinks are the
    writer-declared manifest ports, and an export surface narrower than
    eval — ``onnx_streams``/``onnx_tasks`` — is legitimate by design, so
    ``--strict --mode onnx`` stays usable on narrowed configs);
    everything else is a ``"warning"``. The CLI exits non-zero on any
    error-level finding.
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
    """Compile the execution plan for one primary mode (design §3.1).

    `sources` is what the framework provides a priori (the dataset boundary
    for model-side plans; empty for full-pipeline plans). `schema`, when
    given, is the universe of dotted keys wildcard patterns may narrow to
    (design §2.2 rule (d)); `sinks` anchors demand pruning (see module
    docstring). Only non-optional requires create wildcard demand — optional
    ports are consumed-if-present and never force a producer to materialise
    a key. `sink_origins` optionally maps sink keys to a human-readable
    description of WHO demanded them (e.g. the model-side task module behind
    a dataset-plan sink) — used purely to upgrade error messages to the
    design §4.1 attribution bar.

    Errors (all `GraphError` subclasses): `ConfigError` for composite modes,
    instance-name mismatches, wildcard misuse, or non-concrete source/sink
    keys; `ConnectivityError` for missing/duplicate producers or
    schema-invalid wildcard narrowing; `KindError` for consumer/producer kind
    mismatches; `ShapeError` for symbolic-dim/dtype unification conflicts;
    `CycleError` for dependency cycles including wildcard self-feed
    (design §2.2 rule (b)); `AllModesDeadError` for a configured module dead
    in every primary mode (design §1 principle 10).

    Returns
    -------
    Plan
        The frozen, hashed plan.
    """
    _check_primary_mode(mode)
    if sinks is not None and not isinstance(sinks, Mapping):
        sinks = list(sinks)
    res = _resolve(modules, mode, sources, schema, _sinks_for(sinks, mode, mode), sink_origins)
    _check_all_modes_dead(modules, mode, sources, sinks, res)
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
    sources_out: Mapping[str, TensorSpec] = MappingProxyType(dict(res.sources))
    return Plan(
        mode=mode,
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
    """Report produced-but-never-consumed keys for one mode (design §4.2).

    Mode-aware: ports gated out of `mode` by their declarations are expected
    absences and are not reported. Findings cover (a) whole modules dropped
    by demand pruning when `sinks` is given, (b) produced leaves of surviving
    modules with no consumer, and (c) unconsumed source leaves. Unlike
    `compile_plan`, a module dead in every mode is reported, not raised — but
    graph errors (missing producers, cycles via wildcards, ...) still raise.

    Severity (design §4.2, §3.3): an unconsumed ``preds.*`` key in TEST mode
    is an ``"error"`` by default — the model computed a prediction and no
    writer will persist it. An unconsumed ``preds.*`` key in FIT/VAL is
    ``"info"`` — the normal case of no configured metric callback (§3.3).
    A module pruned in ONNX mode is ``"info"`` too (M4.5 unified manifest:
    ONNX sinks are the writer-declared export-manifest ports, and narrowing
    the Athena surface below the eval surface is legitimate — the §4.2
    export-pruning story; never promoted by ``--strict``).
    Everything else is a ``"warning"``.
    The per-task ``expose:`` opt-out (design §4.2, M5 sub-wave D) gates the
    ``preds.*`` port to the configured modes BEFORE this report runs: a
    ``expose: [fit, val]`` task produces no prediction in TEST, so the planner
    prunes it (a warning-level whole-module pruning, never the TEST preds
    error) — the opt-out both silences the error and prunes the task. Once
    configured callbacks' declared requires enter FIT/VAL sinks (§3.1/§3.4,
    landed in M5 sub-wave C), a callback-consumed pred stops appearing here at
    all.

    Returns
    -------
    list[DeadOutput]
        Deterministically ordered findings; empty when everything is consumed.
    """
    _check_primary_mode(mode)
    if sinks is not None and not isinstance(sinks, Mapping):
        sinks = list(sinks)
    res = _resolve(modules, mode, sources, schema, _sinks_for(sinks, mode, mode))
    consumed: dict[str, set[str]] = {}
    for edge in res.edges:
        consumed.setdefault(edge.producer, set()).add(edge.key)
    # ONNX-mode pruning is the writer-manifest narrowing story (M4.5): the
    # export surface is explicitly declared and may legitimately be narrower
    # than eval — info-level, never promoted by --strict (the README/§4.2
    # export-pruning contract). All other modes keep the warning default.
    pruned_suffix = (
        " (narrowed out of the writer-declared export surface — legitimate, M4.5 amendment §4)"
        if mode == Mode.ONNX
        else ""
    )
    out: list[DeadOutput] = [
        DeadOutput(
            name,
            "*",
            f"module pruned in mode {mode.name}: {res.pruned[name]}{pruned_suffix}",
            severity="info" if mode == Mode.ONNX else "warning",
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
# resolution (shared by compile_plan / deadcode / the all-modes-dead probe)
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """Internal per-module resolution state for one mode.

    `rank` is the module's position in the config ``modules`` dict — the
    topological tie-break key (design §3.1: config declaration order).
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
        """Concrete plus narrowed produces, as a fresh flat dict.

        Returns
        -------
        dict[str, TensorSpec]
            All keys this module writes in this mode.
        """
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
) -> _Resolution:
    """Resolve one mode's graph: narrow wildcards, build edges, check, prune.

    Returns
    -------
    _Resolution
        The resolved graph state for `mode`.
    """
    src = _active_sources(sources, mode)
    nodes, inactive = _collect_nodes(modules, mode)
    producer_of = _concrete_producers(src, nodes, mode)
    sink_list = _checked_sink_keys(sink_keys)
    demand = _collect_demand(nodes, sink_list or [])
    _narrow_wildcards(nodes, producer_of, demand, schema, mode, sink_origins)
    edges = _build_edges(
        nodes, producer_of, src, sink_list or [], mode, modules, sources, sink_origins
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

    Returns
    -------
    dict[str, TensorSpec]
        Mode-active source leaves by dotted key.

    Raises
    ------
    ConfigError
        If a source key contains a wildcard component.
    """
    src: dict[str, TensorSpec] = {}
    for key, spec in flatten_spec(sources).items():
        if _is_pattern(key):
            raise ConfigError(
                f"source key {key!r} may not contain wildcards — sources are concrete (design §2.2)"
            )
        if spec.active_in(mode):
            src[key] = spec
    return src


def _collect_nodes(
    modules: dict[str, GraphModule], mode: Mode
) -> tuple[dict[str, _Node], list[str]]:
    """Build per-module nodes with mode-active flattened ports.

    Nodes keep the config declaration order (dict insertion order) — it is
    the topological tie-break (design §3.1) — and record it as `_Node.rank`.

    Returns
    -------
    tuple[dict[str, _Node], list[str]]
        Active nodes in config order, and mode-inactive module names.

    Raises
    ------
    ConfigError
        Protocol/name violations, reserved names, or wildcard misuse
        (design §2.2, §3.1).
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
        io = module.declare_io(mode)
        # M1 seam for §2.2 "only framework-shipped producers may declare patterns":
        # the attribute is framework-internal, not user API. TODO(M2): bind the
        # capability to shipped code (module-path check or a framework registry)
        # so user classes cannot grant it to themselves.
        allow_wildcards = bool(getattr(module, "allow_wildcards", False))
        requires: dict[str, TensorSpec] = {}
        produces: dict[str, TensorSpec] = {}
        patterns: dict[str, TensorSpec] = {}
        for key, spec in flatten_spec(io.requires).items():
            if _is_pattern(key):
                raise ConfigError(
                    f"module {name!r} declares wildcard require {key!r} — only framework "
                    "producers may declare patterns, and only in produces (design §2.2)"
                )
            if spec.active_in(mode):
                requires[key] = spec
        for key, spec in flatten_spec(io.produces).items():
            if not spec.active_in(mode):
                continue
            if _is_pattern(key):
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
    """Map each concrete produced key to its single producer.

    Returns
    -------
    dict[str, str]
        ``{dotted_key: producer_name}``; sources map to `SOURCES`.

    Raises
    ------
    ConnectivityError
        If any key has two producers (design §3.1).
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

    Returns
    -------
    list[str] | None
        The validated keys, or None when demand is unknown.

    Raises
    ------
    ConfigError
        If a sink key contains a wildcard component.
    """
    if sink_keys is None:
        return None
    for key in sink_keys:
        split_key(key)
        if _is_pattern(key):
            raise ConfigError(f"sink key {key!r} may not contain wildcards — sinks are concrete")
    return sink_keys


def _collect_demand(nodes: dict[str, _Node], sink_keys: list[str]) -> dict[str, list[str]]:
    """Collect concrete demand: non-optional requires plus sink keys.

    Returns
    -------
    dict[str, list[str]]
        ``{demanded_key: [consumer names]}`` (`SINKS` for sink demand).
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
    """Render one demand consumer for an error message (§4.1 attribution).

    Returns
    -------
    str
        The module name (repr) for module consumers; for the `SINKS`
        sentinel, the configured origin description when one is known
        (never the raw ``'<sinks>'`` placeholder).
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

    Concrete producers beat wildcards (rule (a)); narrowed keys are validated
    against `schema` when given (rule (d)); the result is written into each
    node's `narrowed` dict and frozen into the plan (rule (c)).

    Raises
    ------
    ConnectivityError
        Two wildcard producers match one demanded key, or a narrowed key is
        not in the schema.
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
) -> list[Edge]:
    """Bind every require/sink to its producer; check kinds and unify shapes.

    Missing producers raise `ConnectivityError` (via `_raise_missing_producer`)
    and unification conflicts raise `ShapeError` (via `_unify_edge`).

    Returns
    -------
    list[Edge]
        The resolved edges (pre-prune). Optional-but-absent requires bind no
        edge; bound requires are recorded on each node.

    Raises
    ------
    KindError
        Consumer port kind differs from producer leaf kind (design §2.2).
    """
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
            _unify_edge(key, producer, name, pspec, spec, dims, mode)
    for key in sink_keys:
        producer = producer_of.get(key)
        if producer is None:
            _raise_missing_producer(SINKS, key, mode, producer_of, modules, sources, sink_origins)
        edges.append(Edge(producer, key, SINKS))
    return edges


def _drop_unconsumed_narrowed(alive: dict[str, _Node], edges: list[Edge]) -> None:
    """Drop narrowed wildcard keys whose only demand was removed by pruning.

    Narrowing runs against pre-prune demand (design §2.2): once demand pruning
    removes a consumer, a key it alone demanded must not stay in the wildcard
    producer's plan — wildcard producers only materialise demanded keys (rule
    (c)), and `Bundle.merge` enforces the frozen key set exactly. Keys with a
    surviving edge (alive consumer, optional or not, or a sink) are kept.
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
    """Whether a no-current-produces module is a genuine terminal consumer/no-op.

    A module producing nothing in THIS mode anchors demand (like a writer) ONLY
    if it has no CONCRETE produced port active in any OTHER mode. This keeps the
    two legitimate no-current-produces shapes alive:

    - true terminal consumers (writers): no produces in any mode at all;
    - demand-driven wildcard producers (`Labels`'s ``labels.**``): only a
      pattern port, which narrows to nothing in a mode with no demand (the
      module's own docstring: "the kernel keeps wildcard producers with bound
      requires alive as terminal consumers").

    It does NOT keep an `expose: [fit, val]` task alive in test/onnx: that task
    declares a CONCRETE ``preds.*`` port active in fit/val, so it is a prunable
    producer here, not a sink — the opt-out the §4.2 mechanism relies on.

    Returns
    -------
    bool
        True when `module` has no concrete (non-wildcard) produced port active
        in any primary mode.
    """
    for m in PRIMARY_MODES:
        for key, spec in flatten_spec(module.declare_io(m).produces).items():
            if not _is_pattern(key) and spec.active_in(m):
                return False
    return True


def _demand_closure(nodes: dict[str, _Node], sink_keys: list[str]) -> set[str]:
    """Reverse demand walk: modules whose outputs (transitively) reach a sink.

    Terminal consumers (active requires, no produces — writers) anchor demand
    alongside the explicit sink keys (design §3.1). A module that produces
    nothing only because every one of its produced ports is mode-gated OUT of
    this mode (the design §4.2 per-task ``expose: [fit, val]`` opt-out — its
    ``preds.*``/``losses.*`` are inactive here) is NOT such a terminal: it has
    live produces in other modes, so it is a prunable producer here, never a
    sink. Only a module producing nothing in ANY primary mode is a genuine
    terminal consumer (a writer-shaped node).

    Returns
    -------
    set[str]
        Names of needed modules.
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
    """Module-level adjacency from edges, keeping one binding key per pair.

    Returns
    -------
    dict[str, dict[str, str]]
        ``adj[producer][consumer] = key`` (modules only; sentinels excluded).
    """
    adj: dict[str, dict[str, str]] = {}
    for edge in edges:
        if edge.producer in nodes and edge.consumer in nodes:
            adj.setdefault(edge.producer, {}).setdefault(edge.consumer, edge.key)
    return adj


def _check_wildcard_self_feed(nodes: dict[str, _Node], edges: list[Edge], mode: Mode) -> None:
    """Reject narrowed wildcard outputs that transitively feed the producer's own inputs.

    Design §2.2 rule (b): checked explicitly so the error explains the
    wildcard, not just the resulting cycle.

    Raises
    ------
    CycleError
        Naming the wildcard producer and the key-level cycle path.
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
    """BFS from `starts` edges back to `target`, returning the (key, module) path.

    Returns
    -------
    tuple[tuple[str, str], ...] | None
        The path as ``((key, module), ...)`` ending at `target`, or None.
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

    Design §3.1, documented loudly: ties between independent modules are
    broken by the order their names appear in the config ``modules`` dict
    (`_Node.rank`), so reordering independent modules in YAML is the
    supported way to nudge execution order (e.g. peak memory, risk 3).

    Returns
    -------
    list[str]
        Module names in execution order.

    Raises
    ------
    CycleError
        If the graph has a cycle, naming it as a key-level chain.
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
    """Find one cycle in the residual graph and format it as a key-level chain.

    Returns
    -------
    str
        E.g. ``"a -(k1)-> b -(k2)-> a"``.
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
    """Union-find over symbolic dims with concrete bindings (design §2.2)."""

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
        """Bind a symbolic dim to a concrete size; conflict raises ShapeError.

        Raises
        ------
        ShapeError
            Naming both endpoints and the conflicting sizes (design §4.1).
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
        """Unify two symbolic dims; conflicting concrete bindings raise ShapeError.

        Raises
        ------
        ShapeError
            Naming both binding endpoints and the conflicting sizes.
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
    """Unify one edge's producer/consumer specs (shape rank, dims, dtype).

    Raises
    ------
    ShapeError
        Rank, concrete-size, symbolic-binding, or dtype conflict.
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
# error-message helpers (design §4.1 quality bar)
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
    """Raise the missing-producer error meeting the §4.1 quality bar.

    Names the consumer (the demanding module behind a sink, when
    `sink_origins` knows it), the key, nearest-key suggestions, the
    available keys, modes in which the key would exist, and a concrete fix.

    Raises
    ------
    ConnectivityError
        Always.
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
    """Find modules/sources that could produce `key` in another primary mode.

    Returns
    -------
    list[str]
        Human-readable notes, deterministically ordered.
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
                if pkey == key or (_is_pattern(pkey) and _pattern_matches(pkey, key)):
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
# mode/sink plumbing and the all-modes-dead check (design §1 principle 10)
# ---------------------------------------------------------------------------


def _check_primary_mode(mode: Mode) -> None:
    """Reject composite modes — one plan is compiled per primary mode (design §3.1).

    Raises
    ------
    ConfigError
        If `mode` is not one of `PRIMARY_MODES`.
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

    Returns
    -------
    list[str] | None
        Sorted sink keys for `mode`, or None when demand is unknown.
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
    """Raise if any configured module is dead in every primary mode (principle 10).

    A module absent from the compiled plan (mode-inactive or demand-pruned)
    is probed in each other primary mode; modes that fail to resolve are
    leniently treated as alive — only provably-dead-everywhere errors.

    Raises
    ------
    AllModesDeadError
        Naming the module, its class, and the fix (design §3.1).
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

    Returns
    -------
    bool
        True if the module is alive in `mode`, or if `mode` cannot be
        resolved at all (deadness must be provable).
    """
    try:
        probe = _resolve(modules, mode, sources, None, sink_keys)
    except GraphError:
        return True
    return name in probe.alive


# ---------------------------------------------------------------------------
# wildcard patterns and plan hashing
# ---------------------------------------------------------------------------


def _is_pattern(key: str) -> bool:
    """Check whether a dotted key contains a wildcard component.

    Returns
    -------
    bool
        True if any component is ``"*"`` or ``"**"``.
    """
    return any(part in _WILDCARD_PARTS for part in key.split(KEY_SEP))


def _pattern_matches(pattern: str, key: str) -> bool:
    """Match a concrete dotted key against a wildcard pattern (design §2.2).

    ``"*"`` matches exactly one component; ``"**"`` matches one or more.

    Returns
    -------
    bool
        True if `key` matches `pattern`.
    """
    return _match_parts(tuple(pattern.split(KEY_SEP)), tuple(key.split(KEY_SEP)))


def _match_parts(pattern: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    if not pattern:
        return not parts
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_match_parts(rest, parts[i:]) for i in range(1, len(parts) + 1))
    if not parts:
        return False
    if head in {"*", parts[0]}:
        return _match_parts(rest, parts[1:])
    return False


def _spec_payload(spec: TensorSpec) -> dict[str, Any]:
    """Canonical JSON-able form of a TensorSpec for hashing.

    Returns
    -------
    dict[str, Any]
        Plain-type payload, stable across runs and machines.
    """
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
    """sha256 over a canonical plan serialisation (design §3.1).

    Step order is significant (it is the execution order); dict keys are
    sorted by the JSON encoder, so insertion order never leaks into the hash.
    The mode name is deliberately NOT part of the payload: the hash is purely
    structural, so two modes with identical step/edge/source structure hash
    equal — that makes the FIT/VAL plan-identity assertion a cheap hash
    comparison (design §11 risk 5). The mode lives in `Plan.mode`; resume
    comparison only uses the FIT-plan hash (design §3.1), so no collision.

    Returns
    -------
    str
        Hex digest, stable across runs and machines for the same config.
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
