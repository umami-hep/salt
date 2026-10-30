"""Plan compilation for the salt v2 graph kernel.

Flattens `declare_io` to dotted ports, checks connectivity/kinds/shapes,
demand-prunes per mode against the sinks, and emits a frozen, hashed `Plan`.
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

from salt.graph.errors import (
    SUGGESTION_CUTOFF,
    AllModesDeadError,
    ConfigError,
    ConnectivityError,
    CycleError,
    GraphError,
    KindError,
    ShapeError,
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
    "deadcode",
]

SOURCES = "<sources>"  # edge producer sentinel: key provided a priori by the framework
SINKS = "<sinks>"  # edge consumer sentinel: key demanded by a sink
_PRIMARY_NAMES = "/".join(m.name for m in PRIMARY_MODES)  # "FIT/VAL/TEST/ONNX" in error texts

Sinks: TypeAlias = "Iterable[str] | Mapping[Mode, Iterable[str]] | None"
"""Sink keys: flat iterable (compiled mode only) or per-mode mapping."""

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
    ``Bundle.merge(expected=...)``. `rewrites` is the subset of `produces`
    this step overwrites in place — the executor relaxes write-once for
    exactly these keys. `compile_plan` builds them as read-only
    `MappingProxyType` views, so the frozen plan is genuinely immutable
    (mutation raises TypeError); hashing uses `name` only.
    """

    name: str
    module: GraphModule = field(compare=False, repr=False)
    requires: Mapping[str, TensorSpec] = field(default_factory=dict, hash=False)
    produces: Mapping[str, TensorSpec] = field(default_factory=dict, hash=False)
    rewrites: Mapping[str, TensorSpec] = field(default_factory=dict, hash=False)


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
    source leaves. Severity: unconsumed ``preds.*`` in TEST is ``"error"``
    (a prediction silently vanishing from eval files); ``preds.*`` in
    FIT/VAL and modules pruned from the ONNX plan are ``"info"`` (expected,
    never promoted by ``--strict``); everything else is ``"warning"``. The
    CLI exits non-zero on any error-level finding.
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
    sinks = _checked_mode_sinks(mode, sinks)
    res = _resolve(modules, mode, sources, schema, _sinks_for(sinks, mode, mode), sink_origins)
    _check_all_modes_dead(modules, mode, sources, sinks, res)
    return _assemble_plan(res)


def _assemble_plan(res: _Resolution) -> Plan:
    """Topo-sort a resolved graph and freeze it into a hashed `Plan`.

    The tail of `compile_plan`: deterministic Kahn order, immutable
    `PlanStep`/`Edge` views, and the structural `plan_hash` (each `TensorSpec`
    serialised via `_spec_payload`).
    """
    steps = tuple(
        PlanStep(
            name=node.name,
            module=node.module,
            requires=MappingProxyType(dict(node.bound)),
            produces=MappingProxyType({**node.all_produces(), **node.rewrites}),
            rewrites=MappingProxyType(dict(node.rewrites)),
        )
        for node in (res.alive[name] for name in _topo_order(res))
    )
    edges = tuple(sorted(res.edges))
    sources_out: Mapping[str, Any] = MappingProxyType(dict(res.sources))
    plan_hash = _plan_hash(steps, edges, sources_out)
    return Plan(mode=res.mode, steps=steps, edges=edges, sources=sources_out, plan_hash=plan_hash)


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

    Severity is per `DeadOutput`. The per-task ``expose:`` opt-out gates
    ports out of a mode before this runs, so an opted-out ``preds.*`` prunes
    its module (warning) rather than hitting the TEST error. Returns
    deterministically ordered findings; empty when everything is consumed.
    """
    sinks = _checked_mode_sinks(mode, sinks)
    res = _resolve(modules, mode, sources, schema, _sinks_for(sinks, mode, mode))
    consumed = _consumed_keys(res.edges)
    # ONNX export surface is writer-declared and may legitimately be
    # narrower than eval — info-level, never promoted by --strict
    pruned_suffix = ""
    if mode == Mode.ONNX:
        pruned_suffix = " (narrowed out of the writer-declared export surface — legitimate)"
    out: list[DeadOutput] = []
    for name in sorted(res.pruned):
        # a conversion producer (salt.outputs node) pruned in a non-ONNX mode is
        # by design (its export sink is inactive there) — demote to info so
        # configs carrying ONNX nodes still pass --strict --mode test
        info = mode == Mode.ONNX or isinstance(getattr(modules.get(name), "output_key", None), str)
        reason = f"module pruned in mode {mode.name}: {res.pruned[name]}{pruned_suffix}"
        out.append(DeadOutput(name, "*", reason, severity="info" if info else "warning"))
    unconsumed_reason = f"produced but never consumed in mode {mode.name}"
    preds_in_test_reason = (
        f"{unconsumed_reason} — an unconsumed preds.* port in TEST means a computed prediction "
        "is never persisted; wire a writer or drop the port"
    )
    preds_in_training_reason = (
        f"{unconsumed_reason} (normal: no configured metric callback consumes this prediction)"
    )
    for name, node in res.alive.items():
        used = consumed.get(name, set())
        for key in {**node.all_produces(), **node.rewrites}:
            if key in used:
                continue
            if key.partition(KEY_SEP)[0] != "preds":
                out.append(DeadOutput(name, key, unconsumed_reason))
            elif mode == Mode.TEST:
                out.append(DeadOutput(name, key, preds_in_test_reason, severity="error"))
            else:
                out.append(DeadOutput(name, key, preds_in_training_reason, severity="info"))
    source_reason = f"source never consumed in mode {mode.name}"
    used = consumed.get(SOURCES, set())
    out.extend(DeadOutput(SOURCES, key, source_reason) for key in res.sources if key not in used)
    return out


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
    rewrites: dict[str, TensorSpec] = field(default_factory=dict)

    def all_produces(self) -> dict[str, TensorSpec]:
        """Concrete plus narrowed produces, as a fresh flat dict."""
        return {**self.produces, **self.narrowed}


@dataclass
class _Resolution:
    """Internal result of resolving one mode's graph."""

    mode: Mode
    sources: dict[str, TensorSpec]
    alive: dict[str, _Node]  # post demand-prune, in config order
    edges: list[Edge]  # among alive modules (+ SOURCES/SINKS), sorted
    pruned: dict[str, str]  # demand-pruned module -> reason


def _resolve(
    modules: dict[str, GraphModule],
    mode: Mode,
    sources: NestedSpec,
    schema: Collection[str] | None,
    sink_keys: list[str] | None,
    sink_origins: Mapping[str, str] | None = None,
) -> _Resolution:
    """Resolve one mode's graph: narrow wildcards, build edges, check, prune."""
    src, wildcard = _concrete_active(sources, mode)
    if wildcard is not None:
        raise ConfigError(
            f"source key {wildcard!r} may not contain wildcards — sources are concrete"
        )
    nodes = _collect_nodes(modules, mode)
    producer_of = _concrete_producers(src, nodes, mode)
    sink_list = _checked_sink_keys(sink_keys)
    sinks = sink_list or []
    _narrow_wildcards(nodes, producer_of, _collect_demand(nodes, sinks), schema, mode, sink_origins)
    edges = _build_edges(nodes, producer_of, src, sinks, mode, modules, sources, sink_origins)
    _check_wildcard_self_feed(nodes, edges, mode)
    if sink_list is None:
        alive, pruned = dict(nodes), {}
    else:
        needed = _demand_closure(nodes, sink_list)
        reason = f"outputs reach no sink in mode {mode.name}"
        pruned = {name: reason for name in nodes if name not in needed}
        alive = {name: node for name, node in nodes.items() if name in needed}
        # SOURCES only ever produces and SINKS only ever consumes
        kept = {*alive, SOURCES, SINKS}
        edges = [edge for edge in edges if edge.producer in kept and edge.consumer in kept]
        _drop_unconsumed_narrowed(alive, edges)
    return _Resolution(mode=mode, sources=src, alive=alive, edges=sorted(edges), pruned=pruned)


def _concrete_active(nested: NestedSpec, mode: Mode) -> tuple[dict[str, TensorSpec], str | None]:
    """Flatten `nested` to its `mode`-active leaves; also return its first wildcard key
    (in flatten order, active or not — the caller rejects it), or None.
    """
    flat = flatten_spec(nested)
    wildcard = next((key for key in flat if _has_wildcard(key)), None)
    return {key: spec for key, spec in flat.items() if spec.active_in(mode)}, wildcard


def _collect_nodes(modules: dict[str, GraphModule], mode: Mode) -> dict[str, _Node]:
    """Build per-module nodes with mode-active flattened ports, in config
    declaration order (the topological tie-break, recorded as `_Node.rank`).

    Returns active nodes. Raises `ConfigError` on protocol/name violations,
    reserved names, or wildcard misuse.
    """
    nodes: dict[str, _Node] = {}
    for rank, (name, module) in enumerate(modules.items()):
        if name in {SOURCES, SINKS}:
            raise ConfigError(
                f"module instance name {name!r} collides with a reserved planner sentinel "
                f"({SOURCES!r}/{SINKS!r}) — rename the module"
            )
        if not isinstance(module, GraphModule):
            raise ConfigError(
                f"module {name!r} ({type(module).__name__}) does not implement the GraphModule "
                "protocol (name + declare_io)"
            )
        if module.name != name:
            raise ConfigError(
                f"module mapped at key {name!r} declares name={module.name!r} — instance names "
                "must match their config keys"
            )
        io = module.declare_io(mode)
        # framework-internal seam, not user API. TODO: bind the capability to
        # shipped code so user classes cannot grant it to themselves.
        allow_wildcards = bool(getattr(module, "allow_wildcards", False))
        requires, wildcard = _concrete_active(io.requires, mode)
        if wildcard is not None:
            raise ConfigError(
                f"module {name!r} declares wildcard require {wildcard!r} — only framework "
                "producers may declare patterns, and only in produces"
            )
        active = {k: s for k, s in flatten_spec(io.produces).items() if s.active_in(mode)}
        patterns = {key: spec for key, spec in active.items() if _has_wildcard(key)}
        produces = {key: spec for key, spec in active.items() if key not in patterns}
        if patterns and not allow_wildcards:  # the first mode-active wildcard produce
            raise ConfigError(
                f"module {name!r} declares wildcard produces {next(iter(patterns))!r} but is "
                "not a framework wildcard producer (allow_wildcards is not set) — user "
                "modules may not declare wildcards"
            )
        rewrites, wildcard = _concrete_active(getattr(io, "rewrites", {}), mode)
        if wildcard is not None:
            raise ConfigError(
                f"module {name!r} declares wildcard rewrite {wildcard!r} — rewrites are concrete"
            )
        for key in rewrites:
            if key not in requires:
                raise ConfigError(
                    f"module {name!r} rewrites {key!r} but does not require it — a rewriter "
                    "reads the key it replaces"
                )
            if key in produces or key in patterns:
                raise ConfigError(
                    f"module {name!r} both produces and rewrites {key!r} — a key is produced "
                    "OR rewritten, never both"
                )
        if requires or produces or patterns or rewrites:
            nodes[name] = _Node(name, rank, module, requires, produces, patterns, rewrites=rewrites)
    return nodes


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
                    "— every bundle key must have exactly one producer"
                )
            producer_of[key] = name
    return producer_of


def _checked_sink_keys(sink_keys: list[str] | None) -> list[str] | None:
    """Validate sink keys are well-formed concrete dotted keys.

    Raises `ConfigError` if a sink key contains a wildcard component.
    """
    for key in sink_keys or ():
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
    return (sink_origins or {}).get(key, "the configured sinks")


def _narrow_wildcards(
    nodes: dict[str, _Node],
    producer_of: dict[str, str],
    demand: dict[str, list[str]],
    schema: Collection[str] | None,
    mode: Mode,
    sink_origins: Mapping[str, str] | None = None,
) -> None:
    """Narrow wildcard patterns against concrete demand.

    Concrete producers beat wildcards; narrowed keys are validated against
    `schema` when given; the result is written into each node's `narrowed`
    dict and frozen into the plan.
    Raises `ConnectivityError` when two wildcard producers match one
    demanded key, or a narrowed key is not in the schema.
    """
    wildcard_owner: dict[str, str] = {}
    for name, node in nodes.items():
        for pattern, spec in sorted(node.patterns.items()):
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
                        "producer"
                    )
                if schema is not None and key not in schema:
                    consumers = ", ".join(
                        _describe_consumer(c, key, sink_origins) for c in demand[key]
                    )
                    near = get_close_matches(key, sorted(schema), n=3, cutoff=SUGGESTION_CUTOFF)
                    hint = f"\n  nearest schema keys: {', '.join(near)}" if near else ""
                    raise ConnectivityError(
                        f"[mode={mode.name}] key {key!r} (demanded by {consumers}) is not in "
                        f"the declared schema — {name!r} can only serve schema-backed keys "
                        f"(wildcard {pattern!r}).{hint}\n"
                        f"  fix: correct the key in the demanding module's config"
                    )
                wildcard_owner[key] = name
                node.narrowed[key] = spec
    for name, node in nodes.items():
        producer_of.update(dict.fromkeys(node.narrowed, name))


def _rewriters(nodes: dict[str, _Node], producer_of: dict[str, str], mode: Mode) -> dict[str, str]:
    """Map each rewritten key to its single rewriter module name.

    Raises `ConnectivityError` if a rewrite key has no producer to rewrite, or
    if two modules rewrite the same key.
    """
    rewriter_of: dict[str, str] = {}
    for name, node in nodes.items():
        for key in node.rewrites:
            if key not in producer_of:
                raise ConnectivityError(
                    f"[mode={mode.name}] module {name!r} rewrites {key!r} but no module or "
                    "source produces it"
                )
            owner = rewriter_of.get(key)
            if owner is not None:
                raise ConnectivityError(
                    f"[mode={mode.name}] key {key!r} has two rewriters: {owner!r} and {name!r} "
                    "— at most one module may rewrite a key"
                )
            rewriter_of[key] = name
    return rewriter_of


def _produced_spec(node: _Node, key: str) -> TensorSpec:
    """Look up a produced key's spec, preferring a node's own rewrite over its produces."""
    return node.rewrites[key] if key in node.rewrites else node.all_produces()[key]


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

    Every bound edge is kind-matched and shape-unified (`_unify_edge`). Missing
    producers raise `ConnectivityError` (via `_raise_missing_producer`);
    unification conflicts raise `ShapeError`; kind mismatches raise
    `KindError`. Returns the resolved edges (pre-prune); optional-but-absent
    requires bind no edge, and bound requires are recorded on each node.

    Rewriters are resolved first (`_rewriters`, key -> rewriter module name):
    any consumer OTHER than the rewriter itself binds the rewriter instead of
    the original producer — the rewriter's own require edge still binds the
    original producer (it reads the key it replaces). A rewriter's rewrite spec
    may narrow but never widen a concrete dim of the producer's spec (`ShapeError`).
    """
    rewriter_of = _rewriters(nodes, producer_of, mode)
    dims = DimTable(mode)
    edges: list[Edge] = []
    for name, node in nodes.items():
        for key, spec in node.requires.items():
            producer = producer_of.get(key)
            if producer is None:
                if spec.optional:
                    continue
                _raise_missing_producer(name, key, mode, producer_of, modules, sources)
            rewriter = rewriter_of.get(key)
            if rewriter is not None and rewriter != name:
                producer = rewriter
            node.bound[key] = spec
            edge = Edge(producer, key, name)
            edges.append(edge)
            pspec = src[key] if producer == SOURCES else _produced_spec(nodes[producer], key)
            if rewriter == name:  # the rewriter binds the original producer:
                # a rewrite may narrow, never widen
                rspec = node.rewrites[key]
                if pspec.shape and rspec.shape and len(pspec.shape) == len(rspec.shape):
                    for i, (pdim, rdim) in enumerate(zip(pspec.shape, rspec.shape, strict=True)):
                        if isinstance(pdim, int) and isinstance(rdim, int) and rdim > pdim:
                            raise ShapeError(
                                f"[mode={mode.name}] module {name!r} rewrites {key!r} to "
                                f"{rspec.shape} but producer {producer!r} serves {pspec.shape} "
                                "— a rewrite (e.g. a ConstituentSelection pad_max) cannot "
                                f"exceed the reader's width at dim {i}; raise the reader's "
                                "pad_max instead"
                            )
            if pspec.kind != spec.kind:
                raise KindError(
                    f"[mode={mode.name}] module {name!r} port {key!r} expects "
                    f"kind={spec.kind!r} but producer {producer!r} provides "
                    f"kind={pspec.kind!r}"
                )
            _unify_edge(edge, pspec, spec, dims, mode)
    for key in sink_keys:
        producer = producer_of.get(key)
        if producer is None:
            _raise_missing_producer(SINKS, key, mode, producer_of, modules, sources, sink_origins)
        edges.append(Edge(rewriter_of.get(key, producer), key, SINKS))
    return edges


def _drop_unconsumed_narrowed(alive: dict[str, _Node], edges: list[Edge]) -> None:
    """Drop narrowed wildcard keys whose only demand was removed by pruning.

    Narrowing runs against pre-prune demand: once a sole consumer is
    pruned, its key must not stay in the wildcard producer's plan
    (`Bundle.merge` enforces the frozen key set exactly). Keys with any
    surviving edge are kept.
    """
    consumed = _consumed_keys(edges)
    for name, node in alive.items():
        used = consumed.get(name, set())
        for key in [key for key in node.narrowed if key not in used]:
            del node.narrowed[key]


def _consumed_keys(edges: Iterable[Edge]) -> dict[str, set[str]]:
    """Map each edge producer (module name or `SOURCES`) to the keys it serves."""
    consumed: dict[str, set[str]] = {}
    for edge in edges:
        consumed.setdefault(edge.producer, set()).add(edge.key)
    return consumed


def _is_terminal_consumer(module: GraphModule) -> bool:
    """Whether a no-current-produces module is a genuine terminal consumer.

    True only if `module` has no CONCRETE produced port AND no rewrite
    active in ANY primary mode — covers true terminal consumers (writers)
    and demand-driven wildcard producers (a pattern port narrowing to
    nothing here). An `expose: [fit, val]` task that's merely inactive in
    this mode still has a concrete port elsewhere, so it's a prunable
    producer, not a sink — the opt-out mechanism relies on this. A rewriter
    is never a terminal consumer: its rewrite key is live iff someone else
    demands it (see `_demand_closure`).
    """
    for m in PRIMARY_MODES:
        io = module.declare_io(m)
        for key, spec in flatten_spec(io.produces).items():
            if not _has_wildcard(key) and spec.active_in(m):
                return False
        for spec in flatten_spec(getattr(io, "rewrites", {})).values():
            if spec.active_in(m):
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
        if not node.all_produces() and not node.rewrites and _is_terminal_consumer(node.module):
            needed.add(name)
            needed_keys |= set(node.bound)
    changed = True
    while changed:
        changed = False
        for name, node in nodes.items():
            if name in needed:
                continue
            if needed_keys & (set(node.all_produces()) | set(node.rewrites)):
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
        fed = [edge for edge in edges if edge.producer == name and edge.consumer in nodes]
        starts = [(edge.consumer, edge.key) for edge in fed if edge.key in node.narrowed]
        path = _path_back(adj, starts, name)
        if path is not None:
            chain = name + "".join(f" -({key})-> {module}" for key, module in path)
            raise CycleError(
                f"[mode={mode.name}] wildcard producer {name!r}: narrowed outputs transitively "
                f"feed its own inputs: {chain}"
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
    adj = _module_adjacency(res.edges, alive)  # adj[producer][consumer], distinct pairs
    indegree = dict.fromkeys(alive, 0)
    for consumers in adj.values():
        for consumer in consumers:
            indegree[consumer] += 1
    heap = [(alive[name].rank, name) for name, count in indegree.items() if count == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, name = heapq.heappop(heap)
        order.append(name)
        for consumer in sorted(adj.get(name, {})):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(heap, (alive[consumer].rank, consumer))
    if len(order) != len(alive):
        chain = _find_cycle(set(alive) - set(order), adj)
        raise CycleError(
            f"[mode={res.mode.name}] dependency cycle: {chain} — a module's outputs may not "
            "feed its own inputs; cycles are always a config error"
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
                cycle = names[names.index(nxt) :]
                links = "".join(f" -({adj[a][b]})-> {b}" for a, b in pairwise(cycle))
                return f"{cycle[0]}{links} -({key})-> {nxt}"
            if state[nxt] == 0:
                state[nxt] = 1
                stack.append((nxt, iter(sorted(adj.get(nxt, {}).items()))))
    return "<no cycle found>"  # pragma: no cover - residual implies a cycle


# ---------------------------------------------------------------------------
# kind / shape / dtype checks
# ---------------------------------------------------------------------------


class DimTable:
    """Union-find over symbolic dims with concrete bindings."""

    def __init__(self, mode: Mode | None = None, error: type[GraphError] = ShapeError) -> None:
        """Create an empty table; conflicts raise `error`, tagged ``[mode=X]`` if `mode`."""
        self._tag = f"[mode={mode.name}] " if mode is not None else ""
        self._error = error
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
        """Bind a symbolic dim to a concrete size; raises the table's error on
        conflict, naming both endpoints and the conflicting sizes.
        """
        root = self._find(dim)
        previous = self._size.get(root)
        if previous is not None and previous[0] != size:
            raise self._error(
                f"{self._tag}symbolic dim {dim!r} is {previous[0]} at "
                f"{previous[1]} but {size} at {endpoint} — conflicting sizes"
            )
        if previous is None:
            self._size[root] = (size, endpoint)

    def union(self, a: str, b: str, endpoint: str) -> None:
        """Unify two symbolic dims; raises the table's error on conflicting
        concrete bindings, naming both binding endpoints and the sizes.
        """
        root_a, root_b = self._find(a), self._find(b)
        if root_a == root_b:
            return
        size_a, size_b = self._size.get(root_a), self._size.get(root_b)
        if size_a is not None and size_b is not None and size_a[0] != size_b[0]:
            raise self._error(
                f"{self._tag}unifying {a!r} with {b!r} at {endpoint}: "
                f"{a!r} is {size_a[0]} (from {size_a[1]}) but {b!r} is {size_b[0]} "
                f"(from {size_b[1]}) — conflicting sizes"
            )
        self._parent[root_b] = root_a
        if size_a is None and size_b is not None:
            self._size[root_a] = size_b

    def size_of(self, dim: str) -> int | None:
        """Return the resolved concrete size of a symbolic dim, or None if unbound."""
        entry = self._size.get(self._find(dim))
        return entry[0] if entry is not None else None


def _unify_edge(
    edge: Edge, pspec: TensorSpec, cspec: TensorSpec, dims: DimTable, mode: Mode
) -> None:
    """Unify one edge's producer/consumer specs (shape rank, dims, dtype);
    raises `ShapeError` on rank, size, symbolic-binding, or dtype conflict.
    """
    key, producer, consumer = edge.key, edge.producer, edge.consumer
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
        origins = sink_origins or {}
        demander = f" (demanded by {origins[key]})" if key in origins else ""
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
    near = get_close_matches(key, available, n=3, cutoff=SUGGESTION_CUTOFF)
    if near:
        lines.append(f"  did you mean: {', '.join(repr(k) for k in near)}?")
    if available:
        shown = available[:12]
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
    for other in (m for m in PRIMARY_MODES if m != mode):
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


def _checked_mode_sinks(mode: Mode, sinks: Sinks) -> Sinks:
    """Reject composite modes (one plan per primary mode), then materialise a flat
    `sinks` iterable as a list (a per-mode mapping and None pass through).

    Raises `ConfigError` if `mode` is not one of `PRIMARY_MODES`.
    """
    if mode not in PRIMARY_MODES:
        raise ConfigError(
            f"plans are compiled per primary mode ({_PRIMARY_NAMES}); got {mode!r} — compile one "
            "plan per mode"
        )
    return list(sinks) if sinks is not None and not isinstance(sinks, Mapping) else sinks


def _sinks_for(sinks: Sinks, mode: Mode, compiled_mode: Mode) -> list[str] | None:
    """Resolve the sink keys applying to `mode`; None means demand is unknown.

    A flat iterable applies only to the compiled mode; a ``{Mode: keys}``
    mapping applies to every primary mode its (possibly composite) keys
    overlap. Modes with no overlapping mapping entry have unknown demand —
    they are never demand-pruned, keeping the all-modes-dead check lenient.
    Returns sorted sink keys for `mode`, or None when demand is unknown.
    """
    if isinstance(sinks, Mapping):
        if not any(mode_key & mode for mode_key in sinks):
            return None
        return sorted({key for mode_key, keys in sinks.items() if mode_key & mode for key in keys})
    return None if sinks is None or mode != compiled_mode else sorted(set(sinks))


def _check_all_modes_dead(
    modules: dict[str, GraphModule], mode: Mode, sources: NestedSpec, sinks: Sinks, res: _Resolution
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
            raise AllModesDeadError(
                f"module {name!r} ({type(modules[name]).__name__}) is dead in every mode "
                f"({_PRIMARY_NAMES}): "
                "no port is active, or its outputs reach no sink in any mode — a configured "
                "module must do something; remove it or wire a consumer"
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


def _spec_payload(spec: TensorSpec) -> dict[str, Any]:
    """Canonical JSON-able form of a `TensorSpec` leaf for the structural plan hash."""
    return {
        "shape": list(spec.shape) if spec.shape is not None else None,
        "dtype": spec.dtype,
        "kind": spec.kind,
        "modes": [m.name for m in PRIMARY_MODES if spec.active_in(m)],
        "optional": spec.optional,
        "fields": list(spec.fields) if spec.fields is not None else None,
    }


def _specs_payload(specs: Mapping[str, TensorSpec]) -> dict[str, Any]:
    """`_spec_payload` over a flat ``{dotted_key: spec}`` mapping."""
    return {k: _spec_payload(v) for k, v in specs.items()}


def _plan_hash(
    steps: tuple[PlanStep, ...], edges: tuple[Edge, ...], sources: Mapping[str, TensorSpec]
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
                "requires": _specs_payload(step.requires),
                "produces": _specs_payload(step.produces),
                **({"rewrites": _specs_payload(step.rewrites)} if step.rewrites else {}),
            }
            for step in steps
        ],
        "edges": [[edge.producer, edge.key, edge.consumer] for edge in edges],
        "sources": _specs_payload(sources),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
