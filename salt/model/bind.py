"""Two-phase bind support: `ResolvedSchema` (dotted key -> feature width/fields)
built statically from compiled `Plan`s and handed to ``module.bind``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches

from salt.graph.errors import _SUGGESTION_CUTOFF, GraphError
from salt.graph.planner import Plan
from salt.graph.spec import GraphModule, TensorSpec, is_symbolic_dim
from salt.model.base import SaltModelModule

__all__ = ["BindError", "ResolvedSchema", "bind_all", "materialise_all", "resolve_bind_schema"]


class BindError(GraphError):
    """A width or field lookup failed during two-phase bind."""


@dataclass(frozen=True)
class ResolvedSchema:
    """Resolved per-key widths and fields for ``module.bind``.

    `widths` maps dotted bundle keys to their concrete last-dim size;
    `fields` maps keys to their declared last-dim column names. Keys whose
    width is not statically resolvable (meta leaves, scalar losses,
    data-dependent shapes) are simply absent — `width` raises a `BindError`
    naming the nearest known keys.
    """

    widths: Mapping[str, int]
    fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def width(self, key: str) -> int:
        """Return the concrete feature width (last dim) of a bundle key.

        Raises
        ------
        BindError
            If the key has no statically resolved width, with nearest-key
            suggestions.
        """
        try:
            return self.widths[key]
        except KeyError:
            near = get_close_matches(key, sorted(self.widths), n=3, cutoff=_SUGGESTION_CUTOFF)
            hint = f"; nearest: {', '.join(near)}" if near else ""
            raise BindError(
                f"no statically resolved width for bundle key {key!r} — the key is either "
                f"absent from the compiled plans or its last dim never binds to a concrete "
                f"size (design §2.3){hint}"
            ) from None

    def fields_of(self, key: str) -> tuple[str, ...]:
        """Return the declared last-dim field names of a bundle key, in declaration order.

        Raises
        ------
        BindError
            If no spec for the key declares `fields`.
        """
        try:
            return self.fields[key]
        except KeyError:
            raise BindError(
                f"no declared fields for bundle key {key!r} — column names come from the "
                f"producing declaration (e.g. the dataset-side Features variables, "
                f"design §2.4/§6.2); known field-carrying keys: {sorted(self.fields)}"
            ) from None


def resolve_bind_schema(plans: Plan | Iterable[Plan]) -> ResolvedSchema:
    """Build the `ResolvedSchema` from one or more compiled plans.

    Collects every spec observed per key across the given plans (sources,
    producer produces, consumer requires), unifies their shapes positionally
    (symbolic dims union-find, concrete sizes bind them), and records the
    resolved last-dim width and any declared fields per key. Widths are
    config-fixed, so unifying across modes is sound; the planner has already
    rejected genuinely conflicting declarations on connected edges.

    Parameters
    ----------
    plans : Plan | Iterable[Plan]
        Compiled plans, e.g. the FIT and TEST plans of one model.

    Raises
    ------
    BindError
        If two observations bind the same symbolic dim (or the same key's
        last dim) to different concrete sizes.
    """
    if isinstance(plans, Plan):
        plans = (plans,)
    observed: dict[str, list[TensorSpec]] = {}
    for plan in plans:
        for key, spec in plan.sources.items():
            observed.setdefault(key, []).append(spec)
        for step in plan.steps:
            for key, spec in step.produces.items():
                observed.setdefault(key, []).append(spec)
            for key, spec in step.requires.items():
                observed.setdefault(key, []).append(spec)

    dims = _DimBindings()
    for key, specs in observed.items():
        shaped = [s for s in specs if s.shape is not None]
        for i, a in enumerate(shaped):
            for b in shaped[i + 1 :]:
                if len(a.shape) != len(b.shape):  # type: ignore[arg-type]
                    continue  # connected-edge rank conflicts are the planner's error to raise
                for pos, (da, db) in enumerate(zip(a.shape, b.shape, strict=True)):  # type: ignore[arg-type]
                    where = f"key {key!r} dim {pos}"
                    if is_symbolic_dim(da) and isinstance(db, int):
                        dims.bind(str(da), db, where)
                    elif isinstance(da, int) and is_symbolic_dim(db):
                        dims.bind(str(db), da, where)
                    elif is_symbolic_dim(da) and is_symbolic_dim(db):
                        dims.union(str(da), str(db), where)

    # fields define the last dim by name: bind any symbolic last dim observed
    # on a field-carrying key to the declared column count. This covers
    # boundaries that declare fields WITHOUT a shape (rank is reader-inferred)
    # and lets the width propagate to downstream keys sharing the symbol.
    for key, specs in observed.items():
        declared = next((s.fields for s in specs if s.fields is not None), None)
        if declared is None:
            continue
        for spec in specs:
            if spec.shape:
                last = spec.shape[-1]
                if is_symbolic_dim(last):
                    dims.bind(str(last), len(declared), f"key {key!r} declared fields")

    widths: dict[str, int] = {}
    fields: dict[str, tuple[str, ...]] = {}
    for key, specs in observed.items():
        for spec in specs:
            if spec.fields is not None and key not in fields:
                fields[key] = spec.fields
            if spec.shape is None or len(spec.shape) == 0 or key in widths:
                continue
            last = spec.shape[-1]
            if isinstance(last, int):
                widths[key] = last
            else:
                size = dims.size_of(str(last))
                if size is not None:
                    widths[key] = size
        # when no shape observation binds a width (shape=None, rank is
        # reader-inferred), the declared column count IS the width.
        if key not in widths and key in fields:
            widths[key] = len(fields[key])

    # Second phase: modules whose produced width is a FUNCTION of resolved input
    # widths (not a single shared symbol the dim table can unify) contribute it
    # via the optional duck-typed `derived_widths(widths) -> {key: int}` hook
    # (e.g. VectorConcat's ``Dsum = sum(D_i)`` has no concrete edge to bind, so
    # the dim table can't resolve it; the concat module alone knows the sum).
    # Conflicting contributions raise — a derived width disagreeing with an
    # already-resolved one is a real bug, not a silent drop.
    #
    # The derived widths and the symbol back-binding (below) run to a JOINT
    # fixpoint: a derived width on key K may carry a symbolic last dim SHARED
    # with a not-yet-resolved key (e.g. a pool output sharing its input's width
    # symbol). Binding K's symbol to its resolved width then resolves the
    # sharers, which may in turn unlock further derivations.
    while True:
        _apply_derived_widths(plans, widths)
        if not _back_bind_symbols(observed, widths, dims):
            break
    return ResolvedSchema(widths=widths, fields=fields)


def _back_bind_symbols(
    observed: Mapping[str, list[TensorSpec]],
    widths: dict[str, int],
    dims: _DimBindings,
) -> bool:
    """Bind symbolic last dims from resolved key widths, then re-resolve sharers.

    For every key with a known width whose observed specs carry a symbolic last
    dim, bind that symbol to the width. Then resolve any still-unknown key whose
    symbolic last dim is now concrete. Returns True if any new width was
    resolved, so the caller can re-run `derived_widths` (a newly resolved width
    may unlock a derivation).
    """
    for key, width in list(widths.items()):
        for spec in observed.get(key, ()):
            if spec.shape and is_symbolic_dim(spec.shape[-1]):
                dims.bind(str(spec.shape[-1]), width, f"key {key!r} resolved width")
    changed = False
    for key, specs in observed.items():
        if key in widths:
            continue
        for spec in specs:
            if not spec.shape:
                continue
            last = spec.shape[-1]
            if is_symbolic_dim(last):
                size = dims.size_of(str(last))
                if size is not None:
                    widths[key] = size
                    changed = True
                    break
    return changed


def _apply_derived_widths(plans: Iterable[Plan], widths: dict[str, int]) -> None:
    """Let plan-step modules contribute widths derived from resolved input widths.

    Each module exposing a callable ``derived_widths`` is asked for
    ``{produced_key: int}`` given the widths resolved so far (empty dict if
    not yet resolvable). Runs to a FIXPOINT so a chain of derived widths
    resolves regardless of plan order. Raises `BindError` when a derived
    width disagrees with an already-resolved one.
    """
    modules = _unique_derived_modules(plans)
    max_sweeps = len(modules) + 1
    for _ in range(max_sweeps):
        changed = False
        for module in modules:
            for key, size in module.derived_widths(widths).items():  # type: ignore[attr-defined]
                previous = widths.get(key)
                if previous is not None and previous != size:
                    raise BindError(
                        f"module {getattr(module, 'name', '?')!r} derives width {size} for "
                        f"{key!r} but the resolved schema already has {previous} — conflicting "
                        "widths (design §2.2/§6.6)"
                    )
                if previous is None:
                    widths[key] = size
                    changed = True
        if not changed:
            return


def _unique_derived_modules(plans: Iterable[Plan]) -> list[GraphModule]:
    """The plan-step modules that expose a callable ``derived_widths``, deduped by identity.

    First-seen order across the supplied plans (one entry per live module).
    """
    seen: set[int] = set()
    derived: list[GraphModule] = []
    for plan in plans:
        for step in plan.steps:
            module = step.module
            if id(module) in seen:
                continue
            seen.add(id(module))
            if callable(getattr(module, "derived_widths", None)):
                derived.append(module)
    return derived


def bind_all(modules: Mapping[str, SaltModelModule | GraphModule], schema: ResolvedSchema) -> None:
    """Call ``bind(schema)`` on every `SaltModelModule` in `modules`, in dict order.

    `SaltModule._graph_modules` (the production caller argument, see
    `salt.model.saltmodule`) is model-only by construction — but some test
    fixtures call this directly on a per-mode LOCAL module dict with a
    terminal sink folded in (mirroring `SaltModule.compile_mode`'s own fold,
    e.g. an `OnnxExportSink` under `salt.tests.unit.onnx.test_adapter`), so
    this stays an explicit `SaltModelModule`-partitioned direct call, not an
    unconditional one — a sink has no `bind`, exactly as the pre-plan-49
    getattr-discovery silently skipped it. `SaltModelModule.bind` is a
    documented no-op default, so no further discovery is needed for the
    modules the partition DOES call. Bind is config-only — building the
    schema from compiled plans (which are config-derived) keeps that
    property.
    """
    for module in modules.values():
        if isinstance(module, SaltModelModule):
            module.bind(schema)


def materialise_all(modules: Mapping[str, SaltModelModule | GraphModule]) -> None:
    """Call ``materialise()`` on every `SaltModelModule` in `modules`, in dict order.

    Same `SaltModelModule` partition as `bind_all` (see its docstring) — a
    folded sink has no `materialise`. File-touching value loads happen ONLY
    here, before a fresh fit; the caller skips this on checkpoint load
    (values arrive via the state_dict).
    """
    for module in modules.values():
        if isinstance(module, SaltModelModule):
            module.materialise()


class _DimBindings:
    """Union-find over symbolic dims with concrete bindings (mirrors planner `_DimTable`)."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, tuple[int, str]] = {}  # root -> (size, where)

    def _find(self, dim: str) -> str:
        self._parent.setdefault(dim, dim)
        root = dim
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[dim] != root:  # path compression
            self._parent[dim], dim = root, self._parent[dim]
        return root

    def bind(self, dim: str, size: int, where: str) -> None:
        """Bind a symbolic dim to a concrete size; conflicts raise `BindError`."""
        root = self._find(dim)
        previous = self._size.get(root)
        if previous is not None and previous[0] != size:
            raise BindError(
                f"symbolic dim {dim!r} resolves to {previous[0]} at {previous[1]} but "
                f"{size} at {where} — conflicting widths (design §2.2)"
            )
        if previous is None:
            self._size[root] = (size, where)

    def union(self, a: str, b: str, where: str) -> None:
        """Unify two symbolic dims; conflicting concrete bindings raise `BindError`."""
        root_a, root_b = self._find(a), self._find(b)
        if root_a == root_b:
            return
        size_a, size_b = self._size.get(root_a), self._size.get(root_b)
        if size_a is not None and size_b is not None and size_a[0] != size_b[0]:
            raise BindError(
                f"unifying {a!r} with {b!r} at {where}: {a!r} is {size_a[0]} "
                f"(from {size_a[1]}) but {b!r} is {size_b[0]} (from {size_b[1]}) — "
                "conflicting widths (design §2.2)"
            )
        self._parent[root_b] = root_a
        if size_a is None and size_b is not None:
            self._size[root_a] = size_b

    def size_of(self, dim: str) -> int | None:
        """Return the resolved concrete size of a symbolic dim, or None if unbound."""
        entry = self._size.get(self._find(dim))
        return entry[0] if entry is not None else None
