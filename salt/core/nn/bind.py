"""Two-phase bind support: the resolved schema handed to ``module.bind`` (design §2.3, §2.5).

Design §2.3: after ``compile_plans`` the framework calls ``bind(schema)`` on
every module so width-dependent layers can be built from the *resolved*
graph — shape inference replaces YAML-anchor arithmetic, and ``bind`` needs
only the config (checkpoints load on data-less machines). `ResolvedSchema`
is that argument: a static map from dotted bundle key to its feature width
(last dim) and, where declared, its field names.

`resolve_bind_schema` derives it from compiled `Plan`s: every spec observed
for a key (source leaf, producer's produce spec, each consumer's require
spec) is positionally unified, and symbolic dims (``"F:tracks"``) are bound
to the concrete sizes declared elsewhere in the graph — the same union-find
discipline as the planner's shape check (planner.py `_DimTable`), rebuilt
here because the frozen `Plan` does not expose its dim table. Resolution is
static: no data files, no tensors (design principle 8).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches

from salt.core.graph.errors import GraphError
from salt.core.graph.planner import Plan
from salt.core.graph.spec import GraphModule, TensorSpec, is_symbolic_dim

__all__ = ["BindError", "ResolvedSchema", "bind_all", "materialise_all", "resolve_bind_schema"]

_SUGGESTION_CUTOFF = 0.5


class BindError(GraphError):
    """A width or field lookup failed during two-phase bind (design §2.3)."""


@dataclass(frozen=True)
class ResolvedSchema:
    """Resolved per-key widths and fields for ``module.bind`` (design §2.3, §2.5).

    `widths` maps dotted bundle keys to their concrete last-dim size;
    `fields` maps keys to their declared last-dim column names (design §2.2:
    column lookups resolve by name). Keys whose width is not statically
    resolvable (meta leaves, scalar losses, data-dependent shapes) are simply
    absent — `width` raises a `BindError` naming the nearest known keys.
    """

    widths: Mapping[str, int]
    fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def width(self, key: str) -> int:
        """Return the concrete feature width (last dim) of a bundle key.

        Returns
        -------
        int
            The resolved last-dim size.

        Raises
        ------
        BindError
            If the key has no statically resolved width, with nearest-key
            suggestions (design §4.1 quality bar).
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
        """Return the declared last-dim field names of a bundle key.

        Returns
        -------
        tuple[str, ...]
            The column names, in declaration order.

        Raises
        ------
        BindError
            If no spec for the key declares `fields` (design §2.2: the
            dataset-side `Features` declaration is the one place column
            order is defined).
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
    """Build the `ResolvedSchema` from one or more compiled plans (design §2.3).

    Collects every spec observed per key across the given plans (sources,
    producer produces, consumer requires), unifies their shapes positionally
    (symbolic dims union-find, concrete sizes bind them), and records the
    resolved last-dim width and any declared fields per key. Widths are
    config-fixed (design §2.3), so unifying across modes is sound; the
    planner has already rejected genuinely conflicting declarations on
    connected edges.

    Parameters
    ----------
    plans : Plan | Iterable[Plan]
        Compiled plans, e.g. the FIT and TEST plans of one model.

    A `BindError` propagates from the dim table if two observations bind the
    same symbolic dim (or the same key's last dim) to different concrete
    sizes.

    Returns
    -------
    ResolvedSchema
        The resolved widths and fields.
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

    # fields define the last dim by name (design §2.2; TensorSpec enforces
    # len(fields) == concrete last dim): bind any symbolic last dim observed
    # on a field-carrying key to the declared column count. This covers
    # boundaries that declare fields WITHOUT a shape — the dataset-side
    # Features produce (stream rank is reader-inferred, design §6.2) — and
    # lets the width propagate to downstream keys sharing the symbol (e.g.
    # the Normaliser's F:<name>.<stream> on normed.*).
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
        # fields define the last dim by name (design §2.2; TensorSpec enforces
        # len(fields) == concrete last dim): when no shape observation binds a
        # width — e.g. the dataset-side Features boundary declares shape=None
        # because stream rank is reader-inferred (design §6.2) — the declared
        # column count IS the width.
        if key not in widths and key in fields:
            widths[key] = len(fields[key])

    # Second pass: modules whose produced width is a FUNCTION of resolved input
    # widths (not a single shared symbol the dim table can unify) contribute it
    # via the optional duck-typed `derived_widths(widths) -> {key: int}` hook
    # (design §6.6: VectorConcat's ``Dsum`` "unified at bind" — there is no
    # concrete edge to bind ``Dsum``, so the dim table cannot resolve it; the
    # concat module alone knows ``Dsum = sum(D_i)``). The plan steps hold live
    # module references, so no signature change is needed. Generic: any module
    # may derive widths; conflicting contributions raise (a derived width that
    # disagrees with an already-resolved one is a real bug, not a silent drop).
    _apply_derived_widths(plans, widths)
    return ResolvedSchema(widths=widths, fields=fields)


def _apply_derived_widths(plans: Iterable[Plan], widths: dict[str, int]) -> None:
    """Let plan-step modules contribute widths derived from resolved input widths.

    Each unique plan-step module exposing a callable ``derived_widths`` is asked
    for ``{produced_key: int}`` given the widths resolved so far (design §6.6).
    The hook returns an empty dict when its inputs are not yet resolvable (a
    width-less mode), so this is order-insensitive across the supplied plans.

    Run to a FIXPOINT (re-sweep until no width changes), so a CHAIN of derived
    widths — module B's input width is itself derived by module A — resolves
    regardless of plan order: if B is visited before A on the first sweep its
    inputs aren't bound yet and the hook returns ``{}``, but a later sweep
    (after A bound them) re-runs B. A single sweep would leave B's output
    unbound, surfacing only later as a (loud) `ShapeError`; the fixpoint loop
    removes that order-dependence. The shipped configs chain no derived widths
    (so the loop converges after one productive sweep), but the loop keeps the
    surface robust to future chained derivations. Termination: each productive
    sweep binds ≥1 new width and widths are never unbound, so the loop runs at
    most ``num modules + 1`` times.

    Raises
    ------
    BindError
        When a derived width disagrees with an already-resolved width for the
        same key (a genuine inconsistency, surfaced loudly per design §2.2).
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

    Returns
    -------
    list[GraphModule]
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


def bind_all(modules: Mapping[str, GraphModule], schema: ResolvedSchema) -> None:
    """Call ``bind(schema)`` on every module that defines it, in dict order.

    Design §2.3: bind is config-only — building the schema from compiled
    plans (which are config-derived) keeps that property.
    """
    for module in modules.values():
        bind = getattr(module, "bind", None)
        if callable(bind):
            bind(schema)


def materialise_all(modules: Mapping[str, GraphModule]) -> None:
    """Call ``materialise()`` on every module that defines it, in dict order.

    Design §2.3: file-touching value loads happen ONLY here, before a fresh
    fit; the caller skips this on checkpoint load (values arrive via the
    state_dict).
    """
    for module in modules.values():
        materialise = getattr(module, "materialise", None)
        if callable(materialise):
            materialise()


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
        """Bind a symbolic dim to a concrete size; conflicts raise `BindError`.

        Raises
        ------
        BindError
            Naming both observation sites and the conflicting sizes.
        """
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
        """Unify two symbolic dims; conflicting concrete bindings raise `BindError`.

        Raises
        ------
        BindError
            Naming both binding sites and the conflicting sizes.
        """
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
        """Return the resolved concrete size of a symbolic dim, if bound.

        Returns
        -------
        int | None
            The size, or None when the dim never bound to a concrete value.
        """
        entry = self._size.get(self._find(dim))
        return entry[0] if entry is not None else None
