"""The v2 cut vocabulary: one string-form `Cut` compiled to a restricted `eval`
predicate, `Aggregation` (its ``sum(...)`` constituent-axis reductions) and
`GlobalObjectCuts` (sample-axis row eligibility at index-build).
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, NamedTuple

import numpy as np

from salt.graph.errors import ConfigError

__all__ = ["Aggregation", "Cut", "GlobalObjectCuts"]

VALID_FIELD = "valid"
"""The per-constituent validity field every jagged stream carries."""

_FUNCTIONS: dict[str, Any] = {"abs": np.abs, "log": np.log}
"""The only callables a cut expression may invoke besides ``sum``."""


@cache
def _compile(source: str) -> Any:
    """Compile `source` to a code object, cached per string (never pickled)."""
    return compile(source, "<cut>", "eval")


def _eval(source: str, columns: Mapping[str, Any] | np.ndarray) -> Any:
    """Evaluate `source` with NO builtins; a `NameError` becomes a `KeyError` naming the field."""
    names = columns.dtype.names or () if isinstance(columns, np.ndarray) else tuple(columns)
    scope = {**_FUNCTIONS, **{n: columns[n] for n in names}}
    try:
        return eval(_compile(source), {"__builtins__": {}}, scope)
    except NameError as exc:
        raise KeyError(
            f"cut field {exc.name!r} is not available; present: {sorted(names)}. "
            "Add it to the stream so it is read."
        ) from exc


@dataclass(frozen=True)
class Aggregation:
    """One ``sum(...)`` reduction of a stream's constituent axis onto the row axis.

    An `Aggregation` cannot be evaluated inline: the reader reads `fields` off
    `stream`, calls `evaluate`, and serves the result back as the synthetic
    row-axis column `key` — that is what makes ``sum(jets.valid) >= 4``
    expressible as a row cut at all.

    Parameters
    ----------
    source : str
        The call as written, normalised by ``ast.unparse`` (e.g.
        ``"sum(jets.valid)"``).
    stream : str
        The stream whose constituent axis is reduced.
    fields : tuple[str, ...]
        Bare constituent field names read, in first-seen order.
    inner : str
        `source`'s argument with every ``<stream>.`` qualifier stripped.
    """

    source: str
    stream: str
    fields: tuple[str, ...]
    inner: str

    @property
    def key(self) -> str:
        """The synthetic row-axis column this reduction's value is served under."""
        digest = hashlib.sha1(self.source.encode()).hexdigest()[:12]  # noqa: S324 - non-crypto key
        return f"__agg_{digest}__"

    def evaluate(self, columns: Mapping[str, Any]) -> np.ndarray:
        """``(n_rows,)`` sums of `inner` over jagged constituent columns (a predicate counts)."""
        import awkward as ak

        return np.asarray(ak.to_numpy(ak.sum(_eval(self.inner, columns), axis=1)))


class _Rewrite(ast.NodeTransformer):
    """Strip ``<stream>.`` qualifiers; collect fields/streams; sub each ``sum(...)`` for its key."""

    def __init__(self, source: str, *, inner: bool) -> None:
        self.source = source
        self.inner = inner
        self.fields: dict[str, None] = {}
        self.streams: dict[str, None] = {}
        self.aggs: dict[str, Aggregation] = {}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
        if name in _FUNCTIONS:
            return self.generic_visit(node)
        if name != "sum":
            raise ConfigError(
                f"cut {self.source!r}: unknown function {name!r} — only abs, log and sum "
                "may be called"
            )
        if self.inner:
            raise ConfigError(
                f"cut {self.source!r}: sum() nested inside another reduction — a "
                "constituent axis is collapsed once"
            )
        if len(node.args) != 1 or node.keywords:
            raise ConfigError(f"cut {self.source!r}: sum() takes exactly one argument")
        agg = _reduction(self.source, node)
        self.aggs.setdefault(agg.key, agg)
        return ast.Name(id=agg.key, ctx=ast.Load())

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if not isinstance(node.value, ast.Name):
            raise ConfigError(
                f"cut {self.source!r}: {ast.unparse(node)!r} is not a '<stream>.<field>' qualifier"
            )
        self.streams.setdefault(node.value.id)
        self.fields.setdefault(node.attr)
        return ast.Name(id=node.attr, ctx=ast.Load())

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in _FUNCTIONS or node.id == "sum":
            return node
        if self.inner:
            raise ConfigError(
                f"cut {self.source!r}: {node.id!r} needs a '<stream>.' qualifier inside "
                "sum() — the reduction has to name the stream it collapses"
            )
        self.fields.setdefault(node.id)
        return node


def _reduction(source: str, call: ast.Call) -> Aggregation:
    """Build the `Aggregation` for one validated ``sum(...)`` call."""
    text = ast.unparse(call)  # BEFORE the visit: NodeTransformer mutates in place
    rw = _Rewrite(source, inner=True)
    inner = ast.unparse(ast.fix_missing_locations(rw.visit(call.args[0])))
    if not rw.streams:
        raise ConfigError(
            f"cut {source!r}: {text!r} references no field — a reduction must name at "
            "least one '<stream>.<field>'"
        )
    if len(rw.streams) > 1:
        raise ConfigError(
            f"cut {source!r}: {text!r} mixes streams {sorted(rw.streams)} inside one reduction"
        )
    return Aggregation(text, next(iter(rw.streams)), tuple(rw.fields), inner)


class _Parsed(NamedTuple):
    outer: str
    fields: tuple[str, ...]
    aggregations: tuple[Aggregation, ...]


@cache
def _parse(source: str) -> _Parsed:
    """Parse + validate one cut expression string (cached per string)."""
    if not source:
        raise ConfigError("cut expression must be a non-empty string")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ConfigError(f"cut expression {source!r} is not valid syntax: {exc}") from exc
    if not isinstance(tree.body, ast.Compare):
        raise ConfigError(
            f"cut expression {source!r} must be a comparison producing a keep mask, e.g. 'd0 < 3.5'"
        )
    rw = _Rewrite(source, inner=False)
    outer = ast.unparse(ast.fix_missing_locations(rw.visit(tree.body)))
    if not rw.fields and not rw.aggs:
        raise ConfigError(
            f"cut expression {source!r} references no field — a cut must compare at "
            "least one stream field"
        )
    return _Parsed(outer, tuple(rw.fields), tuple(rw.aggs.values()))


@dataclass(frozen=True)
class Cut:
    """One keep predicate: an expression string with exactly one comparison.

    Field names may carry a ``<stream>.`` prefix (stripped); ``abs``/``log``
    are elementwise; ``sum(<predicate or value>)`` reduces a
    ``<stream>.``-qualified constituent axis onto the row (``&`` joins
    parenthesised predicates inside it). NaN fails every ordering/equality
    cut (numpy semantics; only ``!=`` is True vs NaN).

    Parameters
    ----------
    expr : str
        The cut expression, e.g. ``"pt >= 20000"`` or ``"sum(jets.valid) >= 4"``.

    Raises
    ------
    ConfigError
        On a syntax error, a non-comparison root, an unknown function, a
        malformed reduction, or an expression referencing no field.
    """

    expr: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "expr", self.expr.strip())
        _parse(self.expr)  # validate eagerly

    @classmethod
    def parse(cls, spec: Cut | str) -> Cut:
        """Normalise a config entry (a `Cut` or an expression string) to a `Cut`.

        Returns
        -------
        Cut
            The normalised cut.

        Raises
        ------
        ConfigError
            When `spec` is neither a `Cut` nor a string.
        """
        if isinstance(spec, Cut):
            return spec
        if isinstance(spec, str):
            return cls(expr=spec)
        raise ConfigError(f"cut entry {spec!r} is not a cut expression string")

    @property
    def fields(self) -> tuple[str, ...]:
        """The bare ROW-axis field names this cut reads (reduced fields excluded)."""
        return _parse(self.expr).fields

    @property
    def aggregations(self) -> tuple[Aggregation, ...]:
        """The constituent-axis reductions this cut needs precomputed."""
        return _parse(self.expr).aggregations

    def mask(self, source: Mapping[str, Any] | np.ndarray) -> Any:
        """Evaluate to a keep mask (True = PASSES) over a structured array or column mapping
        carrying every referenced field (plus, for a reduction, its `Aggregation.key` column).
        """
        return _eval(_parse(self.expr).outer, source)


def _parse_cuts(specs: Sequence[Cut | str], where: str) -> tuple[Cut, ...]:
    """Normalise a config cut sequence to `Cut` instances, naming `where` on failure.

    Returns
    -------
    tuple[Cut, ...]
        The normalised cuts.

    Raises
    ------
    ConfigError
        On an entry that is not a valid cut.
    """
    out = []
    for spec in specs:
        try:
            out.append(Cut.parse(spec))
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
    return tuple(out)


@dataclass(frozen=True)
class GlobalObjectCuts:
    """Sample-axis row eligibility (AND of `global_cuts`), applied once at index-build.

    The reader-uniform ``cuts:`` surface: cuts here drop whole ROWS (jets for a
    jet reader, events for an event reader) and so change ``__len__``. Per-
    constituent filtering is `ConstituentSelection`, never this.

    Parameters
    ----------
    global_cuts : Sequence[Cut | str], optional
        AND-combined cuts. Entries normalise through `Cut.parse`, so YAML may
        write plain strings.

    Raises
    ------
    ConfigError
        On an entry that is not a cut.
    """

    global_cuts: tuple[str, ...] = ()
    _cuts: tuple[Cut, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        cuts = _parse_cuts(tuple(self.global_cuts), "GlobalObjectCuts.global_cuts")
        object.__setattr__(self, "global_cuts", tuple(c.expr for c in cuts))
        object.__setattr__(self, "_cuts", cuts)

    def eligible(self, rows: np.ndarray) -> np.ndarray:
        """Bool ``(N_rows,)`` mask: True where every cut passes `rows` (all-True with no cuts);
        `KeyError` propagates from `Cut.mask` for a field absent from `rows`.
        """
        keep = np.ones(len(rows), dtype=bool)
        for c in self._cuts:
            keep &= c.mask(rows)
        return keep

    def fields(self) -> tuple[str, ...]:
        """Bare ROW-axis field names referenced, deduplicated in first-seen order (the
        reader's read-at-index-build contract; reduced fields are `aggregations`, not here).
        """
        seen: dict[str, None] = {}
        for c in self._cuts:
            for f in c.fields:
                seen.setdefault(f, None)
        return tuple(seen)

    def aggregations(self) -> tuple[Aggregation, ...]:
        """Constituent-axis reductions the cuts need precomputed, deduplicated by key (the
        reader evaluates each and adds it to the row record under `Aggregation.key` first).
        """
        seen: dict[str, Aggregation] = {}
        for c in self._cuts:
            for agg in c.aggregations:
                seen.setdefault(agg.key, agg)
        return tuple(seen.values())
