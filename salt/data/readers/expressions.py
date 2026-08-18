"""`Expression` — an ``ast``-whitelist evaluator for cut predicates: numeric
arithmetic over stream fields, a small function whitelist (``abs``/``log``
elementwise, ``sum``/``count`` reducing a constituent axis), ``&``/``|`` between
predicates inside a reduction, and exactly one top-level comparison, evaluated
vectorised with numpy. No ``eval``, no new dependency.
"""

from __future__ import annotations

import ast
import hashlib
import operator
from collections.abc import Iterator, Mapping
from functools import cache
from typing import Any

import numpy as np

from salt.graph.errors import ConfigError

__all__ = ["Aggregation", "Expression", "parse_expression"]

_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_COMBINATORS: dict[type[ast.operator], Any] = {
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}
"""``&`` / ``|`` — the only operators that may join two comparisons, and only
where a comparison is legal in the first place (inside a reduction). NOT ``and``
/ ``or``: those call ``__bool__`` and would collapse an array to one truth value.
Python binds ``&`` TIGHTER than a comparison, so the operands must be
parenthesised — ``a > 1 & b < 2`` parses as ``a > (1 & b) < 2`` and is rejected as
a chained comparison."""

_UNARYOPS: dict[type[ast.unaryop], Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_COMPARES: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_SYMBOLS: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}
_ELEMENTWISE: dict[str, Any] = {"abs": np.abs, "log": np.log}
"""Shape-preserving functions: applied where they stand, on whatever axis the
sub-expression already has. Their absence is why ``|eta| < 4.5`` has to be
spelled as two cuts and a b-tag discriminant as a ratio against ``exp(D)``."""

_AGGREGATIONS = ("sum", "count")
"""Reductions of a stream's CONSTITUENT axis onto the row axis: ``sum(x)`` adds
``x`` up over the constituents of each row (a bool predicate counts its passes),
``count(x)`` is that row's constituent multiplicity. Their argument must name a
stream explicitly (``sum(jets.valid)``) — the reduction has to know which
stream's axis it collapses, and nothing else in an expression does."""


def _name_path(node: ast.expr) -> str:
    """The dotted name a `Name`/`Attribute` chain spells.

    Returns
    -------
    str
        The dotted name.

    Raises
    ------
    ConfigError
        When the node is not a pure name path (an attribute on an expression).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_path(node.value)}.{node.attr}"
    raise ConfigError(
        "cut expression: attribute access is only allowed to spell a dotted field name "
        f"(``<stream>.<field>``), got {ast.dump(node)}"
    )


def _bare_field(node: ast.expr) -> str:
    """The bare field a name node addresses (leading ``<stream>.`` prefix stripped).

    Returns
    -------
    str
        The bare field name.
    """
    return _name_path(node).rsplit(".", 1)[-1]


def _call_name(node: ast.Call, source: str) -> str:
    """The whitelisted function a `Call` node invokes.

    Returns
    -------
    str
        The function name.

    Raises
    ------
    ConfigError
        When the callee is not a bare whitelisted name, or the call is not a
        single positional argument.
    """
    known = sorted([*_ELEMENTWISE, *_AGGREGATIONS])
    if not isinstance(node.func, ast.Name) or node.func.id not in known:
        called = ast.unparse(node.func)
        raise ConfigError(
            f"cut expression {source!r}: {called!r} is not a whitelisted function — "
            f"expected one of {known}"
        )
    if len(node.args) != 1 or node.keywords:
        raise ConfigError(
            f"cut expression {source!r}: {node.func.id}() takes exactly one positional "
            f"argument, got {len(node.args)} positional and {len(node.keywords)} keyword"
        )
    return node.func.id


def _check(node: ast.expr, source: str, *, allow_compare: bool, in_agg: bool) -> None:
    """Reject every node outside the whitelist (arithmetic, calls, one comparison, fields, numbers).

    ``allow_compare`` is True only at the root and directly inside an aggregation
    argument: a comparison anywhere else is either a chained comparison or a
    parenthesised one, and both are ambiguous as cuts. ``in_agg`` bans nested
    reductions — the constituent axis is collapsed once.

    Raises
    ------
    ConfigError
        On a disallowed node, operator, chained/nested comparison, a non-numeric
        constant, an unknown function, or a nested aggregation.
    """
    if isinstance(node, ast.Compare):
        if not allow_compare:
            raise ConfigError(
                f"cut expression {source!r}: a comparison may only appear once, at the top "
                f"level, or as the argument of {list(_AGGREGATIONS)} — write one cut per "
                "comparison"
            )
        if len(node.ops) != 1:
            raise ConfigError(
                f"cut expression {source!r}: chained comparisons are not supported — "
                "write one cut per comparison"
            )
        if type(node.ops[0]) not in _COMPARES:
            raise ConfigError(
                f"cut expression {source!r}: unsupported comparison "
                f"{type(node.ops[0]).__name__} — expected one of {sorted(_SYMBOLS.values())}"
            )
        _check(node.left, source, allow_compare=False, in_agg=in_agg)
        _check(node.comparators[0], source, allow_compare=False, in_agg=in_agg)
    elif isinstance(node, ast.BinOp):
        combines = type(node.op) in _COMBINATORS
        if not combines and type(node.op) not in _BINOPS:
            raise ConfigError(
                f"cut expression {source!r}: unsupported operator {type(node.op).__name__} — "
                "only + - * / are allowed (and & | to combine two predicates)"
            )
        # only & / | pass the right to hold a comparison through to their operands
        sub = allow_compare if combines else False
        _check(node.left, source, allow_compare=sub, in_agg=in_agg)
        _check(node.right, source, allow_compare=sub, in_agg=in_agg)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise ConfigError(
                f"cut expression {source!r}: unsupported unary operator "
                f"{type(node.op).__name__} — only + and - are allowed"
            )
        _check(node.operand, source, allow_compare=False, in_agg=in_agg)
    elif isinstance(node, ast.Call):
        name = _call_name(node, source)
        if name in _ELEMENTWISE:
            _check(node.args[0], source, allow_compare=False, in_agg=in_agg)
        elif in_agg:
            raise ConfigError(
                f"cut expression {source!r}: {name}() is nested inside another reduction — "
                "a constituent axis can only be collapsed once"
            )
        else:
            _check(node.args[0], source, allow_compare=True, in_agg=True)
    elif isinstance(node, ast.Attribute | ast.Name):
        _name_path(node)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ConfigError(
                f"cut expression {source!r}: only numeric constants are allowed, got {node.value!r}"
            )
    else:
        raise ConfigError(
            f"cut expression {source!r}: {type(node).__name__} is not allowed — cut "
            f"expressions may only use field names, numbers, + - * /, parentheses, "
            f"{sorted([*_ELEMENTWISE, *_AGGREGATIONS])}, & | between predicates, and "
            "one comparison"
        )


def _is_aggregation(node: ast.expr) -> bool:
    """Whether `node` is a validated reduction call (``sum``/``count``)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id in _AGGREGATIONS)
    )


def _walk_names(node: ast.expr) -> Iterator[str]:
    """The bare ROW-axis field names a validated node references, in source order.

    Names under an aggregation are deliberately NOT yielded: they address the
    constituent axis, which the reader reads and reduces separately (see
    `Aggregation.fields`).

    Yields
    ------
    str
        Each referenced bare field name (duplicates included).
    """
    if _is_aggregation(node):
        return
    if isinstance(node, ast.Compare):
        yield from _walk_names(node.left)
        yield from _walk_names(node.comparators[0])
    elif isinstance(node, ast.BinOp):
        yield from _walk_names(node.left)
        yield from _walk_names(node.right)
    elif isinstance(node, ast.UnaryOp):
        yield from _walk_names(node.operand)
    elif isinstance(node, ast.Call):
        yield from _walk_names(node.args[0])
    elif isinstance(node, ast.Attribute | ast.Name):
        yield _bare_field(node)


def _walk_paths(node: ast.expr) -> Iterator[str]:
    """The DOTTED field paths a validated node references, in source order.

    Yields
    ------
    str
        Each referenced name path, exactly as written (duplicates included).
    """
    if isinstance(node, ast.Compare):
        yield from _walk_paths(node.left)
        yield from _walk_paths(node.comparators[0])
    elif isinstance(node, ast.BinOp):
        yield from _walk_paths(node.left)
        yield from _walk_paths(node.right)
    elif isinstance(node, ast.UnaryOp):
        yield from _walk_paths(node.operand)
    elif isinstance(node, ast.Call):
        yield from _walk_paths(node.args[0])
    elif isinstance(node, ast.Attribute | ast.Name):
        yield _name_path(node)


def _walk_aggregations(node: ast.expr) -> Iterator[ast.Call]:
    """The reduction calls a validated node contains, in source order.

    Yields
    ------
    ast.Call
        Each ``sum``/``count`` call node.
    """
    if _is_aggregation(node):
        assert isinstance(node, ast.Call)
        yield node
        return
    if isinstance(node, ast.Compare):
        yield from _walk_aggregations(node.left)
        yield from _walk_aggregations(node.comparators[0])
    elif isinstance(node, ast.BinOp):
        yield from _walk_aggregations(node.left)
        yield from _walk_aggregations(node.right)
    elif isinstance(node, ast.UnaryOp):
        yield from _walk_aggregations(node.operand)
    elif isinstance(node, ast.Call):
        yield from _walk_aggregations(node.args[0])


def _column(source: Mapping[str, Any] | np.ndarray, name: str) -> Any:
    """One named column off a structured array or a column mapping.

    Returns
    -------
    Any
        The column array.

    Raises
    ------
    KeyError
        When `name` is not present in `source`.
    """
    names = source.dtype.names or () if isinstance(source, np.ndarray) else tuple(source)
    if name not in names:
        raise KeyError(
            f"cut field {name!r} is not available; present: {sorted(names)}. Add it to the "
            "stream so it is read."
        )
    return source[name]


def _evaluate(node: ast.expr, source: Mapping[str, Any] | np.ndarray) -> Any:
    """Evaluate a validated node against a column source.

    Returns
    -------
    Any
        The node's value (a numpy array for field-bearing sub-expressions).
    """
    if isinstance(node, ast.Compare):
        return _COMPARES[type(node.ops[0])](
            _evaluate(node.left, source), _evaluate(node.comparators[0], source)
        )
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op)) or _COMBINATORS[type(node.op)]
        return op(_evaluate(node.left, source), _evaluate(node.right, source))
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_evaluate(node.operand, source))
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        if node.func.id in _ELEMENTWISE:
            return _ELEMENTWISE[node.func.id](_evaluate(node.args[0], source))
        # a reduction is precomputed by the reader onto the row axis and served
        # as a synthetic column (see `Aggregation.key`) — it cannot be evaluated
        # here, where only row-axis columns are in scope
        return _column(source, _agg_key(ast.unparse(node)))
    if isinstance(node, ast.Constant):
        return node.value
    return _column(source, _bare_field(node))


def _agg_key(source: str) -> str:
    """The synthetic row-axis column name a reduction's value is served under.

    Content-addressed and alphanumeric so it is a legal numpy structured-dtype
    field name (the cut record is a structured array), and so the same reduction
    written in two cuts resolves to one column.
    """
    digest = hashlib.sha1(source.encode()).hexdigest()[:12]  # noqa: S324 - non-crypto column key
    return f"__agg_{digest}__"


class Aggregation:
    """One reduction of a stream's constituent axis onto the row axis.

    An `Expression` cannot evaluate a reduction itself: it is handed row-axis
    columns, and the constituents have already been collapsed by then. So a
    reduction is a CONTRACT — it names the stream and the constituent fields it
    needs, the reader reads those, calls `evaluate`, and serves the result back
    as the synthetic row column `key`.

    Parameters
    ----------
    node : ast.Call
        A validated ``sum``/``count`` call.

    Attributes
    ----------
    func : str
        ``"sum"`` or ``"count"``.
    source : str
        The call as written (normalised by ``ast.unparse``) — for error messages.
    key : str
        The synthetic row-axis column name (`_agg_key` of `source`).
    stream : str
        The stream whose constituent axis is reduced.
    fields : tuple[str, ...]
        The bare constituent field names read, in first-seen order.

    Raises
    ------
    ConfigError
        When the argument references no field, references fields of more than one
        stream, or spells a field without (or with more than) a ``<stream>.``
        qualifier.
    """

    __slots__ = ("_arg", "fields", "func", "key", "source", "stream")

    def __init__(self, node: ast.Call) -> None:
        assert isinstance(node.func, ast.Name)
        self.func = node.func.id
        self.source = ast.unparse(node)
        self.key = _agg_key(self.source)
        self._arg = node.args[0]
        paths = tuple(_walk_paths(self._arg))
        if not paths:
            raise ConfigError(
                f"cut expression: {self.source!r} references no field — a reduction must "
                "name at least one constituent field to reduce"
            )
        unqualified = sorted({p for p in paths if p.count(".") != 1})
        if unqualified:
            raise ConfigError(
                f"cut expression: {self.source!r} spells {unqualified} without exactly one "
                "'<stream>.' qualifier — a reduction has to name the stream whose "
                "constituent axis it collapses (e.g. 'sum(jets.valid) >= 4')"
            )
        streams = sorted({p.split(".", 1)[0] for p in paths})
        if len(streams) > 1:
            raise ConfigError(
                f"cut expression: {self.source!r} mixes streams {streams} inside one "
                "reduction — the constituent axes of two streams are not aligned"
            )
        self.stream = streams[0]
        self.fields = tuple(dict.fromkeys(p.split(".", 1)[1] for p in paths))

    def evaluate(self, columns: Mapping[str, Any]) -> np.ndarray:
        """Reduce this stream's constituent columns onto the row axis.

        Parameters
        ----------
        columns : Mapping[str, Any]
            ``{bare field: jagged ``[row][constituent]`` array}`` for `stream`,
            already cut/truncated to what the reader will actually serve.

        Returns
        -------
        np.ndarray
            The ``(n_rows,)`` reduced values.
        """
        import awkward as ak

        inner = _evaluate(self._arg, columns)
        reduced = ak.num(inner, axis=1) if self.func == "count" else ak.sum(inner, axis=1)
        return np.asarray(ak.to_numpy(reduced))

    def __repr__(self) -> str:
        return f"Aggregation({self.source!r})"


def _constant(node: ast.expr) -> float | int | None:
    """The numeric value of a constant (optionally sign-prefixed) node, else None.

    Returns
    -------
    float | int | None
        The constant's value, or None when the node is not a constant.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int | float)
    ):
        value: float | int = _UNARYOPS[type(node.op)](node.operand.value)
        return value
    return None


class Expression:
    """A validated cut expression: arithmetic over stream fields and one comparison.

    Parameters
    ----------
    source : str
        The expression text, e.g. ``"d0 < 3.5"``,
        ``"(numberOfPixelSharedHits + numberOfSCTSharedHits / 2) < 1.1"`` or
        ``"sum(jets.valid) >= 4"``. Field names may carry a ``<stream>.``
        prefix, which is stripped (and is REQUIRED inside a reduction). Only
        field names, numeric constants, ``+ - * /``, parentheses, unary
        ``+``/``-``, the whitelisted functions ``abs``/``log`` (elementwise) and
        ``sum``/``count`` (reducing a constituent axis), ``&``/``|`` joining two
        predicates where a comparison is legal, and exactly one top-level
        comparison (``== != < <= > >=``) are accepted; everything else (other
        calls, attributes on non-names, subscripts, lambdas, comprehensions,
        ``and``/``or``, ...) is rejected.

    Attributes
    ----------
    fields : tuple[str, ...]
        The bare ROW-axis field names referenced, deduplicated in first-seen
        order — the read-planner contract (these fields must be read at bind).
        Fields under a reduction are excluded; they are `Aggregation.fields`.
    aggregations : tuple[Aggregation, ...]
        The reductions this expression contains, deduplicated by
        `Aggregation.key`. The reader evaluates each and serves its value as a
        synthetic row column before calling `evaluate`.

    Raises
    ------
    ConfigError
        On a syntax error, a non-comparison root, a chained comparison, a
        disallowed node/operator/function, a non-numeric constant, a malformed
        reduction, or an expression referencing no field at all.
    """

    __slots__ = ("_tree", "aggregations", "fields", "source")

    def __init__(self, source: str) -> None:
        self.source = source.strip()
        if not self.source:
            raise ConfigError("cut expression must be a non-empty string")
        try:
            tree = ast.parse(self.source, mode="eval")
        except SyntaxError as exc:
            raise ConfigError(f"cut expression {self.source!r} is not valid syntax: {exc}") from exc
        body = tree.body
        if not isinstance(body, ast.Compare):
            raise ConfigError(
                f"cut expression {self.source!r} must be a comparison producing a keep mask, "
                "e.g. 'd0 < 3.5'"
            )
        _check(body, self.source, allow_compare=True, in_agg=False)
        self._tree = body
        aggs: dict[str, Aggregation] = {}
        for call in _walk_aggregations(body):
            try:
                agg = Aggregation(call)
            except ConfigError as exc:
                raise ConfigError(f"cut expression {self.source!r}: {exc}") from exc
            aggs.setdefault(agg.key, agg)
        self.aggregations = tuple(aggs.values())
        seen: dict[str, None] = dict.fromkeys(_walk_names(body))
        if not seen and not self.aggregations:
            raise ConfigError(
                f"cut expression {self.source!r} references no field — a cut must compare "
                "at least one stream field"
            )
        self.fields = tuple(seen)

    @property
    def simple(self) -> tuple[str, str, float | int] | None:
        """``(field, op, value)`` when this is a bare ``<field> <op> <number>``, else None."""
        left, right = self._tree.left, self._tree.comparators[0]
        if not isinstance(left, ast.Attribute | ast.Name):
            return None
        value = _constant(right)
        if value is None:
            return None
        return _bare_field(left), _SYMBOLS[type(self._tree.ops[0])], value

    def evaluate(self, source: Mapping[str, Any] | np.ndarray) -> Any:
        """Evaluate to a keep mask over a structured array or a column mapping.

        Plain numpy comparison semantics: a NaN value FAILS every ordering /
        equality cut (only ``!=`` is True against NaN), matching the ftag
        ``TrackSelector`` convention.

        Parameters
        ----------
        source : Mapping[str, Any] | np.ndarray
            A structured array or a mapping of column arrays. When this
            expression has `aggregations`, it must also carry each one's
            precomputed value under its `Aggregation.key`.

        Returns
        -------
        Any
            The bool keep mask, shaped like the source columns.
        """
        return _evaluate(self._tree, source)

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"


@cache
def parse_expression(source: str) -> Expression:
    """Parse (and cache) one cut expression — the per-batch evaluation entry point.

    Returns
    -------
    Expression
        The validated expression.
    """
    return Expression(source)
