"""`Expression` — an ``ast``-whitelist evaluator for cut predicates: numeric
arithmetic over stream fields plus exactly one comparison, evaluated vectorised
with numpy. No ``eval``, no new dependency.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Iterator, Mapping
from functools import cache
from typing import Any

import numpy as np

from salt.graph.errors import ConfigError

__all__ = ["Expression", "parse_expression"]

_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
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


def _check(node: ast.expr, source: str) -> None:
    """Reject every node outside the whitelist (arithmetic, one comparison, fields, numbers).

    Raises
    ------
    ConfigError
        On a disallowed node, operator, chained comparison, or non-numeric constant.
    """
    if isinstance(node, ast.Compare):
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
        _check(node.left, source)
        _check(node.comparators[0], source)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ConfigError(
                f"cut expression {source!r}: unsupported operator {type(node.op).__name__} — "
                "only + - * / are allowed"
            )
        _check(node.left, source)
        _check(node.right, source)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise ConfigError(
                f"cut expression {source!r}: unsupported unary operator "
                f"{type(node.op).__name__} — only + and - are allowed"
            )
        _check(node.operand, source)
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
            "expressions may only use field names, numbers, + - * /, parentheses and one "
            "comparison"
        )


def _walk_names(node: ast.expr) -> Iterator[str]:
    """The bare field names a validated node references, in left-to-right source order.

    Yields
    ------
    str
        Each referenced bare field name (duplicates included).
    """
    if isinstance(node, ast.Compare):
        yield from _walk_names(node.left)
        yield from _walk_names(node.comparators[0])
    elif isinstance(node, ast.BinOp):
        yield from _walk_names(node.left)
        yield from _walk_names(node.right)
    elif isinstance(node, ast.UnaryOp):
        yield from _walk_names(node.operand)
    elif isinstance(node, ast.Attribute | ast.Name):
        yield _bare_field(node)


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
        return _BINOPS[type(node.op)](_evaluate(node.left, source), _evaluate(node.right, source))
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_evaluate(node.operand, source))
    if isinstance(node, ast.Constant):
        return node.value
    return _column(source, _bare_field(node))


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
        The expression text, e.g. ``"d0 < 3.5"`` or
        ``"(numberOfPixelSharedHits + numberOfSCTSharedHits / 2) < 1.1"``. Field
        names may carry a ``<stream>.`` prefix, which is stripped. Only field
        names, numeric constants, ``+ - * /``, parentheses, unary ``+``/``-`` and
        exactly one comparison (``== != < <= > >=``) are accepted; everything
        else (calls, attributes on non-names, subscripts, lambdas,
        comprehensions, boolean operators, ...) is rejected.

    Attributes
    ----------
    fields : tuple[str, ...]
        The bare field names referenced, deduplicated in first-seen order — the
        read-planner contract (these fields must be read at bind).

    Raises
    ------
    ConfigError
        On a syntax error, a non-comparison root, a chained comparison, a
        disallowed node/operator, a non-numeric constant, or an expression
        referencing no field at all.
    """

    __slots__ = ("_tree", "fields", "source")

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
        _check(body, self.source)
        self._tree = body
        seen: dict[str, None] = dict.fromkeys(_walk_names(body))
        if not seen:
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
            A structured array or a mapping of column arrays.

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
