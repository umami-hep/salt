"""Unit tests for the ast-whitelist cut-expression evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from salt.data.readers.expressions import Expression, parse_expression
from salt.graph.errors import ConfigError


def _tracks() -> np.ndarray:
    rec = np.empty(4, dtype=[("d0", "f4"), ("npix", "u1"), ("nsct", "u1"), ("label", "i4")])
    rec["d0"] = [0.5, 4.0, np.nan, -2.0]
    rec["npix"] = [0, 1, 2, 0]
    rec["nsct"] = [0, 0, 0, 3]
    rec["label"] = [1, 2, 3, 4]
    return rec


# --------------------------------------------------------------------------- #
# whitelist rejections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "src",
    [
        "abs(d0) < 3.5",  # call
        "d0[0] < 3.5",  # subscript
        "(lambda x: x)(d0) < 1",  # lambda
        "[x for x in d0] < 1",  # comprehension
        "d0 < 3.5 and npix > 1",  # boolean operator
        "not d0 < 3.5",  # unary not
        "d0 ** 2 < 3.5",  # unsupported binary operator
        "d0 % 2 < 1",  # unsupported binary operator
        "d0 < 'x'",  # non-numeric constant
        "d0 is None",  # unsupported comparison
        "d0 in (1, 2)",  # unsupported comparison
        "d0 < 3.5 < npix",  # chained comparison
        "d0",  # not a comparison
        "d0 + 1",  # not a comparison
        "1 < 2",  # references no field
        "",  # empty
        "d0 <",  # syntax error
    ],
)
def test_whitelist_rejects(src: str) -> None:
    with pytest.raises(ConfigError):
        Expression(src)


def test_attribute_chain_is_a_dotted_field_not_a_method() -> None:
    """``tracks.d0`` spells a dotted field; the ``<stream>.`` prefix is stripped."""
    expr = Expression("tracks.d0 < 3.5")
    assert expr.fields == ("d0",)
    np.testing.assert_array_equal(expr.evaluate(_tracks()), [True, False, False, True])


def test_attribute_on_non_name_rejected() -> None:
    with pytest.raises(ConfigError):
        Expression("(d0 + 1).real < 3.5")


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def test_precedence_matches_python() -> None:
    """``a + b / 2`` binds the division tighter — not ``(a + b) / 2``."""
    tracks = _tracks()
    expr = Expression("npix + nsct / 2 < 1.1")
    got = expr.evaluate(tracks)
    want = (tracks["npix"] + tracks["nsct"] / 2) < 1.1
    np.testing.assert_array_equal(got, want)
    # parentheses change the grouping
    other = Expression("(npix + nsct) / 2 < 1.1").evaluate(tracks)
    assert not np.array_equal(got, other)


def test_int_float_promotion_matches_numpy() -> None:
    """Unsigned ints divided by an int constant promote to float (no integer truncation)."""
    tracks = _tracks()
    got = Expression("nsct / 2 >= 0.5").evaluate(tracks)
    np.testing.assert_array_equal(got, tracks["nsct"] / 2 >= 0.5)
    assert bool(got[3])  # nsct=3 -> 1.5, would be 1 under integer division


def test_unary_minus_and_parentheses() -> None:
    tracks = _tracks()
    np.testing.assert_array_equal(Expression("d0 > -1.0").evaluate(tracks), tracks["d0"] > -1.0)
    np.testing.assert_array_equal(Expression("-(d0) < 1.0").evaluate(tracks), -tracks["d0"] < 1.0)


def test_nan_comparison_evaluates_false() -> None:
    """A NaN value FAILS every ordering/equality cut (numpy semantics; only != is True)."""
    tracks = _tracks()
    for op in ("<", "<=", ">", ">=", "=="):
        assert not Expression(f"d0 {op} 3.5").evaluate(tracks)[2]
    assert Expression("d0 != 3.5").evaluate(tracks)[2]


def test_referenced_field_extraction_order_and_dedup() -> None:
    expr = Expression("(npix + nsct / 2) * npix < d0")
    assert expr.fields == ("npix", "nsct", "d0")


def test_evaluate_on_a_column_mapping() -> None:
    cols = {"d0": np.array([1.0, 9.0]), "npix": np.array([0, 5])}
    np.testing.assert_array_equal(Expression("d0 + npix < 5").evaluate(cols), [True, False])


def test_missing_field_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        Expression("nope < 1").evaluate(_tracks())


# --------------------------------------------------------------------------- #
# simple-form normalisation + caching
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("src", "want"),
    [
        ("d0 < 3.5", ("d0", "<", 3.5)),
        ("tracks.d0 >= 2", ("d0", ">=", 2)),
        ("d0 != -1", ("d0", "!=", -1)),
    ],
)
def test_simple_normalisation(src: str, want: tuple[str, str, float]) -> None:
    assert Expression(src).simple == want


@pytest.mark.parametrize("src", ["npix + nsct / 2 < 1.1", "d0 < npix", "3.5 > d0"])
def test_not_simple(src: str) -> None:
    assert Expression(src).simple is None


def test_parse_expression_is_cached() -> None:
    assert parse_expression("d0 < 3.5") is parse_expression("d0 < 3.5")
