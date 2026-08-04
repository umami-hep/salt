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
        "sorted(d0) < 3.5",  # call to a non-whitelisted function
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


# --------------------------------------------------------------------------- #
# whitelisted functions: abs/log elementwise, sum/count reducing a constituent axis
# --------------------------------------------------------------------------- #


def test_abs_and_log_are_elementwise() -> None:
    src = {"eta": np.array([-4.6, -1.0, 2.0]), "r": np.array([0.5, 1.0, 2.0])}
    np.testing.assert_array_equal(Expression("abs(eta) < 4.5").evaluate(src), [False, True, True])
    np.testing.assert_array_equal(
        Expression("log(r) > -0.385").evaluate(src), np.log(src["r"]) > -0.385
    )


def test_elementwise_call_keeps_its_fields_on_the_row_axis() -> None:
    expr = Expression("abs(eta) < 4.5")
    assert expr.fields == ("eta",)
    assert expr.aggregations == ()


def test_aggregation_is_not_a_row_field() -> None:
    expr = Expression("sum(jets.valid) >= 4")
    assert expr.fields == ()  # `valid` lives on the constituent axis, not the row
    (agg,) = expr.aggregations
    assert (agg.func, agg.stream, agg.fields) == ("sum", "jets", ("valid",))
    assert agg.source == "sum(jets.valid)"


def test_aggregation_key_is_a_legal_structured_dtype_name() -> None:
    (agg,) = Expression("sum(jets.valid) >= 4").aggregations
    rec = np.zeros(3, dtype=[(agg.key, "i8")])  # must not raise
    assert rec.dtype.names == (agg.key,)


def test_aggregation_reads_its_value_from_the_precomputed_column() -> None:
    (agg,) = Expression("sum(jets.valid) >= 4").aggregations
    src = {agg.key: np.array([3, 4, 5])}
    np.testing.assert_array_equal(
        Expression("sum(jets.valid) >= 4").evaluate(src), [False, True, True]
    )


def test_aggregations_deduplicate_within_one_expression() -> None:
    expr = Expression("sum(jets.valid) + sum(jets.valid) >= 8")
    assert len(expr.aggregations) == 1


def test_two_spellings_of_one_reduction_share_a_key() -> None:
    a = Expression("sum( jets.valid ) >= 4").aggregations[0]
    b = Expression("sum(jets.valid) >= 2").aggregations[0]
    assert a.key == b.key


@pytest.mark.parametrize(
    ("src", "match"),
    [
        ("foo(d0) < 1", "not a whitelisted function"),
        ("np.log(d0) < 1", "not a whitelisted function"),
        ("sum(jets.pt, 1) >= 2", "exactly one positional"),
        ("sum(jets.pt, axis=1) >= 2", "exactly one positional"),
        ("sum(sum(jets.pt)) >= 2", "collapsed once"),
        ("sum(jets.pt) + count(sum(jets.pt)) >= 2", "collapsed once"),
        ("sum(valid) >= 4", "qualifier"),
        ("sum(a.b.c) >= 4", "qualifier"),
        ("sum(jets.pt + tracks.pt) >= 4", "mixes streams"),
        ("sum(4) >= 4", "references no field"),
        ("(d0 < 1) < 1", "only appear once"),
        ("abs(d0 < 1) < 1", "only appear once"),
    ],
)
def test_rejected_call_forms(src: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Expression(src)


def test_aggregation_evaluate_reduces_the_constituent_axis() -> None:
    ak = pytest.importorskip("awkward")
    cols = {"pt": ak.Array([[10.0, 20.0, 30.0], [], [40.0]])}
    (agg,) = Expression("sum(jets.pt) > 1").aggregations
    np.testing.assert_array_equal(agg.evaluate(cols), [60.0, 0.0, 40.0])
    (cnt,) = Expression("count(jets.pt) > 1").aggregations
    np.testing.assert_array_equal(cnt.evaluate(cols), [3, 0, 1])


def test_aggregation_evaluate_counts_a_predicate() -> None:
    ak = pytest.importorskip("awkward")
    cols = {"pt": ak.Array([[10.0, 20.0, 30.0], [], [40.0]])}
    (agg,) = Expression("sum(jets.pt > 15.0) >= 2").aggregations
    np.testing.assert_array_equal(agg.evaluate(cols), [2, 0, 1])


def test_predicates_combine_inside_a_reduction() -> None:
    ak = pytest.importorskip("awkward")
    cols = {
        "pt": ak.Array([[10.0, 20.0, 30.0], [], [40.0]]),
        "eta": ak.Array([[0.5, 3.0, 1.0], [], [0.1]]),
    }
    (agg,) = Expression("sum((jets.pt > 15.0) & (abs(jets.eta) < 2.5)) >= 2").aggregations
    np.testing.assert_array_equal(agg.evaluate(cols), [1, 0, 1])  # jet 1 fails |eta|
    (either,) = Expression("sum((jets.pt > 35.0) | (abs(jets.eta) < 0.6)) >= 1").aggregations
    np.testing.assert_array_equal(either.evaluate(cols), [1, 0, 1])
    assert agg.fields == ("pt", "eta")


@pytest.mark.parametrize(
    ("src", "match"),
    [
        # & binds tighter than a comparison, so unparenthesised operands are a
        # chained comparison, not a conjunction
        ("sum(jets.pt > 15.0 & jets.eta < 2.5) >= 2", "chained comparisons"),
        # a conjunction is only legal where a comparison is
        ("(d0 > 1) & (npix < 2) > 0", "only appear once"),
        ("d0 < 3.5 and npix > 1", "not allowed"),
    ],
)
def test_rejected_combinator_forms(src: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Expression(src)
