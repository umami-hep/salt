"""Unit tests for `salt.data.readers.cuts` (Cut, Aggregation, GlobalObjectCuts)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from salt.data import Cut, GlobalObjectCuts
from salt.graph.errors import ConfigError


def _jets(pt: list[float], label: list[int]) -> np.ndarray:
    rec = np.empty(len(pt), dtype=[("pt", "f4"), ("flavour_label", "i4")])
    rec["pt"] = pt
    rec["flavour_label"] = label
    return rec


def _tracks() -> np.ndarray:
    rec = np.empty(4, dtype=[("d0", "f4"), ("npix", "u1"), ("nsct", "u1"), ("label", "i4")])
    rec["d0"] = [0.5, 4.0, np.nan, -2.0]
    rec["npix"] = [0, 1, 2, 0]
    rec["nsct"] = [0, 0, 0, 3]
    rec["label"] = [1, 2, 3, 4]
    return rec


# --------------------------------------------------------------------------- #
# Cut
# --------------------------------------------------------------------------- #


def test_cut_validates_op() -> None:
    with pytest.raises(ConfigError):
        Cut(expr="pt ~= 0")


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        (">=", 20.0, [False, True, True]),
        (">", 20.0, [False, False, True]),
        ("<=", 20.0, [True, True, False]),
        ("<", 20.0, [True, False, False]),
        ("==", 20.0, [False, True, False]),
        ("!=", 20.0, [True, False, True]),
    ],
)
def test_cut_mask_each_operator(op: str, value: float, expected: list[bool]) -> None:
    jets = _jets(pt=[10.0, 20.0, 30.0], label=[0, 4, 5])
    cut = Cut(expr=f"pt {op} {value}")
    np.testing.assert_array_equal(cut.mask(jets), expected)


def test_cut_bare_field_strips_stream_prefix() -> None:
    jets = _jets(pt=[10.0, 30.0], label=[0, 5])
    # dotted "jets.pt" resolves to the bare "pt" field
    cut = Cut(expr="jets.pt >= 20.0")
    assert cut.fields == ("pt",)
    np.testing.assert_array_equal(cut.mask(jets), [False, True])


def test_cut_mask_unknown_field_raises_keyerror() -> None:
    jets = _jets(pt=[10.0], label=[0])
    with pytest.raises(KeyError):
        Cut(expr="nope > 0").mask(jets)


# --------------------------------------------------------------------------- #
# GlobalObjectCuts
# --------------------------------------------------------------------------- #


def test_global_object_cuts_empty_keeps_all() -> None:
    jets = _jets(pt=[10.0, 20.0, 30.0], label=[0, 4, 5])
    keep = GlobalObjectCuts().eligible(jets)
    assert keep.dtype == bool
    np.testing.assert_array_equal(keep, [True, True, True])


def test_global_object_cuts_global_cuts_and_combined() -> None:
    jets = _jets(pt=[10.0, 25.0, 30.0, 40.0], label=[0, 5, 4, 5])
    spec = GlobalObjectCuts(global_cuts=("pt >= 20.0", "flavour_label == 5"))
    # pt>=20 AND label==5 -> indices 1 and 3
    np.testing.assert_array_equal(spec.eligible(jets), [False, True, False, True])


def test_global_object_cuts_rejects_non_cut_entries() -> None:
    with pytest.raises(ConfigError):
        GlobalObjectCuts(global_cuts=(42,))  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        GlobalObjectCuts(global_cuts=({"field": "pt", "op": ">", "value": 1},))  # type: ignore[arg-type]


def test_global_object_cuts_eligible_unknown_field_raises_keyerror() -> None:
    jets = _jets(pt=[10.0], label=[0])
    spec = GlobalObjectCuts(global_cuts=("missing > 0",))
    with pytest.raises(KeyError):
        spec.eligible(jets)


def test_global_object_cuts_fields_union() -> None:
    spec = GlobalObjectCuts(global_cuts=("pt >= 20.0", "flavour_label == 5", "eta < 2.5"))
    assert spec.fields() == ("pt", "flavour_label", "eta")


def test_global_object_cuts_count_parity_invariant() -> None:
    """Passing + failing == total (the index-build count-parity guarantee)."""
    rng = np.random.default_rng(0)
    pt = rng.uniform(0, 100, size=500)
    label = rng.integers(0, 6, size=500)
    jets = _jets(pt=list(pt), label=list(label))
    spec = GlobalObjectCuts(global_cuts=("pt >= 50.0",))
    keep = spec.eligible(jets)
    assert int(keep.sum()) + int((~keep).sum()) == len(jets)
    assert int(keep.sum()) == int((pt >= 50.0).sum())


def test_global_object_cuts_accept_expression_strings() -> None:
    jets = _jets(pt=[10.0, 25.0, 30.0], label=[0, 5, 4])
    spec = GlobalObjectCuts(global_cuts=("pt >= 20", "flavour_label == 5"))
    np.testing.assert_array_equal(spec.eligible(jets), [False, True, False])
    assert spec.fields() == ("pt", "flavour_label")
    assert spec.global_cuts == ("pt >= 20", "flavour_label == 5")


# --------------------------------------------------------------------------- #
# expression-form Cut: multi-field / derived expressions
# --------------------------------------------------------------------------- #


def test_expression_cut_keeps_derived_expressions() -> None:
    cut = Cut(expr="(npix + nsct / 2) < 1.1")
    assert cut.fields == ("npix", "nsct")
    rec = np.empty(3, dtype=[("npix", "u1"), ("nsct", "u1")])
    rec["npix"], rec["nsct"] = [0, 1, 2], [0, 1, 0]
    np.testing.assert_array_equal(cut.mask(rec), [True, False, False])


def test_cut_parse_accepts_str_and_cut_only() -> None:
    assert Cut.parse("pt > 20").expr == "pt > 20"
    c = Cut(expr="pt > 20")
    assert Cut.parse(c) is c
    with pytest.raises(ConfigError):
        Cut.parse({"field": "pt", "op": ">", "value": 20})  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        Cut.parse({"expr": "pt > 20"})  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        Cut.parse(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# constituent-axis reductions on the row-cut surface
# --------------------------------------------------------------------------- #


def test_global_cuts_expose_their_reductions() -> None:
    cuts = GlobalObjectCuts(global_cuts=("sum(jets.valid) >= 4",))
    (agg,) = cuts.aggregations()
    assert (agg.stream, agg.fields) == ("jets", ("valid",))
    assert cuts.fields() == ()  # nothing to read on the row axis


def test_reductions_deduplicate_across_cuts() -> None:
    cuts = GlobalObjectCuts(
        global_cuts=("sum(jets.valid) >= 4", "sum(jets.valid) >= 6", "sum(jets.pt) >= 1")
    )
    assert [a.source for a in cuts.aggregations()] == ["sum(jets.valid)", "sum(jets.pt)"]


def test_a_reduction_cut_evaluates_off_its_precomputed_column() -> None:
    cuts = GlobalObjectCuts(global_cuts=("sum(jets.valid) >= 4",))
    (agg,) = cuts.aggregations()
    rows = np.empty(3, dtype=[(agg.key, "i8")])
    rows[agg.key] = [3, 4, 5]
    np.testing.assert_array_equal(cuts.eligible(rows), [False, True, True])


def test_simple_cuts_have_no_reductions() -> None:
    assert Cut(expr="pt > 20").aggregations == ()


# --------------------------------------------------------------------------- #
# Cut: one compiled predicate — pickling, repr
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cut",
    [
        Cut(expr="jets.pt >= 20.0"),
        Cut(expr="(pt + flavour_label / 2) < 25.0"),
        Cut(expr="sum(jets.valid) >= 4"),
    ],
)
def test_cut_pickle_round_trip(cut: Cut) -> None:
    revived = pickle.loads(pickle.dumps(cut))
    assert revived == cut
    assert hash(revived) == hash(cut)
    assert repr(revived) == repr(cut)
    assert revived.fields == cut.fields
    assert revived.aggregations == cut.aggregations
    if cut.aggregations:
        (agg,) = cut.aggregations
        rows = np.empty(3, dtype=[(agg.key, "i8")])
        rows[agg.key] = [3, 4, 5]
        np.testing.assert_array_equal(revived.mask(rows), cut.mask(rows))
    else:
        jets = _jets(pt=[10.0, 20.0, 30.0], label=[0, 4, 5])
        np.testing.assert_array_equal(revived.mask(jets), cut.mask(jets))


def test_cut_repr_has_no_predicate_field() -> None:
    assert repr(Cut(expr="pt >= 20.0")) == "Cut(expr='pt >= 20.0')"


# --------------------------------------------------------------------------- #
# expression evaluation (folded from the now-deleted expressions unit tests)
# --------------------------------------------------------------------------- #


def test_attribute_chain_is_a_dotted_field_not_a_method() -> None:
    """``tracks.d0`` spells a dotted field; the ``<stream>.`` prefix is stripped."""
    cut = Cut(expr="tracks.d0 < 3.5")
    assert cut.fields == ("d0",)
    np.testing.assert_array_equal(cut.mask(_tracks()), [True, False, False, True])


def test_attribute_on_non_name_rejected() -> None:
    with pytest.raises(ConfigError, match="qualifier"):
        Cut(expr="(d0 + 1).real < 3.5")


def test_precedence_matches_python() -> None:
    """``a + b / 2`` binds the division tighter — not ``(a + b) / 2``."""
    tracks = _tracks()
    got = Cut(expr="npix + nsct / 2 < 1.1").mask(tracks)
    want = (tracks["npix"] + tracks["nsct"] / 2) < 1.1
    np.testing.assert_array_equal(got, want)
    # parentheses change the grouping
    other = Cut(expr="(npix + nsct) / 2 < 1.1").mask(tracks)
    assert not np.array_equal(got, other)


def test_int_float_promotion_matches_numpy() -> None:
    """Unsigned ints divided by an int constant promote to float (no integer truncation)."""
    tracks = _tracks()
    got = Cut(expr="nsct / 2 >= 0.5").mask(tracks)
    np.testing.assert_array_equal(got, tracks["nsct"] / 2 >= 0.5)
    assert bool(got[3])  # nsct=3 -> 1.5, would be 1 under integer division


def test_unary_minus_and_parentheses() -> None:
    tracks = _tracks()
    np.testing.assert_array_equal(Cut(expr="d0 > -1.0").mask(tracks), tracks["d0"] > -1.0)
    np.testing.assert_array_equal(Cut(expr="-(d0) < 1.0").mask(tracks), -tracks["d0"] < 1.0)


def test_nan_comparison_evaluates_false() -> None:
    """A NaN value FAILS every ordering/equality cut (numpy semantics; only != is True)."""
    tracks = _tracks()
    for op in ("<", "<=", ">", ">=", "=="):
        assert not Cut(expr=f"d0 {op} 3.5").mask(tracks)[2]
    assert Cut(expr="d0 != 3.5").mask(tracks)[2]


def test_referenced_field_extraction_order_and_dedup() -> None:
    cut = Cut(expr="(npix + nsct / 2) * npix < d0")
    assert cut.fields == ("npix", "nsct", "d0")


def test_evaluate_on_a_column_mapping() -> None:
    cols = {"d0": np.array([1.0, 9.0]), "npix": np.array([0, 5])}
    np.testing.assert_array_equal(Cut(expr="d0 + npix < 5").mask(cols), [True, False])


def test_missing_field_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        Cut(expr="nope < 1").mask(_tracks())


def test_abs_and_log_are_elementwise() -> None:
    src = {"eta": np.array([-4.6, -1.0, 2.0]), "r": np.array([0.5, 1.0, 2.0])}
    np.testing.assert_array_equal(Cut(expr="abs(eta) < 4.5").mask(src), [False, True, True])
    np.testing.assert_array_equal(
        Cut(expr="log(r) > -0.385").mask(src), np.log(src["r"]) > -0.385
    )


def test_elementwise_call_keeps_its_fields_on_the_row_axis() -> None:
    cut = Cut(expr="abs(eta) < 4.5")
    assert cut.fields == ("eta",)
    assert cut.aggregations == ()


def test_aggregation_is_not_a_row_field() -> None:
    cut = Cut(expr="sum(jets.valid) >= 4")
    assert cut.fields == ()  # `valid` lives on the constituent axis, not the row
    (agg,) = cut.aggregations
    assert (agg.stream, agg.fields, agg.source) == ("jets", ("valid",), "sum(jets.valid)")


def test_aggregation_key_is_a_legal_structured_dtype_name() -> None:
    (agg,) = Cut(expr="sum(jets.valid) >= 4").aggregations
    rec = np.zeros(3, dtype=[(agg.key, "i8")])  # must not raise
    assert rec.dtype.names == (agg.key,)


def test_aggregation_reads_its_value_from_the_precomputed_column() -> None:
    (agg,) = Cut(expr="sum(jets.valid) >= 4").aggregations
    src = {agg.key: np.array([3, 4, 5])}
    np.testing.assert_array_equal(Cut(expr="sum(jets.valid) >= 4").mask(src), [False, True, True])


def test_aggregations_deduplicate_within_one_expression() -> None:
    cut = Cut(expr="sum(jets.valid) + sum(jets.valid) >= 8")
    assert len(cut.aggregations) == 1


def test_two_spellings_of_one_reduction_share_a_key() -> None:
    a = Cut(expr="sum( jets.valid ) >= 4").aggregations[0]
    b = Cut(expr="sum(jets.valid) >= 2").aggregations[0]
    assert a.key == b.key


@pytest.mark.parametrize(
    ("src", "match"),
    [
        ("foo(d0) < 1", "unknown function"),
        ("np.log(d0) < 1", "unknown function"),
        ("sum(jets.pt, 1) >= 2", "exactly one argument"),
        ("sum(jets.pt, axis=1) >= 2", "exactly one argument"),
        ("sum(sum(jets.pt)) >= 2", "nested"),
        ("sum(valid) >= 4", "qualifier"),
        ("sum(a.b.c) >= 4", "qualifier"),
        ("sum(jets.pt + tracks.pt) >= 4", "mixes streams"),
        ("sum(4) >= 4", "references no field"),
    ],
)
def test_rejected_call_forms(src: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Cut(expr=src)


@pytest.mark.parametrize(
    ("src", "match"),
    [
        ("sorted(d0) < 3.5", "unknown function"),
        ("d0 < 3.5 and npix > 1", "must be a comparison"),
        ("not d0 < 3.5", "must be a comparison"),
        ("d0", "must be a comparison"),
        ("d0 + 1", "must be a comparison"),
        ("1 < 2", "references no field"),
        ("", "non-empty string"),
        ("   ", "non-empty string"),
        ("d0 <", "is not valid syntax"),
    ],
)
def test_rejected_at_parse_time(src: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        Cut(expr=src)


def test_aggregation_evaluate_reduces_the_constituent_axis() -> None:
    ak = pytest.importorskip("awkward")
    cols = {"pt": ak.Array([[10.0, 20.0, 30.0], [], [40.0]])}
    (agg,) = Cut(expr="sum(jets.pt) > 1").aggregations
    np.testing.assert_array_equal(agg.evaluate(cols), [60.0, 0.0, 40.0])


def test_aggregation_evaluate_counts_a_predicate() -> None:
    ak = pytest.importorskip("awkward")
    cols = {"pt": ak.Array([[10.0, 20.0, 30.0], [], [40.0]])}
    (agg,) = Cut(expr="sum(jets.pt > 15.0) >= 2").aggregations
    np.testing.assert_array_equal(agg.evaluate(cols), [2, 0, 1])


def test_predicates_combine_inside_a_reduction() -> None:
    ak = pytest.importorskip("awkward")
    cols = {
        "pt": ak.Array([[10.0, 20.0, 30.0], [], [40.0]]),
        "eta": ak.Array([[0.5, 3.0, 1.0], [], [0.1]]),
    }
    (agg,) = Cut(expr="sum((jets.pt > 15.0) & (abs(jets.eta) < 2.5)) >= 2").aggregations
    np.testing.assert_array_equal(agg.evaluate(cols), [1, 0, 1])  # jet 1 fails |eta|
    assert agg.fields == ("pt", "eta")


def test_rejected_combinator_forms() -> None:
    """``and`` (a BoolOp) is not even a comparison at the root."""
    with pytest.raises(ConfigError, match="must be a comparison"):
        Cut(expr="d0 < 3.5 and npix > 1")


# --------------------------------------------------------------------------- #
# shipped cut strings — masks pinned as literals (plan 06 engine-parity gate)
# --------------------------------------------------------------------------- #

_GN3_TRACK_CUTS = ("d0 < 3.5", "(numberOfPixelSharedHits + numberOfSCTSharedHits / 2) < 1.1")
_PHYSLITE_JET_CUTS = ("pt > 20000", "eta < 4.5", "eta > -4.5", "NNJvtPass > 0.5")
_PHYSLITE_EVENT_CUTS = (
    "sum((jets.pt > 20000) & (jets.eta < 4.5) & (jets.eta > -4.5) & (jets.NNJvtPass > 0.5)) >= 4",
    "sum((log(jets.GN2v01_pb / (0.2 * jets.GN2v01_pc + 0.8 * jets.GN2v01_pu)) > -0.385) "
    "& (abs(jets.eta) < 2.5) & (jets.pt > 20000) & (jets.NNJvtPass > 0.5)) >= 2",
)


def test_shipped_track_cut_masks_are_pinned() -> None:
    """GN3_baseline.yaml track cuts on 5 tracks: d0 = [0, 3.5, -3.6, nan, 1]; npix+nsct/2 = [0, 1, 1.5, 2.5, 1]."""
    tracks = np.array(
        [(0.0, 0, 0), (3.5, 1, 0), (-3.6, 0, 3), (np.nan, 2, 1), (1.0, 0, 2)],
        dtype=[("d0", "f4"), ("numberOfPixelSharedHits", "u1"), ("numberOfSCTSharedHits", "u1")],
    )
    want = {
        _GN3_TRACK_CUTS[0]: [True, False, True, False, True],  # 3.5 is not < 3.5; NaN fails
        _GN3_TRACK_CUTS[1]: [True, True, False, False, True],  # 1 < 1.1, 1.5 and 2.5 are not
    }
    for expr, mask in want.items():
        np.testing.assert_array_equal(Cut(expr=expr).mask(tracks), mask)


def test_shipped_jet_cut_masks_are_pinned() -> None:
    """physlite_events.yaml jet_selection cuts on 6 jets (boundary values sit exactly on the thresholds)."""
    jets = np.array(
        [
            (25000.0, 0.1, 1),
            (20000.0, 4.5, 1),
            (15000.0, -4.5, 0),
            (30000.0, -2.0, 1),
            (np.nan, 1.0, 0),
            (40000.0, -4.6, 1),
        ],
        dtype=[("pt", "f4"), ("eta", "f4"), ("NNJvtPass", "i1")],
    )
    want = {
        "pt > 20000": [True, False, False, True, False, True],  # 20000 is not > 20000; NaN fails
        "eta < 4.5": [True, False, True, True, True, True],  # 4.5 is not < 4.5
        "eta > -4.5": [True, True, False, True, True, False],  # -4.5 is not > -4.5; -4.6 fails
        "NNJvtPass > 0.5": [True, True, False, True, False, True],
    }
    for expr, mask in want.items():
        np.testing.assert_array_equal(Cut(expr=expr).mask(jets), mask)


def test_shipped_event_reductions_are_pinned() -> None:
    """physlite_events.yaml row cuts over 4 events (5 / 0 / 4 / 4 on-disk jets).

    Event 0: jets pass the >=4 preselection 4x (jet 4 has pt 19000) and the b-tag
    predicate 1x (jet 1 D = ln(0.1/0.54) < -0.385, jets 2-3 |eta| >= 2.5, jet 4 pt).
    Event 2: jet 3 fails NNJvt -> 3 and 3. Event 3: everything passes -> 4 and 4.
    """
    ak = pytest.importorskip("awkward")
    cols = {
        "pt": ak.Array([
            [25000.0, 30000.0, 40000.0, 21000.0, 19000.0],
            [],
            [50000.0, 60000.0, 70000.0, 80000.0],
            [30000.0] * 4,
        ]),
        "eta": ak.Array([
            [0.1, 2.0, -2.6, 4.4, 0.0],
            [],
            [1.0, -1.0, 2.4, -2.4],
            [0.5, -0.5, 1.5, -1.5],
        ]),
        "NNJvtPass": ak.Array([[1, 1, 1, 1, 1], [], [1, 1, 1, 0], [1, 1, 1, 1]]),
        "GN2v01_pb": ak.Array([[0.9, 0.1, 0.5, 0.9, 0.9], [], [0.9] * 4, [0.9] * 4]),
        "GN2v01_pc": ak.Array([[0.05, 0.3, 0.25, 0.05, 0.05], [], [0.05] * 4, [0.05] * 4]),
        "GN2v01_pu": ak.Array([[0.05, 0.6, 0.25, 0.05, 0.05], [], [0.05] * 4, [0.05] * 4]),
    }
    four, two = (Cut(expr=e) for e in _PHYSLITE_EVENT_CUTS)
    (agg4,), (agg2,) = four.aggregations, two.aggregations
    np.testing.assert_array_equal(agg4.evaluate(cols), [4, 0, 3, 4])
    np.testing.assert_array_equal(agg2.evaluate(cols), [1, 0, 3, 4])
    rows = np.empty(4, dtype=[(agg4.key, "i8"), (agg2.key, "i8")])
    rows[agg4.key], rows[agg2.key] = agg4.evaluate(cols), agg2.evaluate(cols)
    np.testing.assert_array_equal(four.mask(rows), [True, False, False, True])
    np.testing.assert_array_equal(two.mask(rows), [False, False, True, True])
