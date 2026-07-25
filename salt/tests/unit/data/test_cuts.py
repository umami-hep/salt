"""Unit tests for `salt.data.readers.cuts` (Cut, GlobalObjectCuts, ConstituentCuts)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from salt.data import ConstituentCuts, Cut, CutSpec, GlobalObjectCuts
from salt.graph.errors import ConfigError


def _cc(cuts: Any, on_fail: str) -> ConstituentCuts:
    """Build `ConstituentCuts` off the loose config surface (expression strings).

    Returns
    -------
    ConstituentCuts
        The normalised container.
    """
    return ConstituentCuts(cuts=cuts, on_fail=on_fail)


def _jets(pt: list[float], label: list[int]) -> np.ndarray:
    rec = np.empty(len(pt), dtype=[("pt", "f4"), ("flavour_label", "i4")])
    rec["pt"] = pt
    rec["flavour_label"] = label
    return rec


# --------------------------------------------------------------------------- #
# Cut
# --------------------------------------------------------------------------- #


def test_cut_validates_op() -> None:
    with pytest.raises(ConfigError):
        Cut(field="pt", op="~=", value=0)


def test_cut_validates_field() -> None:
    with pytest.raises(ConfigError):
        Cut(field="", op=">", value=0)


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
    cut = Cut(field="pt", op=op, value=value)
    np.testing.assert_array_equal(cut.mask(jets), expected)


def test_cut_bare_field_strips_stream_prefix() -> None:
    jets = _jets(pt=[10.0, 30.0], label=[0, 5])
    # dotted "jets.pt" resolves to the bare "pt" field
    cut = Cut(field="jets.pt", op=">=", value=20.0)
    assert cut.bare_field == "pt"
    np.testing.assert_array_equal(cut.mask(jets), [False, True])


def test_cut_mask_unknown_field_raises_keyerror() -> None:
    jets = _jets(pt=[10.0], label=[0])
    with pytest.raises(KeyError):
        Cut(field="nope", op=">", value=0).mask(jets)


# --------------------------------------------------------------------------- #
# CutSpec
# --------------------------------------------------------------------------- #


def test_cutspec_empty_keeps_all() -> None:
    jets = _jets(pt=[10.0, 20.0, 30.0], label=[0, 4, 5])
    keep = CutSpec().eligible(jets, None)
    assert keep.dtype == bool
    np.testing.assert_array_equal(keep, [True, True, True])


def test_cutspec_global_cuts_and_combined() -> None:
    jets = _jets(pt=[10.0, 25.0, 30.0, 40.0], label=[0, 5, 4, 5])
    spec = CutSpec(global_cuts=(Cut("pt", ">=", 20.0), Cut("flavour_label", "==", 5)))
    # pt>=20 AND label==5 -> indices 1 and 3
    np.testing.assert_array_equal(spec.eligible(jets, None), [False, True, False, True])


def test_cutspec_per_split_adds_to_global() -> None:
    jets = _jets(pt=[10.0, 25.0, 30.0, 40.0], label=[0, 5, 4, 5])
    spec = CutSpec(
        global_cuts=(Cut("pt", ">=", 20.0),),
        per_split={"train": (Cut("flavour_label", "==", 5),)},
    )
    # global-only (None / val): pt>=20 -> 1,2,3
    np.testing.assert_array_equal(spec.eligible(jets, None), [False, True, True, True])
    np.testing.assert_array_equal(spec.eligible(jets, "val"), [False, True, True, True])
    # train: pt>=20 AND label==5 -> 1,3
    np.testing.assert_array_equal(spec.eligible(jets, "train"), [False, True, False, True])


def test_cutspec_for_split_returns_global_plus_split() -> None:
    g = Cut("pt", ">=", 20.0)
    t = Cut("flavour_label", "==", 5)
    spec = CutSpec(global_cuts=(g,), per_split={"train": (t,)})
    assert spec.for_split(None) == (g,)
    assert spec.for_split("val") == (g,)
    assert spec.for_split("train") == (g, t)


def test_cutspec_unknown_stage_key_raises() -> None:
    with pytest.raises(ConfigError):
        CutSpec(per_split={"validation": (Cut("pt", ">", 0),)})


def test_cutspec_rejects_non_cut_entries() -> None:
    with pytest.raises(ConfigError):
        CutSpec(global_cuts=("not a cut",))  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        CutSpec(per_split={"train": ("not a cut",)})  # type: ignore[dict-item]


def test_cutspec_eligible_unknown_field_raises_keyerror() -> None:
    jets = _jets(pt=[10.0], label=[0])
    spec = CutSpec(global_cuts=(Cut("missing", ">", 0),))
    with pytest.raises(KeyError):
        spec.eligible(jets, None)


def test_cutspec_fields_union_and_per_split() -> None:
    spec = CutSpec(
        global_cuts=(Cut("pt", ">=", 20.0),),
        per_split={"train": (Cut("flavour_label", "==", 5),), "test": (Cut("eta", "<", 2.5),)},
    )
    # union across global + every per-split, first-seen order
    assert spec.fields() == ("pt", "flavour_label", "eta")
    # per-split: global + that split
    assert spec.fields("train") == ("pt", "flavour_label")
    assert spec.fields("val") == ("pt",)


def test_cutspec_count_parity_invariant() -> None:
    """Passing + failing == total (the index-build count-parity guarantee)."""
    rng = np.random.default_rng(0)
    pt = rng.uniform(0, 100, size=500)
    label = rng.integers(0, 6, size=500)
    jets = _jets(pt=list(pt), label=list(label))
    spec = CutSpec(global_cuts=(Cut("pt", ">=", 50.0),))
    keep = spec.eligible(jets, None)
    assert int(keep.sum()) + int((~keep).sum()) == len(jets)
    assert int(keep.sum()) == int((pt >= 50.0).sum())


def test_cutspec_is_the_global_object_cuts_alias() -> None:
    assert CutSpec is GlobalObjectCuts


def test_global_object_cuts_accept_expression_strings() -> None:
    jets = _jets(pt=[10.0, 25.0, 30.0], label=[0, 5, 4])
    specs: Any = ("pt >= 20", "flavour_label == 5")
    spec = GlobalObjectCuts(global_cuts=specs)
    np.testing.assert_array_equal(spec.eligible(jets, None), [False, True, False])
    assert spec.fields() == ("pt", "flavour_label")


# --------------------------------------------------------------------------- #
# expression-form Cut
# --------------------------------------------------------------------------- #


def test_expression_cut_normalises_bare_comparison_to_the_simple_form() -> None:
    cut = Cut(expr="jets.pt >= 20.0")
    assert (cut.field, cut.op, cut.value) == ("pt", ">=", 20.0)
    assert cut.fields == ("pt",)
    jets = _jets(pt=[10.0, 30.0], label=[0, 5])
    np.testing.assert_array_equal(cut.mask(jets), [False, True])


def test_expression_cut_keeps_derived_expressions() -> None:
    cut = Cut(expr="(npix + nsct / 2) < 1.1")
    assert not cut.field  # not normalisable to the simple form
    assert cut.fields == ("npix", "nsct")
    rec = np.empty(3, dtype=[("npix", "u1"), ("nsct", "u1")])
    rec["npix"], rec["nsct"] = [0, 1, 2], [0, 1, 0]
    np.testing.assert_array_equal(cut.mask(rec), [True, False, False])


def test_cut_rejects_both_forms_and_neither() -> None:
    with pytest.raises(ConfigError):
        Cut(field="pt", op=">", value=0, expr="pt > 0")
    with pytest.raises(ConfigError):
        Cut()
    with pytest.raises(ConfigError):
        Cut(field="pt", op=">")  # no value


def test_cut_parse_accepts_str_mapping_and_cut() -> None:
    assert Cut.parse("pt > 20").field == "pt"
    assert Cut.parse({"field": "pt", "op": ">", "value": 20}).value == 20
    c = Cut("pt", ">", 20)
    assert Cut.parse(c) is c
    with pytest.raises(ConfigError):
        Cut.parse(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# ConstituentCuts
# --------------------------------------------------------------------------- #


def _tracks(n_rows: int = 2, t_dim: int = 4) -> np.ndarray:
    """A (n_rows, t_dim) constituent batch: 3 valid slots per row, 1 pad.

    Returns
    -------
    np.ndarray
        The structured constituent batch.
    """
    rec = np.zeros(
        (n_rows, t_dim),
        dtype=[("d0", "f4"), ("npix", "u1"), ("label", "i4"), ("flag", "?"), ("valid", "?")],
    )
    rec["d0"] = [[0.5, 9.0, 1.0, 0.0], [4.0, 0.25, 7.0, 0.0]]
    rec["npix"] = [[1, 2, 3, 0], [4, 5, 6, 0]]
    rec["label"] = [[10, 11, 12, -1], [13, 14, 15, -1]]
    rec["flag"] = True
    rec["valid"] = [[True, True, True, False], [True, True, True, False]]
    return rec


def test_constituent_cuts_requires_explicit_on_fail() -> None:
    with pytest.raises(ConfigError, match="on_fail"):
        _cc(("d0 < 3.5",), "")
    with pytest.raises(ConfigError, match="on_fail"):
        _cc(("d0 < 3.5",), "nan")


def test_constituent_cuts_fields_dedup_union() -> None:
    cc = _cc(("d0 < 3.5", "d0 + npix < 9"), "mask")
    assert cc.fields == ("d0", "npix")


def test_constituent_cuts_mask_blanks_failing_slots_per_dtype() -> None:
    """Mask keeps the slot: float->NaN, signed int->-1, unsigned->0, bool->False."""
    batch = _tracks()
    _cc(("d0 < 3.5",), "mask").apply(batch)
    # row 0: slot 1 (d0=9) fails; row 1: slot 0 (4.0) and 2 (7.0) fail
    assert np.isnan(batch["d0"][0, 1])
    np.testing.assert_array_equal(np.isnan(batch["d0"]), [[0, 1, 0, 0], [1, 0, 1, 0]])
    np.testing.assert_array_equal(batch["npix"][0], [1, 0, 3, 0])
    np.testing.assert_array_equal(batch["label"][0], [10, -1, 12, -1])
    np.testing.assert_array_equal(batch["flag"][0], [True, False, True, True])
    np.testing.assert_array_equal(batch["valid"][0], [True, False, True, False])


def test_constituent_cuts_mask_preserves_positions_and_multiplicity() -> None:
    batch = _tracks()
    before = batch["d0"][0, 2]
    _cc(("d0 < 3.5",), "mask").apply(batch)
    assert batch.shape == (2, 4)
    assert batch["d0"][0, 2] == before  # a passing constituent never moves


def test_constituent_cuts_mask_leaves_padding_untouched() -> None:
    """Already-invalid slots are never blanked (they were not 'removed' by the cut)."""
    batch = _tracks()
    _cc(("d0 < 3.5",), "mask").apply(batch)
    assert not np.isnan(batch["d0"][0, 3])  # pad slot keeps its 0.0
    assert batch["label"][0, 3] == -1


def test_constituent_cuts_drop_compacts_and_repads() -> None:
    batch = _tracks()
    out = _cc(("d0 < 3.5",), "drop").apply(batch)
    # row 0 keeps slots 0, 2 (order preserved); row 1 keeps slot 1
    np.testing.assert_array_equal(out["d0"][0], [0.5, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["label"][0], [10, 12, -1, -1])
    np.testing.assert_array_equal(out["valid"][0], [True, True, False, False])
    np.testing.assert_array_equal(out["d0"][1], [0.25, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["label"][1], [14, -1, -1, -1])
    np.testing.assert_array_equal(out["valid"][1], [True, False, False, False])
    np.testing.assert_array_equal(out["npix"][1], [5, 0, 0, 0])


def test_constituent_cuts_drop_never_keeps_a_failing_constituent() -> None:
    batch = _tracks()
    out = _cc(("d0 < 3.5",), "drop").apply(batch)
    kept = out["d0"][out["valid"]]
    assert (kept < 3.5).all()


def test_constituent_cuts_derived_expression_masks_like_the_oracle() -> None:
    batch = _tracks()
    npix, valid = batch["npix"].copy(), batch["valid"].copy()
    _cc(("npix + npix / 2 < 3.0",), "mask").apply(batch)
    want_removed = valid & ~((npix + npix / 2) < 3.0)
    np.testing.assert_array_equal(batch["valid"], valid & ~want_removed)
    np.testing.assert_array_equal(np.isnan(batch["d0"]), want_removed)


def test_constituent_cuts_no_cuts_is_a_noop() -> None:
    batch = _tracks()
    before = batch.copy()
    out = _cc((), "mask").apply(batch)
    assert out.tobytes() == before.tobytes()


def test_constituent_cuts_need_the_valid_field() -> None:
    rec = np.zeros((1, 2), dtype=[("d0", "f4")])
    with pytest.raises(KeyError):
        _cc(("d0 < 1",), "mask").apply(rec)
