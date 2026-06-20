"""Unit tests for `salt.core.data.cuts` (Cut + CutSpec, plan 19, Track C).

These are pure in-memory tests (no sample file): `CutSpec` is reader-agnostic and
operates on a structured jet-scalar numpy array. The reader-integration count
parity / per-split tests live in ``test_ftag1lite_reader.py`` (which reads the
real sample).
"""

from __future__ import annotations

import numpy as np
import pytest

from salt.core.data import Cut, CutSpec
from salt.core.graph.errors import ConfigError


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
