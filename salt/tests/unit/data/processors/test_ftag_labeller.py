"""Unit tests for `salt.data.processors.ftag_labeller` (FtagLabeller)."""

from __future__ import annotations

import numpy as np
import pytest
from ftag import Labeller

from salt.data import FtagLabeller
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode

SMALL_TYPE = 0
LARGE_TYPE = 1

SMALL_CLASSES = [
    "ghostbjets",
    "ghostcjets",
    "ghostsjets",
    "ghostudjets",
    "ghostgjets",
    "ghosttaujets",
]
LARGE_CLASSES = ["htautauhad", "hbb", "hcc", "top", "qcdbb", "qcdbx", "qcdcx", "qcdll", "Wqq"]

# (HadronGhostTruthLabelID, PartonTruthLabelID), one per small-R class, in class order
_SMALL_TRUTH = [(5, 5), (4, 4), (0, 3), (0, 1), (0, 21), (15, 15)]
# (R10TruthLabel_R22v1, GhostBHadronsFinalCount, GhostCHadronsFinalCount), in class order
_LARGE_TRUTH = [
    (16, 0, 0),
    (11, 2, 0),
    (12, 0, 2),
    (1, 1, 0),
    (10, 2, 0),
    (10, 1, 0),
    (10, 0, 1),
    (10, 0, 0),
    (2, 0, 1),
]

_DTYPE = [
    ("jet_type", "i8"),
    ("HadronGhostTruthLabelID", "i4"),
    ("PartonTruthLabelID", "i4"),
    ("R10TruthLabel_R22v1", "i4"),
    ("GhostBHadronsFinalCount", "i4"),
    ("GhostCHadronsFinalCount", "i4"),
]


def _joint_jets(small: list[tuple[int, int]], large: list[tuple[int, int, int]]) -> np.ndarray:
    """Small-R then large-R rows; the other type's truth fields are -1 (the joint reader's fill)."""
    jets = np.full(len(small) + len(large), -1, dtype=_DTYPE)
    for i, (ghost, parton) in enumerate(small):
        jets[i] = (SMALL_TYPE, ghost, parton, -1, -1, -1)
    for i, (r10, n_b, n_c) in enumerate(large, start=len(small)):
        jets[i] = (LARGE_TYPE, -1, -1, r10, n_b, n_c)
    return jets


def _labels(labeller: FtagLabeller, jets: np.ndarray) -> np.ndarray:
    out = labeller.process({"raw.jets": jets}, slice(0, len(jets)), Mode.FIT)
    key = f"labels.jets.{labeller.label}"
    assert set(out) == {key}
    return out[key]


def _small_labeller(**kw) -> FtagLabeller:  # noqa: ANN003
    kw = {"select_field": "jet_type", "select_value": SMALL_TYPE, **kw}
    return FtagLabeller(label="small_flavour", class_names=SMALL_CLASSES, **kw)


def _large_labeller(**kw) -> FtagLabeller:  # noqa: ANN003
    kw = {"select_field": "jet_type", "select_value": LARGE_TYPE, **kw}
    return FtagLabeller(label="large_flavour", class_names=LARGE_CLASSES, **kw)


# (hadron ghost label, parton label) satisfying exactly one class each, by class index
CLASS_ROWS = {0: (5, 5), 1: (4, 4), 2: (0, 3), 3: (0, 1), 4: (0, 21), 5: (15, 15)}
UNLABELLED_ROW = (0, 7)
SEL = {"select_field": "jet_type", "select_value": 0}
VARIABLES = list(dict.fromkeys(Labeller(SMALL_CLASSES).variables))


def _raw(rows: list[tuple[int, int]], jet_types: list[int]) -> np.ndarray:
    raw = np.zeros(len(rows), dtype=[(v, np.int32) for v in [*VARIABLES, "jet_type"]])
    for var in VARIABLES:
        assert "Parton" in var or "HadronGhost" in var, var
        raw[var] = [parton if "Parton" in var else hadron for hadron, parton in rows]
    raw["jet_type"] = jet_types
    return raw


def _labeller(**kwargs) -> FtagLabeller:
    mod = FtagLabeller(class_names=SMALL_CLASSES, **kwargs)
    mod.name = "ftag_labeller"
    return mod


def test_labels_every_object_without_selection() -> None:
    jets = _joint_jets(_SMALL_TRUTH, [])
    labels = _labels(FtagLabeller(class_names=SMALL_CLASSES), jets)
    np.testing.assert_array_equal(labels, np.arange(len(SMALL_CLASSES)))
    assert labels.dtype == np.int64


def test_mixed_types_raise_without_selection() -> None:
    jets = _joint_jets(_SMALL_TRUTH, _LARGE_TRUTH)
    with pytest.raises(ValueError, match="not labelled"):
        _labels(FtagLabeller(class_names=SMALL_CLASSES), jets)


def test_selection_labels_small_rows_and_ignores_large_rows() -> None:
    jets = _joint_jets(_SMALL_TRUTH, _LARGE_TRUTH)
    labels = _labels(_small_labeller(), jets)
    assert labels.shape == jets.shape
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels[: len(_SMALL_TRUTH)], np.arange(len(SMALL_CLASSES)))
    assert (labels[len(_SMALL_TRUTH) :] == -1).all()


def test_selection_labels_large_rows_and_ignores_small_rows() -> None:
    jets = _joint_jets(_SMALL_TRUTH, _LARGE_TRUTH)
    labels = _labels(_large_labeller(), jets)
    assert labels.shape == jets.shape
    assert (labels[: len(_SMALL_TRUTH)] == -1).all()
    np.testing.assert_array_equal(labels[len(_SMALL_TRUTH) :], np.arange(len(LARGE_CLASSES)))


def test_selection_keeps_row_positions_when_types_interleave() -> None:
    jets = _joint_jets(_SMALL_TRUTH, _LARGE_TRUTH)
    order = np.random.default_rng(0).permutation(len(jets))
    for labeller in (_small_labeller(), _large_labeller()):
        expected = _labels(labeller, jets)[order]
        np.testing.assert_array_equal(_labels(labeller, jets[order]), expected)


def test_selection_with_no_selected_rows_is_all_ignored() -> None:
    jets = _joint_jets(_SMALL_TRUTH, [])
    labels = _labels(_large_labeller(), jets)
    assert labels.shape == jets.shape
    assert labels.dtype == np.int64
    assert (labels == -1).all()


def test_require_labels_still_applies_inside_the_selection() -> None:
    orphan = (0, 9)  # ghost-light with a non-light parton: none of the six classes
    jets = _joint_jets([*_SMALL_TRUTH, orphan], _LARGE_TRUTH)
    with pytest.raises(ValueError, match="not labelled"):
        _labels(_small_labeller(), jets)
    assert (_labels(_large_labeller(), jets)[: len(_SMALL_TRUTH) + 1] == -1).all()


def test_select_field_is_read_from_the_stream() -> None:
    plain = FtagLabeller(class_names=SMALL_CLASSES).read_fields(None)["jets"]
    assert set(plain) == {"HadronGhostTruthLabelID", "PartonTruthLabelID"}
    selected = _small_labeller().read_fields(None)["jets"]
    assert set(selected) == {"HadronGhostTruthLabelID", "PartonTruthLabelID", "jet_type"}


def test_missing_select_field_raises() -> None:
    jets = _joint_jets(_SMALL_TRUTH, _LARGE_TRUTH)
    without = jets[[name for name in jets.dtype.names if name != "jet_type"]]
    with pytest.raises(ValueError, match=r"missing select_field \['jet_type'\]"):
        _labels(_small_labeller(), without)


@pytest.mark.parametrize("kw", [{"select_field": "jet_type"}, {"select_value": 0}])
def test_select_field_and_value_are_set_together(kw: dict) -> None:
    with pytest.raises(ConfigError, match="select_field and select_value are set together"):
        FtagLabeller(class_names=SMALL_CLASSES, **kw)


def test_selection_needs_require_labels() -> None:
    with pytest.raises(ConfigError, match=r"require_labels=True .* batch alignment"):
        _small_labeller(require_labels=False)


def test_selection_on_the_joint_ftag1lite_stream(tmp_path) -> None:  # noqa: ANN001
    """The reader's own fill values and jet-type column drive the selection."""
    pytest.importorskip("uproot")
    pytest.importorskip("awkward")
    from salt.tests._fixtures import joint_ftag1lite_corpus as corpus

    corpus.build_joint_corpus(tmp_path)
    reader = corpus.make_reader(tmp_path)
    batch = reader.read(slice(0, len(reader)), Mode.FIT)
    jets = batch["raw.jets"]
    small = jets["jet_type"] == corpus.SMALL_TYPE
    assert small.any()
    assert (~small).any()

    small_labels = _labels(_small_labeller(select_value=corpus.SMALL_TYPE), jets)
    assert small_labels.shape == jets.shape
    assert ((small_labels[small] >= 0) & (small_labels[small] < len(SMALL_CLASSES))).all()
    assert (small_labels[~small] == -1).all()

    # after its own cut the corpus draws R10TruthLabel_R22v1 from {1, 2, 10, 11}; no single
    # class covers every W jet (2), so those rows are left out
    jets = jets[jets["R10TruthLabel_R22v1"] != 2]
    small = jets["jet_type"] == corpus.SMALL_TYPE
    large_labels = _labels(
        FtagLabeller(
            label="large_flavour",
            class_names=["top", "hbb", "qcd"],
            select_field="jet_type",
            select_value=corpus.LARGE_TYPE,
        ),
        jets,
    )
    assert (~small).any()
    assert (large_labels[small] == -1).all()
    assert (large_labels[~small] >= 0).all()


class TestConstructor:
    def test_selection_attributes_stored(self):
        mod = _labeller(select_field="jet_type", select_value=1)
        assert (mod.select_field, mod.select_value) == ("jet_type", 1)
        assert (_labeller().select_field, _labeller().select_value) == (None, None)


class TestReadFields:
    def test_read_fields_demands_select_field_and_labeller_variables(self):
        fields = _labeller(**SEL).read_fields(None)
        assert set(fields) == {"jets"}
        assert set(fields["jets"]) == {*VARIABLES, "jet_type"}
        assert set(fields["jets"].values()) == {"ftag_labeller"}

    def test_read_fields_without_selection_has_no_select_field(self):
        assert set(_labeller().read_fields(None)["jets"]) == set(VARIABLES)


class TestProcess:
    def test_process_mixed_batch_labels_selected_rows_only(self):
        classes = np.array([0, 1, 2, 3, 4, 5, 3, 0])
        jt = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        out = _labels(_labeller(**SEL), _raw([CLASS_ROWS[c] for c in classes], list(jt)))
        assert (len(out), out.dtype) == (len(classes), np.int64)
        np.testing.assert_array_equal(out[jt == 1], -1)
        np.testing.assert_array_equal(out[jt == 0], classes[jt == 0])

    def test_process_selects_other_value(self):
        raw = _raw([CLASS_ROWS[c] for c in (0, 1, 2, 3)], [0, 1, 1, 0])
        out = _labels(_labeller(select_field="jet_type", select_value=1), raw)
        np.testing.assert_array_equal(out, [-1, 1, 2, -1])

    def test_unlabelled_unselected_row_does_not_raise(self):
        raw = _raw([CLASS_ROWS[0], UNLABELLED_ROW, CLASS_ROWS[4]], [0, 1, 0])
        out = _labels(_labeller(**SEL), raw)
        np.testing.assert_array_equal(out, [0, -1, 4])
        assert out.dtype == np.int64

    def test_selection_absent_behaviour_unchanged(self):
        classes = [5, 4, 3, 2, 1, 0]
        raw = _raw([CLASS_ROWS[c] for c in classes], [0, 1, 0, 1, 0, 1])
        out = _labels(_labeller(), raw)
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out, Labeller(SMALL_CLASSES).get_labels(raw))
        np.testing.assert_array_equal(out, classes)

    def test_selection_absent_require_labels_false_still_drops(self):
        raw = _raw([CLASS_ROWS[0], UNLABELLED_ROW, CLASS_ROWS[3]], [0, 0, 0])
        np.testing.assert_array_equal(_labels(_labeller(require_labels=False), raw), [0, 3])
