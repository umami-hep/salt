"""Unit tests for `FtagLabeller` row selection (``select_field`` / ``select_value``)."""

from __future__ import annotations

import numpy as np
import pytest
from ftag import Labeller

from salt.data.processors.ftag_labeller import FtagLabeller
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode

CLASS_NAMES = [f"ghost{f}jets" for f in ("b", "c", "s", "ud", "g", "tau")]
# (hadron ghost label, parton label) satisfying exactly one class each, by class index
CLASS_ROWS = {0: (5, 5), 1: (4, 4), 2: (0, 3), 3: (0, 1), 4: (0, 21), 5: (15, 15)}
UNLABELLED_ROW = (0, 7)  # matches no ghost class
SEL = {"select_field": "jet_type", "select_value": 0}
VARIABLES = list(dict.fromkeys(Labeller(CLASS_NAMES).variables))


def _raw(rows: list[tuple[int, int]], jet_types: list[int]) -> np.ndarray:
    raw = np.zeros(len(rows), dtype=[(v, np.int32) for v in [*VARIABLES, "jet_type"]])
    for var in VARIABLES:
        assert "Parton" in var or "HadronGhost" in var, var
        raw[var] = [parton if "Parton" in var else hadron for hadron, parton in rows]
    raw["jet_type"] = jet_types
    return raw


def _labeller(**kwargs) -> FtagLabeller:
    mod = FtagLabeller(class_names=CLASS_NAMES, **kwargs)
    mod.name = "ftag_labeller"
    return mod


def _process(mod: FtagLabeller, raw: np.ndarray) -> np.ndarray:
    out = mod.process({"raw.jets": raw}, slice(0, len(raw)), Mode.FIT)
    assert set(out) == {"labels.jets.flavour_label"}
    return out["labels.jets.flavour_label"]


class TestConstructor:
    def test_select_field_without_value_raises(self):
        with pytest.raises(ConfigError, match="select_field and select_value"):
            FtagLabeller(class_names=CLASS_NAMES, select_field="jet_type")

    def test_select_value_without_field_raises(self):
        with pytest.raises(ConfigError, match="select_field and select_value"):
            FtagLabeller(class_names=CLASS_NAMES, select_value=0)

    def test_selection_with_require_labels_false_raises(self):
        with pytest.raises(ConfigError, match="batch alignment"):
            FtagLabeller(class_names=CLASS_NAMES, require_labels=False, **SEL)

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
        out = _process(_labeller(**SEL), _raw([CLASS_ROWS[c] for c in classes], list(jt)))
        assert (len(out), out.dtype) == (len(classes), np.int64)
        np.testing.assert_array_equal(out[jt == 1], -1)
        np.testing.assert_array_equal(out[jt == 0], classes[jt == 0])

    def test_process_selects_other_value(self):
        raw = _raw([CLASS_ROWS[c] for c in (0, 1, 2, 3)], [0, 1, 1, 0])
        out = _process(_labeller(select_field="jet_type", select_value=1), raw)
        np.testing.assert_array_equal(out, [-1, 1, 2, -1])

    def test_unlabelled_selected_row_raises(self):
        raw = _raw([CLASS_ROWS[0], UNLABELLED_ROW, CLASS_ROWS[1]], [0, 0, 1])
        with pytest.raises(ValueError, match="not labelled"):
            _process(_labeller(**SEL), raw)

    def test_unlabelled_unselected_row_does_not_raise(self):
        raw = _raw([CLASS_ROWS[0], UNLABELLED_ROW, CLASS_ROWS[4]], [0, 1, 0])
        out = _process(_labeller(**SEL), raw)
        np.testing.assert_array_equal(out, [0, -1, 4])
        assert out.dtype == np.int64

    def test_no_selected_rows_gives_all_minus_one(self):
        raw = _raw([CLASS_ROWS[0], UNLABELLED_ROW, CLASS_ROWS[2]], [1, 1, 1])
        out = _process(_labeller(**SEL), raw)
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out, [-1, -1, -1])

    def test_missing_select_field_raises(self):
        raw_no_type = _raw([CLASS_ROWS[0]], [0])[VARIABLES]
        with pytest.raises(ValueError, match="missing select_field"):
            _process(_labeller(**SEL), raw_no_type)

    def test_selection_absent_behaviour_unchanged(self):
        classes = [5, 4, 3, 2, 1, 0]
        raw = _raw([CLASS_ROWS[c] for c in classes], [0, 1, 0, 1, 0, 1])
        out = _process(_labeller(), raw)
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out, Labeller(CLASS_NAMES).get_labels(raw))
        np.testing.assert_array_equal(out, classes)

    def test_selection_absent_require_labels_false_still_drops(self):
        raw = _raw([CLASS_ROWS[0], UNLABELLED_ROW, CLASS_ROWS[3]], [0, 0, 0])
        np.testing.assert_array_equal(_process(_labeller(require_labels=False), raw), [0, 3])
