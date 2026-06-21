"""Parity gate: run the REAL salt pipeline/cli test bodies against schema-
generated data.

Strategy: import the existing test modules (``salt.tests.test_pipeline`` and
``salt.tests.test_cli``), monkeypatch the ``write_dummy_file`` /
``write_dummy_norm_dict`` names *inside those modules* to the schema-backed
adapters, then call the real test functions. If they pass, the schema-generated
file is a drop-in for ``write_dummy_file``'s output.

The class_dict length==output_size resolution is what eliminates the
CrossEntropyLoss weight-size RuntimeError in test_GN2 / test_param_* / etc.
"""

from __future__ import annotations

import pytest

from salt.core.testing.datagen.parity import (
    schema_write_dummy_file,
    schema_write_dummy_norm_dict,
)

# import the real salt test modules
import salt.tests.test_pipeline as tp
import salt.tests.test_cli as tc


@pytest.fixture(autouse=True)
def _patch_dummy_writers(monkeypatch):
    """Redirect the dummy-file writers inside the salt test modules to the
    schema-backed adapters for the duration of each test."""
    for mod in (tp, tc):
        if hasattr(mod, "write_dummy_file"):
            monkeypatch.setattr(mod, "write_dummy_file", schema_write_dummy_file)
        if hasattr(mod, "write_dummy_norm_dict"):
            monkeypatch.setattr(mod, "write_dummy_norm_dict", schema_write_dummy_norm_dict)
    # also patch the source module in case anything resolves through it
    import salt.utils.inputs as inputs_mod

    monkeypatch.setattr(inputs_mod, "write_dummy_file", schema_write_dummy_file)
    monkeypatch.setattr(inputs_mod, "write_dummy_norm_dict", schema_write_dummy_norm_dict)
    yield


# --------------------------------------------------------------------------- #
# Core parity: the configs that read class_dict directly (CrossEntropyLoss
# weight-size constraint). These are the tests the design doc calls out:
# test_GN2 / test_GN2_muP / test_param_concat / test_param_featurewise /
# test_truncate_inputs (track_subset).
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings(tp.w)
def test_parity_GN2(tmp_path):
    tp.test_GN2(tmp_path)


@pytest.mark.filterwarnings(tp.w)
def test_parity_GN2_muP(tmp_path):
    tp.test_GN2_muP(tmp_path)


@pytest.mark.filterwarnings(tp.w)
def test_parity_param_concat(tmp_path):
    tp.test_param_concat(tmp_path)


@pytest.mark.filterwarnings(tp.w)
def test_parity_param_featurewise(tmp_path):
    tp.test_param_featurewise(tmp_path)


@pytest.mark.filterwarnings(tp.w)
def test_parity_truncate_inputs(tmp_path):
    tp.test_truncate_inputs(tmp_path)


# --------------------------------------------------------------------------- #
# GN3 family (is_gn3=True -> flavour_label 6-class class_dict)
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings(tp.w)
def test_parity_GN3V00(tmp_path):
    tp.test_GN3V00(tmp_path)


# --------------------------------------------------------------------------- #
# MaskFormer: exercises the barcode linkage -> object_masks (1000,5,40)
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings(tp.w)
def test_parity_maskformer(tmp_path):
    tp.test_maskformer(tmp_path)


# --------------------------------------------------------------------------- #
# regression configs (nan_regression exercises HadronConeExclTruthLabelLxy NaN)
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings(tp.w)
def test_parity_nan_regression(tmp_path):
    tp.test_nan_regression(tmp_path)


@pytest.mark.filterwarnings(tp.w)
def test_parity_regression(tmp_path):
    tp.test_regression(tmp_path)


# --------------------------------------------------------------------------- #
# eval path (truth_hadrons shape + tracks len) via GN2 with onnx off
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings(tp.w)
def test_parity_GN2emu(tmp_path):
    tp.test_GN2emu(tmp_path)


@pytest.mark.filterwarnings(tp.w)
def test_parity_GN2XE(tmp_path):
    tp.test_GN2XE(tmp_path)


# --------------------------------------------------------------------------- #
# CLI initialization (make_xbb=True) -- test_cli body
# --------------------------------------------------------------------------- #
def test_parity_cli_initialization(tmp_path):
    # build dummy files exactly as test_cli's fixture does, but schema-backed
    train_h5_path = tmp_path / "dummy_train_inputs.h5"
    nd_path = tmp_path / "dummy_norm_dict.yaml"
    cd_path = tmp_path / "dummy_class_dict.yaml"
    schema_write_dummy_norm_dict(nd_path, cd_path)
    schema_write_dummy_file(train_h5_path, nd_path, make_xbb=True)

    files = {
        "train_h5_path": train_h5_path,
        "nd_path": nd_path,
        "cd_path": cd_path,
        "tmpdir": tmp_path,
    }
    tc.test_initialization(files)
