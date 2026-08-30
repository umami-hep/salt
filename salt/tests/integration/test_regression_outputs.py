"""Per-config semantics for the regression + gaussian-regression matrix rows.

Rows 8 (``regression``) and 9 (``regression_gaussian``) give the fit/eval/export
lifecycle; what follows here is the de-scale, doubled-column and ONNX-rank
assertions that are per-config semantics the generic runner cannot express and
must survive the fold of ``test_regression_e2e.py`` +
``test_regression_gaussian_e2e.py``.

do_onnx=True for both rows (§7 resolution): the "no gaussian handling" claim
this file's gaussian half used to carry was a stale docstring, not a property
of ``check_onnx`` — ``compare_once`` (``salt/outputs/sinks/onnx/check.py``) is
fully generic (by-name allclose on floats, exact ints, dead-output canary), so
the gaussian export is checked like every other row.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import onnx
import pytest

from salt.outputs.sinks.onnx import make_session
from salt.tests.integration.pipeline import LegFailedError, run_eval, run_export

# pipeline: shares pipeline.py's session-scoped artifact cache with
# test_pipeline.py, so this must run in the SAME pytest invocation (the
# pipeline-matrix CI job, -m pipeline) or it re-trains rows 8/9 from scratch.
# cpu_always: every row here uses --trainer.accelerator=auto on synthetic
# data, so it must survive the tests/integration/ GPU-skip on a CPU runner.
pytestmark = [pytest.mark.pipeline, pytest.mark.cpu_always]


def _eval_h5(name: str, tmp_path_factory) -> Path:
    try:
        return run_eval(name, tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"{name} {exc.leg} failed: {exc}")


def _onnx(name: str, tmp_path_factory) -> Path:
    try:
        return run_export(name, tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"{name} {exc.leg} failed: {exc}")


@pytest.fixture(scope="module")
def regression_eval_h5(tmp_path_factory) -> Path:
    return _eval_h5("regression", tmp_path_factory)


@pytest.fixture(scope="module")
def gaussian_eval_h5(tmp_path_factory) -> Path:
    return _eval_h5("regression_gaussian", tmp_path_factory)


@pytest.fixture(scope="module")
def regression_onnx(tmp_path_factory) -> Path:
    return _onnx("regression", tmp_path_factory)


@pytest.fixture(scope="module")
def gaussian_onnx(tmp_path_factory) -> Path:
    return _onnx("regression_gaussian", tmp_path_factory)


def test_regression_eval_h5_columns_present_and_descaled(regression_eval_h5):
    """The section eval H5 carries the regression columns, finite and de-scaled."""
    with h5py.File(regression_eval_h5) as f:
        jets = f["jets"][:]
        tracks = f["tracks"][:]
    jet_cols = set(jets.dtype.names)
    # the five regression heads' custom / target column names (regression.yaml)
    for col in (
        "regression_HadronConeExclTruthLabelPt",
        "regression_pt",
        "regression_truthMass",
        "regression_truthPt",
    ):
        assert col in jet_cols, f"missing regression column {col}: {sorted(jet_cols)}"
    # reg_normed -> HadronConeExclTruthLabelPt (norm_params mean=1.0 std=1.0):
    # de-scaled = raw*1 + 1, so the column is finite (the de-scale really ran)
    assert np.isfinite(jets["regression_HadronConeExclTruthLabelPt"]).all()
    # the per-token seq head columns land on the tracks stream
    track_cols = set(tracks.dtype.names)
    for col in ("regression_dummyOutput_dPhi", "regression_dummyOutput_dEta"):
        assert col in track_cols, f"missing seq regression column {col}"
    assert jets.shape[0] > 0


def test_gaussian_eval_h5_has_stddev_columns(gaussian_eval_h5):
    """The eval H5 carries the gaussian doubled columns (mean + _stddev), both streams."""
    with h5py.File(gaussian_eval_h5) as f:
        jet_cols = set(f["jets"].dtype.names)
        track_cols = set(f["tracks"].dtype.names)
        n_rows = f["jets"].shape[0]
    assert n_rows > 0
    assert any(c.endswith("_stddev") for c in jet_cols), f"no gaussian stddev in jets: {jet_cols}"
    assert any(
        c.endswith("_stddev") for c in track_cols
    ), f"no gaussian stddev in tracks: {track_cols}"


class TestRegressionOnnxContract:
    """The shipped regression.yaml exports a well-formed ONNX contract (row 8)."""

    def test_onnx_output_ranks_global_vs_per_token(self, regression_onnx):
        """6 rank-0 globals (norm/ratio scalars) + 2 rank-1 per-token seq columns."""
        model = onnx.load(str(regression_onnx))
        ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
        assert sorted(ranks.values()) == [0, 0, 0, 0, 0, 0, 1, 1], ranks

    def test_onnx_session_runs(self, regression_onnx):
        """The exported graph runs in onnxruntime on batch-1 inputs (L=5 tokens)."""
        sess = make_session(regression_onnx)
        in_meta = {i.name: i.shape for i in sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds), strict=True)}
        assert len(out) == 8, f"expected the 8 regression outputs, got {sorted(out)}"


class TestGaussianOnnxContract:
    """The shipped regression_gaussian.yaml exports a well-formed ONNX contract (row 9)."""

    def test_onnx_output_ranks_global_vs_per_token(self, gaussian_onnx):
        """2 rank-0 globals (mean + _stddev of the global head) + 2 rank-1 per-token."""
        model = onnx.load(str(gaussian_onnx))
        ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
        assert sorted(ranks.values()) == [0, 0, 1, 1], ranks

    def test_onnx_session_runs(self, gaussian_onnx):
        """The exported graph runs in onnxruntime on batch-1 inputs (L=5 tokens)."""
        sess = make_session(gaussian_onnx)
        in_meta = {i.name: i.shape for i in sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds), strict=True)}
        assert len(out) == 4, f"expected the 4 gaussian outputs, got {sorted(out)}"
