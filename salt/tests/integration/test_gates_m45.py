"""Tests for the U1 gate harness (`salt.tests.integration.gates_m45`, plan 08 stage C).

plan-29 W4 retires the M4.5 writer-derived ONNX manifest: U1 is now a FOLDED-export
coherence gate (the exported ONNX outputs come from the OnnxExportSink, the eval H5
columns stay TaskWriter-derived). U1 runs at reduced sweep scale here; the python-only
``corruption`` hook proves the coherence comparison has teeth — a corrupted ONNX output
surface must FAIL the gate while the surface/control checks stay green (locality, the
gates_m3/m4 pattern).
"""

from __future__ import annotations

import pytest

from salt.tests.integration.gates_m45 import MODEL_NAME, main, run_u1

SMALL = {"batch_size": 100, "num_test": 300, "check_trials": 1, "check_max_length": 6}


def _rename_pb(names: list[str]) -> list[str]:
    """The U1 corruption hook: rename the observed pb output.

    Simulates an exporter output surface that diverged from the OnnxExportSink
    declarations — exactly the drift class the folded export contract kills.

    Returns
    -------
    list[str]
        The corrupted observed-name list.
    """
    return [f"{MODEL_NAME}_qb" if name == f"{MODEL_NAME}_pb" else name for name in names]


class TestU1:
    @pytest.fixture(scope="class")
    def u1(self, tmp_path_factory):
        outdir = tmp_path_factory.mktemp("u1")
        code, report = run_u1(outdir, **SMALL)
        return code, report, outdir

    def test_gate_passes(self, u1):
        code, report, outdir = u1
        assert code == 0
        assert report["passed"]
        assert (outdir / "u1_report.json").exists()

    def test_surface_runs_succeeded(self, u1):
        _, report, _ = u1
        assert all(report["surface"].values()), report["surface"]

    def test_coherence_checks(self, u1):
        _, report, _ = u1
        coherence = report["coherence"]
        assert all(coherence.values()), coherence
        # the observed graph is the folded gn2v2-dummy export (pb/pc/pu globals,
        # then TrackOrigin int8, VertexIndex int8 — no combine/custom-writer outputs)
        observed = report["observed_output_names"]
        assert observed == [
            f"{MODEL_NAME}_pb",
            f"{MODEL_NAME}_pc",
            f"{MODEL_NAME}_pu",
            f"{MODEL_NAME}_TrackOrigin",
            f"{MODEL_NAME}_VertexIndex",
        ]

    def test_in_gate_negative_controls(self, u1):
        _, report, _ = u1
        # the unblessed-stub + config-declared export.outputs migration error
        assert all(report["controls"].values()), report["controls"]

    def test_corruption_hook_fails_the_gate_locally(self, tmp_path):
        # negative control: an exporter output renamed away from the OnnxExportSink
        # declarations must FAIL U1 — and locally: the surface runs and the in-gate
        # controls stay green
        code, report = run_u1(
            tmp_path,
            batch_size=100,
            num_test=200,
            check_trials=1,
            check_max_length=4,
            corruption=_rename_pb,
        )
        assert code == 1
        assert not report["passed"]
        assert report["config"]["corrupted_by_test_hook"]
        coherence = report["coherence"]
        assert not coherence["onnx_graph_equals_sink_manifest"]
        assert not coherence["onnx_graph_equals_pinned_reference"]
        assert not coherence["gnn_config_output_names_match_graph"]
        assert not coherence["global_suffixes_equal_modulo_prefix"]
        # locality: the failure is the coherence comparison, not the runs/controls
        assert all(report["surface"].values())
        assert all(report["controls"].values())


class TestCli:
    def test_main_runs_u1(self, tmp_path):
        code = main([
            "u1",
            "--outdir",
            str(tmp_path),
            "--batch-size",
            "100",
            "--num-test",
            "200",
            "--check-trials",
            "1",
            "--check-max-length",
            "4",
        ])
        assert code == 0
        assert (tmp_path / "u1_report.json").exists()
