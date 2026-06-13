"""Tests for the M5 regression-family gates harness (`salt.core.gates_m5`, plan 10 sub-wave A).

Each gate runs against its own in-repo dummy fixtures (no machine paths) and is
paired with a python-only ``corruption`` hook proving the parity comparison has
teeth: a perturbed v2 value (or loss) must FAIL the gate while the surface
checks stay green (the gates_m2/m3/m4 pattern, never exposed on the CLI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import salt.core.gates_m5 as gm5
from salt.core.gates_m5 import main, run_r1, run_r2, run_r3, run_r4


class TestR1:
    def test_pass(self, tmp_path):
        code, report = run_r1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "r1_report.json").is_file()
        # every measured diff is at/under the parity bound
        assert all(v <= 1e-6 for v in report["max_abs_diffs"].values())
        # genuine v1-vs-v2 cross-impl parity (independent head), not only
        # self-consistency vs the executor's own composed object
        assert report["checks"]["normparams_fit_pred_vs_independent_v1"]
        assert report["checks"]["normparams_fit_loss_vs_independent_v1"]
        # the functional-scaler per-token SEQUENCE de-scaling is exercised
        assert report["checks"]["scaler_seq_test_descale_vs_independent_v1"]
        assert report["checks"]["scaler_seq_pred_is_3d"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 prediction -> both the self-consistency and the
        # independent-v1 cross-impl forward parity checks must FAIL, but the
        # bind-time denominator rule (a control) stays green
        code, report = run_r1(tmp_path, corruption=lambda p: p + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["normparams_fit_pred_self"]
        assert not report["checks"]["normparams_fit_pred_vs_independent_v1"]
        assert report["checks"]["denominator_not_a_feature_is_bind_error"]


class TestR2:
    def test_pass(self, tmp_path):
        code, report = run_r2(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert report["checks"]["output_size_is_2R"]
        assert report["checks"]["gaussian_test_one_array_parity"]
        # genuine cross-impl parity vs an independent v1 GaussianRegressionTask
        assert report["checks"]["gaussian_fit_loss_vs_independent_v1"]
        assert report["checks"]["gaussian_test_vs_independent_v1"]
        # the ONNX gaussian plan is COMPILED + RUN (not asserted on the TEST array)
        assert report["checks"]["gaussian_onnx_array_is_2R_wide"]
        assert report["checks"]["gaussian_onnx_equals_test"]
        # the design-conformance ONNX-stddev sub-assertion is recorded as such
        assert "NOT a v1-byte-parity gate" in report["onnx_stddev_is_design_conformance"]
        assert report["checks"]["onnx_stddev_design_conformance_positive"]
        # the multi-target R>1 gaussian deferral is documented in the report
        assert "R=1 ONLY" in report["multi_target_gaussian_descale"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 NLL loss -> the FIT loss parity check must FAIL
        code, report = run_r2(tmp_path, corruption=lambda loss: loss + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["gaussian_fit_loss_parity"]


class TestR3:
    def test_pass(self, tmp_path):
        code, report = run_r3(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert report["checks"]["sample_weight_nonuniform_loss_parity"]
        assert report["checks"]["sample_weight_zero_loss_parity"]
        # genuine cross-impl parity vs an independent v1 RegressionTask
        assert report["checks"]["sample_weight_nonuniform_loss_vs_independent_v1"]
        assert report["checks"]["nan_loss_is_finite"]
        assert report["checks"]["nan_masking_loss_parity"]
        # encoder-less pooling forward parity on legacy/dips.yaml (plan 10 R3)
        assert report["checks"]["dips_plan_has_no_registers"]
        assert report["checks"]["dips_encoderless_pool_bitwise"]
        assert report["checks"]["dips_encoderless_pool_parity"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the nonuniform-weight loss + the pooled vector -> their parity
        # checks must FAIL, the NaN-masking check (independent) stays green and
        # the dips no-registers plan check (a static control) stays green
        code, report = run_r3(tmp_path, corruption=lambda loss: loss * 2.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["sample_weight_nonuniform_loss_parity"]
        assert not report["checks"]["dips_encoderless_pool_bitwise"]
        assert report["checks"]["nan_masking_loss_parity"]
        assert report["checks"]["dips_plan_has_no_registers"]


class TestR4:
    def test_pass(self, tmp_path):
        code, report = run_r4(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert report["checks"]["custom_target_chain_bitwise"]
        assert report["checks"]["custom_target_chain_no_nan_left"]
        assert report["checks"]["target_mode_raw_base_bitwise"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 replaced target -> the bitwise check must FAIL
        code, report = run_r4(tmp_path, corruption=lambda arr: arr + np.float32(1.0))
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["custom_target_chain_bitwise"]


class TestCli:
    def test_main_runs_each_gate(self, tmp_path):
        for gate in ("r1", "r2", "r3", "r4"):
            code = main([gate, "--outdir", str(tmp_path / gate)])
            assert code == 0
            assert (tmp_path / gate / f"{gate}_report.json").is_file()


def test_no_machine_paths_in_module():
    """The gate file ships no absolute machine path (design §5 placeholder policy)."""
    src = Path(gm5.__file__).read_text()
    for forbidden in ("/home/", "/data/", "/eos/"):
        assert forbidden not in src, f"machine path {forbidden!r} leaked into gates_m5.py"
