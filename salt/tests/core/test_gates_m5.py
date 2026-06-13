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
from salt.core.gates_m5 import main, run_l1, run_l2, run_l3, run_r1, run_r2, run_r3, run_r4


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


class TestL1:
    def test_pass(self, tmp_path):
        code, report = run_l1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "l1_report.json").is_file()
        # >= 2 task losses are genuinely combined (a 1-task GLS is the identity)
        assert report["checks"]["gls_has_two_task_losses"]
        assert report["config"]["n_losses"] >= 2
        # geometric-mean parity vs an INDEPENDENT v1 ModelWrapper.total_loss (GLS)
        assert report["checks"]["gls_total_bitwise_vs_independent_v1"]
        assert report["checks"]["gls_total_parity_vs_independent_v1"]
        assert report["max_abs_diffs"]["gls_total_vs_independent_v1"] <= 1e-6
        # the reference is v1's actual method, not a re-implementation
        assert "ModelWrapper.total_loss" in report["v1_reference"]
        # it is a geometric mean, not a sum
        assert report["checks"]["gls_is_not_the_sum"]
        # the all-weights==1.0 guard fires on BOTH surfaces and is silent at 1.0
        assert report["checks"]["module_weight_not_one_raises"]
        assert report["checks"]["module_weight_one_accepted"]
        assert report["checks"]["task_weight_not_one_raises"]
        assert report["checks"]["task_weight_one_accepted"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb a per-task loss -> the geometric-mean parity vs the independent
        # v1 reference must FAIL, while the weight-guard checks (independent of
        # the loss values) stay green
        code, report = run_l1(tmp_path, corruption=lambda loss: loss + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["gls_total_bitwise_vs_independent_v1"]
        assert not report["checks"]["gls_total_parity_vs_independent_v1"]
        assert report["checks"]["module_weight_not_one_raises"]
        assert report["checks"]["task_weight_not_one_raises"]


class TestL2:
    def test_pass(self, tmp_path):
        code, report = run_l2(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "l2_report.json").is_file()
        # hybrid encoder forward parity vs an INDEPENDENT v1 Transformer (BITWISE)
        assert report["checks"]["hybrid_encoded_bitwise_vs_independent_v1"]
        assert report["checks"]["hybrid_encoded_parity_vs_independent_v1"]
        assert report["max_abs_diffs"]["hybrid_encoded_vs_independent_v1"] <= 1e-6
        # the reference is an independently-constructed v1 Transformer
        assert "salt.models.Transformer" in report["v1_reference"]
        # the hybrid flag genuinely reached every composed v1 EncoderLayer
        assert report["checks"]["all_layers_hybrid"]
        assert report["checks"]["all_layers_force_qk_v_norm"]
        assert report["checks"]["depth0_pre_residual_identity_norm"]
        assert report["checks"]["deeper_layers_none_residual_real_norm"]
        # matched-init A/B control: pre-norm encoder loaded with the hybrid
        # weights differs ONLY by placement, so the copy must be complete and
        # the output must still differ
        assert report["checks"]["matched_init_no_missing_pre_keys"]
        assert report["checks"]["matched_init_unexpected_are_qk_v_norms"]
        assert report["checks"]["hybrid_differs_from_pre"]
        assert report["checks"]["unknown_norm_type_raises"]
        # no ONNX claim is made (documented honestly in the report)
        assert "no ONNX claim" in report["no_onnx_claim"] or "ONNX" in report["no_onnx_claim"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 encoded.seq -> the bitwise + parity checks vs the
        # independent v1 Transformer must FAIL, while the flag-reached-the-layers
        # and reject-unknown-norm_type checks (independent of the values) stay green
        code, report = run_l2(tmp_path, corruption=lambda enc: enc + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["hybrid_encoded_bitwise_vs_independent_v1"]
        assert not report["checks"]["hybrid_encoded_parity_vs_independent_v1"]
        assert report["checks"]["all_layers_hybrid"]
        assert report["checks"]["all_layers_force_qk_v_norm"]
        assert report["checks"]["unknown_norm_type_raises"]


class TestL3:
    def test_pass(self, tmp_path):
        code, report = run_l3(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "l3_report.json").is_file()
        # VectorConcat output == the literal v1 cat([pooled, global]) BITWISE
        assert report["checks"]["vconcat_bitwise_vs_v1_cat"]
        assert report["checks"]["vconcat_parity_vs_v1_cat"]
        assert report["max_abs_diffs"]["vconcat_vs_v1_cat"] == 0.0
        # the reference is the literal saltmodel.py:175-177 cat (no v1 class)
        assert "saltmodel.py:175-177" in report["v1_reference"]
        # order: pooled FIRST, global features LAST + Dsum = sum
        assert report["checks"]["order_pooled_first"]
        assert report["checks"]["order_global_last"]
        assert report["checks"]["dsum_is_input_width_sum"]
        # alias: a COMPILED + RUN Mode.ONNX plan (identity clone + name gather)
        assert report["checks"]["identity_alias_is_clone"]
        assert report["checks"]["onnx_identity_runs_and_matches_eager"]
        assert report["max_abs_diffs"]["onnx_identity_vs_eager"] <= 1e-6
        assert report["checks"]["name_gather_index_resolves_by_name"]
        assert report["checks"]["name_gather_reorders_columns"]
        assert report["checks"]["name_gather_is_not_identity"]
        # the ONNX claim is honestly backed by a real Mode.ONNX run
        assert "Mode.ONNX" in report["onnx_claim"]
        # loud construction / bind surfaces
        assert report["checks"]["duplicate_inputs_raise"]
        assert report["checks"]["self_feed_out_in_inputs_raises"]
        assert report["checks"]["alias_missing_column_raises"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 vconcat.global -> the order/Dsum/parity checks vs the
        # literal v1 cat must FAIL, while the alias-run and loud-surface checks
        # (independent of the FIT concat values) stay green
        code, report = run_l3(tmp_path, corruption=lambda vc: vc + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["vconcat_bitwise_vs_v1_cat"]
        assert not report["checks"]["vconcat_parity_vs_v1_cat"]
        # the +1.0 shift breaks the pooled/global column split too
        assert not report["checks"]["order_pooled_first"]
        assert not report["checks"]["order_global_last"]
        # Dsum (a width, not a value) and the alias/loud checks stay green
        assert report["checks"]["dsum_is_input_width_sum"]
        assert report["checks"]["identity_alias_is_clone"]
        assert report["checks"]["onnx_identity_runs_and_matches_eager"]
        assert report["checks"]["name_gather_reorders_columns"]
        assert report["checks"]["duplicate_inputs_raise"]
        assert report["checks"]["alias_missing_column_raises"]


class TestCli:
    def test_main_runs_each_gate(self, tmp_path):
        for gate in ("r1", "r2", "r3", "r4", "l1", "l2", "l3"):
            code = main([gate, "--outdir", str(tmp_path / gate)])
            assert code == 0
            assert (tmp_path / gate / f"{gate}_report.json").is_file()


def test_no_machine_paths_in_module():
    """The gate file ships no absolute machine path (design §5 placeholder policy)."""
    src = Path(gm5.__file__).read_text()
    for forbidden in ("/home/", "/data/", "/eos/"):
        assert forbidden not in src, f"machine path {forbidden!r} leaked into gates_m5.py"
