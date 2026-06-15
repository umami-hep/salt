"""Tests for the M6 gates harness (`salt.core.gates_m6`, plan 12 sub-wave A).

Each gate runs against its own in-repo dummy fixtures (no machine paths) and is
paired with a python-only ``corruption`` hook proving the parity comparison has
teeth: a perturbed v2 value must FAIL the gate while the independent guard
checks stay green (the gates_m2/m3/m4/m5 pattern, never exposed on the CLI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import salt.core.gates_m6 as gm6
from salt.core.gates_m6 import main, run_conv, run_lb1, run_mu1, run_mu2, run_vs1


class TestLB1:
    def test_pass(self, tmp_path):
        code, report = run_lb1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "lb1_report.json").is_file()
        # GN3X 9-class, require_labels:True — bitwise int64 parity vs v1
        assert report["checks"]["gn3x_require_true_bitwise"]
        assert report["checks"]["gn3x_dtype_is_int64"]
        assert report["checks"]["gn3x_spans_nine_classes"]
        # GN2X_qcdsplit-style 7-class, require_labels:False — drops unlabelled
        assert report["checks"]["qcdsplit_require_false_bitwise"]
        assert report["checks"]["qcdsplit_drops_unlabelled"]
        assert report["config"]["n_qcdsplit_labelled"] < report["config"]["n_qcdsplit_rows"]
        # read_fields declares the labeller cut variables, not the derived label
        assert report["checks"]["read_fields_declares_cut_variables"]
        assert report["checks"]["read_fields_excludes_derived_label"]
        # named-error guards all fire
        assert report["checks"]["empty_class_guard_raises_configerror"]
        assert report["checks"]["missing_field_guard_raises_valueerror"]
        assert report["checks"]["require_labels_true_raises_on_unlabelled"]
        # the reference is an INDEPENDENT v1 recomputation, not the v2 output
        assert "INDEPENDENT v1 reference" in report["v1_reference"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 GN3X derived labels -> the bitwise parity check must
        # FAIL, while the named-error guards (independent of label values) stay
        # green
        code, report = run_lb1(tmp_path, corruption=lambda arr: arr + np.int64(1))
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["gn3x_require_true_bitwise"]
        assert report["checks"]["empty_class_guard_raises_configerror"]
        assert report["checks"]["missing_field_guard_raises_valueerror"]
        assert report["checks"]["require_labels_true_raises_on_unlabelled"]


class TestVS1:
    """VS1 — the model-side rank-2 [B, F] vector-stream embed (plan 12 sub-wave D).

    Pins the bitwise forward parity vs the INDEPENDENT v1 no-encoder/no-pool DL1
    path, the rank-2 / no-encoder structural checks, the ONNX no-T leg (B the
    sole dynamic axis), and the corruption teeth (a perturbed v2 logit fails the
    parity while the structural checks stay green).
    """

    def test_pass(self, tmp_path):
        code, report = run_vs1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "vs1_report.json").is_file()
        # rank-2 [B, F] -> [B, D] embed, no token axis
        assert report["checks"]["embed_input_is_rank2"]
        assert report["checks"]["embed_output_is_rank2"]
        assert report["checks"]["pred_is_rank2_b_nclasses"]
        # no-encoder / no-pool plan; head reads the embed directly
        assert report["checks"]["no_encoder_in_plan"]
        assert report["checks"]["no_pool_in_plan"]
        assert report["checks"]["no_seq_or_pooled_edges"]
        assert report["checks"]["head_consumes_embed_directly"]
        # bitwise forward parity vs the INDEPENDENT v1 reference
        assert report["checks"]["forward_bitwise_vs_v1"]
        assert "INDEPENDENT v1 reference" in report["v1_reference"]
        # ONNX no-T leg: B sole dynamic axis, onnxruntime agrees with eager
        assert report["checks"]["no_token_dynamic_axis"]
        # the no-T claim is non-vacuous — a sequence:true PROBE over the SAME
        # plan DOES register an n_<stream> axis (negative control)
        assert report["checks"]["no_token_axis_check_is_non_vacuous"]
        assert report["checks"]["onnx_input_is_rank2"]
        assert report["checks"]["onnx_batch_axis_is_dynamic"]
        assert report["checks"]["onnx_runtime_matches_eager_multibatch"]
        assert report["checks"]["onnx_check_passes"]
        # the chosen approach is the StreamEmbed rank-2 path (recorded in report)
        assert "StreamEmbed rank-2" in report["config"]["approach"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 logits -> the bitwise parity check must FAIL, while the
        # structural (rank-2 / no-encoder / no-T) checks stay green
        code, report = run_vs1(tmp_path, corruption=lambda t: t + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["forward_bitwise_vs_v1"]
        # the structural + onnx checks are independent of the logit values
        assert report["checks"]["embed_input_is_rank2"]
        assert report["checks"]["embed_output_is_rank2"]
        assert report["checks"]["no_encoder_in_plan"]
        assert report["checks"]["no_pool_in_plan"]
        assert report["checks"]["no_token_dynamic_axis"]
        assert report["checks"]["no_token_axis_check_is_non_vacuous"]
        assert report["checks"]["onnx_input_is_rank2"]


class TestMU1:
    """MU1 — the muP architectural port (plan 12 sub-wave B).

    Pins: the structural muP wiring (mup-Dense embed + zeroed-init MuReadout
    out-proj), the BITWISE forward-init parity vs the INDEPENDENT v1
    Transformer(mup=True), the MuReadout->plain-Linear export fold (bitwise at the
    unit width_mult, <=1e-6 for a non-unit probe), and the corruption teeth (a
    perturbed v2 encoded sequence fails the parity while the structural + fold
    checks stay green).
    """

    def test_pass(self, tmp_path):
        code, report = run_mu1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "mu1_report.json").is_file()
        # structural muP wiring: mup-Dense embed + MuReadout out-proj zeroed at init
        assert report["checks"]["embed_dense_is_mup"]
        assert report["checks"]["embed_net_built_with_mup"]
        assert report["checks"]["encoder_is_mup"]
        assert report["checks"]["out_proj_is_mu_readout"]
        assert report["checks"]["out_proj_weight_zeroed_at_init"]
        assert report["checks"]["out_proj_bias_zeroed_at_init"]
        # BITWISE forward-init parity vs the INDEPENDENT v1 Transformer(mup=True)
        assert report["checks"]["encoder_forward_bitwise_vs_v1_mup"]
        assert "INDEPENDENT v1 reference" in report["v1_reference"]
        # MuReadout -> plain-Linear export fold: bitwise at unit mult, idempotent
        assert report["checks"]["out_proj_folded_to_plain_linear"]
        assert report["checks"]["export_fold_equals_mu_readout_bitwise_at_unit_mult"]
        assert report["checks"]["export_fold_within_tolerance"]
        assert report["checks"]["export_fold_is_idempotent"]
        # the <=1e-6 leg is non-vacuous: a non-unit width_mult probe still folds
        assert report["checks"]["nonunit_width_mult_fold_within_tolerance"]
        assert report["config"]["nonunit_probe_width_mult"] != 1.0
        # the honest faithfulness note (v1 encoder flag wires only the out-proj)
        assert "does NOT pass mup down to its EncoderLayers" in report["faithfulness_note"]
        assert "MuReadout out-proj swap" in report["faithfulness_note"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 encoded sequence -> the bitwise parity check must FAIL,
        # while the structural (mup-Dense / MuReadout) and export-fold checks stay
        # green (they are independent of the encoded values)
        code, report = run_mu1(tmp_path, corruption=lambda t: t + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["encoder_forward_bitwise_vs_v1_mup"]
        # structural + fold checks are value-independent and stay green
        assert report["checks"]["embed_dense_is_mup"]
        assert report["checks"]["encoder_is_mup"]
        assert report["checks"]["out_proj_is_mu_readout"]
        assert report["checks"]["out_proj_weight_zeroed_at_init"]
        assert report["checks"]["out_proj_folded_to_plain_linear"]
        assert report["checks"]["export_fold_within_tolerance"]
        assert report["checks"]["nonunit_width_mult_fold_within_tolerance"]


class TestMU2:
    """MU2 — the muP routing validator (plan 12 sub-wave B; design §3.4 line 695).

    Pins: the validator rules (apply_to without a mup init_arg / a non-existent
    module ERRORS; a mup:true module outside apply_to WARNS; unknown key / empty
    apply_to ERROR), the MuAdamW swap (selected when mup is configured, AdamW
    otherwise), the setup_mup entry-point casing fix, the end-to-end salt2
    mup-shapes + graph validate on a real config (with the shape file applied at
    bind so width_mult resolves to the real base ratio), and the corruption teeth
    (an apply_to'd non-mup module flips the validator to ERROR).
    """

    def test_pass(self, tmp_path):
        code, report = run_mu2(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "mu2_report.json").is_file()
        # validator rules (design §3.4 line 695)
        assert report["checks"]["valid_apply_to_accepted"]
        assert report["checks"]["apply_to_non_mup_module_errors"]
        assert report["checks"]["apply_to_unknown_module_errors"]
        assert report["checks"]["mup_on_module_outside_apply_to_warns"]
        assert report["checks"]["unknown_mup_key_errors"]
        assert report["checks"]["empty_apply_to_errors"]
        # MuAdamW swap + AdamW fallback
        assert report["checks"]["muadamw_selected_when_mup_configured"]
        assert report["checks"]["adamw_selected_when_no_mup"]
        # entry-point casing fix
        assert report["checks"]["setup_mup_entry_point_callable"]
        assert "salt.core.mup:setup_mup" in report["entry_point_fix"]
        assert "main_muP.py (capital P)" in report["entry_point_fix"]
        # end-to-end tooling: shapes generated, applied at bind, validate OK
        assert report["checks"]["salt2_mup_shapes_runs"]
        assert report["checks"]["shape_file_applied_at_bind_width_mult_2"]
        assert report["config"]["resolved_width_mult"] == 2.0
        assert report["checks"]["salt2_graph_validate_mup_routing_ok"]
        # the v1->v2 routing reference (regex zip -> explicit name list)
        assert "configuration_muP.py:98" in report["v1_reference"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # inject a non-mup module (pool) into apply_to -> the validator ERRORS,
        # so valid_apply_to_accepted flips False and the gate exits 1
        def break_apply_to(block):
            return {"apply_to": ["track_embed", "pool"]}

        code, report = run_mu2(tmp_path, corruption=break_apply_to)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["valid_apply_to_accepted"]
        # the value-independent validator-rule checks stay green (they build their
        # own module dicts, untouched by the corruption hook)
        assert report["checks"]["apply_to_non_mup_module_errors"]
        assert report["checks"]["muadamw_selected_when_mup_configured"]


class TestConv:
    """M6-CONV — the M7-slice acceptance for the M6-authored v2-native configs.

    The gate drives the REAL ``salt2 graph validate`` (fit/test/onnx) on every
    config landed so far (``_CONV_M6_CONFIGS`` — sub-wave A: GN3X, GN2X_qcdsplit;
    sub-wave D: DL1; sub-wave B: GN2_muP) and embeds the authoritative list
    verbatim. These tests pin:
    the file-present check, the per-config mode bookkeeping (all standard traces
    -> onnx validated; GN2_muP's MuReadout out-proj folded to plain Linear at
    trace time), the muP shape-generation prerequisite, and the corruption teeth
    (a missing file fails the gate).
    """

    def test_structure_and_files_present(self, tmp_path):
        _code, report = run_conv(tmp_path)
        assert (tmp_path / "conv_report.json").is_file()
        # every embedded config file exists in salt/core/configs/
        assert report["checks"]["all_m6_conv_config_files_present"]
        assert report["config"]["missing_config_files"] == []
        # waves A + D + B land exactly these 4 configs, names embedded verbatim
        names = {c["name"] for c in report["configs"]}
        assert names == {"GN3X", "GN2X_qcdsplit", "DL1", "GN2_muP"}
        # the report embeds the authoritative _CONV_M6_CONFIGS list verbatim
        assert report["conv_m6_configs"] == ["GN3X", "GN2X_qcdsplit", "DL1", "GN2_muP"]
        assert report["config"]["total_configs"] == 4
        # no --strict (flow/truth_hadrons preflight warnings inherent — documented)
        assert report["config"]["strict"] is False
        assert "no --strict" in report["no_strict_rationale"]
        # the gate makes NO forward-parity claim (owned by LB1)
        assert "NO forward-parity claim" in report["scope_note"]

    def test_all_configs_validate_all_modes(self, tmp_path):
        # GN3X + GN2X_qcdsplit + DL1 + GN2_muP convert+validate+plan-compile in
        # fit/test/onnx — all export-representable (GN2_muP via the MuReadout
        # fold), so onnx is validated for each
        code, report = run_conv(tmp_path)
        for c in report["configs"]:
            assert c["validateFit"], c
            assert c["validateTest"], c
            assert c["validateOnnx"] is True, c
            assert c["rc"]["fit"] == 0, c
            assert c["rc"]["test"] == 0, c
            assert c["rc"]["onnx"] == 0, c
        assert report["config"]["validated_configs"] == 4
        assert code == 0
        assert report["passed"]

    def test_mup_config_shapes_generated(self, tmp_path):
        # the GN2_muP entry needs its base/delta infshapes generated data-free
        # before validation (the committed shape_path is a placeholder; sub-wave B)
        _code, report = run_conv(tmp_path)
        assert report["checks"]["GN2_muP:mup_shapes_generated"]
        gn2_mup = next(c for c in report["configs"] if c["name"] == "GN2_muP")
        assert Path(gn2_mup["shape_path"]).is_file()
        assert gn2_mup["family"] == "mup"

    def test_corruption_missing_file_fails(self, tmp_path):
        # point a config at a non-existent file -> the file-present check fails
        # AND the per-mode validation is skipped/failed, so the gate exits 1
        def break_gn3x(configs):
            out = [dict(c) for c in configs]
            out[0]["cfg"] = ("does_not_exist_m6_conv_probe.yaml",)
            return out

        code, report = run_conv(tmp_path, corruption=break_gn3x)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["all_m6_conv_config_files_present"]
        assert "GN3X" in report["config"]["missing_config_files"]


class TestCli:
    def test_main_runs_each_gate(self, tmp_path):
        gates = ("lb1", "vs1", "mu1", "mu2", "conv")
        for gate in gates:
            code = main([gate, "--outdir", str(tmp_path / gate)])
            assert code == 0
            assert (tmp_path / gate / f"{gate}_report.json").is_file()


def test_no_machine_paths_in_module():
    """The gate file ships no absolute machine path (design §5 placeholder policy)."""
    src = Path(gm6.__file__).read_text()
    for forbidden in ("/home/", "/data/", "/eos/"):
        assert forbidden not in src, f"machine path {forbidden!r} leaked into gates_m6.py"
