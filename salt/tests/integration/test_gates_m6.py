"""Tests for the M6 gates harness (`salt.tests.integration.gates_m6`, plan 12 sub-wave A).

Each gate runs against its own in-repo dummy fixtures (no machine paths) and is
paired with a python-only ``corruption`` hook proving the parity comparison has
teeth: a perturbed v2 value must FAIL the gate while the independent guard
checks stay green (the gates_m2/m3/m4/m5 pattern, never exposed on the CLI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jsonargparse import Namespace

import salt.tests.integration.gates_m6 as gm6
from salt.tests.integration.gates_m6 import (
    main,
    run_cm1,
    run_cm2,
    run_conv,
    run_ed1,
    run_ed2,
    run_lb1,
    run_lr1,
    run_mu1,
    run_mu2,
    run_s31,
    run_vs1,
)


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
        # rank-agnostic embed specs (M7 W1.5 wave R): the rank-2 [B, F] -> [B, D]
        # boundary is proven end-to-end (compiled plan + forward shape + ONNX)
        assert report["checks"]["embed_input_is_rank_agnostic"]
        assert report["checks"]["embed_output_is_rank_agnostic"]
        assert report["checks"]["embed_out_dim_via_derived_widths"]
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
        assert report["checks"]["embed_input_is_rank_agnostic"]
        assert report["checks"]["embed_output_is_rank_agnostic"]
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
        assert not np.isclose(report["config"]["nonunit_probe_width_mult"], 1.0)
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
        assert np.isclose(report["config"]["resolved_width_mult"], 2.0)
        assert report["checks"]["salt2_graph_validate_mup_routing_ok"]
        # the v1->v2 routing reference (regex zip -> explicit name list)
        assert "configuration_muP.py:98" in report["v1_reference"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # inject a non-mup module (pool) into apply_to -> the validator ERRORS,
        # so valid_apply_to_accepted flips False and the gate exits 1
        def break_apply_to(_block):
            return {"apply_to": ["track_embed", "pool"]}

        code, report = run_mu2(tmp_path, corruption=break_apply_to)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["valid_apply_to_accepted"]
        # the value-independent validator-rule checks stay green (they build their
        # own module dicts, untouched by the corruption hook)
        assert report["checks"]["apply_to_non_mup_module_errors"]
        assert report["checks"]["muadamw_selected_when_mup_configured"]


class TestED1:
    """ED1 — the edge path end-to-end (plan 12 sub-wave C; FD §6.7 1418-1431).

    Pins the bitwise forward parity of EdgeFeatures vs an INDEPENDENT v1
    EdgeConstructor (per-feature for dR/kt/z), of EdgeEmbed vs an INDEPENDENT v1
    Dense, AND of the FULL GN2XE-shaped encoder edge path vs an INDEPENDENT v1
    Transformer(edge_embed_dim>0, update_edges=True); the rank-4 / raw-input /
    shared-T-symbol structural checks (incl. the encoder's edge port); the
    dynamic-edge-T ONNX-trace assertion (both edge T axes dynamic + shape-derived
    register pad, proven by an onnxruntime length sweep from a single trace); the
    write-once guarantee; the named-error guards; and the corruption teeth (a
    perturbed v2 edge tensor fails the parity while the structural + ONNX + guard
    checks stay green).
    """

    def test_pass(self, tmp_path):
        code, report = run_ed1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "ed1_report.json").is_file()
        # declared-IO structure: raw input (not normed), pad mask, square rank-4
        assert report["checks"]["edge_features_requires_raw_input"]
        assert report["checks"]["edge_features_not_consuming_normed"]
        assert report["checks"]["edge_features_requires_pad_mask"]
        assert report["checks"]["edges_tensor_is_rank4"]
        assert report["checks"]["edges_both_token_axes_share_stream_symbol"]
        assert report["checks"]["edges_last_dim_is_feature_count"]
        assert report["checks"]["edge_embed_is_rank4"]
        assert report["checks"]["edge_embed_token_axes_share_stream_symbol"]
        assert report["checks"]["edge_embed_out_dim_matches"]
        # indices_map resolved by NAME from the declared fields (v1 parity)
        assert report["checks"]["indices_map_resolved_by_name"]
        # runtime shapes [B, T, T, E] / [B, T, T, D_e]
        assert report["checks"]["edges_runtime_shape"]
        assert report["checks"]["edge_embed_runtime_shape"]
        # write-once: inputs.tracks never mutated (both the edge-only + encoder plans)
        assert report["checks"]["inputs_not_mutated"]
        assert report["checks"]["encoder_inputs_not_mutated"]
        # bitwise forward parity vs the INDEPENDENT v1 references
        assert report["checks"]["edge_features_bitwise_vs_v1"]
        # all 5 edge features pinned column-by-column (kinematic + non-kinematic)
        assert report["checks"]["edge_feature_dR_bitwise_vs_v1"]
        assert report["checks"]["edge_feature_z_bitwise_vs_v1"]
        assert report["checks"]["edge_feature_kt_bitwise_vs_v1"]
        assert report["checks"]["edge_feature_subjetIndex_bitwise_vs_v1"]
        assert report["checks"]["edge_feature_isSelfLoop_bitwise_vs_v1"]
        assert report["checks"]["edge_embed_bitwise_vs_v1"]
        assert "INDEPENDENT v1 reference" in report["v1_reference"]
        # encoder edge path: requires the rank-4 edge port (both axes T:tracks),
        # encoded.seq matches an INDEPENDENT v1 Transformer(edge, update_edges)
        assert report["checks"]["encoder_requires_edge_port"]
        assert report["checks"]["encoder_edge_port_is_rank4"]
        assert report["checks"]["encoder_edge_port_token_axes_share_stream_symbol"]
        assert report["checks"]["encoder_edge_port_width_matches"]
        assert report["checks"]["encoder_encoded_runtime_shape"]
        assert report["checks"]["encoder_edge_forward_bitwise_vs_v1"]
        # the dynamic-edge-T ONNX trace: track axis dynamic, register pad
        # shape-derived, onnxruntime agrees across multiple track counts
        assert report["checks"]["edge_track_axis_is_dynamic"]
        assert report["checks"]["onnx_track_input_token_dim_symbolic"]
        assert report["checks"]["register_pad_shape_derived_ops_present"]
        assert report["checks"]["onnx_dynamic_edge_T_sweep_passes"]
        assert report["checks"]["onnx_sweep_covered_multiple_lengths"]
        assert report["dynamic_axes"]["track_features"] == {"0": "n_tracks"}
        # named-error guards all fire
        assert report["checks"]["unknown_feature_raises_configerror"]
        assert report["checks"]["empty_features_raises_configerror"]
        assert report["checks"]["missing_required_var_raises_valueerror"]
        assert report["checks"]["edge_embed_width_key_raises_configerror"]
        # the scope note records that the GN2XE corpus config is the next stage
        assert "GN2XE corpus config is the next sub-wave C stage" in report["config"]["stage_scope"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 edge tensor AND the encoded sequence -> the bitwise
        # parities must FAIL, while the structural + ONNX + guard checks stay green
        code, report = run_ed1(tmp_path, corruption=lambda t: t + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["edge_features_bitwise_vs_v1"]
        assert not report["checks"]["edge_feature_dR_bitwise_vs_v1"]
        assert not report["checks"]["edge_feature_subjetIndex_bitwise_vs_v1"]
        assert not report["checks"]["edge_feature_isSelfLoop_bitwise_vs_v1"]
        assert not report["checks"]["encoder_edge_forward_bitwise_vs_v1"]
        # structural + ONNX + guard checks are independent of the edge values
        assert report["checks"]["edges_tensor_is_rank4"]
        assert report["checks"]["edge_features_requires_raw_input"]
        assert report["checks"]["edge_embed_out_dim_matches"]
        assert report["checks"]["encoder_requires_edge_port"]
        assert report["checks"]["onnx_dynamic_edge_T_sweep_passes"]
        assert report["checks"]["register_pad_shape_derived_ops_present"]
        assert report["checks"]["unknown_feature_raises_configerror"]
        assert report["checks"]["missing_required_var_raises_valueerror"]
        assert report["checks"]["edge_embed_width_key_raises_configerror"]


class TestED2:
    """ED2 — the edge bind-time validators (plan 12 sub-wave C; FD §6.7 1425-1431).

    Pins the two named constraints (edge-stream-first / EdgeAttention-backend
    forcing), the ctor-level edge guards, the design-conformance that the SAME
    validator runs in SaltModule.__init__ + is exported for the CLI path, and the
    corruption teeth (reordering the concat so the edge stream is no longer first
    flips the validator to ERROR).
    """

    def test_pass(self, tmp_path):
        code, report = run_ed2(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "ed2_report.json").is_file()
        # rule (a) edge-stream-first
        assert report["checks"]["valid_edge_order_accepted"]
        assert report["checks"]["edge_stream_not_first_errors"]
        assert report["checks"]["edge_port_without_concat_errors"]
        # rule (b) EdgeAttention-backend forcing
        assert report["checks"]["non_edge_backend_errors"]
        assert report["checks"]["no_edge_encoder_is_noop"]
        # ctor-level edge guards
        assert report["checks"]["edges_without_dim_raises_configerror"]
        assert report["checks"]["edge_dim_without_edges_raises_configerror"]
        assert report["checks"]["update_edges_without_edges_raises_configerror"]
        # same validator both places (SaltModule.__init__ + CLI export)
        assert report["checks"]["saltmodule_accepts_valid_edge_order"]
        assert report["checks"]["saltmodule_rejects_misordered_edge_concat"]
        assert report["checks"]["validate_edge_port_exported"]
        # the reference records v1 had NO validators (the two silent hacks)
        assert "v1 had NO edge validators" in report["v1_reference"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # reorder the concat so the edge stream (tracks) is no longer first ->
        # the valid-order acceptance check flips False and the gate exits 1
        code, report = run_ed2(tmp_path, corruption=lambda streams: list(reversed(streams)))
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["valid_edge_order_accepted"]
        # the value-independent validator-rule checks stay green (they build their
        # own module dicts, untouched by the corruption hook)
        assert report["checks"]["edge_stream_not_first_errors"]
        assert report["checks"]["non_edge_backend_errors"]
        assert report["checks"]["saltmodule_rejects_misordered_edge_concat"]


class TestConv:
    """M6-CONV — the M7-slice acceptance for the M6-authored v2-native configs.

    The gate drives the REAL ``salt2 graph validate`` (fit/test/onnx) on every
    config in ``_CONV_M6_CONFIGS`` — sub-wave A: GN3X, GN2X_qcdsplit; sub-wave D:
    DL1; sub-wave B: GN2_muP; sub-wave C: GN2XE (the list is COMPLETE at 5, the
    FINAL config-gating wave) — and embeds the authoritative list
    verbatim. These tests pin:
    the file-present check, the per-config mode bookkeeping (all standard traces
    -> onnx validated; GN2_muP's MuReadout out-proj folded to plain Linear at
    trace time; GN2XE's edge path tracing with dynamic edge T-axes, ED1), the muP
    shape-generation prerequisite, and the corruption teeth (a missing file fails
    the gate).
    """

    def test_structure_and_files_present(self, tmp_path):
        _code, report = run_conv(tmp_path)
        assert (tmp_path / "conv_report.json").is_file()
        # every embedded config file exists in salt/core/configs/
        assert report["checks"]["all_m6_conv_config_files_present"]
        assert report["config"]["missing_config_files"] == []
        # waves A + D + B + C land exactly these 5 configs, names embedded verbatim
        names = {c["name"] for c in report["configs"]}
        assert names == {"GN3X", "GN2X_qcdsplit", "DL1", "GN2_muP", "GN2XE"}
        # the report embeds the authoritative _CONV_M6_CONFIGS list verbatim
        assert report["conv_m6_configs"] == [
            "GN3X",
            "GN2X_qcdsplit",
            "DL1",
            "GN2_muP",
            "GN2XE",
        ]
        assert report["config"]["total_configs"] == 5
        # no --strict (flow/truth_hadrons preflight warnings inherent — documented)
        assert report["config"]["strict"] is False
        assert "no --strict" in report["no_strict_rationale"]
        # the gate makes NO forward-parity claim (owned by LB1)
        assert "NO forward-parity claim" in report["scope_note"]

    def test_all_configs_validate_all_modes(self, tmp_path):
        # GN3X + GN2X_qcdsplit + DL1 + GN2_muP + GN2XE convert+validate+
        # plan-compile in fit/test/onnx — all export-representable (GN2_muP via
        # the MuReadout fold, GN2XE via the dynamic-edge-T trace proven by ED1),
        # so onnx is validated for each
        code, report = run_conv(tmp_path)
        for c in report["configs"]:
            assert c["validateFit"], c
            assert c["validateTest"], c
            assert c["validateOnnx"] is True, c
            assert c["rc"]["fit"] == 0, c
            assert c["rc"]["test"] == 0, c
            assert c["rc"]["onnx"] == 0, c
        assert report["config"]["validated_configs"] == 5
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

    def test_edge_config_validates_onnx(self, tmp_path):
        # GN2XE (sub-wave C, edges family) is the FINAL config-gating mover. Its
        # --mode onnx STAYS in the gate (static plan-compile here; the load-bearing
        # dynamic-T trace is proven by ED1) — NOT scoped out. The 39-denominator
        # closes with this 5th config.
        _code, report = run_conv(tmp_path)
        gn2xe = next(c for c in report["configs"] if c["name"] == "GN2XE")
        assert gn2xe["family"] == "edges"
        assert gn2xe["validateFit"]
        assert gn2xe["validateTest"]
        assert gn2xe["validateOnnx"] is True
        assert gn2xe["rc"]["onnx"] == 0
        # the scope note records the 39-denominator is now CLOSED
        assert "CLOSED" in report["scope_note"]
        assert "NOT scoped out" in report["scope_note"]

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


class TestCM1:
    """CM1 — Comet logger + LearningRateMonitor wiring (plan 12 sub-wave E; NON-gating).

    Pins the comet-before-lightning import order, the fit-stage Comet wiring
    (experiment_name + run-name label + online-auto-false + offline dir), the
    test-path logger=False contract, the base2 LearningRateMonitor entry, and the
    corruption teeth (blanking the logger block no-ops the wiring).
    """

    def test_pass(self, tmp_path):
        code, report = run_cm1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "cm1_report.json").is_file()
        # comet-before-lightning import order (v1 main.py:5)
        assert report["checks"]["comet_imported_before_lightning"]
        # fit-stage Comet wiring (experiment_name + name label + online + offline)
        assert report["checks"]["logger_experiment_name_is_run_name"]
        assert report["checks"]["logger_run_name_label_set"]
        assert report["checks"]["online_auto_set_false_without_api_key"]
        assert report["checks"]["offline_directory_set_and_created"]
        assert report["checks"]["comet_logger_instantiates"]
        # test path forces logger off + LearningRateMonitor in base2 callbacks dict
        assert report["checks"]["test_path_disables_logger_in_source"]
        assert report["checks"]["lr_monitor_in_base2_callbacks_dict"]
        assert report["checks"]["lr_monitor_instantiates"]
        # NON-gating UX (no model-reproduction assertion)
        assert report["config"]["non_gating"] is True

    def test_corruption_fails_the_gate(self, tmp_path):
        # blank the logger block on the fit namespace -> the wiring no-ops, so the
        # logger-wiring checks fail; the import-order / base2 invariants stay green
        def blank_logger(cfg: Namespace) -> Namespace:
            cfg.trainer.logger = False
            return cfg

        code, report = run_cm1(tmp_path, corruption=blank_logger)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["logger_block_configured"]
        assert not report["checks"]["logger_experiment_name_is_run_name"]
        # value-independent invariants stay green
        assert report["checks"]["comet_imported_before_lightning"]
        assert report["checks"]["lr_monitor_in_base2_callbacks_dict"]


class TestCM2:
    """CM2 — CometLogger default-FLIP + gate hygiene (plan-24 Wave 0; NON-gating).

    Pins the flipped base2 default (the documented CometLogger block, no longer
    logger: false), the online-auto-false + offline-dir wiring ON that default,
    the no-offline-archive contract under a REAL salt2 fit --trainer.logger false,
    the lr_monitor kept-with-logger / dropped-without assembly rule, and the
    corruption teeth (reverting the default to False fails the flip check).
    """

    def test_pass(self, tmp_path):
        code, report = run_cm2(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "cm2_report.json").is_file()
        # (a) the flipped default IS the documented CometLogger block
        assert report["checks"]["default_logger_is_cometlogger_block"]
        assert report["checks"]["default_logger_not_false"]
        assert report["checks"]["default_logger_project_is_salt"]
        # (b) online auto-false + offline dir on that default without an api key
        assert report["checks"]["online_auto_false_on_default_without_api_key"]
        assert report["checks"]["offline_directory_set_and_created_on_default"]
        # (c) a REAL salt2 fit --trainer.logger false runs + writes NO *.zip archive
        assert report["checks"]["logger_off_fit_runs"]
        assert report["checks"]["no_offline_archive_when_logger_off"]
        assert report["checks"]["offline_dir_not_set_when_logger_off"]
        # (d) lr_monitor present-with-logger / dropped-without
        assert report["checks"]["lr_monitor_needs_logger_flagged"]
        assert report["checks"]["lr_monitor_kept_with_logger"]
        assert report["checks"]["lr_monitor_dropped_without_logger"]
        # NON-gating UX (no model-reproduction assertion)
        assert report["config"]["non_gating"] is True

    def test_corruption_fails_the_gate(self, tmp_path):
        # revert the base2 default back to `logger: false` -> the flip checks must
        # FAIL, while the value-independent lr_monitor assembly rule stays green
        def revert_to_false(base2: dict) -> dict:
            base2.setdefault("trainer", {})["logger"] = False
            return base2

        code, report = run_cm2(tmp_path, corruption=revert_to_false)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["default_logger_is_cometlogger_block"]
        assert not report["checks"]["default_logger_not_false"]
        # the logger-off-run hygiene + the assembly rule are independent invariants
        assert report["checks"]["no_offline_archive_when_logger_off"]
        assert report["checks"]["lr_monitor_kept_with_logger"]
        assert report["checks"]["lr_monitor_dropped_without_logger"]


class TestLR1:
    """LR1 — lion/HybridMuonAdamW explicit routing (plan 12 sub-wave E; NON-gating).

    Pins the default==v1-regex behaviour, the explicit exclude (forces AdamW) /
    include (overrides the regex) lists, the zero-match warning, the shipped
    optimizer selection, and the corruption teeth.
    """

    def test_pass(self, tmp_path):
        code, report = run_lr1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "lr1_report.json").is_file()
        # default empty-list policy == v1 regex-only routing
        assert report["checks"]["default_partition_nonempty"]
        assert report["checks"]["default_matches_v1_regex_routing"]
        # explicit exclude/include lists (the v2 hardening)
        assert report["checks"]["exclude_moves_layers_to_adamw"]
        assert report["checks"]["include_overrides_regex_exclude"]
        assert report["checks"]["include_targets_were_regex_excluded"]
        # zero-match validator warning
        assert report["checks"]["zero_match_token_warns"]
        assert report["checks"]["matching_token_does_not_warn"]
        # shipped optimizer selection intact
        assert report["checks"]["hybrid_selectable"]
        assert report["checks"]["adamw_default"]
        assert report["config"]["non_gating"] is True
        assert "already shipped" in report["v1_reference"].lower()

    def test_corruption_fails_the_gate(self, tmp_path):
        # blank the include list -> the include-override check fails; the
        # value-independent default-routing + zero-match checks stay green
        code, report = run_lr1(tmp_path, corruption=lambda _kw: {})
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["include_overrides_regex_exclude"]
        assert report["checks"]["default_matches_v1_regex_routing"]
        assert report["checks"]["zero_match_token_warns"]


class TestS31:
    """S31 — move_files_temp / S3 staging smoke (plan 12 sub-wave E; NON-gating).

    Pins the default-off (unchanged) read path, the opt-in staging lifecycle
    (prepare_data copies, setup repoints, teardown removes) on a dummy local
    fixture, the S3 helper surface, and the corruption teeth (forcing
    move_files_temp=None breaks the active-path checks).
    """

    def test_pass(self, tmp_path):
        code, report = run_s31(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "s31_report.json").is_file()
        # default-off path unchanged
        assert report["checks"]["default_off_staging_inactive"]
        assert report["checks"]["default_off_paths_untouched"]
        # opt-in staging lifecycle
        assert report["checks"]["accepts_move_files_temp_arg"]
        assert report["checks"]["prepare_data_copies_to_temp"]
        assert report["checks"]["originals_survive_copy"]
        assert report["checks"]["setup_repoints_to_temp"]
        assert report["checks"]["teardown_removes_staged_copies"]
        assert report["checks"]["teardown_keeps_originals"]
        # S3 helper surface ported
        assert report["checks"]["s3_helpers_importable"]
        assert report["config"]["non_gating"] is True

    def test_corruption_fails_the_gate(self, tmp_path):
        # force move_files_temp=None -> staging inactive on the "set" path, so the
        # active-path checks (accept/copy/repoint/teardown) fail; the default-off
        # checks stay green
        code, report = run_s31(tmp_path, corruption=lambda _mft: None)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["prepare_data_copies_to_temp"]
        assert report["checks"]["default_off_staging_inactive"]
        assert report["checks"]["s3_helpers_importable"]


class TestCli:
    def test_main_runs_each_gate(self, tmp_path):
        gates = (
            "lb1", "vs1", "mu1", "mu2", "ed1", "ed2", "cm1", "cm2", "lr1", "s31", "conv",
        )
        for gate in gates:
            code = main([gate, "--outdir", str(tmp_path / gate)])
            assert code == 0
            assert (tmp_path / gate / f"{gate}_report.json").is_file()


def test_no_machine_paths_in_module():
    """The gate file ships no absolute machine path (design §5 placeholder policy)."""
    src = Path(gm6.__file__).read_text()
    for forbidden in ("/home/", "/data/", "/eos/"):
        assert forbidden not in src, f"machine path {forbidden!r} leaked into gates_m6.py"
