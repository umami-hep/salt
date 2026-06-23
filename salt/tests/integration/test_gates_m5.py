"""Tests for the M5 regression-family gates harness (`salt.tests.integration.gates_m5`, plan 10 sub-wave A).

Each gate runs against its own in-repo dummy fixtures (no machine paths) and is
paired with a python-only ``corruption`` hook proving the parity comparison has
teeth: a perturbed v2 value (or loss) must FAIL the gate while the surface
checks stay green (the gates_m2/m3/m4 pattern, never exposed on the CLI).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

import salt.tests.integration.gates_m5 as gm5
from salt.tests.integration.gates_m5 import (
    main,
    run_conv,
    run_d1,
    run_d2cfg,
    run_d2ckpt,
    run_d2expose,
    run_l1,
    run_l2,
    run_l3,
    run_mf1a,
    run_mf1c,
    run_mf2,
    run_r1,
    run_r2,
    run_r3,
    run_r4,
)
from salt.core.onnx.reduces import registered_reduces


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
        # _max_abs is a non-negative float, so <= 0.0 is exactly the "== 0.0" claim
        assert report["max_abs_diffs"]["vconcat_vs_v1_cat"] <= 0.0
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


class TestMF1a:
    def test_pass(self, tmp_path):
        code, report = run_mf1a(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "mf1a_report.json").is_file()
        # every measured diff is BITWISE (0.0) — a pure forward comparison; _max_abs
        # returns a non-negative float, so <= 0.0 is exactly the "== 0.0" claim
        assert all(v <= 0.0 for v in report["max_abs_diffs"].values())
        # drop_registers BITWISE parity vs an INDEPENDENT v1 Transformer
        assert report["checks"]["drop_registers_encoded_bitwise_vs_independent_v1"]
        assert report["checks"]["encoded_seq_is_register_free"]
        assert report["checks"]["test_plan_has_no_registers"]
        assert report["checks"]["onnx_plan_has_no_registers"]
        assert report["checks"]["v1_pad_dropped_registers"]
        # registers were genuinely VISIBLE to attention (not a no-op slice)
        assert report["checks"]["registers_visible_to_encoder"]
        # MaskDecoder forward BITWISE parity vs an INDEPENDENT v1 MaskDecoder
        assert report["checks"]["decoder_embed_bitwise_vs_independent_v1"]
        assert report["checks"]["decoder_class_logits_bitwise_vs_independent_v1"]
        assert report["checks"]["decoder_class_probs_bitwise_vs_independent_v1"]
        assert report["checks"]["decoder_masks_bitwise_vs_independent_v1"]
        # the dummy token is stripped (masks span T)
        assert report["checks"]["decoder_masks_unpadded_to_T"]
        assert report["checks"]["class_probs_is_distribution"]
        # the references are independently constructed, not the v2 composed objects
        assert "build_independent_v1_transformer_drop" in report["v1_reference"]
        assert "build_independent_v1_mask_decoder" in report["v1_reference"]
        # the ONNX claim is backed by a real Mode.ONNX run (incl. zero-constituent edge)
        assert "Mode.ONNX" in report["onnx_claim"]
        assert report["checks"]["onnx_encoded_is_register_free"]
        assert report["checks"]["onnx_objects_finite"]
        assert report["checks"]["zero_constituent_jet_is_finite"]
        # loud surfaces
        assert report["checks"]["missing_n_heads_raises"]
        assert report["checks"]["missing_class_output_size_raises"]
        assert report["checks"]["class_net_width_key_raises"]
        assert report["checks"]["embed_dim_mismatch_at_bind_raises"]
        # the gate claims ONLY the decoder slice (matched loss is a separate module)
        assert "decoder slice" in report["scope_note"].lower()
        assert "MaskFormerMatchedLoss" in report["scope_note"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 encoded.seq + objects.embed -> their parity checks vs the
        # independent v1 references must FAIL, while the un-corrupted decoder masks
        # parity, the registers-visible control, the loud-surface checks and the
        # zero-constituent ONNX control (all independent of the perturbed values)
        # stay green
        code, report = run_mf1a(tmp_path, corruption=lambda t: t + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["drop_registers_encoded_bitwise_vs_independent_v1"]
        assert not report["checks"]["decoder_embed_bitwise_vs_independent_v1"]
        # the masks/class outputs were NOT corrupted -> still bitwise-equal
        assert report["checks"]["decoder_masks_bitwise_vs_independent_v1"]
        assert report["checks"]["decoder_class_probs_bitwise_vs_independent_v1"]
        # static / loud controls stay green
        assert report["checks"]["registers_visible_to_encoder"]
        assert report["checks"]["missing_n_heads_raises"]
        assert report["checks"]["embed_dim_mismatch_at_bind_raises"]
        assert report["checks"]["zero_constituent_jet_is_finite"]


class TestMF1c:
    def test_pass(self, tmp_path):
        code, report = run_mf1c(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "mf1c_report.json").is_file()
        # every measured diff is BITWISE (0.0) — pure forward comparisons
        assert all(v <= 0.0 for v in report["max_abs_diffs"].values())
        # the three v1-decidable loss components are BITWISE vs the INDEPENDENT v1 loss
        assert report["checks"]["matched_object_class_ce_bitwise_vs_independent_v1"]
        assert report["checks"]["matched_mask_dice_bitwise_vs_independent_v1"]
        assert report["checks"]["matched_mask_focal_bitwise_vs_independent_v1"]
        # the matcher assignment matches the independent v1 matcher BITWISE
        assert report["checks"]["matcher_assignment_bitwise_vs_independent_v1"]
        # the regression is the FD matched alignment (design-conformance) and genuinely
        # DIFFERS from v1's query-order loss
        assert report["checks"]["matched_regression_design_conformance"]
        assert report["checks"]["matched_regression_differs_from_query_order"]
        # NO in-place permute: the decoder's objects.* are byte-unchanged; matched.* are new
        assert report["checks"]["objects_class_logits_not_mutated"]
        assert report["checks"]["objects_masks_not_mutated"]
        assert report["checks"]["objects_regression_not_mutated"]
        assert report["checks"]["matched_objects_are_new_keys"]
        # MaskFormerTargets parity vs v1 (no in-place id mutation)
        assert report["checks"]["targets_object_class_bitwise_vs_v1_classmap"]
        assert report["checks"]["targets_masks_bitwise_vs_v1_build_target_masks"]
        assert report["checks"]["targets_regression_labels_bitwise_vs_raw"]
        assert report["checks"]["targets_no_inplace_id_mutation"]
        assert report["checks"]["targets_null_class_is_last"]
        # loud surfaces
        assert report["checks"]["null_not_last_raises"]
        assert report["checks"]["missing_null_class_raises"]
        assert report["checks"]["unknown_loss_weight_key_raises"]
        assert report["checks"]["zero_matcher_weights_raises"]
        assert report["checks"]["regression_width_mismatch_at_bind_raises"]
        # the reference is independently constructed, not the v2 composed v1_loss
        assert "build_independent_v1_matched_loss" in report["v1_reference"]
        assert "NOT the v2 module's composed v1_loss" in report["v1_reference"]
        # the alignment change (MF1b) is documented honestly in the report
        assert "QUERY-ORDER" in report["alignment_note"]
        assert "MATCHED" in report["alignment_note"]
        # the gate claims ONLY the matched-loss/matcher/targets slice
        assert "matched-loss" in report["scope_note"].lower()
        assert "MF2" in report["scope_note"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 object_class_ce + regression losses -> their parity / conformance
        # checks must FAIL, while the un-corrupted mask losses, the matcher assignment, the
        # no-in-place-permute controls, the targets parity and the loud-surface checks (all
        # independent of the perturbed values) stay green
        code, report = run_mf1c(tmp_path, corruption=lambda t: t + 1.0)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["matched_object_class_ce_bitwise_vs_independent_v1"]
        assert not report["checks"]["matched_regression_design_conformance"]
        # the un-corrupted components stay bitwise-equal
        assert report["checks"]["matched_mask_dice_bitwise_vs_independent_v1"]
        assert report["checks"]["matched_mask_focal_bitwise_vs_independent_v1"]
        assert report["checks"]["matcher_assignment_bitwise_vs_independent_v1"]
        # static / loud controls + targets parity stay green
        assert report["checks"]["objects_class_logits_not_mutated"]
        assert report["checks"]["targets_masks_bitwise_vs_v1_build_target_masks"]
        assert report["checks"]["null_not_last_raises"]
        assert report["checks"]["regression_width_mismatch_at_bind_raises"]


class TestMF2:
    def test_pass(self, tmp_path):
        code, report = run_mf2(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "mf2_report.json").is_file()
        # TEST byte-parity vs the v1 op chain (predictionwriter.py:276-308)
        assert report["checks"]["test_class_probs_bitwise_vs_v1"]
        assert report["checks"]["test_class_target_bitwise_vs_v1"]
        assert report["checks"]["test_mask_index_bitwise_vs_v1"]
        assert report["checks"]["test_truth_mask_bitwise_vs_v1"]
        assert report["checks"]["test_mask_logits_bitwise_vs_v1"]
        # both MaskIndex sentinels are exercised (the byte-parity has teeth)
        assert report["checks"]["test_mask_index_has_padded_sentinel"]
        assert report["checks"]["test_mask_index_has_no_object_sentinel"]
        # OBJECT_INDEX imported, never re-declared (merge condition 4)
        assert report["checks"]["object_index_imported_not_redeclared"]
        # the ONNX reduces produce the correct dtypes in a COMPILED + RUN Mode.ONNX plan
        assert report["checks"]["onnx_leading_object_is_float32"]
        assert report["checks"]["onnx_object_index_is_int8"]
        assert report["checks"]["onnx_object_index_has_dynamic_token_axis"]
        assert report["checks"]["onnx_index_suffix_is_pinned_HadronIndex"]
        assert report["checks"]["onnx_sink_names_object_leaves"]  # W4: folded sink leaves
        # torch and onnxruntime agree (int8 exact, leading floats where finite)
        assert report["checks"]["onnx_object_index_int8_exact"]
        assert report["checks"]["onnx_leading_object_floats_agree_where_finite"]
        assert report["max_abs_diffs"]["onnx_leading_object_worst_abs_diff"] <= 1e-4
        # loud surfaces
        assert report["checks"]["empty_object_classes_raises"]
        assert report["checks"]["missing_regression_task_raises"]
        assert report["checks"]["non_sequence_constituent_raises"]
        # the report records the pinned OBJECT_INDEX pair + the documented value divergence
        assert report["config"]["onnx_index_test_suffix"] == "MaskIndex"
        assert report["config"]["onnx_index_onnx_suffix"] == "HadronIndex"
        assert "remapped" in report["divergence_note"].lower()
        # the gate claims ONLY the object-writer slice (decoder is MF1a, loss/targets MF1c)
        assert "MaskFormerObjectWriter" in report["scope_note"]
        assert "run_mf1a" in report["scope_note"]
        assert "run_mf1c" in report["scope_note"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # perturb the v2 writer's MFrun_pb TEST column -> the class-probs byte-parity
        # check must FAIL, while the un-corrupted columns, the ONNX reduce dtypes /
        # torch-onnx agreement, the OBJECT_INDEX import and the loud surfaces (all
        # independent of the perturbed column) stay green
        code, report = run_mf2(tmp_path, corruption=lambda col: col + np.float32(1.0))
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["test_class_probs_bitwise_vs_v1"]
        # the un-corrupted TEST columns stay byte-equal
        assert report["checks"]["test_class_target_bitwise_vs_v1"]
        assert report["checks"]["test_mask_index_bitwise_vs_v1"]
        assert report["checks"]["test_truth_mask_bitwise_vs_v1"]
        assert report["checks"]["test_mask_logits_bitwise_vs_v1"]
        # the ONNX export + loud controls (independent of the perturbed column) stay green
        assert report["checks"]["onnx_object_index_is_int8"]
        assert report["checks"]["onnx_object_index_int8_exact"]
        assert report["checks"]["object_index_imported_not_redeclared"]
        assert report["checks"]["empty_object_classes_raises"]
        assert report["checks"]["non_sequence_constituent_raises"]


class TestD1:
    def test_pass(self, tmp_path):
        code, report = run_d1(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "d1_report.json").is_file()
        # FIT/VAL callback-sink assembly: baseline (no callback) is loss.total only
        assert report["checks"]["no_callback_fit_sinks_are_loss_only"]
        assert report["checks"]["no_callback_val_sinks_are_loss_only"]
        # the negative control WITH teeth: the aux producer is pruned without a sink
        assert report["checks"]["aux_producer_pruned_without_callback"]
        # a callback-declared require becomes a FIT/VAL sink (the loss anchor stays first)
        assert report["checks"]["loss_anchor_stays_first"]
        assert report["checks"]["callback_require_becomes_fit_sink"]
        assert report["checks"]["callback_require_becomes_val_sink"]
        # the once-pruned producer is kept alive in BOTH FIT and VAL through compile_plan
        assert report["checks"]["aux_producer_kept_alive_in_fit"]
        assert report["checks"]["aux_producer_kept_alive_in_val"]
        # shipped callback classes participate in the same surface
        assert report["checks"]["shipped_confusion_matrix_demand_is_assembled"]
        assert report["checks"]["maskformer_metrics_declares_matched_sinks"]
        # callback demand is TRAINING-only (empty in TEST/ONNX)
        assert report["checks"]["callback_demand_empty_in_test"]
        assert report["checks"]["callback_demand_empty_in_onnx"]
        # register_reduce live-registry validation + reduce-declared dtypes
        assert report["checks"]["registered_reduce_in_live_registry"]
        assert report["checks"]["reduce_declared_dtype_is_queryable"]
        assert report["checks"]["registered_reduce_validates_in_manifest"]
        assert report["checks"]["unregistered_reduce_rejected_loudly"]
        assert report["checks"]["reduce_dtype_mismatch_rejected_loudly"]
        assert report["checks"]["unregistered_reduce_dtype_query_raises"]
        # the public register_reduce loud surfaces (duplicate / bad dtype / non-string)
        assert report["checks"]["duplicate_register_reduce_raises"]
        assert report["checks"]["bad_dtype_register_reduce_raises"]
        assert report["checks"]["non_string_name_register_reduce_raises"]
        # the gate restores the process-global registry — no probe-reduce residue leaks
        assert report["checks"]["register_reduce_residue_cleared"]
        # plan-29 W4: the five SHIPPED reduces are RETIRED (folded into conversion
        # nodes) — the registry carries NONE of them after the gate runs (the probe
        # was cleaned up, so the in-process registry is empty of shipped reduces)
        assert {
            "split_scalars",
            "argmax",
            "vertex_union_find",
            "leading_object",
            "object_index",
        }.isdisjoint(set(registered_reduces()))
        # the gate claims ONLY the sub-wave-D C-prereqs slice (NO model parity)
        assert "C-prereqs" in report["scope_note"]
        assert "NO model-parity" in report["scope_note"]
        # it exercises the REAL surfaces, not a re-implementation
        assert "compile_plan" in report["surfaces_exercised"]
        assert "_resolve_output" in report["surfaces_exercised"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # drop the callback-declared sink from the WITH-callback sink list -> the
        # "callback require becomes a fit sink" + "kept alive" checks must FAIL, while
        # the val-side, the register_reduce / TRAINING-only / loud-surface checks (all
        # independent of the corrupted fit-sink list) stay green
        code, report = run_d1(
            tmp_path, corruption=lambda sinks: [s for s in sinks if s != "preds.jets.aux_d1"]
        )
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["callback_require_becomes_fit_sink"]
        assert not report["checks"]["aux_producer_kept_alive_in_fit"]
        # the loss anchor + the baseline pruning control stay green
        assert report["checks"]["loss_anchor_stays_first"]
        assert report["checks"]["aux_producer_pruned_without_callback"]
        # the register_reduce slice + TRAINING-only controls are independent and stay green
        assert report["checks"]["registered_reduce_validates_in_manifest"]
        assert report["checks"]["unregistered_reduce_rejected_loudly"]
        assert report["checks"]["reduce_dtype_mismatch_rejected_loudly"]
        assert report["checks"]["callback_demand_empty_in_test"]
        assert report["checks"]["duplicate_register_reduce_raises"]


class TestD2ckpt:
    def test_pass(self, tmp_path):
        code, report = run_d2ckpt(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "d2ckpt_report.json").is_file()
        # the v1 filename/monitor contract (stem + save_top_k=-1 + monitor + no auto-insert)
        assert report["checks"]["filename_stem_is_v1_contract"]
        assert report["checks"]["save_top_k_is_minus_one"]
        assert report["checks"]["monitor_is_monitor_loss"]
        assert report["checks"]["auto_insert_metric_name_off"]
        assert report["checks"]["formatted_name_has_loss_tag"]
        # the run-dir path inference: setup forces ckpts/ AND _best_checkpoint resolves it
        assert report["checks"]["setup_forces_ckpts_dir"]
        assert report["checks"]["inference_resolves_lowest_loss"]
        assert report["checks"]["non_fit_setup_does_not_force_ckpts"]
        assert report["checks"]["fast_dev_run_does_not_force_ckpts"]
        assert report["checks"]["s3_log_dir_rejected_loudly"]
        # the ProgressBar smoke (stock TQDMProgressBar under a salt-owned name)
        assert report["checks"]["progressbar_attaches_without_error"]
        assert report["checks"]["progressbar_is_stock_tqdm"]
        assert report["checks"]["progressbar_refresh_rate_passthrough"]
        # the gate claims ONLY the Checkpoint/ProgressBar slice + the real surfaces
        assert "Checkpoint + ProgressBar slice" in report["scope_note"]
        assert "NON-GATING" in report["scope_note"]
        assert "_best_checkpoint" in report["surfaces_exercised"]
        assert "ModelCheckpoint.setup" in report["surfaces_exercised"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # drop the 'loss=' tag from the written checkpoint stem -> _best_checkpoint
        # can no longer resolve it (raises), so the run-dir inference check FAILS,
        # while the contract / setup-dir / progress checks (independent of the
        # written filename) stay green
        code, report = run_d2ckpt(tmp_path, corruption=lambda name: name.replace("loss=", "val="))
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["inference_resolves_lowest_loss"]
        # the contract, setup-dir, s3 and progress checks are independent and stay green
        assert report["checks"]["filename_stem_is_v1_contract"]
        assert report["checks"]["setup_forces_ckpts_dir"]
        assert report["checks"]["s3_log_dir_rejected_loudly"]
        assert report["checks"]["progressbar_is_stock_tqdm"]


class TestD2expose:
    def test_pass(self, tmp_path):
        code, report = run_d2expose(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "d2expose_report.json").is_file()
        # name-based origin_weighting: names resolve to v1's hardcoded ids + parity
        assert report["checks"]["names_pending_before_resolution"]
        assert report["checks"]["names_resolve_to_v1_heavy_ids"]
        assert report["checks"]["names_resolve_to_v1_fake_ids"]
        assert report["checks"]["origin_names_match_independent_v1"]
        # the loud surfaces (unknown name, no-schema bind, integer no-op)
        assert report["checks"]["unknown_name_rejected"]
        assert report["checks"]["name_based_no_schema_bind_fails"]
        assert report["checks"]["integer_ids_resolution_is_noop"]
        # the expose opt-out: pred-port gating + plan pruning + real dead-preds path
        assert report["checks"]["exposed_pred_active_in_fit"]
        assert report["checks"]["exposed_pred_inactive_in_test"]
        assert report["checks"]["exposed_task_kept_in_fit_plan"]
        assert report["checks"]["exposed_task_pruned_from_test_plan"]
        assert report["checks"]["default_task_triggers_dead_preds_error"]
        assert report["checks"]["dead_preds_error_advertises_expose"]
        assert report["checks"]["expose_silences_dead_preds_error"]
        # the gate claims ONLY the model-feature slice + names the real surfaces
        assert "expose opt-out + name-based" in report["scope_note"]
        assert "INDEPENDENT v1 VertexingTask" in report["surfaces_exercised"]
        assert "_model_sinks(Mode.TEST)" in report["surfaces_exercised"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # shift every origin id by +1 so the name-resolved heavy/fake ids no
        # longer match v1's hardcoded (3,4,5)/1 -> the INDEPENDENT-v1 parity
        # check FAILS, while the expose / resolution-count / loud-surface checks
        # (independent of the corrupted ids) stay green
        code, report = run_d2expose(
            tmp_path, corruption=lambda index: {name: i + 1 for name, i in index.items()}
        )
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["origin_names_match_independent_v1"]
        # the resolution itself still produced v1 ids (corruption happens AFTER) and
        # the expose checks are wholly independent of the origin ids — all green
        assert report["checks"]["names_resolve_to_v1_heavy_ids"]
        assert report["checks"]["exposed_task_pruned_from_test_plan"]
        assert report["checks"]["default_task_triggers_dead_preds_error"]
        assert report["checks"]["expose_silences_dead_preds_error"]


class TestD2cfg:
    def test_pass(self, tmp_path):
        code, report = run_d2cfg(tmp_path)
        assert code == 0, report["checks"]
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "d2cfg_report.json").is_file()
        # the producing leaf the validator unifies against is data/float32
        assert report["checks"]["producer_leaf_is_data_float32"]
        # writer TensorSpec kind/dtype validation: matching + unconstrained pass
        assert report["checks"]["matching_require_passes"]
        assert report["checks"]["unconstrained_dtype_require_passes"]
        # the loud surfaces: wrong dtype/kind and missing non-optional producer
        assert report["checks"]["wrong_dtype_raises_shape_error"]
        assert report["checks"]["wrong_kind_raises_kind_error"]
        assert report["checks"]["missing_nonoptional_require_raises_config_error"]
        assert report["checks"]["optional_missing_require_is_silent"]
        # the END-TO-END half: the SAME kind/dtype unification now runs inside
        # `salt2 graph validate` (cli._cmd_validate, TEST) — a wrong writer
        # kind/dtype FAILS the canonical static-validator command data-free, a
        # matching writer PASSES it (the medium critic finding's fix)
        assert report["checks"]["wrong_kind_writer_fails_graph_validate"]
        assert report["checks"]["wrong_dtype_writer_fails_graph_validate"]
        assert report["checks"]["matching_writer_passes_graph_validate"]
        assert report["config"]["wrong_kind_rc"] != 0
        assert report["config"]["wrong_dtype_rc"] != 0
        assert report["config"]["matching_rc"] == 0
        # lrs_config -> lrs migration: v2 config loads, no stale key, old key rejected
        assert report["checks"]["migrated_config_graph_validate_ok"]
        assert report["checks"]["no_v2_config_retains_lrs_config_key"]
        assert report["checks"]["stale_lrs_config_key_rejected"]
        # the migration check actually scanned the shipped configs (empty = clean)
        assert report["config"]["stale_configs_found"] == []
        # the gate claims ONLY the cfg slice + names the real surfaces
        assert "writer TensorSpec kind/dtype validation + lrs_config->lrs" in report["scope_note"]
        assert "WriterCallback.validate_specs" in report["surfaces_exercised"]
        assert "salt2 graph validate" in report["surfaces_exercised"]

    def test_corruption_fails_the_gate(self, tmp_path):
        # flip the jets_classification producer dtype to int64 so the MATCHING
        # data/float32 probe is no longer dtype-compatible -> matching_require_passes
        # FAILS, while the wrong-kind/dtype, existence and lrs-migration checks
        # (independent of that producer dtype) stay green
        def corrupt(specs):
            key = "preds.jets.jets_classification"
            specs[key] = dataclasses.replace(specs[key], dtype="int64")
            return specs

        code, report = run_d2cfg(tmp_path, corruption=corrupt)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["matching_require_passes"]
        # the independent checks stay green
        assert report["checks"]["wrong_dtype_raises_shape_error"]
        assert report["checks"]["wrong_kind_raises_kind_error"]
        assert report["checks"]["missing_nonoptional_require_raises_config_error"]
        assert report["checks"]["migrated_config_graph_validate_ok"]
        assert report["checks"]["no_v2_config_retains_lrs_config_key"]
        assert report["checks"]["stale_lrs_config_key_rejected"]


class TestConv:
    """M5-CONV — the consolidated convert+validate+plan-compile acceptance gate.

    The gate drives the REAL ``salt2 graph validate`` (fit/test/onnx) on EVERY
    needs-M5 config and embeds the authoritative list in its report. These tests
    pin: the completeness check (every matrix-🔶 config present + the adjudicated
    regression_multi_target), the per-config mode bookkeeping, the
    event_classifier ONNX = ``na`` carve-out, and the corruption teeth.

    MaskFormer parity (M5-CONV fix): MaskFormer.yaml now validates end-to-end in
    all 3 modes. The fix honoured the v1 objects-stream pad-mask exemption (v1
    ``RegressionTask.forward`` / ``ClassificationTask.forward``:
    ``salt/models/task.py:547`` ``if pad_masks is not None and self.input_name
    != "objects"``) — a sequence-mode task on the fixed-count ``objects`` query
    bank no longer requires/consumes ``masks.objects`` (``_NO_PAD_MASK_STREAMS``
    in ``tasks.py``). It also wired the matched-loss contract end-to-end: the
    object-regression head is named ``regression`` (so it produces
    ``preds.objects.regression``), publishes its scaled targets as
    ``targets.objects.regression`` (``publish_targets: true``, the v1
    maskformer_loss.py:329-332 contract), discards its standalone loss (the
    matched loss owns ``losses.regression``), is ``expose: [fit, val, onnx]``
    (v1 never writes object-regression preds in eval), and the encoder gains a
    ``Split`` (track_origin reads ``encoded.tracks``). The MF1a/MF1c/MF2 gates'
    ``_ObjectRegressionStub`` covers forward parity; M5-CONV proves the full
    config assembly. So ALL 24 needs-M5 configs validate and the gate PASSES.
    """

    def test_structure_and_completeness(self, tmp_path):
        _code, report = run_conv(tmp_path)
        assert (tmp_path / "conv_report.json").is_file()
        # the list is COMPLETE vs the matrix §2 🔶 rows + adjudicated multi_target
        assert report["checks"]["all_matrix_needs_m5_configs_present"]
        assert report["config"]["missing_matrix_configs"] == []
        assert report["config"]["total_configs"] == 24
        # every embedded config name is a matrix needs-M5 name (no stray entries)
        names = {c["name"] for c in report["configs"]}
        assert names == set(report["matrix_needs_m5"])
        # regression_multi_target is included (matrix-🔷, plan-10-adjudicated M5)
        assert "regression_multi_target" in names
        # event_classifier validates fit+test only — ONNX carve-out (no export)
        ev = next(c for c in report["configs"] if c["name"] == "event_classifier")
        assert ev["validateFit"]
        assert ev["validateTest"]
        assert ev["validateOnnx"] == "na"
        assert "onnx" not in ev["rc"]  # the gate never invoked an onnx validate
        # no --strict (data-free no-schema warning is inherent — documented)
        assert report["config"]["strict"] is False
        assert "no --strict" in report["no_strict_rationale"]
        # the gate makes NO forward-parity claim (owned by R/L/MF)
        assert "NO forward-parity claim" in report["scope_note"]

    def test_all_24_configs_validate(self, tmp_path):
        # EVERY needs-M5 config (incl. MaskFormer) convert+validates+plan-compiles
        # in all applicable modes — the M5-CONV acceptance criterion is met
        code, report = run_conv(tmp_path)
        for c in report["configs"]:
            assert c["validateFit"], c
            assert c["validateTest"], c
            assert c["validateOnnx"] in {True, "na"}, c
        assert report["config"]["validated_configs"] == 24
        assert code == 0
        assert report["passed"]

    def test_maskformer_validates_all_modes(self, tmp_path):
        # MaskFormer.yaml validates fit/test/onnx after the objects-stream pad-mask
        # exemption + the matched-loss contract wiring (see the class docstring).
        _code, report = run_conv(tmp_path)
        mf = next(c for c in report["configs"] if c["name"] == "MaskFormer")
        assert mf["rc"]["fit"] == 0
        assert mf["rc"]["test"] == 0
        assert mf["rc"]["onnx"] == 0
        assert mf["validateFit"]
        assert mf["validateTest"]
        assert mf["validateOnnx"]

    def test_corruption_missing_config_fails_completeness(self, tmp_path):
        # drop a config from the embedded list -> the completeness check fails
        # (the gate exits non-zero if any matrix 🔶 config is missing)
        def drop_regression(configs):
            return [c for c in configs if c["name"] != "regression"]

        code, report = run_conv(tmp_path, corruption=drop_regression)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["all_matrix_needs_m5_configs_present"]
        assert "regression" in report["config"]["missing_matrix_configs"]

    def test_corruption_bogus_config_fails(self, tmp_path):
        # inject a config whose stack does not exist -> salt2 graph validate
        # errors (rc != 0), so the gate fails on that config's modes
        def add_bogus(configs):
            return [
                *configs,
                {
                    "name": "regression",  # reuse a matrix name so completeness still passes
                    "cfg": ("does_not_exist_conv_probe.yaml",),
                    "norm_global": False,
                    "onnx": "validate",
                    "family": "regression",
                    "note": "bogus corruption probe",
                },
            ]

        code, report = run_conv(tmp_path, corruption=add_bogus)
        assert code == 1
        assert not report["passed"]


class TestCli:
    def test_main_runs_each_gate(self, tmp_path):
        gates = (
            "r1",
            "r2",
            "r3",
            "r4",
            "l1",
            "l2",
            "l3",
            "mf1a",
            "mf1c",
            "mf2",
            "d1",
            "d2ckpt",
            "d2expose",
            "d2cfg",
        )
        for gate in gates:
            code = main([gate, "--outdir", str(tmp_path / gate)])
            assert code == 0
            assert (tmp_path / gate / f"{gate}_report.json").is_file()

    def test_conv_runs_and_writes_report(self, tmp_path):
        # conv passes (rc=0) — all 24 needs-M5 configs validate; the per-config
        # pass/fail semantics are covered by TestConv above.
        code = main(["conv", "--outdir", str(tmp_path / "conv")])
        assert code == 0
        assert (tmp_path / "conv" / "conv_report.json").is_file()


def test_no_machine_paths_in_module():
    """The gate file ships no absolute machine path (design §5 placeholder policy)."""
    src = Path(gm5.__file__).read_text()
    for forbidden in ("/home/", "/data/", "/eos/"):
        assert forbidden not in src, f"machine path {forbidden!r} leaked into gates_m5.py"
