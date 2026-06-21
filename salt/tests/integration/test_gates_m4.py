"""Tests for the M4 gates harness (`salt.tests.integration.gates_m4`, plan 07 stage C).

O1/O3 run at reduced sweep scale (the full L=0..39 x trials sweeps live in
experiment 06); O2/O4 additionally run with their python-only ``corruption``
hooks, proving the identity comparisons have teeth (a vacuous comparison
would pass the corrupted run); O5 runs as shipped — it is itself the
negative-control gate, so a green O5 here certifies the corrupted-weights
and model-name controls fire and the ``gnn_config`` envelope matches v1.
"""

from __future__ import annotations

import numpy as np
import pytest

from salt.tests.integration.gates_m4 import (
    ATHENA_SUBSET_KEYS,
    MODEL_NAME,
    V1_GNN_KEYS,
    main,
    run_o1,
    run_o2,
    run_o3,
    run_o4,
    run_o5,
)

TRIALS = 1
MAX_LENGTH = 6  # includes the L=0 and L=1 zero-edge cases


def _swap_jet_probs(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The O2 corruption hook: value-swap the v2 pb/pc scalars.

    Swapping VALUES (not names) keeps the IO contract intact, so a pass of
    the corrupted run would mean the value comparison is vacuous — the
    gates_m3 W3 pattern.

    Returns
    -------
    dict[str, np.ndarray]
        The corrupted v2 outputs.
    """
    corrupted = dict(outputs)
    corrupted[f"{MODEL_NAME}_pb"] = outputs[f"{MODEL_NAME}_pc"]
    corrupted[f"{MODEL_NAME}_pc"] = outputs[f"{MODEL_NAME}_pb"]
    return corrupted


class TestO1:
    def test_reduced_sweep_passes_at_1e6(self, tmp_path):
        code, report = run_o1(tmp_path, trials=TRIALS, max_length=MAX_LENGTH)
        assert code == 0
        assert report["passed"]
        assert report["checks"]["case_count_is_full_sweep"]
        assert report["n_cases"] == TRIALS * MAX_LENGTH
        assert all(diff <= 1e-6 for diff in report["worst_abs_diff"].values())
        assert (tmp_path / "o1_report.json").exists()


class TestO2:
    def test_identity_at_reduced_sweep(self, tmp_path):
        code, report = run_o2(tmp_path, trials=TRIALS, max_length=MAX_LENGTH)
        assert code == 0
        assert report["passed"]
        checks = report["checks"]
        assert checks["input_names_equal"]
        assert checks["output_names_equal"]
        assert checks["graph_dims_equal"]
        assert checks["doc_strings_equal"]
        assert checks["athena_metadata_subset_equal"]
        # int8 aux outputs must be BITWISE in every case (never tolerant)
        per_output = report["identity"]["per_output"]
        for name in (f"{MODEL_NAME}_TrackOrigin", f"{MODEL_NAME}_VertexIndex"):
            assert per_output[name]["n_bitwise"] == per_output[name]["n_cases"]
            assert per_output[name]["n_failed"] == 0

    def test_corruption_hook_fails_exactly_the_swapped_outputs(self, tmp_path):
        # negative control: the identity sweep must FAIL on a real value
        # difference, localised to the corrupted outputs (teeth + locality)
        code, report = run_o2(tmp_path, trials=TRIALS, max_length=4, corruption=_swap_jet_probs)
        assert code == 1
        assert not report["passed"]
        assert report["identity"]["corrupted_by_test_hook"]
        failing = {
            name for name, record in report["identity"]["per_output"].items() if record["n_failed"]
        }
        assert failing == {f"{MODEL_NAME}_pb", f"{MODEL_NAME}_pc"}
        # the contract checks themselves still pass — the failure is the values
        assert report["checks"]["output_names_equal"]


class TestO3:
    def test_two_axis_grid_with_zeros(self, tmp_path):
        code, report = run_o3(
            tmp_path, trials=TRIALS, track_lengths=(0, 2, 5), electron_lengths=(0, 3)
        )
        assert code == 0
        assert report["passed"]
        checks = report["checks"]
        assert checks["grid_includes_double_zero"]
        assert checks["v1_cross_check_passed"]
        assert checks["independent_axes_shapes"]
        # every int8 output — incl. the v2-only ElectronOrigin, which the
        # v1 cross-check cannot cover — took >= 2 distinct values over the
        # traced sweep (M4-review fix: a collapsed argmax would compare
        # int8-exactly while proving nothing)
        assert checks["int8_outputs_nondegenerate_in_sweep"]
        assert len(report["int8_distinct"][f"{MODEL_NAME}_ElectronOrigin"]) >= 2
        # the adjudicated risk-7 mechanism is visible in the compiled plan
        assert checks["concat_publishes_seq_offsets_in_onnx_mode"]
        assert checks["split_consumes_seq_offsets_in_onnx_mode"]
        assert "index_select" in report["split_decision_evidence"]
        # the v2-only electron head is the ONLY unmatched module
        assert report["config"]["v2_only_head_missing_keys"]
        assert all(
            key.startswith("electron_origin.")
            for key in report["config"]["v2_only_head_missing_keys"]
        )


class TestO4:
    def test_triple_identity(self, tmp_path):
        code, report = run_o4(tmp_path, trials=TRIALS, max_length=MAX_LENGTH)
        assert code == 0
        assert report["passed"]
        assert report["checks"]["all_triple_exact"]
        assert report["checks"]["vertex_assignments_nondegenerate"]
        assert report["n_mismatches"] == 0

    def test_corruption_hook_fails(self, tmp_path):
        # shifting every assignment by one breaks the triple identity at
        # every L >= 1 (L=0 is empty either way)
        code, report = run_o4(tmp_path, trials=TRIALS, max_length=4, corruption=lambda arr: arr + 1)
        assert code == 1
        assert not report["passed"]
        assert report["n_mismatches"] > 0


class TestO5:
    @pytest.fixture(scope="class")
    def o5(self, tmp_path_factory):
        return run_o5(tmp_path_factory.mktemp("o5"))

    def test_gate_passes(self, o5):
        code, report = o5
        assert code == 0
        assert report["passed"]

    def test_corrupted_weights_control(self, o5):
        _, report = o5
        control = report["control_corrupted_weights"]
        assert control["corrupted_check_failed"]
        assert control["corrupted_failures_reported"]
        assert control["restored_check_passed"]
        # the negative-control evidence trail must carry CONTENT — the old
        # splitlines()[0] truncation blanked assert_allclose messages
        # (M4-review fix, check_onnx)
        assert control["corrupted_failures_nonempty"]
        assert report["corrupted_check_failures"]
        assert all(f.strip() for f in report["corrupted_check_failures"])

    def test_model_name_control(self, o5):
        _, report = o5
        control = report["control_model_name"]
        assert control["config_error_raised"]
        assert control["message_names_offender"]
        assert control["message_states_rule"]
        assert control["export_graph_rejects_before_writing"]
        assert control["default_name_strips_run_name"]

    def test_metadata_envelope(self, o5):
        _, report = o5
        checks = report["metadata_checks"]
        assert checks["v1_envelope_is_the_13_key_set"]
        assert checks["v2_keys_are_v1_prefix_plus_plan_hash"]
        assert checks["athena_subset_byte_equal"]
        assert checks["all_v1_keys_equal_or_justified"]
        # the RAW stored v2 string is a literal byte prefix-extension of
        # v1's (strictly stronger than per-key parse-redump — M4-review fix)
        assert checks["raw_bytes_v1_prefix_plus_plan_hash"]
        assert report["raw_byte_comparison"]["v2_equals_v1_prefix_plus_plan_hash_entry"]
        assert checks["plan_hash_recorded"]
        # any non-equal key must be the justified salt_export_hash, with the
        # justification carried in the report (never silent)
        for record in report["metadata_keys"]:
            assert record["equal"] or (record["key"] == "salt_export_hash" and record["note"]), (
                record
            )

    def test_constants_match_the_v1_contract(self):
        assert len(V1_GNN_KEYS) == 13
        assert set(ATHENA_SUBSET_KEYS) <= set(V1_GNN_KEYS)


class TestCli:
    def test_main_runs_o1(self, tmp_path):
        code = main([
            "o1",
            "--outdir",
            str(tmp_path),
            "--trials",
            "1",
            "--max-length",
            "3",
        ])
        assert code == 0
        assert (tmp_path / "o1_report.json").exists()
