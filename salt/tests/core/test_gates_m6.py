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
from salt.core.gates_m6 import main, run_conv, run_lb1


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


class TestConv:
    """M6-CONV — the M7-slice acceptance for the M6-authored v2-native configs.

    The gate drives the REAL ``salt2 graph validate`` (fit/test/onnx) on every
    config landed so far (``_CONV_M6_CONFIGS`` — this wave: GN3X, GN2X_qcdsplit)
    and embeds the authoritative list verbatim. These tests pin: the
    file-present check, the per-config mode bookkeeping (both standard traces ->
    onnx validated), and the corruption teeth (a missing file fails the gate).
    """

    def test_structure_and_files_present(self, tmp_path):
        _code, report = run_conv(tmp_path)
        assert (tmp_path / "conv_report.json").is_file()
        # every embedded config file exists in salt/core/configs/
        assert report["checks"]["all_m6_conv_config_files_present"]
        assert report["config"]["missing_config_files"] == []
        # this wave lands exactly the 2 Labeller configs, names embedded verbatim
        names = {c["name"] for c in report["configs"]}
        assert names == {"GN3X", "GN2X_qcdsplit"}
        # the report embeds the authoritative _CONV_M6_CONFIGS list verbatim
        assert report["conv_m6_configs"] == ["GN3X", "GN2X_qcdsplit"]
        assert report["config"]["total_configs"] == 2
        # no --strict (flow/truth_hadrons preflight warnings inherent — documented)
        assert report["config"]["strict"] is False
        assert "no --strict" in report["no_strict_rationale"]
        # the gate makes NO forward-parity claim (owned by LB1)
        assert "NO forward-parity claim" in report["scope_note"]

    def test_both_configs_validate_all_modes(self, tmp_path):
        # GN3X + GN2X_qcdsplit convert+validate+plan-compile in fit/test/onnx —
        # both are standard traces, so onnx is validated for each
        code, report = run_conv(tmp_path)
        for c in report["configs"]:
            assert c["validateFit"], c
            assert c["validateTest"], c
            assert c["validateOnnx"] is True, c
            assert c["rc"]["fit"] == 0, c
            assert c["rc"]["test"] == 0, c
            assert c["rc"]["onnx"] == 0, c
        assert report["config"]["validated_configs"] == 2
        assert code == 0
        assert report["passed"]

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
        gates = ("lb1", "conv")
        for gate in gates:
            code = main([gate, "--outdir", str(tmp_path / gate)])
            assert code == 0
            assert (tmp_path / gate / f"{gate}_report.json").is_file()


def test_no_machine_paths_in_module():
    """The gate file ships no absolute machine path (design §5 placeholder policy)."""
    src = Path(gm6.__file__).read_text()
    for forbidden in ("/home/", "/data/", "/eos/"):
        assert forbidden not in src, f"machine path {forbidden!r} leaked into gates_m6.py"
