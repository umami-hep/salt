"""Tests for the M7 W1 gates (CV1 converter acceptance, CV2 hard-error path).

Each gate is driven green AND its negative-control (corruption) must-fail path is
exercised — the gates_m2..m6 paired-test convention: a python-only ``corruption``
hook (never on the CLI) proves the gate's assertions are not vacuous.
"""

from __future__ import annotations

import salt.core.gates_m7 as gm7
from salt.core.gates_m7 import run_cv1, run_cv2

# the authoritative embedded lists live as module privates on gm7 (the gate is
# the single source of truth); reference them through the module so the test does
# not import private names directly.
_CV1_CONFIGS = gm7._CV1_CONFIGS  # noqa: SLF001
_CV1_FIXTURE_SUBSET = gm7._CV1_FIXTURE_SUBSET  # noqa: SLF001
_CV2_CONFIGS = gm7._CV2_CONFIGS  # noqa: SLF001
_CV2_HEALTHY_PROBE = gm7._CV2_HEALTHY_PROBE  # noqa: SLF001


class TestCV1:
    """CV1 — convert-config acceptance over the FULL 39-config matrix §4 denominator.

    Pins: the embedded authoritative config list (29 fixture-bearing needs-M5/M6
    + 10 reproducible-today ✅), every config converts + validates + plan-compiles,
    the fixture-plan semantic equivalence (exact for non-subset fixture configs;
    flagged for the pinned fixture-subset configs; N/A for the ✅ configs), the
    task-count faithfulness fact, and the corruption teeth (a bit-rotted v1 source
    breaks conversion -> gate red).
    """

    def test_pass(self, tmp_path):
        code, report = run_cv1(tmp_path)
        assert code == 0, {k: v for k, v in report["checks"].items() if not v}
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "cv1_report.json").is_file()

    def test_authoritative_list_embedded_verbatim(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        # the report embeds the authoritative _CV1_CONFIGS names verbatim
        assert report["cv1_config_names"] == [e["name"] for e in _CV1_CONFIGS]
        # 24 needs-M5 + 5 needs-M6 + 10 reproducible-today (✅) = 39 (the FULL
        # matrix §4 / plan 13 non-dropped denominator; incl. conditional-drop Dipz)
        assert report["config"]["total_configs"] == 39
        assert report["config"]["denominator"] == 39
        tiers = [e["tier"] for e in _CV1_CONFIGS]
        assert tiers.count("m5") == 24
        assert tiers.count("m6") == 5
        assert tiers.count("today") == 10
        assert report["config"]["fixture_bearing_configs"] == 29
        assert report["config"]["reproducible_today_configs"] == 10
        # Dipz is INCLUDED as the conditional drop
        dipz = next(c for c in report["configs"] if c["name"] == "Dipz")
        assert dipz["conditional_drop"]
        assert dipz["converted"]

    def test_reproducible_today_configs_present_and_gated(self, tmp_path):
        # the high-severity 29-vs-39 fix: the 10 ✅ configs (no fixture) are in
        # CV1, gated on convert+validate+plan-compile (cond. 1-3), fixture_match N/A
        _code, report = run_cv1(tmp_path)
        today = [c for c in report["configs"] if c["today"]]
        assert len(today) == 10
        expected = {
            "GN2",
            "gn2v1-opendata",
            "GN2_extended",
            "GN2_charded_neutral_loose_aux",
            "GN2X",
            "GN2XTau",
            "flow",
            "tutorial",
            "TPLTmu",
            "PLITel",
        }
        assert {c["name"] for c in today} == expected
        for c in today:
            # ✅ configs have NO fixture -> fixture_match is not asserted
            assert c["fix_stack"] == []
            assert report["checks"][f"{c['name']}:accept"]
            if c.get("expect_convert_error"):
                # tutorial: non-standard vertex label -> converter correctly
                # hard-errors per FD §10 (gated on the EXPECTED hard-error)
                assert c["hard_errored_as_expected"], c["name"]
                assert not c["converted"]
            else:
                assert c["converted"], f"{c['name']}: {c.get('convert_error')}"
                assert c["validate"].get("fit") == 0, c["name"]
                assert c["validate"].get("test") == 0, c["name"]

    def test_tutorial_is_expected_hard_error(self, tmp_path):
        # the broader 39-denominator surfaced a real v1<->v2 incompatibility:
        # tutorial.yaml's const_vertexing uses the non-standard `truth_vertex_idx`
        # label the v2 VertexingTaskModule cannot map. The converter HARD-ERRORS
        # (FD §10) instead of emitting output that crashes opaquely downstream.
        _code, report = run_cv1(tmp_path)
        tut = next(c for c in report["configs"] if c["name"] == "tutorial")
        assert tut["expect_convert_error"]
        assert tut["hard_errored_as_expected"]
        assert not tut["converted"]
        assert "VertexIndex" in tut.get("convert_error", "")
        assert report["config"]["expected_hard_error_names"] == ["tutorial"]
        assert report["config"]["accepted"] == report["config"]["total_configs"]

    def test_every_convertible_config_converts(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        for c in report["configs"]:
            if c.get("expect_convert_error"):
                continue  # tutorial: expected hard-error, not convert-success
            assert c["converted"], f"{c['name']} failed to convert: {c.get('convert_error')}"
        # every config is ACCEPTED (converted+validated OR expected hard-error)
        assert report["config"]["accepted"] == report["config"]["total_configs"]
        assert (
            report["config"]["converted"] + report["config"]["expected_hard_error_configs"]
            == report["config"]["total_configs"]
        )

    def test_every_convertible_config_validates_plan_compiles(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        for c in report["configs"]:
            if c.get("expect_convert_error"):
                continue
            assert c["validate"].get("fit") == 0, c["name"]
            assert c["validate"].get("test") == 0, c["name"]
            # onnx only where the config declares an export representation
            if "onnx" in c["validate"]:
                assert c["validate"]["onnx"] == 0, c["name"]
        # data-free, NOT --strict (the M5/M6-CONV precedent)
        assert report["config"]["strict"] is False
        assert "no --strict" in report["no_strict_rationale"]

    def test_non_subset_fixture_configs_plan_match_the_fixture(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        for c in report["configs"]:
            # only the FIXTURE-bearing, non-subset configs plan-match; the ✅
            # today configs have no fixture (skipped)
            if c["today"] or c["name"] in _CV1_FIXTURE_SUBSET:
                continue
            assert c["fixture_match"], (
                f"{c['name']} converted plan != hand-written fixture plan: "
                f"modes={c['mode_match']} eval={c['eval_manifest_match']} "
                f"onnx={c['onnx_manifest_match']}"
            )
            assert c["eval_manifest_match"], c["name"]
            assert c["onnx_manifest_match"], c["name"]
            assert c["converter_faithful_to_v1"], c["name"]

    def test_fixture_subset_reproduces_full_v1_task_count(self, tmp_path):
        # the low-severity finding: converter_faithful_to_v1 for the subset
        # configs is now a CHECKED fact (conv_task_count == v1 >= fixture), not a
        # human note
        _code, report = run_cv1(tmp_path)
        for c in report["configs"]:
            if c["name"] not in _CV1_FIXTURE_SUBSET:
                continue
            assert c["task_counts_faithful"], (
                f"{c['name']}: converter does NOT reproduce the full v1 task count "
                f"(conv={c['conv_task_count']} v1={c['v1_task_count']} "
                f"fixture={c['fixture_task_count']})"
            )
            assert c["conv_task_count"] == c["v1_task_count"]
            assert c["conv_task_count"] >= c["fixture_task_count"]
            assert report["checks"][f"{c['name']}:reproduces_full_v1_tasks"]

    def test_fixture_subset_configs_flagged_and_diverge(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        # the fixture-subset set is pinned (exactly these diverge)
        assert report["config"]["fixture_subset_configs"] == sorted(_CV1_FIXTURE_SUBSET)
        assert report["checks"]["fixture_subset_set_pinned"]
        for c in report["configs"]:
            if c["name"] not in _CV1_FIXTURE_SUBSET:
                continue
            # subset configs: converter is MORE faithful than the fixture -> the
            # exact plan match does NOT hold, but they still convert + validate
            assert c["fixture_subset"]
            assert not c["fixture_match"], (
                f"{c['name']} is flagged fixture-subset but now MATCHES the fixture — the "
                "accounting changed; re-audit whether the fixture was made faithful"
            )
            assert c["converter_faithful_to_v1"]
            assert c["converted"]

    def test_forward_parity_deferral_is_explicit(self, tmp_path):
        # the medium-severity finding: matrix §4 cond. 4's numerical forward-parity
        # + v1==v2 weight-parity are NOT in CV1 — the report states the deferral
        # explicitly so it does not over-claim full §4 coverage
        _code, report = run_cv1(tmp_path)
        assert "deferred_forward_parity" in report
        assert "DEFERRED" in report["deferred_forward_parity"]
        assert "forward-parity" in report["deferred_forward_parity"]
        # the criterion text names the deferral too
        assert "DEFERRED" in report["criterion"]

    def test_qcdsplit_labeller_override_recorded(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        qcd = next(c for c in report["configs"] if c["name"] == "GN2X_qcdsplit")
        # the converter faithfully emits the v1 labeller classes (incl. qcdxx);
        # CV1 substitutes qcdll at plan-compile so it compiles in this ftag build
        ovr = qcd.get("labeller_override_applied")
        assert ovr is not None
        assert "qcdxx" in ovr["v1_faithful"]
        assert "qcdxx" not in ovr["compiled_with"]
        assert qcd["validate"].get("fit") == 0

    def test_corruption_fails_the_gate(self, tmp_path):
        # repoint a needs-M5 config's v1 source at a bit-rotted config -> the
        # conversion hard-errors, so its `converted` check flips False and the
        # gate goes red (proving the acceptance assertions are not vacuous).
        def corrupt(configs):
            for c in configs:
                if c["name"] == "regression":
                    c["v1"] = ("GN2/GN2_open_data.yaml",)  # bit-rotted -> ConvertError
            return configs

        code, report = run_cv1(tmp_path, corruption=corrupt)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["regression:converted"]
        reg = next(c for c in report["configs"] if c["name"] == "regression")
        assert not reg["converted"]
        assert "convert_error" in reg


class TestCV2:
    """CV2 — convert-config hard-error path on the bit-rotted/parked configs.

    Pins: the embedded authoritative list, every dropped config hard-errors with
    a ConvertError naming the config + the removed/parked v1-class marker, and the
    corruption teeth (a silently-converting config -> gate red).
    """

    def test_pass(self, tmp_path):
        code, report = run_cv2(tmp_path)
        assert code == 0, {k: v for k, v in report["checks"].items() if not v}
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "cv2_report.json").is_file()

    def test_authoritative_list_embedded_verbatim(self, tmp_path):
        _code, report = run_cv2(tmp_path)
        assert report["cv2_config_names"] == [e["name"] for e in _CV2_CONFIGS]
        # 5 bit-rotted + 1 parked
        assert report["config"]["total_configs"] == 6
        kinds = [e["kind"] for e in _CV2_CONFIGS]
        assert kinds.count("bit-rotted") == 5
        assert kinds.count("parked") == 1

    def test_every_dropped_config_hard_errors(self, tmp_path):
        _code, report = run_cv2(tmp_path)
        for c in report["configs"]:
            assert c["hard_errored"], f"{c['name']} did NOT hard-error"
            assert c["is_convert_error"], f"{c['name']} raised a non-ConvertError"
            assert c["names_config"], f"{c['name']} error does not name the config"
            assert c["has_marker"], f"{c['name']} error lacks the {c['marker']} marker"
        assert report["config"]["hard_errored"] == report["config"]["total_configs"]

    def test_markers_are_the_removed_or_parked_classes(self, tmp_path):
        _code, report = run_cv2(tmp_path)
        markers = {c["name"]: c["marker"] for c in report["configs"]}
        assert markers["GN2_tracks_neutral_CA"] == "salt.models.TransformerCrossAttentionEncoder"
        assert markers["Baseline_Xbb"] == "salt.models.R21Xbb"
        # the bit-rotted GN2 family names the removed TransformerEncoder
        assert markers["GN2_open_data"] == "salt.models.TransformerEncoder"

    def test_corruption_fails_the_gate(self, tmp_path):
        # swap a dropped config's v1 source for a HEALTHY config (GN3X) that
        # converts cleanly -> the "must hard-error" assertion flips False and the
        # gate goes red (the negative control: a dropped config that silently
        # converted FAILS the gate).
        def corrupt(configs):
            for c in configs:
                if c["name"] == "GN2_open_data":
                    c["v1"] = _CV2_HEALTHY_PROBE
            return configs

        code, report = run_cv2(tmp_path, corruption=corrupt)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["GN2_open_data:hard_errors"]
        bad = next(c for c in report["configs"] if c["name"] == "GN2_open_data")
        assert not bad["hard_errored"]


def test_cli_dispatch_cv1(tmp_path):
    assert gm7.main(["cv1", "--outdir", str(tmp_path)]) == 0


def test_cli_dispatch_cv2(tmp_path):
    assert gm7.main(["cv2", "--outdir", str(tmp_path)]) == 0
