"""Tests for the M7 W1 gates (CV1 converter acceptance, CV2 hard-error path).

Each gate is driven green AND its negative-control (corruption) must-fail path is
exercised — the gates_m2..m6 paired-test convention: a python-only ``corruption``
hook (never on the CLI) proves the gate's assertions are not vacuous.
"""

from __future__ import annotations

import salt.core.gates_m7 as gm7
from salt.core.gates_m7 import run_cv1, run_cv2, run_cvf, run_ren1, run_rs1

# the authoritative embedded lists live as module privates on gm7 (the gate is
# the single source of truth); reference them through the module so the test does
# not import private names directly.
_CV1_CONFIGS = gm7._CV1_CONFIGS  # noqa: SLF001
_CV1_FIXTURE_SUBSET = gm7._CV1_FIXTURE_SUBSET  # noqa: SLF001
_CV1_RESTORED = gm7._CV1_RESTORED  # noqa: SLF001
_CV2_CONFIGS = gm7._CV2_CONFIGS  # noqa: SLF001
_CV2_HEALTHY_PROBE = gm7._CV2_HEALTHY_PROBE  # noqa: SLF001
_CVF_FIXTURE_EXCEPTIONS = gm7._CVF_FIXTURE_EXCEPTIONS  # noqa: SLF001
_REN1_SHIPPED_CONFIGS = gm7._REN1_SHIPPED_CONFIGS  # noqa: SLF001

# the 2 documented INTENTIONAL KEEPS after Wave F2 (event_classifier: export-
# omission correct by design; regression_multi_target: the fixture is the faithful
# F1a artifact). The other 6 once-subset fixtures were restored to full v1 fidelity.
# the v1 import sites the M7 W2a relocation removed (now repointed to salt.core.*)
_W2A_RELOCATED_MODULES = {
    "salt.utils.array_utils",
    "salt.utils.scalers",
    "salt.utils.union_find",
    "salt.utils.file_utils",
    "salt.data.samplers",
    "salt.optim",
    "salt.data.datasets",  # get_dtype
}
# the v1 import SITES the M7 W2b inline-math + mask_utils relocation removed
# (RS1 18 -> 11). These are pinned at (file, module) granularity, NOT module-level:
# salt.models.maskformer is BOTH removed here (maskdecoder.py:51 get_masks, reduces.py
# maskformer get_maskformer_outputs) AND retained at a DIFFERENT W2c site
# (maskdecoder.py MaskDecoderLayer), so a module-level "gone" assertion (the W2a
# style) would be wrong. The salt.utils.* relocations (mask_utils/tensor_utils/
# edge_features) ARE total — no salt.utils import survives W2b anywhere in core.
_W2B_RELOCATED_SITES = {
    ("callbacks.py", "salt.utils.mask_utils"),  # mask_from_logits + reco_metrics
    ("writers/maskformer.py", "salt.utils.mask_utils"),  # indices_from_mask
    ("nn/modules.py", "salt.utils.edge_features"),  # calculate_edge_features + check
    ("nn/modules.py", "salt.utils.tensor_utils"),  # attach_context (+ add_dims)
    ("onnx/reduces.py", "salt.models.maskformer"),  # get_maskformer_outputs inline
    ("onnx/reduces.py", "salt.models.task"),  # the task import reduces no longer needs
}
# NOT a W2b-gone SITE here: the maskdecoder.py get_masks inline removed
# (nn/maskdecoder.py, salt.models.maskformer) at the OLD line, but a DIFFERENT
# salt.models.maskformer import (MaskDecoderLayer) is RETAINED at that same file
# as a W2c item — so (file, module) cannot distinguish removed from retained
# there. Its removal is instead pinned by _W2C_RESIDUAL_WORKLIST below: maskdecoder
# carries EXACTLY ONE salt.models.maskformer import (MaskDecoderLayer), not two —
# proving the get_masks import is gone (and after W2c-2 the V1Dense import too).
# the W2c-3 END-STATE worklist: the FINAL absorption has LANDED. After W2c-1 (task
# family absorbed -> nn/tasks.py salt.models.task imports gone), W2c-2 (the
# Dense/Transformer/pooling family absorbed into nn/modules.py + maskdecoder.py
# V1Dense gone), and W2c-3 (the MaskFormer decoder-layer + matched-loss/matcher
# family absorbed VERBATIM into nn/maskdecoder.py + nn/maskformer_loss.py), the
# residual is EMPTY — the 18 -> 11 -> 7 -> 3 -> 0 monotonic drop is now complete and
# gate-verified. salt.core is fully decoupled from the v1 tree (RS1 count == 0, the
# W2 self-containment success criterion). The pin stays EMPTY so any regression that
# re-introduces a flagged v1 import trips this test (and test_w2c_residual_is_zero
# below asserts the count is 0 explicitly).
_W2C_RESIDUAL_WORKLIST: set[tuple[str, int, str]] = set()
_EXPECTED_KEEPS = {"event_classifier", "regression_multi_target"}
# the 6 Wave-F2-RESTORED configs (fixture restored to full v1 capability).
_EXPECTED_RESTORED = {
    "regression",
    "regression_gaussian",
    "regression_weighted",
    "nan_regression",
    "GN3V01",
    "GN2XE",
}


class TestCV1:
    """CV1 — convert-config acceptance over the FULL 39-config matrix §4 denominator.

    Pins: the embedded authoritative config list (29 fixture-bearing needs-M5/M6
    + 10 reproducible-today ✅), every config converts + validates + plan-compiles,
    the fixture-plan semantic equivalence (exact for unflagged fixture configs;
    structural-equivalence + full-v1-capability for the 6 Wave-F2-RESTORED configs;
    fixture-divergent for the 2 pinned intentional keeps; N/A for the ✅ configs),
    the task-count faithfulness fact, and the corruption teeth (a bit-rotted v1
    source breaks conversion -> gate red).
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

    def test_unflagged_fixture_configs_plan_match_the_fixture(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        for c in report["configs"]:
            # only the FIXTURE-bearing configs that are NEITHER restored NOR
            # intentional-keeps plan-match exactly; the ✅ today configs have no
            # fixture (skipped), the 6 restored configs are structurally
            # equivalent (not byte-exact), the 2 keeps are divergent by design
            if c["today"] or c["restored"] or c["name"] in _CV1_FIXTURE_SUBSET:
                continue
            assert c["fixture_match"], (
                f"{c['name']} converted plan != hand-written fixture plan: "
                f"modes={c['mode_match']} eval={c['eval_manifest_match']} "
                f"onnx={c['onnx_manifest_match']}"
            )
            assert c["eval_manifest_match"], c["name"]
            assert c["onnx_manifest_match"], c["name"]
            assert c["converter_faithful_to_v1"], c["name"]

    def test_restored_set_is_pinned_to_the_six_f2_restores(self, tmp_path):
        # Wave F2 restored exactly 6 once-subset fixtures to full v1 fidelity;
        # the restored set is pinned (a regression back to a subset changes the
        # accounting). The restored set and the keep set are disjoint.
        _code, report = run_cv1(tmp_path)
        assert set(_CV1_RESTORED) == _EXPECTED_RESTORED
        assert report["config"]["restored_configs"] == sorted(_CV1_RESTORED)
        assert report["checks"]["restored_set_pinned"]
        assert set(_CV1_RESTORED).isdisjoint(set(_CV1_FIXTURE_SUBSET))

    def test_restored_configs_reproduce_full_v1_capability(self, tmp_path):
        # the 6 F2-restored configs: the converter + the (now-full-fidelity)
        # fixture both carry the FULL v1 task set AND are structurally equivalent
        # (module-class multiset equal in every shared mode). They are NO LONGER
        # subset-divergent — the gate asserts full-v1-capability, not divergence.
        _code, report = run_cv1(tmp_path)
        for c in report["configs"]:
            if not c["restored"]:
                continue
            assert c["converted"], c["name"]
            assert c["validate"].get("fit") == 0, c["name"]
            assert c["validate"].get("test") == 0, c["name"]
            # full v1 capability: conv == v1 == fixture task count
            assert c["task_counts_faithful"], (
                f"{c['name']}: conv={c['conv_task_count']} v1={c['v1_task_count']} "
                f"fixture={c['fixture_task_count']}"
            )
            assert c["conv_task_count"] == c["v1_task_count"] == c["fixture_task_count"], c["name"]
            # structural equivalence: same module-class multiset every shared mode
            assert c["structurally_equivalent"], (
                f"{c['name']} converter is NOT structurally equivalent to the restored fixture "
                f"(module-class set differs): {c['structural_equivalence']}"
            )
            assert c["reproduces_full_v1_capability"], c["name"]
            assert c["converter_faithful_to_v1"], c["name"]
            assert report["checks"][f"{c['name']}:reproduces_full_v1_capability"], c["name"]
            assert report["checks"][f"{c['name']}:structurally_equivalent"], c["name"]
            # the restored configs are NOT flagged as intentional keeps
            assert c["name"] not in _CV1_FIXTURE_SUBSET
            assert not c["fixture_subset"]

    def test_intentional_keeps_reproduce_full_v1_tasks(self, tmp_path):
        # the 2 documented keeps still reproduce the FULL v1 task count
        # (conv == v1 >= fixture) — the divergence is the FIXTURE's, by design.
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

    def test_intentional_keeps_pinned_named_and_diverge(self, tmp_path):
        _code, report = run_cv1(tmp_path)
        # the keep set shrank from 8 (pre-F2) to exactly the 2 documented keeps
        assert set(_CV1_FIXTURE_SUBSET) == _EXPECTED_KEEPS
        assert report["config"]["fixture_subset_configs"] == sorted(_CV1_FIXTURE_SUBSET)
        assert report["checks"]["fixture_subset_set_pinned"]
        for c in report["configs"]:
            if c["name"] not in _CV1_FIXTURE_SUBSET:
                continue
            # keep configs: the fixture is the faithful artifact by design -> the
            # exact plan match does NOT hold, but they still convert + validate
            assert c["fixture_subset"]
            assert not c["restored"]
            assert not c["fixture_match"], (
                f"{c['name']} is a flagged intentional keep but now MATCHES the fixture — the "
                "accounting changed; re-audit whether it should be a restored config instead"
            )
            # each keep documents WHY it diverges (the codex CVF finding)
            assert c["keep_reason"], f"{c['name']} keep lacks a documented keep_reason"
            assert report["checks"][f"{c['name']}:intentional_keep_divergent"]
            assert report["checks"][f"{c['name']}:keep_reason_documented"]
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


class TestCVF:
    """CVF — converter-output target-producer fidelity (the validate/plan-compile MISS).

    Pins: every converted task's target is PRODUCED by a data module (check (i)),
    every v1-declared synthetic multi_target handle is produced CONCRETELY (not
    served only by the Labels wildcard), the converter task/stream set equals the
    v1 resolved set for faithful configs (check (ii)), the named fixture exceptions
    are excluded from fixture equality but NOT from check (i) (check (iii)), and the
    corruption teeth (deleting a config's MultiTarget producer -> the synthetic
    target falls back to the wildcard -> gate red).
    """

    def test_pass(self, tmp_path):
        code, report = run_cvf(tmp_path)
        assert code == 0, {k: v for k, v in report["checks"].items() if not v}
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "cvf_report.json").is_file()

    def test_runs_over_cv1_denominator_minus_expected_hard_errors(self, tmp_path):
        _code, report = run_cvf(tmp_path)
        # CVF iterates the SAME CV1 config set, skipping only the
        # expected-hard-error configs (tutorial: no converter output to check)
        expected_hard_err = {e["name"] for e in _CV1_CONFIGS if e.get("expect_convert_error")}
        cvf_names = {c["name"] for c in report["configs"]}
        cv1_names = {e["name"] for e in _CV1_CONFIGS}
        assert cvf_names == cv1_names - expected_hard_err
        assert "tutorial" not in cvf_names  # the one expected-hard-error config
        assert report["config"]["total_configs"] == len(cv1_names) - len(expected_hard_err)

    def test_multi_target_now_produces_its_target_concretely(self, tmp_path):
        # the heart of F1a: regression_multi_target's pt_label_handle custom_target
        # is now PRODUCED by a concrete MultiTarget data module, not phantom-served
        # by the Labels wildcard (the validate/plan-compile MISS that let the
        # broken config pass CV1).
        _code, report = run_cvf(tmp_path)
        rmt = next(c for c in report["configs"] if c["name"] == "regression_multi_target")
        assert rmt["converted"]
        # the v1 source declares the synthetic pt_label_handle custom_target
        assert "labels.jets.pt_label_handle" in rmt["v1_synthetic_targets"]
        # it is consumed by a task in the converted FIT plan...
        prod = rmt["task_target_producers"]
        assert "labels.jets.pt_label_handle" in prod
        entry = prod["labels.jets.pt_label_handle"]
        # ...and produced by a CONCRETE MultiTarget (NOT the Labels wildcard)
        assert "MultiTarget" in entry["producers"]
        assert not entry["wildcard_only"], "pt_label_handle is phantom-served by the wildcard"
        assert "RegressionTaskModule" in entry["consumers"]
        # the gate's per-config target check passes for it
        assert rmt["synthetic_targets_concretely_produced"]
        assert report["checks"]["regression_multi_target:targets_produced"]

    def test_every_task_target_has_a_producer(self, tmp_path):
        _code, report = run_cvf(tmp_path)
        for c in report["configs"]:
            if not c["converted"]:
                continue
            assert c["all_targets_have_producer"], c["name"]
            assert report["checks"][f"{c['name']}:targets_produced"], (
                f"{c['name']}: a task target is not concretely produced "
                f"(producers={c['task_target_producers']})"
            )

    def test_converter_faithful_to_v1_task_and_stream_set(self, tmp_path):
        # check (ii): for EVERY convertible config (incl. the fixture exceptions,
        # which are faithful to the v1 SOURCE — only their fixture diverges) the
        # converter task-name + stream set equals the v1 resolved set
        _code, report = run_cvf(tmp_path)
        for c in report["configs"]:
            if not c["converted"]:
                continue
            assert c["converter_v1_task_match"], (
                f"{c['name']}: conv tasks {c['conv_tasks']} != v1 tasks {c['v1_tasks']}"
            )
            assert c["converter_v1_stream_match"], (
                f"{c['name']}: conv streams {c['conv_streams']} != v1 {c['v1_streams']}"
            )
            assert report["checks"][f"{c['name']}:faithful_to_v1"], c["name"]

    def test_fixture_exceptions_named_and_pinned(self, tmp_path):
        # check (iii): event_classifier + regression_multi_target are the NAMED
        # fixture exceptions, pinned to the CV1 fixture-subset set; they are
        # excluded from fixture equality but regression_multi_target is still NOT
        # excused from check (i) below
        _code, report = run_cvf(tmp_path)
        assert report["cvf_fixture_exceptions"] == list(_CVF_FIXTURE_EXCEPTIONS)
        assert set(_CVF_FIXTURE_EXCEPTIONS) == {"event_classifier", "regression_multi_target"}
        assert report["checks"]["fixture_exceptions_pinned"]
        # both exceptions are real CV1 fixture-subset configs
        assert set(_CVF_FIXTURE_EXCEPTIONS) <= set(_CV1_FIXTURE_SUBSET)
        for name in _CVF_FIXTURE_EXCEPTIONS:
            row = next(c for c in report["configs"] if c["name"] == name)
            assert row["fixture_exception"]
            # NOT excused from check (i): each must still pass targets_produced
            assert report["checks"][f"{name}:targets_produced"], name

    def test_corruption_fails_the_gate(self, tmp_path):
        # the negative control: DELETE the MultiTarget data module from the
        # converted regression_multi_target config. Its pt_label_handle target then
        # falls back to the demand-driven Labels wildcard (plan-compile STILL
        # succeeds — the MISS), so the target's wildcard_only flag flips True and
        # CVF goes red, proving check (i) has teeth.
        def corrupt(cfg):
            cfg["data"]["modules"].pop("multi_target", None)
            return cfg

        code, report = run_cvf(tmp_path, corruption=corrupt)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["regression_multi_target:targets_produced"]
        rmt = next(c for c in report["configs"] if c["name"] == "regression_multi_target")
        # the synthetic target is now wildcard-only (phantom) -> not concretely produced
        assert not rmt["synthetic_targets_concretely_produced"]
        entry = rmt["task_target_producers"]["labels.jets.pt_label_handle"]
        assert entry["wildcard_only"]
        assert "MultiTarget" not in entry["producers"]


class TestREN1:
    """REN1 — vector->global_object rename + StreamEmbed flag collapse (W1.5 wave R).

    Pins: (a) no residual `vector` FLAG use in salt/core (only VectorConcat / prose
    / the 3 named historical references allowed); (b) every shipped M5/M6-CONV
    config validates fit/test/onnx with the new global_object flag; (c) the same
    flag-free StreamEmbed infers rank-2 [B,F] -> [B,D] AND rank-3 [B,T,F] ->
    [B,T,D]; and the corruption teeth (an injected residual-flag hit -> gate red).
    """

    def test_pass(self, tmp_path):
        code, report = run_ren1(tmp_path)
        assert code == 0, {k: v for k, v in report["checks"].items() if not v}
        assert report["passed"]
        assert all(report["checks"].values())
        assert (tmp_path / "ren1_report.json").is_file()

    def test_no_residual_vector_flag(self, tmp_path):
        # (a) the rename left NO flag-shaped `vector` use in salt/core
        _code, report = run_ren1(tmp_path)
        assert report["checks"]["no_residual_vector_flag"], (
            f"residual vector FLAG hits: {report['residual_flag_hits']}"
        )
        assert report["residual_flag_hits"] == []
        assert report["config"]["residual_flag_hits"] == 0

    def test_every_shipped_config_validates_with_global_object(self, tmp_path):
        # (b) every M5/M6-CONV shipped config validates fit/test/onnx data-free
        _code, report = run_ren1(tmp_path)
        for r in report["shipped_config_validation"]:
            assert r["files_present"], r["name"]
            assert r["validate"].get("fit") == 0, r["name"]
            assert r["validate"].get("test") == 0, r["name"]
            if "onnx" in r["validate"]:
                assert r["validate"]["onnx"] == 0, r["name"]
            assert r["all_modes_ok"], r["name"]
        # data-free, NOT --strict (the M5/M6-CONV precedent)
        assert report["config"]["strict"] is False
        assert "NOT --strict" in report["no_strict_rationale"]
        assert (
            report["config"]["shipped_configs_validated"] == report["config"]["shipped_configs"]
        )
        # the denominator is the authoritative M5-CONV (24) + M6-CONV (5) lists
        assert report["config"]["shipped_configs"] == len(_REN1_SHIPPED_CONFIGS)

    def test_streamembed_infers_rank_for_rank2_and_rank3(self, tmp_path):
        # (c) the SAME flag-free StreamEmbed: rank-2 [B,F] -> [B,D] (no token
        # axis) AND rank-3 [B,T,F] -> [B,T,D] (token axis preserved)
        _code, report = run_ren1(tmp_path)
        rank = report["rank_inference"]
        r2, r3 = rank["rank2"], rank["rank3"]
        # rank-2 [B,F] bound input -> rank-2 [B,D] embed (no token axis)
        assert r2["input_rank"] == 2
        assert r2["embed_rank"] == 2
        # rank-3 [B,T,F] bound input -> rank-3 [B,T,D] embed (token axis preserved)
        assert r3["input_rank"] == 3
        assert r3["embed_rank"] == 3
        # the same out_dim width contributed via derived_widths in both cases
        assert r2["derived_width"] == r3["derived_width"]
        # the StreamEmbed carries NO `vector` rank flag (the param is DELETED)
        assert r2["embed_has_no_vector_param"]
        assert r3["embed_has_no_vector_param"]
        for cname in (
            "rank2_embed_is_rank2",
            "rank3_embed_is_rank3",
            "out_dim_via_derived_widths",
            "streamembed_has_no_vector_attr",
        ):
            assert report["checks"][f"rank:{cname}"], cname

    def test_historical_refs_and_vectorconcat_are_pinned(self, tmp_path):
        # the allowed exemptions are recorded: the out-of-scope VectorConcat
        # surface + the 3 named historical references that explain the collapse
        _code, report = run_ren1(tmp_path)
        assert "VectorConcat" in report["vectorconcat_tokens_allowed"]
        suffixes = {h["file_suffix"] for h in report["historical_refs_allowed"]}
        assert {"nn/modules.py", "configs/DL1.yaml", "convert.py"} == suffixes

    def test_corruption_fails_the_gate(self, tmp_path):
        # the negative control: INJECT a fake residual-flag hit -> the
        # no_residual_vector_flag check flips False and the gate goes red,
        # proving the grep check (a) is not vacuous.
        def corrupt(state):
            state["residual"] = [
                {"file": "data/reader.py", "line": "88", "text": "vector: bool | None = None",
                 "pattern": "fake"}
            ]
            return state

        code, report = run_ren1(tmp_path, corruption=corrupt)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["no_residual_vector_flag"]
        assert report["config"]["residual_flag_hits"] == 1


class TestRS1:
    """RS1 — residual v1-import scanner (the M7 W2 worklist + W2c end-state gate).

    Pins: the gate RUNS and emits the authoritative residual production-import
    list; it is AST-based (so the convert.py:119 comment + :355 f-string that NAME
    salt.models in prose are NOT counted); the W2a-relocated helpers (array_utils
    / scalers / union_find / file_utils / samplers / optim / get_dtype) are GONE
    from the residual list; the W2b inline-math + mask_utils relocation sites
    (tensor_utils / edge_features / mask_utils + the maskformer get_masks /
    get_maskformer_outputs / task reduce imports) are GONE too — the salt.utils
    tree is fully decoupled and the residual is the pinned W2c worklist (the
    18 -> 11 monotonic drop is gate-verified); and the corruption teeth (an
    injected synthetic salt.models residual -> gate red).
    """

    def test_runs_and_emits_residual_list(self, tmp_path):
        # the gate RUNS and writes its report; during W2a/W2b rc may be nonzero
        # (residual W2b+W2c imports remain) — the gate is the worklist, not yet
        # green. The report carries the authoritative residual list either way.
        code, report = run_rs1(tmp_path)
        assert code in {0, 1}
        assert (tmp_path / "rs1_report.json").is_file()
        assert "residual_imports" in report
        assert report["config"]["residual_import_count"] == len(report["residual_imports"])
        # green iff the residual list is empty (the W2c end-state)
        assert report["passed"] == (report["residual_imports"] == [])

    def test_ast_based_not_text_grep(self, tmp_path):
        # the convert.py:119 comment + :355 f-string NAME salt.models in prose;
        # an AST scan must NOT count them. So NO residual hit may be reported on a
        # convert.py line that is a comment / string literal.
        _code, report = run_rs1(tmp_path)
        assert report["config"]["ast_based"] is True
        convert_hits = [h for h in report["residual_imports"] if h["file"] == "convert.py"]
        # convert.py only NAMES salt.models in prose (a comment + an f-string) —
        # it has no live v1 import node, so the AST scan reports zero hits for it.
        assert convert_hits == [], convert_hits

    def test_w2a_relocations_are_gone_from_residual(self, tmp_path):
        # every W2a-relocated helper import has been repointed to salt.core.* —
        # none may remain in the residual list.
        _code, report = run_rs1(tmp_path)
        residual_modules = {h["module"] for h in report["residual_imports"]}
        leftover = _W2A_RELOCATED_MODULES & residual_modules
        assert leftover == set(), f"W2a relocations still residual: {sorted(leftover)}"

    def test_w2b_relocations_are_gone_from_residual(self, tmp_path):
        # the M7 W2b inline-math + mask_utils relocation wave: every W2b-removed
        # import SITE has been inlined/repointed to salt.core.* — none survive in
        # the residual. Asserted at (file, module) granularity because
        # salt.models.maskformer persists at a DIFFERENT (W2c) site, so a
        # module-level check would mis-fire. Plus the categorical W2b outcome: NO
        # salt.utils.* import survives anywhere in production salt.core (the
        # tensor_utils / edge_features / mask_utils relocations are total).
        _code, report = run_rs1(tmp_path)
        residual_sites = {(h["file"], h["module"]) for h in report["residual_imports"]}
        leftover = _W2B_RELOCATED_SITES & residual_sites
        assert leftover == set(), f"W2b-removed import sites still residual: {sorted(leftover)}"
        # the salt.utils tree is fully decoupled (W2a + W2b utils relocations):
        utils_residual = [
            h for h in report["residual_imports"] if h["module"].startswith("salt.utils")
        ]
        assert utils_residual == [], f"salt.utils still imported by core: {utils_residual}"

    def test_w2c_residual_worklist_is_pinned(self, tmp_path):
        # the W2c-3 END-STATE: the FINAL absorption has landed, so the pinned
        # worklist is EMPTY and the residual must match it (count 0). Matched at
        # (file, module) multiset granularity (robust to benign line shifts; a
        # re-added flagged v1 import or a reverted W2c-3 absorption still trips it).
        _code, report = run_rs1(tmp_path)
        residual_fm = sorted((h["file"], h["module"]) for h in report["residual_imports"])
        worklist_fm = sorted((f, m) for f, _lineno, m in _W2C_RESIDUAL_WORKLIST)
        assert residual_fm == worklist_fm, (
            f"residual != pinned W2c worklist\n  residual: {residual_fm}\n  worklist: {worklist_fm}"
        )
        assert report["config"]["residual_import_count"] == len(_W2C_RESIDUAL_WORKLIST)
        # the W2c-3 success criterion: ZERO residual v1 imports (salt.core fully
        # decoupled from the v1 tree). The gate PASSES (rc 0 GREEN) at this count.
        assert report["config"]["residual_import_count"] == 0
        assert report["checks"]["no_residual_v1_imports"] is True
        assert _code == 0

    def test_harness_files_are_excluded(self, tmp_path):
        # the gate harnesses + v1 adapters import v1 deliberately and are scoped
        # OUT — no residual hit may be reported on one of them.
        _code, report = run_rs1(tmp_path)
        excluded = set(report["config"]["excluded_harness_basenames"])
        prefixes = tuple(report["config"]["excluded_harness_prefixes"])
        for h in report["residual_imports"]:
            base = h["file"].rsplit("/", 1)[-1]
            assert base not in excluded, h
            assert not base.startswith(prefixes), h

    def test_flagged_packages_are_the_v1_tree(self, tmp_path):
        _code, report = run_rs1(tmp_path)
        assert set(report["config"]["flagged_packages"]) == {
            "models",
            "data",
            "utils",
            "onnx",
            "optim",
            "modelwrapper",
            "callbacks",
            "submit",
            "stypes",
        }
        # every residual hit's package is one of the flagged v1 packages
        for h in report["residual_imports"]:
            assert h["pkg"] in report["config"]["flagged_packages"], h

    def test_corruption_fails_the_gate(self, tmp_path):
        # the negative control: INJECT a synthetic salt.models import residual ->
        # the no_residual_v1_imports check flips False and the gate goes red,
        # proving the scanner is not vacuous.
        def corrupt(state):
            state["residual"] = [
                {
                    "file": "nn/synthetic.py",
                    "lineno": 1,
                    "module": "salt.models",
                    "pkg": "models",
                }
            ]
            return state

        code, report = run_rs1(tmp_path, corruption=corrupt)
        assert code == 1
        assert not report["passed"]
        assert not report["checks"]["no_residual_v1_imports"]
        assert report["config"]["residual_import_count"] == 1


def test_cli_dispatch_cv1(tmp_path):
    assert gm7.main(["cv1", "--outdir", str(tmp_path)]) == 0


def test_cli_dispatch_cv2(tmp_path):
    assert gm7.main(["cv2", "--outdir", str(tmp_path)]) == 0


def test_cli_dispatch_cvf(tmp_path):
    assert gm7.main(["cvf", "--outdir", str(tmp_path)]) == 0


def test_cli_dispatch_ren1(tmp_path):
    assert gm7.main(["ren1", "--outdir", str(tmp_path)]) == 0
