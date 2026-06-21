"""Tests for the M2 gates harness (plan 05, stage D) on the dummy-file generators.

G1/G2/G3/G5 run at small K against a tmp dummy file (the formal gate run on
the real open-data file is the 04_m2_gates experiment's job — paths are CLI
args there, never repo constants). G4 gets a tiny-file machinery smoke with
the threshold disabled (sub-second timings on a 1000-jet file measure noise,
not throughput). Negative controls prove the comparisons have teeth: G1 must
FAIL on a mis-mapped label key, G3 on a perturbed transferred weight — both
via the Python-only test hooks (`run_g1` ``corruption`` / `run_g3`
``perturb``), which the CLI deliberately does not expose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from salt.tests.integration.gates_m2 import g3_budget, main, run_g1, run_g2, run_g3, run_g4, run_g5
from salt.core.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2_fixture import write_parity_norm_dict
from salt.utils.inputs import write_dummy_file


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    # dummy H5 + parity norm dict (distinct per-variable constants) + schema artifact
    base = tmp_path_factory.mktemp("gates_m2")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "cd": cd_path, "schema": schema_path}


class TestG1:
    def test_pass_on_dummy_file(self, data, tmp_path):
        code, report = run_g1(data["h5"], data["nd"], tmp_path, n_batches=3, batch_size=300)
        assert code == 0
        assert report["passed"]
        assert report["config"]["length_match"]
        assert all(record["bitwise"] for record in report["comparisons"])
        assert (tmp_path / "g1_report.json").is_file()

    def test_compares_all_leaves_and_tail(self, data, tmp_path):
        code, report = run_g1(data["h5"], data["nd"], tmp_path, n_batches=2, batch_size=400)
        assert code == 0
        keys = {record["key"] for record in report["comparisons"]}
        assert keys == {
            "inputs.jets",
            "inputs.tracks",
            "masks.tracks",
            "labels.jets.flavour_label",
            "labels.tracks.ftagTruthOriginLabel",
            "labels.tracks.ftagTruthVertexIndex",
        }
        # the tail probe reaches the file's final row
        assert report["config"]["slices"][-1][1] == report["config"]["n_rows"]

    def test_mismapped_label_key_fails(self, data, tmp_path):
        # negative control: v2 labels swapped between two track labels — the
        # comparison must catch the mis-mapping. Origin vs Type, NOT Origin
        # vs VertexIndex: the dummy-file generator fills those two from the
        # same draw (identical arrays), which would make the control vacuous.
        def swap(batch):
            tracks = batch["labels"]["tracks"]
            tracks["ftagTruthOriginLabel"], tracks["ftagTruthTypeLabel"] = (
                tracks["ftagTruthTypeLabel"],
                tracks["ftagTruthOriginLabel"],
            )
            return batch

        code, report = run_g1(
            data["h5"],
            data["nd"],
            tmp_path,
            n_batches=1,
            batch_size=200,
            track_labels=("ftagTruthOriginLabel", "ftagTruthTypeLabel"),
            corruption=swap,
        )
        assert code == 1
        assert not report["passed"]
        failing = {r["key"] for r in report["comparisons"] if not r["passed"]}
        assert failing == {
            "labels.tracks.ftagTruthOriginLabel",
            "labels.tracks.ftagTruthTypeLabel",
        }


class TestG2:
    def test_fit_smoke_passes(self, data, tmp_path):
        code, report = run_g2(
            data["h5"],
            data["h5"],
            data["nd"],
            tmp_path,
            schema=data["schema"],
            steps=12,
        )
        assert code == 0
        assert report["passed"]
        assert all(report["checks"].values()), report["checks"]
        assert len(report["train_losses"]) >= 10
        assert np.isfinite(report["train_losses"]).all()
        assert report["last_quartile_mean"] < report["first_quartile_mean"]
        assert (tmp_path / "g2_report.json").is_file()


class TestG3:
    def test_budget_shape(self):
        assert g3_budget(0) == pytest.approx(1e-6)
        assert g3_budget(3) == pytest.approx(8e-6)
        # the cap keeps large --steps runs gated (critic finding: 1e-6*2**k
        # exceeds typical loss magnitudes beyond step ~23)
        assert g3_budget(30) == pytest.approx(1e-3)
        assert g3_budget(100) == pytest.approx(1e-3)

    def test_parity_passes(self, data, tmp_path):
        code, report = run_g3(data["h5"], data["nd"], data["cd"], tmp_path, steps=4, batch_size=100)
        assert code == 0
        assert report["passed"]
        assert len(report["curve"]) == 4
        for point in report["curve"]:
            assert point["abs_diff"] <= point["budget"]
            for task in ("jets_classification", "track_origin", "track_vertexing"):
                assert point["tasks"][task]["abs_diff"] <= point["budget"]
        # step 0 is the strict forward-equivalence bound (no drift yet)
        assert report["curve"][0]["abs_diff"] <= 1e-6
        assert (tmp_path / "g3_report.json").is_file()

    def test_weighting_parity_block(self, data, tmp_path):
        _, report = run_g3(data["h5"], data["nd"], data["cd"], tmp_path, steps=1)
        weighting = report["weighting_parity"]
        assert weighting["task_weights_match"]
        assert weighting["transferred_buffer_matches_v1"]
        assert weighting["transferred_buffer_matches_dict"]
        assert weighting["materialised_buffer_matches_dict"]
        # the fixture class dict carries genuinely non-uniform weights, so
        # the parity assertions above are not vacuous
        weights = weighting["class_weights_from_dict"]
        assert len(weights) == 8
        assert len(set(weights)) > 1

    def test_perturbed_weight_fails(self, data, tmp_path):
        # negative control: a 1e-2 nudge on one transferred weight must blow
        # the step-0 budget (1e-6) and fail the gate
        code, report = run_g3(data["h5"], data["nd"], data["cd"], tmp_path, steps=2, perturb=1e-2)
        assert code == 1
        assert not report["passed"]
        assert not report["curve"][0]["passed"]
        assert report["config"]["perturbed_by_test_hook"]


class TestG4:
    def test_machinery_smoke_tiny_file(self, data, tmp_path):
        # 1000 jets time in milliseconds — this smokes the harness machinery
        # (both loaders, column equalisation, report shape); the 0.97 ratio
        # gate only means something on the real file (the experiment's job),
        # so the threshold is disabled here.
        code, report = run_g4(
            data["h5"],
            data["nd"],
            tmp_path,
            batch_size=250,
            num_workers=0,
            warmup_epochs=1,
            timed_epochs=1,
            repeats=3,
            threshold=0.0,
        )
        assert code == 0
        # drop_last: 4 full batches of 250 = all 1000 jets, both sides
        assert report["v1"]["jets"] == report["v2"]["jets"] == 1000
        assert report["same_jet_count"]
        assert report["v1"]["jets_per_s"] > 0
        assert report["v2"]["jets_per_s"] > 0
        assert report["ratio_v2_over_v1"] > 0
        # interleaved repeats: per-repeat samples + spread are reported and
        # the verdict ratio is the median of the per-repeat ratios
        assert len(report["v1"]["jets_per_s_per_repeat"]) == 3
        assert len(report["v2"]["jets_per_s_per_repeat"]) == 3
        assert len(report["ratios_per_repeat"]) == 3
        assert report["ratio_v2_over_v1"] == pytest.approx(
            float(np.median(report["ratios_per_repeat"]))
        )
        assert report["v1"]["spread_frac"] >= 0
        # the demanded model columns are identical both sides; the v2 read
        # set is features + labels + valid exactly (demand narrowing)
        assert set(report["v2"]["read_columns"]["jets"]) <= set(
            report["v1"]["read_columns"]["jets"]
        )
        # v1 reads MORE jets columns on the dummy file (read-all
        # amplification), so the equal-work control must have run with v2's
        # read set expanded to v1's
        assert report["equal_work"]["ran"]
        assert set(report["equal_work"]["read_columns"]["jets"]) == set(
            report["v1"]["read_columns"]["jets"]
        )
        assert report["equal_work"]["v2_equal_work"]["jets_per_s"] > 0
        assert report["equal_work"]["ratio_vs_v1_production_median"] > 0
        assert (tmp_path / "g4_report.json").is_file()


class TestG5:
    def test_resume_and_compile_roundtrip(self, data, tmp_path):
        code, report = run_g5(data["h5"], data["nd"], tmp_path, k=2, batch_size=100)
        assert code == 0
        assert report["passed"]
        assert all(report["checks"].values()), report["checks"]
        losses = report["losses"]
        assert len(losses["uninterrupted"]) == 4
        assert len(losses["resumed_second_half"]) == 2
        assert report["continuity_max_abs_diff"] <= 1e-7
        assert report["pre_resume_max_abs_diff"] <= 1e-7
        assert (tmp_path / "g5_plain.ckpt").is_file()
        assert (tmp_path / "g5_compiled.ckpt").is_file()
        assert (tmp_path / "g5_report.json").is_file()


class TestCLI:
    def test_g1_subcommand(self, data, tmp_path):
        rc = main([
            "g1",
            "--file",
            str(data["h5"]),
            "--norm-dict",
            str(data["nd"]),
            "--outdir",
            str(tmp_path),
            "--n-batches",
            "1",
            "--batch-size",
            "200",
        ])
        assert rc == 0
        assert (tmp_path / "g1_report.json").is_file()
