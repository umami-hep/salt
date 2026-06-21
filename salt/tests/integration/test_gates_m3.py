"""Tests for the M3 gates harness (`salt.tests.integration.gates_m3`, plan 06 stage C).

Dummy-file runs of W1/W3/W4/W5 plus a W2 smoke on a tiny file carrying the
open-data variable names. W3 doubles as the harness's own negative control
(its corrupted sub-run must FAIL W1 on exactly the swapped columns), so a
green W3 here proves the comparison is not vacuous.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from salt.tests.integration.gates_m3 import (
    _dummy_eval_data,
    _opendata_variables,
    run_w1,
    run_w2,
    run_w3,
    run_w4,
    run_w5,
)
from salt.tests._fixtures.gn2_fixture import ELECTRON_VARIABLES
from salt.utils.inputs import write_dummy_file

BATCH = 100
NUM = 300  # 3 batches — enough to exercise multi-batch streaming + row alignment


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    """The shared dummy fixture data (parity dicts + 1000-jet ttbar-stem file)."""
    return _dummy_eval_data(tmp_path_factory.mktemp("m3_gates_data"))


def make_opendata_style_data(base: Path) -> tuple[Path, Path]:
    """A tiny dummy file + norm dict carrying the OPEN-DATA variable names.

    The opendata-config track list (lifetimeSigned* renames) drives the
    dummy generator, so W2's variable-override path is exercised without
    the real 104 MB file.
    """
    variables = _opendata_variables()
    sd = {
        stream: {
            v: {"mean": round(0.1 * (i + 1), 6), "std": round(1.0 + 0.05 * (i + 1), 6)}
            for i, v in enumerate(names)
        }
        for stream, names in variables.items()
    }
    sd["electrons"] = {v: {"mean": 0.5, "std": 1.5} for v in ELECTRON_VARIABLES}
    nd_path = base / "norm_dict.yaml"
    with open(nd_path, "w") as fh:
        yaml.dump(sd, fh, sort_keys=False)
    h5_path = base / "pp_output_smoke.h5"
    write_dummy_file(h5_path, nd_path)
    return h5_path, nd_path


class TestW1:
    def test_passes_on_dummy_file(self, dummy, tmp_path):
        # batch_size 96 deliberately does NOT divide num_test=300 (final
        # batch of 12 rows): the partial-final-batch path is part of the
        # suite, mirroring the shipped gate default (M3-review fix)
        code, report = run_w1(
            tmp_path, file=dummy.file, norm_dict=dummy.norm_dict, batch_size=96, num_test=NUM
        )
        assert code == 0
        assert report["passed"]
        comparison = report["comparison"]
        assert comparison["datasets_match"]
        assert comparison["n_failed"] == 0
        assert comparison["n_columns"] == comparison["n_bitwise"] + comparison["n_justified"]
        assert all(group["schema_match"] for group in comparison["groups"].values())
        # the v1 per-group layout bar: copies first, task columns, mask LAST
        tracks = comparison["groups"]["tracks"]["v1_columns"]
        assert tracks[-1] == "mask"
        assert "VertexIndex" in tracks
        # report artifact exists
        assert (tmp_path / "w1_report.json").exists()

    def test_file_without_norm_dict_rejected(self, dummy, tmp_path):
        with pytest.raises(ValueError, match="together"):
            run_w1(tmp_path, file=dummy.file)


class TestW3:
    def test_negative_controls_fire(self, tmp_path):
        code, report = run_w3(tmp_path, batch_size=BATCH, num_test=200)
        assert code == 0
        assert report["passed"]
        corrupted = report["control_corrupted_comparison"]
        assert corrupted["w1_failed"]
        assert corrupted["v2_cli_succeeded"]  # the FAILURE is the comparison, not a crash
        assert corrupted["failed_columns"] == ["jets.salt_pb", "jets.salt_pc"]
        dead = report["control_dead_preds"]
        assert dead["errored"]
        assert dead["message_names_dead_preds"]
        assert dead["message_names_both_tasks"]
        # M3-review fix: the dead-preds sub-run points writers.output at a
        # FRESH path and asserts THAT file absent (the old count-only check
        # was vacuous against an overwrite of the pre-existing W1 file)
        assert dead["no_new_eval_file"]
        assert not (tmp_path / "dead_preds_run" / "dead_preds_eval.h5").exists()


class TestW4:
    def test_custom_writer_journey(self, tmp_path):
        code, report = run_w4(tmp_path, batch_size=BATCH, num_test=200)
        assert code == 0
        assert report["passed"]
        assert report["override_lines"] <= 4
        assert report["checks"]["new_column_values_match_source"]
        assert report["checks"]["shipped_columns_survive_merge"]


class TestW5:
    def test_metrics_parity(self, dummy, tmp_path):
        code, report = run_w5(
            tmp_path, file=dummy.file, norm_dict=dummy.norm_dict, n_batches=2, batch_size=BATCH
        )
        assert code == 0
        assert report["passed"]
        assert set(report["tasks"]) == {"jets_classification", "track_origin"}
        for task, record in report["tasks"].items():
            assert record["lists_equal"], task
            assert record["matrix_equal"], task
            assert record["ignored_equal"], task
            assert record["n_entries"] > 0
            # the M3-review non-degeneracy bar: predictions spread over >= 2
            # classes (the untrained fixture used to argmax-collapse)
            assert record["pred_diversity_ok"], (task, record["pred_classes"])
        # the per-track task must actually exercise the -1 padding reduction
        assert report["tasks"]["track_origin"]["v1_ignored"] > 0


class TestW2Smoke:
    def test_smoke_on_tiny_opendata_style_file(self, tmp_path):
        h5_path, nd_path = make_opendata_style_data(tmp_path)
        code, report = run_w2(h5_path, nd_path, tmp_path / "out", batch_size=BATCH, num_test=200)
        assert code == 0
        assert report["passed"]
        # the open-data renames really flowed through both sides
        assert "lifetimeSignedD0Significance" in report["config"]["variables"]["tracks"]
        tracks = report["comparison"]["groups"]["tracks"]["v1_columns"]
        assert "lifetimeSignedD0Significance" in tracks
        assert report["comparison"]["n_failed"] == 0
