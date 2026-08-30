"""Tests for salt.profiling (the dataset line profiler + model torch.profiler)."""

import json
from types import SimpleNamespace

import pytest

from salt.profiling import (
    DEFAULT_DATASET_FUNCTIONS,
    DEFAULT_STEPS,
    TorchProfilerCallback,
    _dataset_summary,
    _model_overlay,
    _resolve_target,
    _split,
    _structure,
    main,
    resolve_schedule,
)


class TestResolveTarget:
    def test_resolves_a_method(self):
        from salt.data.readers.reader import H5StructuredReader

        owner, attr, function = _resolve_target("salt.data.readers.reader.H5StructuredReader.read")
        assert owner is H5StructuredReader
        assert attr == "read"
        assert function is H5StructuredReader.read

    def test_resolves_a_module_function(self):
        owner, attr, _ = _resolve_target("salt.profiling._split")
        assert owner.__name__ == "salt.profiling"
        assert attr == "_split"

    def test_every_default_target_resolves(self):
        # a rename in the read path must break this test, not silently empty the profile
        for dotted in DEFAULT_DATASET_FUNCTIONS:
            _resolve_target(dotted)

    def test_unknown_attribute_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            _resolve_target("salt.profiling.no_such_function")

    def test_unimportable_raises(self):
        with pytest.raises(ValueError, match="cannot import"):
            _resolve_target("definitely_not_a_module.thing")

    def test_non_callable_raises(self):
        with pytest.raises(ValueError, match="not callable"):
            _resolve_target("salt.profiling.DEFAULT_DATASET_FUNCTIONS")


class TestStructure:
    def test_nested_shapes_and_dtypes(self):
        import numpy as np

        left = _structure({"inputs": {"jets": np.zeros((2, 3), dtype=np.float32)}})
        right = _structure({"inputs": {"jets": np.zeros((2, 3), dtype=np.float32)}})
        assert left == right
        assert "(2, 3)" in left["inputs"]["jets"]

    def test_shape_change_is_detected(self):
        import numpy as np

        a = _structure({"x": np.zeros((2, 3))})
        b = _structure({"x": np.zeros((2, 4))})
        assert a != b

    def test_missing_key_is_detected(self):
        import numpy as np

        assert _structure({"x": np.zeros(2)}) != _structure({"x": np.zeros(2), "y": np.zeros(2)})


class TestDatasetSummary:
    def test_folds_timings_into_functions_and_hot_lines(self):
        stats = SimpleNamespace(
            unit=1e-6,
            timings={
                ("reader.py", 471, "read"): [(489, 10, 900_000), (510, 10, 100_000)],
                ("features.py", 79, "process"): [(88, 10, 200_000)],
            },
        )
        summary = _dataset_summary(stats, ["a"], {"b": "why"}, batches=10, elapsed=2.0)
        assert [entry["function"] for entry in summary["functions"]] == ["read", "process"]
        assert summary["functions"][0]["total_s"] == pytest.approx(1.0)
        assert summary["functions"][0]["per_batch_ms"] == pytest.approx(100.0)
        assert summary["hot_lines"][0]["line"] == 489
        assert summary["s_per_batch"] == pytest.approx(0.2)
        assert summary["skipped_functions"] == {"b": "why"}
        json.dumps(summary)  # must be serialisable


class TestTorchProfilerCallback:
    def test_rejects_an_empty_window(self, tmp_path):
        with pytest.raises(ValueError, match="active >= 1"):
            TorchProfilerCallback(dirpath=tmp_path, active=0)

    def test_total_steps_is_the_schedule_length(self, tmp_path):
        cb = TorchProfilerCallback(dirpath=tmp_path, wait=2, warmup=3, active=4)
        assert cb.total_steps == 9

    def test_hooks_are_inert_before_fit_start(self, tmp_path):
        cb = TorchProfilerCallback(dirpath=tmp_path)
        cb.on_train_batch_start(None, None, None, 0)
        cb.on_before_backward(None, None, None)
        cb.on_after_backward(None, None)
        cb.on_train_batch_end(None, None, None, None, 0)
        cb.on_fit_end(None, None)
        assert cb._seen == 0


class TestCli:
    def test_split(self):
        assert _split("a, b ,,c") == ("a", "b", "c")
        assert _split(None) == ()

    def test_help_exits_zero(self, capsys):
        assert main(["--help"]) == 0
        out = capsys.readouterr().out
        assert "dataset" in out
        assert "model" in out
        assert "--steps" in out

    def test_no_args_exits_nonzero(self):
        assert main([]) == 1

    def test_unknown_subcommand_exits_nonzero(self, capsys):
        assert main(["notasubcommand"]) == 1
        assert "unknown subcommand" in capsys.readouterr().err

    def test_both_subcommands_default_to_the_same_steps(self):
        from salt.profiling import _dataset_parser, _model_parser

        assert _model_parser().parse_args(["--config", "x.yaml"]).steps == DEFAULT_STEPS
        # the dataset parser defaults to None; the dispatcher applies DEFAULT_STEPS
        assert _dataset_parser().parse_args(["--config", "x.yaml"]).steps is None

    def test_model_reports_a_schedule_that_cannot_fit(self, capsys):
        rc = main(["model", "--config", "x.yaml", "--steps", "4", "--active", "10"])
        assert rc == 1
        assert "does not fit" in capsys.readouterr().err

    def test_model_rejects_a_malformed_set_override(self, capsys):
        rc = main(["model", "--config", "x.yaml", "--set", "nokey"])
        assert rc == 1
        assert "KEY=VALUE" in capsys.readouterr().err


class TestResolveSchedule:
    def test_default_steps_capture_the_tail(self):
        assert resolve_schedule(DEFAULT_STEPS) == {"wait": 75, "warmup": 5, "active": 20}

    @pytest.mark.parametrize("steps", [3, 4, 7, 10, 25, 30, 100, 1000])
    def test_the_window_always_fits_and_is_never_empty(self, steps):
        schedule = resolve_schedule(steps)
        assert sum(schedule.values()) <= steps
        assert all(value >= 1 for value in schedule.values())

    def test_the_whole_budget_is_used(self):
        # nothing is left running unprofiled after the window closes
        for steps in (3, 12, 40, 100):
            assert sum(resolve_schedule(steps).values()) == steps

    def test_explicit_phases_are_honoured(self):
        assert resolve_schedule(50, wait=1, warmup=2, active=3) == {
            "wait": 1,
            "warmup": 2,
            "active": 3,
        }

    def test_a_partially_explicit_schedule_fills_the_rest(self):
        schedule = resolve_schedule(40, active=10)
        assert schedule["active"] == 10
        assert sum(schedule.values()) == 40

    def test_too_few_steps_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            resolve_schedule(2)

    def test_an_oversized_explicit_schedule_raises(self):
        with pytest.raises(ValueError, match="does not fit"):
            resolve_schedule(10, wait=5, warmup=5, active=5)

    def test_a_non_positive_phase_raises(self):
        with pytest.raises(ValueError, match="must each be >= 1"):
            resolve_schedule(50, active=0)


class TestModelOverlay:
    def test_caps_the_fit_and_strips_the_training_furniture(self):
        import yaml

        overlay = yaml.safe_load(open(_model_overlay(37)))
        assert overlay["trainer"]["limit_train_batches"] == 37
        assert overlay["trainer"]["max_epochs"] == 1
        assert overlay["trainer"]["limit_val_batches"] == 0
        assert overlay["trainer"]["logger"] is False
        assert overlay["trainer"]["enable_checkpointing"] is False
        # deleting these dict entries is how a salt config drops a callback
        assert overlay["callbacks"] == {"checkpoint": None, "lr_monitor": None, "artifacts": None}
