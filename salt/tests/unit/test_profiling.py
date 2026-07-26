"""Tests for salt.profiling (plan 05: the dataset line profiler + model torch.profiler)."""

import json
from types import SimpleNamespace

import pytest

from salt.profiling import (
    DEFAULT_DATASET_FUNCTIONS,
    TorchProfilerCallback,
    _dataset_summary,
    _resolve_target,
    _split,
    _structure,
    main,
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
        assert "salt profile dataset" in capsys.readouterr().out

    def test_no_args_exits_nonzero(self):
        assert main([]) == 1

    def test_unknown_subcommand_exits_nonzero(self, capsys):
        assert main(["model"]) == 1
        assert "unknown subcommand" in capsys.readouterr().err
