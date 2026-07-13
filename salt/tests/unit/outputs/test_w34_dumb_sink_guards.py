"""PLAN 34 W34.2 dumb-sink guard gates — dup-name, zero-output, no-double-split naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.core.graph.errors import ConfigError
from salt.core.outputs import H5OutputSink, OnnxExportSink
from salt.core.outputs.run_task_output import RunTaskOutput
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules

pytestmark = pytest.mark.cpu_always


def _modules(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    nd = tmp_path / "norm_dict.yaml"
    cd = tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return build_gn2v2_modules(nd)


def _bound_run_task(tmp_path, tasks):
    modules = _modules(tmp_path)
    rt = RunTaskOutput(tasks=tasks)
    rt.name = "run_tasks"
    rt.bind_model_modules(modules)
    return rt


class TestDumbH5SinkSectionResolution:
    def test_dumb_h5_resolves_columns_from_section(self, tmp_path):
        """The dumb H5 sink resolves per-class columns from the section RunTaskOutput."""
        rt = _bound_run_task(tmp_path, ["jets_classification", "track_origin"])
        sink = H5OutputSink()  # no init args — dumb
        sink.bind_output_section({"run_tasks": rt})
        cols = sink._resolve_columns("GN2v2")  # noqa: SLF001 - direct guard check
        keys = [c.key for c in cols]
        # one per-field leaf per class column, in task-then-field order
        assert "outputs.jets.jets_classification.pb" in keys
        assert "outputs.tracks.track_origin.pPrimary" in keys
        # each per-field leaf is a single-suffix column
        for c in cols:
            assert len(list(c.suffixes)) == 1

    def test_dumb_h5_zero_output_hard_fails(self, tmp_path):
        """A bound section with no RunTaskOutput is a zero-output hard fail."""
        sink = H5OutputSink()
        # an empty section (no RunTaskOutput) -> no final H5 column
        from salt.core.outputs.input_copy_writer import InputCopyWriter

        icw = InputCopyWriter(streams=["jets"])
        icw.name = "inputs_copy"
        sink.bind_output_section({"inputs_copy": icw})
        with pytest.raises(ConfigError, match="no RunTaskOutput task with a final H5 column"):
            sink._resolve_columns("GN2v2")  # noqa: SLF001 - direct guard check

    def test_dumb_h5_dup_column_hard_fails(self, tmp_path):
        """Two RunTaskOutputs minting the same flat H5 column is a ConfigError."""
        # two RunTaskOutputs both serialising jets_classification -> the SAME
        # outputs.jets.jets_classification.pb column appears twice
        rt1 = _bound_run_task(tmp_path / "a", ["jets_classification"])
        rt1.name = "run_a"
        rt2 = _bound_run_task(tmp_path / "b", ["jets_classification"])
        rt2.name = "run_b"
        sink = H5OutputSink()
        sink.bind_output_section({"run_a": rt1, "run_b": rt2})
        with pytest.raises(ConfigError, match="minted by BOTH"):
            sink._resolve_columns("GN2v2")  # noqa: SLF001 - direct guard check


class TestDumbOnnxSinkNoDoubleSplit:
    def test_onnx_global_scalars_are_single_name_not_split(self, tmp_path):
        """The dumb ONNX sink names each global per-class scalar via a SINGLE `name` (no split)."""
        rt = _bound_run_task(tmp_path, ["jets_classification", "track_origin"])
        sink = OnnxExportSink(model_name="GN2v2")
        sink.bind_output_section({"run_tasks": rt})
        leaves = sink._resolve_section_leaves()  # noqa: SLF001 - direct guard check
        # every leaf is single-name (the no-double-split invariant): no leaf carries
        # a plural `names` split, because get_output already split per-class.
        for leaf in leaves:
            assert leaf.names is None, (
                f"leaf {leaf.key!r} carries a plural-names split — the dumb ONNX sink must NAME "
                "the already-scalar get_output values, NOT re-split (plan §4 W34.2 LOCKED)"
            )
            assert leaf.name is not None
        # the global pb/pc/pu scalars precede the per-token TrackOrigin argmax (the
        # canonical Athena tuple order: globals -> per-token aux)
        names = [leaf.name for leaf in leaves]
        assert names.index("pb") < names.index("TrackOrigin")
        assert names.index("pu") < names.index("TrackOrigin")

    def test_onnx_output_names_prefixed_and_ordered(self, tmp_path):
        """The flat ONNX output names are {model_name}_{suffix} in globals->per-token order."""
        rt = _bound_run_task(tmp_path, ["jets_classification", "track_origin"])
        sink = OnnxExportSink(model_name="GN2v2")
        sink.bind_output_section({"run_tasks": rt})
        assert sink.output_names() == ["GN2v2_pb", "GN2v2_pc", "GN2v2_pu", "GN2v2_TrackOrigin"]
        assert sink.output_dtypes() == ["float32", "float32", "float32", "int8"]

    def test_onnx_zero_output_hard_fails(self, tmp_path):
        """A bound ONNX section with no ONNX leaf is a hard fail."""
        sink = OnnxExportSink(model_name="GN2v2")
        sink.bind_output_section({})  # empty section
        with pytest.raises(ConfigError, match="no RunTaskOutput field with an ONNX leaf"):
            sink._resolve_section_leaves()  # noqa: SLF001 - direct guard check
