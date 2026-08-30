"""Gates for the public sink bases + the `JSONLOutputSink` worked example."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.outputs import (
    H5OutputSink,
    JSONLOutputSink,
    Node,
    OnnxExportSink,
    OutputSink,
    RuntimeSink,
    SinkContext,
    is_test_persistence_sink,
)
from salt.outputs.sinks.h5_sink import _SinkCallback  # noqa: PLC2701 - the alias under test
from salt.outputs.input_copy_writer import InputCopyWriter
from salt.outputs.run_task_output import RunTaskOutput
from salt.tests._fixtures.gn2v2_fixture import (  # noqa: PLC2701 - shared test fixtures
    build_gn2v2_modules,
    write_parity_norm_dict,
)

pytestmark = pytest.mark.cpu_always

_RUN_NAME = "GN2v2"
# tasks default to write_targets: true, so the section also mints this label
# column. It is NOT run-name prefixed (labels are model-independent).
_TARGET = "target_jets_classification"


def _bound_run_task(tmp_path: Path, tasks: list[str]) -> RunTaskOutput:
    """A `RunTaskOutput` bound to the GN2v2 fixture module dict."""  # noqa: DOC201 - test helper
    tmp_path.mkdir(parents=True, exist_ok=True)
    nd = tmp_path / "norm_dict.yaml"
    cd = tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    rt = RunTaskOutput(tasks=tasks)
    rt.name = "run_tasks"
    rt.bind_model_modules(build_gn2v2_modules(nd))
    return rt


def _section(tmp_path: Path, tasks: list[str] | None = None) -> dict:
    """A minimal outputs: section (one RunTaskOutput + one InputCopyWriter)."""  # noqa: DOC201 - test helper
    icw = InputCopyWriter(streams=["jets"])
    icw.name = "inputs_copy"
    return {
        "inputs_copy": icw,
        "run_tasks": _bound_run_task(tmp_path, tasks or ["jets_classification"]),
    }


def _ctx(tmp_path: Path, src: str = "pp_output_test_ttbar.h5") -> SinkContext:
    """The open-time context carrying the three things a sink reads."""  # noqa: DOC201 - test helper
    ckpts = tmp_path / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)
    return SinkContext(
        run_name=_RUN_NAME,
        datamodule=SimpleNamespace(test_dset=SimpleNamespace(reader=SimpleNamespace(filename=src))),
        ckpt_path=str(ckpts / "epoch=009-val_loss=0.64.ckpt"),
    )


def _bundle(probs: torch.Tensor, start: int = 0) -> Bundle:
    """A bundle carrying every leaf the section mints for jets_classification.

    That is one leaf per class PLUS the ``target_{task}`` label leaf (tasks
    ship ``write_targets: true`` by default), and the ``meta.rows`` anchor.
    """  # noqa: DOC201 - test helper
    leaves = {f"p{name}": probs[:, idx] for idx, name in enumerate(("b", "c", "u"))}
    leaves[_TARGET] = torch.zeros(probs.shape[0], dtype=torch.int64)
    return Bundle({
        "outputs": {"jets": {"jets_classification": leaves}},
        "meta": {"rows": torch.tensor([start, start + probs.shape[0]])},
    })


def _read(path: Path) -> list[dict]:
    """Parse a JSONL file back into a list of records (strict JSON per line)."""  # noqa: DOC201 - test helper
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# (1) the base-class promotion --------------------------------------------------


class TestOutputSinkPromotion:
    def test_the_private_alias_still_resolves(self):
        """`_SinkCallback` is kept as an alias of the deprecated public `OutputSink`."""
        assert _SinkCallback is OutputSink

    @pytest.mark.parametrize("cls", [H5OutputSink, OnnxExportSink, JSONLOutputSink])
    def test_shipped_sinks_subclass_the_public_base(self, cls):
        """Every shipped sink derives from the documented extension point."""
        assert issubclass(cls, Node)

    @pytest.mark.parametrize("cls", [H5OutputSink, JSONLOutputSink])
    def test_sinks_with_a_lifecycle_subclass_runtime_sink(self, cls):
        """A sink the test loop drives per batch carries the lifecycle base."""
        assert issubclass(cls, RuntimeSink)

    def test_the_onnx_manifest_is_declare_only(self):
        """Export runs no test loop, so its node has no lifecycle to inherit."""
        assert not issubclass(OnnxExportSink, RuntimeSink)

    def test_output_sink_still_works_but_deprecates_on_subclassing(self):
        """Third-party `class MySink(OutputSink)` keeps working, loudly."""
        assert issubclass(OutputSink, RuntimeSink)
        with pytest.warns(DeprecationWarning, match="deprecated alias"):

            class _ThirdParty(OutputSink):
                name = "third_party"

        assert issubclass(_ThirdParty, RuntimeSink)

    def test_base_marks_terminal_and_defaults_to_empty_io(self):
        """The base is a terminal node declaring nothing until a subclass overrides."""
        sink = RuntimeSink()
        assert sink.is_sink() is True
        assert flatten_spec(sink.declare_io(Mode.TEST).requires) == {}
        assert sink.is_test_sink() is False  # empty TEST requires

    def test_shared_selector_picks_only_the_primary_test_sink(self):
        """`is_test_persistence_sink` is the one selector runtime + graph tooling share."""
        assert is_test_persistence_sink(H5OutputSink()) is True
        assert is_test_persistence_sink(OnnxExportSink()) is False
        assert is_test_persistence_sink(JSONLOutputSink()) is False
        # a duck-typed sink WITHOUT is_test_sink still counts as the primary
        assert is_test_persistence_sink(SimpleNamespace(writer_demand=lambda *_a, **_k: {})) is True


# (2) schema resolution — the JSONL sink reads the SAME section as the H5 sink --


class TestJSONLSectionSchema:
    def test_columns_match_the_h5_sink_exactly(self, tmp_path):
        """Both sinks derive from the section, so their column names cannot drift."""
        section = _section(tmp_path)
        h5 = H5OutputSink()
        h5.bind_output_section(section)
        jsonl = JSONLOutputSink()
        jsonl.bind_output_section(section)
        jsonl._run_name = _RUN_NAME  # noqa: SLF001 - direct schema check
        h5_names = [n for c in h5._resolve_columns(_RUN_NAME) for n in c.column_names(_RUN_NAME)]  # noqa: SLF001
        jsonl_names = [
            n
            for c in jsonl._ensure_columns()  # noqa: SLF001 - direct schema check
            for n in c.column_names(_RUN_NAME)
        ]
        assert jsonl_names == h5_names
        assert f"{_RUN_NAME}_pb" in jsonl_names

    def test_unbound_section_is_a_config_error(self):
        """A sink with no section bound cannot know its columns."""
        with pytest.raises(ConfigError, match="no `outputs:` section bound"):
            JSONLOutputSink()._ensure_columns()  # noqa: SLF001 - direct guard check

    def test_unknown_column_selection_fails_loudly(self, tmp_path):
        """A typo in `columns:` names what the section does not mint."""
        sink = JSONLOutputSink(columns=["GN2v2_pb", "GN2v2_pnonsense"])
        sink.bind_output_section(_section(tmp_path))
        with pytest.raises(ConfigError, match="pnonsense"):
            sink.open_schema(_ctx(tmp_path))


# (3) the graph-node surface — declare_io drives writer_demand -------------------


class TestJSONLGraphSurface:
    def test_declare_io_requires_the_selected_leaves_plus_meta_rows(self, tmp_path):
        """TEST requires exactly the section leaves it writes, plus the row anchor."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        req = flatten_spec(sink.declare_io(Mode.TEST).requires)
        assert "meta.rows" in req
        assert "outputs.jets.jets_classification.pb" in req
        assert flatten_spec(sink.declare_io(Mode.TEST).produces) == {}

    @pytest.mark.parametrize("mode", [Mode.FIT, Mode.VAL, Mode.ONNX])
    def test_non_test_modes_declare_nothing(self, tmp_path, mode):
        """Outside TEST the sink is inert, so the planner prunes it."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        assert flatten_spec(sink.declare_io(mode).requires) == {}

    def test_writer_demand_mirrors_declare_io(self, tmp_path):
        """The demand is GENERATED from declare_io — the two can never disagree."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        demand = sink.writer_demand({}, None)
        assert set(demand) == set(flatten_spec(sink.declare_io(Mode.TEST).requires))
        assert all("JSONLOutputSink" in who for who in demand.values())

    def test_demand_is_a_subset_of_the_primary_sink_demand(self, tmp_path):
        """An auxiliary sink never widens the plan — it rides the H5 sink's demand."""
        section = _section(tmp_path, ["jets_classification", "track_origin"])
        h5 = H5OutputSink()
        h5.bind_output_section(section)
        jsonl = JSONLOutputSink(columns=["pb"])
        jsonl.bind_output_section(section)
        h5_req = set(flatten_spec(h5.declare_io(Mode.TEST).requires))
        jsonl_req = set(flatten_spec(jsonl.declare_io(Mode.TEST).requires))
        assert jsonl_req <= h5_req


# (4) the round trip ------------------------------------------------------------


class TestJSONLRoundTrip:
    def test_round_trips_a_small_batch(self, tmp_path):
        """open_schema -> consume -> flush writes one strict-JSON object per row."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        ctx = _ctx(tmp_path)
        sink.open_schema(ctx)
        probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        sink.consume(_bundle(probs))
        sink.flush()
        records = _read(sink.output_path)
        assert len(records) == 2
        assert set(records[0]) == {
            f"{_RUN_NAME}_pb",
            f"{_RUN_NAME}_pc",
            f"{_RUN_NAME}_pu",
            _TARGET,  # un-prefixed: the label column is model-independent
        }
        assert records[0][f"{_RUN_NAME}_pb"] == pytest.approx(0.7, abs=1e-6)
        assert records[1][f"{_RUN_NAME}_pu"] == pytest.approx(0.6, abs=1e-6)

    def test_output_path_is_deterministic_and_beside_the_eval_h5(self, tmp_path):
        """The template renders the eval-H5 name with a .jsonl suffix."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        sink.open_schema(_ctx(tmp_path))
        sink.flush()
        assert sink.output_path.name == "epoch=009-val_loss=0.64__test_ttbar.jsonl"
        assert sink.output_path.parent == tmp_path / "ckpts"

    def test_multiple_batches_append_in_row_order(self, tmp_path):
        """Consecutive batches append rather than truncate."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        sink.open_schema(_ctx(tmp_path))
        sink.consume(_bundle(torch.tensor([[1.0, 0.0, 0.0]]), start=0))
        sink.consume(_bundle(torch.tensor([[0.0, 1.0, 0.0]]), start=1))
        sink.flush()
        records = _read(sink.output_path)
        assert [r[f"{_RUN_NAME}_pb"] for r in records] == [1.0, 0.0]

    def test_column_filter_narrows_the_record(self, tmp_path):
        """`columns:` keeps only the named columns (flat name or bare suffix)."""
        sink = JSONLOutputSink(columns=[f"{_RUN_NAME}_pb", "pu"])
        sink.bind_output_section(_section(tmp_path))
        sink.open_schema(_ctx(tmp_path))
        sink.consume(_bundle(torch.tensor([[0.7, 0.2, 0.1]])))
        sink.flush()
        assert set(_read(sink.output_path)[0]) == {f"{_RUN_NAME}_pb", f"{_RUN_NAME}_pu"}

    def test_non_finite_values_become_json_null(self, tmp_path):
        """NaN/inf map to null — JSON has no NaN literal, and allow_nan=False guards it."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        sink.open_schema(_ctx(tmp_path))
        sink.consume(_bundle(torch.tensor([[float("nan"), float("inf"), 0.5]])))
        sink.flush()
        record = _read(sink.output_path)[0]
        assert record[f"{_RUN_NAME}_pb"] is None
        assert record[f"{_RUN_NAME}_pc"] is None
        assert record[f"{_RUN_NAME}_pu"] == pytest.approx(0.5, abs=1e-6)

    def test_overwrite_false_refuses_to_clobber(self, tmp_path):
        """A second run against an existing file fails instead of truncating."""
        section = _section(tmp_path)
        first = JSONLOutputSink()
        first.bind_output_section(section)
        first.open_schema(_ctx(tmp_path))
        first.flush()
        second = JSONLOutputSink(overwrite=False)
        second.bind_output_section(section)
        with pytest.raises(ConfigError, match="refuses to overwrite"):
            second.open_schema(_ctx(tmp_path))

    def test_missing_ckpt_path_is_a_config_error(self, tmp_path):
        """The output name is derived from the checkpoint, so it is required."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        ctx = replace(_ctx(tmp_path), ckpt_path=None)
        with pytest.raises(ConfigError, match=r"needs a checkpoint path"):
            sink.open_schema(ctx)

    def test_close_if_open_is_idempotent(self, tmp_path):
        """Cleanup after an interrupted test closes once and no-ops thereafter."""
        sink = JSONLOutputSink()
        sink.bind_output_section(_section(tmp_path))
        sink.open_schema(_ctx(tmp_path))
        sink.close_if_open()
        sink.close_if_open()
        sink.flush()  # also a no-op once closed
