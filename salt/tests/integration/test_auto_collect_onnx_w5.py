"""W5.1 integration gate (c) — ONNX auto-collect == /tmp/w4_oracle golden == explicit.

Plan 31 W5.1 gate (c): on a weight-matched GN2v2 export, the folded ONNX path
driven by an AUTO-COLLECT `OnnxExportSink` (omitted ``outputs:``) produces the
SAME Athena contract — output names (ORDERED), per-output dtypes, dynamic axes,
tuple length — as BOTH the W4 golden (``/tmp/w4_oracle/gn2v2.json``) AND the
EXPLICIT `OnnxExportSink` form (the W4 wiring). This proves auto-collect
reproduces the contract exactly, ordered per the Athena tuple rule
(globals -> combines -> per-token aux), without re-listing the leaves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    compile_onnx_plan,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.core.outputs import (
    ClassProbs,
    OnnxExportLeaf,
    OnnxExportSink,
    SeqClassIndex,
    VertexUnionFind,
)
from salt.tests._fixtures.gn2_fixture import JET_VARIABLES, TRACK_VARIABLES, build_test_gn2
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules

W4_ORACLE = Path("/tmp/w4_oracle")
VARIABLES = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}

pytestmark = pytest.mark.cpu_always


def _export_cfg() -> ExportConfig:
    return ExportConfig(
        model_name="GN2v2",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
        ],
    )


def _producers(tmp_path):
    """The folded GN2v2 ONNX conversion producers (without the export sink)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    v1 = build_test_gn2(tmp_path)
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    jp = ClassProbs(task="jets_classification", stream="jets"); jp.name = "jet_probs"
    ti = SeqClassIndex(task="track_origin", stream="tracks"); ti.name = "track_origin_index"
    vi = VertexUnionFind(task="track_vertexing", stream="tracks"); vi.name = "track_vertex_index"
    modules.update({"jet_probs": jp, "track_origin_index": ti, "track_vertex_index": vi})
    return v1, modules


def _export(tmp_path, sink, fname):
    v1, modules = _producers(tmp_path)
    sink.name = "onnx_export"
    modules["onnx_export"] = sink
    resolved = resolve_export_config(_export_cfg(), "GN2_v2")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    nn.ModuleDict({k: v for k, v in modules.items() if isinstance(v, nn.Module)}).load_state_dict(
        map_v1_state_dict(v1.state_dict(), modules), strict=False
    )
    return export_graph(modules, _export_cfg(), VARIABLES, tmp_path / fname, outputs=[],
                        run_name="GN2_v2")


def _explicit_sink() -> OnnxExportSink:
    return OnnxExportSink(outputs=[
        OnnxExportLeaf(key="outputs.jets.jets_classification", names=["pb", "pc", "pu"]),
        OnnxExportLeaf(key="outputs.tracks.track_origin", name="TrackOrigin", dtype="int8",
                       per_token=True),
        OnnxExportLeaf(key="outputs.tracks.track_vertexing", name="VertexIndex", dtype="int8",
                       per_token=True),
    ])


@pytest.mark.skipif(
    not (W4_ORACLE / "gn2v2.json").is_file(),
    reason="W4 oracle golden /tmp/w4_oracle/gn2v2.json not present",
)
def test_onnx_auto_collect_contract_matches_golden_and_explicit(tmp_path):
    """Auto-collect ONNX contract == /tmp/w4_oracle golden == explicit (names/dtypes/axes/order)."""
    golden = json.loads((W4_ORACLE / "gn2v2.json").read_text())

    auto = _export(tmp_path / "auto", OnnxExportSink(), "auto.onnx").adapter
    explicit = _export(tmp_path / "explicit", _explicit_sink(), "explicit.onnx").adapter

    # 1) auto-collect == the W4 golden contract (ORDERED list-equality)
    assert auto.output_names == golden["output_names"]
    assert auto.output_dtypes == golden["output_dtypes"]
    assert json.loads(json.dumps(auto.dynamic_axes)) == golden["dynamic_axes"]
    assert len(auto.output_names) == golden["output_tuple_len"]

    # 2) auto-collect == the explicit form (the W4 wiring), field-for-field
    assert auto.output_names == explicit.output_names
    assert auto.output_dtypes == explicit.output_dtypes
    assert auto.dynamic_axes == explicit.dynamic_axes

    # 3) the exported ONNX session outputs match the golden names, in order
    session = make_session(_export(tmp_path / "auto2", OnnxExportSink(), "auto2.onnx").onnx_path)
    assert [o.name for o in session.get_outputs()] == golden["output_names"]


@pytest.mark.skipif(
    not (W4_ORACLE / "gn2v2.json").is_file(),
    reason="W4 oracle golden /tmp/w4_oracle/gn2v2.json not present",
)
def test_onnx_auto_collect_runs_at_L0(tmp_path):
    """The auto-collect export runs (onnxruntime) including the L=0 zero-token jet."""
    result = _export(tmp_path, OnnxExportSink(), "auto.onnx")
    adapter = result.adapter
    for length in (0, 5):
        example = adapter.example_inputs(sequence_length=length)
        with torch.no_grad():
            out = adapter(*example)
        assert len(out) == 5  # pb, pc, pu, TrackOrigin, VertexIndex
