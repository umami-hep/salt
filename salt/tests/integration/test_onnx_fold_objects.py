"""End-to-end ONNX export gates for the MaskFormer object nodes."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from salt.model.modules import bind_all, resolve_bind_schema
from salt.outputs.sinks.onnx import (
    ExportConfig,
    ExportInput,
    compile_onnx_plan,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.outputs import (
    MaskFormerObject,
    MaskFormerObjects,
    MFLeadVertexDecorator,
    OnnxExportSink,
)
from salt.graph.render import dot_source
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)
from salt.tests._fixtures.v2_builders import (
    MASKFORMER_WRITER_REG_TARGETS,
    build_maskformer_writer_modules,
)

VARIABLES = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}

# These are pure-CPU ONNX-trace gates (the MaskFormer object-node export
# contract + traces). They are NOT GPU/heavy despite living under
# tests/integration/ — the `cpu_always` marker (conftest.py) opts them OUT of
# the GPU skip so they run on EVERY CI invocation, with or without
# --run-integration. Skipping them would let an object-node export regression
# ship unnoticed.
pytestmark = pytest.mark.cpu_always


def _mf_export_cfg() -> ExportConfig:
    return ExportConfig(
        model_name="MaskFormer",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
        ],
    )





_LEADING_NAMES = [f"leading_objects_{t}" for t in MASKFORMER_WRITER_REG_TARGETS]
_N_REG = len(MASKFORMER_WRITER_REG_TARGETS)


def _bind_producers(modules) -> None:
    """Bind the model modules to every producer/sink that resolves names from them."""
    for module in modules.values():
        if callable(getattr(module, "bind_model_modules", None)):
            module.bind_model_modules(modules)


# MaskFormerObject: ONE node mints BOTH object leaves (leading_object +
# object_index). Self-consistency (torch == ort) is NaN-aware because the
# random decoder weights yield a NaN leading_object by design (null-suppression
# semantics).


def _build_maskformer_folded(tmp_path):
    """Folded single-node maskformer export (MaskFormerObject -> both leaves)."""
    torch.manual_seed(42)
    modules = build_maskformer_writer_modules(tmp_path / "norm_dict.yaml")
    mf = MaskFormerObject(
        n_reg=_N_REG,
        stream="objects",
        constituent_stream="tracks",
        leading_name="leading_object",
        index_name="HadronIndex",
    )
    mf.name = "mf_obj"
    sink = OnnxExportSink()
    sink.name = "onnx_export"
    modules.update({"mf_obj": mf, "onnx_export": sink})
    # the node names its own leaves: leading_objects_<target> off the regression
    # task's targets, and the int8 per-token HadronIndex off `index_name`.
    _bind_producers(modules)
    resolved = resolve_export_config(_mf_export_cfg(), "MaskFormer")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    modules["norm"].materialise()
    torch.manual_seed(42)
    return export_graph(
        modules, _mf_export_cfg(), VARIABLES, tmp_path / "folded_mf.onnx",
        outputs=[], run_name="MaskFormer",
    )


@pytest.fixture(scope="module")
def maskformer(tmp_path_factory):
    """The folded one-node maskformer export."""
    tmp = tmp_path_factory.mktemp("maskformer_fold")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    return SimpleNamespace(folded=_build_maskformer_folded(tmp))


def test_maskformer_folded_export_contract(maskformer):
    """ONE node mints BOTH leaves: the leading split + the per-token int8 index axis."""
    adapter = maskformer.folded.adapter
    assert adapter.output_names == [f"MaskFormer_{name}" for name in _LEADING_NAMES] + [
        "MaskFormer_HadronIndex",
    ]
    assert adapter.output_dtypes == ["float32", "float32", "float32", "int8"]
    assert adapter.dynamic_axes["MaskFormer_HadronIndex"] == {0: "n_tracks"}


def test_maskformer_folded_torch_vs_ort_nan_aware(maskformer):
    """torch == onnxruntime, NaN-aware, incl L=0 — the folded node traces correctly."""
    adapter = maskformer.folded.adapter
    session = make_session(maskformer.folded.onnx_path)
    names = adapter.output_names
    gen = torch.Generator().manual_seed(9)
    for length in (0, 1, 2, 7, 13):
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen)
        tracks = torch.rand(length, len(TRACK_VARIABLES), generator=gen)
        feed = {"jet_features": jets.numpy(), "track_features": tracks.numpy()}
        with torch.no_grad():
            torch_out = dict(zip(names, (o.cpu().numpy() for o in adapter(jets, tracks)), strict=True))
        ort_out = dict(zip(names, session.run(None, feed), strict=True))
        np.testing.assert_array_equal(
            torch_out["MaskFormer_HadronIndex"], ort_out["MaskFormer_HadronIndex"],
            err_msg=f"HadronIndex mismatch at L={length}",
        )
        for name in names:
            if name == "MaskFormer_HadronIndex":
                continue
            np.testing.assert_allclose(
                torch_out[name], ort_out[name], atol=1e-4, equal_nan=True, err_msg=f"{name} at L={length}"
            )


# The TWO-NODE MaskFormer split (design decision 2026-06-22): the MaskFormerObjects
# reconstruction node exposes the per-vertex leaves; the MFLeadVertexDecorator
# reads them and emits jet-level scalars. End-to-end ONNX export (no oracle —
# the decorator is a NEW capability, unit-tested separately in tests/unit).


def _build_two_node_mf(tmp_path):
    """Export MaskFormerObjects + MFLeadVertexDecorator + OnnxExportSink (the two-node chain)."""
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    torch.manual_seed(42)
    modules = build_maskformer_writer_modules(tmp_path / "norm_dict.yaml")
    mf = MaskFormerObjects(
        n_reg=_N_REG, stream="objects", constituent_stream="tracks", index_name="HadronIndex"
    )
    mf.name = "mf_obj"
    dec = MFLeadVertexDecorator(
        source="outputs.objects.vertices_class_probs",
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 2},
        pt_index=0,
        pv_class_index=0,
        pnull_threshold=0.5,
    )
    dec.name = "lead_vertex"
    # narrow the auto-collected tuple to the index + the decorator scalars (the
    # reconstruction node also declares its leading_objects_* globals)
    sink = OnnxExportSink(consumes=["outputs.tracks.HadronIndex", "outputs.jet.*"])
    sink.name = "onnx_export"
    modules.update({"mf_obj": mf, "lead_vertex": dec, "onnx_export": sink})
    _bind_producers(modules)
    resolved = resolve_export_config(_mf_export_cfg(), "MaskFormer")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    modules["norm"].materialise()
    torch.manual_seed(42)
    return export_graph(
        modules,
        _mf_export_cfg(),
        VARIABLES,
        tmp_path / "two_node_mf.onnx",
        outputs=[],
        run_name="MaskFormer",
    )


@pytest.fixture(scope="module")
def two_node_mf(tmp_path_factory):
    """The two-node MaskFormer export (reconstruction node + lead-vertex decorator)."""
    return _build_two_node_mf(tmp_path_factory.mktemp("two_node_mf"))


def test_two_node_mf_export_contract(two_node_mf):
    """The two-node chain wires object_index (per-token int8) + the jet-level decorator scalars."""
    adapter = two_node_mf.adapter
    # both nodes are model modules, so they share one manifest group and the
    # tuple is that group's globals then its per-token leaves — the decorator's
    # two jet-level scalars, then the reconstruction node's index.
    assert adapter.output_names == [
        "MaskFormer_lead_vertex_pt",
        "MaskFormer_lead_vertex_mass",
        "MaskFormer_HadronIndex",
    ]
    assert adapter.output_dtypes == ["float32", "float32", "int8"]
    # only the per-token index carries a dynamic axis; the jet-level scalars are global
    assert adapter.dynamic_axes["MaskFormer_HadronIndex"] == {0: "n_tracks"}
    assert "MaskFormer_lead_vertex_pt" not in adapter.dynamic_axes
    assert "MaskFormer_lead_vertex_mass" not in adapter.dynamic_axes


def test_two_node_mf_traces_and_runs_including_zero_tokens(two_node_mf):
    """The exported two-node chain runs in onnxruntime across L=0..N (the decorator is trace-safe)."""
    session = make_session(two_node_mf.onnx_path)
    names = two_node_mf.adapter.output_names
    gen = torch.Generator().manual_seed(11)
    for length in (0, 1, 2, 7, 13):
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen).numpy()
        tracks = torch.rand(length, len(TRACK_VARIABLES), generator=gen).numpy()
        run = session.run(None, {"jet_features": jets, "track_features": tracks})
        out = dict(zip(names, run, strict=True))
        assert out["MaskFormer_HadronIndex"].shape == (length,)
        for jet_out in ("MaskFormer_lead_vertex_pt", "MaskFormer_lead_vertex_mass"):
            value = np.asarray(out[jet_out]).reshape(-1)
            assert value.shape == (1,)
            # a well-defined scalar — finite (a qualifying vertex) OR NaN (none qualifies)
            assert np.isfinite(value).all() or np.isnan(value).all()


class _DecoratorOnnxWrapper(nn.Module):
    """Wrap MFLeadVertexDecorator.forward as a 2-tensor-in / N-scalar-out module for export."""

    def __init__(self, decorator: MFLeadVertexDecorator) -> None:
        super().__init__()
        self.decorator = decorator

    def forward(self, class_probs: torch.Tensor, regression: torch.Tensor):
        from salt.graph.bundle import Bundle  # noqa: PLC0415 - test-local
        from salt.graph.spec import Mode  # noqa: PLC0415 - test-local

        b = Bundle()
        b.set(self.decorator.source, class_probs)
        b.set(self.decorator.regression_source, regression)
        out = self.decorator.forward(b, Mode.ONNX)
        return tuple(out[key] for key in self.decorator.output_keys)


def test_lead_vertex_decorator_onnx_reproduces_non_nan_selection(tmp_path):
    """The decorator's NON-NaN selection survives ONNX export."""
    dec = MFLeadVertexDecorator(
        source="outputs.objects.vertices_class_probs",
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 2},
        pt_index=0,
        pv_class_index=0,
        pnull_threshold=0.5,
    )
    dec.name = "lead_vertex"
    wrapper = _DecoratorOnnxWrapper(dec).eval()

    # trace inputs (2 jets, 3 vertices, 3 classes [pv, sv, null], 3 reg [pt, Lxy, mass]):
    #   jet 0: v0 PV (excluded), v1 sv pt 4, v2 sv pt 7 -> lead = v2 (pt 7, mass 70)
    #   jet 1: all PV -> NaN
    class_probs = torch.tensor([
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.7, 0.1]],
        [[0.9, 0.05, 0.05], [0.85, 0.1, 0.05], [0.8, 0.15, 0.05]],
    ])
    regression = torch.tensor([
        [[99.0, 1.0, 100.0], [4.0, 2.0, 40.0], [7.0, 3.0, 70.0]],
        [[10.0, 0.0, 100.0], [20.0, 0.0, 200.0], [30.0, 0.0, 300.0]],
    ])
    onnx_path = tmp_path / "decorator.onnx"
    torch.onnx.export(
        wrapper,
        (class_probs, regression),
        str(onnx_path),
        input_names=["class_probs", "regression"],
        output_names=["lead_vertex_pt", "lead_vertex_mass"],
        dynamic_axes={
            "class_probs": {0: "B", 1: "M"},
            "regression": {0: "B", 1: "M"},
        },
        opset_version=16,
    )
    session = make_session(onnx_path)
    run = session.run(None, {"class_probs": class_probs.numpy(), "regression": regression.numpy()})
    pt, mass = (np.asarray(r).reshape(-1) for r in run)
    # jet 0: v2 selected (highest-pT non-PV non-null) — ONNX reproduces the eager pick
    assert pt[0] == pytest.approx(7.0)
    assert mass[0] == pytest.approx(70.0)
    # jet 1: all-PV -> NaN through ORT
    assert np.isnan(pt[1])
    assert np.isnan(mass[1])
    # cross-check eager agrees with the exported ORT result
    eager = wrapper(class_probs, regression)
    np.testing.assert_allclose(np.asarray(eager[0]).reshape(-1), pt, equal_nan=True, atol=1e-6)
    np.testing.assert_allclose(np.asarray(eager[1]).reshape(-1), mass, equal_nan=True, atol=1e-6)


def test_two_node_mf_render_card_and_node_to_node_edge(tmp_path):
    """Render: the MaskFormer object nodes render the onnx_export card + node cards."""
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    torch.manual_seed(42)
    modules = build_maskformer_writer_modules(tmp_path / "norm_dict.yaml")
    mf = MaskFormerObjects(
        n_reg=_N_REG, stream="objects", constituent_stream="tracks", index_name="HadronIndex"
    )
    mf.name = "maskformer_objects"
    dec = MFLeadVertexDecorator(
        source="outputs.objects.vertices_class_probs",
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 2},
        pt_index=0,
        pv_class_index=0,
    )
    dec.name = "mf_lead_vertex"
    sink = OnnxExportSink(consumes=["outputs.tracks.HadronIndex", "outputs.jet.*"])
    sink.name = "onnx_export"
    modules.update({"maskformer_objects": mf, "mf_lead_vertex": dec, "onnx_export": sink})
    _bind_producers(modules)
    resolved = resolve_export_config(_mf_export_cfg(), "GN3Mask")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    dot = dot_source(plan, modules)
    assert "<sinks>" not in dot  # the off-graph manifest sentinel is gone
    assert "onnx_export" in dot
    assert "OnnxExportSink" in dot
    assert "maskformer_objects" in plan.module_names
    assert "mf_lead_vertex" in plan.module_names
    # named-consumer edges flow into the NAMED export node (both object nodes feed it)
    sink_edges = set(re.findall(r'"(\w+)" -> "onnx_export"', dot))
    assert {"maskformer_objects", "mf_lead_vertex"} <= sink_edges
    # the node->node edge: the decorator reads the reconstruction node's vertices leaf
    assert r'"maskformer_objects" -> "mf_lead_vertex"' in dot


def test_two_node_mf_object_index_equals_single_node_fold(two_node_mf, maskformer):
    """The two-node object_index is bitwise-equal to the single-node fold (same reconstruction)."""
    s_two = make_session(two_node_mf.onnx_path)
    s_single = make_session(maskformer.folded.onnx_path)
    two_names = two_node_mf.adapter.output_names
    single_names = maskformer.folded.adapter.output_names
    gen = torch.Generator().manual_seed(19)
    for length in (0, 1, 2, 7, 13):
        jets = torch.rand(1, len(JET_VARIABLES), generator=gen).numpy()
        tracks = torch.rand(length, len(TRACK_VARIABLES), generator=gen).numpy()
        feed = {"jet_features": jets, "track_features": tracks}
        two = dict(zip(two_names, s_two.run(None, feed), strict=True))
        single = dict(zip(single_names, s_single.run(None, feed), strict=True))
        np.testing.assert_array_equal(
            two["MaskFormer_HadronIndex"],
            single["MaskFormer_HadronIndex"],
            err_msg=f"object_index drift at L={length}",
        )
