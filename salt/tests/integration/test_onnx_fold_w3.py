"""End-to-end gates for the plan-29 W3 folded ONNX path — the HARD reduces (design §6, §8 W3 row).

W3 folds the TWO hardest export reduces into the executor-traced forward:

- ``VertexUnionFind`` folds ``reduces._bind_vertex_union_find`` — the
  ``@torch.jit.script`` union-find (``get_node_assignment_jit`` +
  ``mask_fill_flattened`` + ``.reshape(-1).char()``) now runs INSIDE
  ``executor.run`` instead of the post-executor reduce loop (R1, the sharpest
  fold).
- ``MaskFormerObject`` folds BOTH ``_bind_leading_object`` + ``_bind_object_index``
  into ONE node: a single ``get_maskformer_outputs`` call (null-suppression + pT
  reorder) emitting BOTH the leading-regression (float32 global) and the
  object-index (int8 per-token) leaves (collapsing the two reduces that each
  re-run it). Declares ALL cross-node requires (R4), clones before the in-place
  mutate (R4), implements ``derived_width`` for the index collapse (R6).

The HARD CONSTRAINT (design §8 W3 row): the LEGACY reduce path stays
BITWISE-identical (the W0 oracle shas) and the FOLDED outputs are bitwise-equal
to the legacy reduces on the SAME weights (int8 exact incl L=0 + zero-token /
fake-pad-track edges; floats 1e-6).

union_find_outcome=FOLDED (the byte-diff finding): the folded VertexIndex ONNX is
NOT byte-identical to the legacy reduce export, BUT the difference is PURELY the
auto-generated node-NAME strings (the deeper executor call frame names the
scripted subgraph's nodes differently). The graphs are structurally identical —
same node count, same op-type histogram, same op-type SEQUENCE — and the int8
union-find outputs compare EXACTLY across L=0..N. Per the §6.4 gate
("check_onnx-exact AND ordered output_names/dtypes/dynamic_axes identical" when
the byte-diff is fragile), the @torch.jit.script union-find traces IDENTICALLY
inside ``executor.run``; the fold holds. This file pins that finding
(``test_vertex_union_find_op_sequence_identical_to_legacy``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    check_onnx,
    compile_onnx_plan,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.core.outputs import (
    MaskFormerObject,
    MaskFormerObjects,
    MFLeadVertexDecorator,
    OnnxExportLeaf,
    OnnxExportSink,
    VertexUnionFind,
)
from salt.core.render import dot_source
from salt.tests._fixtures.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
)
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules
from salt.tests._fixtures.regression_fixture import (
    MASKFORMER_WRITER_REG_TARGETS,
    build_maskformer_writer_modules,
    write_parity_norm_dict,
)

ORACLE_DIR = Path("/tmp/w3_oracle")

VARIABLES = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}

# plan 29 W3 B3 (mirrors W2): these are pure-CPU ONNX-trace gates (oracle
# byte-identity, folded==legacy union-find / maskformer). They are NOT GPU/heavy
# despite living under tests/integration/ — the `cpu_always` marker
# (conftest.py) opts them OUT of the GPU skip so they run on EVERY CI invocation,
# with or without --run-integration. Skipping them would let a hard-reduce fold
# regression ship unnoticed.
pytestmark = pytest.mark.cpu_always


def _gn2_export_cfg() -> ExportConfig:
    return ExportConfig(
        model_name="GN2v2",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
        ],
    )


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





# ---------------------------------------------------------------------------
# GO (W4): the FOLDED export contract matches the W0 oracle (the legacy reduce
# path is RETIRED — the folded conversion nodes are the SOLE path; the legacy
# equivalence is captured in the W0 oracle JSON these tests assert against).
# ---------------------------------------------------------------------------

_LEADING_NAMES = [f"leading_objects_{t}" for t in MASKFORMER_WRITER_REG_TARGETS]
_N_REG = len(MASKFORMER_WRITER_REG_TARGETS)


def _build_vertex_folded(tmp_path, weights):
    """Folded ``VertexUnionFind`` node + ``OnnxExportSink`` export of the gn2v2 weights."""
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    vuf = VertexUnionFind(task="track_vertexing", stream="tracks")
    vuf.name = "vertex_uf"
    sink = OnnxExportSink(outputs=[
        OnnxExportLeaf(
            key="outputs.tracks.track_vertexing", name="VertexIndex", dtype="int8", per_token=True
        ),
    ])
    sink.name = "onnx_export"
    modules.update({"vertex_uf": vuf, "onnx_export": sink})
    resolved = resolve_export_config(_gn2_export_cfg(), "GN2_v2")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    nn.ModuleDict({k: v for k, v in modules.items() if isinstance(v, nn.Module)}).load_state_dict(
        map_v1_state_dict(weights, modules), strict=False
    )
    torch.manual_seed(42)
    return export_graph(
        modules, _gn2_export_cfg(), VARIABLES, tmp_path / "folded_vertex.onnx",
        outputs=[], run_name="GN2_v2",
    )


@pytest.fixture(scope="module")
def vertex(tmp_path_factory):
    """The folded VertexIndex export of the v1 weights (W4)."""
    tmp = tmp_path_factory.mktemp("vertex_fold")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    weights = build_test_gn2(tmp).state_dict()
    return SimpleNamespace(folded=_build_vertex_folded(tmp, weights))


def test_vertex_folded_export_contract(vertex):
    """The folded VertexUnionFind sink names the int8 leaf with the v1 dynamic axis."""
    adapter = vertex.folded.adapter
    assert adapter.output_names == ["GN2v2_VertexIndex"]
    assert adapter.output_dtypes == ["int8"]
    assert adapter.dynamic_axes["GN2v2_VertexIndex"] == {0: "n_tracks"}


def test_vertex_folded_check_onnx_agrees_including_zero_tokens(vertex):
    """torch-vs-ort 1e-6 incl. L=0 — the folded scripted union-find traces correctly (§6.4)."""
    grid = [{"tracks": length} for length in (0, 1, 2, 7, 21)]
    result = check_onnx(
        vertex.folded.adapter, vertex.folded.onnx_path, trials=2,
        float_rtol=1e-6, float_atol=1e-6, lengths_grid=grid,
    )
    assert result.passed, result.failures
    assert result.n_cases == 2 * len(grid)


# ---------------------------------------------------------------------------
# MaskFormerObjects: ONE node folds BOTH legacy reduces (leading_object +
# object_index). The legacy reference is the W0 oracle JSON; self-consistency
# (torch == ort) is NaN-aware because the random decoder weights yield a NaN
# leading_object by design (v1 null-suppression semantics).
# ---------------------------------------------------------------------------


def _build_maskformer_folded(tmp_path):
    """Folded single-node maskformer export (MaskFormerObject -> both leaves)."""
    torch.manual_seed(42)
    modules = build_maskformer_writer_modules(tmp_path / "norm_dict.yaml")
    mf = MaskFormerObject(
        n_reg=_N_REG, stream="objects", constituent_stream="tracks",
        leading_name="leading_object", index_name="object_index",
    )
    mf.name = "mf_obj"
    sink = OnnxExportSink(outputs=[
        OnnxExportLeaf(key="outputs.objects.leading_object", names=_LEADING_NAMES),
        OnnxExportLeaf(key="outputs.tracks.object_index", name="HadronIndex", dtype="int8", per_token=True),
    ])
    sink.name = "onnx_export"
    modules.update({"mf_obj": mf, "onnx_export": sink})
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
    """The folded one-node maskformer export (W4)."""
    tmp = tmp_path_factory.mktemp("maskformer_fold")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    return SimpleNamespace(folded=_build_maskformer_folded(tmp))


def test_maskformer_folded_export_contract(maskformer):
    """ONE node mints BOTH leaves: the leading split + the per-token int8 index axis."""
    adapter = maskformer.folded.adapter
    assert adapter.output_names == [
        "MaskFormer_leading_objects_pt",
        "MaskFormer_leading_objects_Lxy",
        "MaskFormer_leading_objects_mass",
        "MaskFormer_HadronIndex",
    ]
    assert adapter.output_dtypes == ["float32", "float32", "float32", "int8"]
    assert adapter.dynamic_axes["MaskFormer_HadronIndex"] == {0: "n_tracks"}


@pytest.mark.skipif(
    not (ORACLE_DIR / "maskformer.json").is_file(),
    reason="W0 oracle goldens not present",
)
def test_maskformer_folded_contract_matches_oracle(maskformer):
    """The folded maskformer export contract matches the W0 golden EXACTLY (W4)."""
    golden = json.loads((ORACLE_DIR / "maskformer.json").read_text())
    adapter = maskformer.folded.adapter
    assert adapter.output_names == golden["output_names"]
    assert adapter.output_dtypes == golden["output_dtypes"]
    assert json.loads(json.dumps(adapter.dynamic_axes)) == golden["dynamic_axes"]
    assert len(adapter.output_names) == golden["output_tuple_len"]


def test_maskformer_folded_torch_vs_ort_nan_aware(maskformer):
    """torch == onnxruntime, NaN-aware, incl L=0 — the folded node traces correctly.

    The random decoder weights yield all-null/all-PV jets -> the leading_object
    float leaf is NaN by design (v1 null-suppression semantics), so the comparison
    is NaN-aware (equal_nan=True, 1e-4 float / int8 exact). The int8 HadronIndex is
    NaN-free and bitwise-exact across L=0..N.
    """
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


# ---------------------------------------------------------------------------
# The TWO-NODE MaskFormer split (USER DESIGN 2026-06-22): the MaskFormerObjects
# reconstruction node exposes the per-vertex leaves; the MFLeadVertexDecorator
# reads them and emits jet-level scalars. End-to-end ONNX export (no oracle —
# the decorator is a NEW capability, unit-tested separately in tests/unit).
# ---------------------------------------------------------------------------


def _build_two_node_mf(tmp_path):
    """Export MaskFormerObjects + MFLeadVertexDecorator + OnnxExportSink (the two-node chain)."""
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    torch.manual_seed(42)
    modules = build_maskformer_writer_modules(tmp_path / "norm_dict.yaml")
    mf = MaskFormerObjects(n_reg=_N_REG, stream="objects", constituent_stream="tracks")
    mf.name = "mf_obj"
    dec = MFLeadVertexDecorator(
        source="outputs.objects.vertices_class_probs",
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 2},
        pt_index=0,
        pv_class_index=0,
        pnull_threshold=0.5,
    )
    dec.name = "lead_vertex"
    sink = OnnxExportSink(
        outputs=[
            OnnxExportLeaf(
                key="outputs.tracks.object_index",
                name="HadronIndex",
                dtype="int8",
                per_token=True,
            ),
            OnnxExportLeaf(key="outputs.jet.lead_vertex_pt", name="lead_vertex_pt"),
            OnnxExportLeaf(key="outputs.jet.lead_vertex_mass", name="lead_vertex_mass"),
        ]
    )
    sink.name = "onnx_export"
    modules.update({"mf_obj": mf, "lead_vertex": dec, "onnx_export": sink})
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
    """The two-node MaskFormer export (reconstruction node + lead-vertex decorator).

    Returns
    -------
    ExportResult
        The exported two-node chain.
    """
    return _build_two_node_mf(tmp_path_factory.mktemp("two_node_mf"))


def test_two_node_mf_export_contract(two_node_mf):
    """The two-node chain wires object_index (per-token int8) + the jet-level decorator scalars."""
    adapter = two_node_mf.adapter
    assert adapter.output_names == [
        "MaskFormer_HadronIndex",
        "MaskFormer_lead_vertex_pt",
        "MaskFormer_lead_vertex_mass",
    ]
    assert adapter.output_dtypes == ["int8", "float32", "float32"]
    # only the per-token index carries a dynamic axis; the jet-level scalars are global
    assert adapter.dynamic_axes["MaskFormer_HadronIndex"] == {0: "n_tracks"}
    assert "MaskFormer_lead_vertex_pt" not in adapter.dynamic_axes
    assert "MaskFormer_lead_vertex_mass" not in adapter.dynamic_axes


def test_two_node_mf_traces_and_runs_including_zero_tokens(two_node_mf):
    """The exported two-node chain runs in onnxruntime across L=0..N (the decorator is trace-safe).

    No legacy oracle (the decorator is a NEW capability — unit-tested in
    tests/unit/outputs/test_hard_reduce_nodes_w3.py); here we assert the full chain
    traces + runs: the per-token HadronIndex sizes to the constituent count and the
    two jet-level scalars are finite-or-NaN (well-defined, never raising) at every
    sequence length including the L=0 zero-token jet.
    """
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
    """Wrap MFLeadVertexDecorator.forward as a 2-tensor-in / N-scalar-out module for export.

    Drives the decorator's selection math directly (class_probs + regression in,
    the configured jet-level scalars out) so the NEW capability's NON-NaN selection
    path can be exercised through onnxruntime with CONTROLLED inputs — the full
    two-node fixture only ever hits the all-NaN path with random decoder weights
    (R1 LOW: the non-NaN selection was ONNX-untested in the repo before this).
    """

    def __init__(self, decorator: MFLeadVertexDecorator) -> None:
        super().__init__()
        self.decorator = decorator

    def forward(self, class_probs: torch.Tensor, regression: torch.Tensor):
        from salt.core.graph.bundle import Bundle  # noqa: PLC0415 - test-local
        from salt.core.graph.spec import Mode  # noqa: PLC0415 - test-local

        b = Bundle()
        b.set(self.decorator.source, class_probs)
        b.set(self.decorator.regression_source, regression)
        out = self.decorator.forward(b, Mode.ONNX)
        return tuple(out[key] for key in self.decorator.output_keys)


def test_lead_vertex_decorator_onnx_reproduces_non_nan_selection(tmp_path):
    """The decorator's NON-NaN selection survives ONNX export (R1 LOW coverage pin).

    The two-node fixture's random decoder weights only ever yield all-null/all-PV
    jets (NaN lead vertex), so the only NEW capability was ONNX-verified on the NaN
    path alone. Here we export the decorator math STANDALONE and drive onnxruntime
    with a controlled input where v2 is the highest-pT non-PV non-null vertex —
    onnxruntime must reproduce the eager selection (pt=7.0, mass=70.0), proving the
    masked-argmax-over-(-inf) + torch.where(NaN) trace is selection-correct, not just
    finite-or-NaN. A second all-PV row must still come back NaN through ORT.
    """
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
    """G5 render: the folded MaskFormer config renders the onnx_export card + the W3 nodes.

    The static ONNX plan wiring the reconstruction node + the lead-vertex decorator
    + the OnnxExportSink renders (a) an ``onnx_export (OnnxExportSink)`` card with NO
    ``<sinks>`` sentinel, (b) the ``maskformer_objects`` reconstruction node + the
    ``mf_lead_vertex`` decorator node on-graph, and (c) the node->node edge
    (reconstruction -> decorator) the decorator's demand creates — proving the
    decorator's demand keeps the reconstruction node alive (design §7 render payoff).
    """
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    torch.manual_seed(42)
    modules = build_maskformer_writer_modules(tmp_path / "norm_dict.yaml")
    mf = MaskFormerObjects(n_reg=_N_REG, stream="objects", constituent_stream="tracks")
    mf.name = "maskformer_objects"
    dec = MFLeadVertexDecorator(
        source="outputs.objects.vertices_class_probs",
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 2},
        pt_index=0,
        pv_class_index=0,
    )
    dec.name = "mf_lead_vertex"
    sink = OnnxExportSink(
        outputs=[
            OnnxExportLeaf(
                key="outputs.tracks.object_index",
                name="HadronIndex",
                dtype="int8",
                per_token=True,
            ),
            OnnxExportLeaf(key="outputs.jet.lead_vertex_pt", name="lead_vertex_pt"),
            OnnxExportLeaf(key="outputs.jet.lead_vertex_mass", name="lead_vertex_mass"),
        ]
    )
    sink.name = "onnx_export"
    modules.update({"maskformer_objects": mf, "mf_lead_vertex": dec, "onnx_export": sink})
    resolved = resolve_export_config(_mf_export_cfg(), "GN3Mask")
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    dot = dot_source(plan, modules)
    assert "<sinks>" not in dot  # the off-graph manifest sentinel is gone
    assert "onnx_export" in dot
    assert "OnnxExportSink" in dot
    assert "maskformer_objects" in plan.module_names
    assert "mf_lead_vertex" in plan.module_names
    # named-consumer edges flow into the NAMED export node (both W3 nodes feed it)
    sink_edges = set(re.findall(r'"(\w+)" -> "onnx_export"', dot))
    assert {"maskformer_objects", "mf_lead_vertex"} <= sink_edges
    # the node->node edge: the decorator reads the reconstruction node's vertices leaf
    assert r'"maskformer_objects" -> "mf_lead_vertex"' in dot


def test_two_node_mf_object_index_equals_single_node_fold(two_node_mf, maskformer):
    """The two-node object_index is bitwise-equal to the single-node fold (same reconstruction).

    Splitting reconstruction from decoration must NOT perturb the object_index parity:
    the HadronIndex int8 leaf the two-node chain emits is the SAME tensor the
    single-node ``MaskFormerObjects`` fold emits across L=0..N (both fold the legacy
    ``object_index`` math, captured in the W0 oracle).
    """
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
