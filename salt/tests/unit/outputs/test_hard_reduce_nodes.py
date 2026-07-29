"""Unit gates for the hard-reduce conversion nodes (declare_io / widths / forward)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.onnx.reduces import (
    get_maskformer_outputs,
)
from salt.outputs import (
    H5OutputSink,
    MaskFormerObject,
    MaskFormerObjects,
    MFLeadVertexDecorator,
    OutputColumn,
)

pytestmark = pytest.mark.cpu_always


# MaskFormerObject (folds _bind_leading_object + _bind_object_index — ONE node)


def test_maskformer_object_declares_all_cross_node_requires():
    """ALL three maskformer reads are declared so the demand-closure keeps the decoder alive (R4)."""
    node = MaskFormerObject(n_reg=3, stream="objects", constituent_stream="tracks")
    node.name = "mf_obj"
    io = node.declare_io(Mode.ONNX)
    assert sorted(flatten_spec(io.requires)) == [
        "objects.class_probs",
        "objects.masks",
        "preds.objects.regression",
    ]
    # the two-node split (USER DESIGN 2026-06-22): the reconstruction node mints the
    # leading + per-token index folds AND exposes the reordered per-vertex leaves
    assert sorted(flatten_spec(io.produces)) == [
        "outputs.objects.leading_object",
        "outputs.objects.vertices_class_probs",
        "outputs.objects.vertices_regression",
        "outputs.tracks.object_index",
    ]
    produces = flatten_spec(io.produces)
    assert produces["outputs.objects.leading_object"].dtype == "float32"
    assert produces["outputs.tracks.object_index"].dtype == "int8"
    assert produces["outputs.objects.vertices_class_probs"].dtype == "float32"
    assert produces["outputs.objects.vertices_regression"].dtype == "float32"


def test_maskformer_object_non_default_regression_task_threaded():
    """A non-default ``regression_task`` is threaded into the reg port (never hardcoded)."""
    node = MaskFormerObject(n_reg=2, regression_task="obj_reg", stream="objects")
    node.name = "mf_obj"
    assert "preds.objects.obj_reg" in flatten_spec(node.declare_io(Mode.ONNX).requires)


def test_maskformer_object_derived_widths():
    """The index leaf collapses to 1 (R6); leading + vertices_regression keep n_reg."""
    node = MaskFormerObject(n_reg=3, stream="objects", constituent_stream="tracks")
    node.name = "mf_obj"
    assert node.derived_widths({}) == {
        "outputs.tracks.object_index": 1,
        "outputs.objects.leading_object": 3,
        "outputs.objects.vertices_regression": 3,
    }


def test_maskformer_object_rejects_bad_n_reg():
    """``n_reg`` must be a positive int (the leading-object target count)."""
    with pytest.raises(ConfigError, match="n_reg"):
        MaskFormerObject(n_reg=0, stream="objects")
    with pytest.raises(ConfigError, match="n_reg"):
        MaskFormerObject(n_reg=True, stream="objects")  # bool is not a valid count


def test_maskformer_object_forward_matches_inlined_reduces():
    """ONE forward reproduces BOTH legacy reduces' tensors (folds the two calls)."""
    n_reg, n_obj, n_tracks, n_classes = 3, 5, 7, 3
    gen = torch.Generator().manual_seed(13)
    class_probs = torch.randn(1, n_obj, n_classes, generator=gen).softmax(-1)
    masks = torch.randn(1, n_obj, n_tracks, generator=gen)
    reg = torch.randn(1, n_obj, n_reg, generator=gen)
    b = Bundle()
    b.set("objects.class_probs", class_probs)
    b.set("objects.masks", masks)
    b.set("preds.objects.regression", reg)
    node = MaskFormerObject(n_reg=n_reg, stream="objects", constituent_stream="tracks")
    node.name = "mf_obj"
    out = node.forward(b, Mode.ONNX)
    # the reference: one direct get_maskformer_outputs call on CLONED inputs
    leading_reg, indices, _, _ = get_maskformer_outputs(
        {
            "class_probs": class_probs.clone(),
            "masks": masks.clone(),
            "regression": reg.clone(),
        },
        apply_reorder=True,
    )
    torch.testing.assert_close(
        out["outputs.objects.leading_object"],
        leading_reg[:, :n_reg],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        out["outputs.tracks.object_index"], indices.reshape(-1).char(), rtol=0, atol=0
    )


def test_maskformer_object_forward_does_not_mutate_bundle_leaves():
    """The node clones before ``get_maskformer_outputs``'s in-place mutate."""
    n_reg, n_obj, n_tracks, n_classes = 3, 5, 7, 3
    gen = torch.Generator().manual_seed(17)
    class_probs = torch.randn(1, n_obj, n_classes, generator=gen).softmax(-1)
    masks = torch.randn(1, n_obj, n_tracks, generator=gen)
    reg = torch.randn(1, n_obj, n_reg, generator=gen)
    b = Bundle()
    b.set("objects.class_probs", class_probs)
    b.set("objects.masks", masks)
    b.set("preds.objects.regression", reg)
    before = {
        "objects.class_probs": class_probs.clone(),
        "objects.masks": masks.clone(),
        "preds.objects.regression": reg.clone(),
    }
    node = MaskFormerObject(n_reg=n_reg, stream="objects", constituent_stream="tracks")
    node.name = "mf_obj"
    node.forward(b, Mode.ONNX)
    for key, snapshot in before.items():
        torch.testing.assert_close(b.get(key), snapshot, rtol=0, atol=0)


def test_maskformer_objects_exposes_reordered_per_vertex_leaves():
    """The reconstruction node EXPOSES the reordered per-vertex class_probs + regression."""
    n_reg, n_obj, n_tracks, n_classes = 3, 5, 7, 3
    gen = torch.Generator().manual_seed(23)
    class_probs = torch.randn(1, n_obj, n_classes, generator=gen).softmax(-1)
    masks = torch.randn(1, n_obj, n_tracks, generator=gen)
    reg = torch.randn(1, n_obj, n_reg, generator=gen)
    b = Bundle()
    b.set("objects.class_probs", class_probs)
    b.set("objects.masks", masks)
    b.set("preds.objects.regression", reg)
    node = MaskFormerObjects(n_reg=n_reg, stream="objects", constituent_stream="tracks")
    node.name = "mf_obj"
    out = node.forward(b, Mode.ONNX)
    _, _, exp_cp, exp_reg = get_maskformer_outputs(
        {"class_probs": class_probs.clone(), "masks": masks.clone(), "regression": reg.clone()},
        apply_reorder=True,
    )
    torch.testing.assert_close(
        out["outputs.objects.vertices_class_probs"], exp_cp, rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(
        out["outputs.objects.vertices_regression"], exp_reg, rtol=0, atol=0, equal_nan=True
    )


# MFLeadVertexDecorator (NEW jet-level capability — NO legacy oracle; GO3 unit gate)

_CP = "outputs.objects.vertices_class_probs"
_REG = "outputs.objects.vertices_regression"


def _decorator(**kw):
    """A 3-class (pv/sv/null) decorator: pt at reg index 0, mass at reg index 2."""
    node = MFLeadVertexDecorator(
        source=_CP,
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 2},
        pt_index=0,
        pv_class_index=0,
        pnull_threshold=0.5,
        **kw,
    )
    node.name = "lead_vertex"
    return node


def _vbundle(class_probs, regression):
    b = Bundle()
    b.set(_CP, class_probs)
    b.set(_REG, regression)
    return b


def test_lead_vertex_decorator_declares_vertex_sources_and_jet_outputs():
    """The decorator reads the two per-vertex leaves and produces jet-level GLOBAL scalars."""
    node = _decorator()
    io = node.declare_io(Mode.ONNX)
    assert sorted(flatten_spec(io.requires)) == [_CP, _REG]
    assert sorted(flatten_spec(io.produces)) == [
        "outputs.jet.lead_vertex_mass",
        "outputs.jet.lead_vertex_pt",
    ]
    assert node.derived_widths({}) == {
        "outputs.jet.lead_vertex_pt": 1,
        "outputs.jet.lead_vertex_mass": 1,
    }


def test_lead_vertex_decorator_selects_highest_pt_non_pv_non_null():
    """LEAD = highest-pt vertex with pnull<thr AND argmax-class != pv_class_index."""
    # class_probs [1, 3, 3]: v0 PV-dominant, v1 SV-dominant low-null, v2 NULL-dominant
    class_probs = torch.tensor([[
        [0.8, 0.1, 0.1],  # v0: argmax 0 = PV -> excluded
        [0.1, 0.7, 0.2],  # v1: argmax 1 = SV, pnull 0.2 < 0.5 -> qualifies
        [0.1, 0.2, 0.7],  # v2: argmax 2 = null, pnull 0.7 >= 0.5 -> excluded
    ]])
    # regression [1, 3, 3]: cols = [pt, Lxy, mass]; v0 pt big (but PV), v1 pt 5, v2 pt 9 (null)
    regression = torch.tensor([[
        [99.0, 1.0, 100.0],  # v0 PV (excluded) — its big pt must NOT win
        [5.0, 2.0, 50.0],  # v1 SV — the lead
        [9.0, 3.0, 70.0],  # v2 null (excluded) — its bigger pt must NOT win
    ]])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    # lead is v1: pt=5.0, mass=50.0 (NOT v0's 99/100 nor v2's 9/70)
    torch.testing.assert_close(out["outputs.jet.lead_vertex_pt"], torch.tensor([5.0]))
    torch.testing.assert_close(out["outputs.jet.lead_vertex_mass"], torch.tensor([50.0]))


def test_lead_vertex_decorator_pnull_cut_excludes_high_pt_null_vertex():
    """A high-pt vertex above the pnull threshold is excluded (the null cut)."""
    class_probs = torch.tensor([[
        [0.1, 0.85, 0.05],  # v0: SV, pnull 0.05 < 0.5 -> qualifies, pt 3
        [0.1, 0.3, 0.6],  # v1: pnull 0.6 >= 0.5 -> EXCLUDED despite huge pt
    ]])
    regression = torch.tensor([[
        [3.0, 0.0, 30.0],  # v0 the only qualifier
        [99.0, 0.0, 99.0],  # v1 high-pt but null-suppressed by the pnull cut
    ]])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    torch.testing.assert_close(out["outputs.jet.lead_vertex_pt"], torch.tensor([3.0]))


def test_lead_vertex_decorator_argmax_null_below_pnull_thr_excluded():
    """argmax==null but pnull<threshold (thin-spread) is EXCLUDED (3rd cut, user 2026-06-22)."""
    class_probs = torch.tensor([[
        [0.1, 0.8, 0.1],   # v0: SV (argmax 1), pnull 0.1 < 0.5 -> qualifies, pt 3
        [0.3, 0.3, 0.4],   # v1: argmax 2 = NULL but pnull 0.4 < 0.5 -> EXCLUDED by the argmax!=null cut
    ]])
    regression = torch.tensor([[
        [3.0, 0.0, 30.0],   # v0 the only real-class qualifier
        [99.0, 0.0, 99.0],  # v1 high-pt, argmax==null -> must NOT win (old rule would have picked it)
    ]])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    torch.testing.assert_close(out["outputs.jet.lead_vertex_pt"], torch.tensor([3.0]))
    torch.testing.assert_close(out["outputs.jet.lead_vertex_mass"], torch.tensor([30.0]))


def test_lead_vertex_decorator_pv_exclusion():
    """The argmax-PV vertex is excluded even if it is the highest-pt vertex."""
    class_probs = torch.tensor([[
        [0.9, 0.05, 0.05],  # v0: PV (argmax 0), highest pt -> EXCLUDED
        [0.1, 0.8, 0.1],  # v1: SV -> the lead
    ]])
    regression = torch.tensor([[
        [50.0, 0.0, 500.0],  # v0 PV (excluded)
        [4.0, 0.0, 40.0],  # v1 the lead
    ]])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    torch.testing.assert_close(out["outputs.jet.lead_vertex_pt"], torch.tensor([4.0]))
    torch.testing.assert_close(out["outputs.jet.lead_vertex_mass"], torch.tensor([40.0]))


def test_lead_vertex_decorator_no_qualifying_vertex_fills_nan():
    """When NO vertex qualifies (all PV or all null) every jet-level scalar is NaN."""
    # both vertices are PV (argmax 0) -> none qualify
    class_probs = torch.tensor([[
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
    ]])
    regression = torch.tensor([[
        [10.0, 0.0, 100.0],
        [20.0, 0.0, 200.0],
    ]])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    assert torch.isnan(out["outputs.jet.lead_vertex_pt"]).all()
    assert torch.isnan(out["outputs.jet.lead_vertex_mass"]).all()


def test_lead_vertex_decorator_all_null_fills_nan():
    """An all-null jet (every pnull >= threshold) yields NaN scalars (the null path)."""
    class_probs = torch.tensor([[
        [0.1, 0.2, 0.7],  # pnull 0.7 >= 0.5
        [0.2, 0.1, 0.7],  # pnull 0.7 >= 0.5
    ]])
    regression = torch.tensor([[[5.0, 0.0, 50.0], [9.0, 0.0, 90.0]]])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    assert torch.isnan(out["outputs.jet.lead_vertex_pt"]).all()


def test_lead_vertex_decorator_empty_object_axis_fills_nan():
    """M == 0 (no object queries) yields all-NaN jet scalars (the explicit empty path)."""
    class_probs = torch.zeros(1, 0, 3)
    regression = torch.zeros(1, 0, 3)
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    assert out["outputs.jet.lead_vertex_pt"].shape == (1,)
    assert torch.isnan(out["outputs.jet.lead_vertex_pt"]).all()


def test_lead_vertex_decorator_nan_class_vertex_excluded():
    """DEFENSIVE: a NaN class-probs row is excluded (NaN cmp = False) — belt-and-braces."""
    # v0 has NaN class probs (a row Node 1a never mints — defensive coverage); v1 is a real SV
    class_probs = torch.tensor([[
        [float("nan"), float("nan"), float("nan")],  # NaN-class row -> excluded (NaN cmp False)
        [0.1, 0.8, 0.1],  # v1 SV -> the lead
    ]])
    regression = torch.tensor([[
        [99.0, 0.0, 99.0],  # the NaN-class vertex (excluded)
        [4.0, 0.0, 40.0],  # v1 the lead
    ]])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    torch.testing.assert_close(out["outputs.jet.lead_vertex_pt"], torch.tensor([4.0]))


def test_lead_vertex_decorator_dummy_path_through_node1a_fills_nan():
    """Node 1a's ``not null_preds.any()`` dummy path -> decorator emits NaN lead-vertex scalars."""
    n_reg, n_obj, n_tracks = 3, 4, 6
    # all class_probs have LOW null prob (last class), so null_preds = (p_null > 0.5)
    # is all-False -> get_maskformer_outputs takes the `not null_preds.any()` dummy path
    class_probs = torch.tensor([[
        [0.6, 0.3, 0.1],  # null prob 0.1 < 0.5
        [0.2, 0.7, 0.1],  # null prob 0.1 < 0.5
        [0.5, 0.4, 0.1],  # null prob 0.1 < 0.5
        [0.3, 0.6, 0.1],  # null prob 0.1 < 0.5
    ]])
    masks = torch.randn(1, n_obj, n_tracks, generator=torch.Generator().manual_seed(31))
    reg = torch.randn(1, n_obj, n_reg, generator=torch.Generator().manual_seed(32))
    # drive the REAL Node 1a (the writer) -> get the exposed per-vertex leaves
    writer = MaskFormerObjects(n_reg=n_reg, stream="objects", constituent_stream="tracks")
    writer.name = "mf_obj"
    wb = Bundle()
    wb.set("objects.class_probs", class_probs)
    wb.set("objects.masks", masks)
    wb.set("preds.objects.regression", reg)
    w_out = writer.forward(wb, Mode.ONNX)
    # confirm the dummy path was taken: regression is all-NaN, class_probs is real (not NaN)
    assert torch.isnan(w_out["outputs.objects.vertices_regression"]).all()
    assert not torch.isnan(w_out["outputs.objects.vertices_class_probs"]).any()
    # feed the exposed leaves into the decorator -> the qualify mask passes vertices
    # but their regression is NaN, so the lead-vertex scalars are NaN (documented).
    dec = _decorator()
    out = dec.forward(
        _vbundle(
            w_out["outputs.objects.vertices_class_probs"],
            w_out["outputs.objects.vertices_regression"],
        ),
        Mode.ONNX,
    )
    assert torch.isnan(out["outputs.jet.lead_vertex_pt"]).all()
    assert torch.isnan(out["outputs.jet.lead_vertex_mass"]).all()


def test_lead_vertex_decorator_per_jet_independent_selection():
    """Selection is per-jet (batched): different jets pick different lead vertices / NaN."""
    # jet 0: v1 qualifies (lead pt 5); jet 1: none qualify (both PV) -> NaN
    class_probs = torch.tensor([
        [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]],  # jet 0: v0 PV, v1 SV -> lead v1
        [[0.9, 0.05, 0.05], [0.85, 0.1, 0.05]],  # jet 1: both PV -> NaN
    ])
    regression = torch.tensor([
        [[50.0, 0.0, 500.0], [5.0, 0.0, 50.0]],
        [[10.0, 0.0, 100.0], [20.0, 0.0, 200.0]],
    ])
    node = _decorator()
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    pt = out["outputs.jet.lead_vertex_pt"]
    assert pt[0].item() == pytest.approx(5.0)
    assert torch.isnan(pt[1])


def test_lead_vertex_decorator_custom_null_index():
    """A non-default null_index uses the configured class column for the pnull cut."""
    # null is class 0 here (null_index=0); v0 high-pt but null, v1 the SV lead
    class_probs = torch.tensor([[
        [0.7, 0.2, 0.1],  # null_index=0 -> pnull 0.7 >= 0.5 -> excluded (argmax 0 also null/pv?)
        [0.1, 0.1, 0.8],  # argmax 2 (SV-ish), pnull(col0)=0.1 < 0.5 -> qualifies
    ]])
    regression = torch.tensor([[[99.0, 0.0, 99.0], [4.0, 0.0, 40.0]]])
    node = MFLeadVertexDecorator(
        source=_CP,
        outputs={"lead_vertex_pt": 0},
        pt_index=0,
        pv_class_index=1,  # PV is class 1 here (neither vertex is class 1)
        pnull_threshold=0.5,
        null_index=0,
    )
    node.name = "lead_vertex"
    out = node.forward(_vbundle(class_probs, regression), Mode.ONNX)
    torch.testing.assert_close(out["outputs.jet.lead_vertex_pt"], torch.tensor([4.0]))


def test_lead_vertex_decorator_does_not_mutate_bundle_leaves():
    """The decorator is a pure reader — it never mutates the per-vertex bundle leaves."""
    class_probs = torch.tensor([[[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]])
    regression = torch.tensor([[[50.0, 0.0, 500.0], [4.0, 0.0, 40.0]]])
    b = _vbundle(class_probs, regression)
    before_cp = class_probs.clone()
    before_reg = regression.clone()
    _decorator().forward(b, Mode.ONNX)
    torch.testing.assert_close(b.get(_CP), before_cp, rtol=0, atol=0)
    torch.testing.assert_close(b.get(_REG), before_reg, rtol=0, atol=0)


def test_lead_vertex_decorator_rejects_bad_config():
    """Config validation: non-outputs source, empty outputs, negative indices."""
    with pytest.raises(ConfigError, match="source"):
        MFLeadVertexDecorator(
            source="preds.objects.foo", outputs={"x": 0}, pt_index=0, pv_class_index=0
        )
    with pytest.raises(ConfigError, match="outputs"):
        MFLeadVertexDecorator(source=_CP, outputs={}, pt_index=0, pv_class_index=0)
    with pytest.raises(ConfigError, match="pt_index"):
        MFLeadVertexDecorator(source=_CP, outputs={"x": 0}, pt_index=-1, pv_class_index=0)
    with pytest.raises(ConfigError, match=r"outputs\[.*\]"):
        MFLeadVertexDecorator(source=_CP, outputs={"x": -2}, pt_index=0, pv_class_index=0)


def test_lead_vertex_decorator_leaf_packs_into_h5_output_column():
    """The decorator's jet-level scalar is a NORMAL outputs.* leaf the H5OutputSink serialises."""
    sink = H5OutputSink()
    # OutputColumn is the sink's internal value object (the explicit-table
    # config surface is retired); seed it directly for this
    # white-box packing test.
    sink._columns = (  # noqa: SLF001
        OutputColumn(key="outputs.jets.lead_vertex_pt", suffixes=["lead_vertex_pt"]),
    )
    sink._columns_resolved = True  # noqa: SLF001
    sink._run_name = "GN3X"  # noqa: SLF001 - exercising the packing path without open_schema
    sink._seq_lengths = {}  # noqa: SLF001 - jets is a GLOBAL stream (no per-token re-expansion)
    b = Bundle()
    b.set("outputs.jets.lead_vertex_pt", torch.tensor([5.0, float("nan"), 7.0]))
    frags = sink._output_fragments(b)  # noqa: SLF001 - the per-batch packer under test
    arr = frags["jets"]
    assert arr.dtype.names == ("GN3X_lead_vertex_pt",)
    assert arr.shape == (3,)
    vals = arr["GN3X_lead_vertex_pt"]
    assert vals[0] == pytest.approx(5.0)
    assert np.isnan(vals[1])  # the NaN fill round-trips into the H5 column
    assert vals[2] == pytest.approx(7.0)
