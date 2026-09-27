"""Unit gates for the hard-reduce conversion nodes (declare_io / widths / forward)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.outputs.sinks.onnx.reduces import (
    get_maskformer_outputs,
)
from salt.outputs import (
    H5OutputSink,
    MaskFormerObjects,
    MFLeadVertexDecorator,
    OutputColumn,
)

pytestmark = pytest.mark.cpu_always


# MaskFormerObjects (folds _bind_leading_object + _bind_object_index — ONE node)


def test_maskformer_object_declares_all_cross_node_requires():
    """ALL three maskformer reads are declared so the demand-closure keeps the decoder alive."""
    node = MaskFormerObjects(n_reg=3, stream="objects", constituent_stream="tracks")
    node.name = "mf_obj"
    io = node.declare_io(Mode.ONNX)
    assert sorted(flatten_spec(io.requires)) == [
        "objects.class_probs",
        "objects.masks",
        "preds.objects.regression",
    ]
    # the two-node split: the reconstruction node mints the
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
    node = MaskFormerObjects(n_reg=2, regression_task="obj_reg", stream="objects")
    node.name = "mf_obj"
    assert "preds.objects.obj_reg" in flatten_spec(node.declare_io(Mode.ONNX).requires)


def test_maskformer_object_derived_widths():
    """The index leaf collapses to 1; leading + vertices_regression keep n_reg."""
    node = MaskFormerObjects(n_reg=3, stream="objects", constituent_stream="tracks")
    node.name = "mf_obj"
    assert node.derived_widths({}) == {
        "outputs.tracks.object_index": 1,
        "outputs.objects.leading_object": 3,
        "outputs.objects.vertices_regression": 3,
    }


def test_maskformer_object_rejects_bad_n_reg():
    """``n_reg`` must be a positive int (the leading-object target count)."""
    with pytest.raises(ConfigError, match="n_reg"):
        MaskFormerObjects(n_reg=0, stream="objects")
    with pytest.raises(ConfigError, match="n_reg"):
        MaskFormerObjects(n_reg=True, stream="objects")  # bool is not a valid count


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
    node = MaskFormerObjects(n_reg=n_reg, stream="objects", constituent_stream="tracks")
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
    node = MaskFormerObjects(n_reg=n_reg, stream="objects", constituent_stream="tracks")
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


# get_maskformer_outputs (reduces) — regression tests for the early-return fix
#
# The deleted early returns were: `n_tracks == 0` (batch-1 dummy shapes) and
# `not null_preds.any()` (fired when NO slot was null — inverted, batch-1 dummy
# shapes). These tests pin the single straight-line path's per-batch shapes and
# NaN/index semantics at every combination the old branches covered.


def _class_probs_with_null(p_null: torch.Tensor) -> torch.Tensor:
    """``[..., 3]`` (b, c, null) class probs with the given per-slot null probability."""
    rest = (1.0 - p_null) / 2
    return torch.stack([rest, rest, p_null], dim=-1)


def test_get_maskformer_outputs_all_real_objects_no_early_return():
    """Every slot real (low p_null) -> per-batch shapes, finite outputs, pT-sorted
    class_probs/regression, valid index range. The deleted ``not null_preds.any()``
    branch used to fire exactly here (no slot null) and return batch-1 NaN shapes.
    """
    b, m, length, r = 3, 5, 7, 3
    gen = torch.Generator().manual_seed(41)
    class_probs = _class_probs_with_null(torch.full((b, m), 0.1))
    masks = torch.randn(b, m, length, generator=gen)
    regression = torch.randn(b, m, r, generator=gen)
    leading, indices, _out_cp, out_reg = get_maskformer_outputs(
        {"class_probs": class_probs, "masks": masks.clone(), "regression": regression.clone()},
        apply_reorder=True,
    )
    assert leading.shape == (b, r)
    assert not torch.isnan(leading).any()
    assert out_reg.shape == (b, m, r)
    assert not torch.isnan(out_reg).any()
    # pT-sorted: regression[:, :, 0] non-increasing per jet
    assert (out_reg[:, :-1, 0] >= out_reg[:, 1:, 0]).all()
    assert indices.shape == (b, length)
    assert set(indices.unique().tolist()) <= set(range(m)) | {-2}
    # leading == the max-pt slot's row of the ORIGINAL regression (a permutation of rows
    # never changes which row wins the argmax, so this is a valid independent oracle).
    top = regression[:, :, 0].argmax(-1).view(b, 1, 1).expand(b, 1, r)
    expected = regression.gather(1, top).squeeze(1)
    torch.testing.assert_close(leading, expected, rtol=0, atol=0)


def test_get_maskformer_outputs_all_null_objects_gives_nan_and_neg2():
    """Every slot null (high p_null) -> real per-batch NaN shapes, ``-2`` indices everywhere.

    The old code handled this correctly ALREADY via the normal path (the removed branch
    fired on the OPPOSITE condition) — kept green here as a refactor guard.
    """
    b, m, length, r = 3, 5, 7, 3
    gen = torch.Generator().manual_seed(42)
    class_probs = _class_probs_with_null(torch.full((b, m), 0.9))
    masks = torch.randn(b, m, length, generator=gen)
    regression = torch.randn(b, m, r, generator=gen)
    leading, indices, _out_cp, out_reg = get_maskformer_outputs(
        {"class_probs": class_probs, "masks": masks.clone(), "regression": regression.clone()},
        apply_reorder=True,
    )
    assert leading.shape == (b, r)
    assert torch.isnan(leading).all()
    assert out_reg.shape == (b, m, r)
    assert torch.isnan(out_reg).all()
    assert indices.shape == (b, length)
    assert (indices == -2).all()


def test_get_maskformer_outputs_mixed_batch_per_jet_independent():
    """A mixed batch (one all-null jet, one jet with two real slots) resolves per-jet,
    at the correct per-batch shape (not the old code's hardcoded batch-1 shapes).
    """
    m, length, r = 5, 7, 3
    gen = torch.Generator().manual_seed(43)
    p_null = torch.tensor([
        [0.9, 0.9, 0.9, 0.9, 0.9],  # jet 0: all null
        [0.1, 0.1, 0.9, 0.9, 0.9],  # jet 1: two real slots
    ])
    class_probs = _class_probs_with_null(p_null)
    masks = torch.randn(2, m, length, generator=gen)
    regression = torch.randn(2, m, r, generator=gen)
    leading, indices, _out_cp, out_reg = get_maskformer_outputs(
        {"class_probs": class_probs, "masks": masks.clone(), "regression": regression.clone()},
        apply_reorder=True,
    )
    assert leading.shape == (2, r)
    assert torch.isnan(leading[0]).all()
    assert not torch.isnan(leading[1]).any()
    assert out_reg.shape == (2, m, r)
    assert torch.isnan(out_reg[0]).all()
    assert (~torch.isnan(out_reg[1, :, 0])).sum() == 2
    assert indices.shape == (2, length)
    assert (indices[0] == -2).all()


def test_get_maskformer_outputs_zero_length_no_early_return():
    """``L == 0`` (no tracks) still runs the single straight-line path: leading/regression
    come from the sorted regression tensor, indices are ``[B, 0]``. Regression test for the
    deleted ``n_tracks == 0`` early return (which used to return batch-1 ``(1, n_obj)`` /
    ``(1, n_obj, n_reg)`` dummy shapes and ``None`` indices instead).
    """
    b, m, r = 2, 5, 3
    gen = torch.Generator().manual_seed(44)
    p_null = torch.tensor([[0.1, 0.1, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9]])
    class_probs = _class_probs_with_null(p_null)
    masks = torch.randn(b, m, 0, generator=gen)
    regression = torch.randn(b, m, r, generator=gen)
    leading, indices, _out_cp, out_reg = get_maskformer_outputs(
        {"class_probs": class_probs, "masks": masks.clone(), "regression": regression.clone()},
        apply_reorder=True,
    )
    assert leading.shape == (b, r)
    assert not torch.isnan(leading[0]).any()  # jet 0 has real slots
    assert torch.isnan(leading[1]).all()  # jet 1 all null
    assert out_reg.shape == (b, m, r)
    assert indices.shape == (b, 0)


# MFLeadVertexDecorator (jet-level capability with NO legacy oracle)

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
    """argmax==null but pnull<threshold (thin-spread) is EXCLUDED (the third cut)."""
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


def test_lead_vertex_decorator_all_real_objects_gives_real_lead_scalars():
    """Regression test for the reduces fix: a no-null-slot jet gives REAL lead-vertex
    scalars. Before the fix, ``get_maskformer_outputs``'s inverted ``not null_preds.any()``
    branch fired on exactly these inputs (no slot null) and returned an all-NaN
    ``vertices_regression`` while ``vertices_class_probs`` flowed through real — so the
    decorator's qualify mask passed vertices whose regression was undefined and every
    lead-vertex scalar came out NaN. The expected slot is computed independently here
    from the raw inputs (no `get_maskformer_outputs` call as the oracle): a row
    permutation never changes which underlying (class, regression) pair wins the
    qualify-then-argmax-pt selection, so this is valid regardless of reordering.
    """
    n_reg, n_obj, n_tracks = 3, 4, 6
    # all class_probs have LOW null prob (last class) -> no slot is suppressed
    class_probs = torch.tensor([[
        [0.6, 0.3, 0.1],  # argmax 0 = PV -> excluded
        [0.2, 0.7, 0.1],  # argmax 1 -> qualifies
        [0.5, 0.4, 0.1],  # argmax 0 = PV -> excluded
        [0.3, 0.6, 0.1],  # argmax 1 -> qualifies
    ]])
    masks = torch.randn(1, n_obj, n_tracks, generator=torch.Generator().manual_seed(31))
    reg = torch.randn(1, n_obj, n_reg, generator=torch.Generator().manual_seed(32))
    # the independent oracle: pv_class_index=0, null_index defaults to C-1=2, threshold 0.5
    pred_class = class_probs.argmax(-1)[0]
    pnull = class_probs[..., -1][0]
    qualify = (pnull < 0.5) & (pred_class != 0) & (pred_class != 2)
    masked_pt = torch.where(qualify, reg[0, :, 0], torch.full_like(reg[0, :, 0], -torch.inf))
    winner = masked_pt.argmax()
    expected_pt = reg[0, winner, 0]
    expected_mass = reg[0, winner, 2]

    writer = MaskFormerObjects(n_reg=n_reg, stream="objects", constituent_stream="tracks")
    writer.name = "mf_obj"
    wb = Bundle()
    wb.set("objects.class_probs", class_probs)
    wb.set("objects.masks", masks)
    wb.set("preds.objects.regression", reg)
    w_out = writer.forward(wb, Mode.ONNX)
    assert not torch.isnan(w_out["outputs.objects.vertices_regression"]).any()
    assert not torch.isnan(w_out["outputs.objects.vertices_class_probs"]).any()

    dec = _decorator()
    out = dec.forward(
        _vbundle(
            w_out["outputs.objects.vertices_class_probs"],
            w_out["outputs.objects.vertices_regression"],
        ),
        Mode.ONNX,
    )
    torch.testing.assert_close(out["outputs.jet.lead_vertex_pt"], expected_pt.unsqueeze(0))
    torch.testing.assert_close(out["outputs.jet.lead_vertex_mass"], expected_mass.unsqueeze(0))


def test_lead_vertex_decorator_declare_io_fit_val_empty_test_equals_onnx_ports():
    """FIT/VAL declare nothing (mode-gated to TEST|ONNX); TEST and ONNX ports are the same
    keys (only their `modes` tag differs).
    """
    node = _decorator()
    for mode in (Mode.FIT, Mode.VAL):
        io = node.declare_io(mode)
        assert flatten_spec(io.requires) == {}
        assert flatten_spec(io.produces) == {}
    test_io = node.declare_io(Mode.TEST)
    onnx_io = node.declare_io(Mode.ONNX)
    assert set(flatten_spec(test_io.requires)) == set(flatten_spec(onnx_io.requires))
    assert set(flatten_spec(test_io.produces)) == set(flatten_spec(onnx_io.produces))


def _mf14_row(argmax_idx: int, pnull: float) -> list[float]:
    """A 14-class row (pv=0 .. null=13) with the given argmax index and null probability."""
    row = [0.0] * 14
    if argmax_idx == 13:
        row[13] = pnull
        rest = (1.0 - pnull) / 13
        for i in range(13):
            row[i] = rest
    else:
        row[13] = pnull
        row[argmax_idx] = 1.0 - pnull - 0.01 * 12
        for i in range(14):
            if i not in (argmax_idx, 13):
                row[i] = 0.01
    return row


def test_lead_vertex_decorator_14class_vertexing_matches_test_and_onnx_chains():
    """A realistic 14-class (pv..null) config: the PV slot and the null slot are excluded
    even with the largest pT; the highest-pT qualifying slot wins; the TEST chain
    (`MaskFormerObjects.forward(Mode.TEST)` -> decorator) and the ONNX chain
    (`forward(Mode.ONNX)` -> decorator) agree on the jet-level scalars.
    """
    m, r = 15, 5
    class_rows = [_mf14_row(13, 0.9) for _ in range(m)]  # remaining slots: null-dominant
    class_rows[0] = _mf14_row(0, 0.05)  # PV, largest pt -> excluded
    class_rows[3] = _mf14_row(13, 0.9)  # argmax IS null, large pt -> excluded
    class_rows[7] = _mf14_row(5, 0.05)  # the winner
    class_rows[9] = _mf14_row(2, 0.05)  # qualifies, loses on pt
    class_probs = torch.tensor([class_rows])  # [1, 15, 14]

    reg = torch.zeros(1, m, r)
    reg[0, 0] = torch.tensor([50.0, 0.0, 0.0, 0.0, 9.9])  # PV, excluded
    reg[0, 3] = torch.tensor([99.0, 0.0, 0.0, 0.0, 8.8])  # null, excluded
    reg[0, 7] = torch.tensor([12.0, 0.0, 0.0, 0.0, 3.1])  # the winner
    reg[0, 9] = torch.tensor([9.0, 0.0, 0.0, 0.0, 1.0])  # qualifies, loses on pt
    masks = torch.randn(1, m, 6, generator=torch.Generator().manual_seed(51))
    pad = torch.zeros(1, 6, dtype=torch.bool)

    writer = MaskFormerObjects(n_reg=r, stream="objects", constituent_stream="tracks")
    writer.name = "mf_obj"
    dec = MFLeadVertexDecorator(
        source=_CP, outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 4},
        pt_index=0, pv_class_index=0,
    )
    dec.name = "lead_vertex"

    b_onnx = Bundle()
    b_onnx.set("objects.class_probs", class_probs)
    b_onnx.set("objects.masks", masks)
    b_onnx.set("preds.objects.regression", reg)
    onnx_out = writer.forward(b_onnx, Mode.ONNX)
    onnx_dec = dec.forward(
        _vbundle(
            onnx_out["outputs.objects.vertices_class_probs"],
            onnx_out["outputs.objects.vertices_regression"],
        ),
        Mode.ONNX,
    )
    assert onnx_dec["outputs.jet.lead_vertex_pt"].item() == pytest.approx(12.0)
    assert onnx_dec["outputs.jet.lead_vertex_mass"].item() == pytest.approx(3.1)

    b_test = Bundle()
    b_test.set("objects.class_probs", class_probs)
    b_test.set("objects.masks", masks)
    b_test.set("preds.objects.regression", reg)
    b_test.set("masks.tracks", pad)
    test_out = writer.forward(b_test, Mode.TEST)
    test_dec = dec.forward(
        _vbundle(
            test_out["outputs.objects.vertices_class_probs"],
            test_out["outputs.objects.vertices_regression"],
        ),
        Mode.TEST,
    )
    torch.testing.assert_close(
        test_dec["outputs.jet.lead_vertex_pt"], onnx_dec["outputs.jet.lead_vertex_pt"],
        equal_nan=True,
    )
    torch.testing.assert_close(
        test_dec["outputs.jet.lead_vertex_mass"], onnx_dec["outputs.jet.lead_vertex_mass"],
        equal_nan=True,
    )


def test_lead_vertex_decorator_14class_all_null_gives_nan():
    """An all-null 14-class jet (every slot argmax==null) yields NaN jet-level scalars."""
    m, r = 15, 5
    class_probs = torch.tensor([[_mf14_row(13, 0.9) for _ in range(m)]])
    reg = torch.randn(1, m, r, generator=torch.Generator().manual_seed(52))
    masks = torch.randn(1, m, 6, generator=torch.Generator().manual_seed(53))
    writer = MaskFormerObjects(n_reg=r, stream="objects", constituent_stream="tracks")
    writer.name = "mf_obj"
    b = Bundle()
    b.set("objects.class_probs", class_probs)
    b.set("objects.masks", masks)
    b.set("preds.objects.regression", reg)
    out = writer.forward(b, Mode.ONNX)
    dec = MFLeadVertexDecorator(
        source=_CP, outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 4},
        pt_index=0, pv_class_index=0,
    )
    dec.name = "lead_vertex"
    dec_out = dec.forward(
        _vbundle(
            out["outputs.objects.vertices_class_probs"], out["outputs.objects.vertices_regression"]
        ),
        Mode.ONNX,
    )
    assert torch.isnan(dec_out["outputs.jet.lead_vertex_pt"]).all()
    assert torch.isnan(dec_out["outputs.jet.lead_vertex_mass"]).all()


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
    # OutputColumn is the sink's internal value object; seed it directly for
    # this white-box packing test.
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


def test_lead_vertex_decorator_test_manifest_resolves_h5_columns():
    """`MFLeadVertexDecorator.manifest_fields(Mode.TEST)` resolves into H5OutputSink's TEST
    column table (the manifest path, not a hand-seeded `OutputColumn` like the test above).
    """
    dec = MFLeadVertexDecorator(
        source=_CP,
        outputs={"lead_vertex_pt": 0, "lead_vertex_mass": 4},
        pt_index=0,
        pv_class_index=0,
        jet_stream="jets",
    )
    dec.name = "mf_lead_vertex"
    sink = H5OutputSink()
    sink.bind_output_section({})
    sink.bind_model_modules({"mf_lead_vertex": dec})
    columns = sink._resolve_columns("MFrun")  # noqa: SLF001
    by_key = {col.key: col for col in columns}
    assert "outputs.jets.lead_vertex_pt" in by_key
    assert "outputs.jets.lead_vertex_mass" in by_key
    assert by_key["outputs.jets.lead_vertex_pt"].stream == "jets"
    assert by_key["outputs.jets.lead_vertex_pt"].column_names("MFrun") == ["MFrun_lead_vertex_pt"]
    assert by_key["outputs.jets.lead_vertex_mass"].column_names("MFrun") == [
        "MFrun_lead_vertex_mass"
    ]
