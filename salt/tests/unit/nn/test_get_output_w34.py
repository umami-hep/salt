"""W34.1 unit gate — `ClassificationTaskModule.get_output` parity (plan 34).

Plan 34 folds the eval conversion (softmax / masked-softmax / argmax) back ONTO
the task as ``get_output(b, mode, run_name) -> list[OutputField]``, retiring the
standalone `ConversionOp` producer indirection on the live path. This wave is
STRICTLY ADDITIVE: classification ``forward`` is already raw-everywhere, so
``get_output`` is a pure addition exercised only by these tests. The gate proves,
on WEIGHT-MATCHED fixtures, that ``get_output``'s PER-LEAF ``value`` is
byte/semantic-identical to:

- (a) the corresponding `ConversionOp` producer forward (``ClassProbs`` /
  ``SeqClassProbs`` / ``SeqClassIndex``), per column;
- (b) the task's own ``get_h5`` floats (after the sink would prefix/pack) and the
  ``onnx_outputs`` naming (``class_suffixes`` / ``pascal_case(name)``).

Covered: the global (pooled) head, the seq head probs (H5 modes) + argmax index
(ONNX), the L=0 zero-token edge case, the masked-softmax pad behaviour (padded
tokens read 0.0), and the BCE sigmoid branch. ``output_time_requires`` is
asserted to return the pad-mask key for seq heads and nothing for the global
head (the W34.2 RunTaskOutput declaration surface).

The metadata contract (bare suffix / dtype / axis / no run-name prefix / no
half-precision / torch value, NOT a packed numpy struct) is asserted field-by-
field so the dumb sink keeps sole ownership of prefix + pack + downcast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from salt.core.graph import Mode
from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
    _TaskModuleBase,  # noqa: PLC2701 - base default under test
)
from salt.core.outputs import ClassProbs, Regression, SeqClassIndex, SeqClassProbs, VertexUnionFind
from salt.core.outputs.producers import OutputField
from salt.core.outputs.names import VERTEX_INDEX, pascal_case

_FLOAT_TOL = 1e-6
_STREAM_J = "jets"
_STREAM_T = "tracks"
_RUN = "GN2"


# ---------------------------------------------------------------------------
# helpers — build + bind a real (weight-matched) task head, run get_output
# ---------------------------------------------------------------------------


def _bind_classification(stream, label, class_names, sequence, *, loss=None, input_key=None):
    """Build + bind a `ClassificationTaskModule` against a hand-built schema.

    Returns
    -------
    ClassificationTaskModule
        The bound module (``module.task`` is the absorbed v1 head ORACLE).
    """
    module = ClassificationTaskModule(
        stream=stream,
        label=label,
        class_names=class_names,
        input=input_key,
        sequence=sequence,
        loss=loss,
    )
    module.name = f"{stream}_cls" if not sequence else "track_origin"
    schema = ResolvedSchema(widths={module.input_key: 8})
    module.bind(schema)
    return module


def _pred_bundle(module, logits, mask=None):
    """A bundle carrying the task's RAW ``preds.*`` leaf (+ pad mask for a seq head).

    Returns
    -------
    Bundle
        The output-time bundle `get_output` reads from.
    """
    data: dict = {"preds": {module.stream: {module.name: logits}}}
    if mask is not None:
        data["masks"] = {module.stream: mask}
    return Bundle(data)


def _fields_by_h5(fields):
    """Map ``h5_name -> OutputField`` (skips ONNX-only fields).

    Returns
    -------
    dict
        ``{h5_name: OutputField}`` for the H5-bearing fields.
    """
    return {f.h5_name: f for f in fields if f.h5_name is not None}


# ===========================================================================
# GATE A — global (pooled) head: probs (H5) / split-scalars (ONNX)
# ===========================================================================


def test_global_get_output_value_matches_class_probs_producer():
    """Global head: each `OutputField.value` == the per-class column of `ClassProbs`.

    The producer forward writes the FULL ``[B, C]`` softmax leaf; ``get_output``
    mints one field per class whose ``value`` is that column. Asserted per-class.
    """
    torch.manual_seed(0)
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )
    logits = torch.randn(7, 3)

    # (a) producer ORACLE: full [B, C] softmax leaf
    producer = ClassProbs(task=module.name, stream=_STREAM_J, name="out")
    producer.name = "p"
    producer_out = producer.forward(_pred_bundle(module, logits.clone()), Mode.TEST)[
        f"outputs.{_STREAM_J}.out"
    ]

    fields = module.get_output(_pred_bundle(module, logits.clone()), Mode.TEST, _RUN)

    assert [f.h5_name for f in fields] == ["pb", "pc", "pu"]  # class_suffixes order
    for c, f in enumerate(fields):
        torch.testing.assert_close(f.value, producer_out[..., c], rtol=0, atol=_FLOAT_TOL)


def test_global_get_output_value_matches_get_h5_columns():
    """Global head: per-leaf ``value`` == the f4 floats ``get_h5`` packs (sink prefixes).

    ``get_h5`` softmaxes the raw logits then packs ``{run_name}_{px}`` f4 columns;
    ``get_output`` applies the same softmax and exposes the bare per-class column.
    The floats must match (the run-name prefix + packing are sink-side).
    """
    torch.manual_seed(1)
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )
    logits = torch.randn(5, 3)

    structured = module.get_h5(_pred_bundle(module, logits.clone()), run_name=_RUN)
    fields = module.get_output(_pred_bundle(module, logits.clone()), Mode.TEST, _RUN)

    # get_h5 column {RUN}_{px} <-> get_output field px (bare, sink prefixes)
    for f in fields:
        col = structured[f"{_RUN}_{f.h5_name}"]
        np.testing.assert_allclose(f.value.numpy(), col, rtol=0, atol=_FLOAT_TOL)


def test_global_get_output_onnx_names_match_onnx_outputs():
    """Global head ONNX: field suffixes == ``onnx_outputs`` ``class_suffixes`` (split-scalars).

    The global head's ONNX export splits the prob leaf into per-class scalars
    under the SAME ``class_suffixes`` the H5 columns use; ``get_output`` mints one
    field per class with ``resolved_onnx_name`` == that suffix and the matching
    per-class scalar value.

    The ONNX value is gated against the LIVE sink's actual
    ``torch.split(probs, 1, -1).squeeze()`` (``OnnxExportSink.named_outputs``,
    sinks.py:1714-1717), NOT the un-squeezed ``probs[..., c]`` — on the batch-1
    ONNX trace the sink mints a 0-dim scalar ``[]`` while ``probs[..., c]`` is
    ``[1]``, so a self-consistency-only check would miss the rank divergence.
    Asserted on BOTH B=1 (the real ONNX trace shape) and B=4.
    """
    torch.manual_seed(2)
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )

    (export_entry,) = module.onnx_outputs()  # one split_scalars entry

    # B=1 is the real ONNX export trace shape (adapter keeps batch dim 1 for
    # globals) — the case where the sink's squeeze() drops to a 0-dim scalar.
    for batch in (1, 4):
        logits = torch.randn(batch, 3)
        fields = module.get_output(_pred_bundle(module, logits.clone()), Mode.ONNX, _RUN)

        assert [f.resolved_onnx_name for f in fields] == list(export_entry.names)
        # the live sink's exact per-class scalar: split the [B, C] probs leaf into
        # C columns of [B, 1] then squeeze() (no dim → drops ALL size-1 dims).
        probs = torch.softmax(logits, dim=-1)
        sink_scalars = [part.squeeze() for part in torch.split(probs, 1, -1)]
        for c, f in enumerate(fields):
            assert f.onnx_dtype == "float32"
            # SHAPE must match the sink (0-dim [] at B=1), not just the values
            assert f.value.shape == sink_scalars[c].shape
            torch.testing.assert_close(f.value, sink_scalars[c], rtol=0, atol=_FLOAT_TOL)
        if batch == 1:
            # the rank divergence the un-squeezed probs[..., c] would have:
            # this is what the OLD value=probs[..., c] produced (rank 1), proving
            # the squeeze in get_output's ONNX branch is what closes the gap.
            assert probs[..., 0].shape == (1,)
            assert sink_scalars[0].shape == ()


def test_global_get_output_bce_uses_sigmoid():
    """Global BCE head: ``get_output`` sigmoids (matches ``ClassProbs(bce=True)``)."""
    torch.manual_seed(3)
    module = _bind_classification(
        _STREAM_J,
        "flavour_label",
        ["sig"],
        sequence=False,
        loss={"class_path": "torch.nn.BCEWithLogitsLoss"},
    )
    logits = torch.randn(6, 1)

    producer = ClassProbs(task=module.name, stream=_STREAM_J, name="out", bce=True)
    producer.name = "p"
    oracle = producer.forward(_pred_bundle(module, logits.clone()), Mode.TEST)[
        f"outputs.{_STREAM_J}.out"
    ]

    (field,) = module.get_output(_pred_bundle(module, logits.clone()), Mode.TEST, _RUN)
    torch.testing.assert_close(field.value, oracle[..., 0], rtol=0, atol=_FLOAT_TOL)


def test_global_get_output_field_metadata_contract():
    """Global head fields: bare suffix, f4/global, value is a full-precision torch tensor.

    No run-name prefix in the name, no half-precision in the value/dtype, no
    packed numpy struct — the sink owns all three (plan 34 §4/§5).
    """
    torch.manual_seed(4)
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )
    fields = module.get_output(_pred_bundle(module, torch.randn(3, 3)), Mode.TEST, _RUN)
    for f in fields:
        assert isinstance(f, OutputField)
        assert not f.h5_name.startswith(f"{_RUN}_")  # bare suffix, sink prefixes
        assert f.dtype == "f4"
        assert f.axis == "global"
        assert f.final is True
        assert isinstance(f.value, torch.Tensor)
        assert f.value.dtype == torch.float32  # full precision; sink downcasts to f2


def test_global_output_time_requires_empty():
    """Global head declares NO output-time deps (plain softmax needs only the pred)."""
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )
    assert module.output_time_requires(Mode.TEST) == []
    assert module.output_time_requires(Mode.ONNX) == []


# ===========================================================================
# GATE B — sequence (per-token) head: probs (H5) vs argmax index (ONNX)
# ===========================================================================


def test_seq_get_output_h5_probs_match_seq_class_probs_producer():
    """Seq head (H5): each ``value`` == the per-class column of `SeqClassProbs` (masked softmax)."""
    torch.manual_seed(5)
    b, t, c = 5, 6, 8
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(c)], sequence=True
    )
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 3:] = True  # padded tail
    mask[1, :] = True  # fully padded jet

    producer = SeqClassProbs(task=module.name, stream=_STREAM_T, name="probs")
    producer.name = "sp"
    producer_out = producer.forward(_pred_bundle(module, logits.clone(), mask), Mode.TEST)[
        f"outputs.{_STREAM_T}.probs"
    ]

    fields = module.get_output(_pred_bundle(module, logits.clone(), mask), Mode.TEST, _RUN)
    assert [f.h5_name for f in fields] == module.class_suffixes
    for ci, f in enumerate(fields):
        assert f.axis == "per_token"
        assert f.onnx_name is None  # H5-only; ONNX side is the argmax index
        torch.testing.assert_close(f.value, producer_out[..., ci], rtol=0, atol=_FLOAT_TOL)


def test_seq_get_output_h5_probs_match_get_h5_columns():
    """Seq head (H5): per-leaf ``value`` == the per-class f4 columns ``get_h5`` packs."""
    torch.manual_seed(6)
    b, t, c = 4, 5, 8
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(c)], sequence=True
    )
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 2:] = True

    structured = module.get_h5(_pred_bundle(module, logits.clone(), mask), run_name=_RUN)
    fields = module.get_output(_pred_bundle(module, logits.clone(), mask), Mode.TEST, _RUN)
    for f in fields:
        col = structured[f"{_RUN}_{f.h5_name}"]
        np.testing.assert_allclose(f.value.numpy(), col, rtol=0, atol=_FLOAT_TOL)


def test_seq_get_output_pad_positions_zeroed():
    """Seq head (H5): the masked softmax zeroes padded tokens (v1 eval byte-parity quirk)."""
    torch.manual_seed(7)
    b, t, c = 3, 5, 8
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(c)], sequence=True
    )
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 2:] = True
    mask[2, :] = True  # fully padded

    fields = module.get_output(_pred_bundle(module, logits.clone(), mask), Mode.TEST, _RUN)
    by_h5 = _fields_by_h5(fields)
    for f in by_h5.values():
        # every padded (mask True) position reads exactly 0.0 (the masked-softmax quirk)
        padded = f.value[mask]
        torch.testing.assert_close(padded, torch.zeros_like(padded), rtol=0, atol=0)
    # probabilities over classes sum to 1 on valid tokens (sanity)
    stacked = torch.stack([by_h5[s].value for s in module.class_suffixes], dim=-1)
    valid = ~mask
    torch.testing.assert_close(
        stacked[valid].sum(-1), torch.ones(int(valid.sum())), rtol=0, atol=_FLOAT_TOL
    )


def test_seq_get_output_onnx_index_matches_seq_class_index_producer():
    """Seq head (ONNX): the single field's ``value`` == `SeqClassIndex` int8 argmax leaf.

    Reproduces the zero-row-append/strip ``argmax`` trick VERBATIM and names the
    leaf ``pascal_case(task)`` (e.g. ``TrackOrigin``), int8, per-token, H5-only-None.
    """
    torch.manual_seed(8)
    length, classes = 6, 8
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(classes)], sequence=True
    )
    logits = torch.randn(1, length, classes)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, 4:] = True  # padded tail

    producer = SeqClassIndex(task=module.name, stream=_STREAM_T, name="index")
    producer.name = "si"
    oracle = producer.forward(_pred_bundle(module, logits.clone(), mask), Mode.ONNX)[
        f"outputs.{_STREAM_T}.index"
    ]

    (field,) = module.get_output(_pred_bundle(module, logits.clone(), mask), Mode.ONNX, _RUN)
    assert field.h5_name is None  # ONNX-only leaf
    assert field.resolved_onnx_name == pascal_case(module.name)  # TrackOrigin
    assert field.onnx_dtype == "int8"
    assert field.axis == "per_token"
    assert field.value.dtype == torch.int8
    assert field.value.shape == (length,)  # [L] — batch squeezed (Athena per-token)
    torch.testing.assert_close(field.value, oracle, rtol=0, atol=0)  # int — EXACT


def test_seq_get_output_onnx_name_matches_onnx_outputs():
    """Seq head ONNX: the field name == the ``onnx_outputs`` argmax entry name + dtype."""
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(8)], sequence=True
    )
    (export_entry,) = module.onnx_outputs()  # one argmax int8 entry
    assert export_entry.name == pascal_case(module.name)
    assert export_entry.dtype == "int8"

    bundle = _pred_bundle(module, torch.randn(1, 4, 8), torch.zeros(1, 4, dtype=torch.bool))
    (field,) = module.get_output(bundle, Mode.ONNX, _RUN)
    assert field.resolved_onnx_name == export_entry.name
    assert field.onnx_dtype == "int8"


def test_seq_get_output_probs_and_index_have_distinct_names():
    """Write-once: the H5 probs suffixes and the ONNX index name never collide.

    Probs columns are the per-class suffixes (``o0``..); the ONNX argmax leaf is
    the pascal-case task name (``TrackOrigin``) — distinct leaf identities, so the
    two representations can co-mint without a write-once ``outputs.*`` clash
    (today's ``track_origin_probs`` vs ``track_origin_index`` split).
    """
    torch.manual_seed(9)
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(8)], sequence=True
    )
    logits, mask = torch.randn(2, 4, 8), torch.zeros(2, 4, dtype=torch.bool)
    h5_fields = module.get_output(_pred_bundle(module, logits.clone(), mask), Mode.TEST, _RUN)
    onnx_fields = module.get_output(
        _pred_bundle(module, logits[:1].clone(), mask[:1]), Mode.ONNX, _RUN
    )
    h5_names = {f.h5_name for f in h5_fields}
    onnx_names = {f.resolved_onnx_name for f in onnx_fields}
    assert h5_names.isdisjoint(onnx_names)


def test_seq_get_output_onnx_zero_token_jet():
    """Seq head ONNX L=0: the zero-row trick yields an empty int8 vector, never an error."""
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(8)], sequence=True
    )
    logits = torch.randn(1, 0, 8)  # zero tokens
    mask = torch.zeros(1, 0, dtype=torch.bool)
    (field,) = module.get_output(_pred_bundle(module, logits, mask), Mode.ONNX, _RUN)
    assert field.value.dtype == torch.int8
    assert field.value.shape == (0,)


def test_seq_get_output_onnx_argmax_invariant_to_softmax():
    """Seq head ONNX int8 == argmax of the RAW logits on valid tokens (argmax invariance)."""
    torch.manual_seed(10)
    length, classes = 7, 8
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(classes)], sequence=True
    )
    logits = torch.randn(1, length, classes)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, 5:] = True

    (field,) = module.get_output(_pred_bundle(module, logits.clone(), mask), Mode.ONNX, _RUN)
    raw_argmax = torch.argmax(logits.squeeze(0), dim=-1).char()
    valid = ~mask.squeeze(0)
    torch.testing.assert_close(field.value[valid], raw_argmax[valid], rtol=0, atol=0)


def test_seq_output_time_requires_pad_mask():
    """Seq head declares ``masks.<stream>`` as an output-time dep (W34.2 RunTaskOutput surface)."""
    module = _bind_classification(
        _STREAM_T, "origin_label", [f"o{i}" for i in range(8)], sequence=True
    )
    assert module.output_time_requires(Mode.TEST) == [f"masks.{_STREAM_T}"]
    assert module.output_time_requires(Mode.ONNX) == [f"masks.{_STREAM_T}"]


def test_seq_output_time_requires_empty_when_no_pad_mask():
    """A seq head with no pad mask (objects query bank) declares no output-time deps."""
    module = _bind_classification(
        "objects", "origin_label", [f"o{i}" for i in range(8)], sequence=True
    )
    # objects stream is the no-pad-mask query bank (has_pad_mask False)
    assert module.has_pad_mask is False
    assert module.output_time_requires(Mode.TEST) == []


def test_seq_get_output_no_pad_mask_plain_softmax():
    """A no-pad-mask seq head (objects) uses a plain softmax (no mask read), per-class probs."""
    torch.manual_seed(11)
    b, t, c = 3, 4, 8
    module = _bind_classification(
        "objects", "origin_label", [f"o{i}" for i in range(c)], sequence=True
    )
    logits = torch.randn(b, t, c)
    fields = module.get_output(_pred_bundle(module, logits.clone()), Mode.TEST, _RUN)
    oracle = torch.softmax(logits, dim=-1)
    for ci, f in enumerate(fields):
        torch.testing.assert_close(f.value, oracle[..., ci], rtol=0, atol=_FLOAT_TOL)


# ===========================================================================
# GATE C — base default + additive guard
# ===========================================================================


def test_base_get_output_raises_for_unsupported_family():
    """The `_TaskModuleBase` default raises the unsupported-family ConfigError."""
    base = _TaskModuleBase.__new__(_TaskModuleBase)  # bypass __init__ (no config needed)
    base.name = "weird"
    with pytest.raises(ConfigError, match="ships no output rendering"):
        base.get_output(Bundle({}), Mode.TEST, _RUN)
    # the base output_time_requires is the empty default
    assert _TaskModuleBase.output_time_requires(base, Mode.TEST) == []


# ===========================================================================
# GATE D — vertexing (plan 34 W34.3): union-find int8 (TEST i8 / ONNX char index)
# ===========================================================================


def _bind_vertexing(stream=_STREAM_T):
    """Build + bind a `VertexingTaskModule` against a hand-built schema.

    Returns
    -------
    VertexingTaskModule
        The bound module (``module.task`` is the absorbed v1 head ORACLE that owns
        ``run_inference`` union-find).
    """
    module = VertexingTaskModule(
        stream=stream,
        label="ftagTruthVertexIndex",
        origin_label="ftagTruthOriginLabel",
        context="pooled.global",
        dense={"hidden_layers": [4], "activation": "ReLU"},
    )
    module.name = "track_vertexing"
    schema = ResolvedSchema(widths={module.input_key: 8, module.context: 4})
    module.bind(schema)
    return module


def _vtx_bundle(module, edge_scores, mask):
    """A bundle carrying the RAW vertexing edge scores + the stream pad mask.

    Returns
    -------
    Bundle
        The output-time bundle the vertexing `get_output` reads.
    """
    return Bundle({
        "preds": {module.stream: {module.name: edge_scores}},
        "masks": {module.stream: mask},
    })


def test_vtx_output_time_requires_pad_mask():
    """The vertexing head declares ``masks.<stream>`` as its output-time dep (W34.2 surface)."""
    module = _bind_vertexing()
    assert module.output_time_requires(Mode.TEST) == [f"masks.{_STREAM_T}"]
    assert module.output_time_requires(Mode.ONNX) == [f"masks.{_STREAM_T}"]


def test_vtx_get_output_onnx_matches_vertex_union_find_producer():
    """Vertexing (ONNX): the field ``value`` == the `VertexUnionFind` producer leaf.

    The producer runs ``get_node_assignment_jit`` -> ``mask_fill_flattened`` ->
    ``.reshape(-1).char()`` on the RAW edge scores; ``get_output`` folds the IDENTICAL
    chain. Names the leaf ``VertexIndex`` int8 per-token (the shared cross-mode const).
    """
    module = _bind_vertexing()
    gen = torch.Generator().manual_seed(3)
    n_tracks = 4
    n_edges = n_tracks * (n_tracks - 1)
    edge_scores = torch.rand(n_edges, 1, generator=gen)
    mask = torch.zeros(1, n_tracks, dtype=torch.bool)

    producer = VertexUnionFind(task=module.name, stream=_STREAM_T)
    producer.name = "track_vertex_index"
    oracle = producer.forward(_vtx_bundle(module, edge_scores, mask), Mode.ONNX)[
        f"outputs.{_STREAM_T}.track_vertexing"
    ]

    (field,) = module.get_output(_vtx_bundle(module, edge_scores, mask), Mode.ONNX, _RUN)
    assert field.h5_name is None  # ONNX-only leaf
    assert field.resolved_onnx_name == VERTEX_INDEX
    assert field.onnx_dtype == "int8"
    assert field.axis == "per_token"
    assert field.value.dtype == torch.int8
    torch.testing.assert_close(field.value, oracle, rtol=0, atol=0)  # int — EXACT


def test_vtx_get_output_test_matches_get_h5_run_inference():
    """Vertexing (TEST): the field ``value`` == the int-cast union-find ``get_h5`` packs.

    ``get_h5`` ``run_inference``-s the raw edge scores (union-find) then ``.int()`` ->
    u2s i8; ``get_output`` mints the SAME int-cast union-find value as a ``[B, L, 1]``
    leaf (bare ``VertexIndex`` i8, prefix follows ``prefix_vertex_column``).
    """
    module = _bind_vertexing()
    gen = torch.Generator().manual_seed(7)
    # the head emits one edge score per ordered pair of VALID tracks (the compressed
    # adjacency graph), so n_edges = n_valid * (n_valid - 1). Use an all-valid mask
    # so the edge count matches (the union-find chain is shaped for the valid graph).
    n_tracks = 5
    n_edges = n_tracks * (n_tracks - 1)
    edge_scores = torch.rand(n_edges, 1, generator=gen)
    mask = torch.zeros(1, n_tracks, dtype=torch.bool)

    structured = module.get_h5(_vtx_bundle(module, edge_scores, mask), run_name=_RUN)
    (field,) = module.get_output(_vtx_bundle(module, edge_scores, mask), Mode.TEST, _RUN)
    assert field.h5_name == VERTEX_INDEX
    assert field.onnx_name is None  # H5-only (ONNX side is the .char() index)
    assert field.dtype == "i8"
    assert field.axis == "per_token"
    assert field.prefix is False  # bare VertexIndex (v1 byte-parity, default)
    # the get_h5 column (bare VertexIndex) and the get_output int value must match
    np.testing.assert_array_equal(
        field.value.cpu().numpy().reshape(structured[VERTEX_INDEX].shape),
        structured[VERTEX_INDEX],
    )


def test_vtx_get_output_manifest_mirrors_get_output():
    """Vertexing `get_output_manifest` == `get_output` field metadata (value-free)."""
    module = _bind_vertexing()
    for mode in (Mode.TEST, Mode.ONNX):
        (mf,) = module.get_output_manifest(mode, _RUN)
        assert mf.value is None
        assert mf.axis == "per_token"
        assert mf.dtype == ("int8" if mode & Mode.ONNX else "i8")
        if mode & Mode.ONNX:
            assert mf.h5_name is None
            assert mf.resolved_onnx_name == VERTEX_INDEX
        else:
            assert mf.h5_name == VERTEX_INDEX
            assert mf.onnx_name is None
            assert mf.prefix is False


def test_vtx_get_output_prefix_follows_prefix_vertex_column():
    """``prefix_vertex_column=True`` flips the H5 VertexIndex column to run-name-prefixed."""
    module = VertexingTaskModule(
        stream=_STREAM_T,
        label="ftagTruthVertexIndex",
        origin_label="ftagTruthOriginLabel",
        context="pooled.global",
        dense={"hidden_layers": [4], "activation": "ReLU"},
        prefix_vertex_column=True,
    )
    module.name = "track_vertexing"
    module.bind(ResolvedSchema(widths={module.input_key: 8, module.context: 4}))
    edge_scores = torch.rand(4 * 3, 1)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    (field,) = module.get_output(_vtx_bundle(module, edge_scores, mask), Mode.TEST, _RUN)
    assert field.prefix is True


# ===========================================================================
# GATE E — regression (plan 34 W34.3): de-scale (TEST f4 / ONNX squeezed scalars)
# ===========================================================================


def _bind_regression(stream, targets, *, sequence, denoms=None, norm=None, fields=(), gaussian=False):
    """Build + bind a `RegressionTaskModule` against a hand-built schema.

    Returns
    -------
    RegressionTaskModule
        The bound module (``module.task`` is the absorbed v1 head ORACLE owning
        ``run_inference`` de-scaling).
    """
    module = RegressionTaskModule(
        stream=stream,
        targets=targets,
        sequence=sequence,
        target_denominators=denoms,
        norm_params=norm,
        gaussian=gaussian,
    )
    module.name = f"{stream}_reg"
    schema = ResolvedSchema(
        widths={module.input_key: 8}, fields={module.input_feature_key: fields}
    )
    module.bind(schema)
    return module


def _reg_bundle(module, preds, *, mask=None, labels=None, inputs=None):
    """A bundle carrying the RAW (scaled) regression preds + denom source / pad mask.

    Returns
    -------
    Bundle
        The output-time bundle the regression `get_output` reads.
    """
    data: dict = {"preds": {module.stream: {module.name: preds}}}
    if mask is not None:
        data["masks"] = {module.stream: mask}
    if labels is not None:
        data["labels"] = {module.stream: labels}
    if inputs is not None:
        data["inputs"] = {module.stream: inputs}
    return Bundle(data)


def test_reg_norm_params_get_output_matches_get_h5():
    """Regression norm_params global head: get_output value == get_h5 columns."""
    torch.manual_seed(20)
    module = _bind_regression(
        _STREAM_J, ["mHH", "dR"], sequence=False, norm={"mean": [1.0, 2.0], "std": [3.0, 4.0]}
    )
    preds = torch.randn(6, 2)
    structured = module.get_h5(_reg_bundle(module, preds.clone()), run_name=_RUN)
    fields = module.get_output(_reg_bundle(module, preds.clone()), Mode.TEST, _RUN)
    assert [f.h5_name for f in fields] == ["mHH", "dR"]
    for f in fields:
        assert f.dtype == "f4" and f.axis == "global"
        np.testing.assert_allclose(
            f.value.numpy(), structured[f"{_RUN}_{f.h5_name}"], rtol=0, atol=_FLOAT_TOL
        )


def test_reg_get_output_value_matches_regression_descale_producer():
    """Regression: each get_output value == the per-column of the `Regression` producer leaf."""
    torch.manual_seed(21)
    module = _bind_regression(_STREAM_J, ["mHH"], sequence=False, norm={"mean": 1.0, "std": 2.0})
    preds = torch.randn(5, 1)
    producer = Regression(
        task=module.name, stream=_STREAM_J, name="out", targets=["mHH"],
        norm_params={"mean": 1.0, "std": 2.0},
    )
    producer.name = "p"
    producer_out = producer.forward(_reg_bundle(module, preds.clone()), Mode.TEST)[
        f"outputs.{_STREAM_J}.out"
    ]
    (field,) = module.get_output(_reg_bundle(module, preds.clone()), Mode.TEST, _RUN)
    torch.testing.assert_close(field.value, producer_out[..., 0], rtol=0, atol=_FLOAT_TOL)


def test_reg_get_output_onnx_global_squeezed_scalar():
    """Regression global ONNX: each value is the squeezed 0-dim scalar at B=1 (split_scalars)."""
    torch.manual_seed(22)
    module = _bind_regression(_STREAM_J, ["mHH"], sequence=False, norm={"mean": 0.5, "std": 2.0})
    preds = torch.randn(1, 1)  # batch-1 ONNX trace shape
    (field,) = module.get_output(_reg_bundle(module, preds.clone()), Mode.ONNX, _RUN)
    # the v1 split_scalars shape: split [1, R] into [1,1] then .squeeze() -> []
    descaled = module._descaled_preds(_reg_bundle(module, preds.clone()), Mode.ONNX)  # noqa: SLF001
    sink_scalar = descaled[..., 0].squeeze()
    assert field.value.shape == () == sink_scalar.shape
    torch.testing.assert_close(field.value, sink_scalar, rtol=0, atol=_FLOAT_TOL)


def test_reg_ratio_output_time_requires_mode_split():
    """Ratio head output_time_requires: label denom in TEST, input Feature in ONNX (+pad mask)."""
    module = _bind_regression(
        _STREAM_J, ["m_over_mHH"], sequence=False, denoms=["mHH"], fields=("mHH",)
    )
    assert module.output_time_requires(Mode.TEST) == [f"labels.{_STREAM_J}.mHH"]
    assert module.output_time_requires(Mode.ONNX) == [module.input_feature_key]


def test_reg_ratio_get_output_test_matches_get_h5():
    """Ratio head (TEST): get_output value == get_h5 (de-scale via labels.<stream>.<denom>)."""
    torch.manual_seed(23)
    module = _bind_regression(
        _STREAM_J, ["m_over_mHH"], sequence=False, denoms=["mHH"], fields=("mHH",)
    )
    preds = torch.randn(7, 1)
    denom = torch.rand(7)
    b1 = _reg_bundle(module, preds.clone(), labels={"mHH": denom.clone()})
    b2 = _reg_bundle(module, preds.clone(), labels={"mHH": denom.clone()})
    structured = module.get_h5(b1, run_name=_RUN)
    (field,) = module.get_output(b2, Mode.TEST, _RUN)
    np.testing.assert_allclose(
        field.value.numpy(), structured[f"{_RUN}_m_over_mHH"], rtol=0, atol=_FLOAT_TOL
    )


def test_reg_gaussian_get_output_matches_get_h5_one_array():
    """Gaussian head: get_output mints 2R fields (means ‖ _stddev) == get_h5 columns."""
    torch.manual_seed(24)
    module = _bind_regression(
        _STREAM_J, ["mu"], sequence=False, norm={"mean": 0.0, "std": 2.0}, gaussian=True
    )
    preds = torch.randn(5, 2)  # gaussian head: [B, 2R] (mean ‖ raw var)
    structured = module.get_h5(_reg_bundle(module, preds.clone()), run_name=_RUN)
    fields = module.get_output(_reg_bundle(module, preds.clone()), Mode.TEST, _RUN)
    assert [f.h5_name for f in fields] == ["mu", "mu_stddev"]
    for f in fields:
        np.testing.assert_allclose(
            f.value.numpy(), structured[f"{_RUN}_{f.h5_name}"], rtol=0, atol=_FLOAT_TOL
        )


def test_reg_get_output_manifest_mirrors_get_output():
    """Regression `get_output_manifest` == `get_output` field names/dtypes (value-free)."""
    module = _bind_regression(
        _STREAM_J, ["mHH", "dR"], sequence=False, norm={"mean": [1.0, 2.0], "std": [3.0, 4.0]}
    )
    for mode in (Mode.TEST, Mode.ONNX):
        manifest = module.get_output_manifest(mode, _RUN)
        assert [m.h5_name for m in manifest] == ["mHH", "dR"]
        for m in manifest:
            assert m.value is None and m.dtype == "f4" and m.axis == "global"


# --- W34.3 critic-fix: write-once + per-token agreement of get_output ---------


def test_reg_get_output_does_not_mutate_raw_preds_leaf():
    """``get_output`` de-scaling must NOT mutate the bundle's RAW ``preds.*`` leaf.

    ``_descaled_preds`` clones before ``run_inference`` (which de-scales IN PLACE),
    so the raw leaf is byte-unchanged after get_output and a SECOND read does not
    double-de-scale (write-once §2.1; pre-fix this rewrote ``preds.jets.reg`` and a
    repeat read double-scaled it).
    """
    torch.manual_seed(40)
    module = _bind_regression(_STREAM_J, ["mHH"], sequence=False, norm={"mean": 1.0, "std": 2.0})
    preds = torch.randn(5, 1)
    raw_before = preds.clone()
    bundle = _reg_bundle(module, preds)  # store the SAME tensor object in the bundle
    first = module.get_output(bundle, Mode.TEST, _RUN)[0].value.clone()
    # the raw leaf the bundle holds is byte-unchanged (no in-place de-scale leak)
    torch.testing.assert_close(bundle.get(module.pred_key), raw_before, rtol=0, atol=0)
    # a second get_output on the SAME bundle yields the SAME de-scaled value
    # (no double-de-scale because the leaf was never mutated)
    second = module.get_output(bundle, Mode.TEST, _RUN)[0].value
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_reg_seq_get_output_matches_producer_per_token_scaled():
    """Per-token SCALED seq head: get_output == the `Regression` producer on every token.

    The W34.3 axis fix makes the task's de-scale last-axis, so a per-token
    norm_params head's get_output matches the producer leaf column-for-column,
    token-for-token (pre-fix the task scaled only tokens 0..R-1 and crashed at L=0).
    """
    torch.manual_seed(41)
    norm = {"mean": [10.0, 20.0], "std": [2.0, 0.5]}
    module = _bind_regression(_STREAM_T, ["a", "b"], sequence=True, norm=norm)
    preds = torch.randn(3, 4, 2)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 3] = True  # a real padded position
    producer = Regression(
        task=module.name, stream=_STREAM_T, name="out", targets=["a", "b"],
        norm_params=norm, sequence=True,
    )
    producer.name = "p"
    b_prod = _reg_bundle(module, preds.clone(), mask=mask.clone())
    producer_out = producer.forward(b_prod, Mode.TEST)[f"outputs.{_STREAM_T}.out"]
    b_sec = _reg_bundle(module, preds.clone(), mask=mask.clone())
    fields = module.get_output(b_sec, Mode.TEST, _RUN)
    for i, f in enumerate(fields):
        assert f.axis == "per_token"
        torch.testing.assert_close(
            f.value, producer_out[..., i], rtol=0, atol=_FLOAT_TOL, equal_nan=True
        )


def test_reg_gaussian_seq_get_output_matches_producer_per_token():
    """Per-token gaussian seq head: get_output stddev == producer on valid positions.

    The shipped ``regression_gaussian.yaml gaussian_seq_out`` head. Pre-fix the
    task's gaussian de-scale was first-axis (corrupting tracks, the stddev formula
    never reached the var channel) while the producer was last-axis — they diverged
    ~2.0. Post-fix both are last-axis: the per-token means ‖ stddev agree, incl. NaN
    at padded positions.
    """
    torch.manual_seed(42)
    norm = {"mean": [1.0], "std": [1.0]}
    module = _bind_regression(_STREAM_T, ["dphi"], sequence=True, norm=norm, gaussian=True)
    preds = torch.randn(3, 4, 2)  # [B, L, 2R]
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[1, 3] = True
    producer = Regression(
        task=module.name, stream=_STREAM_T, name="out", targets=["dphi"],
        norm_params=norm, gaussian=True, sequence=True,
    )
    producer.name = "p"
    producer_out = producer.forward(
        _reg_bundle(module, preds.clone(), mask=mask.clone()), Mode.TEST
    )[f"outputs.{_STREAM_T}.out"]
    b_sec = _reg_bundle(module, preds.clone(), mask=mask.clone())
    fields = module.get_output(b_sec, Mode.TEST, _RUN)
    assert [f.h5_name for f in fields] == ["dphi", "dphi_stddev"]
    for i, f in enumerate(fields):
        torch.testing.assert_close(
            f.value, producer_out[..., i], rtol=0, atol=_FLOAT_TOL, equal_nan=True
        )
    # the stddev column at a valid position is sqrt(softplus(var))*std, NOT the
    # pre-fix mean formula (var channel was reached, not corrupted)
    stddev = fields[1].value
    assert torch.isfinite(stddev[0, 0])
    assert torch.isnan(stddev[1, 3])  # masked position NaN-filled
