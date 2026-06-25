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
    _TaskModuleBase,  # noqa: PLC2701 - base default under test
)
from salt.core.outputs import ClassProbs, SeqClassIndex, SeqClassProbs
from salt.core.outputs.producers import OutputField
from salt.core.writers.names import pascal_case

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
