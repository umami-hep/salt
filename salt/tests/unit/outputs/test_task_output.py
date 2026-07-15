"""P1 unit gates for `TaskOutput` + its thin subclasses (split from test_producers.py)."""

from __future__ import annotations

import torch

from salt.core.graph import (
    IO,
    Mode,
    TensorSpec,
    compile_plan,
    sym_dim,
    unflatten_spec,
)
from salt.core.graph.bundle import Bundle
from salt.core.nn import resolve_bind_schema
from salt.core.outputs import (
    ClassProbs,
    ClassProbsOp,
    ConversionOp,
    SeqClassIndex,
    SeqClassIndexOp,
    SeqClassProbs,
    SeqClassProbsOp,
    TaskOutput,
)
from salt.tests.unit.outputs.conftest import (  # noqa: PLC2701 — shared split helpers
    _STREAM_J,
    _STREAM_T,
    _producer,
)


def test_class_probs_subclass_forwards_like_op():
    """The thin `ClassProbs` subclass == a `TaskOutput` carrying `ClassProbsOp`."""
    torch.manual_seed(3)
    logits = torch.randn(4, 3)
    sub = ClassProbs(task="t", stream=_STREAM_J, name="out")
    sub.name = "p"
    generic = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    b = Bundle({"preds": {_STREAM_J: {"t": logits}}})
    torch.testing.assert_close(
        sub.forward(b, Mode.TEST)[f"outputs.{_STREAM_J}.out"],
        generic.forward(Bundle({"preds": {_STREAM_J: {"t": logits}}}), Mode.TEST)[
            f"outputs.{_STREAM_J}.out"
        ],
        rtol=0,
        atol=0,
    )


def test_seq_class_index_subclass_forwards_like_op():
    """The thin `SeqClassIndex` subclass == a `TaskOutput` carrying `SeqClassIndexOp`."""
    torch.manual_seed(12)
    logits = torch.randn(3, 4, 8)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 2:] = True
    sub = SeqClassIndex(task="t", stream=_STREAM_T, name="origin")
    sub.name = "p"
    generic = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassIndexOp())
    generic.name = "g"
    b1 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    b2 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    torch.testing.assert_close(
        sub.forward(b1, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        generic.forward(b2, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        rtol=0,
        atol=0,
    )


def test_seq_class_probs_subclass_forwards_like_op():
    """The thin `SeqClassProbs` subclass == a `TaskOutput` carrying `SeqClassProbsOp`."""
    torch.manual_seed(42)
    logits = torch.randn(3, 4, 8)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 2:] = True
    sub = SeqClassProbs(task="t", stream=_STREAM_T, name="origin")
    sub.name = "p"
    generic = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassProbsOp())
    generic.name = "g"
    b1 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    b2 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    torch.testing.assert_close(
        sub.forward(b1, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        generic.forward(b2, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        rtol=0,
        atol=0,
    )


# demand-gating / width resolution carries over for the conversion ops
# (design §4 risk 6) — the gate (b) of P0, re-asserted per op


def _stub_source(pred_key, width, *, modes=Mode.ALL):
    """A minimal source module producing ``pred_key`` of last-dim `width`."""

    class _Stub:
        name = "src"

        def declare_io(self, mode):
            del mode
            return IO(
                produces=unflatten_spec(
                    {pred_key: TensorSpec(shape=("B", width), dtype="float32", modes=modes)}
                )
            )

    return _Stub()


def test_class_probs_width_resolves_in_test_only_bind():
    """``ClassProbs`` preserves the class width in a TEST-only bind (design §4 risk 6)."""
    pred_key = f"preds.{_STREAM_J}.t"
    out_key = f"outputs.{_STREAM_J}.out"
    src = _stub_source(pred_key, 3)
    producer = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    modules = {"src": src, "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 3  # softmax preserves the class dim


def test_seq_class_index_width_collapses_to_one():
    """``SeqClassIndex`` collapses the class dim to one index column at bind."""
    pred_key = f"preds.{_STREAM_T}.t"
    out_key = f"outputs.{_STREAM_T}.origin"
    src = _stub_source(pred_key, 8)
    # also need a source for the demanded pad mask
    mask_key = f"masks.{_STREAM_T}"

    class _MaskSrc:
        name = "msrc"

        def declare_io(self, mode):
            del mode
            return IO(
                produces=unflatten_spec(
                    {
                        mask_key: TensorSpec(
                            shape=("B", sym_dim("T", _STREAM_T)), dtype="bool", kind="pad_mask"
                        )
                    }
                )
            )

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassIndexOp())
    producer.name = "producer"
    modules = {"src": src, "msrc": _MaskSrc(), "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 1  # argmax collapses the class dim


def test_seq_class_probs_width_preserved_in_test_only_bind():
    """``SeqClassProbs`` preserves the class dim (C probs out, unlike argmax's collapse)."""
    pred_key = f"preds.{_STREAM_T}.t"
    out_key = f"outputs.{_STREAM_T}.origin"
    src = _stub_source(pred_key, 8)
    mask_key = f"masks.{_STREAM_T}"

    class _MaskSrc:
        name = "msrc"

        def declare_io(self, mode):
            del mode
            return IO(
                produces=unflatten_spec(
                    {
                        mask_key: TensorSpec(
                            shape=("B", sym_dim("T", _STREAM_T)), dtype="bool", kind="pad_mask"
                        )
                    }
                )
            )

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassProbsOp())
    producer.name = "producer"
    modules = {"src": src, "msrc": _MaskSrc(), "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 8  # softmax preserves the class dim


def test_identity_op_still_clones_p0_contract():
    """The default `ConversionOp` is still the P0 identity clone (no regression)."""
    preds = torch.randn(3, 4)
    b = Bundle({"preds": {_STREAM_J: {"t": preds}}})
    producer = TaskOutput(task="t", stream=_STREAM_J, name="out")  # default op
    assert isinstance(producer.op, ConversionOp)
    out = producer.forward(b, Mode.TEST)[f"outputs.{_STREAM_J}.out"]
    torch.testing.assert_close(out, preds, rtol=0, atol=0)
    assert out is not preds  # cloned, not aliased (write-once §2.1)
