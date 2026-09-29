"""P1 unit gates for `TaskOutput` + its thin subclasses (split from test_producers.py)."""

from __future__ import annotations

import torch

from salt.graph.bundle import Bundle
from salt.graph.planner import compile_plan
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.modules import resolve_bind_schema
from salt.outputs import (
    ClassProbs,
    ClassProbsOp,
    ConversionOp,
    TaskOutput,
)
from salt.tests.unit.outputs.conftest import (  # noqa: PLC2701 — shared split helpers
    _STREAM_J,
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


# demand-gating / width resolution carries over for the conversion ops


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
    """``ClassProbs`` preserves the class width in a TEST-only bind."""
    pred_key = f"preds.{_STREAM_J}.t"
    out_key = f"outputs.{_STREAM_J}.out"
    src = _stub_source(pred_key, 3)
    producer = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    modules = {"src": src, "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 3  # softmax preserves the class dim


def test_identity_op_still_clones_p0_contract():
    """The default `ConversionOp` is still the P0 identity clone (no regression)."""
    preds = torch.randn(3, 4)
    b = Bundle({"preds": {_STREAM_J: {"t": preds}}})
    producer = TaskOutput(task="t", stream=_STREAM_J, name="out")  # default op
    assert isinstance(producer.op, ConversionOp)
    out = producer.forward(b, Mode.TEST)[f"outputs.{_STREAM_J}.out"]
    torch.testing.assert_close(out, preds, rtol=0, atol=0)
    assert out is not preds  # cloned, not aliased (write-once)
