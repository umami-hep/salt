"""`TaskOutput` — the generic conversion producer, plus its thin pre-wired subclasses."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.base import SaltModelModule
from salt.outputs.conversion_ops import (
    ClassProbsOp,
    ConversionOp,
    SeqClassIndexOp,
    SeqClassProbsOp,
)


class TaskOutput(SaltModelModule):
    """Generic producer: ``preds.<stream>.<task>`` -> ``outputs.<stream>.<name>``.

    The copy/softmax/argmax majority needs no dedicated class — one
    parameterised producer applies a `ConversionOp` (the eval math) to a
    task's published prediction and writes the result. The default op is an
    identity copy; the thin `ClassProbs` / `SeqClassIndex` / `SeqClassProbs`
    subclasses pre-select a P1 op.

    Both ports declare ``shape=None`` (rank-agnostic — a global head is
    ``[B, C]``, a sequence head ``[B, T, C]``); the output last-dim width is
    resolved via `derived_widths`, which maps the bound input width through
    the op's `derived_width` (so it resolves from a TEST-only bind alone).

    Parameters
    ----------
    task : str
        The source task's instance name — the producer reads
        ``preds.<stream>.<task>``.
    stream : str
        The stream the source task publishes under; also the stream the
        output is written under.
    name : str, optional
        The output leaf name; defaults to `task`.
    op : ConversionOp | None, optional
        The eval-math conversion op, by default the identity copy.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        op: ConversionOp | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.stream = stream
        self.output_name = name if name is not None else task
        self.op = op if op is not None else ConversionOp()
        self.pred_key = f"preds.{stream}.{task}"
        self.output_key = f"outputs.{stream}.{self.output_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Requires ``preds.<stream>.<task>`` (+ op extras); produces the ``outputs.*`` leaf."""
        del mode
        pred_spec = TensorSpec(shape=None, dtype="float32")
        out_spec = TensorSpec(shape=None, dtype="float32")
        requires: dict[str, TensorSpec] = {self.pred_key: pred_spec}
        requires.update(self.op.extra_requires(self.stream))
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({self.output_key: out_spec}),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Map the bound input prediction width through the op onto the output.

        Returns ``{}`` until the input width is resolved (keeps the hook
        order-insensitive across plans).
        """
        pred_width = widths.get(self.pred_key)
        if pred_width is None:
            return {}
        return {self.output_key: self.op.derived_width(pred_width)}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Apply the op to the prediction leaf, writing the ``outputs.*`` leaf."""
        converted = self.op.convert(b, mode, pred_key=self.pred_key, stream=self.stream)
        return {self.output_key: converted}


class ClassProbs(TaskOutput):
    """Global-classification probability producer (a `TaskOutput` pre-wired with `ClassProbsOp`).

    Reads a pooled classification head's logits and writes the per-class
    probabilities (``softmax`` / ``sigmoid``). Width-preserving.

    Parameters
    ----------
    task : str
        The source classification task's instance name.
    stream : str
        The stream the head publishes under (also the output stream).
    name : str | None, optional
        The output leaf name, by default `task`.
    bce : bool, optional
        Whether the head uses ``BCEWithLogitsLoss`` (sigmoid) instead of
        ``CrossEntropyLoss`` (softmax), by default False.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        bce: bool = False,
    ) -> None:
        super().__init__(task=task, stream=stream, name=name, op=ClassProbsOp(bce=bce))


class SeqClassIndex(TaskOutput):
    """Per-token class-index producer (a `TaskOutput` pre-wired with `SeqClassIndexOp`).

    Reads a sequence classification head's logits, applies masked softmax +
    ``argmax``, and writes the integer per-token class index (e.g.
    ``TrackOrigin``). The class dim collapses, so the output width is 1.

    Parameters
    ----------
    task : str
        The source classification task's instance name.
    stream : str
        The constituent stream the head publishes under (also the output
        stream).
    name : str | None, optional
        The output leaf name, by default `task`.
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True
        (False for a fixed-count query bank).
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        has_pad_mask: bool = True,
    ) -> None:
        super().__init__(
            task=task, stream=stream, name=name, op=SeqClassIndexOp(has_pad_mask=has_pad_mask)
        )


class SeqClassProbs(TaskOutput):
    """Per-token class-probability producer (a `TaskOutput` pre-wired with `SeqClassProbsOp`).

    Reads a sequence classification head's logits, applies masked softmax,
    and writes the ``[B, L, C]`` per-token per-class probabilities — the
    eval-H5 columns (`SeqClassIndex` is the ONNX argmax counterpart).

    Parameters
    ----------
    task : str
        The source classification task's instance name.
    stream : str
        The constituent stream the head publishes under (also the output stream).
    name : str | None, optional
        The output leaf name, by default `task`.
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
        has_pad_mask: bool = True,
    ) -> None:
        super().__init__(
            task=task, stream=stream, name=name, op=SeqClassProbsOp(has_pad_mask=has_pad_mask)
        )
