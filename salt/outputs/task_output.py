"""`TaskOutput` — the generic conversion producer, plus its thin pre-wired subclasses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.base import SaltModelModule
from salt.outputs.conversion_ops import (
    ClassProbsOp,
    ConversionOp,
    SeqClassIndexOp,
    SeqClassProbsOp,
)
from salt.outputs.output_schema import OutputField


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
        # the model module dict, bound at compile so the producer can resolve the
        # source task whose manifest NAMES its ONNX output(s).
        self._model_modules: Mapping[str, Any] | None = None

    def bind_model_modules(self, model_modules: Mapping[str, Any]) -> None:
        """Capture the model module dict so the producer can resolve its source task."""
        self._model_modules = model_modules

    def manifest_fields(self, mode: Mode) -> list[tuple[str, OutputField]]:
        """The converted leaf's ONNX field manifest, derived from the source task.

        ONNX-ONLY: the eval-H5 representation of these tasks rides the
        ``outputs:`` section's `RunTaskOutput`, so declaring TEST fields here
        would duplicate its columns. Each field the task's
        ``get_output_manifest(Mode.ONNX)`` names is tagged with this producer's
        own leaf key, so the ONNX name and dtype come from the task and are
        typed nowhere. Empty when the op has no ONNX representation.

        Raises
        ------
        ConfigError
            When the model modules are unbound, or the source task is missing
            or ships no manifest surface.
        """  # noqa: DOC201 - contract stated in the summary
        if not (mode & Mode.ONNX) or not self.op.has_onnx_manifest:
            return []
        task = self._resolved_task()
        return [
            (self.output_key, field)
            for field in task.get_output_manifest(Mode.ONNX, "salt")
            if field.resolved_onnx_name is not None
        ]

    def _resolved_task(self) -> Any:
        """The live source task, resolved from the bound model modules."""  # noqa: DOC201, DOC501 - private helper, raises documented on manifest_fields
        who = f"{type(self).__name__} {self.output_name!r}"
        if self._model_modules is None:
            raise ConfigError(
                f"{who} has no model modules bound — it derives its ONNX output names from "
                f"task {self.task!r}'s own manifest; ensure the producer is composed with the "
                "model (bind_model_modules is called at compile)"
            )
        task = self._model_modules.get(self.task)
        if task is None:
            raise ConfigError(
                f"{who}: task {self.task!r} is not a model module — candidates are "
                f"{sorted(self._model_modules)}"
            )
        if not callable(getattr(task, "get_output_manifest", None)):
            raise ConfigError(
                f"{who}: task {self.task!r} ({type(task).__name__}) ships no "
                "get_output_manifest — the ONNX sink needs the output NAMES/DTYPES before "
                "any batch runs"
            )
        return task

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
