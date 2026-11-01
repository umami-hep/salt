"""`FixtureProducer` — a test-only ``preds.*`` -> ``outputs.*`` copy/softmax producer."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.base import SaltModelModule


class FixtureProducer(SaltModelModule):
    """Copy (or softmax) ``preds.<stream>.<task>`` into ``outputs.<stream>.<name or task>``.

    Rank-agnostic float32 ports; declares no ONNX manifest (it names no output).
    """

    def __init__(self, task: str, stream: str, name: str | None = None, *, softmax: bool = False):
        super().__init__()
        self.softmax = softmax
        self.pred_key = f"preds.{stream}.{task}"
        self.output_key = f"outputs.{stream}.{name or task}"

    def declare_io(self, mode: Mode) -> IO:
        del mode
        spec = TensorSpec(shape=None, dtype="float32")
        return IO(unflatten_spec({self.pred_key: spec}), unflatten_spec({self.output_key: spec}))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """The output width is the bound input width (empty until it resolves)."""
        return {self.output_key: widths[self.pred_key]} if self.pred_key in widths else {}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        del mode
        preds = b.get(self.pred_key)
        return {self.output_key: torch.softmax(preds, dim=-1) if self.softmax else preds.clone()}
