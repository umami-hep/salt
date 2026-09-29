"""Conversion ops — the eval-math strategies behind `TaskOutput`."""

from __future__ import annotations

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.spec import Mode, TensorSpec

# Conversion ops: small strategy objects (not nn.Module) holding config and
# reproducing one task family's eval math. Three hooks parameterise the
# generic TaskOutput: extra_requires (extra input ports), derived_width
# (output width given input width), convert (the forward conversion).


class ConversionOp:
    """Eval-math strategy behind `TaskOutput`; base is an identity passthrough."""

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Extra input ports this op reads beyond the source ``preds.*`` leaf (none by default)."""
        del stream
        return {}

    def derived_width(self, in_width: int) -> int:
        """The output leaf's last-dim width given the input width (unchanged by default)."""
        return in_width

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Identity copy of the prediction leaf (cloned so the output never aliases it)."""
        del mode, stream
        return b.get(pred_key).clone()


class ClassProbsOp(ConversionOp):
    """Global classification probabilities: sigmoid (BCE) or softmax over classes.

    Parameters
    ----------
    bce : bool, optional
        Use per-class sigmoid (a ``BCEWithLogitsLoss`` head) instead of
        softmax (``CrossEntropyLoss``), by default False.
    """

    def __init__(self, bce: bool = False) -> None:
        self.bce = bool(bce)

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Sigmoid (BCE) or softmax over the class dim."""
        del mode, stream
        preds = b.get(pred_key)
        if self.bce:
            return torch.sigmoid(preds)
        assert preds.ndim == 2, (
            "ClassProbsOp is a global (pooled) head; per-token heads publish through RunTaskOutput"
        )
        return torch.softmax(preds, dim=-1)
