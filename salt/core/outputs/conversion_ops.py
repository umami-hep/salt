"""Conversion ops — the eval-math strategies behind `TaskOutput`."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import Mode, TensorSpec, sym_dim
from salt.core.outputs.names import pascal_case
from salt.core.outputs.output_field import OutputField, _masked_softmax

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

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """Field manifest for this op's output leaf, resolved from the wrapped task.

        The base (identity) mirrors the wrapped task's own ``output_names``:
        one field per column, bare suffix (run-name prefix stripped — the sink
        re-prefixes), dtype from the task descriptor, axis from the task's
        sequence flag.

        Raises
        ------
        ConfigError
            When the wrapped task ships no ``output_names`` rendering.
        """
        names = task.output_names(run_name)
        sequence = bool(getattr(task, "sequence", False))
        axis = "per_token" if sequence else "global"
        fields: list[OutputField] = []
        for column, dtype in names:
            suffix = column[len(run_name) + 1 :] if column.startswith(f"{run_name}_") else column
            prefix = column.startswith(f"{run_name}_")
            fields.append(
                OutputField(h5_name=suffix, dtype=str(dtype), axis=axis, final=True, prefix=prefix)
            )
        return fields

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Identity copy of the prediction leaf (cloned so the output never aliases it)."""
        del mode, stream
        return b.get(pred_key).clone()


class IdentityOp(ConversionOp):
    """Explicit alias for the identity conversion op (the `ConversionOp` base)."""


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

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float global field per class (``class_suffixes``); same suffixes for H5 and ONNX."""
        del run_name
        return [
            OutputField(h5_name=px, dtype="f4", axis="global", final=True)
            for px in task.class_suffixes
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Sigmoid (BCE) or softmax over the class dim."""
        del mode, stream
        preds = b.get(pred_key)
        if self.bce:
            return torch.sigmoid(preds)
        assert preds.ndim == 2, (
            "ClassProbsOp is a global (pooled) head; use SeqClassIndex for per-token heads"
        )
        return torch.softmax(preds, dim=-1)


class SeqClassIndexOp(ConversionOp):
    """Per-token class index (masked-softmax then ``argmax``, e.g. ``TrackOrigin``).

    The class dim collapses, so the output last dim is 1. The pad mask is
    read from ``masks.<stream>`` — padded tokens are ``-inf`` before the
    softmax (argmax is invariant under it).

    Parameters
    ----------
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True.
        False for a fixed-count query bank (e.g. the MaskFormer ``objects``
        stream, which has no ``masks.objects``).
    """

    def __init__(self, has_pad_mask: bool = True) -> None:
        self.has_pad_mask = bool(has_pad_mask)

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the stream's pad mask for the masked softmax (when `has_pad_mask`)."""
        if not self.has_pad_mask:
            return {}
        return {
            f"masks.{stream}": TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        }

    def derived_width(self, in_width: int) -> int:
        """The argmax collapses the class dim to a single index column (always 1)."""
        del in_width
        return 1

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One int8 per-token ONNX-only field named ``pascal_case(task)`` (e.g. ``TrackOrigin``).

        No H5 column (``h5_name=None``): the TEST H5 carries the full prob
        vector via the sibling `SeqClassProbs` producer instead.
        """
        del run_name
        return [
            OutputField(
                h5_name=None,
                onnx_name=pascal_case(task.name),
                dtype="int8",
                axis="per_token",
                final=True,
            )
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Masked-softmax then per-token ``argmax`` over the class dim (mode-branched).

        TEST returns ``[B, L]`` int64 indices. ONNX returns the int8 ``[L]``
        leaf using the zero-row append/strip trick: a zero row is appended
        along the token axis so the traced argmax stays valid for zero-token
        jets, then stripped and cast to int8 for the Athena output.
        """
        logits = b.get(pred_key)
        mask = b.get(f"masks.{stream}") if self.has_pad_mask else None
        probs = _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
        if mode & Mode.ONNX:
            # zero-row append/strip keeps the traced argmax valid for zero-token
            # jets; strip it back off before casting to the int8 [L] output
            probs = torch.concatenate([probs, torch.zeros((1, 1, probs.shape[-1]))], dim=1)
            out = torch.argmax(probs, dim=-1)[:, :-1]
            return out.squeeze(0).char()
        return torch.argmax(probs, dim=-1)


class SeqClassProbsOp(ConversionOp):
    """Per-token class probabilities (masked-softmax; the sequence eval-H5 columns).

    The eval-H5 counterpart of `SeqClassIndexOp` (which collapses to the ONNX
    argmax index): the TEST H5 keeps the full prob columns, the ONNX export
    keeps the argmax. Padded positions read ``0.0`` (the masked softmax zeroes
    them).

    Parameters
    ----------
    has_pad_mask : bool, optional
        Whether the stream carries a per-token pad mask, by default True
        (False for a fixed-count query bank).
    """

    def __init__(self, has_pad_mask: bool = True) -> None:
        self.has_pad_mask = bool(has_pad_mask)

    def extra_requires(self, stream: str) -> dict[str, TensorSpec]:
        """Demand the stream's pad mask for the masked softmax (when `has_pad_mask`)."""
        if not self.has_pad_mask:
            return {}
        return {
            f"masks.{stream}": TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
        }

    def output_columns(self, task: Any, run_name: str) -> list[OutputField]:
        """One float per-token H5-only field per class (``class_suffixes``); no ONNX leaf."""
        del run_name
        return [
            OutputField(
                h5_name=px, onnx_name=None, dtype="f4", axis="per_token", final=True
            )
            for px in task.class_suffixes
        ]

    def convert(self, b: Bundle, mode: Mode, *, pred_key: str, stream: str) -> Tensor:
        """Masked-softmax over the class dim; ``[B, L, C]`` per-token probabilities."""
        del mode
        logits = b.get(pred_key)
        mask = b.get(f"masks.{stream}") if self.has_pad_mask else None
        return _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
