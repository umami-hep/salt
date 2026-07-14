"""GlobalAttentionPooling GraphModule."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import (
    IO,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.core.nn.base import SaltModelModule
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.transformer_encoder import _SEQ_LEN
from salt.core.utils.tensor_utils import (
    flatten_tensor_dict,
    masked_softmax,
)


class GlobalAttentionPooling(SaltModelModule):
    """Config-constructed global attention pooling, with explicit input/out ports.

    The gate layer (``gate_nn``) is built at `bind` once the input width is
    known. Pools the register-augmented sequence with the post-register mask
    order streams-then-REGISTERS. Two wirings, one class:

    - **With an encoder**: ``input`` is ``encoded.seq`` and `TransformerEncoder`
      publishes ``masks.registers`` (register rows sit after every stream).
    - **Encoder-less** (DiPS/DeepSets family, every regression config): no
      ``encoder:`` block, so nothing produces ``masks.registers``. ``input``
      points at ``seq.x`` and ``masks.registers`` is declared OPTIONAL — absent
      from the plan when no producer exists, so the pad dict is just
      ``{"seq": seq.mask}``.
    """

    def __init__(self, input: str = "encoded.seq", out: str = "pooled.global") -> None:  # noqa: A002
        """Capture the explicit input/output ports."""
        super().__init__()
        self.input_key = input
        self.out_key = out
        # the gate layer — built at bind (input width known there)
        self.gate_nn: nn.Linear | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + masks -> the pooled vector (width symbol shared with the input)."""
        del mode
        width = sym_dim("D", self.name)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", sym_dim("L", self.name), width), dtype="float32"
                ),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                # OPTIONAL: produced by `TransformerEncoder` on the WITH-encoder path,
                # ABSENT on the encoder-less path — the planner drops it when no
                # module produces it, so encoder-less configs still plan-compile.
                "masks.registers": TensorSpec(
                    shape=("B", sym_dim("R", "registers")),
                    dtype="bool",
                    kind="pad_mask",
                    optional=True,
                ),
            }),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", width), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the gate layer with the inferred input width."""
        self.gate_nn = nn.Linear(schema.width(self.input_key), 1)

    def pool(
        self,
        x: dict[str, Tensor] | dict,
        pad_mask: dict | None = None,
    ) -> Tensor:
        """Apply global attention pooling over the concatenated node embeddings.

        Returns
        -------
        Tensor
            Pooled tensor of shape ``[B, D]``.
        """
        x_flat = flatten_tensor_dict(x, exclude=["objects"])

        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1).unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x_flat), pad_mask, dim=1)
        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x_flat.shape[0], 1, x_flat.shape[2]), device=x_flat.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x_flat = torch.cat([x_flat, x_pad], dim=1)

        return (x_flat * weights).sum(dim=1)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pool the sequence with the (optionally register-augmented) mask dict."""
        del mode
        assert self.gate_nn is not None, "forward before bind()"
        x = {"seq": b.get(self.input_key)}
        # streams-then-REGISTERS dict order is numerics-critical: pooling cats
        # mask values in dict order. The encoder-less path has no register row,
        # so the pad dict is just {"seq": seq.mask}.
        pad = {"seq": b.get("seq.mask")}
        if "masks.registers" in b:
            pad["REGISTERS"] = b.get("masks.registers")
        return {self.out_key: self.pool(x, pad_mask=pad)}
