"""GlobalAttentionPooling and ClassAttentionPooling GraphModules."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from salt.graph.bundle import Bundle
from salt.graph.spec import (
    IO,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.model.base import SaltModelModule
from salt.model.bind import ResolvedSchema
from salt.model.modules.transformer_encoder import _SEQ_LEN
from salt.model.nn.transformer import DecoderLayer
from salt.utils.tensor_utils import masked_softmax


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
        requires = {
            self.input_key: TensorSpec(
                shape=("B", sym_dim("L", self.name), width), dtype="float32"
            ),
            "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
            # OPTIONAL: produced by `TransformerEncoder` on the WITH-encoder path,
            # ABSENT on the encoder-less path — the planner drops it when no
            # module produces it, so encoder-less configs still plan-compile.
            "masks.registers": TensorSpec(
                shape=("B", sym_dim("R", "registers")), dtype="bool", kind="pad_mask", optional=True
            ),
        }
        produces = {self.out_key: TensorSpec(shape=("B", width), dtype="float32")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

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
        x_flat = torch.cat([v for k, v in x.items() if k != "objects"], dim=1)

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
        # streams-then-REGISTERS dict order is numerics-critical: pooling cats
        # mask values in dict order. The encoder-less path has no register row,
        # so the pad dict is just {"seq": seq.mask}.
        pad = {"seq": b.get("seq.mask")}
        if "masks.registers" in b:
            pad["REGISTERS"] = b.get("masks.registers")
        return {self.out_key: self.pool({"seq": b.get(self.input_key)}, pad_mask=pad)}


class ClassAttentionPooling(SaltModelModule):
    """Class attention pooling (ParT / DeParT / CaiT style).

    A single learned class token cross-attends to the encoded sequence through
    `num_layers` cross-attention-only `DecoderLayer` blocks, and the resulting
    token is the pooled vector. The class token is prepended to the keys/values
    of every block, so it is always a valid key and fully padded jets stay
    NaN-free. Wired like `GlobalAttentionPooling`: the register rows are attended
    to when the encoder publishes ``masks.registers`` (set ``drop_registers`` on
    the encoder to pool over the constituents only). The class token and blocks
    are built at `bind` once the input width is known.
    """

    def __init__(
        self,
        input: str = "encoded.seq",  # noqa: A002
        out: str = "pooled.global",
        num_layers: int = 1,
        ls_init: float | None = 1e-3,
        dense: dict[str, Any] | None = None,
        attention: dict[str, Any] | None = None,
        norm: str = "LayerNorm",
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        input : str, optional
            The encoded sequence to pool, by default ``"encoded.seq"``.
        out : str, optional
            Produced pooled key, by default ``"pooled.global"``.
        num_layers : int, optional
            Number of cross-attention blocks, by default 1.
        ls_init : float | None, optional
            Initial LayerScale value of each block, by default ``1e-3``; None disables it.
        dense : dict[str, Any] | None, optional
            Feed-forward (GLU) kwargs of each block (``activation``, ``gated``, ...),
            by default None.
        attention : dict[str, Any] | None, optional
            Attention kwargs of each block (``num_heads``, ...), by default None.
        norm : str, optional
            Normalisation layer name, by default ``"LayerNorm"``.
        """
        super().__init__()
        self.input_key = input
        self.out_key = out
        self.num_layers = num_layers
        self.ls_init = ls_init
        self.dense_cfg = dict(dense or {})
        self.attn_cfg = dict(attention or {})
        self.norm = norm
        # the class token and blocks — built at bind (input width known there)
        self.class_token: nn.Parameter | None = None
        self.layers: nn.ModuleList | None = None

    # same ports as GlobalAttentionPooling
    declare_io = GlobalAttentionPooling.declare_io

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the class token and the cross-attention blocks with the inferred input width."""
        dim = schema.width(self.input_key)
        self.class_token = nn.Parameter(torch.normal(torch.zeros((1, dim)), std=1e-4))
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=dim,
                norm=self.norm,
                ls_init=self.ls_init,
                dense_kwargs=self.dense_cfg,
                attn_kwargs=self.attn_cfg,
                self_attn=False,
            )
            for _ in range(self.num_layers)
        ])

    def pool(self, x: Tensor, mask: Tensor) -> Tensor:
        """Pool ``x`` ``[B, L, D]`` with its pad mask ``[B, L]`` (True = padded) -> ``[B, D]``."""
        q = self.class_token.expand(x.shape[0], -1, -1)
        valid = torch.zeros((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)
        kv_mask = torch.cat([valid, mask], dim=1)
        for layer in self.layers:
            q = layer(q, kv=torch.cat([q, x], dim=1), kv_mask=kv_mask)
        return q.squeeze(1)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pool the sequence with the (optionally register-augmented) mask."""
        del mode
        assert self.layers is not None, "forward before bind()"
        masks = [b.get("seq.mask")]
        if "masks.registers" in b:
            masks.append(b.get("masks.registers"))
        return {self.out_key: self.pool(b.get(self.input_key), torch.cat(masks, dim=1))}
