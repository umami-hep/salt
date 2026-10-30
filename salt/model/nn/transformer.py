"""Transformer/EncoderLayer and residual wrappers."""

from __future__ import annotations

from functools import partial
from typing import Any, final

import torch
from torch import BoolTensor, Tensor, nn

from salt.model.nn.attention import ATTN_TYPES, Attention, EdgeAttention
from salt.model.nn.dense import GLU
from salt.model.nn.layernorm import _LAYERNORMS
from salt.utils.tensor_utils import (
    redo_padding,
    undo_padding,
)

try:
    from mup import MuReadout as _MuReadout
except ImportError:
    _MuReadout = None


class NormResidual(nn.Module):
    """Residual wrapper around ``fn`` (PreNorm/PostNorm/NoNorm by ``norm_type``).

    ``pre``: ``x + fn(norm(x))``; ``post``: ``norm(x + fn(x))``; anything else (``none``):
    ``x + fn(x)``. ``embed_dim=0`` falls back to ``fn.embed_dim``; forwards edge features
    through for `EdgeAttention`.
    """

    def __init__(
        self,
        fn: GLU | Attention | EdgeAttention,
        norm: str = "LayerNorm",
        embed_dim: int = 0,
        norm_type: str = "pre",
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        dim = embed_dim or fn.embed_dim
        assert dim > 0, "Could not determine embed_dim from fn"
        self.fn = fn
        if self.norm_type != "none":
            self.norm = _LAYERNORMS[norm](dim)
        self.edges = isinstance(fn, EdgeAttention)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]:
        """Apply the residual wrapper; returns ``(out, edge_out)`` for an `EdgeAttention` ``fn``."""
        out = self.fn(self.norm(x) if self.norm_type == "pre" else x, *args, **kwargs)
        fn_out, edge_out = out if self.edges else (out, None)
        res = x + fn_out
        if self.norm_type == "post":
            res = self.norm(res)
        return (res, edge_out) if self.edges else res


@final
class EncoderLayer(nn.Module):
    """Transformer encoder layer: self-attention + feed-forward.

    ``edge_embed_dim > 0`` swaps `Attention` for `EdgeAttention`;
    ``norm_type="hybrid"`` forces qk/v norm and a pre-FFN norm.
    """

    def __init__(
        self,
        embed_dim: int,
        norm: str = "LayerNorm",
        depth: int = 1,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        norm_type: str = "pre",
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        mup: bool = False,
    ) -> None:
        super().__init__()
        self.mup = mup
        self.update_edges = update_edges
        self.embed_dim = embed_dim
        self.norm_type = norm_type
        attn_kwargs = dict(attn_kwargs or {})
        dense_kwargs = dict(dense_kwargs or {})

        residual_norm_type = norm_type
        if norm_type == "hybrid":
            # layer 0 keeps pre-norm residuals; deeper layers use plain residuals + a pre-FFN norm
            attn_kwargs |= {"do_qk_norm": True, "do_v_norm": True}
            residual_norm_type = "pre" if depth == 0 else "none"
            self.norm = nn.Identity(embed_dim) if depth == 0 else _LAYERNORMS[norm](embed_dim)
        if self.mup:
            attn_kwargs["mup"] = dense_kwargs["mup"] = True

        attn_class: type[Attention | EdgeAttention] = Attention
        if edge_embed_dim > 0:
            attn_class = EdgeAttention
            attn_kwargs |= {"edge_embed_dim": edge_embed_dim, "update_edges": update_edges}
            self.edge_prenorm = _LAYERNORMS[norm](edge_embed_dim)
            if self.update_edges:
                self.edge_postnorm = _LAYERNORMS[norm](edge_embed_dim)

        residual = partial(NormResidual, norm=norm, norm_type=residual_norm_type)
        self.attn = residual(attn_class(embed_dim, **attn_kwargs))
        self.dense = residual(GLU(embed_dim, **dense_kwargs))

    def forward(
        self, x: Tensor, edge_x: Tensor | None = None, **kwargs: Any
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Self-attention + feed-forward; returns updated edges too when ``edge_x`` is given."""
        if edge_x is None:
            x = self.attn(x, **kwargs)
        else:
            x, edge_x = self.attn(x, edge_x=edge_x, **kwargs)
            if self.update_edges:
                edge_x = edge_x + self.edge_postnorm(edge_x)
        x = self.dense(self.norm(x) if self.norm_type == "hybrid" else x)
        return x if edge_x is None else (x, edge_x)


def _append_registers(
    seq: Tensor | dict[str, Tensor], reg: Tensor, dim: int
) -> Tensor | dict[str, Tensor]:
    """Append ``reg`` as a ``"REGISTERS"`` entry of a dict (in place), else concat on ``dim``."""
    if isinstance(seq, dict):
        seq["REGISTERS"] = reg
        return seq
    return torch.cat([seq, reg], dim=dim)


@final
class Transformer(nn.Module):
    """Transformer encoder stack with optional registers and output projection.

    Parameters
    ----------
    num_layers : int
        Number of encoder layers.
    embed_dim : int
        Embedding dimension.
    out_dim : int | None, optional
        Optional output projection dimension. If ``None``, equals ``embed_dim``.
    norm : str, optional
        Normalization style. The default is ``"LayerNorm"``.
    attn_type : str, optional
        Attention backend. The default is ``"torch-math"``.
    do_final_norm : bool, optional
        Whether to apply a final normalization layer. The default is ``True``.
    num_registers : int, optional
        Number of learned register tokens. The default is ``1``.
    drop_registers : bool, optional
        If ``True``, registers are dropped from outputs. The default is ``False``.
    edge_embed_dim : int, optional
        Model embedding dimension for edge features. The default is ``0``.
    update_edges : bool, optional
        If ``True``, edge features are updated after attention. The default is ``False``
    mup: bool, optional
        Whether to use μP parameterization. The default is ``False``.
    **kwargs : Any
        Extra keyword arguments forwarded to :class:`EncoderLayer`.

    Raises
    ------
    ValueError
        If ``num_registers < 1``.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        out_dim: int | None = None,
        norm: str = "LayerNorm",
        attn_type: str = "torch-math",
        do_final_norm: bool = True,
        num_registers: int = 1,
        drop_registers: bool = False,
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        mup: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if num_registers < 1:
            raise ValueError(
                "Some global objects (graphs) might have no constituents (nodes), "
                "which causes NaNs in the attention scores. "
                "To avoid this, set num_registers to at least 1",
            )

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.out_dim = out_dim or embed_dim
        self.do_final_norm = do_final_norm
        self.do_out_proj = out_dim is not None
        self.attn_type = attn_type
        self.num_registers = num_registers
        self.drop_registers = drop_registers
        self.edge_embed_dim = edge_embed_dim
        self.update_edges = update_edges
        self.mup = mup

        if self.update_edges:
            assert edge_embed_dim > 0, "Cannot update edges with edge_embed_dim=0"
        if self.mup:
            assert _MuReadout is not None, "mup is not installed!"
            assert self.do_out_proj, (
                "Need the out_dim layer for muP, as this is the last layer of the muP-part of "
                "the model"
            )

        # attn_type is meaningless for EdgeAttention (edge_embed_dim > 0): it has no
        # pluggable backend, so only set it when the plain Attention path is used
        if edge_embed_dim == 0:
            kwargs["attn_kwargs"]["attn_type"] = self.attn_type

        self.layers = torch.nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                norm=norm,
                depth=depth,
                edge_embed_dim=edge_embed_dim,
                update_edges=update_edges,
                **kwargs,
            )
            for depth in range(num_layers)
        ])

        if self.edge_embed_dim == 0:
            assert self.attn_type in ATTN_TYPES, "Invalid attention type!"
            self.set_backend(self.attn_type)

        if self.do_out_proj:
            # the plain Linear is always built first: its init draws from the RNG either way
            self.out_proj = nn.Linear(self.embed_dim, self.out_dim)
            if self.mup:
                self.out_proj = _MuReadout(embed_dim, self.out_dim)
                self.out_proj.bias.data.zero_()
                self.out_proj.weight.data.zero_()
        if self.do_final_norm:
            self.out_norm = _LAYERNORMS[norm](self.out_dim)
        self.registers = nn.Parameter(
            torch.normal(torch.zeros((self.num_registers, self.embed_dim)), std=1e-4)
        )
        self.register_buffer("register_mask", torch.zeros(num_registers, dtype=torch.bool))
        self.featurewise = nn.ModuleList()

    def set_backend(self, attn_type: str) -> None:
        """Set the attention backend for all layers."""
        self.attn_type = attn_type
        for layer in self.layers:
            self.attn_type = layer.attn.fn.set_backend(self.attn_type)

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        pad_mask: BoolTensor | dict[str, BoolTensor],
        inputs: Any | None = None,
        edge_x: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, BoolTensor | dict[str, BoolTensor]]:
        """Run the encoder stack; returns ``(encoded [B, L, D_out], pad_mask)``."""
        x, pad_mask = self._add_registers(x, pad_mask)
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)
        mask = torch.cat(list(pad_mask.values()), dim=1) if isinstance(pad_mask, dict) else pad_mask

        # zero-pad edges to the register-augmented sequence length (registers attend
        # to each other/tokens with no edge features, so they contribute zero bias)
        if edge_x is not None:
            pad = (edge_x.shape[0], x.shape[1] - edge_x.shape[1], edge_x.shape[2], edge_x.shape[3])
            edge_x = torch.cat([edge_x, torch.zeros(pad, device=edge_x.device)], dim=1)
            pad = (edge_x.shape[0], edge_x.shape[1], x.shape[1] - edge_x.shape[2], edge_x.shape[3])
            edge_x = torch.cat([edge_x, torch.zeros(pad, device=edge_x.device)], dim=2)

        if self.attn_type == "flash-varlen":
            x, kwargs["culens"], kwargs["maxlen"] = undo_padding(x, mask)

        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
            if edge_x is None:
                x = layer(x, mask=mask, **kwargs)
            else:
                x, edge_x = layer(x, edge_x=edge_x, mask=mask, **kwargs)

        if self.do_out_proj:
            x = self.out_proj(x)
        if self.do_final_norm:
            x = self.out_norm(x)

        if self.attn_type == "flash-varlen":
            x = redo_padding(x, mask)

        if self.drop_registers:
            x = x[:, : -self.num_registers]
            if isinstance(pad_mask, dict):
                del pad_mask["REGISTERS"]
            elif isinstance(pad_mask, Tensor):
                pad_mask = pad_mask[:, : -self.num_registers]

        return x, pad_mask

    def _add_registers(
        self, x: Tensor | dict[str, Tensor], pad_mask: BoolTensor | dict[str, BoolTensor] | None
    ) -> tuple[Tensor | dict[str, Tensor], BoolTensor | dict[str, BoolTensor] | None]:
        """Append the learnable registers to the input sequence (and mask)."""
        batch_size = next(iter(x.values())).size(0) if isinstance(x, dict) else x.size(0)
        x = _append_registers(x, self.registers.expand(batch_size, -1, -1), dim=1)
        if pad_mask is not None:
            pad_mask = _append_registers(
                pad_mask, self.register_mask.expand(batch_size, -1), dim=-1
            )
        return x, pad_mask
