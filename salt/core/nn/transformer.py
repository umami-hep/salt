"""Transformer/EncoderLayer and residual wrappers."""

from __future__ import annotations

from functools import partial
from typing import Any, final

import torch
from torch import BoolTensor, Tensor, nn

from salt.core.nn.attention import ATTN_TYPES, Attention, EdgeAttention
from salt.core.nn.glu import GLU
from salt.core.nn.layernorm import _LAYERNORMS
from salt.core.utils.tensor_utils import (
    redo_padding,
    undo_padding,
)

try:
    from mup import MuReadout as _MuReadout

except ImportError:
    _MuReadout = None


class LayerScale(nn.Module):
    """Applies the LayerScale operation from CaiT (stabilizes deep transformers).

    Reference: https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Scale the input by a learnable vector ``gamma``."""
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth / drop-path regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Randomly drop residual paths (no-op outside training)."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class NormResidual(nn.Module):
    """Residual wrapper (PostNorm/PreNorm/NoNorm) around ``fn``, with LayerScale/DropPath.

    ``ls_init=None`` disables LayerScale; ``embed_dim=0`` falls back to
    ``fn.embed_dim``; forwards edge features through for `EdgeAttention`.
    """

    def __init__(
        self,
        fn: GLU | Attention | EdgeAttention,
        norm: str = "LayerNorm",
        ls_init: float | None = None,
        drop_path: float = 0.0,
        embed_dim: int = 0,
        norm_type: str = "pre",
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        dim = embed_dim or fn.embed_dim
        assert dim > 0, "Could not determine embed_dim from fn"
        self.fn = fn
        if self.norm_type != "none":
            self.norm = getattr(_LAYERNORMS, norm)(dim)
        self.ls = LayerScale(dim, ls_init) if ls_init is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

        self.edges = bool(isinstance(fn, EdgeAttention))

    def _forward_edges(self, x: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Residual wrapper around an ``fn`` that also returns edge features."""
        if self.norm_type == "pre":
            fn_out, edge_out = self.fn(self.norm(x), *args, **kwargs)
            res_out = x + self.drop_path(self.ls(fn_out))
            return res_out, edge_out
        if self.norm_type == "post":
            fn_out, edge_out = self.fn(x, *args, **kwargs)
            res_out = self.norm(x + self.drop_path(self.ls(fn_out)))
            return res_out, edge_out
        fn_out, edge_out = self.fn(x, *args, **kwargs)
        res_out = x + self.drop_path(self.ls(fn_out))
        return res_out, edge_out

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]:
        """Apply the residual wrapper; dispatches to the edge-returning path for `EdgeAttention`."""
        if self.edges:
            return self._forward_edges(x, *args, **kwargs)
        if self.norm_type == "pre":
            return x + self.drop_path(self.ls(self.fn(self.norm(x), *args, **kwargs)))
        if self.norm_type == "post":
            return self.norm(x + self.drop_path(self.ls(self.fn(x, *args, **kwargs))))
        return x + self.drop_path(self.ls(self.fn(x, *args, **kwargs)))


@final
class EncoderLayer(nn.Module):
    """Transformer encoder layer: self-attention + feed-forward.

    ``edge_embed_dim > 0`` swaps `Attention` for `EdgeAttention`;
    ``norm_type="hybrid"`` forces qk/v norm and a pre-FFN norm; ``num_dense``
    stacks that many `GLU` blocks in the feed-forward.
    """

    def __init__(
        self,
        embed_dim: int,
        norm: str = "LayerNorm",
        ls_init: float | None = None,
        drop_path: float = 0.0,
        depth: int = 1,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        norm_type: str = "pre",
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        mup: bool = False,
        num_dense: int = 1,
    ) -> None:
        super().__init__()
        self.mup = mup
        self.num_dense = num_dense

        assert num_dense >= 1, "num_dense must be at least 1"
        self.update_edges = update_edges

        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        self.embed_dim = embed_dim
        self.norm_type = norm_type
        if norm_type == "hybrid":
            attn_kwargs["do_qk_norm"] = True
            attn_kwargs["do_v_norm"] = True
            residual_norm_type = "pre" if depth == 0 else "none"
            self.norm = (
                nn.Identity(embed_dim) if depth == 0 else getattr(_LAYERNORMS, norm)(embed_dim)
            )
        else:
            residual_norm_type = norm_type

        if self.mup:
            attn_kwargs["mup"] = True
            dense_kwargs["mup"] = True

        attn_class: type[Attention | EdgeAttention]
        if edge_embed_dim > 0:
            attn_class = EdgeAttention
            attn_kwargs["edge_embed_dim"] = edge_embed_dim
            attn_kwargs["update_edges"] = update_edges

            self.edge_prenorm = getattr(_LAYERNORMS, norm)(edge_embed_dim)
            if self.update_edges:
                self.edge_postnorm = getattr(_LAYERNORMS, norm)(edge_embed_dim)
        else:
            attn_class = Attention

        residual = partial(
            NormResidual,
            norm=norm,
            ls_init=ls_init,
            drop_path=drop_path,
            norm_type=residual_norm_type,
        )
        self.attn = residual(attn_class(embed_dim, **attn_kwargs))
        if num_dense == 1:
            self.dense = residual(GLU(embed_dim, **dense_kwargs))
        else:
            self.dense = nn.Sequential(*[
                residual(GLU(embed_dim, **dense_kwargs)) for _ in range(num_dense)
            ])

    def forward(
        self, x: Tensor, edge_x: Tensor | None = None, **kwargs: Any
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Self-attention + feed-forward; returns updated edges too when ``edge_x`` is given."""
        if edge_x is not None:
            x, edge_x = self.attn(x, edge_x=edge_x, **kwargs)
            if self.update_edges:
                edge_x = edge_x + self.edge_postnorm(edge_x)
        else:
            x = self.attn(x, **kwargs)

        x = self.dense(self.norm(x)) if self.norm_type == "hybrid" else self.dense(x)

        if edge_x is not None:
            return x, edge_x
        return x


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
                "Need the out_dim layer for muP, \
                as this is the last layer of the muP-part of the model"
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
            self.out_proj = nn.Linear(self.embed_dim, self.out_dim)
            if self.mup and _MuReadout is not None:
                self.out_proj = _MuReadout(embed_dim, self.out_dim)
                self.out_proj.bias.data.zero_()
                self.out_proj.weight.data.zero_()
        if self.do_final_norm:
            self.out_norm = getattr(_LAYERNORMS, norm)(self.out_dim)
        if self.num_registers:
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
        if self.num_registers:
            x, pad_mask = self._add_registers(x, pad_mask)

        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)
        mask = torch.cat(list(pad_mask.values()), dim=1) if isinstance(pad_mask, dict) else pad_mask

        # zero-pad edges to the register-augmented sequence length (registers attend
        # to each other/tokens with no edge features, so they contribute zero bias)
        if edge_x is not None:
            edge_x = torch.cat(
                [
                    edge_x,
                    torch.zeros(
                        (
                            edge_x.shape[0],
                            x.shape[1] - edge_x.shape[1],
                            edge_x.shape[2],
                            edge_x.shape[3],
                        ),
                        device=edge_x.device,
                    ),
                ],
                dim=1,
            )
            edge_x = torch.cat(
                [
                    edge_x,
                    torch.zeros(
                        (
                            edge_x.shape[0],
                            edge_x.shape[1],
                            x.shape[1] - edge_x.shape[2],
                            edge_x.shape[3],
                        ),
                        device=edge_x.device,
                    ),
                ],
                dim=2,
            )

        if self.attn_type == "flash-varlen":
            x, kwargs["culens"], kwargs["maxlen"] = undo_padding(x, mask)

        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
            if edge_x is not None:
                x, edge_x = layer(x, edge_x=edge_x, mask=mask, **kwargs)
            else:
                x = layer(x, mask=mask, **kwargs)

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

        reg = self.registers.expand(batch_size, -1, -1)
        if isinstance(x, dict):
            x["REGISTERS"] = reg
        else:
            x = torch.cat([x, reg], dim=1)

        if pad_mask is not None:
            reg_mask = self.register_mask.expand(batch_size, -1)
            if isinstance(pad_mask, dict):
                pad_mask["REGISTERS"] = reg_mask
            else:
                pad_mask = torch.cat([pad_mask, reg_mask], dim=-1)

        return x, pad_mask
