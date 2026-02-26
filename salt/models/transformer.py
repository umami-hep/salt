"""Efficient Transformer implementation.

Updated transformer implementation based on
https://github.com/mistralai/mistral-src

Features
--------
- native SDP kernels (including flash)
- gated linear units https://arxiv.org/abs/2002.05202
- RMSNorm https://arxiv.org/abs/1910.07467
"""

from functools import partial
from typing import Any, final

import torch
from torch import BoolTensor, Tensor, nn

import salt.models.layernorm as layernorms
from salt.models.attention import ATTN_TYPES, Attention, EdgeAttention
from salt.stypes import Tensors
from salt.utils.tensor_utils import redo_padding, undo_padding

try:
    from mup import MuReadout as _MuReadout

except ImportError:
    _MuReadout = None


def change_attn_backends(module: nn.Module, backend: str) -> None:
    """Recursively change the attention backend on a module and its children.

    Used primarily to switch back to ``torch-math`` for ONNX exports.

    Parameters
    ----------
    module : nn.Module
        Root module to traverse.
    backend : str
        Backend name, one of ``{"torch-math", "torch-flash", "torch-meff", "flash-varlen"}``.
    """
    if isinstance(module, Transformer):
        module.set_backend(backend)
        return
    if isinstance(module, Attention):
        module.set_backend(backend)
        return
    for child in module.children():
        change_attn_backends(child, backend)


class GLU(nn.Module):
    """Dense update with a (gated) linear unit.

    See https://arxiv.org/abs/2002.05202.

    Parameters
    ----------
    embed_dim : int
        Input/output embedding dimension.
    hidden_dim : int | None, optional
        Hidden dimension. If ``None``, defaults to ``2 * embed_dim``.
    activation : str, optional
        Name of the activation class in ``torch.nn`` (e.g., ``"SiLU"``).
    dropout : float, optional
        Dropout probability. The default is ``0.0``.
    bias : bool, optional
        Whether to include bias terms. The default is ``True``.
    gated : bool, optional
        If ``True``, uses a gated branch (splits hidden in two). The default is ``False``.
    mup : bool, optional
        Whether to use μP parameterization. The default is ``False``.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
        mup: bool = False,
    ):
        super().__init__()
        self.mup = mup

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        if self.mup:
            for proj in [self.in_proj, self.out_proj]:
                nn.init.normal_(proj.weight, mean=0.0, std=1.0 / (proj.weight.shape[0] ** 0.5))
                if bias:
                    nn.init.zeros_(proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the GLU block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[B, L, D]``.

        Returns
        -------
        Tensor
            Output tensor of shape ``[B, L, D]``.
        """
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)


class LayerScale(nn.Module):
    """Applies the LayerScale operation from CaiT (stabilizes deep transformers).

    Reference: https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Scale the input by a learnable vector ``gamma``.

        Parameters
        ----------
        x : Tensor
            Input as Tensor

        Returns
        -------
        Tensor
            Scaled inputs as Tensor
        """
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth / drop-path regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Randomly drop residual paths during training.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor with stochastic depth applied when training.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class NormResidual(nn.Module):
    """Residual wrapper with normalization, LayerScale, and DropPath.

    Can represent common transformer patterns:

    - PostNorm: ``x = norm(x + drop(scale * fn(x)))``
    - PreNorm: ``x = x + drop(scale * fn(norm(x)))``
    - NoNorm:  ``x = x + drop(scale * fn(x))``

    Parameters
    ----------
    fn : GLU | Attention | EdgeAttention
        The wrapped non-resizing module.
    norm : str, optional
        Normalization class name from :mod:`salt.models.layernorm`. The default is ``"LayerNorm"``.
    ls_init : float | None, optional
        Initial value for LayerScale. If ``None``, LayerScale is disabled.
    drop_path : float, optional
        Drop-path rate for stochastic depth. The default is ``0.0``.
    embed_dim : int, optional
        Input/output dimension. If ``0``, attempts to read ``fn.embed_dim``.
    norm_type : str, optional
        One of ``{"pre", "post", "none"}``. The default is ``"pre"``.
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
            self.norm = getattr(layernorms, norm)(dim)
        self.ls = LayerScale(dim, ls_init) if ls_init is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

        self.edges = bool(isinstance(fn, EdgeAttention))

    def _forward_edges(self, x: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Apply residual wrapper around ``fn`` that returns edge features.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        *args : Any
            Positional arguments forwarded to ``fn``.
        **kwargs : Any
            Keyword arguments forwarded to ``fn``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output tensor with residual, normalization (if enabled), LayerScale, and DropPath.
            Also returns edge features from ``fn``.
        """
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
        """Apply residual wrapper around ``fn``.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        *args : Any
            Positional arguments forwarded to ``fn``.
        **kwargs : Any
            Keyword arguments forwarded to ``fn``.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            Output tensor with residual, normalization (if enabled), LayerScale, and DropPath.
            If ``fn`` is an :class:`EdgeAttention`, returns a tuple of ``(output, edge_out)``.
        """
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

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    norm : str, optional
        Normalization style (class name from :mod:`salt.models.layernorm`).
        The default is ``"LayerNorm"``.
    ls_init : float | None, optional
        Initial LayerScale value. If ``None``, LayerScale is disabled.
    drop_path : float, optional
        Drop-path rate. The default is ``0.0``.
    depth : int, optional
        Layer depth index, used for differential attention weighting. The default is ``1``.
    dense_kwargs : dict | None, optional
        Keyword args for :class:`GLU`.
    attn_kwargs : dict | None, optional
        Keyword args for :class:`Attention`.
    norm_type : str, optional
        One of ``{"pre", "post", "hybrid"}``. The default is ``"pre"``.
    edge_embed_dim : int, optional
        Model embedding dimension for edge features. The default is ``0``.
    update_edges : bool, optional
        If ``True``, edge features are updated after attention. The default is ``False``
    mup: bool, optional
        Whether to use μP parameterization. The default is ``False``.
    num_dense: int, optional
        Number of dense layers to stack in the feed-forward block. The default is ``1``.
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

        # Safe defaults
        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        # Attributes
        self.embed_dim = embed_dim
        self.norm_type = norm_type
        if norm_type == "hybrid":
            attn_kwargs["do_qk_norm"] = True
            attn_kwargs["do_v_norm"] = True
            residual_norm_type = "pre" if depth == 0 else "none"
            self.norm = (
                nn.Identity(embed_dim) if depth == 0 else getattr(layernorms, norm)(embed_dim)
            )
        else:
            residual_norm_type = norm_type

        if self.mup:
            attn_kwargs["mup"] = True
            dense_kwargs["mup"] = True

        # Choose attention type
        attn_class: type[Attention | EdgeAttention]
        if edge_embed_dim > 0:
            attn_class = EdgeAttention
            attn_kwargs["edge_embed_dim"] = edge_embed_dim
            attn_kwargs["update_edges"] = update_edges

            self.edge_prenorm = getattr(layernorms, norm)(edge_embed_dim)
            if self.update_edges:
                self.edge_postnorm = getattr(layernorms, norm)(edge_embed_dim)
        else:
            attn_class = Attention

        # Submodules
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
        """Apply self-attention and feed-forward.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[B, L, D]``.
        edge_x : Tensor | None, optional
            Edge feature tensor of shape ``[B, L, L, D_edge]`` (if using edge features).
        **kwargs : Any
            Extra arguments forwarded to the :class:`Attention` module
            (e.g., ``mask``, ``attn_mask``, etc.).

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            If ``edge_x is None``: the updated token embeddings ``x`` of shape ``[B, N, D]``.
            If ``edge_x`` is provided: ``(x, edge_x)`` with updated tokens and edges.
        """
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
class DecoderLayer(nn.Module):
    """Transformer decoder layer with self- and cross-attention.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    norm : str, optional
        Normalization style (class name from :mod:`salt.models.layernorm`).
        The default is ``"LayerNorm"``.
    ls_init : float | None, optional
        Initial LayerScale value. The default is ``1e-3``.
    drop_path : float, optional
        Drop-path rate. The default is ``0.0``.
    dense_kwargs : dict | None, optional
        Keyword args for :class:`GLU`.
    attn_kwargs : dict | None, optional
        Keyword args for :class:`Attention`.
    norm_type : str, optional
        One of ``{"pre", "post", "none"}``. The default is ``"pre"``.
    """

    def __init__(
        self,
        embed_dim: int,
        norm: str = "LayerNorm",
        ls_init: float | None = 1e-3,
        drop_path: float = 0.0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        norm_type: str = "pre",
    ):
        super().__init__()

        # Safe defaults
        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}

        # Attributes
        self.embed_dim = embed_dim

        # Submodules
        residual = partial(
            NormResidual, norm=norm, ls_init=ls_init, drop_path=drop_path, norm_type=norm_type
        )
        self.self_attn = residual(Attention(embed_dim=embed_dim, **attn_kwargs))
        self.cross_attn = residual(Attention(embed_dim=embed_dim, **attn_kwargs))
        self.dense = residual(GLU(embed_dim, **dense_kwargs))

    def forward(
        self,
        x: Tensor,
        *,  # Indicates that kv is required
        kv: Tensor,
        mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply self-attention, cross-attention, and feed-forward.

        Parameters
        ----------
        x : Tensor
            Decoder input of shape ``[B, L, D]``.
        kv : Tensor
            Encoder memory of shape ``[B, L_enc, D]``.
        mask : Tensor | None, optional
            Padding mask for ``x`` where padded positions are ``True``.
        kv_mask : Tensor | None, optional
            Padding mask for ``kv`` where padded positions are ``True``.

        Returns
        -------
        Tensor
            Output tensor of shape ``[B, L, D]``.
        """
        x = self.self_attn(x, kv_mask=mask)
        x = self.cross_attn(x, kv=kv, kv_mask=kv_mask)
        return self.dense(x)


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
        Normalization style (class name from :mod:`salt.models.layernorm`).
        The default is ``"LayerNorm"``.
    attn_type : str, optional
        Attention backend, one of ``{"torch-math", "torch-flash", "torch-meff", "flash-varlen"}``.
        The default is ``"torch-math"``.
    do_final_norm : bool, optional
        Whether to apply a final normalization layer. The default is ``True``.
    num_registers : int, optional
        Number of learned register tokens appended to the end of the sequence. The default is ``1``.
    drop_registers : bool, optional
        If ``True``, registers are dropped from outputs. The default is ``False``.
    edge_embed_dim : int, optional
        Model embedding dimension for edge features. The default is ``0``.
    update_edges : bool, optional
        If ``True``, edge features are updated after attention. The default is ``False``
    mup: bool, optional
        Whether to use μP parameterization. The default is ``False``.
    **kwargs : Any
        Extra keyword arguments forwarded to :class:`EncoderLayer` (e.g., ``attn_kwargs``,
        ``dense_kwargs``, ``ls_init``, etc.).

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

        # Check the inputs
        if num_registers < 1:
            raise ValueError(
                "Some jets have no tracks, which causes NaNs in the attention scores. "
                "To avoid this, set num_registers to at least 1",
            )

        # Attributes
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

        # Set the attention type if no edge features are used
        if edge_embed_dim == 0:
            kwargs["attn_kwargs"]["attn_type"] = self.attn_type

        # Submodules
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

        # Only set the attention type if no edge features are used
        if self.edge_embed_dim == 0:
            # Check and set the attention type
            assert self.attn_type in ATTN_TYPES, "Invalid attention type!"
            self.set_backend(self.attn_type)

        # Optional submodules
        if self.do_out_proj:
            self.out_proj = nn.Linear(self.embed_dim, self.out_dim)
            if self.mup and _MuReadout is not None:
                self.out_proj = _MuReadout(embed_dim, self.out_dim)
                self.out_proj.bias.data.zero_()
                self.out_proj.weight.data.zero_()
        if self.do_final_norm:
            self.out_norm = getattr(layernorms, norm)(self.out_dim)
        if self.num_registers:
            self.registers = nn.Parameter(
                torch.normal(torch.zeros((self.num_registers, self.embed_dim)), std=1e-4)
            )
            self.register_buffer("register_mask", torch.zeros(num_registers, dtype=torch.bool))
        self.featurewise = nn.ModuleList()

    def set_backend(self, attn_type: str) -> None:
        """Set the attention backend for all layers.

        Parameters
        ----------
        attn_type : str
            Backend name to apply to all encoder layers.
        """
        self.attn_type = attn_type
        for layer in self.layers:
            self.attn_type = layer.attn.fn.set_backend(self.attn_type)

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        pad_mask: BoolTensor | dict[str, BoolTensor],
        inputs: Tensors | None = None,
        edge_x: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, BoolTensor | dict[str, BoolTensor]]:
        """Run the encoder stack.

        Parameters
        ----------
        x : Tensor | dict[str, Tensor]
            Input sequence(s) of shape ``[B, L, D]`` or dict of such tensors.
        pad_mask : BoolTensor | dict[str, BoolTensor]
            Padding mask(s) aligned with ``x``; padded positions are ``True``.
        inputs : Tensors | None, optional
            Original input dictionary for featurewise transforms (if any).
        edge_x : Tensor | None, optional
            Edge feature tensor of shape ``[B, L, L, D_edge]`` (if using edge features).
        **kwargs : Any
            Extra arguments forwarded to encoder layers (e.g., attention masks or
            varlen flash arguments).

        Returns
        -------
        tuple[Tensor, BoolTensor | dict[str, BoolTensor]]
            Tuple of ``(encoded, pad_mask)`` where ``encoded`` has shape ``[B, L, D_out]``.
        """
        # Add the registers to the sequence and the mask
        if self.num_registers:
            x, pad_mask = self._add_registers(x, pad_mask)

        # Combine the input sequences if they are dictionaries (don't overwrite pad_mask)
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)
        mask = torch.cat(list(pad_mask.values()), dim=1) if isinstance(pad_mask, dict) else pad_mask

        # Pad edges by num_registers if using edge features
        if edge_x is not None:
            edge_x = torch.cat(
                [
                    edge_x,
                    torch.zeros(
                        (edge_x.shape[0], self.num_registers, edge_x.shape[2], edge_x.shape[3]),
                        device=edge_x.device,
                    ),
                ],
                dim=1,
            )
            edge_x = torch.cat(
                [
                    edge_x,
                    torch.zeros(
                        (edge_x.shape[0], edge_x.shape[1], self.num_registers, edge_x.shape[3]),
                        device=edge_x.device,
                    ),
                ],
                dim=2,
            )

        # If using the varlen backend, pack the sequence and store the cumulative lengths
        if self.attn_type == "flash-varlen":
            x, kwargs["culens"], kwargs["maxlen"] = undo_padding(x, mask)

        # Run through the main transformer encoder layers
        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
            if edge_x is not None:
                x, edge_x = layer(x, edge_x=edge_x, mask=mask, **kwargs)
            else:
                x = layer(x, mask=mask, **kwargs)

        # Run through the optional layers
        if self.do_out_proj:
            x = self.out_proj(x)
        if self.do_final_norm:
            x = self.out_norm(x)

        # If using the varlen backend, unpack the sequence
        if self.attn_type == "flash-varlen":
            x = redo_padding(x, mask)

        # Optionally drop the registers from the output
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
        """Add the learnable registers to the end of the input sequence (and mask).

        Parameters
        ----------
        x : Tensor | dict[str, Tensor]
            Input sequence(s) of shape ``[B, L, D]`` or dict of such tensors.
        pad_mask : BoolTensor | dict[str, BoolTensor] | None
            Padding mask(s) aligned with ``x``; padded positions are ``True``.

        Returns
        -------
        tuple[Tensor | dict[str, Tensor], BoolTensor | dict[str, BoolTensor] | None]
            Updated ``(x, pad_mask)`` including appended registers.
        """
        # Get the batch size and expand the registers to match
        batch_size = next(iter(x.values())).size(0) if isinstance(x, dict) else x.size(0)

        # Add as a key or concatenate at the end
        reg = self.registers.expand(batch_size, -1, -1)
        if isinstance(x, dict):
            x["REGISTERS"] = reg
        else:
            x = torch.cat([x, reg], dim=1)

        # Also include a mask for the registers
        if pad_mask is not None:
            reg_mask = self.register_mask.expand(batch_size, -1)
            if isinstance(pad_mask, dict):
                pad_mask["REGISTERS"] = reg_mask
            else:
                pad_mask = torch.cat([pad_mask, reg_mask], dim=-1)

        return x, pad_mask
