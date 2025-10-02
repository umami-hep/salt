"""Efficient Transformer implementation.

Updated transformer implementation based on
https://github.com/mistralai/mistral-src

Features
--------
- native SDP kernels (including flash)
- gated linear units https://arxiv.org/abs/2002.05202
- RMSNorm https://arxiv.org/abs/1910.07467
"""

import math
import warnings
from functools import partial
from typing import Any

import torch
from torch import BoolTensor, Size, Tensor, nn
from torch.nn import functional

import salt.models.layernorm as layernorms
from salt.stypes import Tensors
from salt.utils.tensor_utils import redo_padding, undo_padding

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None

ATTN_TYPES = ["torch-math", "torch-flash", "torch-meff", "flash-varlen"]


def merge_masks(
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Size,
) -> BoolTensor | None:
    """Create a full attention mask which incorporates padding information.

    Using PyTorch transformer convention for padding:
        ``False``: real token
        ``True``: zero padded

    Using PyTorch transformer convention for attention mask:
        ``False``: not allowed in attention
        ``True``: allowed in attention

    We design the mask such that padded tokens can't **send** information, but
    they can **receive** it. This prevents NaNs in attention scores due to softmax.

    Parameters
    ----------
    kv_mask : BoolTensor | None
        Mask for keys/values of shape ``[B, L_kv]`` where padded positions are ``True``.
    attn_mask : BoolTensor | None
        Full attention mask of shape ``[B, L_q, L_kv]`` where allowed positions are ``True``.
    q_shape : Size
        Shape of the query tensor, expected as ``(B, L_q, D)``.

    Returns
    -------
    BoolTensor | None
        Combined mask of shape ``[B, 1, L_q, L_kv]`` (broadcastable over heads), or ``None``.
    """
    mask = None

    # If the kv_mask exists, ensure padded tokens never send information
    if kv_mask is not None:
        mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        mask = ~mask  # convert the mask so that True indicates a valid token

    # Combine with the explicit attention mask if present
    if attn_mask is not None:
        mask = attn_mask if mask is None else attn_mask & mask

    # Unsqueeze for head broadcasting
    if mask is not None:
        mask = mask.unsqueeze(1)

    return mask


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int) -> tuple[Tensor, Tensor]:
    """Repeat keys and values along a dimension.

    Parameters
    ----------
    keys : Tensor
        Key tensor.
    values : Tensor
        Value tensor.
    repeats : int
        Number of repeats.
    dim : int
        Dimension along which to repeat.

    Returns
    -------
    tuple[Tensor, Tensor]
        Repeated ``(keys, values)`` tensors.
    """
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


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
    if isinstance(module, TransformerV2):
        module.set_backend(backend)
        return
    if isinstance(module, Attention):
        module.set_backend(backend)
        return
    for child in module.children():
        change_attn_backends(child, backend)


def projection_packed(
    q: Tensor,
    kv: Tensor | None,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Efficient input projection for MHA using a single packed linear layer.

    Essentially the same as ``torch.nn.functional._in_projection_packed`` but uses
    ``chunk`` (substantially faster than ``unflatten`` here).

    Parameters
    ----------
    q : Tensor
        Queries tensor of shape ``[B, L_q, D]``.
    kv : Tensor | None
        Keys/values tensor for cross-attention of shape ``[B, L_kv, D]``. If ``None``,
        self-attention is assumed.
    weight : Tensor
        Packed projection weight of shape ``[3D, D]``.
    bias : Tensor | None, optional
        Packed projection bias of shape ``[3D]``. The default is ``None``.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Projected queries, keys, and values: ``(Q, K, V)``.
    """
    if kv is None:
        return functional.linear(q, weight, bias).chunk(3, dim=-1)

    dim = q.size(-1)
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)

    q_proj = functional.linear(q, w_q, b_q)
    k_proj, v_proj = functional.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    return q_proj, k_proj, v_proj


def torch_attn(
    q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor | None, dropout: float, backend: str
) -> Tensor:
    """Scaled dot-product attention with a switchable torch backend.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape ``[B, H, L_q, D_h]``.
    k : Tensor
        Key tensor of shape ``[B, H, L_kv, D_h]``.
    v : Tensor
        Value tensor of shape ``[B, H, L_kv, D_h]``.
    mask : BoolTensor | None
        Attention mask of shape ``[B, 1, L_q, L_kv]`` (broadcastable), or ``None``.
    dropout : float
        Dropout probability applied inside attention.
    backend : str
        One of ``{"torch-math", "torch-flash", "torch-meff"}`` (flash-varlen handled elsewhere).

    Returns
    -------
    Tensor
        Attention output of shape ``[B, H, L_q, D_h]``.
    """
    with torch.backends.cuda.sdp_kernel(
        enable_math=True,  # always enabled as a fallback
        enable_mem_efficient=(backend == "torch-meff"),
        enable_flash=(backend == "torch-flash"),
    ):
        return functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)


class Attention(nn.Module):
    """Multihead attention module with optional differential attention and norms.

    Parameters
    ----------
    embed_dim : int
        Input (and output) embedding dimension.
    num_heads : int, optional
        Number of attention heads. The default is ``1``.
    attn_type : str, optional
        Backend kernel to use. One of ``{"torch-math", "torch-flash", "torch-meff",
        "flash-varlen"}``. The default is ``"torch-meff"``.
    dropout : float, optional
        Dropout rate applied in attention. The default is ``0.0``.
    bias : bool, optional
        Whether to include bias terms in projections. The default is ``True``.
    diff_attention : bool, optional
        Enable differential attention (splits heads in two branches). The default is ``False``.
    depth : int, optional
        Layer depth index (used to set differential attention weights). The default is ``1``.
    do_qk_norm : bool, optional
        Whether to apply RMSNorm to Q and K per head. The default is ``False``.
    do_v_norm : bool, optional
        Whether to apply RMSNorm to V per head. The default is ``False``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        attn_type: str = "torch-meff",
        dropout: float = 0.0,
        bias: bool = True,
        diff_attention: bool = False,
        depth: int = 1,
        do_qk_norm: bool = False,
        do_v_norm: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Dim not div by the number of heads!"
        assert attn_type in ATTN_TYPES, "Invalid attention type!"

        # Attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias
        self.attn_type = attn_type
        self.diff_attention = diff_attention
        self.depth = depth
        self.do_qk_norm = do_qk_norm
        self.do_v_norm = do_v_norm

        if self.diff_attention:
            self.head_dim = self.head_dim // 2
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
            self.lambda_q1 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_k1 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_q2 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_k2 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )

            self.subln = layernorms.RMSNorm(2 * self.head_dim)

        if self.do_qk_norm:
            self.q_norm = layernorms.RMSNorm(self.head_dim)
            self.k_norm = layernorms.RMSNorm(self.head_dim)
        if self.do_v_norm:
            self.v_norm = layernorms.RMSNorm(self.head_dim)

        # Better parallelism for self-attention when using parameters directly
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()
        self.set_backend(attn_type)

    def set_backend(self, attn_type: str) -> str:
        """Set and validate the attention backend.

        Parameters
        ----------
        attn_type : str
            Backend name.

        Returns
        -------
        str
            Effective backend set (may fall back to ``"torch-math"``).
        """
        # Check the attention backend
        self.attn_type = attn_type
        if self.attn_type == "flash-varlen":
            why_not_flash = ""
            if _flash_attn_func is None:
                why_not_flash = (
                    "Requires the flash_attn package, CUDA 12+, and A100+, and must be installed "
                    "separately or using the [flash] extra. See requirements-flash.txt."
                )
            elif not torch.cuda.is_available():
                why_not_flash = "No GPU available."
            if why_not_flash:
                warnings.warn(
                    f"Cannot use flash-varlen backend. {why_not_flash} Reverting to torch-math.",
                    stacklevel=2,
                )
                self.attn_type = "torch-math"
            else:
                self._flash_attn = _flash_attn_func
        return self.attn_type

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.0)
        self.out_proj.reset_parameters()

    def _weight_by_lambda(self, attn1: Tensor, attn2: Tensor) -> Tensor:
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(
            attn1
        )
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(
            attn1
        )
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        return self.subln(attn) * (1 - self.lambda_init)

    def _flash_forward(self, x: Tensor, culens: Tensor, maxlen: int) -> Tensor:
        """FlashAttention backend.

        Parameters
        ----------
        x : Tensor
            Packed sequence of shape ``[N_total, D]``.
        culens : Tensor
            Cumulative sequence lengths (for varlen flash).
        maxlen : int
            Maximum sequence length.

        Returns
        -------
        Tensor
            Output of shape ``[N_total, D]``.
        """
        # Perform the packed input projection
        qkv = functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

        if self.do_qk_norm:
            dtype = qkv.dtype
            if self.do_v_norm:
                q, k, v = qkv.unbind(1)
                q = self.q_norm(q)
                k = self.k_norm(k)
                v = self.v_norm(v)
                qkv = torch.stack([q, k, v], dim=1).to(dtype)
            else:
                q, k, v = qkv.unbind(1)
                q = self.q_norm(q)
                k = self.k_norm(k)
                qkv = torch.stack([q, k, v], dim=1).to(dtype)

        # Run the flash-varlen backend
        dropout = self.dropout if self.training else 0.0
        a_out = self._flash_attn(qkv, culens, maxlen, dropout)
        a_out = a_out.reshape(-1, self.embed_dim)

        # Mix with final linear layer
        return self.out_proj(a_out)

    def _flash_diff_forward(self, x: Tensor, culens: Tensor, maxlen: int) -> Tensor:
        """FlashAttention backend (differential attention variant).

        Parameters
        ----------
        x : Tensor
            Inputs as Tensor
        culens : Tensor
            Culens as Tensor
        maxlen : int
            Max length as int

        Returns
        -------
        Tensor
            Tensor output of the flash diff
        """
        # Perform the packed input projection
        qkv = functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(-1, 3, self.num_heads, 2, self.head_dim)

        q1, q2 = qkv[:, 0, :, 0], qkv[:, 0, :, 1]
        k1, k2 = qkv[:, 1, :, 0], qkv[:, 1, :, 1]
        v1, v2 = qkv[:, 2, :, 0], qkv[:, 2, :, 1]

        qkv11 = torch.stack([q1, k1, v1], dim=1)
        qkv12 = torch.stack([q1, k1, v2], dim=1)
        qkv21 = torch.stack([q2, k2, v1], dim=1)
        qkv22 = torch.stack([q2, k2, v2], dim=1)

        # Run the flash-varlen backend
        dropout = self.dropout if self.training else 0.0
        a_out11 = self._flash_attn(qkv11, culens, maxlen, dropout)
        a_out12 = self._flash_attn(qkv12, culens, maxlen, dropout)
        a_out1 = torch.cat([a_out11, a_out12], dim=-1)

        a_out21 = self._flash_attn(qkv21, culens, maxlen, dropout)
        a_out22 = self._flash_attn(qkv22, culens, maxlen, dropout)
        a_out2 = torch.cat([a_out21, a_out22], dim=-1)

        a_out = self._weight_by_lambda(a_out1, a_out2)
        a_out = a_out.reshape(-1, self.embed_dim)

        # Mix with final linear layer
        return self.out_proj(a_out)

    def _torch_forward(
        self, x: Tensor, kv: Tensor, mask: BoolTensor, kv_mask: BoolTensor, attn_mask: BoolTensor
    ) -> Tensor:
        """Attention using PyTorch SDPA backends.

        Parameters
        ----------
        x : Tensor
            Query input of shape ``[B, L_q, D]``.
        kv : Tensor
            Key/value input of shape ``[B, L_kv, D]`` (use ``x`` for self-attention).
        mask : BoolTensor
            Padding mask for ``x`` where padded positions are ``True``.
        kv_mask : BoolTensor
            Padding mask for ``kv`` where padded positions are ``True``.
        attn_mask : BoolTensor
            Attention mask of shape ``[B, L_q, L_kv]`` where allowed positions are ``True``.

        Returns
        -------
        Tensor
            Output of shape ``[B, L_q, D]``.
        """
        b, s, d = x.shape

        q, k, v = projection_packed(x, kv, self.in_proj_weight, self.in_proj_bias)

        shape = (b, -1, self.num_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.do_v_norm:
            v = self.v_norm(v)

        s_mask = mask if kv is None else kv_mask  # Who is sending, x or kv
        mask = merge_masks(s_mask, attn_mask, q.shape)
        dropout = self.dropout if self.training else 0.0
        a_out = torch_attn(q, k, v, mask, dropout, self.attn_type)

        a_out = a_out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(a_out)

    def _torch_diff_forward(
        self,
        x: Tensor,
        kv: Tensor,
        mask: BoolTensor,
        kv_mask: BoolTensor,
        attn_mask: BoolTensor,
    ) -> Tensor:
        """Attention using PyTorch SDPA backends (differential attention variant).

        Parameters
        ----------
        x : Tensor
            Input as Tensor
        kv : Tensor
            kv as Tensor
        mask : BoolTensor
            Mask as BoolTensor
        kv_mask : BoolTensor
            kv Mask as BoolTensor
        attn_mask : BoolTensor
            Attention mask as BoolTensor

        Returns
        -------
        Tensor
            Attention as Tensor
        """
        b, s, d = x.shape

        q, k, v = projection_packed(x, kv, self.in_proj_weight, self.in_proj_bias)

        shape = (b, -1, self.num_heads, 2, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        # Split q, k, v into two parts
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        s_mask = mask if kv is None else kv_mask
        mask = merge_masks(s_mask, attn_mask, q.shape)
        dropout = self.dropout if self.training else 0.0

        a_out11 = torch_attn(q1, k1, v1, mask, dropout, self.attn_type)
        a_out12 = torch_attn(q1, k1, v2, mask, dropout, self.attn_type)
        a_out1 = torch.cat([a_out11, a_out12], dim=-1)

        a_out21 = torch_attn(q2, k2, v1, mask, dropout, self.attn_type)
        a_out22 = torch_attn(q2, k2, v2, mask, dropout, self.attn_type)
        a_out2 = torch.cat([a_out21, a_out22], dim=-1)

        a_out = self._weight_by_lambda(a_out1, a_out2)

        a_out = a_out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(a_out)

    def forward(
        self,
        x: Tensor,
        kv: Tensor | None = None,
        mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
        culens: Tensor | None = None,
        maxlen: int | None = None,
    ) -> Tensor:
        """Attention forward pass, dispatching to the appropriate backend.

        Parameters
        ----------
        x : Tensor
            Query input of shape ``[B, L_q, D]``.
        kv : Tensor | None, optional
            Optional key/value input of shape ``[B, L_kv, D]`` for cross-attention.
            If ``None``, self-attention is used.
        mask : BoolTensor | None, optional
            Padding mask for ``x`` where padded positions are ``True``.
        kv_mask : BoolTensor | None, optional
            Padding mask for ``kv`` where padded positions are ``True``.
        attn_mask : BoolTensor | None, optional
            Attention mask of shape ``[B, L_q, L_kv]`` where allowed positions are ``True``.
        culens : Tensor | None, optional
            Cumulative lengths for varlen flash. Required for ``attn_type="flash-varlen"``.
        maxlen : int | None, optional
            Maximum sequence length. Required for ``attn_type="flash-varlen"``.

        Returns
        -------
        Tensor
            Output of shape ``[B, L_q, D]``.
        """
        if self.attn_type == "flash-varlen":
            assert kv is None, "flash-varlen only supports self attention!"
            assert attn_mask is None, "flash-varlen does not support attention masks!"
            assert culens is not None, "flash-varlen requires culens!"
            assert maxlen is not None, "flash-varlen requires maxlen!"
            if self.diff_attention:
                return self._flash_diff_forward(x, culens, maxlen)
            return self._flash_forward(x, culens, maxlen)

        if self.diff_attention:
            return self._torch_diff_forward(x, kv, mask, kv_mask, attn_mask)
        return self._torch_forward(x, kv, mask, kv_mask, attn_mask)


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
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

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
    fn : nn.Module
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
        fn: nn.Module,
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

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
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
        Tensor
            Output tensor with residual, normalization (if enabled), LayerScale, and DropPath.
        """
        if self.norm_type == "pre":
            return x + self.drop_path(self.ls(self.fn(self.norm(x), *args, **kwargs)))
        if self.norm_type == "post":
            return self.norm(x + self.drop_path(self.ls(self.fn(x, *args, **kwargs))))
        return x + self.drop_path(self.ls(self.fn(x, *args, **kwargs)))


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
    ) -> None:
        super().__init__()

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

        # Submodules
        residual = partial(
            NormResidual,
            norm=norm,
            ls_init=ls_init,
            drop_path=drop_path,
            norm_type=residual_norm_type,
        )
        self.attn = residual(Attention(embed_dim, depth=depth, **attn_kwargs))
        self.dense = residual(GLU(embed_dim, **dense_kwargs))

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Apply self-attention and feed-forward.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[B, L, D]``.
        **kwargs : Any
            Extra arguments forwarded to the :class:`Attention` module
            (e.g., ``mask``, ``attn_mask``, etc.).

        Returns
        -------
        Tensor
            Output tensor of shape ``[B, L, D]``.
        """
        x = self.attn(x, **kwargs)
        if self.norm_type == "hybrid":
            return self.dense(self.norm(x))
        return self.dense(x)


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
    depth : int, optional
        Layer depth index. The default is ``1``.
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
        depth: int = 1,
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
        self.self_attn = residual(Attention(embed_dim=embed_dim, depth=depth, **attn_kwargs))
        self.cross_attn = residual(Attention(embed_dim=embed_dim, depth=depth, **attn_kwargs))
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


class TransformerV2(nn.Module):
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

        # Submodules
        kwargs["attn_kwargs"]["attn_type"] = self.attn_type
        self.layers = torch.nn.ModuleList([
            EncoderLayer(embed_dim=embed_dim, norm=norm, depth=depth, **kwargs)
            for depth in range(num_layers)
        ])

        # Check and set the attention type
        assert self.attn_type in ATTN_TYPES, "Invalid attention type!"
        self.set_backend(self.attn_type)

        # Optional submodules
        if self.do_out_proj:
            self.out_proj = nn.Linear(self.embed_dim, out_dim)
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

        # If using the varlen backend, pack the sequence and store the cumulative lengths
        if self.attn_type == "flash-varlen":
            x, kwargs["culens"], kwargs["maxlen"] = undo_padding(x, mask)

        # Run through the main transformer encoder layers
        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
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
