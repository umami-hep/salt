"""Attention/EdgeAttention and SDPA helpers."""

from __future__ import annotations

import math
import warnings

import torch
from torch import BoolTensor, Size, Tensor, nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

from salt.model.nn.layernorm import RMSNorm

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None


def check_flash_attn() -> str:
    """Check if Flash Attention is available; empty string if so, else a reason why not."""
    if not torch.cuda.is_available():
        return "No GPU available."

    gpu_name = torch.cuda.get_device_name(0)
    # (major, minor), e.g. (8, 0)
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    sm_version = float(f"{major}.{minor}")

    if sm_version < 8.0:
        return (
            f"GPU '{gpu_name}' with SM {sm_version} is not compatible. "
            "Flash Attention 2 requires SM 8.0 or newer (Ampere+)."
        )
    if _flash_attn_func is None:
        return (
            "Flash attention is required but not found! Please ensure the package is installed "
            "correctly. If not, please install the flash attention package as described in "
            "https://ftag-salt.docs.cern.ch/setup/#install-the-salt-package"
        )
    return ""


ATTN_TYPES = ["torch-math", "torch-flash", "torch-meff", "flash-varlen"]


def merge_masks(
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Size,
) -> BoolTensor | None:
    """Merge kv-pad + attn masks into ``[B,1,L_q,L_kv]``; padded tokens receive but never send."""
    mask = None

    if kv_mask is not None:
        mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        mask = ~mask  # flip: True now means "valid token" (not padded)

    if attn_mask is not None:
        mask = attn_mask if mask is None else attn_mask & mask

    if mask is not None:
        mask = mask.unsqueeze(1)  # broadcast over heads

    return mask


def projection_packed(
    q: Tensor,
    kv: Tensor | None,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Efficient input projection for MHA using a single packed linear layer -> ``(Q, K, V)``."""
    if kv is None:
        return functional.linear(q, weight, bias).chunk(3, dim=-1)

    dim = q.size(-1)
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)

    q_proj = functional.linear(q, w_q, b_q)
    k_proj, v_proj = functional.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    return q_proj, k_proj, v_proj


def torch_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: BoolTensor | None,
    dropout: float,
    softmax_scale: float,
    backend: str,
) -> Tensor:
    """Scaled dot-product attention with a switchable torch backend; returns ``[B,H,L_q,D_h]``."""
    backends = [SDPBackend.MATH]
    if backend == "torch-flash":
        backends += [SDPBackend.FLASH_ATTENTION]
    elif backend == "torch-meff":
        backends += [SDPBackend.EFFICIENT_ATTENTION]
    with sdpa_kernel(backends=backends):
        return functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout, scale=softmax_scale
        )


def _init_packed_mha(
    mod: nn.Module,
    embed_dim: int,
    num_heads: int,
    dropout: float,
    bias: bool,
    do_qk_norm: bool,
    do_v_norm: bool,
    mup: bool,
) -> None:
    """Set up the shared `Attention`/`EdgeAttention` attributes, norms and projections.

    Head geometry, optional q/k/v RMSNorms, packed QKV + out projection. Registration order
    (norms, ``in_proj_weight``, ``in_proj_bias``, ``out_proj``) is the ``state_dict`` key order
    and the ``nn.Linear`` RNG draw order — keep it.
    """
    mod.embed_dim = embed_dim
    mod.num_heads = num_heads
    mod.head_dim = embed_dim // num_heads
    mod.dropout = dropout
    mod.bias = bias
    mod.do_qk_norm = do_qk_norm
    mod.do_v_norm = do_v_norm
    mod.mup = mup
    mod.scale = 1 / mod.head_dim if mup else 1 / math.sqrt(mod.head_dim)
    if do_qk_norm:
        mod.q_norm = RMSNorm(mod.head_dim)
        mod.k_norm = RMSNorm(mod.head_dim)
    if do_v_norm:
        mod.v_norm = RMSNorm(mod.head_dim)
    # packed QKV projection: better parallelism for self-attention than 3 separate Linears
    mod.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
    mod.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
    mod.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


def _reset_in_proj(mod: nn.Module) -> None:
    """Init the packed QKV projection (muP: https://arxiv.org/abs/2203.03466) and zero its bias."""
    if mod.mup:
        nn.init.normal_(mod.in_proj_weight, mean=0.0, std=1.0 / mod.head_dim**0.5)  # K,V proj
        nn.init.constant_(mod.in_proj_weight[: mod.embed_dim, :], 0.0)  # Q projection
    else:
        nn.init.xavier_uniform_(mod.in_proj_weight)
    if mod.bias:
        nn.init.constant_(mod.in_proj_bias, 0.0)


def _qkv_heads(
    mod: nn.Module,
    x: Tensor,
    b: int,
    kv: Tensor | None,
    mask: BoolTensor | None,
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
) -> tuple[Tensor, Tensor, Tensor, BoolTensor | None]:
    """Project to per-head ``[B, H, L, D_h]`` q/k/v (+ q/k/v norms); merge the sender mask."""
    q, k, v = projection_packed(x, kv, mod.in_proj_weight, mod.in_proj_bias)

    shape = (b, -1, mod.num_heads, mod.head_dim)
    q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

    if mod.do_qk_norm:
        q = mod.q_norm(q)
        k = mod.k_norm(k)
    if mod.do_v_norm:
        v = mod.v_norm(v)

    s_mask = mask if kv is None else kv_mask  # who is sending, x or kv
    return q, k, v, merge_masks(s_mask, attn_mask, q.shape)


class Attention(nn.Module):
    """Multihead attention: packed QKV projection, switchable backend, optional Q/K/V norms.

    ``mup=True`` rescales the dot-product scale ``sqrt(head_dim) -> head_dim``
    and changes init (https://arxiv.org/abs/2203.03466).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        attn_type: str = "torch-meff",
        dropout: float = 0.0,
        bias: bool = True,
        do_qk_norm: bool = False,
        do_v_norm: bool = False,
        mup: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Dim not div by the number of heads!"
        assert attn_type in ATTN_TYPES, "Invalid attention type!"
        self.attn_type = attn_type
        _init_packed_mha(self, embed_dim, num_heads, dropout, bias, do_qk_norm, do_v_norm, mup)
        self.reset_parameters()
        self.set_backend(attn_type)

    def set_backend(self, attn_type: str) -> str:
        """Set the attention backend; falls back to ``"torch-math"`` (warns) if unavailable."""
        self.attn_type = attn_type
        if self.attn_type == "flash-varlen":
            why_not_flash = check_flash_attn()
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
        _reset_in_proj(self)
        if not self.mup:
            self.out_proj.reset_parameters()
            return
        nn.init.normal_(self.out_proj.weight, std=(1.0 / self.embed_dim) ** 0.5)  # Output proj
        if self.bias:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _flash_forward(self, x: Tensor, culens: Tensor, maxlen: int) -> Tensor:
        """FlashAttention backend; returns ``[N_total, D]``."""
        qkv = functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

        if self.do_qk_norm or self.do_v_norm:
            dtype = qkv.dtype
            q, k, v = qkv.unbind(1)
            if self.do_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            if self.do_v_norm:
                v = self.v_norm(v)
            qkv = torch.stack([q, k, v], dim=1).to(dtype)

        dropout = self.dropout if self.training else 0.0
        a_out = self._flash_attn(qkv, culens, maxlen, dropout, softmax_scale=self.scale)
        a_out = a_out.reshape(-1, self.embed_dim)

        return self.out_proj(a_out)

    def _torch_forward(
        self, x: Tensor, kv: Tensor, mask: BoolTensor, kv_mask: BoolTensor, attn_mask: BoolTensor
    ) -> Tensor:
        """Attention using PyTorch SDPA backends; returns ``[B, L_q, D]``."""
        b, s, d = x.shape
        q, k, v, merged = _qkv_heads(self, x, b, kv, mask, kv_mask, attn_mask)
        dropout = self.dropout if self.training else 0.0
        a_out = torch_attn(
            q, k, v, merged, dropout=dropout, softmax_scale=self.scale, backend=self.attn_type
        )

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
        """Attention forward pass, dispatching to the flash or torch-SDPA backend."""
        if self.attn_type == "flash-varlen":
            assert kv is None, "flash-varlen only supports self attention!"
            assert attn_mask is None, "flash-varlen does not support attention masks!"
            assert culens is not None, "flash-varlen requires culens!"
            assert maxlen is not None, "flash-varlen requires maxlen!"
            return self._flash_forward(x, culens, maxlen)

        return self._torch_forward(x, kv, mask, kv_mask, attn_mask)


class EdgeAttention(nn.Module):
    """Multihead attention with edge features biasing the scores and gating the output.

    No pluggable backend (always raw torch attention — see `set_backend`).
    ``update_edges`` also emits updated edge features. ``mup=True`` rescales
    the dot-product scale and init (https://arxiv.org/abs/2203.03466).
    """

    def __init__(
        self,
        embed_dim: int,
        edge_embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        do_qk_norm: bool = False,
        do_v_norm: bool = False,
        update_edges: bool = False,
        mup: bool = False,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Dim not div by the number of heads!"
        self.edge_embed_dim = edge_embed_dim
        self.edge_head_dim = edge_embed_dim // num_heads
        self.update_edges = update_edges
        _init_packed_mha(self, embed_dim, num_heads, dropout, bias, do_qk_norm, do_v_norm, mup)

        self.linear_e = nn.Linear(self.edge_embed_dim, self.num_heads, bias=bias)
        self.linear_g = nn.Linear(self.edge_embed_dim, self.num_heads, bias=bias)
        if self.update_edges:
            self.linear_e_out = nn.Linear(self.num_heads, self.edge_embed_dim, bias=bias)
        else:
            self.register_buffer("linear_e_out", None)

        self.reset_parameters()

    def set_backend(self, attn_type: str) -> str:
        warnings.warn(
            "EdgeAttention does not support different backends yet. Using raw attention.",
            stacklevel=2,
        )
        return attn_type

    def reset_parameters(self) -> None:
        """Initialize the parameters (muP keeps the ``out_proj`` weight at its default init)."""
        _reset_in_proj(self)
        layers = [self.linear_e, self.linear_g, self.out_proj]
        if self.update_edges:
            layers.append(self.linear_e_out)
        for layer in layers:
            if not self.mup:
                layer.reset_parameters()
            elif layer is not self.out_proj:
                nn.init.normal_(layer.weight, std=(1.0 / layer.in_features) ** 0.5)
            if self.bias:
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        edge_x: Tensor,
        kv: Tensor | None = None,
        mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Attention with edge features biasing the scores and gating the output.

        Returns ``(out [B, L_q, D], edge_out [B, L_q, L_kv, E])``.
        """
        b, s, d = x.shape
        q, k, v, merged = _qkv_heads(self, x, b, kv, mask, kv_mask, attn_mask)
        e = self.linear_e(edge_x)  # (B, L_q, L_kv, num_heads)
        g = functional.sigmoid(self.linear_g(edge_x))  # (B, L_q, L_kv, num_heads)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, L_q, L_kv)
        attn_scores = attn_scores + e.permute(0, 3, 1, 2)  # add edge embeddings

        if self.dropout > 0.0 and self.training:
            attn_scores = functional.dropout(attn_scores, p=self.dropout)

        edge_out = edge_x
        if self.update_edges:
            edge_out = self.linear_e_out(
                attn_scores.permute(0, 2, 3, 1)  # (B, L_q, L_kv, num_heads)
            )
        # Compute attention weights
        masked_scores = (
            torch.masked_fill(attn_scores, ~merged, float("-inf"))
            if merged is not None
            else attn_scores
        )
        attn_weights = torch.softmax(masked_scores, dim=-1)  # (B, num_heads, L_q, L_kv)

        attn_weights = attn_weights * g.permute(0, 3, 1, 2)  # apply gating

        a_out = torch.matmul(attn_weights, v)  # (B, num_heads, L_q, head_dim)

        a_out = a_out.transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(a_out), edge_out
