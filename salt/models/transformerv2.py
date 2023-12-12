"""Efficient Transformer implementation.

Updated transformer implementation based on
https://github.com/mistralai/mistral-src

Features:
- use native pytorch sdp operators (including flash)
- support gated linear units from https://arxiv.org/abs/2002.05202
- RMSNorm from https://arxiv.org/abs/1910.07467
"""

from abc import ABC

import torch
from torch import BoolTensor, Tensor, nn

from salt.models.attention import merge_masks


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def torch_meff_attn(q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor, dropout: float) -> Tensor:
    # masking can lead to nans, see
    # - https://github.com/pytorch/pytorch/issues/110213
    # - https://github.com/pytorch/pytorch/issues/103749
    # TODO: change mask convention
    if mask is not None:
        mask = mask.contiguous()
        mask = ~mask
        mask = (1.0 - mask.to(q.dtype)) * torch.finfo(q.dtype).min

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=True
    ):
        return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout
        )


def torch_flash_attn(q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor, dropout: float) -> Tensor:
    assert mask is None, "Flash attention does not support attention masks"
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout
        )


ATTN_TYPES = {
    "torch-meff": torch_meff_attn,
    "torch-flash": torch_flash_attn,
}


class Attention(nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_type: str = "torch-meff",
        n_kv_heads: int | None = None,
        window_size: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.n_kv_heads = num_heads if n_kv_heads is None else n_kv_heads
        assert self.n_kv_heads is not None
        self.repeats = self.num_heads // self.n_kv_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.bias = bias

        self.attn_type = attn_type
        self.attn_func = ATTN_TYPES[self.attn_type]
        self.backend = self._flash_backend if self.attn_type == "flash" else self._torch_backend
        if window_size is None:
            self.window_size = (-1, -1)
        else:
            assert attn_type == "flash"
            assert window_size % 2 == 0
            self.window_size = (window_size // 2, window_size // 2)

        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=self.bias)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=self.bias)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=self.bias)
        self.add_zero_attn = True

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
    ) -> Tensor:
        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            batch = q.shape[0]
            zero_attn_shape = (batch, 1, self.dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1), value=False)
            if kv_mask is not None:
                kv_mask = nn.functional.pad(kv_mask, (0, 1), value=False)

        # combine masks
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        # input projections
        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        # run attention
        output = self.backend(q, k, v, attn_mask)

        # return output
        return self.wo(output)

    def _torch_backend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: BoolTensor | None = None):
        batch, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        q = q.view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # repeat keys and values to match number of query heads
        if self.repeats > 1:
            k, v = repeat_kv(k, v, self.repeats, dim=-2)

        # reshape mask
        if attn_mask is not None:
            attn_mask = attn_mask.view(batch, 1, q_len, kv_len).expand(-1, self.num_heads, -1, -1)

        # run attention
        output = self.attn_func(q, k, v, mask=attn_mask, dropout=self.dropout)  # type: ignore

        # reshape output and return
        return output.transpose(1, 2).contiguous().view(batch, -1, self.dim)

    def _flash_backend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: BoolTensor | None = None):
        assert attn_mask is None

        batch, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        q_p = q.view(batch, q_len, self.num_heads, self.head_dim)
        k_p = k.view(batch, kv_len, self.n_kv_heads, self.head_dim)
        v_p = v.view(batch, kv_len, self.n_kv_heads, self.head_dim)

        # repeat keys and values to match number of query heads
        if self.repeats > 1:
            k_p, v_p = repeat_kv(k_p, v_p, self.repeats, dim=-2)

        # run attention
        output = self.attn_func(q_p, k_p, v_p, dropout=self.dropout, window_size=self.window_size)  # type: ignore

        # reshape output and return
        return output.view_as(q)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, **kwargs)
        self.norm = RMSNorm(self.dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.norm(x)
        return self.attention(x, x, x, **kwargs)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, **kwargs)
        self.norm_q = RMSNorm(self.dim)
        self.norm_kv = RMSNorm(self.dim)

    def forward(self, q: Tensor, kv: Tensor, **kwargs) -> Tensor:
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        return self.attention(q, kv, kv, **kwargs)


class GLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        activation: str = "Mish",
        bias: bool = True,
        gated: bool = False,
    ):
        """Gated linear unit from https://arxiv.org/abs/2002.05202."""
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 2

        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.gate = None
        if gated:
            self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = getattr(nn, activation)()

    def forward(self, x) -> Tensor:
        x = self.norm(x)
        out = self.activation(self.w1(x))
        if self.gate:
            out = out * self.gate(x)
        return self.w2(out)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """RNMSNorm layer from https://arxiv.org/abs/1910.07467."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight


class TransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim: int,
        ff_dim_scale: int = 2,
        gated: bool = False,
        activation: str = "Mish",
        **attn_kwargs,
    ):
        super().__init__()
        if attn_kwargs is None:
            attn_kwargs = {}
        self.num_heads = num_heads
        self.dim = dim
        self.attention = SelfAttention(dim=dim, num_heads=num_heads, **attn_kwargs)
        self.dense = GLU(dim, dim * ff_dim_scale, activation=activation, gated=gated)

    def forward(self, x: Tensor, pad_mask: BoolTensor) -> Tensor:
        x = x + self.attention(x, kv_mask=pad_mask)
        return x + self.dense(x)


class TransformerV2(nn.Module):
    def __init__(self, num_layers: int, dim: int, out_dim: int | None = None, **layer_config):
        """Transformer model consisting of a series of stacked Transformer encoder layers.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        dim : int
            Dimension of the embeddings at each layer.
        out_dim : int | None, optional
            Optionally project the output to a different dimension.
        layer_config : dict
            Configuration for each layer.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim

        self.layers = torch.nn.ModuleList(
            [TransformerLayer(dim=dim, **layer_config) for _ in range(num_layers)]
        )
        self.out_norm = RMSNorm(dim if out_dim is None else out_dim)
        if out_dim is not None:
            self.out_proj = nn.Linear(self.dim, out_dim)
        else:
            self.out_proj = None

    def forward(self, x: Tensor, pad_mask: BoolTensor) -> Tensor:
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)
        if isinstance(pad_mask, dict):
            pad_mask = torch.cat(list(pad_mask.values()), dim=1)

        for layer in self.layers:
            x = layer(x, pad_mask)
        if self.out_proj is not None:
            x = self.out_proj(x)
        return self.out_norm(x)
