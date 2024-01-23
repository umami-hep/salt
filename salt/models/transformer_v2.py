"""Efficient Transformer implementation.

Updated transformer implementation based on
https://github.com/mistralai/mistral-src

Features:
- native SDP kernels (including flash)
- gated linear units https://arxiv.org/abs/2002.05202
- RMSNorm https://arxiv.org/abs/1910.07467
"""

from abc import ABC

import torch
from torch import BoolTensor, Tensor, nn

import salt.models.layernorm as layernorms
from salt.models.attention import merge_masks


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def torch_meff_attn(q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor, dropout: float) -> Tensor:
    # masking can lead to nans, see
    # - https://github.com/pytorch/pytorch/issues/110213
    # - https://github.com/pytorch/pytorch/issues/103749
    # to get round this, can transform the mask from a bool to float
    # mask = (1.0 - mask.to(q.dtype)) * torch.finfo(q.dtype).min
    # but don't need this if add_zero_attn is True

    # TODO: change mask convention
    # https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/-/issues/47
    if mask is not None:
        mask = ~mask.contiguous()
        # mask = (1.0 - mask.to(q.dtype)) * torch.finfo(q.dtype).min

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


ATTN_BACKENDS = {
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
        add_zero_attn: bool = True,
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
        self.add_zero_attn = add_zero_attn

        self.attn_type = attn_type
        self.attn_func = ATTN_BACKENDS[self.attn_type]
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

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
    ) -> Tensor:
        """Attention forward pass.

        Parameters
        ----------
        q : Tensor
            Queries of shape (batch, q_len, dim).
        k : Tensor
            Keys of shape (batch, kv_len, dim).
        v : Tensor
            Values of shape (batch, kv_len, dim).
        q_mask : BoolTensor, optional
            Mask for the queries, by default None.
        kv_mask : BoolTensor, optional
            Mask for the keys and values, by default None.
        attn_mask : BoolTensor, optional
            Full attention mask, by default None.

        Returns
        -------
        Tensor
            Output of shape (batch, q_len, dim).
        """
        # combine masks
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        # input projections
        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        # add a dummy token to attend to - avoids nan when all tokens are padded
        if self.add_zero_attn:
            batch = q.shape[0]
            zero_attn_shape = (batch, 1, self.dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1), value=False)
            if kv_mask is not None:
                kv_mask = nn.functional.pad(kv_mask, (0, 1), value=False)

        # run attention
        output = self.backend(q, k, v, attn_mask)

        # return output projection
        return self.wo(output)

    def _torch_backend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: BoolTensor | None = None):
        batch, q_len, _ = q.shape
        _, kv_len, _ = k.shape

        # transform tensors to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, kv_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # repeat keys and values to match number of query heads
        if self.repeats > 1:
            k, v = repeat_kv(k, v, self.repeats, dim=-2)

        # expand mask to (batch, num_heads, q_len, kv_len)
        if attn_mask is not None:
            attn_mask = attn_mask.view(batch, 1, q_len, kv_len).expand(-1, self.num_heads, -1, -1)

        # run attention
        output = self.attn_func(q, k, v, mask=attn_mask, dropout=self.dropout)

        # reshape output and return
        return output.transpose(1, 2).contiguous().view(batch, -1, self.dim)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, **kwargs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.attention(x, x, x, **kwargs)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim=dim, **kwargs)

    def forward(self, q: Tensor, kv: Tensor, **kwargs) -> Tensor:
        return self.attention(q, kv, kv, **kwargs)


class GLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        activation: str = "ReLU",
        bias: bool = True,
        gated: bool = True,
    ):
        """Gated linear unit from https://arxiv.org/abs/2002.05202."""
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 2

        self.in_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.gate = None
        if gated:
            self.gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(self.in_proj(x))
        if self.gate:
            out = out * self.gate(x)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ):
        super().__init__()
        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}
        self.dim = dim
        self.attn = SelfAttention(dim=dim, **attn_kwargs)
        self.attn_norm = getattr(layernorms, norm)(dim)
        self.dense = GLU(dim, **dense_kwargs)
        self.dense_norm = getattr(layernorms, norm)(dim)

    def forward(self, x: Tensor, pad_mask: BoolTensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), kv_mask=pad_mask)
        return x + self.dense(self.dense_norm(x))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ):
        super().__init__()
        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}
        self.dim = dim
        self.attn = CrossAttention(dim=dim, **attn_kwargs)
        self.q_norm = getattr(layernorms, norm)(dim)
        self.kv_norm = getattr(layernorms, norm)(dim)
        self.dense = GLU(dim, **dense_kwargs)
        self.dense_norm = getattr(layernorms, norm)(dim)

    def forward(self, x: Tensor, kv: Tensor, pad_mask: BoolTensor) -> Tensor:
        x = x + self.attn(self.q_norm(x), self.kv_norm(kv), kv_mask=pad_mask)
        return x + self.dense(self.dense_norm(x))


class TransformerV2(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        out_dim: int | None = None,
        norm: str = "LayerNorm",
        **kwargs,
    ):
        """Transformer model consisting of a series of stacked Transformer encoder layers.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        dim : int
            Dimension of the embeddings at each layer.
        out_dim : int | None, optional
            Optionally project the output to a different dimension.
        norm : str, optional
            Normalization style, by default "LayerNorm".
        kwargs : dict
            Keyword arguments for [salt.models.transformerv2.EncoderLayer].
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim

        self.layers = torch.nn.ModuleList([
            EncoderLayer(dim=dim, norm=norm, **kwargs) for _ in range(num_layers)
        ])
        self.out_norm = getattr(layernorms, norm)(dim if out_dim is None else out_dim)
        self.out_proj = None
        if out_dim is not None:
            self.out_proj = nn.Linear(self.dim, out_dim)

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
