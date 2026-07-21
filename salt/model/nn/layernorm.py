"""LayerNorm/RMSNorm variants."""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Faster LayerNorm by setting elementwise_affine=False."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, elementwise_affine=False)


class RMSNorm(torch.nn.Module):
    """RMSNorm from https://arxiv.org/abs/1910.07467 (LLaMA implementation)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class _Layernorms:
    """Namespace exposing `LayerNorm`/`RMSNorm` by name for ``getattr(_LAYERNORMS, norm)``."""

    LayerNorm = LayerNorm
    RMSNorm = RMSNorm


_LAYERNORMS = _Layernorms()
