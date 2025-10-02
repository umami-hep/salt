import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Faster LayerNorm by seting elementwise_affine=False."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, elementwise_affine=False)


class RMSNorm(torch.nn.Module):
    """RNMSNorm from https://arxiv.org/abs/1910.07467. Slower than LayerNorm.
    Follows the LLaMA implementation (https://github.com/meta-llama/llama3/blob/main/llama/model.py#L35).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
