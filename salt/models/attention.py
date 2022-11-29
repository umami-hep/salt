import numpy as np
import torch
import torch.nn as nn
from torch import BoolTensor, Tensor


def masked_softmax(scores, mask, dim=-1):
    # apply masking
    if mask is not None:
        scores = scores.masked_fill(mask, -np.inf)

    # softmax
    attention_weights = torch.softmax(scores, dim=-1)

    # reapply the mask to remove NaNs
    if mask is not None:
        attention_weights = attention_weights.masked_fill(mask, 0)

    return attention_weights


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: nn.Module,
        k_dim: int = None,
        v_dim: int = None,
        out_proj: bool = True,
    ):
        """Generic multihead attention.

        Parameters
        ----------
        embed_dim : int
            Model embedding dimension (query dim only if k_dim and v_dim also provided).
        num_heads : int
            Number of attention heads. The embed_dim is split into num_heads chunks.
        attention : nn.Module
            Type of attention to use.
        k_dim : int, optional
            Key dimension, by default None
        v_dim : int, optional
            Value dimension, by default None
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = attention
        self.out_proj = out_proj

        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim

        self.head_dim = embed_dim // self.num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dim))

        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.k_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.v_dim, self.embed_dim)
        if self.out_proj:
            self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.register_buffer("linear_out", None)

    def input_projection(self, q, k, v, B):
        """Linear input projections, allowing for varying sequence lengths."""
        q = self.linear_q(q).view((B, q.shape[1], self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.linear_k(k).view((B, k.shape[1], self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.linear_v(v).view((B, v.shape[1], self.num_heads, self.head_dim)).transpose(1, 2)
        return q, k, v

    def merge_masks(self, q_mask, k_mask, q_shape, k_shape):
        """Create a combined mask to handle query and key padding."""
        combined_mask = None

        # if either mask exists, create a merged mask
        if q_mask is not None or k_mask is not None:
            k_mask = k_mask if k_mask is not None else torch.ones(k_shape[:-1], dtype=torch.bool)
            q_mask = q_mask if q_mask is not None else torch.ones(q_shape[:-1], dtype=torch.bool)
            combined_mask = (q_mask.unsqueeze(-1) | k_mask.unsqueeze(-2)).unsqueeze(-3)

        return combined_mask

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, q_mask: BoolTensor = None, k_mask: BoolTensor = None
    ):
        """Forward pass.

        input projection -> attention-weighted sum -> output projection.
        """

        # input shape (B, L, D)
        batch, length, dim = k.shape

        # get mask
        mask = self.merge_masks(q_mask, k_mask, q.shape, k.shape)

        # project inputs to (B, H, L, HD)
        q, k, v = self.input_projection(q, k, v, B=batch)

        # calculate attention scores (B, H, Lq, Lk)
        attention = self.attention(q, k, self.scale, mask)

        # outputs
        out = torch.matmul(attention, v).transpose(1, 2).contiguous()
        out = out.view(batch, -1, self.num_heads * self.head_dim)
        if self.out_proj:
            out = self.linear_out(out)

        return out


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention, commonly used in transformers."""

    def __init__(self):
        super().__init__()

    def forward(self, q: Tensor, k: Tensor, scale: Tensor, mask: BoolTensor = None):
        # inputs are (B, H, L, HD)

        # dot product between queries and keys
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # softmax
        attention_weights = masked_softmax(scores, mask)

        return attention_weights


class GATv2Attention(nn.Module):
    """GATv2 attention, used in the original implementation of GN1.

    https://arxiv.org/abs/2105.14491
    """

    def __init__(self, num_heads: int, head_dim: int, activation: nn.Module = nn.SiLU()):
        super().__init__()
        self.attention = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1, 1, head_dim)))
        self.activation = activation
        nn.init.xavier_uniform_(self.attention)

    def forward(self, q: Tensor, k: Tensor, scale: Tensor = None, mask: BoolTensor = None):
        # inputs are (B, H, Lq/k, D)
        B, H, Lq, D = q.shape

        # sum each pair of tracks within a batch
        # shape: (B, H, Lq, Lk, D)
        summed = q.unsqueeze(-2) + k.unsqueeze(-3)

        # after activation, dot product with learned vector
        # shape: (B, H, Lq, Lk)
        scores = (self.activation(summed) * self.attention).sum(dim=-1)

        # softmax
        attention_weights = masked_softmax(scores, mask)

        return attention_weights
