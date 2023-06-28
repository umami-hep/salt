import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from lightning.pytorch.cli import instantiate_class
from torch import BoolTensor, Size, Tensor
from torch.nn.functional import sigmoid, softmax

from salt.models.dense import add_dims


def masked_softmax(x: Tensor, mask: BoolTensor, dim: int = -1) -> Tensor:
    """Applies softmax over a tensor without including padded elements."""
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


def merge_masks(
    q_mask: BoolTensor | None,
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Size,
    k_shape: Size,
    device: torch.device,
) -> BoolTensor:
    """Create a full attention mask which incoporates the padding information.

    Using pytorch transformer convention:
        False: Real node
        True:  Zero padded
    """
    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = torch.full(q_shape[:-1], False, device=device)
        if kv_mask is None:
            kv_mask = torch.full(k_shape[:-1], False, device=device)
        merged_mask = q_mask.unsqueeze(-1) | kv_mask.unsqueeze(-2)

    # If attention mask exists then it must be included
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask | merged_mask

    return merged_mask


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: nn.Module,
        edge_embed_dim: int = 0,
        k_dim: int | None = None,
        v_dim: int | None = None,
        out_proj: bool = True,
        update_edges: bool = False,
    ) -> None:
        """Generic multihead attention.

        Takes in three sequences with dim: (batch, sqeuence, features)
        - q: The primary sequence queries (determines output sequence length)
        - k: The attending sequence keys (determines incoming information)
        - v: The attending sequence values

        In a message passing sense you can think of q as your receiver nodes, v and k
        are the information coming from the sender nodes.

        When q == k(and v) this is a SELF attention operation
        When q != k(and v) this is a CROSS attention operation

        Block operations:

        1) Uses three linear layers to embed the sequences.
        - q = q_linear * q
        - k = k_linear * k
        - v = v_linear * v

        2) Outputs are reshaped to add a head dimension, and transposed for matmul.
        - dim becomes: batch, heads, sequence, features

        3) Passes these through to the attention module (message passing)
        - In standard transformers this is the scaled dot product attention
        - Also takes additional dropout layer to mask the attention

        4) Flatten out the head dimension, and optionally one more linear layer
        - results are same as if attention was done seperately for each head and concat
        - dim: batch, sequence, features*heads

        Parameters
        ----------
        embed_dim : int
            Model embedding dimension (query dim only if k_dim and v_dim also provided).
        num_heads : int
            Number of attention heads. The embed_dim is split into num_heads chunks.
        attention : nn.Module
            Type of attention (pooling operation) to use.
        edge_embed_dim: int, optional
            Model embedding dimension for edge features.
        k_dim : int, optional
            Key dimension, by default None where it assumes embed_dim
        v_dim : int, optional
            Value dimension, by default None where it assumes embed_dim
        out_proj : bool
            An optional output linear layer
        update_edges : bool, optional
            Indicate whether to update edge features.
        """
        super().__init__()

        # Check that the dimension of each heads makes internal sense
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        elif edge_embed_dim % num_heads != 0:
            raise ValueError(
                f"edge_embed_dim {edge_embed_dim} must be divisible by num_heads {num_heads}"
            )

        # Model base attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_proj = out_proj
        self.edge_embed_dim = edge_embed_dim
        self.edge_head_dim = edge_embed_dim // num_heads
        self.k_dim = k_dim or embed_dim
        self.v_dim = v_dim or embed_dim
        self.scale = math.sqrt(self.head_dim)
        self.update_edges = update_edges

        # Explicitly instantiate the attention class if passed as a dictionary
        if isinstance(attention, Mapping):
            attention = instantiate_class((), attention)
        self.attention = attention

        # The different linear projection layers, output is optional
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.k_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.v_dim, self.embed_dim)
        if self.edge_embed_dim > 0:
            self.linear_e = nn.Linear(self.edge_embed_dim, self.num_heads)
            self.linear_g = nn.Linear(self.edge_embed_dim, self.num_heads)
            if self.update_edges:
                self.linear_e_out = nn.Linear(self.num_heads, self.edge_embed_dim)
            else:
                self.register_buffer("linear_e_out", None)
        if self.out_proj:
            self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.register_buffer("linear_out", None)

    def input_projections(self, q, k, v) -> tuple:
        """Perform input linear projections, output shapes are (B,L,H,HD)."""
        shape = (k.shape[0], -1, self.num_heads, self.head_dim)
        q_proj = self.linear_q(q).view(shape).transpose(1, 2)
        k_proj = self.linear_k(k).view(shape).transpose(1, 2)
        v_proj = self.linear_v(v).view(shape).transpose(1, 2)
        return q_proj, k_proj, v_proj

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        edges: Tensor | None = None,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        """Full forward pass through the model.

        Parameters
        ----------
        q : Tensor
            The main sequence used to generate the queries.
        k : Optional[Tensor], optional
            Seperate sequence from which to generate the keys, by default None
        v : Optional[Tensor], optional
            Seperate sequence from which to generate the values, by default None
        edges : Optional[Tensor], optional
            Main sequence for edge features (used to calculate E and G)
        q_mask : Optional[BoolTensor], optional
            Shows which elements of q are real verses padded, by default None
        kv_mask : Optional[BoolTensor], optional
            Shows which elements of k and v are real verses padded, by default None
        attn_mask : Optional[BoolTensor], optional
            Extra mask for the attention (adjacency) matrix, by default None
        attn_bias : Optional[Tensor], optional
            Extra values to further augment the attention matrix, by default None

        Returns
        -------
        Tensor
            Output with the same shape as q
        """
        # If only q and q_mask are provided then we automatically apply self attention
        if k is None:
            k = q
            if kv_mask is None:
                kv_mask = q_mask
        v = v if v is not None else k

        # input shape
        b_size, _seq_len, _features = q.shape

        # Work out the masking situation, with padding, peaking, etc
        attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)

        # Apply the input projections (B,H,L,HD)
        q_proj, k_proj, v_proj = self.input_projections(q, k, v)

        # Calculate edge feature matrices E, G and reshape them to (B,L,L,H)
        if edges is not None:
            e = self.linear_e(edges)
            g = sigmoid(self.linear_g(edges))
            attn_bias = e if attn_bias is None else attn_bias + e

        # Calculate attention scores (B,H,Lq,Lk)
        attn_weights = self.attention(
            q_proj, k_proj, self.scale, attn_mask, attn_bias, self.update_edges
        )
        if self.update_edges:
            attn_weights, attn_scores = attn_weights

        # Apply gating to attention scores
        if edges is not None:
            attn_weights = attn_weights * g.permute(0, 3, 1, 2)

        # Use the scores for pooling and reshape (B, Lv, F)
        out = torch.matmul(attn_weights, v_proj)
        out = out.transpose(1, 2).contiguous().view(b_size, -1, self.embed_dim)

        # update edges with dot product attention scores (if desired)
        edge_out = None
        if self.update_edges:
            edge_out = self.linear_e_out(attn_scores.permute(0, 2, 3, 1))

        # Optional output layer
        if self.out_proj:
            out = self.linear_out(out)

        if edges is not None:
            return out, edge_out
        else:
            return out


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention, commonly used in transformers.

    Contains dropout layer to stochastically mask messages, used alot in language
    processing.

    Allows for addition of extra bias term in the attention matrix as used in the
    particle transformer: https://arxiv.org/abs/2202.03772
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        scale: Tensor,
        mask: BoolTensor | None = None,
        attn_bias: Tensor | None = None,
        return_scores: bool = False,
    ) -> Tensor:
        # inputs are of shape (batch, heads, sequence, head_dim)

        # dot product between queries and keys
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # add the bias terms if present
        if attn_bias is not None:  # Move the head dimension to the first
            scores = scores + attn_bias.permute(0, 3, 1, 2)

        # apply the dropout to the scores
        scores = self.dropout(scores)

        # softmax, use the mask to not included padded information
        attention_weights = masked_softmax(scores, mask)

        if return_scores:
            return attention_weights, scores
        else:
            return attention_weights


class GATv2Attention(nn.Module):
    """GATv2 attention, used in the original implementation of GN1.

    https://arxiv.org/abs/2105.14491
    """

    def __init__(self, num_heads: int, head_dim: int, activation: str = "SiLU") -> None:
        super().__init__()
        self.attention = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1, 1, head_dim)))
        self.activation = getattr(nn, activation)()
        nn.init.xavier_uniform_(self.attention)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        scale: Tensor,
        mask: BoolTensor | None = None,
        attn_bias: Tensor | None = None,
        return_scores: bool = False,
    ) -> Tensor:
        # inputs are (B, H, Lq/k, D)
        B, H, Lq, D = q.shape

        # sum each pair of tracks within a batch
        # shape: (B, H, Lq, Lk, D)
        summed = q.unsqueeze(-2) + k.unsqueeze(-3)

        # after activation, dot product with learned vector
        # shape: (B, H, Lq, Lk)
        scores = (self.activation(summed) * self.attention).sum(dim=-1)

        # add the optional bias
        if attn_bias is not None:
            scores = scores + attn_bias

        # softmax
        attention_weights = masked_softmax(scores, mask)

        if return_scores:
            return attention_weights, scores
        else:
            return attention_weights
