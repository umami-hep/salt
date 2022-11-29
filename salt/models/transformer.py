import torch
import torch.nn as nn

from salt.models.attention import MultiheadAttention
from salt.models.dense import Dense


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        num_heads: int,
        attention: nn.Module,
        activation: str,
        residual: bool = True,
        norm_layer: str = None,
        dropout: float = 0.0,
        out_proj: bool = True,
    ):
        """Self attention followed by a dense layer."""
        super().__init__()

        self.residual = residual

        if norm_layer:
            self.norm = getattr(torch.nn, norm_layer)(embd_dim, elementwise_affine=False)
        else:
            self.register_buffer("norm", None)

        self.mha = MultiheadAttention(embd_dim, num_heads, attention=attention, out_proj=out_proj)

        self.dense = Dense(
            embd_dim,
            embd_dim,
            [embd_dim],
            activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        if self.norm:
            x = self.norm(x)

        x_mha = self.mha(x, x, x, q_mask=mask, k_mask=mask)

        if self.residual:
            x_mha = x + x_mha

        x_dense = self.dense(x_mha)

        if self.residual:
            x_dense = x_mha + x_dense

        return x_dense


class Transformer(nn.Module):
    """Stacked Self attention followed by a dense layer.

    TODO: make general stacked. num, kwargs
    """

    def __init__(
        self,
        embd_dim: int,
        num_heads: int,
        num_layers: int,
        attention: nn.Module,
        activation: str,
        residual: bool = True,
        norm_layer: str = None,
        dropout: float = 0.0,
        out_proj: bool = True,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                SelfAttentionBlock(
                    embd_dim,
                    num_heads,
                    attention,
                    activation,
                    residual,
                    norm_layer,
                    dropout,
                    out_proj,
                )
            )
        self.net = nn.Sequential(*layers)

        self.final_dense = Dense(embd_dim, embd_dim, [embd_dim], activation, norm_layer=norm_layer)

    def forward(self, x, mask=None):
        for layer in self.net:
            x = layer(x, mask=mask)

        return self.final_dense(x)
