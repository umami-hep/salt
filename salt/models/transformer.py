import torch
import torch.nn as nn

from salt.models.dense import Dense


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        num_heads: int,
        activation: str,
        residual: bool = True,
        norm_layer: str = None,
        dropout: float = 0.0,
    ):
        """Transformer block.

        Parameters
        ----------
        input_size : int
            Number of input features per track
        output_size : int
            Number of output classes
        hidden_layers : list
            Number of nodes per hidden layer
        activation : nn.Module
            Activation function
        """
        super().__init__()

        self.residual = residual

        if norm_layer:
            self.norm = getattr(torch.nn, norm_layer)(
                embd_dim, elementwise_affine=False
            )
        else:
            self.register_buffer("norm", None)

        self.mha = nn.MultiheadAttention(
            embd_dim, num_heads, batch_first=True, add_zero_attn=True
        )

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

        x_mha, _ = self.mha(
            x,
            x,
            x,
            key_padding_mask=mask,
            need_weights=False,
        )

        if self.residual:
            x_mha = x + x_mha

        x_dense = self.dense(x_mha)

        if self.residual:
            x_dense = x_mha + x_dense

        return x_dense


class Transformer(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        num_heads: int,
        num_layers: int,
        activation: str,
        residual: bool = True,
        norm_layer: str = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                SelfAttentionBlock(
                    embd_dim, num_heads, activation, residual, norm_layer, dropout
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        for layer in self.net:
            x = layer(x, mask=mask)
        return x
