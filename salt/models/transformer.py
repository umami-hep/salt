from typing import Mapping, Optional

import torch.nn as nn
from torch import BoolTensor, Tensor

from salt.models.attention import MultiheadAttention
from salt.models.dense import Dense


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer based on the GPT-2+Normformer style
    arcitecture.

    We choose Normformer as it has often proved to be the most stable to train
    https://arxiv.org/abs/2210.06423
    https://arxiv.org/abs/2110.09456

    It contains:
    - Multihead(self)Attention block
    - A feedforward network (which can take optional context information)
    - Layernorm is applied before each operation
    - Residual connections are used to bypass each operation
    """

    def __init__(
        self,
        embed_dim: int,
        mha_config: Mapping,
        dense_config: Mapping,
        context_dim: int = 0,
    ) -> None:
        """Init method of TransformerEncoderBlock.

        Parameters
        ----------
        embed_dim : int
            The embedding dimension of the transformer block
        mha_config : int
            Configuration for the MultiheadAttention block
        dense_config : nn.Module
            Configuration for the Dense network
        context_dim: int
            The size of the context tensor
        """
        super().__init__()
        self.embed_dim = embed_dim

        # The main blocks in the transformer
        self.mha = MultiheadAttention(embed_dim, **mha_config)
        self.dense = Dense(
            input_size=embed_dim, output_size=embed_dim, context_size=context_dim, **dense_config
        )

        # The multiple normalisation layers to keep it all stable
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[BoolTensor] = None,
        context: Optional[Tensor] = None,
        attn_mask: Optional[BoolTensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.norm2(
            self.mha(self.norm1(x), q_mask=mask, attn_mask=attn_mask, attn_bias=attn_bias)
        )
        x = x + self.dense(x, context)
        return x


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final
    normalisation step.

    A message passing network which does not change the dimension of the
    nodes, and uses multiheaded attention to facilitate message
    importance.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        mha_config: Mapping,
        dense_config: Mapping,
        context_dim: int = 0,
        out_dim: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            Feature size for input, output, and all intermediate layers
        num_layers : int
            Number of encoder layers used
        mha_config : nn.Module
            Keyword arguments for the mha block
        dense_config: int
            Keyword arguments for the dense network in each layer
        context_dim: int
            Dimension of the context inputs
        out_dim: int
            If set, a final linear layer resizes the outputs of the model
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, mha_config, dense_config, context_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.out_dim:
            self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            x = layer(x, **kwargs)
        x = self.final_norm(x)

        # Optinal resizing layer
        if self.out_dim:
            x = self.final_linear(x)
        return x
