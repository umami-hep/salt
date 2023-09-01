from collections.abc import Mapping

import torch
from torch import Tensor, nn, randn

from salt.models.transformer import TransformerCrossAttentionLayer
from salt.utils.tensor_utils import masked_softmax


class Pooling(nn.Module):
    ...


class GlobalAttentionPooling(Pooling):
    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = nn.Linear(input_size, 1)

    def forward(self, x: Tensor, mask: Tensor = None):
        if mask is not None:
            mask = mask.unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x), mask, dim=1)

        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x = torch.cat([x, x_pad], dim=1)

        return (x * weights).sum(dim=1)


class CrossAttentionPooling(Pooling):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        mha_config: Mapping,
        dense_config: Mapping | None = None,
        context_dim: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        self.ca_layers = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(input_size, mha_config, dense_config, context_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(input_size)

        # Initialise class token which is the query for the cross-attention
        self.class_token = nn.Parameter(randn(1, 1, input_size))

    def forward(self, x: Tensor, mask: Tensor = None, context: Tensor | None = None):
        # Expand class token to match batch size
        class_token = self.class_token.expand(x.shape[0], 1, self.input_size)

        # pass class token through all layers
        for layer in self.ca_layers:
            class_token = layer(class_token, x, key_value_mask=mask, context=context)

        class_token = self.final_norm(class_token)

        return class_token.squeeze(1)
