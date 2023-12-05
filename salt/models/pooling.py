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

    def forward(self, x: Tensor | dict, pad_mask: dict | None = None):
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)

        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1).unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x), pad_mask, dim=1)

        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x = torch.cat([x, x_pad], dim=1)

        return (x * weights).sum(dim=1)


class BaseCrossAttentionPooling(Pooling):
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
        self.class_token = nn.Parameter(randn(1, 1, input_size))

    def expand_class_token(self, x: Tensor | dict):
        if isinstance(x, dict):
            return self.class_token.expand(x[list(x.keys())[0]].shape[0], 1, self.input_size)

        return self.class_token.expand(x.shape[0], 1, self.input_size)


class DictCrossAttentionPooling(BaseCrossAttentionPooling):
    def forward(self, x: dict, pad_mask: dict | None = None, context: Tensor | None = None):
        class_token = self.expand_class_token(x)
        for layer in self.ca_layers:
            new_class_token = torch.zeros_like(class_token)
            for input_name in x:
                new_class_token += layer(
                    class_token,
                    x[input_name],
                    key_value_mask=pad_mask[input_name] if pad_mask else None,
                    context=context,
                )
            class_token = new_class_token

        class_token = self.final_norm(class_token)
        return class_token.squeeze(1)


class TensorCrossAttentionPooling(BaseCrossAttentionPooling):
    def forward(self, x: Tensor, pad_mask: dict | None = None, context: Tensor | None = None):
        class_token = self.expand_class_token(x)
        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1)

        for layer in self.ca_layers:
            class_token = layer(class_token, x, key_value_mask=pad_mask, context=context)

        class_token = self.final_norm(class_token)
        return class_token.squeeze(1)
