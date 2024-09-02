from collections.abc import Mapping

import torch
from torch import Tensor, nn, randn

from salt.models.transformer import TransformerCrossAttentionLayer
from salt.stypes import Tensors
from salt.utils.tensor_utils import flatten_tensor_dict, masked_softmax


class Pooling(nn.Module): ...


class GlobalAttentionPooling(Pooling):
    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn = nn.Linear(input_size, 1)

    def forward(
        self,
        x: dict[str, Tensor] | dict,
        pad_mask: dict | None = None,
    ):
        x_flat = flatten_tensor_dict(x, exclude=["objects"])

        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1).unsqueeze(-1)

        weights = masked_softmax(self.gate_nn(x_flat), pad_mask, dim=1)
        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x_flat.shape[0], 1, x_flat.shape[2]), device=x_flat.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x_flat = torch.cat([x_flat, x_pad], dim=1)

        return (x_flat * weights).sum(dim=1)


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
        self.ca_layers = nn.ModuleList([
            TransformerCrossAttentionLayer(input_size, mha_config, dense_config, context_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(input_size)
        self.class_token = nn.Parameter(randn(1, 1, input_size))

    def expand_class_token(self, x: Tensor | dict):
        if isinstance(x, dict):
            return self.class_token.expand(x[list(x.keys())[0]].shape[0], 1, self.input_size)

        return self.class_token.expand(x.shape[0], 1, self.input_size)


class DictCrossAttentionPooling(BaseCrossAttentionPooling):
    def forward(
        self,
        x: Tensors,
        pad_mask: dict | None = None,
        context: Tensor | None = None,
    ):
        class_token = self.expand_class_token(x)
        for layer in self.ca_layers:
            new_class_token = torch.zeros_like(class_token)
            for input_name in x:
                if input_name == "objects":
                    continue
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
    def forward(
        self,
        x: Tensors,
        pad_mask: dict | None = None,
        context: Tensor | None = None,
    ):
        x_flat = flatten_tensor_dict(x, exclude=["objects"])
        class_token = self.expand_class_token(x_flat)
        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1)

        for layer in self.ca_layers:
            class_token = layer(class_token, x_flat, key_value_mask=pad_mask, context=context)

        class_token = self.final_norm(class_token)
        return class_token.squeeze(1)


class NodeQueryGAP(Pooling):
    def __init__(self, input_size: int):
        super().__init__()
        self.gate_nn_1 = nn.Linear(input_size, 1)
        self.gate_nn_2 = nn.Linear(input_size, 1)

    def forward(
        self,
        x: Tensors,
        pad_mask: dict | None = None,
    ):
        # Global Attention Pooling applied to both the decoder kv embeddings and
        # the encoder embeddings.
        x_nodes = flatten_tensor_dict(x, exclude=["objects"])

        if pad_mask is not None:
            pad_mask = torch.cat(list(pad_mask.values()), dim=1).unsqueeze(-1)

        weights = masked_softmax(self.gate_nn_1(x_nodes), pad_mask, dim=1)

        # add padded track to avoid error in onnx model when there are no tracks in the jet
        weight_pad = torch.zeros((weights.shape[0], 1, weights.shape[2]), device=weights.device)
        x_pad = torch.zeros((x_nodes.shape[0], 1, x_nodes.shape[2]), device=x_nodes.device)
        weights = torch.cat([weights, weight_pad], dim=1)
        x_nodes = torch.cat([x_nodes, x_pad], dim=1)
        pooled_nodes = (x_nodes * weights).sum(dim=1)

        # get pooled queries
        emb_queries = x["objects"]["embed"]
        query_pad = torch.zeros(
            (emb_queries.shape[0], 1, emb_queries.shape[2]),
            device=emb_queries.device,
        )
        padded_queries = torch.cat([emb_queries, query_pad], dim=1)
        weights = self.gate_nn_2(padded_queries).softmax(1)
        pooled_queries = (padded_queries * weights).sum(dim=1)

        # concatenate pooled nodes and queries
        return torch.cat([pooled_nodes, pooled_queries], dim=-1)
