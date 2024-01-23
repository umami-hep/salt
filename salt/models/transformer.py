from collections.abc import Mapping
from itertools import combinations

from mup import MuReadout
from torch import BoolTensor, Tensor, cat, nn

from salt.models.attention import MultiheadAttention
from salt.models.dense import Dense


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer.

    We switched from normformer style to pre-ln style in MR !132.

    It contains:
    - Multihead attention block
    - A feedforward network (which can take optional context information)
    - Layernorms
    - Residual connections
    - Optional edge update blocks
    """

    def __init__(
        self,
        embed_dim: int,
        mha_config: Mapping,
        dense_config: Mapping | None = None,
        context_dim: int = 0,
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        muP: bool = False,
    ) -> None:
        """Init method of TransformerEncoderBlock.

        Parameters
        ----------
        embed_dim : int
            The embedding dimension of the transformer block
        mha_config : int
            Configuration for the MultiheadAttention block
        dense_config : Mapping
            Configuration for the Dense network
        context_dim: int
            The size of the context tensor
        edge_embed_dim : int
            The embedding dimension of the transformer block for edge features
        update_edges: bool
            Value indicating whether to update edge features via attention
        muP: bool, optional,
            Whether to use the muP parametrisation (impacts initialisation).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.update_edges = update_edges
        self.muP = muP

        # The main blocks in the transformer
        self.mha = MultiheadAttention(
            embed_dim,
            edge_embed_dim=edge_embed_dim,
            update_edges=update_edges,
            muP=self.muP,
            **mha_config,
        )
        if dense_config:
            self.dense = Dense(
                input_size=embed_dim,
                output_size=embed_dim,
                context_size=context_dim,
                muP=self.muP,
                **dense_config,
            )
        else:
            self.register_buffer("dense", None)

        # The multiple normalisation layers to keep it all stable
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.edge_embed_dim > 0:
            self.enorm1 = nn.LayerNorm(edge_embed_dim)
            if self.update_edges:
                self.enorm2 = nn.LayerNorm(edge_embed_dim)

    def forward(
        self,
        x: Tensor,
        edge_x: Tensor | None = None,
        pad_mask: BoolTensor | None = None,
        context: Tensor | None = None,
        attn_mask: BoolTensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        if edge_x is not None:
            xi, edge_xi = self.mha(
                self.norm1(x),
                edges=self.enorm1(edge_x),
                q_mask=pad_mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
        else:
            xi = self.mha(
                self.norm1(x),
                q_mask=pad_mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        x = x + xi
        if self.update_edges:
            edge_x = edge_x + self.enorm2(edge_xi)
        if self.dense:
            x = x + self.dense(self.norm2(x), context)

        if edge_x is not None:
            return x, edge_x

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
        dense_config: Mapping | None = None,
        context_dim: int = 0,
        out_dim: int = 0,
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        muP: bool = False,
    ) -> None:
        """Transformer encoder module.

        Parameters
        ----------
        embed_dim : int
            Feature size for input, output, and all intermediate layers
        num_layers : int
            Number of encoder layers used
        mha_config : nn.Module
            Keyword arguments for the mha block
        dense_config: Mapping
            Keyword arguments for [`salt.models.Dense`][salt.models.Dense]
        context_dim: int
            Dimension of the context inputs
        out_dim: int
            If set, a final linear layer resizes the outputs of the model
        edge_embed_dim : int
            Feature size for input and output of edge features
        update_edges: bool
            If set, edge features are updated in each encoder layer
        muP: bool, optional,
            Whether to use the muP parametrisation (impacts initialisation).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.update_edges = update_edges
        self.muP = muP

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim,
                mha_config,
                dense_config,
                context_dim,
                edge_embed_dim,
                update_edges if i != num_layers - 1 else False,
                self.muP,
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.muP:
            assert self.out_dim, "Need the out_dim layer for muP, \
                as this is the last layer of the muP-part of the model"
        if self.out_dim:
            if self.muP:
                self.final_linear = MuReadout(self.embed_dim, self.out_dim)
                self.final_linear.bias.data.zero_()
                self.final_linear.weight.data.zero_()
            else:
                self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

    def forward(
        self,
        x: Tensor | dict,
        edge_x: Tensor = None,
        pad_mask: Tensor | dict | None = None,
        **kwargs,
    ) -> Tensor:
        """Pass the input through all layers sequentially."""
        if isinstance(x, dict):
            x = cat(list(x.values()), dim=1)

        if isinstance(pad_mask, dict):
            pad_mask = cat(list(pad_mask.values()), dim=1)

        for layer in self.layers:
            if edge_x is not None:
                x, edge_x = layer(x, edge_x, pad_mask=pad_mask, **kwargs)
            else:
                x = layer(x, pad_mask=pad_mask, **kwargs)
        x = self.final_norm(x)

        # optional resizing layer
        if self.out_dim:
            x = self.final_linear(x)
        return x


class TransformerCrossAttentionLayer(TransformerEncoderLayer):
    """A transformer encoder layer with cross-attention."""

    def __init__(
        self,
        embed_dim: int,
        mha_config: Mapping,
        dense_config: Mapping | None = None,
        context_dim: int = 0,
        muP: bool = False,
    ) -> None:
        super().__init__(embed_dim, mha_config, dense_config, context_dim, muP)
        self.norm0 = nn.LayerNorm(embed_dim)

    def forward(  # type: ignore[override]
        self,
        query: Tensor,
        key_value: Tensor,
        query_mask: BoolTensor | None = None,
        key_value_mask: BoolTensor | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        additive = self.mha(
            self.norm1(query),
            self.norm0(key_value),
            q_mask=query_mask,
            kv_mask=key_value_mask,
        )
        if self.dense:
            additive = self.dense(self.norm2(additive), context)
        # The cross attention does not return residual+additive but only the additive
        # It is ment to be used as Ai+1 += CA.forward(Ai, Bi) if one wants to
        # have a residual connection
        # and Ai+1 = CA.forward(Ai, Bi) if a simple feed forward connection is intended
        # we also recoomend to use clones of Ai and Bi and avoid A += CA.forward(A, B)
        return additive


class TransformerCrossAttentionEncoder(nn.Module):
    """A stack of N transformer encoder layers interspersed with cross-attention
    layers between inputs types.
    """

    def __init__(
        self,
        input_names: list[str],
        embed_dim: int,
        num_layers: int,
        mha_config: Mapping,
        sa_dense_config: Mapping | None = None,
        ca_dense_config: Mapping | None = None,
        context_dim: int = 0,
        out_dim: int = 0,
        ca_every_layer: bool = False,
        update_edges: bool = False,
        muP: bool = False,
    ):
        super().__init__()
        self.input_names = input_names
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.ca_every_layer = ca_every_layer
        self.update_edges = update_edges
        self.muP = muP

        # Layers for each input type
        # need to use ModuleDict so device is set correctly
        self.type_layers = nn.ModuleDict({
            input_name: nn.ModuleList([
                TransformerEncoderLayer(
                    embed_dim,
                    mha_config,
                    sa_dense_config,
                    context_dim,
                    update_edges if i != num_layers - 1 else False,
                    muP,
                )
                for i in range(num_layers)
            ])
            for input_name in self.input_names
        })

        ca_layers = num_layers if ca_every_layer else 2 if num_layers > 1 else 1

        # module dict only supports string keys
        self.cross_layers = nn.ModuleDict({
            f"{input_name1}_{input_name2}": nn.ModuleList([
                TransformerCrossAttentionLayer(
                    embed_dim,
                    mha_config,
                    ca_dense_config,
                    context_dim,
                    muP,
                )
                for _ in range(ca_layers)
            ])
            for input_name1, input_name2 in combinations(self.input_names, 2)
        })

        self.final_norm = nn.LayerNorm(embed_dim)

        if self.muP:
            assert self.out_dim, "Need the out_dim layer for muP, \
                as this is the last layer of the muP-part of the model"

        # For resizing the output tokens
        if self.out_dim:
            if muP:
                self.final_linear = MuReadout(self.embed_dim, self.out_dim)
                self.final_linear.bias.data.zero_()
                self.final_linear.weight.data.zero_()
            else:
                self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

    def forward(
        self, x: dict[str, Tensor], pad_mask: dict[str, Tensor], edge_x: Tensor = None, **kwargs
    ) -> Tensor:
        """Pass the input through all layers sequentially."""
        if edge_x is not None:
            raise ValueError("Edge updates of Cross Attention Encoder not yet supported")

        # Merge inputs as specified
        for i in range(self.num_layers):
            # Self-attention for each type
            for it in self.input_names:
                x[it] = self.type_layers[it][i](x[it], pad_mask=pad_mask[it], **kwargs)

            # Cross-attention between pairs of types - symmetric update
            # i % len(self.cross_layers[layer_key]) evals to 0 for first layer
            # and final layer and 1 everywhere else to give behaviour we want
            prev_x = {k: x[k].clone() for k in x}
            if self.ca_every_layer:
                for it1, it2 in combinations(self.input_names, 2):
                    layer_key = f"{it1}_{it2}"
                    x[it1] += self.cross_layers[layer_key][i](
                        prev_x[it1], prev_x[it2], pad_mask[it1], pad_mask[it2], **kwargs
                    )
                    x[it2] += self.cross_layers[layer_key][i](
                        prev_x[it2], prev_x[it1], pad_mask[it2], pad_mask[it1], **kwargs
                    )
            elif i in {0, self.num_layers - 1}:
                for it1, it2 in combinations(self.input_names, 2):
                    layer_key = f"{it1}_{it2}"
                    x[it1] += self.cross_layers[layer_key][i // (self.num_layers - 1)](
                        prev_x[it1], prev_x[it2], pad_mask[it1], pad_mask[it2], **kwargs
                    )
                    x[it2] += self.cross_layers[layer_key][i // (self.num_layers - 1)](
                        prev_x[it2], prev_x[it1], pad_mask[it2], pad_mask[it1], **kwargs
                    )

        # Apply final normalization
        for it in self.input_names:
            x[it] = self.final_norm(x[it])

        # Optional resizing layer
        if self.out_dim:
            for it in self.input_names:
                x[it] = self.final_linear(x[it])

        return x
