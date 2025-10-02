from collections.abc import Mapping
from itertools import combinations

from torch import BoolTensor, Tensor, cat, nn

from salt.models.attention import MultiheadAttention
from salt.models.dense import Dense
from salt.stypes import Tensors

try:
    from mup import MuReadout as _MuReadout

except ImportError:
    _MuReadout = None


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with optional edge updates and context FFN.

    This layer applies:
    - Multi-head self-attention (optionally reading/updating edge features),
    - a feed-forward network (optionally conditioned on a context vector),
    - LayerNorm (pre-LN),
    - residual connections.

    Notes
    -----
    Switched from NormFormer-style to pre-LN in MR !132.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension for inputs/outputs and attention projections.
    mha_config : Mapping
        Keyword arguments forwarded to :class:`salt.models.attention.MultiheadAttention`.
    dense_config : Mapping | None, optional
        Keyword arguments forwarded to :class:`salt.models.dense.Dense`. If ``None``,
        the dense block is omitted. The default is ``None``.
    context_dim : int, optional
        Dimensionality of the optional context vector used by the dense block.
        The default is ``0``.
    edge_embed_dim : int, optional
        Edge embedding dimensionality (if using edge features). The default is ``0``.
    update_edges : bool, optional
        If ``True``, edge features are updated in the attention block. The default is ``False``.
    mup : bool, optional
        Whether to use μP parameterization. The default is ``False``.
    """

    def __init__(
        self,
        embed_dim: int,
        mha_config: Mapping,
        dense_config: Mapping | None = None,
        context_dim: int = 0,
        edge_embed_dim: int = 0,
        update_edges: bool = False,
        mup: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.update_edges = update_edges
        self.mup = mup

        # The main blocks in the transformer
        self.mha = MultiheadAttention(
            embed_dim,
            edge_embed_dim=edge_embed_dim,
            update_edges=update_edges,
            mup=self.mup,
            **mha_config,
        )
        if dense_config:
            self.dense = Dense(
                input_size=embed_dim,
                output_size=embed_dim,
                context_size=context_dim,
                mup=self.mup,
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
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply attention (and optional edge update) + feed-forward with residuals.

        Parameters
        ----------
        x : Tensor
            Node/track embeddings of shape ``[B, N, D]`` (``D = embed_dim``).
        edge_x : Tensor | None, optional
            Edge embeddings of shape ``[B, N, N, E]`` (``E = edge_embed_dim``). If provided,
            the attention block can read/update edge features. The default is ``None``.
        pad_mask : BoolTensor | None, optional
            Boolean padding mask of shape ``[B, N]`` where padded positions are ``True``.
            The default is ``None``.
        context : Tensor | None, optional
            Optional context vector of shape ``[B, C]`` passed into the dense block.
            The default is ``None``.
        attn_mask : BoolTensor | None, optional
            Attention mask of shape ``[B, N, N]`` where allowed positions are ``True``.
            The default is ``None``.
        attn_bias : Tensor | None, optional
            Optional attention bias (broadcastable to attention logits). The default is ``None``.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            If ``edge_x is None``: the updated token embeddings ``x`` of shape ``[B, N, D]``.
            If ``edge_x`` is provided: ``(x, edge_x)`` with updated tokens and edges.
        """
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
    """Stack of transformer encoder layers with final normalization (and optional projection).

    A message-passing encoder that preserves token dimensionality across layers and
    uses multi-head attention for token-token interactions.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension for inputs/outputs and attention projections.
    num_layers : int
        Number of encoder layers.
    mha_config : Mapping
        Keyword arguments forwarded to :class:`salt.models.attention.MultiheadAttention`
        (used inside each :class:`TransformerEncoderLayer`).
    dense_config : Mapping | None, optional
        Keyword arguments forwarded to :class:`salt.models.dense.Dense` (per-layer).
        If ``None``, the dense block in each layer is omitted. The default is ``None``.
    context_dim : int, optional
        Dimensionality of the optional context vector for dense blocks. The default is ``0``.
    out_dim : int, optional
        If non-zero, apply a final linear projection from ``embed_dim`` to ``out_dim``.
        The default is ``0``.
    edge_embed_dim : int, optional
        Edge embedding dimension (if using edge features). The default is ``0``.
    update_edges : bool, optional
        If ``True``, layers update edges (except the last layer). The default is ``False``.
    mup : bool, optional
        Whether to use μP parameterization. The default is ``False``.
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
        mup: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.update_edges = update_edges
        self.mup = mup
        self.featurewise = nn.ModuleList()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim,
                mha_config,
                dense_config,
                context_dim,
                edge_embed_dim,
                update_edges if i != num_layers - 1 else False,
                self.mup,
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.mup:
            assert self.out_dim, "Need the out_dim layer for muP, \
                as this is the last layer of the muP-part of the model"
        if self.out_dim:
            if self.mup and _MuReadout:
                self.final_linear = _MuReadout(self.embed_dim, self.out_dim)
                self.final_linear.bias.data.zero_()
                self.final_linear.weight.data.zero_()
            else:
                self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

    def forward(
        self,
        x: Tensor | dict,
        edge_x: Tensor | None = None,
        pad_mask: Tensor | dict | None = None,
        inputs: Tensors | None = None,
        **kwargs,
    ) -> Tensor:
        """Encode a sequence (optionally with edges) through all layers.

        Parameters
        ----------
        x : Tensor | dict
            Input token embeddings. If a dict, values are concatenated along the
            sequence dimension: each ``Tensor`` of shape ``[B, N_i, D]`` leading to
            concatenated ``[B, sum_i N_i, D]``.
        edge_x : Tensor | None, optional
            Optional edge embeddings of shape ``[B, N, N, E]``. The default is ``None``.
        pad_mask : Tensor | dict | None, optional
            Padding mask(s). If a dict, values (``[B, N_i]``) are concatenated to ``[B, N]``.
            The default is ``None``.
        inputs : Tensors | None, optional
            Original per-stream inputs passed to any configured featurewise transforms.
            The default is ``None``.
        **kwargs
            Extra keyword arguments forwarded to each layer's attention/FFN (e.g. ``attn_mask``,
            ``attn_bias``, etc.).

        Returns
        -------
        Tensor
            Encoded token embeddings of shape ``[B, N, D_out]``, where ``D_out = out_dim``
            if a final projection is configured, else ``embed_dim``.
        """
        if isinstance(x, dict):
            x = cat(list(x.values()), dim=1)

        if isinstance(pad_mask, dict):
            pad_mask = cat(list(pad_mask.values()), dim=1)

        for i, layer in enumerate(self.layers):
            if len(self.featurewise) > 0:
                x = self.featurewise[i](inputs, x)
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
    """Transformer layer that computes **additive** cross-attention updates.

    Unlike a standard encoder layer, this module returns *only* the additive update
    (no residual is applied internally). Use it as:

    - residual style: ``A_{i+1} = A_i + CA(A_i, B_i)``
    - feed-forward style: ``A_{i+1} = CA(A_i, B_i)``

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension for inputs/outputs and attention projections.
    mha_config : Mapping
        Keyword arguments forwarded to :class:`salt.models.attention.MultiheadAttention`.
    dense_config : Mapping | None, optional
        Keyword arguments forwarded to :class:`salt.models.dense.Dense`. If ``None``,
        the dense block is omitted. The default is ``None``.
    context_dim : int, optional
        Dimensionality of the optional context vector for the dense block. The default is ``0``.
    mup : bool, optional
        Whether to use μP parameterization. The default is ``False``.
    """

    def __init__(
        self,
        embed_dim: int,
        mha_config: Mapping,
        dense_config: Mapping | None = None,
        context_dim: int = 0,
        mup: bool = False,
    ) -> None:
        super().__init__(embed_dim, mha_config, dense_config, context_dim, mup)
        self.norm0 = nn.LayerNorm(embed_dim)

    def forward(  # type: ignore[override]
        self,
        query: Tensor,
        key_value: Tensor,
        query_mask: BoolTensor | None = None,
        key_value_mask: BoolTensor | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        """Compute the **additive** cross-attention update.

        Parameters
        ----------
        query : Tensor
            Query sequence of shape ``[B, N_q, D]``.
        key_value : Tensor
            Key/value sequence of shape ``[B, N_kv, D]``.
        query_mask : BoolTensor | None, optional
            Padding mask for the query sequence of shape ``[B, N_q]``. The default is ``None``.
        key_value_mask : BoolTensor | None, optional
            Padding mask for the key/value sequence of shape ``[B, N_kv]``.
            The default is ``None``.
        context : Tensor | None, optional
            Optional context vector for the dense block of shape ``[B, C]``.
            The default is ``None``.

        Returns
        -------
        Tensor
            Additive update tensor of shape ``[B, N_q, D]``. No residual is added internally.
        """
        additive = self.mha(
            self.norm1(query),
            self.norm0(key_value),
            q_mask=query_mask,
            kv_mask=key_value_mask,
        )
        if self.dense:
            additive = self.dense(self.norm2(additive), context)
        # The cross attention does not return residual+additive but only the additive.
        # Recommended usage: A_{i+1} = A_i + CA(A_i, B_i) for residual style.
        return additive


class TransformerCrossAttentionEncoder(nn.Module):
    """Interleaved self-attention encoders with symmetric cross-attention updates.

    For each input stream, a stack of self-attention layers is applied. Between
    streams, symmetric cross-attention updates are optionally applied either at
    every layer (``ca_every_layer=True``) or only at the first/last layer.

    Parameters
    ----------
    input_names : list[str]
        Ordered list of input stream names (used to key the dictionaries).
    embed_dim : int
        Token embedding dimension for inputs/outputs and attention projections.
    num_layers : int
        Number of self-attention layers per input stream.
    mha_config : Mapping
        Keyword arguments forwarded to :class:`salt.models.attention.MultiheadAttention`.
    sa_dense_config : Mapping | None, optional
        Keyword arguments for the self-attention layers' dense blocks. The default is ``None``.
    ca_dense_config : Mapping | None, optional
        Keyword arguments for the cross-attention layers' dense blocks. The default is ``None``.
    context_dim : int, optional
        Dimensionality of the optional context vector for dense blocks. The default is ``0``.
    out_dim : int, optional
        If non-zero, apply a final linear projection from ``embed_dim`` to ``out_dim``.
        The default is ``0``.
    ca_every_layer : bool, optional
        If ``True``, apply cross-attention after **every** self-attention layer;
        otherwise only on the first and last layers. The default is ``False``.
    update_edges : bool, optional
        Reserved for parity with other encoders; edge updates are not supported here.
        The default is ``False``.
    mup : bool, optional
        Whether to use μP parameterization. The default is ``False``.
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
        mup: bool = False,
    ):
        super().__init__()
        self.input_names = input_names
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.ca_every_layer = ca_every_layer
        self.update_edges = update_edges
        self.mup = mup

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
                    mup,
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
                    mup,
                )
                for _ in range(ca_layers)
            ])
            for input_name1, input_name2 in combinations(self.input_names, 2)
        })

        self.final_norm = nn.LayerNorm(embed_dim)

        if self.mup:
            assert self.out_dim, "Need the out_dim layer for mup, \
                as this is the last layer of the mup-part of the model"

        # For resizing the output tokens
        if self.out_dim:
            if mup and _MuReadout:
                self.final_linear = _MuReadout(self.embed_dim, self.out_dim)
                self.final_linear.bias.data.zero_()
                self.final_linear.weight.data.zero_()
            else:
                self.final_linear = nn.Linear(self.embed_dim, self.out_dim)

    def forward(
        self,
        x: dict[str, Tensor],
        pad_mask: dict[str, Tensor],
        edge_x: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Run interleaved self-attention per stream with symmetric cross-attention updates.

        Parameters
        ----------
        x : dict[str, Tensor]
            Mapping from input name to token embeddings of shape ``[B, N_i, D]``.
            The set of keys must match ``input_names``.
        pad_mask : dict[str, Tensor]
            Mapping from input name to boolean padding masks of shape ``[B, N_i]``.
        edge_x : Tensor | None, optional
            Unused (edge updates are not supported). If provided, a ``ValueError`` is raised.
        **kwargs
            Extra keyword arguments forwarded to layers (e.g. ``attn_mask``, ``attn_bias``,
            and/or ``context`` for dense blocks).

        Returns
        -------
        dict[str, Tensor]
            Mapping with the same keys as ``x``, each containing the encoded sequence of shape
            ``[B, N_i, D_out]``, where ``D_out = out_dim`` if a final projection is configured,
            else ``embed_dim``.

        Raises
        ------
        ValueError
            If edge_x is given (Edge updates of Cross Attention are not yet supported)
        """
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
