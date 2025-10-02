from collections.abc import Mapping

import torch
from torch import Tensor, nn

from salt.models import MaskFormerLoss
from salt.models.transformer_v2 import GLU, Attention
from salt.stypes import Tensors
from salt.utils.mask_utils import indices_from_mask


class MaskDecoder(nn.Module):
    """Mask decoder for Salt.

    Uses constituent/node embeddings to generate a fixed number of object
    queries. From these queries, it produces (i) classification logits/probabilities
    and (ii) mask logits over the input sequence. Optional auxiliary outputs at
    intermediate layers can be returned for deep supervision. If labels are
    provided, the loss is computed via a MaskFormer-style matching/loss.

    The design takes inspiration from:
      * https://github.com/facebookresearch/MaskFormer
      * https://github.com/facebookresearch/Mask2Former

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the query and node embeddings.
    num_layers : int
        Number of decoder layers.
    md_config : Mapping
        Configuration mapping passed to each :class:`MaskDecoderLayer`
        (e.g. ``n_heads``, ``mask_attention``, ``bidirectional_ca``).
    class_net : nn.Module
        Head mapping query embeddings to class logits (shape ``[B, M, C]`` or ``[B, M, 1]``).
    mask_net : nn.Module
        Head mapping query embeddings to mask tokens (used to build attention/masks).
    num_objects : int
        Number of object queries ``M``.
    loss_config : Mapping
        Configuration mapping forwarded to :class:`MaskFormerLoss`.
    aux_loss : bool, optional
        If ``True``, store intermediate predictions after each decoder layer
        for auxiliary losses, by default ``False``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        md_config: Mapping,
        class_net: nn.Module,
        mask_net: nn.Module,
        num_objects: int,
        loss_config: Mapping,
        aux_loss: bool = False,
    ):
        super().__init__()
        self.aux_loss = aux_loss

        self.inital_q = nn.Parameter(torch.empty((num_objects, embed_dim)))
        nn.init.normal_(self.inital_q)

        self.class_net = class_net
        self.mask_net = mask_net

        self.layers = nn.ModuleList([
            MaskDecoderLayer(embed_dim, mask_net=mask_net, **md_config) for _ in range(num_layers)
        ])
        # Two norm layers may be overkill but it should help with model stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mask_loss = MaskFormerLoss(**loss_config, num_objects=num_objects)

    def get_preds(
        self,
        queries: Tensor,
        mask_tokens: Tensor,
        pad_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute class probabilities/logits and mask logits from queries.

        Parameters
        ----------
        queries : Tensor
            Query embeddings of shape ``[B, M, E]``.
        mask_tokens : Tensor
            Input/node embeddings ``x`` over which masks are predicted,
            of shape ``[B, L, E]`` (often the encoder outputs).
        pad_mask : Tensor | None , optional
            Boolean/byte mask for padded positions in ``mask_tokens`` of
            shape ``[B, L]``; padded positions are suppressed in mask logits,
            by default ``None``.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with:
            - ``"class_logits"``: logits of shape ``[B, M, C]`` (or ``[B, M, 1]``).
            - ``"class_probs"``: probabilities of shape ``[B, M, C]`` (sigmoid-expanded if binary).
            - ``"masks"``: mask logits over input positions of shape ``[B, M, L]``.
        """
        # get class predictions from queries
        class_logits = self.class_net(queries)
        if class_logits.shape[-1] == 1:
            class_probs = class_logits.sigmoid()
            class_probs = torch.cat([1 - class_probs, class_probs], dim=-1)
        else:
            class_probs = class_logits.softmax(-1)

        # get mask predictions from queries and mask tokens
        pred_masks = get_masks(mask_tokens, queries, self.mask_net, pad_mask)

        return {"class_logits": class_logits, "class_probs": class_probs, "masks": pred_masks}

    def forward(
        self,
        preds: Tensors,
        tasks: nn.ModuleList,
        pad_mask: Tensor = None,
        labels: Tensors | None = None,
    ):
        """Run the decoder and optionally compute the loss.

        Utilises the encoder embeddings to generate ``M`` query vectors, from which
        a classification and mask prediction are performed. If ``labels`` are provided,
        a Hungarian-style matching and loss computation is applied via
        :class:`MaskFormerLoss`. Intermediate outputs can be stored when ``aux_loss=True``.

        Parameters
        ----------
        preds : Tensors
            Existing predictions containing at least ``"embed_xs"`` with encoder
            embeddings of shape ``[B, L, E]``.
        tasks : nn.ModuleList
            Additional task heads. Any heads consuming the ``"objects"`` stream
            will be invoked inside the loss object.
        pad_mask : Tensor, optional
            Padding mask for encoder embeddings of shape ``[B, L]``; ``True``/``1``
            denotes padded positions, by default ``None``.
        labels : Tensors | None, optional
            Ground-truth labels used by the loss, by default ``None``.

        Returns
        -------
        tuple
            If ``labels is None``:
                ``(preds, labels, {})`` where
                - ``preds["objects"]`` contains keys
                  ``"embed"`` (``[B, M, E]``), ``"x"`` (unpadded ``[B, L, E]``),
                  ``"class_logits"``, ``"class_probs"``, ``"masks"``.
                - ``labels`` is ``None``.
                - Third element is an empty dict.
            If ``labels is not None``:
                Return value of :meth:`MaskFormerLoss.__call__`, typically
                ``(preds, labels, loss_dict)``.

        Notes
        -----
        - A dummy padded token is appended to the inputs to maintain ONNX
          compatibility; it is removed again before returning.
        - When ``aux_loss`` is enabled, intermediate predictions after each
          decoder layer are stored in ``preds["intermediate_outputs"]``.
        """
        # MF only supports one input, if we have multiple then we have no way of knowing
        # what section of the embedding relates to objects we want to generate masks for
        if isinstance(pad_mask, dict):
            assert len(pad_mask) == 1, "Maskformer only supports one input."
            pad_mask = next(iter(pad_mask.values()))

        x = preds["embed_xs"]
        # apply norm
        q = self.norm1(self.inital_q.expand(x.shape[0], -1, -1))
        x = self.norm2(x)

        # Add a dummy track to the inputs (and to pad mask) to stop onnx complaining
        xpad = torch.zeros((x.shape[0], 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        x = torch.cat([x, xpad], dim=1)
        if pad_mask is not None:
            padpad_mask = torch.zeros(
                (pad_mask.shape[0], 1), device=pad_mask.device, dtype=pad_mask.dtype
            )
            pad_mask = torch.cat([pad_mask, padpad_mask], dim=1)

        intermediate_outputs: list | None = [] if self.aux_loss else None
        for layer in self.layers:
            if self.aux_loss:
                assert intermediate_outputs is not None
                intermediate_outputs.append({"embed": q, **self.get_preds(q, x, pad_mask)})
            q, x = layer(q, x, kv_mask=pad_mask)
        mf_preds = self.get_preds(q, x, pad_mask)

        # Un-pad the embedding x, get the mf_predictions, and then unpad them as well
        preds["objects"] = {"embed": q, "x": x[:, :-1, :], **mf_preds}
        preds["objects"]["masks"] = preds["objects"]["masks"][:, :, :-1]
        if self.aux_loss:
            preds["intermediate_outputs"] = intermediate_outputs

        if labels is not None:
            return self.mask_loss(preds, tasks, labels)

        return preds, labels, {}


def get_masks(
    x: Tensor,
    q: Tensor,
    mask_net: nn.Module,
    input_pad_mask: Tensor | None = None,
) -> Tensor:
    """Compute mask logits over input tokens conditioned on queries.

    Parameters
    ----------
    x : Tensor
        Input/node embeddings of shape ``[B, L, E]``.
    q : Tensor
        Query embeddings of shape ``[B, M, E]``.
    mask_net : nn.Module
        Module mapping queries to mask tokens; expected to output
        ``mask_tokens = mask_net(q)`` with shape ``[B, M, E]``.
    input_pad_mask : Tensor | None, optional
        Padding mask for inputs of shape ``[B, L]``; padded positions are set
        to the minimum representable value in the output, by default ``None``.

    Returns
    -------
    Tensor
        Mask logits of shape ``[B, M, L]`` computed as
        ``einsum('bqe,ble->bql')`` between mask tokens and inputs.
    """
    mask_tokens = mask_net(q)
    pred_masks = torch.einsum("bqe,ble->bql", mask_tokens, x)

    if input_pad_mask is not None:
        pred_masks[input_pad_mask.unsqueeze(1).expand_as(pred_masks)] = torch.finfo(
            pred_masks.dtype
        ).min

    return pred_masks


def get_maskformer_outputs(
    objects: Mapping[str, Tensor],
    max_null: float = 0.5,
    apply_reorder: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert raw MaskFormer-style outputs to convenient per-object tensors.

    This helper:
      1. Thresholds the "null" class probability and suppresses masks/regression
         for objects with ``p_null > max_null``.
      2. Converts per-position mask logits into sparse mask indices via
         :func:`salt.utils.mask_utils.indices_from_mask`.
      3. Optionally reorders objects so that the "leading" object is first
         (highest regression[0], e.g. pT in vertexing).

    Parameters
    ----------
    objects : Mapping[str, Tensor]
        Dictionary with keys at least:
        - ``"masks"``: mask logits of shape ``[B, M, L]``.
        - ``"class_probs"``: class probabilities of shape ``[B, M, C]`` (last class is null).
        - ``"regression"``: regression targets/predictions of shape ``[B, M, R]``.
    max_null : float, optional
        Maximum allowed null probability ``p_null`` for an object to be kept,
        by default ``0.5``.
    apply_reorder : bool, optional
        If ``True``, reorder objects in descending order of ``regression[..., 0]``,
        by default ``True``.

    Returns
    -------
    leading_regression : torch.Tensor
        Tensor of shape ``[B, R]`` for the leading object (after optional reordering).
    obj_indices : torch.Tensor | None
        Sparse indices of masks per object with shape ``[B, M]``; values are
        positions in ``[0, L)`` (or ``NaN`` when undefined). May be ``None``
        if there are no tracks (``L == 0``).
    class_probs : torch.Tensor
        Possibly-reordered class probabilities of shape ``[B, M, C]``.
    regression : torch.Tensor
        Possibly-reordered regression tensor of shape ``[B, M, R]`` with ``NaN``
        for objects deemed null.

    Notes
    -----
    - If there are no input tracks/tokens (``L == 0``), dummy tensors filled with
      ``NaN`` are returned for indices and regression.
    - Masks are thresholded at ``0.5`` after a sigmoid to produce boolean masks
      prior to conversion to indices.
    """
    # Convert the (N,M) -> (M,) mask indices
    masks = objects["masks"]
    class_probs = objects["class_probs"]
    regression = objects["regression"]
    n_tracks = masks.shape[-1]
    n_obj = masks.shape[1]
    n_reg = regression.shape[-1]

    # If we have a jet with no tracks,
    if n_tracks == 0:
        return (
            torch.full((1, n_obj), torch.nan),
            None,
            class_probs,
            torch.full((1, n_obj, n_reg), torch.nan),
        )
    # For testing purposes - this will likely blow up our fake rate
    null_preds = class_probs[:, :, -1] > max_null
    if not null_preds.any():
        # If we have no predicted objects, we return dummy values
        return (
            torch.full((1, n_obj), torch.nan),
            torch.arange(n_tracks).unsqueeze(0).expand(1, n_tracks),
            class_probs,
            torch.full((1, n_obj, n_reg), torch.nan),
        )

    masks = masks.sigmoid() > 0.5
    expanded_null = null_preds.unsqueeze(-1).expand(-1, -1, masks.size(-1))
    masks[expanded_null] = torch.zeros_like(masks)[expanded_null]
    regression[null_preds] = torch.nan

    if apply_reorder:
        # Define the leading object as the one with the highest regression[0] value
        # in vertexing case, this is the pT. We first set these values to non-nans, as
        # argsort will otherwise not work correctly when in athena, and then set them back
        regression[null_preds] = -torch.inf
        order = torch.argsort(regression[:, :, 0], descending=True)
        regression[null_preds] = torch.nan
        order_expanded = order.unsqueeze(-1).expand(-1, -1, masks.size(-1))

        # Use gather to reorder tensors along a specific dimension
        masks = torch.gather(masks, 1, order_expanded)
        class_probs = torch.gather(
            class_probs, 1, order.unsqueeze(-1).expand(-1, -1, class_probs.size(-1))
        )
        regression = torch.gather(
            regression, 1, order.unsqueeze(-1).expand(-1, -1, regression.size(-1))
        )
        # Define the leading object as that with the highest [0] (pt for vertexing)
    leading_regression = regression[:, 0]

    # Convert our masks (N,M), now in pT order, to be (M,) indices
    obj_indices = indices_from_mask(masks)

    return leading_regression, obj_indices, class_probs, regression


class MaskDecoderLayer(nn.Module):
    """Single decoder layer used in :class:`MaskDecoder`.

    Applies (1) cross-attention from queries to inputs, (2) self-attention
    among queries, (3) a gated feed-forward (GLU) update, and optionally
    (4) a bidirectional cross-attention update from inputs to queries.
    Mask-guided attention can be enabled to sparsify cross-attention.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension ``E``.
    n_heads : int
        Number of attention heads.
    mask_attention : bool
        If ``True``, build a boolean attention mask from predicted masks to
        restrict cross-attention to confident positions.
    bidirectional_ca : bool
        If ``True``, also update inputs via cross-attention from queries.
    mask_net : nn.Module
        Module mapping queries to mask tokens used when ``mask_attention=True``.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mask_attention: bool,
        bidirectional_ca: bool,
        mask_net: nn.Module,
    ) -> None:
        super().__init__()

        self.mask_attention = mask_attention
        self.bidirectional_ca = bidirectional_ca

        self.q_ca = Attention(embed_dim=embed_dim, num_heads=n_heads)
        self.q_sa = Attention(embed_dim=embed_dim, num_heads=n_heads)
        self.q_dense = GLU(embed_dim)
        if bidirectional_ca:
            self.kv_ca = Attention(embed_dim=embed_dim, num_heads=n_heads)
            self.kv_dense = GLU(embed_dim)
        self.mask_net = mask_net

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        kv_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one decoder layer step.

        Parameters
        ----------
        q : Tensor
            Query embeddings of shape ``[B, M, E]``.
        kv : Tensor
            Input/key-value embeddings of shape ``[B, L, E]``.
        kv_mask : Tensor | None, optional
            Padding mask for ``kv`` of shape ``[B, L]``, by default ``None``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple ``(q, kv)`` with updated query and (optionally) updated
            input embeddings, each maintaining their original shapes.
        """
        attn_mask = None
        # Return the q, kv
        # if we want to do mask attention
        if self.mask_attention:
            # New attention masking convention with transformers 2
            # Positions with True are allowed while False are masked
            # Compute masks and apply sigmoid
            attn_mask = get_masks(kv, q, self.mask_net, kv_mask).sigmoid()

            # Threshold and detach
            attn_mask = (attn_mask > 0.9).detach()
            # Check if all values along the last dimension are 0 (equivalent to `False` in boolean)
            # If so, set them to 1 (equivalent to `True` in boolean)
            newmask = torch.all(attn_mask == 0, dim=-1, keepdim=True).expand(attn_mask.shape)

            attn_mask = attn_mask | newmask

        # update queries with cross attention from nodes
        q = q + self.q_ca(q, kv=kv, kv_mask=kv_mask, attn_mask=attn_mask)

        # update queries with self attention
        q = q + self.q_sa(q)

        # dense update
        q = q + self.q_dense(q)

        # update nodes with cross attention from queries and dense layer
        if self.bidirectional_ca:
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(1, 2)
                newmask = torch.all(attn_mask == 1, dim=-1, keepdim=True).expand(attn_mask.shape)
                attn_mask = attn_mask | ~newmask.bool()

            kv = kv + self.kv_ca(kv, q, attn_mask=attn_mask)
            kv = kv + self.kv_dense(kv)
        return q, kv
