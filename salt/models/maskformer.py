from collections.abc import Mapping

import torch
from torch import Tensor, nn

from salt.models import MaskFormerLoss
from salt.models.transformer_v2 import GLU, Attention
from salt.stypes import Tensors
from salt.utils.mask_utils import indices_from_mask


class MaskDecoder(nn.Module):
    """Mask decoder for Salt - Uses constituent embeddings to generate object queries,
    from which a classification task and mask prediction task, as well as any further object
    level tasks, are performed.

    Based on
    - https://github.com/facebookresearch/MaskFormer
    - https://github.com/facebookresearch/Mask2Former
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

    def get_preds(self, queries: Tensor, mask_tokens: Tensor, pad_mask: Tensor | None = None):
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
        """Forward pass through the MaskDecoder. Utilises the encoder embeddings to
        generate M query vectors, from which a classification task and mask prediction are
        performed. If labels are provided, the HungarianMatcher algorithm is implemented
        to find the optimal match between predictions and labels. The model predictions are
        then re-ordered to match the labels, and a final loss is calculated.

        Parameters
        ----------
        preds : dict[str, Tensor]
            Dictionary containing existing predictions from the encoder, should contain the
            key 'embed_xs' which is the encoder embeddings.
        tasks : nn.ModuleList
            The tasks for the model to perform, excluding the object mask prediction and
            classification. Any tasks that use the input stream 'objects' will be used
        pad_mask : Tensor, optional
            Masks for input padding, by default None
        labels : dict[str, Tensor] | None, optional
            Dict containing labels, by default None

        Returns
        -------
        preds : dict[str, Tensor]
            Dictionary containing the updated model predictions, with all object predictions
            stored in preds["objects"]
        labels : dict[str, Tensor] | None
            Updated labels, containing all default labels and any additional labels generated
            by the model (e.g. regression targets). If input labels are None, returns None.
        loss :
            Loss value if labels are provided, otherwise returns an empty dict.
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


def get_masks(x: Tensor, q: Tensor, mask_net: nn.Module, input_pad_mask: Tensor | None = None):
    mask_tokens = mask_net(q)
    pred_masks = torch.einsum("bqe,ble->bql", mask_tokens, x)

    if input_pad_mask is not None:
        pred_masks[input_pad_mask.unsqueeze(1).expand_as(pred_masks)] = torch.finfo(
            pred_masks.dtype
        ).min

    return pred_masks


def get_maskformer_outputs(
    objects,
    max_null=0.9,
    apply_reorder=True,
):
    """Takes objects for a single MF global prediction and converts them to a more useful
    form.

        -  Keep only objects which have `pnull < max_null`
        - Converts the mask logits to mask indices, see salt.utils.mask_utils.indices_from_mask

    """
    # Convert the (N,M) -> (M,) mask indices
    masks = objects["masks"]
    class_probs = objects["class_probs"]
    regression = objects["regression"]
    object_leading = objects["regression"]
    n_tracks = masks.shape[-1]
    n_obj = masks.shape[1]
    n_reg = regression.shape[-1]

    # If we have a jet with no tracks,
    if n_tracks == 0:
        return (
            torch.ones((1, n_obj)) * torch.nan,
            None,
            class_probs,
            torch.ones((1, n_obj, n_reg)) * torch.nan,
        )
    # For testing purposes - this will likely blow up our fake rate
    null_preds = class_probs[:, :, -1] > max_null
    if not null_preds.any():
        # If we have no predicted objects, we return dummy values
        return (
            torch.ones((1, n_obj)) * torch.nan,
            torch.zeros((1, n_obj, n_tracks), dtype=torch.bool),
            class_probs,
            torch.ones((1, n_obj, n_reg)) * torch.nan,
        )

    masks = masks.sigmoid() > 0.5
    object_leading[null_preds] = -999
    regression[null_preds] = torch.nan

    if apply_reorder:
        # Define the leading object as the one with the highest regression[0] value
        # in vertexing case, this is the pT
        order = torch.argsort(object_leading[:, :, 0], descending=True)
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

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor | None = None) -> Tensor:
        attn_mask = None
        # return q, kv
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
