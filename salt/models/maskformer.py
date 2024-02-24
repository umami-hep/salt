from collections.abc import Mapping

import torch
from torch import Tensor, nn

from salt.models import MaskFormerLoss
from salt.models.transformer_v2 import GLU, CrossAttention, SelfAttention
from salt.stypes import Tensors


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

        intermediate_outputs: list | None = [] if self.aux_loss else None
        for layer in self.layers:
            if self.aux_loss:
                assert intermediate_outputs is not None
                intermediate_outputs.append({"embed": q, **self.get_preds(q, x, pad_mask)})
            q, x = layer(q, x, kv_mask=pad_mask)

        preds["objects"] = {"embed": q, "x": x, **self.get_preds(q, x, pad_mask)}
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

        self.q_ca = CrossAttention(embed_dim=embed_dim, num_heads=n_heads)
        self.q_sa = SelfAttention(embed_dim=embed_dim, num_heads=n_heads)
        self.q_dense = GLU(embed_dim)
        if bidirectional_ca:
            self.kv_ca = CrossAttention(embed_dim=embed_dim, num_heads=n_heads)
            self.kv_dense = GLU(embed_dim)
        self.mask_net = mask_net

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor | None = None) -> Tensor:
        attn_mask = None

        # if we want to do mask attention
        if self.mask_attention:
            # If a BoolTensor is provided, positions with ``True`` are not allowed
            # to attend while ``False`` values will be unchanged.
            attn_mask = (get_masks(kv, q, self.mask_net, kv_mask).sigmoid() < 0.1).detach()

            # if the attn mask is invalid for a given query, allow it to attend everywhere
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        # update queries with cross attention from nodes
        q = q + self.q_ca(q, kv, kv_mask=kv_mask, attn_mask=attn_mask)

        # update queries with self attention
        q = q + self.q_sa(q)

        # dense update
        q = q + self.q_dense(q)

        # update nodes with cross attention from queries and dense layer
        if self.bidirectional_ca:
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(1, 2)
            kv = kv + self.kv_ca(kv, q, q_mask=kv_mask, attn_mask=attn_mask)
            kv = kv + self.kv_dense(kv)

        return q, kv
