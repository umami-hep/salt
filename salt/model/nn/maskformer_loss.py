"""MaskFormer (DETR-style) matched loss over Hungarian assignments."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from salt.model.nn.matcher import HungarianMatcher

__all__ = ["MaskFormerLoss"]


@torch.jit.script
def dice_loss(inputs: Tensor, labels: Tensor):
    """DICE loss (similar to generalized IOU for masks); returns a scalar."""
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * labels).sum(-1)
    denominator = inputs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / len(inputs)


@torch.jit.script
def mask_ce_loss(inputs: Tensor, labels: Tensor):
    """Binary cross-entropy loss for masks, mean-reduced per example; returns a scalar."""
    loss = functional.binary_cross_entropy_with_logits(inputs, labels, reduction="none")
    loss = loss.mean(1)
    return loss.sum() / len(inputs)


@torch.jit.script
def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2):
    """Sigmoid focal loss (RetinaNet, https://arxiv.org/abs/1708.02002); returns a scalar.

    ``alpha<0`` disables the positive/negative balance weighting.
    """
    prob = inputs.sigmoid()
    ce_loss = functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / len(inputs)


class MaskFormerLoss(nn.Module):
    """MaskFormer (DETR-style) loss terms plus the Hungarian matcher that aligns preds to truth.

    ``class_weights`` may be length ``num_classes`` (null weight appended) or
    ``num_classes + 1`` (used as-is).
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict,
        matcher_weights: dict,
        null_class_weight: float = 0.5,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.null_class_weight = null_class_weight
        # num_classes == 1 is a binary task and ignores class_weights.
        if self.num_classes == 1:
            empty_weight = torch.tensor([self.null_class_weight])
        elif class_weights is not None:
            if len(class_weights) == self.num_classes + 1:
                # class_weights already includes the null class weight
                empty_weight = torch.tensor(class_weights)
            elif len(class_weights) == self.num_classes:
                # append the null class weight at the end
                empty_weight = torch.tensor([*class_weights, self.null_class_weight])
            else:
                raise ValueError(
                    f"Invalid class_weights length: {len(class_weights)}. "
                    f"Expected {self.num_classes} or {self.num_classes + 1}."
                )
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.null_class_weight
        self.register_buffer("empty_weight", empty_weight)
        self.loss_weights = loss_weights

        self.matcher = HungarianMatcher(
            num_classes=num_classes,
            num_objects=num_objects,
            loss_weights=matcher_weights,
        )

    def loss_labels(
        self,
        preds: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Classification (NLL) loss on object classes -> ``{"object_class_ce": loss}``.

        Binary cross-entropy when there's a single class logit, else cross-entropy.
        """
        flav_pred_logits = preds["class_logits"].flatten(0, 1)
        flavour_labels = labels["object_class"].flatten(0, 1)
        if flav_pred_logits.shape[1] == 1:
            loss = functional.binary_cross_entropy_with_logits(
                flav_pred_logits.squeeze(), flavour_labels.float(), pos_weight=self.empty_weight
            )
        else:
            loss = functional.cross_entropy(flav_pred_logits, flavour_labels, self.empty_weight)
        return {"object_class_ce": loss}

    def loss_masks(
        self,
        preds: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Mask losses (``mask_dice``/``mask_focal``/``mask_ce``) over valid (non-null) objects.

        Only the components present with a truthy ``self.loss_weights`` entry are computed.
        """
        valid_idx = labels["object_class"] != self.num_classes
        target_masks = labels["masks"][valid_idx].float()
        pred_masks = preds["masks"][valid_idx]

        losses: dict[str, torch.Tensor] = {}
        if self.loss_weights.get("mask_dice"):
            losses["mask_dice"] = dice_loss(pred_masks, target_masks)
        if self.loss_weights.get("mask_focal"):
            losses["mask_focal"] = sigmoid_focal_loss(pred_masks, target_masks)
        if self.loss_weights.get("mask_ce"):
            losses["mask_ce"] = mask_ce_loss(pred_masks, target_masks)
        return losses
