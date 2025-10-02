"""Implements the loss of MaskFormer, utilising a Hungarian matcher.

Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former
"""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from salt.models.matcher import HungarianMatcher


@torch.jit.script
def dice_loss(inputs: Tensor, labels: Tensor):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Parameters
    ----------
    inputs : Tensor
        The predictions for each example.
    labels : Tensor
        A float tensor with the same shape as inputs. Stores the binary classification label
        for each element in inputs (0 for the negative class and 1 for the positive class).

    Returns
    -------
    Tensor
        Single-element loss tensor
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * labels).sum(-1)
    denominator = inputs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / len(inputs)


@torch.jit.script
def mask_ce_loss(inputs: Tensor, labels: Tensor):
    """Computes cross entropy loss for masks.

    Parameters
    ----------
    inputs: Tensor
            A float tensor of arbitrary shape representing the predictions for each example.
    labels: Tensor
        A float tensor with the same shape as inputs. Stores the binary classification label
        for each element in inputs (0 for the negative class and 1 for the positive class).

    Returns
    -------
    Tensor
        Single-element loss tensor
    """
    loss = functional.binary_cross_entropy_with_logits(inputs, labels, reduction="none")
    # find the mean loss for each mask
    loss = loss.mean(1)

    # take the average over all masks
    return loss.sum() / len(inputs)


@torch.jit.script
def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2):
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Parameters
    ----------
    inputs: Tensor
        A float tensor of arbitrary shape representing the predictions for each example.
    targets: Tensor
        A float tensor with the same shape as inputs. Stores the binary classification label for
        each element in inputs (0 for the negative class and 1 for the positive class).
    alpha: float, optional
        Weighting factor in range (0,1) to balance positive vs negative examples.
        Default = -1 (no weighting).
    gamma: float, optional
        Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Default is 2

    Returns
    -------
    Tensor
        Single-element loss tensor
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
    """Compute the loss of MaskFormer, based on DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the preds of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box).

    Parameters
    ----------
    num_classes: int
        Number of object categories, omitting the special no-object category
    num_objects: int
        Number of objects to detect
    loss_weights: dict
        Dict containing as key the names of the losses and as values their relative weight
    matcher_weights: dict | None, optional
        Same as loss_weights but for the matching cost, by default None
    null_class_weight: float, optional
        Relative classification weight applied to the no-object category, by default 0.5
    losses: list[str] | None, optional
        List of all the losses to be applied. See get_loss for list of available losses,
        by default None
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict,
        matcher_weights: dict | None = None,
        null_class_weight: float = 0.5,
        losses: list[str] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.null_class_weight = null_class_weight
        assert self.num_classes > 0
        if self.num_classes == 1:
            empty_weight = torch.tensor([self.null_class_weight])
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.null_class_weight
        self.register_buffer("empty_weight", empty_weight)
        self.loss_weights = loss_weights
        if matcher_weights is None:
            matcher_weights = loss_weights
        self.losses = losses if losses is not None else ["labels", "masks"]

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
        """Compute the classification (NLL) loss on object classes.

        Parameters
        ----------
        preds : dict[str, torch.Tensor]
            Dictionary of prediction tensors. Must contain a key
            ``"class_logits"`` of shape ``(batch, n_queries, n_classes)``.
        labels : dict[str, torch.Tensor]
            Dictionary of label tensors. Must contain a key
            ``"object_class"`` of shape ``(batch, n_queries)``.

        Returns
        -------
        dict[str, torch.Tensor]
            A single-key dictionary ``{"object_class_ce": loss}`` containing
            the cross-entropy or binary cross-entropy loss.
        """
        # use the new indices to calculate the loss
        # process full inidices
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
        """Compute the mask-related losses: dice, focal and cross-entropy.

        Parameters
        ----------
        preds : dict[str, torch.Tensor]
            Dictionary of prediction tensors. Must contain a key
            ``"masks"`` of shape ``(batch, n_queries, h, w)``.
        labels : dict[str, torch.Tensor]
            Dictionary of label tensors. Must contain keys:

            * ``"object_class"``: class indices of shape ``(batch, n_queries)``.
            * ``"masks"``: ground-truth masks of shape
            ``(batch, n_queries, h, w)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of the requested mask losses. Keys may include
            ``"mask_dice"``, ``"mask_focal"``, ``"mask_ce"`` depending on
            ``self.loss_weights``.
        """
        # select valid masks via flavour label
        valid_idx = labels["object_class"] != self.num_classes
        target_masks = labels["masks"][valid_idx].float()
        pred_masks = preds["masks"][valid_idx]

        # compute losses on valid masks
        losses: dict[str, torch.Tensor] = {}
        if self.loss_weights.get("mask_dice"):
            losses["mask_dice"] = dice_loss(pred_masks, target_masks)
        if self.loss_weights.get("mask_focal"):
            losses["mask_focal"] = sigmoid_focal_loss(pred_masks, target_masks)
        if self.loss_weights.get("mask_ce"):
            losses["mask_ce"] = mask_ce_loss(pred_masks, target_masks)
        return losses

    def get_loss(
        self,
        loss: str,
        preds: dict[str, Any],
        labels: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Select and compute one type of loss on the given predictions.

        Parameters
        ----------
        loss : str
            Name of the loss to compute (``"labels"`` or ``"masks"``).
        preds : dict[str, Any]
            Predictions, typically ``preds["objects"]`` from the model.
        labels : dict[str, Any]
            Labels, typically ``labels["objects"]`` corresponding to ``preds``.

        Returns
        -------
        dict[str, torch.Tensor]
            Loss dictionary returned by the underlying loss function,
            with weights applied via :meth:`weight_loss`.
        """
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return self.weight_loss(loss_map[loss](preds["objects"], labels["objects"]))

    def weight_loss(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the configured loss weights to a loss dictionary.

        Parameters
        ----------
        losses : dict[str, torch.Tensor]
            Dictionary mapping loss names to loss tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Same dictionary with each loss scaled by ``self.loss_weights``.
        """
        for k in list(losses.keys()):
            losses[k] *= self.loss_weights[k]
        return losses

    def forward(
        self,
        preds: dict[str, Any],
        tasks: list[Any],
        labels: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
        """Calculate the full MaskFormer loss via optimal assignment.

        Parameters
        ----------
        preds : dict[str, Any]
            Model predictions. May contain key ``"intermediate_outputs"`` for
            auxiliary layers and key ``"objects"`` for the final layer.
        tasks : list[Any]
            A list of task objects that can be applied to ``preds["objects"]``
            and ``labels["objects"]`` to generate additional predictions and
            targets (e.g. regression tasks).
        labels : dict[str, Any]
            Ground truth labels corresponding to the predictions.

        Returns
        -------
        tuple
            ``(preds, labels, losses)`` where:

            * ``preds`` : dict — predictions with any updated tasks included.
            * ``labels`` : dict — labels with any task targets added.
            * ``losses`` : dict[str, torch.Tensor] — combined losses from
            all requested loss functions.
        """
        losses: dict[str, torch.Tensor] = {}

        # loop over intermediate outputs and compute losses
        if "intermediate_outputs" in preds:
            for i, aux_pred in enumerate(preds["intermediate_outputs"]):
                # add regression prediction for cost
                for task in tasks:
                    if task.input_name == "objects":
                        aux_pred.update(task(aux_pred, labels))

                aux_idx = self.matcher(aux_pred, labels)
                for k, v in aux_pred.items():
                    if k in {"x", "embed_xs", "global_rep"}:
                        continue
                    aux_pred[k] = v[aux_idx]

                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_pred, labels)
                    l_dict = {k + f"_layer{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # run tasks on the objects (e.g. regression) for the main predictions
        for task in tasks:
            if task.input_name == "objects":
                # Get the scaled targets for this task and store them in the labels dict
                # for the matcher to use
                task_targets = task.get_targets(labels)
                task_pred, _ = task(preds["objects"]["embed"], labels)
                preds["objects"].update({task.name: task_pred})
                labels["objects"][task.name] = task_targets

        # get the optimal assignment of the predictions to the labels
        idx = self.matcher(preds["objects"], labels["objects"])

        # warning: don't put this into a function or comprehension
        for k, v in preds["objects"].items():
            if k in {"x", "embed"}:
                continue

            if k != "intermediate_outputs":  # don't permute input reps
                preds["objects"][k] = v[idx]

        # compute the requested losses
        for loss in self.losses:
            losses.update(self.get_loss(loss, preds, labels))

        return preds, labels, losses
