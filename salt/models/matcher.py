from typing import Any

import scipy
import torch
from torch import Tensor, nn
from torch.nn import functional


@torch.jit.script
def batch_dice_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute batched DICE loss for all input-target permutations.

    The loss is computed for every pair of prediction and target within each
    batch element, analogous to a generalized IoU for masks.

    Parameters
    ----------
    inputs : Tensor
        Predicted mask logits of shape ``[B, N, C]``.
    targets : Tensor
        Target masks (0/1) of shape ``[B, M, C]``.

    Returns
    -------
    Tensor
        Pairwise DICE loss matrix of shape ``[B, N, M]`` where entry ``(b, n, m)``
        is the DICE loss between prediction ``n`` and target ``m`` for batch ``b``.
    """
    inputs = inputs.sigmoid()

    # inputs has shape (B, N, C), targets has shape (B, M, C)
    # We want to compute the DICE loss for each combination of N and M for each batch
    # Using torch.einsum to handle the batched matrix multiplication
    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)

    # Compute the denominator using sum over the last dimension (C) and broadcasting
    denominator = inputs.sum(-1).unsqueeze(2) + targets.sum(-1).unsqueeze(1)

    return 1 - (numerator + 1) / (denominator + 1)


@torch.jit.script
def batch_sigmoid_ce_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute batched sigmoid cross-entropy cost for all permutations.

    Parameters
    ----------
    inputs : Tensor
        Predicted mask logits of shape ``[B, N, C]``.
    targets : Tensor
        Target masks (0/1) of shape ``[B, M, C]``.

    Returns
    -------
    Tensor
        Pairwise cross-entropy cost matrix of shape ``[B, N, M]``.
    """
    pos = functional.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = functional.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    loss = torch.einsum("bnc,bmc->bnm", pos, targets) + torch.einsum(
        "bnc,bmc->bnm", neg, (1 - targets)
    )
    return loss / inputs.shape[2]


@torch.jit.script
def batch_sigmoid_focal_cost(
    inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2
) -> Tensor:
    """Compute batched focal loss for all input-target permutations.

    Parameters
    ----------
    inputs : Tensor
        Predicted mask logits of shape ``[B, N, C]``.
    targets : Tensor
        Target masks (0/1) of shape ``[B, M, C]``.
    alpha : float, optional
        Class balancing factor. If negative, no reweighting is applied.
        The default is ``-1``.
    gamma : float, optional
        Focusing parameter controlling down-weighting of easy examples.
        The default is ``2``.

    Returns
    -------
    Tensor
        Pairwise focal loss matrix of shape ``[B, N, M]``.
    """
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * functional.binary_cross_entropy_with_logits(
        inputs,
        torch.ones_like(inputs),
        reduction="none",
    )
    focal_neg = (prob**gamma) * functional.binary_cross_entropy_with_logits(
        inputs,
        torch.zeros_like(inputs),
        reduction="none",
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)
    loss = torch.einsum("bnc,bmc->bnm", focal_pos, targets) + torch.einsum(
        "bnc,bmc->bnm", focal_neg, (1 - targets)
    )
    return loss / inputs.shape[2]


@torch.jit.script
def batch_mae_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute batched mean absolute error for all permutations.

    Parameters
    ----------
    inputs : Tensor
        Predicted values of shape ``[B, N, C]``.
    targets : Tensor
        Target values of shape ``[B, M, C]``.

    Returns
    -------
    Tensor
        Pairwise MAE matrix of shape ``[B, N, M]`` computed by averaging over
        the last dimension ``C``.
    """
    return (inputs[:, :, None] - targets[:, None, :]).abs().mean(-1)


class HungarianMatcher(nn.Module):
    """Solve LSAP matching between predictions and targets via Hungarian algorithm.

    The module aggregates multiple cost terms (classification, mask losses, optional
    regression) into a single cost matrix per batch element and solves the linear
    sum assignment problem to obtain a 1-to-1 matching.

    Parameters
    ----------
    num_classes : int
        Number of object classes, excluding the special ``no_object`` class.
    num_objects : int
        Number of object slots (typically ``num_classes + 1`` including ``no_object``).
    loss_weights : dict[str, float]
        Weights for individual loss components, e.g.
        ``{"object_class_ce": 1.0, "mask_dice": 1.0, "mask_ce": 0.0, "mask_focal": 0.0,
        "regression": 0.0}``.

    Notes
    -----
    The sum of ``loss_weights`` must be positive.
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict[str, float],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.loss_weights = loss_weights
        assert sum(self.loss_weights.values()) != 0, "Sum of loss weights must be positive"

        self.global_step = 0

    def get_batch_cost(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Build the pairwise cost matrix for the whole batch.

        Parameters
        ----------
        preds : dict[str, Tensor]
            Model predictions with keys:
            - ``"class_probs"``: class probabilities of shape ``[B, N, C]``.
            - ``"masks"``: mask logits of shape ``[B, N, L]``.
            - ``"regression"`` (optional): regression predictions of shape ``[B, N, R]``.
        targets : dict[str, Tensor]
            Ground-truth targets with keys:
            - ``"object_class"``: class indices of shape ``[B, M]`` where ``num_classes``
              denotes ``no_object``.
            - ``"masks"``: target masks of shape ``[B, M, L]``.
            - ``"regression"`` (optional): regression targets of shape ``[B, M, R]``.

        Returns
        -------
        Tensor
            Cost tensor ``C`` of shape ``[B, N, M]``. Entries corresponding to invalid
            target objects are set to ``NaN`` and ignored later in LSAP.
        Tensor
            Tensor of shape ``[B, 1]`` with the valid number of target objects per batch element.
        """
        # get some useful things
        bs = len(targets["object_class"])
        dev = preds["class_probs"].device

        obj_class_tgt = targets["object_class"].detach()
        obj_class_pred = preds["class_probs"].detach()
        mask_pred = preds["masks"].detach()
        mask_tgt = targets["masks"].detach().to(mask_pred.dtype)

        valid_obj_idx = obj_class_tgt != self.num_classes
        batch_obj_lengths = torch.sum(valid_obj_idx, dim=1)

        # compute the object class loss
        obj_class_tgt = (
            obj_class_tgt[:, : self.num_classes].unsqueeze(1).expand(-1, obj_class_pred.size(1), -1)
        )
        valid_obj_mask = obj_class_tgt != self.num_classes
        output = torch.gather(obj_class_pred, 2, obj_class_tgt * valid_obj_mask) * valid_obj_mask
        obj_class_cost = torch.zeros((bs, self.num_objects, self.num_objects), device=dev)
        obj_class_cost[:, :, : self.num_classes] = -output

        # initialize the cost matrix with the object class loss
        cost_matrix = self.loss_weights["object_class_ce"] * obj_class_cost

        # add mask costs
        if self.loss_weights.get("mask_dice"):
            cost_mask_dice = batch_dice_cost(mask_pred, mask_tgt)
            cost_matrix += self.loss_weights["mask_dice"] * cost_mask_dice
        if self.loss_weights.get("mask_ce"):
            cost_mask_ce = batch_sigmoid_ce_cost(mask_pred, mask_tgt)
            cost_matrix += self.loss_weights["mask_ce"] * cost_mask_ce
        if self.loss_weights.get("mask_focal"):
            cost_mask_focal = batch_sigmoid_focal_cost(mask_pred, mask_tgt)
            cost_matrix += self.loss_weights["mask_focal"] * cost_mask_focal

        # add regression costs
        if "regression" in preds and self.loss_weights.get("regression"):
            reg_pred = preds["regression"]
            reg_tgt = targets["regression"] * valid_obj_idx.unsqueeze(-1)
            cost_matrix += self.loss_weights["regression"] * batch_mae_loss(reg_pred, reg_tgt)

        # set entries corresponding to invalid objects to nan
        # (these are removed later when running LSAP)
        batch_obj_lengths = batch_obj_lengths.unsqueeze(-1)
        col_indices = torch.arange(obj_class_cost.size(-1), device=dev).unsqueeze(0)
        null_obj_cost_mask = (col_indices < batch_obj_lengths).unsqueeze(1).expand_as(cost_matrix)
        cost_matrix[~null_obj_cost_mask] = torch.nan

        return cost_matrix, batch_obj_lengths

    @torch.no_grad()
    def forward(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute optimal assignments for each batch element.

        Parameters
        ----------
        preds : dict[str, Tensor]
            Model predictions; see :meth:`get_batch_cost` for required keys/shapes.
        targets : dict[str, Tensor]
            Ground-truth targets; see :meth:`get_batch_cost` for required keys/shapes.

        Returns
        -------
        Tensor
            Batch indices of shape ``[B, M]`` suitable for advanced indexing.
        Tensor
            Assigned target indices per batch of shape ``[B, M]``; unassigned
            slots are filled to cover all ``num_objects`` by appending remaining indices.
        """
        batch_size = preds["class_logits"].shape[0]

        idxs: list[list[int]] = []
        self.default_idx = set(range(self.num_objects))

        # Get the full cost matrix, then run lsap on each batch element
        full_cost, n_batch = self.get_batch_cost(preds, targets)
        full_cost = full_cost.to(torch.float32).cpu().numpy()

        for batch_idx in range(batch_size):
            # get the cost matrix for this batch element
            cost_matrix = full_cost[batch_idx][:, : n_batch[batch_idx]]

            # get the optimal assignment
            idx = self.lap(cost_matrix)

            idxs.append(idx)

        # get the device so we can put the indices on the same device as the predictions
        d = preds["class_logits"].device
        # format indices to allow simple indexing
        idxs_tensor = torch.tensor(idxs).to(d)
        batch_arange = torch.arange(len(idxs)).unsqueeze(1).to(d)
        idxs_tuple = (batch_arange, idxs_tensor)  # shape-compatible indexing tuple

        self.global_step += 1
        return idxs_tuple

    def lap(self, cost: Any) -> list[int]:
        """Solve the linear sum assignment problem for a single cost matrix.

        Parameters
        ----------
        cost : Any
            Cost matrix of shape ``[N, M]``.

        Returns
        -------
        list[int]
            Ordered list of selected target indices ``idx`` aligned with sources,
            extended by appending any remaining indices to cover all ``num_objects``.
        """
        src_idx, tgt_idx = scipy.optimize.linear_sum_assignment(cost)
        idx = src_idx[tgt_idx]
        return list(idx) + sorted(self.default_idx - set(idx))
