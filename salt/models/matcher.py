import scipy
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.jit.script
def batch_dice_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the DICE loss for the entire batch, similar to generalized IOU for masks,
    over all permutations of the inputs and targets.
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
    """Compute the cross entropy loss over all permutations of the inputs and targets."""
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("bnc,bmc->bnm", pos, targets) + torch.einsum(
        "bnc,bmc->bnm", neg, (1 - targets)
    )
    return loss / inputs.shape[2]


@torch.jit.script
def batch_sigmoid_focal_cost(
    inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2
) -> Tensor:
    """Compute the focal loss over all permutations of the inputs and targets for batches."""
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
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
    """Compute the mean average error over all permutations of the inputs and targets
    for batches.
    """
    return (inputs[:, :, None] - targets[:, None, :]).abs().mean(-1)


class HungarianMatcher(nn.Module):
    """Module to compute the matching cost and solve the corresponding LSAP using the
    Hungarian algorithm.

    Based on
    - https://github.com/facebookresearch/MaskFormer
    - https://github.com/facebookresearch/Mask2Former
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict,
    ):
        """Compute the optimal assignment between the targets and the predictions of the network.

        For efficiency reasons, the targets don't include the no_object null class. Because of this,
        in general, there are more predictions than targets. In this case, we do a 1-to-1 matching
        of the best predictions, while the others are un-matched (and thus treated as
        non-objects).

        Parameters
        ----------
        num_classes: int
            Number of object classes, omitting the special no_object class
        num_objects: int
            Number of object classes + 1 (the no_object class)
        loss_weights: dict
            Dictionary containing the weights for the different losses
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.loss_weights = loss_weights
        assert sum(self.loss_weights.values()) != 0, "Sum of loss weights must be positive"

        self.global_step = 0

    def get_batch_cost(self, preds, targets):
        """Gets the cost matrix for the whole batch.

        Parameters
        ----------
        preds : dict of tensors
            The predictions from the model
        targets : dict of tensors
            The targets for the model

        Returns
        -------
        C : tensor
            The cost matrix for the whole batch, with a large value set in the input pad region
        batch_obj_lengths : tensor
            The number of inputs in each batch element
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
        C = self.loss_weights["object_class_ce"] * obj_class_cost

        # add mask costs
        if self.loss_weights.get("mask_dice"):
            cost_mask_dice = batch_dice_cost(mask_pred, mask_tgt)
            C += self.loss_weights["mask_dice"] * cost_mask_dice
        if self.loss_weights.get("mask_ce"):
            cost_mask_ce = batch_sigmoid_ce_cost(mask_pred, mask_tgt)
            C += self.loss_weights["mask_ce"] * cost_mask_ce
        if self.loss_weights.get("mask_focal"):
            cost_mask_focal = batch_sigmoid_focal_cost(mask_pred, mask_tgt)
            C += self.loss_weights["mask_focal"] * cost_mask_focal

        # add regression costs
        if "regression" in preds and self.loss_weights.get("regression"):
            reg_pred = preds["regression"]
            reg_tgt = targets["regression"] * valid_obj_idx.unsqueeze(-1)
            C += self.loss_weights["regression"] * batch_mae_loss(reg_pred, reg_tgt)

        # set entries corresponding to invalid objects to nan
        # (these are removed later when running LSAP)
        batch_obj_lengths = batch_obj_lengths.unsqueeze(-1)
        col_indices = torch.arange(obj_class_cost.size(-1), device=dev).unsqueeze(0)
        null_obj_cost_mask = (col_indices < batch_obj_lengths).unsqueeze(1).expand_as(C)
        C[~null_obj_cost_mask] = torch.nan

        return C, batch_obj_lengths

    @torch.no_grad()
    def forward(self, preds, targets):
        """Compute the optimal assignment between the targets and the predictions of the network."""
        batch_size = preds["class_logits"].shape[0]

        idxs = []
        self.default_idx = set(range(self.num_objects))

        # Get the full cost matrix, then run lsap on each batch element
        full_cost, batch_N = self.get_batch_cost(preds, targets)
        full_cost = full_cost.to(torch.float32).cpu().numpy()

        for batch_idx in range(batch_size):
            # get the cost matrix for this batch element
            C = full_cost[batch_idx][:, : batch_N[batch_idx]]

            # get the optimal assignment
            idx = self.lap(C)

            idxs.append(idx)

        # format indices to allow simple indexing
        idxs = torch.stack(idxs)
        idxs = (torch.arange(len(idxs)).unsqueeze(1), idxs)
        self.global_step += 1
        return idxs

    def lap(self, cost):
        src_idx, tgt_idx = scipy.optimize.linear_sum_assignment(cost)
        idx = src_idx[tgt_idx]
        idx = list(idx) + sorted(self.default_idx - set(idx))
        return torch.as_tensor(idx)
