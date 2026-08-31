"""Hungarian matcher: LSAP query-to-truth assignment for the MaskFormer losses."""

from __future__ import annotations

import torch
from py_lap_solver.solvers import Solvers
from torch import Tensor, nn
from torch.nn import functional

__all__ = ["HungarianMatcher"]


# Maps solver name -> solver instance for the installed py_lap_solver build.
# Validate requested solver names against this registry (no silent fallback — a
# silent swap would mask a regressed container missing the expected solver).
SOLVER_REGISTRY = Solvers.get_available_solvers()


def fill_unmatched_assignments_vectorized(assignments: Tensor, num_predictions: int) -> Tensor:
    """Fill unmatched assignments (-1 values) with the remaining unused prediction indices."""
    batch_size, _num_objects = assignments.shape
    device = assignments.device

    # Create mask for unmatched assignments
    unmatched_mask = assignments < 0

    # Early exit if no unmatched assignments
    if not unmatched_mask.any():
        return assignments

    all_indices = torch.arange(num_predictions, device=device).unsqueeze(0).expand(batch_size, -1)

    # Mark which indices are already used (True == used)
    used_mask = torch.zeros(batch_size, num_predictions, dtype=torch.bool, device=device)
    valid_mask = assignments >= 0
    if valid_mask.any():
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(assignments)
        used_mask[batch_indices[valid_mask], assignments[valid_mask]] = True

    # Get unused indices (available for assignment), sorted per batch element
    available_mask = ~used_mask
    sort_keys = all_indices.float() + (~available_mask).float() * 1e10
    sorted_indices = torch.argsort(sort_keys, dim=1)
    sorted_available = torch.gather(all_indices, 1, sorted_indices)

    # Rank each unmatched slot within its batch element, then gather its filler index
    unmatched_ranks = torch.cumsum(unmatched_mask.long(), dim=1) - 1
    filled_indices = torch.gather(sorted_available, 1, unmatched_ranks.clamp(min=0))

    # Replace -1 values with the filled indices (only where unmatched_mask is True)
    return torch.where(unmatched_mask, filled_indices, assignments)


# ---------------------------------------------------------------------------
# HungarianMatcher cost helpers
# ---------------------------------------------------------------------------


def batch_dice_cost_eager(inputs: Tensor, targets: Tensor) -> Tensor:
    """Pairwise DICE cost ``[B, N, M]`` for every prediction/target permutation."""
    inputs = inputs.sigmoid()

    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)
    denominator = inputs.sum(-1).unsqueeze(2) + targets.sum(-1).unsqueeze(1)

    return 1 - (numerator + 1) / (denominator + 1)


@torch.jit.script
def batch_dice_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """TorchScript wrapper for :func:`batch_dice_cost_eager`."""
    return batch_dice_cost_eager(inputs, targets)


def batch_sigmoid_ce_cost_eager(inputs: Tensor, targets: Tensor) -> Tensor:
    """Pairwise sigmoid cross-entropy cost ``[B, N, M]`` for every permutation."""
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
def batch_sigmoid_ce_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """TorchScript wrapper for :func:`batch_sigmoid_ce_cost_eager`."""
    return batch_sigmoid_ce_cost_eager(inputs, targets)


def batch_sigmoid_focal_cost_eager(
    inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2
) -> Tensor:
    """Pairwise sigmoid focal cost ``[B, N, M]``; ``alpha<0`` disables class balancing."""
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
def batch_sigmoid_focal_cost(
    inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2
) -> Tensor:
    """TorchScript wrapper for :func:`batch_sigmoid_focal_cost_eager`."""
    return batch_sigmoid_focal_cost_eager(inputs, targets, alpha, gamma)


def batch_mae_loss_eager(inputs: Tensor, targets: Tensor) -> Tensor:
    """Pairwise MAE cost ``[B, N, M]``, averaged over the last dimension."""
    return (inputs[:, :, None] - targets[:, None, :]).abs().mean(-1)


@torch.jit.script
def batch_mae_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """TorchScript wrapper for :func:`batch_mae_loss_eager`."""
    return batch_mae_loss_eager(inputs, targets)


class HungarianMatcher(nn.Module):
    """Solve LSAP matching between predictions and targets via the Hungarian algorithm.

    Aggregates the classification/mask/regression cost terms (per
    ``loss_weights``, sum must be positive) into one cost matrix per batch
    element. ``solver_name`` must be a registered `py_lap_solver` solver.
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict[str, float],
        solver_name: str = "BatchedScipyOMP",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.loss_weights = loss_weights
        assert sum(self.loss_weights.values()) != 0, "Sum of loss weights must be positive"

        # No silent fallback: a silent solver swap would mask a regressed container
        # that dropped the requested solver.
        if solver_name not in SOLVER_REGISTRY:
            available_solvers = ", ".join(sorted(SOLVER_REGISTRY))
            msg = f"Unknown LAP solver '{solver_name}'. Available solvers: {available_solvers}"
            raise ValueError(msg)
        self.solver_name = solver_name

        self.global_step = 0

    def get_batch_cost(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Build the pairwise cost matrix ``[B, N, M]`` for the whole batch.

        Invalid target-object entries are set to NaN (excluded later in LSAP).
        Returns ``(cost, valid_target_counts [B, 1])``.
        """
        bs = len(targets["object_class"])
        dev = preds["class_probs"].device

        obj_class_tgt = targets["object_class"].detach()
        obj_class_pred = preds["class_probs"].detach()
        mask_pred = preds["masks"].detach()
        # clamp mask logits to a finite range: guards against inf/NaN cost -> garbage
        # LSAP assignment.
        mask_pred = mask_pred.clamp(min=-1e4, max=1e4)
        mask_tgt = targets["masks"].detach().to(mask_pred.dtype)

        valid_obj_idx = obj_class_tgt != self.num_classes
        batch_obj_lengths = torch.sum(valid_obj_idx, dim=1)

        obj_class_tgt = (
            obj_class_tgt[:, : self.num_classes].unsqueeze(1).expand(-1, obj_class_pred.size(1), -1)
        )
        valid_obj_mask = obj_class_tgt != self.num_classes
        output = torch.gather(obj_class_pred, 2, obj_class_tgt * valid_obj_mask) * valid_obj_mask
        obj_class_cost = torch.zeros((bs, self.num_objects, self.num_objects), device=dev)
        obj_class_cost[:, :, : self.num_classes] = -output

        cost_matrix = self.loss_weights["object_class_ce"] * obj_class_cost

        if self.loss_weights.get("mask_dice"):
            cost_mask_dice = batch_dice_cost(mask_pred, mask_tgt)
            cost_matrix += self.loss_weights["mask_dice"] * cost_mask_dice
        if self.loss_weights.get("mask_ce"):
            cost_mask_ce = batch_sigmoid_ce_cost(mask_pred, mask_tgt)
            cost_matrix += self.loss_weights["mask_ce"] * cost_mask_ce
        if self.loss_weights.get("mask_focal"):
            cost_mask_focal = batch_sigmoid_focal_cost(mask_pred, mask_tgt)
            cost_matrix += self.loss_weights["mask_focal"] * cost_mask_focal

        if "regression" in preds and self.loss_weights.get("regression"):
            reg_pred = preds["regression"]
            reg_tgt = targets["regression"] * valid_obj_idx.unsqueeze(-1)
            # sanitise regression targets: guards against nan/inf targets producing
            # nan MAE cost -> garbage LSAP assignment.
            reg_tgt = torch.nan_to_num(reg_tgt, nan=0.0, posinf=0.0, neginf=0.0)
            cost_matrix += self.loss_weights["regression"] * batch_mae_loss(reg_pred, reg_tgt)

        # invalid target objects get nan cost; removed later when running LSAP.
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
        """Solve the optimal assignment per batch element.

        Returns ``(batch_arange, assignments)`` ``[B, M]`` each, suitable for
        advanced indexing; unassigned slots are filled with remaining indices.
        """
        device = preds["class_logits"].device

        full_cost, batch_n = self.get_batch_cost(preds, targets)
        batch_n = batch_n.squeeze(-1).cpu().numpy()

        # Transpose [B, N, M] -> [B, M, N] so targets are the solver rows and
        # predictions the columns; num_valid restricts each problem to its valid
        # target rows (padded/NaN rows M >= batch_n are excluded). batch_solve
        # returns [B, M] where entry [b, i] is the prediction column assigned to
        # target row i (-1 for padded rows).
        solver = SOLVER_REGISTRY[self.solver_name]
        full_cost = full_cost.transpose(1, 2).to(torch.float32).cpu().numpy()
        assignments = solver.batch_solve(full_cost, num_valid=batch_n)
        assignments = torch.from_numpy(assignments).to(torch.int64).to(device)

        # full_cost is [B, M, N]; shape[1] is the prediction count (square cost).
        assignments = fill_unmatched_assignments_vectorized(assignments, full_cost.shape[1])

        self.global_step += 1
        # (batch_arange [B, 1], assignments [B, M]) consumed downstream as
        # preds["objects"][k][idx] to permute predictions.
        batch_arange = torch.arange(len(assignments)).unsqueeze(1).to(device)
        return (batch_arange, assignments)
