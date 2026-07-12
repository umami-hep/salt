"""Hungarian-matched MaskFormer loss.

Given the decoder's object predictions and truth object labels, solves the
optimal 1-to-1 assignment of queries to truth objects (Hungarian matching)
and computes the matched classification/mask/regression loss components.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from py_lap_solver.solvers import Solvers
from torch import Tensor, nn
from torch.nn import functional

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema

# Maps solver name -> solver instance for the installed py_lap_solver build.
# Validate requested solver names against this registry (no silent fallback — a
# silent swap would mask a regressed container missing the expected solver).
SOLVER_REGISTRY = Solvers.get_available_solvers()


def fill_unmatched_assignments_vectorized(assignments: Tensor, num_predictions: int) -> Tensor:
    """Fill unmatched assignments (-1 values) with remaining prediction indices (vectorized).

    Parameters
    ----------
    assignments : Tensor
        Tensor of shape (batch_size, num_objects) containing assignment indices.
        Values of -1 indicate unmatched objects.
    num_predictions : int
        Total number of predictions available.

    Returns
    -------
    Tensor
        Assignments with -1 values replaced by unused prediction indices.
    """
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

__all__ = ["HungarianMatcher", "MaskFormerLoss", "MaskFormerMatchedLoss"]


# ---------------------------------------------------------------------------
# HungarianMatcher cost helpers
# ---------------------------------------------------------------------------


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

    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)
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
    solver_name : str, optional
        Name of the py_lap_solver LAP solver to use, by default upstream's
        ``"BatchedScipyOMP"`` (the OpenMP batched solver the container provides).

    Raises
    ------
    ValueError
        If ``solver_name`` is not available in ``SOLVER_REGISTRY``.

    Notes
    -----
    The sum of ``loss_weights`` must be positive.
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


# ---------------------------------------------------------------------------
# MaskFormerLoss loss helpers (M7 W2c-3 verbatim copy of v1
# maskformer_loss.py:17-97)
# ---------------------------------------------------------------------------


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
    loss = loss.mean(1)
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
    class_weights: list[float] | None, optional
        Optional per-class weights folded into the ``empty_weight`` CE balance buffer.
        May be of length ``num_classes`` (the null weight is appended) or
        ``num_classes + 1`` (used as-is), by default None.
    losses: list[str] | None, optional
        List of all the losses to be applied. See get_loss for list of available losses,
        by default None

    Raises
    ------
    ValueError
        If ``class_weights`` has an invalid length.
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict,
        matcher_weights: dict | None = None,
        null_class_weight: float = 0.5,
        class_weights: list[float] | None = None,
        losses: list[str] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.null_class_weight = null_class_weight
        assert self.num_classes > 0
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

        if "intermediate_outputs" in preds:
            for i, aux_pred in enumerate(preds["intermediate_outputs"]):
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

        for task in tasks:
            if task.input_name == "objects":
                # store the scaled targets in labels for the matcher to use
                task_targets = task.get_targets(labels)
                task_pred, _ = task(preds["objects"]["embed"], labels)
                preds["objects"].update({task.name: task_pred})
                labels["objects"][task.name] = task_targets

        idx = self.matcher(preds["objects"], labels["objects"])

        # warning: don't put this into a function or comprehension
        for k, v in preds["objects"].items():
            if k in {"x", "embed"}:
                continue

            if k != "intermediate_outputs":  # don't permute input reps
                preds["objects"][k] = v[idx]

        for loss in self.losses:
            losses.update(self.get_loss(loss, preds, labels))

        return preds, labels, losses


_UNNAMED = "unnamed"
"""Placeholder instance name, assigned before compile."""

# bundle stream name for the reconstructed objects (matches the MaskDecoder
# out_stream and the MaskFormerTargets object stream).
_OBJECT_STREAM = "objects"


class MaskFormerMatchedLoss(nn.Module):
    """Hungarian-matched MaskFormer loss over the decoder's object predictions.

    A FIT|VAL-only module. Runs the matcher on the scaled object
    predictions/targets, then publishes the matcher-permuted predictions plus the
    truth labels as new ``matched.objects.*`` keys (no in-place permute) and emits
    ``losses.{object_class_ce, mask_dice, mask_focal, regression}`` for whichever
    components have a positive ``loss_weights`` entry.

    The regression component is the matched L1 over valid (non-null) objects.

    Parameters
    ----------
    num_classes : int
        The number of non-null object classes. The null/no-object class index is
        ``num_classes``. MUST equal the decoder's ``class_net.output_size - 1``.
    num_objects : int
        The number of object queries ``M``. Must equal the decoder's
        ``num_objects`` and the truth-object slot count. MUST be >= 1.
    loss_weights : Mapping[str, float]
        Per-component loss weights; keys among ``object_class_ce``, ``mask_dice``,
        ``mask_focal``, ``mask_ce``, ``regression``. A component is produced only
        when its weight is present and truthy. ``object_class_ce`` is always applied.
    matcher_weights : Mapping[str, float] | None, optional
        Per-component matcher cost weights, defaulting to ``loss_weights``.
    null_class_weight : float, optional
        The class-balance weight on the null category in the CE, by default 0.5.
    class_weights : list[float] | None, optional
        Optional per-class CE balance weights forwarded to the composed
        ``MaskFormerLoss``, by default None.
    input_stream : str, optional
        The decoder's object stream name, by default ``objects`` — the
        ``<input_stream>.{class_logits,class_probs,masks}`` keys it reads.

    Raises
    ------
    ConfigError
        On a non-positive ``num_classes``, an unknown ``loss_weights`` key, an
        all-zero matcher-weight sum, or no requested loss component.
    """

    _KNOWN_COMPONENTS = ("object_class_ce", "mask_dice", "mask_focal", "mask_ce", "regression")

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: Mapping[str, float],
        matcher_weights: Mapping[str, float] | None = None,
        null_class_weight: float = 0.5,
        class_weights: list[float] | None = None,
        input_stream: str = "objects",
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        if num_classes < 1:
            raise ConfigError(
                f"MaskFormerMatchedLoss: num_classes (non-null classes) must be >= 1, got "
                f"{num_classes}"
            )
        if num_objects < 1:
            raise ConfigError(
                f"MaskFormerMatchedLoss: num_objects (query count M) must be >= 1, got "
                f"{num_objects}"
            )
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.input_stream = input_stream

        self.loss_weights = {k: float(v) for k, v in dict(loss_weights).items()}
        if unknown := sorted(set(self.loss_weights) - set(self._KNOWN_COMPONENTS)):
            raise ConfigError(
                f"MaskFormerMatchedLoss: unknown loss_weights keys {unknown} — known components "
                f"are {list(self._KNOWN_COMPONENTS)} (v1 maskformer_loss.py loss_map + regression)"
            )
        mw = self.loss_weights if matcher_weights is None else dict(matcher_weights)
        self.matcher_weights = {k: float(v) for k, v in mw.items()}
        if unknown := sorted(set(self.matcher_weights) - set(self._KNOWN_COMPONENTS)):
            raise ConfigError(
                f"MaskFormerMatchedLoss: unknown matcher_weights keys {unknown} — known components "
                f"are {list(self._KNOWN_COMPONENTS)}"
            )
        # get_batch_cost reads object_class_ce unconditionally — default it to the
        # loss weight if absent.
        self.matcher_weights.setdefault(
            "object_class_ce", self.loss_weights.get("object_class_ce", 1.0)
        )
        # the matcher asserts the cost-weight sum is positive; raise a clear
        # ConfigError here instead of a bare AssertionError deeper in construction.
        if sum(self.matcher_weights.values()) == 0:
            raise ConfigError(
                "MaskFormerMatchedLoss: the matcher cost weights sum to 0 — at least one "
                f"matcher_weights entry must be positive (got {self.matcher_weights}; v1 "
                "HungarianMatcher asserts this, matcher.py:167)"
            )

        # object_class_ce is always computed; the others only when their loss
        # weight is truthy. regression is the matched extension.
        self.components: tuple[str, ...] = tuple(
            c for c in self._KNOWN_COMPONENTS if c == "object_class_ce" or self.loss_weights.get(c)
        )
        if not self.components:
            raise ConfigError(
                "MaskFormerMatchedLoss: no loss component requested — set at least "
                "object_class_ce in loss_weights"
            )

        self.null_class_weight = float(null_class_weight)
        self.class_weights = list(class_weights) if class_weights is not None else None

        # composes the HungarianMatcher + empty_weight buffer + loss_labels/loss_masks
        self.v1_loss = MaskFormerLoss(
            num_classes=num_classes,
            num_objects=num_objects,
            loss_weights=self.loss_weights,
            matcher_weights=self.matcher_weights,
            null_class_weight=self.null_class_weight,
            class_weights=self.class_weights,
        )

    @property
    def matcher(self) -> HungarianMatcher:
        """The composed `HungarianMatcher` (owned by the loss)."""
        return self.v1_loss.matcher

    def _reg_pred_key(self) -> str:
        """The scaled object-regression prediction key (``preds.<stream>.regression``)."""
        return f"preds.{self.input_stream}.regression"

    def _reg_tgt_key(self) -> str:
        """The scaled object-regression target key (``targets.<stream>.regression``)."""
        return f"targets.{self.input_stream}.regression"

    def declare_io(self, mode: Mode) -> IO:
        """Declare the object preds + truth labels -> ``matched.objects.*`` + ``losses.*``.

        FIT|VAL only — empty IO in TEST/ONNX (the matched loss is mode-inactive
        there). Requires the decoder's ``<stream>.{class_logits,class_probs,masks}``
        and the truth ``labels.objects.{object_class,masks}``; additionally the
        scaled ``preds.<stream>.regression`` + ``targets.<stream>.regression`` when a
        regression component is requested.
        """
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})

        m = self.num_objects
        tok = sym_dim("T", self.name)
        emb = sym_dim("E", self.name)
        n_classes = self.num_classes + 1
        f = Mode.TRAINING

        requires: dict[str, TensorSpec] = {
            f"{self.input_stream}.class_logits": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"{self.input_stream}.class_probs": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"{self.input_stream}.masks": TensorSpec(shape=("B", m, tok), dtype="float32", modes=f),
            f"labels.{_OBJECT_STREAM}.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label", modes=f
            ),
            f"labels.{_OBJECT_STREAM}.masks": TensorSpec(
                shape=("B", m, tok), dtype="bool", kind="label", modes=f
            ),
        }
        if "regression" in self.components:
            r = sym_dim("R", self.name)
            requires[self._reg_pred_key()] = TensorSpec(shape=("B", m, r), dtype="float32", modes=f)
            requires[self._reg_tgt_key()] = TensorSpec(shape=("B", m, r), dtype="float32", modes=f)

        produces: dict[str, TensorSpec] = {
            f"matched.{_OBJECT_STREAM}.embed": TensorSpec(
                shape=("B", m, emb), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.class_logits": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.class_probs": TensorSpec(
                shape=("B", m, n_classes), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.masks": TensorSpec(
                shape=("B", m, tok), dtype="float32", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.object_class": TensorSpec(
                shape=("B", m), dtype="int64", kind="label", modes=f
            ),
            f"matched.{_OBJECT_STREAM}.target_masks": TensorSpec(
                shape=("B", m, tok), dtype="bool", kind="label", modes=f
            ),
        }
        requires[f"{self.input_stream}.embed"] = TensorSpec(
            shape=("B", m, emb), dtype="float32", modes=f
        )
        if "regression" in self.components:
            r = sym_dim("R", self.name)
            produces[f"matched.{_OBJECT_STREAM}.regression"] = TensorSpec(
                shape=("B", m, r), dtype="float32", modes=f
            )
            produces[f"matched.{_OBJECT_STREAM}.target_regression"] = TensorSpec(
                shape=("B", m, r), dtype="float32", modes=f
            )
        for component in self.components:
            produces[f"losses.{component}"] = TensorSpec(shape=(), kind="loss", modes=f)
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Validate the regression prediction/target widths agree (element-wise L1).

        ``bind_all`` runs ``bind`` on every configured module regardless of the
        compiled mode, but this loss is FIT|VAL-only, so in a TEST/ONNX-only bind
        schema the regression pred/target keys are absent and their widths never
        resolve statically — skip validation when either key is missing from
        ``schema.widths``. A genuinely missing regression producer in a training
        plan still fails earlier as a planner `ConnectivityError`, so this guard
        cannot mask a real training-mode wiring bug.

        Raises
        ------
        ConfigError
            If a requested regression component has mismatched pred/target widths.
        """
        if "regression" not in self.components:
            return
        reg_pred_key, reg_tgt_key = self._reg_pred_key(), self._reg_tgt_key()
        if reg_pred_key not in schema.widths or reg_tgt_key not in schema.widths:
            return
        wp = schema.width(reg_pred_key)
        wt = schema.width(reg_tgt_key)
        if wp != wt:
            raise ConfigError(
                f"MaskFormerMatchedLoss {self.name!r}: regression prediction width {wp} != "
                f"target width {wt} ({reg_pred_key!r} vs {reg_tgt_key!r}) — "
                "the matched L1 is element-wise (v1 maskformer_loss.py regression cost)"
            )

    def _matched_regression_loss(
        self, reg_pred: Tensor, reg_tgt: Tensor, object_class: Tensor
    ) -> Tensor:
        """The matched object-regression L1 over valid (non-null) objects.

        The matcher-permuted regression predictions are aligned to the truth-order
        targets, and the L1 is averaged over the valid objects only
        (``object_class != num_classes``).

        Returns
        -------
        Tensor
            A scalar matched-regression L1 (0.0 if no valid object in the batch).
        """
        valid = object_class != self.num_classes  # [B, M]
        if not valid.any():
            return reg_pred.new_zeros(())
        # element-wise L1 over the valid objects' targets; the matcher-permuted preds
        # are already truth-aligned. mean over (valid objects x R), v1-style reduction.
        return torch.nn.functional.l1_loss(reg_pred[valid], reg_tgt[valid])

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Match queries to truth, then emit the matched predictions + the loss components.

        Solves the assignment on the scaled preds/targets, gathers the permuted
        predictions into new ``matched.*`` keys (no in-place permute), and computes
        the classification/mask losses via the composed v1 methods plus the matched
        regression L1.

        Returns
        -------
        dict[str, Tensor]
            The new ``matched.objects.*`` + ``losses.*`` keys only.
        """
        del mode
        class_logits = b.get(f"{self.input_stream}.class_logits")
        class_probs = b.get(f"{self.input_stream}.class_probs")
        masks = b.get(f"{self.input_stream}.masks")
        embed = b.get(f"{self.input_stream}.embed")
        object_class = b.get(f"labels.{_OBJECT_STREAM}.object_class")
        target_masks = b.get(f"labels.{_OBJECT_STREAM}.masks")

        # matcher cost stays in scaled space
        pred_for_match: dict[str, Tensor] = {
            "class_logits": class_logits,
            "class_probs": class_probs,
            "masks": masks,
        }
        tgt_for_match: dict[str, Tensor] = {
            "object_class": object_class,
            "masks": target_masks.to(masks.dtype),
        }
        reg_pred = reg_tgt = None
        if "regression" in self.components:
            reg_pred = b.get(self._reg_pred_key())
            reg_tgt = b.get(self._reg_tgt_key())
            pred_for_match["regression"] = reg_pred
            tgt_for_match["regression"] = reg_tgt

        idx = self.matcher(pred_for_match, tgt_for_match)

        # advanced indexing returns a fresh tensor, so matched.* never aliases the
        # decoder output (no in-place permute).
        m_class_logits = class_logits[idx]
        m_class_probs = class_probs[idx]
        m_masks = masks[idx]
        m_embed = embed[idx]

        out: dict[str, Tensor] = {
            f"matched.{_OBJECT_STREAM}.embed": m_embed,
            f"matched.{_OBJECT_STREAM}.class_logits": m_class_logits,
            f"matched.{_OBJECT_STREAM}.class_probs": m_class_probs,
            f"matched.{_OBJECT_STREAM}.masks": m_masks,
            f"matched.{_OBJECT_STREAM}.object_class": object_class,
            f"matched.{_OBJECT_STREAM}.target_masks": target_masks,
        }

        permuted_preds = {"objects": {"class_logits": m_class_logits, "masks": m_masks}}
        truth_labels = {"objects": {"object_class": object_class, "masks": target_masks}}
        losses: dict[str, Tensor] = {}
        losses.update(self.v1_loss.get_loss("labels", permuted_preds, truth_labels))
        if any(self.loss_weights.get(c) for c in ("mask_dice", "mask_focal", "mask_ce")):
            losses.update(self.v1_loss.get_loss("masks", permuted_preds, truth_labels))

        for component in self.components:
            if component == "regression":
                assert reg_pred is not None
                assert reg_tgt is not None
                m_reg = reg_pred[idx]
                reg_loss = self._matched_regression_loss(m_reg, reg_tgt, object_class)
                out[f"matched.{_OBJECT_STREAM}.regression"] = m_reg
                out[f"matched.{_OBJECT_STREAM}.target_regression"] = reg_tgt
                out["losses.regression"] = self.loss_weights["regression"] * reg_loss
            else:
                out[f"losses.{component}"] = losses[component]
        return out
