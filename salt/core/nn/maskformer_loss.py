"""Config-constructed MaskFormer matched loss + Hungarian matcher (FD 1158-1170).

M5 sub-wave C (plan 10). The v2 spelling of v1's matcher-driven MaskFormer loss
(``salt.models.maskformer_loss.MaskFormerLoss.forward`` + ``salt.models.matcher.
HungarianMatcher``). A FIT|VAL-only `GraphModule` that, given the decoder's object
predictions and the truth object labels, solves the optimal 1-to-1 assignment of
queries to truth objects and emits the four MaskFormer loss components.

M2/M5 porting policy (plan 05): this COMPOSES the verbatim v1 building blocks — a
v1 ``MaskFormerLoss`` instance OWNS the ``HungarianMatcher`` (matcher.py), the
``empty_weight`` class-balance buffer, and the three loss methods
(``loss_labels``/``loss_masks``, maskformer_loss.py:154-227) — and reproduces v1's
matching + the three v1-decidable loss components (``object_class_ce``,
``mask_dice``, ``mask_focal``) BYTE-faithfully. Full code absorption is M7.

What is DELIBERATELY DIFFERENT from v1 (FD 1158-1170, the design's single-ownership
+ no-in-place-permute rules; the alignment change is the MF1b sign-off item):

- **NO in-place permute.** v1 mutates ``preds["objects"][k] = v[idx]`` in place
  (maskformer_loss.py:338-343), corrupting the prediction dict every other module
  shares. v2 gathers the matcher-permuted predictions into NEW ``matched.objects.*``
  bundle keys (write-once, design §2.1); the decoder's ``objects.*`` are never
  touched. `MaskformerMetrics` consumes ``matched.objects.*`` (its sub-wave-C wiring).
- **The object regression loss is MATCHED, not query-order.** v1's *effective*
  regression loss is QUERY-ORDER: the object-regression task runs in
  ``SaltModel.run_tasks`` with ``labels`` (saltmodel.py:218-219) and computes its L1
  there, aligning query-i to truth-object-i WITHOUT the matcher; the matcher only
  uses ``regression`` as a *cost* term, and ``"regression"`` is NOT in
  ``MaskFormerLoss.losses`` (= ``["labels", "masks"]``), so the matched loop never
  re-computes it. v2 makes the regression loss MATCHED (the matcher-permuted
  predictions vs the truth-order targets) and OWNED here — the single-ownership
  rule. This is a USER-VISIBLE alignment change (FD 1168-1170, §12): the MF1b human
  sign-off characterises it; the code gate (MF1a/this module) asserts the three
  mask/class components BYTE vs v1 and the matched-regression as a DESIGN-conformance
  property (no v1 byte reference exists for a matched object-regression loss).
- **NO double task execution.** v1 runs the object task twice (saltmodel.py:218-219
  for the loss + maskformer_loss.py:330 for the matcher cost). v2 runs the
  regression task ONCE; its scaled predictions/targets arrive here as the declared
  ``preds.objects.regression`` / ``targets.objects.regression`` keys.
- **NO ``aux_loss`` deep supervision** (maskformer_loss.py:306-322). PARKED (FD §10;
  shipped MaskFormer.yaml:39 sets ``aux_loss: false``) — re-deferred, not implemented.

Matching costs stay in SCALED space (v1 behaviour, FD 1166): the matcher reads the
scaled ``regression`` predictions/targets exactly as v1's
``get_batch_cost`` does (matcher.py:236-239).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import scipy
import torch
from torch import Tensor, nn
from torch.nn import functional

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema

# M7 W2c-3 (the FINAL absorption): the MaskFormer matched loss + the Hungarian
# matcher are now v2-NATIVE, copied VERBATIM into this module from v1
# ``salt.models.maskformer_loss.MaskFormerLoss`` (maskformer_loss.py:100-349) and
# ``salt.models.matcher.HungarianMatcher`` (matcher.py:134-317), together with
# their ``@torch.jit.script`` cost/loss helpers. The matcher cost + the scipy
# ``linear_sum_assignment`` LAP are byte-for-byte v1's, so the matched assignment
# (matched indices) and the matched loss stay BITWISE identical vs v1 (the MF1c
# gate oracle builds a fresh v1 ``MaskFormerLoss`` and compares ``torch.equal``).
# The composed ``MaskFormerLoss`` below carries ONLY the config-deterministic
# ``empty_weight`` buffer (matcher has no parameters), so it remains
# state_dict-compatible with a fresh v1 ``MaskFormerLoss``.

__all__ = ["HungarianMatcher", "MaskFormerLoss", "MaskFormerMatchedLoss"]


# ---------------------------------------------------------------------------
# HungarianMatcher cost helpers (M7 W2c-3 verbatim copy of v1 matcher.py:9-131)
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

    M7 W2c-3 v2-native absorption of v1 ``salt.models.matcher.HungarianMatcher``
    (matcher.py:134-317), COPIED VERBATIM — the same cost matrix assembly
    (``get_batch_cost``) and the same per-batch scipy ``linear_sum_assignment``
    LAP (``lap``), so the matched assignment is byte-for-byte v1's (the MF1c gate
    builds a fresh v1 ``HungarianMatcher`` and asserts ``torch.equal`` on the
    permuted predictions). The v1 original is UNTOUCHED as the gate oracle.

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

    M7 W2c-3 v2-native absorption of v1 ``salt.models.maskformer_loss.MaskFormerLoss``
    (maskformer_loss.py:100-349), COPIED VERBATIM — the same ``empty_weight`` buffer,
    ``HungarianMatcher`` (now the absorbed one above), ``loss_labels`` / ``loss_masks``
    / ``get_loss`` / ``weight_loss`` methods, so the three v1-decidable loss components
    are byte-for-byte v1's (the MF1c gate builds a fresh v1 ``MaskFormerLoss`` and
    asserts ``torch.equal``). The v1 original is UNTOUCHED as the gate oracle; this
    class carries ONLY the config-deterministic ``empty_weight`` buffer, so its
    ``state_dict()`` stays loadable into a fresh v1 ``MaskFormerLoss``.

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


_UNNAMED = "unnamed"
"""Placeholder instance name — the config dict key is assigned before compile (design §2.2)."""

# the v2 bundle stream name for the reconstructed objects (matches the MaskDecoder
# out_stream and the MaskFormerTargets object stream — FD uses objects.* / labels.objects.*)
_OBJECT_STREAM = "objects"


class MaskFormerMatchedLoss(nn.Module):
    """Hungarian-matched MaskFormer loss over the decoder's object predictions (FD 1158-1170).

    A FIT|VAL-only `GraphModule`. It composes a v1 ``MaskFormerLoss`` (which holds
    the ``HungarianMatcher`` + the ``empty_weight`` buffer + the v1 ``loss_labels``/
    ``loss_masks`` methods), runs the matcher on the SCALED object predictions/targets
    (v1 ``get_batch_cost``, matcher.py:171-248), then:

    - publishes the matcher-permuted predictions + the truth labels as NEW
      ``matched.objects.*`` keys (no in-place permute — design §2.1); and
    - emits ``losses.{object_class_ce, mask_dice, mask_focal, regression}`` (only the
      components with a positive ``loss_weights`` entry are produced, v1
      maskformer_loss.py:221-226).

    The three mask/class components are computed by the composed v1 methods on the
    permuted predictions (byte-faithful v1). The regression component is the MATCHED
    L1 over valid (non-null) objects (the FD alignment change; no v1 byte reference).

    Lifecycle (design §2.3): ``__init__`` builds the composed v1 loss (the matcher +
    buffer are width-free), `declare_io` is static (FIT|VAL only), `bind` validates
    the regression prediction/target widths agree when a regression component is
    requested. forward runs the matcher + the loss methods and returns ONLY the new
    ``matched.objects.*`` + ``losses.*`` keys (write-once).

    Parameters
    ----------
    num_classes : int
        The number of NON-null object classes (v1 ``loss_config.num_classes``,
        MaskFormer.yaml:58). The null/no-object class index is ``num_classes`` (the
        sentinel the matcher + target masks use, matcher.py:209). MUST equal the
        decoder's ``class_net.output_size - 1``.
    num_objects : int
        The number of object queries ``M`` (v1 ``num_objects``, passed from the
        ``MaskDecoder`` to the loss, maskformer.py:74; MaskFormer.yaml:36). The
        matcher cost matrix is ``[B, M, M]`` (matcher.py:218), so ``M`` must equal
        the decoder's ``num_objects`` AND the truth-object slot count. MUST be >= 1.
    loss_weights : Mapping[str, float]
        Per-component LOSS weights (v1 ``loss_config.loss_weights``,
        MaskFormer.yaml:59-63): keys among ``object_class_ce``, ``mask_dice``,
        ``mask_focal``, ``mask_ce``, ``regression``. A component is produced only
        when its weight is present and truthy (v1 maskformer_loss.py:221-226). The
        ``object_class_ce`` weight is always applied (v1 always computes labels).
    matcher_weights : Mapping[str, float] | None, optional
        Per-component MATCHER cost weights (v1 ``matcher_weights``), defaulting to
        ``loss_weights`` (v1 maskformer_loss.py:144-145).
    null_class_weight : float, optional
        The class-balance weight on the null category in the CE
        (v1 ``null_class_weight``, maskformer_loss.py:130,135-142), by default 0.5.
    input_stream : str, optional
        The decoder's object stream name, by default ``objects`` — the
        ``<input_stream>.{class_logits,class_probs,masks}`` keys it reads.

    Raises
    ------
    ConfigError
        On a non-positive ``num_classes``, an unknown ``loss_weights`` key, an
        all-zero matcher-weight sum (the matcher asserts it positive,
        matcher.py:167), or no requested loss component.
    """

    # the loss components this module knows how to emit (v1 maskformer_loss.py loss_map
    # + the matched-regression extension); mask_ce is v1-supported but NOT in the shipped
    # MaskFormer.yaml weights, so it is emitted only if weighted.
    _KNOWN_COMPONENTS = ("object_class_ce", "mask_dice", "mask_focal", "mask_ce", "regression")

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: Mapping[str, float],
        matcher_weights: Mapping[str, float] | None = None,
        null_class_weight: float = 0.5,
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
        # v1 requires object_class_ce in the matcher cost (get_batch_cost reads it
        # unconditionally, matcher.py:222) — default it to the loss weight if absent.
        self.matcher_weights.setdefault(
            "object_class_ce", self.loss_weights.get("object_class_ce", 1.0)
        )
        # the composed v1 matcher asserts the cost-weight sum is positive
        # (matcher.py:167, a bare AssertionError); promote it to the loud config
        # surface so a zero-cost config fails as a ConfigError at construction.
        if sum(self.matcher_weights.values()) == 0:
            raise ConfigError(
                "MaskFormerMatchedLoss: the matcher cost weights sum to 0 — at least one "
                f"matcher_weights entry must be positive (got {self.matcher_weights}; v1 "
                "HungarianMatcher asserts this, matcher.py:167)"
            )

        # the produced loss components: object_class_ce is always computed (v1 always
        # runs loss_labels), the others only when their loss weight is truthy
        # (v1 maskformer_loss.py:221-226). regression is the matched extension.
        self.components: tuple[str, ...] = tuple(
            c for c in self._KNOWN_COMPONENTS if c == "object_class_ce" or self.loss_weights.get(c)
        )
        if not self.components:
            raise ConfigError(
                "MaskFormerMatchedLoss: no loss component requested — set at least "
                "object_class_ce in loss_weights"
            )

        self.null_class_weight = float(null_class_weight)

        # compose the absorbed MaskFormerLoss (M7 W2c-3, v2-native above): it owns the
        # absorbed HungarianMatcher, the empty_weight class-balance buffer, and the three
        # loss methods. The matcher asserts sum(matcher_weights) != 0 (matcher.py:167); v1
        # passes the FULL loss_weights to MaskFormerLoss and forwards matcher_weights to the
        # matcher (maskformer_loss.py:148-152). ``losses=["labels", "masks"]`` is the v1
        # default; we drive loss_labels/loss_masks directly so the list is unused. The
        # absorbed loss is byte-for-byte v1's (the attribute name ``v1_loss`` is kept: the
        # MF1c gate oracle copies ``composed.v1_loss.state_dict()`` into a fresh v1 loss,
        # and the absorbed loss carries the SAME single ``empty_weight`` buffer key).
        self.v1_loss = MaskFormerLoss(
            num_classes=num_classes,
            num_objects=num_objects,  # the matcher cost matrix is [B, M, M] (matcher.py:218)
            loss_weights=self.loss_weights,
            matcher_weights=self.matcher_weights,
            null_class_weight=self.null_class_weight,
        )

    @property
    def matcher(self) -> HungarianMatcher:
        """The composed (absorbed v2-native) `HungarianMatcher` (owned by the loss).

        Returns
        -------
        HungarianMatcher
            The matcher instance the forward drives.
        """
        return self.v1_loss.matcher

    def _reg_pred_key(self) -> str:
        """The scaled object-regression PREDICTION key (``preds.<stream>.regression``).

        Returns
        -------
        str
            The bundle key the object-regression task publishes.
        """
        return f"preds.{self.input_stream}.regression"

    def _reg_tgt_key(self) -> str:
        """The scaled object-regression TARGET key (``targets.<stream>.regression``).

        Returns
        -------
        str
            The bundle key the object-regression task publishes in FIT|VAL.
        """
        return f"targets.{self.input_stream}.regression"

    def declare_io(self, mode: Mode) -> IO:
        """Declare the object preds + truth labels -> ``matched.objects.*`` + ``losses.*``.

        FIT|VAL only (the matched loss is pruned from TEST/ONNX, FD 1159). Requires
        the decoder's ``<stream>.{class_logits,class_probs,masks}`` and the truth
        ``labels.objects.{object_class,masks}``; additionally the scaled
        ``preds.<stream>.regression`` + ``targets.<stream>.regression`` when a
        regression component is requested. Produces the matcher-permuted predictions
        + the truth labels as NEW ``matched.objects.*`` keys (consumed by
        `MaskformerMetrics`) and one scalar ``losses.<component>`` per requested
        component.

        Returns
        -------
        IO
            Empty in TEST/ONNX (the module is mode-inactive there).
        """
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})

        m = self.num_objects  # the query count M (concrete; matcher cost is [B, M, M])
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
            # the matcher-permuted predictions + the truth labels, for MaskformerMetrics
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
        # the decoder's embed feeds the permuted matched.embed (MaskformerMetrics reads
        # the matched query embeddings); only required when the embed is available.
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

        The query count ``M`` is config-known (``num_objects``, baked into the
        composed matcher at ``__init__``), so nothing structural binds here. When a
        regression component is requested, the scaled prediction width
        (``preds.<stream>.regression``) and target width
        (``targets.<stream>.regression``) must match — the matched L1 is element-wise
        (v1 ``batch_mae_loss`` cost, matcher.py:131).

        Raises
        ------
        ConfigError
            If a requested regression component has mismatched pred/target widths.
        """
        if "regression" in self.components:
            wp = schema.width(self._reg_pred_key())
            wt = schema.width(self._reg_tgt_key())
            if wp != wt:
                raise ConfigError(
                    f"MaskFormerMatchedLoss {self.name!r}: regression prediction width {wp} != "
                    f"target width {wt} ({self._reg_pred_key()!r} vs {self._reg_tgt_key()!r}) — "
                    "the matched L1 is element-wise (v1 maskformer_loss.py regression cost)"
                )

    def _matched_regression_loss(
        self, reg_pred: Tensor, reg_tgt: Tensor, object_class: Tensor
    ) -> Tensor:
        """The MATCHED object-regression L1 over valid (non-null) objects (FD 1168-1170).

        The FD alignment change: the matcher-permuted regression predictions are
        aligned to the truth-order targets, and the L1 is averaged over the VALID
        objects only (``object_class != num_classes``) — the same validity mask v1
        uses for the mask losses (maskformer_loss.py:215). No v1 byte reference: v1's
        regression loss is query-order (see module docstring); this is the design's
        matched alignment, gated as a design-conformance property.

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

        Reproduces v1's matched loop (maskformer_loss.py:334-347) WITHOUT the
        in-place permute: solve the assignment (matcher reads the SCALED preds/targets,
        v1 get_batch_cost), gather the permuted predictions into NEW ``matched.*`` keys,
        and compute the three v1-decidable losses (``object_class_ce``, ``mask_dice``,
        ``mask_focal``) via the composed v1 methods on the permuted predictions plus
        the MATCHED regression L1.

        Returns
        -------
        dict[str, Tensor]
            The new ``matched.objects.*`` + ``losses.*`` keys only (design §2.5).
        """
        del mode
        class_logits = b.get(f"{self.input_stream}.class_logits")
        class_probs = b.get(f"{self.input_stream}.class_probs")
        masks = b.get(f"{self.input_stream}.masks")
        embed = b.get(f"{self.input_stream}.embed")
        object_class = b.get(f"labels.{_OBJECT_STREAM}.object_class")
        target_masks = b.get(f"labels.{_OBJECT_STREAM}.masks")

        # the matcher cost dict (v1 keys: class_logits/class_probs/masks [+regression],
        # matcher.py:181-190). Costs stay in SCALED space (FD 1166).
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

        # solve the optimal assignment: idx = (batch_arange[B,1], tgt_idx[B,M]) — the
        # advanced-indexing tuple v1 applies as v[idx] (matcher.py:293-296).
        idx = self.matcher(pred_for_match, tgt_for_match)

        # permute the predictions into NEW tensors (NO in-place; v1 does
        # preds["objects"][k] = v[idx] in place, maskformer_loss.py:338-343). Advanced
        # indexing returns a fresh tensor, so matched.* never aliases the decoder output.
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

        # the three v1-decidable losses, on the PERMUTED predictions vs truth-order
        # labels, via the composed v1 methods (loss_labels/loss_masks already apply
        # the loss weights via weight_loss, maskformer_loss.py:252-254 -> 256-271).
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
