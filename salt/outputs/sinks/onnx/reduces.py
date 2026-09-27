"""Shared MaskFormer export math used by the folded conversion nodes in `salt.outputs`."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from salt.utils.mask_utils import indices_from_mask

__all__ = ["get_maskformer_outputs"]


# ---------------------------------------------------------------------------
# shared MaskFormer export math
# ---------------------------------------------------------------------------
# `get_maskformer_outputs` (null suppression + pT reorder + index math) is a pure
# function the folded MaskFormerObjects conversion node composes.


def get_maskformer_outputs(
    objects: Mapping[str, Tensor],
    max_null: float = 0.5,
    apply_reorder: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert raw MaskFormer-style outputs to convenient per-object tensors.

    Thresholds the "null" class probability and suppresses masks/regression for
    objects with ``p_null > max_null``; converts per-position mask logits into
    sparse mask indices; optionally reorders objects so the "leading" object
    (highest ``regression[0]``, e.g. pT) is first.

    Parameters
    ----------
    objects : Mapping[str, Tensor]
        Keys: ``"masks"`` (mask logits ``[B, M, L]``), ``"class_probs"``
        (``[B, M, C]``, last class is null), ``"regression"`` (``[B, M, R]``).
    max_null : float, optional
        Maximum allowed null probability for an object to be kept, by default ``0.5``.
    apply_reorder : bool, optional
        Reorder objects in descending order of ``regression[..., 0]``, by default ``True``.

    Returns
    -------
    leading_regression : torch.Tensor
        ``[B, R]`` for the leading object (after optional reordering); all
        ``NaN`` for a jet where every object is null.
    obj_indices : torch.Tensor
        The per-token owning-object index, ``[B, L]`` int64 (`indices_from_mask`):
        ``-2`` where no object claims a token; ``[B, 0]`` when ``L == 0``.
    class_probs : torch.Tensor
        Possibly-reordered class probabilities, ``[B, M, C]``.
    regression : torch.Tensor
        Possibly-reordered regression tensor, ``[B, M, R]``, ``NaN`` for null objects.

    Notes
    -----
    Deliberately no data-dependent early return: this is one straight-line
    tensor program for every batch/length, so the eager output is identical
    to the graph ``torch.onnx.export`` traces at any length. An earlier
    version special-cased ``n_tracks == 0`` and ``not null_preds.any()``
    (inherited from v1 `2dfc10d`, 2024-07-30) — the second branch fired when
    NO object was null (i.e. every object real) rather than when all were,
    and both returned batch-1-shaped dummy tensors instead of the real
    per-batch shapes. An all-null jet is handled correctly by the single path
    below: every mask position is suppressed, `indices_from_mask` reports
    ``-2`` everywhere, and `regression`/`leading_regression` are ``NaN``.
    """
    masks = objects["masks"]
    class_probs = objects["class_probs"]
    regression = objects["regression"]

    null_preds = class_probs[:, :, -1] > max_null

    masks = masks.sigmoid() > 0.5
    expanded_null = null_preds.unsqueeze(-1).expand(-1, -1, masks.size(-1))
    masks = masks & ~expanded_null
    null_reg = null_preds.unsqueeze(-1).expand_as(regression)
    regression = torch.where(null_reg, torch.full_like(regression, torch.nan), regression)

    if apply_reorder:
        # leading object = highest regression[0] (e.g. pT); argsort doesn't handle
        # NaN reliably in Athena, so null entries go to -inf for the sort
        # (regression is already NaN there, from above)
        sort_key = torch.where(
            null_preds, torch.full_like(regression[:, :, 0], -torch.inf), regression[:, :, 0]
        )
        order = torch.argsort(sort_key, descending=True)
        order_expanded = order.unsqueeze(-1).expand(-1, -1, masks.size(-1))

        masks = torch.gather(masks, 1, order_expanded)
        class_probs = torch.gather(
            class_probs, 1, order.unsqueeze(-1).expand(-1, -1, class_probs.size(-1))
        )
        regression = torch.gather(
            regression, 1, order.unsqueeze(-1).expand(-1, -1, regression.size(-1))
        )
    leading_regression = regression[:, 0]

    obj_indices = indices_from_mask(masks)

    return leading_regression, obj_indices, class_probs, regression
