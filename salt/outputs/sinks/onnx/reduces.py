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
        ``[B, R]`` for the leading object (after optional reordering).
    obj_indices : torch.Tensor | None
        Sparse mask indices per object, ``[B, M]``, values in ``[0, L)`` (``NaN``
        when undefined); ``None`` when there are no tracks (``L == 0``).
    class_probs : torch.Tensor
        Possibly-reordered class probabilities, ``[B, M, C]``.
    regression : torch.Tensor
        Possibly-reordered regression tensor, ``[B, M, R]``, ``NaN`` for null objects.
    """
    masks = objects["masks"]
    class_probs = objects["class_probs"]
    regression = objects["regression"]
    n_tracks = masks.shape[-1]
    n_obj = masks.shape[1]
    n_reg = regression.shape[-1]

    if n_tracks == 0:
        return (
            torch.full((1, n_obj), torch.nan),
            None,
            class_probs,
            torch.full((1, n_obj, n_reg), torch.nan),
        )
    null_preds = class_probs[:, :, -1] > max_null
    if not null_preds.any():
        return (
            torch.full((1, n_obj), torch.nan),
            torch.arange(n_tracks).unsqueeze(0).expand(1, n_tracks),
            class_probs,
            torch.full((1, n_obj, n_reg), torch.nan),
        )

    masks = masks.sigmoid() > 0.5
    expanded_null = null_preds.unsqueeze(-1).expand(-1, -1, masks.size(-1))
    masks[expanded_null] = torch.zeros_like(masks)[expanded_null]
    regression[null_preds] = torch.nan

    if apply_reorder:
        # leading object = highest regression[0] (e.g. pT); argsort doesn't handle
        # NaN reliably in Athena, so null entries go to -inf for the sort then
        # back to NaN afterward
        regression[null_preds] = -torch.inf
        order = torch.argsort(regression[:, :, 0], descending=True)
        regression[null_preds] = torch.nan
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
