"""Mask helpers used by production ``salt`` code (``indices_from_mask``,
``sanitise_mask``, ``mask_effs_purs``, ``reco_metrics``).
"""

import torch
from torch import BoolTensor, Tensor


def indices_from_mask(mask: BoolTensor, noindex: int = -2) -> Tensor:
    """Convert a sparse boolean mask to dense indices.

    Each column's index is the row holding its single ``True``; masks must have
    exactly one ``True`` per column (or all ``False`` for "no index").

    Examples
    --------
    >>> m = torch.tensor([[True, False, False],
    ...                   [False, True,  True]])
    >>> indices_from_mask(m)
    tensor([0, 1, 1])

    Parameters
    ----------
    mask : BoolTensor
        Mask of shape ``[K, L]`` or ``[B, K, L]`` (``K`` masks over ``L`` columns).
    noindex : int, optional
        Value used where a column has no ``True`` entry, by default ``-2``.

    Returns
    -------
    Tensor
        If ``mask.ndim == 2``: tensor of shape ``[L]``.
        If ``mask.ndim == 3``: tensor of shape ``[B, L]``.

    Raises
    ------
    ValueError
        If ``mask`` is not 2D or 3D.

    Notes
    -----
    Trace-safe: no data-dependent Python control flow, so this is safe under
    ``torch.onnx.export``. A column claimed by several objects gets the highest
    claiming object index.
    """
    mask = torch.as_tensor(mask)
    if mask.ndim not in {2, 3}:
        raise ValueError("mask must be 2D for single sample or 3D for batch")
    # owning object per column = highest claiming object index k (== the previous
    # nonzero/index_put last-write-wins), via a reduction instead of a scatter with
    # duplicate indices (undefined in ONNX ScatterND)
    k = torch.arange(mask.shape[-2], dtype=torch.long, device=mask.device).unsqueeze(-1)
    claimed = torch.where(mask, k, torch.full_like(k, -1))  # [..., K, L]
    # append one unclaimed (-1) column along L so the reductions below never see an
    # empty input when L == 0 (onnxruntime mis-shapes ReduceMax on empty data); it is
    # sliced off after the amax. Built from k (never empty along L), not from claimed.
    pad_col = torch.full_like(k, -1).expand(*claimed.shape[:-1], 1)  # [..., K, 1]
    claimed = torch.cat([claimed, pad_col], dim=-1)  # [..., K, L + 1]
    # pad one all -1 row on the object axis so amax is defined when K == 0
    claimed = torch.nn.functional.pad(claimed, (0, 0, 1, 0), value=-1)
    indices = claimed.amax(dim=-2)[..., :-1]  # [L] or [B, L]; -1 = unclaimed (drop pad column)
    # rebase so claimed indices start from 0 (batch-global min), as a tensor op so
    # torch.onnx.export does not bake the trace sample's min in as a constant
    exists = indices >= 0
    sentinel = torch.iinfo(torch.long).max
    masked = torch.where(exists, indices, torch.full_like(indices, sentinel))
    flat = torch.cat([masked.reshape(-1), masked.new_full((1,), sentinel)])
    minval = flat.min()
    minval = torch.where(minval == sentinel, torch.zeros_like(minval), minval)
    return torch.where(exists, indices - minval, torch.full_like(indices, noindex))


def sanitise_mask(
    mask: BoolTensor,
    input_pad_mask: BoolTensor | None = None,
    object_class_preds: Tensor | None = None,
) -> BoolTensor:
    """Sanitise predicted masks by removing padded inputs and null-class predictions.

    Parameters
    ----------
    mask : BoolTensor
        Predicted mask of shape ``[B, N_obj, N_inp]``.
    input_pad_mask : BoolTensor | None, optional
        Boolean padding mask over inputs with shape ``[B, N_inp]`` where ``True`` marks
        padded inputs to be removed, by default ``None``.
    object_class_preds : Tensor | None, optional
        Class logits or probabilities of shape ``[B, N_obj, C]``. If provided,
        the null class is assumed to be the last index (``C-1``) and masks for
        objects predicted as null are zeroed out, by default ``None``.

    Returns
    -------
    BoolTensor
        Sanitised mask with the same shape as ``mask``.
    """
    if input_pad_mask is not None:
        mask.transpose(1, 2)[input_pad_mask] = False
    if object_class_preds is not None:
        pred_null = object_class_preds.argmax(-1) == object_class_preds.shape[-1] - 1
        mask[pred_null] = False
    return mask


def mask_effs_purs(m_pred: BoolTensor, m_tgt: BoolTensor) -> tuple[Tensor, Tensor]:
    """Compute per-object efficiency and purity tensors.

    Parameters
    ----------
    m_pred : BoolTensor
        Predicted mask of shape ``[B, N_obj, N_inp]``.
    m_tgt : BoolTensor
        Target mask of shape ``[B, N_obj, N_inp]``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(eff, pur)`` where each has shape ``[B, N_obj]``.
        Efficiency is ``(m_pred & m_tgt).sum(-1) / m_tgt.sum(-1)``.
        Purity is     ``(m_pred & m_tgt).sum(-1) / m_pred.sum(-1)``.
    """
    eff = (m_pred & m_tgt).sum(-1) / m_tgt.sum(-1)
    pur = (m_pred & m_tgt).sum(-1) / m_pred.sum(-1)
    return eff, pur


def reco_metrics(
    pred_mask: BoolTensor,
    tgt_mask: BoolTensor,
    pred_valid: Tensor | None = None,
    reduce: bool = False,
    min_recall: float = 1.0,
    min_purity: float = 1.0,
    min_constituents: int = 0,
) -> tuple[Tensor, Tensor]:
    """Compute object-level reconstruction metrics (efficiency and fake rate).

    An object is considered **valid** if it has at least one predicted constituent
    (or as provided by ``pred_valid``) and, optionally, if it has at least
    ``min_constituents`` constituents. A valid object passes if both its efficiency
    and purity meet the provided thresholds; otherwise it is counted as fake.

    Parameters
    ----------
    pred_mask : BoolTensor
        Predicted mask of shape ``[B, N_obj, N_inp]``.
    tgt_mask : BoolTensor
        Target mask of shape ``[B, N_obj, N_inp]``.
    pred_valid : Tensor | None, optional
        Optional boolean tensor ``[B, N_obj]`` indicating which predictions are considered
        valid before thresholding. If ``None``, validity is ``pred_mask.sum(-1) > 0``,
        by default ``None``.
    reduce : bool, optional
        If ``True``, return mean values (over valid targets/predictions). If ``False``,
        return per-object boolean tensors, by default ``False``.
    min_recall : float, optional
        Minimum per-object efficiency to be considered correct, by default ``1.0``.
    min_purity : float, optional
        Minimum per-object purity to be considered correct, by default ``1.0``.
    min_constituents : int, optional
        Minimum number of predicted constituents required for an object to be valid,
        by default ``0``.

    Returns
    -------
    tuple[Tensor, Tensor]
        If ``reduce=False``: two boolean tensors ``(eff, fake)`` of shape ``[B, N_obj]``,
        where ``eff[b, i]`` is ``True`` if object ``i`` in batch ``b`` passes both
        thresholds, and ``fake[b, i]`` indicates a valid but failed prediction.
        If ``reduce=True``: two 0-D tensors (scalars) giving the mean efficiency over
        targets with at least one constituent and the mean fake rate over valid predictions.
    """
    if pred_valid is None:
        pred_valid = pred_mask.sum(-1) > 0
    else:
        pred_valid = pred_valid.clone()
        pred_valid &= pred_mask.sum(-1) > 0

    eff, pur = mask_effs_purs(pred_mask, tgt_mask)
    pass_cuts = (eff >= min_recall) & (pur >= min_purity)

    if min_constituents > 0:
        pred_valid &= pred_mask.sum(-1) >= min_constituents

    eff = pred_valid & pass_cuts
    fake = pred_valid & ~pass_cuts

    if reduce:
        valid_tgt = tgt_mask.sum(-1) > 0
        eff = eff[valid_tgt].float().mean()
        fake = fake[pred_valid].float().mean()

    return eff, fake
