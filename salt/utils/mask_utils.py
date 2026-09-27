"""Mask helpers used by production ``salt`` code (``indices_from_mask``,
the ``mask_from_logits`` chain, ``reco_metrics``).
"""

import torch
from torch import BoolTensor, Tensor


def mask_from_indices(indices: Tensor, num_masks: int | None = None) -> BoolTensor:
    """Convert dense indices to a sparse boolean mask.

    Indices are assumed to be non-negative integers starting at 0. Negative entries
    are treated as "no index" and will be masked out (all-False in their column).

    Examples
    --------
    >>> mask_from_indices(torch.tensor([0, 1, 1]), num_masks=3)
    tensor([[ True, False, False],
            [False,  True,  True]])

    >>> mask_from_indices(torch.tensor([0, 1, 2]))
    tensor([[ True, False, False],
            [False,  True, False],
            [False, False,  True]])

    Parameters
    ----------
    indices : Tensor
        1D tensor of shape ``[L]`` or 2D tensor of shape ``[B, L]`` with integer indices.
    num_masks : int | None, optional
        The number of mask rows (i.e., maximum index + 1). If ``None``, it is inferred
        as ``indices.max() + 1``, by default ``None``.

    Returns
    -------
    BoolTensor
        If ``indices.ndim == 1``: mask of shape ``[num_masks, L]``.
        If ``indices.ndim == 2``: mask of shape ``[B, num_masks, L]``.
    """
    assert indices.ndim in {1, 2}, "indices must be 1D for single sample or 2D for batch"
    if num_masks is None:
        num_masks = indices.max() + 1
    else:
        assert num_masks > indices.max(), (
            "num_masks must be greater than the maximum value in indices"
        )

    indices = torch.as_tensor(indices)
    kwargs = {"dtype": torch.bool, "device": indices.device}
    if indices.ndim == 1:
        mask = torch.zeros((num_masks, indices.shape[-1]), **kwargs)
        mask[indices, torch.arange(indices.shape[-1], device=indices.device)] = True
        mask.transpose(0, 1)[indices < 0] = False  # handle negative indices
    else:
        mask = torch.zeros((indices.shape[0], num_masks, indices.shape[-1]), **kwargs)
        mask[
            torch.arange(indices.shape[0], device=indices.device).unsqueeze(-1),
            indices,
            torch.arange(indices.shape[-1], device=indices.device),
        ] = True
        mask.transpose(1, 2)[indices < 0] = False  # handle negative indices

    return mask


def indices_from_mask(mask: BoolTensor, noindex: int = -2) -> Tensor:
    """Convert a sparse boolean mask to dense indices.

    The inverse of :func:`mask_from_indices` for masks with exactly one ``True`` per
    column (or all ``False`` for "no index").

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


def sigmoid_mask(
    mask_logits: Tensor,
    threshold: float = 0.5,
    **kwargs,
) -> BoolTensor:
    """Compute a mask by thresholding mask logits with a sigmoid.

    Parameters
    ----------
    mask_logits : Tensor
        Logits of shape ``[B, N_obj, N_inp]``.
    threshold : float, optional
        Threshold applied after ``sigmoid``, by default ``0.5``.
    **kwargs
        Forwarded to :func:`sanitise_mask` (e.g. ``input_pad_mask=...``,
        ``object_class_preds=...``).

    Returns
    -------
    BoolTensor
        Boolean mask of shape ``[B, N_obj, N_inp]``.
    """
    mask = mask_logits.sigmoid() > threshold
    return sanitise_mask(mask, **kwargs)


def argmax_mask(
    mask_logits: Tensor,
    weighted: bool = False,
    **kwargs,
) -> BoolTensor:
    """Compute a mask by taking the argmax over objects for each input.

    If ``weighted`` is ``True``, logits are first converted to probabilities (softmax
    over the object dimension) and multiplied by per-object class confidence
    (maximum over classes), following the MaskFormer-style weighting.

    Parameters
    ----------
    mask_logits : Tensor
        Logits of shape ``[B, N_obj, N_inp]``.
    weighted : bool, optional
        If ``True``, weight by per-object class confidence. Requires
        ``object_class_preds`` in ``kwargs``, by default ``False``.
    **kwargs
        Forwarded to :func:`sanitise_mask`. When ``weighted=True``, must include
        ``object_class_preds: Tensor`` with shape ``[B, N_obj, C]``.

    Returns
    -------
    BoolTensor
        Boolean mask of shape ``[B, N_obj, N_inp]``.

    Raises
    ------
    ValueError
        If ``weighted=True`` and ``object_class_preds`` is not supplied.
    """
    if weighted and kwargs.get("object_class_preds") is None:
        raise ValueError("weighted argmax requires object_class_preds")

    if not weighted:
        idx = mask_logits.argmax(-2)
    else:
        confidence = kwargs["object_class_preds"].max(-1)[0].unsqueeze(-1)
        assert (  # noqa: PT018
            confidence.min() >= 0.0 and confidence.max() <= 1.0
        ), "confidence must be between 0 and 1"
        idx = (mask_logits.softmax(-2) * confidence).argmax(-2)
    mask = mask_from_indices(idx, num_masks=mask_logits.shape[-2])

    return sanitise_mask(mask, **kwargs)


def mask_from_logits(
    logits: Tensor,
    mode: str,
    input_pad_mask: BoolTensor | None = None,
    object_class_preds: Tensor | None = None,
) -> BoolTensor:
    """Dispatch helper to convert logits to a mask by different strategies.

    Parameters
    ----------
    logits : Tensor
        Logits of shape ``[B, N_obj, N_inp]``.
    mode : str
        One of ``{"sigmoid", "argmax", "weighted_argmax"}``.
    input_pad_mask : BoolTensor | None, optional
        Input padding mask (``[B, N_inp]``), by default ``None``.
    object_class_preds : Tensor | None, optional
        Object class logits/probs (``[B, N_obj, C]``) required for
        ``mode="weighted_argmax"``, by default ``None``.

    Returns
    -------
    BoolTensor
        Boolean mask of shape ``[B, N_obj, N_inp]``.

    Raises
    ------
    ValueError
        If ``mode`` is not supported.
    """
    modes = {"sigmoid", "argmax", "weighted_argmax"}
    if mode == "sigmoid":
        return sigmoid_mask(
            logits, input_pad_mask=input_pad_mask, object_class_preds=object_class_preds
        )
    if mode == "argmax":
        return argmax_mask(
            logits, input_pad_mask=input_pad_mask, object_class_preds=object_class_preds
        )
    if mode == "weighted_argmax":
        return argmax_mask(
            logits,
            weighted=True,
            input_pad_mask=input_pad_mask,
            object_class_preds=object_class_preds,
        )

    raise ValueError(f"mode must be one of {modes}")


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


def mask_eff_pur(
    m_pred: BoolTensor,
    m_tgt: BoolTensor,
    flat: bool = False,
    reduce: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute efficiency and purity, either per object or globally.

    Parameters
    ----------
    m_pred : BoolTensor
        Predicted mask of shape ``[B, N_obj, N_inp]``.
    m_tgt : BoolTensor
        Target mask of shape ``[B, N_obj, N_inp]``.
    flat : bool, optional
        If ``True``, compute a single global efficiency/purity across all objects
        (edgewise). If ``False``, compute per-object and optionally reduce with
        ``nanmean``, by default ``False``.
    reduce : bool, optional
        When ``flat=False``, if ``True`` reduce to scalars using ``nanmean``,
        by default ``False``.

    Returns
    -------
    tuple[Tensor, Tensor]
        If ``flat=True``: two 0-D tensors (scalars).
        If ``flat=False`` and ``reduce=False``: two tensors of shape ``[B, N_obj]``.
        If ``flat=False`` and ``reduce=True``: two 0-D tensors (scalars).
    """
    if flat:
        # per assignment metric (i.e. edgewise)
        eff = (m_pred & m_tgt).sum() / m_tgt.sum()
        pur = (m_pred & m_tgt).sum() / m_pred.sum()
    else:
        # per object metric (nanmean avoids invalid indices)
        eff, pur = mask_effs_purs(m_pred, m_tgt)
        if reduce:
            eff, pur = eff.nanmean(), pur.nanmean()
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
