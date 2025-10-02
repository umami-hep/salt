import torch
from torch import BoolTensor, Tensor
from torch.nn.functional import pad, softmax

from salt.stypes import Tensors


def maybe_flatten_tensors(x: Tensor | Tensors) -> Tensor:
    """Return a single tensor, flattening a tensor-dict if needed.

    If ``x`` is a ``dict[str, Tensor]``, the tensors are concatenated
    along the feature dimension using :func:`flatten_tensor_dict`.
    Otherwise, the input tensor is returned unchanged.

    Parameters
    ----------
    x : Tensor | Tensors
        Either a single tensor or a dictionary of tensors.

    Returns
    -------
    Tensor
        A single (possibly concatenated) tensor.
    """
    if isinstance(x, dict):
        return flatten_tensor_dict(x)
    return x


def flatten_tensor_dict(
    x: dict[str, Tensor],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Tensor:
    """Flatten (concatenate) a dictionary of tensors into one tensor.

    All tensors are concatenated along ``dim=1`` (feature dimension).
    You may either select an explicit subset with ``include`` **or**
    omit specific keys with ``exclude``—but not both.

    Parameters
    ----------
    x : dict[str, Tensor]
        Dictionary of tensors to concatenate. Each tensor must share
        the same batch and (if present) sequence dimensions so that
        concatenation along ``dim=1`` is valid.
    include : list[str] | None, optional
        Keys to include in the concatenation. If provided, only these
        tensors are concatenated. Mutually exclusive with ``exclude``.
    exclude : list[str] | None, optional
        Keys to exclude from the concatenation. Mutually exclusive with
        ``include``.

    Returns
    -------
    Tensor
        Single tensor formed by concatenating the selected tensors
        along ``dim=1``.

    Raises
    ------
    ValueError
        If both ``include`` and ``exclude`` are provided.
    """
    if include and exclude:
        raise ValueError("Cannot use 'include' and 'exclude' together")
    if include:
        return torch.cat([x[emb] for emb in include], dim=1)
    if exclude:
        return torch.cat([x[emb] for emb in x if emb not in exclude], dim=1)
    return torch.cat(list(x.values()), dim=1)


def masked_softmax(x: Tensor, mask: BoolTensor | None, dim: int = -1) -> Tensor:
    """Apply softmax while ignoring (masking) padded elements.

    Elements where ``mask`` is ``True`` are excluded from the softmax by
    setting them to ``-inf`` before the operation, and then zeroed after.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    mask : BoolTensor | None
        Padding mask broadcastable to ``x`` after expanding with
        :func:`add_dims`. A value of ``True`` indicates a padded element
        to be ignored. If ``None``, no masking is applied.
    dim : int, optional
        Dimension over which to apply the softmax, by default ``-1``.

    Returns
    -------
    Tensor
        Tensor after masked softmax.
    """
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


def undo_padding(seq: Tensor, mask: BoolTensor) -> tuple[Tensor, Tensor, int]:
    """Remove padded elements and return packed sequence info.

    Converts a padded sequence to a packed (unpadded) form using ``mask``.
    Here, the convention is:
    - ``mask == True``  → padded element
    - ``mask == False`` → valid element

    This function flips the mask internally to match valid positions.

    Parameters
    ----------
    seq : Tensor
        Padded sequence tensor of shape ``(B, S, D)`` (or similar).
    mask : BoolTensor
        Padding mask of shape ``(B, S)`` where ``True`` indicates padding.

    Returns
    -------
    unpadded : Tensor
        Tensor containing only valid (non-padded) positions, flattened
        along the sequence dimension (shape ``(sum(valid_lens), D)`` for
        a 3D input).
    culens : Tensor
        Cumulative lengths tensor of shape ``(B + 1,)`` with ``int32`` dtype,
        suitable for variable-length attention kernels.
    maxlen : int
        The maximum valid sequence length in the batch.
    """
    mask = ~mask  # convert mask: True -> valid token
    seqlens = mask.sum(dim=-1)
    maxlen = int(seqlens.max().item())
    culens = pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seq[mask], culens, maxlen


def redo_padding(unpadded_seq: Tensor, mask: BoolTensor) -> Tensor:
    """Re-apply padding to an unpadded sequence.

    Parameters
    ----------
    unpadded_seq : Tensor
        Unpadded sequence tensor where valid elements were stacked.
    mask : BoolTensor
        Original padding mask of shape ``(B, S)`` where ``True`` indicates padding.

    Returns
    -------
    Tensor
        Padded tensor of shape ``(B, S, D)`` (matching the original layout),
        filled with zeros at padded positions and values from ``unpadded_seq``
        at valid positions.
    """
    mask = ~mask  # convert mask: True -> valid token
    shape = (*mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[mask] = unpadded_seq
    return out


def add_dims(x: Tensor, ndim: int) -> Tensor:
    """Add singleton dimensions to reach a target rank.

    The new singleton dimensions are inserted after the batch dimension
    (i.e., at position 1 repeatedly) until ``x.ndim == ndim``.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    ndim : int
        Target number of dimensions.

    Returns
    -------
    Tensor
        Tensor reshaped with added singleton dimensions.

    Raises
    ------
    ValueError
        If ``ndim`` is smaller than ``x.ndim``.
    """
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is smaller than input ndim ({x.dim()})")

    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])

    return x


def attach_context_single(x: Tensor, context: Tensor) -> Tensor:
    """Concatenate a context tensor to a single tensor with broadcasting.

    The ``context`` tensor is expanded (via :func:`add_dims` and broadcast)
    so its rank matches ``x``; it is then concatenated with ``x`` along
    the last dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(B, ..., F)``.
    context : Tensor
        Context tensor of shape ``(B, F_ctx)`` or broadcastable to
        ``(B, ..., F_ctx)``.

    Returns
    -------
    Tensor
        Concatenation of ``context`` and ``x`` along the feature dimension,
        with shape ``(B, ..., F_ctx + F)``.

    Raises
    ------
    RuntimeError
        If ``context`` is ``None``.
    ValueError
        If the provided context has more dimensions than the input.
    """
    if context is None:
        raise RuntimeError("Expected context is missing from forward pass")

    if (dim_diff := x.dim() - context.dim()) < 0:
        raise ValueError(
            f"Provided context has more dimensions ({context.dim()}) than inputs ({x.dim()})"
        )

    if dim_diff > 0:
        context = add_dims(context, x.dim())
        context = context.expand(*x.shape[:-1], -1)

    return torch.cat([context, x], dim=-1)


def attach_context(x: Tensor | dict[str, Tensor], context: Tensor) -> Tensors:
    """Concatenate a context tensor to inputs (tensor or dict of tensors).

    This is a convenience wrapper over :func:`attach_context_single` that
    applies the operation to either a single tensor or every tensor in a
    dictionary.

    Parameters
    ----------
    x : Tensor | dict[str, Tensor]
        Input tensor or dictionary of tensors to which the context will be
        concatenated along the last dimension.
    context : Tensor
        Context tensor of shape ``(B, F_ctx)`` (or broadcastable to each
        input).

    Returns
    -------
    Tensors
        If ``x`` is a tensor, returns a tensor with context concatenated.
        If ``x`` is a dict, returns a dict with each value concatenated
        with the context.
    """
    if isinstance(x, dict):
        return {key: attach_context_single(val, context) for key, val in x.items()}
    return attach_context_single(x, context)
