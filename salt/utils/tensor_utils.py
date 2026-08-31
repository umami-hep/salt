"""Shared tensor-shape helpers."""

from __future__ import annotations

import torch
from torch import BoolTensor, Tensor
from torch.nn.functional import pad, softmax


def add_dims(x: Tensor, ndim: int) -> Tensor:
    """Add singleton dimensions (after the batch dim) to reach a target rank.

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
    """Broadcast ``context`` to match ``x`` and concatenate along the last dim.

    Returns ``cat([context, x], dim=-1)`` — context is PREPENDED.

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


def attach_context(x: Tensor | dict[str, Tensor], context: Tensor) -> Tensor | dict[str, Tensor]:
    """Concatenate a context tensor to inputs (tensor or dict of tensors)."""
    if isinstance(x, dict):
        return {key: attach_context_single(val, context) for key, val in x.items()}
    return attach_context_single(x, context)


def flatten_tensor_dict(
    x: dict[str, Tensor],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Tensor:
    """Concatenate a dict of tensors along ``dim=1`` (``include``/``exclude`` are exclusive).

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
    """Softmax that ignores padded elements: masked (``True``) entries are set to
    ``-inf`` before the softmax and zeroed after.
    """
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


@torch.compiler.disable
def undo_padding(seq: Tensor, mask: BoolTensor) -> tuple[Tensor, Tensor, int]:
    """Remove padded elements; return the packed sequence + flash-varlen metadata.

    ``mask == True`` means padded (the mask is flipped internally).

    ``torch.compiler.disable`` keeps this out of the compiled region — it is a
    no-op unless the model is compiled. Two things here are not compilable:
    ``.item()`` (a scalar read dynamo cannot trace without
    ``capture_scalar_outputs``) and the boolean index ``seq[mask]``, whose output
    shape depends on the data. The latter lowers to ``aten.nonzero``, which
    **inductor refuses on CUDA** — so capturing it is not an option on the GPUs
    this runs on, and the cheap fix is to take one deliberate graph break here
    and let the whole layer stack compile with static shapes around it.
    ``maybe_mark_dynamic`` then stops the packed length from re-specialising
    every batch.

    Returns
    -------
    tuple[Tensor, Tensor, int]
        The packed (unpadded) sequence, the cumulative lengths (``int32``), and
        the maximum valid sequence length.
    """
    mask = ~mask  # convert mask: True -> valid token
    seqlens = mask.sum(dim=-1)
    maxlen = int(seqlens.max().item())
    culens = pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    packed = seq[mask]
    # the packed token count varies per batch: mark it dynamic so the compiled
    # region downstream guards on "any length" instead of recompiling per batch
    torch._dynamo.maybe_mark_dynamic(packed, 0)  # noqa: SLF001 - public-in-practice dynamo API
    return packed, culens, maxlen


@torch.compiler.disable
def redo_padding(unpadded_seq: Tensor, mask: BoolTensor) -> Tensor:
    """Re-apply padding to an unpadded sequence (zeros at padded positions).

    Disabled for the compiler for the same reason as `undo_padding`: the masked
    scatter ``out[mask] = ...`` is the same data-dependent ``aten.nonzero`` that
    inductor rejects on CUDA.
    """
    mask = ~mask  # convert mask: True -> valid token
    shape = (*mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[mask] = unpadded_seq
    return out
