import torch
from torch import BoolTensor, Tensor
from torch.nn.functional import pad, softmax

from salt.stypes import Tensors


def maybe_flatten_tensors(
    x: Tensor | Tensors,
) -> Tensor:
    if isinstance(x, dict):
        return flatten_tensor_dict(x)
    return x


def flatten_tensor_dict(
    x: dict[str, Tensor],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Tensor:
    """Flattens a dictionary of tensors into a single tensor.

    Parameters
    ----------
    x: dict[str, Tensor]
        Dictionary of tensors to flatten.
    include: list[str] | None, optional
        List of keys defining the tensors to be concatenated. If None, all tensors will be
        concatenated unless defined by 'exclude'. Cannot be used with 'exclude'.
    exclude: list[str] | None, optional
        List of keys to exclude from the concatenation. If None, all tensors will be
        concatenated unless defined by 'include'. Cannot be used with 'include'.

    Returns
    -------
    flattened : Tensor
        A single tensor containing the concatenated values of the input dictionary.
    """
    if include and exclude:
        raise ValueError("Cannot use 'include' and 'exclude' together")
    if include:
        return torch.cat([x[emb] for emb in include], dim=1)
    if exclude:
        return torch.cat([x[emb] for emb in x if emb not in exclude], dim=1)
    return torch.cat(list(x.values()), dim=1)


def masked_softmax(x: Tensor, mask: BoolTensor, dim: int = -1) -> Tensor:
    """Applies softmax over a tensor without including padded elements."""
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


def undo_padding(seq: Tensor, mask: BoolTensor) -> tuple:
    """Remove all padded elements from a tensor and return the sequence lengths."""
    mask = ~mask  # convert the mask such that True is a valid token
    seqlens = mask.sum(dim=-1)
    maxlen = seqlens.max().item()
    culens = pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seq[mask], culens, maxlen


def redo_padding(unpadded_seq: Tensor, mask: BoolTensor) -> Tensor:
    """Redo the padding and return a zero-padded tensor."""
    mask = ~mask  # convert the mask such that True is a valid token
    shape = (*mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[mask] = unpadded_seq
    return out


def add_dims(x: Tensor, ndim: int):
    """Adds dimensions to a tensor to match the shape of another tensor."""
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is smaller than input ndim ({x.dim()})")

    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])

    return x


def attach_context_single(x: Tensor, context: Tensor) -> Tensor:
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


def attach_context(x: Tensor, context: Tensor) -> Tensor:
    """Concatenate a context tensor to an input tensor with broadcasting.

    The idea behind this is to allow a context tensor less or equal dimensions to be
    concatenated to an input with more dimensions.
    This function checks the dimension difference and reshapes the context to have
    the same dimension as the constituents so broadcast concatenation can apply.
    The shape change always assumes that the first dimension is the batch and the last
    are the features.

    Here is a basic use case: concatenating the high level jet variables to the
    constituents during a forward pass through the network
    - The jet variables will be of shape: [batch, j_features]
    - The constituents will be of shape: [batch, num_nodes, n_features]
    Here the jet variable will be shaped to [batch, 1, j_features] allowing broadcast
    concatenation with the constituents

    Another example is using the edge features [b,n,n,ef] and concatenating the
    high level jet variables, which will be expanded to [b,1,1,jf] or the conditioning
    on the node features which will be expanded to [b,1,n,nf]
    """
    if isinstance(x, dict):
        return {key: attach_context_single(val, context) for key, val in x.items()}

    return attach_context_single(x, context)
