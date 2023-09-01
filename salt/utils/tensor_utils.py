import torch
from torch import BoolTensor, Tensor
from torch.nn.functional import softmax


def masked_softmax(x: Tensor, mask: BoolTensor, dim: int = -1) -> Tensor:
    """Applies softmax over a tensor without including padded elements."""
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)

    x = softmax(x, dim=dim)

    if mask is not None:
        x = x.masked_fill(mask, 0)

    return x


def add_dims(x: Tensor, ndim: int):
    """Adds dimensions to a tensor to match the shape of another tensor."""
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is larger than input ndim ({x.dim()})")

    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])

    return x


def attach_context(x: Tensor, context: Tensor) -> Tensor:
    """Concatenates a context tensor to an input tensor with considerations for
    broadcasting.

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
    if context is None:
        raise RuntimeError("Expected context is missing from forward pass")

    # Check if the context information has less dimensions and the broadcast is needed
    if (dim_diff := x.dim() - context.dim()) < 0:
        raise ValueError(
            f"Provided context has more dimensions ({context.dim()}) than inputs ({x.dim()})"
        )

    # If reshaping is required
    if dim_diff > 0:
        # Reshape the context inputs with 1's after the batch dim
        context = add_dims(context, x.dim())

        # Use expand to allow for broadcasting as expand does not allocate memory
        context = context.expand(*x.shape[:-1], -1)

    # Apply the concatenation on the final dimension
    return torch.cat([x, context], dim=-1)
