from typing import Optional

from torch import Tensor, concat, nn


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
    dim_diff = x.dim() - context.dim()

    # Context cant have more dimensions
    if dim_diff < 0:
        raise ValueError(
            f"Provided context has more dimensions ({context.dim()}) than inputs ({x.dim()})"
        )

    # If reshaping is required
    if dim_diff > 0:
        # Reshape the context inputs with 1's after the batch dim
        context = context.view(context.shape[0], *dim_diff * (1,), *context.shape[1:])

        # Use expand to allow for broadcasting as expand does not allocate memory
        context = context.expand(*x.shape[:-1], -1)

    # Apply the concatenation on the final dimension
    return concat([x, context], dim=-1)


class Dense(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list,
        activation: str,
        final_activation: str = None,
        norm_layer: str = None,
        norm_final_layer: bool = False,
        dropout: float = 0.0,
        context_size: int = 0,
    ) -> None:
        """A simple fully connected feed forward neural network, which can take
        in additional contextual information.

        Parameters
        ----------
        input_size : int
            Input size
        output_size : int
            Output size
        hidden_layers : list
            Number of nodes per layer
        activation : str
            Activation function for hidden layers
        final_activation : str, optional
            Activation function for the output layer, by default None
        norm_layer : str, optional
            Normalisation layer, by default None
        norm_final_layer : bool, optional
            Whether to use normalisation on the final layer, by default False
        dropout : float, optional
            Apply dropout with the supplied probability, by default 0.0
        context_size : int
            Size of the context tensor, 0 means no context information is provided
        """
        super().__init__()

        # Save the networks input and output sizes
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size

        # build nodelist
        node_list = [input_size + context_size, *hidden_layers, output_size]

        # input and hidden layers
        layers = []

        num_layers = len(node_list) - 1
        for i in range(num_layers):
            is_final_layer = i == num_layers - 1

            # normalisation first
            if norm_layer and (norm_final_layer or not is_final_layer):
                layers.append(getattr(nn, norm_layer)(node_list[i], elementwise_affine=False))

            # then dropout
            if dropout and (norm_final_layer or not is_final_layer):
                layers.append(nn.Dropout(dropout))

            # linear projection
            layers.append(nn.Linear(node_list[i], node_list[i + 1]))

            # activation
            if not is_final_layer:
                layers.append(getattr(nn, activation)())

            # final layer: return logits by default, otherwise apply activation
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        # build the net
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.context_size:
            x = attach_context(x, context)
        return self.net(x)
