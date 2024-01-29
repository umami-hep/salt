from torch import Tensor, nn

from salt.utils.tensor_utils import attach_context


class Dense(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: list[int] | None = None,
        hidden_dim_scale: int = 2,
        activation: str = "ReLU",
        final_activation: str | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        context_size: int = 0,
        muP: bool = False,
    ) -> None:
        """A fully connected feed forward neural network, which can take
        in additional contextual information.

        Parameters
        ----------
        input_size : int
            Input size
        output_size : int
            Output size. If not specified this will be the same as the input size.
        hidden_layers : list, optional
            Number of nodes per layer, if not specified, the network will have
            a single hidden layer with size `input_size * hidden_dim_scale`.
        hidden_dim_scale : int, optional
            Scale factor for the hidden layer size.
        activation : str
            Activation function for hidden layers.
            Must be a valid torch.nn activation function.
        final_activation : str, optional
            Activation function for the output layer.
            Must be a valid torch.nn activation function.
        dropout : float, optional
            Apply dropout with the supplied probability.
        bias : bool, optional
            Whether to use bias in the linear layers.
        context_size : int
            Size of the context tensor, 0 means no context information is provided.
        muP: bool, optional,
            Whether to use the muP parametrisation (impacts initialisation).
        """
        super().__init__()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]

        # Save the networks input and output sizes
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.muP = muP

        # build nodelist
        self.node_list = [input_size + context_size, *hidden_layers, output_size]

        # input and hidden layers
        layers = []

        num_layers = len(self.node_list) - 1
        for i in range(num_layers):
            if dropout:
                layers.append(nn.Dropout(dropout))

            # linear projection
            layers.append(nn.Linear(self.node_list[i], self.node_list[i + 1], bias=bias))

            # activation for all but the final layer
            if i != num_layers - 1:
                layers.append(getattr(nn, activation)())

            # final layer: return logits by default, or activation if specified
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        # build the net
        self.net = nn.Sequential(*layers)

        if self.muP:
            self._reset_parameters()

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        if self.context_size:
            x = attach_context(x, context)
        return self.net(x)

    def _reset_parameters(self):
        """Initialise the weights and biases for muP."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                std = 1.0 / layer.weight.shape[0] ** 0.5
                nn.init.normal_(layer.weight, std=std)
                nn.init.constant_(layer.bias, 0)
