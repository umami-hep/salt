"""Dense MLP blocks: the fully-connected `Dense` stack and the `GLU` feed-forward unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from salt.graph.errors import ConfigError
from salt.utils.tensor_utils import (
    attach_context,
)


class Dense(nn.Module):
    """A fully connected feed forward neural network, with optional context.

    Parameters
    ----------
    input_size : int
        Input size
    output_size : int | None, optional
        Output size. If not specified this will be the same as the input size, by default None
    hidden_layers : list[int] | None, optional
        Number of nodes per layer, if not specified, the network will have
        a single hidden layer with size `input_size * hidden_dim_scale`, by default None
    hidden_dim_scale : int, optional
        Scale factor for the hidden layer size, by default 2
    activation : str, optional
        Activation function for hidden layers. Must be a valid torch.nn activation function.
        By default "ReLU"
    final_activation : str | None, optional
        Activation function for the output layer. Must be a valid torch.nn activation function.
        By default None
    dropout : float, optional
        Apply dropout with the supplied probability, by default 0.0
    bias : bool, optional
        Whether to use bias in the linear layers, by default True
    context_size : int, optional
        Size of the context tensor, 0 means no context information is provided, by default 0
    mup : bool, optional
        Whether to use the muP parametrisation (impacts initialisation), by default None
    """

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
        mup: bool = False,
    ) -> None:
        super().__init__()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.mup = mup

        self.node_list = [input_size + context_size, *hidden_layers, output_size]

        layers = []

        num_layers = len(self.node_list) - 1
        for i in range(num_layers):
            if dropout:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(self.node_list[i], self.node_list[i + 1], bias=bias))

            if i != num_layers - 1:
                layers.append(getattr(nn, activation)())
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

        if self.mup:
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


class GLU(nn.Module):
    """Dense update with a (gated) linear unit. See https://arxiv.org/abs/2002.05202.

    ``hidden_dim`` defaults to ``2 * embed_dim``; ``gated=True`` splits the
    hidden layer in two (one half gates the other).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
        mup: bool = False,
    ):
        super().__init__()
        self.mup = mup

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        if self.mup:
            for proj in [self.in_proj, self.out_proj]:
                nn.init.normal_(proj.weight, mean=0.0, std=1.0 / (proj.weight.shape[0] ** 0.5))
                if bias:
                    nn.init.zeros_(proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the GLU block; returns ``[B, L, D]``."""
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)


def _reject_width_keys(who: str, cfg: Mapping[str, Any] | None, banned: tuple[str, ...]) -> None:
    """Reject configured width keys (inferred at bind); raises `ConfigError` naming them."""
    if cfg and (bad := sorted(set(cfg) & set(banned))):
        raise ConfigError(
            f"{who}: dense config must not set {bad} — widths are inferred at bind from the "
            "resolved schema (no YAML width arithmetic)"
        )
