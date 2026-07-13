"""Gated linear unit block (v1 salt/models/transformer.py GLU absorption)."""

from __future__ import annotations

from torch import Tensor, nn


class GLU(nn.Module):
    """Dense update with a (gated) linear unit. See https://arxiv.org/abs/2002.05202.

    Parameters
    ----------
    embed_dim : int
        Input/output embedding dimension.
    hidden_dim : int | None, optional
        Hidden dimension. If ``None``, defaults to ``2 * embed_dim``.
    activation : str, optional
        Name of the activation class in ``torch.nn`` (e.g., ``"SiLU"``).
    dropout : float, optional
        Dropout probability. The default is ``0.0``.
    bias : bool, optional
        Whether to include bias terms. The default is ``True``.
    gated : bool, optional
        If ``True``, uses a gated branch (splits hidden in two). The default is ``False``.
    mup : bool, optional
        Whether to use μP parameterization. The default is ``False``.
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
        """Apply the GLU block.

        Returns
        -------
        Tensor
            Output tensor of shape ``[B, L, D]``.
        """
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)
