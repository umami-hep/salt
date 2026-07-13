"""PositionalEncoder GraphModule (v1 salt/models/posenc.py absorption)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import _UNNAMED

_POSENC_SYM_VARS: frozenset[str] = frozenset({"phi"})
"""Variables whose positional encoding is symmetric (sin/cos of the sin/cos)."""


class PositionalEncoder(nn.Module):
    """Sin/cos positional encoding over coordinate variables.

    Evenly shares the embedding space between the encoded variables; any
    remaining dimensions are left as zeros. Parameter-free (``@torch.no_grad``).

    Parameters
    ----------
    variables : Sequence[str]
        Variable names to encode. Symmetric variables (``phi``) get the
        sin-of-sin / sin-of-cos symmetric encoding.
    dim : int
        Total positional-encoding width, split evenly across variables
        (``per_input_dim = dim // (2 * len(variables))``, remainder zero-padded).
    alpha : int, optional
        Frequency scaling factor, by default 100.

    Raises
    ------
    ConfigError
        If `variables` is empty or `dim` is too small to give each variable at
        least one frequency band.
    """

    def __init__(self, variables: Sequence[str], dim: int, alpha: int = 100) -> None:
        super().__init__()
        self.name = _UNNAMED
        self.variables = tuple(variables)
        if not self.variables:
            raise ConfigError("PositionalEncoder: variables must be a non-empty sequence")
        self.dim = int(dim)
        self.alpha = int(alpha)
        self.per_input_dim = self.dim // (2 * len(self.variables))
        self.last_dim = self.dim % (2 * len(self.variables))
        if self.per_input_dim < 1:
            raise ConfigError(
                f"PositionalEncoder: dim={self.dim} too small for {len(self.variables)} variables "
                f"(per_input_dim = dim // (2*n_vars) = {self.per_input_dim} < 1)"
            )

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        """Encode each coordinate column (in ``variables`` order); concat along the last dim.

        Returns
        -------
        Tensor
            The positional encoding ``[..., dim]``.
        """
        encodings: list[Tensor] = []
        for i, var in enumerate(self.variables):
            symmetric = var in _POSENC_SYM_VARS
            encodings.append(self.pos_enc(inputs[..., i], self.per_input_dim, symmetric=symmetric))
        if self.last_dim > 0:
            encodings.append(torch.zeros_like(encodings[0][..., : self.last_dim]))
        return torch.cat(encodings, dim=-1)

    def pos_enc(self, xs: Tensor, dim: int, symmetric: bool = False) -> Tensor:
        """One variable's sin/cos encoding.

        Returns
        -------
        Tensor
            The ``[..., 2*dim]`` encoding for this variable.
        """
        xs = xs.unsqueeze(-1)
        kwargs = {"device": xs.device, "dtype": xs.dtype}
        omegas = self.alpha * torch.logspace(0, 2 / (dim) - 1, dim, 10_000, **kwargs)
        if symmetric:
            p1 = (xs.sin() * omegas).sin()
            p2 = (xs.cos() * omegas).sin()
        else:
            p1 = (xs * omegas).sin()
            p2 = (xs * omegas).cos()
        return torch.cat((p1, p2), dim=-1)
