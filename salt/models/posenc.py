import torch
from torch import Tensor, nn

SYM_VARS = {"phi"}


class PositionalEncoder(nn.Module):
    def __init__(self, variables: list[str], dim: int, alpha: int = 100):
        """Positional encoder.

        Evenly share the embedding space between the different variables to be encoded.
        Any remaining dimensions are left as zeros.

        TODO: alpha should be set for each variable

        Parameters
        ----------
        variables : list[str]
            List of variables to apply the positional encoding to.
        dim : int
            Dimension of the positional encoding.
        alpha : int, optional
            Scaling factor for the positional encoding, by default 100.
        """
        super().__init__()
        self.variables = variables
        self.dim = dim
        self.alpha = alpha
        print(dim, len(self.variables))
        self.per_input_dim = self.dim // (2 * len(self.variables))
        self.last_dim = self.dim % (2 * len(self.variables))
        print(self.per_input_dim, self.last_dim)

    @torch.no_grad()
    def forward(self, inputs: Tensor):
        """Apply positional encoding to the inputs.

        Parameters
        ----------
        inputs : Tensor
            Input tensor for coordinates to be encoded.

        Returns
        -------
        Tensor
            Positional encoding of the input variables.
        """
        encodings = []
        for i, var in enumerate(self.variables):
            kwargs = {"symmetric": var in SYM_VARS}
            encodings.append(self.pos_enc(inputs[..., i], self.per_input_dim, **kwargs))

        if self.last_dim > 0:
            encodings.append(torch.zeros_like(encodings[0][..., : self.last_dim]))
        print([e.shape for e in encodings])
        return torch.cat(encodings, dim=-1)

    def pos_enc(self, xs: Tensor, dim: int, symmetric: bool = False) -> Tensor:
        """Positional encoding.

        Parameters
        ----------
        xs : Tensor
            Input tensor.
        dim : int
            Dimension of the positional encoding.
        symmetric : bool, optional
            Whether to use symmetric encoding, by default False.

        Returns
        -------
        Tensor
            Positional encoding.
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
