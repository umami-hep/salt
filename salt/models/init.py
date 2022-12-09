from torch import nn

from salt.models import Dense


class InitNet(nn.Module):
    def __init__(
        self,
        name: str,
        net: Dense,
    ):
        """Initialiser network. Just a named dense network.

        Parameters
        ----------
        Name : str
            Name of the input.
        net : Dense
            Dense network for performing the the initial embedding.
        """
        super().__init__()

        self.name = name
        self.net = net

    def forward(self, inputs: dict):
        return self.net(inputs[self.name])
