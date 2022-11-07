import torch.nn as nn

from salt.models import Dense


class Task(nn.Module):
    def __init__(
        self,
        name: str,
        net: Dense,
        loss: nn.Module,
        weight: float = 1.0,
    ):
        """Task head.

        A wrapper around a dense network, a loss function, a label and a weight.

        Parameters
        ----------
        Name : str
            Name of this task
        net : Dense
            Dense network for performing this task
        loss : nn.Module
            Task loss
        weight : float
            Weight in the overall loss
        """
        super().__init__()

        self.name = name
        self.weight = weight
        self.net = net
        self.loss = loss

    def forward(self, x, labels):
        preds = self.net(x)
        loss = None
        if labels is not None:
            loss = self.loss(preds, labels) * self.weight
        return preds, loss
