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


class ClassificationTask(Task):
    def forward(self, x, labels, mask=None):
        preds = self.net(x)
        if mask is not None:
            preds = preds[~mask]
            if labels is not None:
                labels = labels[~mask]

        loss = None
        if labels is not None:
            loss = self.loss(preds, labels) * self.weight

        return preds, loss


class RegressionTask(Task):
    """Gaussian regression tasks.

    Applies softplus activation to sigmas to ensure positivty.
    """

    def forward(self, x, labels, mask=None):
        if x.ndim != 2 or mask is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level"
                " predictions."
            )

        preds = self.net(x)

        # split outputs into means and sigmas
        assert preds.shape[-1] % 2 == 0
        means, sigmas = preds.tensor_split(2, -1)
        sigmas = nn.functional.softplus(sigmas)  # enforce positive variance

        loss = None
        if labels is not None:
            loss = self.loss(means, labels, var=sigmas) * self.weight

        return preds, loss
