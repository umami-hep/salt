from torch import Tensor, nn

from salt.models import Dense


class Task(nn.Module):
    def __init__(
        self,
        name: str,
        label: str,
        net: Dense,
        loss: nn.Module,
        weight: float = 1.0,
    ):
        """Task head.

        A wrapper around a dense network, a loss function, a label and a weight.

        Parameters
        ----------
        Name : str
            Name of the task
        lbael : str
            Label name for the task
        net : Dense
            Dense network for performing the task
        loss : nn.Module
            Task loss
        weight : float
            Weight in the overall loss
        """
        super().__init__()

        self.name = name
        self.label = label
        self.net = net
        self.loss = loss
        self.weight = weight


class ClassificationTask(Task):
    def forward(self, x: Tensor, labels: Tensor, mask: Tensor = None):
        preds = self.net(x)

        # could use ignore_index instead of the mask here
        if mask is not None:
            preds = preds[~mask]
            if labels is not None:
                labels = labels[~mask]

        loss = None
        if labels is not None:
            loss = self.loss(preds, labels) * self.weight

        return preds, loss


class RegressionTask(Task):
    """Gaussian regression task.

    Applies softplus activation to sigmas to ensure positivty.
    """

    def forward(self, x: Tensor, labels: Tensor, mask: Tensor = None):
        if x.ndim != 2 or mask is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
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
