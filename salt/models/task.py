from typing import Mapping

import torch
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
        label : str
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
    def forward(
        self, x: Tensor, labels_dict: Mapping, mask: Tensor = None, context: Tensor = None
    ):
        preds = self.net(x, context)
        labels = labels_dict[self.name] if labels_dict else None

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

    def forward(
        self, x: Tensor, labels_dict: Mapping, mask: Tensor = None, context: Tensor = None
    ):
        if x.ndim != 2 or mask is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
            )

        preds = self.net(x, context)
        labels = labels_dict[self.name] if labels_dict else None

        # split outputs into means and sigmas
        assert preds.shape[-1] % 2 == 0
        means, sigmas = preds.tensor_split(2, -1)
        sigmas = nn.functional.softplus(sigmas)  # enforce positive variance

        loss = None
        if labels is not None:
            loss = self.loss(means, labels, var=sigmas) * self.weight

        return preds, loss


class VertexingTask(Task):
    """Vertexing task."""

    def forward(
        self, x: Tensor, labels_dict: Mapping, mask: Tensor = None, context: Tensor = None
    ):
        b, n, d = x.shape
        ex_size = (b, n, n, d)
        if mask is None:
            t_mask = torch.ones(b, n)
        else:
            t_mask = ~mask
        adjmat = t_mask.unsqueeze(-1) * t_mask.unsqueeze(-2)
        adjmat = adjmat & ~torch.diag_embed(torch.ones_like(t_mask).bool())
        adjmat = adjmat.bool()

        # Deal with context
        context_matrix = None
        if context is not None:
            context_d = context.shape[-1]
            context = context.unsqueeze(1).expand(b, n, context_d)
            context_matrix = torch.zeros(
                (adjmat.sum(), 2 * context_d), device=x.device, dtype=x.dtype
            )
            context_matrix = context.unsqueeze(-2).expand((b, n, n, context_d))[adjmat]

        # Create the track-track matrix as a compressed tensor
        tt_matrix = torch.zeros((adjmat.sum(), d * 2), device=x.device, dtype=x.dtype)
        tt_matrix[:, :d] = x.unsqueeze(-2).expand(ex_size)[adjmat]
        tt_matrix[:, d:] = x.unsqueeze(-3).expand(ex_size)[adjmat]
        pred = self.net(tt_matrix, context_matrix)
        loss = None
        if labels_dict is not None:
            loss = self.calculate_loss(pred, labels_dict, adjmat=adjmat)

        return pred, loss

    def calculate_loss(self, pred, labels_dict, adjmat):
        labels = labels_dict[self.name]

        match_matrix = labels.unsqueeze(-1) == labels.unsqueeze(-2)

        # Remove matching pairs if either of them come from the negative class
        unique_matrix = labels == -2
        unique_matrix = unique_matrix.unsqueeze(-1) | unique_matrix.unsqueeze(-2)
        match_matrix = match_matrix * ~unique_matrix

        # Compress the matrix using the adjacenty matrix (no self connections)
        match_matrix = match_matrix[adjmat].float()

        # Compare the match_matrix to the vertx predictions using the BCE loss
        loss = self.loss(pred.squeeze(-1), match_matrix)

        # If reduction is none and have weight labels, weight the loss
        weights = self.get_weights(labels_dict["track_classification"], adjmat)
        loss = (loss * weights).mean()

        return loss * self.weight

    def get_weights(self, labels, adjmat):
        weights = torch.clip(sum(labels == i for i in (3, 4, 5)), 0, 1) - (labels == 1).int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        weights = 1 + weights
        return weights
