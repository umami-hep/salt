from collections.abc import Mapping

import torch
from torch import Tensor, nn

from salt.models import Dense


class Task(nn.Module):
    def __init__(
        self,
        name: str,
        input_type: str,
        label: str,
        net: Dense,
        loss: nn.Module,
        weight: float = 1.0,
        label_map: Mapping = None,
        label_denominator: str = None,
    ):
        """Task head.

        A wrapper around a dense network, a loss function, a label and a weight.

        Parameters
        ----------
        name : str
            Name of the task
        input_type : str
            Which type of object is input to the task e.g. jet/track/flow
        label : str
            Label name for the task
        net : Dense
            Dense network for performing the task
        loss : nn.Module
            Task loss
        weight : float
            Weight in the overall loss
        label_map : Mapping
            Remap integer labels for training (e.g. 0,4,5 -> 0,1,2)
        label_denominator : str
            Name of the denominator label for the task (regression only)
        """
        super().__init__()

        self.name = name
        self.input_type = input_type
        self.label = label
        self.net = net
        self.loss = loss
        self.weight = weight
        self.label_map = label_map
        self.label_denominator = label_denominator
        assert not (self.label_map and self.label_denominator)

    def input_type_mask(self, masks):
        return torch.cat(
            [torch.ones(m.shape[1]) * (t == self.input_type) for t, m in masks.items()]
        ).bool()


class ClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.label_denominator is None

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        masks: Mapping = None,
        context: Tensor = None,
    ):
        if masks is not None:
            input_type_mask = self.input_type_mask(masks)
            preds = self.net(x[:, input_type_mask], context)
            mask = masks[self.input_type]
        else:
            preds = self.net(x, context)
            mask = None

        # get labels
        labels = labels_dict[self.name] if labels_dict else None
        if labels is not None and self.label_map is not None:
            for k, v in self.label_map.items():
                labels[labels == k] = v

        # could use ignore_index instead of the mask here
        # TODO remove when https://gitlab.cern.ch/atlas/athena/-/merge_requests/60199
        # is in the samples
        if mask is not None:
            if labels is not None:
                mask = torch.masked_fill(mask, labels == -2, 1)
            preds = preds[~mask]
            if labels is not None:
                labels = labels[~mask]

        loss = None
        if labels is not None:
            loss = self.loss(preds, labels) * self.weight

        return preds, loss


class RegressionTask(Task):
    """Regression task without uncertainty prediction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.label_map is None

    def forward(
        self, x: Tensor, labels_dict: Mapping, masks: Mapping = None, context: Tensor = None
    ):
        if x.ndim != 2 or masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
            )

        preds = self.net(x, context).squeeze(-1)
        labels = labels_dict[self.name] if labels_dict else None
        if labels_dict and f"{self.name}_denominator" in labels_dict:
            labels = torch.div(labels_dict[self.name], labels_dict[f"{self.name}_denominator"])

        loss = None
        if labels is not None:
            loss = self.loss(preds, labels) * self.weight

        return preds, loss


class GaussianRegressionTask(Task):
    """Gaussian regression task, enabling uncertainty prediction.

    Applies softplus activation to sigmas to ensure positivty.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.label_map is None

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        masks: Mapping = None,
        context: Tensor = None,
    ):
        if x.ndim != 2 or masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
            )

        preds = self.net(x, context)
        labels = labels_dict[self.name] if labels_dict else None
        if labels_dict and f"{self.name}_denominator" in labels_dict:
            labels = torch.div(labels_dict[self.name], labels_dict[f"{self.name}_denominator"])

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.label_map is None

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        masks: Tensor = None,
        context: Tensor = None,
    ):
        if masks is not None:
            input_type_mask = self.input_type_mask(masks)
            mask = masks[self.input_type]
            x = x[:, input_type_mask]
        else:
            mask = None
        b, n, d = x.shape
        ex_size = (b, n, n, d)
        t_mask = torch.ones(b, n) if mask is None else ~mask
        t_mask = torch.cat([t_mask, torch.zeros(b, 1)], dim=1)  # pad t_mask for onnx compatibility
        adjmat = t_mask.unsqueeze(-1) * t_mask.unsqueeze(-2)
        adjmat = adjmat.bool() & ~torch.eye(n + 1, n + 1).repeat(b, 1, 1).bool().to(adjmat.device)

        # Deal with context
        context_matrix = None
        if context is not None:
            context_d = context.shape[-1]
            context = context.unsqueeze(1).expand(b, n, context_d)
            context_matrix = torch.zeros(
                (adjmat.sum(), 2 * context_d), device=x.device, dtype=x.dtype
            )
            context_matrix = context.unsqueeze(-2).expand((b, n, n, context_d))[adjmat[:, :-1, :-1]]

        # Create the track-track matrix as a compressed tensor
        tt_matrix = torch.zeros((adjmat.sum(), d * 2), device=x.device, dtype=x.dtype)
        tt_matrix[:, :d] = x.unsqueeze(-2).expand(ex_size)[adjmat[:, :-1, :-1]]
        tt_matrix[:, d:] = x.unsqueeze(-3).expand(ex_size)[adjmat[:, :-1, :-1]]
        pred = self.net(tt_matrix, context_matrix)
        loss = None
        if labels_dict:
            loss = self.calculate_loss(pred, labels_dict, adjmat=adjmat[:, :-1, :-1])

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
        weights = self.get_weights(labels_dict[f"{self.input_type}_origin"], adjmat)
        loss = (loss * weights).mean()

        return loss * self.weight

    def get_weights(self, labels, adjmat):
        weights = torch.clip(sum(labels == i for i in (3, 4, 5)), 0, 1) - (labels == 1).int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        weights = 1 + weights
        return weights
