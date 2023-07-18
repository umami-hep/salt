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
        """
        super().__init__()

        self.name = name
        self.input_type = input_type
        self.label = label
        self.net = net
        self.loss = loss
        self.weight = weight

    def input_type_mask(self, masks):
        return torch.cat(
            [torch.ones(m.shape[1]) * (t == self.input_type) for t, m in masks.items()]
        ).bool()


class ClassificationTask(Task):
    def __init__(self, label_map: Mapping | None = None, **kwargs):
        """Classification task.

        Parameters
        ----------
        label_map : Mapping | None, optional
            Remap integer labels for training (e.g. 0,4,5 -> 0,1,2), by default None
        **kwargs
            Keyword arguments for Task
        """
        super().__init__(**kwargs)
        self.label_map = label_map

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        masks: Mapping | None = None,
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

    def run_inference(self, preds: Tensor, mask: Tensor | None = None):
        if mask is None:
            assert preds.ndim == 2
            probs = torch.softmax(preds, dim=-1)

        else:
            assert preds.ndim == 2
            probs = torch.softmax(preds, dim=-1)

            # the preds are a flat tensor of only valid tracks
            # add back in the track dimension using the mask
            out = torch.full((*mask.shape[:2], preds.shape[1]), torch.nan, device=preds.device)
            out[~mask] = probs
            probs = out
        return probs


class RegressionTaskBase(Task):
    def __init__(
        self, label_denominator: str | None = None, norm_params: dict | None = None, **kwargs
    ):
        """Base class for regression tasks.

        Parameters
        ----------
        label_denominator : str | None, optional
            Name of the denominator label for the task, by default None
        norm_params : dict | None, optional
            Normalization parameters for the task, by default None
        **kwargs
            Keyword arguments for Task
        """
        super().__init__(**kwargs)
        self.label_denominator = label_denominator
        self.norm_params = norm_params
        if self.label_denominator is not None and self.norm_params is not None:
            raise ValueError("Cannot use label_denominator and norm_params at the same time.")
        self.denominator_key = f"{self.name}_denominator"

    def get_labels(self, labels_dict: Mapping):
        labels = labels_dict[self.name] if labels_dict else None
        if labels is not None:
            if self.label_denominator in labels_dict:
                labels = torch.div(labels_dict[self.name], labels_dict[self.denominator_key])
            elif self.norm_params is not None:
                labels = (labels - self.norm_params["mean"]) / self.norm_params["std"]
        return labels

    def run_inference(self, preds: Tensor, labels_dict: Mapping):
        if self.label_denominator is not None:
            preds = preds * labels_dict[self.denominator_key]
        elif self.norm_params is not None:
            preds = preds * self.norm_params["std"] + self.norm_params["mean"]
        return preds


class RegressionTask(RegressionTaskBase):
    def forward(
        self, x: Tensor, labels_dict: Mapping, masks: Mapping | None = None, context: Tensor = None
    ):
        if x.ndim != 2 or masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
            )

        preds = self.net(x, context).squeeze(-1)
        labels = self.get_labels(labels_dict)

        loss = None
        if labels is not None:
            loss = self.loss(preds, labels) * self.weight

        return preds, loss


class GaussianRegressionTask(RegressionTaskBase):
    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        masks: Mapping | None = None,
        context: Tensor = None,
    ):
        if x.ndim != 2 or masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
            )

        preds = self.net(x, context)
        labels = self.get_labels(labels_dict)  # type:ignore

        # split outputs into means and sigmas
        assert preds.shape[-1] % 2 == 0
        means, sigmas = preds.tensor_split(2, -1)
        sigmas = nn.functional.softplus(sigmas)  # enforce positive variance

        loss = None
        if labels is not None:
            loss = self.loss(means, labels, var=sigmas) * self.weight

        return preds, loss


class VertexingTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        t_mask = torch.ones(b, n, device=x.device) if mask is None else ~mask
        t_mask = torch.cat(
            [t_mask, torch.zeros(b, 1, device=x.device)], dim=1
        )  # pad t_mask for onnx compatibility
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
        unique_matrix = labels < 0
        unique_matrix = unique_matrix.unsqueeze(-1) | unique_matrix.unsqueeze(-2)
        match_matrix = match_matrix * ~unique_matrix

        # Compress the matrix using the adjacenty matrix (no self connections)
        match_matrix = match_matrix[adjmat].float()

        # Compare the match_matrix to the vertx predictions using the BCE loss
        loss = self.loss(pred.squeeze(-1), match_matrix)

        # If reduction is none and have weight labels, weight the loss
        weights = self.get_weights(labels_dict[f"{self.input_type}_origin"], adjmat)
        weighted_loss = loss * weights

        # Calculate the number of non-masked elements
        num_non_masked_elements = match_matrix.sum()

        # Take average over the non-masked elements
        loss = weighted_loss.sum() / num_non_masked_elements

        return loss * self.weight

    def get_weights(self, labels, adjmat):
        weights = torch.clip(sum(labels == i for i in (3, 4, 5)), 0, 1) - (labels == 1).int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        return 1 + weights
