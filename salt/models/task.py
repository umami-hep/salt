from collections.abc import Mapping

import torch
from torch import Tensor, nn

from salt.models import Dense
from salt.utils.tensor_utils import masked_softmax


class Task(nn.Module):
    def __init__(
        self,
        name: str,
        input_type: str,
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
        self.net = net
        self.loss = loss
        self.weight = weight

    def input_type_mask(self, masks):
        return torch.cat(
            [
                torch.ones(m.shape[1], device=m.device) * (t == self.input_type)
                for t, m in masks.items()
            ],
        ).bool()


class ClassificationTask(Task):
    def __init__(
        self,
        label: str,
        class_names: list[str] | None = None,
        label_map: Mapping | None = None,
        **kwargs,
    ):
        """Classification task.

        Parameters
        ----------
        label : str
            Label name for the task
        class_names : list[str] | None, optional
            List of class names, ordered by output index, by default None
        label_map : Mapping | None, optional
            Remap integer labels for training (e.g. 0,4,5 -> 0,1,2), by default None
        **kwargs
            Keyword arguments for Task
        """
        super().__init__(**kwargs)
        self.label = label
        self.class_names = class_names
        self.label_map = label_map
        if self.label_map is not None and self.class_names is None:
            raise ValueError("Specify class names when using label_map.")
        if hasattr(self.loss, "ignore_index"):
            self.loss.ignore_index = -1

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

        if mask is not None and labels is not None:
            # mask out dodgey labels
            # TODO remove when https://gitlab.cern.ch/atlas/athena/-/merge_requests/60199 is in
            mask = torch.masked_fill(mask, labels == -2, True)

            # update the labels based on the mask (in case not done already)
            labels = torch.masked_fill(labels, mask, -1)

        loss = None
        if labels is not None:
            if preds.ndim == 3:
                loss = self.loss(preds.permute(0, 2, 1), labels) * self.weight
            else:
                loss = self.loss(preds, labels) * self.weight

        return preds, loss

    def run_inference(self, preds: Tensor, mask: Tensor | None = None):
        if mask is None:
            assert preds.ndim == 2
            probs = torch.softmax(preds, dim=-1)
        else:
            assert preds.ndim == 3
            probs = masked_softmax(preds, mask.unsqueeze(-1))
        return probs


class RegressionTaskBase(Task):
    def __init__(
        self,
        targets: list[str] | str,
        target_denominators: list[str] | str | None = None,
        norm_params: dict | None = None,
        **kwargs,
    ):
        """Base class for regression tasks.

        Parameters
        ----------
        targets : list[str] | str
            Target names for the task
        target_denominators : list[str] | str | None, optional
            Name of the target denominator for the task, by default None
        norm_params : dict | None, optional
            Normalization parameters for the task, by default None
        **kwargs
            Keyword arguments for Task
        """
        super().__init__(**kwargs)

        def listify(maybe_list):
            if maybe_list is None:
                return None
            if isinstance(maybe_list, list):
                return maybe_list
            return [maybe_list]

        self.targets = listify(targets)
        self.target_denominators = listify(target_denominators)
        if norm_params:
            norm_params["mean"] = listify(norm_params["mean"])
            norm_params["std"] = listify(norm_params["std"])
        self.norm_params = norm_params
        if self.target_denominators is not None and self.norm_params is not None:
            raise ValueError("Cannot use target_denominators and norm_params at the same time.")
        self.denominator_key = f"{self.name}_denominators"
        if self.target_denominators and len(self.targets) != len(self.target_denominators):
            raise ValueError(
                f"{self.name}: "
                f"Number of targets ({len(self.targets)}) does not match "
                f"number of target denominators ({len(self.target_denominators)})"
            )
        if self.norm_params and len(self.norm_params["mean"]) != self.net.output_size:
            raise ValueError(
                f"{self.name}: "
                f"Number of means in norm_params ({len(self.norm_params['mean'])}) does not match "
                f"number of targets ({len(self.targets)})"
            )
        if self.norm_params and len(self.norm_params["std"]) != self.net.output_size:
            raise ValueError(
                f"{self.name}: "
                f"Number of stds in norm_params ({len(self.norm_params['std'])}) does not match "
                f"number of targets ({len(self.targets)})"
            )

    def nan_loss(self, preds, targets, **kwargs):
        """Calculates the loss function, and excludes any NaNs.
        If Nans are included in the targets, then the loss should be instansiated
        with the `reduction="none"` option, and this function will take the mean
        excluding any nans.
        """
        mask = torch.isnan(targets)
        preds = torch.where(mask, torch.zeros_like(preds), preds)
        targets = torch.where(mask, torch.zeros_like(targets), targets)

        if "var" in kwargs:
            kwargs["var"] = torch.where(mask, torch.zeros_like(kwargs["var"]), kwargs["var"])

        loss = self.loss(preds, targets, **kwargs)

        if len(loss.shape) == 0:
            if torch.isnan(loss):
                raise ValueError(
                    "Regression loss is NaN. This may be due to NaN targets,"
                    + " check configs/nan_regression.yaml for options to deal with this."
                )
            return loss

        nanmean = torch.nanmean(loss)
        if torch.isnan(nanmean):
            raise ValueError("NanRegression is NaN. This means all model predictions are NaN")
        return nanmean

    def get_targets(self, targets_dict: Mapping):
        targets = targets_dict[self.name] if targets_dict else None
        if targets is not None:
            if self.target_denominators is not None:
                targets = torch.div(targets, targets_dict[self.denominator_key])
            if targets.ndim == 1:
                targets = targets.unsqueeze(-1)
            if self.norm_params is not None:
                for i in range(len(self.norm_params["mean"])):
                    targets[:, i] = (targets[:, i] - self.norm_params["mean"][i]) / (
                        self.norm_params["std"][i]
                    )
        return targets

    def run_inference(self, preds: Tensor, targets_dict: Mapping):
        if self.target_denominators is not None:
            if preds.shape[-1] == 1:
                preds = preds * targets_dict[self.denominator_key].unsqueeze(-1)
            else:
                preds = preds * targets_dict[self.denominator_key]
        elif self.norm_params is not None:
            for i in range(len(self.norm_params["mean"])):
                preds[:, i] = preds[:, i] * self.norm_params["std"][i] + self.norm_params["mean"][i]
        return preds


class RegressionTask(RegressionTaskBase):
    def forward(
        self, x: Tensor, targets_dict: Mapping, masks: Mapping | None = None, context: Tensor = None
    ):
        if x.ndim != 2 or masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
            )

        preds = self.net(x, context)
        targets = self.get_targets(targets_dict)

        loss = None
        if targets is not None:
            loss = self.nan_loss(preds, targets) * self.weight

        return preds, loss


class GaussianRegressionTask(RegressionTaskBase):
    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        masks: Mapping | None = None,
        context: Tensor = None,
    ):
        if x.ndim != 2 or masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for jet-level predictions."
            )

        preds = self.net(x, context)
        targets = self.get_targets(targets_dict)  # type:ignore

        # split outputs into means and sigmas
        assert preds.shape[-1] % 2 == 0
        means, sigmas = preds.tensor_split(2, -1)
        sigmas = nn.functional.softplus(sigmas)  # enforce positive variance

        loss = None
        if targets is not None:
            loss = self.nan_loss(means, targets, var=sigmas) * self.weight

        return preds, loss


class VertexingTask(Task):
    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self.label = label

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
        adjmat = (
            adjmat.bool() & ~torch.eye(n + 1, n + 1, device=adjmat.device).repeat(b, 1, 1).bool()
        )

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
