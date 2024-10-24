from abc import ABC
from collections.abc import Mapping

import numpy as np
import torch
from ftag import Flavours
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from torch import Tensor, nn

from salt.models import Dense
from salt.stypes import Tensors
from salt.utils.array_utils import listify
from salt.utils.class_names import CLASS_NAMES
from salt.utils.scalers import RegressionTargetScaler
from salt.utils.tensor_utils import masked_softmax
from salt.utils.union_find import get_node_assignment_jit


class TaskBase(nn.Module, ABC):
    def __init__(
        self,
        name: str,
        input_name: str,
        dense_config: dict,
        loss: nn.Module,
        weight: float = 1.0,
    ):
        """Task head base class.

        Tasks wrap a dense network, a loss, a label, and a weight.

        Parameters
        ----------
        name : str
            Arbitrary name of the task, used for logging and inference.
        input_name : str
            Which type of object is input to the task e.g. jet/track/flow.
        dense_config : dict
            Keyword arguments for [`salt.models.Dense`][salt.models.Dense],
            the dense network producing the task outputs.
        loss : nn.Module
            Loss function applied to the dense network outputs.
        weight : float
            Weight in the overall loss.
        """
        super().__init__()

        self.name = name
        self.input_name = input_name
        self.net = Dense(**dense_config)
        self.loss = loss
        self.weight = weight

    def input_name_mask(self, pad_masks: Mapping):
        return torch.cat(
            [
                torch.ones(m.shape[1], device=m.device) * (t == self.input_name)
                for t, m in pad_masks.items()
            ],
        ).bool()


class ClassificationTask(TaskBase):
    def __init__(
        self,
        label: str,
        class_names: list[str] | None = None,
        label_map: Mapping | None = None,
        sample_weight: str | None = None,
        use_class_dict: bool = False,
        **kwargs,
    ):
        """Classification task.

        Parameters
        ----------
        label : str
            Label name for the task
        class_names : list[str] | None, optional
            List of class names, ordered by output index. If not specified attempt to
            automatically determine these from the label name.
        label_map : Mapping | None, optional
            Remap integer labels for training (e.g. 0,4,5 -> 0,1,2).
        sample_weight : str | None, optional
            Name of a per sample weighting to apply in the loss function.
        use_class_dict : bool, optional
            If True, read class weights for the loss from the class_dict file.
        **kwargs
            Keyword arguments for [`salt.models.TaskBase`][salt.models.TaskBase].
        """
        super().__init__(**kwargs)
        self.label = label
        self.class_names = class_names
        self.label_map = label_map
        if self.label_map is not None and self.class_names is None:
            raise ValueError("Specify class names when using label_map.")
        if hasattr(self.loss, "ignore_index"):
            self.loss.ignore_index = -1
        self.sample_weight = sample_weight
        if self.sample_weight is not None:
            assert (
                self.loss.reduction == "none"
            ), "Sample weights only supported for reduction='none'"
        if self.class_names is None:
            self.class_names = CLASS_NAMES[self.label]
        if len(self.class_names) != self.net.output_size:
            raise ValueError(
                f"{self.name}: "
                f"Number of outputs ({self.net.output_size}) does not match "
                f"number of class names ({len(self.class_names)}). Class names: {self.class_names}"
            )
        self.use_class_dict = use_class_dict

    @property
    def output_names(self) -> list[str]:
        assert self.class_names is not None
        pxs = [f"{Flavours[c].px}" if c in Flavours else f"p{c}" for c in self.class_names]
        return [f"{self.model_name}_{px}" for px in pxs]

    def apply_sample_weight(self, loss: Tensor, labels_dict: Mapping) -> Tensor:
        """Apply per sample weights, if specified."""
        if self.sample_weight is None:
            return loss
        return (loss * labels_dict[self.input_name][self.sample_weight]).mean()

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor = None,
    ):
        # get predictions and mask
        if pad_masks is not None:
            input_name_mask = self.input_name_mask(pad_masks)
            preds = self.net(x[:, input_name_mask], context)
            pad_mask = pad_masks[self.input_name]
        else:
            preds = self.net(x, context)
            pad_mask = None

        # get labels and remap them if necessary
        labels = labels_dict[self.input_name][self.label] if labels_dict else None
        if labels is not None and self.label_map is not None:
            labels_original = torch.clone(labels)
            for k, v in self.label_map.items():
                labels[labels_original == k] = v

        # use the mask to remove padded values from the loss (ignore_index=-1 is set by default)
        if pad_mask is not None and labels is not None:
            # mask out dodgey labels
            # TODO: remove when is in the samples
            # https://gitlab.cern.ch/atlas/athena/-/merge_requests/60199
            pad_mask = torch.masked_fill(pad_mask, labels == -2, True)

            # update the labels based on the mask (in case not done already)
            labels = torch.masked_fill(labels, pad_mask, -1)

        loss = None
        if labels is not None:
            if preds.ndim == 3:
                loss = self.loss(preds.permute(0, 2, 1), labels)
            else:
                loss = self.loss(preds, labels)
            loss = self.apply_sample_weight(loss, labels_dict)
            loss *= self.weight

        return preds, loss

    def run_inference(self, preds: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        if pad_mask is None:
            assert preds.ndim == 2
            probs = torch.softmax(preds, dim=-1)
        else:
            assert preds.ndim == 3
            probs = masked_softmax(preds, pad_mask.unsqueeze(-1))
        return probs

    def get_h5(self, preds: Tensor, pad_mask: Tensor | None = None) -> np.ndarray:
        probs = self.run_inference(preds, pad_mask)
        dtype = np.dtype([(n, "f4") for n in self.output_names])
        return u2s(probs.float().cpu().numpy(), dtype)

    def get_onnx(self, preds: Tensor, pad_mask: Tensor | None = None) -> tuple:
        probs = self.run_inference(preds, pad_mask)
        return tuple(output.squeeze() for output in torch.split(probs, 1, -1))


class RegressionTaskBase(TaskBase, ABC):
    def __init__(
        self,
        targets: list[str] | str,
        scaler: RegressionTargetScaler | None = None,
        target_denominators: list[str] | str | None = None,
        norm_params: dict | None = None,
        **kwargs,
    ):
        """Base class for regression tasks.

        Parameters
        ----------
        targets : list[str] | str
            Regression target(s).
        scaler : RegressionTargetScaler
            Functional scaler for regression targets
                - cannot be used with other target scaling options.
        target_denominators : list[str] | str | None, optional
            Variables to divide regression target(s) by (i.e. for regressing a ratio).
                - cannot be used with other target scaling options.
        norm_params : dict | None, optional
            Mean and std normalization parameters for each target, used for scaling.
                - cannot be used with other target scaling options..
        **kwargs
            Keyword arguments for [`salt.models.TaskBase`][salt.models.TaskBase].
        """
        super().__init__(**kwargs)
        self.scaler = scaler
        self.targets = listify(targets)
        self.target_denominators = listify(target_denominators)
        if norm_params:
            norm_params["mean"] = listify(norm_params["mean"])
            norm_params["std"] = listify(norm_params["std"])
        self.norm_params = norm_params

        if [scaler, target_denominators, norm_params].count(None) not in {2, 3}:
            raise ValueError("Can only use a single scaling method")

        if self.scaler:
            for target in self.targets:
                self.scaler.scale(target, torch.Tensor(1))
        if self.target_denominators and len(self.targets) != len(self.target_denominators):
            raise ValueError(
                f"{self.name}: "
                f"Number of targets ({len(self.targets)}) does not match "
                f"number of target denominators ({len(self.target_denominators)})"
            )
        if self.norm_params and len(self.norm_params["mean"]) != len(self.targets):
            raise ValueError(
                f"{self.name}: "
                f"Number of means in norm_params ({len(self.norm_params['mean'])}) does not match "
                f"number of targets ({len(self.targets)})"
            )
        if self.norm_params and len(self.norm_params["std"]) != len(self.targets):
            raise ValueError(
                f"{self.name}: "
                f"Number of stds in norm_params ({len(self.norm_params['std'])}) does not match "
                f"number of targets ({len(self.targets)})"
            )

    def nan_loss(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Calculates the loss function, and excludes any NaNs.

        If Nans are included in the targets, then the loss should be instansiated
        with the `reduction="none"` option, and this function will take the mean
        excluding any nans.
        """
        invalid = torch.isnan(targets)
        preds = torch.where(invalid, torch.zeros_like(preds), preds)
        targets = torch.where(invalid, torch.zeros_like(targets), targets)

        if "var" in kwargs:
            kwargs["var"] = torch.where(invalid, torch.zeros_like(kwargs["var"]), kwargs["var"])

        loss = self.loss(preds, targets, **kwargs)

        if len(loss.shape) == 0:
            if torch.isnan(loss):
                raise ValueError(
                    "Regression loss is NaN. This may be due to NaN targets,"
                    " check configs/nan_regression.yaml for options to deal with this."
                )
            return loss

        nanmean = torch.nanmean(loss)
        if torch.isnan(nanmean):
            raise ValueError("NanRegression is NaN. This means all model predictions are NaN")
        return nanmean

    def get_targets(self, targets_dict: Mapping):
        targets = None
        if targets_dict:
            targets = torch.stack(
                [targets_dict[self.input_name][target] for target in self.targets], dim=1
            )

        if targets is not None:
            if self.scaler is not None:
                for i in range(len(self.targets)):
                    targets[:, i] = self.scaler.scale(self.targets[i], targets[:, i])
            if self.target_denominators is not None:
                for i in range(len(self.targets)):
                    targets[:, i] = torch.div(
                        targets[:, i], targets_dict[self.input_name][self.target_denominators[i]]
                    )
            if self.norm_params is not None:
                for i in range(len(self.norm_params["mean"])):
                    targets[:, i] = (targets[:, i] - self.norm_params["mean"][i]) / (
                        self.norm_params["std"][i]
                    )

            # We stack targets dict always over the first dimension to allow consistency
            # when scaling, but for queries we want the regression target to be in the final
            # dimension. This allows us to keep the same code for both global and query scaling
            if len(targets.shape) == 3:
                targets = targets.transpose(1, 2)
        return targets


class RegressionTask(RegressionTaskBase):
    def __init__(self, scaler=None, **kwargs):
        """Regression task.

        Parameters
        ----------
        scaler
            dummy text
        **kwargs
            Keyword arguments for
            [`salt.models.RegressionTaskBase`][salt.models.RegressionTaskBase].
        """
        super().__init__(**kwargs)
        if self.net.output_size != len(self.targets):
            raise ValueError(
                f"{self.name}: "
                f"Number of outputs ({self.net.output_size}) does not match "
                f"number of targets ({len(self.targets)})"
            )
        self.scaler = scaler

    @property
    def output_names(self) -> list[str]:
        return [f"{self.model_name}_{self.name}_{x}" for x in self.targets]

    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor = None,
    ):
        if (x.ndim != 2 or pad_masks is not None) and self.input_name != "objects":
            raise NotImplementedError(
                "Regression tasks are currently only supported for global object "
                "and object query level predictions."
            )
        if pad_masks is not None:
            pass
        preds = self.net(x, context)
        targets = self.get_targets(targets_dict)

        loss = None
        if targets is not None:
            loss = self.nan_loss(preds, targets) * self.weight

        return preds, loss

    def run_inference(self, preds: Tensor, labels: Tensors) -> Tensor:
        if self.target_denominators is not None:
            for i in range(len(self.targets)):
                preds[:, i] *= labels[self.input_name][self.target_denominators[i]]
        elif self.norm_params is not None:
            for i in range(len(self.norm_params["mean"])):
                preds[:, i] *= self.norm_params["std"][i]
                preds[:, i] += self.norm_params["mean"][i]
        elif self.scaler is not None:
            for i in range(len(self.targets)):
                preds[:, :, i] = self.scaler.inverse(self.targets[i], preds[:, :, i])
        return preds

    def get_h5(self, preds: Tensor, labels: Tensors) -> np.ndarray:
        preds = self.run_inference(preds, labels)
        dtype = np.dtype([(x, "f4") for x in self.output_names])
        return u2s(preds.float().cpu().numpy(), dtype)


class GaussianRegressionTask(RegressionTaskBase):
    def __init__(self, **kwargs):
        """Regression task that outputs a mean and variance for each target.
        The loss function is the negative log likelihood of a Gaussian distribution.

        Parameters
        ----------
        **kwargs
            Keyword arguments for
            [`salt.models.RegressionTaskBase`][salt.models.RegressionTaskBase].
        """
        super().__init__(**kwargs)
        if self.net.output_size != 2 * len(self.targets):
            raise ValueError(
                f"{self.name}: "
                f"Number of targets ({len(self.targets)}) is not twice the "
                f"number of outputs ({self.net.output_size})"
            )

    @property
    def output_names(self) -> list[str]:
        outputs = [*self.targets, *[f"{x}_stddev" for x in self.targets]]
        return [f"{self.model_name}_{x}" for x in outputs]

    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor = None,
    ):
        if x.ndim != 2 or pad_masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for global object level predictions."
            )

        preds = self.net(x, context)
        targets = self.get_targets(targets_dict)

        # split outputs into means and sigmas
        means, variances = preds.tensor_split(2, -1)
        variances = nn.functional.softplus(variances)  # ensure positiveness of variance

        loss = None
        if targets is not None:
            loss = self.nan_loss(means, targets, var=variances) * self.weight

        return preds, loss

    def run_inference(self, preds: Tensor, labels: Tensors | None = None) -> tuple[Tensor, Tensor]:
        if self.target_denominators is not None and labels is not None:
            for i in range(len(self.targets)):
                preds[:, i] *= labels[self.input_name][self.target_denominators[i]]
                preds[:, i + 1] *= labels[self.input_name][self.target_denominators[i]]
        elif self.norm_params is not None:
            for i in range(len(self.norm_params["mean"])):
                preds[:, i] *= self.norm_params["std"][i]
                preds[:, i] += self.norm_params["mean"][i]
                # return stddev as sqrt(var)
                preds[:, i + 1] = (
                    torch.sqrt(nn.functional.softplus(preds[:, i + 1])) * self.norm_params["std"][i]
                )
        else:
            raise ValueError("Inference for Gaussian regression requires scaling parameters.")
        means, stds = preds.tensor_split(2, -1)
        return means, stds

    def get_h5(self, preds: Tensor, labels: Tensors | None = None) -> np.ndarray:
        means, stds = self.run_inference(preds, labels)
        mean_dtype = np.dtype([(f"{self.name}_{t}", "f4") for t in self.targets])
        stds_dtype = np.dtype([(f"{self.name}_{t}_stddev", "f4") for t in self.targets])
        means = u2s(means.float().cpu().numpy(), mean_dtype)
        stds = u2s(stds.float().cpu().numpy(), stds_dtype)
        return means, stds

    def get_onnx(self, preds: Tensor) -> tuple:
        means, stds = self.run_inference(preds)
        return means.squeeze(), stds.squeeze()


class VertexingTask(TaskBase):
    def __init__(self, label: str, **kwargs):
        """Edge classification task for vertexing.

        Parameters
        ----------
        label : str
            Label name for the target object IDs.
        **kwargs
            Keyword arguments for [`salt.models.TaskBase`][salt.models.TaskBase].
        """
        super().__init__(**kwargs)
        self.label = label

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        pad_masks: Tensor = None,
        context: Tensor = None,
    ):
        if pad_masks is not None:
            input_name_mask = self.input_name_mask(pad_masks)
            mask = pad_masks[self.input_name]
            x = x[:, input_name_mask]
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
        labels = labels_dict[self.input_name][self.label]

        match_matrix = labels.unsqueeze(-1) == labels.unsqueeze(-2)

        # Remove matching pairs if either of them come from the negative class
        unique_matrix = labels < 0
        unique_matrix = unique_matrix.unsqueeze(-1) | unique_matrix.unsqueeze(-2)
        match_matrix *= ~unique_matrix

        # Compress the matrix using the adjacenty matrix (no self connections)
        match_matrix = match_matrix[adjmat].float()

        # Compare the match_matrix to the vertx predictions using the BCE loss
        loss = self.loss(pred.squeeze(-1), match_matrix)

        # If reduction is none and have weight labels, weight the loss
        origin_label = self.label.replace("VertexIndex", "OriginLabel")
        weights = self.get_weights(labels_dict[self.input_name][origin_label], adjmat)
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

    def run_inference(self, preds: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        preds = get_node_assignment_jit(preds, pad_mask)
        return mask_fill_flattened(preds, pad_mask)

    def get_h5(self, preds: Tensor, pad_mask: Tensor | None = None) -> np.ndarray:
        preds = self.run_inference(preds, pad_mask)
        dtype = np.dtype([("VertexIndex", "i8")])
        return u2s(preds.int().cpu().numpy(), dtype)


# convert flattened array to shape of mask (ntracks, ...) -> (njets, maxtracks, ...)
@torch.jit.script
def mask_fill_flattened(flat_array, mask):
    filled = torch.full((mask.shape[0], mask.shape[1], flat_array.shape[1]), float("-inf"))
    mask = mask.to(torch.bool)
    start_index = end_index = 0

    for i in range(mask.shape[0]):
        if mask[i].shape[0] > 0:
            end_index += (~mask[i]).to(torch.long).sum()
            filled[i, : end_index - start_index] = flat_array[start_index:end_index]
            start_index = end_index

    return filled
