from abc import ABC
from collections.abc import Mapping
from typing import Any

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
    """Base class for task heads.

    Tasks wrap a dense network, a loss, a target label, and a scalar weight.

    Parameters
    ----------
    name : str
        Arbitrary name of the task, used for logging and inference.
    input_name : str
        Name of the input stream consumed by this task (e.g., ``"jet"``,
        ``"track"``, ``"objects"``).
    dense_config : dict
        Keyword arguments for :class:`salt.models.Dense`, the head producing
        the task outputs.
    loss : nn.Module
        Loss function applied to the head outputs.
    weight : float, optional
        Scalar multiplier for the task loss in the overall objective.
        The default is ``1.0``.
    """

    def __init__(
        self,
        name: str,
        input_name: str,
        dense_config: dict,
        loss: nn.Module,
        weight: float = 1.0,
    ):
        super().__init__()

        self.name = name
        self.input_name = input_name
        self.net = Dense(**dense_config)
        self.loss = loss
        self.weight = weight

    def input_name_mask(self, pad_masks: Mapping) -> Tensor:
        """Build a boolean mask selecting tokens from the configured input stream.

        Parameters
        ----------
        pad_masks : Mapping
            Mapping from stream name to padding mask tensors of shape
            ``[B, L_i]`` for each stream.

        Returns
        -------
        Tensor
            Boolean mask of shape ``[L]`` (concatenated across streams) that is
            ``True`` for positions belonging to ``self.input_name``.
        """
        return torch.cat(
            [
                torch.ones(m.shape[1], device=m.device) * (1 if (t == self.input_name) else 0)
                for t, m in pad_masks.items()
            ],
        ).bool()


class ClassificationTask(TaskBase):
    """Multi-class or binary classification task head.

    Parameters
    ----------
    label : str
        Label name for the task.
    class_names : list[str] | None, optional
        Ordered class names (index-aligned with outputs). If ``None``,
        attempt to infer from the label via :data:`CLASS_NAMES`.
    label_map : Mapping | None, optional
        Mapping to remap integer labels for training (e.g., {0,4,5} → {0,1,2}).
    sample_weight : str | None, optional
        Key of a per-sample weight found in ``labels_dict[self.input_name]``.
        Requires the configured loss to have ``reduction="none"``.
    use_class_dict : bool, optional
        If ``True``, read class weights for the loss from a class dictionary.
    **kwargs
        Forwarded to :class:`TaskBase`.

    Raises
    ------
    ValueError
        If a label map is provided without class names, or if the number of
        outputs does not match the number of classes.
    """

    def __init__(
        self,
        label: str,
        class_names: list[str] | None = None,
        label_map: Mapping | None = None,
        sample_weight: str | None = None,
        use_class_dict: bool = False,
        **kwargs,
    ):
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
        """Return a list of the per-class output field names.

        Returns
        -------
        list[str]
            List of the per-class output field names.
        """
        assert self.class_names is not None
        pxs = [f"{Flavours[c].px}" if c in Flavours else f"p{c}" for c in self.class_names]
        return [f"{self.model_name}_{px}" for px in pxs]

    def apply_sample_weight(self, loss: Tensor, labels_dict: Mapping) -> Tensor:
        """Apply per-sample weights to a loss tensor if configured.

        Parameters
        ----------
        loss : Tensor
            Loss tensor, typically with ``reduction="none"``.
        labels_dict : Mapping
            Labels mapping containing the sample-weight field.

        Returns
        -------
        Tensor
            Weighted mean loss if ``sample_weight`` is set; otherwise the input loss.
        """
        if self.sample_weight is None:
            return loss
        return (loss * labels_dict[self.input_name][self.sample_weight]).mean()

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute logits and classification loss.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[B, L, D]`` or ``[B, D]`` depending on task.
        labels_dict : Mapping
            Mapping providing ground-truth labels (and optional sample weights).
        pad_masks : Mapping | None, optional
            Per-stream padding masks used to select positions and to mask labels.
            The default is ``None``.
        context : Tensor | None, optional
            Optional context passed to the dense head. The default is ``None``.

        Returns
        -------
        Tensor
            Predicted logits of shape ``[B, C]`` or ``[B, L, C]`` depending on task configuration.
        Tensor | None
            Loss tensor if labels are provided; otherwise ``None``.
        """
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
            mapped_labels = torch.clone(labels)
            for k, v in self.label_map.items():
                mapped_labels[labels == k] = v
            labels = mapped_labels

        # use the mask to remove padded values from the loss (ignore_index=-1 is set by default)
        if pad_mask is not None and labels is not None:
            # mask out dodgey labels
            # TODO @npond: remove when is in the samples
            # https://gitlab.cern.ch/atlas/athena/-/merge_requests/60199
            pad_mask = torch.masked_fill(pad_mask, labels == -2, True)

            # update the labels based on the mask (in case not done already)
            labels = torch.masked_fill(labels, pad_mask, -1)

        loss: Tensor | None = None
        if labels is not None:
            if preds.ndim == 3:
                loss = self.loss(preds.permute(0, 2, 1), labels)
            elif isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
                loss = self.loss(preds.squeeze(-1), labels.float())
            else:
                loss = self.loss(preds, labels)
            loss = self.apply_sample_weight(loss, labels_dict)
            loss *= self.weight

        return preds, loss

    def run_inference(self, preds: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Convert logits to probabilities, optionally with padding-aware softmax.

        Parameters
        ----------
        preds : Tensor
            Logits of shape ``[B, C]`` or ``[B, L, C]``.
        pad_mask : Tensor | None, optional
            If provided and ``preds.ndim == 3``, apply a padding-aware softmax.
            The default is ``None``.

        Returns
        -------
        Tensor
            Probabilities with the same leading dimensions as ``preds``.
        """
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            probs = torch.sigmoid(preds)
        elif pad_mask is None:
            assert preds.ndim == 2
            probs = torch.softmax(preds, dim=-1)
        else:
            assert preds.ndim == 3
            probs = masked_softmax(preds, pad_mask.unsqueeze(-1))
        return probs

    def get_h5(self, preds: Tensor, pad_mask: Tensor | None = None) -> np.ndarray:
        """Convert predictions to a structured NumPy array suitable for HDF5.

        Parameters
        ----------
        preds : Tensor
            Logits of shape ``[B, C]`` or ``[B, L, C]``.
        pad_mask : Tensor | None, optional
            Optional padding mask aligned with the sequence dimension. The default is ``None``.

        Returns
        -------
        np.ndarray
            Structured array with one field per class in :attr:`output_names`.
        """
        probs = self.run_inference(preds, pad_mask)
        dtype = np.dtype([(n, "f4") for n in self.output_names])
        return u2s(probs.float().cpu().numpy(), dtype)

    def get_onnx(self, preds: Tensor, **kwargs) -> tuple:
        """Return ONNX-friendly outputs (tuple of per-class probability tensors).

        Parameters
        ----------
        preds : Tensor
            Logits of shape ``[B, C]`` or ``[B, L, C]``.
        **kwargs
            Additional keyword arguments (e.g., ``pad_mask`` for sequence inputs).

        Returns
        -------
        tuple
            Per-class probability tensors, each squeezed along the last dimension.
        """
        probs = self.run_inference(preds, kwargs.get("pad_mask", None))
        return tuple(output.squeeze() for output in torch.split(probs, 1, -1))


class RegressionTaskBase(TaskBase, ABC):
    """Base class for regression tasks with optional target scaling.

    Parameters
    ----------
    targets : list[str] | str
        Regression target name(s).
    scaler : RegressionTargetScaler | None, optional
        Functional scaler for targets. Mutually exclusive with other scaling options.
    target_denominators : list[str] | str | None, optional
        Denominator variable(s) for forming ratios as targets. Mutually exclusive
        with other scaling options.
    norm_params : dict | None, optional
        Mean/std normalization parameters for each target. Mutually exclusive
        with other scaling options. Expected keys: ``"mean"``, ``"std"``.
    custom_output_names : list[str] | str | None, optional
        Optional custom output names overriding the default.
    **kwargs
        Forwarded to :class:`TaskBase`.

    Raises
    ------
    ValueError
        If multiple scaling methods are set simultaneously or if parameter
        counts do not match the number of targets.
    """

    def __init__(
        self,
        targets: list[str] | str,
        scaler: RegressionTargetScaler | None = None,
        target_denominators: list[str] | str | None = None,
        norm_params: dict | None = None,
        custom_output_names: list[str] | str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scaler = scaler
        self.targets = listify(targets)
        self.target_denominators = listify(target_denominators)
        self.custom_output_names = listify(custom_output_names)
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
        """Compute a loss that ignores NaN targets.

        If NaNs are present in ``targets``, the underlying loss should be
        instantiated with ``reduction="none"``. This method masks NaNs and
        returns the mean over valid entries.

        Parameters
        ----------
        preds : Tensor
            Predicted values with the same shape as ``targets``.
        targets : Tensor
            Target values; NaNs are ignored in the mean.
        **kwargs
            Additional keyword arguments forwarded to the loss call (e.g., ``var``).

        Returns
        -------
        Tensor
            Mean loss over non-NaN elements.

        Raises
        ------
        ValueError
            If the resulting loss becomes NaN (e.g., all predictions are NaN).
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

    def get_targets(self, targets_dict: Mapping) -> Tensor | None:
        """Assemble and scale regression targets from a mapping.

        Parameters
        ----------
        targets_dict : Mapping
            Mapping providing target tensors for the configured input stream.

        Returns
        -------
        Tensor | None
            Targets tensor of shape ``[B, R]`` (or ``[B, L, R]`` for queries),
            potentially scaled depending on configuration. ``None`` if no targets.
        """
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
    """Standard regression task head.

    Parameters
    ----------
    scaler : RegressionTargetScaler | None, optional
        Backward-compatibility placeholder; if provided, stored on the instance.
    **kwargs
        Forwarded to :class:`RegressionTaskBase`.

    Raises
    ------
    ValueError
        If the number of outputs does not match the number of regression targets.
    """

    def __init__(self, scaler: RegressionTargetScaler | None = None, **kwargs):
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
        """List of output field names for HDF5/ONNX export."""
        if self.custom_output_names is not None:
            assert len(self.custom_output_names) == len(self.targets)
            return [f"{self.model_name}_{x}" for x in self.custom_output_names]
        return [f"{self.model_name}_{x}" for x in self.targets]

    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute regression predictions and loss.

        Parameters
        ----------
        x : Tensor
            Input tensor (global or query-level).
        targets_dict : Mapping
            Mapping that provides target tensors.
        pad_masks : Mapping | None, optional
            Currently unused for regression unless operating on queries.
        context : Tensor | None, optional
            Optional context tensor. The default is ``None``.

        Returns
        -------
        Tensor
            Predicted values of shape ``[B, R]`` (or ``[B, L, R]`` for queries).
        Tensor | None
            Loss tensor if targets are provided; otherwise ``None``.

        Raises
        ------
        NotImplementedError
            If called on non-supported shapes/inputs.
        """
        if (x.ndim != 2 or pad_masks is not None) and self.input_name != "objects":
            raise NotImplementedError(
                "Regression tasks are currently only supported for global object "
                "and object query level predictions."
            )
        if pad_masks is not None:
            pass
        preds = self.net(x, context)
        targets = self.get_targets(targets_dict)

        loss: Tensor | None = None
        if targets is not None:
            loss = self.nan_loss(preds, targets) * self.weight

        return preds, loss

    def run_inference(self, preds: Tensor, labels: Tensors | None = None) -> Tensor:
        """Invert target scaling to obtain values in the original space.

        Parameters
        ----------
        preds : Tensor
            Predicted values.
        labels : Tensors | None, optional
            Optional labels used to re-apply denominators when configured.

        Returns
        -------
        Tensor
            De-scaled predictions in the original target space.
        """
        preds = preds.float()
        if self.target_denominators is not None and labels is not None:
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
        """Convert predictions to a structured NumPy array suitable for HDF5.

        Parameters
        ----------
        preds : Tensor
            Predictions as Tensor.
        labels : Tensors
            Labels as Tensors.

        Returns
        -------
        np.ndarray
            Predictions as np.ndarray.
        """
        preds = self.run_inference(preds, labels)
        dtype = np.dtype([(x, "f4") for x in self.output_names])
        return u2s(preds.float().cpu().numpy(), dtype)

    def get_onnx(self, preds: Tensor, **kwargs) -> tuple:
        """Return ONNX-friendly outputs (tuple of per-target tensors).

        Parameters
        ----------
        preds : Tensor
            Predictions as Tensor.
        **kwargs
            Additional keyword arguments (e.g., ``labels`` for de-scaling).

        Returns
        -------
        tuple
            Tuple of per-target tensors.
        """
        means = self.run_inference(preds, kwargs.get("labels", None))
        return tuple(output.squeeze() for output in torch.split(means, 1, -1))


class GaussianRegressionTask(RegressionTaskBase):
    """Gaussian regression task head (predicts mean and variance).

    The head outputs ``2 * len(targets)`` values per example (means and
    variances). The loss is the negative log-likelihood under a Gaussian.

    Parameters
    ----------
    **kwargs : Any
        Forwarded to :class:`RegressionTaskBase`.

    Raises
    ------
    ValueError
        If the number of outputs is not ``2 * len(targets)``.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.net.output_size != 2 * len(self.targets):
            raise ValueError(
                f"{self.name}: "
                f"Number of targets ({len(self.targets)}) is not twice the "
                f"number of outputs ({self.net.output_size})"
            )

    @property
    def output_names(self) -> list[str]:
        """List of output field names including stddev fields."""
        outputs = [*self.targets, *[f"{x}_stddev" for x in self.targets]]
        return [f"{self.model_name}_{x}" for x in outputs]

    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute mean/variance predictions and Gaussian NLL loss.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[B, D]``.
        targets_dict : Mapping
            Mapping providing regression targets.
        pad_masks : Mapping | None, optional
            Not supported; if provided for non-query global predictions, raises.
        context : Tensor | None, optional
            Optional context tensor. The default is ``None``.

        Returns
        -------
        Tensor
            Concatenated means and variances of shape ``[B, 2R]``.
        Tensor | None
            Loss tensor if targets are provided; otherwise ``None``.

        Raises
        ------
        NotImplementedError
            If called with unsupported shapes/inputs.
        """
        if x.ndim != 2 or pad_masks is not None:
            raise NotImplementedError(
                "Regression tasks are currently only supported for global object level predictions."
            )

        preds = self.net(x, context)
        targets = self.get_targets(targets_dict)

        # split outputs into means and sigmas
        means, variances = preds.tensor_split(2, -1)
        variances = nn.functional.softplus(variances)  # ensure positiveness of variance

        loss: Tensor | None = None
        if targets is not None:
            loss = self.nan_loss(means, targets, var=variances) * self.weight

        return preds, loss

    def run_inference(self, preds: Tensor, labels: Tensors | None = None) -> tuple[Tensor, Tensor]:
        """Invert scaling for means and (sqrt of) variances to stddevs.

        Parameters
        ----------
        preds : Tensor
            Concatenated means/variances of shape ``[B, 2R]``.
        labels : Tensors | None, optional
            Optional labels for denominator-based scaling.

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple of de-scaled ``(means, stds)`` each of shape ``[B, R]``.

        Raises
        ------
        ValueError
            If called without the necessary scaling parameters.
        """
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

    def get_h5(self, preds: Tensor, labels: Tensors | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Convert Gaussian outputs to structured NumPy arrays (means and stddevs).

        Parameters
        ----------
        preds : Tensor
            Predictions as Tensor.
        labels : Tensors | None, optional
            Labels as Tensors, by default None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple with the means and the stds.
        """
        means, stds = self.run_inference(preds, labels)
        mean_dtype = np.dtype([(f"{self.name}_{t}", "f4") for t in self.targets])
        stds_dtype = np.dtype([(f"{self.name}_{t}_stddev", "f4") for t in self.targets])
        means = u2s(means.float().cpu().numpy(), mean_dtype)
        stds = u2s(stds.float().cpu().numpy(), stds_dtype)
        return means, stds

    def get_onnx(self, preds: Tensor, **kwargs) -> tuple:
        """Return ONNX-friendly outputs: ``(means, stds)`` as squeezed tensors.

        Parameters
        ----------
        preds : Tensor
            Predictions as Tensor.
        **kwargs
            Additional keyword arguments (e.g., ``labels`` for de-scaling).

        Returns
        -------
        tuple
            Tuple of the means and the stds.
        """
        # This might need to be fixed to run inference correctly with denominators
        means, stds = self.run_inference(preds, kwargs.get("labels", None))
        return means.squeeze(), stds.squeeze()


class VertexingTask(TaskBase):
    """Edge classification task for vertexing.

    Parameters
    ----------
    label : str
        Label name for the target object IDs.
    **kwargs
        Forwarded to :class:`TaskBase`.
    """

    def __init__(self, label: str, **kwargs):
        super().__init__(**kwargs)
        self.label = label

    def forward(
        self,
        x: Tensor,
        labels_dict: Mapping,
        pad_masks: Tensor | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute pair classification for vertexing and its loss.

        Parameters
        ----------
        x : Tensor
            Node embeddings of shape ``[B, L, D]``.
        labels_dict : Mapping
            Mapping providing per-node integer labels for the configured stream.
        pad_masks : Tensor | None, optional
            Boolean padding mask of shape ``[B, L]``. The default is ``None``.
        context : Tensor | None, optional
            Optional global context tensor of shape ``[B, C]``. The default is ``None``.

        Returns
        -------
        Tensor
            Predicted edge logits of shape ``[E, 1]`` after compression by the adjacency mask.
        Tensor | None
            Scalar loss if labels are available; otherwise ``None``.
        """
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
        loss: Tensor | None = None
        if labels_dict:
            loss = self.calculate_loss(pred, labels_dict, adjmat=adjmat[:, :-1, :-1])

        return pred, loss

    def calculate_loss(self, pred: Tensor, labels_dict: Mapping, adjmat: Tensor) -> Tensor:
        """Compute the vertexing loss against pairwise matching labels.

        Parameters
        ----------
        pred : Tensor
            Predicted edge logits of shape ``[E, 1]``.
        labels_dict : Mapping
            Mapping containing per-node labels under ``self.label``.
        adjmat : Tensor
            Boolean adjacency mask of shape ``[B, L, L]`` (no self-edges).

        Returns
        -------
        Tensor
            Weighted average loss scaled by :attr:`weight`.
        """
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

    def get_weights(self, labels: Tensor, adjmat: Tensor) -> Tensor:
        """Compute per-edge weights based on origin labels.

        Parameters
        ----------
        labels : Tensor
            Per-node origin labels of shape ``[B, L]``.
        adjmat : Tensor
            Boolean adjacency mask of shape ``[B, L, L]``.

        Returns
        -------
        Tensor
            Per-edge weights of shape ``[E]`` after compression by ``adjmat``.
        """
        weights = torch.clip(sum(labels == i for i in (3, 4, 5)), 0, 1) - (labels == 1).int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        return 1 + weights

    def run_inference(self, preds: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Return per-node assignments from edge predictions.

        Parameters
        ----------
        preds : Tensor
            Edge prediction logits.
        pad_mask : Tensor | None, optional
            Optional padding mask for nodes. The default is ``None``.

        Returns
        -------
        Tensor
            Flattened per-node assignments with paddings filled to ``-inf`` via
            :func:`mask_fill_flattened`.
        """
        preds = get_node_assignment_jit(preds, pad_mask)
        return mask_fill_flattened(preds, pad_mask)

    def get_h5(self, preds: Tensor, pad_mask: Tensor | None = None) -> np.ndarray:
        """Convert vertex assignments to a structured NumPy array for HDF5.

        Parameters
        ----------
        preds : Tensor
            Predictions as Tensor.
        pad_mask : Tensor | None, optional
            Mask with the zero-padding, by default None.

        Returns
        -------
        np.ndarray
            Structured np.ndarray with the vertex assignments.
        """
        preds = self.run_inference(preds, pad_mask)
        dtype = np.dtype([("VertexIndex", "i8")])
        return u2s(preds.int().cpu().numpy(), dtype)


# convert flattened array to shape of mask (ntracks, ...) -> (njets, maxtracks, ...)
@torch.jit.script
def mask_fill_flattened(flat_array: Tensor, mask: Tensor) -> Tensor:
    """Unflatten a per-node array back to a batch-shaped tensor using a mask.

    Parameters
    ----------
    flat_array : Tensor
        Tensor of shape ``[N, F]`` with concatenated (valid) per-node values.
    mask : Tensor
        Boolean mask of shape ``[B, L]`` where valid (non-padded) positions are ``False``.

    Returns
    -------
    Tensor
        Filled tensor of shape ``[B, L, F]`` where padded positions are set to ``-inf``.
    """
    filled = torch.full((mask.shape[0], mask.shape[1], flat_array.shape[1]), float("-inf"))
    mask = mask.to(torch.bool)
    start_index = end_index = 0

    for i in range(mask.shape[0]):
        if mask[i].shape[0] > 0:
            end_index += (~mask[i]).to(torch.long).sum()
            filled[i, : end_index - start_index] = flat_array[start_index:end_index]
            start_index = end_index

    return filled
