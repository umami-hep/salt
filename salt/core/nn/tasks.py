"""Config-constructed task modules (classification, regression, vertexing).

Each task composes a v1 task head (loss math kept verbatim) at `bind`, declares
its label/mask/context dependencies, and publishes ``preds.<stream>.<task>``
plus ``losses.<task>`` (FIT|VAL only).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.modules import Dense, _reject_width_keys, _stream_len
from salt.core.onnx.config import ExportOutput
from salt.core.onnx.reduces import mask_fill_flattened
from salt.core.outputs.producers import OutputField
from salt.core.utils.array_utils import listify
from salt.core.utils.scalers import RegressionTargetScaler
from salt.core.utils.union_find import get_node_assignment_jit
from salt.core.outputs.names import VERTEX_INDEX, pascal_case

__all__ = ["ClassificationTaskModule", "RegressionTaskModule", "VertexingTaskModule"]

_UNNAMED = "unnamed"
_WIDTH_KEYS = ("input_size", "output_size", "context_size")

# Streams that are a fixed-count query bank (e.g. MaskFormer's `objects`, M
# learnable queries with no padding) rather than a variable-length pad-masked
# sequence. A sequence-mode task on such a stream must NOT require/consume a
# pad mask, or graph validation fails with a ConnectivityError (no module
# produces `masks.objects`).
_NO_PAD_MASK_STREAMS = frozenset({"objects"})

_DEFAULT_CLS_LOSS: dict[str, Any] = {"class_path": "torch.nn.CrossEntropyLoss"}
_DEFAULT_VTX_LOSS: dict[str, Any] = {
    "class_path": "torch.nn.BCEWithLogitsLoss",
    "init_args": {"reduction": "none"},
}
_DEFAULT_REG_LOSS: dict[str, Any] = {"class_path": "torch.nn.MSELoss"}
_DEFAULT_GAUSS_LOSS: dict[str, Any] = {"class_path": "torch.nn.GaussianNLLLoss"}


# ===========================================================================
# Task-head family: math kept verbatim from ``salt.models.task`` (loss,
# ignore_index=-1 / ``-2`` label fold, nan_loss, target scaling, origin
# weighting) since other code compares against it bitwise.
# ===========================================================================


def _add_dims(x: Tensor, ndim: int) -> Tensor:
    """Add singleton dims after the batch dim to reach ``ndim``.

    Raises
    ------
    ValueError
        If ``ndim`` is smaller than ``x.ndim``.
    """
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(f"Target ndim ({ndim}) is smaller than input ndim ({x.dim()})")
    if dim_diff > 0:
        x = x.view(x.shape[0], *dim_diff * (1,), *x.shape[1:])
    return x


def _masked_softmax(x: Tensor, mask: Tensor | None, dim: int = -1) -> Tensor:
    """Softmax ignoring padded elements: masked positions set to -inf before softmax, zeroed after."""
    if mask is not None:
        mask = _add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)
    x = torch.softmax(x, dim=dim)
    if mask is not None:
        x = x.masked_fill(mask, 0)
    return x


# convert flattened array to shape of mask (ntracks, ...) -> (njets, maxtracks, ...)
@torch.jit.script
def _mask_fill_flattened(flat_array: Tensor, mask: Tensor) -> Tensor:
    """Unflatten a per-node array back to a batch-shaped tensor; padded positions read -inf.

    Returns
    -------
    Tensor
        Filled tensor of shape ``[B, L, F]``.
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


class _AbsorbedTaskBase(nn.Module):
    """Wraps a `Dense` head, a loss, an ``input_name`` stream tag, and a scalar ``weight``."""

    def __init__(
        self,
        name: str,
        input_name: str,
        dense_config: dict,
        loss: nn.Module,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.input_name = input_name
        self.net = Dense(**dense_config)
        self.loss = loss
        self.weight = weight

    def input_name_mask(self, pad_masks: Mapping) -> Tensor:
        """Boolean mask selecting tokens from ``self.input_name``.

        Returns
        -------
        Tensor
            Boolean mask of shape ``[L]``, True for positions in ``self.input_name``.
        """
        return torch.cat(
            [
                torch.ones(m.shape[1], device=m.device) * (1 if (t == self.input_name) else 0)
                for t, m in pad_masks.items()
            ],
        ).bool()


class _AbsorbedClassificationTask(_AbsorbedTaskBase):
    """Classification head: CE/BCE loss with ignore_index=-1 and a ``-2`` pad-label fold;
    ``run_inference`` applies sigmoid/softmax/padded-softmax.
    """

    def __init__(
        self,
        label: str,
        class_names: list[str],
        label_map: Mapping | None = None,
        sample_weight: str | None = None,
        **kwargs,
    ) -> None:
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
            assert self.loss.reduction == "none", (
                "Sample weights only supported for reduction='none'"
            )
        if len(self.class_names) != self.net.output_size:
            raise ValueError(
                f"{self.name}: "
                f"Number of outputs ({self.net.output_size}) does not match "
                f"number of class names ({len(self.class_names)}). Class names: {self.class_names}"
            )

    def apply_sample_weight(self, loss: Tensor, labels_dict: Mapping) -> Tensor:
        """Apply per-sample weights to a loss tensor if configured.

        Returns
        -------
        Tensor
            Weighted mean loss if ``sample_weight`` is set; otherwise the input.
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

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted logits and the loss (``None`` when no labels).
        """
        if pad_masks is not None:
            input_name_mask = self.input_name_mask(pad_masks)
            preds = self.net(x[:, input_name_mask], context)
            pad_mask = pad_masks[self.input_name]
        else:
            preds = self.net(x, context)
            pad_mask = None

        labels = labels_dict[self.input_name][self.label] if labels_dict else None
        if labels is not None and self.label_map is not None:
            mapped_labels = torch.clone(labels)
            for k, v in self.label_map.items():
                mapped_labels[labels == k] = v
            labels = mapped_labels

        if pad_mask is not None and labels is not None:
            # TODO @npond: remove once fixed upstream (MR!60199)
            pad_mask = torch.masked_fill(pad_mask, labels == -2, True)
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
        """Convert logits to probabilities.

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
            probs = _masked_softmax(preds, pad_mask.unsqueeze(-1))
        return probs


class _AbsorbedRegressionTaskBase(_AbsorbedTaskBase):
    """Base regression head: single-scaling guard, NaN-masked loss, and target scaling/stacking."""

    def __init__(
        self,
        targets: list[str] | str,
        scaler: RegressionTargetScaler | None = None,
        target_denominators: list[str] | str | None = None,
        norm_params: dict | None = None,
        custom_output_names: list[str] | str | None = None,
        sample_weight: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.scaler = scaler
        self.targets = listify(targets)
        self.target_denominators = listify(target_denominators)
        self.custom_output_names = listify(custom_output_names)
        if norm_params:
            norm_params["mean"] = listify(norm_params["mean"])
            norm_params["std"] = listify(norm_params["std"])
        self.norm_params = norm_params
        self.sample_weight = sample_weight

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
        if self.sample_weight is not None:
            assert self.loss.reduction == "none", (
                "Sample weights only supported for reduction='none'"
            )

    def nan_loss(self, preds: Tensor, targets: Tensor, targets_dict: Mapping, **kwargs) -> Tensor:
        """Loss that ignores NaN targets.

        Returns
        -------
        Tensor
            Mean loss over non-NaN elements.

        Raises
        ------
        ValueError
            If the resulting loss becomes NaN.
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

        if self.sample_weight is not None:
            weights = targets_dict[self.input_name][self.sample_weight]
            weights = weights.unsqueeze(1)
            # If multiple regression targets, expand the weights to match the shape
            if loss.shape[1] > 1:
                weights = weights.expand(-1, loss.shape[1])
            loss = loss * weights

        nanmean = torch.nanmean(loss)
        if torch.isnan(nanmean):
            raise ValueError("NanRegression is NaN. This means all model predictions are NaN")
        return nanmean

    def get_targets(self, targets_dict: Mapping) -> Tensor | None:
        """Assemble and scale regression targets.

        Returns
        -------
        Tensor | None
            Targets of shape ``[B, R]`` (or ``[B, L, R]`` for queries), scaled
            per configuration; ``None`` when there are no targets.
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

            # targets are stacked over dim 0 for scaling consistency, but a query
            # target needs the target axis last, so transpose for that case only
            if len(targets.shape) == 3:
                targets = targets.transpose(1, 2)

        # Must run BEFORE forward's pad-mask NaN fill and nan_loss's isnan-masking:
        # this zeroes NaN/inf in the *data* targets, while padding NaNs (added by
        # forward after this returns) stay NaN and are masked out later — the two
        # mechanisms act on disjoint sets and don't double-count.
        if targets is not None:
            targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        return targets


class _AbsorbedRegressionTask(_AbsorbedRegressionTaskBase):
    """Plain regression head: ``output_size == len(targets)``; forward NaN-fills padded
    targets before ``nan_loss``; ``run_inference`` inverts the scaling.
    """

    def __init__(self, scaler: RegressionTargetScaler | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.net.output_size != len(self.targets):
            raise ValueError(
                f"{self.name}: "
                f"Number of outputs ({self.net.output_size}) does not match "
                f"number of targets ({len(self.targets)})"
            )
        self.scaler = scaler

    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute regression predictions and loss.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted values and the loss (``None`` when no targets).
        """
        if pad_masks is not None and self.input_name != "objects":
            input_name_mask = self.input_name_mask(pad_masks)
            preds = self.net(x[:, input_name_mask], context)
            pad_mask = pad_masks[self.input_name]
        else:
            preds = self.net(x, context)
            pad_mask = None

        targets = self.get_targets(targets_dict)

        # NaN-fill padded targets so nan_loss excludes them
        if pad_mask is not None and targets is not None:
            targets = torch.masked_fill(targets, pad_mask.unsqueeze(-1), torch.nan)

        loss: Tensor | None = None
        if targets is not None:
            loss = self.nan_loss(preds, targets, targets_dict) * self.weight

        return preds, loss

    def run_inference(
        self, preds: Tensor, labels: Mapping | None = None, pad_mask: Tensor | None = None
    ) -> Tensor:
        """Invert target scaling to the original space.

        Indexes the trailing (target-channel) axis ``preds[..., i]`` so a
        per-token ``[B, L, R]`` head de-scales column ``i`` of every token
        (not just token ``i``); bit-identical to indexing axis 1 for a global
        ``[B, R]`` head.

        Returns
        -------
        Tensor
            De-scaled predictions with NaN padding.
        """
        preds = preds.float()
        if self.target_denominators is not None and labels is not None:
            for i in range(len(self.targets)):
                preds[..., i] *= labels[self.input_name][self.target_denominators[i]]
        elif self.norm_params is not None:
            for i in range(len(self.norm_params["mean"])):
                preds[..., i] *= self.norm_params["std"][i]
                preds[..., i] += self.norm_params["mean"][i]
        elif self.scaler is not None:
            for i in range(len(self.targets)):
                preds[..., i] = self.scaler.inverse(self.targets[i], preds[..., i])

        if pad_mask is not None:
            preds = torch.masked_fill(preds, pad_mask.unsqueeze(-1), np.nan)

        return preds


class _AbsorbedGaussianRegressionTask(_AbsorbedRegressionTaskBase):
    """Mu/sigma regression head: ``output_size == 2 * len(targets)``; softplus variance,
    Gaussian NLL loss; ``run_inference`` returns de-scaled ``(means, stds)``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.net.output_size != 2 * len(self.targets):
            raise ValueError(
                f"{self.name}: "
                f"Number of targets ({len(self.targets)}) is not twice the "
                f"number of outputs ({self.net.output_size})"
            )

    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute mean/variance predictions and Gaussian NLL loss.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Concatenated means/variances ``[B, 2R]`` and the loss.
        """
        if pad_masks is not None:
            input_name_mask = self.input_name_mask(pad_masks)
            preds = self.net(x[:, input_name_mask], context)
            pad_mask = pad_masks[self.input_name]
        else:
            preds = self.net(x, context)
            pad_mask = None

        targets = self.get_targets(targets_dict)

        means, variances = preds.tensor_split(2, -1)
        variances = nn.functional.softplus(variances)  # ensure variance stays positive

        # NaN-fill padded targets so nan_loss excludes them
        if pad_mask is not None and targets is not None:
            targets = torch.masked_fill(targets, pad_mask.unsqueeze(-1), torch.nan)

        loss: Tensor | None = None
        if targets is not None:
            loss = self.nan_loss(means, targets, targets_dict, var=variances) * self.weight

        return preds, loss

    def run_inference(
        self, preds: Tensor, labels: Mapping | None = None, pad_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Invert scaling for means + (sqrt of) variances.

        Indexes the trailing axis (``preds[..., i]`` / ``preds[..., i + 1]``) so a
        per-token ``[B, L, 2R]`` head de-scales column ``i``/``i + 1`` of every
        token; bit-identical to indexing axis 1 for a global ``[B, 2R]`` head.

        Returns
        -------
        tuple[Tensor, Tensor]
            De-scaled ``(means, stds)`` each of shape ``[..., R]``.

        Raises
        ------
        ValueError
            If called without the necessary scaling parameters.
        """
        if self.target_denominators is not None and labels is not None:
            for i in range(len(self.targets)):
                preds[..., i] *= labels[self.input_name][self.target_denominators[i]]
                preds[..., i + 1] *= labels[self.input_name][self.target_denominators[i]]
        elif self.norm_params is not None:
            for i in range(len(self.norm_params["mean"])):
                preds[..., i] *= self.norm_params["std"][i]
                preds[..., i] += self.norm_params["mean"][i]
                # return stddev as sqrt(var)
                preds[..., i + 1] = (
                    torch.sqrt(nn.functional.softplus(preds[..., i + 1]))
                    * self.norm_params["std"][i]
                )
        else:
            raise ValueError("Inference for Gaussian regression requires scaling parameters.")
        means, stds = preds.tensor_split(2, -1)

        if pad_mask is not None:
            means = torch.masked_fill(means, pad_mask.unsqueeze(-1), np.nan)
            stds = torch.masked_fill(stds, pad_mask.unsqueeze(-1), np.nan)

        return means, stds


class _AbsorbedVertexingTask(_AbsorbedTaskBase):
    """Edge-classification vertexing head: builds the compressed track-track matrix,
    weights per-edge BCE by origin labels (``get_weights``), and ``run_inference``
    returns per-node union-find assignments. ``_OriginWeightedVertexing`` overrides
    ``get_weights``.
    """

    def __init__(self, label: str, **kwargs) -> None:
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

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted edge logits ``[E, 1]`` and the scalar loss.
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

        context_matrix = None
        if context is not None:
            context_d = context.shape[-1]
            context = context.unsqueeze(1).expand(b, n, context_d)
            context_matrix = torch.zeros(
                (adjmat.sum(), 2 * context_d), device=x.device, dtype=x.dtype
            )
            context_matrix = context.unsqueeze(-2).expand((b, n, n, context_d))[adjmat[:, :-1, :-1]]

        # compressed track-track matrix (one row per valid edge, not [B, N, N])
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

        Returns
        -------
        Tensor
            Weighted average loss scaled by ``self.weight``.
        """
        labels = labels_dict[self.input_name][self.label]

        match_matrix = labels.unsqueeze(-1) == labels.unsqueeze(-2)

        # negative-class labels never count as a match, even to each other
        unique_matrix = labels < 0
        unique_matrix = unique_matrix.unsqueeze(-1) | unique_matrix.unsqueeze(-2)
        match_matrix *= ~unique_matrix

        match_matrix = match_matrix[adjmat].float()

        loss = self.loss(pred.squeeze(-1), match_matrix)

        origin_label = self.label.replace("VertexIndex", "OriginLabel")
        weights = self.get_weights(labels_dict[self.input_name][origin_label], adjmat)
        weighted_loss = loss * weights

        num_non_masked_elements = match_matrix.sum()
        loss = weighted_loss.sum() / num_non_masked_elements

        return loss * self.weight

    def get_weights(self, labels: Tensor, adjmat: Tensor) -> Tensor:
        """Per-edge weights from hardcoded heavy/fake origin ids (3, 4, 5 / 1).

        ``_OriginWeightedVertexing`` overrides this with config-driven ids.

        Returns
        -------
        Tensor
            Per-edge weights ``[E]`` after adjacency compression.
        """
        weights = torch.clip(sum(labels == i for i in (3, 4, 5)), 0, 1) - (labels == 1).int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        return 1 + weights

    def run_inference(self, preds: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Per-node assignments from edge predictions.

        Returns
        -------
        Tensor
            Flattened per-node assignments with paddings filled to ``-inf``.
        """
        preds = get_node_assignment_jit(preds, pad_mask)
        return _mask_fill_flattened(preds, pad_mask)


class _TaskModuleBase(nn.Module):
    """Shared config capture + loss construction for the v2 task modules."""

    def __init__(
        self,
        stream: str,
        label: str,
        input: str | None,  # noqa: A002 - matches the YAML config key name
        context: str | None,
        dense: dict[str, Any] | None,
        loss: str | dict[str, Any] | None,
        weight: float,
        default_loss: dict[str, Any],
        expose: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        _reject_width_keys(type(self).__name__, dense, _WIDTH_KEYS)
        self.stream = stream
        self.label = label
        self.input_key = input if input is not None else f"encoded.{stream}"
        self.context = context
        self.dense_cfg = dict(dense or {})
        self.loss_cfg = _loss_cfg(loss, default_loss)
        self.weight = float(weight)
        self.expose_modes = _parse_expose(expose, type(self).__name__)
        self.task: nn.Module | None = None

    def _pred_spec(self, spec: TensorSpec) -> TensorSpec:
        """Restrict a prediction port spec to the configured ``expose`` modes.

        Returns
        -------
        TensorSpec
            `spec` re-stamped to the exposed modes (its kind/shape/dtype kept).
        """
        if self.expose_modes == Mode.ALL:
            return spec
        return replace(spec, modes=spec.modes & self.expose_modes)

    @property
    def pred_key(self) -> str:
        """The published prediction key.

        Returns
        -------
        str
            ``preds.<stream>.<instance-name>``.
        """
        return f"preds.{self.stream}.{self.name}"

    @property
    def loss_key(self) -> str:
        """The published loss key, auto-collected by `LossSum`.

        Returns
        -------
        str
            ``losses.<instance-name>``.
        """
        return f"losses.{self.name}"

    @property
    def label_key(self) -> str:
        """The declared label dependency.

        Returns
        -------
        str
            ``labels.<stream>.<label>``.
        """
        return f"labels.{self.stream}.{self.label}"

    @property
    def has_pad_mask(self) -> bool:
        """Whether this (sequence) head consumes a per-stream pad mask.

        True for a normal variable-length constituent stream; False for a
        non-sequence head or a fixed-count query bank (e.g. MaskFormer
        ``objects``, which has no padding).

        Returns
        -------
        bool
            ``self.sequence and self.stream not in _NO_PAD_MASK_STREAMS``.
        """
        return self.sequence and self.stream not in _NO_PAD_MASK_STREAMS

    # -- output rendering: TEST columns + values, ONNX manifest -----------------
    #
    # Per-family rendering lives on the task; `TaskWriter` only orchestrates
    # (groups, prefixes-by-stream, pads, writes) and never branches on task
    # family. A family that ships no rendering inherits the base methods below,
    # which raise a ConfigError naming the missing rendering.

    onnx_renameable: bool = False
    """Whether `TaskWriter` ``onnx_names`` may override this task's ONNX suffix.

    True for classification (needed when class names collide across two
    classification tasks). False for vertexing (fixed `VERTEX_INDEX` suffix)
    and regression (suffixes are its own ``custom_output_names``).
    """

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """The TEST column schema for this task — ``(column_name, np_dtype_str)``.

        Parameters
        ----------
        run_name : str
            The run ``name:`` — the TEST column prefix.

        Raises
        ------
        ConfigError
            For a task family that ships no TEST rendering.
        """
        del run_name
        raise ConfigError(self._no_render_msg("TEST columns"))

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Render this task's formatted TEST values as a structured array.

        Reads the task's published ``preds.*`` leaf from `b` and renders it to
        a structured array whose dtype is ``np.dtype(self.output_names(run_name))``.

        Parameters
        ----------
        b : Bundle
            The executed TEST bundle (carries the converted ``preds.*`` leaf).
        run_name : str
            The run ``name:`` — the TEST column prefix (matches `output_names`).

        Raises
        ------
        ConfigError
            For a task family that ships no TEST rendering.
        """
        del b, run_name
        raise ConfigError(self._no_render_msg("TEST values"))

    def onnx_outputs(self) -> list[ExportOutput]:
        """Render this task's ONNX-manifest entries.

        One `ExportOutput` per family entry, referencing a shipped reduce by
        key: global classification -> ``split_scalars`` per-class suffixes;
        sequence classification -> one ``argmax`` int8 entry; vertexing ->
        ``vertex_union_find`` int8 on the shared `VERTEX_INDEX` constant;
        regression -> ``split_scalars`` per-target suffixes. `TaskWriter`
        decides WHICH tasks export; the task only renders its own entry.

        Raises
        ------
        ConfigError
            For a task family that ships no ONNX rendering.
        """
        raise ConfigError(self._no_render_msg("ONNX output"))

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Render this task's converted, graph-visible output fields.

        Reads the task's RAW ``preds.*`` leaf (loss-space) plus any output-time
        deps (`output_time_requires`) from `b`, applies the eval conversion
        (softmax / masked-softmax / argmax / descale / union-find) in
        TRACEABLE torch ops so ONNX sees them in-graph, and returns one
        `OutputField` per serialisation leaf. Each field carries the converted
        torch ``value`` plus the bare suffix + dtype + axis metadata — it does
        NOT pack a structured numpy array, prefix the run/model name, or
        downcast precision (the sink does all three).

        Parameters
        ----------
        b : Bundle
            The executed bundle (carries the RAW ``preds.*`` leaf + any
            output-time dep, e.g. ``masks.<stream>``).
        mode : Mode
            The execution mode — selects the H5 (probs) vs ONNX
            (split-scalars / argmax index) representation.
        run_name : str
            Accepted for symmetry with `get_h5`/`output_names` but NOT baked
            into the field names (the sink prefixes).

        Raises
        ------
        ConfigError
            For a task family that ships no output rendering.
        """
        del b, mode, run_name
        raise ConfigError(self._no_render_msg("output"))

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free serialisation-leaf metadata for `mode`.

        The bundle-free twin of `get_output`: the SAME `OutputField` list
        (names / dtypes / axis / final / prefix, same order) but with every
        ``value`` left ``None``, so sinks can resolve column names/dtypes/order
        before any batch runs.

        Parameters
        ----------
        mode : Mode
            The execution mode — selects the H5 vs ONNX representation,
            exactly as `get_output`.
        run_name : str
            Accepted for symmetry; NOT baked into the names (the sink prefixes).

        Raises
        ------
        ConfigError
            For a task family that ships no output rendering.
        """
        del mode, run_name
        raise ConfigError(self._no_render_msg("output manifest"))

    def output_time_requires(self, mode: Mode) -> list[str]:
        """The non-pred bundle keys `get_output` needs at output time.

        E.g. a per-token (sequence) head needs the stream pad mask
        (``masks.<stream>``) for the masked softmax; a global (pooled) head
        needs nothing extra. The base returns ``[]``; per-family overrides
        declare their own.

        Parameters
        ----------
        mode : Mode
            The execution mode (for families whose deps are mode-split).

        Returns
        -------
        list[str]
            Dotted bundle keys (empty by default).
        """
        del mode
        return []

    def _no_render_msg(self, what: str) -> str:
        """The unsupported-family error message.

        Returns
        -------
        str
            Naming the task instance, its type, and the missing rendering.
        """
        return (
            f"task {self.name!r} ({type(self).__name__}) ships no {what} rendering — "
            "supported families are ClassificationTaskModule, VertexingTaskModule and "
            "RegressionTaskModule; give a custom task module output_names/get_h5/onnx_outputs "
            "methods, or write a custom Writer for its outputs (design §8)"
        )


class ClassificationTaskModule(_TaskModuleBase):
    """Classification head over a pooled vector or a per-stream sequence.

    ``class_names`` is REQUIRED and explicit. When the dataset schema names the
    label's classes, the configured list is cross-checked (set and order) by
    `salt.core.saltmodule.check_class_names`. The head width is
    ``len(class_names)``; input/context widths are inferred at `bind`.

    Declarative class weights: ``weight_source: {from_class_dict: <path>}``
    allocates the CE ``weight`` buffer at `bind`, fills it at `materialise()`
    on fresh fits (the only file I/O), and inherits it from the checkpoint on
    resume. A literal ``loss.init_args.weight`` list is exclusive with
    ``weight_source``.
    """

    onnx_renameable = True
    """Classification ONNX suffixes are overridable via ``onnx_names``."""

    def __init__(
        self,
        stream: str,
        label: str,
        class_names: Sequence[str],
        input: str | None = None,  # noqa: A002 - matches the YAML config key name
        context: str | None = None,
        sequence: bool | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        weight_source: Mapping[str, str] | None = None,
        label_map: dict[int, int] | None = None,
        expose: Sequence[str] | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        class_names : Sequence[str]
            Ordered class names — REQUIRED, index-aligned with outputs.
        sequence : bool | None, optional
            Whether the task is per-token, by default inferred from `input`.
        weight_source : Mapping[str, str] | None, optional
            ``{"from_class_dict": <path>}`` declarative class-weight source
            (see class docstring), by default None.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in, by default all modes.
            ``[fit, val]`` opts a train-only aux task out of TEST/ONNX.

        Raises
        ------
        ConfigError
            On empty/duplicate class names, malformed `weight_source`, a
            literal loss weight combined with `weight_source`, or a bad
            `expose` list.
        """
        super().__init__(
            stream, label, input, context, dense, loss, weight, _DEFAULT_CLS_LOSS, expose
        )
        if not class_names:
            raise ConfigError(
                f"ClassificationTaskModule: class_names is required and explicit for label "
                f"{label!r} (design §3.3 — no CLASS_NAMES fallback)"
            )
        if len(set(class_names)) != len(tuple(class_names)):
            raise ConfigError(
                f"ClassificationTaskModule: duplicate class names in {tuple(class_names)}"
            )
        self.class_names = tuple(class_names)
        self.sequence = sequence if sequence is not None else input is None
        self.label_map = dict(label_map) if label_map is not None else None
        self.weight_source = _checked_weight_source(weight_source)
        if self.weight_source is not None and "weight" in self.loss_cfg.get("init_args", {}):
            raise ConfigError(
                "ClassificationTaskModule: class weights already specified in the loss config — "
                "remove them or drop weight_source (v1 use_class_dict contract, cli.py:476-479)"
            )

    def declare_io(self, mode: Mode) -> IO:
        """Declare input/context/masks (+ FIT|VAL labels) -> preds (+ FIT|VAL loss).

        Returns
        -------
        IO
            The declared requires/produces; label and loss ports carry
            ``modes=TRAINING``.
        """
        del mode
        n_classes = len(self.class_names)
        width = sym_dim("D", self.name)
        if self.sequence:
            input_spec = TensorSpec(shape=("B", _stream_len(self.stream), width), dtype="float32")
            label_spec = TensorSpec(
                shape=("B", _stream_len(self.stream)),
                dtype="int64",
                kind="label",
                modes=Mode.TRAINING,
            )
            pred_spec = TensorSpec(
                shape=("B", _stream_len(self.stream), n_classes), dtype="float32"
            )
        else:
            input_spec = TensorSpec(shape=("B", width), dtype="float32")
            label_spec = TensorSpec(shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING)
            pred_spec = TensorSpec(shape=("B", n_classes), dtype="float32")
        requires: dict[str, TensorSpec] = {self.input_key: input_spec}
        if self.context is not None:
            requires[self.context] = TensorSpec(shape=None, dtype="float32")
        if self.has_pad_mask:
            requires[f"masks.{self.stream}"] = TensorSpec(
                shape=("B", _stream_len(self.stream)), dtype="bool", kind="pad_mask"
            )
        requires[self.label_key] = label_spec
        produces: dict[str, TensorSpec] = {
            self.pred_key: self._pred_spec(pred_spec),
            self.loss_key: TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the composed head with inferred widths.

        When `weight_source` is set, the loss is constructed with a
        ones-initialised ``weight`` buffer sized ``len(class_names)`` so it
        lives in the state_dict; `materialise` fills it on fresh fits.
        """
        init_args = dict(self.loss_cfg.get("init_args", {}))
        if self.weight_source is not None:
            init_args["weight"] = torch.ones(len(self.class_names))
        elif isinstance(init_args.get("weight"), (list, tuple)):
            init_args["weight"] = torch.as_tensor(init_args["weight"], dtype=torch.float32)
        loss_module = _loss_class(self.loss_cfg)(**init_args)
        dense_config = {
            "input_size": schema.width(self.input_key),
            "output_size": len(self.class_names),
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
        }
        self.task = _AbsorbedClassificationTask(
            name=self.name,
            input_name=self.stream,
            label=self.label,
            class_names=list(self.class_names),
            label_map=self.label_map,
            loss=loss_module,
            weight=self.weight,
            dense_config=dense_config,
        )

    def materialise(self) -> None:
        """Resolve ``weight_source`` from the class dict (the ONLY file I/O).

        Raises
        ------
        RuntimeError
            If called before `bind`.
        ValueError
            If the class dict lacks the stream/label or the weight count
            does not match ``class_names``.
        """
        if self.weight_source is None:
            return
        if self.task is None:
            raise RuntimeError(f"{type(self).__name__} {self.name!r}: materialise before bind()")
        path = self.weight_source["from_class_dict"]
        with open(path) as fh:
            class_dict = yaml.safe_load(fh)
        try:
            values = class_dict[self.stream][self.label]
        except (KeyError, TypeError):
            raise ValueError(
                f"Label {self.label!r} for stream {self.stream!r} not found in class dict "
                f"{path} — drop weight_source and specify class weights manually "
                f"(cli.py:483-488 semantics)"
            ) from None
        if len(values) != len(self.class_names):
            raise ValueError(
                f"class dict {path} has {len(values)} weights for {self.stream}.{self.label} "
                f"but the task declares {len(self.class_names)} class_names (design §3.3)"
            )
        with torch.no_grad():
            self.task.loss.weight.copy_(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the composed head; publishes RAW logits in EVERY non-training mode.

        The eval conversion (softmax / masked softmax) is owned by the
        `ClassProbs`/`SeqClassProbs`/`SeqClassIndex` producers and by `get_h5`,
        both of which read this raw leaf and convert exactly once — so there is
        no double-softmax.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        assert self.task is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        # objects-stream (query-bank) heads have no pad mask
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        if mode & Mode.TRAINING:
            labels_dict = {self.stream: {self.label: b.get(self.label_key)}}
            pad_masks = {self.stream: mask} if self.has_pad_mask else None
            preds, loss = self.task(x, labels_dict, pad_masks, context=ctx)
            return {self.pred_key: preds, self.loss_key: loss}
        preds, _ = self.task(x, None, None, context=ctx)
        return {self.pred_key: preds}

    # -- output rendering ---------------------------------------------------

    @property
    def class_suffixes(self) -> list[str]:
        """Per-class logical suffixes — ``Flavours[c].px`` else ``p{c}``.

        Returns
        -------
        list[str]
            One suffix per ``class_names`` entry, in class order.
        """
        # Must stay a LOCAL import: `Flavours` (ftag LabelContainer) raises
        # KeyError (not AttributeError) from __getattr__ on unknown names, which
        # breaks jsonargparse's hasattr(value, "__args__") protocol walk over
        # this module's globals if imported at module level — the whole
        # model.modules config would fail to parse.
        from ftag import Flavours  # noqa: PLC0415

        return [Flavours[c].px if c in Flavours else f"p{c}" for c in self.class_names]

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """One ``f4`` column per class, named ``{run_name}_{px}``.

        Returns
        -------
        list[tuple[str, str]]
            ``(column, "f4")`` pairs, one per class, in class order.
        """
        return [(f"{run_name}_{px}", "f4") for px in self.class_suffixes]

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Softmax the RAW TEST logits then pack as ``f4`` columns (padded positions read 0.0).

        Returns
        -------
        np.ndarray
            ``[B]`` (global) or ``[B, L]`` (sequence) structured array.
        """
        assert self.task is not None, "get_h5 before bind()"
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        preds = self.task.run_inference(b.get(self.pred_key), mask)
        dtype = np.dtype(self.output_names(run_name))
        return u2s(preds.float().cpu().numpy(), dtype)

    def onnx_outputs(self) -> list[ExportOutput]:
        """Global -> per-class ``split_scalars``; sequence -> one ``argmax`` int8.

        A global (pooled) head emits one float32 scalar per class (suffixes =
        `class_suffixes`); a per-token sequence head emits a single int8 argmax
        entry named by the Pascal-case task name (``track_origin -> TrackOrigin``).

        Returns
        -------
        list[ExportOutput]
            One entry (per-class scalars, or a single argmax).
        """
        if not self.sequence:
            return [ExportOutput(port=self.pred_key, names=list(self.class_suffixes))]
        return [
            ExportOutput(
                port=self.pred_key,
                name=pascal_case(self.name),
                reduce="argmax",
                dtype="int8",
            )
        ]

    def output_time_requires(self, mode: Mode) -> list[str]:
        """The non-pred deps `get_output` reads: the stream pad mask for a seq head.

        Mode-independent for this family.

        Returns
        -------
        list[str]
            ``["masks.<stream>"]`` for a padded seq head, else ``[]``.
        """
        del mode
        # has_pad_mask already implies sequence, so this covers all three cases:
        # global [] / no-pad-mask seq [] / padded seq [masks.<stream>]
        if self.has_pad_mask:
            return [f"masks.{self.stream}"]
        return []

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Softmax/argmax the RAW logits into graph-visible `OutputField`s.

        Reads the RAW ``preds.*`` logits + the stream pad mask (seq head) and
        runs the eval conversion in TRACEABLE torch ops. Per mode:

        - **Global (pooled) head**: ``sigmoid`` (BCE) else ``softmax(dim=-1)``.
          One ``f4`` field per class (``class_suffixes``). H5 modes keep the
          per-class column ``[B]`` unsqueezed; ONNX squeezes to the rank-0
          scalar the export sink expects.
        - **Sequence (per-token) head**: masked softmax (padded tokens -> 0.0).
          H5 modes emit one ``f4`` field per class (``[B, L]``, H5-only). ONNX
          instead emits a single int8 argmax field named ``pascal_case(name)``.

        ``run_name`` is not baked into the field names (the sink prefixes it).

        Returns
        -------
        list[OutputField]
            Per-class probability fields (H5 modes), or a single argmax-index
            field (ONNX seq head); each carries a torch ``value``.
        """
        del run_name
        assert self.task is not None, "get_output before bind()"
        logits = b.get(self.pred_key)
        if not self.sequence:
            if isinstance(self.task.loss, torch.nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(logits)
            else:
                assert logits.ndim == 2, "global classification head expects [B, C] logits"
                probs = torch.softmax(logits, dim=-1)
            # ONNX sink squeezes [1, C] per-class columns to rank-0 scalars; H5
            # sink packs the full [B] column, so only squeeze in ONNX mode
            squeeze_global = bool(mode & Mode.ONNX)
            return [
                OutputField(
                    h5_name=px,
                    dtype="f4",
                    axis="global",
                    final=True,
                    value=probs[..., c].squeeze() if squeeze_global else probs[..., c],
                )
                for c, px in enumerate(self.class_suffixes)
            ]
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        probs = _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
        if mode & Mode.ONNX:
            # zero-row append/strip -> [L] int8 argmax leaf under the pascal-case task name
            padded = torch.concatenate([probs, torch.zeros((1, 1, probs.shape[-1]))], dim=1)
            index = torch.argmax(padded, dim=-1)[:, :-1].squeeze(0).char()
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=pascal_case(self.name),
                    dtype="int8",
                    axis="per_token",
                    final=True,
                    value=index,
                )
            ]
        # H5-only: the ONNX side of a seq head is the argmax leaf above, not these
        # per-class probs. Callers MUST select ONNX outputs by calling with
        # mode=Mode.ONNX (which returns before reaching here), not by scanning
        # resolved_onnx_name — these fields fall back to a (misleading) per-class
        # ONNX suffix since onnx_name=None.
        return [
            OutputField(
                h5_name=px,
                onnx_name=None,
                dtype="f4",
                axis="per_token",
                final=True,
                value=probs[..., c],
            )
            for c, px in enumerate(self.class_suffixes)
        ]

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free field metadata mirroring `get_output` for `mode` (``value=None``).

        Returns
        -------
        list[OutputField]
            The value-free serialisation fields, in field order.
        """
        del run_name
        if not self.sequence:
            return [
                OutputField(h5_name=px, dtype="f4", axis="global", final=True)
                for px in self.class_suffixes
            ]
        if mode & Mode.ONNX:
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=pascal_case(self.name),
                    dtype="int8",
                    axis="per_token",
                    final=True,
                )
            ]
        return [
            OutputField(h5_name=px, onnx_name=None, dtype="f4", axis="per_token", final=True)
            for px in self.class_suffixes
        ]


class VertexingTaskModule(_TaskModuleBase):
    """Edge-classification vertexing head.

    The origin-label dependency is declared (``origin_label:`` config ->
    ``labels.<stream>.<origin_label>``). ``origin_weighting`` gives the
    heavy/fake origins as integer ids OR class names (defaults reproduce the
    hardcoded 3,4,5 / 1). Names are resolved to ids at fit/test setup against
    the dataset schema's origin class-name attr (`resolve_origin_names`,
    called before `bind`); a name-based config without a schema artifact is a
    `ConfigError` at bind. Mixing names and ids in one heavy/fake list is
    rejected.

    TEST publishes per-node vertex assignments (union-find); ONNX publishes
    RAW edge scores for the export-side ``vertex_union_find`` reduce.
    """

    def __init__(
        self,
        stream: str,
        label: str,
        origin_label: str,
        input: str | None = None,  # noqa: A002 - matches the YAML config key name
        context: str | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        origin_weighting: Mapping[str, Sequence[int | str]] | None = None,
        prefix_vertex_column: bool = False,
        expose: Sequence[str] | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        label : str
            Vertex-index label; must contain ``"VertexIndex"`` (the composed
            loss derives the origin key by string-replace).
        origin_label : str
            Declared origin-label dependency for edge weighting (and, for
            name-based weighting, the schema attr the names resolve against).
        loss : str | dict[str, Any] | None, optional
            Loss config, by default ``BCEWithLogitsLoss(reduction="none")``.
            Reduction MUST be ``none`` — per-edge weighting multiplies the
            unreduced loss.
        origin_weighting : Mapping[str, Sequence[int | str]] | None, optional
            ``{"heavy": [...], "fake": [...]}`` origin ids OR class names, by
            default ``{"heavy": [3, 4, 5], "fake": [1]}``. Names are resolved
            at setup against the schema's origin class-name attr; ids bind
            directly.
        prefix_vertex_column : bool, optional
            Name the TEST vertexing column ``{run_name}_VertexIndex`` instead
            of the bare ``VertexIndex``, by default False. The ONNX name is
            always the shared `VERTEX_INDEX` constant.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in, by default all modes.
            ``[fit, val]`` opts a train-only aux task out of TEST/ONNX.

        Raises
        ------
        ConfigError
            If `label` lacks ``"VertexIndex"``, `origin_weighting` has unknown
            keys, a heavy/fake list mixes integer ids with class names, or
            `expose` is a bad mode list.
        """
        super().__init__(
            stream, label, input, context, dense, loss, weight, _DEFAULT_VTX_LOSS, expose
        )
        if "VertexIndex" not in label:
            raise ConfigError(
                f"VertexingTaskModule: label {label!r} must contain 'VertexIndex' — the "
                "composed v1 loss derives the origin key as "
                "label.replace('VertexIndex', 'OriginLabel') (task.py:937; absorbed at M7)"
            )
        self.origin_label = origin_label
        self.prefix_vertex_column = bool(prefix_vertex_column)
        weighting = dict(origin_weighting or {"heavy": [3, 4, 5], "fake": [1]})
        if unknown := sorted(set(weighting) - {"heavy", "fake"}):
            raise ConfigError(
                f"VertexingTaskModule: unknown origin_weighting keys {unknown} — expected "
                "'heavy' and 'fake' (design §3.3)"
            )
        heavy = tuple(weighting.get("heavy", (3, 4, 5)))
        fake = tuple(weighting.get("fake", (1,)))
        # ids bind standalone; names defer to resolve_origin_names(reader) at setup,
        # so heavy_ids/fake_ids stay None until resolved (a name-based bind without
        # a schema then fails loudly instead of silently mis-weighting)
        self._heavy_cfg, self._fake_cfg = heavy, fake
        self._names_pending = _is_name_weighting(heavy, fake)
        if self._names_pending:
            self.heavy_ids: tuple[int, ...] | None = None
            self.fake_ids: tuple[int, ...] | None = None
        else:
            self.heavy_ids = _coerce_origin_ids(heavy, "heavy")
            self.fake_ids = _coerce_origin_ids(fake, "fake")

    @property
    def origin_label_key(self) -> str:
        """The declared origin-label dependency.

        Returns
        -------
        str
            ``labels.<stream>.<origin_label>``.
        """
        return f"labels.{self.stream}.{self.origin_label}"

    def resolve_origin_names(self, reader: Any) -> bool:
        """Resolve name-based ``origin_weighting`` to ids against the schema.

        No-op for integer-id weighting. For name-based weighting, the origin
        label's class-name attr (``schema_group(stream).attrs[origin_label]``)
        maps each name to its index. Must be called (via
        `salt.core.saltmodule.resolve_origin_weighting`) before `bind`.

        Parameters
        ----------
        reader : Any
            The stage dataset reader; consulted via ``schema_group(stream)``
            (duck-typed — a reader without schema support resolves nothing,
            leaving a name-based config to fail loudly at bind).

        Returns
        -------
        bool
            True when names were resolved here, False when there was nothing
            to resolve (integer ids) or the reader has no schema artifact.

        Raises
        ------
        ConfigError
            When the stream/origin-label class-name attr is absent or a
            configured name is not among the schema's origin classes.
        """
        if not self._names_pending:
            return False
        schema_group = getattr(reader, "schema_group", None)
        if not callable(schema_group):
            return False
        gschema = schema_group(self.stream)
        attr = gschema.attrs.get(self.origin_label) if gschema is not None else None
        if not (
            isinstance(attr, (list, tuple)) and attr and all(isinstance(item, str) for item in attr)
        ):
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: name-based origin_weighting needs the "
                f"origin label's class names, but the schema artifact has no string-list "
                f"attr {self.origin_label!r} on the {self.stream!r} group (config: "
                f"model.modules.{self.name}.init_args.origin_weighting; design §5.1, §2.6) — "
                "dump the schema with the origin class names, or use integer origin ids"
            )
        index = {name: i for i, name in enumerate(attr)}
        self.heavy_ids = self._resolve_names("heavy", self._heavy_cfg, index, attr)
        self.fake_ids = self._resolve_names("fake", self._fake_cfg, index, attr)
        self._names_pending = False
        return True

    def _resolve_names(
        self, role: str, names: Sequence[Any], index: Mapping[str, int], classes: Sequence[str]
    ) -> tuple[int, ...]:
        """Map one heavy/fake class-name list to integer origin ids.

        Returns
        -------
        tuple[int, ...]
            The resolved integer ids, in config order.

        Raises
        ------
        ConfigError
            On any name absent from the schema's origin classes.
        """
        ids: list[int] = []
        for name in names:
            if name not in index:
                raise ConfigError(
                    f"VertexingTaskModule {self.name!r}: origin_weighting {role!r} class "
                    f"{name!r} is not among the {self.origin_label!r} classes {list(classes)} "
                    f"(config: model.modules.{self.name}.init_args.origin_weighting; design §5.1)"
                )
            ids.append(index[name])
        return tuple(ids)

    def declare_io(self, mode: Mode) -> IO:
        """Declare input/context/mask (+ FIT|VAL vertex AND origin labels) -> preds/loss.

        The prediction spec is shape-unconstrained: the edge count is
        data-dependent in FIT|VAL|ONNX (``[E, 1]`` raw scores) and the TEST
        assignments are per-node.

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        width = sym_dim("D", self.name)
        label_spec = TensorSpec(
            shape=("B", _stream_len(self.stream)),
            dtype="int64",
            kind="label",
            modes=Mode.TRAINING,
        )
        requires: dict[str, TensorSpec] = {
            self.input_key: TensorSpec(
                shape=("B", _stream_len(self.stream), width), dtype="float32"
            ),
            f"masks.{self.stream}": TensorSpec(
                shape=("B", _stream_len(self.stream)), dtype="bool", kind="pad_mask"
            ),
            self.label_key: label_spec,
            self.origin_label_key: label_spec,
        }
        if self.context is not None:
            requires[self.context] = TensorSpec(shape=None, dtype="float32")
        produces: dict[str, TensorSpec] = {
            self.pred_key: self._pred_spec(TensorSpec(shape=None, dtype=None)),
            self.loss_key: TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the composed vertexing head with inferred widths.

        ``input_size = 2 * width(input)`` (pair concat); ``context_size = width(context)``.

        Raises
        ------
        ConfigError
            If the configured loss does not use ``reduction="none"`` (per-edge
            weighting requires the unreduced loss), or name-based
            ``origin_weighting`` was never resolved before bind.
        """
        if self._names_pending or self.heavy_ids is None or self.fake_ids is None:
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: name-based origin_weighting "
                f"(heavy={list(self._heavy_cfg)}, fake={list(self._fake_cfg)}) was not resolved "
                "to integer ids before bind — a dataset schema artifact carrying the "
                f"{self.origin_label!r} class names is required (design §5.1, §2.6). "
                "resolve_origin_names(reader) runs at fit/test setup; standalone bind needs "
                "integer origin ids instead"
            )
        init_args = dict(self.loss_cfg.get("init_args", {}))
        loss_module = _loss_class(self.loss_cfg)(**init_args)
        if getattr(loss_module, "reduction", "none") != "none":
            raise ConfigError(
                f"VertexingTaskModule {self.name!r}: loss reduction must be 'none' — the "
                "origin weighting multiplies the per-edge loss (task.py:938-947)"
            )
        width = schema.width(self.input_key)
        dense_config = {
            "input_size": 2 * width,
            "output_size": 1,
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
        }
        self.task = _OriginWeightedVertexing(
            heavy_ids=self.heavy_ids,
            fake_ids=self.fake_ids,
            name=self.name,
            input_name=self.stream,
            label=self.label,
            loss=loss_module,
            weight=self.weight,
            dense_config=dense_config,
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the composed head; publishes RAW ``[E, 1]`` edge scores in EVERY non-training mode.

        The labels dict carries the origin labels under the derived key
        (``label.replace("VertexIndex", "OriginLabel")``) so the composed loss
        finds them — the graph dependency is the declared ``origin_label`` port.

        The union-find conversion (edge scores -> per-node assignments) is
        owned by ``get_output`` on the live path and by ``get_h5`` on the
        oracle path, both of which read this raw leaf and convert exactly once.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        assert self.task is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        mask = b.get(f"masks.{self.stream}")
        pad_masks = {self.stream: mask}
        if mode & Mode.TRAINING:
            derived = self.label.replace("VertexIndex", "OriginLabel")
            labels_dict = {
                self.stream: {
                    self.label: b.get(self.label_key),
                    derived: b.get(self.origin_label_key),
                }
            }
            preds, loss = self.task(x, labels_dict, pad_masks, context=ctx)
            return {self.pred_key: preds, self.loss_key: loss}
        preds, _ = self.task(x, None, pad_masks, context=ctx)
        return {self.pred_key: preds}

    # -- output rendering ---------------------------------------------------

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """A single ``('VertexIndex', 'i8')`` column.

        Bare ``VertexIndex`` by default; with ``prefix_vertex_column`` it is
        ``{run_name}_VertexIndex``.

        Returns
        -------
        list[tuple[str, str]]
            One ``(column, "i8")`` pair.
        """
        column = f"{run_name}_{VERTEX_INDEX}" if self.prefix_vertex_column else VERTEX_INDEX
        return [(column, "i8")]

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Per-node vertex assignments as one ``i8`` column.

        Union-finds the raw ``preds.*`` edge scores (``get_node_assignment_jit``
        -> ``_mask_fill_flattened``) then casts to int. Padded positions read
        the int32 cast of ``-inf`` (-2147483648).

        Returns
        -------
        np.ndarray
            ``[B, L]`` structured array with one ``i8`` field.
        """
        assert self.task is not None, "get_h5 before bind()"
        mask = b.get(f"masks.{self.stream}")
        preds = self.task.run_inference(b.get(self.pred_key), mask)
        dtype = np.dtype(self.output_names(run_name))
        return u2s(preds.int().cpu().numpy(), dtype)

    def onnx_outputs(self) -> list[ExportOutput]:
        """One ``vertex_union_find`` int8 entry on the shared `VERTEX_INDEX` suffix.

        The export suffix is the same `VERTEX_INDEX` constant the TEST column
        uses (the exporter prepends ``{model_name}_``). The in-graph union-find
        lives in the shipped ``vertex_union_find`` reduce; ONNX publishes raw
        edge scores that the reduce consumes.

        Returns
        -------
        list[ExportOutput]
            One ``vertex_union_find`` int8 entry.
        """
        return [
            ExportOutput(
                port=self.pred_key,
                name=VERTEX_INDEX,
                reduce="vertex_union_find",
                dtype="int8",
            )
        ]

    def output_time_requires(self, mode: Mode) -> list[str]:
        """The non-pred dep `get_output` reads: the stream pad mask.

        A vertexing head always has a pad mask (its ``forward`` requires
        ``masks.<stream>`` unconditionally), so this is mode-independent.

        Returns
        -------
        list[str]
            ``["masks.<stream>"]``.
        """
        del mode
        return [f"masks.{self.stream}"]

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Union-find the RAW edge scores into a graph-visible int8 leaf.

        Reads the RAW ``preds.*`` ``[E, 1]`` edge scores + the stream pad mask
        and runs the union-find conversion in TRACEABLE torch ops. Per mode:

        - **ONNX**: ``get_node_assignment_jit`` -> ``mask_fill_flattened`` ->
          ``.reshape(-1).char()`` -> one int8 ``[L]`` per-token field under the
          shared `VERTEX_INDEX` suffix.
        - **H5 modes** (TEST): ``run_inference`` -> ``.int()`` — the exact value
          ``get_h5`` packs. One int8 per-token field, ``onnx_name=None``. The
          column ``prefix`` follows ``prefix_vertex_column``.

        ``run_name`` is not baked in (the sink prefixes it).

        Returns
        -------
        list[OutputField]
            One int8 vertex-index field (the per-token union-find assignment).
        """
        del run_name
        assert self.task is not None, "get_output before bind()"
        edge_scores = b.get(self.pred_key)
        mask = b.get(f"masks.{self.stream}")
        if mode & Mode.ONNX:
            vertex_indices = get_node_assignment_jit(edge_scores, mask)
            vertex_list = mask_fill_flattened(vertex_indices, mask)
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=VERTEX_INDEX,
                    dtype="int8",
                    axis="per_token",
                    final=True,
                    value=vertex_list.reshape(-1).char(),
                )
            ]
        # H5 (TEST): union-find then cast to int (-inf padding -> int32 -2147483648)
        preds = self.task.run_inference(edge_scores, mask).int()
        return [
            OutputField(
                h5_name=VERTEX_INDEX,
                onnx_name=None,
                dtype="i8",
                axis="per_token",
                final=True,
                prefix=self.prefix_vertex_column,
                value=preds,
            )
        ]

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free field metadata mirroring `get_output` for `mode` (``value=None``).

        Returns
        -------
        list[OutputField]
            The value-free serialisation field.
        """
        del run_name
        if mode & Mode.ONNX:
            return [
                OutputField(
                    h5_name=None,
                    onnx_name=VERTEX_INDEX,
                    dtype="int8",
                    axis="per_token",
                    final=True,
                )
            ]
        return [
            OutputField(
                h5_name=VERTEX_INDEX,
                onnx_name=None,
                dtype="i8",
                axis="per_token",
                final=True,
                prefix=self.prefix_vertex_column,
            )
        ]


class RegressionTaskModule(_TaskModuleBase):
    """Scalar/vector regression head.

    Targets are read from labels (``labels.<stream>.<target>``) and the head
    publishes ``preds.<stream>.<name>`` of shape ``[B, R]`` (global) or
    ``[B, L, R]`` (per-token sequence), where ``R = len(targets)``.

    A single scaling method: exactly one of ``target_denominators`` (ratio
    targets), ``norm_params`` (mean/std), or ``scaler`` (per-target functional
    `RegressionTargetScaler`) — or none (raw targets).

    Mode-split de-scaling (same math, different denominator source):

    - **FIT|VAL**: targets and ratio denominators come from
      ``labels.<stream>.<var>``. Predictions publish raw/scaled, loss in
      scaled space.
    - **TEST**: predictions are de-scaled; the ratio denominator is read from
      ``labels.<stream>.<denom>``.
    - **ONNX**: the ratio denominator MUST be a declared input Feature
      (resolved by name at `bind`) since the export graph has no label source
      — a denominator absent from the input Features is a `bind`-time error.

    ``norm_params`` and ``scaler`` need no external source, so de-scaling is
    mode-independent for them.

    Gaussian regression: ``gaussian: true`` emits ``2 * len(targets)`` outputs
    (means ‖ raw variances), a Gaussian NLL loss, and inference returns means +
    ``stddev = sqrt(softplus(var))``. This module publishes ONE ``[B, 2R]``
    array (means then stddevs); the writer owns the ``_stddev`` suffix split.

    Per-sample weighting (``sample_weight:``) and NaN-target masking are
    handled inside ``nan_loss``: invalid (NaN) targets are masked to 0 and the
    loss reduced with ``torch.nanmean``; ``sample_weight`` multiplies the
    per-element loss before the mean. Both require ``loss.reduction == 'none'``.

    The `MultiTarget` processor (row-wise target replacement) is a
    dataset-side sibling (`salt.core.data.processors.MultiTarget`), not a task
    flag — it runs over the label bundle before the task forward.
    """

    def __init__(
        self,
        stream: str,
        targets: str | Sequence[str],
        input: str | None = None,  # noqa: A002 - matches the YAML config key name
        context: str | None = None,
        sequence: bool | None = None,
        target_denominators: str | Sequence[str] | None = None,
        norm_params: Mapping[str, Any] | None = None,
        scaler: Mapping[str, Mapping[str, Any]] | None = None,
        custom_output_names: str | Sequence[str] | None = None,
        gaussian: bool = False,
        sample_weight: str | None = None,
        publish_targets: bool = False,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        expose: Sequence[str] | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        targets : str | Sequence[str]
            Regression target name(s), demanded as
            ``labels.<stream>.<target>``. The head width is ``len(targets)``.
        sequence : bool | None, optional
            Whether the task is per-token, by default inferred from `input`.
        target_denominators : str | Sequence[str] | None, optional
            Per-target ratio denominator variable(s), by default None.
            Mutually exclusive with `norm_params`/`scaler`.
        norm_params : Mapping[str, Any] | None, optional
            ``{"mean": <scalar|list>, "std": <scalar|list>}`` per-target
            mean/std normalisation, by default None. Mutually exclusive.
        scaler : Mapping[str, Mapping[str, Any]] | None, optional
            Per-target functional scaling
            ``{<target>: {op, x_scale, x_off, op_scale, op_off}}``, by default
            None. Mutually exclusive.
        custom_output_names : str | Sequence[str] | None, optional
            Output column suffix(es) overriding the target names, by default
            None. Count-checked against `targets`.
        gaussian : bool, optional
            Compose a mu/sigma head (``output_size = 2 * len(targets)``,
            softplus variance, Gaussian NLL) instead of plain regression, by
            default False. Requires `norm_params` or `target_denominators`
            (its inference has no ``scaler`` branch).
        sample_weight : str | None, optional
            Per-sample loss-reweighting label (FIT|VAL only), by default None.
            Requires ``loss.reduction == 'none'``.
        publish_targets : bool, optional
            Also publish the scaled regression targets under
            ``targets.<stream>.<instance-name>`` in FIT|VAL, for
            `MaskFormerMatchedLoss`'s Hungarian-matcher cost term, by default
            False. A plain regression head leaves this off.
        loss : str | dict[str, Any] | None, optional
            Loss config, by default ``MSELoss`` (``GaussianNLLLoss`` when
            ``gaussian`` is set). A ``nan_regression`` config that expects NaN
            targets must set ``reduction: none``.
        weight : float, optional
            Scalar task-loss weight, by default 1.0.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in, by default all modes.
            ``[fit, val]`` opts a train-only aux task out of TEST/ONNX.

        Raises
        ------
        ConfigError
            On empty targets, a custom-output-name count mismatch, more than
            one scaling method, a ``sample_weight`` with a non-``none`` loss
            reduction, a gaussian head combined with a functional ``scaler`` /
            no scaling method, or a bad `expose` list.
        """
        self.gaussian = bool(gaussian)
        default_loss = _DEFAULT_GAUSS_LOSS if self.gaussian else _DEFAULT_REG_LOSS
        super().__init__(stream, "", input, context, dense, loss, weight, default_loss, expose)
        # regression has no single `label` field: one label demanded per target below
        self.targets = _opt_tuple(targets) or ()
        if not self.targets:
            raise ConfigError(
                "RegressionTaskModule: targets is required and non-empty (design §3.3)"
            )
        self.sequence = sequence if sequence is not None else input is None
        self.target_denominators = _opt_tuple(target_denominators)
        self.norm_params = self._checked_norm_params(norm_params)
        self.scaler_scales = dict(scaler) if scaler is not None else None
        self.custom_output_names = _opt_tuple(custom_output_names)
        self.sample_weight = sample_weight
        self.publish_targets = bool(publish_targets)
        n_methods = sum(
            x is not None for x in (self.target_denominators, self.norm_params, self.scaler_scales)
        )
        if n_methods > 1:
            raise ConfigError(
                f"RegressionTaskModule: only a single scaling method is allowed — set at most "
                f"one of target_denominators/norm_params/scaler (v1 task.py:355), got "
                f"{n_methods}"
            )
        if self.custom_output_names is not None and len(self.custom_output_names) != len(
            self.targets
        ):
            raise ConfigError(
                f"RegressionTaskModule: custom_output_names {list(self.custom_output_names)} "
                f"({len(self.custom_output_names)}) must match targets {list(self.targets)} "
                f"({len(self.targets)}) (v1 task.py:515)"
            )
        if self.target_denominators is not None and len(self.target_denominators) != len(
            self.targets
        ):
            raise ConfigError(
                f"RegressionTaskModule: target_denominators {list(self.target_denominators)} "
                f"({len(self.target_denominators)}) must match targets {list(self.targets)} "
                f"({len(self.targets)}) (v1 task.py:361-366)"
            )
        if self.gaussian and self.scaler_scales is not None:
            raise ConfigError(
                "RegressionTaskModule: a gaussian head cannot use a functional 'scaler' — v1 "
                "GaussianRegressionTask.run_inference has no scaler branch (task.py:753-766); use "
                "norm_params or target_denominators (or none with no inference de-scaling)"
            )
        if self.sample_weight is not None and (
            self.loss_cfg.get("init_args", {}).get("reduction", "mean") != "none"
        ):
            raise ConfigError(
                f"RegressionTaskModule: sample_weight {self.sample_weight!r} requires the loss "
                "reduction to be 'none' — the per-sample weight multiplies the unreduced loss "
                "before nanmean (v1 task.py:379-382,429-435); set loss: "
                "{class_path: torch.nn.MSELoss, init_args: {reduction: none}}"
            )
        # resolved at bind: input-Feature column order, for ONNX to gather denominators by name
        self._input_fields: tuple[str, ...] = ()

    @staticmethod
    def _checked_norm_params(
        norm_params: Mapping[str, Any] | None,
    ) -> dict[str, list[float]] | None:
        """Normalise + validate the ``norm_params`` mapping.

        Returns
        -------
        dict[str, list[float]] | None
            ``{"mean": [...], "std": [...]}`` with both listified, or None.

        Raises
        ------
        ConfigError
            If the mapping is present but lacks ``mean``/``std``.
        """
        if norm_params is None:
            return None
        if set(norm_params) < {"mean", "std"}:
            raise ConfigError(
                f"RegressionTaskModule: norm_params must carry 'mean' and 'std', got "
                f"{sorted(norm_params)} (v1 task.py:349-351)"
            )
        return {
            "mean": [float(x) for x in listify(norm_params["mean"])],
            "std": [float(x) for x in listify(norm_params["std"])],
        }

    @property
    def output_suffixes(self) -> tuple[str, ...]:
        """Per-output column suffixes (custom names override the targets).

        For a gaussian head the suffix list is doubled — the R mean suffixes
        followed by the R ``<suffix>_stddev`` suffixes, index-aligned with the
        published ``[B, 2R]`` array.

        Returns
        -------
        tuple[str, ...]
            R suffixes (plain regression) or 2R suffixes (gaussian: means then
            ``_stddev``), in column order.
        """
        base = self.custom_output_names if self.custom_output_names is not None else self.targets
        if self.gaussian:
            return (*base, *(f"{s}_stddev" for s in base))
        return base

    @property
    def target_label_keys(self) -> tuple[str, ...]:
        """The declared target-label dependencies (one per target).

        Returns
        -------
        tuple[str, ...]
            ``labels.<stream>.<target>`` keys, in target order.
        """
        return tuple(f"labels.{self.stream}.{t}" for t in self.targets)

    @property
    def denom_label_keys(self) -> tuple[str, ...]:
        """The declared ratio-denominator label dependencies (FIT|VAL|TEST source).

        Returns
        -------
        tuple[str, ...]
            ``labels.<stream>.<denom>`` keys, in target order, or empty when
            there are no ratio denominators.
        """
        if self.target_denominators is None:
            return ()
        return tuple(f"labels.{self.stream}.{d}" for d in self.target_denominators)

    @property
    def input_feature_key(self) -> str:
        """The raw-input key carrying the ONNX denominator columns.

        Returns
        -------
        str
            ``inputs.<stream>``.
        """
        return f"inputs.{self.stream}"

    @property
    def weight_label_key(self) -> str:
        """The declared per-sample weight dependency (FIT|VAL only).

        Returns
        -------
        str
            ``labels.<stream>.<sample_weight>`` (only meaningful when
            ``sample_weight`` is set).
        """
        return f"labels.{self.stream}.{self.sample_weight}"

    @property
    def targets_key(self) -> str:
        """The published scaled-targets key (``publish_targets``, FIT|VAL only).

        Returns
        -------
        str
            ``targets.<stream>.<instance-name>`` — the matched-loss matcher's
            scaled-space target source.
        """
        return f"targets.{self.stream}.{self.name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare input/context/masks (+ target/denominator deps) -> preds (+ loss).

        Mode-split denominator dependency: FIT|VAL|TEST demand the
        denominators from ``labels.<stream>.<denom>``; ONNX demands the raw
        ``inputs.<stream>`` Feature tensor. Targets, the per-sample weight,
        and the FIT|VAL loss are TRAINING-gated; predictions are produced in
        ALL modes. A gaussian head publishes a ``2R``-wide prediction (means
        then stddevs).

        Returns
        -------
        IO
            The declared requires/produces.
        """
        n_outputs = 2 * len(self.targets) if self.gaussian else len(self.targets)
        width = sym_dim("D", self.name)
        if self.sequence:
            input_spec = TensorSpec(shape=("B", _stream_len(self.stream), width), dtype="float32")
            label_shape: tuple[int | str, ...] = ("B", _stream_len(self.stream))
            pred_spec = TensorSpec(
                shape=("B", _stream_len(self.stream), n_outputs), dtype="float32"
            )
        else:
            input_spec = TensorSpec(shape=("B", width), dtype="float32")
            label_shape = ("B",)
            pred_spec = TensorSpec(shape=("B", n_outputs), dtype="float32")
        requires: dict[str, TensorSpec] = {self.input_key: input_spec}
        if self.context is not None:
            requires[self.context] = TensorSpec(shape=None, dtype="float32")
        if self.has_pad_mask:
            requires[f"masks.{self.stream}"] = TensorSpec(
                shape=("B", _stream_len(self.stream)), dtype="bool", kind="pad_mask"
            )
        for key in self.target_label_keys:
            requires[key] = TensorSpec(
                shape=label_shape, dtype="float32", kind="label", modes=Mode.TRAINING
            )
        if self.sample_weight is not None:
            requires[self.weight_label_key] = TensorSpec(
                shape=label_shape, dtype="float32", kind="label", modes=Mode.TRAINING
            )
        if self.target_denominators is not None:
            # FIT|VAL|TEST read denominators from labels; ONNX reads them from
            # the raw input Feature tensor instead (no label source at export)
            for key in self.denom_label_keys:
                requires[key] = TensorSpec(
                    shape=label_shape,
                    dtype="float32",
                    kind="label",
                    modes=Mode.FIT | Mode.VAL | Mode.TEST,
                )
            if mode & Mode.ONNX:
                requires[self.input_feature_key] = TensorSpec(
                    shape=("B", sym_dim("F", f"{self.name}.{self.stream}")),
                    dtype="float32",
                    modes=Mode.ONNX,
                )
        produces: dict[str, TensorSpec] = {self.pred_key: self._pred_spec(pred_spec)}
        # a publish_targets head emits NO losses.<name>: MaskFormerMatchedLoss owns
        # losses.regression instead, and emitting both would collide on that key
        if not self.publish_targets:
            produces[self.loss_key] = TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING)
        else:
            # scaled targets for the matched-loss matcher, width R (never doubled,
            # even for a gaussian head)
            tgt_shape: tuple[int | str, ...] = (
                ("B", _stream_len(self.stream), len(self.targets))
                if self.sequence
                else ("B", len(self.targets))
            )
            produces[self.targets_key] = TensorSpec(
                shape=tgt_shape, dtype="float32", modes=Mode.TRAINING
            )
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the composed regression head + resolve the ONNX denominator source.

        ``output_size = len(targets)`` (``2 * len(targets)`` for a gaussian
        head). When ``target_denominators`` is set, every denominator must be
        a declared column of ``inputs.<stream>`` so the ONNX graph can gather
        it by name — otherwise the ONNX de-scaling has no source and `bind`
        raises. A gaussian head with neither ``norm_params`` nor
        ``target_denominators`` also raises here.

        Raises
        ------
        ConfigError
            If a ratio denominator is not a declared input Feature, or a
            gaussian head has no scaling method for inference de-scaling.
        """
        if self.gaussian and self.norm_params is None and self.target_denominators is None:
            raise ConfigError(
                f"RegressionTaskModule {self.name!r}: a gaussian head requires norm_params or "
                "target_denominators — v1 GaussianRegressionTask.run_inference raises without "
                "scaling params (task.py:765-766)"
            )
        init_args = dict(self.loss_cfg.get("init_args", {}))
        loss_module = _loss_class(self.loss_cfg)(**init_args)
        scaler = RegressionTargetScaler(self.scaler_scales) if self.scaler_scales else None
        dense_config = {
            "input_size": schema.width(self.input_key),
            "output_size": (2 if self.gaussian else 1) * len(self.targets),
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
        }
        # the composed task mutates norm_params in place; pass a fresh copy to
        # keep this module's config immutable
        norm_params = (
            {"mean": list(self.norm_params["mean"]), "std": list(self.norm_params["std"])}
            if self.norm_params is not None
            else None
        )
        common = {
            "name": self.name,
            "input_name": self.stream,
            "targets": list(self.targets),
            "target_denominators": (
                list(self.target_denominators) if self.target_denominators is not None else None
            ),
            "norm_params": norm_params,
            "custom_output_names": (
                list(self.custom_output_names) if self.custom_output_names is not None else None
            ),
            "sample_weight": self.sample_weight,
            "loss": loss_module,
            "weight": self.weight,
            "dense_config": dense_config,
        }
        # only the plain head takes the functional `scaler` (gaussian has no
        # scaler branch — guarded at __init__)
        self.task = (
            _AbsorbedGaussianRegressionTask(**common)
            if self.gaussian
            else _AbsorbedRegressionTask(scaler=scaler, **common)
        )
        if self.target_denominators is not None:
            self._input_fields = schema.fields_of(self.input_feature_key)
            features = set(self._input_fields)
            if missing := [d for d in self.target_denominators if d not in features]:
                raise ConfigError(
                    f"RegressionTaskModule {self.name!r}: ratio denominators {missing} are not "
                    f"declared columns of {self.input_feature_key!r} ({sorted(features)}) — the "
                    f"ONNX export graph de-scales from the input Feature tensor "
                    f"(to_onnx.py:381-398), so a denominator must be an input variable. Add it "
                    f"to data.modules.features.init_args.variables.{self.stream}, or drop the "
                    f"ratio target (FD §3.3 mode-split de-scaling)"
                )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the composed head; publishes RAW (scaled) preds in EVERY non-training mode.

        The head is handed a single-stream targets dict; the per-sample weight
        (when configured) rides in that dict so ``nan_loss`` finds it.

        The eval de-scaling (incl. the gaussian means‖stddev concat) is owned
        by ``get_output`` on the live path and by ``get_h5`` on the oracle
        path, both of which read this raw leaf and de-scale exactly once. The
        mode-split denominator source (labels in TEST, the input Feature by
        name in ONNX) lives on ``get_output``/``get_h5``.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        assert self.task is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        # objects-stream (query-bank) heads have no pad mask
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        pad_masks = {self.stream: mask} if self.has_pad_mask else None
        if mode & Mode.TRAINING:
            targets_dict = {
                self.stream: {
                    t: b.get(k) for t, k in zip(self.targets, self.target_label_keys, strict=True)
                }
            }
            for denom, key in zip(
                self.target_denominators or (), self.denom_label_keys, strict=True
            ):
                targets_dict[self.stream][denom] = b.get(key)
            if self.sample_weight is not None:
                targets_dict[self.stream][self.sample_weight] = b.get(self.weight_label_key)
            preds, loss = self.task(x, targets_dict, pad_masks, context=ctx)
            if self.publish_targets:
                # matched-loss feature head: publish preds + scaled targets for the
                # matcher; the standalone loss is discarded (MaskFormerMatchedLoss
                # owns losses.regression instead)
                return {
                    self.pred_key: preds,
                    self.targets_key: self.task.get_targets(targets_dict),
                }
            return {self.pred_key: preds, self.loss_key: loss}
        preds, _ = self.task(x, {}, pad_masks, context=ctx)
        return {self.pred_key: preds}

    def _descale_source(self, b: Bundle, mode: Mode) -> dict[str, dict[str, Tensor]]:
        """Build the per-denominator de-scaling source dict (mode-split).

        ``run_inference`` reads ``labels[input_name][denom]``, so this returns
        that exact nesting — sourced from the label group in TEST and gathered
        by name from the raw input Feature tensor in ONNX (the only
        export-time source).

        Returns
        -------
        dict[str, dict[str, Tensor]]
            ``{stream: {denom: tensor}}`` for every ratio denominator.
        """
        assert self.target_denominators is not None
        if mode & Mode.ONNX:
            columns = b.get(self.input_feature_key)
            field_index = {name: i for i, name in enumerate(self._input_fields)}
            return {
                self.stream: {
                    denom: columns[..., field_index[denom]] for denom in self.target_denominators
                }
            }
        return {
            self.stream: {
                denom: b.get(key)
                for denom, key in zip(self.target_denominators, self.denom_label_keys, strict=True)
            }
        }

    def _descaled_preds(self, b: Bundle, mode: Mode) -> Tensor:
        """De-scale the RAW ``preds.*`` leaf to physical values.

        Reads the RAW ``preds.*`` leaf + the mode-split denominator source
        (labels in TEST, the input Feature by name in ONNX) + the pad mask,
        and runs ``self.task.run_inference``. A gaussian head's
        ``run_inference`` returns a ``(means, stds)`` tuple, re-concatenated
        here to ONE ``[..., 2R]`` array. Both ``get_h5`` and ``get_output``
        call this so the de-scaling happens exactly once and is identical
        between the two.

        Returns
        -------
        Tensor
            The de-scaled physical predictions: ``[..., R]`` (plain) or
            ``[..., 2R]`` (gaussian, means ‖ stds).
        """
        assert self.task is not None, "de-scale before bind()"
        # clone before the in-place de-scale: run_inference mutates preds[..., i]
        # in place, so a bare b.get(...) would corrupt the bundle's RAW preds.*
        # leaf and double-de-scale on any later read
        preds = b.get(self.pred_key).float().clone()
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        labels = self._descale_source(b, mode) if self.target_denominators is not None else None
        descaled = self.task.run_inference(preds, labels=labels, pad_mask=mask)
        if self.gaussian:
            # run_inference returns (means, stds); publish ONE [..., 2R] array
            # (means ‖ stds) for the single-leaf graph contract
            means, stds = descaled
            return torch.cat([means, stds], dim=-1)
        return descaled

    # -- output rendering ---------------------------------------------------

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """One ``f4`` column per output, named ``{run_name}_{suffix}``.

        The suffixes ARE `output_suffixes` (``custom_output_names`` else the
        targets; doubled for a gaussian head — R means then R ``_stddev``),
        the same suffixes the ONNX manifest carries.

        Returns
        -------
        list[tuple[str, str]]
            ``(column, "f4")`` pairs, one per output, in column order.
        """
        return [(f"{run_name}_{suffix}", "f4") for suffix in self.output_suffixes]

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """The de-scaled values as ``f4`` columns.

        Returns
        -------
        np.ndarray
            ``[B]`` (global) or ``[B, L]`` (sequence) structured array.
        """
        preds = self._descaled_preds(b, Mode.TEST)
        dtype = np.dtype(self.output_names(run_name))
        return u2s(preds.float().cpu().numpy(), dtype)

    def onnx_outputs(self) -> list[ExportOutput]:
        """The legacy ``split_scalars`` manifest entry — retired on the export path.

        No longer drives a real ONNX export (the live ONNX de-scale lives on
        ``get_output`` instead, wired via an ``OnnxExportSink``); a regression
        config without an ``OnnxExportSink`` cannot ONNX-export at all. Kept
        for manifest-gate assertions only.

        Returns
        -------
        list[ExportOutput]
            One ``split_scalars`` entry (the legacy manifest API; export-retired).
        """
        return [
            ExportOutput(
                port=self.pred_key,
                names=list(self.output_suffixes),
                reduce="split_scalars",
            )
        ]

    def output_time_requires(self, mode: Mode) -> list[str]:
        """The non-pred deps `get_output` reads: denom source(s) + pad mask, mode-aware.

        - a ratio head (``target_denominators``) reads its denominator source —
          ``labels.<stream>.<denom>`` in FIT|VAL|TEST, the raw input Feature
          ``inputs.<stream>`` (gathered by name) in ONNX.
        - a padded sequence head reads the stream pad mask ``masks.<stream>``
          (the ``run_inference`` NaN-fill). A global / objects-stream head
          needs no pad mask.

        ``norm_params`` / ``scaler`` need no external source (mode-independent).

        Returns
        -------
        list[str]
            The dotted bundle keys `get_output` reads at output time, in order.
        """
        deps: list[str] = []
        if self.target_denominators is not None:
            if mode & Mode.ONNX:
                deps.append(self.input_feature_key)
            else:
                deps.extend(self.denom_label_keys)
        if self.has_pad_mask:
            deps.append(f"masks.{self.stream}")
        return deps

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """De-scale the RAW preds into graph-visible `OutputField`s.

        Reads the RAW ``preds.*`` leaf + the mode-split denominator source +
        the pad mask, and de-scales in TRACEABLE torch ops via
        ``_descaled_preds``. Mints ONE ``f4`` `OutputField` per output column
        (``output_suffixes``, doubled for a gaussian head). H5 and ONNX share
        the same suffixes (``onnx_name`` defaults to ``h5_name``). The
        per-column value shape is mode-faithful to the live sink:

        - **H5 modes** (TEST): the per-column ``[B]`` (global) / ``[B, L]``
          (seq) value. No squeeze.
        - **ONNX**: squeezed to drop the size-1 batch dim — a global head's
          ``[1, R]`` de-scaled preds give a rank-0 scalar per column; a
          per-token head's ``[1, L, R]`` gives a ``[L]`` per-token vector.

        ``run_name`` is not baked in (the sink prefixes it).

        Returns
        -------
        list[OutputField]
            One ``f4`` field per output column (de-scaled physical value), in
            column order.
        """
        del run_name
        preds = self._descaled_preds(b, mode)
        axis = "per_token" if self.sequence else "global"
        squeeze = bool(mode & Mode.ONNX)
        return [
            OutputField(
                h5_name=suffix,
                dtype="f4",
                axis=axis,
                final=True,
                value=preds[..., i].squeeze() if squeeze else preds[..., i],
            )
            for i, suffix in enumerate(self.output_suffixes)
        ]

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free field metadata mirroring `get_output` for `mode` (``value=None``).

        One ``f4`` field per ``output_suffixes`` entry (gaussian-doubled);
        H5/ONNX suffixes are identical, so each field carries both.

        Returns
        -------
        list[OutputField]
            The value-free serialisation fields, in column order.
        """
        del run_name, mode
        axis = "per_token" if self.sequence else "global"
        return [
            OutputField(h5_name=suffix, dtype="f4", axis=axis, final=True)
            for suffix in self.output_suffixes
        ]


class _OriginWeightedVertexing(_AbsorbedVertexingTask):
    """`_AbsorbedVertexingTask` with config-driven heavy/fake origin ids.

    With the default ids (3,4,5 / 1), `get_weights` is bit-identical to the
    hardcoded base version.
    """

    def __init__(self, heavy_ids: Sequence[int], fake_ids: Sequence[int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._heavy_ids = tuple(heavy_ids)
        self._fake_ids = tuple(fake_ids)

    def get_weights(self, labels: Tensor, adjmat: Tensor) -> Tensor:
        """Compute per-edge weights from configured origin ids.

        Returns
        -------
        Tensor
            Per-edge weights of shape ``[E]`` after adjacency compression.
        """
        heavy = torch.clip(sum(labels == i for i in self._heavy_ids), 0, 1)
        fake = torch.clip(sum(labels == i for i in self._fake_ids), 0, 1)
        weights = heavy - fake.int()
        weights = weights.unsqueeze(-1) & weights.unsqueeze(-2)
        weights = weights[adjmat]
        return 1 + weights


def _opt_tuple(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Listify a scalar-or-list config value to a tuple, preserving ``None``.

    Returns
    -------
    tuple[str, ...] | None
        The listified tuple, or None when the input is None.
    """
    if value is None:
        return None
    return tuple(listify(value))


def _is_name_weighting(heavy: Sequence[Any], fake: Sequence[Any]) -> bool:
    """Whether an ``origin_weighting`` config is class-name based.

    Name-based iff ANY heavy/fake entry is a string. A list mixing strings and
    ints is rejected here so the caller fails at construction, not silently
    half-resolved.

    Returns
    -------
    bool
        True for a (consistent) name-based config; False for all-integer ids.

    Raises
    ------
    ConfigError
        If any single role list mixes integer ids with class names.
    """
    for role, entries in (("heavy", heavy), ("fake", fake)):
        has_name = any(isinstance(e, str) for e in entries)
        has_id = any(not isinstance(e, str) for e in entries)
        if has_name and has_id:
            raise ConfigError(
                f"VertexingTaskModule: origin_weighting {role!r} list mixes integer ids with "
                f"class names ({list(entries)!r}) — use one or the other (design §5.1)"
            )
    return any(isinstance(e, str) for e in (*heavy, *fake))


def _coerce_origin_ids(entries: Sequence[Any], role: str) -> tuple[int, ...]:
    """Coerce an all-integer ``origin_weighting`` role list to a tuple of ids.

    Returns
    -------
    tuple[int, ...]
        The integer origin ids, in config order.

    Raises
    ------
    ConfigError
        On a non-integer entry (e.g. a float) — bool is rejected too (an origin
        id is never True/False).
    """
    ids: list[int] = []
    for e in entries:
        if isinstance(e, bool) or not isinstance(e, int):
            raise ConfigError(
                f"VertexingTaskModule: origin_weighting {role!r} entries must be INTEGER origin "
                f"ids or class NAMES (got {e!r} in {list(entries)!r}); v1's hardcoded ids are "
                "heavy: [3, 4, 5], fake: [1] (design §5.1)"
            )
        ids.append(int(e))
    return tuple(ids)


def _parse_expose(expose: Sequence[str] | None, cls: str) -> Mode:
    """Parse a task ``expose:`` mode-name list into a `Mode` flag.

    ``None`` (the default) means all modes. A list of mode names
    (case-insensitive, ``fit``/``val``/``test``/``onnx``) gates the prediction
    port to exactly those modes: a train-only aux task uses
    ``expose: [fit, val]`` so its prediction is pruned from the TEST/ONNX plans.

    Returns
    -------
    Mode
        The exposed-modes flag (``Mode.ALL`` for the default).

    Raises
    ------
    ConfigError
        On a non-list value, an empty list, or an unknown mode name.
    """
    if expose is None:
        return Mode.ALL
    if isinstance(expose, str) or not isinstance(expose, Sequence):
        raise ConfigError(
            f"{cls}: expose must be a list of mode names (e.g. [fit, val]), got {expose!r} "
            "(design §4.2)"
        )
    if not expose:
        raise ConfigError(
            f"{cls}: expose may not be an empty list — a task exposed in no mode is dead; "
            "omit expose for all modes, or remove the task (design §4.2)"
        )
    valid = {m.name.lower(): m for m in (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX)}
    modes = Mode.FIT & Mode.TEST  # empty seed (no mode); accumulate the named ones
    for raw in expose:
        if not isinstance(raw, str) or raw.lower() not in valid:
            raise ConfigError(
                f"{cls}: unknown expose mode {raw!r} — valid modes are "
                f"{sorted(valid)} (design §4.2)"
            )
        modes |= valid[raw.lower()]
    return modes


def _checked_weight_source(weight_source: Mapping[str, str] | None) -> dict[str, str] | None:
    """Validate a ``weight_source`` mapping.

    Returns
    -------
    dict[str, str] | None
        The validated mapping, or None.

    Raises
    ------
    ConfigError
        If the mapping is not exactly ``{"from_class_dict": <path>}``.
    """
    if weight_source is None:
        return None
    if set(weight_source) != {"from_class_dict"} or not isinstance(
        weight_source["from_class_dict"], (str, Path)
    ):
        raise ConfigError(
            f"weight_source must be {{'from_class_dict': <path>}}, got {dict(weight_source)!r} "
            "(design §3.3)"
        )
    return {"from_class_dict": str(weight_source["from_class_dict"])}


def _loss_cfg(loss: str | dict[str, Any] | None, default: dict[str, Any]) -> dict[str, Any]:
    """Normalise a loss config to ``{class_path, init_args}`` form.

    Returns
    -------
    dict[str, Any]
        The normalised config (a fresh dict).

    Raises
    ------
    ConfigError
        On malformed configs.
    """
    if loss is None:
        cfg: dict[str, Any] = {k: dict(v) if isinstance(v, dict) else v for k, v in default.items()}
        return cfg
    if isinstance(loss, str):
        return {"class_path": loss if "." in loss else f"torch.nn.{loss}"}
    if isinstance(loss, Mapping) and "class_path" in loss:
        return {
            "class_path": str(loss["class_path"]),
            "init_args": dict(loss.get("init_args", {})),
        }
    raise ConfigError(
        f"loss config must be a torch.nn class name or {{class_path, init_args}}, got {loss!r}"
    )


def _loss_class(cfg: Mapping[str, Any]) -> type[nn.Module]:
    """Resolve a loss ``class_path`` to its class (config-only, no instantiation).

    Returns
    -------
    type[nn.Module]
        The loss class.

    Raises
    ------
    ConfigError
        If the path does not resolve to an ``nn.Module`` subclass.
    """
    path = cfg["class_path"]
    module_path, _, cls_name = path.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError, ValueError) as err:
        raise ConfigError(f"cannot resolve loss class_path {path!r}: {err}") from err
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ConfigError(f"loss class_path {path!r} is not an nn.Module subclass")
    return cls
