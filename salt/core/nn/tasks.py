"""Config-constructed task modules for the GN2v2 surface (design §3.3, §5.1, §9.2).

M2 porting policy (plan 05): each task module composes a FRESH v1 task head
(`salt.models.task.ClassificationTask` / `VertexingTask`) built at `bind`
from the resolved schema — loss math (ignore_index=-1, the label ``-2`` pad
fold, origin-weighted vertexing) stays verbatim v1; full absorption is M7.
What is new: the declared label/mask/context dependencies, the two-phase
bind, the per-mode output semantics, and the declarative class-weight source.

Per design §3.3, tasks consume PER-STREAM tensors produced by an explicit
`Split` (``encoded.<stream>``) — the v1 ``input_name_mask`` reconstruction
from pad-mask dict order (task.py:58-78) is gone. The composed v1 head is
therefore handed single-stream dicts, under which its internal slicing is
the identity; outputs are mathematically equal to v1's full-sequence path
but NOT guaranteed bitwise (the M2 gates use 1e-6/curve criteria, plan 05).

Mode semantics (design §3.3, P1.5 flip): ``preds.<stream>.<task>`` is
published in ALL modes — raw logits / raw edge scores / scaled targets.
For the CLASSIFICATION family the eval conversion (softmax / padded-aware
softmax) has been REMOVED from ``forward`` in TEST (the P1.5 flip, design §2
"no per-task forward branching"): classification ``forward`` now publishes
RAW logits in FIT|VAL|TEST, and the conversion is owned by the
``ClassProbs``/``SeqClassProbs`` producers (live path) and by the task's
``get_h5`` (the transitional M4.5 oracle path) — both read the raw leaf and
``run_inference`` it once. ONNX is the carve-out (deferred to P4): the
classification ``forward`` still publishes converted probs in ONNX so the
export trace + tests are unchanged. Regression / vertexing / maskformer are
NOT yet flipped (P2/P3): their TEST ``forward`` keeps converting (de-scale /
union-find) and their ``get_h5`` reads the already-converted leaf. Vertexing
remains the documented per-family exception: TEST publishes per-node vertex
assignments (union-find, v1 writer semantics, task.py:985-986,1003); ONNX
publishes RAW edge scores so the export-side ``vertex_union_find`` reduce
owns the in-graph union-find (v1 placement, to_onnx.py:426-431). Labels are
required and ``losses.<task>`` produced in FIT|VAL only.
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
from salt.core.writers.names import VERTEX_INDEX, pascal_case

__all__ = ["ClassificationTaskModule", "RegressionTaskModule", "VertexingTaskModule"]

_UNNAMED = "unnamed"
_WIDTH_KEYS = ("input_size", "output_size", "context_size")

# Streams that are a fixed-count query bank rather than a variable-length,
# pad-masked constituent sequence. The MaskFormer ``objects`` stream is M
# learnable object queries from the `MaskDecoder` (`objects.embed` is
# ``[B, M, D]`` with M fixed) — there is NO padding and so NO ``masks.objects``
# pad mask anywhere in the graph. A sequence-mode task on such a stream must NOT
# require/consume a pad mask, mirroring v1 ``RegressionTask.forward`` /
# ``ClassificationTask.forward`` which EXEMPT the objects stream (task.py:547:
# ``if pad_masks is not None and self.input_name != "objects"``). Without this,
# an objects-stream sequence head demands ``masks.objects`` and fails graph
# validation with a ConnectivityError (no module produces it).
_NO_PAD_MASK_STREAMS = frozenset({"objects"})

_DEFAULT_CLS_LOSS: dict[str, Any] = {"class_path": "torch.nn.CrossEntropyLoss"}
_DEFAULT_VTX_LOSS: dict[str, Any] = {
    "class_path": "torch.nn.BCEWithLogitsLoss",
    "init_args": {"reduction": "none"},
}
_DEFAULT_REG_LOSS: dict[str, Any] = {"class_path": "torch.nn.MSELoss"}
_DEFAULT_GAUSS_LOSS: dict[str, Any] = {"class_path": "torch.nn.GaussianNLLLoss"}


# ===========================================================================
# Absorbed v1 task-head family (M7 W2c-1) — v2-native, math copied VERBATIM
# from ``salt.models.task`` (the read-only gate ORACLE). The v2 ``*TaskModule``
# classes below compose these v2-native heads at ``bind`` (was: the v1
# ``salt.models.task.*`` classes); ``_OriginWeightedVertexing`` subclasses the
# v2-native ``_AbsorbedVertexingTask``. The v1 originals are NEVER edited — the
# G3 / M5 / parity_gn2 gates still import ``salt.models.task`` to compare
# BITWISE, so this family reproduces the v1 forward/loss/inference EXACTLY:
# origin-weighting (v1 task.py:957-964), the ignore_index=-1 / ``-2`` label fold
# (task.py:218-227), ``nan_loss`` (task.py:384-440), target scaling
# (task.py:442-602) and the Gaussian/Classification/Vertexing specifics.
# ===========================================================================


def _add_dims(x: Tensor, ndim: int) -> Tensor:
    """Add singleton dims after the batch dim to reach ``ndim`` (v1 tensor_utils.py:168-198).

    Inlined byte-faithfully from v1 ``salt.utils.tensor_utils.add_dims`` — the
    ``masked_softmax`` mask-broadcast helper the classification ``run_inference``
    relies on (sequence-mode padded softmax).

    Returns
    -------
    Tensor
        ``x`` reshaped with the added singleton dimensions.

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
    """Softmax ignoring padded elements (v1 tensor_utils.py:74-107, byte-faithful).

    Inlined VERBATIM from v1 ``salt.utils.tensor_utils.masked_softmax`` — the
    padded-aware softmax the absorbed ``ClassificationTask.run_inference`` uses
    for sequence (per-token) heads (v1 task.py:263). Elements where ``mask`` is
    ``True`` are set to ``-inf`` before the softmax and zeroed after.

    Returns
    -------
    Tensor
        Tensor after the masked softmax.
    """
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
    """Unflatten a per-node array back to a batch-shaped tensor (v1 task.py:1009-1035).

    Inlined VERBATIM from v1 ``salt.models.task.mask_fill_flattened`` — the
    ``@torch.jit.script`` decorator is PRESERVED (the scripted loop semantics are
    load-bearing for vertexing inference parity). Padded positions read ``-inf``.

    Returns
    -------
    Tensor
        Filled tensor of shape ``[B, L, F]``; padded positions set to ``-inf``.
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
    """Absorbed v1 ``TaskBase`` (task.py:20-78) — v2-native, math verbatim.

    Wraps a `Dense` head (the W2c-2 v2-native absorbed ``salt.core.nn.modules.Dense``), a loss, an
    ``input_name`` stream tag and a scalar ``weight``. ``input_name_mask`` is the
    v1 cross-stream selection helper; the v2 modules hand SINGLE-STREAM dicts so
    it is the identity, but it is reproduced verbatim for bitwise parity.
    """

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
        """Boolean mask selecting tokens from ``self.input_name`` (v1 task.py:58-78).

        Returns
        -------
        Tensor
            Boolean mask of shape ``[L]`` (concatenated across streams), ``True``
            for positions belonging to ``self.input_name``.
        """
        return torch.cat(
            [
                torch.ones(m.shape[1], device=m.device) * (1 if (t == self.input_name) else 0)
                for t, m in pad_masks.items()
            ],
        ).bool()


class _AbsorbedClassificationTask(_AbsorbedTaskBase):
    """Absorbed v1 ``ClassificationTask`` (task.py:81-301) — v2-native, math verbatim.

    ``class_names`` is REQUIRED (the v2 modules always pass it; the v1
    ``CLASS_NAMES`` h5-attr fallback is intentionally dropped — design §3.3).
    The forward reproduces the ignore_index=-1 default, the label-map remap, and
    the ``-2`` pad fold (task.py:218-227); ``run_inference`` reproduces the
    sigmoid / softmax / padded-softmax branch (task.py:240-264).
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
        """Apply per-sample weights to a loss tensor if configured (v1 task.py:153-170).

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
        """Compute logits and classification loss (v1 task.py:172-238, verbatim).

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted logits and the loss (``None`` when no labels).
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
        """Convert logits to probabilities (v1 task.py:240-264, verbatim).

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
    """Absorbed v1 ``RegressionTaskBase`` (task.py:304-482) — v2-native, verbatim.

    Owns the single-scaling guard, ``nan_loss`` (NaN-target masking +
    per-sample weighting + ``torch.nanmean``, task.py:384-440) and ``get_targets``
    (stack + scaler/denominator/norm_params scaling, task.py:442-482).
    """

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
        """Loss that ignores NaN targets (v1 task.py:384-440, verbatim).

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
        """Assemble and scale regression targets (v1 task.py:442-482, verbatim).

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

            # We stack targets dict always over the first dimension to allow consistency
            # when scaling, but for queries we want the regression target to be in the final
            # dimension. This allows us to keep the same code for both global and query scaling
            if len(targets.shape) == 3:
                targets = targets.transpose(1, 2)

        # MFU-1 (upstream task.py RegressionTaskBase.get_targets): robust target
        # sanitisation. Runs here -- BEFORE forward's pad-mask NaN fill (forward
        # line ~524) and before nan_loss (line ~412). The two NaN mechanisms act on
        # DISJOINT sets and do not double-count:
        #   * nan_to_num zeroes NaN/inf in the *data* targets (treats them as valid
        #     0.0 targets), matching upstream's behaviour on dirty inputs.
        #   * nan_loss's isnan-masking removes the PADDING entries, which forward
        #     fills with NaN AFTER get_targets returns -- so they are still NaN at
        #     loss time and remain correctly masked out.
        # No-op on clean data: real targets are finite, so nan_to_num leaves them
        # unchanged; padding NaNs are added downstream and untouched here.
        if targets is not None:
            targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        return targets


class _AbsorbedRegressionTask(_AbsorbedRegressionTaskBase):
    """Absorbed v1 ``RegressionTask`` (task.py:485-642) — v2-native, verbatim.

    Plain regression head: ``output_size == len(targets)``; forward fills padded
    targets with NaN before ``nan_loss``; ``run_inference`` inverts the scaling
    (denominator / norm_params / scaler, task.py:567-602).
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
        """Compute regression predictions and loss (v1 task.py:519-565, verbatim).

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

        # fill targets with nan where invalid to remove them from loss calculation
        if pad_mask is not None and targets is not None:
            targets = torch.masked_fill(targets, pad_mask.unsqueeze(-1), torch.nan)

        loss: Tensor | None = None
        if targets is not None:
            loss = self.nan_loss(preds, targets, targets_dict) * self.weight

        return preds, loss

    def run_inference(
        self, preds: Tensor, labels: Mapping | None = None, pad_mask: Tensor | None = None
    ) -> Tensor:
        """Invert target scaling to the original space (v1 task.py:567-602).

        plan 34 W34.3 critic-fix (per-token axis correctness): the de-scale loops
        now index the LAST (target-channel) axis ``preds[..., i]`` instead of v1's
        first axis ``preds[:, i]``. For a GLOBAL ``[B, R]`` head ``[..., i]`` is
        bit-identical to v1's ``[:, i]`` (axis 1 IS the channel axis) — so every
        shipped global head is byte-unchanged. For a per-token ``[B, L, R]`` head
        v1's ``[:, i]`` addressed TOKEN ``i`` (scaling tokens 0..R-1, leaving later
        tokens raw, and crashing at ``L == 0``); ``[..., i]`` correctly de-scales
        column ``i`` of EVERY token, matching the ``RegressionDescaleOp`` producer
        (``producers.py:907-915``) so the producer / get_output / get_h5 paths
        de-scale identically on a per-token head. The scaler branch already used the
        per-token-faithful trailing index — it is folded into the same ``[..., i]``
        form (global-safe: a scaler is only configured on per-token MaskFormer-style
        heads today, but ``[..., i]`` works for both ranks).

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

        # apply mask if available
        if pad_mask is not None:
            preds = torch.masked_fill(preds, pad_mask.unsqueeze(-1), np.nan)

        return preds


class _AbsorbedGaussianRegressionTask(_AbsorbedRegressionTaskBase):
    """Absorbed v1 ``GaussianRegressionTask`` (task.py:645-821) — v2-native, verbatim.

    Mu/sigma head: ``output_size == 2 * len(targets)``; softplus variance, Gaussian
    NLL loss, and ``run_inference`` returning de-scaled ``(means, stds)`` with
    ``stddev = sqrt(softplus(var))`` (task.py:729-774).
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
        """Compute mean/variance predictions and Gaussian NLL loss (v1 task.py:677-727).

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

        # split outputs into means and sigmas
        means, variances = preds.tensor_split(2, -1)
        variances = nn.functional.softplus(variances)  # ensure positiveness of variance

        # fill targets with nan where padded to remove them from loss calculation
        if pad_mask is not None and targets is not None:
            targets = torch.masked_fill(targets, pad_mask.unsqueeze(-1), torch.nan)

        loss: Tensor | None = None
        if targets is not None:
            loss = self.nan_loss(means, targets, targets_dict, var=variances) * self.weight

        return preds, loss

    def run_inference(
        self, preds: Tensor, labels: Mapping | None = None, pad_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Invert scaling for means + (sqrt of) variances (v1 task.py:729-774).

        plan 34 W34.3 critic-fix (per-token axis correctness): the de-scale loops
        index the LAST axis ``preds[..., i]`` / ``preds[..., i + 1]`` instead of
        v1's first axis ``preds[:, i]`` / ``preds[:, i + 1]``. For a GLOBAL ``[B, 2R]``
        head this is bit-identical to v1 (axis 1 IS the 2R channel axis), so the
        shipped global gaussian heads (Dipz/hitz/regression_gaussian global) are
        byte-unchanged. For a per-token ``[B, L, 2R]`` head v1's first-axis index
        addressed TOKEN 0/1 (corrupting track data, never reaching the stddev
        channel, and crashing at ``L == 0``); the trailing index correctly de-scales
        column ``i`` (mean) / ``i + 1`` (var) of EVERY token, matching the
        ``RegressionDescaleOp._convert_gaussian`` producer (``producers.py:959-970``)
        so the producer / get_output / get_h5 paths agree on a per-token gaussian
        head. The ``i`` / ``i + 1`` index pairing is the v1 means‖vars layout for an
        R=1 head (all shipped gaussian heads are R=1) — preserved verbatim (the
        producer shares the same pairing), only the AXIS is corrected.

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

        # apply mask if available
        if pad_mask is not None:
            means = torch.masked_fill(means, pad_mask.unsqueeze(-1), np.nan)
            stds = torch.masked_fill(stds, pad_mask.unsqueeze(-1), np.nan)

        return means, stds


class _AbsorbedVertexingTask(_AbsorbedTaskBase):
    """Absorbed v1 ``VertexingTask`` (task.py:824-1005) — v2-native, verbatim.

    Edge-classification vertexing: builds the compressed track-track matrix,
    runs the dense head per edge, weights the per-edge BCE by origin labels
    (``get_weights``, task.py:949-967), and ``run_inference`` returns per-node
    union-find assignments (task.py:969-986). ``_OriginWeightedVertexing``
    subclasses this v2-native base and overrides ``get_weights``.
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
        """Compute pair classification for vertexing and its loss (v1 task.py:839-902).

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
        """Compute the vertexing loss against pairwise matching labels (v1 task.py:904-947).

        Returns
        -------
        Tensor
            Weighted average loss scaled by ``self.weight``.
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
        """Per-edge weights from origin labels (v1 task.py:949-967, hardcoded 3,4,5/1).

        ``_OriginWeightedVertexing`` overrides this with config-driven heavy/fake
        ids; with the v1 defaults the two are bit-identical.

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
        """Per-node assignments from edge predictions (v1 task.py:969-986, verbatim).

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
        input: str | None,  # noqa: A002 - design §3.3 YAML surface name
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
        """Apply the ``expose:`` opt-out to a prediction port spec (design §4.2).

        The ``preds.*`` port is active only in the configured ``expose`` modes
        (default: all modes). A train-only aux task (``expose: [fit, val]``)
        therefore produces NO prediction in TEST/ONNX, so the planner prunes it
        from those plans and the TEST dead-preds hard error never fires — the
        real opt-out the dead-preds message points users at, replacing the
        ``--model.modules.X=null`` workaround.

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
        """The published prediction key (design §3.3).

        Returns
        -------
        str
            ``preds.<stream>.<instance-name>``.
        """
        return f"preds.{self.stream}.{self.name}"

    @property
    def loss_key(self) -> str:
        """The published loss key, auto-collected by `LossSum` (design §3.3).

        Returns
        -------
        str
            ``losses.<instance-name>``.
        """
        return f"losses.{self.name}"

    @property
    def label_key(self) -> str:
        """The declared label dependency (demand-driven `Labels`, design §3.3).

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
        non-sequence head OR a fixed-count query bank (the MaskFormer
        ``objects`` stream — M learnable queries with no padding, so no
        ``masks.objects`` exists). Mirrors v1's objects-stream exemption
        (task.py:547 ``self.input_name != "objects"``): such a head neither
        declares nor consumes ``masks.<stream>``.

        Returns
        -------
        bool
            ``self.sequence and self.stream not in _NO_PAD_MASK_STREAMS``.
        """
        return self.sequence and self.stream not in _NO_PAD_MASK_STREAMS

    # -- output rendering: TEST columns + values, ONNX manifest (mirrors v1) ----
    #
    # The per-family rendering knowledge lives on the task — v1's `output_names`
    # / `get_h5` / `get_onnx` placement (task.py). `TaskWriter` is pure
    # orchestration: it groups, prefixes-by-stream, pads, and writes; it never
    # branches on the task family. A family that ships no rendering inherits
    # these guards, so a writer-role task with no representation raises a loud
    # `ConfigError` at the point the writer asks for its columns / export entry
    # (the unsupported-family error, design §8 — moved here from the writer).

    onnx_renameable: bool = False
    """Whether `TaskWriter` ``onnx_names`` may override this task's ONNX suffix.

    Classification is renameable (overlapping class names across two
    classification tasks is the collision-fix path, amendment merge condition
    6). Vertexing's suffix is the shared cross-mode `VERTEX_INDEX` constant and
    regression's suffixes ARE its ``custom_output_names`` — both reject the
    writer-side ``onnx_names`` rename (single ownership, amendment §2.2). The
    family owns this naming policy alongside its rendering; the writer reads it
    instead of branching on the task type.
    """

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """The TEST column schema for this task — ``(column_name, np_dtype_str)``.

        Mirrors v1's per-family ``output_names`` property plus the column
        dtype the writer's old ``_task_descr`` carried. The ``run_name``
        prefix matches the existing data flow (v1's ``model_name`` column
        prefix, task.py:140-151); a bare property cannot see it. The
        per-family override returns the same names+formats the writer used to
        derive in its isinstance ladder, so the eval byte-schema is unchanged.
        The base raises for a task family that ships no TEST rendering (the
        unsupported-family guard, moved here from `TaskWriter`).

        Parameters
        ----------
        run_name : str
            The run ``name:`` — the TEST column prefix (v1 ``model_name``).

        Raises
        ------
        ConfigError
            For a task family that ships no TEST rendering.
        """
        del run_name
        raise ConfigError(self._no_render_msg("TEST columns"))

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Render this task's formatted TEST values as a structured array (v1 ``get_h5``).

        Mirrors v1's per-family ``get_h5`` (task.py:266/604/988): reads the
        task's published ``preds.*`` leaf from `b` and renders it to the
        structured array whose dtype is exactly
        ``np.dtype(self.output_names(run_name))`` — the FINAL run-name-prefixed
        field names, so the values line up with the columns `columns` declares.
        (The v1 per-family bundle inputs — pad mask, de-scaling labels — were
        consumed by the task's TEST forward, which publishes the already
        converted/de-scaled values, so this only needs the ``preds.*`` leaf.)
        The writer owns padding to the file sequence length and the I/O; this
        is the value-formatting the writer's ``write``/``_convert`` used to do.
        The base raises for a task family that ships no TEST rendering.

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
        """Render this task's ONNX-manifest entries (mirrors v1 ``get_onnx`` + naming).

        One `ExportOutput` per family entry, referencing a shipped reduce by
        key (no change to the reduce registry): global classification ->
        ``split_scalars`` per-class suffixes; sequence classification -> one
        ``argmax`` int8 entry; vertexing -> ``vertex_union_find`` int8 on the
        shared `VERTEX_INDEX` constant; regression -> ``split_scalars``
        per-target suffixes. The exporter prepends ``{model_name}_`` to the
        suffixes (amendment §5 rule 3) — the task carries them bare, the SAME
        suffixes its TEST columns use.

        `TaskWriter` decides WHICH tasks export (the ``onnx``/``onnx_tasks``/
        ``onnx_streams`` narrowing, the global-before-sequence emission order,
        the ``onnx_names`` per-instance overrides); the task only knows how to
        render ITS own entry. The base raises for a task family that ships no
        ONNX rendering.

        Raises
        ------
        ConfigError
            For a task family that ships no ONNX rendering.
        """
        raise ConfigError(self._no_render_msg("ONNX output"))

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Render this task's converted, graph-visible output fields (plan 34 W34.1).

        The unifying serialisation-space entry point: it reads the task's RAW
        ``preds.*`` leaf (loss-space — ``forward`` no longer converts) plus any
        output-time deps (`output_time_requires`, e.g. the stream pad mask) from
        `b`, applies the eval conversion (softmax / masked-softmax / argmax /
        descale / union-find — the SAME math the family's ``run_inference`` /
        the legacy `ConversionOp` producer runs, VERBATIM, in TRACEABLE torch
        ops so ONNX sees them in-graph), and returns one `OutputField` per
        SERIALISATION LEAF. Each field carries the converted torch tensor as
        its ``value`` plus the BARE suffix + dtype + axis + ordering metadata —
        it does NOT pack a structured numpy array (the H5 sink packs), does NOT
        prefix the run/model name (the sink prefixes) and does NOT downcast to
        half precision (the sink downcasts). The H5-vs-ONNX representation split
        is keyed on `mode` (probs for H5 modes, per-class scalars / argmax index
        for ONNX) — see the per-family override.

        This UNIFIES the conversion + naming responsibilities of `get_h5`
        (TEST values) and `onnx_outputs` (ONNX manifest) onto one method; those
        two legacy methods stay for the transitional oracle path. The base
        raises for a task family that ships no output rendering (the
        unsupported-family guard, mirroring `get_h5`/`onnx_outputs`).

        Parameters
        ----------
        b : Bundle
            The executed bundle (carries the RAW ``preds.*`` leaf + any
            output-time dep, e.g. ``masks.<stream>``).
        mode : Mode
            The execution mode — selects the H5 (probs) vs ONNX
            (split-scalars / argmax index) representation.
        run_name : str
            The run ``name:`` — accepted for symmetry with `get_h5`/`output_names`
            but NOT baked into the field names (the sink prefixes), so the
            classification override deletes it.

        Raises
        ------
        ConfigError
            For a task family that ships no output rendering. The per-family
            override returns ``list[OutputField]`` (one field per serialisation
            leaf, each carrying a torch ``value``).
        """
        del b, mode, run_name
        raise ConfigError(self._no_render_msg("output"))

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free serialisation-leaf metadata for `mode` (plan 34 W34.2).

        The bundle-free twin of `get_output`: returns the SAME `OutputField`
        list (names / dtypes / axis / final / prefix, in the SAME field order),
        but with every ``value`` left ``None`` — so the dumb sinks can resolve
        the column NAMES / DTYPES / ORDER at declare/open time, before any batch
        runs (``get_output`` reads ``preds.*`` and cannot run without a bundle).
        The per-family override returns the metadata; the base raises for a task
        family that ships no output rendering (mirroring `get_output`).

        Parameters
        ----------
        mode : Mode
            The execution mode — selects the H5 (probs columns) vs ONNX
            (split-scalars / argmax index) representation, exactly as
            `get_output`.
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
        """The NON-pred bundle keys `get_output` needs at output time (plan 34 W34.1).

        Declares the extra dependencies (beyond the task's own ``preds.*`` leaf)
        that `get_output` reads — for `RunTaskOutput` to add to its
        ``declare_io`` requires in W34.2. A per-token (sequence) head needs the
        stream pad mask (``masks.<stream>``) for the masked softmax; a global
        (pooled) head needs nothing extra. Mirrors the legacy
        `ConversionOp.extra_requires` set, but task-side and mode-aware (a global
        head returns ``[]`` in every mode; a seq head returns the pad-mask key
        whenever it has one, in every mode that runs `get_output`).

        The base returns ``[]`` (a task family with no output-time deps); the
        per-family override declares its own.

        Parameters
        ----------
        mode : Mode
            The execution mode (accepted for families whose deps are mode-split;
            the base ignores it).

        Returns
        -------
        list[str]
            Dotted bundle keys (empty by default).
        """
        del mode
        return []

    def _no_render_msg(self, what: str) -> str:
        """The unsupported-family error message (moved off the writer).

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
    """Classification head over a pooled vector or a per-stream sequence (design §3.3).

    ``class_names`` is REQUIRED and explicit (no ``CLASS_NAMES`` fallback).
    Whenever the reader carries a schema artifact whose group attrs name the
    label's classes, the configured list is cross-checked — set AND order —
    by `salt.core.saltmodule.check_class_names`, default-on at
    `SaltModule.setup` and in ``salt2 graph validate`` (design §2.6, §4.1).
    The head width is ``len(class_names)``; the input/context widths are
    inferred at `bind` — no width keys in config (design §2.3).

    Declarative class weights (design §3.3, reproducing v1 ``use_class_dict``,
    cli.py:446-491): ``weight_source: {from_class_dict: <path>}`` allocates
    the CE ``weight`` buffer at `bind` (it lives in the loss's state_dict,
    ``torch.nn._WeightedLoss``), fills it at `materialise()` on fresh fits
    (the ONLY file I/O), and inherits it from the checkpoint on resume —
    v1's bake-on-resume contract (cli.py:253-267). Freezing the resolved
    values into the run-dir config is the config surface's job (M2 stage C).
    A literal ``loss.init_args.weight`` list remains valid and is exclusive
    with ``weight_source``.
    """

    onnx_renameable = True
    """Classification ONNX suffixes are overridable via ``onnx_names`` (condition 6)."""

    def __init__(
        self,
        stream: str,
        label: str,
        class_names: Sequence[str],
        input: str | None = None,  # noqa: A002 - design §3.3 YAML surface name
        context: str | None = None,
        sequence: bool | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        weight_source: Mapping[str, str] | None = None,
        label_map: dict[int, int] | None = None,
        expose: Sequence[str] | None = None,
    ) -> None:
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The labelled stream (``jets``, ``tracks``, ...).
        label : str
            Label field name, demanded as ``labels.<stream>.<label>``.
        class_names : Sequence[str]
            Ordered class names — REQUIRED, index-aligned with outputs.
        input : str | None, optional
            Input key, by default ``encoded.<stream>`` (the `Split` output).
        context : str | None, optional
            Context key (e.g. ``pooled.global``), by default None.
        sequence : bool | None, optional
            Whether the task is per-token, by default inferred: True when
            `input` is the per-stream default, False for an explicit input
            (e.g. ``pooled.global``). Set explicitly for per-token tasks on
            non-default inputs.
        dense : dict[str, Any] | None, optional
            Extra v1 `Dense` kwargs (no width keys), by default None.
        loss : str | dict[str, Any] | None, optional
            Loss config: a ``torch.nn`` class name or a
            ``{class_path, init_args}`` mapping, by default
            CrossEntropyLoss.
        weight : float, optional
            Scalar task-loss weight (applied INSIDE the composed v1 head,
            task.py:243), by default 1.0.
        weight_source : Mapping[str, str] | None, optional
            ``{"from_class_dict": <path>}`` declarative class-weight source
            (see class docstring), by default None.
        label_map : dict[int, int] | None, optional
            Integer label remap for training, by default None.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in (design §4.2), by
            default None (all modes). ``[fit, val]`` opts a train-only aux task
            out of the TEST/ONNX plans.

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
            ``modes=TRAINING`` and are filtered by the planner elsewhere.
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
        """Build the composed v1 head with inferred widths (design §2.3, §3.3).

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
            does not match ``class_names`` (v1 contract, cli.py:480-491).
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
        """Run the composed v1 head; RAW logits in EVERY non-training mode.

        The v1 head is handed SINGLE-STREAM label/mask dicts: its internal
        ``input_name_mask`` slicing is the identity on per-stream inputs,
        and the loss masking (pad fold to ``ignore_index=-1``, the ``-2``
        label fold, task.py:218-227) runs verbatim.

        Mode semantics (P4 atomic cutover — design §2 "no per-task forward
        branching" + §6/R9): the task now publishes RAW logits in EVERY
        non-training mode (FIT/VAL unchanged; TEST flipped at P1.5; ONNX
        flipped HERE at P4). The eval conversion (softmax / masked softmax) is
        OWNED by the classification conversion producers (`ClassProbs` /
        `SeqClassProbs` / `SeqClassIndex`) on the live path and by `get_h5` on
        the transitional M4.5 oracle path (both read this raw leaf and
        `run_inference` it once — design §4a). Pre-P4 ONNX kept softmaxing here
        (the deferred carve-out so the export trace was untouched); now the
        folded `ClassProbs` node owns the single softmax inside the traced
        graph, so there is NO double-softmax.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        assert self.task is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        # objects-stream (query-bank) heads have no pad mask (v1 task.py:547)
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        if mode & Mode.TRAINING:
            labels_dict = {self.stream: {self.label: b.get(self.label_key)}}
            pad_masks = {self.stream: mask} if self.has_pad_mask else None
            preds, loss = self.task(x, labels_dict, pad_masks, context=ctx)
            return {self.pred_key: preds, self.loss_key: loss}
        # TEST and ONNX (P4 atomic cutover): publish RAW logits in EVERY non-
        # training mode — the classification task no longer softmaxes anywhere.
        # The eval conversion (softmax / masked softmax) is OWNED by the
        # classification conversion producers (`ClassProbs`/`SeqClassProbs`/
        # `SeqClassIndex`) on the live path and by `get_h5` on the transitional
        # M4.5 oracle path (both read this raw leaf and `run_inference` it once).
        # Pre-P4 ONNX kept publishing the converted probs (the deferred carve-out
        # so the export trace was untouched); now the folded `ClassProbs` node
        # owns the softmax, so a single softmax happens in the traced graph (no
        # double-softmax — design §6 / R9).
        preds, _ = self.task(x, None, None, context=ctx)
        return {self.pred_key: preds}

    # -- output rendering (v1 ClassificationTask.output_names/get_h5/get_onnx) --

    @property
    def class_suffixes(self) -> list[str]:
        """Per-class logical suffixes — ``Flavours[c].px`` else ``p{c}`` (v1 task.py:140-151).

        ONE owner, BOTH modes: the TEST columns prefix these with the run name
        (`output_names`), the ONNX manifest carries them bare for the
        exporter's ``{model_name}_`` prefix (`onnx_outputs`).

        Returns
        -------
        list[str]
            One suffix per ``class_names`` entry, in class order.
        """
        # Imported lazily, NOT at module level: ``Flavours`` is a ftag
        # ``LabelContainer`` instance whose ``__getattr__`` raises ``KeyError``
        # (not ``AttributeError``) for unknown names. jsonargparse >=4.48 walks a
        # function's module globals with ``hasattr(value, "__args__")`` when
        # resolving annotations (``implements_protocol`` against ``GraphModule``);
        # a module-level ``Flavours`` makes that ``hasattr`` raise, so this
        # module's task classes fail the ``GraphModule`` protocol check and the
        # whole ``model.modules`` config is rejected at parse time. Keeping the
        # import local removes ``Flavours`` from the module globals.
        from ftag import Flavours  # noqa: PLC0415

        return [Flavours[c].px if c in Flavours else f"p{c}" for c in self.class_names]

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """One ``f4`` column per class, named ``{run_name}_{px}`` (v1 task.py:140-151,282).

        Returns
        -------
        list[tuple[str, str]]
            ``(column, "f4")`` pairs, one per class, in class order.
        """
        return [(f"{run_name}_{px}", "f4") for px in self.class_suffixes]

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Softmax the RAW TEST logits then pack as ``f4`` columns (v1 task.py:266-283).

        Since the P1.5 flip the TEST ``forward`` publishes RAW logits (design
        §2), so this M4.5-oracle path now does the eval conversion ITSELF: it
        ``run_inference``-s the raw ``preds.*`` leaf (softmax / masked softmax,
        the SAME math the forward used to run) before the ``probs -> f4 u2s``
        packing (padded positions read 0.0). The conversion happens exactly
        ONCE per leaf on this path — the producer path converts independently
        (no double-conversion, design §4 risk "double-conversion"). The mask is
        read from ``masks.<stream>`` for the masked softmax (present in the TEST
        bundle: the per-token head declares it as a require and the writer
        machinery / PadMaskWriter demands it). The OUTPUT of this method is
        byte-identical to pre-flip (softmax-then-pack == flip-then-softmax-then-
        pack).

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

        Mirrors v1's two classification export shapes (``to_onnx.py:258-292``):
        a global (pooled) classification head emits one float32 scalar per
        class (suffixes = `class_suffixes`); a per-token sequence head emits a
        single int8 argmax entry whose suffix is the Pascal-case of the task
        instance name (``track_origin -> TrackOrigin``). The `TaskWriter`
        owns ``onnx_names`` overrides and the global-before-sequence order.

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

        A per-token (sequence) classification head's masked softmax reads
        ``masks.<stream>`` (the same `extra_requires` the legacy
        `SeqClassProbsOp`/`SeqClassIndexOp` declared). A global (pooled) head's
        plain softmax / sigmoid needs nothing beyond the pred leaf. The
        objects-stream query-bank seq head has no pad mask (`has_pad_mask` False)
        so even a sequence head returns ``[]`` there — exactly v1's objects-stream
        exemption (task.py:547). Mode-independent for this family.

        Returns
        -------
        list[str]
            ``["masks.<stream>"]`` for a padded seq head, else ``[]``.
        """
        del mode
        # `has_pad_mask` already implies `sequence` (tasks.py:881:
        # `self.sequence and self.stream not in _NO_PAD_MASK_STREAMS`), so the
        # bare `has_pad_mask` guard covers all three cases: global [] / no-pad-mask
        # seq [] / padded seq [masks.<stream>].
        if self.has_pad_mask:
            return [f"masks.{self.stream}"]
        return []

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Softmax/argmax the RAW logits into graph-visible `OutputField`s (plan 34 W34.1).

        Reads the RAW ``preds.*`` logits (forward is loss-space) + the stream pad
        mask (seq head) and runs the eval conversion in TRACEABLE torch ops,
        folding the EXACT math of the legacy `ClassProbsOp` / `SeqClassProbsOp` /
        `SeqClassIndexOp` (``producers.py``) which itself mirrors the absorbed
        ``run_inference`` (``tasks.py``). Per mode:

        - **Global (pooled) head**: ``sigmoid`` (BCE) else ``softmax(dim=-1)``.
          BOTH H5 and ONNX use the per-class ``class_suffixes`` (the legacy
          ``output_names`` H5 columns AND ``split_scalars`` ONNX scalars share the
          suffixes). One ``f4`` global field PER CLASS, ``value`` = that class's
          column ``probs[..., c]``. The value shape is MODE-FAITHFUL to the sink it
          feeds:

          - **H5 modes** (FIT/VAL/TEST): the per-class column ``[B]`` — exactly the
            float column ``get_h5`` packs (tasks.py:1364). No squeeze (a B=1 H5 leaf
            must stay ``[1]`` to match ``get_h5``).
          - **ONNX**: the 0-dim scalar ``[]`` the live ONNX sink mints. The sink
            (`OnnxExportSink.named_outputs`, sinks.py:1714-1717) takes the full
            ``[1, C]`` leaf and does ``torch.split(value, 1, -1)`` then
            ``part.squeeze()`` (NO dim arg → drops ALL size-1 dims). On the batch-1
            ONNX trace the global logits are ``[1, C]`` (the adapter keeps batch dim
            1 for globals), so ``probs[..., c]`` is ``[1]`` and ``.squeeze()`` yields
            ``[]`` — the rank-0 scalar Athena/v1 expects. So ``probs[..., c]`` alone
            (``[1]``) does NOT match the v1 trace; ``get_output`` squeezes in ONNX
            mode to reproduce it byte-for-byte.

          ``run_name`` is NOT baked in (the sink prefixes), so it is deleted.

          .. note:: W34.2 sink-wiring decision — get_output now owns the global ONNX
             squeeze (each per-class field value is already the 0-dim scalar). When
             the dumb ``OnnxExportSink`` is wired to consume these per-class field
             values DIRECTLY (the "sink supplies names, does no math" contract, plan
             §4), it must NOT re-split / re-squeeze: it names the already-scalar
             values. The legacy plural-names ``leaf.names`` split path stays only for
             the static (value-less) plan-31 leaves.
        - **Sequence (per-token) head**:
            - **H5 modes** (FIT/VAL/TEST): masked softmax (padded tokens read
              ``0.0`` — the v1 eval byte-parity quirk). One ``f4`` per-token field
              PER CLASS, ``onnx_name=None`` (H5-only — the ONNX side is the argmax
              index), ``value`` = ``probs[..., c]`` (shape ``[B, L]``).
            - **ONNX**: masked softmax then the zero-row-append/strip ``argmax``
              VERBATIM from `SeqClassIndexOp.convert` (``producers.py:560-567`` /
              ``to_onnx.py:418-421``), yielding the ``[L]`` int8 leaf. ONE field
              with ``h5_name=None`` (ONNX-only), ``onnx_name=pascal_case(name)``
              (e.g. ``TrackOrigin``), ``dtype="int8"`` (per-token), ``value`` =
              the int8 tensor.

        The probs (H5) and index (ONNX) leaves carry DISTINCT logical names (the
        per-class suffixes vs the pascal-case task name) so the write-once
        ``outputs.*`` constraint holds when both are minted (today's
        ``track_origin_probs`` vs ``track_origin_index`` split — the RunTaskOutput
        leaf name disambiguates in W34.2). The H5 packing (``u2s``), the
        ``{run_name}``/``{model_name}`` prefix and the f4->f2 downcast all stay in
        the sink — this method returns full-precision torch tensors only.

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
            # global (pooled) head: sigmoid (BCE) / softmax(dim=-1) — verbatim
            # ClassProbsOp.convert (producers.py:441-448) / run_inference
            # (tasks.py:329-333).
            if isinstance(self.task.loss, torch.nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(logits)
            else:
                assert logits.ndim == 2, "global classification head expects [B, C] logits"
                probs = torch.softmax(logits, dim=-1)
            # The per-class value shape is MODE-FAITHFUL to the live sinks: in ONNX
            # the sink mints a 0-dim scalar (`OnnxExportSink.named_outputs`,
            # sinks.py:1714-1717: `torch.split(probs, 1, -1)` then `part.squeeze()` —
            # NO dim arg → drops ALL size-1 dims; on the batch-1 ONNX trace the global
            # logits are `[1, C]`, so `probs[..., c]` is `[1]` and `.squeeze()` yields
            # the `[]` scalar Athena/v1 expects). In H5/TEST modes the sink packs the
            # full `[B]` per-class column (`get_h5`, tasks.py:1364), so the value MUST
            # stay `[B]` (squeezing there would diverge from get_h5 at B=1). We mirror
            # the sink exactly: squeeze only in ONNX mode.
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
        # sequence (per-token) head: masked softmax (padded tokens -> 0.0) —
        # verbatim SeqClassProbsOp/SeqClassIndexOp (producers.py:559) /
        # run_inference (tasks.py:334-336).
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        probs = _masked_softmax(logits, mask.unsqueeze(-1) if mask is not None else None)
        if mode & Mode.ONNX:
            # ONNX argmax index: zero-row append/strip VERBATIM (producers.py:560-567 /
            # to_onnx.py:418-421) -> [L] int8 leaf under the pascal-case task name.
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
        # These seq probs fields are H5-ONLY (the ONNX side of a seq head is the
        # argmax index leaf above, NOT per-class probs). The H5-only intent is
        # carried by `onnx_name=None` + `axis=per_token`, but it is NOT enforceable
        # via `resolved_onnx_name`: that property falls back to `h5_name` when
        # `onnx_name` is None (producers.py:194), so these fields REPORT a per-class
        # ONNX suffix. They are reached only in H5 modes because get_output mode-keys
        # (Mode.ONNX returns the argmax leaf above and never reaches here). The W34.2
        # RunTaskOutput/sink consumer MUST select ONNX outputs by MODE (call
        # get_output(b, Mode.ONNX, ...)), NOT by enumerating resolved_onnx_name across
        # all fields — otherwise these H5-only probs would falsely claim ONNX names
        # and collide with the argmax leaf.
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
        """The value-free field metadata mirroring `get_output` for `mode` (plan 34 W34.2).

        Returns the SAME `OutputField`s `get_output` mints (names / dtypes / axis /
        final / prefix, SAME field order) but with ``value=None`` — so the dumb
        sinks resolve the column schema before any batch runs. The mode split is
        IDENTICAL to `get_output`:

        - global head -> one ``f4`` global field per class (``class_suffixes``).
        - seq head, ONNX -> one int8 per-token argmax field (``pascal_case(name)``).
        - seq head, H5 modes -> one ``f4`` per-token field per class (H5-only).

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
    """Edge-classification vertexing head (design §3.3, composes v1 `VertexingTask`).

    The origin-label dependency is DECLARED (``origin_label:`` config →
    ``labels.<stream>.<origin_label>``) instead of derived by string-replace
    and free-riding on another task's label collection (task.py:937).
    Origin weighting is explicit config: ``origin_weighting`` gives the
    heavy/fake origins as integer *ids* OR class *names* (defaults reproduce
    v1's hardcoded 3,4,5 / 1, task.py:964 — bit-identical math). Names (design
    §5.1, the GN3 origin surface) are resolved to ids at fit/test setup against
    the origin label's class-name attr in the dataset schema artifact
    (``schema_group(stream).attrs[origin_label]``, the same source as the §2.6
    class-names cross-check) — `resolve_origin_names`, driven by
    `salt.core.saltmodule.resolve_origin_weighting` BEFORE bind. A name-based
    config without a schema artifact is a loud `ConfigError` at bind (the names
    cannot be resolved); an integer-id config never needs a schema and binds
    standalone (unit-test path). Mixing names and ids inside one heavy/fake
    list is rejected.

    Mode exception (design §3.3): TEST publishes per-node vertex
    assignments (v1 ``run_inference``: union-find + ``mask_fill_flattened``,
    task.py:985-986,1003); ONNX publishes RAW edge scores for the
    export-side ``vertex_union_find`` reduce (to_onnx.py:426-431).
    """

    def __init__(
        self,
        stream: str,
        label: str,
        origin_label: str,
        input: str | None = None,  # noqa: A002 - design §3.3 YAML surface name
        context: str | None = None,
        dense: dict[str, Any] | None = None,
        loss: str | dict[str, Any] | None = None,
        weight: float = 1.0,
        origin_weighting: Mapping[str, Sequence[int | str]] | None = None,
        prefix_vertex_column: bool = False,
        expose: Sequence[str] | None = None,
    ) -> None:
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The track-like stream.
        label : str
            Vertex-index label (must contain ``"VertexIndex"`` — the
            composed v1 loss derives the origin key by string-replace,
            task.py:937; absorbed at M7).
        origin_label : str
            Declared origin-label dependency for edge weighting (and, for
            name-based weighting, the schema attr the names resolve against).
        input : str | None, optional
            Input key, by default ``encoded.<stream>``.
        context : str | None, optional
            Context key, by default None.
        dense : dict[str, Any] | None, optional
            Extra v1 `Dense` kwargs (no width keys), by default None.
        loss : str | dict[str, Any] | None, optional
            Loss config, by default ``BCEWithLogitsLoss(reduction="none")``
            (reduction MUST be ``none`` — per-edge weighting multiplies the
            unreduced loss, task.py:938-947).
        weight : float, optional
            Scalar task-loss weight, by default 1.0.
        origin_weighting : Mapping[str, Sequence[int | str]] | None, optional
            ``{"heavy": [...], "fake": [...]}`` origin ids OR class names, by
            default v1's ``{"heavy": [3, 4, 5], "fake": [1]}``. Names are
            resolved at setup against the schema's origin class-name attr
            (design §5.1); ids bind directly.
        prefix_vertex_column : bool, optional
            Name the TEST vertexing column ``{run_name}_VertexIndex`` instead
            of the v1-compatible bare ``VertexIndex``, by default False. The
            task owns its column naming now (moved off `TaskWriter`); the W1
            byte-schema bar is v1's bare output (``task.py:1003``), and the
            design §8 run-name-prefix fix is opt-in behind this flag (the
            config converter sets the compat flag, design §8). The ONNX name
            is ALWAYS the shared `VERTEX_INDEX` constant (the exporter
            prefixes ``{model_name}_`` — v1 ``to_onnx.py:287``), so flipping
            the flag aligns the two names up to the prefix value.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in (design §4.2), by
            default None (all modes). ``[fit, val]`` opts a train-only aux task
            out of the TEST/ONNX plans.

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
        # ids bind standalone; names defer to resolve_origin_names(reader) at
        # setup. heavy_ids/fake_ids stay None until resolved so a name-based
        # bind without a schema fails loudly instead of silently mis-weighting.
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
        """The declared origin-label dependency (design §3.3).

        Returns
        -------
        str
            ``labels.<stream>.<origin_label>``.
        """
        return f"labels.{self.stream}.{self.origin_label}"

    def resolve_origin_names(self, reader: Any) -> bool:
        """Resolve name-based ``origin_weighting`` to ids against the schema (design §5.1).

        No-op for integer-id weighting (``heavy_ids``/``fake_ids`` already set
        in ``__init__``). For name-based weighting, the origin label's
        class-name attr (``schema_group(stream).attrs[origin_label]``, the §2.6
        class-names source) maps each name to its index — the integer origin id
        the v1 weighting math compares against (task.py:957-964). Called from
        `salt.core.saltmodule.resolve_origin_weighting` at fit/test setup, after
        the boundary readers exist and BEFORE bind, so the resolved ids are in
        place when `bind` builds the composed head.

        Parameters
        ----------
        reader : Any
            The stage dataset reader; consulted via ``schema_group(stream)``
            (duck-typed — a reader without schema support resolves nothing,
            leaving a name-based config to fail loudly at bind).

        Returns
        -------
        bool
            True when names were resolved here (one resolution counted), False
            when there was nothing to resolve (integer ids) or the reader has
            no schema artifact.

        Raises
        ------
        ConfigError
            When the stream/origin-label class-name attr is absent or a
            configured name is not among the schema's origin classes (design
            §4.1 quality bar: names the unknown name and the known classes).
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
        assignments are per-node (v1 ``mask_fill_flattened`` layout).

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
        """Build the composed v1 vertexing head with inferred widths (design §2.3).

        ``input_size = 2 * width(input)`` (pair concat, task.py:894-896);
        ``context_size = width(context)`` (task.py:884-891).

        Raises
        ------
        ConfigError
            If the configured loss does not use ``reduction="none"`` (per-edge
            weighting requires the unreduced loss), or name-based
            ``origin_weighting`` was never resolved (no schema artifact reached
            `resolve_origin_names` before bind).
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
        """Run the composed v1 head; RAW edge scores in EVERY non-training mode (plan 34 W34.3).

        The labels dict carries the origin labels under the v1-derived key
        (``label.replace("VertexIndex", "OriginLabel")``) so the composed
        loss finds them — the *graph* dependency is the declared
        ``origin_label`` port.

        Mode semantics (plan 34 W34.3 atomic forward-flip — design §2): the
        task now publishes the RAW ``[E, 1]`` edge scores in EVERY non-training
        mode (FIT/VAL unchanged; **TEST flipped HERE** — it used to run
        ``run_inference`` (union-find) and publish the per-node assignments;
        ONNX was ALREADY raw, the union-find living in the export reduce). The
        union-find conversion is now OWNED by ``get_output`` on the live path
        (the ``RunTaskOutput`` consumer reads this raw leaf and union-finds it
        once) and by ``get_h5`` on the transitional M4.5 oracle path (it reads
        this raw leaf and ``run_inference``-s it once) — exactly the W34.1
        classification pattern. The conversion happens exactly ONCE per leaf on
        either path (no double-conversion, no zero-conversion — plan §4).

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
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
        # TEST and ONNX (plan 34 W34.3 atomic flip): publish the RAW [E, 1] edge
        # scores in EVERY non-training mode — the vertexing task no longer runs
        # union-find anywhere. The eval conversion (union-find -> per-node int8
        # assignments) is OWNED by the conversion producer / get_output on the
        # live path and by get_h5 on the transitional oracle path (both read this
        # raw leaf and run the union-find ONCE).
        preds, _ = self.task(x, None, pad_masks, context=ctx)
        return {self.pred_key: preds}

    # -- output rendering (v1 VertexingTask.output_names/get_h5 + naming) -------

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """A single ``('VertexIndex', 'i8')`` column (v1 task.py:1003-1004).

        The column is BARE ``VertexIndex`` by default (v1 byte parity); with
        ``prefix_vertex_column`` it is ``{run_name}_VertexIndex`` (design §8).

        Returns
        -------
        list[tuple[str, str]]
            One ``(column, "i8")`` pair.
        """
        column = f"{run_name}_{VERTEX_INDEX}" if self.prefix_vertex_column else VERTEX_INDEX
        return [(column, "i8")]

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """Per-node vertex assignments as one ``i8`` column (v1 task.py:988-1005).

        Since the plan 34 W34.3 flip the TEST ``forward`` publishes the RAW
        ``[E, 1]`` edge scores (design §2), so this M4.5-oracle path now runs the
        union-find conversion ITSELF: it ``run_inference``-s the raw ``preds.*``
        leaf (``get_node_assignment_jit`` -> ``_mask_fill_flattened``, the SAME
        math the forward used to run) before the EXACT v1 op chain
        ``preds.int().cpu()`` -> u2s i8. The conversion happens exactly ONCE on
        this path (the producer / get_output path converts independently — no
        double-conversion, plan §4). The mask is read from ``masks.<stream>``
        (present in the TEST bundle — the vertexing head declares it as a forward
        require). Padded positions read the int32 cast of ``-inf``
        (-2147483648). The OUTPUT is byte-identical to pre-flip (union-find-then-
        pack == flip-then-union-find-then-pack).

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

        The export suffix is the SAME `VERTEX_INDEX` constant the TEST column
        uses (amendment §2.2; v1 ``to_onnx.py:286-288``) — the exporter
        prepends ``{model_name}_``. The in-graph union-find lives in the
        shipped ``vertex_union_find`` reduce (referenced by key, no registry
        change); ONNX publishes raw edge scores that the reduce consumes.

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
        """The non-pred dep `get_output` reads: the stream pad mask (plan 34 W34.3).

        The vertexing union-find reads the stream's ``masks.<stream>`` pad mask
        (the all-valid / padded mask the union-find + ``mask_fill_flattened``
        consume — the SAME ``extra_requires`` the legacy `VertexUnionFind`
        producer declared, producers.py:1522). A vertexing head always has a pad
        mask (its ``forward`` requires ``masks.<stream>`` unconditionally), so
        this is mode-independent.

        Returns
        -------
        list[str]
            ``["masks.<stream>"]``.
        """
        del mode
        return [f"masks.{self.stream}"]

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """Union-find the RAW edge scores into a graph-visible int8 leaf (plan 34 W34.3).

        Reads the RAW ``preds.*`` ``[E, 1]`` edge scores (forward is loss-space
        since the W34.3 flip) + the stream pad mask and runs the union-find
        conversion in TRACEABLE torch ops, folding the EXACT chain of the legacy
        `VertexUnionFind` producer (producers.py:1567-1569) / the task's own
        ``run_inference`` (tasks.py:773-782). Per mode:

        - **ONNX**: the export-shaped chain VERBATIM —
          ``get_node_assignment_jit`` -> ``mask_fill_flattened`` ->
          ``.reshape(-1).char()`` (the IDENTICAL ops the `VertexUnionFind`
          producer runs, so the traced subgraph is byte-identical) — one int8
          ``[L]`` per-token field under the shared `VERTEX_INDEX` suffix (the
          exporter prepends ``{model_name}_``).
        - **H5 modes** (TEST): ``run_inference`` (``get_node_assignment_jit`` ->
          ``_mask_fill_flattened`` -> ``[B, L, 1]`` with ``-inf`` padding) then
          ``.int()`` — the EXACT value ``get_h5`` packs (tasks.py:2031). One int8
          per-token field, ``axis="per_token"``, ``dtype="i8"`` (the v1 int64
          ``VertexIndex`` column), ``onnx_name=None`` (the ONNX side is the
          ``.char()`` index leaf above). The column ``prefix`` follows
          ``prefix_vertex_column`` (bare ``VertexIndex`` by default, v1 byte-
          parity).

        ``run_name`` is NOT baked in (the sink prefixes), so it is deleted.

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
            # ONNX export-shaped union-find — VERBATIM the VertexUnionFind producer
            # chain (producers.py:1567-1569) so the traced subgraph is identical.
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
        # H5 (TEST): run_inference union-find -> .int(), the EXACT value get_h5 packs
        # ([B, L, 1] with -inf padding cast to int32 -> u2s i8 by the sink).
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
        """The value-free field metadata mirroring `get_output` for `mode` (plan 34 W34.3).

        Returns the SAME `OutputField` `get_output` mints (name / dtype / axis /
        final / prefix, value-free) so the dumb sinks resolve the column schema
        before any batch runs:

        - ONNX -> one int8 per-token field under the shared `VERTEX_INDEX` ONNX
          suffix (``h5_name=None``).
        - H5 modes -> one ``i8`` per-token ``VertexIndex`` field (``onnx_name=None``,
          ``prefix`` per ``prefix_vertex_column``).

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
    """Scalar/vector regression head (design §3.3, composes v1 `RegressionTask`).

    Targets are read from labels (``labels.<stream>.<target>``) and the head
    publishes ``preds.<stream>.<name>`` of shape ``[B, R]`` (global) or
    ``[B, L, R]`` (per-token sequence), where ``R = len(targets)``. The
    composed v1 head owns the scaling math verbatim (task.py:442-482 forward
    scaling, 567-602 ``run_inference`` de-scaling); what is new is the
    declared label/denominator dependency set and the **mode-split
    de-scaling** wiring.

    A single scaling method (v1 single-scaling guard, task.py:355): exactly
    one of ``target_denominators`` (ratio targets), ``norm_params``
    (mean/std), or ``scaler`` (per-target functional `RegressionTargetScaler`)
    — or none (raw targets). Each is config scalar-or-list, count-checked
    against ``targets`` inside the composed v1 head.

    Mode-split de-scaling (FD §3.3): the de-scaling SOURCE for ratio targets
    differs per mode — same math, different denominator provider.

    - **FIT|VAL**: targets and ratio denominators come from
      ``labels.<stream>.<var>`` (training labels). Predictions are published
      RAW/SCALED (cheap, consumed by metrics callbacks), loss in scaled
      space (v1 forward, task.py:519-565).
    - **TEST**: predictions are DE-SCALED (v1 ``run_inference``). The ratio
      denominator falls back to ``labels.<stream>.<denom>`` when it is not a
      declared input Feature (v1 ``get_h5`` reads ``labels[input_name][denom]``,
      task.py:587-589,604,621); ``event_classifier.yaml:65``'s ``mHH`` IS a
      feature, but the TEST writer still re-reads it from the label group, so
      TEST always sources denominators from labels.
    - **ONNX**: the ratio denominator MUST be a declared input Feature
      (resolved by NAME at `bind` against the stream's declared columns) —
      v1 ``get_onnx(labels=structured_input_dict)`` reads it from the
      structured INPUT dict, not the labels (to_onnx.py:381-398). A
      denominator absent from the input Features is a `bind`-time error, since
      the export graph has no other source for it.

    ``norm_params`` and ``scaler`` need no external source, so their de-scaling
    is mode-independent.

    Gaussian regression (M5 sub-wave A3, FD 1567-1568): ``gaussian: true``
    composes a v1 ``GaussianRegressionTask`` (task.py:645) — the head emits
    ``2 * len(targets)`` outputs (means ‖ raw variances), the loss is the
    Gaussian NLL, and inference returns means + ``stddev = sqrt`` of the
    de-scaled softplus variance (task.py:753-774). Unlike v1's TEST-path
    ``(means, stds)`` TUPLE, this module publishes ONE ``[B, 2R]`` (or
    ``[B, L, 2R]``) array — means in the first R columns, stddevs in the last
    R — so the rest of the graph sees the same single-leaf contract as a plain
    regression head; the writer owns the ``_stddev`` suffix split (one owner,
    both modes). The **ONNX stddev representation** is a M4.5 deferral-(b)
    design-conformance: ``stddev = sqrt(softplus(var))`` is traced in-graph by
    the same de-scaling math and declared in the unified manifest — there is NO
    v1 ONNX golden (v1 ``onnx/to_onnx.py`` has zero Gaussian handling).

    Per-sample weighting (``sample_weight:``) and NaN-target masking are
    handled verbatim inside the composed v1 ``nan_loss`` (task.py:384-440):
    invalid (NaN) targets are masked to 0 and the loss is reduced with
    ``torch.nanmean``; a configured ``sample_weight`` label multiplies the
    per-element loss (unsqueeze + multi-target expand) before the mean. Both
    require ``loss.reduction == 'none'`` so the per-element loss survives —
    surfaced at config time for ``sample_weight`` and documented for NaN
    targets.

    The `MultiTarget` processor (conditional row-wise target replacement) is a
    DATASET-side sibling — `salt.core.data.processors.MultiTarget` — not a task
    flag; it runs over the label bundle BEFORE the task forward.
    """

    def __init__(
        self,
        stream: str,
        targets: str | Sequence[str],
        input: str | None = None,  # noqa: A002 - design §3.3 YAML surface name
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
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The regressed stream (``jets``, ``tracks``, ``objects``, ...).
        targets : str | Sequence[str]
            Regression target name(s), demanded as
            ``labels.<stream>.<target>``. The head width is ``len(targets)``.
        input : str | None, optional
            Input key, by default ``encoded.<stream>`` (the `Split` output).
        context : str | None, optional
            Context key (e.g. ``pooled.global``), by default None.
        sequence : bool | None, optional
            Whether the task is per-token, by default inferred: True when
            `input` is the per-stream default, False for an explicit input
            (e.g. ``pooled.global``). Set explicitly for per-token tasks on
            non-default inputs.
        target_denominators : str | Sequence[str] | None, optional
            Per-target ratio denominator variable(s) (``target/denom``),
            by default None. Mutually exclusive with `norm_params`/`scaler`.
        norm_params : Mapping[str, Any] | None, optional
            ``{"mean": <scalar|list>, "std": <scalar|list>}`` per-target
            mean/std normalisation, by default None. Mutually exclusive.
        scaler : Mapping[str, Mapping[str, Any]] | None, optional
            Per-target functional scaling
            ``{<target>: {op, x_scale, x_off, op_scale, op_off}}`` (built
            into a `RegressionTargetScaler`), by default None. Mutually
            exclusive.
        custom_output_names : str | Sequence[str] | None, optional
            Output column suffix(es) overriding the target names (the v1
            ``custom_output_names``, task.py:514-516), by default None
            (suffixes ARE the target names). Count-checked against `targets`.
        gaussian : bool, optional
            Compose a v1 ``GaussianRegressionTask`` instead of the plain
            ``RegressionTask`` (mu/sigma head, ``output_size = 2 *
            len(targets)``, softplus variance, Gaussian NLL, stddev =
            ``sqrt`` in inference; v1 task.py:645-774), by default False.
            A Gaussian head needs a non-functional scaling method
            (``norm_params`` or ``target_denominators``): v1's gaussian
            ``run_inference`` has no ``scaler`` branch and RAISES without
            scaling params (task.py:765-766), so a ``scaler:`` or
            scaling-free gaussian head is a config-time error.
        sample_weight : str | None, optional
            Per-sample loss-reweighting label (``labels.<stream>.<weight>``,
            FIT|VAL only), multiplied INTO the per-element loss inside the
            composed v1 ``nan_loss`` (unsqueeze + multi-target expand,
            task.py:429-435), by default None. REQUIRES ``loss.reduction ==
            'none'`` (the v1 assert, task.py:379-382) — surfaced at config
            time.
        publish_targets : bool, optional
            Also publish the SCALED regression targets under
            ``targets.<stream>.<instance-name>`` in FIT|VAL (the v1
            ``maskformer_loss.py:329-332`` contract: the object-regression
            head's ``get_targets`` output is stored in the labels dict so the
            Hungarian matcher can use the scaled-space targets as a cost term),
            by default False. The ONLY consumer is `MaskFormerMatchedLoss`,
            which requires ``targets.objects.regression`` alongside
            ``preds.objects.regression``; a plain regression head leaves it off
            and publishes ONLY its prediction. The scaling applied is exactly
            the head's own (``scaler``/``norm_params``/``target_denominators``)
            via the composed v1 ``get_targets`` — matched-space, FD §5.2.
        dense : dict[str, Any] | None, optional
            Extra v1 `Dense` kwargs (no width keys), by default None.
        loss : str | dict[str, Any] | None, optional
            Loss config, by default ``MSELoss`` (``GaussianNLLLoss`` when
            ``gaussian`` is set). NaN-target masking is unconditional inside
            the composed v1 ``nan_loss`` (mask invalid -> 0, ``torch.nanmean``
            over the loss, task.py:412-437); a config that EXPECTS NaN
            targets (``nan_regression``) must set ``reduction: none`` so the
            per-element mask survives to the ``nanmean`` (task.py:421-427).
        weight : float, optional
            Scalar task-loss weight (applied INSIDE the composed v1 head,
            task.py:563), by default 1.0.
        expose : Sequence[str] | None, optional
            Modes the ``preds.*`` port is published in (design §4.2), by
            default None (all modes). ``[fit, val]`` opts a train-only aux task
            out of the TEST/ONNX plans.

        Raises
        ------
        ConfigError
            On empty targets, a custom-output-name count mismatch, more than
            one scaling method (the v1 single-scaling guard surfaced at
            config time), a ``sample_weight`` with a non-``none`` loss
            reduction, a gaussian head combined with a functional ``scaler`` /
            no scaling method, or a bad `expose` list.
        """
        self.gaussian = bool(gaussian)
        default_loss = _DEFAULT_GAUSS_LOSS if self.gaussian else _DEFAULT_REG_LOSS
        super().__init__(stream, "", input, context, dense, loss, weight, default_loss, expose)
        # regression has no single `label` field; the demand is one label per
        # target (set below). `_TaskModuleBase.label` is left empty.
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
        # resolved at bind: the declared input-Feature column order, captured
        # so the ONNX de-scaling can gather denominators by NAME (FD §3.3).
        self._input_fields: tuple[str, ...] = ()

    @staticmethod
    def _checked_norm_params(
        norm_params: Mapping[str, Any] | None,
    ) -> dict[str, list[float]] | None:
        """Normalise + validate the ``norm_params`` mapping (v1 task.py:349-351).

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

        Reproduces v1 ``RegressionTask.output_names`` minus the ``model_name``
        prefix (task.py:511-517); the writer adds the run-name prefix. For a
        gaussian head the suffix list is doubled — the R mean suffixes followed
        by the R ``<suffix>_stddev`` suffixes (v1 ``GaussianRegressionTask.
        output_names``, task.py:672-675; the FD 1567-1568 consistency fix honours
        ``custom_output_names`` for the means, which v1's ``output_names``
        dropped), index-aligned with the published ``[B, 2R]`` array.

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
        """The declared target-label dependencies (one per target, design §3.3).

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
        """The published SCALED-targets key (``publish_targets``, FIT|VAL only).

        Returns
        -------
        str
            ``targets.<stream>.<instance-name>`` — the matched-loss matcher's
            scaled-space target source (v1 maskformer_loss.py:332).
        """
        return f"targets.{self.stream}.{self.name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare input/context/masks (+ target/denominator deps) -> preds (+ loss).

        Mode-split denominator dependency (FD §3.3): FIT|VAL|TEST demand the
        denominators from ``labels.<stream>.<denom>``; ONNX demands the raw
        ``inputs.<stream>`` Feature tensor (the export-side source). Targets,
        the per-sample weight, and the FIT|VAL loss are TRAINING-gated;
        predictions are produced in ALL modes. A gaussian head publishes a
        ``2R``-wide prediction (means ‖ stddevs).

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
        # targets are training labels (continuous → float32; v1 reads them as
        # the dataset dtype and stacks/divides, task.py:458-460)
        for key in self.target_label_keys:
            requires[key] = TensorSpec(
                shape=label_shape, dtype="float32", kind="label", modes=Mode.TRAINING
            )
        if self.sample_weight is not None:
            # the per-sample weight is a FIT|VAL training label; it multiplies
            # the per-element loss inside nan_loss (task.py:429-435)
            requires[self.weight_label_key] = TensorSpec(
                shape=label_shape, dtype="float32", kind="label", modes=Mode.TRAINING
            )
        if self.target_denominators is not None:
            # FIT|VAL|TEST: denominators come from the label group; ONNX reads
            # them from the raw input Feature tensor instead (de-scaling source
            # split, FD §3.3 — to_onnx.py:381-398 vs task.py:587-589).
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
        # The matched-loss object head (publish_targets) is a pure feature +
        # scaled-target producer: v1 DISCARDS its standalone loss (``task_pred, _
        # = task(...)``, maskformer_loss.py:330) and the MaskFormerMatchedLoss
        # owns ``losses.regression`` (the matched regression component) — so this
        # head emits NO ``losses.<name>`` to avoid a two-producer collision on
        # that key. A plain head emits its own FIT|VAL loss as usual.
        if not self.publish_targets:
            produces[self.loss_key] = TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING)
        else:
            # the SCALED targets (width R = len(targets), even for a gaussian
            # head — targets are never doubled) for the matched-loss matcher,
            # FIT|VAL only (v1 maskformer_loss.py:329-332)
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
        """Build the composed v1 regression head + resolve the ONNX denominator source.

        ``output_size = len(targets)`` (``2 * len(targets)`` for a gaussian
        head); the input/context widths are inferred from the resolved schema
        (design §2.3). When ``target_denominators`` is set, every denominator
        must be a declared column of ``inputs.<stream>`` so the ONNX graph can
        gather it by name (to_onnx.py:381-398) — otherwise the ONNX de-scaling
        has no source and `bind` raises. A gaussian head with neither
        ``norm_params`` nor ``target_denominators`` also raises here: v1's
        gaussian ``run_inference`` cannot de-scale without scaling params
        (task.py:765-766).

        Raises
        ------
        ConfigError
            If a ratio denominator is not a declared input Feature (the ONNX
            mode has no other source for it, FD §3.3), or a gaussian head has
            no scaling method for inference de-scaling.
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
        # norm_params is mutated in place by v1 RegressionTaskBase (listify),
        # so hand it a fresh copy to keep this module's config immutable.
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
        # the plain head additionally takes the functional `scaler` (the
        # gaussian head has no scaler branch — guarded at __init__)
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
        """Run the composed v1 head; RAW scaled preds in EVERY non-training mode (plan 34 W34.3).

        The v1 head is handed a SINGLE-STREAM targets dict (its
        ``input_name_mask`` slicing is the identity on per-stream inputs). The
        per-sample weight (when configured) rides in that dict so the composed
        ``nan_loss`` finds it (task.py:430).

        Mode semantics (plan 34 W34.3 atomic forward-flip — design §2): the task
        now publishes the RAW (training-space, SCALED) ``[..., R]`` predictions
        in EVERY non-training mode (FIT/VAL unchanged; **TEST/ONNX flipped HERE**
        — they used to ``run_inference``-de-scale, returning the physical values,
        gaussian as a re-concatenated ``[..., 2R]`` means‖stds array). The eval
        de-scaling (incl. the gaussian means‖softplus-stddev one-array concat) is
        now OWNED by ``get_output`` on the live path (the ``RunTaskOutput``
        consumer reads this raw leaf and de-scales it once) and by ``get_h5`` on
        the transitional M4.5 oracle path (it reads this raw leaf and de-scales
        once). The de-scaling happens exactly ONCE per leaf on either path (no
        double-de-scale, no zero-de-scale — plan §4). The mode-split denominator
        SOURCE (FD §3.3: labels in TEST, the input Feature by name in ONNX) moves
        with the de-scaling onto ``get_output`` / ``get_h5``.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        assert self.task is not None, "forward before bind()"
        x = b.get(self.input_key)
        ctx = b.get(self.context) if self.context is not None else None
        # objects-stream (query-bank) heads have no pad mask (v1 task.py:547):
        # the composed v1 head's own exemption keys on input_name == "objects",
        # so passing pad_masks=None here matches its non-masking branch exactly.
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
                # matched-loss feature head: publish preds + the SCALED targets
                # the matcher uses (v1 maskformer_loss.py:329 stores
                # task.get_targets(labels) under labels["objects"][name]); the
                # standalone loss is DISCARDED (v1 :330) — the matched loss owns
                # losses.regression
                return {
                    self.pred_key: preds,
                    self.targets_key: self.task.get_targets(targets_dict),
                }
            return {self.pred_key: preds, self.loss_key: loss}
        # TEST|ONNX (plan 34 W34.3 atomic flip): publish the RAW (scaled) preds in
        # EVERY non-training mode — the regression task no longer de-scales here.
        # The eval de-scaling (incl. the gaussian means‖stds concat) is OWNED by
        # get_output on the live path and by get_h5 on the transitional oracle
        # path (both read this raw leaf and de-scale ONCE). v1 forward with an
        # empty targets dict returns the raw scaled preds.
        preds, _ = self.task(x, {}, pad_masks, context=ctx)
        return {self.pred_key: preds}

    def _descale_source(self, b: Bundle, mode: Mode) -> dict[str, dict[str, Tensor]]:
        """Build the per-denominator de-scaling source dict (FD §3.3 mode split).

        ``run_inference`` reads ``labels[input_name][denom]`` (task.py:587-589),
        so this returns that exact nesting — sourced from the label group in
        TEST and gathered by NAME from the raw input Feature tensor in ONNX
        (the only export-time source, to_onnx.py:381-398).

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
        """De-scale the RAW ``preds.*`` leaf to physical values (plan 34 W34.3).

        The de-scaling math the W34.3 flip moved OFF ``forward`` (which now
        publishes RAW scaled preds) and ONTO the serialisation path. Reads the
        RAW ``preds.*`` leaf + the mode-split denominator source (FD §3.3: labels
        in TEST, the input Feature by name in ONNX) + the pad mask, and runs
        ``self.task.run_inference`` VERBATIM — the SAME call the old TEST/ONNX
        forward made (tasks.py pre-flip). A gaussian head's ``run_inference``
        returns a ``(means, stds)`` TUPLE re-concatenated to ONE ``[..., 2R]``
        array (means ‖ stds) here (FD 1567-1568 one-array contract). Both
        ``get_h5`` (oracle path) and ``get_output`` (live path) call this so the
        de-scaling happens once and is byte-identical between the two.

        Returns
        -------
        Tensor
            The de-scaled physical predictions: ``[..., R]`` (plain) or
            ``[..., 2R]`` (gaussian, means ‖ stds).
        """
        assert self.task is not None, "de-scale before bind()"
        # clone before the in-place de-scale: `run_inference` does `preds.float()`
        # (a no-op ALIAS when the leaf is already float32) then mutates it in place
        # (`preds[..., i] *= ...`), so a bare `b.get(...)` would rewrite the bundle's
        # RAW `preds.*` leaf — a write-once §2.1 violation that double-de-scales on
        # any second read of the leaf. `.float().clone()` matches the producer's
        # identical write-once guard (`producers.py:902`) so both paths leave the
        # raw leaf byte-unchanged (plan §4: de-scale exactly ONCE per leaf).
        preds = b.get(self.pred_key).float().clone()
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        labels = self._descale_source(b, mode) if self.target_denominators is not None else None
        descaled = self.task.run_inference(preds, labels=labels, pad_mask=mask)
        if self.gaussian:
            # v1 GaussianRegressionTask.run_inference returns (means, stds);
            # publish ONE [..., 2R] array (means ‖ stds) for the single-leaf
            # graph contract (FD 1567-1568)
            means, stds = descaled
            return torch.cat([means, stds], dim=-1)
        return descaled

    # -- output rendering (v1 RegressionTask.output_names/get_h5/get_onnx) ------

    def output_names(self, run_name: str) -> list[tuple[str, str]]:
        """One ``f4`` column per output, named ``{run_name}_{suffix}`` (v1 task.py:511-517).

        The suffixes ARE `output_suffixes` (``custom_output_names`` else the
        targets; doubled for a gaussian head — R means then R ``_stddev``),
        the SAME suffixes the ONNX manifest carries.

        Returns
        -------
        list[tuple[str, str]]
            ``(column, "f4")`` pairs, one per output, in column order.
        """
        return [(f"{run_name}_{suffix}", "f4") for suffix in self.output_suffixes]

    def get_h5(self, b: Bundle, run_name: str) -> np.ndarray:
        """The de-scaled values as ``f4`` columns (v1 task.py:604-623).

        Since the plan 34 W34.3 flip the TEST ``forward`` publishes RAW scaled
        preds (design §2), so this M4.5-oracle path now de-scales ITSELF via
        ``_descaled_preds`` (``run_inference`` + the gaussian means‖stds concat —
        the SAME math the forward used to run) before the ``de-scaled values ->
        f4 u2s`` pack. The conversion happens exactly ONCE on this path (the
        producer / get_output path de-scales independently — no double-de-scale,
        plan §4). The OUTPUT is byte-identical to pre-flip.

        Returns
        -------
        np.ndarray
            ``[B]`` (global) or ``[B, L]`` (sequence) structured array.
        """
        preds = self._descaled_preds(b, Mode.TEST)
        dtype = np.dtype(self.output_names(run_name))
        return u2s(preds.float().cpu().numpy(), dtype)

    def onnx_outputs(self) -> list[ExportOutput]:
        """The LEGACY M4.5 ``split_scalars`` manifest entry — RETIRED on the export path.

        This is the v1 ``get_onnx`` (task.py:625-642) manifest shape: one
        ``split_scalars`` entry naming the R suffixes (`output_suffixes` minus the
        run-name prefix — the exporter prepended ``{model_name}_``). It is the M4.5
        ``WriterCallback.onnx_manifest`` API and is still ASSERTED by the manifest
        gates (names/reduce/suffix lockstep), but it NO LONGER drives a real ONNX
        export: plan-29 W4 retired the off-graph reduce manifest, so ``split_scalars``
        is unregistered (``reduces.py``), ``export_graph`` rejects a non-empty
        ``outputs=`` manifest, and ``OnnxAdapter`` requires a folded ``OnnxExportSink``.

        CRITICAL (plan 34 W34.3): the entry's ``port`` is the RAW ``preds.*`` leaf
        the W34.3 forward-flip now publishes (pre-flip the forward de-scaled, so this
        port held PHYSICAL values; post-flip it holds RAW scaled values). Because
        ``split_scalars`` is a naming-only splitter (it never de-scales), this entry
        would emit RAW values IF it could ever run — but it cannot (export-retired,
        above). The LIVE ONNX de-scale lives on the ``Regression`` (``RegressionDescaleOp``)
        producer / the ``outputs:`` section ``get_output`` (``_descaled_preds`` ->
        traceable ``run_inference``), which a config wires via an ``OnnxExportSink``.
        A regression config WITHOUT an ``OnnxExportSink`` cannot ONNX-export its
        regression head at all (it does not silently export raw).

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
        """The non-pred deps `get_output` reads: denom source(s) + pad mask (plan 34 W34.3).

        Mirrors the legacy `RegressionDescaleOp.extra_requires` (producers.py:808)
        plus the pad mask, mode-aware (FD §3.3 mode split):

        - a ratio head (``target_denominators``) reads its denominator source —
          ``labels.<stream>.<denom>`` in FIT|VAL|TEST, the raw input Feature
          ``inputs.<stream>`` (gathered by NAME) in ONNX.
        - a padded sequence head reads the stream pad mask ``masks.<stream>`` (the
          ``run_inference`` NaN-fill). A global / objects-stream head needs no
          pad mask.

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
        """De-scale the RAW preds into graph-visible `OutputField`s (plan 34 W34.3).

        Reads the RAW ``preds.*`` leaf (forward is loss-space since the W34.3
        flip) + the mode-split denominator source + the pad mask, and de-scales
        in TRACEABLE torch ops via ``_descaled_preds`` (``self.task.run_inference``
        VERBATIM — the SAME math the legacy `RegressionDescaleOp` producer /
        the pre-flip forward ran; incl. the gaussian means‖softplus-stddev
        one-array concat). It then mints ONE ``f4`` `OutputField` PER output
        column (``output_suffixes`` — ``custom_output_names`` else the targets,
        DOUBLED for a gaussian head: R means then R ``_stddev``). The H5 and ONNX
        suffixes are the SAME (regression's ``output_names`` H5 columns AND
        ``split_scalars`` ONNX scalars share the suffixes), so each field carries
        both (``onnx_name`` defaults to ``h5_name``). The per-column value shape
        is MODE-FAITHFUL to the live sink:

        - **H5 modes** (TEST): the per-column ``[B]`` (global) / ``[B, L]`` (seq)
          value — exactly the float column ``get_h5`` packs. No squeeze.
        - **ONNX**: the v1 ``split_scalars`` shape — ``torch.split(preds, 1, -1)``
          then ``.squeeze()`` (drops ALL size-1 dims). On the batch-1 ONNX trace a
          global head's de-scaled preds are ``[1, R]`` so ``preds[..., i]`` is
          ``[1]`` -> ``.squeeze()`` yields the ``[]`` scalar Athena/v1 expects; a
          per-token head's ``[1, L, R]`` -> ``preds[..., i]`` is ``[1, L]`` ->
          ``.squeeze()`` yields the ``[L]`` per-token vector (batch dim dropped).
          The dumb `OnnxExportSink` then NAMES each already-shaped value directly
          (single-``name`` leaf, no re-split — the W34.2 LOCKED no-double-split
          decision).

        ``run_name`` is NOT baked in (the sink prefixes), so it is deleted.

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
        """The value-free field metadata mirroring `get_output` for `mode` (plan 34 W34.3).

        Returns the SAME `OutputField`s `get_output` mints (one ``f4`` field per
        ``output_suffixes`` entry — gaussian-doubled — global or per_token,
        value-free) so the dumb sinks resolve the column schema before any batch
        runs. The H5/ONNX suffixes are identical, so each field carries both
        (``onnx_name`` defaults to ``h5_name``).

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
    """Absorbed v2-native `VertexingTask` with config-driven heavy/fake origin ids (design §3.3).

    Subclasses the M7 W2c-1 ABSORBED `_AbsorbedVertexingTask` base (was: the v1
    `salt.models.task.VertexingTask`); the subclass relationship now holds
    v2-natively. With the default ids (3,4,5 / 1) `get_weights` is bit-identical
    to v1's hardcoded version (task.py:957-964): both build the heavy indicator
    via clipped sums and AND them pairwise over the adjacency.
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

    Mirrors v1 ``listify`` (array_utils.py:40) but returns a tuple (this
    module's immutable-config convention) and never explodes on ``None``.

    Returns
    -------
    tuple[str, ...] | None
        The listified tuple, or None when the input is None.
    """
    if value is None:
        return None
    return tuple(listify(value))


def _is_name_weighting(heavy: Sequence[Any], fake: Sequence[Any]) -> bool:
    """Whether an ``origin_weighting`` config is class-name based (design §5.1).

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
    """Parse a task ``expose:`` mode-name list into a `Mode` flag (design §4.2).

    ``None`` (the default) means all modes — the task publishes ``preds.*`` in
    fit/val/test/onnx as before. A list of mode names (case-insensitive,
    ``fit``/``val``/``test``/``onnx``) gates the prediction port to exactly
    those modes: a train-only aux task uses ``expose: [fit, val]`` so its
    prediction is pruned from the TEST/ONNX plans (and the TEST dead-preds
    error is silenced) instead of needing the ``--model.modules.X=null``
    deletion workaround.

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
    """Validate a ``weight_source`` mapping (design §3.3).

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
