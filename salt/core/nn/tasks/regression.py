"""Regression task module, plain + gaussian (+ the absorbed v1 regression heads)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.stream_embed import _stream_len
from salt.core.nn.tasks.base import _AbsorbedTaskBase, _loss_class, _TaskModuleBase
from salt.core.onnx.config import ExportOutput
from salt.core.outputs.producers import OutputField
from salt.core.utils.array_utils import listify
from salt.core.utils.scalers import RegressionTargetScaler

_DEFAULT_REG_LOSS: dict[str, Any] = {"class_path": "torch.nn.MSELoss"}
_DEFAULT_GAUSS_LOSS: dict[str, Any] = {"class_path": "torch.nn.GaussianNLLLoss"}


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
