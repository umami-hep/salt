"""Regression task module, plain + gaussian."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.model.bind import ResolvedSchema
from salt.model.modules.stream_embed import _stream_len
from salt.model.modules.tasks.base import _loss_class, _TaskModuleBase
from salt.model.nn.dense import Dense
from salt.outputs.output_schema import OutputField
from salt.utils.array_utils import listify
from salt.utils.scalers import RegressionTargetScaler

_DEFAULT_REG_LOSS: dict[str, Any] = {"class_path": "torch.nn.MSELoss"}
_DEFAULT_GAUSS_LOSS: dict[str, Any] = {"class_path": "torch.nn.GaussianNLLLoss"}


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
    dataset-side sibling (`salt.data.processors.MultiTarget`), not a task
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
        write_targets: bool = True,
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
        write_targets : bool, optional
            Also emit the UNSCALED physical target(s) as the TEST eval columns
            ``target_{task}_{target}`` (one f4 per target; padded positions
            NaN), by default True. Never emitted in ONNX/export mode. Distinct
            from ``publish_targets`` (the FIT|VAL scaled-targets port for the
            MaskFormer matcher).

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
        super().__init__(
            stream, "", input, context, dense, loss, weight, default_loss, expose, write_targets
        )
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
        # the built `RegressionTargetScaler` (bind constructs it from scaler_scales)
        self.scaler: RegressionTargetScaler | None = None
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
        """Normalise ``norm_params`` to ``{"mean": [...], "std": [...]}``
        (listified), or None; raises `ConfigError` if present but missing
        either key.
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

    # -- the head math (loss + target scaling + inference de-scaling) -----------

    def nan_loss(self, preds: Tensor, targets: Tensor, targets_dict: Mapping, **kwargs) -> Tensor:
        """Loss that ignores NaN targets.

        Returns
        -------
        Tensor
            Mean loss over non-NaN elements.

        Raises
        ------
        ValueError
            If the resulting loss becomes NaN. Eager only: the check reads a
            tensor value, which is data-dependent control flow and would force a
            graph break, so it is skipped inside a compiled region.
        """
        invalid = torch.isnan(targets)
        preds = torch.where(invalid, torch.zeros_like(preds), preds)
        targets = torch.where(invalid, torch.zeros_like(targets), targets)

        if "var" in kwargs:
            kwargs["var"] = torch.where(invalid, torch.zeros_like(kwargs["var"]), kwargs["var"])

        loss = self.loss(preds, targets, **kwargs)

        if len(loss.shape) == 0:
            if not torch.compiler.is_compiling() and torch.isnan(loss):
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
        if not torch.compiler.is_compiling() and torch.isnan(nanmean):
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

    def head_forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute regression predictions and loss.

        A plain head predicts ``[..., R]`` values; a gaussian head predicts
        ``[..., 2R]`` (means ‖ raw variances) and takes the Gaussian NLL of
        the means against the (softplus) variances.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Predicted values and the loss (``None`` when no targets).
        """
        if self.gaussian:
            if pad_masks is not None:
                preds = self.net(x[:, self.input_name_slice(pad_masks)], context)
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

        if pad_masks is not None and self.input_name != "objects":
            preds = self.net(x[:, self.input_name_slice(pad_masks)], context)
            pad_mask = pad_masks[self.input_name]
        else:
            preds = self.net(x, context)
            pad_mask = None

        targets = self.get_targets(targets_dict)

        # NaN-fill padded targets so nan_loss excludes them
        if pad_mask is not None and targets is not None:
            targets = torch.masked_fill(targets, pad_mask.unsqueeze(-1), torch.nan)

        loss = None
        if targets is not None:
            loss = self.nan_loss(preds, targets, targets_dict) * self.weight

        return preds, loss

    def run_inference(
        self, preds: Tensor, labels: Mapping | None = None, pad_mask: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Invert target scaling to the original space.

        Indexes the trailing (target-channel) axis ``preds[..., i]`` so a
        per-token ``[B, L, R]`` head de-scales column ``i`` of every token
        (not just token ``i``); bit-identical to indexing axis 1 for a global
        ``[B, R]`` head. A gaussian head de-scales means + (sqrt of softplus)
        variances and returns the ``(means, stds)`` pair instead.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            De-scaled predictions with NaN padding (plain head), or de-scaled
            ``(means, stds)`` each of shape ``[..., R]`` (gaussian head).

        Raises
        ------
        ValueError
            If a gaussian head is called without the necessary scaling
            parameters.
        """
        if self.gaussian:
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

    @property
    def output_suffixes(self) -> tuple[str, ...]:
        """Per-output column suffixes (custom names override targets); doubled for
        a gaussian head — R means then R ``<suffix>_stddev``, index-aligned with
        the ``[B, 2R]`` array.
        """
        base = self.custom_output_names if self.custom_output_names is not None else self.targets
        if self.gaussian:
            return (*base, *(f"{s}_stddev" for s in base))
        return base

    @property
    def target_label_keys(self) -> tuple[str, ...]:
        """``labels.<stream>.<target>`` keys, in target order."""
        return tuple(f"labels.{self.stream}.{t}" for t in self.targets)

    @property
    def denom_label_keys(self) -> tuple[str, ...]:
        """``labels.<stream>.<denom>`` keys (FIT|VAL|TEST source), in target order, or empty."""
        if self.target_denominators is None:
            return ()
        return tuple(f"labels.{self.stream}.{d}" for d in self.target_denominators)

    @property
    def input_feature_key(self) -> str:
        """``inputs.<stream>`` — the raw-input key carrying the ONNX denominator columns."""
        return f"inputs.{self.stream}"

    @property
    def weight_label_key(self) -> str:
        """``labels.<stream>.<sample_weight>`` (FIT|VAL only; meaningful iff
        `sample_weight` set).
        """
        return f"labels.{self.stream}.{self.sample_weight}"

    @property
    def targets_key(self) -> str:
        """``targets.<stream>.<instance-name>`` — the published scaled-targets key
        (``publish_targets``, FIT|VAL only).
        """
        return f"targets.{self.stream}.{self.name}"

    def declare_io(self, mode: Mode) -> IO:
        """Denominator dep is mode-split: labels in FIT|VAL|TEST, the raw ``inputs.<stream>``
        Feature in ONNX. Targets/sample-weight/loss are TRAINING-gated; a gaussian head
        publishes a ``2R``-wide prediction (means then stddevs).
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
        """``output_size = len(targets)`` (doubled for gaussian); when `target_denominators` is
        set, every denominator must be a declared column of ``inputs.<stream>`` (else `bind`
        raises) so the ONNX graph can gather it by name.
        """
        if self.gaussian and self.norm_params is None and self.target_denominators is None:
            raise ConfigError(
                f"RegressionTaskModule {self.name!r}: a gaussian head requires norm_params or "
                "target_denominators — v1 GaussianRegressionTask.run_inference raises without "
                "scaling params (task.py:765-766)"
            )
        init_args = dict(self.loss_cfg.get("init_args", {}))
        self.loss = _loss_class(self.loss_cfg)(**init_args)
        # only the plain head takes the functional `scaler` (gaussian has no
        # scaler branch — guarded at __init__); an unknown target surfaces at
        # the first get_targets/run_inference call, as before
        self.scaler = RegressionTargetScaler(self.scaler_scales) if self.scaler_scales else None
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
        self.net = Dense(
            input_size=schema.width(self.input_key),
            output_size=(2 if self.gaussian else 1) * len(self.targets),
            **({"context_size": schema.width(self.context)} if self.context else {}),
            **self.dense_cfg,
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
        """Publishes RAW (scaled) preds in every non-training mode; de-scaling
        happens exactly once downstream in `get_output`. A
        `publish_targets` head returns scaled targets instead of a loss.
        """
        assert self.net is not None, "forward before bind()"
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
            preds, loss = self.head_forward(x, targets_dict, pad_masks, context=ctx)
            if self.publish_targets:
                # matched-loss feature head: publish preds + scaled targets for the
                # matcher; the standalone loss is discarded (MaskFormerMatchedLoss
                # owns losses.regression instead)
                return {
                    self.pred_key: preds,
                    self.targets_key: self.get_targets(targets_dict),
                }
            return {self.pred_key: preds, self.loss_key: loss}
        preds, _ = self.head_forward(x, {}, pad_masks, context=ctx)
        return {self.pred_key: preds}

    def _descale_source(self, b: Bundle, mode: Mode) -> dict[str, dict[str, Tensor]]:
        """``{stream: {denom: tensor}}`` in the nesting `run_inference` expects:
        from the label group in TEST, gathered by name from the raw input
        Feature tensor in ONNX.
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
        """De-scale the RAW ``preds.*`` leaf via `run_inference` to ``[..., R]`` (plain) or
        ``[..., 2R]`` means-then-stds (gaussian); shared by `get_output` calls so de-scaling
        happens exactly once.
        """
        assert self.net is not None, "de-scale before bind()"
        # clone before the in-place de-scale: run_inference mutates preds[..., i]
        # in place, so a bare b.get(...) would corrupt the bundle's RAW preds.*
        # leaf and double-de-scale on any later read
        preds = b.get(self.pred_key).float().clone()
        mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
        labels = self._descale_source(b, mode) if self.target_denominators is not None else None
        descaled = self.run_inference(preds, labels=labels, pad_mask=mask)
        if self.gaussian:
            # run_inference returns (means, stds); publish ONE [..., 2R] array
            # (means ‖ stds) for the single-leaf graph contract
            means, stds = descaled
            return torch.cat([means, stds], dim=-1)
        return descaled

    # -- output rendering ---------------------------------------------------

    def output_time_requires(self, mode: Mode) -> list[str]:
        """A ratio head's denominator source (labels in FIT|VAL|TEST, ``inputs.<stream>`` in
        ONNX) plus the pad mask for a padded sequence head; ``norm_params``/``scaler`` need
        no external source. In TEST a ``write_targets`` head also demands its target label
        keys (never in ONNX).
        """
        deps: list[str] = []
        if self.target_denominators is not None:
            if mode & Mode.ONNX:
                deps.append(self.input_feature_key)
            else:
                deps.extend(self.denom_label_keys)
        if self.has_pad_mask:
            deps.append(f"masks.{self.stream}")
        if self._emit_targets(mode):
            deps.extend(k for k in self.target_label_keys if k not in deps)
        return deps

    def get_output(self, b: Bundle, mode: Mode, run_name: str) -> list[OutputField]:
        """One ``f4`` field per `output_suffixes` column, de-scaled via `_descaled_preds`.
        H5 modes keep the per-column ``[B]``/``[B, L]`` value; ONNX squeezes the size-1
        batch dim to a rank-0 scalar (global) or ``[L]`` vector (per-token).
        """
        del run_name
        preds = self._descaled_preds(b, mode)
        axis = "per_token" if self.sequence else "global"
        squeeze = bool(mode & Mode.ONNX)
        fields = [
            OutputField(
                h5_name=suffix,
                dtype="f4",
                axis=axis,
                final=True,
                value=preds[..., i].squeeze() if squeeze else preds[..., i],
            )
            for i, suffix in enumerate(self.output_suffixes)
        ]
        if self._emit_targets(mode):
            mask = b.get(f"masks.{self.stream}") if self.has_pad_mask else None
            for field, key in zip(self._target_fields(), self.target_label_keys, strict=True):
                raw = b.get(key).float()
                if mask is not None:
                    # padded positions NaN, matching the de-scaled prediction columns
                    raw = torch.masked_fill(raw, mask, torch.nan)
                fields.append(replace(field, value=raw))
        return fields

    def _target_fields(self) -> list[OutputField]:
        """One value-free target-label field per target: the UNSCALED physical target
        as an unprefixed ``target_{task}_{target}`` f4 column (one per `targets`
        entry — R, not 2R, for a gaussian head).
        """
        axis = "per_token" if self.sequence else "global"
        return [
            OutputField(
                h5_name=f"target_{self.name}_{target}",
                onnx_name=None,
                dtype="f4",
                axis=axis,
                final=True,
                prefix=False,
            )
            for target in self.targets
        ]

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """The value-free field metadata mirroring `get_output` for `mode` (``value=None``)."""
        del run_name
        axis = "per_token" if self.sequence else "global"
        fields = [
            OutputField(h5_name=suffix, dtype="f4", axis=axis, final=True)
            for suffix in self.output_suffixes
        ]
        if self._emit_targets(mode):
            fields.extend(self._target_fields())
        return fields


def _opt_tuple(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Listify a scalar-or-list config value to a tuple, preserving ``None``."""
    if value is None:
        return None
    return tuple(listify(value))
