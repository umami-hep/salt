"""Mixture density regression task module: a K-component Gaussian mixture per target value."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.model.modules.tasks.regression import RegressionTaskModule
from salt.outputs.output_schema import OutputField


class MixtureDensityTaskModule(RegressionTaskModule):
    """Regression head that models one target with a K-component Gaussian mixture.

    Useful when the target has several likely values, where a single Gaussian regresses
    towards the prior mean with a large variance. The head publishes the raw ``[..., 3K]``
    output (component means ‖ pre-softplus variances ‖ mixture-weight logits) and is
    trained with `MixtureGaussianNLLLoss` in scaled target space, built internally.

    The outputs describe the dominant component (largest mixture weight): its de-scaled
    mean as ``<target>`` and its weight as ``<target>_modeweight``. ``<target>_stddev`` is
    the dominant component's own sigma in the H5 output, and in the ONNX output
    ``sigma_fair = sqrt(Var[y] + (mean_mix - mean_dom)^2)``, the law-of-total-variance
    spread of the mixture around the dominant mean, since the dominant sigma understates
    the uncertainty whenever rival modes exist. The H5 output also carries the mixture
    mean ``_mubar``, ``_sigmix = sqrt(Var[y])``, ``_sigfair`` and every component in raw
    order, ``_mu{k}``, ``_sigma{k}`` and ``_pi{k}``.

    Parameters
    ----------
    stream : str
        Stream the target label lives on (``labels.<stream>.<target>``).
    targets : str | Sequence[str]
        The regression target, exactly one.
    n_components : int
        Number of mixture components K, at least 1.
    input : str | None, optional
        Input key, by default ``encoded.<stream>``.
    context : str | None, optional
        Context key for the dense head, by default None.
    sequence : bool | None, optional
        Whether the task is per-token, by default inferred from `input`.
    target_denominators : str | Sequence[str] | None, optional
        Ratio denominator variable. Exactly one of this and `norm_params` is required.
    norm_params : Mapping[str, Any] | None, optional
        ``{"mean": ..., "std": ...}`` target normalisation.
    custom_output_names : str | Sequence[str] | None, optional
        Output column name overriding the target name, by default None.
    sample_weight : str | None, optional
        Per-sample loss-reweighting label (FIT|VAL only), by default None.
    dense : dict[str, Any] | None, optional
        `Dense` head config; its output width is ``3 * n_components``.
    weight : float, optional
        Scalar task-loss weight, by default 1.0.
    eps : float, optional
        Added to the softplus variance for numerical stability, by default 1e-6.
    expose : Sequence[str] | None, optional
        Modes the ``preds.*`` port is published in, by default all modes.
    write_targets : bool, optional
        Also emit the unscaled target as the TEST column ``target_{task}_{target}``,
        by default True.

    Raises
    ------
    ConfigError
        On more than one target, ``n_components < 1``, or neither `norm_params` nor
        `target_denominators` set, besides the `RegressionTaskModule` checks.
    """

    def __init__(
        self,
        stream: str,
        targets: str | Sequence[str],
        n_components: int,
        input: str | None = None,  # noqa: A002 - matches the YAML config key name
        context: str | None = None,
        sequence: bool | None = None,
        target_denominators: str | Sequence[str] | None = None,
        norm_params: Mapping[str, Any] | None = None,
        custom_output_names: str | Sequence[str] | None = None,
        sample_weight: str | None = None,
        dense: dict[str, Any] | None = None,
        weight: float = 1.0,
        eps: float = 1e-6,
        expose: Sequence[str] | None = None,
        write_targets: bool = True,
    ) -> None:
        # the mixture NLL is always per sample, so the base check of a configured
        # loss reduction does not apply: sample_weight is set after the base init
        super().__init__(
            stream,
            targets,
            input=input,
            context=context,
            sequence=sequence,
            target_denominators=target_denominators,
            norm_params=norm_params,
            custom_output_names=custom_output_names,
            dense=dense,
            loss={
                "class_path": "salt.model.nn.losses.MixtureGaussianNLLLoss",
                "init_args": {"eps": eps},
            },
            weight=weight,
            expose=expose,
            write_targets=write_targets,
        )
        self.sample_weight = sample_weight
        self.weight_label_key = f"labels.{stream}.{sample_weight}"
        self.n_components = n_components
        self.eps = eps
        name = type(self).__name__
        if len(self.targets) != 1:
            raise ConfigError(
                f"{name}: supports exactly one target, got {len(self.targets)} "
                f"({list(self.targets)})"
            )
        if n_components < 1:
            raise ConfigError(f"{name}: n_components must be >= 1, got {n_components}")
        if self.norm_params is None and self.target_denominators is None:
            raise ConfigError(
                f"{name}: de-scaling the outputs requires norm_params or target_denominators"
            )
        suffix = (self.custom_output_names or self.targets)[0]
        self.output_suffixes = (suffix, f"{suffix}_stddev", f"{suffix}_modeweight")

    @property
    def _pred_width(self) -> int:
        """Head output width: means, raw variances and logits, ``3 * n_components``."""
        return 3 * self.n_components

    def nan_loss(self, preds: Tensor, targets: Tensor, targets_dict: Mapping, **kwargs) -> Tensor:
        """Mixture NLL averaged over the non-NaN (unpadded) targets, optionally
        sample-weighted; raises `ValueError` (eager only) when no valid target remains.
        """
        del kwargs
        assert self.loss is not None, "loss before bind()"
        invalid = torch.isnan(targets)
        safe = torch.where(invalid, torch.zeros_like(targets), targets)
        nll = torch.where(invalid, torch.nan, self.loss(*preds.tensor_split(3, -1), safe))
        if self.sample_weight is not None:
            nll = nll * targets_dict[self.input_name][self.sample_weight].unsqueeze(1)
        loss = torch.nanmean(nll)
        if not torch.compiler.is_compiling() and torch.isnan(loss):
            raise ValueError(
                f"{self.name}: mixture NLL is NaN. Either all targets are padded or the model "
                "predictions are NaN."
            )
        return loss

    def _descaled_preds(self, b: Bundle, mode: Mode) -> Tensor:
        """The `get_output_manifest` columns, de-scaled, NaN at padded positions: in ONNX the
        dominant mean, sigma_fair and dominant weight; in H5 modes the dominant mean, sigma
        and weight, mixture mean, sigmix, sigma_fair, then all means, sigmas and weights.
        """
        preds = b.get(self.pred_key).float()
        means, raw_vars, logits = preds.tensor_split(3, -1)
        stds = torch.sqrt(nn.functional.softplus(raw_vars) + self.eps)
        weights = torch.softmax(logits, dim=-1)
        denoms, norm = self._denominators(b, mode), self.norm_params
        if denoms:
            den = denoms[0].float().unsqueeze(-1)
            means, stds = means * den, stds * den
        elif norm is not None:
            means = means * norm["std"][0] + norm["mean"][0]
            stds = stds * norm["std"][0]
        k = weights.argmax(dim=-1, keepdim=True)
        mu_dom, pi_dom = torch.gather(means, -1, k), torch.gather(weights, -1, k)
        mu_bar = (weights * means).sum(-1, keepdim=True)
        var = (weights * (stds**2 + (means - mu_bar) ** 2)).sum(-1, keepdim=True)
        sig_fair = torch.sqrt(var + (mu_bar - mu_dom) ** 2)
        if mode & Mode.ONNX:
            out = torch.cat([mu_dom, sig_fair, pi_dom], dim=-1)
        else:
            sg_dom = torch.gather(stds, -1, k)
            summary = [mu_dom, sg_dom, pi_dom, mu_bar, torch.sqrt(var), sig_fair]
            out = torch.cat([*summary, means, stds, weights], dim=-1)
        if self.has_pad_mask:
            out = torch.masked_fill(out, b.get(f"masks.{self.stream}").unsqueeze(-1), torch.nan)
        return out

    def get_output_manifest(self, mode: Mode, run_name: str) -> list[OutputField]:
        """H5 modes add the mixture summaries and every component after the three
        dominant-component columns.
        """
        fields = super().get_output_manifest(mode, run_name)
        if mode & Mode.ONNX:
            return fields
        suffix = self.output_suffixes[0]
        components = [f"{p}{i}" for p in ("mu", "sigma", "pi") for i in range(self.n_components)]
        extra = [
            replace(fields[0], h5_name=f"{suffix}_{n}")
            for n in ("mubar", "sigmix", "sigfair", *components)
        ]
        return [*fields[:3], *extra, *fields[3:]]
