"""Mixture density network (MDN) regression task head.

A :class:`MixtureDensityTask` models the regression target with a K-component
Gaussian mixture instead of a single Gaussian. This is useful when the
posterior over the target is multimodal (e.g. several discrete candidate
values): a unimodal head fitted to a multimodal posterior regresses
towards the prior mean with a large variance, whereas a mixture can keep a
component per candidate mode. At inference time single component can be chosen
based on, for example, highest probability (pi_k) of lowest sigma (var_k).
"""

import math
from collections.abc import Mapping

import numpy as np
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from torch import Tensor, nn

from salt.models.task import RegressionTask, RegressionTaskBase
from salt.stypes import Tensors
from salt.utils.scalers import RegressionTargetScaler


class MixtureGaussianNLLLoss(nn.Module):
    """Negative log-likelihood of a 1D K-component Gaussian mixture.

    Computed with 'logsumexp' for numerical stability:
    NLL = -logsumexp_k[ log pi_k - 0.5*(log(2*pi*var_k) + (y - mu_k)^2 / var_k) ]

    Parameters
    ----------
    eps : float, optional
        Small constant added to the softplus-activated variance for numerical
        stability. The default is 1e-6.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Per-sample (unreduced) output;
        # Also satisfies the RegressionTaskBase assertion 'loss.reduction == "none"' that is
        # triggered when `sample_weight` is configured.
        self.reduction = "none"

    def forward(self, means: Tensor, raw_vars: Tensor, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the per-sample mixture NLL.

        Parameters
        ----------
        means : Tensor
            Component means of shape [..., K] (in scaled target space).
        raw_vars : Tensor
            Pre-softplus component variances of shape [..., K].
        logits : Tensor
            Unnormalized mixture weights of shape [..., K].
        targets : Tensor
            Targets of shape [..., 1] (broadcast against the K components).

        Returns
        -------
        Tensor
            Per-sample NLL of shape [..., 1].
        """
        var = nn.functional.softplus(raw_vars) + self.eps
        log_pi = nn.functional.log_softmax(logits, dim=-1)
        log_comp = -0.5 * (
            math.log(2 * math.pi) + torch.log(var) + (targets - means).square() / var
        )
        return -torch.logsumexp(log_pi + log_comp, dim=-1, keepdim=True)


class MixtureDensityTask(RegressionTask):
    """Gaussian mixture density regression head (single target, K components).

    The dense head outputs 3 * n_components values per object, split (like
    GaussianRegressionTask's tensor_split(2, -1) pattern) as
    means, raw_vars, logits = preds.tensor_split(3, -1):

    - [..., 0:K]   component means (in scaled target space),
    - [..., K:2K]  raw variances (softplus is applied in the loss/inference,
      NOT to the returned 'preds', same convention as GaussianRegressionTask),
    - [..., 2K:3K] mixture weight logits (softmax applied in loss/inference).

    Target extraction and scaling (norm_params / target_denominators /
    scaler) are reused unchanged from RegressionTaskBase.get_targets.

    Parameters
    ----------
    name : str
        Task name, used for logging and inference output prefixes.
    input_name : str
        Input stream (e.g. "jets" for a global task).
    dense_config : dict
        Keyword args for :class:'salt.models.Dense'; 'output_size' must be
        '3 * n_components'.
    targets : list[str] | str
        The regression target (exactly one).
    n_components : int
        Number of Gaussian mixture components K.
    loss : nn.Module | None, optional
        Must be 'None' (default) or a :class:'MixtureGaussianNLLLoss'
        instance; anything else raises "omit it from your YAML config".
    weight : float, optional
        Scalar multiplier for the task loss. The default is 1.0.
    scaler : RegressionTargetScaler | None, optional
        Functional target scaler (training only; inference de-scaling is
        implemented for norm_params / target_denominators).
    target_denominators : list[str] | str | None, optional
        Denominator variable for ratio targets.
    norm_params : dict | None, optional
        {"mean": ..., "std": ...} normalization of the target.
    custom_output_names : list[str] | str | None, optional
        Optional custom output name overriding the target name.
    sample_weight : str | None, optional
        Key of a per-sample weight in 'labels[input_name]'.
    eps : float, optional
        Numerical-stability constant for the variance. The default is 1e-6.

    Raises
    ------
    TypeError
        If 'loss' is provided but is not a 'MixtureGaussianNLLLoss'.
    ValueError
        If 'n_components < 1', more than one target is configured, or
        'dense_config.output_size != 3 * n_components'.
    """

    def __init__(
        self,
        name: str,
        input_name: str,
        dense_config: dict,
        targets: list[str] | str,
        n_components: int,
        loss: nn.Module | None = None,
        weight: float = 1.0,
        scaler: RegressionTargetScaler | None = None,
        target_denominators: list[str] | str | None = None,
        norm_params: dict | None = None,
        custom_output_names: list[str] | str | None = None,
        sample_weight: str | None = None,
        eps: float = 1e-6,
    ):
        if loss is None:
            loss = MixtureGaussianNLLLoss(eps=eps)
        # For now, omitting is the design option, later when/if
        # CRPSLoss/energy score are implemented, should be set
        # in config
        if not isinstance(loss, MixtureGaussianNLLLoss):
            raise TypeError(
                "MixtureDensityTask computes its own mixture NLL; omit `loss` from the "
                "config or pass a MixtureGaussianNLLLoss instance."
            )
        # Deliberately skip RegressionTask.__init__: its
        # output_size == len(targets) check does not apply to a 3K-output
        # mixture head. RegressionTaskBase.__init__ performs all the scaling
        # config validation that is needed.
        # Idk if the jsonargparse/LightningCLI can handle this grandparent call
        # instead of normal super() call. So every parameter is spelled out.
        RegressionTaskBase.__init__(
            self,
            name=name,
            input_name=input_name,
            dense_config=dense_config,
            loss=loss,
            weight=weight,
            targets=targets,
            scaler=scaler,
            target_denominators=target_denominators,
            norm_params=norm_params,
            custom_output_names=custom_output_names,
            sample_weight=sample_weight,
        )
        self.n_components = n_components

        if n_components < 1:
            raise ValueError(f"{self.name}: n_components must be >= 1, got {n_components}")
        if len(self.targets) != 1:
            raise ValueError(
                f"{self.name}: MixtureDensityTask supports exactly one target, "
                f"got {len(self.targets)} ({self.targets})"
            )
        if self.net.output_size != 3 * n_components:
            raise ValueError(
                f"{self.name}: dense head output_size ({self.net.output_size}) must be "
                f"3 * n_components ({3 * n_components}): [means | raw variances | logits]"
            )

    @property
    def output_names(self) -> list[str]:
        """Output field names (mode mean, mode stddev, mode weight)."""
        t = self.custom_output_names[0] if self.custom_output_names else self.targets[0]
        return [
            f"{self.model_name}_{t}",
            f"{self.model_name}_{t}_stddev",
            f"{self.model_name}_{t}_modeweight",
        ]

    def forward(
        self,
        x: Tensor,
        targets_dict: Mapping,
        pad_masks: Mapping | None = None,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute raw mixture parameters and the mixture NLL loss.

        Parameters
        ----------
        x : Tensor
            Input tensor ([B, D] for a global task).
        targets_dict : Mapping
            Nested labels dict (labels[input_name][target]); 'None' or
            empty at inference.
        pad_masks : Mapping | None, optional
            Per-stream padding masks ('None' for global tasks, this is what
            'SaltModel.run_tasks' passes for the global-object).
        context : Tensor | None, optional
            Optional context tensor. The default is 'None'.

        Returns
        -------
        Tensor
            Raw head output of shape [B, 3K] (no softplus/softmax applied).
        Tensor | None
            Scalar weighted loss if targets are available, else 'None'.
        """
        if pad_masks is not None and self.input_name != "objects":
            input_name_mask = self.input_name_mask(pad_masks)
            preds = self.net(x[:, input_name_mask], context)
            pad_mask = pad_masks[self.input_name]
        else:
            preds = self.net(x, context)
            pad_mask = None

        targets = self.get_targets(targets_dict)

        # fill targets with nan where padded to remove them from loss calculation
        if pad_mask is not None and targets is not None:
            targets = torch.masked_fill(targets, pad_mask.unsqueeze(-1), torch.nan)

        loss: Tensor | None = None
        if targets is not None:
            means, raw_vars, logits = preds.tensor_split(3, -1)
            loss = self.mixture_nan_loss(means, raw_vars, logits, targets, targets_dict)
            loss = loss * self.weight

        return preds, loss

    def mixture_nan_loss(
        self,
        means: Tensor,
        raw_vars: Tensor,
        logits: Tensor,
        targets: Tensor,
        targets_dict: Mapping,
    ) -> Tensor:
        """Mixture NLL that ignores NaN targets (mirrors 'nan_loss').

        NaN targets (e.g. padding) are zeroed before the loss
        call to avoid NaN propagation, then their per-sample NLL entries are
        excluded from the mean via 'nanmean'.

        Parameters
        ----------
        means : Tensor
            Component means, shape [..., K].
        raw_vars : Tensor
            Pre-softplus variances, shape [..., K].
        logits : Tensor
            Mixture weight logits, shape [..., K].
        targets : Tensor
            Targets, shape [..., 1]; NaN entries are excluded.
        targets_dict : Mapping
            Labels dict, used for the optional per-sample weight.

        Returns
        -------
        Tensor
            Scalar mean NLL over valid entries.

        Raises
        ------
        ValueError
            If the loss is NaN (all targets invalid or predictions NaN).
        """
        invalid = torch.isnan(targets)
        safe_targets = torch.where(invalid, torch.zeros_like(targets), targets)

        nll = self.loss(means, raw_vars, logits, safe_targets)  # [..., 1]
        # exclude invalid entries from the mean entirely
        nll = torch.where(invalid, torch.full_like(nll, torch.nan), nll)

        if self.sample_weight is not None:
            weights = targets_dict[self.input_name][self.sample_weight]
            weights = weights.unsqueeze(1)
            nll = nll * weights

        loss = torch.nanmean(nll)
        if torch.isnan(loss):
            raise ValueError(
                f"{self.name}: mixture NLL is NaN. Either all targets are NaN/padded or "
                "the model predictions are NaN."
            )
        return loss

    def _descaled_params(
        self, preds: Tensor, labels: Tensors | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """De-scale ALL mixture parameters to the original target space.

        Parameters
        ----------
        preds : Tensor
            Raw head output of shape [B, 3K].
        labels : Tensors | None, optional
            Labels, used to re-apply 'target_denominators' when configured.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            (means, stds, weights), each of shape [B, K], de-scaled.

        Raises
        ------
        ValueError
            If neither 'target_denominators' (+ labels) nor 'norm_params'
            are available for de-scaling.
        """
        preds = preds.float()
        means, raw_vars, logits = preds.tensor_split(3, -1)
        stds = torch.sqrt(nn.functional.softplus(raw_vars) + self.loss.eps)
        weights = torch.softmax(logits, dim=-1)
        if self.target_denominators is not None and labels is not None:
            den = labels[self.input_name][self.target_denominators[0]]
            if not torch.is_tensor(den):
                den = torch.as_tensor(np.asarray(den), dtype=means.dtype)
            den = den.to(device=means.device, dtype=means.dtype).unsqueeze(-1)
            means = means * den
            stds = stds * den
        elif self.norm_params is not None:
            means = means * self.norm_params["std"][0] + self.norm_params["mean"][0]
            stds = stds * self.norm_params["std"][0]
        else:
            raise ValueError(
                "Inference for mixture density regression requires scaling parameters "
                "(norm_params or target_denominators)."
            )
        return means, stds, weights

    def run_inference(
        self, preds: Tensor, labels: Tensors | None = None, pad_mask: Tensor | None = None
    ) -> Tensor:
        """De-scale the dominant-mode parameters of the mixture.

        Selects the component with the largest mixture weight pi_k ("the mode") and
        returns its de-normalized mean, its de-normalized standard deviation,
        and its mixture weight.

        Parameters
        ----------
        preds : Tensor
            Raw head output of shape [B, 3K].
        labels : Tensors | None, optional
            Labels, used to re-apply 'target_denominators' when configured.
        pad_mask : Tensor | None, optional
            If provided, padded entries are replaced with NaN.

        Returns
        -------
        Tensor
            Tensor of shape [B, 3]: (mode_mean, mode_stddev, mode_weight)
            in the original target space.

        Raises
        ------
        ValueError
            If neither 'target_denominators' (+ labels) nor 'norm_params'
            are available for de-scaling.
        """
        preds = preds.float()
        means, raw_vars, logits = preds.tensor_split(3, -1)
        stds = torch.sqrt(nn.functional.softplus(raw_vars) + self.loss.eps)
        weights = torch.softmax(logits, dim=-1)

        k = weights.argmax(dim=-1, keepdim=True)
        mode_mean = torch.gather(means, -1, k)
        mode_std = torch.gather(stds, -1, k)
        mode_weight = torch.gather(weights, -1, k)

        if self.target_denominators is not None and labels is not None:
            den = labels[self.input_name][self.target_denominators[0]]
            if not torch.is_tensor(den):
                # ONNX export passes numpy arrays (get_structured_input_dict)
                den = torch.as_tensor(np.asarray(den), dtype=mode_mean.dtype)
            den = den.to(device=mode_mean.device, dtype=mode_mean.dtype).unsqueeze(-1)
            mode_mean = mode_mean * den
            mode_std = mode_std * den
        elif self.norm_params is not None:
            mode_mean = mode_mean * self.norm_params["std"][0] + self.norm_params["mean"][0]
            mode_std = mode_std * self.norm_params["std"][0]
        else:
            raise ValueError(
                "Inference for mixture density regression requires scaling parameters "
                "(norm_params or target_denominators)."
            )

        out = torch.cat([mode_mean, mode_std, mode_weight], dim=-1)
        if pad_mask is not None:
            out = torch.masked_fill(out, pad_mask.unsqueeze(-1), np.nan)
        return out

    def get_h5(
        self, preds: Tensor, labels: Tensors | None = None, pad_mask: Tensor | None = None
    ) -> np.ndarray:
        """Convert predictions to a single structured array for the H5 writer.

        A single structured array (not a tuple) is returned on purpose: the
        untouched 'PredictionWriter' writes tuple returns only for
        GaussianRegressionTask instances; for plain RegressionTask
        instances, which this is by inheritance, the return value is stored
        as is.

        Field names follow the 'GaussianRegressionTask.get_h5' convention of
        prefixing with the task 'name' (note: 'RegressionTask.get_h5'
        instead prefixes with 'model_name' via 'output_names').

        Parameters
        ----------
        preds : Tensor
            Raw head output of shape [B, 3K].
        labels : Tensors | None, optional
            Labels, forwarded to :meth:'run_inference'.
        pad_mask : Tensor | None, optional
            If provided, padded entries are NaN-filled.

        Returns
        -------
        np.ndarray
            Structured array with the dominant-mode triplet ({name}_{target},
            _stddev = dominant-mode sigma, _modeweight), the mixture summaries
            (_mubar, _sigmix = sqrt(Var[z]) via the law of total variance,
            _sigfair = sqrt(Var[z] + (mubar - mu_dom)^2)) and the full mixture
            (_mu{k}, _sigma{k}, _pi{k} in raw component order).
        """
        means, stds, weights = self._descaled_params(preds, labels)
        k = weights.argmax(dim=-1, keepdim=True)
        mu_dom = torch.gather(means, -1, k)
        sg_dom = torch.gather(stds, -1, k)
        pi_dom = torch.gather(weights, -1, k)
        mu_bar = (weights * means).sum(-1, keepdim=True)
        var = (weights * stds**2).sum(-1, keepdim=True) + (weights * (means - mu_bar) ** 2).sum(
            -1, keepdim=True
        )
        sig_mix = torch.sqrt(var)
        sig_fair = torch.sqrt(var + (mu_bar - mu_dom) ** 2)
        # Full-mixture serialization: the dominant-mode triplet comes first (backward
        # compatible), then the law-of-total-variance summaries, then every component
        # in RAW component order (identity preserved; consumers may sort by pi).
        out = torch.cat(
            [mu_dom, sg_dom, pi_dom, mu_bar, sig_mix, sig_fair, means, stds, weights], dim=-1
        )
        if pad_mask is not None:
            out = torch.masked_fill(out, pad_mask.unsqueeze(-1), np.nan)
        t = self.targets[0]
        kk = self.n_components
        fields = [
            (f"{self.name}_{t}", "f4"),
            (f"{self.name}_{t}_stddev", "f4"),
            (f"{self.name}_{t}_modeweight", "f4"),
            (f"{self.name}_{t}_mubar", "f4"),
            (f"{self.name}_{t}_sigmix", "f4"),
            (f"{self.name}_{t}_sigfair", "f4"),
        ]
        fields += [(f"{self.name}_{t}_mu{i}", "f4") for i in range(kk)]
        fields += [(f"{self.name}_{t}_sigma{i}", "f4") for i in range(kk)]
        fields += [(f"{self.name}_{t}_pi{i}", "f4") for i in range(kk)]
        return u2s(out.float().cpu().numpy(), np.dtype(fields))

    def get_onnx(self, preds: Tensor, **kwargs) -> tuple:
        """ONNX-friendly outputs: squeezed (mode_mean, sigma_fair, mode_weight).

        Parameters
        ----------
        preds : Tensor
            Raw head output of shape [B, 3K].
        **kwargs
            Optional 'labels' and 'pad_mask' forwarded to inference.

        Returns
        -------
        tuple
            Tuple of per-output tensors matching :attr:'output_names'.
        """
        # The "stddev" output carries sigma_fair = sqrt(Var[z] + (mu_bar - mu_dom)^2):
        # the uncertainty consistent with the reported dominant-mode mean. The conditional
        # dominant-mode sigma understates the risk whenever rival modes exist.
        means, stds, weights = self._descaled_params(preds, kwargs.get("labels"))
        k = weights.argmax(dim=-1, keepdim=True)
        mu_dom = torch.gather(means, -1, k)
        pi_dom = torch.gather(weights, -1, k)
        mu_bar = (weights * means).sum(-1, keepdim=True)
        var = (weights * stds**2).sum(-1, keepdim=True) + (weights * (means - mu_bar) ** 2).sum(
            -1, keepdim=True
        )
        sig_fair = torch.sqrt(var + (mu_bar - mu_dom) ** 2)
        out = torch.cat([mu_dom, sig_fair, pi_dom], dim=-1)
        return tuple(o.squeeze() for o in torch.split(out, 1, -1))
