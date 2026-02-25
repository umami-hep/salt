from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class MuonParamPolicy:
    """Policy for deciding which parameters should be optimized by Muon.

    This policy determines whether a parameter is assigned to the Muon optimizer
    based on both its tensor dimensionality (must be 2D) and its name. Certain
    parameter name patterns (e.g. biases, normalization parameters, embeddings,
    classifier heads) are excluded by default.

    Parameters
    ----------
    exclude_name_patterns : tuple[str, ...]
        Case-insensitive regex patterns. If a parameter name matches any of
        these patterns, it will be excluded from Muon optimization even if it is 2D.

    Attributes
    ----------
    exclude_name_patterns : tuple[str, ...]
        Regex patterns used to filter out parameters that should *not* be
        optimized by Muon. Typically includes biases, LayerNorm/BatchNorm
        parameters, embeddings, and output heads.
    """

    exclude_name_patterns: tuple[str, ...] = (
        r"\bbias\b",
        r"layernorm|layer_norm|\bln\d*\b|norm",
        r"embedding|embeddings|\bembed\b",
        r"\bhead\b|classifier|output|out_proj|final",
    )

    def is_muon_param(self, name: str, param: nn.Parameter) -> bool:
        """Return whether a parameter should be optimized by Muon.

        Parameters
        ----------
        name : str
            Parameter name (from ``named_parameters()``).
        param : nn.Parameter
            Parameter tensor.

        Returns
        -------
        bool
            True if the parameter should go to Muon, False otherwise.
        """
        if not param.requires_grad:
            return False
        if param.ndim != 2:
            return False

        lname = name.lower()
        return all(not re.search(pat, lname) for pat in self.exclude_name_patterns)


class HybridMuonAdamW(Optimizer):
    """Hybrid optimizer: Muon for selected 2D matrices, AdamW for the rest.

    This wrapper presents a single-optimizer interface to Lightning/SALT, but
    internally maintains and steps two optimizers:

    - ``torch.optim.Muon`` for (selected) 2D weight matrices
    - ``torch.optim.AdamW`` for all remaining parameters

    Use ``model.named_parameters()`` as input to enable name-based exclusions
    (biases/norms/embeddings/heads).

    Notes
    -----
    - If initialized with ``model.parameters()`` (no names), the selection falls
      back to a simple rule: **Muon for params with ``ndim == 2``**, AdamW for
      everything else.

    Parameters
    ----------
    params : Iterable[nn.Parameter | tuple[str, nn.Parameter]]
        Either an iterable of Parameters (as returned by ``model.parameters()``)
        or an iterable of ``(name, Parameter)`` pairs (as returned by
        ``model.named_parameters()``).
    lr : float
        Base learning rate. Used for both Muon and AdamW unless overridden by
        ``lr_muon`` / ``lr_adamw``.
    weight_decay : float, optional
        Base decoupled weight decay. Used for both Muon and AdamW unless
        overridden by ``muon_weight_decay`` / ``adamw_weight_decay``.
    lr_muon : float | None, optional
        Muon learning rate override. If None, uses ``lr``.
    lr_adamw : float | None, optional
        AdamW learning rate override. If None, uses ``lr``.
    muon_weight_decay : float | None, optional
        Muon weight decay override. If None, uses ``weight_decay``.
    adamw_weight_decay : float | None, optional
        AdamW weight decay override. If None, uses ``weight_decay``.
    momentum : float, optional
        Muon momentum.
    nesterov : bool, optional
        Whether Muon uses Nesterov momentum.
    ns_coefficients : tuple[float, float, float], optional
        Newton-Schulz iteration coefficients for Muon.
    eps : float, optional
        Numerical stability epsilon used in Muon.
    ns_steps : int, optional
        Number of Newton-Schulz steps for Muon.
    adjust_lr_fn : str | None, optional
        Muon LR adjustment mode. Common values: ``"original"`` or
        ``"match_rms_adamw"`` (per PyTorch docs). If None, disables adjustment.
    betas : tuple[float, float], optional
        AdamW betas.
    adamw_eps : float, optional
        AdamW epsilon.
    policy : MuonParamPolicy | None, optional
        Name-based policy for selecting Muon parameters. If None, uses the
        default policy.

    Raises
    ------
    ValueError
        If no parameters are selected for Muon.
        If no parameters are selected for AdamW
    TypeError
        If parameters to HybridMuonAdamW are not Parameters or (name, Parameter) pairs.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter | tuple[str, nn.Parameter]],
        *,
        lr: float,
        weight_decay: float = 1e-5,
        lr_muon: float | None = None,
        lr_adamw: float | None = None,
        muon_weight_decay: float | None = None,
        adamw_weight_decay: float | None = None,
        # Muon hyperparams
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = "original",
        # AdamW hyperparams
        betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        # Selection policy
        policy: MuonParamPolicy | None = None,
    ) -> None:
        self.policy = policy or MuonParamPolicy()

        items = list(params)
        if not items:
            raise ValueError("HybridMuonAdamW received an empty parameter list.")

        named = _looks_like_named_params(items)

        muon_params: list[nn.Parameter] = []
        adamw_params: list[nn.Parameter] = []
        self._muon_names: list[str] = []
        self._adamw_names: list[str] = []

        if named:
            for name, p in items:  # type: ignore[misc]
                if not p.requires_grad:
                    continue
                if self.policy.is_muon_param(name, p):
                    muon_params.append(p)
                    self._muon_names.append(name)
                else:
                    adamw_params.append(p)
                    self._adamw_names.append(name)
        else:
            # No names available: fall back to ndim-based selection
            for idx, p in enumerate(items):  # type: ignore[assignment]
                if not isinstance(p, nn.Parameter):
                    raise TypeError(
                        "HybridMuonAdamW expected Parameters or (name, Parameter) pairs."
                    )
                if not p.requires_grad:
                    continue
                name = f"param_{idx}"
                if p.ndim == 2:
                    muon_params.append(p)
                    self._muon_names.append(name)
                else:
                    adamw_params.append(p)
                    self._adamw_names.append(name)

        if len(muon_params) == 0:
            raise ValueError(
                "HybridMuonAdamW: no parameters selected for Muon. "
                "If you passed model.parameters(), this can happen if there are no 2D parameters. "
                "If you passed model.named_parameters(), check your exclusion policy."
            )
        if len(adamw_params) == 0:
            raise ValueError(
                "HybridMuonAdamW: no parameters selected for AdamW. "
                "This is unusual; check your model and selection policy."
            )

        # Interface-compatible defaults
        lr_muon = lr if lr_muon is None else lr_muon
        lr_adamw = lr if lr_adamw is None else lr_adamw
        muon_weight_decay = weight_decay if muon_weight_decay is None else muon_weight_decay
        adamw_weight_decay = weight_decay if adamw_weight_decay is None else adamw_weight_decay

        # Store LR ratios so schedulers can drive a single base LR on the wrapper.
        self._lr_ratio_muon = lr_muon / lr
        self._lr_ratio_adamw = lr_adamw / lr

        # Initialize as a normal Optimizer over all params so Lightning sees one optimizer
        super().__init__(
            [
                {
                    "params": muon_params + adamw_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            ],
            defaults={"lr": lr, "weight_decay": weight_decay},
        )

        self.muon = torch.optim.Muon(
            muon_params,
            lr=lr_muon,
            weight_decay=muon_weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )

        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=lr_adamw,
            weight_decay=adamw_weight_decay,
            betas=betas,
            eps=adamw_eps,
        )

        # Ensure internal optimizers start consistent with the wrappers base LR.
        self._sync_lrs_from_wrapper()

    def _sync_lrs_from_wrapper(self) -> None:
        """Synchronize internal optimizer learning rates from wrapper param groups.

        Torch schedulers update the learning rate stored in ``self.param_groups`` of this
        wrapper optimizer. This method propagates that base learning rate to the internal
        Muon and AdamW optimizers.

        Notes
        -----
        The wrapper treats ``self.param_groups[0]["lr"]`` as a *base LR* and applies
        constant ratios computed at initialization time:

        - ``lr_muon  = base_lr * self._lr_ratio_muon``
        - ``lr_adamw = base_lr * self._lr_ratio_adamw``

        This allows schedulers like OneCycleLR to work unchanged while preserving a fixed
        ratio between the Muon and AdamW learning rates when overrides are provided.
        """
        base_lr = float(self.param_groups[0]["lr"])
        lr_muon = base_lr * self._lr_ratio_muon
        lr_adamw = base_lr * self._lr_ratio_adamw

        for group in self.muon.param_groups:
            group["lr"] = lr_muon
        for group in self.adamw.param_groups:
            group["lr"] = lr_adamw

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        """Perform a single optimization step.

        Parameters
        ----------
        closure : Callable[[], Any] | None, optional
            A callable that re-evaluates the model and returns a value (typically
            the loss). If provided, it is executed once under ``enable_grad()``
            and its return value is propagated.

        Returns
        -------
        Any
            The value returned by ``closure`` if provided; otherwise ``None``.
        """
        loss: Any = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Propagate scheduler-driven LR changes into internal optimizers.
        self._sync_lrs_from_wrapper()

        self.muon.step()
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients of all optimized parameters.

        Parameters
        ----------
        set_to_none : bool, optional
            Whether to set gradients to None instead of zeroing in place.
        """
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the optimizer.

        Returns
        -------
        dict
            A dictionary containing state for both internal optimizers and
            informational parameter name lists.
        """
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "muon_names": list(self._muon_names),
            "adamw_names": list(self._adamw_names),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the optimizer state.

        Parameters
        ----------
        state_dict : dict[str, Any]
            State dictionary as produced by :meth:`state_dict`.
        """
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self._muon_names = list(state_dict.get("muon_names", self._muon_names))
        self._adamw_names = list(state_dict.get("adamw_names", self._adamw_names))

    @property
    def muon_param_names(self) -> list[str]:
        """Return names (or synthetic names) of Muon-optimized parameters.

        Returns
        -------
        list[str]
            Parameter names (from ``named_parameters()``) or synthetic names if
            the optimizer was created from ``parameters()``.
        """
        return list(self._muon_names)

    @property
    def adamw_param_names(self) -> list[str]:
        """Return names (or synthetic names) of AdamW-optimized parameters.

        Returns
        -------
        list[str]
            Parameter names (from ``named_parameters()``) or synthetic names if
            the optimizer was created from ``parameters()``.
        """
        return list(self._adamw_names)


def _looks_like_named_params(items: Sequence[Any]) -> bool:
    """Heuristically determine whether an iterable looks like named parameters.

    Parameters
    ----------
    items : Sequence[Any]
        Materialized sequence passed into the optimizer.

    Returns
    -------
    bool
        True if items look like ``(name, Parameter)`` pairs, False otherwise.
    """
    first = items[0]
    if not isinstance(first, tuple) or len(first) != 2:
        return False
    name, param = first
    return isinstance(name, str) and isinstance(param, nn.Parameter)
