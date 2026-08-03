from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class MuonParamPolicy:
    """Policy for deciding which parameters should be optimized by Muon.

    Determines whether a parameter is assigned to the Muon optimizer based on its
    tensor dimensionality (must be 2D), explicit module-name include/exclude
    lists, and — as a fallback — broad name regexes.

    Selection order:

    1. Non-2D or non-trainable parameters always go to AdamW (Muon needs 2D
       weight matrices).
    2. ``exclude`` — an explicit module-name substring match forces the
       parameter to AdamW (highest priority).
    3. ``include`` — an explicit module-name substring match forces the
       parameter to Muon, overriding the broad ``exclude_name_patterns`` regexes.
    4. ``exclude_name_patterns`` — broad regex defaults: a 2D parameter whose
       lowercased name matches any pattern goes to AdamW.

    A policy whose ``include`` / ``exclude`` list matches ZERO parameters is
    surfaced as a warning by `HybridMuonAdamW` (a likely typo or renamed module).

    Parameters
    ----------
    exclude_name_patterns : tuple[str, ...]
        Case-insensitive regex patterns. If a 2D parameter's name matches any
        of these (and no ``include`` entry overrides it), it is excluded from
        Muon.
    include : tuple[str, ...]
        Explicit module-name substrings that force a 2D parameter onto Muon,
        overriding ``exclude_name_patterns`` (but not ``exclude``). Empty by
        default.
    exclude : tuple[str, ...]
        Explicit module-name substrings that force a parameter onto AdamW,
        taking precedence over everything else. Empty by default.
    """

    exclude_name_patterns: tuple[str, ...] = (
        r"\bbias\b",
        r"layernorm|layer_norm|\bln\d*\b|norm",
        r"embedding|embeddings|\bembed\b",
        r"\bhead\b|classifier|output|out_proj|final",
    )
    include: tuple[str, ...] = field(default_factory=tuple)
    exclude: tuple[str, ...] = field(default_factory=tuple)

    def is_muon_param(self, name: str, param: nn.Parameter) -> bool:
        """Return whether a parameter should be optimized by Muon.

        Applies the selection order documented on the class: 2D/trainable gate,
        explicit ``exclude``, explicit ``include``, then the broad
        ``exclude_name_patterns`` defaults.
        """
        if not param.requires_grad:
            return False
        if param.ndim != 2:
            return False

        # explicit exclude wins over everything (AdamW)
        if any(token in name for token in self.exclude):
            return False
        # explicit include overrides the broad regexes (Muon)
        if any(token in name for token in self.include):
            return True

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

    Notes
    -----
    - If initialized with ``model.parameters()`` (no names), the selection falls
      back to a simple rule: **Muon for params with ``ndim == 2``**, AdamW for
      everything else.
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
            all_names = [name for name, _ in items]  # type: ignore[misc]
            self._warn_dead_routing(self.policy, all_names)
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

    @staticmethod
    def _warn_dead_routing(policy: MuonParamPolicy, names: Sequence[str]) -> None:
        """Warn when an explicit ``include``/``exclude`` token matches no parameter
        name (likely a typo/renamed module). Staticmethod so it's testable
        without constructing the optimizer (whose ``torch.optim.Muon`` needs
        torch>=2.9).
        """
        for label, tokens in (("include", policy.include), ("exclude", policy.exclude)):
            for token in tokens:
                if not any(token in name for name in names):
                    warnings.warn(
                        f"HybridMuonAdamW policy.{label} entry {token!r} matched 0 parameter "
                        f"names — a dead routing entry (typo or renamed module?). Known prefixes: "
                        f"{sorted({name.split('.')[0] for name in names})} .",
                        stacklevel=3,
                    )

    def _sync_lrs_from_wrapper(self) -> None:
        """Propagate the wrapper's scheduler-driven ``param_groups[0]["lr"]`` to
        Muon/AdamW as ``base_lr * self._lr_ratio_{muon,adamw}``, preserving the
        fixed ratio so schedulers like OneCycleLR work unchanged.
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
        """One optimization step across both internal optimizers; `closure`, if
        given, runs once under ``enable_grad()``.
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
        """Clear gradients of all optimized parameters."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the optimizer (both internal optimizers plus wrapper metadata)."""
        return {
            "wrapper": super().state_dict(),
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "muon_names": list(self._muon_names),
            "adamw_names": list(self._adamw_names),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the optimizer state, as produced by :meth:`state_dict`."""
        if "wrapper" in state_dict:
            super().load_state_dict(state_dict["wrapper"])

        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self._muon_names = list(state_dict.get("muon_names", self._muon_names))
        self._adamw_names = list(state_dict.get("adamw_names", self._adamw_names))
        self._sync_lrs_from_wrapper()

    @property
    def muon_param_names(self) -> list[str]:
        """Return names (or synthetic names) of Muon-optimized parameters."""
        return list(self._muon_names)

    @property
    def adamw_param_names(self) -> list[str]:
        """Return names (or synthetic names) of AdamW-optimized parameters."""
        return list(self._adamw_names)


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum), batched over parameters with ``torch._foreach_*``.

    Same update as the reference ``lion-pytorch`` implementation, op for op —
    this class only changes how many kernels it takes to get there. The
    reference walks ``group["params"]`` in Python and issues five ops per
    parameter; on GN3V00 that is 595 launches per step for 1.33 ms of actual
    kernel work (``docs/profiling.md``). The ``_foreach_`` form issues seven
    launches per parameter *group* instead, whatever the parameter count.

    The update, for each parameter ``p`` with gradient ``g`` and state
    ``exp_avg``::

        p        <- p * (1 - lr * wd)
        update   <- sign(exp_avg * beta1 + g * (1 - beta1))
        p        <- p - lr * update
        exp_avg  <- exp_avg * beta2 + g * (1 - beta2)

    Note the ordering that makes Lion Lion: the *momentum* used for the update
    is the pre-update ``exp_avg`` interpolated towards ``g`` with ``beta1``,
    while the state carried forward uses ``beta2``. The ops are spelled as
    ``mul`` + ``add(alpha=...)`` and NOT as ``lerp`` — algebraically identical,
    but a different rounding, and this class exists to be a drop-in for weights
    already trained with the reference. ``salt/tests/unit/test_optim.py``
    gates that bit-for-bit against ``lion_pytorch.Lion``.

    Parameters
    ----------
    params : Iterable
        Parameters or parameter groups, as any `torch.optim.Optimizer`.
    lr : float, optional
        Learning rate, by default 1e-4. Must be positive.
    betas : tuple[float, float], optional
        ``(beta1, beta2)`` — interpolation for the update direction and for the
        momentum state respectively, by default ``(0.9, 0.99)``. Both in [0, 1].
    weight_decay : float, optional
        Decoupled step-weight decay applied as ``p *= 1 - lr * wd``, by default 0.
    decoupled_weight_decay : bool, optional
        When True, ``wd`` is divided by the initial learning rate so the decay
        per step is independent of the schedule, by default False (matches the
        reference default).

    Raises
    ------
    ValueError
        Non-positive `lr`, a beta outside [0, 1], or a torch build without the
        ``_foreach_`` ops this implementation is built on.
    """

    _FOREACH_OPS = ("_foreach_mul", "_foreach_mul_", "_foreach_add_", "_foreach_sign_")

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Lion needs lr > 0, got {lr}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"Lion needs both betas in [0, 1], got {betas}")
        missing = [name for name in self._FOREACH_OPS if not hasattr(torch, name)]
        if missing:
            raise ValueError(
                f"this torch build ({torch.__version__}) lacks {', '.join(missing)} — "
                "use optimizer: lion-pytorch for the per-parameter reference implementation"
            )
        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        """One Lion step over every parameter group.

        Parameters
        ----------
        closure : Callable[[], Any] | None, optional
            Re-evaluates the model and returns the loss, by default None.

        Returns
        -------
        Any
            The closure's return value, or None when no closure was given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            # the reference divides by the lr the optimizer was BUILT with, not
            # the scheduled one — so the decay per step stops tracking the cycle
            if self.decoupled_wd:
                weight_decay = weight_decay / self._init_lr

            params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)
                params.append(param)
                grads.append(param.grad)
                exp_avgs.append(state["exp_avg"])
            if not params:
                continue

            # p *= 1 - lr * wd   (unconditional, as the reference: at wd == 0 the
            # factor is exactly 1.0 and the multiply is a no-op on every finite value)
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)

            # the update direction: the sign of the beta1-interpolated momentum,
            # applied to the parameters scaled by the learning rate
            updates = torch._foreach_mul(exp_avgs, beta1)  # noqa: SLF001 - torch's foreach API
            torch._foreach_add_(updates, grads, alpha=1.0 - beta1)
            torch._foreach_sign_(updates)
            torch._foreach_add_(params, updates, alpha=-lr)

            # the momentum carried forward, interpolated with beta2 instead
            torch._foreach_mul_(exp_avgs, beta2)
            torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta2)

        return loss


def _looks_like_named_params(items: Sequence[Any]) -> bool:
    """Heuristically determine whether an iterable looks like named parameters."""
    first = items[0]
    if not isinstance(first, tuple) or len(first) != 2:
        return False
    name, param = first
    return isinstance(name, str) and isinstance(param, nn.Parameter)
