"""Standalone regression losses, selected through a task's ``loss:`` config."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BetaNLLLoss(nn.Module):
    """Beta-NLL loss from Seitzer et al. (2022), https://arxiv.org/abs/2203.09168.

    Weights the Gaussian NLL by ``sigma^(2*beta)`` (detached), reducing the dominance of
    low-uncertainty predictions in the gradient.

    Parameters
    ----------
    beta : float, optional
        Exponent of the variance weight, in ``[0, 1]``, by default 0.0. ``beta=0`` is the
        Gaussian NLL, ``beta=1`` removes its ``1/sigma^2`` down-weighting of the squared error.
    reduction : str, optional
        One of ``'none'``, ``'mean'``, ``'sum'``, by default ``'none'``.
    """

    def __init__(self, beta: float = 0.0, reduction: str = "none") -> None:
        super().__init__()
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        assert reduction in {"none", "mean", "sum"}
        self.beta = beta
        self.clamp = 1e-6
        self.reduction = reduction

    def forward(self, means: Tensor, targets: Tensor, var: Tensor) -> Tensor:
        """Compute the loss of predicted means and variances against the targets.

        Parameters
        ----------
        means : Tensor
            Predicted means, shape ``[B, R]``.
        targets : Tensor
            Target values, shape ``[B, R]``.
        var : Tensor
            Predicted variances, shape ``[B, R]``, clamped at ``1e-6``.

        Returns
        -------
        Tensor
            Loss of shape ``[B, R]`` for ``reduction='none'``, else a scalar.

        Raises
        ------
        ValueError
            If the loss has a non-finite value before reduction. Eager only: the check is
            data-dependent control flow, so it is skipped inside a compiled region.
        """
        # clamp outside no_grad, so the clamp also acts on the gradient
        var = var.clamp(min=self.clamp)
        loss = 0.5 * (torch.log(var) + (targets - means) ** 2 / var)
        if self.beta > 0:
            loss = loss * (var.detach() ** self.beta)
        if not torch.compiler.is_compiling() and not torch.isfinite(loss).all():
            num_bad = (~torch.isfinite(loss)).sum().item()
            raise ValueError(
                f"Regression loss has {num_bad}/{loss.numel()} non-finite values before reduction."
            )
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
