"""Unit tests for BetaNLLLoss."""

from __future__ import annotations

import pytest
import torch

from salt.model.nn.losses import BetaNLLLoss


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    means = torch.randn(4, 2)
    targets = torch.randn(4, 2)
    var = torch.rand(4, 2) + 0.1
    return means, targets, var


def test_beta_nll_beta_zero_is_gaussian_nll(inputs):
    """beta=0 is the plain Gaussian NLL, computed by hand."""
    means, targets, var = inputs
    loss = BetaNLLLoss(beta=0.0)(means, targets, var)
    expected = 0.5 * (torch.log(var) + (targets - means) ** 2 / var)
    torch.testing.assert_close(loss, expected)


def test_beta_nll_beta_weighting(inputs):
    """beta=1 weights the Gaussian NLL by the variance."""
    means, targets, var = inputs
    base = BetaNLLLoss(beta=0.0)(means, targets, var)
    weighted = BetaNLLLoss(beta=1.0)(means, targets, var)
    torch.testing.assert_close(weighted, base * var)


def test_beta_nll_reductions(inputs):
    """'none' keeps the element shape; 'mean'/'sum' reduce it."""
    means, targets, var = inputs
    none = BetaNLLLoss(reduction="none")(means, targets, var)
    assert none.shape == means.shape
    mean = BetaNLLLoss(reduction="mean")(means, targets, var)
    torch.testing.assert_close(mean, none.mean())
    total = BetaNLLLoss(reduction="sum")(means, targets, var)
    torch.testing.assert_close(total, none.sum())


def test_beta_nll_clamps_zero_variance(inputs):
    """A zero variance is clamped, so the loss stays finite."""
    means, targets, _ = inputs
    loss = BetaNLLLoss()(means, targets, torch.zeros_like(means))
    assert torch.isfinite(loss).all()


def test_beta_nll_raises_on_nan_target(inputs):
    """A NaN target makes the loss non-finite, which raises."""
    means, _, var = inputs
    targets = torch.full_like(means, torch.nan)
    with pytest.raises(ValueError, match="non-finite"):
        BetaNLLLoss()(means, targets, var)


def test_beta_nll_invalid_args():
    """beta outside [0, 1] and an unknown reduction are rejected."""
    with pytest.raises(AssertionError):
        BetaNLLLoss(beta=2.0)
    with pytest.raises(AssertionError):
        BetaNLLLoss(reduction="max")
