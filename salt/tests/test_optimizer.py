from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from salt.optim.hybrid_muon_adamw import HybridMuonAdamW, MuonParamPolicy


class TinyModel(nn.Module):
    """A small module with a mix of 2D and non-2D parameters.

    The parameter names are chosen to exercise MuonParamPolicy exclusions:
    - `linear.weight` (2D) -> should be eligible for Muon
    - `linear.bias` (1D, and name matches bias) -> AdamW
    - `norm.weight` (1D, and name matches norm) -> AdamW
    - `head.weight` (2D but name matches head) -> excluded from Muon -> AdamW
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=True)
        self.norm = nn.LayerNorm(3)
        self.head = nn.Linear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.linear(x)
        x = self.norm(x)
        x = self.head(x)
        return x


def _set_all_grads(model: nn.Module, value: float = 0.01) -> None:
    """Populate gradients for all trainable parameters.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters should receive gradients.
    value : float, optional
        Constant gradient value.
    """
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, value)


# -----------------------------
# MuonParamPolicy tests
# -----------------------------


def test_muon_policy_requires_grad_false() -> None:
    policy = MuonParamPolicy()
    p = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
    assert policy.is_muon_param("linear.weight", p) is False


def test_muon_policy_rejects_non_2d() -> None:
    policy = MuonParamPolicy()
    p = nn.Parameter(torch.zeros(3), requires_grad=True)
    assert policy.is_muon_param("linear.bias", p) is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("linear.weight", True),  # 2D and not excluded
        ("linear.bias", False),  # excluded via bias pattern, and also non-2D in practice
        ("norm.weight", False),  # excluded via norm pattern
        ("layer_norm.weight", False),  # excluded via layer_norm pattern
        ("embedding.weight", False),  # excluded via embedding pattern
        ("head.weight", False),  # excluded via head pattern (even if 2D)
        ("classifier.weight", False),  # excluded via classifier pattern
        ("out_proj.weight", False),  # excluded via out_proj pattern
    ],
)
def test_muon_policy_name_exclusions(name: str, expected: bool) -> None:
    policy = MuonParamPolicy()
    p = nn.Parameter(torch.zeros(3, 3), requires_grad=True)  # 2D
    assert policy.is_muon_param(name, p) is expected


# -----------------------------
# HybridMuonAdamW constructor tests
# -----------------------------


def test_hybrid_raises_on_empty_params() -> None:
    with pytest.raises(ValueError, match="empty parameter list"):
        HybridMuonAdamW([], lr=1e-3, weight_decay=1e-5)


def test_hybrid_named_splits_params_by_policy() -> None:
    model = TinyModel()
    opt = HybridMuonAdamW(model.named_parameters(), lr=1e-3, weight_decay=1e-5)

    # We expect:
    # - linear.weight -> Muon
    # Everything else -> AdamW (bias, norm, head.* excluded)
    assert any("linear.weight" == n for n in opt.muon_param_names)
    assert all("linear.weight" != n for n in opt.adamw_param_names)

    # Confirm expected AdamW members
    assert any("linear.bias" == n for n in opt.adamw_param_names)
    assert any("norm.weight" == n for n in opt.adamw_param_names)
    assert any("norm.bias" == n for n in opt.adamw_param_names)
    assert any("head.weight" == n for n in opt.adamw_param_names)


def test_hybrid_unnamed_falls_back_to_ndim_rule() -> None:
    model = TinyModel()
    opt = HybridMuonAdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # In unnamed mode, naming is synthetic: param_{idx}
    # We can still check that some 2D params went to Muon and some non-2D went to AdamW.
    assert len(opt.muon_param_names) > 0
    assert len(opt.adamw_param_names) > 0


def test_hybrid_raises_if_no_muon_params_named_all_excluded() -> None:
    model = TinyModel()
    # Custom policy that excludes everything by matching any name
    policy = MuonParamPolicy(exclude_name_patterns=(r".*",))
    with pytest.raises(ValueError, match="no parameters selected for Muon"):
        HybridMuonAdamW(model.named_parameters(), lr=1e-3, weight_decay=1e-5, policy=policy)


def test_hybrid_raises_if_no_adamw_params_all_muon_2d_only() -> None:
    # Model with exactly one 2D parameter and nothing else -> all params become Muon -> should raise
    class Only2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(3, 3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = Only2D()
    with pytest.raises(ValueError, match="no parameters selected for AdamW"):
        HybridMuonAdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)


def test_hybrid_unnamed_raises_on_invalid_item_type() -> None:
    # If items are not named pairs, we take the unnamed path and require actual Parameters.
    bad_items: list[Any] = [("not-a-param", "still-not-a-param")]
    with pytest.raises(TypeError, match="expected Parameters"):
        HybridMuonAdamW(bad_items, lr=1e-3, weight_decay=1e-5)


# -----------------------------
# Functional tests: step/zero_grad/state_dict
# -----------------------------


def test_hybrid_zero_grad_set_to_none() -> None:
    model = TinyModel()
    opt = HybridMuonAdamW(model.named_parameters(), lr=1e-3, weight_decay=1e-5)

    _set_all_grads(model, value=0.01)
    opt.zero_grad(set_to_none=True)

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is None


def test_hybrid_step_updates_parameters() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    opt = HybridMuonAdamW(model.named_parameters(), lr=1e-2, weight_decay=0.0)

    # Save copies of parameters
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Add gradients and step
    _set_all_grads(model, value=0.01)
    opt.step()

    after = {n: p.detach().clone() for n, p in model.named_parameters()}

    # At least one parameter must change
    changed = any(not torch.equal(before[n], after[n]) for n in before)
    assert changed is True


def test_hybrid_step_with_closure_returns_value() -> None:
    model = TinyModel()
    opt = HybridMuonAdamW(model.named_parameters(), lr=1e-3, weight_decay=0.0)

    _set_all_grads(model, value=0.01)

    def closure() -> float:
        return 123.0

    out = opt.step(closure=closure)
    assert out == 123.0


def test_hybrid_state_dict_roundtrip() -> None:
    torch.manual_seed(0)
    model1 = TinyModel()
    opt1 = HybridMuonAdamW(model1.named_parameters(), lr=1e-3, weight_decay=1e-5)

    # Take a step to populate internal state
    _set_all_grads(model1, value=0.01)
    opt1.step()

    sd = opt1.state_dict()

    # New model + optimizer with same structure
    torch.manual_seed(0)
    model2 = TinyModel()
    opt2 = HybridMuonAdamW(model2.named_parameters(), lr=1e-3, weight_decay=1e-5)

    # Load state dict should not raise
    opt2.load_state_dict(sd)

    # Ensure name lists were restored (informational but useful)
    assert opt2.muon_param_names == opt1.muon_param_names
    assert opt2.adamw_param_names == opt1.adamw_param_names
