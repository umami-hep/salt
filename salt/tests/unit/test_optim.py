"""Unit tests for `salt.optim` — HybridMuonAdamW + MuonParamPolicy."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from salt.optim import HybridMuonAdamW, Lion, MuonParamPolicy


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

    scheduler_lr = 2e-4
    opt1.param_groups[0]["lr"] = scheduler_lr
    sd = opt1.state_dict()
    assert "wrapper" in sd

    # New model + optimizer with same structure
    torch.manual_seed(0)
    model2 = TinyModel()
    opt2 = HybridMuonAdamW(model2.named_parameters(), lr=1e-3, weight_decay=1e-5)

    # Load state dict should not raise
    opt2.load_state_dict(sd)

    # Ensure name lists were restored (informational but useful)
    assert opt2.muon_param_names == opt1.muon_param_names
    assert opt2.adamw_param_names == opt1.adamw_param_names

    # Ensure wrapper state was restored and propagated to the internal optimizers.
    assert opt2.param_groups[0]["lr"] == pytest.approx(scheduler_lr)
    assert opt2.muon.param_groups[0]["lr"] == pytest.approx(scheduler_lr)
    assert opt2.adamw.param_groups[0]["lr"] == pytest.approx(scheduler_lr)


# --------------------------------------------------------------------------- #
# salt.optim.Lion — bitwise equivalence with the lion-pytorch reference
# --------------------------------------------------------------------------- #

reference_lion = pytest.importorskip(
    "lion_pytorch", reason="lion-pytorch is the oracle for the foreach Lion parity gate"
).Lion


class LionNet(nn.Module):
    """Parameters of assorted shapes/sizes — the foreach path groups them."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(7, 5)
        self.b = nn.Linear(5, 3, bias=False)
        self.norm = nn.LayerNorm(3)
        self.scalar = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tiny forward used only to produce gradients.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[N, 7]``.

        Returns
        -------
        torch.Tensor
            Output of shape ``[N, 3]``.
        """
        return self.norm(self.b(torch.relu(self.a(x)))) * self.scalar


def _paired_nets() -> tuple[LionNet, LionNet]:
    """Two nets with byte-identical initial parameters."""
    torch.manual_seed(1234)
    reference = LionNet()
    candidate = LionNet()
    with torch.no_grad():
        for tgt, src in zip(candidate.parameters(), reference.parameters(), strict=True):
            tgt.copy_(src)
    return reference, candidate


def _assert_bitwise(reference: LionNet, candidate: LionNet, ref_opt: Any, cand_opt: Any) -> None:
    """Every parameter AND every ``exp_avg`` state tensor must match exactly."""
    for (name, ref_p), (_, cand_p) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        assert torch.equal(ref_p, cand_p), f"parameter {name} diverged"
        ref_state = ref_opt.state[ref_p]
        cand_state = cand_opt.state[cand_p]
        assert set(ref_state) == set(cand_state) == {"exp_avg"}
        assert torch.equal(ref_state["exp_avg"], cand_state["exp_avg"]), (
            f"exp_avg for {name} diverged"
        )


def _run_paired(
    steps: int = 12,
    weight_decay: float = 1e-5,
    decoupled: bool = False,
    schedule: bool = False,
) -> None:
    """Drive both optimizers with byte-identical gradients and compare each step."""
    reference, candidate = _paired_nets()
    kwargs: dict[str, Any] = {
        "lr": 1e-3,
        "betas": (0.9, 0.99),
        "weight_decay": weight_decay,
        "decoupled_weight_decay": decoupled,
    }
    ref_opt = reference_lion(reference.parameters(), **kwargs)
    cand_opt = Lion(candidate.parameters(), **kwargs)

    generator = torch.Generator().manual_seed(99)
    for step in range(steps):
        if schedule:
            # OneCycleLR mutates param_groups["lr"] every step — the decoupled
            # branch divides by the INITIAL lr, so a moving lr must not desync
            lr = 1e-3 * (1.0 + step)
            ref_opt.param_groups[0]["lr"] = lr
            cand_opt.param_groups[0]["lr"] = lr
        for ref_p, cand_p in zip(reference.parameters(), candidate.parameters(), strict=True):
            grad = torch.randn(ref_p.shape, generator=generator)
            ref_p.grad = grad.clone()
            cand_p.grad = grad.clone()
        ref_opt.step()
        cand_opt.step()
        _assert_bitwise(reference, candidate, ref_opt, cand_opt)


def test_lion_bitwise_parity_with_weight_decay() -> None:
    """The shipped GN3 setting: nonzero weight decay, 12 seeded steps."""
    _run_paired(weight_decay=1e-5)


def test_lion_bitwise_parity_zero_weight_decay() -> None:
    """Zero weight decay must take the same path (the factor is exactly 1.0)."""
    _run_paired(weight_decay=0.0)


def test_lion_bitwise_parity_decoupled_weight_decay() -> None:
    """``decoupled_weight_decay`` divides wd by the initial lr on both sides."""
    _run_paired(weight_decay=1e-2, decoupled=True)


def test_lion_bitwise_parity_under_moving_lr() -> None:
    """A OneCycleLR-style moving lr must not desync the two implementations."""
    _run_paired(weight_decay=1e-5, schedule=True)
    _run_paired(weight_decay=1e-2, decoupled=True, schedule=True)


def test_lion_bitwise_parity_through_autograd() -> None:
    """End-to-end: identical forward/backward, not hand-set gradients."""
    reference, candidate = _paired_nets()
    ref_opt = reference_lion(reference.parameters(), lr=1e-3, weight_decay=1e-5)
    cand_opt = Lion(candidate.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(7)
    for _ in range(8):
        batch = torch.randn(6, 7, generator=generator)
        for net, opt in ((reference, ref_opt), (candidate, cand_opt)):
            opt.zero_grad()
            net(batch).square().mean().backward()
            opt.step()
        _assert_bitwise(reference, candidate, ref_opt, cand_opt)


def test_lion_skips_params_without_grad() -> None:
    """A parameter with no gradient is left untouched and gets no state."""
    net = LionNet()
    opt = Lion(net.parameters(), lr=1e-3, weight_decay=1e-5)
    frozen = net.scalar.detach().clone()
    for param in net.parameters():
        if param is not net.scalar:
            param.grad = torch.ones_like(param)
    opt.step()
    assert torch.equal(net.scalar, frozen)
    assert net.scalar not in opt.state or not opt.state[net.scalar]


def test_lion_multiple_param_groups_use_their_own_hyperparameters() -> None:
    """Per-group lr/betas/weight_decay, matched against the reference."""
    reference, candidate = _paired_nets()
    groups = lambda net: [  # noqa: E731 - one-liner used twice
        {"params": list(net.a.parameters()), "lr": 1e-3, "weight_decay": 1e-4},
        {"params": [*net.b.parameters(), *net.norm.parameters(), net.scalar], "lr": 5e-4},
    ]
    ref_opt = reference_lion(groups(reference), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-5)
    cand_opt = Lion(groups(candidate), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-5)
    generator = torch.Generator().manual_seed(3)
    for _ in range(6):
        for ref_p, cand_p in zip(reference.parameters(), candidate.parameters(), strict=True):
            grad = torch.randn(ref_p.shape, generator=generator)
            ref_p.grad = grad.clone()
            cand_p.grad = grad.clone()
        ref_opt.step()
        cand_opt.step()
        _assert_bitwise(reference, candidate, ref_opt, cand_opt)


def test_lion_state_dict_round_trip_matches_the_reference_layout() -> None:
    """``exp_avg`` is the only state key, so checkpoints interchange."""
    net = LionNet()
    opt = Lion(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for param in net.parameters():
        param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state_dict()
    assert all(set(entry) == {"exp_avg"} for entry in state["state"].values())

    restored = Lion(net.parameters(), lr=1e-3, weight_decay=1e-5)
    restored.load_state_dict(state)
    for param in net.parameters():
        assert torch.equal(restored.state[param]["exp_avg"], opt.state[param]["exp_avg"])


def test_lion_closure_returns_loss() -> None:
    """The closure contract is the stock optimizer one."""
    net = LionNet()
    opt = Lion(net.parameters(), lr=1e-3)
    for param in net.parameters():
        param.grad = torch.ones_like(param)
    assert opt.step(lambda: torch.tensor(1.25)).item() == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr": 0.0}, "lr > 0"),
        ({"lr": -1.0}, "lr > 0"),
        ({"betas": (0.9, 1.5)}, "betas in"),
        ({"betas": (-0.1, 0.99)}, "betas in"),
    ],
)
def test_lion_rejects_invalid_hyperparameters(kwargs: dict[str, Any], match: str) -> None:
    """Bad hyperparameters fail at construction, not at the first step."""
    net = LionNet()
    with pytest.raises(ValueError, match=match):
        Lion(net.parameters(), **kwargs)
