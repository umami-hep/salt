"""Unit tests for `salt.optim` — HybridMuonAdamW + MuonParamPolicy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from salt.model.saltmodule import SaltModule, safe_pct_start
from salt.optim import HybridMuonAdamW, Lion, MuonParamPolicy
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, write_parity_norm_dict


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


# --------------------------------------------------------------------------- #
# safe_pct_start — OneCycleLR must survive a short run
# --------------------------------------------------------------------------- #


class TestSafePctStart:
    """The clamp that keeps both OneCycleLR phases non-empty."""

    @staticmethod
    def _build(pct_start: float, total_steps: int) -> Any:
        """Build the scheduler exactly as `SaltModule.configure_optimizers` does."""
        net = LionNet()
        opt = Lion(net.parameters(), lr=1e-4)
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=1e-3,
            total_steps=total_steps,
            div_factor=10.0,
            final_div_factor=10.0,
            pct_start=safe_pct_start(pct_start, total_steps),
        )

    def test_the_shipped_gn3_setting_at_100_steps_no_longer_divides_by_zero(self) -> None:
        """The exact case `salt profile model --steps 100` hits on GN3V00."""
        scheduler = self._build(0.01, 100)
        for _ in range(100):
            scheduler.step()

    @pytest.mark.parametrize("total_steps", [1, 2, 3, 4, 10, 100, 101, 1000])
    @pytest.mark.parametrize("pct_start", [0.001, 0.01, 0.1, 0.3, 0.99])
    def test_no_schedule_length_raises(self, total_steps: int, pct_start: float) -> None:
        """Every (pct_start, total_steps) pair builds and steps to completion."""
        scheduler = self._build(pct_start, total_steps)
        for _ in range(total_steps):
            scheduler.step()

    def test_a_real_training_run_is_untouched(self) -> None:
        """A run long enough for the configured warm-up keeps its own pct_start."""
        # GN3V00 ships pct_start 0.01; 1.5M jets at batch 1000 is 1500 steps
        assert safe_pct_start(0.01, 1500) == pytest.approx(0.01)
        assert safe_pct_start(0.3, 10_000) == pytest.approx(0.3)

    def test_only_short_runs_are_clamped(self) -> None:
        """The clamp bites exactly when the warm-up would span fewer than two steps."""
        assert safe_pct_start(0.01, 100) == pytest.approx(0.02)
        assert safe_pct_start(0.01, 70) == pytest.approx(2 / 70)
        assert safe_pct_start(0.01, 201) == pytest.approx(0.01)

    def test_an_over_long_warmup_is_clamped_too(self) -> None:
        """pct_start == 1 would collapse the ANNEAL phase instead."""
        assert safe_pct_start(1.0, 100) == pytest.approx(0.99)


# --------------------------------------------------------------------------- #
# The post-W8 per-stage optimizer/scheduler rebuild
#
# `configure_optimizers` was rewritten by the fine-tuning work to rebuild BOTH
# the optimizer and the scheduler at every stage boundary, off the ACTIVE
# stage's merged `lrs`/`optimizer` and its own step allocation. Both of the
# guarantees below were established against the pre-rebuild single-shot form, so
# they need gating against the rebuild path they now live on.
# --------------------------------------------------------------------------- #


@pytest.fixture
def _norm_dict(tmp_path: Any) -> Any:
    """A parity norm-dict for the CPU-safe gn2v2 module fixture (no DataLoader)."""
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


class TestSafePctStartGuardsThePerStageAllocation:
    """`safe_pct_start` must clamp against the STAGE's step allocation.

    A sub-stage of a multi-stage schedule gets a fraction of the run's steps, so
    it reaches OneCycleLR's degenerate-warm-up boundary on runs far longer than a
    single-stage fit ever would. Clamping against the whole-run estimate — the
    figure the pre-rebuild code had — leaves that stage unguarded.
    """

    # 100 epochs, 10k whole-run steps, a 1-epoch warm-up stage -> 100 stage steps,
    # which is exactly where the shipped GN3 pct_start of 0.01 divides by zero.
    WHOLE_RUN = 10_000
    MAX_EPOCHS = 100
    SHIPPED_PCT_START = 0.01

    @staticmethod
    def _model(norm_dict: Any) -> SaltModule:
        return SaltModule(
            build_gn2v2_modules(norm_dict),
            lrs={"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.01},
            training_schedule={
                "stages": {"warmup": {"epochs": 1, "frozen": ["encoder"]}, "full": {}}
            },
        )

    def _stage_steps(self, norm_dict: Any, stage_index: int) -> int:
        """The `total_steps` `configure_optimizers` hands OneCycleLR for a stage."""
        model = self._model(norm_dict)
        model._trainer = SimpleNamespace(  # noqa: SLF001 - stub: the two fields read below
            max_epochs=self.MAX_EPOCHS, estimated_stepping_batches=self.WHOLE_RUN
        )
        model._current_stage_index = stage_index  # noqa: SLF001
        return int(model._stage_total_steps())  # noqa: SLF001

    def test_the_warmup_stage_gets_a_small_slice_of_the_run(self, _norm_dict: Any) -> None:
        """The stage allocation, not the whole-run estimate, is what reaches OneCycle."""
        assert self._stage_steps(_norm_dict, 0) == 100
        assert self._stage_steps(_norm_dict, 1) == self.WHOLE_RUN - 100

    def test_the_whole_run_figure_would_not_have_caught_it(self, _norm_dict: Any) -> None:
        """The silent-failure proof: guarding the OLD figure leaves the stage unguarded.

        Over 10k steps a pct_start of 0.01 is a perfectly healthy 100-step warm-up,
        so a guard fed the whole-run estimate is a no-op — and the stage that only
        gets 100 of those steps still divides by zero.
        """
        assert safe_pct_start(self.SHIPPED_PCT_START, self.WHOLE_RUN) == pytest.approx(
            self.SHIPPED_PCT_START
        )
        with pytest.raises(ZeroDivisionError):
            self._build_unguarded(self.SHIPPED_PCT_START, self._stage_steps(_norm_dict, 0))

    def test_guarding_the_stage_allocation_fixes_it(self, _norm_dict: Any) -> None:
        """Clamped against the stage's own allocation, the schedule builds and runs."""
        stage_steps = self._stage_steps(_norm_dict, 0)
        scheduler = TestSafePctStart._build(self.SHIPPED_PCT_START, stage_steps)
        for _ in range(stage_steps):
            scheduler.step()

    @staticmethod
    def _build_unguarded(pct_start: float, total_steps: int) -> Any:
        """OneCycleLR with the pct_start UNGUARDED — the pre-port behaviour."""
        net = LionNet()
        return torch.optim.lr_scheduler.OneCycleLR(
            torch.optim.AdamW(net.parameters(), lr=1e-4),
            max_lr=1e-3,
            total_steps=total_steps,
            div_factor=10.0,
            final_div_factor=10.0,
            pct_start=pct_start,
        )


class TestLionResolvesThroughThePerStageRebuild:
    """`optimizer: lion` must reach `salt.optim.Lion` on the per-stage path.

    The rebuild resolves the optimizer from `_active_optim_config()` — the
    ACTIVE stage's `optimizer` falling back to the top-level one — so the
    resolution has to key off that value, not off `self.optimizer`.
    """

    @staticmethod
    def _resolve(model: SaltModule, stage_index: int) -> type:
        """Exactly what `configure_optimizers` does to pick the optimizer class."""
        model._current_stage_index = stage_index  # noqa: SLF001
        _lrs, optimizer_name = model._active_optim_config()  # noqa: SLF001
        return model._get_optimizer_class(optimizer_name)  # noqa: SLF001

    def test_a_stage_overriding_the_optimizer_to_lion_resolves(self, _norm_dict: Any) -> None:
        """A per-stage `optimizer: lion` over an AdamW top level resolves to Lion."""
        model = SaltModule(
            build_gn2v2_modules(_norm_dict),
            lrs={"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1},
            optimizer="AdamW",
            training_schedule={
                "stages": {"warmup": {"epochs": 1, "optimizer": "lion"}, "full": {}}
            },
        )
        assert self._resolve(model, 0) is Lion  # the stage override wins
        assert self._resolve(model, 1) is not Lion  # the un-overriding stage falls back

    def test_a_top_level_lion_survives_the_rebuild_on_every_stage(self, _norm_dict: Any) -> None:
        """A top-level `optimizer: lion` is inherited by stages that do not override."""
        model = SaltModule(
            build_gn2v2_modules(_norm_dict),
            lrs={"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1},
            optimizer="lion",
            training_schedule={
                "stages": {"warmup": {"epochs": 1, "frozen": ["encoder"]}, "full": {}}
            },
        )
        assert self._resolve(model, 0) is Lion
        assert self._resolve(model, 1) is Lion

    def test_lion_is_salts_own_foreach_implementation_not_the_reference(
        self, _norm_dict: Any
    ) -> None:
        """`lion` is salt's batched Lion; the reference package is `lion-pytorch`."""
        model = SaltModule(
            build_gn2v2_modules(_norm_dict),
            lrs={"initial": 1e-4, "max": 1e-3, "end": 1e-5, "pct_start": 0.1},
            training_schedule={"stages": {"fit": {"optimizer": "lion"}}},
        )
        assert self._resolve(model, 0) is Lion
        assert Lion.__module__ == "salt.optim"
