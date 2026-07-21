"""W6 reducer-safe freeze helpers (plan 06) — the pure freeze/optimizer-membership
primitives that fix the DDP init-frozen-unfreeze desync.

CPU-safe: no DataLoader, no DDP, no spawn. Exercises `reducer_safe_freeze_required`,
`apply_stage_freeze`, `trainable_named_params`, and `clear_frozen_grads` directly on
a toy `ModuleDict` + fake strategy/schedule so the semantics that the exp-07 gate
proves under a real reducer are also locked at the unit level.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from salt.schedule import (
    StageConfig,
    TrainingSchedule,
    apply_stage_freeze,
    clear_frozen_grads,
    reducer_safe_freeze_required,
    trainable_named_params,
)

MODULES = ("A", "B", "C")


def _net() -> nn.ModuleDict:
    torch.manual_seed(0)
    return nn.ModuleDict({
        "A": nn.Linear(4, 4, bias=False),
        "B": nn.Linear(4, 4, bias=False),
        "C": nn.Linear(4, 1, bias=False),
    })


def _named(net: nn.ModuleDict):
    # mimic SaltModule param names: net.<module>.<param>
    return [(f"net.{n}", p) for n, p in net.named_parameters()]


class _FakeDDP:
    def __init__(self):
        self._ddp_kwargs = {"find_unused_parameters": False}


class _FakeSingleDevice:
    pass  # no _ddp_kwargs


_FLIP = TrainingSchedule(
    [
        StageConfig(name="warmup", epochs=2, trainable=("B", "C")),
        StageConfig(name="mid", epochs=2, trainable=("C",)),
        StageConfig(name="full", trainable=("A", "B", "C")),
    ],
    MODULES,
)
_STATIC = TrainingSchedule(
    [
        StageConfig(name="warmup", epochs=2, frozen=("A",)),
        StageConfig(name="full", frozen=("A",)),
    ],
    MODULES,
)
_LEGACY = TrainingSchedule.desugar_legacy(MODULES)


class TestPredicate:
    def test_ddp_plus_freeze_flip_requires_reducer_safe(self):
        assert reducer_safe_freeze_required(_FakeDDP(), _FLIP) is True

    def test_single_device_never_reducer_safe(self):
        assert reducer_safe_freeze_required(_FakeSingleDevice(), _FLIP) is False

    def test_ddp_static_freeze_not_reducer_safe(self):
        # frozen set never changes across stages -> requires_grad freeze is safe
        assert reducer_safe_freeze_required(_FakeDDP(), _STATIC) is False

    def test_ddp_legacy_single_stage_not_reducer_safe(self):
        assert reducer_safe_freeze_required(_FakeDDP(), _LEGACY) is False

    def test_none_schedule_not_reducer_safe(self):
        assert reducer_safe_freeze_required(_FakeDDP(), None) is False


class TestApplyStageFreeze:
    def test_default_mode_toggles_requires_grad(self):
        net = _net()
        apply_stage_freeze(net, {"A"}, set(), reducer_safe=False)
        assert all(not p.requires_grad for p in net["A"].parameters())
        assert not net["A"].training  # eval
        assert all(p.requires_grad for p in net["B"].parameters())

    def test_default_mode_unfreeze_restores(self):
        net = _net()
        apply_stage_freeze(net, {"A"}, set(), reducer_safe=False)
        apply_stage_freeze(net, set(), {"A"}, reducer_safe=False)
        assert all(p.requires_grad for p in net["A"].parameters())
        assert net["A"].training  # train restored

    def test_reducer_safe_keeps_requires_grad_true(self):
        net = _net()
        apply_stage_freeze(net, {"A"}, set(), reducer_safe=True)
        # frozen but STILL requires_grad (stays in the DDP reducer at wrap)
        assert all(p.requires_grad for p in net["A"].parameters())
        assert not net["A"].training  # but eval()

    def test_reducer_safe_unfreeze_sets_train(self):
        net = _net()
        apply_stage_freeze(net, {"A"}, set(), reducer_safe=True)
        apply_stage_freeze(net, set(), {"A"}, reducer_safe=True)
        assert net["A"].training


class TestTrainableNamedParams:
    def test_excludes_frozen_prefix_when_requires_grad_true(self):
        # reducer-safe: A frozen but requires_grad True -> excluded by prefix
        net = _net()
        names = [n for n, _ in trainable_named_params(_named(net), {"A"})]
        assert not any(n.startswith("net.A.") for n in names)
        assert any(n.startswith("net.B.") for n in names)
        assert any(n.startswith("net.C.") for n in names)

    def test_default_mode_matches_requires_grad_filter(self):
        # frozen via requires_grad=False -> same selection as prefix filter
        net = _net()
        apply_stage_freeze(net, {"A"}, set(), reducer_safe=False)
        by_helper = {n for n, _ in trainable_named_params(_named(net), {"A"})}
        by_reqgrad = {n for n, p in _named(net) if p.requires_grad}
        assert by_helper == by_reqgrad

    def test_no_frozen_returns_all_requires_grad(self):
        net = _net()
        names = {n for n, _ in trainable_named_params(_named(net), set())}
        assert names == {n for n, p in _named(net) if p.requires_grad}

    def test_prefix_not_a_substring_collision(self):
        # "A" must not match "AB": a module named A frozen leaves AB trainable
        net = nn.ModuleDict({"A": nn.Linear(2, 2), "AB": nn.Linear(2, 2)})
        named = [(f"net.{n}", p) for n, p in net.named_parameters()]
        names = {n for n, _ in trainable_named_params(named, {"A"})}
        assert any(n.startswith("net.AB.") for n in names)
        assert not any(n.startswith("net.A.") and not n.startswith("net.AB.") for n in names)


class TestClearFrozenGrads:
    def test_clears_only_frozen(self):
        net = _net()
        x = torch.randn(3, 4)
        out = net["C"](torch.relu(net["B"](torch.relu(net["A"](x)))))
        out.sum().backward()
        assert all(p.grad is not None for p in net["A"].parameters())
        clear_frozen_grads(net, {"A"})
        assert all(p.grad is None for p in net["A"].parameters())
        # B/C untouched
        assert all(p.grad is not None for p in net["B"].parameters())

    def test_prevents_accumulation_across_steps(self):
        net = _net()
        x = torch.randn(3, 4)

        def step():
            out = net["C"](torch.relu(net["B"](torch.relu(net["A"](x)))))
            out.sum().backward()

        step()
        g1 = net["A"].weight.grad.clone()  # single-step grad
        clear_frozen_grads(net, {"A"})
        step()
        g2 = net["A"].weight.grad.clone()
        # after clearing, the second step's grad is a single-step grad, not 2x
        assert torch.allclose(g1, g2)
