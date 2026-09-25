"""Unit tests for `salt.model.checkpoint` — warm-start load/classification/coverage."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.model.checkpoint import (
    WarmStartResult,
    apply_warm_start,
    coverage_mismatch,
    partition_by_module,
    reject_v1_and_strip_orig_mod,
    warm_start_from_checkpoint,
    warm_start_summary,
)


class _Net(nn.Module):
    """Two named modules -> `net.<name>.*` state-dict keys, matching a real `SaltModule`."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 2)})


class _NetAC(nn.Module):
    """Has `a` and `c`, no `b` — the warm-start "extra + missing" ckpt twin."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.ModuleDict({"a": nn.Linear(2, 2), "c": nn.Linear(2, 2)})


class _NetWrongShapeA(nn.Module):
    """Same module names as `_Net`, but `a` has an incompatible shape."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.ModuleDict({"a": nn.Linear(2, 3), "b": nn.Linear(2, 2)})


class TestRejectV1AndStripOrigMod:
    def test_v1_layout_raises(self) -> None:
        state = {"model.pool_net.gate_nn.weight": torch.zeros(1)}
        with pytest.raises(ConfigError, match=r"v1 .*not supported.*29c67a1"):
            reject_v1_and_strip_orig_mod(state)

    def test_orig_mod_prefix_is_stripped_into_a_new_dict(self) -> None:
        t = torch.zeros(1)
        state = {"_orig_mod.net.a.weight": t}
        result = reject_v1_and_strip_orig_mod(state)
        assert result is not state
        assert set(result) == {"net.a.weight"}
        assert result["net.a.weight"] is t

    def test_clean_dict_is_returned_by_identity(self) -> None:
        state = {"net.a.weight": torch.zeros(1)}
        assert reject_v1_and_strip_orig_mod(state) is state

    def test_none_passes_through(self) -> None:
        assert reject_v1_and_strip_orig_mod(None) is None

    def test_empty_dict_passes_through(self) -> None:
        state: dict = {}
        assert reject_v1_and_strip_orig_mod(state) is state


class TestPartitionByModule:
    def test_buckets_by_the_net_dot_name_prefix(self) -> None:
        state = {
            "net.a.weight": torch.zeros(1),
            "net.b.bias": torch.zeros(1),
            "other.x": torch.zeros(1),
        }
        buckets = partition_by_module(state)
        assert set(buckets["a"]) == {"net.a.weight"}
        assert set(buckets["b"]) == {"net.b.bias"}
        assert set(buckets[None]) == {"other.x"}


class TestCoverageMismatch:
    def test_identical_returns_none(self) -> None:
        t = torch.zeros(2, 2)
        current = {"net.a.weight": t.clone()}
        ckpt = {"net.a.weight": t.clone()}
        assert coverage_mismatch(current, ckpt) is None

    def test_key_missing_from_checkpoint(self) -> None:
        current = {"net.a.weight": torch.zeros(2, 2), "net.a.bias": torch.zeros(2)}
        ckpt = {"net.a.weight": torch.zeros(2, 2)}
        msg = coverage_mismatch(current, ckpt)
        assert msg is not None
        assert "missing from checkpoint" in msg

    def test_extra_key_in_checkpoint(self) -> None:
        current = {"net.a.weight": torch.zeros(2, 2)}
        ckpt = {"net.a.weight": torch.zeros(2, 2), "net.a.bias": torch.zeros(2)}
        msg = coverage_mismatch(current, ckpt)
        assert msg is not None
        assert "extra key(s)" in msg

    def test_shape_mismatch(self) -> None:
        current = {"net.a.weight": torch.zeros(2, 2)}
        ckpt = {"net.a.weight": torch.zeros(3, 2)}
        msg = coverage_mismatch(current, ckpt)
        assert msg is not None
        assert "shape mismatch" in msg

    def test_dtype_mismatch(self) -> None:
        current = {"net.a.weight": torch.zeros(2, 2, dtype=torch.float32)}
        ckpt = {"net.a.weight": torch.zeros(2, 2, dtype=torch.float64)}
        msg = coverage_mismatch(current, ckpt)
        assert msg is not None
        assert "dtype mismatch" in msg


class TestWarmStartSummary:
    def test_three_rows_in_loaded_new_dropped_order(self) -> None:
        summary = warm_start_summary(["a"], ["b"], ["c"])
        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        assert lines[0].startswith("loaded   a")
        assert lines[1].startswith("new      b")
        assert lines[2].startswith("dropped  c")

    def test_all_empty(self) -> None:
        assert warm_start_summary([], [], []) == "  (no module-level weights)"


class TestApplyWarmStart:
    def test_partitions_loaded_new_dropped(self) -> None:
        model = _Net()
        ckpt_source = _NetAC()
        before_b = {k: v.clone() for k, v in model.net["b"].state_dict().items()}

        result = apply_warm_start(
            model, ckpt_source.state_dict(), config_modules={"a": None, "b": None}, path="p"
        )

        assert result == WarmStartResult(loaded=["a"], new=["b"], dropped=["c"])
        for key, val in model.net["a"].state_dict().items():
            assert torch.equal(val, ckpt_source.net["a"].state_dict()[key])
        for key, val in model.net["b"].state_dict().items():
            assert torch.equal(val, before_b[key])

    def test_raises_on_a_partial_shape_mismatch(self) -> None:
        model = _Net()
        ckpt_source = _NetWrongShapeA()
        with pytest.raises(ConfigError, match=r"PARTIALLY covered.*swap"):
            apply_warm_start(
                model, ckpt_source.state_dict(), config_modules={"a": None, "b": None}, path="p"
            )


class TestWarmStartFromCheckpoint:
    def test_loads_from_a_real_checkpoint_file(self, tmp_path) -> None:
        model = _Net()
        twin = _NetAC()
        path = tmp_path / "w.ckpt"
        torch.save({"state_dict": twin.state_dict()}, path)

        result = warm_start_from_checkpoint(
            model,
            str(path),
            config_modules={"a": None, "b": None},
            plans={},
            bound=True,
            payload_key="salt_core",
        )
        assert result == WarmStartResult(loaded=["a"], new=["b"], dropped=["c"])

    def test_raises_before_bind(self, tmp_path) -> None:
        model = _Net()
        path = tmp_path / "w.ckpt"
        torch.save({"state_dict": _Net().state_dict()}, path)

        with pytest.raises(ConfigError, match="before bind"):
            warm_start_from_checkpoint(
                model,
                str(path),
                config_modules={"a": None, "b": None},
                plans={},
                bound=False,
                payload_key="salt_core",
            )

    def test_raises_when_checkpoint_carries_no_state_dict(self, tmp_path) -> None:
        model = _Net()
        path = tmp_path / "w.ckpt"
        torch.save({"not_state_dict": {}}, path)

        with pytest.raises(ConfigError, match="no 'state_dict'"):
            warm_start_from_checkpoint(
                model,
                str(path),
                config_modules={"a": None, "b": None},
                plans={},
                bound=True,
                payload_key="salt_core",
            )

    def test_plan_hash_mismatch_is_informational_not_raised(self, tmp_path, caplog) -> None:
        model = _Net()
        twin = _Net()
        path = tmp_path / "w.ckpt"
        torch.save(
            {
                "state_dict": twin.state_dict(),
                "salt_core": {"plan_hashes": {"FIT": "a" * 64}},
            },
            path,
        )
        plans = {Mode.FIT: SimpleNamespace(plan_hash="b" * 64)}

        with caplog.at_level(logging.INFO, logger="salt"):
            warm_start_from_checkpoint(
                model,
                str(path),
                config_modules={"a": None, "b": None},
                plans=plans,
                bound=True,
                payload_key="salt_core",
            )

        assert "plan hash differs" in caplog.text
