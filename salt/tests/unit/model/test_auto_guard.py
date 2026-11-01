"""Tests for the automatic module-contract guard schedule of `SaltModule`."""

from __future__ import annotations

import pytest

from salt.graph.errors import MutationError, UndeclaredAccessError
from salt.graph.executor import Executor
from salt.graph.spec import Mode
from salt.tests.unit.model.test_saltmodule import (  # noqa: F401 - `data` is a fixture
    StopAfterFirstEpoch,
    build_datamodule,
    build_model,
    data,
    make_trainer,
)


@pytest.fixture
def guard_log(monkeypatch) -> list[tuple[Mode, bool]]:
    """Record ``(plan mode, debug)`` for every `Executor.run` call."""
    calls: list[tuple[Mode, bool]] = []
    real_run = Executor.run

    def run(self, bundle, debug=False):
        calls.append((self.plan.mode, debug))
        return real_run(self, bundle, debug=debug)

    monkeypatch.setattr(Executor, "run", run)
    return calls


def flags(calls: list[tuple[Mode, bool]], mode: Mode) -> list[bool]:
    return [debug for run_mode, debug in calls if run_mode is mode]


def patch_forward(monkeypatch, module, before) -> None:
    """Run ``before(bundle)`` ahead of `module`'s real forward (instance-level patch)."""
    original = module.forward

    def forward(b, mode):
        before(b)
        return original(b, mode)

    monkeypatch.setattr(module, "forward", forward)


class TestSanityBatchGuard:
    """``debug`` off: the sanity-check batches still run the contract check."""

    def test_undeclared_read_fails_the_sanity_batch(self, data, tmp_path, monkeypatch):
        model = build_model(data)
        assert model.debug is False
        patch_forward(monkeypatch, model.net["pool"], lambda b: b.get("inputs.jets"))
        trainer = make_trainer(num_sanity_val_steps=2, default_root_dir=tmp_path)
        with pytest.raises(RuntimeError, match=r"module 'pool' read bundle key 'inputs.jets'") as info:
            trainer.fit(model, build_datamodule(data))
        assert isinstance(info.value.__cause__, UndeclaredAccessError)
        assert "[mode=VAL]" in str(info.value)
        assert trainer.global_step == 0  # stopped before any optimizer step

    def test_inplace_mutation_fails_the_sanity_batch(self, data, tmp_path, monkeypatch):
        model = build_model(data)
        patch_forward(monkeypatch, model.net["norm"], lambda b: b.get("inputs.tracks").add_(1.0))
        trainer = make_trainer(num_sanity_val_steps=2, default_root_dir=tmp_path)
        with pytest.raises(RuntimeError, match=r"module 'norm' mutated bundle key 'inputs.tracks'") as info:
            trainer.fit(model, build_datamodule(data))
        assert isinstance(info.value.__cause__, MutationError)
        assert trainer.global_step == 0


class TestGuardSchedule:
    """Which batches the executor runs with ``debug=True``."""

    def test_first_fit_and_val_batch_guarded_then_not(self, data, guard_log, tmp_path):
        trainer = make_trainer(num_sanity_val_steps=0, default_root_dir=tmp_path)
        trainer.fit(build_model(data), build_datamodule(data))
        assert flags(guard_log, Mode.FIT) == [True, False]
        assert flags(guard_log, Mode.VAL) == [True, False]

    def test_sanity_batches_always_guarded(self, data, guard_log, tmp_path):
        trainer = make_trainer(num_sanity_val_steps=2, default_root_dir=tmp_path)
        trainer.fit(build_model(data), build_datamodule(data))
        # two sanity batches, then the epoch's validation: first guarded, second not
        assert flags(guard_log, Mode.VAL) == [True, True, True, False]
        assert flags(guard_log, Mode.FIT) == [True, False]

    def test_first_test_batch_guarded(self, data, guard_log, tmp_path):
        model, dm = build_model(data), build_datamodule(data)
        trainer = make_trainer(num_sanity_val_steps=0, limit_test_batches=2, default_root_dir=tmp_path)
        trainer.fit(model, dm)
        trainer.test(model, dm)
        assert flags(guard_log, Mode.TEST) == [True, False]

    def test_debug_true_guards_every_batch(self, data, guard_log, tmp_path):
        model, dm = build_model(data, debug=True), build_datamodule(data)
        trainer = make_trainer(num_sanity_val_steps=2, limit_test_batches=2, default_root_dir=tmp_path)
        trainer.fit(model, dm)
        trainer.test(model, dm)
        assert len(flags(guard_log, Mode.FIT)) == 2
        assert len(flags(guard_log, Mode.TEST)) == 2
        assert all(debug for _, debug in guard_log)

    def test_resume_rearms_the_first_fit_batch(self, data, guard_log, tmp_path):
        trainer = make_trainer(
            max_epochs=2,
            num_sanity_val_steps=0,
            callbacks=[StopAfterFirstEpoch()],
            default_root_dir=tmp_path,
        )
        trainer.fit(build_model(data), build_datamodule(data))
        ckpt = tmp_path / "epoch0.ckpt"
        trainer.save_checkpoint(ckpt)
        guard_log.clear()
        resumed = make_trainer(max_epochs=2, num_sanity_val_steps=0, default_root_dir=tmp_path)
        resumed.fit(build_model(data), build_datamodule(data), ckpt_path=ckpt)
        assert resumed.global_step == 4  # 2 saved + 2 resumed steps
        assert flags(guard_log, Mode.FIT) == [True, False]
