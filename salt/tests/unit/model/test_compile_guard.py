"""`--compile` with the automatic guard: guarded batches run eagerly and never recompile.

CPU-only and inductor-free (a counting eager backend, tiny fixture models) — runs
unconditionally in the unit-test job, like ``test_compile_regression.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from salt.graph.errors import ConfigError
from salt.graph.spec import Mode
from salt.tests.unit.model.test_saltmodule import (  # noqa: F401 - `data` is a fixture
    build_datamodule,
    build_model,
    data,
)


def setup_test_stage(data, model):
    """Compile + bind the TEST plan through the real `SaltModule.setup`, then materialise."""
    dm = build_datamodule(data)
    dm.set_sinks(model.sink_demand())
    dm.setup("test")
    model._trainer = SimpleNamespace(callbacks=[], datamodule=dm)  # noqa: SLF001 - duck-typed attach
    model.setup("test")  # ends in _apply_compile
    model.on_fit_start()  # fresh weights: materialise the norm stats
    return dm


def test_guarded_batch_runs_eagerly_and_compiled_batches_do_not_recompile(data):
    torch._dynamo.reset()  # noqa: SLF001 - the dynamo test surface
    counter = CompileCounter()
    model = build_model(data)
    model.enable_compile(backend=counter)
    dm = setup_test_stage(data, model)
    batch = dm.test_dset[slice(0, 50)]
    with torch.no_grad():
        guarded = model(batch, Mode.TEST)  # first TEST batch: guarded -> force_eager
        assert counter.frame_count == 0, "a guarded batch reached dynamo"
        compiled = model(batch, Mode.TEST)  # unguarded: compiles here, once
        frames = counter.frame_count
        assert frames > 0, "the unguarded batch did not run compiled"
        model(batch, Mode.TEST)
        model._arm_guard()  # noqa: SLF001 - a second guarded batch after compilation
        model(batch, Mode.TEST)
    assert counter.frame_count == frames, "a later batch recompiled"
    preds = [key for key in compiled.keys() if key.startswith("preds.")]  # noqa: SIM118
    assert preds
    for key in preds:
        assert guarded.get(key).shape == compiled.get(key).shape
        assert torch.isfinite(compiled.get(key)).all()
        torch.testing.assert_close(guarded.get(key), compiled.get(key))


def test_missing_set_stance_is_a_config_error_at_setup(data, monkeypatch):
    monkeypatch.delattr(torch.compiler, "set_stance")
    model = build_model(data)
    model.enable_compile(backend="eager")
    with pytest.raises(ConfigError, match=r"--compile needs torch.compiler.set_stance"):
        setup_test_stage(data, model)
    assert not model._compiled  # noqa: SLF001 - nothing was compiled
