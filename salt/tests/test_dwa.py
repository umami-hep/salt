import types

import pytest
import torch

from salt.modelwrapper import ModelWrapper

TASKS = ["jets_classification", "track_origin", "track_vertexing"]


class _Stub:
    """Minimal stand-in exercising only the DWA machinery of ModelWrapper."""

    total_loss = ModelWrapper.total_loss
    log_losses = ModelWrapper.log_losses
    _dwa_reset_accumulator = ModelWrapper._dwa_reset_accumulator
    _dwa_accumulate = ModelWrapper._dwa_accumulate
    _dwa_update_weights = ModelWrapper._dwa_update_weights
    on_train_epoch_end = ModelWrapper.on_train_epoch_end
    on_save_checkpoint = ModelWrapper.on_save_checkpoint
    on_load_checkpoint = ModelWrapper.on_load_checkpoint

    def __init__(self, temperature=2.0, loss_mode="DWA"):
        self.loss_mode = loss_mode
        self.dwa_temperature = temperature
        self._dwa_prev = {}
        self._dwa_prev2 = {}
        self._dwa_weights = {}
        self.trainer = types.SimpleNamespace(world_size=1, device_ids=[0])
        self.logged = {}
        self._dwa_reset_accumulator()

    def log(self, name, value, **kwargs):
        self.logged[name] = value


def _losses(vals):
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in vals.items()}


def _run_epoch(w, vals, steps=3):
    for _ in range(steps):
        loss = _losses(vals)
        loss["loss"] = w.total_loss(loss)
        w._dwa_accumulate(loss)
    w.on_train_epoch_end()


FLAT = dict.fromkeys(TASKS, 1.0)
DIVERGING = {"jets_classification": 2.0, "track_origin": 0.5, "track_vertexing": 0.5}


def test_first_two_epochs_are_uniform():
    w = _Stub()
    assert w.total_loss(_losses(FLAT)).item() == pytest.approx(3.0)
    _run_epoch(w, FLAT)
    # one epoch in there is still no ratio, so weights stay unset and the total is a plain sum
    assert w._dwa_weights == {}
    assert w.total_loss(_losses(FLAT)).item() == pytest.approx(3.0)


def test_weights_appear_after_two_epochs_and_sum_to_n():
    w = _Stub()
    _run_epoch(w, FLAT)
    _run_epoch(w, {"jets_classification": 0.9, "track_origin": 0.5, "track_vertexing": 0.8})
    assert set(w._dwa_weights) == set(TASKS)
    assert sum(w._dwa_weights.values()) == pytest.approx(len(TASKS))


def test_worsening_task_gains_weight():
    w = _Stub()
    _run_epoch(w, FLAT)
    _run_epoch(w, DIVERGING)
    weights = w._dwa_weights
    # the inverse of GLS: the task that got worse is up-weighted, not abandoned
    assert weights["jets_classification"] > weights["track_origin"]
    assert weights["jets_classification"] > 1.0
    assert weights["track_origin"] < 1.0


def test_higher_temperature_flattens_weights():
    def spread(temp):
        w = _Stub(temperature=temp)
        _run_epoch(w, FLAT)
        _run_epoch(w, DIVERGING)
        vals = list(w._dwa_weights.values())
        return max(vals) - min(vals)

    assert spread(10.0) < spread(0.5)


def test_weights_are_applied_to_the_total():
    w = _Stub()
    _run_epoch(w, FLAT)
    _run_epoch(w, DIVERGING)
    vals = {"jets_classification": 1.0, "track_origin": 2.0, "track_vertexing": 3.0}
    expected = sum(w._dwa_weights[k] * v for k, v in vals.items())
    assert w.total_loss(_losses(vals)).item() == pytest.approx(expected)


def test_gradients_flow_through_weighted_sum():
    w = _Stub()
    _run_epoch(w, FLAT)
    _run_epoch(w, DIVERGING)
    loss = {k: torch.tensor(1.0, dtype=torch.float64, requires_grad=True) for k in TASKS}
    w.total_loss(loss).backward()
    for k in TASKS:
        assert loss[k].grad.item() == pytest.approx(w._dwa_weights[k])


def test_state_survives_a_checkpoint_round_trip():
    w = _Stub()
    _run_epoch(w, FLAT)
    _run_epoch(w, DIVERGING)
    ckpt: dict = {}
    w.on_save_checkpoint(ckpt)
    assert "dwa_state" in ckpt

    fresh = _Stub()
    fresh.on_load_checkpoint(ckpt)
    assert fresh._dwa_weights == w._dwa_weights
    assert fresh._dwa_prev == w._dwa_prev
    assert fresh._dwa_prev2 == w._dwa_prev2


def test_zero_previous_loss_is_rejected():
    # dividing by it would hand that task the whole weight budget and starve the rest
    w = _Stub()
    _run_epoch(w, {"jets_classification": 0.0, "track_origin": 1.0, "track_vertexing": 1.0})
    with pytest.raises(AssertionError):
        _run_epoch(w, FLAT)


def test_a_changed_task_set_is_rejected():
    w = _Stub()
    _run_epoch(w, FLAT)
    with pytest.raises(AssertionError):
        _run_epoch(w, {"jets_classification": 1.0, "track_origin": 1.0})


def test_weights_are_logged():
    w = _Stub()
    _run_epoch(w, FLAT)
    _run_epoch(w, DIVERGING)
    loss = _losses(FLAT)
    loss["loss"] = w.total_loss(loss)
    w.log_losses(loss, stage="train")
    for k in TASKS:
        assert w.logged[f"train/{k}_dwa_weight"] == pytest.approx(w._dwa_weights[k])


def test_other_loss_modes_do_not_accumulate_or_save():
    w = _Stub(loss_mode="wsum")
    w.on_train_epoch_end()
    assert w._dwa_weights == {}
    ckpt: dict = {}
    w.on_save_checkpoint(ckpt)
    assert "dwa_state" not in ckpt
