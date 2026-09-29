import math
import types

import pytest
import torch

from salt.modelwrapper import ModelWrapper

# per-task losses taken from a real diverged training run (epoch 6)
REAL_LOSSES = {
    "jets_classification": 1.9171,
    "track_origin": 0.6487,
    "track_vertexing": 0.3833,
    "track_type": 0.5895,
    "jet_pt_regression": 0.1034,
    "jets_bccharge": 0.6790,
}


def _wrapper(loss_mode="GLS", floor=0.0):
    # total_loss only reads these attributes
    return types.SimpleNamespace(
        loss_mode=loss_mode,
        gls_weight_floor=floor,
        _gls_weights={},
        trainer=types.SimpleNamespace(world_size=1),
    )


class _LogStub:
    """Records what log_losses emits, without a Lightning trainer behind it."""

    total_loss = ModelWrapper.total_loss
    log_losses = ModelWrapper.log_losses

    def __init__(self, loss_mode="GLS"):
        self.loss_mode = loss_mode
        self.gls_weight_floor = 0.0
        self._gls_weights = {}
        self._dwa_weights = {}
        self.trainer = types.SimpleNamespace(world_size=1, device_ids=[0])
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = value


def _tensors(vals, requires_grad=False):
    return {
        k: torch.tensor(v, dtype=torch.float64, requires_grad=requires_grad)
        for k, v in vals.items()
    }


def _reference_gls(loss):
    # the original implementation: pow(prod(losses), 1/N)
    return torch.pow(math.prod(loss.values()), 1.0 / len(loss))


def test_gls_value_matches_geometric_mean():
    loss = _tensors(REAL_LOSSES)
    got = ModelWrapper.total_loss(_wrapper(), loss)
    assert torch.allclose(got, _reference_gls(loss), rtol=0, atol=1e-12)


def test_gls_gradient_matches_geometric_mean():
    ref_in = _tensors(REAL_LOSSES, requires_grad=True)
    new_in = _tensors(REAL_LOSSES, requires_grad=True)
    _reference_gls(ref_in).backward()
    ModelWrapper.total_loss(_wrapper(), new_in).backward()
    for k in ref_in:
        assert torch.allclose(ref_in[k].grad, new_in[k].grad, rtol=0, atol=1e-12)


def test_gls_weights_are_recorded_and_inverse_in_loss():
    w = _wrapper()
    loss = _tensors(REAL_LOSSES)
    total = ModelWrapper.total_loss(w, loss)
    assert w._gls_weights is not None
    assert set(w._gls_weights) == set(REAL_LOSSES)
    n = len(REAL_LOSSES)
    for k, v in loss.items():
        assert torch.allclose(w._gls_weights[k], total / (n * v), rtol=0, atol=1e-12)
    # the worst task must carry the smallest weight
    assert min(w._gls_weights, key=lambda k: w._gls_weights[k].item()) == "jets_classification"


def test_gls_weights_are_detached():
    loss = _tensors(REAL_LOSSES, requires_grad=True)
    w = _wrapper()
    ModelWrapper.total_loss(w, loss)
    assert all(not t.requires_grad for t in w._gls_weights.values())


def test_gls_weight_floor_lifts_only_the_starved_task():
    plain, floored = _wrapper(), _wrapper(floor=0.5)
    ModelWrapper.total_loss(plain, _tensors(REAL_LOSSES))
    ModelWrapper.total_loss(floored, _tensors(REAL_LOSSES))
    n = len(REAL_LOSSES)
    for k in REAL_LOSSES:
        assert floored._gls_weights[k].item() >= 0.5 / n - 1e-12
        if k == "jets_classification":
            assert floored._gls_weights[k].item() > plain._gls_weights[k].item()
        else:
            assert floored._gls_weights[k].item() == pytest.approx(plain._gls_weights[k].item())


def test_gls_floor_zero_is_a_noop():
    a, b = _wrapper(floor=0.0), _wrapper()
    ta = ModelWrapper.total_loss(a, _tensors(REAL_LOSSES))
    tb = ModelWrapper.total_loss(b, _tensors(REAL_LOSSES))
    assert torch.equal(ta, tb)


def test_gls_stays_finite_over_the_representable_range():
    # a running product of these underflows to zero; log-space must not
    vals = dict.fromkeys(REAL_LOSSES, 1e-40)
    w = _wrapper()
    total = ModelWrapper.total_loss(w, _tensors(vals))
    assert torch.isfinite(total)
    assert all(torch.isfinite(t) for t in w._gls_weights.values())


def test_a_zero_loss_is_not_silently_rescaled():
    # clamping the loss would hand this task a ~1e9 weight; NaN reaches the training_step guard
    vals = dict(REAL_LOSSES, jet_pt_regression=0.0)
    total = ModelWrapper.total_loss(_wrapper(), _tensors(vals))
    assert total.isnan()


def test_gls_weights_are_shared_across_ranks():
    # a second rank whose jet classification loss is much lower than on this rank
    other = dict(REAL_LOSSES, jets_classification=0.9)
    w = _wrapper()
    w.trainer.world_size = 2
    w.all_gather = lambda x: torch.stack([x, torch.stack(list(_tensors(other).values()))])
    local = _tensors(REAL_LOSSES, requires_grad=True)
    ModelWrapper.total_loss(w, local).backward()

    averaged = {k: (REAL_LOSSES[k] + other[k]) / 2 for k in REAL_LOSSES}
    ref = _wrapper()
    ModelWrapper.total_loss(ref, _tensors(averaged))
    for k in REAL_LOSSES:
        assert torch.allclose(w._gls_weights[k], ref._gls_weights[k], rtol=0, atol=1e-12)
        assert torch.allclose(local[k].grad, ref._gls_weights[k], rtol=0, atol=1e-12)


def test_gls_weights_are_logged():
    w = _LogStub()
    loss = _tensors(REAL_LOSSES)
    loss["loss"] = w.total_loss(loss)
    w.log_losses(loss, stage="train")
    for k in REAL_LOSSES:
        assert w.logged[f"train/{k}_gls_weight"] == w._gls_weights[k]


def test_wsum_logs_no_weights():
    w = _LogStub(loss_mode="wsum")
    loss = _tensors(REAL_LOSSES)
    loss["loss"] = w.total_loss(loss)
    w.log_losses(loss, stage="train")
    assert not [n for n in w.logged if n.endswith("_gls_weight")]


def test_wsum_is_a_plain_sum_and_sets_no_weights():
    w = _wrapper(loss_mode="wsum")
    loss = _tensors(REAL_LOSSES)
    total = ModelWrapper.total_loss(w, loss)
    assert total.item() == pytest.approx(sum(REAL_LOSSES.values()))
    assert w._gls_weights == {}
