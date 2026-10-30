"""Unit tests for ClassWeight, including bit-equality with KinematicWeight's accumulator."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.model.bind import ResolvedSchema
from salt.model.modules import ClassWeight, KinematicWeight

_KEY = "labels.jets.flavour_class_weight"


def make_cw(**kw) -> ClassWeight:
    cw = ClassWeight(**{"n_classes": 3, **kw})
    cw.name = "class_weight"
    return cw


def make_kw(**kw) -> KinematicWeight:
    km = KinematicWeight(mode="nn", task_name="head", n_classes=3, q_f=None, **kw)
    km.name = "kw"
    km.bind(ResolvedSchema(widths={}, fields={}))
    return km


def _bundle(labels, logits=None):
    b = Bundle()
    b.set("labels.jets.flavour_label", torch.as_tensor(labels))
    if logits is not None:
        b.set("preds.jets.head", torch.as_tensor(logits))
    return b


def _w(cw, labels, mode=Mode.FIT):
    return cw(_bundle(labels), mode)[_KEY]


def _state(cw):
    return [t.clone() for t in (cw.counts, cw.q, cw.frozen)]


def _same(cw, state):
    return all(torch.equal(a, b) for a, b in zip(_state(cw), state, strict=True))


@pytest.mark.parametrize("kwargs", [
    {"n_classes": 1}, {"n_classes": True}, {"n_classes": 2.5},
    {"p_floor": 0.0}, {"p_floor": 1.5}, {"w_max": 0.0},
])
def test_construction_errors(kwargs):
    """Invalid n_classes / p_floor / w_max raise ConfigError."""
    with pytest.raises(ConfigError, match="ClassWeight: "):
        make_cw(**kwargs)


def test_declare_io_training_only_and_custom_keys():
    """FIT/VAL declare the label and weight; TEST/ONNX are empty; keys follow the args."""
    for mode in (Mode.FIT, Mode.VAL):
        io = make_cw().declare_io(mode)
        req, prod = flatten_spec(io.requires), flatten_spec(io.produces)
        assert set(req) == {"labels.jets.flavour_label"} and set(prod) == {_KEY}
        for spec, dtype in ((req["labels.jets.flavour_label"], "int64"), (prod[_KEY], "float32")):
            assert (spec.shape, spec.dtype, spec.kind, spec.modes) == (
                ("B",), dtype, "label", Mode.TRAINING)
    for mode in (Mode.TEST, Mode.ONNX):
        io = make_cw().declare_io(mode)
        assert flatten_spec(io.requires) == flatten_spec(io.produces) == {}
    cw = make_cw(out="cw", stream="large_jets", label="lbl")
    assert (cw.label_key, cw.weight_key) == ("labels.large_jets.lbl", "labels.large_jets.cw")
    assert set(flatten_spec(cw.declare_io(Mode.FIT).produces)) == {"labels.large_jets.cw"}


def test_jet_mean_is_exactly_one():
    """One FIT batch with integer class counts gives w_f = N/(K n_f) and jet-mean 1."""
    cw, n = make_cw(), torch.tensor([50, 30, 20])
    labels = torch.repeat_interleave(torch.arange(3), n)
    w = _w(cw, labels)
    assert w.dtype == torch.float32 and w.requires_grad is False
    torch.testing.assert_close(w, (100 / (3 * n[labels])).float(), rtol=0, atol=1e-6)
    assert abs(w.mean().item() - 1.0) < 1e-6
    torch.testing.assert_close(cw.q, torch.tensor([0.5, 0.3, 0.2]))
    assert torch.equal(cw.counts, torch.tensor([50.0, 30.0, 20.0], dtype=torch.float64))


def test_invalid_rows_weight_one_and_not_counted():
    """Label -1 and out-of-range rows get weight 1.0 and are excluded from the counts."""
    cw = make_cw()
    w = _w(cw, [0, -1, 1, 5, 2])
    assert w[1].item() == w[3].item() == 1.0
    assert torch.equal(cw.counts, torch.ones(3, dtype=torch.float64))


def test_val_never_accumulates():
    """A VAL batch leaves counts and q untouched and is weighted by the unchanged q."""
    cw = make_cw()
    _w(cw, [0, 0, 1, 2])
    before = _state(cw)
    w = _w(cw, [2, 2, 2, 1], Mode.VAL)
    assert _same(cw, before)
    torch.testing.assert_close(w, 1.0 / (3 * cw.q[torch.tensor([2, 2, 2, 1])]))


def test_freeze(caplog):
    """Freeze raises when empty, stores the unfloored q, is idempotent and stops accumulation."""
    with pytest.raises(RuntimeError, match="no accumulated"):
        make_cw().freeze_accumulators(0, 0)
    cw = make_cw()
    _w(cw, [0, 0, 1, 2, 1, 0])
    with caplog.at_level(logging.INFO, logger="salt"):
        cw.freeze_accumulators(3, 0)
    assert cw.accumulators_frozen is True
    assert torch.equal(cw.q, (cw.counts / cw.counts.sum()).to(torch.float32))
    assert "q frozen at global step 3 (epoch 0)" in caplog.text
    frozen = _state(cw)
    cw.freeze_accumulators(4, 1)
    _w(cw, [2, 2, 2])
    assert _same(cw, frozen)


def test_empty_class_capped_at_w_max(caplog):
    """A never-seen class freezes to q=0, warns, and is weighted exactly w_max."""
    cw = make_cw()
    _w(cw, [0, 0, 1])
    with caplog.at_level(logging.WARNING, logger="salt"):
        cw.freeze_accumulators(1, 0)
    assert "class index 2" in caplog.text and cw.q[2].item() == 0.0
    w = _w(cw, [0, 1, 2], Mode.VAL)
    assert w[2].item() == 50.0
    torch.testing.assert_close(w[:2], 1.0 / (3 * cw.q[:2]))


def test_state_dict_round_trip():
    """The three buffers are the state; loading them restores the frozen state."""
    cw = make_cw()
    assert set(cw.state_dict()) == {"counts", "q", "frozen"}
    _w(cw, [0, 1, 1, 2])
    cw.freeze_accumulators(2, 0)
    fresh = make_cw()
    fresh.load_state_dict(cw.state_dict())
    assert fresh.accumulators_frozen is True and _same(fresh, _state(cw))
    _w(fresh, [0, 0, 0])
    assert _same(fresh, _state(cw))


@pytest.mark.parametrize(("p_floor", "batches", "q2"), [
    (1e-4, ([0, 0, 1], [2, -1, 1], [0, 5, 2]), (2 / 7, 2 / 7)),  # -1 / out-of-range rows
    (0.3, ([0, 0, 1, 1, 2],), (0.3, 0.2)),  # class 2 below p_floor: floored only while running
    (1e-4, ([0, 1, 1],), (1e-4, 0.0)),  # class 2 never seen: frozen to 0
])
def test_q_bit_identical_to_kinematic_weight(p_floor, batches, q2):
    """counts/q equal KinematicWeight's q_f_counts/q_f after every FIT batch and at freeze."""
    cw, km = make_cw(p_floor=p_floor), make_kw(p_floor=p_floor)
    for labels in batches:
        km(_bundle(labels, torch.zeros(len(labels), 3)), Mode.FIT)
        cw(_bundle(labels), Mode.FIT)
        assert torch.equal(cw.counts, km.q_f_counts) and torch.equal(cw.q, km.q_f)
    assert cw.q[2].item() == pytest.approx(q2[0])
    for module in (cw, km):
        module.freeze_accumulators(3, 0)
    assert torch.equal(cw.counts, km.q_f_counts) and torch.equal(cw.q, km.q_f)
    assert cw.q[2].item() == pytest.approx(q2[1])


def test_ddp_reduces_each_fit_batch_only(monkeypatch):
    """FIT increments are all-reduced; VAL and the freeze trigger no further reduce."""
    for name, value in (("is_available", True), ("is_initialized", True), ("get_world_size", 2)):
        monkeypatch.setattr(torch.distributed, name, lambda value=value: value)
    fake = Mock(side_effect=lambda t, op=None: t.mul_(2))
    monkeypatch.setattr(torch.distributed, "all_reduce", fake)
    cw = make_cw()
    _w(cw, [0, 0, 1])
    assert torch.equal(cw.counts, torch.tensor([4.0, 2.0, 0.0], dtype=torch.float64))
    _w(cw, [1, 2], Mode.VAL)
    cw.freeze_accumulators(1, 0)
    assert fake.call_count == 1
