"""Tests for `salt.callbacks.PriorClosureMetrics` (per-head AUC / CE of the nn prior heads)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from lightning.pytorch.loggers.comet import CometLogger

from salt.callbacks import PriorClosureMetrics
from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.model.modules.kinematic_weight import KinematicWeight

NAMES, CHAIN = ["b", "c", "u"], "nn_kinematic_weight_iter_1"
LABEL, W_KEY = "labels.jets.flavour_label", f"labels.jets.{CHAIN}"
PRED_A, PRED_B = "preds.jets.kinematic_prior_1", "preds.jets.kinematic_prior"
HEAD_A, HEAD_B = "kinematic_weight_1", "kinematic_weight"
TRAINER = {"logger": None, "global_step": 7, "is_global_zero": True, "sanity_checking": False}


def make_modules(q_f=None, **kwargs) -> dict:
    """SaltModule-shaped `_graph_modules`: head 2 (chain = C_1) listed BEFORE head 1."""
    cls = {"n_classes": 3} if q_f is None else {"q_f": list(q_f)}
    weight = KinematicWeight(task_name="kinematic_prior", mode="nn", chain=[CHAIN], **cls, **kwargs)
    weight_1 = KinematicWeight(task_name="kinematic_prior_1", mode="nn", **cls, **kwargs)
    (weight.name, weight_1.name), prior = (HEAD_B, HEAD_A), SimpleNamespace(class_names=NAMES)
    return {HEAD_B: weight, "kinematic_prior": prior, HEAD_A: weight_1, "kinematic_prior_1": prior}


def make_bundle(y, logits_a, logits_b, w=None) -> Bundle:
    out = Bundle()
    out.set(LABEL, torch.as_tensor(y, dtype=torch.int64))
    out.set(PRED_A, torch.as_tensor(logits_a, dtype=torch.float32))
    out.set(PRED_B, torch.as_tensor(logits_b, dtype=torch.float32))
    out.set(W_KEY, torch.ones(len(y)) if w is None else torch.as_tensor(w, dtype=torch.float32))
    return out


def run(cb, modules, bundles, trainer=None) -> dict | None:
    trainer = trainer or SimpleNamespace(**TRAINER)
    pl_module = SimpleNamespace(_graph_modules=modules)
    cb.setup(trainer, pl_module, stage="fit")
    for i, bundle in enumerate(bundles):
        cb.on_validation_batch_end(trainer, pl_module, {"bundle": bundle}, None, i)
    cb.on_validation_epoch_end(trainer, pl_module)
    return cb.last_payload


def labels(seed: int, n: int = 300) -> np.ndarray:
    return np.random.default_rng(seed).choice(3, size=n, p=[0.5, 0.3, 0.2])


def noisy(y, seed: int) -> np.ndarray:  # float32-exact logits, correlated with the label
    return (np.random.default_rng(seed).normal(size=(len(y), 3)) + np.eye(3)[y]).astype(np.float32)


def constant(y) -> np.ndarray:  # every row's posterior = the validation class marginal
    return np.tile(np.log(np.bincount(y, minlength=3) / len(y)), (len(y), 1))


def perfect(y) -> np.ndarray:
    return 10.0 * np.eye(3)[y]


def softmax(logits) -> np.ndarray:
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float32).double(), -1).numpy()


def pair_auc(s, pos, w) -> float:  # brute-force weighted Mann-Whitney over every pair
    diff = s[pos][:, None] - s[~pos][None]
    pairs = w[pos][:, None] * w[~pos][None] * ((diff > 0) + 0.5 * (diff == 0))
    return float(pairs.sum() / (w[pos].sum() * w[~pos].sum()))


def test_discovery_order_demand_and_config_errors():
    mods, cb = make_modules(), PriorClosureMetrics()
    assert cb.fit_val_demand(mods) == (LABEL, PRED_A, PRED_B, W_KEY)
    assert cb._heads == [(HEAD_A, "kinematic_prior_1", None), (HEAD_B, "kinematic_prior", W_KEY)]
    extra = {f"{HEAD_B}_{k}": KinematicWeight("x", mode="nn", n_classes=3) for k in (10, 2)}
    cb.fit_val_demand({**extra, **mods})  # numeric suffix order, un-suffixed last
    assert [h[0] for h in cb._heads] == [HEAD_A, f"{HEAD_B}_2", f"{HEAD_B}_10", HEAD_B]
    assert PriorClosureMetrics([HEAD_B]).fit_val_demand(mods) == (LABEL, PRED_B, W_KEY)
    bad = [({"p": SimpleNamespace()}, None), (None, None), (mods, ["kinematic_prior"])]
    for modules, names in bad:
        with pytest.raises(ConfigError):
            PriorClosureMetrics(weight_modules=names).fit_val_demand(modules)


def test_constant_head_is_chance_at_the_floor_and_perfect_head_is_one():
    y, cb = labels(1), PriorClosureMetrics()
    p = run(cb, make_modules(), [make_bundle(y, constant(y), perfect(y))])
    a, b = p["heads"][HEAD_A], p["heads"][HEAD_B]
    assert a["auc_macro"] == pytest.approx(0.5, abs=1e-6)
    assert a["ce"] == pytest.approx(a["ce_floor"], abs=1e-6)
    q = np.bincount(y) / len(y)  # unit incoming weight: H(q_w) == the unweighted H(q_val)
    assert a["ce_floor"] == pytest.approx(p["ce_floor"])
    assert p["ce_floor"] == pytest.approx(-(q * np.log(q)).sum())
    assert all(b["auc"][c] == pytest.approx(1.0, abs=1e-12) for c in NAMES)
    assert {f"{HEAD_A}/auc_macro", f"{HEAD_B}/auc_u", "ce_floor", "n_jets"} <= set(cb.last_metrics)
    assert (p["n_jets"], a["iteration"], b["iteration"], b["weight_field"]) == (300, 1, 2, CHAIN)


def test_weighted_ce_auc_floor_and_constant_q_reference():
    y, q_f, logits = labels(3), np.array([0.2, 0.3, 0.5]), noisy(labels(3), 3)
    w = (np.linspace(0.5, 2.0, len(y)) * (1 + (y == 2))).astype(np.float32).astype(np.float64)
    p = run(PriorClosureMetrics(), make_modules(q_f), [make_bundle(y, logits, logits, w)])
    b, q_w, s = p["heads"][HEAD_B], np.bincount(y, weights=w) / w.sum(), softmax(logits)
    assert b["sum_w"] == pytest.approx(w.sum(), rel=1e-9)
    assert b["ce"] == pytest.approx(-(w * np.log(s[np.arange(len(y)), y])).sum() / w.sum())
    assert b["auc"] == pytest.approx({c: pair_auc(s[:, i], y == i, w) for i, c in enumerate(NAMES)})
    assert b["ce_floor"] == pytest.approx(-(q_w * np.log(q_w)).sum())
    assert abs(b["ce_floor"] - p["ce_floor"]) > 1e-3  # weighted H(q_w) != unweighted H(q_val)
    assert b["ce_const_q"] == pytest.approx(-(q_w * np.log(q_f)).sum())
    assert abs(b["ce_const_q"] - b["ce_floor"]) > 1e-3  # q_w != q_f
    ones = np.ones(len(y))  # head 1 (empty chain): unit weights
    unit = {c: pair_auc(s[:, i], y == i, ones) for i, c in enumerate(NAMES)}
    assert p["heads"][HEAD_A]["auc"] == pytest.approx(unit)
    assert max(abs(unit[c] - b["auc"][c]) for c in NAMES) > 1e-4  # the weights mattered


def test_balanced_prior_scores_the_recovered_posterior():
    y, q_f = labels(4), np.array([0.6, 0.3, 0.1])
    logits = noisy(y, 4)  # P_bal logits
    rec = softmax(logits) * q_f.astype(np.float32)  # the module's float32 q_f buffer
    rec /= rec.sum(-1, keepdims=True)  # P = q_f P_bal / sum_g q_g P_bal
    bundles = [make_bundle(y, logits, logits)]
    bal = run(PriorClosureMetrics(), make_modules(q_f, balanced_prior=True), bundles)
    raw = run(PriorClosureMetrics(), make_modules(q_f), bundles)  # same rows, no recovery
    bal, raw = bal["heads"][HEAD_A], raw["heads"][HEAD_A]
    want = {c: pair_auc(rec[:, i], y == i, np.ones(len(y))) for i, c in enumerate(NAMES)}
    assert bal["auc"] == pytest.approx(want)
    assert bal["ce"] == pytest.approx(-np.log(rec[np.arange(len(y)), y]).mean())
    assert max(abs(bal["auc"][c] - raw["auc"][c]) for c in NAMES) > 1e-4
    assert abs(bal["ce"] - raw["ce"]) > 1e-3


def test_absent_class_auc_is_null_with_support_and_excluded_from_macro():
    y, cb = labels(5) % 2, PriorClosureMetrics()  # no "u" jets
    h = run(cb, make_modules(), [make_bundle(y, perfect(y), perfect(y))])["heads"][HEAD_B]
    assert h["auc"]["u"] is None
    assert h["auc_support"]["u"] == {"n_pos": 0, "n_neg": len(y), "status": "no_positives"}
    n_b = int((y == 0).sum())
    assert h["auc_support"]["b"] == {"n_pos": n_b, "n_neg": len(y) - n_b, "status": "ok"}
    assert h["auc_macro"] == pytest.approx(np.mean([h["auc"]["b"], h["auc"]["c"]]))
    assert f"{HEAD_B}/auc_u" not in cb.last_metrics
    zeros = np.zeros(50, dtype=np.int64)  # only "b": no negatives for b, no positives otherwise
    p = run(cb, make_modules(), [make_bundle(zeros, perfect(zeros), perfect(zeros))])
    h = p["heads"][HEAD_B]
    assert h["auc_macro"] is None and h["auc_support"]["b"]["status"] == "no_negatives"
    assert f"{HEAD_B}/auc_macro" not in cb.last_metrics and set(h["auc"].values()) == {None}


def test_json_and_figure_written_invalid_labels_dropped_sanity_skipped(tmp_path):
    y, cb = labels(6), PriorClosureMetrics(output_dir=str(tmp_path))
    bad, w = y.copy(), np.linspace(1.0, 3.0, len(y))
    bad[:20] = -1
    sanity = SimpleNamespace(**TRAINER | {"sanity_checking": True})
    assert run(cb, make_modules(), [make_bundle(y, perfect(y), perfect(y))], sanity) is None
    assert not list(tmp_path.iterdir())
    cb.on_fit_start(SimpleNamespace(**TRAINER), None)  # no CometLogger: no comet_experiment.json
    c, s = constant(y), perfect(y)
    bundles = [make_bundle(bad, c, s, w), make_bundle(y, c, s)]
    run(cb, make_modules(), bundles)
    text = (tmp_path / "step000007.json").read_text()
    assert "NaN" not in text
    data = json.loads(text)
    assert [(n, h["iteration"]) for n, h in data["heads"].items()] == [(HEAD_A, 1), (HEAD_B, 2)]
    assert data["n_jets"] == data["heads"][HEAD_B]["n"] == 2 * len(y) - 20
    assert data["heads"][HEAD_B]["sum_w"] == pytest.approx(w[20:].sum() + len(y), rel=1e-6)
    assert data["heads"][HEAD_A]["weight_field"] is None and data["class_names"] == NAMES
    assert (tmp_path / "auc_ce_vs_iteration_step000007.png").exists()
    assert not (tmp_path / "comet_experiment.json").exists()
    assert len(cb.last_figures["auc_ce_vs_iteration"].axes) == 2


def test_retained_columns_own_their_storage():
    """Stored val columns are fresh CPU copies, never views of the (worker-shared) bundle."""
    y, mods, trainer = labels(8), make_modules(), SimpleNamespace(**TRAINER)
    logits = noisy(y, 8)
    shared, plain = make_bundle(y, logits, logits), make_bundle(y, logits, logits)
    for key in (LABEL, PRED_A, PRED_B, W_KEY):
        shared.get(key).share_memory_()  # simulate DataLoader-worker shared storage
    cb, pl_module = PriorClosureMetrics(), SimpleNamespace(_graph_modules=mods)
    cb.setup(trainer, pl_module, stage="fit")
    cb.on_validation_batch_end(trainer, pl_module, {"bundle": shared}, None, 0)
    (stored,) = cb._cols
    for key in cb.requires:
        src = shared.get(key)
        assert src.is_shared() and not stored[key].is_shared()
        assert stored[key].untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
        assert torch.equal(stored[key], src)
    cb.on_validation_epoch_end(trainer, pl_module)
    want = run(PriorClosureMetrics(), mods, [plain])["heads"][HEAD_B]["ce"]
    assert cb.last_payload["heads"][HEAD_B]["ce"] == want


def test_setup_accepts_validate_and_on_validation_start_writes_comet_json(tmp_path):
    """``salt validate``: setup runs on ``validate``, the Comet receipt is written at val start."""
    cb = PriorClosureMetrics(output_dir=str(tmp_path))
    cb.setup(SimpleNamespace(**TRAINER), SimpleNamespace(_graph_modules=make_modules()), "validate")
    assert cb.requires == (LABEL, PRED_A, PRED_B, W_KEY)
    off = PriorClosureMetrics(output_dir=str(tmp_path))
    off.setup(SimpleNamespace(**TRAINER), SimpleNamespace(_graph_modules=make_modules()), "test")
    assert off.requires == ()
    info = {"key": "k1", "url": "u1", "name": "n1", "project_name": "p1", "workspace": "w1"}
    logger_mock = MagicMock(spec=CometLogger)
    exp = logger_mock.experiment
    exp.get_key.return_value, exp.get_name.return_value = info["key"], info["name"]
    exp.url, exp.project_name, exp.workspace = info["url"], info["project_name"], info["workspace"]
    trainer = SimpleNamespace(**TRAINER | {"logger": logger_mock})
    cb.on_validation_start(trainer)
    cb.on_validation_start(trainer)  # idempotent
    assert json.loads((tmp_path / "comet_experiment.json").read_text()) == info
