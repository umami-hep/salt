"""P1 unit gates for the conversion ops."""

from __future__ import annotations

import numpy as np
import torch

from salt.graph.bundle import Bundle
from salt.graph.spec import Mode
from salt.outputs import ClassProbsOp
from salt.tests.unit.outputs.conftest import (
    _FLOAT_TOL,
    _STREAM_J,
    _bind_classification,
    _producer,
)

# GATE 1: ClassProbs (global classification) == task.run_inference softmax/sigmoid


def test_class_probs_global_softmax_matches_task_run_inference():
    """Global CE head: ``ClassProbs`` op == ``ClassificationTask.run_inference``."""
    torch.manual_seed(0)
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )
    logits = torch.randn(7, 3)

    oracle = module.run_inference(logits.clone())  # v1 softmax (tasks.py:322-324)

    producer = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    out = producer.forward(Bundle({"preds": {_STREAM_J: {"t": logits.clone()}}}), Mode.TEST)
    got = out[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)
    # probabilities sum to ~1 over the class dim (sanity)
    torch.testing.assert_close(got.sum(-1), torch.ones(7), rtol=0, atol=_FLOAT_TOL)


def test_class_probs_global_matches_literal_f4_softmax():
    """``ClassProbs`` op values == the literal f4-packed softmax columns
    (re-anchored from the retired ``task.get_h5`` oracle).
    """
    torch.manual_seed(1)
    logits = torch.randn(5, 3)
    oracle_cols = torch.softmax(logits, dim=-1).numpy().astype("f4")

    producer = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    got = producer.forward(
        Bundle({"preds": {_STREAM_J: {"t": logits.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_J}.out"]

    np.testing.assert_allclose(got.numpy(), oracle_cols, rtol=0, atol=_FLOAT_TOL)


def test_class_probs_bce_uses_sigmoid_matches_task():
    """BCE head: ``ClassProbs(bce=True)`` op == sigmoid run_inference (tasks.py:320-321)."""
    torch.manual_seed(2)
    module = _bind_classification(
        _STREAM_J,
        "flavour_label",
        ["sig"],
        sequence=False,
        loss={"class_path": "torch.nn.BCEWithLogitsLoss"},
    )
    logits = torch.randn(6, 1)
    oracle = module.run_inference(logits.clone())

    producer = _producer(ClassProbsOp(bce=True), stream=_STREAM_J, task="t")
    got = producer.forward(
        Bundle({"preds": {_STREAM_J: {"t": logits.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)
