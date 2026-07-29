"""P1 unit gates for the conversion ops."""

from __future__ import annotations

import numpy as np
import torch

from salt.graph import Mode
from salt.graph.bundle import Bundle
from salt.outputs import ClassProbsOp, SeqClassIndexOp, SeqClassProbsOp, TaskOutput
from salt.utils.tensor_utils import masked_softmax  # parity-oracle access
from salt.tests.unit.outputs.conftest import (
    _FLOAT_TOL,
    _STREAM_J,
    _STREAM_T,
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


# GATE 2: SeqClassIndex (sequence classification) == argmax over masked softmax


def test_seq_class_index_matches_argmax_of_masked_softmax():
    """Sequence head: ``SeqClassIndex`` == ``argmax(masked_softmax(logits))``."""
    torch.manual_seed(4)
    b, t, c = 5, 6, 8
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 3:] = True  # padded tail on jet 0
    mask[1, :] = True  # fully padded jet (zero valid tokens edge case)

    # ORACLE: the exact masked-softmax then argmax over classes
    probs = masked_softmax(logits.clone(), mask.unsqueeze(-1))
    oracle = torch.argmax(probs, dim=-1)

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassIndexOp())
    producer.name = "p"
    out = producer.forward(
        Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}}),
        Mode.TEST,
    )
    got = out[f"outputs.{_STREAM_T}.origin"]

    assert got.dtype == torch.int64
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)  # int — EXACT


def test_seq_class_index_argmax_invariant_to_softmax():
    """``argmax`` over (masked) softmax == ``argmax`` over the raw logits on valid tokens."""
    torch.manual_seed(5)
    b, t, c = 4, 5, 8
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 2:] = True

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassIndexOp())
    producer.name = "p"
    got = producer.forward(
        Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}}), Mode.TEST
    )[f"outputs.{_STREAM_T}.origin"]

    raw_argmax = torch.argmax(logits, dim=-1)
    valid = ~mask
    torch.testing.assert_close(got[valid], raw_argmax[valid], rtol=0, atol=0)


def test_seq_class_index_no_pad_mask_branch():
    """``SeqClassIndexOp(has_pad_mask=False)`` argmaxes a plain softmax (no mask demanded)."""
    torch.manual_seed(6)
    logits = torch.randn(3, 4, 8)
    oracle = torch.argmax(torch.softmax(logits.clone(), dim=-1), dim=-1)

    op = SeqClassIndexOp(has_pad_mask=False)
    assert op.extra_requires("objects") == {}  # no masks.objects demand
    producer = TaskOutput(task="t", stream="objects", name="origin", op=op)
    producer.name = "p"
    got = producer.forward(
        Bundle({"preds": {"objects": {"t": logits.clone()}}}), Mode.TEST
    )["outputs.objects.origin"]
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)


# GATE 2b: SeqClassIndex ONNX branch == reduces._bind_argmax,
#          and is mode-branched (TEST [B,L] int64 unchanged, ONNX [L] int8).


def _argmax_reduce_oracle(probs_1lc: torch.Tensor) -> torch.Tensor:
    """The verbatim ``reduces._bind_argmax`` math on a converted ``[1, L, C]`` probs tensor."""
    scores = torch.concatenate([probs_1lc, torch.zeros((1, 1, probs_1lc.shape[-1]))], dim=1)
    return torch.argmax(scores, dim=-1)[:, :-1].squeeze(0).char()


def test_seq_class_index_onnx_branch_matches_bind_argmax():
    """The ONNX branch carries the v1 zero-row argmax trick VERBATIM (folds ``_bind_argmax``)."""
    torch.manual_seed(7)
    length, classes = 6, 8
    logits = torch.randn(1, length, classes)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, 4:] = True  # padded tail

    op = SeqClassIndexOp()
    b = Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}})
    got = op.convert(b, Mode.ONNX, pred_key=f"preds.{_STREAM_T}.t", stream=_STREAM_T)

    probs = masked_softmax(logits.clone(), mask.unsqueeze(-1))
    oracle = _argmax_reduce_oracle(probs)

    assert got.dtype == torch.int8
    assert got.shape == (length,)  # [L], the batch dim squeezed (Athena per-token)
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)


def test_seq_class_index_onnx_branch_valid_for_zero_token_jet():
    """The zero-row trick keeps the ONNX argmax valid for a zero-token jet (L=0)."""
    op = SeqClassIndexOp()
    logits = torch.randn(1, 0, 8)  # zero tokens
    mask = torch.zeros(1, 0, dtype=torch.bool)
    b = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    got = op.convert(b, Mode.ONNX, pred_key=f"preds.{_STREAM_T}.t", stream=_STREAM_T)
    assert got.dtype == torch.int8
    assert got.shape == (0,)


def test_seq_class_index_test_branch_unchanged_by_onnx_fold():
    """The TEST branch stays ``[B, L]`` int64 — the ONNX zero-row trick is ONNX-only (R2)."""
    torch.manual_seed(8)
    b_, length, classes = 5, 6, 8
    logits = torch.randn(b_, length, classes)
    mask = torch.zeros(b_, length, dtype=torch.bool)
    mask[0, 3:] = True

    op = SeqClassIndexOp()
    test_b = Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}})
    got = op.convert(test_b, Mode.TEST, pred_key=f"preds.{_STREAM_T}.t", stream=_STREAM_T)

    oracle = torch.argmax(masked_softmax(logits.clone(), mask.unsqueeze(-1)), dim=-1)
    assert got.dtype == torch.int64
    assert got.shape == (b_, length)  # [B, L] — full batch, no zero-row strip
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)


def test_seq_class_index_onnx_int8_argmax_invariant_to_softmax():
    """The ONNX int8 leaf == argmax of the RAW logits on valid tokens (argmax invariance)."""
    torch.manual_seed(9)
    length, classes = 7, 8
    logits = torch.randn(1, length, classes)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, 5:] = True

    op = SeqClassIndexOp()
    b = Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}})
    got = op.convert(b, Mode.ONNX, pred_key=f"preds.{_STREAM_T}.t", stream=_STREAM_T)

    raw_argmax = torch.argmax(logits.squeeze(0), dim=-1).char()
    valid = ~mask.squeeze(0)
    torch.testing.assert_close(got[valid], raw_argmax[valid], rtol=0, atol=0)


# GATE 2b: SeqClassProbs (sequence classification PROBS) == masked softmax,
#          and == the literal f4-packed eval-H5 float columns.
#          This is the eval-H5 column counterpart of the SeqClassIndex argmax.


def test_seq_class_probs_matches_masked_softmax():
    """Sequence head: ``SeqClassProbs`` == ``masked_softmax(logits)`` (tasks.py:322-327)."""
    torch.manual_seed(40)
    b, t, c = 5, 6, 8
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 3:] = True  # padded tail on jet 0
    mask[1, :] = True  # fully padded jet (zero valid tokens edge case)

    oracle = masked_softmax(logits.clone(), mask.unsqueeze(-1))

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassProbsOp())
    producer.name = "p"
    got = producer.forward(
        Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}}),
        Mode.TEST,
    )[f"outputs.{_STREAM_T}.origin"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)
    # padded tokens read exactly 0.0 (the v1 eval byte-parity quirk)
    assert int(torch.count_nonzero(got[mask])) == 0
    # valid tokens are a probability simplex (sum to 1)
    valid = ~mask
    torch.testing.assert_close(got[valid].sum(-1), torch.ones(int(valid.sum())), atol=1e-6, rtol=0)


def test_seq_class_probs_matches_task_run_inference_and_literal_f4_pack():
    """``SeqClassProbs`` == the bound task's sequence ``run_inference`` AND the
    literal f4-packed masked-softmax columns (re-anchored from the retired
    ``task.get_h5`` oracle).
    """
    torch.manual_seed(41)
    b, t = 4, 5
    class_names = ["Pileup", "Fake", "Primary", "FromB", "FromBC", "FromC", "FromTau", "Other"]
    module = _bind_classification(
        _STREAM_T, "ftagTruthOriginLabel", class_names, sequence=True
    )
    logits = torch.randn(b, t, len(class_names))
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 2:] = True

    oracle = module.run_inference(logits.clone(), mask)

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassProbsOp())
    producer.name = "p"
    got = producer.forward(
        Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}}),
        Mode.TEST,
    )[f"outputs.{_STREAM_T}.origin"]
    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)

    # == the literal f4-packed per-class masked-softmax columns (what the eval
    # H5 serialises after the sink's f4 downcast) — the retired get_h5 pack.
    packed = masked_softmax(logits.clone(), mask.unsqueeze(-1)).numpy().astype("f4")
    np.testing.assert_allclose(packed, got.numpy(), rtol=0, atol=_FLOAT_TOL)
