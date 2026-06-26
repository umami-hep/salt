"""P1 unit gates for the classification + regression output PRODUCERS (plan 01, P1).

Each conversion op behind the generic ``TaskOutput`` (design §4b) must reproduce
its source task's eval math BITWISE (ints exact, floats ≤1e-6) — the producer
forward is the SAME ``run_inference`` the M4.5 task published in TEST mode
(`salt.core.nn.tasks`), now living in an in-graph producer instead of a
mode-branch inside ``task.forward``. These tests build a real bound task head,
feed synthetic RAW (training-space) predictions, and assert:

- `ClassProbs` (global classification) == ``ClassificationTask.run_inference``
  softmax / sigmoid (``tasks.py:312-328``);
- `SeqClassIndex` (sequence classification) == ``argmax`` over the masked
  softmax (the ONNX ``TrackOrigin`` index math, ``reduces.py:_bind_argmax`` /
  ``tasks.py:118-137``);
- `Regression` (de-scale) == ``RegressionTask.run_inference`` de-scaling for
  the scaler / norm_params / ratio-denominator cases (``tasks.py:522-548``).

Plus the demand-gating / width-resolution contract carries over for every op
(design §4 risk 6): the produced ``outputs.*`` width resolves in a TEST-only
bind, with the ``argmax`` width collapsing to 1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from salt.core.graph import (
    IO,
    ConfigError,
    Mode,
    TensorSpec,
    compile_plan,
    sym_dim,
    unflatten_spec,
)
from salt.core.graph.bundle import Bundle
from salt.core.nn import resolve_bind_schema
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.tasks import ClassificationTaskModule, RegressionTaskModule
from salt.core.outputs import (
    ClassProbs,
    ClassProbsOp,
    ConversionOp,
    Regression,
    RegressionDescaleOp,
    SeqClassIndex,
    SeqClassIndexOp,
    SeqClassProbs,
    SeqClassProbsOp,
    TaskOutput,
)
from salt.core.outputs.producers import _masked_softmax  # noqa: PLC2701 — parity-oracle access

_FLOAT_TOL = 1e-6

_STREAM_J = "jets"
_STREAM_T = "tracks"


# ---------------------------------------------------------------------------
# helpers — build + bind a real task head; run its op on a synthetic bundle
# ---------------------------------------------------------------------------


def _bind_classification(stream, label, class_names, sequence, *, loss=None, input_key=None):
    """Build + bind a `ClassificationTaskModule` against a hand-built schema.

    Returns
    -------
    ClassificationTaskModule
        The bound module (``module.task`` is the absorbed v1 head ORACLE).
    """
    module = ClassificationTaskModule(
        stream=stream,
        label=label,
        class_names=class_names,
        input=input_key,
        sequence=sequence,
        loss=loss,
    )
    module.name = f"{stream}_cls"
    schema = ResolvedSchema(widths={module.input_key: 8})
    module.bind(schema)
    return module


def _bind_regression(
    stream, targets, *, sequence, denoms=None, norm=None, scaler=None, gaussian=False, fields=()
):
    """Build + bind a `RegressionTaskModule` against a hand-built schema.

    Returns
    -------
    RegressionTaskModule
        The bound module (``module.task`` is the absorbed v1 head ORACLE).
    """
    module = RegressionTaskModule(
        stream=stream,
        targets=targets,
        sequence=sequence,
        gaussian=gaussian,
        target_denominators=denoms,
        norm_params=norm,
        scaler=scaler,
    )
    module.name = f"{stream}_reg"
    widths = {module.input_key: 8}
    schema = ResolvedSchema(widths=widths, fields={module.input_feature_key: fields})
    module.bind(schema)
    return module


def _producer(op, *, task="t", stream=_STREAM_J, name="out"):
    """Construct a named `TaskOutput` carrying `op`.

    Returns
    -------
    TaskOutput
        The producer (instance name set, ready to ``forward``).
    """
    producer = TaskOutput(task=task, stream=stream, name=name, op=op)
    producer.name = "producer"
    return producer


# ---------------------------------------------------------------------------
# GATE 1: ClassProbs (global classification) == task.run_inference softmax/sigmoid
# ---------------------------------------------------------------------------


def test_class_probs_global_softmax_matches_task_run_inference():
    """Global CE head: ``ClassProbs`` op == ``ClassificationTask.run_inference``."""
    torch.manual_seed(0)
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )
    logits = torch.randn(7, 3)

    oracle = module.task.run_inference(logits.clone())  # v1 softmax (tasks.py:322-324)

    producer = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    out = producer.forward(Bundle({"preds": {_STREAM_J: {"t": logits.clone()}}}), Mode.TEST)
    got = out[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)
    # probabilities sum to ~1 over the class dim (sanity)
    torch.testing.assert_close(got.sum(-1), torch.ones(7), rtol=0, atol=_FLOAT_TOL)


def test_class_probs_global_matches_task_get_h5_values():
    """``ClassProbs`` op values == the f4 columns ``task.get_h5`` would emit.

    Since the P1.5 flip the task publishes RAW logits in TEST and ``get_h5``
    run_inference-s them itself (softmax) before packing the f4 columns; the
    producer applies the same softmax, so the producer's un-structured floats
    must equal the floats ``get_h5`` writes from the SAME raw logits (run-name
    prefix is sink-side). Both convert ONCE (no double-convert).
    """
    torch.manual_seed(1)
    module = _bind_classification(
        _STREAM_J, "flavour_label", ["bjets", "cjets", "ujets"], sequence=False
    )
    logits = torch.randn(5, 3)
    # the flipped TEST forward publishes RAW logits; get_h5 softmaxes + packs
    b_oracle = Bundle({"preds": {_STREAM_J: {module.name: logits.clone()}}})
    structured = module.get_h5(b_oracle, run_name="GN2")
    oracle_cols = np.stack([structured[n] for n in structured.dtype.names], axis=-1)

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
    oracle = module.task.run_inference(logits.clone())

    producer = _producer(ClassProbsOp(bce=True), stream=_STREAM_J, task="t")
    got = producer.forward(
        Bundle({"preds": {_STREAM_J: {"t": logits.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_class_probs_subclass_forwards_like_op():
    """The thin `ClassProbs` subclass == a `TaskOutput` carrying `ClassProbsOp`."""
    torch.manual_seed(3)
    logits = torch.randn(4, 3)
    sub = ClassProbs(task="t", stream=_STREAM_J, name="out")
    sub.name = "p"
    generic = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    b = Bundle({"preds": {_STREAM_J: {"t": logits}}})
    torch.testing.assert_close(
        sub.forward(b, Mode.TEST)[f"outputs.{_STREAM_J}.out"],
        generic.forward(Bundle({"preds": {_STREAM_J: {"t": logits}}}), Mode.TEST)[
            f"outputs.{_STREAM_J}.out"
        ],
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# GATE 2: SeqClassIndex (sequence classification) == argmax over masked softmax
# ---------------------------------------------------------------------------


def test_seq_class_index_matches_argmax_of_masked_softmax():
    """Sequence head: ``SeqClassIndex`` == ``argmax(masked_softmax(logits))``.

    Reproduces the v1 TrackOrigin index math: the per-token masked softmax the
    sequence ``run_inference`` applies (tasks.py:322-327) then ``argmax`` over
    the class dim (the int index the ONNX argmax reduce emits).
    """
    torch.manual_seed(4)
    b, t, c = 5, 6, 8
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 3:] = True  # padded tail on jet 0
    mask[1, :] = True  # fully padded jet (zero valid tokens edge case)

    # ORACLE: the exact masked-softmax then argmax over classes
    probs = _masked_softmax(logits.clone(), mask.unsqueeze(-1))
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
    """``argmax`` over (masked) softmax == ``argmax`` over the raw logits on valid tokens.

    The ONNX argmax reduce argmaxes RAW logits where the producer argmaxes the
    converted probs; the two agree because argmax is invariant under the
    monotone softmax (reduces.py:28-30). Checked on the VALID positions only
    (padded positions are converted to a tie of zeros, whose argmax is 0 — a
    sink-side concern, not part of this invariance claim).
    """
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


def test_seq_class_index_subclass_forwards_like_op():
    """The thin `SeqClassIndex` subclass == a `TaskOutput` carrying `SeqClassIndexOp`."""
    torch.manual_seed(12)
    logits = torch.randn(3, 4, 8)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 2:] = True
    sub = SeqClassIndex(task="t", stream=_STREAM_T, name="origin")
    sub.name = "p"
    generic = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassIndexOp())
    generic.name = "g"
    b1 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    b2 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    torch.testing.assert_close(
        sub.forward(b1, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        generic.forward(b2, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# GATE 2b (plan-29 W2): SeqClassIndex ONNX branch == reduces._bind_argmax,
#          and is mode-branched (TEST [B,L] int64 unchanged, ONNX [L] int8).
# ---------------------------------------------------------------------------


def _argmax_reduce_oracle(probs_1lc: torch.Tensor) -> torch.Tensor:
    """The verbatim ``reduces._bind_argmax`` math on a converted ``[1, L, C]`` probs tensor.

    Returns
    -------
    Tensor
        The int8 ``[L]`` ONNX argmax output (the zero-row append/strip + char).
    """
    scores = torch.concatenate([probs_1lc, torch.zeros((1, 1, probs_1lc.shape[-1]))], dim=1)
    return torch.argmax(scores, dim=-1)[:, :-1].squeeze(0).char()


def test_seq_class_index_onnx_branch_matches_bind_argmax():
    """The ONNX branch carries the v1 zero-row argmax trick VERBATIM (folds ``_bind_argmax``).

    `SeqClassIndexOp.convert(mode=Mode.ONNX)` must reproduce the v1 ONNX argmax
    reduce (``to_onnx.py:415-423`` / ``reduces._bind_argmax``): masked softmax,
    then the zero-row append/strip trick, ``.squeeze(0).char()`` to int8 ``[L]``.
    Asserted EXACT against the reduce-math oracle on the converted probs.
    """
    torch.manual_seed(7)
    length, classes = 6, 8
    logits = torch.randn(1, length, classes)
    mask = torch.zeros(1, length, dtype=torch.bool)
    mask[0, 4:] = True  # padded tail

    op = SeqClassIndexOp()
    b = Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}})
    got = op.convert(b, Mode.ONNX, pred_key=f"preds.{_STREAM_T}.t", stream=_STREAM_T)

    probs = _masked_softmax(logits.clone(), mask.unsqueeze(-1))
    oracle = _argmax_reduce_oracle(probs)

    assert got.dtype == torch.int8
    assert got.shape == (length,)  # [L], the batch dim squeezed (Athena per-token)
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)


def test_seq_class_index_onnx_branch_valid_for_zero_token_jet():
    """The zero-row trick keeps the ONNX argmax valid for a zero-token jet (L=0).

    The whole point of the appended row (``to_onnx.py:418-421``): ``argmax`` over a
    zero-length token axis would be undefined, so a constant row is appended then
    stripped. With L=0 the output is an empty int8 vector — never an error.
    """
    op = SeqClassIndexOp()
    logits = torch.randn(1, 0, 8)  # zero tokens
    mask = torch.zeros(1, 0, dtype=torch.bool)
    b = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    got = op.convert(b, Mode.ONNX, pred_key=f"preds.{_STREAM_T}.t", stream=_STREAM_T)
    assert got.dtype == torch.int8
    assert got.shape == (0,)


def test_seq_class_index_test_branch_unchanged_by_onnx_fold():
    """The TEST branch stays ``[B, L]`` int64 — the ONNX zero-row trick is ONNX-only (R2).

    Mode-branching is load-bearing: the zero-row append/strip is ONNX-shape-
    specific and must NOT run on a TEST batch (it would corrupt the eval int64
    column / change the width). TEST is unchanged from the pre-W2 behaviour.
    """
    torch.manual_seed(8)
    b_, length, classes = 5, 6, 8
    logits = torch.randn(b_, length, classes)
    mask = torch.zeros(b_, length, dtype=torch.bool)
    mask[0, 3:] = True

    op = SeqClassIndexOp()
    test_b = Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}})
    got = op.convert(test_b, Mode.TEST, pred_key=f"preds.{_STREAM_T}.t", stream=_STREAM_T)

    oracle = torch.argmax(_masked_softmax(logits.clone(), mask.unsqueeze(-1)), dim=-1)
    assert got.dtype == torch.int64
    assert got.shape == (b_, length)  # [B, L] — full batch, no zero-row strip
    torch.testing.assert_close(got, oracle, rtol=0, atol=0)


def test_seq_class_index_onnx_int8_argmax_invariant_to_softmax():
    """The ONNX int8 leaf == argmax of the RAW logits on valid tokens (argmax invariance).

    v1 argmaxed RAW logits where the v2 conversion softmaxes first; argmax is
    invariant under the monotone (masked) softmax, so the int8 output is
    identical on the valid positions (reduces.py:28-30).
    """
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


# ---------------------------------------------------------------------------
# GATE 2b: SeqClassProbs (sequence classification PROBS) == masked softmax,
#          and == the float values the sequence ClassificationTask.get_h5 packs.
#          This is the eval-H5 column counterpart of the SeqClassIndex argmax.
# ---------------------------------------------------------------------------


def test_seq_class_probs_matches_masked_softmax():
    """Sequence head: ``SeqClassProbs`` == ``masked_softmax(logits)`` (tasks.py:322-327).

    The per-token per-class probabilities the eval H5 ``{run_name}_p{origin}``
    columns carry; padded tokens read 0.0 (the masked softmax zeroes them).
    """
    torch.manual_seed(40)
    b, t, c = 5, 6, 8
    logits = torch.randn(b, t, c)
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[0, 3:] = True  # padded tail on jet 0
    mask[1, :] = True  # fully padded jet (zero valid tokens edge case)

    oracle = _masked_softmax(logits.clone(), mask.unsqueeze(-1))

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


def test_seq_class_probs_matches_task_run_inference_and_get_h5():
    """``SeqClassProbs`` == the bound task's sequence ``run_inference`` AND its ``get_h5`` floats.

    Builds a real bound sequence `ClassificationTaskModule`; the producer probs
    equal both ``task.run_inference`` (the conversion) and the float values
    ``task.get_h5`` packs into the eval columns (tasks.py:1257-1271) — the
    end-to-end eval-H5 value the M4.5 ``TaskWriter`` writes.
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

    oracle = module.task.run_inference(logits.clone(), mask)

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassProbsOp())
    producer.name = "p"
    got = producer.forward(
        Bundle({"preds": {_STREAM_T: {"t": logits.clone()}}, "masks": {_STREAM_T: mask}}),
        Mode.TEST,
    )[f"outputs.{_STREAM_T}.origin"]
    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)

    # == the float values the task's get_h5 packs (what the M4.5 writer serialises).
    # Since the P1.5 flip get_h5 reads the RAW logits leaf (+ masks.<stream>) and
    # run_inference-s them itself, so feed it the SAME raw logits + mask; the
    # packed floats then equal the producer probs (both convert ONCE).
    structured = module.get_h5(
        Bundle(
            {"preds": {_STREAM_T: {module.name: logits.clone()}}, "masks": {_STREAM_T: mask}}
        ),
        "RUN",
    )
    packed = np.stack([structured[f"RUN_p{c}"] for c in class_names], axis=-1)
    np.testing.assert_allclose(packed, got.numpy(), rtol=0, atol=_FLOAT_TOL)


def test_seq_class_probs_subclass_forwards_like_op():
    """The thin `SeqClassProbs` subclass == a `TaskOutput` carrying `SeqClassProbsOp`."""
    torch.manual_seed(42)
    logits = torch.randn(3, 4, 8)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 2:] = True
    sub = SeqClassProbs(task="t", stream=_STREAM_T, name="origin")
    sub.name = "p"
    generic = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassProbsOp())
    generic.name = "g"
    b1 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    b2 = Bundle({"preds": {_STREAM_T: {"t": logits}}, "masks": {_STREAM_T: mask}})
    torch.testing.assert_close(
        sub.forward(b1, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        generic.forward(b2, Mode.TEST)[f"outputs.{_STREAM_T}.origin"],
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# GATE 3: Regression de-scale == RegressionTask.run_inference
# ---------------------------------------------------------------------------


def test_regression_norm_params_matches_task_run_inference():
    """Norm_params head: ``Regression`` op == ``RegressionTask.run_inference`` (tasks.py:536-9)."""
    torch.manual_seed(7)
    norm = {"mean": [3.0, -1.0], "std": [2.0, 0.5]}
    module = _bind_regression(_STREAM_J, ["mHH", "dR"], sequence=False, norm=norm)
    preds = torch.randn(9, 2)

    oracle = module.task.run_inference(preds.clone())  # v1 de-norm, no mask

    op = RegressionDescaleOp(stream=_STREAM_J, targets=["mHH", "dR"], norm_params=norm)
    got = _producer(op, stream=_STREAM_J, task="t").forward(
        Bundle({"preds": {_STREAM_J: {"t": preds.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_regression_scaler_matches_task_run_inference():
    """Functional scaler head: ``Regression`` op == the scaler branch of run_inference.

    v1's scaler branch indexes ``preds[:, :, i]`` (sequence), so this exercises
    a per-token head (tasks.py:540-542).
    """
    torch.manual_seed(8)
    scales = {"pt": {"op": "log", "x_scale": 5}, "Lxy": {"op": "linear", "x_scale": 2, "x_off": 1}}
    module = _bind_regression(_STREAM_T, ["pt", "Lxy"], sequence=True, scaler=scales)
    preds = torch.randn(4, 5, 2)
    mask = torch.zeros(4, 5, dtype=torch.bool)

    # v1 run_inference masks with NaN; the descale MATH is the scaler.inverse —
    # compare with an all-valid mask so no NaN fill enters (the NaN fill is a
    # sink-side serialisation concern, design §2 layer 2)
    oracle = module.task.run_inference(preds.clone(), pad_mask=mask)

    op = RegressionDescaleOp(stream=_STREAM_T, targets=["pt", "Lxy"], scaler=scales)
    got = _producer(op, stream=_STREAM_T, task="t").forward(
        Bundle({"preds": {_STREAM_T: {"t": preds.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_T}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_regression_ratio_denominator_test_mode_matches_task():
    """Ratio-denominator head (TEST): ``Regression`` op == v1 de-scale via labels.

    The denominator source in TEST is ``labels.<stream>.<denom>`` (the same
    nesting v1 ``run_inference`` reads, tasks.py:587-589). mHH is also a
    declared input Feature so the ONNX path has a source.
    """
    torch.manual_seed(9)
    module = _bind_regression(
        _STREAM_J, ["m_over_mHH"], sequence=False, denoms=["mHH"], fields=("mHH", "pt")
    )
    preds = torch.randn(6, 1)
    denom = torch.rand(6) + 0.5

    # v1 run_inference(labels={stream: {denom: tensor}}) (tasks.py:533-535)
    oracle = module.task.run_inference(preds.clone(), labels={_STREAM_J: {"mHH": denom}})

    op = RegressionDescaleOp(stream=_STREAM_J, targets=["m_over_mHH"], target_denominators=["mHH"])
    op.bind(("mHH", "pt"))  # capture input-Feature field order (ONNX gather)
    got = _producer(op, stream=_STREAM_J, task="t").forward(
        Bundle(
            {
                "preds": {_STREAM_J: {"t": preds.clone()}},
                "labels": {_STREAM_J: {"mHH": denom}},
            }
        ),
        Mode.TEST,
    )[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_regression_ratio_denominator_onnx_gathers_by_name():
    """ratio-denominator head (ONNX): the denominator is gathered by NAME from inputs.<stream>.

    Mirrors v1 ``get_onnx`` (tasks.py:2281-2294): the export graph has no label
    group, so the denominator comes from the raw input Feature tensor at the
    bound column index — must equal the TEST result fed the same denominator.
    """
    torch.manual_seed(10)
    fields = ("pt", "mHH", "eta")
    preds = torch.randn(5, 1)
    denom = torch.rand(5) + 0.5
    # inputs.<stream> = [B, F] with mHH at column index 1
    inputs = torch.zeros(5, len(fields))
    inputs[:, fields.index("mHH")] = denom

    op = RegressionDescaleOp(stream=_STREAM_J, targets=["m_over_mHH"], target_denominators=["mHH"])
    op.bind(fields)

    onnx = _producer(op, stream=_STREAM_J, task="t").forward(
        Bundle({"preds": {_STREAM_J: {"t": preds.clone()}}, "inputs": {_STREAM_J: inputs}}),
        Mode.ONNX,
    )[f"outputs.{_STREAM_J}.out"]

    # same math as TEST sourcing the denominator from the label group
    expected = preds.clone()
    expected[..., 0] *= denom
    torch.testing.assert_close(onnx, expected.float(), rtol=0, atol=_FLOAT_TOL)


def test_regression_subclass_forwards_like_op():
    """The thin `Regression` subclass == a `TaskOutput` carrying `RegressionDescaleOp`."""
    torch.manual_seed(11)
    norm = {"mean": [1.0], "std": [3.0]}
    preds = torch.randn(4, 1)
    sub = Regression(task="t", stream=_STREAM_J, targets=["mHH"], name="out", norm_params=norm)
    sub.name = "p"
    op = RegressionDescaleOp(stream=_STREAM_J, targets=["mHH"], norm_params=norm)
    generic = _producer(op, stream=_STREAM_J, task="t")
    torch.testing.assert_close(
        sub.forward(Bundle({"preds": {_STREAM_J: {"t": preds}}}), Mode.TEST)[
            f"outputs.{_STREAM_J}.out"
        ],
        generic.forward(Bundle({"preds": {_STREAM_J: {"t": preds}}}), Mode.TEST)[
            f"outputs.{_STREAM_J}.out"
        ],
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# W34.3 critic-fix GATE: the gaussian + sequence producer branches == the task's
# run_inference oracle (the branches test_producers.py previously left untested).
# After the per-token axis fix (run_inference now indexes [..., i]), the producer
# (already last-axis) and the task path agree on global AND per-token heads.
# ---------------------------------------------------------------------------


def _gaussian_concat(means_stds: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Concat the task's ``(means, stds)`` into the producer's one ``[..., 2R]`` array.

    Returns
    -------
    torch.Tensor
        The ``means ‖ stds`` one-array form the gaussian producer publishes.
    """
    means, stds = means_stds
    return torch.cat([means, stds], dim=-1)


def test_regression_gaussian_global_producer_matches_task_run_inference():
    """Global gaussian head: ``RegressionDescaleOp(gaussian=True)`` == means‖stds oracle.

    The gaussian producer ``_convert_gaussian`` REIMPLEMENTS the v1 loop rather
    than calling ``run_inference`` — so it is asserted directly against the task's
    ``GaussianRegressionTask.run_inference`` (concatenated means‖stds, the FD
    1567-1568 one-array contract). R=1 (every shipped gaussian head is R=1).
    """
    torch.manual_seed(20)
    norm = {"mean": [2.0], "std": [3.0]}
    module = _bind_regression(_STREAM_J, ["mHH"], sequence=False, gaussian=True, norm=norm)
    preds = torch.randn(8, 2)  # [B, 2R] = mean ‖ raw-var

    oracle = _gaussian_concat(module.task.run_inference(preds.clone()))

    op = RegressionDescaleOp(stream=_STREAM_J, targets=["mHH"], norm_params=norm, gaussian=True)
    got = _producer(op, stream=_STREAM_J, task="t").forward(
        Bundle({"preds": {_STREAM_J: {"t": preds.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_regression_gaussian_per_token_producer_matches_task_run_inference():
    """Per-token gaussian seq head: producer == task on EVERY token (W34.3 axis fix).

    The shipped ``regression_gaussian.yaml gaussian_seq_out`` head is per-token
    gaussian. Pre-fix the task indexed the first (token) axis and diverged from
    the producer's last-axis ``_convert_gaussian``; post-fix both index the target
    channel so means‖stds agree on every token, incl. NaN at masked positions.
    """
    torch.manual_seed(21)
    norm = {"mean": [1.0], "std": [1.0]}
    module = _bind_regression(_STREAM_T, ["dphi"], sequence=True, gaussian=True, norm=norm)
    preds = torch.randn(3, 4, 2)  # [B, L, 2R]
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 3] = True  # a real padded position -> NaN-filled means + stds

    oracle = _gaussian_concat(module.task.run_inference(preds.clone(), pad_mask=mask))

    op = RegressionDescaleOp(
        stream=_STREAM_T, targets=["dphi"], norm_params=norm, gaussian=True, sequence=True
    )
    got = _producer(op, stream=_STREAM_T, task="t").forward(
        Bundle({"preds": {_STREAM_T: {"t": preds.clone()}}, "masks": {_STREAM_T: mask}}), Mode.TEST
    )[f"outputs.{_STREAM_T}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL, equal_nan=True)
    # the masked position is NaN on both sides (sanity on the nan-fill path)
    assert torch.isnan(got[0, 3]).all()
    assert torch.isnan(oracle[0, 3]).all()


def test_regression_sequence_nan_fill_producer_matches_task_with_real_padding():
    """Per-token scaled seq head: producer ``_nan_fill`` == task run_inference NaN-fill.

    Uses a norm_params per-token head with a pad mask that has REAL padded rows
    (the existing scaler test deliberately used an all-valid mask, leaving the
    nan-fill path untested). The W34.3 axis fix makes the task's de-scale loop
    last-axis, so producer == task on the valid rows AND both NaN-fill the padded
    rows identically.
    """
    torch.manual_seed(22)
    norm = {"mean": [10.0, 20.0], "std": [2.0, 0.5]}
    module = _bind_regression(_STREAM_T, ["a", "b"], sequence=True, norm=norm)
    preds = torch.randn(2, 5, 2)  # [B, L, R]
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 4] = True
    mask[1, 3] = True
    mask[1, 4] = True  # real padded rows on both jets

    oracle = module.task.run_inference(preds.clone(), pad_mask=mask)

    op = RegressionDescaleOp(stream=_STREAM_T, targets=["a", "b"], norm_params=norm, sequence=True)
    got = _producer(op, stream=_STREAM_T, task="t").forward(
        Bundle({"preds": {_STREAM_T: {"t": preds.clone()}}, "masks": {_STREAM_T: mask}}), Mode.TEST
    )[f"outputs.{_STREAM_T}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL, equal_nan=True)
    # the de-scale reached EVERY token's columns (not just tokens 0,1 — the pre-fix
    # first-axis bug left later tokens raw); token 2 of jet 0 is a valid scaled row
    assert torch.isfinite(got[0, 2]).all()
    assert torch.isnan(got[0, 4]).all()


# ---------------------------------------------------------------------------
# producer config-error guards (mirror the task module's guards)
# ---------------------------------------------------------------------------


def test_regression_descale_rejects_multiple_scaling_methods():
    """Two scaling methods is a `ConfigError` (the v1 single-scaling guard)."""
    with pytest.raises(ConfigError):
        RegressionDescaleOp(
            stream=_STREAM_J,
            targets=["mHH"],
            norm_params={"mean": [0.0], "std": [1.0]},
            target_denominators=["mHH"],
        )


def test_regression_descale_rejects_denominator_count_mismatch():
    """Denominator count != target count is a `ConfigError`."""
    with pytest.raises(ConfigError):
        RegressionDescaleOp(stream=_STREAM_J, targets=["a", "b"], target_denominators=["d"])


def test_regression_descale_bind_rejects_missing_input_feature():
    """A denominator absent from the input Features fails the ONNX-source bind check."""
    op = RegressionDescaleOp(stream=_STREAM_J, targets=["m"], target_denominators=["mHH"])
    with pytest.raises(ConfigError):
        op.bind(("pt", "eta"))  # mHH not declared


def test_regression_descale_empty_targets_rejected():
    """Empty targets is a `ConfigError`."""
    with pytest.raises(ConfigError):
        RegressionDescaleOp(stream=_STREAM_J, targets=[])


# ---------------------------------------------------------------------------
# write-once: the op never mutates the source preds.* leaf
# ---------------------------------------------------------------------------


def test_descale_does_not_mutate_source_preds_leaf():
    """The de-scale clones — the bundle's float32 ``preds.*`` leaf is untouched."""
    norm = {"mean": [10.0], "std": [4.0]}
    preds = torch.ones(3, 1, dtype=torch.float32)
    b = Bundle({"preds": {_STREAM_J: {"t": preds}}})
    op = RegressionDescaleOp(stream=_STREAM_J, targets=["mHH"], norm_params=norm)
    _producer(op, stream=_STREAM_J, task="t").forward(b, Mode.TEST)
    # source leaf unchanged (still ones), not de-scaled in place
    torch.testing.assert_close(b.get(f"preds.{_STREAM_J}.t"), torch.ones(3, 1), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# demand-gating / width resolution carries over for the conversion ops
# (design §4 risk 6) — the gate (b) of P0, re-asserted per op
# ---------------------------------------------------------------------------


def _stub_source(pred_key, width, *, modes=Mode.ALL):
    """A minimal source module producing ``pred_key`` of last-dim `width`.

    Returns
    -------
    object
        A `GraphModule`-shaped stub.
    """

    class _Stub:
        name = "src"

        def declare_io(self, mode):
            del mode
            return IO(
                produces=unflatten_spec(
                    {pred_key: TensorSpec(shape=("B", width), dtype="float32", modes=modes)}
                )
            )

    return _Stub()


def test_class_probs_width_resolves_in_test_only_bind():
    """``ClassProbs`` preserves the class width in a TEST-only bind (design §4 risk 6)."""
    pred_key = f"preds.{_STREAM_J}.t"
    out_key = f"outputs.{_STREAM_J}.out"
    src = _stub_source(pred_key, 3)
    producer = _producer(ClassProbsOp(), stream=_STREAM_J, task="t")
    modules = {"src": src, "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 3  # softmax preserves the class dim


def test_seq_class_index_width_collapses_to_one():
    """``SeqClassIndex`` collapses the class dim to one index column at bind."""
    pred_key = f"preds.{_STREAM_T}.t"
    out_key = f"outputs.{_STREAM_T}.origin"
    src = _stub_source(pred_key, 8)
    # also need a source for the demanded pad mask
    mask_key = f"masks.{_STREAM_T}"

    class _MaskSrc:
        name = "msrc"

        def declare_io(self, mode):
            del mode
            return IO(
                produces=unflatten_spec(
                    {
                        mask_key: TensorSpec(
                            shape=("B", sym_dim("T", _STREAM_T)), dtype="bool", kind="pad_mask"
                        )
                    }
                )
            )

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassIndexOp())
    producer.name = "producer"
    modules = {"src": src, "msrc": _MaskSrc(), "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 1  # argmax collapses the class dim


def test_seq_class_probs_width_preserved_in_test_only_bind():
    """``SeqClassProbs`` preserves the class dim (C probs out, unlike argmax's collapse)."""
    pred_key = f"preds.{_STREAM_T}.t"
    out_key = f"outputs.{_STREAM_T}.origin"
    src = _stub_source(pred_key, 8)
    mask_key = f"masks.{_STREAM_T}"

    class _MaskSrc:
        name = "msrc"

        def declare_io(self, mode):
            del mode
            return IO(
                produces=unflatten_spec(
                    {
                        mask_key: TensorSpec(
                            shape=("B", sym_dim("T", _STREAM_T)), dtype="bool", kind="pad_mask"
                        )
                    }
                )
            )

    producer = TaskOutput(task="t", stream=_STREAM_T, name="origin", op=SeqClassProbsOp())
    producer.name = "producer"
    modules = {"src": src, "msrc": _MaskSrc(), "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 8  # softmax preserves the class dim


def test_regression_width_resolves_in_test_only_bind():
    """``Regression`` de-scale preserves the target width in a TEST-only bind."""
    pred_key = f"preds.{_STREAM_J}.t"
    out_key = f"outputs.{_STREAM_J}.out"
    src = _stub_source(pred_key, 2)
    op = RegressionDescaleOp(
        stream=_STREAM_J, targets=["mHH", "dR"], norm_params={"mean": [0, 0], "std": [1, 1]}
    )
    producer = _producer(op, stream=_STREAM_J, task="t")
    modules = {"src": src, "producer": producer}
    test = compile_plan(modules, Mode.TEST, sources={}, sinks=[out_key])
    schema = resolve_bind_schema([test])
    assert schema.width(out_key) == 2  # de-scale preserves the target dim


def test_identity_op_still_clones_p0_contract():
    """The default `ConversionOp` is still the P0 identity clone (no regression)."""
    preds = torch.randn(3, 4)
    b = Bundle({"preds": {_STREAM_J: {"t": preds}}})
    producer = TaskOutput(task="t", stream=_STREAM_J, name="out")  # default op
    assert isinstance(producer.op, ConversionOp)
    out = producer.forward(b, Mode.TEST)[f"outputs.{_STREAM_J}.out"]
    torch.testing.assert_close(out, preds, rtol=0, atol=0)
    assert out is not preds  # cloned, not aliased (write-once §2.1)
