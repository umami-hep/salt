"""P1 unit gates for `RegressionDescaleOp` (split from test_producers.py along §3.4)."""

from __future__ import annotations

import pytest
import torch

from salt.core.graph import ConfigError, Mode
from salt.core.graph.bundle import Bundle
from salt.core.outputs import RegressionDescaleOp
from salt.tests.unit.outputs.conftest import (  # noqa: PLC2701 — shared split helpers
    _FLOAT_TOL,
    _STREAM_J,
    _STREAM_T,
    _bind_regression,
    _producer,
)


# GATE 3: Regression de-scale == RegressionTask.run_inference


def test_regression_norm_params_matches_task_run_inference():
    """Norm_params head: ``Regression`` op == ``RegressionTask.run_inference`` (tasks.py:536-9)."""
    torch.manual_seed(7)
    norm = {"mean": [3.0, -1.0], "std": [2.0, 0.5]}
    module = _bind_regression(_STREAM_J, ["mHH", "dR"], sequence=False, norm=norm)
    preds = torch.randn(9, 2)

    oracle = module.run_inference(preds.clone())  # v1 de-norm, no mask

    op = RegressionDescaleOp(stream=_STREAM_J, targets=["mHH", "dR"], norm_params=norm)
    got = _producer(op, stream=_STREAM_J, task="t").forward(
        Bundle({"preds": {_STREAM_J: {"t": preds.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_regression_scaler_matches_task_run_inference():
    """Functional scaler head: ``Regression`` op == the scaler branch of run_inference."""
    torch.manual_seed(8)
    scales = {"pt": {"op": "log", "x_scale": 5}, "Lxy": {"op": "linear", "x_scale": 2, "x_off": 1}}
    module = _bind_regression(_STREAM_T, ["pt", "Lxy"], sequence=True, scaler=scales)
    preds = torch.randn(4, 5, 2)
    mask = torch.zeros(4, 5, dtype=torch.bool)

    # v1 run_inference masks with NaN; the descale MATH is the scaler.inverse —
    # compare with an all-valid mask so no NaN fill enters (the NaN fill is a
    # sink-side serialisation concern, design §2 layer 2)
    oracle = module.run_inference(preds.clone(), pad_mask=mask)

    op = RegressionDescaleOp(stream=_STREAM_T, targets=["pt", "Lxy"], scaler=scales)
    got = _producer(op, stream=_STREAM_T, task="t").forward(
        Bundle({"preds": {_STREAM_T: {"t": preds.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_T}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_regression_ratio_denominator_test_mode_matches_task():
    """Ratio-denominator head (TEST): ``Regression`` op == v1 de-scale via labels."""
    torch.manual_seed(9)
    module = _bind_regression(
        _STREAM_J, ["m_over_mHH"], sequence=False, denoms=["mHH"], fields=("mHH", "pt")
    )
    preds = torch.randn(6, 1)
    denom = torch.rand(6) + 0.5

    # v1 run_inference(labels={stream: {denom: tensor}}) (tasks.py:533-535)
    oracle = module.run_inference(preds.clone(), labels={_STREAM_J: {"mHH": denom}})

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
    """ratio-denominator head (ONNX): the denominator is gathered by NAME from inputs.<stream>."""
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


# W34.3 critic-fix GATE: the gaussian + sequence producer branches == the task's
# run_inference oracle (the branches test_producers.py previously left untested).
# After the per-token axis fix (run_inference now indexes [..., i]), the producer
# (already last-axis) and the task path agree on global AND per-token heads.


def _gaussian_concat(means_stds: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Concat the task's ``(means, stds)`` into the producer's one ``[..., 2R]`` array."""
    means, stds = means_stds
    return torch.cat([means, stds], dim=-1)


def test_regression_gaussian_global_producer_matches_task_run_inference():
    """Global gaussian head: ``RegressionDescaleOp(gaussian=True)`` == means‖stds oracle."""
    torch.manual_seed(20)
    norm = {"mean": [2.0], "std": [3.0]}
    module = _bind_regression(_STREAM_J, ["mHH"], sequence=False, gaussian=True, norm=norm)
    preds = torch.randn(8, 2)  # [B, 2R] = mean ‖ raw-var

    oracle = _gaussian_concat(module.run_inference(preds.clone()))

    op = RegressionDescaleOp(stream=_STREAM_J, targets=["mHH"], norm_params=norm, gaussian=True)
    got = _producer(op, stream=_STREAM_J, task="t").forward(
        Bundle({"preds": {_STREAM_J: {"t": preds.clone()}}}), Mode.TEST
    )[f"outputs.{_STREAM_J}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL)


def test_regression_gaussian_per_token_producer_matches_task_run_inference():
    """Per-token gaussian seq head: producer == task on EVERY token (W34.3 axis fix)."""
    torch.manual_seed(21)
    norm = {"mean": [1.0], "std": [1.0]}
    module = _bind_regression(_STREAM_T, ["dphi"], sequence=True, gaussian=True, norm=norm)
    preds = torch.randn(3, 4, 2)  # [B, L, 2R]
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[0, 3] = True  # a real padded position -> NaN-filled means + stds

    oracle = _gaussian_concat(module.run_inference(preds.clone(), pad_mask=mask))

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
    """Per-token scaled seq head: producer ``_nan_fill`` == task run_inference NaN-fill."""
    torch.manual_seed(22)
    norm = {"mean": [10.0, 20.0], "std": [2.0, 0.5]}
    module = _bind_regression(_STREAM_T, ["a", "b"], sequence=True, norm=norm)
    preds = torch.randn(2, 5, 2)  # [B, L, R]
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 4] = True
    mask[1, 3] = True
    mask[1, 4] = True  # real padded rows on both jets

    oracle = module.run_inference(preds.clone(), pad_mask=mask)

    op = RegressionDescaleOp(stream=_STREAM_T, targets=["a", "b"], norm_params=norm, sequence=True)
    got = _producer(op, stream=_STREAM_T, task="t").forward(
        Bundle({"preds": {_STREAM_T: {"t": preds.clone()}}, "masks": {_STREAM_T: mask}}), Mode.TEST
    )[f"outputs.{_STREAM_T}.out"]

    torch.testing.assert_close(got, oracle, rtol=0, atol=_FLOAT_TOL, equal_nan=True)
    # the de-scale reached EVERY token's columns (not just tokens 0,1 — the pre-fix
    # first-axis bug left later tokens raw); token 2 of jet 0 is a valid scaled row
    assert torch.isfinite(got[0, 2]).all()
    assert torch.isnan(got[0, 4]).all()


# producer config-error guards (mirror the task module's guards)


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


# write-once: the op never mutates the source preds.* leaf


def test_descale_does_not_mutate_source_preds_leaf():
    """The de-scale clones — the bundle's float32 ``preds.*`` leaf is untouched."""
    norm = {"mean": [10.0], "std": [4.0]}
    preds = torch.ones(3, 1, dtype=torch.float32)
    b = Bundle({"preds": {_STREAM_J: {"t": preds}}})
    op = RegressionDescaleOp(stream=_STREAM_J, targets=["mHH"], norm_params=norm)
    _producer(op, stream=_STREAM_J, task="t").forward(b, Mode.TEST)
    # source leaf unchanged (still ones), not de-scaled in place
    torch.testing.assert_close(b.get(f"preds.{_STREAM_J}.t"), torch.ones(3, 1), rtol=0, atol=0)
