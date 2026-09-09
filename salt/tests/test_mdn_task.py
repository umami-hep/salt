"""Unit tests for salt.models.mdn_task.MixtureDensityTask.run_inference.

These unit tests evaluate 'run_inference' in isolation:
dominant-mode selection, the two de-scaling paths
('norm_params' and 'target_denominators'), NaN padding,
and the error raised when no scaling is configured..
"""

import numpy as np
import pytest
import torch
from torch.nn.functional import softplus

from salt.models.mdn_task import MixtureDensityTask

EPS = 1e-6


def make_task(n_components: int = 3, **kwargs) -> MixtureDensityTask:
    """Build a minimal task; only 'run_inference' is exercised."""
    return MixtureDensityTask(
        name="pvz_mdn",
        input_name="jets",
        dense_config={
            "input_size": 8,
            "output_size": 3 * n_components,
            "hidden_layers": [8],
            "activation": "ReLU",
        },
        targets="TruthJetPVz",
        n_components=n_components,
        **kwargs,
    )


# preds layout is [means (K) | raw_vars (K) | logits (K)]; here K = 3, B = 2.
# Row 0 dominant component -> index 1 (largest logit); row 1 -> index 2.
PREDS = torch.tensor(
    [
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
        [10.0, 20.0, 30.0, -2.0, -2.0, 2.0, 1.0, 0.0, 9.0],
    ]
)
DOM_IDX = [1, 2]
NORM = {"mean": 5.0, "std": 2.0}


def _expected(preds: torch.Tensor, dom_idx: list[int], descale_mean, descale_std) -> torch.Tensor:
    """Recompute (mode_mean, mode_std, mode_weight) via an independent path."""
    means, raw_vars, logits = preds.tensor_split(3, -1)
    stds = torch.sqrt(softplus(raw_vars) + EPS)
    weights = torch.softmax(logits, dim=-1)
    rows = []
    for b, k in enumerate(dom_idx):
        rows.append(
            [descale_mean(means[b, k]), descale_std(stds[b, k]), float(weights[b, k])]
        )
    return torch.tensor(rows)


def test_run_inference_selects_dominant_mode_and_descales_with_norm_params():
    task = make_task(norm_params=dict(NORM))
    out = task.run_inference(PREDS)

    assert out.shape == (2, 3)
    expected = _expected(
        PREDS,
        DOM_IDX,
        descale_mean=lambda m: float(m) * NORM["std"] + NORM["mean"],
        descale_std=lambda s: float(s) * NORM["std"],
    )
    torch.testing.assert_close(out, expected)
    # de-scaled means are exact and independent of the softplus/softmax formulas:
    torch.testing.assert_close(out[:, 0], torch.tensor([9.0, 65.0]))
    # a standard deviation is strictly positive.
    assert (out[:, 1] > 0).all()


def test_run_inference_picks_largest_weight_not_largest_mean():
    # dominant weight sits on the SMALLEST mean -> guards against argmax-on-mean bugs.
    preds = torch.tensor([[9.0, 1.0, 5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]])
    out = make_task(norm_params=dict(NORM)).run_inference(preds)
    # component 0 wins (logit 7); mean 9.0 -> 9*2 + 5 = 23.0
    torch.testing.assert_close(out[0, 0], torch.tensor(23.0))


def test_run_inference_single_component_weight_is_one():
    # K = 1: the only component is always dominant and softmax weight == 1.
    preds = torch.tensor([[4.0, 0.5, 0.0]])
    out = make_task(n_components=1, norm_params=dict(NORM)).run_inference(preds)
    assert out.shape == (1, 3)
    torch.testing.assert_close(out[0, 0], torch.tensor(4.0 * 2.0 + 5.0))  # 13.0
    torch.testing.assert_close(out[0, 2], torch.tensor(1.0))


def test_run_inference_target_denominator_descaling():
    dens = [3.0, 4.0]  # per-jet denominators; this path scales, with no additive offset
    task = make_task(target_denominators=["den"])
    labels = {"jets": {"den": torch.tensor(dens)}}
    out = task.run_inference(PREDS, labels=labels)

    means, raw_vars, logits = PREDS.tensor_split(3, -1)
    stds = torch.sqrt(softplus(raw_vars) + EPS)
    weights = torch.softmax(logits, dim=-1)
    expected = torch.tensor(
        [
            [float(means[b, k]) * d, float(stds[b, k]) * d, float(weights[b, k])]
            for b, (k, d) in enumerate(zip(DOM_IDX, dens, strict=True))
        ]
    )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out[:, 0], torch.tensor([6.0, 120.0]))


def test_run_inference_pad_mask_nan_fills_padded_rows():
    task = make_task(norm_params=dict(NORM))
    pad_mask = torch.tensor([False, True])
    out = task.run_inference(PREDS, pad_mask=pad_mask)

    assert torch.isnan(out[1]).all()  # padded row blanked
    assert torch.isfinite(out[0]).all()  # kept row untouched
    torch.testing.assert_close(out[0, 0], torch.tensor(9.0))


def test_run_inference_requires_scaling_params():
    # no norm_params and no target_denominators -> cannot de-scale.
    task = make_task()
    with pytest.raises(ValueError, match="requires scaling parameters"):
        task.run_inference(PREDS)


def test_run_inference_is_float_and_backend_agnostic_dtype():
    # a float64 input is cast to float32 internally (preds.float()).
    task = make_task(norm_params=dict(NORM))
    out = task.run_inference(PREDS.double())
    assert out.dtype == torch.float32
    assert not np.isnan(out.numpy()).any()
