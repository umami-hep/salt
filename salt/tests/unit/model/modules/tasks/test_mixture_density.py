"""Unit tests for MixtureDensityTaskModule."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn.functional import softplus

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.model.bind import ResolvedSchema
from salt.model.modules.tasks import MixtureDensityTaskModule

EPS = 1e-6
NORM = {"mean": 5.0, "std": 2.0}
# preds layout is [means (K) | raw_vars (K) | logits (K)]; here K = 3, B = 2.
# Row 0 dominant component -> index 1 (largest logit); row 1 -> index 2.
PREDS = torch.tensor([
    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
    [10.0, 20.0, 30.0, -2.0, -2.0, 2.0, 1.0, 0.0, 9.0],
])
DOM_IDX = [1, 2]
FIELDS = {"inputs.jets": ("pt", "den")}


def make_task(n_components: int = 3, **kwargs) -> MixtureDensityTaskModule:
    """A global head on ``pooled.global``, bound to width 8."""
    kwargs.setdefault("write_targets", False)
    task = MixtureDensityTaskModule(
        stream="jets",
        targets="TruthJetPVz",
        n_components=n_components,
        input="pooled.global",
        **kwargs,
    )
    task.name = "pvz_mdn"
    task.bind(ResolvedSchema(widths={"pooled.global": 8, "inputs.jets": 2}, fields=FIELDS))
    return task


def _outputs(task: MixtureDensityTaskModule, preds: torch.Tensor, mode: Mode = Mode.TEST, **keys):
    b = Bundle()
    b.set(task.pred_key, preds)
    for key, value in keys.items():
        b.set(key, value)
    return torch.stack([f.value for f in task.get_output(b, mode, "run")], dim=-1)


def _expected(preds: torch.Tensor, dom_idx: list[int], scale, offset) -> torch.Tensor:
    """The H5 columns per row, one component at a time: dominant mean, sigma and weight,
    mixture mean, sqrt(Var[y]), sigma_fair, then every mean, sigma and weight.
    """
    means, raw_vars, logits = preds.tensor_split(3, -1)
    stds = torch.sqrt(softplus(raw_vars) + EPS)
    weights = torch.softmax(logits, dim=-1)
    rows = []
    for b, k in enumerate(dom_idx):
        s = float(scale[b]) if isinstance(scale, list) else scale
        mu = [float(m) * s + offset for m in means[b]]
        sg = [float(x) * s for x in stds[b]]
        pi = [float(w) for w in weights[b]]
        mu_bar = sum(p * m for p, m in zip(pi, mu, strict=True))
        var = sum(p * (g**2 + (m - mu_bar) ** 2) for p, m, g in zip(pi, mu, sg, strict=True))
        sig_fair = math.sqrt(var + (mu_bar - mu[k]) ** 2)
        rows.append([mu[k], sg[k], pi[k], mu_bar, math.sqrt(var), sig_fair, *mu, *sg, *pi])
    return torch.tensor(rows)


ONNX_COLUMNS = [0, 5, 2]  # dominant mean, sigma_fair, dominant weight


def test_selects_dominant_mode_and_descales_with_norm_params():
    out = _outputs(make_task(norm_params=NORM), PREDS)
    assert out.shape == (2, 6 + 3 * 3)
    torch.testing.assert_close(out, _expected(PREDS, DOM_IDX, NORM["std"], NORM["mean"]))
    # de-scaled means are exact and independent of the softplus/softmax formulas
    torch.testing.assert_close(out[:, 0], torch.tensor([9.0, 65.0]))
    torch.testing.assert_close(out[:, 6:9], PREDS[:, :3] * 2.0 + 5.0)
    assert (out[:, 1] > 0).all()


def test_picks_largest_weight_not_largest_mean():
    # the dominant weight sits on the smallest mean: guards against argmax-on-mean bugs
    preds = torch.tensor([[9.0, 1.0, 5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]])
    out = _outputs(make_task(norm_params=NORM), preds)
    torch.testing.assert_close(out[0, 0], torch.tensor(23.0))  # 9 * 2 + 5


def test_h5_stddev_is_dominant_sigma_and_onnx_stddev_is_sigma_fair():
    """sigma_fair exceeds the dominant component's own sigma when rival modes exist."""
    task = make_task(norm_params=NORM)
    h5 = _outputs(task, PREDS)
    _, raw_vars, _ = PREDS.tensor_split(3, -1)
    dom_sigma = torch.sqrt(softplus(raw_vars) + EPS)[torch.arange(2), DOM_IDX] * NORM["std"]
    torch.testing.assert_close(h5[:, 1], dom_sigma)
    assert (h5[:, 5] > h5[:, 1]).all()
    onnx = _outputs(task, PREDS, Mode.ONNX)
    torch.testing.assert_close(onnx, h5[:, ONNX_COLUMNS])


def test_single_component_weight_is_one_and_stddev_is_its_sigma():
    preds = torch.tensor([[4.0, 0.5, 0.0]])
    task = make_task(n_components=1, norm_params=NORM)
    out = _outputs(task, preds)
    sigma = torch.sqrt(softplus(torch.tensor(0.5)) + EPS) * 2.0
    torch.testing.assert_close(out[0, 0], torch.tensor(13.0))  # 4 * 2 + 5
    torch.testing.assert_close(out[0, 1], sigma)
    torch.testing.assert_close(out[0, 2], torch.tensor(1.0))
    torch.testing.assert_close(out[0, 5], sigma)  # no rival mode: sigma_fair is the sigma
    torch.testing.assert_close(_outputs(task, preds, Mode.ONNX)[1], sigma)  # B=1 squeezed


def test_target_denominator_descaling_test_and_onnx():
    """TEST reads the denominator from labels, ONNX from the input Feature by name."""
    dens = [3.0, 4.0]  # this path scales, with no additive offset
    task = make_task(target_denominators="den")
    expected = _expected(PREDS, DOM_IDX, dens, 0.0)
    test = _outputs(task, PREDS, Mode.TEST, **{"labels.jets.den": torch.tensor(dens)})
    torch.testing.assert_close(test, expected)
    torch.testing.assert_close(test[:, 0], torch.tensor([6.0, 120.0]))
    inputs = torch.tensor([[0.0, dens[0]], [0.0, dens[1]]])
    onnx = _outputs(task, PREDS, Mode.ONNX, **{"inputs.jets": inputs})
    torch.testing.assert_close(onnx, expected[:, ONNX_COLUMNS])


def test_pad_mask_nan_fills_padded_tokens():
    task = MixtureDensityTaskModule(
        stream="tracks", targets="dphi", n_components=3, norm_params=NORM, write_targets=False
    )
    task.name = "dphi_mdn"
    task.bind(ResolvedSchema(widths={"encoded.tracks": 8}))
    mask = torch.tensor([[False, True]])
    out = _outputs(task, PREDS[None], **{"masks.tracks": mask})
    assert torch.isnan(out[0, 1]).all()
    assert torch.isfinite(out[0, 0]).all()
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(9.0))


def test_requires_scaling_params():
    with pytest.raises(ConfigError, match="norm_params or target_denominators"):
        MixtureDensityTaskModule(stream="jets", targets="x", n_components=2, input="pooled.global")


def test_output_is_float32_for_float64_preds():
    out = _outputs(make_task(norm_params=NORM), PREDS.double())
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_output_names_and_target_column():
    task = make_task(norm_params=NORM, custom_output_names="pvz", write_targets=True)
    test = [f.h5_name for f in task.get_output_manifest(Mode.TEST, "run")]
    mixture = [f"pvz_{p}{k}" for p in ("mu", "sigma", "pi") for k in range(3)]
    assert test == [
        "pvz",
        "pvz_stddev",
        "pvz_modeweight",
        "pvz_mubar",
        "pvz_sigmix",
        "pvz_sigfair",
        *mixture,
        "target_pvz_mdn_TruthJetPVz",
    ]
    onnx = [f.resolved_onnx_name for f in task.get_output_manifest(Mode.ONNX, "run")]
    assert onnx == ["pvz", "pvz_stddev", "pvz_modeweight"]


def test_head_width_is_three_per_component():
    task = make_task(n_components=4, norm_params=NORM)
    assert task.net.output_size == 12
    pred = flatten_spec(task.declare_io(Mode.TEST).produces)[task.pred_key]
    assert pred.shape == ("B", 12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"targets": ["a", "b"]}, "exactly one target"),
        ({"n_components": 0}, "n_components must be >= 1"),
    ],
)
def test_config_errors(kwargs, match):
    args = {"stream": "jets", "targets": "x", "n_components": 2, "norm_params": NORM} | kwargs
    with pytest.raises(ConfigError, match=match):
        MixtureDensityTaskModule(input="pooled.global", **args)


def test_fit_loss_is_weighted_mixture_nll_without_padded_tokens():
    """Padded tokens drop out of the mean; the loss is in norm_params-scaled space."""
    task = MixtureDensityTaskModule(
        stream="tracks", targets="dphi", n_components=3, norm_params=NORM, weight=0.5
    )
    task.name = "dphi_mdn"
    task.bind(ResolvedSchema(widths={"encoded.tracks": 8}))
    x, y = torch.randn(2, 3, 8), 4.0 + torch.rand(2, 3)
    mask = torch.tensor([[False, False, True], [False, True, True]])
    b = Bundle()
    b.set("encoded.tracks", x)
    b.set("masks.tracks", mask)
    b.set("labels.tracks.dphi", y)
    loss = task.forward(b, Mode.FIT)[task.loss_key]
    nll = task.loss(*task.net(x, None).tensor_split(3, -1), ((y - 5.0) / 2.0)[..., None])
    torch.testing.assert_close(loss, 0.5 * nll[~mask].mean())


def test_fit_loss_sample_weight():
    task = make_task(norm_params=NORM, sample_weight="w")
    x, y, w = torch.randn(3, 8), torch.tensor([7.0, 3.0, 11.0]), torch.tensor([1.0, 0.0, 2.0])
    b = Bundle()
    b.set("pooled.global", x)
    b.set("labels.jets.TruthJetPVz", y)
    b.set("labels.jets.w", w)
    loss = task.forward(b, Mode.FIT)[task.loss_key]
    nll = task.loss(*task.net(x, None).tensor_split(3, -1), ((y - 5.0) / 2.0)[:, None])
    torch.testing.assert_close(loss, (nll * w[:, None]).mean())
