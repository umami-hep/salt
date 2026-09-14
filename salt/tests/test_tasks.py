import numpy as np
import pytest
import torch
import yaml
from torch import nn

from salt.models.inputnorm import InputNorm
from salt.models.task import (
    ClassificationTask,
    GaussianRegressionTask,
    RegressionTask,
    VertexingTask,
    mask_fill_flattened,
)
from salt.utils.inputs import get_random_mask
from salt.utils.scalers import RegressionTargetScaler

B, L, D = 5, 4, 8


def dense_config(output_size, input_size=D):
    return {"input_size": input_size, "output_size": output_size, "hidden_layers": [8]}


def make_class_task(**kwargs):
    defaults = {
        "name": "jets_classification",
        "input_name": "jets",
        "label": "flavour_label",
        "class_names": ["sig", "bkg"],
        "loss": nn.CrossEntropyLoss(),
        "dense_config": dense_config(2),
    }
    defaults.update(kwargs)
    return ClassificationTask(**defaults)


def test_classification_global_forward():
    torch.manual_seed(0)
    task = make_class_task()
    x = torch.randn(B, D)
    labels_dict = {"jets": {"flavour_label": torch.randint(2, (B,))}}
    preds, loss = task(x, labels_dict)
    assert preds.shape == (B, 2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_classification_forward_no_labels_gives_no_loss():
    task = make_class_task()
    preds, loss = task(torch.randn(B, D), {})
    assert preds.shape == (B, 2)
    assert loss is None


def test_classification_track_forward_masks_padded_labels():
    torch.manual_seed(0)
    task = make_class_task(input_name="tracks", dense_config=dense_config(2))
    x = torch.randn(B, L, D)
    pad_masks = {"tracks": get_random_mask(B, L)}
    labels_dict = {"tracks": {"flavour_label": torch.randint(2, (B, L))}}
    preds, loss = task(x, labels_dict, pad_masks=pad_masks)
    assert preds.shape == (B, L, 2)
    assert torch.isfinite(loss)


def test_classification_label_map_remaps_labels():
    torch.manual_seed(0)
    task = make_class_task(
        class_names=["a", "b", "c"],
        dense_config=dense_config(3),
        label_map={0: 0, 4: 1, 5: 2},
    )
    x = torch.randn(B, D)
    labels_dict = {"jets": {"flavour_label": torch.tensor([0, 4, 5, 4, 0])}}
    _, loss = task(x, labels_dict)
    assert torch.isfinite(loss)


def test_classification_label_map_requires_class_names():
    with pytest.raises(ValueError, match="class names"):
        make_class_task(class_names=None, label_map={0: 0})


def test_classification_class_count_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        make_class_task(class_names=["a", "b", "c"])


def test_classification_infers_class_names():
    task = make_class_task(
        label="ftagTruthTypeLabel", class_names=None, dense_config=dense_config(6)
    )
    assert len(task.class_names) == 6


def test_classification_sample_weight():
    torch.manual_seed(0)
    weights = torch.rand(B)
    labels = torch.randint(2, (B,))
    x = torch.randn(B, D)
    task = make_class_task(loss=nn.CrossEntropyLoss(reduction="none"), sample_weight="weight")
    torch.manual_seed(1)
    preds, loss = task(x, {"jets": {"flavour_label": labels, "weight": weights}})
    expected = (nn.functional.cross_entropy(preds, labels, reduction="none") * weights).mean()
    torch.testing.assert_close(loss, expected)


def test_classification_run_inference_global():
    task = make_class_task()
    probs = task.run_inference(torch.randn(B, 2))
    torch.testing.assert_close(probs.sum(-1), torch.ones(B))


def test_classification_run_inference_tracks_zeros_padded():
    task = make_class_task(input_name="tracks")
    pad_mask = get_random_mask(B, L)
    probs = task.run_inference(torch.randn(B, L, 2), pad_mask)
    assert torch.all(probs[pad_mask] == 0)
    valid = ~pad_mask
    torch.testing.assert_close(probs.sum(-1)[valid], torch.ones(int(valid.sum())))


def test_classification_get_h5_field_names():
    task = make_class_task()
    task.model_name = "test"
    out = task.get_h5(torch.randn(B, 2))
    assert out.dtype.names == ("test_psig", "test_pbkg")
    assert out.shape == (B,)


def make_regression_task(targets="pt", output_size=1, **kwargs):
    defaults = {
        "name": "jet_regression",
        "input_name": "jets",
        "targets": targets,
        "loss": nn.MSELoss(),
        "dense_config": dense_config(output_size),
    }
    defaults.update(kwargs)
    return RegressionTask(**defaults)


def test_regression_forward():
    torch.manual_seed(0)
    task = make_regression_task()
    x = torch.randn(B, D)
    targets_dict = {"jets": {"pt": torch.rand(B)}}
    preds, loss = task(x, targets_dict)
    assert preds.shape == (B, 1)
    assert torch.isfinite(loss)


def test_regression_output_target_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        make_regression_task(output_size=2)


def test_regression_norm_params_round_trip():
    task = make_regression_task(norm_params={"mean": 5.0, "std": 2.0})
    raw = torch.rand(B) + 1
    scaled = task.get_targets({"jets": {"pt": raw}})
    torch.testing.assert_close(scaled[:, 0], (raw - 5.0) / 2.0)
    recovered = task.run_inference(scaled.clone())
    torch.testing.assert_close(recovered[:, 0], raw)


def test_regression_scaler_validates_at_init():
    scales = {"pt": {"op": "linear", "x_scale": 2, "x_off": 1, "op_scale": 3, "op_off": 1}}
    task = make_regression_task(scaler=RegressionTargetScaler(scales))
    assert task.scaler is not None


def test_regression_single_scaling_method_enforced():
    with pytest.raises(ValueError, match="single scaling method"):
        make_regression_task(
            norm_params={"mean": 5.0, "std": 2.0},
            target_denominators="pt",
        )


def test_regression_nan_loss_ignores_nan_targets():
    torch.manual_seed(0)
    task = make_regression_task(loss=nn.MSELoss(reduction="none"))
    preds = torch.rand(B, 1)
    targets = torch.rand(B, 1)
    targets[0, 0] = torch.nan
    loss = task.nan_loss(preds, targets, {})
    assert torch.isfinite(loss)


def test_regression_multi_target_output_names():
    task = make_regression_task(targets=["pt", "mass"], output_size=2)
    task.model_name = "test"
    assert task.output_names == ["test_pt", "test_mass"]
    task.custom_output_names = ["a", "b"]
    assert task.output_names == ["test_a", "test_b"]


def make_gaussian_task(**kwargs):
    defaults = {
        "name": "jet_regression",
        "input_name": "jets",
        "targets": "pt",
        "loss": nn.GaussianNLLLoss(),
        "dense_config": dense_config(2),
        "norm_params": {"mean": 5.0, "std": 2.0},
    }
    defaults.update(kwargs)
    return GaussianRegressionTask(**defaults)


def test_gaussian_regression_forward():
    torch.manual_seed(0)
    task = make_gaussian_task()
    x = torch.randn(B, D)
    preds, loss = task(x, {"jets": {"pt": torch.rand(B)}})
    assert preds.shape == (B, 2)
    assert torch.isfinite(loss)


def test_gaussian_regression_output_size_must_be_twice_targets():
    with pytest.raises(ValueError, match="twice"):
        make_gaussian_task(dense_config=dense_config(3))


def test_gaussian_regression_run_inference():
    task = make_gaussian_task()
    means, stds = task.run_inference(torch.randn(B, 2))
    assert means.shape == (B, 1)
    assert stds.shape == (B, 1)
    assert torch.all(stds >= 0)


def test_gaussian_regression_run_inference_requires_scaling():
    task = make_gaussian_task()
    task.norm_params = None
    with pytest.raises(ValueError, match="requires scaling"):
        task.run_inference(torch.randn(B, 2))


def test_vertexing_forward_and_loss():
    torch.manual_seed(0)
    task = VertexingTask(
        name="track_vertexing",
        input_name="tracks",
        label="VertexIndex",
        loss=nn.BCEWithLogitsLoss(reduction="none"),
        dense_config=dense_config(1, input_size=2 * D),
    )
    x = torch.randn(B, L, D)
    pad_masks = {"tracks": get_random_mask(B, L)}
    labels_dict = {
        "tracks": {
            "VertexIndex": torch.randint(-1, 3, (B, L)),
            "OriginLabel": torch.randint(0, 6, (B, L)),
        }
    }
    preds, loss = task(x, labels_dict, pad_masks=pad_masks)
    assert preds.ndim == 2
    assert preds.shape[1] == 1
    assert torch.isfinite(loss)


def test_mask_fill_flattened():
    mask = torch.tensor([[False, False, True], [False, True, True]])
    flat = torch.arange(3, dtype=torch.float32).unsqueeze(-1)
    out = mask_fill_flattened(flat, mask)
    assert out.shape == (2, 3, 1)
    torch.testing.assert_close(out[0, :2, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(out[1, 0, 0], torch.tensor(2.0))
    assert torch.isinf(out[0, 2, 0])
    assert torch.isinf(out[1, 1, 0])


def test_input_name_mask_two_streams():
    task = make_class_task(input_name="electrons")
    pad_masks = {
        "tracks": torch.zeros(B, 3, dtype=torch.bool),
        "electrons": torch.zeros(B, 2, dtype=torch.bool),
    }
    mask = task.input_name_mask(pad_masks)
    assert mask.tolist() == [False, False, False, True, True]


def test_input_norm(tmp_path):
    norm_dict = {
        "jets": {"pt": {"mean": 2.0, "std": 4.0}, "eta": {"mean": 0.0, "std": 1.0}},
        "tracks": {"d0": {"mean": 1.0, "std": 2.0}},
    }
    nd_path = tmp_path / "norm.yaml"
    nd_path.write_text(yaml.dump(norm_dict))
    variables = {"jets": ["pt", "eta"], "tracks": ["d0"]}
    norm = InputNorm(nd_path, variables, "jets", {k: k for k in variables})

    inputs = {"jets": torch.ones(B, 2), "tracks": torch.ones(B, L, 1)}
    out = norm(dict(inputs))
    expected_jets = torch.stack([
        torch.full((B,), (1.0 - 2.0) / 4.0),
        torch.full((B,), 1.0),
    ], dim=1)
    torch.testing.assert_close(out["jets"], expected_jets)
    torch.testing.assert_close(out["tracks"], torch.zeros(B, L, 1))


def test_input_norm_missing_variable_raises(tmp_path):
    norm_dict = {"jets": {"pt": {"mean": 0.0, "std": 1.0}}}
    nd_path = tmp_path / "norm.yaml"
    nd_path.write_text(yaml.dump(norm_dict))
    with pytest.raises(ValueError, match="Missing variables"):
        InputNorm(nd_path, {"jets": ["pt", "eta"]}, "jets", {"jets": "jets"})


def test_input_norm_zero_std_raises(tmp_path):
    norm_dict = {"jets": {"pt": {"mean": 0.0, "std": 0.0}}}
    nd_path = tmp_path / "norm.yaml"
    nd_path.write_text(yaml.dump(norm_dict))
    with pytest.raises(ValueError, match="Zero standard deviation"):
        InputNorm(nd_path, {"jets": ["pt"]}, "jets", {"jets": "jets"})


def test_input_norm_clamp(tmp_path):
    norm_dict = {"jets": {"pt": {"mean": 0.0, "std": 1.0}, "eta": {"mean": 0.0, "std": 1.0}}}
    nd_path = tmp_path / "norm.yaml"
    nd_path.write_text(yaml.dump(norm_dict))
    norm = InputNorm(nd_path, {"jets": ["pt", "eta"]}, "jets", {"jets": "jets"}, clamp=5.0)
    out = norm({"jets": torch.tensor([[1e6, -1e6], [-7.0, 1.0]])})
    torch.testing.assert_close(out["jets"], torch.tensor([[5.0, -5.0], [-5.0, 1.0]]))


@pytest.mark.parametrize("clamp", [0.0, -1.0])
def test_input_norm_clamp_must_be_positive(tmp_path, clamp):
    norm_dict = {"jets": {"pt": {"mean": 0.0, "std": 1.0}}}
    nd_path = tmp_path / "norm.yaml"
    nd_path.write_text(yaml.dump(norm_dict))
    with pytest.raises(ValueError, match="clamp must be positive"):
        InputNorm(nd_path, {"jets": ["pt"]}, "jets", {"jets": "jets"}, clamp=clamp)


def test_regression_get_h5_uses_output_names():
    task = make_regression_task(norm_params={"mean": 0.0, "std": 1.0})
    task.model_name = "test"
    out = task.get_h5(torch.rand(B, 1), {})
    assert out.dtype.names == ("test_pt",)
    assert out.shape == (B,)
    assert np.isfinite(out["test_pt"]).all()
