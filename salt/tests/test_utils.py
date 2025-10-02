import tempfile
from pathlib import Path

import pytest
import torch
from ftag import get_mock_file

from salt.utils import clean_logs, compare_models, repair_ckpt
from salt.utils.inputs import inputs_concat
from salt.utils.scalers import RegressionTargetScaler


def test_compare_models():
    fname_A = get_mock_file()[0]
    fname_B = get_mock_file()[0]
    args = ["--file_a", fname_A, "--file_b", fname_B, "--tagger_a", "MockTagger"]
    compare_models.main(args)


# Fixture to create a temporary directory with a specified subdirectory
@pytest.fixture
def temp_directory_with_subdir(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir3 = tmp_path / "dir3"

    dir1.mkdir()
    dir2.mkdir()
    dir3.mkdir()

    subdir_path = dir2 / "specified_subdirectory"
    subdir_path.mkdir()

    return tmp_path


# Test delete_dirs_without_subdir function
def test_delete_dirs_without_subdir(temp_directory_with_subdir):
    # Only dir1 and dir3 should be kept after calling the function
    clean_logs.delete_dirs_without_subdir(temp_directory_with_subdir, "specified_subdirectory")

    # Check if the correct directories were deleted
    assert not (temp_directory_with_subdir / "dir1").exists()
    assert (temp_directory_with_subdir / "dir2").exists()
    assert not (temp_directory_with_subdir / "dir3").exists()


# Test the main function by capturing stdout
def test_clean_logs_main(temp_directory_with_subdir):
    print(temp_directory_with_subdir)
    args = [
        "--folder_path",
        str(temp_directory_with_subdir),
        "--subdirectory",
        "specified_subdirectory",
    ]
    clean_logs.main(args=args)


def test_scaler():
    dummy_scales = {
        "var1": {"op": "log", "x_scale": 2, "x_off": 1, "op_scale": 3, "op_off": 1},
        "var2": {"op": "exp", "x_scale": 2, "x_off": 1, "op_scale": 3, "op_off": 1},
        "var3": {"op": "linear", "x_scale": 2, "x_off": 1, "op_scale": 3, "op_off": 1},
    }
    scaler = RegressionTargetScaler(dummy_scales)
    values = torch.rand(500)
    for var in dummy_scales:
        scaled = scaler.scale(var, values)
        unscaled = scaler.inverse(var, scaled)
        assert torch.allclose(values, unscaled, atol=1e-7)


# test inputs_concat function
@pytest.mark.parametrize(
    ("n_batch", "n_track", "n_jet_feat", "n_track_feat"),
    [
        (10, 20, 5, 3),  # Test case 1
        (500, 15, 10, 4),  # Test case 2
    ],
)
def test_inputs_concat(n_batch, n_track, n_jet_feat, n_track_feat):
    # Call the inputs_concat function with the specified inputs
    inputs, mask = inputs_concat(n_batch, n_track, n_jet_feat, n_track_feat)
    assert inputs.shape == (n_batch, 40, n_jet_feat + n_track_feat)
    assert mask.shape == (n_batch, n_track)
    # ensuring mask created with at least on valid tracks
    assert not torch.any(mask[:, 0])


def test_repair_checkpoint(capsys):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Setup: Create a fake checkpoint file
        ckpt_path = Path(tmpdirname) / "test.ckpt"
        state_dict = {
            "_orig_mod.layer.weight": torch.randn(5, 5),
            "_orig_mod.layer.bias": torch.randn(5),
            "layer.activation": torch.randn(5),  # Entry without "_orig_mod"
        }
        torch.save({"state_dict": state_dict}, ckpt_path)

        # test the repair_checkpoint function
        repair_ckpt.repair_checkpoint(ckpt_path)
        repaired_ckpt = torch.load(ckpt_path)
        repaired_state_dict = repaired_ckpt["state_dict"]
        for key in repaired_state_dict:
            assert not key.startswith("_orig_mod."), "Found unmodified key in state_dict"
            assert "layer.activation" in repaired_state_dict, "Unmodified key was wrongly altered"
        assert Path(str(ckpt_path) + ".bak").exists(), "Backup file not found"

        # test the main function
        repair_ckpt.main([str(ckpt_path)])
        captured = capsys.readouterr()
        output = captured.out
        assert (
            "Repaired" in output or "No need to repair" in output
        ), "Unexpected output from main function"
        if "Repaired" in output:
            assert (
                "_orig_mod.layer.weight  ==>  layer.weight" in output
            ), "Expected key rename message missing"
        else:
            assert "No need to repair" in output, "Expected 'No need to repair' message missing"
