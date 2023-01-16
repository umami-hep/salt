import sys
from pathlib import Path

import pytest

from salt.main import main
from salt.to_onnx import main as to_onnx
from salt.utils.get_onnx_metadata import main as get_onnx_metadata
from salt.utils.inputs import (
    write_dummy_scale_dict,
    write_dummy_test_file,
    write_dummy_train_file,
)

w = "ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning:"

N_JET_FEATURES = 2
N_TRACK_FEATURES = 21


def run_train(tmp_path, config_path, train_args):
    tmp_path = Path(tmp_path)
    train_h5_path = tmp_path / "dummy_train_inputs.h5"
    sd_path = tmp_path / "dummy_scale_dict.json"
    write_dummy_scale_dict(sd_path, N_JET_FEATURES, N_TRACK_FEATURES)
    write_dummy_train_file(train_h5_path, sd_path)

    args = ["fit"]
    args += [f"--config={config_path}"]
    args += [f"--data.scale_dict={sd_path}"]
    args += [f"--data.train_file={train_h5_path}"]
    args += [f"--data.val_file={train_h5_path}"]
    args += ["--data.batched_read=True"]
    args += ["--data.num_jets_train=1000"]
    args += ["--data.num_jets_val=1000"]
    args += ["--data.batch_size=100"]
    args += ["--data.num_workers=0"]
    args += ["--trainer.max_epochs=2"]
    args += ["--trainer.accelerator=cpu"]
    args += [f"--trainer.default_root_dir={tmp_path}"]
    args += ["--trainer.logger.offline=True"]
    if train_args:
        args += train_args

    main(args)


def run_eval(tmp_path, train_config_path, sd_path):
    test_h5_path = Path(tmp_path) / "dummy_test_inputs.h5"
    write_dummy_test_file(test_h5_path, sd_path)

    args = ["test"]
    args += [f"--config={train_config_path}"]
    args += [f"--data.test_file={test_h5_path}"]
    args += ["--data.num_jets_test=1000"]
    main(args)


def run_onnx(train_dir, sd_path):
    ckpt_path = list((train_dir / "ckpts").iterdir())[-1]
    args = [f"--config={train_dir / 'config.yaml'}"]
    args += [f"--ckpt_path={ckpt_path}"]
    args += ["--track_selection=dipsLoose202102"]
    args += [f"--sd_path={sd_path}"]
    to_onnx(args)

    args = [str(train_dir / "model.onnx")]
    get_onnx_metadata(args)


def run_combined(tmp_path, config_path, do_eval=True, do_onnx=True, train_args=None):
    sys.argv = [sys.argv[0]]  # ignore pytest cli args when running salt cli

    # run training
    run_train(tmp_path, config_path, train_args)

    if do_eval:
        train_dir = [x for x in tmp_path.iterdir() if x.is_dir() and (x / "config.yaml").exists()]
        assert len(train_dir) == 1
        train_dir = train_dir[0]
        print(f"Using train_dir {train_dir}.")
        train_config_path = train_dir / "config.yaml"
        sd_path = [x for x in train_dir.iterdir() if x.suffix == ".json"][0]
        run_eval(tmp_path, train_config_path, sd_path)
    if do_onnx:
        run_onnx(train_dir, sd_path)


@pytest.mark.filterwarnings(w)
class TestTrainMisc:
    config_path = "configs/GN1.yaml"

    def test_train_batched(self, tmp_path) -> None:
        args = ["--data.batched_read=True"]
        run_combined(tmp_path, self.config_path, do_eval=False, do_onnx=False, train_args=args)

    def test_train_unbatched(self, tmp_path) -> None:
        args = ["--data.batched_read=False"]
        run_combined(tmp_path, self.config_path, do_eval=False, do_onnx=False, train_args=args)

    def test_train_dev(self, tmp_path) -> None:
        args = ["--trainer.fast_dev_run=2"]
        run_combined(tmp_path, self.config_path, do_eval=False, do_onnx=False, train_args=args)

    def test_train_movefilestemp(self, tmp_path) -> None:
        tmp_path = Path(tmp_path)
        move_path = tmp_path / "dev" / "shm"
        args = [f"--data.move_files_temp={move_path}"]
        run_combined(tmp_path, self.config_path, do_eval=False, do_onnx=False, train_args=args)
        assert not Path(move_path).exists()

    def test_train_distributed(self, tmp_path) -> None:
        args = ["--trainer.devices=2", "--data.num_workers=2", "--model.lrs_config.pct_start=0.2"]
        run_combined(tmp_path, self.config_path, do_eval=False, do_onnx=False, train_args=args)


@pytest.mark.filterwarnings(w)
class TestModels:
    def test_GN1(self, tmp_path) -> None:
        run_combined(tmp_path, "configs/GN1.yaml")

    def test_GN1_GATv2(self, tmp_path) -> None:
        args = ["--config=configs/GATv2.yaml"]
        run_combined(tmp_path, "configs/GN1.yaml", train_args=args)

    def test_DIPS(self, tmp_path) -> None:
        run_combined(tmp_path, "configs/dips.yaml", do_eval=True, do_onnx=False)

    def test_regression(self, tmp_path) -> None:
        run_combined(tmp_path, "configs/regression.yaml", do_eval=False, do_onnx=False)
