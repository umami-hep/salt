import sys
from pathlib import Path

import pytest

from salt.main import main
from salt.to_onnx import main as to_onnx
from salt.utils.get_onnx_metadata import main as get_onnx_metadata
from salt.utils.inputs import write_dummy_file, write_dummy_norm_dict

w = "ignore::lightning.fabric.utilities.warnings.PossibleUserWarning:"


def run_train(tmp_path, config_path, train_args):
    tmp_path = Path(tmp_path)
    train_h5_path = tmp_path / "dummy_train_inputs.h5"
    nd_path = tmp_path / "dummy_norm_dict.yaml"
    cd_path = tmp_path / "dummy_class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    write_dummy_file(train_h5_path, nd_path)

    args = ["fit"]
    args += [f"--config={config_path}"]
    args += [f"--data.norm_dict={nd_path}"]
    args += [f"--data.class_dict={cd_path}"]
    args += [f"--data.train_file={train_h5_path}"]
    args += [f"--data.val_file={train_h5_path}"]
    args += ["--data.num_jets_train=500"]
    args += ["--data.num_jets_val=200"]
    args += ["--data.batch_size=100"]
    args += ["--data.num_workers=0"]
    args += ["--trainer.max_epochs=1"]
    args += ["--trainer.accelerator=cpu"]
    args += ["--trainer.devices=1"]
    args += [f"--trainer.default_root_dir={tmp_path}"]
    args += ["--trainer.logger.offline=True"]
    if train_args:
        args += train_args

    main(args)


def run_eval(tmp_path, train_config_path, nd_path):
    test_h5_path = Path(tmp_path) / "dummy_test_sample_inputs.h5"
    write_dummy_file(test_h5_path, nd_path)

    args = ["test"]
    args += [f"--config={train_config_path}"]
    args += [f"--data.test_file={test_h5_path}"]
    args += ["--data.num_jets_test=1000"]
    main(args)


def run_onnx(train_dir, nd_path):
    ckpt_path = [f for f in (train_dir / "ckpts").iterdir() if f.suffix == ".ckpt"][-1]
    args = [f"--config={train_dir / 'config.yaml'}"]
    args += [f"--ckpt_path={ckpt_path}"]
    args += ["--track_selection=dipsLoose202102"]
    args += [f"--nd_path={nd_path}"]
    to_onnx(args)
    get_onnx_metadata([str(train_dir / "network.onnx")])


def run_combined(tmp_path, config, do_eval=True, do_onnx=True, train_args=None):
    sys.argv = [sys.argv[0]]  # ignore pytest cli args when running salt cli
    config_base = Path(__file__).parent.parent / "configs"

    # run training
    run_train(tmp_path, config_base / config, train_args)

    if do_eval:
        train_dir = [x for x in tmp_path.iterdir() if x.is_dir() and (x / "config.yaml").exists()]
        assert len(train_dir) == 1
        train_dir = train_dir[0]
        print(f"Using train_dir {train_dir}.")
        train_config_path = train_dir / "config.yaml"
        nd_path = [x for x in train_dir.iterdir() if x.suffix == ".yaml" and "norm" in str(x)]
        assert len(nd_path) == 1
        nd_path = nd_path[0]
        run_eval(tmp_path, train_config_path, nd_path)
    if do_onnx:
        run_onnx(train_dir, nd_path)


@pytest.mark.filterwarnings(w)
class TestTrainMisc:
    config = "GN1.yaml"

    def test_train_dev(self, tmp_path) -> None:
        args = ["--trainer.fast_dev_run=2"]
        run_combined(tmp_path, self.config, do_eval=False, do_onnx=False, train_args=args)

    def test_train_movefilestemp(self, tmp_path) -> None:
        tmp_path = Path(tmp_path)
        move_path = tmp_path / "dev" / "shm"
        args = [f"--data.move_files_temp={move_path}"]
        run_combined(tmp_path, self.config, do_eval=False, do_onnx=False, train_args=args)
        assert not Path(move_path).exists()

    def test_train_distributed(self, tmp_path) -> None:
        args = ["--trainer.devices=2", "--data.num_workers=2", "--model.lrs_config.pct_start=0.2"]
        run_combined(tmp_path, self.config, do_eval=False, do_onnx=False, train_args=args)

    def test_write_tracks(self, tmp_path) -> None:
        args = ["--trainer.callbacks+=salt.callbacks.PredictionWriter"]
        args += ["--trainer.callbacks.write_tracks=True"]
        args += ["--trainer.callbacks.track_variables=null"]
        run_combined(tmp_path, self.config, do_eval=True, do_onnx=False, train_args=args)


@pytest.mark.filterwarnings(w)
def test_GN1(tmp_path) -> None:
    run_combined(tmp_path, "GN1.yaml")


@pytest.mark.filterwarnings(w)
def test_GN2(tmp_path) -> None:
    run_combined(tmp_path, "GN2.yaml")


@pytest.mark.filterwarnings(w)
def test_GN2emu(tmp_path) -> None:
    run_combined(tmp_path, "GN2emu.yaml", do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_GN1_GATv2(tmp_path) -> None:
    args = [f"--config={Path(__file__).parent.parent / 'configs' / 'GATv2.yaml'}"]
    run_combined(tmp_path, "GN1.yaml", train_args=args)


@pytest.mark.filterwarnings(w)
def test_DIPS(tmp_path) -> None:
    run_combined(tmp_path, "dips.yaml", do_eval=True, do_onnx=True)


@pytest.mark.filterwarnings(w)
def test_regression(tmp_path) -> None:
    run_combined(tmp_path, "regression.yaml", do_eval=False, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_flow(tmp_path) -> None:
    run_combined(tmp_path, "flow.yaml", do_eval=False, do_onnx=False)
