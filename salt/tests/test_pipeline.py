import sys
from pathlib import Path

import h5py
import pytest

from salt.main import main
from salt.to_onnx import main as to_onnx
from salt.utils.get_onnx_metadata import main as get_onnx_metadata
from salt.utils.inputs import write_dummy_file, write_dummy_norm_dict

w = "ignore::lightning.fabric.utilities.warnings.PossibleUserWarning:"
CONFIG = "GN2.yaml"


def run_train(tmp_path, config_path, train_args, do_xbb=False, do_muP=False):
    incl_taus = config_path.name == CONFIG
    tmp_path = Path(tmp_path)
    train_h5_path = tmp_path / "dummy_train_inputs.h5"
    nd_path = tmp_path / "dummy_norm_dict.yaml"
    cd_path = tmp_path / "dummy_class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    write_dummy_file(train_h5_path, nd_path, do_xbb, incl_taus)

    args = ["fit"]
    args += [f"--config={config_path}"]
    args += [f"--data.norm_dict={nd_path}"]
    args += [f"--data.class_dict={cd_path}"]
    args += [f"--data.train_file={train_h5_path}"]
    args += [f"--data.val_file={train_h5_path}"]
    args += ["--data.num_train=500"]
    args += ["--data.num_val=200"]
    args += ["--data.batch_size=100"]
    args += ["--data.num_workers=0"]
    args += ["--trainer.max_epochs=1"]
    args += ["--trainer.accelerator=cpu"]
    args += ["--trainer.devices=1"]
    args += [f"--trainer.default_root_dir={tmp_path}"]
    args += ["--trainer.logger.offline=True"]

    # add another instance of the prediction writer callback with tracks added
    args += ["--trainer.callbacks+=salt.callbacks.PredictionWriter"]
    args += ["--trainer.callbacks.write_tracks=True"]

    if train_args:
        args += train_args

    if do_muP:
        from salt.utils.muP_utils.main_muP import main as main_muP

        # skip fit and callbacks
        main_muP(args=args[1:-2])

    main(args)


def run_eval(tmp_path, train_config_path, nd_path, do_xbb=False):
    test_h5_path = Path(tmp_path) / "dummy_test_sample_inputs.h5"
    write_dummy_file(test_h5_path, nd_path, do_xbb)

    args = ["test"]
    args += [f"--config={train_config_path}"]
    args += [f"--data.test_file={test_h5_path}"]
    args += ["--data.num_test=1000"]
    main(args)

    # check output h5 files are produced
    h5_dir = train_config_path.parent / "ckpts"
    h5_files = [f for f in h5_dir.iterdir() if f.suffix == ".h5"]
    assert len(h5_files) == 1
    h5_file = h5_files[0]
    with h5py.File(h5_file, "r") as f:
        assert "jets" in f
        assert len(f["jets"]) == 1000
        if "GN2" in str(train_config_path):
            assert "tracks" in f
            assert len(f["tracks"]) == 1000


def run_onnx(train_dir, args=None):
    ckpt_path = [f for f in (train_dir / "ckpts").iterdir() if f.suffix == ".ckpt"][-1]
    if args is None:
        args = []
    args += [f"--ckpt_path={ckpt_path}"]
    args += ["--track_selection=dipsLoose202102"]
    args += args
    to_onnx(args)
    get_onnx_metadata([str(train_dir / "network.onnx")])


def run_combined(
    tmp_path,
    config,
    do_eval=True,
    do_onnx=True,
    train_args=None,
    export_args=None,
    do_xbb=False,
    do_muP=False,
):
    sys.argv = [sys.argv[0]]  # ignore pytest cli args when running salt cli
    config_base = Path(__file__).parent.parent / "configs"

    # run training
    run_train(tmp_path, config_base / config, train_args, do_xbb, do_muP)

    if do_eval:
        train_dir = [x for x in tmp_path.iterdir() if x.is_dir() and (x / "config.yaml").exists()]
        assert len(train_dir) == 1
        train_dir = train_dir[0]
        print(f"Using train_dir {train_dir}.")
        train_config_path = train_dir / "config.yaml"
        nd_path = [x for x in train_dir.iterdir() if x.suffix == ".yaml" and "norm" in str(x)]
        assert len(nd_path) == 1
        nd_path = nd_path[0]
        run_eval(tmp_path, train_config_path, nd_path, do_xbb)
    if do_onnx:
        run_onnx(train_dir, export_args)


@pytest.mark.filterwarnings(w)
def test_GN1(tmp_path) -> None:
    run_combined(tmp_path, "GN1.yaml")


@pytest.mark.filterwarnings(w)
def test_GN2(tmp_path) -> None:
    run_combined(tmp_path, CONFIG, export_args=["--include_aux"])


@pytest.mark.filterwarnings(w)
def test_GN2_muP(tmp_path) -> None:
    run_combined(tmp_path, "GN2_muP.yaml", do_muP=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_GN2emu(tmp_path) -> None:
    run_combined(tmp_path, "GN2emu.yaml", do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_GN2XE(tmp_path) -> None:
    run_combined(tmp_path, "GN2XE.yaml", do_onnx=False, do_xbb=True)


@pytest.mark.filterwarnings(w)
def test_GN1_GATv2(tmp_path) -> None:
    args = [f"--config={Path(__file__).parent.parent / 'configs' / 'GATv2.yaml'}"]
    run_combined(tmp_path, "GN1.yaml", train_args=args)


@pytest.mark.filterwarnings(w)
def test_DIPS(tmp_path) -> None:
    run_combined(tmp_path, "dips.yaml", do_eval=True, do_onnx=True)


@pytest.mark.filterwarnings(w)
def test_DL1(tmp_path) -> None:
    run_combined(tmp_path, "DL1.yaml", do_eval=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_regression(tmp_path) -> None:
    run_combined(tmp_path, "regression.yaml", do_eval=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_nan_regression(tmp_path) -> None:
    run_combined(tmp_path, "nan_regression.yaml", do_eval=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_regression_gaussian(tmp_path) -> None:
    run_combined(tmp_path, "regression_gaussian.yaml", do_eval=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_flow(tmp_path) -> None:
    run_combined(tmp_path, "flow.yaml", do_eval=False, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_no_global_inputs(tmp_path) -> None:
    [f"--config={Path(__file__).parent.parent / 'tests' / 'configs' / 'no_global_inputs.yaml'}"]
    run_combined(tmp_path, CONFIG, do_eval=False, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_train_dev(tmp_path) -> None:
    args = ["--trainer.fast_dev_run=2"]
    run_combined(tmp_path, CONFIG, do_eval=False, do_onnx=False, train_args=args)


@pytest.mark.filterwarnings(w)
def test_train_movefilestemp(tmp_path) -> None:
    tmp_path = Path(tmp_path)
    move_path = tmp_path / "dev" / "shm"
    args = [f"--data.move_files_temp={move_path}"]
    run_combined(tmp_path, CONFIG, do_eval=False, do_onnx=False, train_args=args)
    assert not Path(move_path).exists()


@pytest.mark.filterwarnings(w)
def test_train_distributed(tmp_path) -> None:
    args = ["--trainer.devices=2", "--data.num_workers=2", "--model.lrs_config.pct_start=0.2"]
    run_combined(tmp_path, CONFIG, do_eval=False, do_onnx=False, train_args=args)


@pytest.mark.filterwarnings(w)
def test_truncate_inputs(tmp_path) -> None:
    args = ["--data.num_inputs.tracks=10"]
    run_combined(tmp_path, CONFIG, do_eval=True, do_onnx=False, train_args=args)


@pytest.mark.filterwarnings(w)
def test_truncate_inputs_error(tmp_path) -> None:
    args = ["--data.num_inputs.this_should_error=10"]
    with pytest.raises(ValueError, match="must be a subset of"):
        run_combined(tmp_path, CONFIG, do_eval=False, do_onnx=False, train_args=args)


@pytest.mark.filterwarnings(w)
def test_tfv2(tmp_path) -> None:
    args = [f"--config={Path(__file__).parent.parent / 'configs' / 'encoder-v2.yaml'}"]
    run_combined(tmp_path, CONFIG, train_args=args)
