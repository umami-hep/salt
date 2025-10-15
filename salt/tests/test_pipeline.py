import sys
from pathlib import Path

import h5py
import pytest
import yaml

from salt.main import main
from salt.onnx.to_onnx import main as to_onnx
from salt.utils.get_onnx_metadata import main as get_onnx_metadata
from salt.utils.inputs import write_dummy_file, write_dummy_norm_dict

w = "ignore::lightning.fabric.utilities.warnings.PossibleUserWarning:"
CONFIG = "GN2.yaml"
TAU_CONFIGS = {"GN2.yaml", "GN3_baseline.yaml"}


def run_train(tmp_path, config_path, train_args, do_xbb=False, do_mup=False, inc_params=False):
    incl_taus = config_path.name in TAU_CONFIGS
    tmp_path = Path(tmp_path)
    train_h5_path = tmp_path / "dummy_train_inputs.h5"
    nd_path = tmp_path / "dummy_norm_dict.yaml"
    cd_path = tmp_path / "dummy_class_dict.yaml"
    write_dummy_norm_dict(nd_path, cd_path)
    write_dummy_file(train_h5_path, nd_path, do_xbb, incl_taus, inc_params)

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
    args += ["--trainer.logger.init_args.online=False"]

    if train_args:
        args += train_args

    if do_mup:
        from salt.utils.muP_utils.main_muP import main as main_mup

        # skip fit and callbacks
        main_mup(args=args[1:-2])

    main(args)


def run_eval(tmp_path, train_config_path, nd_path, do_xbb=False):
    test_h5_path = Path(tmp_path) / "dummy_test_sample_inputs.h5"
    write_dummy_file(test_h5_path, nd_path, do_xbb)

    # Modify the output config to force writing tracks in the prediction writer
    with open(train_config_path) as f:
        config = yaml.safe_load(f)
        for callback in config["trainer"]["callbacks"]:
            if "PredictionWriter" in callback["class_path"]:
                callback["init_args"]["write_tracks"] = True
                break
    with open(train_config_path, "w") as f:
        yaml.dump(config, f)
    args = ["test"]

    args += [f"--config={train_config_path}"]
    args += [f"--data.test_file={test_h5_path}"]
    args += ["--data.num_test=1000"]
    args += ["--data.batch_size=100"]
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

        if "maskformer" in str(train_config_path):
            assert "truth_hadrons" in f
            assert len(f["truth_hadrons"]) == 1000
            assert f["truth_hadrons"].shape[1] == 5
            required_keys = {
                "MaskFormer_pt",
                "MaskFormer_deta",
                "MaskFormer_dphi",
                "MaskFormer_mass",
                "MaskFormer_Lxy",
                "MaskFormer_pb",
                "MaskFormer_pc",
                "MaskFormer_pnull",
                "class_label",
            }
            print(set(required_keys) - set(f["truth_hadrons"].dtype.names))
            print(set(f["truth_hadrons"].dtype.names) - set(required_keys))
            assert all(k in f["truth_hadrons"].dtype.names for k in required_keys)

            assert "object_masks" in f
            assert f["object_masks"].shape == (1000, 5, 40)
            assert "mask_logits" in f["object_masks"].dtype.names
            assert "truth_mask" in f["object_masks"].dtype.names


def run_onnx(train_dir, args=None):
    ckpt_path = [f for f in (train_dir / "ckpts").iterdir() if f.suffix == ".ckpt"][-1]
    if args is None:
        args = []
    args += [f"--ckpt_path={ckpt_path}"]
    args += ["--track_selection=dipsLoose202102"]

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
    do_mup=False,
    inc_params=False,
):
    sys.argv = [sys.argv[0]]  # ignore pytest cli args when running salt cli

    # look for the config
    config_path = Path(__file__).parent.parent / "configs" / config
    if not config_path.is_file():
        config_path = Path(__file__).parent / "configs" / config

    # run training
    run_train(tmp_path, config_path, train_args, do_xbb, do_mup, inc_params)

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
    run_combined(tmp_path, "GN1.yaml", do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_GN2(tmp_path) -> None:
    run_combined(tmp_path, CONFIG, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_GN3(tmp_path) -> None:
    run_combined(
        tmp_path,
        "GN3_dev/GN3_baseline.yaml",
        export_args=["--tasks", "jets_classification", "track_vertexing", "track_origin"],
    )


@pytest.mark.filterwarnings(w)
def test_GN2_muP(tmp_path) -> None:
    run_combined(tmp_path, "GN2_muP.yaml", do_mup=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_GN2emu(tmp_path) -> None:
    run_combined(tmp_path, "GN2emu.yaml", do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_GN2XE(tmp_path) -> None:
    run_combined(tmp_path, "GN2XE.yaml", do_onnx=False, do_xbb=True)


@pytest.mark.filterwarnings(w)
def test_GN1_GATv2(tmp_path) -> None:
    args = [f"--config={Path(__file__).parent.parent / 'configs' / 'GATv2.yaml'}"]
    run_combined(tmp_path, "GN1.yaml", train_args=args, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_DIPS(tmp_path) -> None:
    run_combined(tmp_path, "dips.yaml", do_eval=True, do_onnx=True)


@pytest.mark.filterwarnings(w)
def test_DL1(tmp_path) -> None:
    run_combined(tmp_path, "DL1.yaml", do_eval=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_regression(tmp_path) -> None:
    run_combined(tmp_path, "regression.yaml", do_eval=True, do_onnx=True)


@pytest.mark.filterwarnings(w)
def test_nan_regression(tmp_path) -> None:
    run_combined(tmp_path, "nan_regression.yaml", do_eval=True, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_regression_gaussian(tmp_path) -> None:
    run_combined(tmp_path, "regression_gaussian.yaml", do_eval=True, do_onnx=True)


@pytest.mark.filterwarnings(w)
def test_flow(tmp_path) -> None:
    run_combined(tmp_path, "flow.yaml", do_eval=False, do_onnx=False)


@pytest.mark.filterwarnings(w)
def test_no_global_inputs(tmp_path) -> None:
    run_combined(tmp_path, "no_global_inputs.yaml", do_eval=False, do_onnx=False)


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


@pytest.mark.filterwarnings(w)
def test_maskformer(tmp_path) -> None:
    run_combined(tmp_path, "MaskFormer.yaml", train_args=None, export_args=["-mf=vertexing"])


@pytest.mark.filterwarnings(w)
def test_param_concat(tmp_path) -> None:
    args = [f"--config={Path(__file__).parent.parent / 'tests' / 'configs' / 'param_concat.yaml'}"]
    run_combined(tmp_path, CONFIG, do_onnx=False, inc_params=True, train_args=args)


@pytest.mark.filterwarnings(w)
def test_param_featurewise(tmp_path) -> None:
    args = [
        f"--config={Path(__file__).parent.parent / 'tests' / 'configs' / 'param_featurewise.yaml'}"
    ]
    run_combined(tmp_path, CONFIG, do_onnx=False, inc_params=True, train_args=args)


@pytest.mark.filterwarnings(w)
def test_gls_weighting(tmp_path) -> None:
    # Ensure that this raises an assertion
    args = ["--model.loss_mode=lol"]
    with pytest.raises(AssertionError):
        run_combined(tmp_path, "dips.yaml", train_args=args)

    # Should fail, as we still have weights here
    args = ["--model.loss_mode=GLS"]
    with pytest.raises(AssertionError):
        run_combined(tmp_path, "GN2.yaml", train_args=args)

    # And this *should* work
    run_combined(tmp_path, "dips.yaml", train_args=args)
