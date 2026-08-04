"""End-to-end smoke for shipped v2-native configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.v2_builders import write_vector_concat_norm_dict
from salt.testing.inputs import write_dummy_file

CONFIGS = [
    "regression",
    "regression_gaussian",
    "regression_weighted",
    "nan_regression",
    "regression_multi_target",
]


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("regression_configs")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # regression.yaml and regression_weighted.yaml declare `mass` as a third
    # jets input variable (target_denominators — must be a declared input
    # Feature for ONNX de-scaling).  write_parity_norm_dict only knows the
    # GN2 parity jets variables [pt_btagJes, eta_btagJes]; adding `mass` here
    # keeps the shared parity fixture untouched (do NOT add mass there —
    # it would change the parity gate's per-variable indexing and break
    # bitwise reproducibility).  Constants follow the same scheme as
    # write_parity_norm_dict: mean_i = 0.1*(i+1), std_i = 1.0+0.05*(i+1)
    # indexed by position in the jets variable list; mass is at index 2.
    with open(nd_path) as f:
        nd = yaml.safe_load(f)
    nd["jets"]["mass"] = {"mean": round(0.1 * 3, 6), "std": round(1.0 + 0.05 * 3, 6)}
    with open(nd_path, "w") as f:
        yaml.dump(nd, f, sort_keys=False)
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


@pytest.mark.parametrize("config", CONFIGS)
def test_config_validates_all_modes(config, data):
    cfg = CONFIG_DIR / f"{config}.yaml"
    rc = salt_main([
        "graph",
        "validate",
        "-c",
        str(cfg),
        "--set",
        f"model.modules.norm.init_args.norm_dict={data['nd']}",
    ])
    assert rc == 0, f"{config} failed graph validate"


@pytest.mark.parametrize("config", CONFIGS)
def test_config_fast_dev_run_fit(config, data, tmp_path):
    cfg = CONFIG_DIR / f"{config}.yaml"
    rc = salt_main([
        "fit",
        "--config",
        str(cfg),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        f"--trainer.default_root_dir={tmp_path}",
        "--trainer.accelerator=cpu",
        # base default-ON CometLogger → off for the fast_dev_run
        # fit so no offline Comet archive is written (and lr_monitor drops)
        "--trainer.logger=false",
        "--trainer.fast_dev_run=2",
        # null-delete the base ProgressBar (D2 default-on); the stock
        # enable_progress_bar=false cannot coexist with a configured bar
        "--callbacks.progress=null",
    ])
    assert rc == 0, f"{config} failed fast_dev_run fit"


def test_dl1_config_validates_all_modes(data):
    """DL1's jets-only MLP path plan-compiles in fit + test + onnx."""
    cfg = CONFIG_DIR / "legacy/DL1.yaml"
    rc = salt_main([
        "graph",
        "validate",
        "-c",
        str(cfg),
        "--set",
        f"model.modules.norm.init_args.norm_dict={data['nd']}",
    ])
    assert rc == 0, "DL1.yaml failed graph validate"


def test_dl1_config_fast_dev_run_fit(data, tmp_path):
    """DL1 runs a real ``salt fit --fast_dev_run`` end-to-end (MLP-only path)."""
    cfg = CONFIG_DIR / "legacy/DL1.yaml"
    rc = salt_main([
        "fit",
        "--config",
        str(cfg),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        f"--trainer.default_root_dir={tmp_path}",
        "--trainer.accelerator=cpu",
        # base default-ON CometLogger → off for the fast_dev_run
        # fit so no offline Comet archive is written (and lr_monitor drops)
        "--trainer.logger=false",
        "--trainer.fast_dev_run=2",
        # DL1 ships the v1-faithful batch_size: 2000, larger than the dummy
        # file's 1000 jets -> zero train batches. Shrink it for the CI fixture
        # (the regression-family configs ship batch_size: 100, so need no
        # override); the graph wiring this test exercises is batch-size invariant
        "--data.batch_size=50",
        # null-delete the base ProgressBar (D2 default-on); the stock
        # enable_progress_bar=false cannot coexist with a configured bar
        "--callbacks.progress=null",
    ])
    assert rc == 0, "DL1.yaml failed fast_dev_run fit"


def test_gn3epclv01_config_validates_all_modes(tmp_path):
    """The GN3 flagship (VectorConcat + alias + norm_type:hybrid) plan-compiles."""
    nd_path, cd_path = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_vector_concat_norm_dict(nd_path, cd_path)
    rc = salt_main([
        "graph",
        "validate",
        "-c",
        str(CONFIG_DIR / "GN3EPCLV01.yaml"),
        "--set",
        f"model.modules.norm.init_args.norm_dict={nd_path}",
        "--set",
        f"model.modules.norm_global.init_args.norm_dict={nd_path}",
        "--mode",
        "fit",
        "--mode",
        "val",
        "--mode",
        "test",
        "--mode",
        "onnx",
    ])
    assert rc == 0, "GN3EPCLV01.yaml failed graph validate"
