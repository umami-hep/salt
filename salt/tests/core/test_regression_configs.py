"""End-to-end smoke for the M5 sub-wave A regression-family configs (plan 10/11).

The four v2-native configs (regression_gaussian, regression_weighted,
nan_regression, regression_multi_target) each: parse + instantiate through the
real ``salt2`` CLI, ``graph validate`` all four modes, and run a real
``salt2 fit --fast_dev_run`` end-to-end on a dummy file (import + forward-run +
loss computed, rc=0). The dummy file carries every label these configs need
(HadronConeExclTruthLabel*, R10TruthLabel_*, sample_weight, the NaN-injected
Lxy, the ID for the MultiTarget selection) — see ``salt.utils.inputs``.

No machine paths: the dummy H5 + norm/class dicts + schema are generated into a
tmp dir, the configs' documented required override is the norm_dict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.core.main import CONFIG_DIR
from salt.core.main import main as salt2_main
from salt.core.schema import dump_schema, save_schema
from salt.tests.core.gn2_fixture import write_parity_norm_dict
from salt.tests.core.regression_fixture import write_vector_concat_norm_dict
from salt.utils.inputs import write_dummy_file

CONFIGS = [
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
    h5_path = base / "pp_output_train.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


@pytest.mark.parametrize("config", CONFIGS)
def test_config_validates_all_modes(config, data):
    cfg = CONFIG_DIR / f"{config}.yaml"
    rc = salt2_main([
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
    rc = salt2_main([
        "fit",
        "--config",
        str(cfg),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        f"--trainer.default_root_dir={tmp_path}",
        "--trainer.accelerator=cpu",
        "--trainer.fast_dev_run=2",
        "--trainer.enable_progress_bar=false",
    ])
    assert rc == 0, f"{config} failed fast_dev_run fit"


def test_gn3v01_config_validates_all_modes(tmp_path):
    """The GN3V01 flagship (VectorConcat + alias + norm_type:hybrid) plan-compiles.

    Validates the shipped ``gn3v01.yaml`` through the real ``salt2 graph
    validate`` in ALL four modes (FIT/VAL/TEST/ONNX) — the sub-wave-B feature
    config (design §6.6). Uses the augmented norm dict (jets/tracks/global) for
    the TWO Normalisers; no H5 is needed (``validate`` is data-free, design
    §2.3/§4.1). The ONNX mode in particular exercises the ``export.inputs
    alias:`` (``inputs.global`` cloned from ``inputs.jets``) and the
    ``VectorConcat`` width resolution at bind.
    """
    nd_path, cd_path = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_vector_concat_norm_dict(nd_path, cd_path)
    rc = salt2_main([
        "graph",
        "validate",
        "-c",
        str(CONFIG_DIR / "gn3v01.yaml"),
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
    assert rc == 0, "gn3v01.yaml failed graph validate"
