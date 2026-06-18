"""End-to-end smoke for shipped v2-native configs (plan 10/11 M5 + plan 12 M6).

The four M5 sub-wave A regression-family configs (regression_gaussian,
regression_weighted, nan_regression, regression_multi_target) each: parse +
instantiate through the real ``salt2`` CLI, ``graph validate`` all four modes,
and run a real ``salt2 fit --fast_dev_run`` end-to-end on a dummy file (import +
forward-run + loss computed, rc=0). The dummy file carries every label these
configs need (HadronConeExclTruthLabel*, R10TruthLabel_*, sample_weight, the
NaN-injected Lxy, the ID for the MultiTarget selection) — see
``salt.utils.inputs``.

Also hosts the dedicated shipped-config CI tests: ``gn3v01`` (M5 sub-wave B
VectorConcat flagship) and ``DL1`` (M6 sub-wave D vector-stream / MLP-only — the
v2-native port of the v1 ``test_pipeline.py:213`` ``test_DL1`` CI fixture the
user asked to preserve; validate-all-modes + a real fast_dev_run fit).

No machine paths: the dummy H5 + norm/class dicts + schema are generated into a
tmp dir, the configs' documented required override is the norm_dict.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from salt.core.main import CONFIG_DIR
from salt.core.main import main as salt2_main
from salt.core.schema import dump_schema, save_schema
from salt.tests.core.gn2_fixture import write_parity_norm_dict
from salt.tests.core.regression_fixture import write_vector_concat_norm_dict
from salt.utils.inputs import write_dummy_file

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
        # null-delete the base2 ProgressBar (D2 default-on); the stock
        # enable_progress_bar=false cannot coexist with a configured bar
        "--callbacks.progress=null",
    ])
    assert rc == 0, f"{config} failed fast_dev_run fit"


def test_dl1_config_validates_all_modes(data):
    """DL1's jets-only MLP path plan-compiles in fit + test + onnx.

    The v2-native port of the v1 ``test_pipeline.py:213`` ``test_DL1`` CI fixture
    (legacy/DL1.yaml) — the MLP-only / vector-stream coverage the user asked to
    PRESERVE (plan 12 sub-wave D non-regression invariant: "DL1's original
    MLP-only CI fixture is preserved unchanged"). DL1 is the canonical exercise
    of the M6-6 deliverable: the rank-2 ``[B, F]`` vector-stream embed (rank
    inferred from the reader ``global_object: true`` boundary) -> a
    ``sequence: false`` head, NO encoder / NO pool.
    Validates the shipped ``DL1.yaml`` through the real ``salt2 graph validate``
    in all default modes (fit/test/onnx); the parity norm dict (jets:
    pt_btagJes/eta_btagJes) is the documented required override.
    """
    cfg = CONFIG_DIR / "DL1.yaml"
    rc = salt2_main([
        "graph",
        "validate",
        "-c",
        str(cfg),
        "--set",
        f"model.modules.norm.init_args.norm_dict={data['nd']}",
    ])
    assert rc == 0, "DL1.yaml failed graph validate"


def test_dl1_config_fast_dev_run_fit(data, tmp_path):
    """DL1 runs a real ``salt2 fit --fast_dev_run`` end-to-end (MLP-only path).

    The behavioural half of the v1 ``test_DL1`` port: a real fit (import +
    rank-2 ``[B, F]`` embed forward + 3-class CE loss computed, rc=0) on the
    SAME dummy H5 the regression-family configs use — the default
    ``write_dummy_file`` writes jets ``flavour_label`` in {0,1,2} with the
    schema attr ``[bjets, cjets, ujets]`` (inputs.py), matching DL1's 3-class
    head index-for-index (the check_class_names cross-check). No constituents
    are needed: DL1's only stream is the global jets vector.
    """
    cfg = CONFIG_DIR / "DL1.yaml"
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
        # DL1 ships the v1-faithful batch_size: 2000, larger than the dummy
        # file's 1000 jets -> zero train batches. Shrink it for the CI fixture
        # (the regression-family configs ship batch_size: 100, so need no
        # override); the graph wiring this test exercises is batch-size invariant
        "--data.batch_size=50",
        # null-delete the base2 ProgressBar (D2 default-on); the stock
        # enable_progress_bar=false cannot coexist with a configured bar
        "--callbacks.progress=null",
    ])
    assert rc == 0, "DL1.yaml failed fast_dev_run fit"


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
