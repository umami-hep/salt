"""Every shipped production config, through the full lifecycle on synthetic data.

The top level of ``salt/configs`` holds the production configs — one per model
family, each family's back-catalogue living a directory down. Each of them runs
the four things a user does, in order:

1. **merge + plot** — the config stack merges and ``salt graph plot`` renders it
2. **train** — ``salt fit`` on synthetic data, producing a checkpoint
3. **eval** — ``salt test`` against that checkpoint, writing an eval H5
4. **export** — ``salt export``, which runs ``check_onnx`` internally and so
   proves the exported graph agrees with the eager torch model

Data comes from ``salt.testing.datagen``: schema-driven recipes, one per config
family, which emit the H5 *and* the matching norm/class dicts. Each config names
the recipe that feeds it in ``RECIPES``. A config with no recipe entry fails
rather than skips — adding a production config means providing data for it.

This supersedes the older per-config smoke tests, which could only reach the
handful of configs the fixed-schema ``write_dummy_file`` happened to serve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.schema import dump_schema, save_schema
from salt.testing.datagen import load_pipeline

pytestmark = pytest.mark.cpu_always

RECIPES_DIR = Path(__file__).resolve().parents[2] / "testing" / "datagen" / "recipes"

# Production config -> the datagen recipe that feeds it.
RECIPES: dict[str, str] = {
    "event_classifier": "event_objects",
    "gn2v2-opendata": "flavour_tagger",
    "GN3EPCLV01": "flavour_tagger",
    "GN3X": "flavour_tagger",
    "hitz": "single_constituent_regression",
    "MaskFormer": "maskformer_truth_hadron",
    "nan_regression": "flavour_tagger",
    "regression": "flavour_tagger",
    "regression_gaussian": "flavour_tagger",
    "regression_multi_target": "flavour_tagger",
    "regression_weighted": "flavour_tagger",
}

# base2.yaml is auto-loaded machinery, never a model in its own right.
MACHINERY = {"base2"}


def _production_configs() -> list[str]:
    """Every top-level config — the production ones, discovered not listed."""
    return sorted(
        p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem not in MACHINERY
    )


PRODUCTION = _production_configs()


def test_every_production_config_has_a_recipe():
    """A new production config must bring data with it, not slip through."""
    missing = sorted(set(PRODUCTION) - set(RECIPES))
    assert not missing, (
        f"production configs with no datagen recipe: {missing}. Add an entry to "
        "RECIPES naming the recipe that feeds it (salt/testing/datagen/recipes)."
    )
    unknown = sorted(set(RECIPES) - set(PRODUCTION))
    assert not unknown, f"RECIPES names configs that are not top-level: {unknown}"


@pytest.fixture(scope="module")
def datasets(tmp_path_factory) -> dict[str, dict[str, Path]]:
    """Run each recipe once; every config using it shares the result."""
    import yaml  # noqa: PLC0415

    from salt.testing.datagen import compute_norm_dict  # noqa: PLC0415

    built: dict[str, dict[str, Path]] = {}
    for recipe in sorted(set(RECIPES.values())):
        out = tmp_path_factory.mktemp(f"datagen_{recipe}")
        pipe = load_pipeline(str(RECIPES_DIR / f"{recipe}.yaml"))
        pipe.set_output_dir(out)
        data = pipe.run()
        h5 = out / f"{recipe}.h5"
        assert h5.is_file(), f"recipe {recipe} wrote no H5"
        schema = out / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        entry = {"h5": h5, "schema": schema, "dir": out}
        for name in ("norm_dict.yaml", "class_dict.yaml"):
            path = out / name
            if path.is_file():
                entry[name.split("_")[0]] = path
        # not every recipe ships a NormWriter (event_objects does not), but every
        # config with a Normaliser needs one — derive it from the produced arrays
        if "norm" not in entry:
            nd = out / "norm_dict.yaml"
            nd.write_text(yaml.dump(compute_norm_dict(data), sort_keys=False))
            entry["norm"] = nd
        built[recipe] = entry
    return built


def _norm_dict_overrides(cfg: Path, norm_dict: Path) -> list[str]:
    """``model.modules.<name>.init_args.norm_dict`` for every Normaliser.

    The modules live under ``model.init_args.modules``; reading ``model.modules``
    instead silently yields no overrides and turns healthy configs into failures.
    Several configs carry more than one Normaliser (``norm`` plus
    ``norm_global``), so overriding only ``norm`` is not enough.
    """
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(cfg.read_text()) or {}
    model = raw.get("model")
    modules = (model or {}).get("init_args", {}).get("modules") if isinstance(model, dict) else None
    return [
        f"model.modules.{name}.init_args.norm_dict={norm_dict}"
        for name, node in (modules or {}).items()
        if isinstance(node, dict)
        and "Normaliser" in str(node.get("class_path", ""))
        and "norm_dict" not in (node.get("init_args") or {})
    ]


def _declares_onnx_export(cfg: Path) -> bool:
    """Whether the config wires an ONNX sink, by a top-level ``export:`` block
    or an ``OnnxExportSink`` in its ``outputs:`` section.
    """
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(cfg.read_text()) or {}
    if raw.get("export"):
        return True
    outputs = raw.get("outputs") or {}
    return any(
        "Onnx" in str(node.get("class_path", ""))
        for node in outputs.values()
        if isinstance(node, dict)
    )


def _data_args(cfg: Path, data: dict[str, Path]) -> list[str]:
    return [
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        # shipped configs assume large training machines
        "--data.num_workers=0",
        # the recipes emit 1000 rows; shipped batch sizes run to 4000
        "--data.batch_size=50",
        *(f"--{o}" for o in _norm_dict_overrides(cfg, data["norm"])),
    ]


def _trainer_args(root: Path) -> list[str]:
    return [
        f"--trainer.default_root_dir={root}",
        # auto, not cpu: on a GPU runner these must exercise the GPU path
        "--trainer.accelerator=auto",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
        # base2 ships a default-ON CometLogger; off so no offline archive lands
        "--trainer.logger=false",
        # null-delete the base2 ProgressBar: the stock enable_progress_bar=false
        # cannot coexist with a configured bar
        "--callbacks.progress=null",
    ]


@pytest.mark.parametrize("config", PRODUCTION)
def test_config_merges_and_plots(config, datasets, tmp_path):
    """Leg 1 — the config stack merges and its graph renders."""
    data = datasets[RECIPES[config]]
    cfg = CONFIG_DIR / f"{config}.yaml"

    sets: list[str] = []
    for override in ["trainer.accelerator=auto", *_norm_dict_overrides(cfg, data["norm"])]:
        sets += ["--set", override]

    argv = ["graph", "validate", "-c", str(cfg)]
    for mode in ("fit", "val", "test", "onnx"):
        argv += ["--mode", mode]
    assert salt_main([*argv, *sets]) == 0, f"{config}: config merge / plan compile failed"

    # `graph plot` renders one mode to one file
    out_path = tmp_path / f"{config}.dot"
    plot_argv = ["graph", "plot", "-c", str(cfg), "--mode", "fit", "-o", str(out_path)]
    assert salt_main([*plot_argv, *sets]) == 0, f"{config}: graph plot failed"
    assert out_path.is_file() or any(tmp_path.iterdir()), (
        f"{config}: graph plot wrote nothing to {tmp_path}"
    )


@pytest.mark.parametrize("config", PRODUCTION)
def test_config_trains_evaluates_and_exports(config, datasets, tmp_path):
    """Legs 2-4 — train, evaluate, then export with torch<->ONNX parity.

    One test because the legs chain: eval and export both need the checkpoint
    training produced, and the saved config that came with it.
    """
    data = datasets[RECIPES[config]]
    cfg = CONFIG_DIR / f"{config}.yaml"

    # -- leg 2: train
    rc = salt_main(["fit", "--config", str(cfg), *_data_args(cfg, data), *_trainer_args(tmp_path)])
    assert rc == 0, f"{config}: salt fit failed"

    ckpts = sorted(tmp_path.rglob("*.ckpt"))
    assert ckpts, f"{config}: training wrote no checkpoint under {tmp_path}"
    ckpt = ckpts[0]
    saved = sorted(tmp_path.rglob("config.yaml"))
    assert saved, f"{config}: training saved no config.yaml"

    # -- leg 3: evaluate
    rc = salt_main([
        "test",
        "--config",
        str(saved[0]),
        f"--ckpt_path={ckpt}",
        f"--data.test_file={data['h5']}",
        "--data.num_workers=0",
        "--trainer.accelerator=auto",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ])
    assert rc == 0, f"{config}: salt test failed"
    evals = sorted(ckpt.parent.glob("*__test_*.h5"))
    assert evals, f"{config}: eval wrote no H5 next to {ckpt}"

    # -- leg 4: export. Skipped only for a config that wires no ONNX sink at
    # all, read from the config rather than a hardcoded list. Today that is
    # event_classifier alone: its mHH_regression divides by a quantity it takes
    # from labels rather than inputs, so there is no export-time source for the
    # de-scaling. Declaring mHH an input feature would make it exportable, but
    # that changes what the network consumes and is a modelling decision.
    if not _declares_onnx_export(cfg):
        pytest.skip(f"{config} wires no ONNX sink — fit/test only by design")

    # `salt export` runs check_onnx itself, so a zero return code IS the
    # torch<->ONNX equivalence assertion.
    rc = salt_main([
        "export",
        "--config",
        str(saved[0]),
        f"--ckpt_path={ckpt}",
        f"--outdir={tmp_path / 'onnx'}",
    ])
    assert rc == 0, (
        f"{config}: salt export failed. The CLI runs check_onnx by default, so "
        "this covers both the export itself and its agreement with eager torch."
    )
    assert sorted((tmp_path / "onnx").glob("*.onnx")), f"{config}: no ONNX written"
