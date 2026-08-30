"""Every shipped production config, through the full lifecycle on synthetic data.

The top level of ``salt/configs`` holds the production configs — one per model
family, each family's back-catalogue living a directory down. Each of them runs
the four things a user does, in order:

1. **merge + plot** — the config stack merges and ``salt graph plot`` renders it
2. **train** — ``salt fit`` on synthetic data, producing a checkpoint
3. **eval** — ``salt test`` against that checkpoint, writing an eval H5
4. **export** — ``salt export``, which runs ``check_onnx`` internally and so
   proves the exported graph agrees with the eager torch model

Most configs are fed from ``salt.testing.datagen``: schema-driven recipes, one
per config family, which emit the H5 *and* the matching norm/class dicts. Each
names the recipe that feeds it in ``RECIPES``.

``ttbar_vs_hh4b_event_tagger`` is fed differently, and has to be: it declares no
reader (pairing it with one of two reader fragments IS the model), and both of
those fragments read ROOT, which no datagen recipe emits — ``salt.testing.datagen``
is H5-only. It is listed in ``PAIRED`` instead, giving the reader fragment to
stack and the ROOT fixture that feeds it. Only the easyjet pairing runs: the
PHYSLITE xAOD POOL layout is not synthesisable, so that leg is held statically by
`test_event_tagger_readers.py` and dynamically only against real files.

A production config in neither table fails rather than skips — adding one means
providing data for it.
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
    "gn2v2-opendata": "flavour_tagger",
    "GN3EPCLV01": "flavour_tagger_global",
    "GN3X": "flavour_tagger",
    "hitz": "hits_regression",
    "MaskFormer": "maskformer_truth_hadron",
}

# Production config -> the reader fragment stacked under it for this gate.
# A config here declares no reader of its own; the fragment supplies one and a
# ROOT fixture supplies the data (datagen is H5-only and cannot feed either).
PAIRED: dict[str, str] = {
    "ttbar_vs_hh4b_event_tagger": "readers/easyjet_events",
}

# base.yaml is auto-loaded machinery, never a model in its own right.
MACHINERY = {"base"}


def _production_configs() -> list[str]:
    """Every top-level config — the production ones, discovered not listed."""
    return sorted(
        p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem not in MACHINERY
    )


PRODUCTION = _production_configs()


def test_every_production_config_brings_data():
    """A new production config must bring data with it, not slip through."""
    missing = sorted(set(PRODUCTION) - set(RECIPES) - set(PAIRED))
    assert not missing, (
        f"production configs with no data source: {missing}. Add an entry to "
        "RECIPES naming the datagen recipe that feeds it "
        "(salt/testing/datagen/recipes), or — for a config that declares no "
        "reader — to PAIRED naming the reader fragment to stack under it."
    )
    unknown = sorted((set(RECIPES) | set(PAIRED)) - set(PRODUCTION))
    assert not unknown, f"RECIPES/PAIRED name configs that are not top-level: {unknown}"
    both = sorted(set(RECIPES) & set(PAIRED))
    assert not both, f"configs claiming two data sources: {both}"


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
        # not every recipe ships a NormWriter, but every config with a
        # Normaliser needs one — derive it from the produced arrays
        if "norm" not in entry:
            nd = out / "norm_dict.yaml"
            nd.write_text(yaml.dump(compute_norm_dict(data), sort_keys=False))
            entry["norm"] = nd
        built[recipe] = entry
    return built


@pytest.fixture(scope="module")
def paired_data(tmp_path_factory) -> dict[str, dict[str, Path]]:
    """Synthetic ROOT data + a sourced copy of each PAIRED config's fragment.

    The fragment copy is derived from the shipped file with only the per-sample
    ``sources:`` filled in, so the groups under test are the shipped groups.
    """
    import yaml  # noqa: PLC0415

    from salt.config_utils import expand_includes  # noqa: PLC0415
    from salt.tests._fixtures.easyjet_minitree import (  # noqa: PLC0415
        write_jets_norm_dict,
        write_sample_pair,
        write_sourced_fragment,
    )

    built: dict[str, dict[str, Path]] = {}
    if not _root_deps():
        # writing a ROOT fixture needs uproot; return empty and let _feed skip
        # the PAIRED configs only. Raising here would error EVERY parametrisation
        # of both lifecycle tests, including the H5-fed ones this cannot touch.
        return built
    for config, fragment in PAIRED.items():
        out = tmp_path_factory.mktemp(f"paired_{config}")
        signal, background = write_sample_pair(out)
        raw = yaml.safe_load(
            Path(expand_includes(str(CONFIG_DIR / f"{fragment}.yaml"))).read_text()
        )
        sourced = write_sourced_fragment(
            raw, out / "reader.yaml", {"signal": signal, "background": background}
        )
        model = yaml.safe_load((CONFIG_DIR / f"{config}.yaml").read_text())
        variables = model["data"]["modules"]["features"]["init_args"]["variables"]["jets"]
        built[config] = {
            "fragment": sourced,
            "norm": write_jets_norm_dict(out / "norm_dict.yaml", variables),
            "dir": out,
        }
    return built


def _root_deps() -> bool:
    """Whether uproot + awkward are importable (the ROOT fixture needs both)."""
    from importlib.util import find_spec  # noqa: PLC0415

    return all(find_spec(m) is not None for m in ("uproot", "awkward"))


def _feed(config: str, datasets: dict, paired_data: dict) -> tuple[list[Path], list[str], Path]:
    """``(config stack, data CLI args, norm_dict)`` for one production config.

    Two feeders: a datagen H5 recipe, or a reader fragment + ROOT fixture. The
    ``schema=`` override is H5-reader-only and must not reach a uproot reader.
    """
    cfg = CONFIG_DIR / f"{config}.yaml"
    if config in PAIRED:
        if config not in paired_data:
            pytest.skip(f"{config} is ROOT-fed; needs `pip install 'salt[root]'`")
        entry = paired_data[config]
        return (
            [cfg, entry["fragment"]],
            [
                "--data.num_workers=0",
                # the minitree fixture holds 6 events per sample
                "--data.batch_size=2",
                *(f"--{o}" for o in _norm_dict_overrides(cfg, entry["norm"])),
            ],
            entry["norm"],
        )
    data = datasets[RECIPES[config]]
    return [cfg], _data_args(cfg, data), data["norm"]


def _norm_dict_overrides(cfg: Path, norm_dict: Path) -> list[str]:
    """``model.modules.<name>.init_args.norm_dict`` for every Normaliser.

    The modules live under ``model.init_args.modules``; reading ``model.modules``
    instead silently yields no overrides and turns healthy configs into failures.
    Several configs carry more than one Normaliser (``norm`` plus
    ``norm_global``), so overriding only ``norm`` is not enough.
    """
    import yaml  # noqa: PLC0415

    from salt.config_utils import expand_includes  # noqa: PLC0415

    raw = yaml.safe_load(Path(expand_includes(str(cfg))).read_text()) or {}
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
    """Whether the config wires an ``OnnxExportSink`` in its ``outputs:`` section."""
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(cfg.read_text()) or {}
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
        # base ships a default-ON CometLogger; off so no offline archive lands
        "--trainer.logger=false",
        # null-delete the base ProgressBar: the stock enable_progress_bar=false
        # cannot coexist with a configured bar
        "--callbacks.progress=null",
    ]


@pytest.mark.parametrize("config", PRODUCTION)
def test_config_merges_and_plots(config, datasets, paired_data, tmp_path):
    """Leg 1 — the config stack merges and its graph renders."""
    stack, _, norm = _feed(config, datasets, paired_data)
    cfg = stack[0]

    sets: list[str] = []
    for override in ["trainer.accelerator=auto", *_norm_dict_overrides(cfg, norm)]:
        sets += ["--set", override]

    argv = ["graph", "validate"]
    for path in stack:
        argv += ["-c", str(path)]
    for mode in ("fit", "val", "test", "onnx"):
        argv += ["--mode", mode]
    assert salt_main([*argv, *sets]) == 0, f"{config}: config merge / plan compile failed"

    # `graph plot` renders one mode to one file
    out_path = tmp_path / f"{config}.dot"
    plot_argv = ["graph", "plot"]
    for path in stack:
        plot_argv += ["-c", str(path)]
    plot_argv += ["--mode", "fit", "-o", str(out_path)]
    assert salt_main([*plot_argv, *sets]) == 0, f"{config}: graph plot failed"
    assert out_path.is_file() or any(tmp_path.iterdir()), (
        f"{config}: graph plot wrote nothing to {tmp_path}"
    )


@pytest.mark.parametrize("config", PRODUCTION)
def test_config_trains_evaluates_and_exports(config, datasets, paired_data, tmp_path):
    """Legs 2-4 — train, evaluate, then export with torch<->ONNX parity.

    One test because the legs chain: eval and export both need the checkpoint
    training produced, and the saved config that came with it.
    """
    stack, data_args, _ = _feed(config, datasets, paired_data)
    cfg = stack[0]

    # -- leg 2: train
    fit_argv = ["fit"]
    for path in stack:
        fit_argv += ["--config", str(path)]
    rc = salt_main([*fit_argv, *data_args, *_trainer_args(tmp_path)])
    assert rc == 0, f"{config}: salt fit failed"

    ckpts = sorted(tmp_path.rglob("*.ckpt"))
    assert ckpts, f"{config}: training wrote no checkpoint under {tmp_path}"
    ckpt = ckpts[0]
    saved = sorted(tmp_path.rglob("config.yaml"))
    assert saved, f"{config}: training saved no config.yaml"

    # -- leg 3: evaluate. `salt test` takes ONE config, so this is the saved run
    # config — which already carries the merged reader, sources and all.
    # A PAIRED config's reader sources per sample, so test_file is a placeholder
    # it ignores and is left as the saved config has it.
    test_argv = [
        "test",
        "--config",
        str(saved[0]),
        f"--ckpt_path={ckpt}",
        "--data.num_workers=0",
        "--trainer.accelerator=auto",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ]
    if config not in PAIRED:
        test_argv.insert(4, f"--data.test_file={datasets[RECIPES[config]]['h5']}")
    rc = salt_main(test_argv)
    assert rc == 0, f"{config}: salt test failed"
    evals = sorted(ckpt.parent.glob("*__test_*.h5"))
    assert evals, f"{config}: eval wrote no H5 next to {ckpt}"

    # -- leg 4: export. Skipped only for a config that wires no ONNX sink at
    # all, read from the config rather than a hardcoded list.
    if not _declares_onnx_export(cfg):
        pytest.skip(f"{config} wires no ONNX sink — fit/test only by design")

    # `salt export` runs check_onnx itself, so a zero return code IS the
    # torch<->ONNX equivalence assertion.
    # deliberately NOT --no-check: the parity check is the point of this leg
    onnx_path = tmp_path / "onnx" / f"{config}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    rc = salt_main([
        "export",
        "--config",
        str(saved[0]),
        f"--ckpt_path={ckpt}",
        f"--output={onnx_path}",
    ])
    assert rc == 0, (
        f"{config}: salt export failed. The CLI runs check_onnx by default, so "
        "this covers both the export itself and its agreement with eager torch."
    )
    assert sorted(onnx_path.parent.glob("*.onnx")), f"{config}: no ONNX written"
