"""Every shipped config is exercised, by construction.

Discovery is a recursive glob of ``salt/configs`` — a config added anywhere in
the tree is picked up automatically and cannot silently go untested.

A shipped file is one of two things, decided from its own content rather than a
list here:

- a **config**: it has a ``model:`` and a ``data.modules.reader``, so it is a
  whole run and stands alone (an overlay declares its bases in its own
  ``include:`` block, so the stack lives in the config, not in a table that
  could drift from it);
- a **fragment**: it has only one half. ``readers/*`` fragments carry a reader
  and no model; ``ttbar_vs_hh4b_event_tagger`` carries a model and no reader,
  because pairing it with either reader fragment IS the demonstration.

A fragment has no plan to compile and nothing to fit, so it is NOT passed alone
to ``graph validate``. Making one validate standalone means giving it a model it
does not have — that was tried on ``readers/physlite`` and reverted. Fragments
are gated by `test_event_tagger_readers.py` instead, which instantiates their
reader (catching every constructor invariant) and holds the cross-format
contract; the ``ttbar_vs_hh4b_event_tagger`` pairings are additionally driven
through the full lifecycle by `test_production_configs.py`.

Two tiers, over the whole configs:

**Tier A** (`test_config_plan_compiles`): all-mode ``salt graph validate``.

**Tier B** (`test_config_fast_dev_run`): a real 2-batch fit for every config a
synthetic fixture can serve. Configs whose streams the shipped dummy writer
does not produce carry an explicit ``xfail`` naming the missing piece, so the
gap is visible in the report and shrinks as fixtures are added — never a silent
skip.

Adding a file to ``salt/configs`` therefore forces one of four explicit
outcomes: it compiles standalone, it declares its stack, it declares why it
cannot yet be fitted, or it is a fragment and says so by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

pytestmark = pytest.mark.cpu_always

# Auto-loaded by SaltCLI for every fit/test, never stacked by a user.
MACHINERY = {"base"}

# Which synthetic fixture flavour a config's jets_classification head needs.
# Unlisted configs take the default 3-class (bjets/cjets/ujets) file.
FIXTURE_FLAVOUR: dict[str, str] = {
    "GN3/GN3V00": "gn3",  # 6-class ghost*
    "GN3/GN3_v00": "gn3",
    "GN3EPCLV01": "gn3",
    "GN3/GN3_Charge": "gn3",  # overlays inherit their base's classes
    "GN3/GN3_Hybrid": "gn3",
    "GN3/GN3_SoftE": "gn3",
    "finetune/finetune_gn3large": "gn3",
    "finetune/finetune_gn3large_new_head": "gn3",
    "GN3/GN3_baseline": "taus",  # 4-class incl. taujets
    "GN3/GN3_baseline_loose": "taus",
    "GN3/GN3_dR": "taus",
    "GN3/GN3_flow": "taus",
    "GN3/GN3_LepID_SMT": "taus",
    "GN3/GN3_tracklabel": "taus",
}

# Configs no synthetic fixture can yet serve, and exactly what is missing. Each
# entry is a promise to a future fixture author, not a permanent excuse.
NO_FIXTURE: dict[str, str] = {
    "legacy/Dipz": "no super_tracks stream in write_dummy_file",
    "hitz": "no hits stream in write_dummy_file",
    "GN2/GN2emu": "no soft-muon global stream in write_dummy_file",
    "GN2/GN2XE": "config truncates tracks to 100; the fixture writes 40",
    "GN2/GN2X_qcdsplit": "fixture flow stream lacks the flow_* field prefix",
    "GN3X": "fixture flow stream lacks the flow_* field prefix",
    "legacy/dips": "labels on raw HadronConeExclTruthLabelID; fixture writes PDG-like values",
    "gn2v2-opendata": "sources its files through input_samples, not data.train_file",
    # measured, not guessed — see the diagnosis in experiment 53
    "GN3EPCLV01": "no global stream in write_dummy_file",
    "GN3/GN3_SoftE": "no global stream in write_dummy_file",
    "GN3/GN3_tracklabel": "no tracks.ftagTruthSourceLabel in write_dummy_file",
    # the finetune templates need a warm start and a multi-epoch schedule, so
    # fast_dev_run (which pins max_epochs to 1) cannot drive them. They are
    # trained through properly by test_finetune_templates.py instead.
    "finetune/finetune_gn3large": "multi-stage schedule needs max_epochs>=5 "
    "and a checkpoint — trained by test_finetune_templates.py",
    "finetune/finetune_gn3large_new_head": "adds a head on jets.large_r_flavour_label, "
    "absent from write_dummy_file — trained by test_finetune_templates.py",
}

# Optional extras. Present in the shipped image (setup/Dockerfile installs both
# --group dev ".[muP]" and ".[root]"), so these skip only on a bare local env.
EXTRAS: dict[str, tuple[str, str]] = {
    "readers/easyjet_flavour": ("awkward", "salt[root]"),
    "readers/ftag1lite_empflow": ("awkward", "salt[root]"),
    "readers/easyjet_events": ("awkward", "salt[root]"),
    "readers/ftag1lite": ("awkward", "salt[root]"),
    "readers/physlite_events": ("awkward", "salt[root]"),
    "readers/physlite_jets": ("awkward", "salt[root]"),
    "GN2/GN2_muP": ("mup", "salt[muP]"),
}


def _discover() -> list[str]:
    """Every shipped file under CONFIG_DIR, relative to it and without suffix."""
    found = sorted(
        p.relative_to(CONFIG_DIR).with_suffix("").as_posix()
        for p in CONFIG_DIR.rglob("*.yaml")
    )
    return [c for c in found if c not in MACHINERY]


def _is_fragment(config: str) -> bool:
    """Whether the config declares only half a run (a reader, or a model, not both).

    Read from the EXPANDED config: an overlay inherits the missing half from the
    base it includes, and judging it on its raw text would call a whole config a
    fragment.
    """
    from salt.config_utils import expand_includes  # noqa: PLC0415

    path = CONFIG_DIR / f"{config}.yaml"
    raw = yaml.safe_load(Path(expand_includes(str(path))).read_text()) or {}
    modules = ((raw.get("data") or {}).get("modules")) or {}
    return not ("model" in raw and "reader" in modules)


ALL_FILES = _discover()
FRAGMENTS = [c for c in ALL_FILES if _is_fragment(c)]
ALL_CONFIGS = [c for c in ALL_FILES if c not in FRAGMENTS]


def _require_extra(config: str) -> None:
    if config not in EXTRAS:
        return
    module, extra = EXTRAS[config]
    pytest.importorskip(module, reason=f"{config} needs `pip install '{extra}'`")


def _stack(config: str) -> list[Path]:
    """Just the config: it declares any bases it needs in its own ``include:``."""
    return [CONFIG_DIR / f"{config}.yaml"]


def _norm_dict_overrides(stack: list[Path], nd_path: Path) -> list[str]:
    """A norm_dict for every Normaliser in the merged stack that ships none.

    The modules live at ``model.init_args.modules``; reading ``model.modules``
    instead yields zero overrides and turns healthy configs into false failures.
    """
    from salt.config_utils import expand_includes  # noqa: PLC0415

    merged: dict = {}
    for path in stack:
        # scan the EXPANDED config: an overlay inherits its Normaliser from the
        # base it includes, so reading the raw file finds nothing and the parse
        # then dies on a required norm_dict
        raw = yaml.safe_load(Path(expand_includes(str(path))).read_text())
        merged = _deep_merge(merged, raw if isinstance(raw, dict) else {})
    model = merged.get("model")
    modules = (model or {}).get("init_args", {}).get("modules") if isinstance(model, dict) else None
    return [
        f"model.modules.{name}.init_args.norm_dict={nd_path}"
        for name, node in (modules or {}).items()
        if isinstance(node, dict)
        and "Normaliser" in str(node.get("class_path", ""))
        and "norm_dict" not in (node.get("init_args") or {})
    ]


def _deep_merge(base: dict, override: dict) -> dict:
    """jsonargparse cross-config semantics: dicts merge, null deletes, lists replace."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for key, value in override.items():
        if value is None and key in out:
            del out[key]
        elif key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> dict[str, dict[str, Path]]:
    """One synthetic H5 + norm/class dict per flavour scheme, built once.

    The three schemes differ only in the ``jets`` group's ``flavour_label``
    class set, which the schema gate compares against each config's declared
    ``class_names``.
    """
    base = tmp_path_factory.mktemp("shipped_configs")
    built: dict[str, dict[str, Path]] = {}
    for flavour, kwargs in (
        ("default", {}),
        ("taus", {"inc_taus": True}),
        ("gn3", {"is_gn3": True}),
    ):
        out = base / flavour
        out.mkdir()
        nd_path, cd_path = out / "norm_dict.yaml", out / "class_dict.yaml"
        write_dummy_norm_dict(nd_path, cd_path, is_gn3=kwargs.get("is_gn3", False))
        h5_path = out / "pp_output_train.h5"
        write_dummy_file(h5_path, nd_path, **kwargs)
        schema_path = out / "schema.yaml"
        save_schema(dump_schema(h5_path), schema_path)
        built[flavour] = {
            "nd": nd_path,
            "cd": cd_path,
            "h5": h5_path,
            "schema": schema_path,
        }
    return built


def _data_for(config: str, fixtures: dict[str, dict[str, Path]]) -> dict[str, Path]:
    return fixtures[FIXTURE_FLAVOUR.get(config, "default")]


def test_tables_name_only_files_that_exist():
    """NO_FIXTURE/EXTRAS/FIXTURE_FLAVOUR name only shipped files."""
    for name, table in (
        ("NO_FIXTURE", NO_FIXTURE),
        ("EXTRAS", EXTRAS),
        ("FIXTURE_FLAVOUR", FIXTURE_FLAVOUR),
    ):
        unknown = sorted(set(table) - set(ALL_FILES))
        assert not unknown, f"{name} names files that do not exist: {unknown}"
    # NO_FIXTURE is a Tier-B statement, and Tier B never runs on a fragment
    overlap = sorted(set(NO_FIXTURE) & set(FRAGMENTS))
    assert not overlap, f"NO_FIXTURE names fragments, which Tier B never runs: {overlap}"


def _reader_node(config: str) -> dict | None:
    """The expanded config's ``data.modules.reader`` node, or None."""
    from salt.config_utils import expand_includes  # noqa: PLC0415

    raw = yaml.safe_load(
        Path(expand_includes(str(CONFIG_DIR / f"{config}.yaml"))).read_text()
    )
    return (((raw or {}).get("data") or {}).get("modules") or {}).get("reader")


READER_FRAGMENTS = [c for c in FRAGMENTS if _reader_node(c) is not None]
MODEL_FRAGMENTS = [c for c in FRAGMENTS if _reader_node(c) is None]


@pytest.mark.parametrize("fragment", READER_FRAGMENTS)
def test_reader_fragment_instantiates(fragment):
    """A reader fragment's reader builds — the gate it gets instead of Tier A.

    A fragment has no model, so there is no plan to compile, but every invariant
    a reader enforces (``unroll`` naming a scalar group, link_branch/target_prefix
    pairing, constituent cuts on a jagged stream naming configured branches, ...)
    is raised from its constructor and is caught here. This is what a fragment is
    for: giving it a model so ``graph validate`` accepts it would be
    reverse-engineering the test rather than modelling the domain.
    """
    _require_extra(fragment)
    from jsonargparse import ArgumentParser  # noqa: PLC0415

    from salt.data.base import Reader  # noqa: PLC0415

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(Reader, "reader")
    cfg = parser.parse_object({"reader": _reader_node(fragment)})
    assert parser.instantiate_classes(cfg).reader is not None


def test_every_model_fragment_is_gated_elsewhere():
    """A model fragment names no reader, so a pairing has to drive it.

    `test_production_configs.py` runs ``ttbar_vs_hh4b_event_tagger`` through the
    full lifecycle on each of its reader pairings, and
    `test_event_tagger_readers.py` holds the cross-format contract. This asserts
    the set has not silently grown past what those cover.
    """
    covered = {"ttbar_vs_hh4b_event_tagger"}
    uncovered = sorted(set(MODEL_FRAGMENTS) - covered)
    assert not uncovered, (
        f"model fragments with no gate: {uncovered}. A config carrying no reader "
        "cannot be passed alone to graph validate — declare its reader pairings in "
        "test_production_configs.py PAIRED."
    )
    stale = sorted(covered - set(MODEL_FRAGMENTS))
    assert not stale, f"named as model fragments but are whole configs: {stale}"


@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_config_plan_compiles(config, fixtures):
    """Tier A — the config plan-compiles in fit, val, test and onnx modes."""
    _require_extra(config)
    data = _data_for(config, fixtures)
    stack = _stack(config)
    argv = ["graph", "validate"]
    for path in stack:
        argv += ["-c", str(path)]
    for mode in ("fit", "val", "test", "onnx"):
        argv += ["--mode", mode]
    # a plan compile is static, but the CLI still instantiates the trainer, so a
    # config shipping accelerator: gpu (a config declaring accelerator: gpu) would fail on a
    # CPU runner for a reason that has nothing to do with its graph. `auto`
    # rather than `cpu`: on a GPU runner these must exercise the GPU path.
    argv += ["--set", "trainer.accelerator=auto"]
    for override in _norm_dict_overrides(stack, data["nd"]):
        argv += ["--set", override]

    rc = salt_main(argv)
    assert rc == 0, (
        f"{config} failed graph validate.\n"
        "  If this is an overlay, declare its bases in the config's own "
        "include: block."
    )


@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_config_fast_dev_run(config, fixtures, tmp_path, request):
    """Tier B — the config builds, forward-passes and steps on synthetic data."""
    _require_extra(config)
    if config in NO_FIXTURE:
        request.node.add_marker(
            pytest.mark.xfail(strict=True, reason=f"no fixture: {NO_FIXTURE[config]}")
        )

    data = _data_for(config, fixtures)
    stack = _stack(config)
    argv = ["fit"]
    for path in stack:
        argv += ["--config", str(path)]
    argv += [
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--trainer.default_root_dir={tmp_path}",
        # auto, not cpu: on a GPU runner these must exercise the GPU path
        "--trainer.accelerator=auto",
        # base ships a default-ON CometLogger; off so the run writes no offline
        # Comet archive (lr_monitor drops with it).
        "--trainer.logger=false",
        "--trainer.fast_dev_run=2",
        # the fixture holds 1000 jets and shipped batch sizes run to 4000, which
        # would yield zero train batches. The wiring under test is batch-size
        # invariant.
        "--data.batch_size=50",
        # shipped configs assume large training machines
        "--data.num_workers=0",
        # null-delete the base ProgressBar: the stock enable_progress_bar=false
        # cannot coexist with a configured bar.
        "--callbacks.progress=null",
    ]
    for override in _norm_dict_overrides(stack, data["nd"]):
        argv.append(f"--{override}")

    rc = salt_main(argv)
    assert rc == 0, f"{config} failed fast_dev_run fit"
