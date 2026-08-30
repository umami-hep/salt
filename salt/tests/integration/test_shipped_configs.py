"""Every shipped config is exercised, by construction.

Discovery is a recursive glob of ``salt/configs`` — a config added anywhere in
the tree is picked up automatically and cannot silently go untested. A shipped
file is either a **config** (has a ``model:`` AND a ``data.modules.reader`` —
a whole run) or one of three fragment kinds, judged on the EXPANDED
(``include:``-resolved) config: a **reader fragment** (has a reader, no
model), a **model fragment** (has a model, no reader), or an **overlay**
(neither — a pure data-behaviour patch, e.g. ``readers/ftag1lite_streaming``,
stacked on top of a reader fragment + model at fit time). None of the three
has a plan to compile, so none is passed alone to ``graph validate``: reader
fragments are gated by `test_reader_fragment_instantiates` below plus
`test_event_tagger_readers.py` (cross-format contract); the one model
fragment (``ttbar_vs_hh4b_event_tagger``'s pairings) is gated by matrix row 6
(``event_tagger_easyjet``, ``pipeline.py`` ``FEEDS``); overlays are named in
``OVERLAY_GATED_BY`` with a reason, checked for completeness by
`test_every_overlay_is_gated`.

Two tiers: **Tier A** — all-mode ``salt graph validate`` for every whole
config, plus a ``salt graph plot --mode fit`` render (all 38 shipped files get
some Tier-A-level gate; the ~32 whole configs get validate+plot, the
remaining ~6 fragments/overlays get the instantiation/completeness checks
above). **Tier B** — a real 2-batch fit for every config a synthetic fixture
can serve and that has no matrix row (a matrix fit strictly dominates a
fast_dev_run, so a config fed by a row skips here instead, naming the row —
§4.4); configs the dummy writer cannot serve AND have no matrix row carry an
explicit ``xfail`` naming the missing piece — never a silent skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.schema import dump_schema, save_schema
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict
from salt.tests.integration.pipeline import MATRIX

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
# gn2v2-opendata/GN3EPCLV01/GN3X/hitz and the two finetune templates used to be
# here too (their fixture gaps are real), but they now have a matrix row
# (pipeline.py MATRIX) that feeds them properly — MATRIX_ROW_FOR below routes
# them to a "covered by row X" skip instead (§4.4), so they are not also an
# xfail-with-an-excuse here.
NO_FIXTURE: dict[str, str] = {
    "legacy/Dipz": "no super_tracks stream in write_dummy_file",
    "GN2/GN2emu": "no soft-muon global stream in write_dummy_file",
    "GN2/GN2XE": "config truncates tracks to 100; the fixture writes 40",
    "GN2/GN2X_qcdsplit": "fixture flow stream lacks the flow_* field prefix",
    "legacy/dips": "labels on raw HadronConeExclTruthLabelID; fixture writes PDG-like values",
    "GN3/GN3_SoftE": "no global stream in write_dummy_file",
    "GN3/GN3_tracklabel": "no tracks.ftagTruthSourceLabel in write_dummy_file",
}

# config_relpath -> matrix test_name, for every shipped config the matrix
# (pipeline.py MATRIX) feeds. Tier B skips these (§4.4): the matrix fit
# (max_epochs=1, limit_batches=2, a real checkpoint) strictly dominates a
# fast_dev_run, so running both here too is duplicated seconds and two owners
# for one config's depth — exactly what the consolidation removes.
MATRIX_ROW_FOR: dict[str, str] = {row.config: row.test_name for row in MATRIX}

# Overlays (fragments with neither a model nor a reader — a pure
# data-behaviour patch stacked on top of a reader fragment + model at fit
# time) cannot be Tier-A validated or Tier-B fit alone. Each must be named
# here with why. See `test_every_overlay_is_gated`.
OVERLAY_GATED_BY: dict[str, str] = {
    "readers/ftag1lite_streaming": "needs a manifest and real DAOD_FTAG1LITE files to build an "
    "IterableSaltDataset — no synthetic fixture emits that corpus shape, so it stays floor-only "
    "(this table entry) rather than Tier A/B.",
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
        p.relative_to(CONFIG_DIR).with_suffix("").as_posix() for p in CONFIG_DIR.rglob("*.yaml")
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
    """NO_FIXTURE/EXTRAS/FIXTURE_FLAVOUR/OVERLAY_GATED_BY name only shipped files."""
    for name, table in (
        ("NO_FIXTURE", NO_FIXTURE),
        ("EXTRAS", EXTRAS),
        ("FIXTURE_FLAVOUR", FIXTURE_FLAVOUR),
        ("OVERLAY_GATED_BY", OVERLAY_GATED_BY),
    ):
        unknown = sorted(set(table) - set(ALL_FILES))
        assert not unknown, f"{name} names files that do not exist: {unknown}"
    # NO_FIXTURE is a Tier-B statement, and Tier B never runs on a fragment
    overlap = sorted(set(NO_FIXTURE) & set(FRAGMENTS))
    assert not overlap, f"NO_FIXTURE names fragments, which Tier B never runs: {overlap}"
    # a config fed by the matrix gets a "covered by row X" skip (§4.4), not an
    # xfail-with-an-excuse — the two tables must stay disjoint
    both = sorted(set(NO_FIXTURE) & set(MATRIX_ROW_FOR))
    assert not both, f"named in both NO_FIXTURE and fed by a matrix row: {both}"


def _expanded(config: str) -> dict:
    """The expanded (``include:``-resolved) top-level config mapping."""
    from salt.config_utils import expand_includes  # noqa: PLC0415

    raw = yaml.safe_load(Path(expand_includes(str(CONFIG_DIR / f"{config}.yaml"))).read_text())
    return raw or {}


def _reader_node(config: str) -> dict | None:
    """The expanded config's ``data.modules.reader`` node, or None."""
    return ((_expanded(config).get("data") or {}).get("modules") or {}).get("reader")


def _has_model(config: str) -> bool:
    """Whether the expanded config declares a ``model:`` section."""
    return "model" in _expanded(config)


# Fragments split three ways on the EXPANDED config (an overlay declares
# neither a reader nor a model of its own — it patches data behaviour and is
# stacked on top of both at fit time; see OVERLAY_GATED_BY above).
READER_FRAGMENTS = [c for c in FRAGMENTS if _reader_node(c) is not None]
_NO_READER = [c for c in FRAGMENTS if _reader_node(c) is None]
MODEL_FRAGMENTS = [c for c in _NO_READER if _has_model(c)]
OVERLAYS = [c for c in _NO_READER if not _has_model(c)]


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

    Matrix row 6 (``event_tagger_easyjet``, ``test_pipeline.py``) runs
    ``ttbar_vs_hh4b_event_tagger`` through the full lifecycle paired with its
    ``readers/easyjet_events`` fragment (``pipeline.py`` ``FEEDS``), and
    `test_event_tagger_readers.py` holds the cross-format contract. This asserts
    the set has not silently grown past what those cover.
    """
    covered = {"ttbar_vs_hh4b_event_tagger"}
    uncovered = sorted(set(MODEL_FRAGMENTS) - covered)
    assert not uncovered, (
        f"model fragments with no gate: {uncovered}. A config carrying no reader "
        "cannot be passed alone to graph validate — declare its reader pairing as a "
        "matrix row in pipeline.py's MATRIX/FEEDS."
    )
    stale = sorted(covered - set(MODEL_FRAGMENTS))
    assert not stale, f"named as model fragments but are whole configs: {stale}"


def test_every_overlay_is_gated():
    """A config with neither a model nor a reader must be named in OVERLAY_GATED_BY.

    Discovered live: ``readers/ftag1lite_streaming`` sets ``data.iterable`` /
    ``block_rows`` / ... but declares no reader and no model of its own, so it
    used to be silently swept into "model fragment" by a two-way split (any
    fragment with no reader was called a model fragment, whether or not it
    actually had a model) — the first time this file ran in CI that would have
    made `test_every_model_fragment_is_gated_elsewhere` fail on a config that
    was never a model fragment to begin with. The three-way split fixes the
    classification; this assertion keeps the overlay set from silently growing
    past what is a documented, floor-only gap.
    """
    uncovered = sorted(set(OVERLAYS) - set(OVERLAY_GATED_BY))
    assert not uncovered, (
        f"overlays with no recorded reason: {uncovered}. A config carrying neither a "
        "reader nor a model cannot be passed alone to graph validate or fast_dev_run — "
        "name it in OVERLAY_GATED_BY with why."
    )
    stale = sorted(set(OVERLAY_GATED_BY) - set(OVERLAYS))
    assert not stale, f"named in OVERLAY_GATED_BY but are not overlays: {stale}"


@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_config_plan_compiles(config, fixtures, tmp_path):
    """Tier A — the config plan-compiles in fit/val/test/onnx modes, then renders fit."""
    _require_extra(config)
    data = _data_for(config, fixtures)
    stack = _stack(config)

    sets: list[str] = ["--set", "trainer.accelerator=auto"]
    for override in _norm_dict_overrides(stack, data["nd"]):
        sets += ["--set", override]

    argv = ["graph", "validate"]
    for path in stack:
        argv += ["-c", str(path)]
    for mode in ("fit", "val", "test", "onnx"):
        argv += ["--mode", mode]
    # a plan compile is static, but the CLI still instantiates the trainer, so a
    # config shipping accelerator: gpu (a config declaring accelerator: gpu) would fail on a
    # CPU runner for a reason that has nothing to do with its graph. `auto`
    # rather than `cpu`: on a GPU runner these must exercise the GPU path.
    rc = salt_main([*argv, *sets])
    assert rc == 0, (
        f"{config} failed graph validate.\n"
        "  If this is an overlay, declare its bases in the config's own "
        "include: block."
    )

    # `graph plot` renders the fit-mode plan — folded into Tier A so a DOT-render
    # regression surfaces on every shipped config, not just the 6 flagships that
    # used to get this leg in test_production_configs.py (§7 ruling 3).
    plot_argv = ["graph", "plot"]
    for path in stack:
        plot_argv += ["-c", str(path)]
    out_path = tmp_path / f"{config.replace('/', '_')}.dot"
    plot_argv += ["--mode", "fit", "-o", str(out_path)]
    assert salt_main([*plot_argv, *sets]) == 0, f"{config}: graph plot failed"
    assert out_path.is_file(), f"{config}: graph plot wrote nothing to {out_path}"


@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_config_fast_dev_run(config, fixtures, tmp_path, request):
    """Tier B — the config builds, forward-passes and steps on synthetic data.

    Skips any config with a matrix row (§4.4): the matrix fit strictly
    dominates a fast_dev_run (a real checkpoint vs two forward/backward
    steps), so running both here too would be duplicated seconds and two
    owners for one config's depth.
    """
    _require_extra(config)
    if config in MATRIX_ROW_FOR:
        pytest.skip(f"covered by matrix row {MATRIX_ROW_FOR[config]!r} (test_pipeline.py)")
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
