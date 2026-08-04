"""Every shipped config is exercised, by construction.

Discovery is a recursive glob of ``salt/configs`` — a config added anywhere in
the tree is picked up automatically and cannot silently go untested. Two tiers:

**Tier A** (`test_config_plan_compiles`, every config): all-mode
``salt graph validate`` on the config's documented stack. Standalone configs
compile alone; overlay fragments compile stacked on their base per ``STACKS``.
A config that is neither standalone nor in ``STACKS`` fails — that is the check
that stops a new fragment sliding in ungated.

**Tier B** (`test_config_fast_dev_run`): a real 2-batch fit for every config a
synthetic fixture can serve. Configs whose streams the shipped dummy writer
does not produce carry an explicit ``xfail`` naming the missing piece, so the
gap is visible in the report and shrinks as fixtures are added — never a silent
skip.

Adding a config to ``salt/configs`` therefore forces one of three explicit
outcomes: it compiles standalone, it declares its stack, or it declares why it
cannot yet be fitted.
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
MACHINERY = {"base2"}

# Overlay fragments -> the stack that precedes them, from each config's own
# header comment. Paths are relative to salt/configs.
STACKS: dict[str, list[str]] = {
    "GN2/gn2v2-dummy-cutover": ["GN2/gn2v2-dummy"],
    "GN2/gn2v2-dummy-cutover34": ["GN2/gn2v2-dummy"],
    "GN2/gn2v2-dummy-onnx-fold": ["GN2/gn2v2-dummy"],
    "GN2/gn2v2-cluster-override": ["gn2v2-opendata"],
    "GN3/GN3_baseline_loose": ["GN3/GN3_baseline"],
    "GN3/GN3_dR": ["GN3/GN3_baseline"],
    "GN3/GN3_flow": ["GN3/GN3_baseline", "GN3/GN3_baseline_loose"],
    "GN3/GN3_LepID_SMT": [
        "GN3/GN3_baseline",
        "GN3/GN3_baseline_loose",
        "GN3/GN3_flow",
    ],
    "GN3/GN3_tracklabel": [
        "GN3/GN3_baseline",
        "GN3/GN3_baseline_loose",
        "GN3/GN3_flow",
        "GN3/GN3_LepID_SMT",
    ],
    "GN3/GN3_Charge": ["GN3/GN3V00"],
    "GN3/GN3_Hybrid": ["GN3/GN3V00"],
    "GN3/GN3_SoftE": ["GN3/GN3V00"],
    # GN3V00 is the in-repo, production-faithful GN3Large stand-in these
    # templates target — the same base salt/tests/unit/test_finetune_configs.py
    # validates their freeze specs against.
    "finetune/finetune_gn3large": ["GN3/GN3V00"],
    "finetune/finetune_gn3large_new_head": ["GN3/GN3V00"],
    # Reader fragments carry no model: block; each is stacked under the model
    # whose inputs it serves, reader first so the model's data: block wins on
    # everything but the reader itself.
    "readers/easyjet_hh4b": ["readers/easyjet_hh4b_ttbar"],
    "readers/ftag1lite": ["readers/ftag1lite_empflow"],
    "readers/physlite": ["readers/ftag1lite_empflow"],
}

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
    "event_classifier": "no events/objects streams in write_dummy_file",
    "GN2/GN2emu": "no soft-muon global stream in write_dummy_file",
    "GN2/GN2XE": "config truncates tracks to 100; the fixture writes 40",
    "GN2/GN2X_qcdsplit": "fixture flow stream lacks the flow_* field prefix",
    "GN3X": "fixture flow stream lacks the flow_* field prefix",
    "legacy/dips": "labels on raw HadronConeExclTruthLabelID; fixture writes PDG-like values",
    "gn2v2-opendata": "sources its files through input_samples, not data.train_file",
    "GN2/gn2v2-cluster-override": "inherits gn2v2-opendata's input_samples sourcing",
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
    "readers/easyjet_hh4b_ttbar": ("awkward", "salt[root]"),
    "readers/ftag1lite_empflow": ("awkward", "salt[root]"),
    "readers/easyjet_hh4b": ("awkward", "salt[root]"),
    "readers/ftag1lite": ("awkward", "salt[root]"),
    "readers/physlite": ("awkward", "salt[root]"),
    "GN2/GN2_muP": ("mup", "salt[muP]"),
}


def _discover() -> list[str]:
    """Every shipped config, as a path relative to CONFIG_DIR without suffix."""
    found = sorted(
        p.relative_to(CONFIG_DIR).with_suffix("").as_posix()
        for p in CONFIG_DIR.rglob("*.yaml")
    )
    return [c for c in found if c not in MACHINERY]


ALL_CONFIGS = _discover()


def _require_extra(config: str) -> None:
    if config not in EXTRAS:
        return
    module, extra = EXTRAS[config]
    pytest.importorskip(module, reason=f"{config} needs `pip install '{extra}'`")


def _stack(config: str) -> list[Path]:
    return [CONFIG_DIR / f"{c}.yaml" for c in [*STACKS.get(config, []), config]]


def _norm_dict_overrides(stack: list[Path], nd_path: Path) -> list[str]:
    """A norm_dict for every Normaliser in the merged stack that ships none.

    The modules live at ``model.init_args.modules``; reading ``model.modules``
    instead yields zero overrides and turns healthy configs into false failures.
    """
    merged: dict = {}
    for path in stack:
        raw = yaml.safe_load(path.read_text())
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


def test_every_fragment_declares_a_stack():
    """STACKS and NO_FIXTURE name only configs that exist — no stale entries."""
    unknown = sorted(set(STACKS) - set(ALL_CONFIGS))
    assert not unknown, f"STACKS names configs that do not exist: {unknown}"
    unknown = sorted(set(NO_FIXTURE) - set(ALL_CONFIGS))
    assert not unknown, f"NO_FIXTURE names configs that do not exist: {unknown}"
    unknown = sorted(set(EXTRAS) - set(ALL_CONFIGS))
    assert not unknown, f"EXTRAS names configs that do not exist: {unknown}"
    unknown = sorted(set(FIXTURE_FLAVOUR) - set(ALL_CONFIGS))
    assert not unknown, f"FIXTURE_FLAVOUR names configs that do not exist: {unknown}"


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
    # config shipping accelerator: gpu (gn2v2-cluster-override) would fail on a
    # CPU runner for a reason that has nothing to do with its graph
    argv += ["--set", "trainer.accelerator=cpu"]
    for override in _norm_dict_overrides(stack, data["nd"]):
        argv += ["--set", override]

    rc = salt_main(argv)
    assert rc == 0, (
        f"{config} failed graph validate.\n"
        f"  stack: {[p.name for p in stack]}\n"
        "  If this is an overlay fragment, add its documented base stack to "
        "STACKS in this file."
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
        "--trainer.accelerator=cpu",
        # base2 ships a default-ON CometLogger; off so the run writes no offline
        # Comet archive (lr_monitor drops with it).
        "--trainer.logger=false",
        "--trainer.fast_dev_run=2",
        # the fixture holds 1000 jets and shipped batch sizes run to 4000, which
        # would yield zero train batches. The wiring under test is batch-size
        # invariant.
        "--data.batch_size=50",
        # shipped configs assume large training machines
        "--data.num_workers=0",
        # null-delete the base2 ProgressBar: the stock enable_progress_bar=false
        # cannot coexist with a configured bar.
        "--callbacks.progress=null",
    ]
    for override in _norm_dict_overrides(stack, data["nd"]):
        argv.append(f"--{override}")

    rc = salt_main(argv)
    assert rc == 0, f"{config} failed fast_dev_run fit"
