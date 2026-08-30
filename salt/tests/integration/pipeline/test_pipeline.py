"""The declarative config-lifecycle matrix: mechanism, fixtures, and tests.

Owns ``MATRIX``, ``FEEDS``, the feeder builders, ``run_row``/``run_eval``/
``run_export`` and their session-scoped artifact cache, the pytest surface
(a compile+plot floor leg every row gets regardless of ``fit``, plus three
parametrized lifecycle legs: fit runs when the row declares ``fit=True``;
eval/export additionally skip when the row declares ``do_eval=False``/
``do_onnx=False``), the xfail table lookup, the matrix<->discovery
completeness checks (see ``FRAGMENTS`` below), the two residual
finetune-template assertions (claims about the chained artifacts that a
matrix row cannot itself express), and
the regression/gaussian-regression per-config semantics (de-scale, doubled-
column and ONNX-rank assertions) that are per-config and cannot be expressed
by the generic runner.

Discovery is a recursive glob of ``salt/configs`` (``_discover``) — a config
added anywhere in the tree is picked up automatically and cannot silently go
untested. Every discovered config (minus ``base.yaml``) must be EITHER a
``MATRIX`` row, or a key in ``FRAGMENTS`` (a config that cannot run alone: an
include-target reader fragment, a fragment paired via ``FEEDS``, or a
data-behaviour overlay) — see ``test_every_discovered_config_is_placed``. A
row that has no synthetic fixture the fit lifecycle can serve declares
``fit=False``: it still gets the compile+plot floor leg (every row does),
just not the fit/eval/export legs.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import onnx
import pytest

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.outputs.sinks.onnx import make_session
from salt.schema import dump_schema, save_schema
from salt.testing.datagen import compute_norm_dict, load_pipeline
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

RECIPES_DIR = Path(__file__).resolve().parents[3] / "testing" / "datagen" / "recipes"
GOLDEN_DIR = Path(__file__).resolve().parents[2] / "_fixtures" / "output_goldens"

# base.yaml is auto-loaded machinery, never a model in its own right.
_MACHINERY = {"base"}

# cpu_always: every row uses --trainer.accelerator=auto on synthetic data, so
# the whole matrix is CPU-safe and must run on every CI invocation (the
# tests/integration/ GPU-skip in conftest.py would otherwise hide it on a
# CPU box).
pytestmark = [pytest.mark.cpu_always]

# ---------------------------------------------------------------------------
# schema (§1.1-1.2)
# ---------------------------------------------------------------------------

Feed = tuple[Literal["recipe", "dummy", "root"], str | None]
"""``("recipe", name)`` | ``("dummy", flavour)`` | ``("root", fragment | None)``."""


@dataclass(frozen=True)
class Row:
    """One matrix row: ``(test_name, config_relpath, do_eval, do_onnx, train_args, fit)``.

    ``fit`` (default ``True``): whether this row gets the fit/eval/export
    lifecycle legs. Every row — ``fit=True`` or not — always gets the
    compile+plot floor leg (``test_compile_plot``): a config with no
    synthetic fixture the lifecycle can run on (``fit=False``) is still a
    real shipped config, and this is what proves its plan compiles and
    renders even though no checkpoint is produced.
    """

    test_name: str
    config: str
    do_eval: bool
    do_onnx: bool
    train_args: tuple[str, ...] = ()
    fit: bool = True


@dataclass
class Artifacts:
    """What a fit leg (and, once run, eval/export) produced for one row."""

    ckpt: Path
    saved_config: Path
    root_dir: Path
    eval_h5: Path | None = None
    onnx: Path | None = None


class LegFailedError(RuntimeError):
    """A row's leg did not complete.

    ``producer`` names the row whose failure is the root cause — itself,
    unless a ``{ckpt:NAME}``/``{config:NAME}`` dependency failed first.
    """

    def __init__(self, name: str, leg: str, reason: str, producer: str | None = None) -> None:
        self.name = name
        self.leg = leg
        self.reason = reason
        self.producer = producer or name
        super().__init__(reason)


class RootDepsMissingError(RuntimeError):
    """A ROOT-fed row's fixture needs ``pip install 'salt[root]'`` (uproot + awkward)."""


# ---------------------------------------------------------------------------
# the matrix (§1.2)
# ---------------------------------------------------------------------------

MATRIX: list[Row] = [
    # -- production flagships (6) --------------------------------------
    Row(
        "gn2v2_opendata",
        "gn2v2-opendata",
        True,
        True,
        (
            # pipeline #15649957: SystemExit 2 on every leg — these three were
            # missing `.init_args`, so jsonargparse rejected the override.
            "--data.modules.input_samples.init_args.files.train={h5}",
            "--data.modules.input_samples.init_args.files.val={h5}",
            "--data.modules.input_samples.init_args.files.test={h5}",
        ),
    ),
    Row("gn3epclv01", "GN3EPCLV01", True, True, ()),
    Row("gn3x", "GN3X", True, True, ()),
    Row("hitz", "hitz", True, True, ()),
    Row("maskformer", "MaskFormer", True, True, ()),
    Row(
        "event_tagger_easyjet",
        "ttbar_vs_hh4b_event_tagger",
        True,
        False,
        ("--config {fragment}", "--data.batch_size=2"),
    ),
    # -- reader pairing (1) ----------------------------------------------
    Row(
        "easyjet_flavour",
        "readers/easyjet_flavour",
        False,
        False,
        (
            "--data.train_file={root}",
            "--data.val_file={root}",
            "--data.batch_size=2",
        ),
    ),
    # -- regression family (5) -------------------------------------------
    Row("regression", "regression/regression", True, True, ()),
    # do_onnx=True (§7 resolution): "no gaussian handling" was a stale
    # docstring claim, not a property of check_onnx — see check.py.
    Row("regression_gaussian", "regression/regression_gaussian", True, True, ()),
    Row("regression_weighted", "regression/regression_weighted", True, True, ()),
    Row("nan_regression", "regression/nan_regression", True, True, ()),
    Row("regression_multi_target", "regression/regression_multi_target", True, True, ()),
    # -- fine-tuning (3) — the reason train_args exists -------------------
    Row("gn3v00_base", "GN3/GN3V00", False, False, ("--trainer.max_epochs=1",)),
    Row(
        "finetune_same_heads",
        "finetune/finetune_gn3large",
        False,
        False,
        (
            "--config {config:gn3v00_base}",
            "--init_from={ckpt:gn3v00_base}",
            "--trainer.max_epochs=6",
            "--trainer.limit_train_batches=1",
            "--trainer.limit_val_batches=1",
        ),
    ),
    Row(
        "finetune_new_head",
        "finetune/finetune_gn3large_new_head",
        False,
        False,
        (
            "--config {config:gn3v00_base}",
            "--init_from={ckpt:gn3v00_base}",
            "--trainer.max_epochs=6",
            "--trainer.limit_train_batches=1",
            "--trainer.limit_val_batches=1",
        ),
    ),
    # -- completeness rows (§2): every remaining shipped config. `fit=True`
    # rows get real fit legs on top of the compile+plot floor every row gets;
    # `fit=False` rows carry forward the old NO_FIXTURE reason (or an EXTRAS
    # gap) as the comment — compile+plot only, no synthetic fixture serves
    # the lifecycle.
    # NO_FIXTURE: no soft-muon global stream in write_dummy_file
    Row("gn2emu", "GN2/GN2emu", False, False, fit=False),
    Row("gn2_mup", "GN2/GN2_muP", False, False),
    # NO_FIXTURE: config truncates tracks to 100; the fixture writes 40
    Row("gn2xe", "GN2/GN2XE", False, False, fit=False),
    # NO_FIXTURE: fixture flow stream lacks the flow_* field prefix
    Row("gn2x_qcdsplit", "GN2/GN2X_qcdsplit", False, False, fit=False),
    Row("gn3_baseline", "GN3/GN3_baseline", False, False),
    Row("gn3_baseline_loose", "GN3/GN3_baseline_loose", False, False),
    Row("gn3_charge", "GN3/GN3_Charge", False, False),
    Row("gn3_dr", "GN3/GN3_dR", False, False),
    Row("gn3_flow", "GN3/GN3_flow", False, False),
    Row("gn3_hybrid", "GN3/GN3_Hybrid", False, False),
    Row("gn3_lepid_smt", "GN3/GN3_LepID_SMT", False, False),
    # NO_FIXTURE: no global stream in write_dummy_file
    Row("gn3_softe", "GN3/GN3_SoftE", False, False, fit=False),
    # NO_FIXTURE: no tracks.ftagTruthSourceLabel in write_dummy_file
    Row("gn3_tracklabel", "GN3/GN3_tracklabel", False, False, fit=False),
    Row("gn3_v00", "GN3/GN3_v00", False, False),
    # NO_FIXTURE: labels on raw HadronConeExclTruthLabelID; fixture writes PDG-like values
    Row("dips", "legacy/dips", False, False, fit=False),
    # NO_FIXTURE: no super_tracks stream in write_dummy_file
    Row("dipz", "legacy/Dipz", False, False, fit=False),
    Row("dl1", "legacy/DL1", False, False),
    # EXTRAS(awkward)-gated; no synthetic FTAG1LITE POOL fixture
    Row("ftag1lite_empflow", "readers/ftag1lite_empflow", False, False, fit=False),
]

# config_relpath -> feeder. Replaces RECIPES/PAIRED/FIXTURE_FLAVOUR/the fixture
# half of NO_FIXTURE, which used to exist across two files.
FEEDS: dict[str, Feed] = {
    "gn2v2-opendata": ("recipe", "flavour_tagger"),
    "GN3EPCLV01": ("recipe", "flavour_tagger_global"),
    "GN3X": ("recipe", "flavour_tagger"),
    "hitz": ("recipe", "hits_regression"),
    "MaskFormer": ("recipe", "maskformer_truth_hadron"),
    "ttbar_vs_hh4b_event_tagger": ("root", "readers/easyjet_events"),
    "readers/easyjet_flavour": ("root", None),
    "regression/regression": ("dummy", "regression"),
    "regression/regression_gaussian": ("dummy", "regression"),
    "regression/regression_weighted": ("dummy", "regression"),
    "regression/nan_regression": ("dummy", "default"),
    "regression/regression_multi_target": ("dummy", "default"),
    "GN3/GN3V00": ("dummy", "gn3"),
    "finetune/finetune_gn3large": ("dummy", "gn3"),
    "finetune/finetune_gn3large_new_head": ("dummy", "gn3_large_r"),
    # completeness rows (§2) — configs not listed above take the plain
    # 3-class "default" fixture (compile+plot never reads the file, only the
    # norm_dict path, so a schema mismatch on a fit=False row is harmless).
    "GN2/GN2emu": ("dummy", "default"),
    "GN2/GN2_muP": ("dummy", "default"),
    "GN2/GN2XE": ("dummy", "default"),
    "GN2/GN2X_qcdsplit": ("dummy", "default"),
    "GN3/GN3_baseline": ("dummy", "taus"),
    "GN3/GN3_baseline_loose": ("dummy", "taus"),
    "GN3/GN3_Charge": ("dummy", "gn3"),
    "GN3/GN3_dR": ("dummy", "taus"),
    "GN3/GN3_flow": ("dummy", "taus"),
    "GN3/GN3_Hybrid": ("dummy", "gn3"),
    "GN3/GN3_LepID_SMT": ("dummy", "taus"),
    "GN3/GN3_SoftE": ("dummy", "gn3"),
    "GN3/GN3_tracklabel": ("dummy", "taus"),
    "GN3/GN3_v00": ("dummy", "gn3"),
    "legacy/dips": ("dummy", "default"),
    "legacy/Dipz": ("dummy", "default"),
    "legacy/DL1": ("dummy", "default"),
    "readers/ftag1lite_empflow": ("dummy", "default"),
}

# config_relpath -> (importable module, pip extra) for a config whose reader
# or model needs an optional dependency the bare image lacks. ROOT-fed MATRIX
# rows (easyjet_flavour, event_tagger_easyjet) already self-gate via
# `_root_deps()`/`RootDepsMissingError` (the "root" FEEDS kind) and don't need
# an entry here; this table is for configs fed some other way (dummy H5)
# whose reader/model class still has a hard import on an optional package,
# plus the reader-fragment entries in FRAGMENTS that
# `test_reader_fragment_instantiates` gates directly.
EXTRAS: dict[str, tuple[str, str]] = {
    "GN2/GN2_muP": ("mup", "salt[muP]"),
    "readers/ftag1lite_empflow": ("awkward", "salt[root]"),
    "readers/easyjet_events": ("awkward", "salt[root]"),
    "readers/ftag1lite": ("awkward", "salt[root]"),
    "readers/physlite_events": ("awkward", "salt[root]"),
    "readers/physlite_jets": ("awkward", "salt[root]"),
}


def _require_extra(config: str) -> None:
    """Skip if ``config``'s optional dependency (EXTRAS) isn't importable."""
    entry = EXTRAS.get(config)
    if entry is None:
        return
    module, extra = entry
    pytest.importorskip(module, reason=f"{config} needs `pip install '{extra}'`")


# Fragments that cannot run alone (an include-target reader fragment, a
# fragment paired via FEEDS, or a pure data-behaviour overlay) — every
# discovered config is either a MATRIX row or a key here (§ discovery,
# test_every_discovered_config_is_placed). The value records how it is
# exercised: "included" (>=1 row's config include:s it — checked via each
# row's own top-level include: block), "paired" (a FEEDS entry references it
# by name), or a free-text reason (floor-only: existence-checked, not
# otherwise exercised by the matrix — the ftag1lite_streaming overlay, ported
# from the old OVERLAY_GATED_BY table, and the two PHYSLITE reader fragments,
# which are documented alternates with no synthetic fixture to pair them).
FRAGMENTS: dict[str, str] = {
    "readers/easyjet_events": "paired",
    "readers/ftag1lite": "included",
    "readers/physlite_events": (
        "documented alternate leg for ttbar_vs_hh4b_event_tagger (the PHYSLITE "
        "side of the easyjet/physlite reader-parity pair, see the config's own "
        "docstring) — no synthetic PHYSLITE/POOL fixture exists to pair it into "
        "a matrix row, so it is exercised only via test_reader_fragment_instantiates."
    ),
    "readers/physlite_jets": (
        "documented JET-axis PHYSLITE reader (ElementLink-dereferenced tracks) — "
        "no synthetic PHYSLITE/POOL fixture exists to pair it into a matrix row, "
        "so it is exercised only via test_reader_fragment_instantiates."
    ),
    "readers/ftag1lite_streaming": (
        "overlay: needs a manifest and real DAOD_FTAG1LITE files to build an "
        "IterableSaltDataset — no synthetic fixture emits that corpus shape, so "
        "it stays floor-only (this table entry) rather than a matrix row."
    ),
}

# The GPU subset (§5.2): green-on-CPU rows only — gn3epclv01/gn3x/hitz are
# deliberately excluded (a strict xfail whose failure was never root-caused
# could plausibly XPASS on a different device, and a strict XPASS is red).
GPU_ROWS: set[str] = {"gn2v2_opendata", "maskformer", "gn3v00_base", "finetune_same_heads"}

# (test_name, leg) -> reason. strict=True: an unexpected PASS is a FAILURE, so
# fixing the product forces the entry to be deleted. Study-record only (§7
# resolution #4) — no tracker links until an upstream MR is on the table.
KNOWN_FAILURES: dict[tuple[str, str], str] = {
    ("gn3epclv01", "fit"): (
        "salt fit rc=1, no traceback captured; not caused by the consolidation "
        "(exp 55 run 3, exp 59 both saw it pre-W6). Root cause not yet isolated."
    ),
    ("gn3x", "fit"): (
        "salt fit rc=1, unchanged since the first lifecycle run (exp 55 run 1) — "
        "looks like a genuine config/model bug, not a harness artefact."
    ),
    ("hitz", "fit"): ("salt fit rc=1, survives the recipe-schema fixes from exp 55 run 2."),
    ("maskformer", "export"): (
        "check_onnx: NaN in torch output for MFv2_leading_objects_pt at "
        "lengths={'tracks': 0} and {'tracks': 1} — a real torch/ONNX zero-token "
        "divergence, not a harness bug. Fit and eval are green."
    ),
}

# A row that do_eval=True but has no committed output golden (generate_goldens.py)
# names itself here with a reason, so the eval leg's schema-parity assertion
# knows to skip rather than error looking for a file that will never exist.
NO_GOLDEN: dict[str, str] = {
    "event_tagger_easyjet": (
        "ttbar_vs_hh4b_event_tagger is ROOT-fed and paired via a reader "
        "fragment — outside the H5 dumb-section golden family generate_goldens.py "
        "captures."
    ),
}

_BY_NAME: dict[str, Row] = {r.test_name: r for r in MATRIX}


def row_by_name(name: str) -> Row:
    """The matrix row named ``name``."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"no matrix row named {name!r}") from None


def _discover() -> list[str]:
    """Every shipped config under ``CONFIG_DIR``, relative path minus suffix, minus base.yaml.

    Recursive (``rglob``) — a config added anywhere in the tree, at any
    depth, is picked up automatically (§ discovery).
    """
    found = sorted(
        p.relative_to(CONFIG_DIR).with_suffix("").as_posix() for p in CONFIG_DIR.rglob("*.yaml")
    )
    return [c for c in found if c not in _MACHINERY]


def _config_includes(config: str) -> set[str]:
    """``config``'s own top-level ``include:`` list, resolved to config_relpaths.

    Deliberately NOT ``expand_includes`` (which recurses and would also
    surface transitive includes) — this is "does THIS config's own
    ``include:`` block name it", matching the docstring convention
    ("declare its bases in the config's own include: block").
    """
    import yaml

    raw = yaml.safe_load((CONFIG_DIR / f"{config}.yaml").read_text()) or {}
    resolved: set[str] = set()
    for inc in raw.get("include") or []:
        candidate = (CONFIG_DIR / config).parent / inc
        if not candidate.is_file():
            candidate = CONFIG_DIR / inc
        resolved.add(candidate.relative_to(CONFIG_DIR).with_suffix("").as_posix())
    return resolved


_TOKEN_RE = re.compile(r"\{(\w+)(?::(\w+))?\}")


def dependencies_of(row: Row) -> set[str]:
    """Row names ``row.train_args`` references via ``{ckpt:NAME}``/``{config:NAME}``."""
    deps: set[str] = set()
    for template in row.train_args:
        for match in _TOKEN_RE.finditer(template):
            key, arg = match.group(1), match.group(2)
            if key in {"ckpt", "config"} and arg:
                deps.add(arg)
    return deps


def golden_path(row: Row) -> Path:
    """The committed output golden this row's eval leg is checked against."""
    return GOLDEN_DIR / f"{Path(row.config).name}.json"


def _load_expanded(config: str) -> dict:
    """A shipped config's include-expanded YAML, as a plain dict."""
    import yaml

    from salt.config_utils import expand_includes

    path = CONFIG_DIR / f"{config}.yaml"
    return yaml.safe_load(Path(expand_includes(str(path))).read_text()) or {}


def _reader_node(config: str) -> dict | None:
    """The expanded config's ``data.modules.reader`` node, or ``None``."""
    return ((_load_expanded(config).get("data") or {}).get("modules") or {}).get("reader")


# ---------------------------------------------------------------------------
# feeders (§1.1: the ``FEEDS`` table drives which of these a row uses)
# ---------------------------------------------------------------------------

_RECIPE_CACHE: dict[str, dict[str, Path]] = {}


def _recipe_context(recipe: str, tmp_path_factory) -> dict[str, Path]:
    """Run a ``salt.testing.datagen`` recipe once; every row using it shares the result."""
    cached = _RECIPE_CACHE.get(recipe)
    if cached is not None:
        return cached
    import yaml

    out = tmp_path_factory.mktemp(f"datagen_{recipe}")
    pipe = load_pipeline(str(RECIPES_DIR / f"{recipe}.yaml"))
    pipe.set_output_dir(out)
    data = pipe.run()
    h5 = out / f"{recipe}.h5"
    if not h5.is_file():
        raise RuntimeError(f"recipe {recipe} wrote no H5")
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    ctx: dict[str, Path] = {"h5": h5, "schema": schema}
    for name in ("norm_dict.yaml", "class_dict.yaml"):
        path = out / name
        if path.is_file():
            ctx[name.split("_")[0]] = path
    # not every recipe ships a NormWriter, but every config with a Normaliser
    # needs one — derive it from the produced arrays
    if "norm" not in ctx:
        nd = out / "norm_dict.yaml"
        nd.write_text(yaml.dump(compute_norm_dict(data), sort_keys=False))
        ctx["norm"] = nd
    _RECIPE_CACHE[recipe] = ctx
    return ctx


def _build_default_dummy(tmp_path_factory) -> dict[str, Path]:
    """The plain 3-class ``write_dummy_file`` fixture — rows 11-12."""
    out = tmp_path_factory.mktemp("dummy_default")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_dummy_norm_dict(nd, cd)
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _build_taus_dummy(tmp_path_factory) -> dict[str, Path]:
    """The 4-class (incl. taujets) ``write_dummy_file`` fixture — the GN3_baseline family.

    The "taus" flavour: same 3-class norm/class dict as the default builder,
    but the H5 itself carries a fourth ``taujets`` flavour_label value.
    """
    out = tmp_path_factory.mktemp("dummy_taus")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_dummy_norm_dict(nd, cd)
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd, inc_taus=True)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _build_regression_dummy(tmp_path_factory) -> dict[str, Path]:
    """The parity fixture plus a ``mass`` jets entry — rows 8-10 (§1.2 "default + mass").

    ``regression``/``regression_weighted`` declare ``mass`` as a jets input
    Feature (a ratio denominator); the other rows in the family share this
    builder too, and simply never ask for the extra column.
    """
    import yaml

    from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

    out = tmp_path_factory.mktemp("dummy_regression")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    raw = yaml.safe_load(nd.read_text())
    raw["jets"]["mass"] = {"mean": round(0.1 * 3, 6), "std": round(1.0 + 0.05 * 3, 6)}
    nd.write_text(yaml.dump(raw, sort_keys=False))
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _build_gn3_dummy(tmp_path_factory) -> dict[str, Path]:
    """The 6-class GN3 ``write_dummy_file`` fixture — rows 13-14."""
    out = tmp_path_factory.mktemp("dummy_gn3")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_dummy_norm_dict(nd, cd, is_gn3=True)
    h5 = out / "pp_output_train.h5"
    write_dummy_file(h5, nd, is_gn3=True)
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": nd, "class_dict": cd}


def _build_gn3_large_r_dummy(tmp_path_factory) -> dict[str, Path]:
    """The ``gn3`` fixture plus a ``large_r_flavour_label`` jets column — row 15.

    ``finetune_gn3large_new_head.yaml`` bolts a head onto exactly that label;
    its absence from the base checkpoint IS the domain shift the template
    demonstrates, so it lives only in this derived fixture, never in the base
    one rows 13-14 share.
    """
    import h5py as _h5py
    import numpy as _np

    from salt.utils.array_utils import join_structured_arrays

    base = _dummy_context("gn3", tmp_path_factory)
    out = tmp_path_factory.mktemp("dummy_gn3_large_r")
    h5 = out / "pp_output_train.h5"
    with _h5py.File(base["h5"]) as src, _h5py.File(h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name, dataset in src.items():
            if name != "jets":
                dst.create_dataset(name, data=dataset[:])
                for key, value in dataset.attrs.items():
                    dst[name].attrs[key] = value
                continue
            jets = dataset[:]
            rng = _np.random.default_rng(42)
            extra = rng.integers(0, 4, size=len(jets)).astype("i4")
            extra = extra.view(_np.dtype([("large_r_flavour_label", "i4")]))
            dst.create_dataset("jets", data=join_structured_arrays([jets, extra]))
            for key, value in dataset.attrs.items():
                dst["jets"].attrs[key] = value
            dst["jets"].attrs["large_r_flavour_label"] = ["hbb", "hcc", "top", "qcd"]
    schema = out / "schema.yaml"
    save_schema(dump_schema(h5), schema)
    return {"h5": h5, "schema": schema, "norm": base["norm"], "class_dict": base["class_dict"]}


_DUMMY_BUILDERS = {
    "default": _build_default_dummy,
    "regression": _build_regression_dummy,
    "gn3": _build_gn3_dummy,
    "gn3_large_r": _build_gn3_large_r_dummy,
    "taus": _build_taus_dummy,
}
_DUMMY_CACHE: dict[str, dict[str, Path]] = {}


def _dummy_context(flavour: str, tmp_path_factory) -> dict[str, Path]:
    cached = _DUMMY_CACHE.get(flavour)
    if cached is not None:
        return cached
    builder = _DUMMY_BUILDERS.get(flavour)
    if builder is None:
        raise ValueError(f"unknown dummy flavour {flavour!r}")
    ctx = builder(tmp_path_factory)
    _DUMMY_CACHE[flavour] = ctx
    return ctx


def _root_deps() -> bool:
    """Whether uproot + awkward are importable (every ROOT fixture needs both)."""
    from importlib.util import find_spec

    return all(find_spec(m) is not None for m in ("uproot", "awkward"))


_ROOT_CACHE: dict[str, dict[str, Path]] = {}


def _root_context(row: Row, fragment: str | None, tmp_path_factory) -> dict[str, Path]:
    cached = _ROOT_CACHE.get(row.config)
    if cached is not None:
        return cached
    if not _root_deps():
        raise RootDepsMissingError(f"{row.test_name} is ROOT-fed; needs `pip install 'salt[root]'`")
    ctx = (
        _paired_root_context(row, fragment, tmp_path_factory)
        if fragment is not None
        else _whole_root_context(row, tmp_path_factory)
    )
    _ROOT_CACHE[row.config] = ctx
    return ctx


def _paired_root_context(row: Row, fragment: str, tmp_path_factory) -> dict[str, Path]:
    """Two synthetic minitrees + a sourced copy of ``fragment`` — row 6."""
    from salt.tests._fixtures.easyjet_minitree import (
        write_jets_norm_dict,
        write_sample_pair,
        write_sourced_fragment,
    )

    out = tmp_path_factory.mktemp(f"root_{row.test_name}")
    signal, background = write_sample_pair(out)
    raw = _load_expanded(fragment)
    sourced = write_sourced_fragment(
        raw, out / "reader.yaml", {"signal": signal, "background": background}
    )
    model = _load_expanded(row.config)
    variables = model["data"]["modules"]["features"]["init_args"]["variables"]["jets"]
    norm = write_jets_norm_dict(out / "norm_dict.yaml", variables)
    return {"fragment": sourced, "norm": norm}


def _whole_root_context(row: Row, tmp_path_factory) -> dict[str, Path]:
    """One synthetic minitree, read via ``--data.train_file``/``--data.val_file`` — row 7."""
    from salt.tests._fixtures.easyjet_minitree import (
        build_fixture_arrays,
        write_jets_norm_dict,
        write_minitree,
    )

    out = tmp_path_factory.mktemp(f"root_{row.test_name}")
    root = write_minitree(out / "data.root", build_fixture_arrays())
    variables = _load_expanded(row.config)["data"]["modules"]["features"]["init_args"]["variables"][
        "jets"
    ]
    norm = write_jets_norm_dict(out / "norm_dict.yaml", variables)
    return {"root": root, "norm": norm}


def _feed_context(row: Row, tmp_path_factory) -> dict[str, Path]:
    """The format-string context (§1.1) available to ``row``'s ``train_args``.

    Gates ``EXTRAS`` first: a config whose reader/model class has a hard
    import on an optional package (mup, or a non-``root``-fed UprootReader
    config) must skip before any fixture-building is attempted, same as the
    ``root``-fed rows already do via ``_root_deps()`` below.
    """
    _require_extra(row.config)
    kind, arg = FEEDS[row.config]
    if kind == "recipe":
        assert arg is not None
        return _recipe_context(arg, tmp_path_factory)
    if kind == "dummy":
        assert arg is not None
        return _dummy_context(arg, tmp_path_factory)
    return _root_context(row, arg, tmp_path_factory)


def _norm_overrides(row: Row, ctx: dict[str, Path]) -> list[str]:
    """``--model.modules.<name>.init_args.norm_dict=<norm>`` for every bare Normaliser.

    Reads the row's own EXPANDED config: an overlay's Normaliser lives in the
    base it includes, and reading the raw file would silently yield none.
    Several configs carry more than one Normaliser (``norm`` plus
    ``norm_global``), so overriding only the first is not enough.
    """
    norm = ctx.get("norm")
    if norm is None:
        return []
    raw = _load_expanded(row.config)
    model = raw.get("model")
    modules = (model or {}).get("init_args", {}).get("modules") if isinstance(model, dict) else None
    return [
        f"--model.modules.{name}.init_args.norm_dict={norm}"
        for name, node in (modules or {}).items()
        if isinstance(node, dict)
        and "Normaliser" in str(node.get("class_path", ""))
        and "norm_dict" not in (node.get("init_args") or {})
    ]


def _norm_set_overrides(row: Row, ctx: dict[str, Path]) -> list[str]:
    """``_norm_overrides`` results as bare ``KEY=VALUE`` (for ``graph``'s ``--set``)."""
    return [o.removeprefix("--") for o in _norm_overrides(row, ctx)]


def _base_data_args(row: Row, ctx: dict[str, Path]) -> list[str]:
    """The per-feed CLI args every row of that feed kind shares.

    Recipe/dummy rows are H5-fed: point the reader at the built file + schema.
    ROOT rows supply their own file wiring via ``train_args`` (``sources:``
    embedded in a paired fragment, or ``{root}``) — the base args here are
    deliberately minimal, so this list must not fight that.
    """
    kind = FEEDS[row.config][0]
    if kind == "root":
        return ["--data.num_workers=0", *_norm_overrides(row, ctx)]
    return [
        f"--data.train_file={ctx['h5']}",
        f"--data.val_file={ctx['h5']}",
        f"--data.modules.reader.init_args.schema={ctx['schema']}",
        # shipped configs assume large training machines
        "--data.num_workers=0",
        # the feeders emit ~1000 rows; shipped batch sizes run to 4000
        "--data.batch_size=50",
        *_norm_overrides(row, ctx),
    ]


def _trainer_args(root: Path) -> list[str]:
    """The shared FIT trainer args (§4.1) — one copy instead of five."""
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


# ---------------------------------------------------------------------------
# train_args expansion (§1.1, §2) — chains into run_row for {ckpt:NAME}/{config:NAME}
# ---------------------------------------------------------------------------


def _expand_train_args(
    row: Row, ctx: dict[str, Path], tmp_path_factory
) -> tuple[list[str], list[str]]:
    """``row.train_args`` with every ``{token}``/``{token:NAME}`` substituted.

    Split into ``(config_argv, override_argv)``: elements that stack an extra
    ``--config`` file (``"--config {fragment}"``, ``"--config {config:NAME}"``)
    go in ``config_argv``; everything else (``--init_from=``, per-row trainer
    overrides, ...) goes in ``override_argv``. ``_do_fit`` stacks
    ``config_argv`` BEFORE this row's own template — a ``{config:NAME}``
    dependency is a producer row's *saved* ``config.yaml``, and salt's fit
    semantics deep-merge later ``--config`` files on top (salt/parser.py), so
    stacking it after the template (or after the shared harness args) would
    let the producer's incidental settings — its own
    ``trainer.default_root_dir``, ``data.train_file``, a null
    ``training_schedule`` — clobber this row's. ``override_argv`` stays last,
    so row-specific overrides still win over both configs and the shared
    harness args.

    Each element may expand to more than one argv token (``"--config {fragment}"``
    -> two), so the whole thing is re-split with :func:`shlex.split` after
    substitution.
    """
    config_argv: list[str] = []
    override_argv: list[str] = []
    for template in row.train_args:
        expanded = _expand_one(row.test_name, template, ctx, tmp_path_factory)
        tokens = shlex.split(expanded)
        (config_argv if template.startswith("--config ") else override_argv).extend(tokens)
    return config_argv, override_argv


def _expand_one(name: str, template: str, ctx: dict[str, Path], tmp_path_factory) -> str:
    def repl(match: re.Match[str]) -> str:
        key, arg = match.group(1), match.group(2)
        if key in {"ckpt", "config"}:
            if arg is None:
                raise ValueError(f"{{{key}}} needs a row name, e.g. {{{key}:gn3v00_base}}")
            try:
                producer = run_row(arg, tmp_path_factory)
            except LegFailedError as exc:
                raise LegFailedError(
                    name, "fit", f"producer row {exc.producer} failed", producer=exc.producer
                ) from exc
            return str(producer.ckpt if key == "ckpt" else producer.saved_config)
        if arg is not None:
            raise ValueError(f"unexpected token {match.group(0)!r} in {name}'s train_args")
        if key not in ctx:
            raise KeyError(
                f"{name}: train_args token {{{key}}} not in this row's feed context ({sorted(ctx)})"
            )
        return str(ctx[key])

    return _TOKEN_RE.sub(repl, template)


# ---------------------------------------------------------------------------
# the floor leg (compile+plot) — every row, ``fit`` or not
# ---------------------------------------------------------------------------


def run_compile_plot(row: Row, tmp_path_factory, tmp_path: Path) -> None:
    """Every row's floor leg: ``graph validate`` (all modes) + ``graph plot --mode fit``.

    Runs regardless of ``row.fit`` — a ``fit=False`` row (no synthetic
    fixture the lifecycle can serve) still proves its plan compiles and
    renders, just with no checkpoint. A chained row (``{config:NAME}`` in its
    ``train_args``) reuses the SAME producer artifact the fit leg would
    (``_expand_train_args``/``run_row``), so a finetune template's compile
    leg genuinely needs its base already fit — this is correct, not
    accidental: the template cannot compile without the base config it warms
    up.

    NOTE: ``--mode`` (``salt/cli.py``) is a plain, non-``append`` argparse
    flag — passing it four times, as the old Tier A did, leaves
    ``mode=onnx`` (the last value wins), not all four. Ported byte-for-byte
    rather than "fixed": this item's scope is completeness + the two named
    bugs (§8), and validating fit/val/test for the first time on ~30 configs
    that have never been checked in those modes is a change with unknown
    blast radius, not something to slip in unannounced here.
    """
    ctx = _feed_context(row, tmp_path_factory)
    config_argv, _ = _expand_train_args(row, ctx, tmp_path_factory)
    sets: list[str] = ["--set", "trainer.accelerator=auto"]
    for override in _norm_set_overrides(row, ctx):
        sets += ["--set", override]

    cfg_path = str(CONFIG_DIR / f"{row.config}.yaml")
    argv = ["graph", "validate", *config_argv, "-c", cfg_path]
    for mode in ("fit", "val", "test", "onnx"):
        argv += ["--mode", mode]
    rc = salt_main([*argv, *sets])
    assert rc == 0, f"{row.test_name} ({row.config}) failed graph validate"

    plot_path = tmp_path / f"{row.test_name}.dot"
    plot_argv = [
        "graph",
        "plot",
        *config_argv,
        "-c",
        cfg_path,
        "--mode",
        "fit",
        "-o",
        str(plot_path),
    ]
    assert salt_main([*plot_argv, *sets]) == 0, f"{row.test_name}: graph plot failed"
    assert plot_path.is_file(), f"{row.test_name}: graph plot wrote nothing to {plot_path}"


_COMPILE_PLOT_PARAMS = [row.test_name for row in MATRIX]


@pytest.mark.parametrize("name", _COMPILE_PLOT_PARAMS)
def test_compile_plot(name, tmp_path_factory, tmp_path):
    """Floor leg — every row's config plan-compiles and its fit-mode DOT renders."""
    row = row_by_name(name)
    try:
        run_compile_plot(row, tmp_path_factory, tmp_path)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.producer != name:
            pytest.skip(f"producer row {exc.producer} failed")
        raise


# ---------------------------------------------------------------------------
# the legs (§4.1) — session-scoped, memoised per row
# ---------------------------------------------------------------------------

# dict[test_name, Artifacts | LegFailedError]. No xdist today (pyproject has
# none, CI runs one pytest process), so a plain module-level dict is correct.
# If xdist is ever added, every matrix row must stay in one module (this
# module already is), and `--dist loadfile` becomes mandatory — this cache
# is not otherwise safe to share across workers.
_ROW_CACHE: dict[str, Artifacts | LegFailedError] = {}


def run_row(name: str, tmp_path_factory) -> Artifacts:
    """Run (or fetch) matrix row ``name`` through the fit leg.

    Memoised — the work happens once, on the first leg (of any row) that
    asks. Raises :class:`LegFailedError`: ``producer == name`` when this row's
    own fit failed, or names the upstream row when a
    ``{ckpt:NAME}``/``{config:NAME}`` dependency failed first.
    """
    cached = _ROW_CACHE.get(name)
    if isinstance(cached, LegFailedError):
        raise cached
    if cached is not None:
        return cached
    row = row_by_name(name)
    try:
        ctx = _feed_context(row, tmp_path_factory)
        config_argv, override_argv = _expand_train_args(row, ctx, tmp_path_factory)
        artifacts = _do_fit(row, ctx, config_argv, override_argv, tmp_path_factory)
    except LegFailedError as exc:
        _ROW_CACHE[name] = exc
        raise
    _ROW_CACHE[name] = artifacts
    return artifacts


def _do_fit(
    row: Row,
    ctx: dict[str, Path],
    config_argv: list[str],
    override_argv: list[str],
    tmp_path_factory,
) -> Artifacts:
    root = tmp_path_factory.mktemp(row.test_name)
    argv = ["fit"]
    # a {config:NAME} dependency is a producer row's *saved* config.yaml —
    # stack it BEFORE this row's own template (docs/tutorials/finetuning.md's
    # worked order: saved base config first, template second), so the
    # template's own settings (training_schedule, the new head, ...) win over
    # the producer's incidental ones on any key both happen to set.
    argv += config_argv
    argv += ["--config", str(CONFIG_DIR / f"{row.config}.yaml")]
    argv += _base_data_args(row, ctx)
    argv += _trainer_args(root)
    # override_argv last: row-specific overrides (finetune epochs/limits, the
    # input_samples fix, ...) must win over both configs and the shared
    # defaults above.
    argv += override_argv
    rc = salt_main(argv)
    if rc != 0:
        raise LegFailedError(row.test_name, "fit", f"salt fit rc={rc}")
    ckpts = sorted(root.rglob("*.ckpt"))
    if not ckpts:
        raise LegFailedError(row.test_name, "fit", f"training wrote no checkpoint under {root}")
    configs = sorted(root.rglob("config.yaml"))
    if not configs:
        raise LegFailedError(row.test_name, "fit", f"training saved no config.yaml under {root}")
    return Artifacts(ckpt=ckpts[0], saved_config=configs[0], root_dir=root)


def run_eval(name: str, tmp_path_factory) -> Path:
    """Run (or fetch) row ``name``'s eval leg: ``salt test`` + schema-parity (§4.2).

    Requires the fit leg's artifacts first (calls :func:`run_row`).
    """
    artifacts = run_row(name, tmp_path_factory)
    if artifacts.eval_h5 is not None:
        return artifacts.eval_h5
    row = row_by_name(name)
    ctx = _feed_context(row, tmp_path_factory)
    argv = [
        "test",
        "--config",
        str(artifacts.saved_config),
        f"--ckpt_path={artifacts.ckpt}",
        "--data.num_workers=0",
        "--trainer.accelerator=auto",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ]
    if "h5" in ctx:
        argv.append(f"--data.test_file={ctx['h5']}")
    rc = salt_main(argv)
    if rc != 0:
        raise LegFailedError(name, "eval", f"salt test rc={rc}")
    evals = sorted(artifacts.ckpt.parent.glob("*__test_*.h5"))
    if not evals:
        raise LegFailedError(name, "eval", f"eval wrote no H5 next to {artifacts.ckpt}")
    eval_h5 = evals[-1]
    artifacts.eval_h5 = eval_h5
    if name not in NO_GOLDEN:
        _assert_eval_h5_matches_golden(eval_h5, row)
    return eval_h5


def run_export(name: str, tmp_path_factory) -> Path:
    """Run (or fetch) row ``name``'s export leg: checked ONNX export (§4.1).

    Deliberately *without* ``--no-check`` — ``rc == 0`` IS the torch<->ONNX
    parity assertion.
    """
    artifacts = run_row(name, tmp_path_factory)
    if artifacts.onnx is not None:
        return artifacts.onnx
    onnx_dir = artifacts.root_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{name}.onnx"
    rc = salt_main([
        "export",
        "--config",
        str(artifacts.saved_config),
        f"--ckpt_path={artifacts.ckpt}",
        f"--output={onnx_path}",
    ])
    if rc != 0:
        raise LegFailedError(name, "export", f"salt export failed rc={rc}")
    if not sorted(onnx_dir.glob("*.onnx")):
        raise LegFailedError(name, "export", f"no ONNX written under {onnx_dir}")
    artifacts.onnx = onnx_path
    return onnx_path


def _assert_eval_h5_matches_golden(eval_h5: Path, row: Row) -> None:
    """The written eval H5's per-group field names/order == the committed golden.

    This compares what was actually written, not what was planned — strictly
    stronger than the run-free static table check
    (``H5OutputSink._resolve_columns`` vs the golden) that the folded
    ``test_config_h5_schema_parity.py`` used to run for this row's config.
    """
    golden = golden_path(row)
    if not golden.is_file():
        raise LegFailedError(
            row.test_name,
            "eval",
            f"no committed golden at {golden} — add one via generate_goldens.py, "
            f"or list {row.test_name!r} in NO_GOLDEN with a reason",
        )
    expected: dict[str, list[str]] = {}
    for col in json.loads(golden.read_text())["h5"]["columns"]:
        expected.setdefault(col["stream"], []).extend(col["column_names"])
    with h5py.File(eval_h5) as f:
        for stream, names in expected.items():
            if stream not in f:
                raise LegFailedError(
                    row.test_name,
                    "eval",
                    f"golden expects H5 group {stream!r}, the eval H5 has none",
                )
            actual = list(f[stream].dtype.names or ())
            if actual != names:
                raise LegFailedError(
                    row.test_name,
                    "eval",
                    f"eval H5 group {stream!r} schema drifted from the committed "
                    f"golden\n  golden: {names}\n  eval H5: {actual}",
                )


# ---------------------------------------------------------------------------
# pytest surface — three parametrized legs per row
# ---------------------------------------------------------------------------

_PARAMS = [
    pytest.param(row.test_name, marks=(pytest.mark.gpu,) if row.test_name in GPU_ROWS else ())
    for row in MATRIX
]


def _apply_known_xfail(request: pytest.FixtureRequest, name: str, leg: str) -> None:
    reason = KNOWN_FAILURES.get((name, leg))
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=reason))


@pytest.mark.parametrize("name", _PARAMS)
def test_fit(name, tmp_path_factory, request):
    """Leg 1 — ``fit=True`` rows: ``salt fit`` produces a checkpoint + saved config."""
    row = row_by_name(name)
    if not row.fit:
        pytest.skip("row declares fit=False — compile+plot only (see test_compile_plot)")
    _apply_known_xfail(request, name, "fit")
    try:
        run_row(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.producer != name:
            pytest.skip(f"producer row {exc.producer} failed")
        raise


@pytest.mark.parametrize("name", _PARAMS)
def test_eval(name, tmp_path_factory, request):
    """Leg 2 — ``do_eval=True`` rows: ``salt test`` + the H5-schema-vs-golden assertion."""
    row = row_by_name(name)
    if not row.do_eval:
        pytest.skip("row declares do_eval=False")
    _apply_known_xfail(request, name, "eval")
    try:
        run_eval(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.leg == "fit":
            pytest.skip(f"fit failed for {name}")
        raise


@pytest.mark.parametrize("name", _PARAMS)
def test_export(name, tmp_path_factory, request):
    """Leg 3 — ``do_onnx=True`` rows: ``salt export``, checked (torch<->ONNX parity)."""
    row = row_by_name(name)
    if not row.do_onnx:
        pytest.skip("row declares do_onnx=False")
    _apply_known_xfail(request, name, "export")
    try:
        run_export(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.leg == "fit":
            pytest.skip(f"fit failed for {name}")
        raise


# --------------------------------------------------------- completeness (§4.3)


def test_every_discovered_config_is_placed():
    """§ discovery: every config is a MATRIX row or a FRAGMENTS entry — never neither, never both.

    Enforces "require ALL configs be defined in test_pipeline" (user,
    verbatim). A config discovered on disk in neither table is an actionable
    failure naming the file and both tables; stale entries (naming a file
    that no longer exists) and rows also listed as fragments are equally a
    failure.
    """
    discovered = set(_discover())
    matrix_configs = {r.config for r in MATRIX}
    fragment_configs = set(FRAGMENTS)

    unplaced = sorted(discovered - matrix_configs - fragment_configs)
    assert not unplaced, (
        f"shipped config(s) with neither a MATRIX row nor a FRAGMENTS entry: {unplaced}. "
        "Add a Row to MATRIX (pipeline/test_pipeline.py) naming a FEEDS source, or a "
        "FRAGMENTS entry naming how it is exercised."
    )
    stale_rows = sorted(matrix_configs - discovered)
    assert not stale_rows, f"MATRIX rows name configs that do not exist: {stale_rows}"
    stale_fragments = sorted(fragment_configs - discovered)
    assert not stale_fragments, f"FRAGMENTS names configs that do not exist: {stale_fragments}"
    both = sorted(matrix_configs & fragment_configs)
    assert not both, f"named in both MATRIX and FRAGMENTS: {both}"


def test_every_fragment_is_exercised():
    """Every FRAGMENTS entry is reachable some way other than sitting unused on disk.

    "included" entries need >=1 MATRIX row's config to ``include:`` them
    directly; "paired" entries need a FEEDS entry to reference them by name;
    anything else is a free-text reason and is only existence-checked (by
    ``test_every_discovered_config_is_placed`` above) — the
    ``ftag1lite_streaming`` overlay and the two undocumented-pairing PHYSLITE
    reader fragments.
    """
    included_by: set[str] = set()
    for row in MATRIX:
        included_by |= _config_includes(row.config)
    paired = {arg for kind, arg in FEEDS.values() if kind == "root" and arg is not None}
    for fragment, how in FRAGMENTS.items():
        if how == "included":
            assert fragment in included_by, (
                f"{fragment} is marked 'included' in FRAGMENTS but no MATRIX row's "
                "config include:s it"
            )
        elif how == "paired":
            assert fragment in paired, (
                f"{fragment} is marked 'paired' in FRAGMENTS but no FEEDS entry "
                "references it"
            )


_READER_FRAGMENTS = [c for c in FRAGMENTS if _reader_node(c) is not None]


@pytest.mark.parametrize("fragment", _READER_FRAGMENTS)
def test_reader_fragment_instantiates(fragment):
    """A reader fragment's reader builds — the gate a fragment gets instead of a matrix row.

    A fragment has no model, so there is no plan to compile, but every
    invariant a reader enforces (``unroll`` naming a scalar group,
    link_branch/target_prefix pairing, constituent cuts on a jagged stream
    naming configured branches, ...) is raised from its constructor and is
    caught here.
    """
    _require_extra(fragment)
    from jsonargparse import ArgumentParser

    from salt.data.base import Reader

    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(Reader, "reader")
    cfg = parser.parse_object({"reader": _reader_node(fragment)})
    assert parser.instantiate_classes(cfg).reader is not None


def test_matrix_test_names_are_unique():
    """``test_name`` is the artifact-cache key and the junit id."""
    names = [r.test_name for r in MATRIX]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"MATRIX test_names are not unique: {dupes}"


def test_matrix_dependencies_are_defined_and_acyclic():
    """Every ``{ckpt:NAME}``/``{config:NAME}`` names a row; the DAG has no cycle."""
    names = {r.test_name for r in MATRIX}
    deps = {r.test_name: dependencies_of(r) for r in MATRIX}
    for name, dep_names in deps.items():
        unknown = sorted(dep_names - names)
        assert not unknown, f"{name}: train_args names unknown row(s) {unknown}"

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in visited:
            return
        if name in visiting:
            raise AssertionError(f"MATRIX dependency cycle: {' -> '.join((*stack, name))}")
        visiting.add(name)
        for dep in deps[name]:
            visit(dep, (*stack, name))
        visiting.discard(name)
        visited.add(name)

    for name in names:
        visit(name, ())


def test_every_matrix_config_has_a_feed():
    """No row can reach the fit leg without a declared data source."""
    missing = sorted({r.config for r in MATRIX} - set(FEEDS))
    assert not missing, f"MATRIX rows with no FEEDS entry: {missing}"


def test_known_failures_name_real_rows_and_legs():
    """Stale KNOWN_FAILURES entries fail loudly instead of silently protecting nothing."""
    names = {r.test_name for r in MATRIX}
    for name, leg in KNOWN_FAILURES:
        assert name in names, f"KNOWN_FAILURES names an unknown row: {name!r}"
        assert leg in {"fit", "eval", "export"}, (
            f"KNOWN_FAILURES names an unknown leg {leg!r} for {name!r}"
        )


def test_no_golden_table_is_honest():
    """Every do_eval=True row has a committed golden, or a reason in NO_GOLDEN — never both."""
    eval_rows = {r.test_name: r for r in MATRIX if r.do_eval}
    for name in NO_GOLDEN:
        assert name in eval_rows, f"NO_GOLDEN names a row that does not do_eval: {name!r}"
    for name, row in eval_rows.items():
        golden = golden_path(row)
        if golden.is_file():
            assert name not in NO_GOLDEN, (
                f"{name} is in NO_GOLDEN but a golden exists at {golden} — the table is stale"
            )
        else:
            assert name in NO_GOLDEN, (
                f"{name} has do_eval=True and no golden at {golden} — add it to "
                "NO_GOLDEN with a reason, or commit a golden"
            )


# ----------------------------------------------- residual finetune assertions (§3)
# What a matrix row cannot express: claims about the CHAINED artifacts
# (rows 13-15), not observable from rc == 0 on a fit.


def test_base_run_carries_the_modules_the_templates_freeze(tmp_path_factory):
    """The saved base config names the head both finetune templates warm up."""
    import yaml

    try:
        artifacts = run_row("gn3v00_base", tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"gn3v00_base fit failed: {exc}")
    modules = yaml.safe_load(artifacts.saved_config.read_text())["model"]["init_args"]["modules"]
    assert "jets_classification" in modules, (
        "finetune_gn3large.yaml warms up `jets_classification`; the base config "
        "no longer defines it"
    )


def test_added_head_is_absent_from_the_pretrained_checkpoint(tmp_path_factory):
    """The new head really is new — the warm start cannot be a no-op."""
    import torch

    try:
        artifacts = run_row("gn3v00_base", tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"gn3v00_base fit failed: {exc}")
    state = torch.load(artifacts.ckpt, map_location="cpu", weights_only=False)
    keys = state.get("state_dict", state)
    assert not any("large_r_jet_classification" in k for k in keys), (
        "the base checkpoint already carries the head the template adds, so this "
        "test would no longer prove the new-head path works"
    )


# ---------------------------------------------------------------------------
# regression + gaussian-regression per-config semantics
# ---------------------------------------------------------------------------
#
# Rows ``regression``/``regression_gaussian`` give the fit/eval/export
# lifecycle above; what follows is the de-scale, doubled-column and ONNX-rank
# assertions that are per-config semantics the generic runner cannot express
# and must survive the fold of ``test_regression_e2e.py`` +
# ``test_regression_gaussian_e2e.py``.
#
# do_onnx=True for both rows (§7 resolution): the "no gaussian handling"
# claim this section's gaussian half used to carry was a stale docstring, not
# a property of ``check_onnx`` — ``compare_once``
# (``salt/outputs/sinks/onnx/check.py``) is fully generic (by-name allclose on
# floats, exact ints, dead-output canary), so the gaussian export is checked
# like every other row.


def _eval_h5(name: str, tmp_path_factory) -> Path:
    try:
        return run_eval(name, tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"{name} {exc.leg} failed: {exc}")


def _onnx(name: str, tmp_path_factory) -> Path:
    try:
        return run_export(name, tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"{name} {exc.leg} failed: {exc}")


@pytest.fixture(scope="module")
def regression_eval_h5(tmp_path_factory) -> Path:
    return _eval_h5("regression", tmp_path_factory)


@pytest.fixture(scope="module")
def gaussian_eval_h5(tmp_path_factory) -> Path:
    return _eval_h5("regression_gaussian", tmp_path_factory)


@pytest.fixture(scope="module")
def regression_onnx(tmp_path_factory) -> Path:
    return _onnx("regression", tmp_path_factory)


@pytest.fixture(scope="module")
def gaussian_onnx(tmp_path_factory) -> Path:
    return _onnx("regression_gaussian", tmp_path_factory)


def test_regression_eval_h5_columns_present_and_descaled(regression_eval_h5):
    """The section eval H5 carries the regression columns, finite and de-scaled."""
    with h5py.File(regression_eval_h5) as f:
        jets = f["jets"][:]
        tracks = f["tracks"][:]
    jet_cols = set(jets.dtype.names)
    # the five regression heads' custom / target column names (regression.yaml)
    for col in (
        "regression_HadronConeExclTruthLabelPt",
        "regression_pt",
        "regression_truthMass",
        "regression_truthPt",
    ):
        assert col in jet_cols, f"missing regression column {col}: {sorted(jet_cols)}"
    # reg_normed -> HadronConeExclTruthLabelPt (norm_params mean=1.0 std=1.0):
    # de-scaled = raw*1 + 1, so the column is finite (the de-scale really ran)
    assert np.isfinite(jets["regression_HadronConeExclTruthLabelPt"]).all()
    # the per-token seq head columns land on the tracks stream
    track_cols = set(tracks.dtype.names)
    for col in ("regression_dummyOutput_dPhi", "regression_dummyOutput_dEta"):
        assert col in track_cols, f"missing seq regression column {col}"
    assert jets.shape[0] > 0


def test_gaussian_eval_h5_has_stddev_columns(gaussian_eval_h5):
    """The eval H5 carries the gaussian doubled columns (mean + _stddev), both streams."""
    with h5py.File(gaussian_eval_h5) as f:
        jet_cols = set(f["jets"].dtype.names)
        track_cols = set(f["tracks"].dtype.names)
        n_rows = f["jets"].shape[0]
    assert n_rows > 0
    assert any(c.endswith("_stddev") for c in jet_cols), f"no gaussian stddev in jets: {jet_cols}"
    assert any(
        c.endswith("_stddev") for c in track_cols
    ), f"no gaussian stddev in tracks: {track_cols}"


class TestRegressionOnnxContract:
    """The shipped regression.yaml exports a well-formed ONNX contract (row 8)."""

    def test_onnx_output_ranks_global_vs_per_token(self, regression_onnx):
        """6 rank-0 globals (norm/ratio scalars) + 2 rank-1 per-token seq columns."""
        model = onnx.load(str(regression_onnx))
        ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
        assert sorted(ranks.values()) == [0, 0, 0, 0, 0, 0, 1, 1], ranks

    def test_onnx_session_runs(self, regression_onnx):
        """The exported graph runs in onnxruntime on batch-1 inputs (L=5 tokens)."""
        sess = make_session(regression_onnx)
        in_meta = {i.name: i.shape for i in sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds), strict=True)}
        assert len(out) == 8, f"expected the 8 regression outputs, got {sorted(out)}"


class TestGaussianOnnxContract:
    """The shipped regression_gaussian.yaml exports a well-formed ONNX contract (row 9)."""

    def test_onnx_output_ranks_global_vs_per_token(self, gaussian_onnx):
        """2 rank-0 globals (mean + _stddev of the global head) + 2 rank-1 per-token."""
        model = onnx.load(str(gaussian_onnx))
        ranks = {o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output}
        assert sorted(ranks.values()) == [0, 0, 1, 1], ranks

    def test_onnx_session_runs(self, gaussian_onnx):
        """The exported graph runs in onnxruntime on batch-1 inputs (L=5 tokens)."""
        sess = make_session(gaussian_onnx)
        in_meta = {i.name: i.shape for i in sess.get_inputs()}
        rng = np.random.default_rng(0)

        def shape_for(dims):
            return tuple(5 if (isinstance(d, str) or d is None) else d for d in dims)

        feeds = {
            name: rng.standard_normal(shape_for(dims)).astype(np.float32)
            for name, dims in in_meta.items()
        }
        out = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, feeds), strict=True)}
        assert len(out) == 4, f"expected the 4 gaussian outputs, got {sorted(out)}"
