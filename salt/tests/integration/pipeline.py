"""The declarative config-lifecycle matrix: mechanism, not tests.

Importable and un-collected (pytest's ``python_files`` does not match this
name). ``test_pipeline.py`` is the pytest surface; this module owns ``MATRIX``,
``FEEDS``, the feeder builders, ``run_row``/``run_eval``/``run_export`` and
their session-scoped artifact cache — the parts a mechanism test can import
without importing a test module.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.schema import dump_schema, save_schema
from salt.testing.datagen import compute_norm_dict, load_pipeline
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

RECIPES_DIR = Path(__file__).resolve().parents[2] / "testing" / "datagen" / "recipes"
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "output_goldens"

# base.yaml is auto-loaded machinery, never a model in its own right.
_MACHINERY = {"base"}

# ---------------------------------------------------------------------------
# schema (§1.1-1.2)
# ---------------------------------------------------------------------------

Feed = tuple[Literal["recipe", "dummy", "root"], str | None]
"""``("recipe", name)`` | ``("dummy", flavour)`` | ``("root", fragment | None)``."""


@dataclass(frozen=True)
class Row:
    """One matrix row: ``(test_name, config_relpath, do_eval, do_onnx, train_args)``."""

    test_name: str
    config: str
    do_eval: bool
    do_onnx: bool
    train_args: tuple[str, ...] = ()


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
            "--data.modules.input_samples.files.train={h5}",
            "--data.modules.input_samples.files.val={h5}",
            "--data.modules.input_samples.files.test={h5}",
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


def production_configs() -> list[str]:
    """The discovered top-level production configs (``salt/configs/*.yaml`` minus base)."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem not in _MACHINERY)


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
    import h5py
    import numpy as np

    from salt.utils.array_utils import join_structured_arrays

    base = _dummy_context("gn3", tmp_path_factory)
    out = tmp_path_factory.mktemp("dummy_gn3_large_r")
    h5 = out / "pp_output_train.h5"
    with h5py.File(base["h5"]) as src, h5py.File(h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name, dataset in src.items():
            if name != "jets":
                dst.create_dataset(name, data=dataset[:])
                for key, value in dataset.attrs.items():
                    dst[name].attrs[key] = value
                continue
            jets = dataset[:]
            rng = np.random.default_rng(42)
            extra = rng.integers(0, 4, size=len(jets)).astype("i4")
            extra = extra.view(np.dtype([("large_r_flavour_label", "i4")]))
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
    """The format-string context (§1.1) available to ``row``'s ``train_args``."""
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
# the legs (§4.1) — session-scoped, memoised per row
# ---------------------------------------------------------------------------

# dict[test_name, Artifacts | LegFailedError]. No xdist today (pyproject has
# none, CI runs one pytest process), so a plain module-level dict is correct.
# If xdist is ever added, every matrix row must stay in one module
# (test_pipeline.py already is), and `--dist loadfile` becomes mandatory —
# this cache is not otherwise safe to share across workers.
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
    That static form has not (yet) been re-added to the Tier-A loop of
    ``test_shipped_configs.py`` (§4.2 scoped it to ``MIGRATED = {regression,
    regression_gaussian}``, both of which are ``do_eval=True`` matrix rows) —
    no coverage loss today, but a config that only ever gets a matrix row and
    never a Tier-A/eval-golden pairing would have neither check.
    """
    import h5py

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
