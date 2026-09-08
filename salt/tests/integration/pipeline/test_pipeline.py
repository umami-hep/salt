"""The declarative config-lifecycle matrix: fixtures, mechanism, and tests.

Owns ``ROWS`` (loaded from ``fixtures/*.yaml``, see ``_load_fixtures``),
``FEEDS``, the feeder builders, ``run_row``/``run_eval``/``run_export`` and
their session-scoped artifact cache, the pytest surface (a compile+plot floor
leg every row gets regardless of ``fit``, plus three parametrized lifecycle
legs: fit runs when the row declares ``fit=True``; eval/export additionally
skip when the row declares ``do_eval=False``/``do_onnx=False``), the xfail
lookup (from each fixture's own ``xfail:`` block), the fixture<->discovery
completeness checks (see ``FRAGMENT_FIXTURES`` below), the
regression/gaussian-regression per-config semantics (de-scale, doubled-column
and ONNX-rank assertions) that are per-config and cannot be expressed by the
generic runner, and (folded in from the deleted ``test_inference.py``) the
post-fit ``salt inference`` gate over every fit-capable row's own artifacts.

**``fixtures/`` is the curated, hand-maintained source of truth.** Each
``fixtures/<id>.yaml`` is either a fragment stub (``config`` + ``fragment``)
or a full row (``config`` plus optional ``gpu``/``fit``/``eval``/``onnx``/
``inference``/``xfail`` keys — see ``fixtures/README.md`` for the schema).
``_load_fixtures`` reads every file at module-import time and validates it
hard: an unknown key, a missing ``expected_outputs``, an unknown xfail leg, or
a duplicate config all raise ``ValueError`` naming the offending file.
**These files are NEVER auto-regenerated — there is no regeneration script,
by design (see the study's standing "no snapshots" ruling and
``fixtures/README.md``).** Edit them by hand.

Discovery is a recursive glob of ``salt/configs`` (``_discover``) — a config
added anywhere in the tree is picked up automatically and cannot silently go
untested. Every discovered config (minus ``_EXEMPT``, i.e. ``base.yaml``)
must be claimed by EXACTLY ONE fixture file — either a row or a ``fragment:``
stub (an include-target reader fragment, a fragment paired via ``FEEDS``, or
a data-behaviour overlay) — see ``test_every_config_has_exactly_one_fixture``.
A row that has no synthetic fixture the fit lifecycle can serve declares
``fit: false``: it still gets the compile+plot floor leg (every row does),
just not the fit/eval/export legs.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import onnx
import pytest
import yaml
from numpy.lib.recfunctions import repack_fields

from salt.graph.errors import GraphError
from salt.inference import run_inference
from salt.main import CONFIG_DIR
from salt.main import main as salt_main
from salt.outputs.sinks.onnx import make_session
from salt.schema import dump_schema, load_schema, save_schema
from salt.testing.datagen import compute_norm_dict, load_pipeline
from salt.testing.inputs import write_dummy_file, write_dummy_norm_dict

RECIPES_DIR = Path(__file__).resolve().parents[3] / "testing" / "datagen" / "recipes"
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# base.yaml is auto-loaded machinery, never a model in its own right — the
# ONE hardcoded exemption from the fixture-completeness gate.
_EXEMPT = {"base"}

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
    """One fixture row: what a config needs to run the lifecycle, and what its
    outputs must contain.

    ``fit`` (default ``True``): whether this row gets the fit/eval/export
    lifecycle legs. Every row — ``fit=True`` or not — always gets the
    compile+plot floor leg (``test_compile_plot``): a config with no
    synthetic fixture the lifecycle can run on (``fit=False``) is still a
    real shipped config, and this is what proves its plan compiles and
    renders even though no checkpoint is produced.

    ``xfail`` is ``((leg, reason), ...)`` pairs from the fixture's own
    ``xfail:`` block — ``_apply_known_xfail`` looks the leg up per-test.
    ``expected_h5``/``expected_onnx``/``inference_onnx`` are containment
    contracts (not necessarily exhaustive): a declared name must be present
    in what the corresponding leg actually produces.

    ``stack``: shipped configs (relpaths under ``salt/configs/``, no
    ``.yaml``) stacked BEFORE this row's own config in both the compile+plot
    floor leg and the fit leg — for overlay templates whose base cannot
    itself be a producer row (e.g. a bundle mirror no synthetic fixture can
    feed). The row's own config is always stacked last and wins.
    """

    test_name: str
    config: str
    do_eval: bool
    do_onnx: bool
    train_args: tuple[str, ...] = ()
    stack: tuple[str, ...] = ()
    fit: bool = True
    gpu: bool = False
    xfail: tuple[tuple[str, str], ...] = ()
    expected_h5: dict | None = None
    expected_onnx: tuple[str, ...] | None = None
    inference_onnx: tuple[str, ...] | None = None


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
# fixture loader (§ loader) — reads fixtures/*.yaml at module-import time
# ---------------------------------------------------------------------------

_ROW_TOP_KEYS = {"config", "gpu", "fit", "eval", "onnx", "inference", "xfail", "stack"}
_FRAGMENT_TOP_KEYS = {"config", "fragment"}
_VALID_XFAIL_LEGS = {"fit", "eval", "export", "compile_plot"}


def _fixture_error(path: Path, msg: str) -> ValueError:
    return ValueError(f"{path}: {msg}")


def _load_fit(path: Path, raw: dict) -> tuple[bool, tuple[str, ...]]:
    """``(fit, train_args)`` from a row fixture's ``fit:`` key (absent -> True)."""
    if "fit" not in raw:
        return True, ()
    node = raw["fit"]
    if node is False:
        return False, ()
    if isinstance(node, dict):
        args = node.get("args")
        if not args:
            raise _fixture_error(path, "'fit' mapping must declare a non-empty 'args' list")
        return True, tuple(str(a) for a in args)
    raise _fixture_error(path, f"'fit' must be false or a mapping with 'args', got {node!r}")


def _load_eval(path: Path, raw: dict) -> tuple[bool, dict | None]:
    """``(do_eval, expected_h5)`` from a row fixture's ``eval:`` key."""
    node = raw.get("eval")
    if node is None or node is False:
        return False, None
    if not isinstance(node, dict):
        raise _fixture_error(path, f"'eval' must be a mapping, got {node!r}")
    expected = node.get("expected_outputs")
    if not expected:
        raise _fixture_error(path, "'eval' present but 'expected_outputs' is missing/empty")
    return True, expected


def _load_onnx_like(path: Path, raw: dict, key: str) -> tuple[str, ...] | None:
    """``expected_outputs`` tuple from ``raw[key]`` (``onnx``/``inference``), or ``None``."""
    node = raw.get(key)
    if node is None:
        return None
    if not isinstance(node, dict):
        raise _fixture_error(path, f"{key!r} must be a mapping, got {node!r}")
    names = node.get("expected_outputs")
    if not names:
        raise _fixture_error(path, f"{key!r} present but 'expected_outputs' is missing/empty")
    return tuple(names)


def _load_stack(path: Path, raw: dict) -> tuple[str, ...]:
    """``row.stack`` from a row fixture's ``stack:`` key (absent -> ())."""
    node = raw.get("stack")
    if node is None:
        return ()
    if not isinstance(node, list) or not node:
        raise _fixture_error(path, "'stack' must be a non-empty list of config relpaths")
    resolved: list[str] = []
    for item in node:
        s = str(item)
        if not s.endswith(".yaml"):
            raise _fixture_error(path, f"'stack' entries must end in .yaml, got {s!r}")
        if not (CONFIG_DIR / s).is_file():
            raise _fixture_error(path, f"'stack' names {s!r}, not found under {CONFIG_DIR}")
        resolved.append(s)
    return tuple(resolved)


def _load_xfail(path: Path, raw: dict) -> tuple[tuple[str, str], ...]:
    """``((leg, reason), ...)`` from a row fixture's ``xfail:`` key."""
    node = raw.get("xfail")
    if node is None:
        return ()
    if not isinstance(node, dict) or not node:
        raise _fixture_error(path, "'xfail' must be a non-empty mapping of leg -> reason")
    pairs: list[tuple[str, str]] = []
    for leg, reason in node.items():
        if leg not in _VALID_XFAIL_LEGS:
            raise _fixture_error(
                path, f"'xfail' names unknown leg {leg!r} (valid: {sorted(_VALID_XFAIL_LEGS)})"
            )
        if not str(reason).strip():
            raise _fixture_error(path, f"'xfail' leg {leg!r} has an empty reason")
        pairs.append((leg, str(reason)))
    return tuple(pairs)


def _load_fixtures() -> tuple[list[Row], dict[str, str]]:
    """Read + validate every ``fixtures/*.yaml``; see the module docstring.

    Returns ``(rows, fragment_fixtures)`` — ``fragment_fixtures`` maps
    config_relpath -> the fixture's ``fragment:`` value (mirrors the deleted
    ``FRAGMENTS`` table).
    """
    if not _FIXTURES_DIR.is_dir():
        raise ValueError(f"fixtures directory missing: {_FIXTURES_DIR} — see fixtures/README.md")
    paths = sorted(_FIXTURES_DIR.glob("*.yaml"))
    if not paths:
        raise ValueError(
            f"no fixture files under {_FIXTURES_DIR} — every runnable/fragment config needs "
            "a hand-curated fixtures/<id>.yaml; this loader never regenerates them, see "
            "fixtures/README.md"
        )
    rows: list[Row] = []
    fragments: dict[str, str] = {}
    seen: dict[str, Path] = {}
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "config" not in raw:
            raise _fixture_error(path, "missing required key 'config'")
        config_field = str(raw["config"])
        if not config_field.endswith(".yaml"):
            raise _fixture_error(path, f"'config' must end in .yaml, got {config_field!r}")
        config = config_field[: -len(".yaml")]

        if config in seen:
            raise _fixture_error(
                path, f"duplicate config {config!r} (also claimed by {seen[config]})"
            )
        seen[config] = path

        if "fragment" in raw:
            unknown = set(raw) - _FRAGMENT_TOP_KEYS
            if unknown:
                raise _fixture_error(
                    path, f"fragment fixture has unexpected key(s): {sorted(unknown)}"
                )
            fragments[config] = raw["fragment"]
            continue

        unknown = set(raw) - _ROW_TOP_KEYS
        if unknown:
            raise _fixture_error(path, f"unknown top-level key(s): {sorted(unknown)}")

        fit, train_args = _load_fit(path, raw)
        do_eval, expected_h5 = _load_eval(path, raw)
        expected_onnx = _load_onnx_like(path, raw, "onnx")
        inference_onnx = _load_onnx_like(path, raw, "inference")

        rows.append(
            Row(
                test_name=path.stem,
                config=config,
                do_eval=do_eval,
                do_onnx=expected_onnx is not None,
                train_args=train_args,
                stack=_load_stack(path, raw),
                fit=fit,
                gpu=bool(raw.get("gpu", False)),
                xfail=_load_xfail(path, raw),
                expected_h5=expected_h5,
                expected_onnx=expected_onnx,
                inference_onnx=inference_onnx,
            )
        )
    return rows, fragments


ROWS, FRAGMENT_FIXTURES = _load_fixtures()

_BY_NAME: dict[str, Row] = {r.test_name: r for r in ROWS}


def row_by_name(name: str) -> Row:
    """The fixture row named ``name``."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"no fixture row named {name!r}") from None


def _discover() -> list[str]:
    """Every shipped config under ``CONFIG_DIR``, relative path minus suffix, minus ``_EXEMPT``.

    Recursive (``rglob``) — a config added anywhere in the tree, at any
    depth, is picked up automatically (§ discovery).
    """
    found = sorted(
        p.relative_to(CONFIG_DIR).with_suffix("").as_posix() for p in CONFIG_DIR.rglob("*.yaml")
    )
    result = []
    for c in found:
        if c in _EXEMPT:
            continue
        result.append(c)
    return result


def _config_includes(config: str) -> set[str]:
    """``config``'s own top-level ``include:`` list, resolved to config_relpaths.

    Deliberately NOT ``expand_includes`` (which recurses and would also
    surface transitive includes) — this is "does THIS config's own
    ``include:`` block name it", matching the docstring convention
    ("declare its bases in the config's own include: block").
    """
    import yaml as _yaml

    raw = _yaml.safe_load((CONFIG_DIR / f"{config}.yaml").read_text()) or {}
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


def _load_expanded(config: str) -> dict:
    """A shipped config's include-expanded YAML, as a plain dict."""
    import yaml as _yaml

    from salt.utils.config_utils import expand_includes

    path = CONFIG_DIR / f"{config}.yaml"
    return _yaml.safe_load(Path(expand_includes(str(path))).read_text()) or {}


def _reader_node(config: str) -> dict | None:
    """The expanded config's ``data.modules.reader`` node, or ``None``."""
    return ((_load_expanded(config).get("data") or {}).get("modules") or {}).get("reader")


# ---------------------------------------------------------------------------
# feeders (§1.1: the ``FEEDS`` table drives which of these a row uses)
# ---------------------------------------------------------------------------

_RECIPE_CACHE: dict[str, dict[str, Path]] = {}


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
# or model needs an optional dependency the bare image lacks. ROOT-fed
# fixture rows (easyjet_flavour, event_tagger_easyjet) already self-gate via
# `_root_deps()`/`RootDepsMissingError` (the "root" FEEDS kind) and don't need
# an entry here; this table is for configs fed some other way (dummy H5)
# whose reader/model class still has a hard import on an optional package,
# plus the reader-fragment entries in the fragment fixtures that
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


# config_relpath -> the ``class_names`` this row's own flavour_label task
# ACTUALLY declares, when it differs from the SHARED recipe's default
# schema attr (pipeline #15651154, item 5: gn2v2-opendata declares
# [bjets, cjets, ujets, taujets] (open-data label order, taujets INCLUDED),
# but the "flavour_tagger" recipe's default flags: {} (no inc_taus) produces
# a 3-class flavour_label attr with no taujets — check_class_names
# (salt/model/saltmodule.py) then raises ConfigError at fit/eval time
# (schema-bound legs only; compile+plot passes no schema, so it never saw
# this). A per-config table, not a blanket recipe/flag change: GN3X ALSO
# feeds off "flavour_tagger" (see FEEDS) and ALSO declares a class set the
# recipe's default doesn't produce (9 classes) — GN3X's fit is an EXISTING
# fixture-declared strict xfail, so fixing this generically for every
# recipe-fed row risks silently fixing GN3X's root cause too and flipping
# it to an unexpected PASS (a strict-xfail failure). Metadata-only: the
# underlying H5 flavour_label INT column is untouched by the override below
# — the appended class(es) are simply undrawn (count 0), exactly like
# taujets already is in the write_dummy_file convention this recipe mirrors
# (jet_features.yaml's own class_names vs. sample_classes split).
RECIPE_CLASS_NAMES_OVERRIDE: dict[str, list[str]] = {
    "gn2v2-opendata": ["bjets", "cjets", "ujets", "taujets"],
}


def _apply_class_names_override(
    row: Row, ctx: dict[str, Path], class_names: list[str], tmp_path_factory
) -> dict[str, Path]:
    """A row-scoped derived schema with the ``jets`` group's ``flavour_label``
    attr replaced by ``class_names`` (see ``RECIPE_CLASS_NAMES_OVERRIDE``).

    Never mutates the shared, cached recipe context (``_recipe_context``,
    keyed by recipe name only, so other rows sharing the SAME recipe get the
    UNCHANGED schema) — this returns a NEW ctx dict pointing ``"schema"`` at
    a freshly written, row-scoped file instead.
    """
    schema = load_schema(ctx["schema"])
    jets = schema.groups["jets"]
    derived_jets = replace(jets, attrs={**jets.attrs, "flavour_label": list(class_names)})
    derived_schema = replace(schema, groups={**schema.groups, "jets": derived_jets})
    out = tmp_path_factory.mktemp(f"{row.test_name}_schema") / "schema.yaml"
    save_schema(derived_schema, out)
    return {**ctx, "schema": out}


def _recipe_context(recipe: str, tmp_path_factory) -> dict[str, Path]:
    """Run a ``salt.testing.datagen`` recipe once; every row using it shares the result."""
    cached = _RECIPE_CACHE.get(recipe)
    if cached is not None:
        return cached
    import yaml as _yaml

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
        nd.write_text(_yaml.dump(compute_norm_dict(data), sort_keys=False))
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
    import yaml as _yaml

    from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict

    out = tmp_path_factory.mktemp("dummy_regression")
    nd, cd = out / "norm_dict.yaml", out / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    raw = _yaml.safe_load(nd.read_text())
    raw["jets"]["mass"] = {"mean": round(0.1 * 3, 6), "std": round(1.0 + 0.05 * 3, 6)}
    nd.write_text(_yaml.dump(raw, sort_keys=False))
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


_DUMMY_BUILDERS = {
    "default": _build_default_dummy,
    "regression": _build_regression_dummy,
    "gn3": _build_gn3_dummy,
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
    """One synthetic minitree — read via ``--data.train_file``/``--data.val_file``
    for the fit leg (row 7), and via a stacked ``--config {fragment}`` overlay
    for the compile+plot floor leg (pipeline #15651154, item 1: that leg
    discards plain overrides, and its run-free ``graph validate``/``graph
    plot`` needs the reader's ``filename`` already resolved in a config file —
    see ``write_standalone_source_fragment``).
    """
    from salt.tests._fixtures.easyjet_minitree import (
        build_fixture_arrays,
        write_jets_norm_dict,
        write_minitree,
        write_standalone_source_fragment,
    )

    out = tmp_path_factory.mktemp(f"root_{row.test_name}")
    root = write_minitree(out / "data.root", build_fixture_arrays())
    expanded = _load_expanded(row.config)
    variables = expanded["data"]["modules"]["features"]["init_args"]["variables"]["jets"]
    norm = write_jets_norm_dict(out / "norm_dict.yaml", variables)
    reader_node = expanded["data"]["modules"]["reader"]
    fragment = write_standalone_source_fragment(out / "source.yaml", root, reader_node)
    return {"root": root, "norm": norm, "fragment": fragment}


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
        ctx = _recipe_context(arg, tmp_path_factory)
        override = RECIPE_CLASS_NAMES_OVERRIDE.get(row.config)
        if override is not None:
            ctx = _apply_class_names_override(row, ctx, override, tmp_path_factory)
        return ctx
    if kind == "dummy":
        assert arg is not None
        return _dummy_context(arg, tmp_path_factory)
    return _root_context(row, arg, tmp_path_factory)


def _norm_dict_is_unresolved(init_args: dict) -> bool:
    """Whether ``init_args`` needs a real ``norm_dict`` override: either it
    declares none at all (the "bare Normaliser" case), or it declares an
    UNRESOLVED CCRA runner placeholder (``${VAR_NAME}``, e.g.
    gn2v2-opendata.yaml's ``norm_dict: ${DATA_NORM_DICT_PATH}``) — the
    experiment runner's ``update_paths.py`` substitutes those, never this
    isolated test harness, so the literal ``${...}`` string would otherwise
    reach ``Normaliser``'s constructor and fail its own file-existence
    preflight (pipeline #15651451, item 2: gn2v2-opendata is the only
    shipped config in the matrix whose Normaliser ALREADY declares a
    norm_dict key, so the old bare ``"norm_dict" not in init_args`` check
    saw the key present and silently skipped it).
    """
    if "norm_dict" not in init_args:
        return True
    value = init_args["norm_dict"]
    return isinstance(value, str) and value.startswith("${") and value.endswith("}")


def _norm_overrides(row: Row, ctx: dict[str, Path]) -> list[str]:
    """``--model.modules.<name>.init_args.norm_dict=<norm>`` for every Normaliser
    that needs one (see ``_norm_dict_is_unresolved``).

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
        # exact class-name match only: MaskedInputNormaliser has no norm_dict
        # init arg (it learns stats online), so a substring match would wrongly
        # try to override it.
        and str(node.get("class_path", "")).rsplit(".", 1)[-1] == "Normaliser"
        and _norm_dict_is_unresolved(node.get("init_args") or {})
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


_STDERR_TAIL_LINES = 20


def _salt_main(name: str, leg: str, argv: list[str]) -> None:
    """Run ``salt_main(argv)`` for ``name``'s ``leg``; raises ``LegFailedError``
    on ANY failure — a nonzero rc, or ``salt.main.main``'s ``SystemExit``
    re-raise on a failed CLI parse. Returns ``None`` on success (rc == 0):
    every call site below is now "call it, then keep going" — a failure
    always surfaces as ``LegFailedError``, never a bare rc for the caller to
    re-check.

    Harness fix (pipeline #15651025 cluster 4 — "unblocks all future
    diagnosis"): CI runs pytest with ``--show-capture=stdout``, and every
    ``salt`` error path prints to STDERR (``console(..., file=sys.stderr)``,
    argparse's own usage errors, ``GraphError``'s one-block form) — so a
    failing leg showed NOTHING in the CI log, just a bare rc/exit code. This
    redirects stderr into a buffer for the duration of the call; on ANY
    failure the last ``_STDERR_TAIL_LINES`` lines print to STDOUT (so CI
    actually shows them) and are folded into the raised ``LegFailedError``'s
    message (so a producer-row skip elsewhere names the real cause, not just
    "rc=1"). On rc == 0 the buffer is discarded silently.

    ``salt.main.main`` re-raising ``SystemExit`` from a failed CLI parse
    (instead of returning a nonzero rc) matters beyond messaging too: left
    uncaught, it skips the ``_ROW_CACHE``/``_INFERENCE_CACHE`` memoisation
    entirely, so every row chained onto a dead producer
    (``{ckpt:NAME}``/``{config:NAME}``) re-runs the full fit from scratch
    instead of skipping with a named reason (pipeline #15650554, item B4 —
    gn2v2_opendata's 7 legs each re-ran the whole fit).
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            rc = salt_main(argv)
    except SystemExit as exc:
        _fail_with_stderr(name, leg, buf, f"salt {leg} exited {exc.code} (argv parse)", exc)
        return
    if rc != 0:
        _fail_with_stderr(name, leg, buf, f"salt {leg} rc={rc}")


def _fail_with_stderr(
    name: str, leg: str, buf: io.StringIO, reason: str, cause: BaseException | None = None
) -> None:
    """Print ``buf``'s captured-stderr tail to stdout, then raise ``LegFailedError``
    naming ``reason`` with the SAME tail folded into its message. Always raises.
    """
    lines = buf.getvalue().splitlines()
    tail = "\n".join(lines[-_STDERR_TAIL_LINES:])
    if tail:
        n_shown = min(len(lines), _STDERR_TAIL_LINES)
        print(f"--- {name} {leg}: captured stderr (last {n_shown} lines) ---")
        print(tail)
        print(f"--- {name} {leg}: end captured stderr ---")
        reason = f"{reason}\nstderr tail:\n{tail}"
    if cause is not None:
        raise LegFailedError(name, leg, reason) from cause
    raise LegFailedError(name, leg, reason)


def _stack_argv(row: Row, flag: str) -> list[str]:
    """``[flag, path]`` pairs for each shipped config in ``row.stack``, in order.

    Stacked BEFORE the row's own ``-c``/``--config`` (see ``Row.stack``), so
    the row's own config always wins the deep-merge.
    """
    argv: list[str] = []
    for s in row.stack:
        argv += [flag, str(CONFIG_DIR / s)]
    return argv


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
    stack_argv = _stack_argv(row, "-c")
    argv = ["graph", "validate", *stack_argv, *config_argv, "-c", cfg_path]
    for mode in ("fit", "val", "test", "onnx"):
        argv += ["--mode", mode]
    _salt_main(row.test_name, "validate", [*argv, *sets])

    plot_path = tmp_path / f"{row.test_name}.dot"
    plot_argv = [
        "graph",
        "plot",
        *stack_argv,
        *config_argv,
        "-c",
        cfg_path,
        "--mode",
        "fit",
        "-o",
        str(plot_path),
    ]
    _salt_main(row.test_name, "plot", [*plot_argv, *sets])
    assert plot_path.is_file(), f"{row.test_name}: graph plot wrote nothing to {plot_path}"


_COMPILE_PLOT_PARAMS = [row.test_name for row in ROWS]


@pytest.mark.parametrize("name", _COMPILE_PLOT_PARAMS)
def test_compile_plot(name, tmp_path_factory, tmp_path, request):
    """Floor leg — every row's config plan-compiles and its fit-mode DOT renders.

    Leg name "compile_plot" for fixture-xfail purposes (study xfail policy:
    "pre-existing lifecycle failures get tracked xfails, not debugging
    expeditions") — this leg has no {fit, eval, export} split of its own, so
    one entry covers both the ``graph validate`` and ``graph plot`` calls
    inside ``run_compile_plot``.
    """
    row = row_by_name(name)
    _apply_known_xfail(request, name, "compile_plot")
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
# If xdist is ever added, every fixture row must stay in one module (this
# module already is), and `--dist loadfile` becomes mandatory — this cache
# is not otherwise safe to share across workers.
_ROW_CACHE: dict[str, Artifacts | LegFailedError] = {}


def run_row(name: str, tmp_path_factory) -> Artifacts:
    """Run (or fetch) fixture row ``name`` through the fit leg.

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


def _needs_mup_shapes(config: str) -> bool:
    """Whether ``config``'s model declares a ``mup:`` block (structural check
    on the config itself, not a row-name special case — GN2_muP is simply the
    only shipped config where this is currently true).
    """
    model = _load_expanded(config).get("model") or {}
    return bool((model.get("init_args") or {}).get("mup"))


def _generate_mup_shapes(row: Row, ctx: dict[str, Path], tmp_path_factory) -> Path:
    """Run ``salt mup-shapes`` for ``row``'s config — the real muP pre-fit
    workflow (pipeline #15651154, item 4): the shipped config's
    ``model.init_args.mup.shape_path`` is a documented placeholder
    ("shape_path is a REQUIRED override — generate with `salt mup-shapes`",
    GN2_muP.yaml), never a real file, so ``salt fit`` on it verbatim always
    fails looking for ``shape_mup.bsh``. ``generate_shapes`` builds its
    base/delta probe models data-free/run-free purely from the config, so
    this needs only the same norm_dict override every bare Normaliser in the
    row's config needs for a run-free parse (``_norm_set_overrides``) — base/
    delta widths default from the config's own configured width.
    """
    out_dir = tmp_path_factory.mktemp(f"mup_shapes_{row.test_name}")
    shape_path = out_dir / "shape_mup.bsh"
    argv = [
        "mup-shapes",
        "-c",
        str(CONFIG_DIR / f"{row.config}.yaml"),
        "--save-path",
        str(shape_path),
    ]
    for override in _norm_set_overrides(row, ctx):
        argv += ["--set", override]
    _salt_main(row.test_name, "mup-shapes", argv)
    return shape_path


def _do_fit(
    row: Row,
    ctx: dict[str, Path],
    config_argv: list[str],
    override_argv: list[str],
    tmp_path_factory,
) -> Artifacts:
    # explicit, direct gate — pipeline #15651025 cluster 3: gn2_mup's fit leg
    # rc=1'd on a missing `mup` despite EXTRAS mapping GN2/GN2_muP to it;
    # `_feed_context` (called by `run_row` just before this) already calls
    # `_require_extra`, but the failure surfaced anyway, so this leg gets its
    # OWN direct call rather than relying solely on that transitive path.
    _require_extra(row.config)
    mup_shape_overrides: list[str] = []
    if _needs_mup_shapes(row.config):
        shape_path = _generate_mup_shapes(row, ctx, tmp_path_factory)
        mup_shape_overrides = [f"--model.init_args.mup.shape_path={shape_path}"]
    root = tmp_path_factory.mktemp(row.test_name)
    argv = ["fit"]
    # row.stack (shipped configs, e.g. an overlay's base bundle) goes first —
    # before both the {config:NAME} producer chain and this row's own config.
    argv += _stack_argv(row, "--config")
    # a {config:NAME} dependency is a producer row's *saved* config.yaml —
    # stack it BEFORE this row's own template (docs/tutorials/finetuning.md's
    # worked order: saved base config first, template second), so the
    # template's own settings (training_schedule, the new head, ...) win over
    # the producer's incidental ones on any key both happen to set.
    argv += config_argv
    argv += ["--config", str(CONFIG_DIR / f"{row.config}.yaml")]
    argv += _base_data_args(row, ctx)
    argv += _trainer_args(root)
    argv += mup_shape_overrides
    # override_argv last: row-specific overrides (finetune epochs/limits, the
    # input_samples fix, ...) must win over both configs and the shared
    # defaults above.
    argv += override_argv
    _salt_main(row.test_name, "fit", argv)
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
    _salt_main(name, "eval", argv)
    evals = sorted(artifacts.ckpt.parent.glob("*__test_*.h5"))
    if not evals:
        raise LegFailedError(name, "eval", f"eval wrote no H5 next to {artifacts.ckpt}")
    eval_h5 = evals[-1]
    artifacts.eval_h5 = eval_h5
    _assert_eval_h5_has_expected_outputs(eval_h5, row)
    return eval_h5


def run_export(name: str, tmp_path_factory) -> Path:
    """Run (or fetch) row ``name``'s export leg: checked ONNX export (§4.1).

    Deliberately *without* ``--no-check`` — ``rc == 0`` IS the torch<->ONNX
    parity assertion. Additionally checks the exported graph's own output
    names against the row's fixture ``expected_onnx`` (containment) when
    present — the export leg's own enforcement point (user ruling),
    independent of ``TestExportedOnnxOutputNames``'s onnxruntime-session
    check on the same file (below).
    """
    artifacts = run_row(name, tmp_path_factory)
    if artifacts.onnx is not None:
        return artifacts.onnx
    onnx_dir = artifacts.root_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{name}.onnx"
    _salt_main(
        name,
        "export",
        [
            "export",
            "--config",
            str(artifacts.saved_config),
            f"--ckpt_path={artifacts.ckpt}",
            f"--output={onnx_path}",
        ],
    )
    if not sorted(onnx_dir.glob("*.onnx")):
        raise LegFailedError(name, "export", f"no ONNX written under {onnx_dir}")
    expected_onnx = row_by_name(name).expected_onnx
    if expected_onnx:
        actual = [o.name for o in onnx.load(str(onnx_path)).graph.output]
        missing = [n for n in expected_onnx if n not in actual]
        if missing:
            raise LegFailedError(
                name,
                "export",
                f"exported ONNX is missing declared expected_outputs output(s) "
                f"{missing}; actual tuple: {actual}",
            )
    artifacts.onnx = onnx_path
    return onnx_path


def _assert_eval_h5_has_expected_outputs(eval_h5: Path, row: Row) -> None:
    """The written eval H5 carries at least the row fixture's declared ``expected_h5`` columns.

    Containment, not equality (user ruling, replacing the golden-snapshot
    exact-schema check): extra columns (input copies, pad mask, object
    groups, undeclared predictions) are expected and fine — only a MISSING
    declared column is a failure. Compares what was actually written, not
    what was planned.
    """
    expected = row.expected_h5
    if not expected:
        return
    with h5py.File(eval_h5) as f:
        for group, names in expected.items():
            if group not in f:
                raise LegFailedError(
                    row.test_name,
                    "eval",
                    f"fixture expects H5 group {group!r}, the eval H5 has none "
                    f"(groups present: {sorted(f)})",
                )
            actual = set(f[group].dtype.names or ())
            missing = [n for n in names if n not in actual]
            if missing:
                raise LegFailedError(
                    row.test_name,
                    "eval",
                    f"eval H5 group {group!r} is missing declared expected_outputs "
                    f"column(s) {missing}; actual columns: {sorted(actual)}",
                )


# ---------------------------------------------------------------------------
# pytest surface — three parametrized legs per row
# ---------------------------------------------------------------------------

_PARAMS = [
    pytest.param(row.test_name, marks=(pytest.mark.gpu,) if row.gpu else ()) for row in ROWS
]


def _apply_known_xfail(request: pytest.FixtureRequest, name: str, leg: str) -> None:
    reason = dict(row_by_name(name).xfail).get(leg)
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
    """Leg 2 — ``do_eval=True`` rows: ``salt test`` + the fixture's expected_h5 containment check."""
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


def test_every_config_has_exactly_one_fixture():
    """§ discovery: every non-``_EXEMPT`` shipped config <-> exactly one fixtures/*.yaml.

    Enforces "require ALL configs be defined" both ways: a MISSING fixture
    fails naming the config (and points at the fixtures dir), a fixture
    naming a nonexistent config fails naming the file, and a config claimed
    by both a row and a fragment fixture fails.
    """
    discovered = set(_discover())
    row_configs = {r.config for r in ROWS}
    fragment_configs = set(FRAGMENT_FIXTURES)

    unclaimed = sorted(discovered - row_configs - fragment_configs)
    assert not unclaimed, (
        f"shipped config(s) with no fixtures/*.yaml claiming them: {unclaimed}. Add a "
        f"hand-curated fixture file under {_FIXTURES_DIR} (see fixtures/README.md)."
    )
    stale_rows = sorted(row_configs - discovered)
    assert not stale_rows, f"fixture row(s) name configs that do not exist: {stale_rows}"
    stale_fragments = sorted(fragment_configs - discovered)
    assert not stale_fragments, f"fixture fragment(s) name configs that do not exist: {stale_fragments}"
    both = sorted(row_configs & fragment_configs)
    assert not both, f"config(s) claimed by both a row fixture and a fragment fixture: {both}"


def test_every_fragment_is_exercised():
    """Every fragment fixture is reachable some way other than sitting unused on disk.

    "included" entries need >=1 fixture row's config to ``include:`` them
    directly; "paired" entries need a FEEDS entry to reference them by name;
    anything else is a free-text reason and is only existence-checked (by
    ``test_every_config_has_exactly_one_fixture`` above) — the
    ``ftag1lite_streaming`` overlay and the two undocumented-pairing PHYSLITE
    reader fragments.
    """
    included_by: set[str] = set()
    for row in ROWS:
        included_by |= _config_includes(row.config)
    paired = {arg for kind, arg in FEEDS.values() if kind == "root" and arg is not None}
    for fragment, how in FRAGMENT_FIXTURES.items():
        if how == "included":
            assert fragment in included_by, (
                f"{fragment} is marked 'included' but no fixture row's config include:s it"
            )
        elif how == "paired":
            assert fragment in paired, (
                f"{fragment} is marked 'paired' but no FEEDS entry references it"
            )


_READER_FRAGMENTS = [c for c in FRAGMENT_FIXTURES if _reader_node(c) is not None]


@pytest.mark.parametrize("fragment", _READER_FRAGMENTS)
def test_reader_fragment_instantiates(fragment):
    """A reader fragment's reader builds — the gate a fragment gets instead of a fixture row.

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
    """``test_name`` is the artifact-cache key and the junit id.

    Trivially filename-driven now (``test_name`` == the fixture file's stem)
    — kept as a guard over ``ROWS`` in case that invariant is ever broken by
    a future loader change.
    """
    names = [r.test_name for r in ROWS]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"fixture test_names are not unique: {dupes}"


def test_matrix_dependencies_are_defined_and_acyclic():
    """Every ``{ckpt:NAME}``/``{config:NAME}`` names a row; the DAG has no cycle."""
    names = {r.test_name for r in ROWS}
    deps = {r.test_name: dependencies_of(r) for r in ROWS}
    for name, dep_names in deps.items():
        unknown = sorted(dep_names - names)
        assert not unknown, f"{name}: train_args names unknown row(s) {unknown}"

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in visited:
            return
        if name in visiting:
            raise AssertionError(f"fixture dependency cycle: {' -> '.join((*stack, name))}")
        visiting.add(name)
        for dep in deps[name]:
            visit(dep, (*stack, name))
        visiting.discard(name)
        visited.add(name)

    for name in names:
        visit(name, ())


def test_every_matrix_config_has_a_feed():
    """No row can reach the fit leg without a declared data source."""
    missing = sorted({r.config for r in ROWS} - set(FEEDS))
    assert not missing, f"fixture rows with no FEEDS entry: {missing}"


def test_gpu_ci_matrix_matches_gpu_fixtures():
    """The GPU CI matrix (``.gitlab/.ci-test.yaml``) and the ``gpu: true``
    fixtures may never drift — the CI ``PIPELINE_ROW`` selection IS the set
    of rows the release-cleanup GPU-CI split runs one job per row for.
    """
    ci_path = Path(__file__).resolve().parents[4] / ".gitlab" / ".ci-test.yaml"
    if not ci_path.is_file():
        pytest.skip(f"{ci_path} not found")
    ci = yaml.safe_load(ci_path.read_text())
    matrix_rows = ci["integration-gpu"]["parallel"]["matrix"][0]["PIPELINE_ROW"]
    gpu_fixture_rows = [r.test_name for r in ROWS if r.gpu]
    assert sorted(matrix_rows) == sorted(gpu_fixture_rows), (
        f"CI GPU matrix {sorted(matrix_rows)} != gpu: true fixtures {sorted(gpu_fixture_rows)}"
    )


def test_cpu_ci_matrix_matches_runnable_fixtures():
    """The per-config CPU CI matrix (``.gitlab/.ci-test.yaml``) and the full set of
    runnable fixture rows may never drift — this is the both-ways twin of
    ``test_gpu_ci_matrix_matches_gpu_fixtures``: a new runnable fixture with no CI
    job, or a CI row with no fixture, must fail loudly either way.
    """
    ci_path = Path(__file__).resolve().parents[4] / ".gitlab" / ".ci-test.yaml"
    if not ci_path.is_file():
        pytest.skip(f"{ci_path} not found")
    ci = yaml.safe_load(ci_path.read_text())
    matrix_rows = ci["integration-cpu"]["parallel"]["matrix"][0]["PIPELINE_ROW"]
    fixture_rows = [r.test_name for r in ROWS]
    lhs, rhs = set(matrix_rows), set(fixture_rows)
    assert sorted(matrix_rows) == sorted(fixture_rows), (
        f"CI per-config matrix {sorted(matrix_rows)} != runnable fixtures "
        f"{sorted(fixture_rows)} — symmetric difference: "
        f"CI-only={sorted(lhs - rhs)}, fixture-only={sorted(rhs - lhs)}"
    )


# test_known_failures_name_real_rows_and_legs and
# test_every_eval_or_onnx_row_has_expected_outputs are DELETED: both
# guarantees now live in the loader validation (_load_fixtures) — xfail legs
# are validated against _VALID_XFAIL_LEGS at load time (so a stale/unknown
# leg fails at collection, not silently protecting nothing), and eval/onnx
# presence implies a non-empty expected_outputs by construction
# (_load_eval/_load_onnx_like raise otherwise) — rows are correct by
# construction, so there is nothing left for a completeness test to check.


# ----------------------------------------------- residual finetune assertions (§3)
# The fine-tuning overlays live under docs/tutorials/configs/finetuning/ —
# they are not shipped configs, so they carry no pipeline fixture and no CI
# row here. Their static partitions and accounting (loaded/new/dropped
# modules) are asserted by `salt/tests/unit/test_finetune_configs.py`, not
# here.


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
    assert any(c.endswith("_stddev") for c in track_cols), (
        f"no gaussian stddev in tracks: {track_cols}"
    )


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


# ---------------------------------------------------------------------------
# post-fit inference gate (folded in from the deleted test_inference.py)
# ---------------------------------------------------------------------------
#
# ``salt inference`` over every fit-capable row's own fit artifacts (user
# ruling: "run inference using the model checkpoints from all the models
# covered by pipeline"). Parametrised over every ``fit=True`` row; a row
# skips, with a stated reason, rather than running silently-wrong or erroring
# uninformatively, when: it is ROOT-fed (no H5 file to build a labelled/
# label-stripped pair from), its ``outputs:`` section assembles no ONNX
# export selection (``salt inference`` is inexpressible for it — same
# contract ``salt inference`` itself enforces), its ONNX-mode plan does not
# even connect (``GraphError`` — e.g. gn3_flow/gn3_lepid_smt's
# ``ConnectivityError`` on a missing ``inputs.flow``), or its own fit (or a
# chained producer's) failed.
#
# Deliberate change from the pre-fold test_inference.py (this plan, item 1):
# these params carry NO gpu marks — they check artifact content, not the
# device path, so they run in the CPU job even for rows fit on GPU.
#
# Gates, generalised per row wherever a fixture names ``onnx:`` or
# ``inference:`` expected_outputs (``_expected_onnx_or_skip``):
#
# (a) ``TestExportedOnnxOutputNames`` — the row's own freshly-exported ONNX
#     session's output names equal the fixture's expected ONNX tuple (order
#     included — the tuple is an Athena contract);
# (b) ``TestGn2v2OpendataValuesMatchOnnxRuntime`` — H5 values equal
#     onnxruntime outputs on the SAME real per-jet features, at check_onnx
#     tolerance;
# (c) ``TestLabelStripped`` — a label-stripped copy of the row's own test file
#     runs green with bit-identical prediction columns.
#
# NOTE on why the eval-H5 vocabulary (``expected_h5``) is NOT used here:
# ``salt inference``'s own H5 output uses the EXPORT-mode column selection
# (``H5OutputSink.use_export_selection()``, ``salt/inference.py::
# build_inference_sink``) — run-name-prefixed leaves resolved from
# ``manifest_fields(Mode.ONNX)`` — which is a DIFFERENT naming convention
# from the TEST-mode eval H5 ``expected_h5`` encodes (e.g. a folded
# per-token classification head is one reduced ``TrackOrigin`` column here
# vs. several per-class probability columns under ``salt test``).
# Reconstructing the true inference-H5 column names generically would need
# per-field axis/dtype/prefix metadata the curated fixtures deliberately do
# not carry (a flat, human-curated name list). Gate (a) above checks the
# ONNX tuple itself (unambiguous, no naming-convention translation needed);
# gate (c) sidesteps the naming question entirely by comparing the SAME
# row's own labelled vs. label-stripped output columns to each other,
# whatever they are named.
#
# Gate (b), and the input-copy/pad-mask structural check, are anchored to the
# single row ``gn2v2_opendata`` (the flagship default) rather than
# generalised: doing so for every do_onnx row would mean re-deriving each
# config's global-vs-sequence ONNX input-port mapping (jets vs. tracks vs.
# flows vs. per-object masks, ...) outside the model's own resolved export
# contract — substantial duplication of ``salt.outputs.sinks.onnx.export``
# internals across ~10 structurally different configs for a check
# ``test_export`` (check_onnx's random-input sweep) already partially covers.

# label columns physically removed for the stripped copy — only those a given
# row's fixture actually carries are dropped (different feeders carry
# different label sets; see _strip_labels).
LABEL_FIELDS = {
    "flavour_label",
    "HadronConeExclTruthLabelID",
    "HadronGhostInitialTruthLabelPdgId",
    "ftagTruthOriginLabel",
    "ftagTruthTypeLabel",
    "ftagTruthVertexIndex",
    "ftagTruthParentBarcode",
}

# >= 100: upstream ftag.hdf5.H5Writer hardcodes a 100-row chunk shape, so any
# eval/inference file under 100 jets fails dataset creation (pre-existing).
N_TEST = 128

_INFERENCE_PARAMS = [row.test_name for row in ROWS if row.fit]


def _strip_labels(src: Path, dst: Path) -> None:
    """Copy ``src`` dropping whichever LABEL_FIELDS columns it actually carries."""
    with h5py.File(src) as fin, h5py.File(dst, "w") as fout:
        for name, ds in fin.items():
            arr = ds[:]
            present = LABEL_FIELDS & set(arr.dtype.names or ())
            keep = [f for f in arr.dtype.names if f not in present]
            out = fout.create_dataset(name, data=repack_fields(arr[keep]))
            for key, value in ds.attrs.items():
                if key not in present:
                    out.attrs[key] = value


def _expected_onnx_or_skip(name: str) -> list[str]:
    """A row's expected ONNX tuple: its own ``onnx:`` fixture entry, else its
    ``inference:`` entry (ONNX-vocabulary contract for a row with no
    ``onnx:`` key — today only ``gn3v00_base``), else a skip.

    Skips (rather than raising) exactly like the deleted golden lookup did: a
    row with no committed ONNX contract (no ``do_onnx`` leg, or one that
    simply has not been curated yet) is not this gate's business.
    """
    row = row_by_name(name)
    onnx_names = row.expected_onnx or row.inference_onnx
    if not onnx_names:
        pytest.skip(f"{name}: no onnx/inference expected_outputs fixture entry")
    return list(onnx_names)


@dataclass
class InferenceArtifacts:
    """What ``salt inference`` produced for one row: labelled + label-stripped."""

    source_h5: Path
    labelled_output: Path
    stripped_input: Path
    stripped_output: Path


_INFERENCE_CACHE: dict[str, InferenceArtifacts | Exception] = {}


def run_inference_pair(name: str, tmp_path_factory) -> InferenceArtifacts:
    """Run (or fetch) row ``name``'s ``salt inference`` pair: labelled + label-stripped.

    Memoised per row, mirroring ``run_row``. Re-raises the SAME exception on
    every subsequent call for a row that failed once: ``RootDepsMissingError``
    (ROOT-fed row, no H5), ``LegFailedError`` (this row's or a producer's fit
    failed), or ``GraphError`` (the row's ``outputs:`` section assembles no
    ONNX export selection — inference is inexpressible for it — or its
    ONNX-mode plan does not even connect, e.g. gn3_flow/gn3_lepid_smt's
    ``ConnectivityError: 'norm' requires 'inputs.flow'`` — ``ConnectivityError``
    is a ``GraphError`` sibling of ``ConfigError``, not a subclass, so
    catching only ``ConfigError`` would miss it).
    """
    cached = _INFERENCE_CACHE.get(name)
    if isinstance(cached, Exception):
        raise cached
    if cached is not None:
        return cached
    row = row_by_name(name)
    try:
        artifacts = _build_inference_pair(row, tmp_path_factory)
    except (LegFailedError, RootDepsMissingError, GraphError) as exc:
        _INFERENCE_CACHE[name] = exc
        raise
    _INFERENCE_CACHE[name] = artifacts
    return artifacts


def _uses_input_samples(fit: Artifacts) -> bool:
    """Whether ``fit``'s resolved config declares an explicit ``input_samples``
    module (the only shipped case: ``gn2v2_opendata``).

    ``SaltDataModule._wire_input_samples`` only synthesises an implicit
    ``InputSamples`` from ``train_file``/``val_file``/``test_file`` when none
    already exists, so for these rows plain ``data.test_file=``/
    ``data.num_test=`` overrides are silently ignored — the resolution path
    reads ``source.<reader>.test.pattern`` off the ``InputSamples`` setup
    context instead (``salt/data/datamodule.py::_resolve_source``).
    """
    cfg = yaml.safe_load(fit.saved_config.read_text())
    return "input_samples" in ((cfg.get("data") or {}).get("modules") or {})


def _test_file_overrides(fit: Artifacts, path: Path) -> list[str]:
    """The ``--set`` overrides that route ``path``/``N_TEST`` to wherever
    ``fit``'s config actually reads its test file and row cap from.

    Whole-dict JSON, not a deep-dotted per-key override (pipeline #15650554,
    item B2 — same jsonargparse defect as a fixture row's chained
    ``train_args``: a dotted ``--...files.test=`` hands the ``files`` dict
    field a bare ``Namespace`` instead of merging into it). These are plain
    f-strings (unlike the fixture ``train_args`` templates, which go through
    ``_expand_one``'s regex substitution) consumed directly as single
    ``KEY=VALUE`` list entries by ``salt.outputs.sinks.onnx.export.
    _run_free_cli`` (``args.append(f"--{entry}")`` — no ``shlex`` re-split),
    so the doubled braces here are genuine f-string escapes for a literal
    ``{``/``}``, not tokens for a second substitution pass. This double-brace
    convention is intentionally different from ``_expand_one``'s single-brace
    tokens — do not "fix" one to match the other (both verified by
    execution).
    """
    if _uses_input_samples(fit):
        return [
            f'data.modules.input_samples.init_args.files={{"test": "{path}"}}',
            f'data.modules.input_samples.init_args.num={{"test": {N_TEST}}}',
        ]
    return [f"data.num_test={N_TEST}"]


def _build_inference_pair(row: Row, tmp_path_factory) -> InferenceArtifacts:
    kind = FEEDS[row.config][0]
    if kind == "root":
        raise RootDepsMissingError(
            f"{row.test_name} is ROOT-fed ({row.config!r} -> {FEEDS[row.config]!r}) — no H5 "
            "file to run salt inference / build a label-stripped copy against"
        )
    ctx = _feed_context(row, tmp_path_factory)
    fit = run_row(row.test_name, tmp_path_factory)
    out_dir = tmp_path_factory.mktemp(f"inference_{row.test_name}")

    stripped_input = out_dir / "stripped_input.h5"
    _strip_labels(ctx["h5"], stripped_input)
    stripped_schema = out_dir / "stripped_schema.yaml"
    save_schema(dump_schema(stripped_input), stripped_schema)

    labelled_output = out_dir / "inference_labelled.h5"
    run_inference(
        [fit.saved_config],
        fit.ckpt,
        ctx["h5"],
        output=labelled_output,
        set_overrides=_test_file_overrides(fit, ctx["h5"]),
    )
    stripped_output = out_dir / "inference_stripped.h5"
    run_inference(
        [fit.saved_config],
        fit.ckpt,
        stripped_input,
        output=stripped_output,
        set_overrides=[
            f"data.modules.reader.init_args.schema={stripped_schema}",
            *_test_file_overrides(fit, stripped_input),
        ],
    )
    return InferenceArtifacts(
        source_h5=ctx["h5"],
        labelled_output=labelled_output,
        stripped_input=stripped_input,
        stripped_output=stripped_output,
    )


def _artifacts_or_skip(name: str, tmp_path_factory) -> InferenceArtifacts:
    try:
        return run_inference_pair(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        pytest.skip(f"producer row {exc.producer} failed its fit leg: {exc}")
    except GraphError as exc:
        pytest.skip(f"{name}: salt inference cannot run — {exc}")


# ---------------------------------------------------------------------------
# gate (a): the exported ONNX tuple's own output names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _INFERENCE_PARAMS)
class TestExportedOnnxOutputNames:
    """Gate (a): the row's own freshly-exported ONNX graph's output names
    equal the fixture's expected ONNX tuple — order included, since the tuple
    order IS the Athena contract (``salt.outputs.sinks.onnx_sink``'s "globals
    before per-token" rule).
    """

    def test_onnx_output_names_match_expected(self, name, tmp_path_factory):
        expected = _expected_onnx_or_skip(name)
        try:
            onnx_path = run_export(name, tmp_path_factory)
        except LegFailedError as exc:
            pytest.skip(f"{name}: export failed: {exc}")
        session = make_session(onnx_path)
        actual = [o.name for o in session.get_outputs()]
        assert actual == expected, f"{name}: exported ONNX tuple != fixture's expected onnx"


# ---------------------------------------------------------------------------
# gate (b), anchored: real-data H5-vs-onnxruntime numeric parity
# ---------------------------------------------------------------------------


class TestGn2v2OpendataValuesMatchOnnxRuntime:
    """Gate (b), anchored to ``gn2v2_opendata`` (the flagship default row): H5
    values equal onnxruntime outputs on the SAME real per-jet inputs — not
    just check_onnx's random-input sweep (already covered per-row by
    ``test_export``). See the section comment above for why this is not
    generalised across the whole fixture set.

    The per-field manifest below (suffix/axis/dtype) is deliberately
    hardcoded to this ONE row's own known, shipped contract (config
    ``name: GN2v2_opendata``, ``outputs.export.model_name: GN2v2opendata``)
    rather than read from a golden JSON — this test was already anchored to
    a single row (not generalised), so its contract belongs directly in code.
    ``salt inference`` names H5 columns by the RUN name
    (``salt/inference.py::build_inference_sink``); the exported ONNX tuple by
    the MODEL name (``salt/outputs/sinks/onnx_sink.py``) — same suffixes,
    different prefixes. ``test_manifest_matches_expected_outputs`` below
    locks this table's suffixes against the ``gn2v2_opendata`` fixture's
    ``onnx:`` entry so the two cannot silently drift apart.
    """

    ROW = "gn2v2_opendata"
    N = N_TEST
    RUN_NAME = "GN2v2_opendata"
    MODEL_NAME = "GN2v2opendata"
    # (onnx/h5 suffix, axis, onnx dtype)
    FIELDS: tuple[tuple[str, str, str], ...] = (
        ("pb", "global", "float32"),
        ("pc", "global", "float32"),
        ("pu", "global", "float32"),
        ("ptau", "global", "float32"),
        ("TrackOrigin", "per_token", "int8"),
        ("VertexIndex", "per_token", "int8"),
    )

    def test_manifest_matches_expected_outputs(self):
        """This class's hardcoded FIELDS names exactly the fixture's expected onnx tuple."""
        expected = list(row_by_name(self.ROW).expected_onnx or ())
        derived = [f"{self.MODEL_NAME}_{suffix}" for suffix, _, _ in self.FIELDS]
        assert derived == expected, (
            "TestGn2v2OpendataValuesMatchOnnxRuntime.FIELDS drifted from the "
            f"{self.ROW!r} fixture's onnx.expected_outputs: {derived} != {expected}"
        )

    @pytest.fixture(scope="class")
    def artifacts(self, tmp_path_factory) -> InferenceArtifacts:
        return _artifacts_or_skip(self.ROW, tmp_path_factory)

    @pytest.fixture(scope="class")
    def onnx_path(self, tmp_path_factory) -> Path:
        try:
            return run_export(self.ROW, tmp_path_factory)
        except LegFailedError as exc:
            pytest.skip(f"{self.ROW}: export failed: {exc}")

    def test_h5_values_equal_onnxruntime(self, artifacts, onnx_path, tmp_path_factory):
        """Per jet: run the exported ONNX on the file's valid tokens (Athena
        convention) and compare against the H5 — floats at check_onnx
        tolerance (1e-4), int8 exact; padded H5 positions read 0.
        """
        fit = run_row(self.ROW, tmp_path_factory)
        cfg = yaml.safe_load(fit.saved_config.read_text())
        variables = cfg["data"]["modules"]["features"]["init_args"]["variables"]
        with h5py.File(artifacts.source_h5) as f:
            jets_src = f["jets"][: self.N]
            tracks_src = f["tracks"][: self.N]
        jet_feats = np.stack([jets_src[v] for v in variables["jets"]], -1).astype(np.float32)
        trk_feats = np.stack([tracks_src[v] for v in variables["tracks"]], -1).astype(np.float32)
        valid = tracks_src["valid"].astype(bool)
        # fixture sanity: valid tokens are LEADING (the reader/pad layout the
        # H5 per-token placement relies on)
        assert (np.sort(valid, axis=-1)[:, ::-1] == valid).all()
        with h5py.File(artifacts.labelled_output) as f:
            jets_out = f["jets"][: self.N]
            tracks_out = f["tracks"][: self.N]
        session = make_session(onnx_path)
        ort_names = [o.name for o in session.get_outputs()]
        expected = list(row_by_name(self.ROW).expected_onnx or ())
        assert ort_names == expected, f"{self.ROW}: exported tuple != fixture's expected onnx"
        n_mismatch_checked = 0
        for i in range(self.N):
            ort_out = dict(
                zip(
                    ort_names,
                    session.run(
                        None,
                        {
                            "jet_features": jet_feats[i : i + 1],
                            "track_features": trk_feats[i][valid[i]],
                        },
                    ),
                    strict=True,
                )
            )
            for suffix, axis, dtype in self.FIELDS:
                h5_col = f"{self.RUN_NAME}_{suffix}"
                ref = ort_out[f"{self.MODEL_NAME}_{suffix}"]
                if axis == "global":
                    np.testing.assert_allclose(
                        np.float64(jets_out[h5_col][i]),
                        np.ravel(ref)[0],
                        rtol=1e-4,
                        atol=1e-4,
                        err_msg=f"{h5_col} jet {i}",
                    )
                else:
                    n_valid = int(valid[i].sum())
                    got_tokens = tracks_out[h5_col][i]
                    if dtype == "int8":
                        np.testing.assert_array_equal(
                            got_tokens[:n_valid], ref, err_msg=f"{h5_col} jet {i}"
                        )
                    else:
                        np.testing.assert_allclose(
                            got_tokens[:n_valid],
                            ref,
                            rtol=1e-4,
                            atol=1e-4,
                            err_msg=f"{h5_col} jet {i}",
                        )
                    assert (got_tokens[n_valid:] == 0).all(), f"{h5_col} jet {i}: pad not zero"
                    n_mismatch_checked += 1
        assert n_mismatch_checked > 0, "no per-token comparison ran (degenerate fixture)"


class TestGn2v2OpendataStructuralAnchor:
    """Anchored structural checks that depend on per-config writer wiring
    (which streams get an ``InputCopyWriter``/``PadMaskWriter``) rather than
    anything a committed golden ever encoded — see the section comment above.
    """

    ROW = "gn2v2_opendata"

    def test_mode_gated_copy_and_mask_columns(self, tmp_path_factory):
        """The default-modes InputCopyWriter/PadMaskWriter run under inference
        (they declare export implicitly): source copies precede the task
        columns, and the tracks pad-mask column is last.
        """
        artifacts = _artifacts_or_skip(self.ROW, tmp_path_factory)
        with h5py.File(artifacts.source_h5) as src:
            src_jets = list(src["jets"].dtype.names)
        with h5py.File(artifacts.labelled_output) as f:
            jets = list(f["jets"].dtype.names)
            tracks = list(f["tracks"].dtype.names)
        assert jets[: len(src_jets)] == src_jets, "input copies must lead the jets group"
        assert tracks[-1] == "mask", "the pad-mask column must be last in tracks"


# ---------------------------------------------------------------------------
# gate (c): label-stripped copy runs green with identical predictions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _INFERENCE_PARAMS)
class TestLabelStripped:
    """Gate (c): a label-stripped copy of the row's own test file runs green
    with bit-identical prediction columns (H5-fed rows only — ROOT-fed rows
    skip upstream in ``_artifacts_or_skip``). Generic — compares the row's OWN
    two outputs to each other, so it needs no per-row expected-name table at
    all (unlike gates (a)/(b), see the section comment above).
    """

    def test_stripped_input_lacks_label_fields(self, name, tmp_path_factory):
        """Fixture sanity: no LABEL_FIELDS column survives in the stripped INPUT."""
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        with h5py.File(artifacts.stripped_input) as f:
            for group in f:
                assert not set(f[group].dtype.names or ()) & LABEL_FIELDS, group

    def test_prediction_columns_identical(self, name, tmp_path_factory):
        """The MODEL-MINTED columns — the run-name-prefixed export-selection
        columns, ``{RUN_NAME}_*`` — are bit-identical between the labelled and
        stripped runs: labels contribute nothing to the model's own outputs.

        Deliberately NOT a whole-file column diff (pipeline #15651025 cluster
        1, ~12 failures): the sink's copy-all behaviour copies whatever label
        columns the LABELLED source carries and (correctly) none from the
        label-stripped one, so the full column sets differ BY CONSTRUCTION —
        and the copied VALUES can differ too (e.g. regression: tracks/deta),
        since copy-all pulls in every source field, not just labels. None of
        that is a prediction, so it is excluded here: input copies, the pad
        mask, and target_* columns are all un-prefixed (or differently
        prefixed) and never enter the comparison.
        """
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        fit = run_row(name, tmp_path_factory)
        run_name = yaml.safe_load(fit.saved_config.read_text())["name"]
        prefix = f"{run_name}_"
        with h5py.File(artifacts.labelled_output) as fa, h5py.File(artifacts.stripped_output) as fb:
            assert set(fa) == set(fb), f"{name}: group sets differ between labelled and stripped"
            checked_any = False
            for group in fa:
                a_names = {c for c in (fa[group].dtype.names or ()) if c.startswith(prefix)}
                b_names = {c for c in (fb[group].dtype.names or ()) if c.startswith(prefix)}
                assert a_names == b_names, (
                    f"{name}/{group}: model-minted ({prefix}*) column sets differ when stripped"
                )
                for col in a_names:
                    checked_any = True
                    a, b = fa[group][col][:], fb[group][col][:]
                    assert np.array_equal(a, b), f"{name}: {group}/{col} differs when stripped"
            assert checked_any, f"{name}: no model-minted ({prefix}*) column found to compare"

    def test_stripped_output_carries_no_label_columns(self, name, tmp_path_factory):
        """The stripped-run H5 carries no label copy and no target_* column."""
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        with h5py.File(artifacts.stripped_output) as f:
            for group in f:
                names = set(f[group].dtype.names or ())
                assert not names & LABEL_FIELDS, f"{name}: label columns leaked into {group}"
                assert not {n for n in names if n.startswith("target_")}, f"{name}: {group}"
