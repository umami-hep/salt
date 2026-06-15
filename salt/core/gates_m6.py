"""M6 gates harness — LB1 + M6-CONV (sub-wave A, Labeller) (plan 12; design §9.5, FD 1303-1304).

Standalone gates, each a subcommand of ``python -m salt.core.gates_m6``, each
writing a machine-readable ``<gate>_report.json`` into ``--outdir`` and a
human-readable table to stdout, exiting non-zero on failure (the
gates_m2/m3/m4/m5 envelope). NO machine data path lives in this file (design §5
placeholder policy): every gate generates its own dummy fixtures into ``--outdir``
(or in-memory), and compares an INDEPENDENT v1 reference against the v2 module
driven through the real plan compiler / two-phase bind / executor (or, for a
pure data-op gate like LB1, the bound processor's ``process``). The M6-CONV gate
drives the REAL ``salt2 graph validate`` subcommand on the newly-authored
v2-native M6 configs (the M5-CONV ``run_conv`` pattern), data-free via the
parity norm dict written into ``--outdir``.

This is the FIRST M6 wave — it bootstraps the harness mirroring gates_m5
EXACTLY (the same ``run_<gate>(outdir, *, corruption=None) -> (rc, report)``
shape, ``checks`` dict + ``passed = all(...)``, ``<gate>_report.json`` + stdout
verdict table, non-zero exit, a python-only ``corruption`` negative-control
hook proving the parity comparison has teeth, never exposed on the CLI). Later
M6 sub-waves append their gates (MU1-2/ED1-2/VS1/CM1/LR1/S31/IG1 + M6-CONV) to
this same file.

Gate criteria (each justified in its ``run_*`` docstring vs the design/v1 ref):

- **LB1 Labeller label-derivation parity (v1-decidable)** — the on-the-fly ftag
  ``Labeller``, wired as opt-in ``init_args`` on the v2 `Labels` processor
  (``data/processors.py``), produces the SAME int labels as v1's
  ``SaltDataset.process_labels`` (``datasets.py:609-626``) on a fixed structured
  fixture, BITWISE, for both shipped shapes: GN3X (9-class, ``require_labels:
  True``) and a GN2X_qcdsplit-style set (7-class, ``require_labels: False`` —
  unlabelled objects dropped). The v1 reference is INDEPENDENT: it composes the
  ftag ``Labeller`` directly and reproduces v1 ``process_labels``'s
  field-subset check + ``get_labels`` + int64 cast, NOT the v2 processor's own
  output. PLUS the two named guards fire: the empty-class guard (v1
  ``LabellerConfig.__post_init__``, ``configs.py:189``) raises a ``ConfigError``
  when ``use_labeller`` is set without ``class_names``; the missing-field guard
  (v1 ``datasets.py:622-624``) raises a ``ValueError`` when a labeller cut
  variable is absent from the raw stream; and ``require_labels: True`` raises
  (ftag ``labeller.py:70``) on an unlabelled object.

- **M6-CONV — the M7-slice acceptance (this wave's slice)** — the
  newly-authored v2-native M6 configs (this wave: GN3X, GN2X_qcdsplit; later M6
  waves EXTEND ``_CONV_M6_CONFIGS``) exist as v2-native fixtures in
  ``salt/core/configs/`` AND pass the REAL ``salt2 graph validate`` (the
  canonical static validator, the d2cfg command path) in fit + test + onnx with
  rc == 0. The authoritative config list is embedded VERBATIM in
  ``_CONV_M6_CONFIGS`` and the report; the gate exits non-zero if any listed
  config is MISSING from ``salt/core/configs/`` or FAILS any applicable mode.
  Mirrors the M5-CONV ``run_conv`` (gates_m5.py:4478): data-free validation via
  the parity norm dict written into ``--outdir`` (jets/tracks/electrons
  resolve; the GN3X flow / GN2X_qcdsplit flow+truth_hadrons streams emit
  non-fatal "missing input type" preflight WARNINGS, not errors — so NO
  ``--strict``, the same rationale as M5-CONV). Both M6 configs are standard
  traces (no muP/edge export hazards — those land in the later B/C waves), so
  onnx stays in the gate for both. Moves the 2 configs 🔷→✅ in the 39
  denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from ftag import Labeller as V1Labeller

from salt.core.data import H5StructuredReader  # noqa: F401 (parity with gates_m5 import surface)
from salt.core.data.base import WorkerCtx
from salt.core.data.processors import Labels
from salt.core.graph import Bundle, Mode
from salt.core.graph.errors import ConfigError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import TensorSpec
from salt.core.main import CONFIG_DIR
from salt.core.main import main as salt2_main
from salt.tests.core.gn2_fixture import write_parity_norm_dict

# ---------------------------------------------------------------------------
# shared report helpers (the gates_m2/m3/m4/m5 envelope, kept standalone)
# ---------------------------------------------------------------------------


def _emit_report(report: dict[str, Any], outdir: Path, gate: str) -> Path:
    """Write the gate report JSON into ``outdir`` and return its path.

    Returns
    -------
    Path
        The written ``<gate>_report.json`` path.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{gate}_report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path


def _print_verdict(gate: str, passed: bool, criterion: str, report_path: Path) -> None:
    """Print the closing verdict block every gate ends with."""
    print("=" * 96)
    print(f"GATE {gate.upper()}: {'PASS' if passed else 'FAIL'} — criterion: {criterion}")
    print(f"report: {report_path}")


def _base_report(gate: str, passed: bool, criterion: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the common report envelope shared with the M2/M3/M4/M5 gates.

    Returns
    -------
    dict[str, Any]
        Envelope with gate name, timestamp, verdict, criterion, config and
        environment fields; gate-specific sections are added by the caller.
    """
    return {
        "gate": gate,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "criterion": criterion,
        "config": config,
        "environment": {"numpy": np.__version__, "device": "cpu"},
    }


def _print_checks(checks: dict[str, bool]) -> None:
    """Print a name/PASS-FAIL table for a checks mapping."""
    for name, ok in checks.items():
        print(f"  {name:<56} {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# LB1 fixture helpers (the parity scaffold — no machine paths)
# ---------------------------------------------------------------------------

# the GN3X 9-class scheme (v1 GN3X.yaml:108, require_labels:True) — every class
# resolves in this ftag version's Flavours catalogue. Cut variables (deduped):
# R10TruthLabel_R22v1, GhostBHadronsFinalCount, GhostCHadronsFinalCount.
GN3X_CLASSES: tuple[str, ...] = (
    "htautauhad",
    "hbb",
    "hcc",
    "top",
    "qcdbb",
    "qcdbx",
    "qcdcx",
    "qcdll",
    "Wqq",
)

# a GN2X_qcdsplit-style 7-class scheme (v1 GN2X_qcdsplit.yaml:102 ships
# require_labels:False with a `qcdxx` class absent from this ftag build, so we
# use a 7-class subset of the resolvable Flavours that exercises the SAME
# behaviour: require_labels:False drops unlabelled objects, labeller.py:73).
QCDSPLIT_CLASSES: tuple[str, ...] = (
    "hbb",
    "hcc",
    "top",
    "qcdbb",
    "qcdbx",
    "qcdcx",
    "qcdll",
)

# one feature row per GN3X class, in class-index order. Each row is a
# R10TruthLabel_R22v1 / GhostBHadronsFinalCount / GhostCHadronsFinalCount triple
# chosen to satisfy exactly that class's ftag cut: htautauhad needs R10==16, hbb
# R10==11, hcc R10==12, top R10 in (1,6,7), qcdbb R10==10 & B>=2, qcdbx R10==10 &
# B==1, qcdcx R10==10 & C>=1 & B==0, qcdll R10==10 & B==0 & C==0, Wqq R10==2 &
# B<2 & C<2.
_GN3X_ROWS_BY_CLASS: tuple[tuple[int, int, int], ...] = (
    (16, 0, 0),  # 0 htautauhad
    (11, 0, 0),  # 1 hbb
    (12, 0, 0),  # 2 hcc
    (1, 0, 0),  # 3 top (also 6, 7)
    (10, 2, 0),  # 4 qcdbb (B>=2)
    (10, 1, 0),  # 5 qcdbx (B==1)
    (10, 0, 1),  # 6 qcdcx (C>=1, B==0)
    (10, 0, 0),  # 7 qcdll (B==0, C==0)
    (2, 0, 0),  # 8 Wqq (B<2, C<2)
)

_LABELLER_DTYPE = [
    ("R10TruthLabel_R22v1", "i4"),
    ("GhostBHadronsFinalCount", "i4"),
    ("GhostCHadronsFinalCount", "i4"),
]

# a row that matches NO class (R10TruthLabel value used by none of the schemes)
_UNLABELLED_ROW = (999, 0, 0)


def _gn3x_fixture() -> np.ndarray:
    """A structured array with exactly one object per GN3X class, shuffled.

    Every object IS labellable (so ``require_labels:True`` does not raise);
    interleaving the classes makes a per-class ordering bug visible.

    Returns
    -------
    np.ndarray
        The ``[R10TruthLabel_R22v1, GhostBHadronsFinalCount,
        GhostCHadronsFinalCount]`` structured fixture.
    """
    rng = np.random.default_rng(13)
    rows = list(_GN3X_ROWS_BY_CLASS)
    # repeat each class a few times so the fixture is larger than n_classes,
    # then shuffle deterministically
    rows = rows * 4
    rng.shuffle(rows)
    return np.array(rows, dtype=_LABELLER_DTYPE)


def _qcdsplit_fixture() -> np.ndarray:
    """A 7-class fixture WITH unlabelled objects (the require_labels:False case).

    Uses the GN3X row recipes for the 7 retained classes plus a sprinkling of
    rows that match no class — under ``require_labels:False`` those are dropped
    (ftag ``labeller.py:73``), so the derived array is shorter than the batch.

    Returns
    -------
    np.ndarray
        The structured fixture with labellable + unlabellable objects mixed.
    """
    rng = np.random.default_rng(29)
    # rows for the 7 qcdsplit classes (same recipes as GN3X for those names)
    by_name = dict(zip(GN3X_CLASSES, _GN3X_ROWS_BY_CLASS, strict=True))
    rows = [by_name[name] for name in QCDSPLIT_CLASSES] * 3
    rows += [_UNLABELLED_ROW] * 5  # unlabelled -> dropped under require_labels:False
    rng.shuffle(rows)
    return np.array(rows, dtype=_LABELLER_DTYPE)


def _bundle(raw: np.ndarray) -> Bundle:
    """A FIT bundle holding the raw structured ``jets`` array.

    Returns
    -------
    Bundle
        ``{raw.jets: raw}`` — the input the `Labels` processor reads.
    """
    bundle = Bundle()
    bundle.set("raw.jets", raw)
    return bundle


def _bound_labels(
    classes: Sequence[str],
    *,
    require_labels: bool,
    stream: str = "jets",
    label: str = "flavour_label",
) -> Labels:
    """Build a `Labels` processor with the labeller on, bound to a one-key plan.

    Returns
    -------
    Labels
        A bound `Labels` whose narrowed produce is exactly
        ``labels.<stream>.<label>`` — ready to ``process``/``read_fields``.
    """
    labels = Labels(
        streams=[stream],
        use_labeller=True,
        class_names=list(classes),
        require_labels=require_labels,
        labeller_stream=stream,
        labeller_label=label,
    )
    labels.name = "labels"
    step = PlanStep(
        name="labels",
        module=labels,
        requires=MappingProxyType({}),
        produces=MappingProxyType({f"labels.{stream}.{label}": TensorSpec(kind="label")}),
    )
    labels.bind(WorkerCtx(mode=Mode.FIT, read_fields={}, seed=0, step=step))
    return labels


def _v1_process_labels(
    classes: Sequence[str],
    require_labels: bool,
    raw: np.ndarray,
) -> np.ndarray:
    """INDEPENDENT v1 reference: reproduce ``SaltDataset.process_labels``.

    Composes the ftag ``Labeller`` directly (``Labeller(class_names,
    require_labels)``, exactly v1 ``datasets.py:205``), reproduces v1's
    field-subset check (``datasets.py:622-624``) and ``get_labels`` call
    (``datasets.py:626``), then the int->torch.long cast (``datasets.py:633``)
    — returned here as the equivalent int64 numpy array. This is NOT the v2
    processor's output; it is a from-scratch v1 recomputation.

    Returns
    -------
    np.ndarray
        The v1 int64 labels for the fixture.

    Raises
    ------
    ValueError
        On a missing labeller field (v1 field-subset check) or, under
        ``require_labels``, an unlabelled object (ftag ``labeller.py:70``).
    """
    labeller = V1Labeller(list(classes), require_labels)
    all_vars = set(raw.dtype.names or ())
    for var in labeller.variables:
        if var not in all_vars:
            raise ValueError("Not enough fields to apply labelling cuts.")
    labels_on_the_fly = labeller.get_labels(raw)
    # v1 datasets.py:633 — dtype=torch.long when integer; we keep numpy int64.
    if np.issubdtype(labels_on_the_fly.dtype, np.integer):
        return labels_on_the_fly.astype(np.int64)
    return np.array(labels_on_the_fly, copy=True)


def _raises(exc: type[BaseException], fn: Callable[..., Any], *args: Any) -> bool:
    """Whether ``fn(*args)`` raises ``exc``.

    Returns
    -------
    bool
        True iff the call raised an instance of ``exc``.
    """
    try:
        fn(*args)
    except exc:
        return True
    except Exception:  # noqa: BLE001 - a different exception type does NOT satisfy the named-error claim
        return False
    return False


# ---------------------------------------------------------------------------
# LB1 — Labeller label-derivation parity vs v1 (sub-wave A)
# ---------------------------------------------------------------------------


def run_lb1(
    outdir: Path | str,
    *,
    corruption: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[int, dict[str, Any]]:
    """LB1: on-the-fly Labeller int-label parity vs v1 ``process_labels``.

    The v2 `Labels` processor (``data/processors.py``) with the labeller
    init_args on derives ``labels.<stream>.<label>`` from the raw structured
    array via the ftag ``Labeller``; LB1 asserts that derived int output is
    BITWISE identical to an INDEPENDENT v1 reference (`_v1_process_labels`,
    which composes the ftag ``Labeller`` directly and reproduces v1
    ``datasets.py:609-626``) on two fixed fixtures:

    - **GN3X (9-class, require_labels:True)** — every object is labellable, so
      ``get_labels`` raises on none; the derived labels span ``range(9)``.
    - **GN2X_qcdsplit-style (7-class, require_labels:False)** — some objects
      match no class and are DROPPED (ftag ``labeller.py:73``), so the derived
      array is shorter than the batch; LB1 asserts the v1 and v2 dropped-length
      arrays match BITWISE.

    Both are a pure data op (no model, no float drift), so the claim is BITWISE.
    The derived dtype is int64 (``dtype_policy: int64-for-int``, v1
    ``datasets.py:633``).

    PLUS the named-error guards (plan 12 LB1 row; the "raise as named errors"
    constraint, v1 ``configs.py:191`` + ``datasets.py:622-623``):

    - the **empty-class guard** raises a ``ConfigError`` when ``use_labeller``
      is set without ``class_names`` (v1 ``LabellerConfig.__post_init__``);
    - the **missing-field guard** raises a ``ValueError`` when a labeller cut
      variable is absent from the raw stream (v1 ``datasets.py:622-624``);
    - **require_labels:True** raises a ``ValueError`` on an unlabelled object
      (ftag ``labeller.py:70``).

    Negative control (``test_gates_m6.py``): the ``corruption`` hook perturbs
    the v2 GN3X derived labels — the bitwise parity check must FAIL while the
    guard checks (independent of the label values) stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("LB1 on-the-fly Labeller label-derivation BITWISE parity vs v1 process_labels")
    print("=" * 96)
    checks: dict[str, bool] = {}

    # -- (a) GN3X 9-class, require_labels:True ---------------------------------
    raw_gn3x = _gn3x_fixture()
    v2_gn3x = _bound_labels(GN3X_CLASSES, require_labels=True).process(
        _bundle(raw_gn3x), np.s_[0 : len(raw_gn3x)], Mode.FIT
    )["labels.jets.flavour_label"]
    if corruption is not None:
        v2_gn3x = corruption(v2_gn3x)
    v1_gn3x = _v1_process_labels(GN3X_CLASSES, True, raw_gn3x)
    checks["gn3x_require_true_bitwise"] = v2_gn3x.shape == v1_gn3x.shape and np.array_equal(
        v2_gn3x, v1_gn3x
    )
    checks["gn3x_dtype_is_int64"] = v2_gn3x.dtype == np.int64
    checks["gn3x_spans_nine_classes"] = set(v1_gn3x.tolist()) == set(range(len(GN3X_CLASSES)))

    # -- (b) GN2X_qcdsplit-style 7-class, require_labels:False (drops) ---------
    raw_qcd = _qcdsplit_fixture()
    v2_qcd = _bound_labels(QCDSPLIT_CLASSES, require_labels=False).process(
        _bundle(raw_qcd), np.s_[0 : len(raw_qcd)], Mode.FIT
    )["labels.jets.flavour_label"]
    v1_qcd = _v1_process_labels(QCDSPLIT_CLASSES, False, raw_qcd)
    checks["qcdsplit_require_false_bitwise"] = v2_qcd.shape == v1_qcd.shape and np.array_equal(
        v2_qcd, v1_qcd
    )
    # require_labels:False drops the unlabelled objects -> derived array shorter
    checks["qcdsplit_drops_unlabelled"] = len(v1_qcd) < len(raw_qcd)
    checks["qcdsplit_dtype_is_int64"] = v2_qcd.dtype == np.int64

    # -- (c) read_fields declares the labeller cut variables ------------------
    labels_gn3x = _bound_labels(GN3X_CLASSES, require_labels=True)
    step = PlanStep(
        name="labels",
        module=labels_gn3x,
        requires=MappingProxyType({}),
        produces=MappingProxyType({"labels.jets.flavour_label": TensorSpec(kind="label")}),
    )
    declared = set(labels_gn3x.read_fields(step).get("jets", {}))
    expected_vars = set(dict.fromkeys(V1Labeller(list(GN3X_CLASSES), True).variables))
    checks["read_fields_declares_cut_variables"] = declared == expected_vars
    # the derived label name itself is NOT read from disk (it has no producer)
    checks["read_fields_excludes_derived_label"] = "flavour_label" not in declared

    # -- (d) named-error guards ------------------------------------------------
    # empty-class guard: use_labeller=True without class_names -> ConfigError
    checks["empty_class_guard_raises_configerror"] = _raises(
        ConfigError, lambda: Labels(use_labeller=True)
    )
    # missing-field guard: a labeller cut variable absent -> ValueError
    raw_missing = np.array([(11,), (10,)], dtype=[("R10TruthLabel_R22v1", "i4")])
    checks["missing_field_guard_raises_valueerror"] = _raises(
        ValueError,
        lambda: _bound_labels(GN3X_CLASSES, require_labels=True).process(
            _bundle(raw_missing), np.s_[0:2], Mode.FIT
        ),
    )
    # require_labels:True raises on an unlabelled object (ftag labeller.py:70)
    raw_unlabelled = np.array([(11, 0, 0), _UNLABELLED_ROW], dtype=_LABELLER_DTYPE)
    checks["require_labels_true_raises_on_unlabelled"] = _raises(
        ValueError,
        lambda: _bound_labels(GN3X_CLASSES, require_labels=True).process(
            _bundle(raw_unlabelled), np.s_[0:2], Mode.FIT
        ),
    )

    passed = all(checks.values())
    criterion = (
        "the v2 Labels-processor on-the-fly Labeller get_labels int output is BITWISE identical to "
        "an INDEPENDENT v1 process_labels (ftag Labeller composed directly, datasets.py:609-626) "
        "for GN3X (9-class require_labels:True, all labelled) and a GN2X_qcdsplit-style set "
        "(7-class require_labels:False, unlabelled objects dropped); read_fields declares the "
        "labeller's cut variables (FD §6.1 1280-1282) and not the derived label; the empty-class "
        "guard (ConfigError), missing-field guard (ValueError) and require_labels:True "
        "unlabelled-object guard (ValueError) all fire as named errors"
    )
    report = _base_report(
        "lb1_labeller_parity",
        passed,
        criterion,
        {
            "gn3x_classes": list(GN3X_CLASSES),
            "qcdsplit_classes": list(QCDSPLIT_CLASSES),
            "n_gn3x_rows": len(raw_gn3x),
            "n_qcdsplit_rows": len(raw_qcd),
            "n_qcdsplit_labelled": len(v1_qcd),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["v1_reference"] = (
        "INDEPENDENT v1 reference: ftag Labeller(class_names, require_labels) composed directly + "
        "v1 process_labels field-subset check + get_labels + int64 cast (datasets.py:609-633)"
    )
    _print_checks(checks)
    _print_verdict("lb1", passed, criterion, _emit_report(report, outdir, "lb1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# M6-CONV — the M7-slice acceptance for the M6-authored configs (sub-wave A
# bootstraps it; later M6 waves EXTEND _CONV_M6_CONFIGS)
# ---------------------------------------------------------------------------

# The AUTHORITATIVE M6-CONV config list (plan 12 M6-CONV row: "embed the
# authoritative _CONV_M6_CONFIGS list verbatim"). This wave (sub-wave A,
# Labeller) lands the FIRST two; later waves append GN2_muP (B), GN2XE (C),
# legacy/DL1 (D) until all 5 config-gating needs-M6 configs are here.
#
# `norm_global` is False for both (neither config carries a SECOND `norm_global`
# Normaliser — that is the VectorConcat/global-stream pattern, absent here).
# `onnx == "validate"` for both: GN3X and GN2X_qcdsplit are STANDARD traces — no
# muP MuReadout fold (B) or edge dynamic-T register pad (C) export hazard — so
# onnx stays in the gate (plan 12 M6-CONV per-config export contracts: "GN3X/
# GN2X_qcdsplit/DL1 are standard traces").
_CONV_M6_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "GN3X",
        "cfg": ("GN3X.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "labeller",
        "note": "boosted GN3X: 9-class on-the-fly Labeller (require_labels:True) + lion + LossGLS",
    },
    {
        "name": "GN2X_qcdsplit",
        "cfg": ("GN2X_qcdsplit.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "labeller",
        "note": (
            "GN2X 7-class QCD-split Labeller (require_labels:False, drop-unlabelled) + "
            "truth_hadrons third stream; LossSum"
        ),
    },
)


def _conv_norm_dict(outdir: Path) -> Path:
    """Write the parity norm/class dicts into ``outdir``; return the norm-dict path.

    Reuses the M2/M5 ``write_parity_norm_dict`` (DISTINCT per-variable
    constants, so a field-order mismatch can't pass silently) — the SAME
    data-free norm-dict source M5-CONV's ``run_conv`` uses (gates_m5.py:451).

    Returns
    -------
    Path
        The norm-dict YAML path passed to each config's Normaliser override.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    nd, cd = outdir / "norm_dict.yaml", outdir / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd


def _conv_set_args(entry: dict[str, Any], norm_dict: Path) -> list[str]:
    """Build the data-free ``--set`` overrides for one config (norm + norm_global).

    Every shipped v2 config materialises its `Normaliser` from a norm dict at
    setup; static validation supplies it data-free via the documented
    ``--set model.modules.norm.init_args.norm_dict=<path>`` (each config header,
    gates_m5.py:4439). A config with a SECOND `norm_global` Normaliser needs its
    own override (neither M6-A config has one).

    Returns
    -------
    list[str]
        The ``--set KEY=VALUE`` flag pairs (already split for ``salt2_main``).
    """
    args = ["--set", f"model.modules.norm.init_args.norm_dict={norm_dict}"]
    if entry["norm_global"]:
        args += ["--set", f"model.modules.norm_global.init_args.norm_dict={norm_dict}"]
    return args


def _conv_modes(entry: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Resolve the (mode, report-key) pairs to validate for one config.

    Always fit + test; onnx only when the config is export-representable (the
    ``onnx`` field is ``"validate"``). Both M6-A configs are standard traces, so
    both validate onnx too.

    Returns
    -------
    tuple[tuple[str, str], ...]
        ``((cli_mode, report_key), ...)`` — the modes `salt2 graph validate` is
        actually invoked for.
    """
    modes = [("fit", "validateFit"), ("test", "validateTest")]
    if entry["onnx"] == "validate":
        modes.append(("onnx", "validateOnnx"))
    return tuple(modes)


def run_conv(
    outdir: Path | str,
    *,
    corruption: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """M6-CONV: the M7-slice acceptance for the M6-authored v2-native configs.

    For EVERY M6 config landed so far (``_CONV_M6_CONFIGS`` — this wave: GN3X,
    GN2X_qcdsplit; later waves EXTEND the list), this drives the canonical static
    validator — the REAL ``salt2 graph validate`` subcommand (``salt2_main``, the
    SAME command path d2cfg / M5-CONV exercise) — in fit + test (+ onnx where the
    config is export-representable) and asserts rc == 0 for every applicable mode.
    The authoritative config list is embedded VERBATIM in the report (plan 12
    M6-CONV row: "embed the authoritative ``_CONV_M6_CONFIGS`` list verbatim").
    The gate exits non-zero if any listed config is MISSING from
    ``salt/core/configs/`` or if any config FAILS a mode (plan 12: "fail if any
    missing or any mode rc != 0").

    Why NO ``--strict`` (identical rationale to M5-CONV's ``run_conv``): a
    data-free validation cannot satisfy ``--strict``. The acceptance level is
    ``rc == 0`` (no ERROR-level finding, design §4.2), which IS the
    convert+validate+plan-compile criterion. ``--strict`` promotes EVERY warning
    to an error (cli.py:919-920), and several warnings are inherent to data-free
    validation and orthogonal to convertibility:

    1. the no-``schema:`` warning (field spellings cannot be checked statically);
    2. warning-level deadcode (per-mode "produced but never consumed" infos);
    3. `Normaliser` preflight warnings for the streams the parity norm dict does
       NOT supply — it covers jets/tracks/electrons but NOT the GN3X ``flow`` or
       the GN2X_qcdsplit ``flow``/``truth_hadrons`` streams, so those emit
       non-fatal "missing input type 'flow'/'truth_hadrons'" preflight warnings.

    The parity norm dict is still written so the jets/tracks preflight resolves
    and the bind succeeds — the remaining flow/truth_hadrons warnings are
    data-shape artifacts, not graph errors.

    Forward-parity: M6-CONV makes NO forward-parity claim. The Labeller's
    label-derivation parity is owned by LB1; this gate is the static
    convert+validate+plan-compile slice for the new configs.

    Parameters
    ----------
    outdir : Path | str
        Report output directory (the parity norm dict is written here too).
    corruption : Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None
        Test-only hook (the gates_m2/m3/m4/m5 pattern): transforms the embedded
        config list before validation (e.g. inject a missing/bogus config) to
        prove the completeness + per-config assertions have teeth. Never on CLI.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every listed config validated every
        applicable mode (and every config file exists).
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("M6-CONV acceptance — salt2 graph validate (fit/test/onnx) on the M6-authored configs")
    print("=" * 96)
    norm_dict = _conv_norm_dict(outdir)

    configs: list[dict[str, Any]] = [dict(e) for e in _CONV_M6_CONFIGS]
    if corruption is not None:
        configs = corruption(configs)

    checks: dict[str, bool] = {}
    # -- every listed config file exists on disk ------------------------------
    missing_files = sorted(
        e["name"] for e in configs if not all((CONFIG_DIR / c).is_file() for c in e["cfg"])
    )
    checks["all_m6_conv_config_files_present"] = missing_files == []

    # -- per-config validation across the applicable modes --------------------
    results: list[dict[str, Any]] = []
    for entry in configs:
        cargs: list[str] = []
        for cfg_name in entry["cfg"]:
            cargs += ["-c", str(CONFIG_DIR / cfg_name)]
        set_args = _conv_set_args(entry, norm_dict)
        row: dict[str, Any] = {
            "name": entry["name"],
            "family": entry["family"],
            "cfg_stack": list(entry["cfg"]),
            "note": entry["note"],
            "validateFit": False,
            "validateTest": False,
            "validateOnnx": entry["onnx"] if entry["onnx"] == "na" else False,
            "rc": {},
        }
        # a missing config file cannot be validated — leave the mode rows False
        # (the file-present check already fails the gate) and skip the calls
        files_present = all((CONFIG_DIR / c).is_file() for c in entry["cfg"])
        if files_present:
            for cli_mode, report_key in _conv_modes(entry):
                rc = salt2_main(["graph", "validate", "--mode", cli_mode, *cargs, *set_args])
                row["rc"][cli_mode] = rc
                ok = rc == 0
                row[report_key] = ok
                checks[f"{entry['name']}:{cli_mode}"] = ok
        else:
            for cli_mode, _report_key in _conv_modes(entry):
                checks[f"{entry['name']}:{cli_mode}"] = False
        results.append(row)

    passed = all(checks.values())
    n_total = len(configs)
    n_validated = sum(
        1
        for r in results
        if r["validateFit"] and r["validateTest"] and r["validateOnnx"] in {True, "na"}
    )

    criterion = (
        "the M7-slice acceptance (plan 12 M6-CONV row): EVERY M6-authored config landed so far "
        f"(_CONV_M6_CONFIGS, {n_total} configs this wave, names embedded in the report) exists as "
        "a v2-native fixture in salt/core/configs/ that the REAL salt2 graph validate (the "
        "canonical static validator, the d2cfg/M5-CONV command path) convert+validates+"
        "plan-compiles in fit + test + onnx with rc == 0; the list is complete and exits non-zero "
        "if any config is missing or fails. No --strict (the no-schema + flow/truth_hadrons "
        "preflight warnings are inherent to data-free validation — see the docstring); no "
        "forward-parity claim (owned by LB1)."
    )
    report = _base_report(
        "m6_conv_acceptance",
        passed,
        criterion,
        {
            "norm_dict": str(norm_dict),
            "strict": False,
            "total_configs": n_total,
            "validated_configs": n_validated,
            "missing_config_files": missing_files,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["configs"] = results
    report["conv_m6_configs"] = [e["name"] for e in _CONV_M6_CONFIGS]
    report["scope_note"] = (
        "M6-CONV is the consolidated convert+validate+plan-compile acceptance for the M6-authored "
        "v2-native configs (plan 12 M6-CONV row): static validation ONLY (salt2 graph validate "
        "fit/test/onnx). It makes NO forward-parity claim — the Labeller label-derivation parity "
        "is owned by LB1. This wave (sub-wave A, Labeller) lands GN3X + GN2X_qcdsplit; later M6 "
        "waves EXTEND _CONV_M6_CONFIGS with GN2_muP (B), GN2XE (C) and legacy/DL1 (D). Both M6-A "
        "configs are standard traces (no muP MuReadout fold or edge dynamic-T register-pad export "
        "hazard), so onnx stays in the gate for both. The 2 configs move 🔷->✅ in the 39 "
        "denominator."
    )
    report["no_strict_rationale"] = (
        "no --strict: a data-free validation cannot satisfy it. --strict promotes EVERY warning to "
        "an error (cli.py:919-920) and several warnings are inherent to data-free validation, "
        "orthogonal to convertibility: (1) the no-schema warning (every config lacking a schema: "
        "artifact); (2) warning-level deadcode / per-mode unconsumed-pred infos; (3) Normaliser "
        "preflight warnings for the GN3X 'flow' / GN2X_qcdsplit 'flow'+'truth_hadrons' streams the "
        "parity norm dict does not supply (it covers jets/tracks/electrons only). The parity norm "
        "dict resolves the jets/tracks preflight so bind succeeds; the flow/truth_hadrons warnings "
        "are data-shape artifacts, not graph errors. Same rationale as M5-CONV / d2cfg / every "
        "config header. rc == 0 (no ERROR-level finding, design §4.2) IS the "
        "convert+validate+plan-compile criterion."
    )

    # -- stdout table ----------------------------------------------------------
    print(f"{'config':<26}{'family':<12}{'fit':>5}{'test':>6}{'onnx':>7}   stack / note")
    for r in results:
        onnx_disp = "na" if r["validateOnnx"] == "na" else ("PASS" if r["validateOnnx"] else "FAIL")
        print(
            f"{r['name']:<26}{r['family']:<12}"
            f"{'PASS' if r['validateFit'] else 'FAIL':>5}"
            f"{'PASS' if r['validateTest'] else 'FAIL':>6}"
            f"{onnx_disp:>7}   {'+'.join(r['cfg_stack'])}"
        )
    if missing_files:
        print(f"\nMISSING config files (no salt/core/configs/ entry): {missing_files}")
    print(f"\n{n_validated}/{n_total} configs validated all applicable modes")
    _print_verdict("conv", passed, criterion, _emit_report(report, outdir, "conv"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the gate subcommand parser (LB1, M6-CONV; later M6 waves append theirs).

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``lb1`` and ``conv`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.gates_m6", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)
    helps = {
        "lb1": "Labeller label-derivation parity vs v1 + named-error guards (sub-wave A)",
        "conv": "M6-CONV: salt2 graph validate (fit/test/onnx) on the M6-authored configs",
    }
    for gate, help_text in helps.items():
        p = sub.add_parser(gate, help=help_text)
        p.add_argument("--outdir", type=Path, required=True, help="report output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an M6 gate from the command line.

    Returns
    -------
    int
        0 if the gate passed, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    runner = {
        "lb1": run_lb1,
        "conv": run_conv,
    }[args.gate]
    code, _ = runner(args.outdir)
    return code


if __name__ == "__main__":
    sys.exit(main())
