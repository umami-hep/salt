"""M6 gates harness — LB1 + VS1 + M6-CONV (sub-waves A/D) (plan 12; design §9.5, FD 1303-1304).

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

- **VS1 vector-stream forward parity (v1-decidable)** — the model-side rank-2
  ``[B, F]`` vector-stream embed (M6-6 / plan 12 sub-wave D): a `StreamEmbed`
  with ``vector: true`` projects ``normed.<s> [B, F]`` -> ``embed.<s> [B, D]``
  (no token axis), consumed DIRECTLY by a ``sequence: false`` task head with NO
  encoder and NO pooling — the v2 reproduction of v1's jets-only DL1 MLP
  (``legacy/DL1.yaml`` single init_net ``attach_global: false`` -> the
  no-encoder/no-pool `SaltModel` path ``saltmodel.py:90-92`` guard, ``:155-156``
  ``embed_xs = flatten(xs)``, ``:170-172`` ``global_rep = embed_xs``). VS1 runs
  the DL1-shaped v2 plan through the real compiler/bind/executor and asserts the
  raw logits are BITWISE identical to an INDEPENDENT v1 reference (a separately
  constructed v1 ``InitNet(attach_global=False)`` + v1 ``ClassificationTask``,
  weight-loaded from the v2 modules, run through v1's no-encoder/no-pool forward
  math). PLUS an ONNX leg: the rank-2 embed is a plain ``nn.Linear``-stack with
  no T axis, so the traced graph carries B as the SOLE dynamic axis and NO token
  (``n_*``) dynamic axis anywhere (mirrors `Normaliser`'s ``[B, F]``
  global-object handling — no special-casing), and onnxruntime agrees with the
  eager adapter. Negative control: perturbing the v2 embed weights breaks the
  bitwise parity while the structural (no-T / rank-2) checks stay green.

- **M6-CONV — the M7-slice acceptance (this wave's slice)** — the
  newly-authored v2-native M6 configs (sub-wave A: GN3X, GN2X_qcdsplit; sub-wave
  D: DL1; later M6 waves EXTEND ``_CONV_M6_CONFIGS``) exist as v2-native fixtures in
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
import torch
from ftag import Labeller as V1Labeller

from salt.core.data import H5StructuredReader  # noqa: F401 (parity with gates_m5 import surface)
from salt.core.data.base import WorkerCtx
from salt.core.data.processors import Labels
from salt.core.graph import Bundle, Executor, Mode, compile_plan
from salt.core.graph.errors import ConfigError
from salt.core.graph.planner import PlanStep
from salt.core.graph.spec import TensorSpec, flatten_spec, unflatten_spec
from salt.core.main import CONFIG_DIR
from salt.core.main import main as salt2_main
from salt.core.nn import (
    LossSum,
    Normaliser,
    StreamEmbed,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.tasks import ClassificationTaskModule
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    ExportOutput,
    OnnxAdapter,
    attach_manifest,
    check_onnx,
    export_graph,
    make_session,
    resolve_export_config,
)
from salt.models.initnet import InitNet as V1InitNet
from salt.models.task import ClassificationTask as V1ClassificationTask
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
# VS1 — vector-stream forward parity vs v1 (sub-wave D, plan 12 M6-6)
# ---------------------------------------------------------------------------

# DL1's two jet input variables (legacy/DL1.yaml:35-36 pt/eta; input_size:2).
# The v2 source declares inputs.jets as the rank-2 [B, F] vector boundary the
# reader's `vector:` flag produces (reader.py GroupConfig -> Features rank-2).
_VS1_JET_VARIABLES: tuple[str, ...] = ("pt_btagJes", "eta_btagJes")
# DL1's 3-class jet flavour head (legacy/DL1.yaml:28 output_size:3).
_VS1_CLASS_NAMES: tuple[str, ...] = ("bjets", "cjets", "ujets")
_VS1_EMBED_DIM = 64  # legacy/DL1.yaml:13 init_net output_size
_VS1_HIDDEN = [128, 128, 128]  # legacy/DL1.yaml:12 init_net hidden_layers
_VS1_HEAD_HIDDEN = [128]  # legacy/DL1.yaml:29 task hidden_layers
_VS1_B = 8  # batch size for the parity fixture


def _vs1_norm_dict(outdir: Path) -> Path:
    """Write a jets-only identity-ish norm dict for the DL1 vector stream.

    VS1's parity is about the embed/head wiring, not normalisation — the norm
    dict supplies the two jet variables so `Normaliser.materialise` resolves
    (distinct per-variable constants, so a field-order bug can't pass silently).

    Returns
    -------
    Path
        The norm-dict YAML path.
    """
    import yaml  # noqa: PLC0415 - keep the module import surface lean

    outdir.mkdir(parents=True, exist_ok=True)
    nd = {
        "jets": {
            "pt_btagJes": {"mean": 1.5, "std": 2.0},
            "eta_btagJes": {"mean": -0.25, "std": 1.25},
        }
    }
    path = outdir / "vs1_norm_dict.yaml"
    path.write_text(yaml.safe_dump(nd))
    return path


def _vs1_modules(norm_dict: Path) -> dict[str, Any]:
    """Build the DL1-shaped v2 module dict: rank-2 embed -> head, no encoder/pool.

    Mirrors the v1 jets-only MLP (legacy/DL1.yaml): a single ``vector: true``
    `StreamEmbed` on the global ``jets`` stream feeds a ``sequence: false``
    `ClassificationTaskModule` reading ``embed.jets`` DIRECTLY — no `Concat`,
    no `TransformerEncoder`, no `Split`, no pooling (v1 saltmodel.py:90-92
    guard / :155-156 / :170-172).

    Returns
    -------
    dict[str, Any]
        Instance-named modules with `LossSum` already narrowed.
    """
    modules: dict[str, Any] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["jets"], global_object="jets"),
        "jets_embed": StreamEmbed(
            stream="jets",
            out_dim=_VS1_EMBED_DIM,
            dense={"hidden_layers": list(_VS1_HIDDEN), "activation": "Mish"},
            vector=True,
        ),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=list(_VS1_CLASS_NAMES),
            input="embed.jets",  # consume the rank-2 embed directly (no pool)
            sequence=False,
            dense={"hidden_layers": list(_VS1_HEAD_HIDDEN), "activation": "Mish"},
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    modules["loss"].narrow(LossSum.collect_loss_keys(modules))
    return modules


def _vs1_sources():
    """The DL1 dataset boundary: a rank-2 vector ``inputs.jets`` + its label.

    ``inputs.jets`` is ``[B, F]`` (the reader ``vector:`` boundary, no token
    axis); the flavour label is ``[B]`` and TRAINING-gated.

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    return unflatten_spec({
        "inputs.jets": TensorSpec(
            shape=("B", len(_VS1_JET_VARIABLES)),
            dtype="float32",
            fields=tuple(_VS1_JET_VARIABLES),
        ),
        "labels.jets.flavour_label": TensorSpec(
            shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
        ),
    })


def _vs1_v1_reference(modules: dict[str, Any], jets: torch.Tensor) -> torch.Tensor:
    """Run the INDEPENDENT v1 no-encoder/no-pool DL1 path on ``normed jets``.

    Constructs a SEPARATE v1 ``InitNet(attach_global=False)`` + v1
    ``ClassificationTask`` (NOT the v2 modules' composed instances) and loads
    them from the v2 modules' weights, then runs v1's exact forward math:

    - ``embed_xs = init_net({"jets": x})`` (InitNet.forward: x = inputs[name];
      attach_global=False so NO context; x = net(x), initnet.py:72-89);
    - ``global_rep = embed_xs`` (no pool_net, saltmodel.py:170-172);
    - ``logits = task.net(global_rep)`` (ClassificationTask raw logits).

    `x` is the SAME ``normed.jets`` the v2 `Normaliser` produced, so the only
    thing under test is the embed/head wiring, not normalisation.

    Returns
    -------
    torch.Tensor
        The ``[B, n_classes]`` raw logits from the v1 reference path.
    """
    embed = modules["jets_embed"]
    head = modules["jets_classification"]
    # an INDEPENDENT v1 InitNet (attach_global=False -> jets-only, no context),
    # weight-loaded from the v2 StreamEmbed's composed Dense
    init_net = V1InitNet(
        input_name="jets",
        dense_config={
            "input_size": len(_VS1_JET_VARIABLES),
            "output_size": _VS1_EMBED_DIM,
            "hidden_layers": list(_VS1_HIDDEN),
            "activation": "Mish",
        },
        variables={"jets": list(_VS1_JET_VARIABLES)},
        global_object="jets",
        attach_global=False,
    )
    init_net.net.load_state_dict(embed.net.state_dict())
    # an INDEPENDENT v1 ClassificationTask, weight-loaded from the v2 head
    ref_task = V1ClassificationTask(
        name="jets_classification",
        input_name="jets",
        label="flavour_label",
        class_names=list(_VS1_CLASS_NAMES),
        loss=torch.nn.CrossEntropyLoss(),
        dense_config={
            "input_size": _VS1_EMBED_DIM,
            "output_size": len(_VS1_CLASS_NAMES),
            "hidden_layers": list(_VS1_HEAD_HIDDEN),
            "activation": "Mish",
        },
    )
    ref_task.net.load_state_dict(head.task.net.state_dict())
    init_net.eval()
    ref_task.eval()
    with torch.no_grad():
        embed_xs = init_net({"jets": jets})  # [B, D] — no token axis, no context
        global_rep = embed_xs  # no pool_net (saltmodel.py:170-172)
        logits, _ = ref_task(global_rep, labels_dict=None, pad_masks=None, context=None)
    return logits


def run_vs1(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """VS1: rank-2 ``[B, F]`` vector-stream embed -> task head, parity vs v1 DL1.

    Drives the DL1-shaped v2 plan (rank-2 ``vector: true`` `StreamEmbed` ->
    ``sequence: false`` `ClassificationTaskModule`, NO encoder/pool — `_vs1_modules`)
    through the REAL compiler / two-phase bind / executor and asserts:

    - **bitwise forward parity** — the v2 raw FIT logits (``preds.jets.
      jets_classification``) are BITWISE identical to an INDEPENDENT v1
      no-encoder/no-pool reference (`_vs1_v1_reference`: a separately built v1
      ``InitNet(attach_global=False)`` + ``ClassificationTask`` weight-loaded
      from the v2 modules, run through v1's ``saltmodel.py:90-92,155-172``
      forward math). A plain ``nn.Linear``-stack on ``[B, F]`` reorders no
      floats, so the claim is BITWISE;
    - **rank-2 structural shape** — the embed declares/produces ``[B, F]`` ->
      ``[B, D]`` (no token axis) and the head's prediction is ``[B, n_classes]``;
    - **no-encoder/no-pool plan** — the compiled plan contains no encoder, no
      pool, and no ``seq.*`` / ``encoded.*`` / ``pooled.*`` edges;
    - **ONNX no-T leg** — the model exports under a real ``Mode.ONNX`` trace
      with B as the SOLE dynamic axis and NO token (``n_*``) dynamic axis (the
      rank-2 embed has no T axis — mirrors `Normaliser`'s ``[B, F]`` global
      handling, no special-casing); the loaded graph's single input is rank-2
      and onnxruntime agrees with the eager adapter within 1e-6. The no-T claim
      is made NON-VACUOUS in-gate by a negative control: a PROBE adapter over
      the SAME ONNX plan but with ``inputs.jets`` flagged ``sequence: true``
      DOES register an ``n_<stream>`` axis (proving the same machinery would
      emit a token axis for a sequence input, so the DL1 graph's absence of one
      is by-design, not a vacuously-passing iteration).

    Negative control (``test_gates_m6.py``): the ``corruption`` hook perturbs
    the v2 logits — the bitwise parity check must FAIL while the structural
    (rank-2 / no-T / no-encoder) checks stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("VS1 rank-2 [B,F] vector-stream embed -> head BITWISE parity vs v1 DL1 no-enc/no-pool")
    print("=" * 96)
    checks: dict[str, bool] = {}

    norm_dict = _vs1_norm_dict(outdir)
    modules = _vs1_modules(norm_dict)

    # -- (a) the embed declares/produces rank-2 [B, F] -> [B, D] (no T axis) ---
    embed_io = modules["jets_embed"].declare_io(Mode.FIT)
    req = flatten_spec(embed_io.requires)
    prod = flatten_spec(embed_io.produces)
    checks["embed_input_is_rank2"] = len(req["normed.jets"].shape) == 2
    checks["embed_output_is_rank2"] = len(prod["embed.jets"].shape) == 2
    checks["embed_out_dim_matches"] = prod["embed.jets"].shape[-1] == _VS1_EMBED_DIM

    # -- (b) compile the DL1 plan: no encoder, no pool, no seq/encoded edges --
    fit_plan = compile_plan(modules, Mode.FIT, sources=_vs1_sources(), sinks=["loss.total"])
    test_plan = compile_plan(
        modules,
        Mode.TEST,
        sources=_vs1_sources(),
        sinks=["preds.jets.jets_classification"],
    )
    names = set(fit_plan.module_names)
    checks["no_encoder_in_plan"] = "encoder" not in names
    checks["no_pool_in_plan"] = "pool" not in names
    checks["no_seq_or_pooled_edges"] = all(
        not (
            edge.key.startswith("seq.")
            or edge.key.startswith("encoded.")
            or edge.key.startswith("pooled.")
        )
        for plan in (fit_plan, test_plan)
        for edge in plan.edges
    )
    # the head consumes the embed DIRECTLY (the embed.jets -> head edge exists)
    checks["head_consumes_embed_directly"] = any(
        edge.key == "embed.jets" for edge in fit_plan.edges
    )

    # -- (c) bitwise forward parity vs the INDEPENDENT v1 reference -----------
    bind_all(modules, resolve_bind_schema([fit_plan, test_plan]))
    materialise_all(modules)
    gen = torch.Generator().manual_seed(17)
    jets = torch.randn(_VS1_B, len(_VS1_JET_VARIABLES), generator=gen)
    labels = torch.randint(0, len(_VS1_CLASS_NAMES), (_VS1_B,), generator=gen)
    b = Bundle()
    b.set("inputs.jets", jets)
    b.set("labels.jets.flavour_label", labels)
    out = Executor(fit_plan).run(b, debug=True)
    v2_logits = out.get("preds.jets.jets_classification")
    checks["pred_is_rank2_b_nclasses"] = tuple(v2_logits.shape) == (
        _VS1_B,
        len(_VS1_CLASS_NAMES),
    )
    if corruption is not None:
        v2_logits = corruption(v2_logits)
    # the v1 reference runs on the SAME normed.jets the v2 Normaliser produced
    normed_jets = out.get("normed.jets")
    v1_logits = _vs1_v1_reference(modules, normed_jets)
    checks["forward_bitwise_vs_v1"] = v2_logits.shape == v1_logits.shape and torch.equal(
        v2_logits, v1_logits
    )

    # -- (d) ONNX no-T leg: B the sole dynamic axis, no n_* token axis --------
    onnx_path = outdir / "vs1_dl1.onnx"
    head = modules["jets_classification"]
    export = ExportConfig(
        model_name="DL1v2",
        inputs=[ExportInput(port="inputs.jets", name="jet_features")],  # global, NOT sequence
    )
    manifest = [
        ExportOutput(port="preds.jets.jets_classification", names=list(head.class_suffixes))
    ]
    result = export_graph(
        modules,
        export,
        {"jets": list(_VS1_JET_VARIABLES)},
        onnx_path,
        outputs=manifest,
        run_name="DL1v2",
    )
    # the adapter declares NO token (n_*) dynamic axis at all (no sequence input,
    # no per-token output) — only B is the moving axis (Normaliser [B,F] parity).
    # A sequence stream would register {0: 'n_<stream>'} (adapter.dynamic_axes);
    # a rank-2 DL1 graph registers none.
    dyn_axes = result.adapter.dynamic_axes
    checks["no_token_dynamic_axis"] = not any(
        str(name).startswith("n_") for axmap in dyn_axes.values() for name in axmap.values()
    )
    # negative control — prove the no-T claim is NON-VACUOUS within the gate
    # (not merely "the empty dynamic_axes mapping iterated zero items"). Build a
    # PROBE adapter over the SAME compiled ONNX plan (result.adapter.plan) but
    # with inputs.jets flagged `sequence: true`, run through the REAL
    # resolve_export_config + attach_manifest + OnnxAdapter machinery (the
    # export.py:267-272 path): it MUST register the default n_<stream> axis.
    # This demonstrates the same adapter code that produced {} for DL1 WOULD
    # emit an n_* axis for a sequence input, so the DL1 graph's absence of one
    # is BY-DESIGN, not a vacuously-passing iteration (adapter.py:201-205,
    # config.py:713 dyn_axis=(entry.dyn_axis or f"n_{stream}")).
    seq_probe_cfg = attach_manifest(
        resolve_export_config(
            ExportConfig(
                model_name="DL1v2SeqProbe",
                inputs=[ExportInput(port="inputs.jets", name="jet_features", sequence=True)],
            ),
            run_name="DL1v2SeqProbe",
        ),
        manifest,
    )
    probe = OnnxAdapter(result.adapter.plan, seq_probe_cfg, {"jets": tuple(_VS1_JET_VARIABLES)})
    probe_axes = probe.dynamic_axes
    checks["no_token_axis_check_is_non_vacuous"] = any(
        str(name).startswith("n_") for axmap in probe_axes.values() for name in axmap.values()
    )
    # inspect the loaded ONNX graph: the single input is rank-2 (no T dim)
    import onnx  # noqa: PLC0415 - heavy import, onnx-leg only

    graph = onnx.load(str(onnx_path)).graph
    in_dims = graph.input[0].type.tensor_type.shape.dim
    checks["onnx_input_is_rank2"] = len(in_dims) == 2
    # B is the SOLE dynamic axis: re-trace with the batch axis marked dynamic
    # and confirm onnxruntime accepts batch > 1 (the [B, F] embed is batch-poly).
    dyn_onnx = outdir / "vs1_dl1_dynB.onnx"
    torch.onnx.export(
        result.adapter,
        result.adapter.example_inputs(),
        str(dyn_onnx),
        opset_version=20,
        input_names=result.adapter.input_names,
        output_names=result.adapter.output_names,
        dynamic_axes={"jet_features": {0: "batch"}},
        dynamo=False,
    )
    session = make_session(dyn_onnx)
    multi = torch.rand(5, len(_VS1_JET_VARIABLES))
    ort_out = session.run(None, {"jet_features": multi.numpy()})
    with torch.no_grad():
        eager_out = result.adapter(multi)
    checks["onnx_batch_axis_is_dynamic"] = ort_out[0].shape[0] == 5
    checks["onnx_runtime_matches_eager_multibatch"] = all(
        np.max(np.abs(np.asarray(o) - e.numpy())) <= 1e-6
        for o, e in zip(ort_out, eager_out, strict=True)
    )
    # torch-vs-onnxruntime sweep on the static (B=1) graph (the O1 pattern)
    sweep = check_onnx(result.adapter, result.onnx_path, trials=2, float_rtol=1e-6, float_atol=1e-6)
    checks["onnx_check_passes"] = sweep.passed

    passed = all(checks.values())
    criterion = (
        "the model-side rank-2 [B, F] vector-stream embed (M6-6 / plan 12 VS1): a vector: true "
        "StreamEmbed projects normed.jets [B, F] -> embed.jets [B, D] (no token axis) consumed "
        "DIRECTLY by a sequence: false ClassificationTaskModule with NO encoder and NO pooling; "
        "the v2 FIT logits are BITWISE identical to an INDEPENDENT v1 no-encoder/no-pool reference "
        "(separately built InitNet(attach_global:false) + ClassificationTask weight-loaded from "
        "the v2 modules, run through v1 saltmodel.py:90-92,155-172 math); the compiled plan has no "
        "encoder/pool/seq/encoded/pooled edges; and the ONNX trace carries B as the SOLE dynamic "
        "axis with NO token (n_*) axis (Normaliser [B, F] parity), onnxruntime agreeing with the "
        "eager adapter (<=1e-6) at batch 1 and batch 5; the no-T claim is non-vacuous — a "
        "sequence:true PROBE adapter over the SAME plan DOES register an n_<stream> axis"
    )
    report = _base_report(
        "vs1_vector_stream_parity",
        passed,
        criterion,
        {
            "jet_variables": list(_VS1_JET_VARIABLES),
            "class_names": list(_VS1_CLASS_NAMES),
            "embed_dim": _VS1_EMBED_DIM,
            "batch": _VS1_B,
            "norm_dict": str(norm_dict),
            "onnx_path": str(onnx_path),
            "approach": "StreamEmbed rank-2 vector path (mirrors Normaliser global_object)",
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["dynamic_axes"] = {k: {str(a): n for a, n in v.items()} for k, v in dyn_axes.items()}
    report["v1_reference"] = (
        "INDEPENDENT v1 reference (instance independence, NOT a math re-implementation): "
        "separately-instantiated v1 modules — a fresh InitNet(attach_global=False) + "
        "ClassificationTask, distinct objects from the v2 modules' composed instances, "
        "weight-loaded from them — run through v1's no-encoder/no-pool forward "
        "(saltmodel.py:90-92 guard, :155-156 embed_xs, :170-172 global_rep=embed_xs). The "
        "bitwise torch.equal therefore tests that the v2 GRAPH PATH (compile -> two-phase "
        "bind -> executor -> sequence:false head reading embed.jets directly, no encoder/"
        "pool) routes tensors through the SAME Dense math as v1's hand-wired saltmodel.py "
        "no-pool path; it does not (and does not claim to) independently re-derive the Dense "
        "arithmetic"
    )
    _print_checks(checks)
    _print_verdict("vs1", passed, criterion, _emit_report(report, outdir, "vs1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# M6-CONV — the M7-slice acceptance for the M6-authored configs (sub-wave A
# bootstraps it; later M6 waves EXTEND _CONV_M6_CONFIGS)
# ---------------------------------------------------------------------------

# The AUTHORITATIVE M6-CONV config list (plan 12 M6-CONV row: "embed the
# authoritative _CONV_M6_CONFIGS list verbatim"). Sub-wave A (Labeller) landed
# the FIRST two; sub-wave D (vector-stream) adds DL1 (now 3); later waves append
# GN2_muP (B) and GN2XE (C) until all 5 config-gating needs-M6 configs are here.
#
# `norm_global` is False for all (none carries a SECOND `norm_global` Normaliser
# — that is the VectorConcat/global-stream pattern, absent here). `onnx ==
# "validate"` for all three: GN3X, GN2X_qcdsplit AND DL1 are STANDARD traces — no
# muP MuReadout fold (B) or edge dynamic-T register pad (C) export hazard — so
# onnx stays in the gate (plan 12 M6-CONV per-config export contracts: "GN3X/
# GN2X_qcdsplit/DL1 are standard traces"; DL1's rank-2 [B, F] embed is a plain
# nn.Linear stack with B the sole dynamic axis, sub-wave D / VS1).
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
    {
        "name": "DL1",
        "cfg": ("DL1.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "vector-stream",
        "note": (
            "jets-only MLP: rank-2 [B, F] vector-stream embed (vector: true) -> sequence: false "
            "head, NO encoder/pool (the M6-6 deliverable, gate VS1); 3-class CE; LossSum"
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

    For EVERY M6 config landed so far (``_CONV_M6_CONFIGS`` — sub-wave A: GN3X,
    GN2X_qcdsplit; sub-wave D: DL1; later waves EXTEND the list), this drives the canonical static
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
        "is owned by LB1, the vector-stream forward parity by VS1. Sub-wave A (Labeller) landed "
        "GN3X + GN2X_qcdsplit; sub-wave D (vector-stream) adds DL1 (now 3); later M6 waves EXTEND "
        "_CONV_M6_CONFIGS with GN2_muP (B) and GN2XE (C). All three current configs are standard "
        "traces (no muP MuReadout fold or edge dynamic-T register-pad export hazard; DL1's rank-2 "
        "[B, F] embed is a plain nn.Linear stack with B the sole dynamic axis), so onnx stays in "
        "the gate for all. The 3 configs move 🔷->✅ in the 39 denominator."
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
        "vs1": "Vector-stream rank-2 [B,F] embed -> head parity vs v1 DL1 + onnx no-T (sub-wave D)",
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
        "vs1": run_vs1,
        "conv": run_conv,
    }[args.gate]
    code, _ = runner(args.outdir)
    return code


if __name__ == "__main__":
    sys.exit(main())
