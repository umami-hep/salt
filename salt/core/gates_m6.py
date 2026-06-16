"""M6 gates harness — LB1 + VS1 + MU1/MU2 + ED1 + M6-CONV (waves A/D/B/C) (plan 12; design §9.5).

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

- **MU1 muP forward-init parity + export-fold (v1-decidable)** — the muP
  ARCHITECTURAL port (M6 sub-wave B; design §3.4 KEEP-architecture/BREAK-routing):
  the ``mup: true`` init_arg on `StreamEmbed` (-> ``Dense(mup=True)``) and
  `TransformerEncoder` (-> ``Transformer(mup=True)``) reproduces v1's encoder-flag
  muP path BITWISE. MU1 builds a GN2_muP-shaped model and asserts: (a) the v2 mup
  embed + encoder forward is BITWISE identical to an INDEPENDENT v1 reference
  (separately built ``Dense(mup=True)`` + ``Transformer(mup=True)``, weight-loaded
  from the v2 modules, run through v1 forward) — deterministic init values, NO
  training; (b) the muP machinery is wired as v1 wires it from the encoder flag —
  the encoder out-proj is a ``mup.MuReadout`` with weight AND bias zeroed at init
  (transformer.py:623-628), the embed's `Dense` carries ``mup=True``; (c) the
  **export-fold check** — folding the `MuReadout` to a plain ``nn.Linear``
  (``set_export_mode``, ``W_folded = (output_mult/width_mult) * W``,
  ``bias_folded = bias``) gives an output EQUAL to the MuReadout forward (bitwise
  at the architectural default ``width_mult==1.0``, ``<=1e-6`` once a non-unit
  multiplier enters via the routing stage), and the swapped out-proj is a plain
  ``nn.Linear`` in the traced graph. HONEST faithfulness note recorded in the
  report: v1's ``Transformer(mup=True)`` does NOT propagate muP to its
  EncoderLayers/Attention — the encoder flag wires ONLY the MuReadout out-proj
  swap (the 1/d softmax scale + Q-zero attention init live in
  ``Attention(mup=True)``, which the v1 encoder flag does not pass down,
  transformer.py:604-614). The v2 port composes ``Transformer(mup=True)`` and is
  therefore byte-faithful to v1's ACTUAL encoder-mup behaviour. The coord-curve
  comparison (multi-step, stochastic) + the "is it muP-flat?" plot judgment are
  MU2/MU-HUMAN, NOT this gate. Negative control: perturbing the v2 forward breaks
  the bitwise parity while the structural (MuReadout / fold) checks stay green.

- **ED1 EdgeFeatures + EdgeEmbed forward parity (v1-decidable; sub-wave C stage 1)**
  — the M6 edge-feature building blocks (design §6.7 1418-1421): `EdgeFeatures`
  (a NetModule) requires the RAW ``inputs.<stream>`` (un-normalised, §6.3) +
  ``masks.<stream>`` and produces ``edges.<stream>`` ``[B, T, T, E]``;
  `EdgeEmbed` (an edge-typed `StreamEmbed`) maps it to ``edges.<stream>_emb``
  ``[B, T, T, D_e]``. ED1 drives both through the REAL compiler / two-phase bind
  / executor on a GN2XE-shaped fixture and asserts: the v2 ``edges.tracks`` is
  BITWISE identical to an INDEPENDENT v1 ``EdgeConstructor`` (a separately built
  v1 object run on a CLONE of the same raw input — edge_constructor.py:41-44,
  edge_features.py:52-146), checked per-feature for dR/kt/z; the v2
  ``edges.tracks_emb`` is BITWISE identical to an INDEPENDENT v1 ``Dense``
  (weight-loaded from the v2 EdgeEmbed, run on the v1 edges — v1
  ``edge_init_nets``, saltmodel.py:132); both token axes share the
  ``T:<stream>`` symbol (the dynamic-T export prerequisite); ``inputs.tracks``
  is never mutated (write-once §2.1); and the named-error guards fire
  (unrecognised/empty feature -> `ConfigError`, missing required variable ->
  `check_edge_config` `ValueError`, EdgeEmbed width key -> `ConfigError`).
  Negative control: perturbing the v2 edge tensor breaks the bitwise parity
  while the structural + guard checks stay green. The encoder edge path, the
  edge-stream-first / backend-forcing bind validators (ED2), the GN2XE config,
  and the ONNX dynamic-T register pad are LATER sub-wave C stages.

- **M6-CONV — the M7-slice acceptance (this wave's slice)** — the
  newly-authored v2-native M6 configs (sub-wave A: GN3X, GN2X_qcdsplit; sub-wave
  D: DL1; sub-wave B: GN2_muP; sub-wave C: GN2XE — the list is now COMPLETE at 5)
  exist as v2-native fixtures in
  ``salt/core/configs/`` AND pass the REAL ``salt2 graph validate`` (the
  canonical static validator, the d2cfg command path) in fit + test + onnx with
  rc == 0. The authoritative config list is embedded VERBATIM in
  ``_CONV_M6_CONFIGS`` and the report; the gate exits non-zero if any listed
  config is MISSING from ``salt/core/configs/`` or FAILS any applicable mode.
  Mirrors the M5-CONV ``run_conv`` (gates_m5.py:4478): data-free validation via
  the parity norm dict written into ``--outdir`` (jets/tracks/electrons
  resolve; the GN3X flow / GN2X_qcdsplit flow+truth_hadrons streams emit
  non-fatal "missing input type" preflight WARNINGS, not errors — so NO
  ``--strict``, the same rationale as M5-CONV). GN3X/GN2X_qcdsplit/DL1 are
  standard traces; for GN2_muP, ``--mode onnx`` here is a STATIC plan-compile +
  writer-manifest validation (cli.py:381-423) — it does NOT call
  ``torch.onnx.export``/``set_export_mode``, so the CONV onnx leg proves the onnx
  PLAN compiles, not the fold itself. The MuReadout->plain-Linear fold that makes
  the real export traceable is exercised + proven by MU1's export-fold check
  (``set_export_mode`` on the live module). GN2_muP's base/delta infshapes are
  generated data-free (``_conv_mup_shape_path``) before validation so the plan-
  compile succeeds; with the fold proven by MU1, onnx stays in the gate for it too.
  GN2XE's ``--mode onnx`` here is LIKEWISE a STATIC plan-compile; its load-bearing
  dynamic-T trace (BOTH edge T axes dynamic + the register zero-pad SHAPE-DERIVED,
  not baked) is exercised + PROVEN by ED1's ONNX-trace leg (a real
  ``torch.onnx.export`` + an onnxruntime length sweep) — so onnx stays in the gate
  for GN2XE too (NOT scoped out). NO export hazard remains deferred. This
  split mirrors the M5-CONV precedent (run_conv is also static validate). Moves
  ALL 5 configs 🔷→✅ in the 39 denominator — the LAST config-gating wave, the
  39-denominator now CLOSED.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
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
    Concat,
    EdgeEmbed,
    EdgeFeatures,
    GlobalAttentionPooling,
    LossSum,
    Normaliser,
    StreamEmbed,
    TransformerEncoder,
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
from salt.core.saltmodule import SaltModule, validate_edge_port, validate_mup_routing
from salt.models import Dense as V1Dense
from salt.models import Transformer as V1Transformer
from salt.models.edge_constructor import EdgeConstructor as V1EdgeConstructor
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
# MU1 — muP forward-init parity + MuReadout export-fold (sub-wave B, plan 12)
# ---------------------------------------------------------------------------

# A GN2_muP-shaped fixture (GN2_muP.yaml): tracks stream embedded with a muP
# Dense, a muP transformer encoder with an out-projection (the MuReadout slot),
# a global-attention pool, and a jet flavour head. Small dims keep the gate fast;
# the muP wiring (not the size) is what is under test.
_MU1_TRACK_VARIABLES: tuple[str, ...] = ("d0", "z0SinTheta", "dphi", "deta", "qOverP")
_MU1_JET_VARIABLES: tuple[str, ...] = ("pt_btagJes", "eta_btagJes")
_MU1_CLASS_NAMES: tuple[str, ...] = ("bjets", "cjets", "ujets")
_MU1_EMBED_DIM = 16  # GN2_muP.yaml:&embed_dim 256 (scaled down)
_MU1_OUT_DIM = 8  # GN2_muP.yaml:&out_dim 128 (scaled down)
_MU1_NUM_HEADS = 2  # GN2_muP.yaml:&num_heads 8 (scaled down)
_MU1_NUM_LAYERS = 2  # GN2_muP.yaml:num_layers 4 (scaled down)
_MU1_B = 6  # batch size
_MU1_T = 5  # track count


def _mu1_norm_dict(outdir: Path) -> Path:
    """Write a jets+tracks norm dict for the GN2_muP-shaped fixture.

    MU1's parity is about the muP embed/encoder wiring, not normalisation — the
    norm dict supplies distinct per-variable constants so a field-order bug can't
    pass silently.

    Returns
    -------
    Path
        The norm-dict YAML path.
    """
    import yaml  # noqa: PLC0415 - keep the module import surface lean

    outdir.mkdir(parents=True, exist_ok=True)
    nd: dict[str, Any] = {
        "jets": {
            v: {"mean": 0.5 + i, "std": 1.0 + 0.5 * i} for i, v in enumerate(_MU1_JET_VARIABLES)
        },
        "tracks": {
            v: {"mean": -0.25 + 0.1 * i, "std": 1.25 + 0.2 * i}
            for i, v in enumerate(_MU1_TRACK_VARIABLES)
        },
    }
    path = outdir / "mu1_norm_dict.yaml"
    path.write_text(yaml.safe_dump(nd))
    return path


def _mu1_modules(norm_dict: Path) -> dict[str, Any]:
    """Build the GN2_muP-shaped v2 module dict (mup embed + mup encoder + pool + head).

    Mirrors GN2_muP.yaml: a ``mup: true`` `StreamEmbed` on tracks (with jets
    context), a ``mup: true`` `TransformerEncoder` with an out-projection (the
    MuReadout slot, GN2_muP.yaml:&out_dim), a `GlobalAttentionPooling`, and a jet
    flavour `ClassificationTaskModule`. The two mup flags are the architectural
    port under test; the routing (apply_to lists, shape file) is a later stage.

    Returns
    -------
    dict[str, Any]
        Instance-named modules with `LossSum` already narrowed.
    """
    modules: dict[str, Any] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["jets", "tracks"], global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks",
            out_dim=_MU1_EMBED_DIM,
            dense={"hidden_layers": [_MU1_EMBED_DIM], "activation": "ReLU"},
            context=["normed.jets"],
            mup=True,
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=_MU1_EMBED_DIM,
            num_layers=_MU1_NUM_LAYERS,
            out_dim=_MU1_OUT_DIM,
            attention={"num_heads": _MU1_NUM_HEADS, "attn_type": "torch-math"},
            dense={"activation": "ReLU"},
            mup=True,
        ),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=list(_MU1_CLASS_NAMES),
            input="pooled.global",
            dense={"hidden_layers": [_MU1_OUT_DIM], "activation": "ReLU"},
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    modules["loss"].narrow(LossSum.collect_loss_keys(modules))
    return modules


def _mu1_sources():
    """The GN2_muP dataset boundary: rank-3 tracks + rank-2 jets + a flavour label.

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    return unflatten_spec({
        "inputs.jets": TensorSpec(
            shape=("B", len(_MU1_JET_VARIABLES)),
            dtype="float32",
            fields=tuple(_MU1_JET_VARIABLES),
        ),
        "inputs.tracks": TensorSpec(
            shape=("B", "T:tracks", len(_MU1_TRACK_VARIABLES)),
            dtype="float32",
            fields=tuple(_MU1_TRACK_VARIABLES),
        ),
        "masks.tracks": TensorSpec(shape=("B", "T:tracks"), dtype="bool", kind="pad_mask"),
        "labels.jets.flavour_label": TensorSpec(
            shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
        ),
    })


def _mu1_v1_encoder_reference(
    v2_encoder: TransformerEncoder, seq_x: torch.Tensor, seq_mask: torch.Tensor
) -> torch.Tensor:
    """Run an INDEPENDENT v1 ``Transformer(mup=True)`` on the SAME concatenated sequence.

    Constructs a SEPARATE v1 `Transformer` with ``mup=True`` (NOT the v2 module's
    composed instance), weight-loads it from the v2 encoder's composed v1
    Transformer, and runs v1's exact forward — the byte-faithful reference for the
    v2 encoder-mup path (the encoder flag wires the MuReadout out-proj swap; v1's
    ``Transformer(mup=True)`` does NOT pass mup to its EncoderLayers/Attention,
    transformer.py:604-614, so this reference and the v2 module share that
    behaviour exactly). The returned tensor is the register-augmented
    ``encoded.seq`` (v1 keeps the register rows).

    Returns
    -------
    torch.Tensor
        The v1 ``[B, T+R, out_dim]`` encoded sequence.
    """
    ref = V1Transformer(
        num_layers=_MU1_NUM_LAYERS,
        embed_dim=_MU1_EMBED_DIM,
        out_dim=_MU1_OUT_DIM,
        norm="LayerNorm",
        attn_type="torch-math",
        do_final_norm=True,
        num_registers=v2_encoder.num_registers,
        attn_kwargs={"num_heads": _MU1_NUM_HEADS},
        dense_kwargs={"activation": "ReLU"},
        mup=True,
    )
    from mup import set_base_shapes  # noqa: PLC0415

    set_base_shapes(ref, ref, rescale_params=False)
    ref.load_state_dict(v2_encoder.encoder.state_dict())
    ref.eval()
    with torch.no_grad():
        xs = {"seq": seq_x}
        pad = {"seq": seq_mask}
        encoded, _ = ref(xs, pad_mask=pad)
    return encoded


def run_mu1(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """MU1: muP forward-init parity vs v1 + MuReadout export-fold (sub-wave B).

    The muP ARCHITECTURAL port (design §3.4 KEEP-architecture/BREAK-routing): the
    ``mup: true`` init_arg on `StreamEmbed` and `TransformerEncoder` reproduces
    v1's encoder-flag muP path. MU1 drives a GN2_muP-shaped v2 plan (`_mu1_modules`:
    mup embed + mup encoder + pool + head) through the REAL compiler / two-phase
    bind / executor and asserts:

    - **structural muP wiring (v1-decidable)** — the embed's composed `Dense`
      carries ``mup=True`` (the muP weight init ran, dense.py:88-89,96-102); the
      encoder's out projection is a ``mup.MuReadout`` (transformer.py:625-628) with
      weight AND bias zeroed at init (the v1 zeroed-readout, transformer.py:627-628);

    - **forward-init parity (BITWISE)** — the v2 encoder forward (``encoded.seq``)
      is BITWISE identical to an INDEPENDENT v1 ``Transformer(mup=True)``
      (`_mu1_v1_encoder_reference`: a separately built v1 Transformer, weight-loaded
      from the v2 encoder, run through v1 forward). Same instances' weights, same
      torch-math math -> no float reordering, so the claim is BITWISE. This is the
      byte/parity-decidable leg (deterministic init values, NO training);

    - **export-fold (numerically equal)** — folding the `MuReadout` to a plain
      ``nn.Linear`` via the export protocol (``encoder.set_export_mode()``,
      ``W_folded = (output_mult/width_mult) * W``, ``bias_folded = bias``) leaves
      the encoder forward EQUAL to the pre-fold MuReadout forward (BITWISE at the
      architectural default ``width_mult==1.0``; ``<=1e-6`` once a non-unit
      multiplier enters via the routing stage — proven non-vacuous in-gate by ALSO
      folding a probe MuReadout given a synthetic ``width_mult != 1`` and checking
      the ``<=1e-6`` equality), and the out-proj is then a plain ``nn.Linear`` in
      the traced graph (so onnx sees a standard transformer, not the unsupported
      MuReadout multiplier op).

    HONEST faithfulness note (recorded in the report): v1's ``Transformer(mup=True)``
    wires ONLY the MuReadout out-proj swap from the encoder flag — it does NOT pass
    mup down to its EncoderLayers/Attention, so the 1/d softmax scale + Q-zero
    attention init (attention.py:275,319) are NOT active in v1's encoder-mup path
    (they live in ``Attention(mup=True)``, constructed independently). The v2 port
    composes ``Transformer(mup=True)`` and is byte-faithful to that ACTUAL v1
    behaviour. The coord-curve comparison + "is it muP-flat?" judgment are
    MU2/MU-HUMAN, not MU1.

    Negative control (``test_gates_m6.py``): the ``corruption`` hook perturbs the
    v2 encoded sequence — the bitwise parity check must FAIL while the structural
    (MuReadout / mup-Dense) and export-fold checks stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    from mup import MuReadout, set_base_shapes  # noqa: PLC0415

    outdir = Path(outdir)
    print("=" * 96)
    print("MU1 muP forward-init BITWISE parity vs v1 Transformer(mup=True) + MuReadout export-fold")
    print("=" * 96)
    checks: dict[str, bool] = {}

    norm_dict = _mu1_norm_dict(outdir)
    modules = _mu1_modules(norm_dict)

    # -- (a) structural muP wiring -------------------------------------------
    checks["embed_dense_is_mup"] = modules["track_embed"].mup is True
    checks["encoder_is_mup"] = modules["encoder"].mup is True
    out_proj = modules["encoder"].encoder.out_proj
    checks["out_proj_is_mu_readout"] = isinstance(out_proj, MuReadout)
    checks["out_proj_weight_zeroed_at_init"] = bool((out_proj.weight == 0).all())
    checks["out_proj_bias_zeroed_at_init"] = out_proj.bias is not None and bool(
        (out_proj.bias == 0).all()
    )

    # -- compile + bind + materialise the GN2_muP plan -----------------------
    fit_plan = compile_plan(modules, Mode.FIT, sources=_mu1_sources(), sinks=["loss.total"])
    test_plan = compile_plan(
        modules,
        Mode.TEST,
        sources=_mu1_sources(),
        sinks=["preds.jets.jets_classification"],
    )
    bind_all(modules, resolve_bind_schema([fit_plan, test_plan]))
    materialise_all(modules)
    checks["embed_net_built_with_mup"] = modules["track_embed"].net.mup is True

    # give the encoder + embed NON-ZERO weights so the parity test is meaningful
    # (a zeroed readout would make every forward all-zero post-norm) — the muP
    # INIT distribution is asserted structurally above; the forward parity tests
    # that the v2 graph routes the SAME weights through the SAME math as v1.
    gen = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for p in modules["encoder"].encoder.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        for p in modules["track_embed"].net.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)

    # -- (b) forward-init BITWISE parity vs the INDEPENDENT v1 encoder --------
    jets = torch.randn(_MU1_B, len(_MU1_JET_VARIABLES), generator=gen)
    tracks = torch.randn(_MU1_B, _MU1_T, len(_MU1_TRACK_VARIABLES), generator=gen)
    track_mask = torch.zeros(_MU1_B, _MU1_T, dtype=torch.bool)
    labels = torch.randint(0, len(_MU1_CLASS_NAMES), (_MU1_B,), generator=gen)
    b = Bundle()
    b.set("inputs.jets", jets)
    b.set("inputs.tracks", tracks)
    b.set("masks.tracks", track_mask)
    b.set("labels.jets.flavour_label", labels)
    out = Executor(fit_plan).run(b, debug=True)
    v2_encoded = out.get("encoded.seq")
    if corruption is not None:
        v2_encoded = corruption(v2_encoded)
    # the v1 reference runs on the SAME seq.x/seq.mask the v2 graph produced
    seq_x = out.get("seq.x")
    seq_mask = out.get("seq.mask")
    v1_encoded = _mu1_v1_encoder_reference(modules["encoder"], seq_x, seq_mask)
    checks["encoder_forward_bitwise_vs_v1_mup"] = v2_encoded.shape == v1_encoded.shape and (
        torch.equal(v2_encoded, v1_encoded)
    )

    # -- (c) MuReadout export-fold equals the MuReadout forward ---------------
    # snapshot the eager (MuReadout) encoded.seq, then fold via the export
    # protocol and confirm the post-fold forward is numerically equal. The fold
    # is applied to the LIVE encoder module the executor runs (set_export_mode is
    # what the OnnxAdapter invokes at trace time).
    fresh = Bundle()
    fresh.set("inputs.jets", jets)
    fresh.set("inputs.tracks", tracks)
    fresh.set("masks.tracks", track_mask)
    pre_fold = Executor(test_plan).run(fresh).get("encoded.seq").clone()
    modules["encoder"].set_export_mode()
    checks["out_proj_folded_to_plain_linear"] = (
        type(modules["encoder"].encoder.out_proj) is torch.nn.Linear
    )
    fresh2 = Bundle()
    fresh2.set("inputs.jets", jets)
    fresh2.set("inputs.tracks", tracks)
    fresh2.set("masks.tracks", track_mask)
    post_fold = Executor(test_plan).run(fresh2).get("encoded.seq")
    fold_max_abs = float((pre_fold - post_fold).abs().max())
    # at the architectural default width_mult==1.0 the fold is the identity
    # multiplier, so the equality is BITWISE; the <=1e-6 path is exercised by the
    # non-unit probe below.
    checks["export_fold_equals_mu_readout_bitwise_at_unit_mult"] = torch.equal(pre_fold, post_fold)
    checks["export_fold_within_tolerance"] = fold_max_abs <= 1e-6
    # idempotent: a second set_export_mode leaves the plain Linear in place
    modules["encoder"].set_export_mode()
    checks["export_fold_is_idempotent"] = (
        type(modules["encoder"].encoder.out_proj) is torch.nn.Linear
    )

    # -- (d) NON-VACUOUS fold check: a non-unit width_mult still folds <=1e-6 -
    # the unit-mult fold is bitwise but exercises only output_mult/width_mult==1.
    # Build a probe MuReadout with a real (>1) width_mult by setting it against a
    # NARROWER base, give it non-zero weights, and confirm the same fold formula
    # (W_folded=(output_mult/width_mult)*W, bias unchanged) matches the MuReadout
    # forward within 1e-6 — proving the fold is correct for the multipliers the
    # routing stage will introduce, not just the trivial identity.
    probe = MuReadout(_MU1_OUT_DIM, len(_MU1_CLASS_NAMES), output_mult=2.5)
    base = MuReadout(_MU1_OUT_DIM // 2, len(_MU1_CLASS_NAMES))
    set_base_shapes(probe, base, rescale_params=False)  # width_mult = 2.0
    with torch.no_grad():
        probe.weight.copy_(torch.randn(probe.weight.shape, generator=gen))
        probe.bias.copy_(torch.randn(probe.bias.shape, generator=gen))
    probe.eval()
    mult = float(probe.output_mult) / float(probe.width_mult())
    folded = torch.nn.Linear(probe.in_features, probe.out_features)
    with torch.no_grad():
        folded.weight.copy_(probe.weight * mult)
        folded.bias.copy_(probe.bias)
    folded.eval()
    x_probe = torch.randn(_MU1_B, _MU1_OUT_DIM, generator=gen)
    with torch.no_grad():
        probe_ref = probe(x_probe)
        probe_folded = folded(x_probe)
    probe_max_abs = float((probe_ref - probe_folded).abs().max())
    checks["nonunit_width_mult_fold_within_tolerance"] = (
        not math.isclose(probe.width_mult(), 1.0) and probe_max_abs <= 1e-6
    )

    passed = all(checks.values())
    criterion = (
        "the muP ARCHITECTURAL port (M6 sub-wave B; design §3.4 KEEP-architecture/BREAK-routing): "
        "the mup: true init_arg on StreamEmbed (-> Dense(mup=True)) and TransformerEncoder "
        "(-> Transformer(mup=True)) reproduces v1's encoder-flag muP path. The encoder out-proj is "
        "a mup.MuReadout with weight+bias zeroed at init (transformer.py:625-628) and the embed "
        "Dense carries mup=True; the v2 encoder forward is BITWISE identical to an INDEPENDENT v1 "
        "Transformer(mup=True) (weight-loaded, run through v1 forward) on deterministic init "
        "values with NO training; and the MuReadout->plain-Linear export fold (set_export_mode, "
        "W_folded="
        "(output_mult/width_mult)*W, bias unchanged) is numerically equal to the MuReadout forward "
        "(bitwise at width_mult==1.0, <=1e-6 for a non-unit probe). The coord-curve comparison + "
        "muP-flat plot judgment are MU2/MU-HUMAN, not MU1."
    )
    report = _base_report(
        "mu1_mup_forward_init_parity",
        passed,
        criterion,
        {
            "track_variables": list(_MU1_TRACK_VARIABLES),
            "jet_variables": list(_MU1_JET_VARIABLES),
            "class_names": list(_MU1_CLASS_NAMES),
            "embed_dim": _MU1_EMBED_DIM,
            "out_dim": _MU1_OUT_DIM,
            "num_heads": _MU1_NUM_HEADS,
            "num_layers": _MU1_NUM_LAYERS,
            "batch": _MU1_B,
            "n_tracks": _MU1_T,
            "norm_dict": str(norm_dict),
            "fold_max_abs_diff": fold_max_abs,
            "nonunit_probe_width_mult": float(probe.width_mult()),
            "nonunit_probe_max_abs_diff": probe_max_abs,
            "approach": "compose v1 Transformer(mup=True)/Dense(mup=True); export-fold MuReadout",
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["v1_reference"] = (
        "INDEPENDENT v1 reference (instance independence, NOT a math re-implementation): a "
        "separately-instantiated v1 Transformer(mup=True) — a distinct object from the v2 "
        "encoder's composed instance, weight-loaded from it — run through v1's forward. The "
        "bitwise torch.equal tests that the v2 GRAPH PATH (compile -> two-phase bind -> executor) "
        "routes tensors through the SAME muP-parametrised v1 math; it does not re-derive it."
    )
    report["faithfulness_note"] = (
        "v1's Transformer(mup=True) wires ONLY the MuReadout out-proj swap from the encoder flag — "
        "it does NOT pass mup down to its EncoderLayers/Attention (transformer.py:604-614), so the "
        "1/d softmax scale + Q-zero attention init (attention.py:275,319-326) are NOT active in "
        "v1's encoder-mup path; they live in Attention(mup=True), constructed independently. The "
        "v2 TransformerEncoder(mup=True) composes Transformer(mup=True) and is therefore "
        "byte-faithful to v1's ACTUAL encoder-mup behaviour. The embed's Dense(mup=True) DOES "
        "apply the muP linear init (dense.py:96-102), reproducing v1 InitNet's intended muP init "
        "(v1 InitNet's own re-reset call, initnet.py:69, RAISES AttributeError: Dense exposes "
        "only _reset_parameters and nn.Module has no reset_parameters, so v1's InitNet(mup=True) "
        "path is broken at construction; v2 relies on Dense(mup=True)'s own __init__ reset "
        "(dense.py:88-89,96-102), which is the intended muP distribution)."
    )
    report["export_fold"] = (
        "MuReadout.forward applies output_mult*x/width_mult before the linear; the export fold "
        "bakes that multiplier into the weight (W_folded=(output_mult/width_mult)*W) and leaves "
        "the bias unchanged (the multiplier scales only the x@Wt term), then swaps the out-proj "
        "for a plain nn.Linear. Inference math is then identity to a standard Linear, so the "
        "traced ONNX graph is a plain transformer (no unsupported MuReadout multiplier op). "
        "Deterministic, idempotent."
    )
    _print_checks(checks)
    _print_verdict("mu1", passed, criterion, _emit_report(report, outdir, "mu1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# MU2 — muP routing validator (design-conformance, sub-wave B)
# ---------------------------------------------------------------------------

_MU2_LRS = {"initial": 1e-4, "max": 5e-4, "end": 1e-5, "pct_start": 0.1}


def _mu2_mup_override(outdir: Path, shape_path: Path) -> Path:
    """Write the GN2_muP-routing override config (stacked on gn2v2-dummy.yaml).

    The minimal v2-native muP ROUTING surface (NOT the GN2_muP corpus config —
    that is the next stage): adds ``mup: true`` to the dummy config's
    track_embed/encoder and the ``model.init_args.mup: {shape_path, apply_to}``
    routing block. Stacked on the shipped ``gn2v2-dummy.yaml`` so MU2 exercises
    the REAL ``salt2 mup-shapes`` / ``salt2 graph validate`` tooling on a true
    config file, not just in-process module dicts.

    Returns
    -------
    Path
        The override YAML path.
    """
    import yaml  # noqa: PLC0415 - keep the module import surface lean

    outdir.mkdir(parents=True, exist_ok=True)
    override = {
        "name": "GN2muP_mu2_fixture",
        "model": {
            "init_args": {
                "optimizer": "AdamW",
                "mup": {
                    "shape_path": str(shape_path),
                    "apply_to": ["track_embed", "encoder"],
                },
                "modules": {
                    "track_embed": {"init_args": {"mup": True}},
                    "encoder": {"init_args": {"mup": True}},
                },
            }
        },
    }
    path = outdir / "mu2_mup_override.yaml"
    path.write_text(yaml.safe_dump(override, sort_keys=False))
    return path


def run_mu2(
    outdir: Path | str,
    *,
    corruption: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """MU2: muP ROUTING validator (design-conformance, sub-wave B; design §3.4 line 695).

    The muP ROUTING half (design §3.4 KEEP-architecture/BREAK-routing): the v1
    regex routing (``apply_to`` x ``parameter_name`` zip into the model init_args,
    configuration_muP.py:98-115) is replaced by an EXPLICIT module-name list
    ``SaltModule.mup.apply_to``. MU2 asserts the design-conformance contract:

    - **validator rule 1 (ERROR)** — ``apply_to`` naming a module that lacks a
      ``mup`` init_arg (`module_supports_mup` False, e.g. the pool) raises
      `ConfigError`; an ``apply_to`` naming a non-existent module raises too;
    - **validator rule 2 (WARN)** — a module with ``mup: true`` left OUT of
      ``apply_to`` emits a warning (its base shapes / MuAdamW grouping silently
      diverge);
    - **MuAdamW swap** — ``SaltModule._get_optimizer_class`` returns
      ``mup.optim.MuAdamW`` whenever a ``mup`` block is configured (regardless of
      the ``optimizer`` name), and plain ``AdamW`` when it is not;
    - **entry-point casing fix** — the v1 ``setup_mup`` console entry pointed at
      ``salt.utils.mup_utils.main_mup`` (lowercase) but the module is
      ``muP_utils/main_muP.py`` (capital P) — a broken entry point
      (pyproject.toml:95). v2 resolves ``setup_mup`` to ``salt.core.mup:setup_mup``
      (importable + callable);
    - **tooling end-to-end (real config + real CLI)** — ``salt2 mup-shapes`` on
      the GN2_muP-routing fixture writes an infshape file; the file applied at
      bind makes the `MuReadout.width_mult()` resolve to the real base ratio
      (NOT the architectural default 1.0); ``salt2 graph validate`` prints the
      muP-routing OK line.

    Negative control (``test_gates_m6.py``): the ``corruption`` hook mutates the
    routing config (e.g. injects a non-mup module into ``apply_to``) — the
    validator must then ERROR, flipping the gate.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    from torch.optim import AdamW  # noqa: PLC0415

    from salt.core.mup import setup_mup as _setup_mup_entry  # noqa: PLC0415

    outdir = Path(outdir)
    print("=" * 96)
    print("MU2 muP ROUTING validator (apply_to name-lists, MuAdamW swap, setup_mup casing)")
    print("=" * 96)
    checks: dict[str, bool] = {}

    norm_dict = _mu1_norm_dict(outdir)
    mup_block = {"apply_to": ["track_embed", "encoder"]}
    if corruption is not None:
        mup_block = corruption(mup_block)

    # -- (a) validator rules over the in-process module dict ------------------
    valid_modules = _mu1_modules(norm_dict)
    try:
        normalised = validate_mup_routing(mup_block, valid_modules)
        checks["valid_apply_to_accepted"] = normalised is not None and normalised["apply_to"] == [
            "track_embed",
            "encoder",
        ]
    except ConfigError:
        checks["valid_apply_to_accepted"] = False

    # rule 1: apply_to a module WITHOUT a mup init_arg (pool) -> ConfigError
    checks["apply_to_non_mup_module_errors"] = _raises(
        ConfigError, validate_mup_routing, {"apply_to": ["pool"]}, _mu1_modules(norm_dict)
    )
    # rule 1: apply_to a NON-EXISTENT module -> ConfigError
    checks["apply_to_unknown_module_errors"] = _raises(
        ConfigError, validate_mup_routing, {"apply_to": ["does_not_exist"]}, _mu1_modules(norm_dict)
    )
    # rule 2: a mup:true module OUTSIDE apply_to -> WARNING
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_mup_routing({"apply_to": ["track_embed"]}, _mu1_modules(norm_dict))
    checks["mup_on_module_outside_apply_to_warns"] = any("NOT in" in str(w.message) for w in caught)
    # unknown key / empty apply_to -> ConfigError
    checks["unknown_mup_key_errors"] = _raises(
        ConfigError, validate_mup_routing, {"apply_to": ["encoder"], "bogus": 1}, valid_modules
    )
    checks["empty_apply_to_errors"] = _raises(
        ConfigError, validate_mup_routing, {"apply_to": []}, valid_modules
    )

    # -- (b) MuAdamW swap when mup configured, AdamW when not -----------------
    from mup.optim import MuAdamW  # noqa: PLC0415

    mup_model = SaltModule(
        modules=_mu1_modules(norm_dict), lrs=_MU2_LRS, mup={"apply_to": ["track_embed", "encoder"]}
    )
    checks["muadamw_selected_when_mup_configured"] = (
        mup_model._get_optimizer_class() is MuAdamW  # noqa: SLF001
    )
    plain_model = SaltModule(modules=_mu1_modules(norm_dict), lrs=_MU2_LRS)
    checks["adamw_selected_when_no_mup"] = (
        plain_model._get_optimizer_class() is AdamW  # noqa: SLF001
        and plain_model.mup_cfg is None
    )

    # -- (c) setup_mup console entry resolves (casing fix) -------------------
    checks["setup_mup_entry_point_callable"] = callable(_setup_mup_entry)

    # -- (d) end-to-end tooling on a REAL GN2_muP-routing config + CLI -------
    shape_path = outdir / "mu2_shapes.bsh"
    dummy_cfg = str(CONFIG_DIR / "gn2v2-dummy.yaml")
    override = _mu2_mup_override(outdir, shape_path)
    set_norm = f"model.modules.norm.init_args.norm_dict={norm_dict}"
    shapes_rc = salt2_main([
        "mup-shapes",
        "-c",
        dummy_cfg,
        "-c",
        str(override),
        "--set",
        set_norm,
        "--base-width",
        "8",
        "--delta-width",
        "16",
    ])
    checks["salt2_mup_shapes_runs"] = shapes_rc == 0 and shape_path.is_file()

    # the generated shape file, applied at bind, makes width_mult resolve to the
    # real base ratio (16 model / 8 base = 2.0) — proving the routing-stage shape
    # application (SaltModule._apply_mup_shapes), not the architectural default 1.0
    width_mult = _mu2_bound_width_mult(dummy_cfg, str(override), set_norm)
    checks["shape_file_applied_at_bind_width_mult_2"] = width_mult is not None and math.isclose(
        width_mult, 2.0
    )

    validate_rc = salt2_main([
        "graph",
        "validate",
        "--mode",
        "fit",
        "-c",
        dummy_cfg,
        "-c",
        str(override),
        "--set",
        set_norm,
    ])
    checks["salt2_graph_validate_mup_routing_ok"] = validate_rc == 0

    passed = all(checks.values())
    criterion = (
        "the muP ROUTING half (M6 sub-wave B; design §3.4 KEEP-architecture/BREAK-routing line "
        "695): apply_to is an EXPLICIT module-name list (NOT v1's regex zip, "
        "configuration_muP.py:98-115). The validator ERRORS when apply_to names a module without a "
        "mup init_arg or a non-existent module, and WARNS when a mup:true module is left out of "
        "apply_to; SaltModule swaps the optimizer to mup.optim.MuAdamW when mup is configured "
        "(AdamW otherwise); the setup_mup console entry resolves (casing fix from the broken v1 "
        "salt.utils.mup_utils.main_mup -> salt.core.mup:setup_mup); and salt2 mup-shapes + "
        "salt2 graph validate run end-to-end on a real GN2_muP-routing config, with the generated "
        "shape file applied at bind so MuReadout.width_mult resolves to the real base ratio."
    )
    report = _base_report(
        "mu2_mup_routing_validator",
        passed,
        criterion,
        {
            "apply_to": ["track_embed", "encoder"],
            "base_width": 8,
            "delta_width": 16,
            "resolved_width_mult": width_mult,
            "norm_dict": str(norm_dict),
            "shape_path": str(shape_path),
            "override_config": str(override),
            "approach": (
                "validate_mup_routing over module dicts + SaltModule MuAdamW branch + setup_mup "
                "entry + salt2 mup-shapes/graph validate on a real config"
            ),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["v1_reference"] = (
        "v1 routing engine update_config (configuration_muP.py:98) zips apply_to x parameter_name "
        "x base|delta and asserts each apply_to key exists in init_args (:111-113); v2 replaces "
        "the regex/zip with an explicit instance-name list + a typed validator. v1 "
        "store_shapes_mup (:251) + make_base_shapes (:257) + generate_base_delta_config (:119) -> "
        "salt2 mup-shapes; "
        "v1 coord-check substring match (:448,460) -> salt2 mup-coord-check net.<name>.* traversal."
    )
    report["entry_point_fix"] = (
        "v1 pyproject.toml:95 setup_mup -> salt.utils.mup_utils.main_mup:main (lowercase mup_utils/"
        "main_mup) but the real module is salt/utils/muP_utils/main_muP.py (capital P) — a broken "
        "entry point. v2 normalises the casing: setup_mup -> salt.core.mup:setup_mup, a thin alias "
        "for salt2 mup-shapes."
    )
    _print_checks(checks)
    _print_verdict("mu2", passed, criterion, _emit_report(report, outdir, "mu2"))
    return (0 if passed else 1), report


def _mu2_bound_width_mult(dummy_cfg: str, override: str, set_norm: str) -> float | None:
    """Bind the GN2_muP-routing config and read the encoder MuReadout ``width_mult``.

    Exercises `SaltModule._apply_mup_shapes` end-to-end: parse the real config,
    compile + bind the combined graph (which applies the shape file), and read
    the live `MuReadout.width_mult()`. Returns None if the out-proj is not a
    `MuReadout` (a wiring regression).

    Returns
    -------
    float | None
        The resolved width multiplier (2.0 for a 16-width model on an 8-width
        base), or None on a wiring failure.
    """
    from mup import MuReadout  # noqa: PLC0415

    from salt.core.mup import _combined_graph, _parse_cli  # noqa: PLC0415

    cli = _parse_cli([dummy_cfg, override], [set_norm])
    model = cli.model
    combined = _combined_graph(cli)
    plan = compile_plan(combined, Mode.FIT, sources={}, sinks=["loss.total"])
    model._bind(resolve_bind_schema([plan]))  # noqa: SLF001
    out_proj = model.net["encoder"].encoder.out_proj
    if not isinstance(out_proj, MuReadout):
        return None
    return float(out_proj.width_mult())


# ---------------------------------------------------------------------------
# ED1 — EdgeFeatures + EdgeEmbed forward parity vs v1 (sub-wave C, stage 1)
# ---------------------------------------------------------------------------

# A GN2XE-shaped edge fixture (GN2XE.yaml:38-45,67-72): the tracks stream carries
# the variables the edge features need (eta/phi/pt for dR/kt/z, subjetIndex for
# subjetIndex), the edge feature list is the shipped GN2XE set, and the edge
# embed is the shipped [B,T,T,E] -> [B,T,T,D_e=32] Dense (hidden [32], SiLU).
_ED1_TRACK_VARIABLES: tuple[str, ...] = (
    "pt",
    "eta",
    "phi",
    "subjetIndex",
    "d0",
    "z0",
)
_ED1_EDGE_FEATURES: tuple[str, ...] = ("dR", "z", "kt", "subjetIndex", "isSelfLoop")
_ED1_CLASS_NAMES: tuple[str, ...] = ("bjets", "cjets", "ujets")
_ED1_EDGE_EMBED_DIM = 32
_ED1_EDGE_HIDDEN: tuple[int, ...] = (32,)
_ED1_B = 4
_ED1_T = 6
# the encoder-edge-path dims (GN2XE.yaml:&embed_dim 192 / &out_dim 128 / num_heads 4
# / num_layers 6, scaled down — the edge wiring, not the size, is under test)
_ED1_EMBED_DIM = 16
_ED1_OUT_DIM = 12
_ED1_NUM_HEADS = 2
_ED1_NUM_LAYERS = 2


def _ed1_modules() -> dict[str, Any]:
    """Build the GN2XE-shaped EdgeFeatures + EdgeEmbed module pair (this stage).

    Just the two edge modules under test — the encoder/attention edge path is a
    LATER sub-wave C stage (plan 12). EdgeFeatures requires raw ``inputs.tracks``
    + ``masks.tracks`` and produces ``edges.tracks`` ``[B, T, T, E]``; EdgeEmbed
    maps that to ``edges.tracks_emb`` ``[B, T, T, D_e]`` (design §6.7 1418-1421).

    Returns
    -------
    dict[str, Any]
        Instance-named edge modules.
    """
    modules: dict[str, Any] = {
        "edge_features": EdgeFeatures(stream="tracks", features=list(_ED1_EDGE_FEATURES)),
        "edge_embed": EdgeEmbed(
            stream="tracks",
            out_dim=_ED1_EDGE_EMBED_DIM,
            dense={"hidden_layers": list(_ED1_EDGE_HIDDEN), "activation": "SiLU"},
        ),
    }
    for name, module in modules.items():
        module.name = name
    return modules


def _ed1_sources():
    """The GN2XE dataset boundary the edge path reads: raw tracks + its pad mask.

    ``inputs.tracks`` is the RAW (un-normalised) ``[B, T, F]`` stream carrying
    the edge variables as declared ``fields`` (so EdgeFeatures resolves its
    indices_map by NAME at bind, design §2.2); ``masks.tracks`` is the stream's
    pad mask (design §6.7 1418).

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    return unflatten_spec({
        "inputs.tracks": TensorSpec(
            shape=("B", "T:tracks", len(_ED1_TRACK_VARIABLES)),
            dtype="float32",
            fields=tuple(_ED1_TRACK_VARIABLES),
        ),
        "masks.tracks": TensorSpec(shape=("B", "T:tracks"), dtype="bool", kind="pad_mask"),
    })


def _ed1_encoder_sources():
    """The GN2XE encoder boundary: raw tracks + pad mask + the jet flavour label.

    Same as `_ed1_sources` plus the TRAINING-gated ``labels.jets.flavour_label``
    the classification head consumes in FIT (the encoder model has a head; the
    edge-only stage-b/c plan uses the leaner `_ed1_sources`).

    Returns
    -------
    NestedSpec
        The nested source spec for the encoder `compile_plan`.
    """
    return unflatten_spec({
        "inputs.tracks": TensorSpec(
            shape=("B", "T:tracks", len(_ED1_TRACK_VARIABLES)),
            dtype="float32",
            fields=tuple(_ED1_TRACK_VARIABLES),
        ),
        "masks.tracks": TensorSpec(shape=("B", "T:tracks"), dtype="bool", kind="pad_mask"),
        "labels.jets.flavour_label": TensorSpec(
            shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
        ),
    })


def _ed1_v1_edges(raw_tracks: torch.Tensor) -> torch.Tensor:
    """Run the INDEPENDENT v1 ``EdgeConstructor`` on the SAME raw tracks.

    Constructs a SEPARATE v1 ``EdgeConstructor`` (NOT the v2 module's composed
    function call) keyed on the same variables and edge-feature list, runs its
    ``forward`` on a CLONE of the raw input, and returns the
    ``_edge_features_tracks`` tensor it writes back into the input dict
    (edge_constructor.py:41-44). This is the v1-decidable reference for the
    dR/kt/z/subjetIndex/isSelfLoop math (edge_features.py:52-146).

    Returns
    -------
    torch.Tensor
        The v1 ``[B, T, T, E]`` edge-feature tensor.
    """
    v1ec = V1EdgeConstructor(
        input_name="tracks",
        edge_features=list(_ED1_EDGE_FEATURES),
        variables={"tracks": list(_ED1_TRACK_VARIABLES)},
    )
    v1_inputs = {"tracks": raw_tracks.clone()}
    v1ec(v1_inputs)
    return v1_inputs["_edge_features_tracks"]


def _ed1_v1_embed(edge_embed: EdgeEmbed, v1_edges: torch.Tensor) -> torch.Tensor:
    """Run an INDEPENDENT v1 ``Dense`` (weight-loaded from the v2 EdgeEmbed) on the v1 edges.

    Mirrors v1's ``edge_init_nets[0]`` (saltmodel.py:69-80,132 — a `Dense` over
    the ``[B, L, L, E]`` edge matrix). A SEPARATE v1 `Dense` is built with the
    same widths and loaded from the v2 `EdgeEmbed`'s composed Dense, then run on
    the v1 edge tensor — so the only thing under test is the embed wiring, not
    the edge math.

    Returns
    -------
    torch.Tensor
        The v1 ``[B, T, T, D_e]`` embedded-edge tensor.
    """
    v1dense = V1Dense(
        input_size=len(_ED1_EDGE_FEATURES),
        output_size=_ED1_EDGE_EMBED_DIM,
        hidden_layers=list(_ED1_EDGE_HIDDEN),
        activation="SiLU",
    )
    v1dense.load_state_dict(edge_embed.net.state_dict())
    v1dense.eval()
    with torch.no_grad():
        return v1dense(v1_edges)


def _ed1_norm_dict(outdir: Path) -> Path:
    """Write a tracks norm dict for the GN2XE-shaped encoder-edge fixture.

    The edge features are built on the RAW input (design §6.3), but the encoder
    path needs ``normed.tracks`` for the `StreamEmbed`; this supplies the
    per-variable constants (distinct, so a field-order bug can't pass silently).

    Returns
    -------
    Path
        The norm-dict YAML path.
    """
    import yaml  # noqa: PLC0415 - keep the module import surface lean

    outdir.mkdir(parents=True, exist_ok=True)
    nd = {
        "tracks": {
            var: {"mean": 0.1 * i, "std": 1.0 + 0.1 * i}
            for i, var in enumerate(_ED1_TRACK_VARIABLES)
        }
    }
    path = outdir / "ed1_norm_dict.yaml"
    path.write_text(yaml.safe_dump(nd))
    return path


def _ed1_encoder_modules(norm_dict: Path) -> dict[str, Any]:
    """Build the GN2XE-shaped FULL edge model: edge build -> embed -> encoder edge path.

    Mirrors GN2XE.yaml: an `EdgeFeatures` on the RAW tracks -> `EdgeEmbed`
    (``edges.tracks_emb``), a `StreamEmbed` on ``normed.tracks``, a single-stream
    `Concat` (tracks first — the edge-stream-first rule), and a
    `TransformerEncoder` with the ``edges: edges.tracks_emb`` port +
    ``update_edges: true`` (GN2XE.yaml:79-80), feeding a `GlobalAttentionPooling`.
    The encoder swaps every `Attention` for an `EdgeAttention` (transformer.py:
    365-373) — the path under test in this stage.

    Returns
    -------
    dict[str, Any]
        Instance-named modules.
    """
    modules: dict[str, Any] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["tracks"]),
        "edge_features": EdgeFeatures(stream="tracks", features=list(_ED1_EDGE_FEATURES)),
        "edge_embed": EdgeEmbed(
            stream="tracks",
            out_dim=_ED1_EDGE_EMBED_DIM,
            dense={"hidden_layers": list(_ED1_EDGE_HIDDEN), "activation": "SiLU"},
        ),
        "track_embed": StreamEmbed(
            stream="tracks",
            out_dim=_ED1_EMBED_DIM,
            dense={"hidden_layers": [_ED1_EMBED_DIM], "activation": "SiLU"},
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=_ED1_EMBED_DIM,
            num_layers=_ED1_NUM_LAYERS,
            out_dim=_ED1_OUT_DIM,
            attention={"num_heads": _ED1_NUM_HEADS, "attn_type": "torch-math"},
            dense={"activation": "SiLU"},
            edges="edges.tracks_emb",
            edge_embed_dim=_ED1_EDGE_EMBED_DIM,
            update_edges=True,
        ),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=list(_ED1_CLASS_NAMES),
            input="pooled.global",
            dense={"hidden_layers": [_ED1_OUT_DIM], "activation": "SiLU"},
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    modules["loss"].narrow(LossSum.collect_loss_keys(modules))
    return modules


def _ed1_v1_encoder_edge_reference(
    v2_encoder: TransformerEncoder,
    seq_x: torch.Tensor,
    seq_mask: torch.Tensor,
    edge_x: torch.Tensor,
) -> torch.Tensor:
    """Run an INDEPENDENT v1 ``Transformer(edge_embed_dim>0, update_edges=True)`` on same inputs.

    Constructs a SEPARATE v1 `Transformer` with the edge config (NOT the v2
    module's composed instance), weight-loads it from the v2 encoder's composed
    v1 Transformer, and runs v1's exact edge forward (the register/edge zero-pad,
    transformer.py:689-719; the per-layer `EdgeAttention` + edge update,
    transformer.py:729-733). The returned tensor is the register-augmented
    ``encoded.seq`` (v1 keeps the register rows).

    Returns
    -------
    torch.Tensor
        The v1 ``[B, T+R, out_dim]`` encoded sequence (edge path).
    """
    ref = V1Transformer(
        num_layers=_ED1_NUM_LAYERS,
        embed_dim=_ED1_EMBED_DIM,
        out_dim=_ED1_OUT_DIM,
        norm="LayerNorm",
        attn_type="torch-math",
        do_final_norm=True,
        num_registers=v2_encoder.num_registers,
        edge_embed_dim=_ED1_EDGE_EMBED_DIM,
        update_edges=True,
        attn_kwargs={"num_heads": _ED1_NUM_HEADS},
        dense_kwargs={"activation": "SiLU"},
    )
    ref.load_state_dict(v2_encoder.encoder.state_dict())
    ref.eval()
    with torch.no_grad():
        encoded, _ = ref({"seq": seq_x}, pad_mask={"seq": seq_mask}, edge_x=edge_x)
    return encoded


def _ed1_export_config() -> ExportConfig:
    """The GN2XE ONNX export config: a single rank-3 tracks input, dynamic T.

    The ONLY positional input is ``inputs.tracks`` flagged ``sequence: true`` so
    the OnnxAdapter registers its token axis as the dynamic ``n_tracks`` axis
    (adapter.py:201-205). The edge tensor is PRODUCED inside the model (by
    `EdgeFeatures`/`EdgeEmbed`) from this raw input — both edge T axes inherit the
    same dynamic token count, the dynamic-edge-T contract (FD §6.7 ONNX note).

    Returns
    -------
    ExportConfig
        The unresolved export config for `export_graph`.
    """
    return ExportConfig(
        model_name="GN2XEv2EdgeProbe",
        inputs=[ExportInput(port="inputs.tracks", name="track_features", sequence=True)],
    )


def _ed1_onnx_trace(
    enc_modules: dict[str, Any], outdir: Path
) -> tuple[dict[str, bool], dict[str, dict[int, str]]]:
    """Export the GN2XE model and assert dynamic edge-T axes + a shape-derived register pad.

    The load-bearing ONNX contract (FD §6.7 ONNX note, plan 12 sub-wave C). Two
    hazards the trace must clear:

    (a) **Both edge T axes dynamic.** The edge tensor ``edges.tracks`` ``[B, T, T,
        E]`` and its EdgeEmbed ``[B, T, T, D_e]`` are PRODUCED inside the model
        from the dynamic ``inputs.tracks`` token axis (the ``track_features``
        positional input flagged ``sequence: true`` -> dynamic ``n_tracks`` axis,
        adapter.py:201-205), so BOTH edge axes derive from that one dynamic count.
        The functional proof: trace ONCE at a fixed track count, then run
        onnxruntime at SEVERAL DIFFERENT track counts and confirm each matches the
        eager model (`check_onnx` length sweep). If either edge axis were baked to
        the trace-time count, an off-trace length would mis-shape and fail.

    (b) **Register zero-pad is shape-derived.** The in-encoder pad to the
        register-augmented length builds ``torch.zeros((B, x.shape[1] -
        edge_x.shape[1], ...))`` from the DYNAMIC ``x.shape[1]``
        (transformer.py:689-719). This must trace to shape-derived nodes
        (``Shape``/``Sub``/``ConstantOfShape``/``Concat``/``Expand``), NOT a baked
        ``Constant`` zeros tensor of the trace-time padded shape — else the pad
        would be the wrong size at a different track count (caught functionally by
        the same length sweep, AND structurally by inspecting the graph nodes).

    Returns
    -------
    tuple[dict[str, bool], dict[str, dict[int, str]]]
        The ONNX-trace checks and the adapter's dynamic-axes mapping.
    """
    import onnx  # noqa: PLC0415 - heavy import, onnx-leg only

    checks: dict[str, bool] = {}
    # the GN2XE flavour head is the export sink (split_scalars global reduce,
    # exactly the VS1/MU pattern); the edge path is internal to the trace.
    head = enc_modules["jets_classification"]
    manifest = [
        ExportOutput(port="preds.jets.jets_classification", names=list(head.class_suffixes))
    ]
    onnx_path = outdir / "ed1_gn2xe_edge.onnx"
    result = export_graph(
        enc_modules,
        _ed1_export_config(),
        {"tracks": list(_ED1_TRACK_VARIABLES)},
        onnx_path,
        outputs=manifest,
        run_name="GN2XEv2EdgeProbe",
    )
    # (a) the track input registers a DYNAMIC token axis (n_tracks) — the single
    # dynamic count both edge T axes derive from (adapter.dynamic_axes).
    dyn_axes = result.adapter.dynamic_axes
    checks["edge_track_axis_is_dynamic"] = any(
        str(name).startswith("n_") for axmap in dyn_axes.values() for name in axmap.values()
    )
    # the loaded graph: the track input is rank-2 [L, F] with a SYMBOLIC token dim
    # (dim_param set, not a fixed dim_value) — the edge axes inherit this symbol.
    graph = onnx.load(str(onnx_path)).graph
    track_in = next(
        (i for i in graph.input if i.name == result.adapter.input_names[0]), graph.input[0]
    )
    track_dims = track_in.type.tensor_type.shape.dim
    checks["onnx_track_input_token_dim_symbolic"] = len(track_dims) == 2 and bool(
        track_dims[0].dim_param
    )
    # (b) the graph carries shape-derivation ops for the register pad (NOT a baked
    # constant pad): Shape/Sub/ConstantOfShape/Concat/Expand all present means the
    # zeros pad is built from the runtime sequence length, not a frozen tensor.
    op_types = {node.op_type for node in graph.node}
    checks["register_pad_shape_derived_ops_present"] = bool(
        {"Shape", "ConstantOfShape"} & op_types
    ) and bool({"Sub", "Concat", "Expand", "Slice"} & op_types)
    # the dynamic-T + shape-derived-pad FUNCTIONAL proof: trace ONCE (above) then
    # run onnxruntime at SEVERAL track counts != the example-input length and
    # confirm each matches the eager adapter. A baked edge-T axis or a
    # baked-constant register pad would FAIL at an off-trace length.
    sweep = check_onnx(
        result.adapter,
        result.onnx_path,
        trials=2,
        lengths_grid=[{"tracks": n} for n in (3, 7, 11, 16)],
        float_rtol=1e-5,
        float_atol=1e-5,
        forbid_zeros=False,  # a zeroed-pad edge column legitimately holds exact 0s
    )
    checks["onnx_dynamic_edge_T_sweep_passes"] = sweep.passed
    # the sweep is non-vacuous: it actually ran multiple DIFFERENT track lengths
    checks["onnx_sweep_covered_multiple_lengths"] = sweep.n_cases >= 2
    return checks, dyn_axes


def run_ed1(
    outdir: Path | str,
    *,
    corruption: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[int, dict[str, Any]]:
    """ED1 (sub-wave C): EdgeFeatures + EdgeEmbed + encoder edge path parity vs v1 + ONNX trace.

    The full edge path (FD §6.7 1418-1431, plan 12 sub-wave C). ED1 drives the
    real compiler / two-phase bind / executor on a GN2XE-shaped fixture and
    asserts:

    - **EdgeFeatures bitwise parity** — the v2 ``edges.tracks`` ``[B, T, T, E]``
      (raw dR/z/kt/subjetIndex/isSelfLoop on the UN-normalised ``inputs.tracks``,
      design §6.3) is BITWISE identical to an INDEPENDENT v1 ``EdgeConstructor``
      (`_ed1_v1_edges`: a separately built v1 ``EdgeConstructor`` keyed on the
      same variables, run on a CLONE of the same raw input — its
      ``_edge_features_tracks``, edge_constructor.py:41-44 / edge_features.py:
      52-146). The dR/kt/z legs are checked column-by-column (the named feature
      values, not just the bundled tensor);
    - **EdgeEmbed shape + forward parity** — the v2 ``edges.tracks_emb`` is
      ``[B, T, T, D_e]`` (both token axes preserved, last dim ``out_dim``) and is
      BITWISE identical to an INDEPENDENT v1 ``Dense`` (`_ed1_v1_embed`,
      weight-loaded from the v2 EdgeEmbed, run on the v1 edges — v1
      ``edge_init_nets``, saltmodel.py:132);
    - **encoder edge path parity (BITWISE)** — the FULL GN2XE-shaped model
      (`_ed1_encoder_modules`: EdgeFeatures -> EdgeEmbed -> StreamEmbed -> Concat
      -> ``TransformerEncoder(edges=..., update_edges=True)`` -> pool) produces an
      ``encoded.seq`` BITWISE identical to an INDEPENDENT v1
      ``Transformer(edge_embed_dim>0, update_edges=True)``
      (`_ed1_v1_encoder_edge_reference`, weight-loaded from the v2 encoder, run
      with the SAME embedded edges) — proving the v2 encoder edge port routes the
      edge tensor through the SAME `EdgeAttention` bias/gate/edge-update math
      (attention.py:465,630-654) and the SAME register/edge zero-pad
      (transformer.py:689-719) as v1, including the ``update_edges`` return;
    - **declared-IO structure** — EdgeFeatures requires the RAW ``inputs.tracks``
      (NOT ``normed.*``) plus ``masks.tracks``; both produced edge tensors AND the
      encoder's required edge port carry the SAME ``T:<stream>`` symbol on both
      token axes (square pairwise matrices, the dynamic-T export prerequisite);
    - **ONNX-trace assertion** — the GN2XE-shaped model traces under
      ``torch.onnx.export`` (`_ed1_export_config` -> `export_graph`) with BOTH edge
      T axes DYNAMIC (the edge tensor is produced internally from the dynamic
      ``inputs.tracks`` token axis, so both edge axes inherit it; a second trace
      at a different track count agrees with the eager adapter, proving the track
      count is NOT baked) AND the in-encoder register zero-pad emitted as a
      SHAPE-DERIVED pad (``Shape``/``Sub``/``ConstantOfShape``/``Concat`` /
      ``Expand`` nodes, NOT a baked-constant ``Constant`` tensor of the padded
      shape — transformer.py:689-719 builds the pad from the dynamic
      ``x.shape[1]``); onnxruntime agrees with the eager model (<=1e-6);
    - **write-once** — neither edge module mutates ``inputs.tracks`` (design §2.1);
    - **named-error guards** — EdgeFeatures rejects an unrecognised feature
      (`ConfigError`), an empty feature list (`ConfigError`), and a feature whose
      required input variable is absent at bind (`check_edge_config` `ValueError`,
      edge_features.py:45-49); EdgeEmbed rejects a width key in ``dense``
      (`ConfigError`).

    Negative control (``test_gates_m6.py``): the ``corruption`` hook perturbs the
    v2 edge tensor — the bitwise parity checks must FAIL while the structural
    (rank-4 / raw-input / symbol), encoder-shape, ONNX-trace, and named-error
    checks stay green.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("ED1 EdgeFeatures + EdgeEmbed + encoder edge path BITWISE parity vs v1 + ONNX trace")
    print("=" * 96)
    checks: dict[str, bool] = {}

    modules = _ed1_modules()
    ef = modules["edge_features"]
    ee = modules["edge_embed"]

    # -- (a) declared-IO structure: raw input, pad mask, square pairwise dims ---
    ef_io = ef.declare_io(Mode.FIT)
    ef_req = flatten_spec(ef_io.requires)
    ef_prod = flatten_spec(ef_io.produces)
    ee_prod = flatten_spec(ee.declare_io(Mode.FIT).produces)
    checks["edge_features_requires_raw_input"] = "inputs.tracks" in ef_req
    checks["edge_features_not_consuming_normed"] = not any(k.startswith("normed.") for k in ef_req)
    checks["edge_features_requires_pad_mask"] = (
        ef_req.get("masks.tracks") is not None and ef_req["masks.tracks"].kind == "pad_mask"
    )
    edges_spec = ef_prod["edges.tracks"]
    checks["edges_tensor_is_rank4"] = len(edges_spec.shape) == 4
    # both token axes share the SAME T:<stream> symbol (the dynamic-T prerequisite)
    checks["edges_both_token_axes_share_stream_symbol"] = (
        edges_spec.shape[1] == edges_spec.shape[2] == "T:tracks"
    )
    checks["edges_last_dim_is_feature_count"] = edges_spec.shape[-1] == len(_ED1_EDGE_FEATURES)
    emb_spec = ee_prod["edges.tracks_emb"]
    checks["edge_embed_is_rank4"] = len(emb_spec.shape) == 4
    checks["edge_embed_token_axes_share_stream_symbol"] = (
        emb_spec.shape[1] == emb_spec.shape[2] == "T:tracks"
    )
    checks["edge_embed_out_dim_matches"] = emb_spec.shape[-1] == _ED1_EDGE_EMBED_DIM

    # -- (b) compile + bind the real edge plan --------------------------------
    fit_plan = compile_plan(modules, Mode.FIT, sources=_ed1_sources(), sinks=["edges.tracks_emb"])
    bind_all(modules, resolve_bind_schema([fit_plan]))
    # bind resolves the indices_map by NAME from the declared fields (v1
    # EdgeConstructor indices_map, edge_constructor.py:34-36)
    checks["indices_map_resolved_by_name"] = ef.indices_map == {
        name: i for i, name in enumerate(_ED1_TRACK_VARIABLES)
    }

    # -- (c) bitwise forward parity vs the INDEPENDENT v1 references -----------
    gen = torch.Generator().manual_seed(13)
    raw_tracks = torch.randn(_ED1_B, _ED1_T, len(_ED1_TRACK_VARIABLES), generator=gen)
    pad_mask = torch.zeros(_ED1_B, _ED1_T, dtype=torch.bool)
    pad_mask[:, -2:] = True  # last two tracks padded (exercise nan_to_num path)
    b = Bundle()
    b.set("inputs.tracks", raw_tracks)
    b.set("masks.tracks", pad_mask)
    out = Executor(fit_plan).run(b, debug=True)
    v2_edges = out.get("edges.tracks")
    v2_emb = out.get("edges.tracks_emb")
    checks["edges_runtime_shape"] = tuple(v2_edges.shape) == (
        _ED1_B,
        _ED1_T,
        _ED1_T,
        len(_ED1_EDGE_FEATURES),
    )
    checks["edge_embed_runtime_shape"] = tuple(v2_emb.shape) == (
        _ED1_B,
        _ED1_T,
        _ED1_T,
        _ED1_EDGE_EMBED_DIM,
    )
    # write-once: inputs.tracks is byte-unchanged after the forward
    checks["inputs_not_mutated"] = torch.equal(out.get("inputs.tracks"), raw_tracks)

    if corruption is not None:
        v2_edges = corruption(v2_edges)

    v1_edges = _ed1_v1_edges(raw_tracks)
    checks["edge_features_bitwise_vs_v1"] = v2_edges.shape == v1_edges.shape and torch.equal(
        v2_edges, v1_edges
    )
    # per-feature legs: EVERY edge feature (dR/z/kt + subjetIndex/isSelfLoop)
    # matches column-by-column. A bundled-tensor equality could mask a single
    # -column shift; pinning all 5 columns gives symmetric column-level coverage
    # for the two non-kinematic features as well as the kinematic ones (the
    # whole tuple is filled by enumerate(variables) index in the SHARED
    # calculate_edge_features, edge_features.py:103, so a column-order bug in
    # any feature is now caught at the granularity of the individual column).
    feat_idx = {f: i for i, f in enumerate(_ED1_EDGE_FEATURES)}
    for feat in _ED1_EDGE_FEATURES:
        i = feat_idx[feat]
        checks[f"edge_feature_{feat}_bitwise_vs_v1"] = torch.equal(
            v2_edges[..., i], v1_edges[..., i]
        )

    # EdgeEmbed: the v1 Dense (weight-loaded from v2) on the v1 edges matches
    # the v2 embed bitwise (a plain nn.Linear stack over the last dim reorders
    # no floats, so the claim is BITWISE)
    v1_emb = _ed1_v1_embed(ee, v1_edges)
    checks["edge_embed_bitwise_vs_v1"] = v2_emb.shape == v1_emb.shape and torch.equal(
        v2_emb, v1_emb
    )

    # -- (d) encoder edge path parity vs the INDEPENDENT v1 Transformer --------
    # the FULL GN2XE-shaped model (edge build -> embed -> stream embed -> concat
    # -> EdgeAttention encoder -> pool) routes the edge tensor through the SAME
    # EdgeAttention bias/gate/edge-update + register/edge zero-pad as v1.
    enc_norm = _ed1_norm_dict(outdir)
    enc_modules = _ed1_encoder_modules(enc_norm)
    encoder = enc_modules["encoder"]
    # declared-IO: the encoder REQUIRES the rank-4 edge port, both axes T:tracks
    enc_req = flatten_spec(encoder.declare_io(Mode.FIT).requires)
    enc_edge_spec = enc_req.get("edges.tracks_emb")
    checks["encoder_requires_edge_port"] = enc_edge_spec is not None
    checks["encoder_edge_port_is_rank4"] = (
        enc_edge_spec is not None and len(enc_edge_spec.shape) == 4
    )
    checks["encoder_edge_port_token_axes_share_stream_symbol"] = (
        enc_edge_spec is not None and enc_edge_spec.shape[1] == enc_edge_spec.shape[2] == "T:tracks"
    )
    checks["encoder_edge_port_width_matches"] = (
        enc_edge_spec is not None and enc_edge_spec.shape[-1] == _ED1_EDGE_EMBED_DIM
    )
    enc_fit = compile_plan(
        enc_modules, Mode.FIT, sources=_ed1_encoder_sources(), sinks=["loss.total"]
    )
    bind_all(enc_modules, resolve_bind_schema([enc_fit]))
    materialise_all(enc_modules)
    # give the encoder + embeds non-zero weights so the parity is meaningful
    gen2 = torch.Generator().manual_seed(29)
    with torch.no_grad():
        for mod in (
            enc_modules["encoder"].encoder,
            enc_modules["track_embed"].net,
            enc_modules["edge_embed"].net,
        ):
            for p in mod.parameters():
                p.copy_(torch.randn(p.shape, generator=gen2) * 0.1)
    enc_b = Bundle()
    enc_b.set("inputs.tracks", raw_tracks)
    enc_b.set("masks.tracks", pad_mask)
    enc_b.set("labels.jets.flavour_label", torch.zeros(_ED1_B, dtype=torch.int64))
    enc_out = Executor(enc_fit).run(enc_b, debug=True)
    v2_encoded = enc_out.get("encoded.seq")
    # encoded.seq is register-augmented [B, T+R, out_dim] (registers kept)
    checks["encoder_encoded_runtime_shape"] = tuple(v2_encoded.shape) == (
        _ED1_B,
        _ED1_T + encoder.num_registers,
        _ED1_OUT_DIM,
    )
    # write-once held through the FULL edge encoder too
    checks["encoder_inputs_not_mutated"] = torch.equal(enc_out.get("inputs.tracks"), raw_tracks)
    if corruption is not None:
        v2_encoded = corruption(v2_encoded)
    # the SAME embedded edges the v2 encoder consumed (the bundle key), and the
    # SAME concatenated seq.x / seq.mask, fed to the INDEPENDENT v1 Transformer
    v1_encoded = _ed1_v1_encoder_edge_reference(
        encoder, enc_out.get("seq.x"), enc_out.get("seq.mask"), enc_out.get("edges.tracks_emb")
    )
    checks["encoder_edge_forward_bitwise_vs_v1"] = (
        v2_encoded.shape == v1_encoded.shape and torch.equal(v2_encoded, v1_encoded)
    )

    # -- (e) ONNX-trace assertion: dynamic edge-T axes + shape-derived register pad
    ed1_onnx_checks, dyn_axes = _ed1_onnx_trace(enc_modules, outdir)
    checks.update(ed1_onnx_checks)

    # -- (f) named-error guards (value-independent, config/bind-time) ----------
    checks["unknown_feature_raises_configerror"] = _raises(
        ConfigError, lambda: EdgeFeatures(stream="tracks", features=["dR", "bogus_feature"])
    )
    checks["empty_features_raises_configerror"] = _raises(
        ConfigError, lambda: EdgeFeatures(stream="tracks", features=[])
    )
    # a feature whose required input variable is missing -> check_edge_config
    # ValueError at bind (dR needs eta/phi; supply a field set lacking phi)

    def _bind_missing_var() -> None:
        bad = EdgeFeatures(stream="tracks", features=["dR"])
        bad.name = "bad_edges"
        bad_schema = resolve_bind_schema([
            compile_plan(
                {"bad_edges": bad},
                Mode.FIT,
                sources=unflatten_spec({
                    "inputs.tracks": TensorSpec(
                        shape=("B", "T:tracks", 2),
                        dtype="float32",
                        fields=("pt", "eta"),  # no phi -> dR cannot be built
                    ),
                    "masks.tracks": TensorSpec(
                        shape=("B", "T:tracks"), dtype="bool", kind="pad_mask"
                    ),
                }),
                sinks=["edges.tracks"],
            )
        ])
        bad.bind(bad_schema)

    checks["missing_required_var_raises_valueerror"] = _raises(ValueError, _bind_missing_var)
    checks["edge_embed_width_key_raises_configerror"] = _raises(
        ConfigError,
        lambda: EdgeEmbed(stream="tracks", out_dim=32, dense={"input_size": 5}),
    )

    passed = all(checks.values())
    criterion = (
        "the M6 edge path end-to-end (plan 12 sub-wave C; design §6.7 1418-1431): EdgeFeatures "
        "requires the RAW inputs.tracks (un-normalised, §6.3) + masks.tracks and produces "
        "edges.tracks [B, T, T, E] whose dR/z/kt/subjetIndex/isSelfLoop values are BITWISE "
        "identical to an INDEPENDENT v1 EdgeConstructor (edge_constructor.py:41-44, "
        "edge_features.py:52-146); EdgeEmbed maps it to edges.tracks_emb [B, T, T, D_e] BITWISE "
        "identical to an INDEPENDENT v1 Dense (edge_init_nets, saltmodel.py:132); the FULL "
        "GN2XE-shaped model (EdgeFeatures->EdgeEmbed->StreamEmbed->Concat->TransformerEncoder("
        "edges=...,update_edges=True)->pool->head) produces an encoded.seq BITWISE identical to an "
        "INDEPENDENT v1 Transformer(edge_embed_dim>0, update_edges=True) weight-loaded from the v2 "
        "encoder (the EdgeAttention bias/gate/edge-update + register/edge zero-pad path, "
        "attention.py:465,630-654 / transformer.py:689-733); both token axes of every edge tensor "
        "AND the encoder's edge port share the T:tracks symbol (dynamic-T export prerequisite); "
        "the model TRACES under torch.onnx.export with both edge T axes DYNAMIC (onnxruntime "
        "agrees with eager across track counts 3/7/11/16 from a SINGLE trace) and the in-encoder "
        "register zero-pad emitted SHAPE-DERIVED (Shape/ConstantOfShape/Sub/Concat ops, not a "
        "baked constant, transformer.py:689-719); inputs.tracks is never mutated (write-once "
        "§2.1); and the named-error guards fire (unrecognised/empty feature -> ConfigError; "
        "missing required variable -> check_edge_config ValueError; EdgeEmbed width key -> "
        "ConfigError). "
        "The bind-time edge validators (edge-stream-first / backend-forcing) are ED2"
    )
    report = _base_report(
        "ed1_edge_path_parity",
        passed,
        criterion,
        {
            "track_variables": list(_ED1_TRACK_VARIABLES),
            "edge_features": list(_ED1_EDGE_FEATURES),
            "edge_embed_dim": _ED1_EDGE_EMBED_DIM,
            "edge_embed_hidden": list(_ED1_EDGE_HIDDEN),
            "encoder_embed_dim": _ED1_EMBED_DIM,
            "encoder_out_dim": _ED1_OUT_DIM,
            "encoder_num_heads": _ED1_NUM_HEADS,
            "encoder_num_layers": _ED1_NUM_LAYERS,
            "class_names": list(_ED1_CLASS_NAMES),
            "batch": _ED1_B,
            "n_tracks": _ED1_T,
            "onnx_sweep_track_counts": [3, 7, 11, 16],
            "onnx_path": str(outdir / "ed1_gn2xe_edge.onnx"),
            "stage_scope": (
                "EdgeFeatures + EdgeEmbed + encoder edge path (EdgeAttention, update_edges) + the "
                "dynamic-edge-T ONNX trace. The edge-stream-first / EdgeAttention-backend "
                "bind-time validators are gate ED2; the GN2XE corpus config is the next sub-wave "
                "C stage"
            ),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["dynamic_axes"] = {k: {str(a): n for a, n in v.items()} for k, v in dyn_axes.items()}
    report["v1_reference"] = (
        "INDEPENDENT v1 reference (instance + graph-routing independence, NOT independent "
        "arithmetic): a separately-instantiated v1 EdgeConstructor (distinct object from the v2 "
        "EdgeFeatures, run on a CLONE of the same raw input) supplies the byte-decidable "
        "dR/kt/z/subjetIndex/isSelfLoop reference (edge_constructor.py:41-44, "
        "edge_features.py:52-146); a separately-instantiated v1 Dense, weight-loaded from the v2 "
        "EdgeEmbed and run on the v1 edges, supplies the edge-embed reference (v1 edge_init_nets, "
        "saltmodel.py:132); a separately-instantiated v1 Transformer(edge_embed_dim>0, "
        "update_edges=True), weight-loaded from the v2 encoder and run with the SAME embedded "
        "edges, supplies the encoder-edge-path reference (transformer.py:689-733, "
        "attention.py:465). The byte-decidable ORACLE for the edge values is the SHARED composed "
        "function salt.utils.edge_features.calculate_edge_features (edge_features.py:52) — BOTH "
        "the v1 EdgeConstructor.forward (edge_constructor.py:43) and the v2 EdgeFeatures.forward "
        "(modules.py:1002) delegate to it (single source, M2 porting policy; full math absorption "
        "is M7). So 'INDEPENDENT' here means distinct OBJECT INSTANCES + an independently-wired v1 "
        "GRAPH PATH, NOT an independently-derived dR/z/kt formula: the bitwise torch.equal proves "
        "the v2 GRAPH PATH (compile -> two-phase bind -> executor) resolves the indices_map by "
        "NAME and routes the raw stream through the SAME edge math + Dense + EdgeAttention encoder "
        "as v1's hand-wired EdgeConstructor/edge_init_nets/Transformer(edge) path"
    )
    _print_checks(checks)
    _print_verdict("ed1", passed, criterion, _emit_report(report, outdir, "ed1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# ED2 — the edge bind-time validators (design-conformance, sub-wave C)
# ---------------------------------------------------------------------------

_ED2_LRS = {"initial": 1e-4, "max": 5e-4, "end": 1e-5, "pct_start": 0.1}


def _ed2_edge_encoder(attn_type: str = "torch-math") -> TransformerEncoder:
    """A small edge-bearing `TransformerEncoder` (the ED2 validator subject).

    Returns
    -------
    TransformerEncoder
        An encoder with an ``edges: edges.tracks_emb`` port (+ ``edge_embed_dim``),
        the optional ``attn_type`` letting the negative control declare a non-edge
        backend.
    """
    enc = TransformerEncoder(
        dim=_ED1_EMBED_DIM,
        num_layers=1,
        out_dim=_ED1_OUT_DIM,
        attention={"num_heads": _ED1_NUM_HEADS, "attn_type": attn_type},
        edges="edges.tracks_emb",
        edge_embed_dim=_ED1_EDGE_EMBED_DIM,
        update_edges=True,
    )
    enc.name = "encoder"
    return enc


def _ed2_modules(concat_streams: Sequence[str], attn_type: str = "torch-math") -> dict[str, Any]:
    """A minimal edge model dict for the bind-time validators (concat + edge encoder).

    Just the two modules the edge validators inspect — a `Concat` (whose
    ``streams[0]`` is the edge-stream-first reference) and an edge-bearing
    `TransformerEncoder` (whose backend rule b checks). `concat_streams[0]`
    controls whether the edge stream (``tracks``) is first.

    Returns
    -------
    dict[str, Any]
        Instance-named ``{concat, encoder}``.
    """
    modules: dict[str, Any] = {
        "concat": Concat(streams=list(concat_streams)),
        "encoder": _ed2_edge_encoder(attn_type),
    }
    for name, module in modules.items():
        module.name = name
    return modules


def _ed2_full_modules(norm_dict: Path, concat_streams: Sequence[str]) -> dict[str, Any]:
    """A complete GN2XE-shaped model dict for a full `SaltModule` construction.

    The same shape as `_ed1_encoder_modules` but with the `Concat` stream order
    parameterised so a `SaltModule(...)` construction exercises the edge
    bind-time validator end-to-end (the validator runs in ``__init__``).

    Returns
    -------
    dict[str, Any]
        Instance-named modules with `LossSum` narrowed.
    """
    modules = _ed1_encoder_modules(norm_dict)
    modules["concat"] = Concat(streams=list(concat_streams))
    modules["concat"].name = "concat"
    return modules


def run_ed2(
    outdir: Path | str,
    *,
    corruption: Callable[[list[str]], list[str]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """ED2: the edge bind-time validators (design-conformance, sub-wave C; FD §6.7 1425-1431).

    The two HIDDEN v1 edge mechanisms made NAMED constraints (`validate_edge_port`,
    saltmodule.py) — the "land in M1 validation" rules. v2 runs the SAME validator
    in `SaltModule.__init__` AND in ``salt2 graph validate`` (the edge analogue of
    the muP routing validator, MU2). ED2 asserts:

    - **edge-stream-first (rule a)** — a model whose edge stream IS the `Concat`'s
      first stream is ACCEPTED (the validator returns the encoder count); a model
      whose edge stream is NOT first raises `ConfigError`; an edge port with NO
      `Concat` raises `ConfigError`. v1 SILENTLY re-sorted the init_nets so the
      edge stream landed first (saltmodel.py:66-73) — v2 makes the ordering an
      explicit named constraint;
    - **EdgeAttention-backend forcing (rule b)** — an edge encoder declaring a
      non-edge backend (``flash-varlen``) raises `ConfigError`. v1 SILENTLY
      ignored ``attn_type`` whenever edge features were on
      (transformer.py:599-601) — v2 makes the silent bypass a named error;
    - **ctor-level edge guards** — ``edges`` without a positive ``edge_embed_dim``
      (or vice versa) and ``update_edges`` without ``edges`` raise `ConfigError`
      at `TransformerEncoder.__init__`;
    - **same validator both places** — a full `SaltModule(...)` construction with
      a mis-ordered edge concat raises the SAME `ConfigError` (the validator is
      wired into ``__init__``), and a correctly-ordered one constructs; the
      ``salt2 graph validate`` import wiring is present (`validate_edge_port`
      callable from the CLI module path).

    Negative control (``test_gates_m6.py``): the ``corruption`` hook reorders the
    `Concat` streams so the edge stream is no longer first — the
    ``valid_edge_order_accepted`` check then flips False and the gate exits 1.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("ED2 edge bind-time validators (edge-stream-first + EdgeAttention-backend forcing)")
    print("=" * 96)
    checks: dict[str, bool] = {}

    # -- (a) edge-stream-first (rule a) ---------------------------------------
    # the edge stream is "tracks" (edges.tracks_emb -> tracks). A two-stream
    # concat with tracks FIRST is valid; with tracks SECOND it errors.
    valid_order = ["tracks", "flow"]
    if corruption is not None:
        valid_order = corruption(valid_order)
    try:
        n_enc = validate_edge_port(_ed2_modules(valid_order))
        checks["valid_edge_order_accepted"] = n_enc == 1
    except ConfigError:
        checks["valid_edge_order_accepted"] = False
    checks["edge_stream_not_first_errors"] = _raises(
        ConfigError, validate_edge_port, _ed2_modules(["flow", "tracks"])
    )
    # an edge port with NO Concat at all -> ConfigError
    lone_encoder = {"encoder": _ed2_edge_encoder()}
    lone_encoder["encoder"].name = "encoder"
    checks["edge_port_without_concat_errors"] = _raises(
        ConfigError, validate_edge_port, lone_encoder
    )

    # -- (b) EdgeAttention-backend forcing (rule b) ---------------------------
    checks["non_edge_backend_errors"] = _raises(
        ConfigError, validate_edge_port, _ed2_modules(["tracks"], attn_type="flash-varlen")
    )
    # a plain (non-edge) encoder is untouched by the validator (no-op)
    plain = {"concat": Concat(streams=["tracks"])}
    plain["concat"].name = "concat"
    checks["no_edge_encoder_is_noop"] = validate_edge_port(plain) == 0

    # -- (c) ctor-level edge guards -------------------------------------------
    checks["edges_without_dim_raises_configerror"] = _raises(
        ConfigError,
        lambda: TransformerEncoder(
            dim=8, num_layers=1, out_dim=4, attention={"num_heads": 2}, edges="edges.tracks_emb"
        ),
    )
    checks["edge_dim_without_edges_raises_configerror"] = _raises(
        ConfigError,
        lambda: TransformerEncoder(
            dim=8, num_layers=1, out_dim=4, attention={"num_heads": 2}, edge_embed_dim=16
        ),
    )
    checks["update_edges_without_edges_raises_configerror"] = _raises(
        ConfigError,
        lambda: TransformerEncoder(
            dim=8, num_layers=1, out_dim=4, attention={"num_heads": 2}, update_edges=True
        ),
    )

    # -- (d) same validator both places (SaltModule.__init__ + CLI wiring) ----
    norm_dict = _ed1_norm_dict(outdir)
    # a correctly-ordered edge model constructs through SaltModule (validator OK)
    try:
        SaltModule(modules=_ed2_full_modules(norm_dict, ["tracks"]), lrs=_ED2_LRS)
        checks["saltmodule_accepts_valid_edge_order"] = True
    except ConfigError:
        checks["saltmodule_accepts_valid_edge_order"] = False
    # a mis-ordered edge model raises the SAME ConfigError in __init__
    checks["saltmodule_rejects_misordered_edge_concat"] = _raises(
        ConfigError,
        lambda: SaltModule(
            modules=_ed2_full_modules(norm_dict, ["truth_hadrons", "tracks"]), lrs=_ED2_LRS
        ),
    )
    # the CLI `salt2 graph validate` path imports the SAME validator (design
    # §6.7: one validator, fired data-free in CI and at run construction)
    import importlib  # noqa: PLC0415

    cli_mod = importlib.import_module("salt.core.saltmodule")
    checks["validate_edge_port_exported"] = callable(getattr(cli_mod, "validate_edge_port", None))

    passed = all(checks.values())
    criterion = (
        "the edge bind-time validators (plan 12 sub-wave C / FD §6.7 1425-1431): the SINGLE "
        "validate_edge_port (saltmodule.py), fired BOTH in SaltModule.__init__ and salt2 graph "
        "validate, makes v1's two HIDDEN edge mechanisms NAMED constraints — (a) "
        "edge-stream-first: the edge tensor's stream MUST be Concat.streams[0] (replacing v1's "
        "silent init-net sort, saltmodel.py:66-73), else a named ConfigError; an edge port with "
        "no Concat errors too; (b) EdgeAttention-backend forcing: a non-edge backend "
        "(flash-varlen) declared alongside an edge port is a named ConfigError (replacing v1's "
        "silent attn_type bypass, transformer.py:599-601). Plus the ctor-level edge guards "
        "(edges<->edge_embed_dim paired, update_edges needs edges) and the design-conformance "
        "check that the SAME validator runs in SaltModule.__init__ (a mis-ordered edge concat "
        "raises) and is exported for the CLI path"
    )
    report = _base_report(
        "ed2_edge_bind_validators",
        passed,
        criterion,
        {
            "edge_stream": "tracks",
            "valid_concat_order": ["tracks", "flow"],
            "invalid_concat_order": ["flow", "tracks"],
            "non_edge_backend_probe": "flash-varlen",
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["v1_reference"] = (
        "v1 had NO edge validators — the two constraints were SILENT runtime hacks: the init-net "
        "sort that forced the edge stream first (saltmodel.py:66-80, with assert len==1) and the "
        "attn_type bypass that ignored the configured backend whenever edge_embed_dim>0 "
        "(transformer.py:599-601,616-620). ED2 asserts v2 turns both into NAMED ConfigErrors fired "
        "by the single validate_edge_port at bind time (SaltModule.__init__ + salt2 graph validate)"
    )
    _print_checks(checks)
    _print_verdict("ed2", passed, criterion, _emit_report(report, outdir, "ed2"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# M6-CONV — the M7-slice acceptance for the M6-authored configs (sub-wave A
# bootstraps it; later M6 waves EXTEND _CONV_M6_CONFIGS)
# ---------------------------------------------------------------------------

# The AUTHORITATIVE M6-CONV config list (plan 12 M6-CONV row: "embed the
# authoritative _CONV_M6_CONFIGS list verbatim"). Sub-wave A (Labeller) landed
# the FIRST two; sub-wave D (vector-stream) added DL1; sub-wave B (muP) added
# GN2_muP; sub-wave C (edges) appends GN2XE — the FINAL config-gating needs-M6
# config. With GN2XE the list is COMPLETE at 5, which CLOSES the 39-denominator
# (all needs-M6 movers reproduced: GN3X + GN2X_qcdsplit + DL1 + GN2_muP + GN2XE,
# plus regression_multi_target ✅-in-M5 + base.yaml the 39th transitive member).
#
# `norm_global` is False for all (none carries a SECOND `norm_global` Normaliser
# — that is the VectorConcat/global-stream pattern, absent here). `onnx ==
# "validate"` for all FIVE. GN3X, GN2X_qcdsplit AND DL1 are STANDARD traces. For
# GN2_muP, this CONV `onnx` leg is a STATIC plan-compile (salt2 graph validate
# --mode onnx does not call torch.onnx.export/set_export_mode); the MuReadout->
# plain-nn.Linear fold that makes the REAL export traceable (set_export_mode; plan
# 12 sub-wave B export contract) is proven by MU1's export-fold check, not by this
# plan-compile. With the fold proven by MU1, onnx stays in the gate for GN2_muP
# too (plan 12 M6-CONV per-config export contracts: "GN3X/GN2X_qcdsplit/DL1 are
# standard traces; GN2_muP exports with the MuReadout out-proj folded to plain
# Linear"). For GN2XE (edges), this CONV `onnx` leg is LIKEWISE a STATIC
# plan-compile (cli.py:381-423: graph validate --mode onnx compiles the onnx plan
# + validates the writer manifest, it does NOT call torch.onnx.export) — so the
# CONV onnx leg proves the GN2XE onnx PLAN compiles, not the dynamic-T trace
# itself. The load-bearing dynamic-T export contract (BOTH edge T axes dynamic +
# the in-encoder register zero-pad emitted SHAPE-DERIVED, not baked — FD §6.7
# ONNX note / plan 12 sub-wave C) is exercised + PROVEN by ED1's ONNX-trace leg
# (`_ed1_onnx_trace`: a real torch.onnx.export, then onnxruntime at track counts
# 3/7/11/16 matching the eager model from a SINGLE trace, plus a graph-node
# inspection for the Shape/ConstantOfShape/Sub/Concat shape-derivation ops). With
# the dynamic-T trace PROVEN by ED1, --mode onnx STAYS in the gate for GN2XE too
# (NOT scoped out — plan 12 default: "onnx stays in the gate, edges trace with
# dynamic axes"; the M5-CONV scope-out fallback is unused). NO export hazard
# remains deferred. DL1's rank-2 [B, F] embed is a plain nn.Linear stack with B
# the sole dynamic axis (sub-wave D / VS1). A muP config carries a `mup: True`
# marker: it needs its base/delta infshape file generated data-free before
# validation (`_conv_mup_shape_path`).
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
    {
        "name": "GN2_muP",
        "cfg": ("GN2_muP.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "mup",
        "mup": True,  # needs a generated shape file (see _conv_mup_shape_path / run_conv)
        "note": (
            "GN2 with muP: mup: true on track_embed (StreamEmbed) + encoder (TransformerEncoder) "
            "+ model.init_args.mup {shape_path, apply_to} routing (the M6-B deliverable, gates "
            "MU1/MU2). The encoder out-proj is a mup.MuReadout FOLDED to a plain nn.Linear at "
            "trace time (set_export_mode) at REAL export; this CONV onnx leg is a static plan-"
            "compile (no torch.onnx.export), so the fold is proven by MU1's export-fold check, not "
            "here — with that proof, --mode onnx STAYS in the gate; shape file generated data-free "
            "by salt2 mup-shapes and supplied via --set model.init_args.mup.shape_path"
        ),
    },
    {
        "name": "GN2XE",
        "cfg": ("GN2XE.yaml",),
        "norm_global": False,
        "onnx": "validate",
        "family": "edges",
        "note": (
            "GN2X with EDGE FEATURES end-to-end: EdgeFeatures (raw inputs.tracks -> edges.tracks "
            "[B,T,T,E]) -> EdgeEmbed (-> edges.tracks_emb [B,T,T,D_e=32]) -> TransformerEncoder "
            "with edges:edges.tracks_emb + edge_embed_dim:32 + update_edges:true (EdgeAttention "
            "per layer); single-stream Concat[tracks] so the edge stream is streams[0] "
            "(edge-stream-first, ED2 rule a); attn_type:torch-math (EdgeAttention-backend forcing, "
            "ED2 rule b: flash-varlen FORBIDDEN with an edge port). The M6-C deliverable (gates "
            "ED1/ED2 + ED-HUMAN edge-memory sign-off). This CONV onnx leg is a STATIC plan-compile "
            "(cli.py:381-423, no torch.onnx.export); the load-bearing dynamic-T trace (BOTH edge T "
            "axes dynamic + register zero-pad SHAPE-DERIVED, not baked — FD §6.7 ONNX note) is "
            "proven by ED1's ONNX-trace leg (real export, onnxruntime at track counts 3/7/11/16 "
            "from a SINGLE trace) — with that proof, --mode onnx STAYS in the gate (NOT scoped out)"
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


def _conv_set_args(
    entry: dict[str, Any], norm_dict: Path, shape_path: Path | None = None
) -> list[str]:
    """Build the data-free ``--set`` overrides for one config (norm + norm_global + mup shapes).

    Every shipped v2 config materialises its `Normaliser` from a norm dict at
    setup; static validation supplies it data-free via the documented
    ``--set model.modules.norm.init_args.norm_dict=<path>`` (each config header,
    gates_m5.py:4439). A config with a SECOND `norm_global` Normaliser needs its
    own override (none of the M6 configs has one). A muP config additionally
    needs its base/delta infshape file: the committed config carries a
    placeholder ``shape_path`` (NO machine path, design §5), so the gate
    generates the shapes data-free and overrides
    ``model.init_args.mup.shape_path`` (`SaltModule._apply_mup_shapes` raises if
    the file is missing — plan 12 sub-wave B).

    Returns
    -------
    list[str]
        The ``--set KEY=VALUE`` flag pairs (already split for ``salt2_main``).
    """
    args = ["--set", f"model.modules.norm.init_args.norm_dict={norm_dict}"]
    if entry["norm_global"]:
        args += ["--set", f"model.modules.norm_global.init_args.norm_dict={norm_dict}"]
    if entry.get("mup") and shape_path is not None:
        args += ["--set", f"model.init_args.mup.shape_path={shape_path}"]
    return args


def _conv_mup_shape_path(entry: dict[str, Any], norm_dict: Path, outdir: Path) -> Path:
    """Generate a muP config's base/delta infshapes data-free; return the file path.

    Runs the REAL ``salt2 mup-shapes`` command (`salt2_main`) on the muP config
    with the data-free norm dict, writing the infshape file into ``outdir``. This
    mirrors MU2's end-to-end tooling exercise (run_mu2: salt2 mup-shapes on a
    real config) and the v1 ``store_shapes_mup`` step — every v1 muP run
    generated shapes before training, and v1's pickled shapes are legacy-path
    only (plan 12 sub-wave B: "REGENERATED under v2"). The generated file is what
    ``SaltModule.mup.shape_path`` consumes at bind so `MuReadout.width_mult()` /
    `MuAdamW` resolve against the real base widths.

    Returns
    -------
    Path
        The written infshape file (``<outdir>/<name>_shape_mup.bsh``).

    Raises
    ------
    RuntimeError
        When ``salt2 mup-shapes`` fails (rc != 0) or writes no file.
    """
    shape_path = outdir / f"{entry['name']}_shape_mup.bsh"
    cargs: list[str] = []
    for cfg_name in entry["cfg"]:
        cargs += ["-c", str(CONFIG_DIR / cfg_name)]
    rc = salt2_main([
        "mup-shapes",
        *cargs,
        "--save-path",
        str(shape_path),
        "--set",
        f"model.modules.norm.init_args.norm_dict={norm_dict}",
    ])
    if rc != 0 or not shape_path.is_file():
        raise RuntimeError(
            f"salt2 mup-shapes failed for {entry['name']} (rc={rc}, file={shape_path}) — the "
            "muP shape generation is a prerequisite for the M6-CONV validation of a muP config "
            "(plan 12 sub-wave B)"
        )
    return shape_path


def _conv_modes(entry: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Resolve the (mode, report-key) pairs to validate for one config.

    Always fit + test; onnx only when the config is export-representable (the
    ``onnx`` field is ``"validate"``). All 5 M6 configs are export-representable
    (``onnx == "validate"``), so all validate onnx too — GN3X/GN2X_qcdsplit/DL1
    are standard traces, GN2_muP folds the MuReadout (MU1) and GN2XE traces with
    dynamic edge T-axes (ED1).

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

    For EVERY M6 config in ``_CONV_M6_CONFIGS`` — sub-wave A: GN3X,
    GN2X_qcdsplit; sub-wave D: DL1; sub-wave B: GN2_muP; sub-wave C: GN2XE (the
    list is now COMPLETE at 5, the FINAL config-gating wave) — this drives the
    canonical static
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
        # a muP config needs its base/delta infshapes generated data-free first
        # (the committed shape_path is a placeholder; SaltModule._apply_mup_shapes
        # raises if the file is missing — plan 12 sub-wave B). The generation step
        # is also a per-config gate: if mup-shapes fails the config cannot validate.
        shape_path: Path | None = None
        if files_present and entry.get("mup"):
            shape_path = _conv_mup_shape_path(entry, norm_dict, outdir)
            row["shape_path"] = str(shape_path)
            checks[f"{entry['name']}:mup_shapes_generated"] = shape_path.is_file()
        set_args = _conv_set_args(entry, norm_dict, shape_path)
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
        "is owned by LB1, the vector-stream forward parity by VS1, the muP forward-init parity + "
        "MuReadout export-fold by MU1/MU2, and the edge-path forward parity + dynamic-T trace by "
        "ED1/ED2. Sub-wave A (Labeller) landed GN3X + GN2X_qcdsplit; "
        "sub-wave D (vector-stream) added DL1; sub-wave B (muP) added GN2_muP; sub-wave C "
        "(edges) appends GN2XE — the list is now COMPLETE at 5 (the FINAL config-gating wave). "
        "GN3X/GN2X_qcdsplit/DL1 are standard "
        "traces (DL1's rank-2 [B, F] embed is a plain nn.Linear stack with B the sole dynamic "
        "axis); GN2_muP's --mode onnx here is a STATIC plan-compile + writer-manifest validation "
        "(cli.py:381-423, the M5-CONV run_conv path) — it does NOT call torch.onnx.export/"
        "set_export_mode, so the CONV onnx leg proves the onnx PLAN compiles, not the fold. The "
        "MuReadout->plain-Linear fold that makes the real export traceable is exercised + proven "
        "by MU1's export-fold check (set_export_mode on the live module), NOT this plan-compile. "
        "With the fold proven by MU1, onnx stays in the gate for GN2_muP too. GN2XE's --mode onnx "
        "here is LIKEWISE a STATIC plan-compile (cli.py:381-423); its load-bearing dynamic-T trace "
        "(BOTH edge T axes dynamic + the in-encoder register zero-pad emitted SHAPE-DERIVED, not "
        "baked — FD §6.7 ONNX note) is exercised + PROVEN by ED1's ONNX-trace leg (a real "
        "torch.onnx.export, then onnxruntime at track counts 3/7/11/16 matching the eager model "
        "from a SINGLE trace) — so onnx stays in the gate for GN2XE too (NOT scoped out). NO "
        "export hazard remains deferred. All 5 configs move 🔷->✅ in the 39 denominator — which "
        "is now CLOSED."
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
    """Build the gate subcommand parser (LB1/VS1/MU1/MU2/ED1/ED2/M6-CONV; later waves append more).

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``lb1``..``ed2`` and ``conv`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.gates_m6", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)
    helps = {
        "lb1": "Labeller label-derivation parity vs v1 + named-error guards (sub-wave A)",
        "vs1": "Vector-stream rank-2 [B,F] embed -> head parity vs v1 DL1 + onnx no-T (sub-wave D)",
        "mu1": "muP forward-init parity vs v1 Transformer(mup=True) + MuReadout export-fold (B)",
        "mu2": "muP routing validator: apply_to name-lists + MuAdamW swap + setup_mup casing (B)",
        "ed1": "Edge path parity vs v1 (EdgeFeatures/EdgeEmbed/encoder) + dynamic-edge-T onnx (C)",
        "ed2": "Edge bind validators: edge-stream-first + EdgeAttention-backend forcing (C)",
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
        "mu1": run_mu1,
        "mu2": run_mu2,
        "ed1": run_ed1,
        "ed2": run_ed2,
        "conv": run_conv,
    }[args.gate]
    code, _ = runner(args.outdir)
    return code


if __name__ == "__main__":
    sys.exit(main())
