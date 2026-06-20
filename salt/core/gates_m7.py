"""M7 W1 gates harness — CV1 (converter acceptance) + CV2 (hard-error path) + CVF.

Standalone gates, each a subcommand of ``python -m salt.core.gates_m7``, each
writing a machine-readable ``<gate>_report.json`` into ``--outdir`` and a
human-readable table to stdout, exiting non-zero on failure (the
gates_m2/m3/m4/m5/m6 envelope, mirrored EXACTLY: each gate is a
``run_<gate>(outdir, *, corruption=None) -> (rc, report)`` argparse subcommand
building a ``checks: dict[str, bool]`` with ``passed = all(...)``, a
``<gate>_report.json`` + stdout verdict table, a non-zero exit on failure, and a
python-only ``corruption`` negative-control hook proving the gate has teeth —
never exposed on the CLI). NO machine data path lives in this file (design §5
placeholder policy): every gate generates its own data-free norm dicts / class
overrides into ``--outdir`` and drives the REAL ``salt2 convert-config`` +
``salt2 graph validate`` subcommands.

This is the FIRST M7 wave (plan 13 W1, the converter) — it does NOT change any
v2 module behaviour; it gates the v1->v2 ``salt2 convert-config`` translator
(``salt.core.convert``, landed W1a) against the 39-config denominator.

Gate criteria (each justified in its ``run_*`` docstring vs the plan/matrix):

- **CV1 — converter acceptance (the FULL 39-config matrix §4 denominator)** (plan
  13 CV1 row; matrix §4 + §4.1, the 4 conditions). CV1 covers the FULL plan 13 /
  matrix §4 non-dropped denominator of 39 configs = the 29 needs-M5 (matrix §3 M5
  / gates_m5 ``_CONV_CONFIGS``, 24 incl. conditional-drop Dipz) + needs-M6
  (gates_m6 ``_CONV_M6_CONFIGS``, 5) configs that have a hand-written v2 fixture,
  PLUS the 10 reproducible-today (✅) configs (matrix §1; gn2v1-opendata, GN2,
  GN2_extended, GN2_charded_neutral_loose_aux, GN2X, GN2XTau, flow, tutorial,
  TPLTmu, PLITel) that have NO M5/M6 fixture but still need the converter for the
  mechanical conversion details (matrix §5.3). The AUTHORITATIVE list is embedded
  VERBATIM in ``_CV1_CONFIGS`` (the ✅ rows carry ``today: True``). CV1 runs
  ``salt2 convert-config`` on the v1 SOURCE stack (``salt/configs/...``) and
  asserts the converted v2 config (a) converts with NO hard-error AND validates +
  plan-compiles fit/test/onnx
  (``salt2 graph validate`` rc == 0 in every applicable mode — the convert+
  validate+plan-compile criterion; data-free, so NOT ``--strict``, the M5/M6-CONV
  precedent: ``--strict`` promotes inherent data-free warnings (no schema,
  warning-level deadcode, missing-input-type preflight) to errors, orthogonal to
  convertibility — see ``_NO_STRICT_RATIONALE``) AND (b) is SEMANTICALLY
  EQUIVALENT to the hand-written fixture: the converted single config and the
  hand-written fixture STACK compile to the SAME plan in every shared mode
  (compared CLASS-CANONICAL — instance names normalised to module classes,
  because the converter derives instance names from v1 task names while the
  fixtures choose short hand-written names; the comparison is over module CLASS
  set, edges (producer-class, key-namespace, consumer-class), per-mode sinks, the
  eval (TEST) column manifest and the ONNX output manifest — the PLANS, not raw
  YAML bytes). Class names are baked at conversion from the fixture's own
  declared ``class_names`` (matrix §5.3 ledger #7: ``class_names`` baked as
  literals), supplied via ``--class-names HEAD=...`` for the ``use_class_dict``
  heads that relied on v1 h5-attr autodiscovery (so the converter never reads a
  data file). Dipz (matrix §1 conditional drop, fired post-M5 once hitz
  validated) is INCLUDED here — the converter still RUNS on it; it is annotated
  ``conditional_drop`` in the report (the actual file removal rides the later
  deletion release, plan 13 Q4). The 10 reproducible-today (✅) configs have NO
  fixture (condition 4 is N/A): they are gated on conditions 1-3 only
  (convert+validate+plan-compile fit/test/onnx), with their global flavour
  ``class_names`` supplied via ``--class-names`` per cardinality (TPLTmu/PLITel
  declare them inline). **Post-Wave-F2 honesty**: 6 fixtures that were once
  curated SUBSETS of the v1 source (regression, regression_gaussian,
  regression_weighted, nan_regression, GN3V01, GN2XE) were RESTORED in Wave F2 to
  FULL v1 fidelity — for these CV1 asserts they REPRODUCE THE FULL V1 CAPABILITY:
  ``conv_task_count == v1_task_count == fixture_task_count`` AND the converter is
  STRUCTURALLY EQUIVALENT to the fixture (the module-CLASS multiset matches in
  every shared mode). The residual converter-vs-fixture differences are
  topology-neutral cosmetics (verbose-v1 vs short head names → class-canonical
  sink keys; the embed.tracks==seq.x edge for a single-stream Concat; the
  fixture's hand-set PadMaskWriter mask column), so a byte-exact ``fixture_match``
  is recorded but NOT required; a DROPPED module would break the structural
  check. Exactly 2 INTENTIONAL KEEPS remain (``event_classifier`` — the converter
  wires FIT/VAL metric sinks the fixture omits by design; ``regression_multi_target``
  — the fixture renames the v1 task and is the curated F1a MultiTarget artifact):
  their hand-written fixture is the faithful v2 artifact and NOT a 1:1 converter
  reproduction, so the exact-plan match cannot hold; CV1 records ``fixture_match``
  + ``converter_faithful_to_v1``, ASSERTS the converter reproduces the FULL v1
  task count (``conv_task_count == v1_task_count >= fixture_task_count``), keeps
  them fixture-DIVERGENT, and requires a ``keep_reason`` naming WHY. BOTH the keep
  set and the restored set are PINNED (a fixture that SILENTLY started matching,
  or regressed back to a subset, changes the gate's accounting), so every
  divergence is a documented, audited fact — not a hidden converter gap. NOTE:
  matrix §4 cond. 4's
  numerical forward-parity spot-check + the v1==v2 weight-parity test are
  explicitly DEFERRED (see the report ``deferred_forward_parity`` field): the
  v2-vs-v1 forward parity for every CV1 family is already bitwise-gated by
  parity_gn2 + gates_m2/m4/m5/m6, and CV1 substitutes class-canonical plan +
  manifest equivalence — a structural proxy, not the numerical criterion itself.

- **CV2 — converter hard-error path (the bit-rotted identifier)** (plan 13 CV2
  row; FD §10, matrix §5.1). The bit-rotted / dropped / parked v1 configs make
  ``salt2 convert-config`` HARD-ERROR (a ``ConvertError``, never an
  approximation) with a message NAMING the config as dropped + the removed/parked
  v1 class. Run on the 6 §5.1 bit-rotted + parked configs (GN2_open_data,
  GN2_tracks_neutral_SA_aux, GN2_tracks_neutral_CA, GN3_dev/GN2_dR,
  legacy/SubjetXbb — bit-rotted ``TransformerEncoder``/``ScaledDotProductAttention``/
  ``TransformerCrossAttentionEncoder``; legacy/Baseline_Xbb — parked
  ``R21Xbb``), each must hard-error with the expected marker AND the config name
  in the message. Negative control (``test_gates_m7.py`` + the in-gate
  ``corruption`` hook): a dropped config that SILENTLY converted (no
  ``ConvertError``) FAILS the gate — the corruption hook swaps a dropped entry's
  v1 source for a HEALTHY config (which converts cleanly), so the "must
  hard-error" assertion flips to False and the gate goes red, proving the
  hard-error check is not vacuous.

- **CVF — converter-output target-producer fidelity** (M7 W1.5; the validate/
  plan-compile MISS). Over the SAME CV1 denominator, CVF asserts every converted
  task's declared target/label is PRODUCED by a data module: (i) every task
  target in the converted FIT plan has a producer, and every v1-declared
  SYNTHETIC handle (a ``multi_target`` ``custom_target`` column that exists
  nowhere on disk) is produced by a CONCRETE (non-wildcard) data module
  (``MultiTarget`` / ``MaskFormerTargets`` / ...), never served only by the
  demand-driven ``Labels`` ``labels.**`` wildcard — the MISS that let the broken
  pre-F1a ``regression_multi_target`` pass CV1 (its regression head consumed a
  never-produced ``pt_label_handle`` yet plan-compiled clean because the wildcard
  narrows ``labels.**`` to ANY demanded key); (ii) the converter's task-NAME +
  stream set equals the v1 RESOLVED set; (iii) the named fixture exceptions
  (``event_classifier``, ``regression_multi_target``) are excluded from
  converter-vs-FIXTURE equality (fixture keeps, not fidelity failures) but NOT
  from check (i) — ``regression_multi_target`` MUST now concretely produce its
  ``pt_label_handle`` target (F1a). Negative control (``test_gates_m7.py`` + the
  in-gate ``corruption`` hook): a converter output whose ``MultiTarget`` producer
  was DELETED falls back to the wildcard (plan-compile still succeeds, the MISS)
  -> the target's ``wildcard_only`` flag flips True and the gate goes red.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import io
import json
import operator
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from salt.core import convert
from salt.core.cli import load_config
from salt.core.convert import ConvertError
from salt.core.gates_m5 import _CONV_CONFIGS as _M5_CONV_CONFIGS
from salt.core.gates_m5 import _conv_set_args as _m5_conv_set_args
from salt.core.gates_m6 import _CONV_M6_CONFIGS as _M6_CONV_CONFIGS
from salt.core.graph import Bundle, Executor, Mode
from salt.core.graph.planner import compile_plan
from salt.core.graph.spec import TensorSpec, unflatten_spec
from salt.core.main import CONFIG_DIR
from salt.core.main import main as salt2_main
from salt.core.nn import (
    Normaliser,
    StreamEmbed,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.tests.core.gn2_fixture import write_parity_norm_dict

# The salt.core package root (the worktree's salt/core), the editable surface
# REN1 greps for a residual ``vector`` FLAG use (reader GroupConfig field /
# StreamEmbed param). CONFIG_DIR is salt/core/configs, so its parent is salt/core.
CORE_DIR = CONFIG_DIR.parent

# The v1 SOURCE config root (the legacy tree, READ-ONLY reference; plan 13:
# v1 stays in-tree dormant). The converter reads YAML from here.
V1_CONFIG_DIR = CONFIG_DIR.parent.parent / "configs"


# ---------------------------------------------------------------------------
# shared report helpers (the gates_m2..m6 envelope, kept standalone)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet() -> Any:
    """Silence stdout+stderr for the wrapped block (the gates run noisy CLIs).

    Yields
    ------
    None
        Within the block, both streams are redirected to throwaway buffers.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


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
    """Build the common report envelope shared with the M2..M6 gates.

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


# ---------------------------------------------------------------------------
# CV1 — the authoritative needs-M5 + needs-M6 denominator
# ---------------------------------------------------------------------------

# The AUTHORITATIVE CV1 config list, embedded VERBATIM (plan 13 CV1 row: "Embed
# the authoritative config list verbatim"). It is the FULL plan 13 / matrix §4
# non-dropped denominator of 39 configs:
#   - the needs-M5 configs (matrix §3 M5; the live gates_m5 `_CONV_CONFIGS`, 24
#     entries incl. the conditional-drop Dipz) + needs-M6 (gates_m6
#     `_CONV_M6_CONFIGS`, 5) — every 🔶/🔷 config that has a hand-written v2
#     fixture in salt/core/configs/ (29 total; gated on all 4 matrix §4
#     conditions: convert + validate + plan-compile + fixture-plan parity); PLUS
#   - the 10 reproducible-today (✅) configs (matrix §1; `today: True`) that have
#     NO M5/M6 fixture but STILL require the converter for the mechanical
#     conversion details that recur everywhere (matrix §5.3: bake class_names +
#     origin_label, use_class_dict->weight_source, num_inputs->truncate,
#     input_map->groups.dataset, callbacks list->dict). They are gated on
#     conditions 1-3 ONLY (no fixture to plan-match; condition 4 is N/A).
# This closes the high-severity 29-vs-39 denominator gap (CV1 previously scoped
# itself to the 29 fixture-bearing configs and the gate report claimed it covered
# "the 39-config denominator" while it covered 29 — the 10 ✅ configs were
# exercised by NO gate: gates_m5/m6 CONV validate hand-written fixtures, not the
# converter).
#
# Per-config fields:
#   name        — the report key (matches the gates_m5/m6 _CONV_* names)
#   v1          — the v1 SOURCE stack under salt/configs/ (the converter input;
#                 overlays carry their FULL declared base chain first, then the
#                 overlay — the converter runs the legacy deep-merge on the
#                 resolved stack, matrix §4.1)
#   fix         — the v2 fixture STACK under salt/core/configs/ (the comparison
#                 target; mirrors the gates_m5/m6 `cfg` tuples). EMPTY () for the
#                 reproducible-today (✅) configs — they have no fixture.
#   tier        — "m5" (needs-M5), "m6" (needs-M6), or "today" (✅ no-fixture) for
#                 the report table
#   today       — True for the 10 reproducible-today (✅) configs: NO fixture, so
#                 gated on conditions 1-3 only (convert + validate + plan-compile).
#   class_names — (today configs only) the baked global flavour-head class_names
#                 supplied to the converter via --class-names (v1 autodiscovers
#                 them from h5 attrs; the converter hard-errors without data).
#                 Omitted for heads that declare class_names inline (TPLTmu/PLITel).
#   onnx        — "validate" when the config declares an export: block (onnx is a
#                 gated mode); "na" when it has no ONNX representation (GN2emu's
#                 soft-muon global; event_classifier's non-feature denominator)
#   labeller_override — True when the v1 labeller class_names cannot materialise
#                 in this ftag build (GN2X_qcdsplit's `qcdxx` is absent from ftag
#                 v0.2.17, matrix §2 / gates_m6.py:287-299; the converter
#                 FAITHFULLY emits `qcdxx`, the fixture substitutes `qcdll`); CV1
#                 applies the SAME documented substitution at plan-compile so the
#                 converter output can be compiled in this environment — the
#                 converter output stays faithful to v1, the override is a data-
#                 environment workaround, not a converter edit.
#   restored — True for the 6 Wave-F2-RESTORED configs (regression,
#                 regression_gaussian, regression_weighted, nan_regression,
#                 GN3V01, GN2XE) whose fixture was a v1 subset but is now restored
#                 to FULL v1 fidelity. CV1 asserts they REPRODUCE THE FULL V1
#                 CAPABILITY (conv == v1 == fixture task count AND module-class
#                 multiset equal to the fixture in every shared mode); a byte-exact
#                 fixture_match is recorded but NOT required (the residual diff is
#                 topology-neutral cosmetics). This set is PINNED.
#   fixture_subset — True for the 2 INTENTIONAL KEEPS (event_classifier,
#                 regression_multi_target) whose hand-written fixture is the
#                 faithful v2 artifact and NOT a 1:1 converter reproduction BY
#                 DESIGN. The exact-plan match cannot hold; the converter is MORE
#                 (event_classifier: metric-sink) / DIFFERENTLY (multi_target:
#                 rename) faithful than the fixture. CV1 records WHY (a keep_reason
#                 field), keeps them fixture-DIVERGENT, and still requires
#                 convert+validate + the full-v1-task-count fact. This set is
#                 PINNED (a fixture silently starting/stopping to match changes the
#                 accounting).
#   keep_reason — (fixture_subset keeps only) the documented reason the fixture
#                 diverges from the converter by design (named per codex CVF
#                 finding; asserted present by the gate).
#   conditional_drop — True for Dipz (matrix §1: drops once hitz validates;
#                 trigger fired post-M5). The converter still runs on it.
#   note        — the matrix feature(s) the config exercises.
_CV1_CONFIGS: tuple[dict[str, Any], ...] = (
    # -- needs-M5 regression family (encoder-less pooling + RegressionTaskModule)
    {
        "name": "regression",
        "v1": ("regression.yaml",),
        "fix": ("regression.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "restored": True,
        "note": "RegressionTask all variants + encoder-less pooling; F2 RESTORED the fixture to the "
        "full v1 5-task set (was a 3-task subset) — converter + fixture now both carry all 5 v1 "
        "tasks (class-canonical module set + edges identical; sink keys differ only by the "
        "verbose-v1 vs short-fixture head names)",
    },
    {
        "name": "regression_gaussian",
        "v1": ("regression_gaussian.yaml",),
        "fix": ("regression_gaussian.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "restored": True,
        "note": "GaussianRegressionTask + ONNX stddev; F2 confirmed full v1 fidelity (both gaussian "
        "heads present, PadMaskWriter wired). Converter + fixture carry the same task count and "
        "the same module-class set; the per-token head's input edge differs cosmetically "
        "(converter embed.tracks vs fixture seq.x — identical for a single-stream Concat([tracks]))",
    },
    {
        "name": "regression_weighted",
        "v1": ("regression_weighted.yaml",),
        "fix": ("regression_weighted.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "restored": True,
        "note": "RegressionTask sample_weight; F2 RESTORED the fixture to the full v1 5-task set "
        "(was 3 + the mass denom var) — converter + fixture now class-canonical-IDENTICAL (module "
        "set + edges + sinks match in every mode; only the head instance names differ)",
    },
    {
        "name": "regression_multi_target",
        "v1": ("regression_multi_target.yaml",),
        "fix": ("regression_multi_target.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "fixture_subset": True,
        "keep_reason": "INTENTIONAL KEEP (1 of 2): the hand-written fixture RENAMES the v1 task and "
        "is therefore not class-canonical-equal to the converter output (the converter preserves "
        "the verbose v1 task name; the fixture is the curated artifact). The fixture is the faithful "
        "v2 artifact kept for the MultiTarget/F1a producer-fidelity scenario (CVF); the converter "
        "stays MORE-verbose-faithful to the v1 source. NOT a converter-fidelity failure.",
        "note": "MultiTarget processor + RegressionTask; fixture renames the v1 task (intentional "
        "keep, not subset-restorable — the fixture IS the faithful artifact). CVF asserts F1a's "
        "concrete MultiTarget producer for pt_label_handle",
    },
    {
        "name": "nan_regression",
        "v1": ("nan_regression.yaml",),
        "fix": ("nan_regression.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "restored": True,
        "note": "RegressionTask NaN-target masking; F2 confirmed full v1 fidelity (all 3 v1 heads, "
        "PadMaskWriter wired). Converter + fixture carry the same task count + module-class set; the "
        "per-token head's input edge differs cosmetically (converter embed.tracks vs fixture seq.x)",
    },
    {
        "name": "dips",
        "v1": ("legacy/dips.yaml",),
        "fix": ("dips.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "note": "encoder-less pooling (the primary CI smoke fixture, KEEP despite legacy/)",
    },
    # -- needs-M5 GN3 family (LossGLS + RegressionTask; standalone + overlay) ----
    {
        "name": "GN3V00",
        "v1": ("GN3v01/GN3V00.yaml",),
        "fix": ("GN3V00.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "note": "LossGLS + RegressionTask (the GN3 dev baseline body)",
    },
    {
        "name": "GN3_v00",
        "v1": ("GN3_dev/GN3_v00.yaml",),
        "fix": ("GN3_v00.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "note": "standalone dev twin of GN3V00 (IP3D-named vars); LossGLS + RegressionTask",
    },
    {
        "name": "GN3_baseline",
        "v1": ("GN3_dev/GN3_baseline.yaml",),
        "fix": ("GN3_baseline.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "note": "LossGLS + track selections (the GN3_dev overlay-stack base)",
    },
    {
        "name": "GN3_Hybrid",
        "v1": ("GN3v01/GN3V00.yaml", "GN3v01/GN3_Hybrid.yaml"),
        "fix": ("GN3V00.yaml", "GN3_Hybrid.yaml"),
        "tier": "m5",
        "onnx": "validate",
        "note": "norm_type:hybrid passthrough overlay on GN3V00; (stack) LossGLS",
    },
    {
        "name": "GN3_Charge",
        "v1": ("GN3v01/GN3V00.yaml", "GN3v01/GN3_Charge.yaml"),
        "fix": ("GN3V00.yaml", "GN3_Charge.yaml"),
        "tier": "m5",
        "onnx": "validate",
        "note": "b-jet charge task overlay on GN3V00; (stack) LossGLS + RegressionTask",
    },
    {
        "name": "GN3_baseline_loose",
        "v1": ("GN3_dev/GN3_baseline.yaml", "GN3_dev/GN3_baseline_loose.yaml"),
        "fix": ("GN3_baseline.yaml", "GN3_baseline_loose.yaml"),
        "tier": "m5",
        "onnx": "validate",
        "note": "selections:null overlay on GN3_baseline; (stack) LossGLS",
    },
    {
        "name": "GN3_dR",
        "v1": ("GN3_dev/GN3_baseline.yaml", "GN3_dev/GN3_dR.yaml"),
        "fix": ("GN3_baseline.yaml", "GN3_dR.yaml"),
        "tier": "m5",
        "onnx": "validate",
        "note": "dR-matched tracks retarget overlay on GN3_baseline; (stack) LossGLS",
    },
    {
        "name": "GN3_flow",
        "v1": (
            "GN3_dev/GN3_baseline.yaml",
            "GN3_dev/GN3_baseline_loose.yaml",
            "GN3_dev/GN3_flow.yaml",
        ),
        "fix": ("GN3_baseline.yaml", "GN3_baseline_loose.yaml", "GN3_flow.yaml"),
        "tier": "m5",
        "onnx": "validate",
        "note": "pflow-stream overlay on GN3_baseline->loose; (stack) LossGLS",
    },
    {
        "name": "GN3_LepID_SMT",
        "v1": (
            "GN3_dev/GN3_baseline.yaml",
            "GN3_dev/GN3_baseline_loose.yaml",
            "GN3_dev/GN3_flow.yaml",
            "GN3_dev/GN3_LepID_SMT.yaml",
        ),
        "fix": (
            "GN3_baseline.yaml",
            "GN3_baseline_loose.yaml",
            "GN3_flow.yaml",
            "GN3_LepID_SMT.yaml",
        ),
        "tier": "m5",
        "onnx": "validate",
        "note": "lepton-ID+SMT data overlay on GN3_baseline->loose->flow; (stack) LossGLS",
    },
    {
        "name": "GN3_tracklabel",
        "v1": (
            "GN3_dev/GN3_baseline.yaml",
            "GN3_dev/GN3_baseline_loose.yaml",
            "GN3_dev/GN3_flow.yaml",
            "GN3_dev/GN3_LepID_SMT.yaml",
            "GN3_dev/GN3_tracklabel.yaml",
        ),
        "fix": (
            "GN3_baseline.yaml",
            "GN3_baseline_loose.yaml",
            "GN3_flow.yaml",
            "GN3_LepID_SMT.yaml",
            "GN3_tracklabel.yaml",
        ),
        "tier": "m5",
        "onnx": "validate",
        "note": "5-task track_type/track_source overlay (deepest stack); (stack) LossGLS",
    },
    # -- needs-M5 concat/global family (VectorConcat + export.inputs alias) ------
    {
        "name": "GN3V01",
        "v1": ("GN3v01/GN3V01.yaml",),
        "fix": ("gn3v01.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "restored": True,
        "note": "flagship GN3: VectorConcat+alias + LossGLS + norm_type:hybrid + RegressionTask; "
        "F2 RESTORED the dropped flows + electrons constituent streams (was tracks-only) — the "
        "converter + fixture now EXACTLY class-canonical-MATCH in every mode (module set + edges + "
        "sinks identical; the converter reproduces tracks+flows+electrons, as does the fixture)",
    },
    {
        "name": "GN2emu",
        "v1": ("GN2/GN2emu.yaml",),
        "fix": ("GN2emu.yaml",),
        "tier": "m5",
        "onnx": "na",
        "note": "VectorConcat (soft-muon global concat) outside GN3; fit+test ONLY (no export: "
        "block — the 14-var soft-muon `global` has no ONNX representation, matrix gates_m5)",
    },
    {
        "name": "GN3_SoftE",
        "v1": ("GN3v01/GN3V00.yaml", "GN3v01/GN3_SoftE.yaml"),
        "fix": ("GN3V00.yaml", "GN3_SoftE.yaml"),
        "tier": "m5",
        "onnx": "validate",
        "note": "electrons stream + global concat overlay on GN3V00; (stack) LossGLS",
    },
    {
        "name": "GN3EPCLV01",
        "v1": ("GN3EPCLV01.yaml",),
        "fix": ("GN3EPCLV01.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "note": "GN3 3-stream: LossGLS + norm_type:hybrid + VectorConcat+alias + RegressionTask",
    },
    # -- needs-M5 Gaussian-PVz / event-level / MaskFormer ----------------------
    {
        "name": "hitz",
        "v1": ("hitz.yaml",),
        "fix": ("hitz.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "note": "GaussianRegressionTask on an encoder (HLT hits PV-z)",
    },
    {
        "name": "Dipz",
        "v1": ("legacy/Dipz.yaml",),
        "fix": ("Dipz.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "conditional_drop": True,
        "note": "encoder-less GaussianRegressionTask; CONDITIONAL DROP (matrix §1: drops once hitz "
        "validates — fired post-M5; the converter still RUNS on it until the deletion release)",
    },
    {
        "name": "event_classifier",
        "v1": ("event_classifier.yaml",),
        "fix": ("event_classifier.yaml",),
        "tier": "m5",
        "onnx": "na",
        "fixture_subset": True,
        "keep_reason": "INTENTIONAL KEEP (2 of 2): the converter's classification head wires the "
        "FIT/VAL metric sinks that the hand-written fixture deliberately omits, so the FIT/VAL "
        "plans diverge BY DESIGN (an export-omission that is correct for this fixture — the TEST/"
        "ONNX plans still match). The fixture is the curated artifact; the divergence is an "
        "intentional metric-sink omission, NOT a converter-fidelity failure or a missing v1 task.",
        "note": "RegressionTask TEST de-scaling from a NON-feature label; fit+test ONLY (no "
        "export: block). Fixture omits the FIT/VAL metric sinks the converter wires for the "
        "classification head (TEST/ONNX plans match)",
    },
    {
        "name": "MaskFormer",
        "v1": ("MaskFormer.yaml",),
        "fix": ("MaskFormer.yaml",),
        "tier": "m5",
        "onnx": "validate",
        "note": "MaskDecoder + MaskFormerMatchedLoss + MaskFormerTargets + object writer + metrics",
    },
    # -- needs-M6 (Labeller / vector-stream / muP / edges) ----------------------
    {
        "name": "GN3X",
        "v1": ("GN3X.yaml",),
        "fix": ("GN3X.yaml",),
        "tier": "m6",
        "onnx": "validate",
        "note": "boosted GN3X: 9-class on-the-fly Labeller (require_labels:True) + lion + LossGLS",
    },
    {
        "name": "GN2X_qcdsplit",
        "v1": ("GN2/GN2X_qcdsplit.yaml",),
        "fix": ("GN2X_qcdsplit.yaml",),
        "tier": "m6",
        "onnx": "validate",
        "labeller_override": True,
        "note": "GN2X 7-class QCD-split Labeller (require_labels:False, drop-unlabelled) + "
        "truth_hadrons stream. The v1 7th class `qcdxx` is absent from ftag v0.2.17 (matrix "
        "§2): the converter FAITHFULLY emits qcdxx; CV1 substitutes qcdll at plan-compile "
        "(the SAME workaround the fixture + gates_m6 LB1 make) so it compiles in this build",
    },
    {
        "name": "DL1",
        "v1": ("legacy/DL1.yaml",),
        "fix": ("DL1.yaml",),
        "tier": "m6",
        "onnx": "validate",
        "note": "jets-only MLP: rank-2 [B,F] vector-stream embed (rank inferred from the "
        "global_object input) -> sequence:false head, NO encoder/pool (the M6-6 deliverable, "
        "gate VS1); 3-class CE; LossSum",
    },
    {
        "name": "GN2_muP",
        "v1": ("GN2/GN2_muP.yaml",),
        "fix": ("GN2_muP.yaml",),
        "tier": "m6",
        "onnx": "validate",
        "mup": True,
        "note": "GN2 with muP: mup:true on track_embed + encoder + model.mup routing (M6-B, gates "
        "MU1/MU2). The plan-compile needs base/delta infshapes generated data-free first",
    },
    {
        "name": "GN2XE",
        "v1": ("GN2/GN2XE.yaml",),
        "fix": ("GN2XE.yaml",),
        "tier": "m6",
        "onnx": "validate",
        "restored": True,
        "note": "GN2X with EDGE FEATURES end-to-end (EdgeFeatures -> EdgeEmbed -> encoder edges:; "
        "M6-C, gates ED1/ED2). F2 RESTORED the jets-context edge v1 attaches to track_embed (v1 "
        "InitNet.attach_global default True, initnet.py:46) — the converter + fixture now EXACTLY "
        "class-canonical-MATCH in every mode (the norm.normed.jets -> track_embed edge is present "
        "on both sides; edge counts FIT 37 / TEST 34 / ONNX 33 identical)",
    },
    # -- reproducible-today (✅) configs: NO hand-written v2 fixture exists (they
    # were never authored as M5/M6 fixtures — every module already shipped at
    # M1-M4.5). They still REQUIRE the converter for the mechanical conversion
    # details that recur everywhere (matrix §5.3: bake class_names + origin_label,
    # use_class_dict -> weight_source, num_inputs -> truncate, input_map ->
    # groups.dataset, callbacks list -> dict). The high-severity critic finding
    # (the 29-vs-39 denominator gap) is closed by adding them here: ``today: True``
    # gates them on conditions 1-3 ONLY (convert + validate + plan-compile
    # fit/test/onnx) — there is no fixture to plan-match against (condition 4 is
    # N/A; the v2-vs-v1 forward parity for the GN2 family is already bitwise-gated
    # by parity_gn2 + gates_m2/m4). The global ``flavour_label`` classification
    # head autodiscovers class_names from h5 attrs in v1 (cli.py:get_object_class_
    # names); the converter HARD-ERRORS without data, so CV1 supplies the canonical
    # FTAG class_names per cardinality via ``class_names`` -> ``--class-names``
    # (the converter never reads a data file). Heads that declare class_names
    # inline (TPLTmu/PLITel) need no override. With these 10, CV1's denominator is
    # 39 = the full matrix §4 / plan 13 non-dropped set (29 needs-M5/M6 + 10 ✅).
    {
        "name": "GN2",
        "v1": ("GN2/GN2.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets", "taujets"]},
        "note": "GN2 nominal (matrix §4 GN2-nominal family): tracks stream, 4-layer encoder, GAP, "
        "3 tasks (use_class_dict flavour + track origin + vertexing). No fixture; gated on "
        "convert+validate+plan-compile (cond. 1-3). class_names baked (ledger #7)",
    },
    {
        "name": "gn2v1-opendata",
        "v1": ("GN2/gn2v1-opendata.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets", "taujets"]},
        "note": "the v1 half of the plan-09 retrain gate (matrix §4 rep #1); gn2v2-opendata.yaml "
        "is its shipped v2 twin (M2/M4/M4.5 gated pair). No M5/M6 fixture row; cond. 1-3",
    },
    {
        "name": "GN2_extended",
        "v1": ("GN2/GN2_extended.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets", "taujets", "ghostjets"]},
        "note": "GN2 with a 5-class flavour head (extended labelling). No fixture; cond. 1-3",
    },
    {
        "name": "GN2_charded_neutral_loose_aux",
        "v1": ("GN2/GN2_charded_neutral_loose_aux.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets"]},
        "note": "3 streams (loose tracks/charged/neutral pflow) + aux tasks on charged; modern "
        "salt.models.Transformer. No fixture; cond. 1-3",
    },
    {
        "name": "GN2X",
        "v1": ("GN2/GN2X.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets", "taujets"]},
        "note": "boosted Xbb GN2X: tracks+flow, 8 layers, registers, num_inputs->truncate. No "
        "fixture; cond. 1-3",
    },
    {
        "name": "GN2XTau",
        "v1": ("GN2/GN2XTau.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets", "taujets", "ghostjets"]},
        "note": "GN2X variant with a 5-class (incl. tau) head. No fixture; cond. 1-3",
    },
    {
        "name": "flow",
        "v1": ("flow.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets"]},
        "note": "2-stream CI smoke (tracks+pflow, 3-class). No fixture; cond. 1-3",
    },
    {
        "name": "tutorial",
        "v1": ("tutorial.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "class_names": {"jets_classification": ["bjets", "cjets", "ujets"]},
        "expect_convert_error": "does not contain 'VertexIndex'",
        "note": "GN2 open-data tutorial (docs/tutorial.md). Its const_vertexing head uses the "
        "non-standard label `truth_vertex_idx` (no 'VertexIndex' -> no derivable OriginLabel "
        "column), which the v2 VertexingTaskModule invariant (tasks.py:702-707) cannot map: the "
        "converter HARD-ERRORS (FD §10 'never approximate') with an actionable TODO. gn2v2-"
        "opendata.yaml is the working v2 counterpart for the tutorial's open-data subject matter; "
        "the W4 docs rewrite retires/rewrites tutorial.md against v2 modules (matrix §5.3 spirit). "
        "Counted in the 39 denominator; gated on the EXPECTED hard-error, not convert-success",
    },
    {
        "name": "TPLTmu",
        "v1": ("TPLTmu.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "note": "prompt-lepton tagger (muons); documents the v1 magic-task-name wart. The global "
        "head declares class_names inline ([muxprompt, npxall]) — NO --class-names needed. "
        "matrix §4 rep 'non-jet global object'. No fixture; cond. 1-3",
    },
    {
        "name": "PLITel",
        "v1": ("PLITel.yaml",),
        "fix": (),
        "tier": "today",
        "onnx": "validate",
        "today": True,
        "note": "prompt-lepton tagger (electrons). The global head declares class_names inline "
        "([elxprompt, npxall]) — NO --class-names needed. No fixture; cond. 1-3",
    },
)

# the PINNED set of fixture-subset configs — the INTENTIONAL KEEPS whose
# hand-written v2 fixture is NOT a 1:1 converter reproduction by design (the
# converter is MORE faithful to the v1 source than the fixture). After Wave F2
# restored 6 previously-subset fixtures to full v1 fidelity, this set shrinks to
# exactly the 2 DOCUMENTED keeps (the codex CVF finding names them explicitly):
#
#   - ``event_classifier`` — the converter wires the FIT/VAL classification-metric
#     sinks the fixture deliberately omits; an EXPORT-OMISSION correct by design
#     (TEST/ONNX plans still match). NOT a missing v1 task.
#   - ``regression_multi_target`` — the fixture RENAMES the v1 task (so it is not
#     class-canonical-equal to the converter), and the fixture IS the faithful v2
#     artifact kept for the F1a MultiTarget producer-fidelity scenario (CVF).
#
# Each keep carries a ``keep_reason`` field naming WHY it diverges (asserted
# present below). A fixture that silently starts/stops matching changes CV1's
# accounting, so this set is PINNED and asserted == exactly these two.
_CV1_FIXTURE_SUBSET: frozenset[str] = frozenset(
    e["name"] for e in _CV1_CONFIGS if e.get("fixture_subset")
)

# the PINNED set of F2-RESTORED configs: previously fixture-subset, now restored
# to FULL v1 fidelity (the fixture reproduces the full v1 task/stream/capability
# set). For these the converter + fixture reproduce the SAME v1 capability — the
# gate asserts they reproduce the full v1 task count AND are STRUCTURALLY
# EQUIVALENT to the fixture (the module-CLASS multiset matches in every shared
# mode), no longer subset-divergent. (Class-canonical sink keys / one documented-
# equivalent embed.tracks==seq.x edge / the fixture's PadMaskWriter mask column
# may still differ from the converter's verbose-v1 naming — cosmetic, topology-
# neutral — so a byte-exact ``fixture_match`` is NOT required; the structural-
# equivalence + full-v1-task-count facts are.)
_CV1_RESTORED: frozenset[str] = frozenset(
    e["name"] for e in _CV1_CONFIGS if e.get("restored")
)

_NO_STRICT_RATIONALE = (
    "no --strict: a data-free validation cannot satisfy it (the M5/M6-CONV precedent + the "
    "converter's own _self_validate, convert.py:1657). --strict promotes EVERY warning to an error "
    "(cli.py:919-920) and several warnings are inherent to data-free validation, orthogonal to "
    "convertibility: (1) the no-schema warning (every config lacking a schema: artifact); "
    "(2) warning-level deadcode / per-mode unconsumed-pred infos; (3) Normaliser preflight "
    "warnings for streams the data-free norm dict does not supply (multi-stream GN3/GN2X configs "
    "emit benign 'missing input type' preflight warnings). The data-free norm dict resolves the "
    "preflight so the bind succeeds; the residual warnings are data-shape artifacts, not graph "
    "errors. rc == 0 (no ERROR-level finding, design §4.2) IS the convert+validate+plan-compile "
    "criterion."
)


# ---------------------------------------------------------------------------
# CV1 helpers (data-free conversion + plan-signature comparison)
# ---------------------------------------------------------------------------


def _merge_yaml(paths: Sequence[Path]) -> dict[str, Any]:
    """Deep-merge a config STACK to one dict (v1 dict-merge, list-replace).

    Mirrors `salt fit` / `salt2` deep-merge: later files override matching dict
    keys; non-dict leaves (incl. lists) are replaced wholesale. Used only to
    READ the stacked fixture's declared ``class_names`` / Normaliser streams /
    labeller classes for the data-free overrides — never to drive a plan.

    Returns
    -------
    dict[str, Any]
        The merged config mapping.
    """
    merged: dict[str, Any] = {}

    def _m(a: dict[str, Any], b: Mapping[str, Any]) -> None:
        for k, v in (b or {}).items():
            if isinstance(v, Mapping) and isinstance(a.get(k), dict):
                _m(a[k], v)
            else:
                a[k] = copy.deepcopy(v)

    for p in paths:
        with open(p) as fh:
            _m(merged, yaml.safe_load(fh) or {})
    return merged


def _fixture_class_names(fix_full: Sequence[Path]) -> dict[str, list[str]]:
    """The fixture's declared ``ClassificationTaskModule`` class_names per head.

    Read from the (stacked) hand-written fixture — the source of truth for the
    baked class names (matrix §5.3 ledger #7). CV1 supplies these to the
    converter via ``--class-names HEAD=...`` so the converter never reads a data
    file for h5-attr autodiscovery (matrix §4 cond. 1, design §9.3).

    Returns
    -------
    dict[str, list[str]]
        ``{head_name: [class names]}`` for every classification head that
        declares ``class_names`` (heads resolving from the embedded
        ``CLASS_NAMES`` lookup — track_origin/type/source — are auto-resolved by
        the converter and omitted here unless the fixture pins them).
    """
    mods = _merge_yaml(fix_full).get("model", {}).get("init_args", {}).get("modules", {})
    out: dict[str, list[str]] = {}
    for name, mod in mods.items():
        if str(mod.get("class_path", "")).endswith("ClassificationTaskModule"):
            cn = (mod.get("init_args") or {}).get("class_names")
            if cn:
                out[name] = list(cn)
    return out


def _fixture_labeller_classes(fix_full: Sequence[Path]) -> list[str] | None:
    """The fixture's ``FtagLabeller`` processor class_names (if any).

    M8 sub-wave 1 extracted the on-the-fly labeller into the standalone
    ``ftag_labeller`` ``data.modules`` entry (it was an opt-in flag on ``labels``),
    so the labeller classes live there now.

    Returns
    -------
    list[str] | None
        The fixture's labeller class names, or None when the fixture has no
        on-the-fly labeller.
    """
    mods = _merge_yaml(fix_full).get("data", {}).get("modules", {})
    return ((mods.get("ftag_labeller") or {}).get("init_args") or {}).get("class_names")


# v2 task-bearing module classes (the converter + fixtures emit these as the
# head modules). A MaskDecoder is the MaskFormer head; the *_classification /
# *_vertexing / *_regression heads are TaskModules. Counting these lets CV1
# machine-check that a converter regression never DROPS a v1 task on a
# fixture-subset config (the low-severity converter_faithful_to_v1 finding).
_V2_TASK_CLASS_SUFFIXES: tuple[str, ...] = ("TaskModule", "MaskDecoder")


def _count_v2_task_modules(cfg: Mapping[str, Any]) -> int:
    """Count task-bearing v2 head modules in a v2 config dict / converter output.

    Returns
    -------
    int
        The number of ``model.init_args.modules`` whose class_path ends in a
        task suffix (``*TaskModule`` / ``MaskDecoder``).
    """
    mods = (cfg.get("model") or {}).get("init_args", {}).get("modules", {}) or {}
    return sum(
        1
        for m in mods.values()
        if isinstance(m, Mapping)
        and any(str(m.get("class_path", "")).endswith(s) for s in _V2_TASK_CLASS_SUFFIXES)
    )


def _count_v2_fixture_task_modules(fix_full: Sequence[Path]) -> int:
    """Count task-bearing v2 head modules in a (stacked) hand-written fixture.

    Returns
    -------
    int
        The number of fixture ``model.modules`` that are v2 task heads.
    """
    return _count_v2_task_modules(_merge_yaml(fix_full))


def _count_v1_tasks(cfg: Mapping[str, Any]) -> int:
    """Count v1 task heads in a RESOLVED v1 stack (the converter's input).

    v1 nests the inner ``SaltModel`` under ``model.model`` (the ``ModelWrapper``
    wraps it); its heads live at
    ``model.model.init_args.tasks.init_args.modules`` (a list) — the SAME path
    ``convert_stack`` reads (convert.py:1069-1076). A ``mask_decoder`` block is
    the MaskFormer head and counts as one task (it owns the per-object
    class/mask/regression heads). This is the independent v1-source task count
    the converter MUST reproduce — so a future converter that silently dropped a
    v1 task on a fixture-subset config is caught (the low-severity finding: make
    ``converter_faithful_to_v1`` a checked fact, not just a human note).

    Returns
    -------
    int
        The v1 task count (len(tasks) + 1 when a ``mask_decoder`` is present).
    """
    inner_init = ((cfg.get("model") or {}).get("model") or {}).get("init_args", {}) or {}
    tasks = (inner_init.get("tasks") or {}).get("init_args", {}).get("modules", []) or []
    n = len(tasks)
    if inner_init.get("mask_decoder") is not None:
        n += 1
    return n


def _normaliser_overrides(paths: Sequence[Path], outdir: Path) -> tuple[list[str], Path]:
    """Build the data-free ``--set ...norm_dict=`` overrides for a config stack.

    Writes a minimal stream-covering norm dict into ``outdir`` (the M5/M6-CONV /
    convert._self_validate convention) and returns one override per Normaliser
    module so its ``materialise`` resolves data-free.

    Returns
    -------
    tuple[list[str], Path]
        The ``--set KEY=VALUE`` flag pairs (already split) and the norm-dict path.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    merged = _merge_yaml(paths)
    mods = merged.get("model", {}).get("init_args", {}).get("modules", {})
    streams: set[str] = set()
    norm_mods: list[str] = []
    for name, mod in mods.items():
        if str(mod.get("class_path", "")).endswith("Normaliser"):
            streams.update((mod.get("init_args") or {}).get("streams", []) or [])
            norm_mods.append(name)
    nd = outdir / f"{paths[-1].stem}__nd.yaml"
    nd.write_text(yaml.safe_dump({s: {} for s in (streams or {"jets", "tracks"})}))
    sets: list[str] = []
    for name in norm_mods:
        sets += ["--set", f"model.modules.{name}.init_args.norm_dict={nd}"]
    return sets, nd


def _mup_shape_override(paths: Sequence[Path], norm_sets: list[str], outdir: Path) -> list[str]:
    """Generate a muP config's base/delta infshapes data-free; return the --set.

    Runs the REAL ``salt2 mup-shapes`` (the gates_m6 _conv_mup_shape_path / MU2
    precedent) so ``SaltModule._apply_mup_shapes`` resolves at plan-compile.

    Returns
    -------
    list[str]
        ``["--set", "model.init_args.mup.shape_path=<file>"]`` or ``[]`` when the
        config carries no ``model.init_args.mup`` block.

    Raises
    ------
    RuntimeError
        When ``salt2 mup-shapes`` fails (rc != 0) or writes no file.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    merged = _merge_yaml(paths)
    init = merged.get("model", {}).get("init_args", {})
    if init.get("mup") is None:
        return []
    sp = outdir / f"{paths[-1].stem}__shape_mup.bsh"
    cargs: list[str] = []
    for p in paths:
        cargs += ["-c", str(p)]
    with _quiet():
        rc = salt2_main(["mup-shapes", *cargs, "--save-path", str(sp), *norm_sets])
    if rc != 0 or not sp.is_file():
        raise RuntimeError(
            f"salt2 mup-shapes failed for {[str(p) for p in paths]} (rc={rc}) — the muP shape "
            "generation is a prerequisite for the CV1 plan-compile of a muP config"
        )
    return ["--set", f"model.init_args.mup.shape_path={sp}"]


def _validate_rc(paths: Sequence[Path], mode: str, sets: list[str]) -> int:
    """Run ``salt2 graph validate --mode <mode>`` on a config stack; return rc.

    Drives the REAL validator (the d2cfg / M5-CONV command path), data-free, NOT
    ``--strict`` (see ``_NO_STRICT_RATIONALE``).

    Returns
    -------
    int
        The validator exit code (0 == convert+validate+plan-compile clean).
    """
    cargs: list[str] = []
    for p in paths:
        cargs += ["-c", str(p)]
    with _quiet():
        return salt2_main(["graph", "validate", "--mode", mode, *cargs, *sets])


def _plan_signature(
    paths: Sequence[Path], outdir: Path, *, extra_sets: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Compile a config stack's per-mode plans and return a CLASS-CANONICAL sig.

    Loads the stack through the REAL `load_config` (the trainer-config adapter,
    the same surface ``salt2 graph validate`` uses), then compiles each declared
    mode. The signature is CLASS-CANONICAL — module instance names are normalised
    to their module CLASS (the converter derives instance names from v1 task
    names while the hand-written fixtures pick short names; the GRAPH is what must
    match, not the labels). Per mode it records:

    - ``classes``  — the sorted multiset of module classes in the plan;
    - ``edges``    — the sorted set of (producer-class, key-namespace,
      consumer-class) triples (SOURCES/SINKS sentinels kept verbatim);
    - ``sinks``    — the sorted per-mode sink keys (the demand boundary).

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{mode_name: {classes, edges, sinks} | {error}}``.
    """
    sets, _ = _normaliser_overrides(paths, outdir)
    sets += _mup_shape_override(paths, sets, outdir)
    if extra_sets:
        sets += extra_sets
    # load_config takes bare KEY=VALUE entries (strip the --set tokens)
    bare = [s for s in sets if s != "--set"]
    with _quiet():
        gc = load_config([str(p) for p in paths], bare)

    def _cls(node: str) -> str:
        return type(gc.modules[node]).__name__ if node in gc.modules else node

    out: dict[str, dict[str, Any]] = {}
    for mode in gc.sinks:
        try:
            plan = compile_plan(
                gc.modules,
                mode,
                gc.sources,
                schema=gc.schema,
                sinks=gc.sinks,
                sink_origins=gc.sink_origins.get(mode),
            )
            out[mode.name] = {
                "classes": sorted(type(gc.modules[n]).__name__ for n in plan.module_names),
                "edges": sorted(
                    (_cls(e.producer), e.key.split(".")[0], _cls(e.consumer)) for e in plan.edges
                ),
                "sinks": sorted(gc.sinks.get(mode, ())),
            }
        except Exception as err:  # noqa: BLE001, PERF203 - compile failure is a per-mode datum
            out[mode.name] = {"error": f"{type(err).__name__}: {err}"}
    return out


def _onnx_manifest(
    paths: Sequence[Path], outdir: Path, *, extra_sets: list[str] | None = None
) -> list[str] | None:
    """The ONNX output manifest (ordered output ports) for a config stack.

    Reuses `load_config`'s writer-callback ONNX manifest (the M4.5 unified output
    manifest, the validate-time ONNX-sink derivation, cli.py:391-392). ``extra_sets``
    carries the same data-environment overrides as `_plan_signature` (e.g. the
    GN2X_qcdsplit labeller substitution) so the converted config loads.

    Returns
    -------
    list[str] | None
        The ordered ONNX output ports, or None when the config has no ONNX
        export representation (no ``export:`` block / no writer manifest).
    """
    sets, _ = _normaliser_overrides(paths, outdir)
    if extra_sets:
        sets += extra_sets
    bare = [s for s in sets if s != "--set"]
    with _quiet():
        gc = load_config([str(p) for p in paths], bare)
        if gc.writers is None or gc.reader is None:
            return None
        try:
            manifest = gc.writers.onnx_manifest(gc.model_modules or gc.modules, gc.reader)
        except Exception:  # noqa: BLE001 - absent/invalid manifest -> no ONNX surface
            return None
    return [out.port for out in manifest] if manifest else None


def _eval_manifest(sig_test: dict[str, Any]) -> list[str]:
    """The eval (TEST) column manifest: the TEST plan's sink keys.

    The TEST sinks ARE the eval-file column demand boundary (preds.* + writer-
    demanded meta/labels/masks; design §4.2/§8).

    Returns
    -------
    list[str]
        The sorted TEST sink keys (empty when the TEST plan failed to compile).
    """
    return list(sig_test.get("sinks", [])) if "error" not in sig_test else []


def _structural_equivalence(
    conv_sig: Mapping[str, Any], fix_sig: Mapping[str, Any]
) -> dict[str, bool]:
    """Per shared mode, whether the converter + fixture compile to the SAME module set.

    The STRUCTURAL-EQUIVALENCE proxy used to gate the F2-RESTORED configs: the
    fixture now reproduces the full v1 capability, so the converter and fixture
    must compile to the SAME module-CLASS multiset in every shared mode (the
    module inventory — what modules run — is identical). This is WEAKER than the
    exact ``fixture_match`` (which also requires identical edges, sinks, eval/ONNX
    manifests) because a RESTORED fixture may still diverge from the converter's
    verbose-v1 output by topology-NEUTRAL cosmetics: class-canonical sink keys
    (the converter keeps verbose v1 task names, the fixture short names), one
    documented-equivalent embed.tracks==seq.x per-token-head edge for a
    single-stream Concat, and the fixture's hand-set PadMaskWriter mask column.
    None of these change WHICH modules run; the module-class set is the faithful
    invariant. A RESTORED config that DROPPED a module (a real regression) would
    change the class set and FAIL this check.

    Returns
    -------
    dict[str, bool]
        ``{mode_name: classes-equal}`` over the modes both signatures compiled.
    """
    out: dict[str, bool] = {}
    for mode in sorted(set(conv_sig) & set(fix_sig)):
        c, f = conv_sig[mode], fix_sig[mode]
        out[mode] = "error" not in c and "error" not in f and c.get("classes") == f.get("classes")
    return out


def run_cv1(
    outdir: Path | str,
    *,
    corruption: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """CV1: ``salt2 convert-config`` acceptance over the needs-M5+M6 denominator.

    For EVERY config in the AUTHORITATIVE ``_CV1_CONFIGS`` (the needs-M5 +
    needs-M6 configs with hand-written v2 fixtures; the list is embedded verbatim
    in the report), CV1:

    1. runs ``salt2 convert-config`` on the v1 SOURCE stack (``salt/configs/...``)
       with the fixture's baked class_names supplied via ``--class-names`` — and
       asserts conversion SUCCEEDS (no ``ConvertError``);
    2. asserts the converted v2 config validates + plan-compiles fit/test/onnx
       (``salt2 graph validate`` rc == 0 in every applicable mode; data-free, NOT
       ``--strict`` — the M5/M6-CONV precedent, ``_NO_STRICT_RATIONALE``);
    3. asserts SEMANTIC EQUIVALENCE to the hand-written fixture STACK: the
       converted single config and the fixture stack compile to the SAME plan in
       every shared mode (CLASS-CANONICAL module set, edges, sinks, the eval
       (TEST) column manifest and the ONNX output manifest) — the PLANS, not raw
       YAML bytes.

    For the PINNED fixture-subset configs (``_CV1_FIXTURE_SUBSET``: the
    hand-written fixture is a curated subset of the v1 source — fewer regression
    tasks / streams / context edges), the exact-plan match cannot hold; CV1
    records ``fixture_match: false`` + ``converter_faithful_to_v1: true`` and the
    config STILL must convert + validate + plan-compile. The subset set is itself
    asserted (a fixture silently starting/stopping to match changes the gate's
    accounting — so the divergence is an audited fact, not a hidden gap).

    Dipz (matrix §1 conditional drop) is INCLUDED — the converter still runs on
    it; it is annotated ``conditional_drop`` in the report.

    Negative control (``test_gates_m7.py`` + the ``corruption`` hook): the hook
    transforms the embedded config list (e.g. point a config's ``v1`` at a
    bit-rotted source) — the conversion then hard-errors, so the per-config
    ``converted`` check flips False and the gate goes red, proving the acceptance
    assertions are not vacuous.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("CV1 convert-config acceptance — convert + validate + plan-compile + fixture-plan parity")
    print("=" * 96)

    configs: list[dict[str, Any]] = [dict(e) for e in _CV1_CONFIGS]
    if corruption is not None:
        configs = corruption(configs)

    checks: dict[str, bool] = {}
    results: list[dict[str, Any]] = []

    # -- every v1 source + v2 fixture file exists on disk ---------------------
    missing: list[str] = [
        f"{e['name']}:v1:{v}"
        for e in configs
        for v in e.get("v1", ())
        if not (V1_CONFIG_DIR / v).is_file()
    ] + [
        f"{e['name']}:fix:{f}"
        for e in configs
        for f in e.get("fix", ())
        if not (CONFIG_DIR / f).is_file()
    ]
    checks["all_cv1_config_files_present"] = missing == []

    for entry in configs:
        name = entry["name"]
        cdir = outdir / name
        cdir.mkdir(parents=True, exist_ok=True)
        v1_full = [V1_CONFIG_DIR / p for p in entry.get("v1", ())]
        fix_full = [CONFIG_DIR / p for p in entry.get("fix", ())]
        files_ok = all(p.is_file() for p in v1_full) and all(p.is_file() for p in fix_full)
        is_today = bool(entry.get("today"))
        expect_err = entry.get("expect_convert_error")
        row: dict[str, Any] = {
            "name": name,
            "tier": entry["tier"],
            "today": is_today,
            "v1_stack": list(entry.get("v1", ())),
            "fix_stack": list(entry.get("fix", ())),
            "fixture_subset": bool(entry.get("fixture_subset")),
            "restored": bool(entry.get("restored")),
            "keep_reason": entry.get("keep_reason"),
            "conditional_drop": bool(entry.get("conditional_drop")),
            "expect_convert_error": expect_err,
            "converted": False,
            "validate": {},
            "fixture_match": None,
            "mode_match": {},
            "eval_manifest_match": None,
            "onnx_manifest_match": None,
            "structural_equivalence": {},
            "structurally_equivalent": None,
            "reproduces_full_v1_capability": None,
            "converter_faithful_to_v1": None,
            "v1_task_count": None,
            "fixture_task_count": None,
            "conv_task_count": None,
            "note": entry["note"],
        }
        if not files_ok:
            checks[f"{name}:converted"] = False
            results.append(row)
            continue

        # -- (1) convert ------------------------------------------------------
        # reproducible-today (✅) configs have NO fixture: take the baked global
        # flavour class_names from the embedded `class_names` map (heads that
        # declare class_names inline need none). Otherwise read them from the
        # hand-written fixture (matrix §5.3 ledger #7).
        class_names = (
            dict(entry.get("class_names") or {}) if is_today else _fixture_class_names(fix_full)
        )
        conv_path = cdir / f"{name}__converted.yaml"
        try:
            with _quiet():
                cfg = convert.convert_config(v1_full, class_names=class_names or None)
            conv_path.write_text(convert._yaml_dump(cfg))  # noqa: SLF001 - same-package dumper
            row["converted"] = True
        except ConvertError as err:
            row["convert_error"] = f"{type(err).__name__}: {err}"
            # `expect_convert_error` configs (tutorial: non-standard vertex label
            # the v2 module cannot map) MUST hard-error with the expected marker
            # (FD §10). The expected hard-error IS the per-config acceptance.
            if expect_err is not None and corruption is None:
                row["hard_errored_as_expected"] = expect_err in str(err)
                checks[f"{name}:accept"] = row["hard_errored_as_expected"]
            else:
                checks[f"{name}:converted"] = False
            results.append(row)
            continue
        # a config the gate EXPECTED to hard-error but which silently converted
        # FAILS (the converter must not approximate a non-convertible config).
        if expect_err is not None and corruption is None:
            row["hard_errored_as_expected"] = False
            checks[f"{name}:accept"] = False
            results.append(row)
            continue
        checks[f"{name}:converted"] = True

        # task-count provenance (the low-severity converter_faithful_to_v1
        # finding): record the INDEPENDENT v1-source task count (from the resolved
        # v1 stack), the converter-output task count, and the fixture's. For the
        # fixture-subset configs this turns "converter is MORE faithful than the
        # fixture" from a human note into a CHECKED fact: the converter output must
        # reproduce the FULL v1 task count (== v1) and carry at least as many tasks
        # as the (subset) fixture (>= fixture).
        try:
            with _quiet():
                v1_resolved = convert.resolve_stack(v1_full)
            row["v1_task_count"] = _count_v1_tasks(v1_resolved)
        except Exception as err:  # noqa: BLE001 - count failure is a per-config datum
            row["v1_task_count_error"] = f"{type(err).__name__}: {err}"
        row["conv_task_count"] = _count_v2_task_modules(cfg)
        if not is_today:
            row["fixture_task_count"] = _count_v2_fixture_task_modules(fix_full)

        # the labeller substitution (GN2X_qcdsplit qcdxx->qcdll): the converter
        # faithfully emits the v1 labeller classes; CV1 applies the SAME fixture
        # substitution at plan-compile so the converter output compiles in this
        # ftag build (the converter output is NOT edited).
        extra_sets: list[str] = []
        conv_lab = (
            cfg.get("data", {}).get("modules", {}).get("ftag_labeller", {}).get("init_args") or {}
        ).get("class_names")
        fix_lab = _fixture_labeller_classes(fix_full)
        if entry.get("labeller_override") and conv_lab and fix_lab and conv_lab != fix_lab:
            extra_sets = ["--set", f"data.modules.ftag_labeller.init_args.class_names={fix_lab}"]
            row["labeller_override_applied"] = {"v1_faithful": conv_lab, "compiled_with": fix_lab}

        # -- (2) validate + plan-compile (fit/test/onnx; data-free, no --strict)
        modes = ["fit", "test"] + (["onnx"] if entry["onnx"] == "validate" else [])
        val_sets, _ = _normaliser_overrides([conv_path], cdir)
        val_sets += _mup_shape_override([conv_path], val_sets, cdir)
        val_sets += extra_sets
        all_modes_ok = True
        for mode in modes:
            rc = _validate_rc([conv_path], mode, val_sets)
            row["validate"][mode] = rc
            mode_ok = rc == 0
            checks[f"{name}:validate:{mode}"] = mode_ok
            all_modes_ok = all_modes_ok and mode_ok

        # -- reproducible-today (✅) configs: NO fixture to plan-match against.
        # They are gated on conditions 1-3 (convert + validate + plan-compile)
        # only (the high-severity 29-vs-39 finding's fix option (a)). The
        # condition-4 fixture-plan parity is N/A (no fixture), and the GN2-family
        # v2-vs-v1 forward parity is already bitwise-gated by parity_gn2 +
        # gates_m2/m4 (see the report scope_note + the medium-finding deferral).
        if is_today:
            checks[f"{name}:accept"] = row["converted"] and all_modes_ok
            results.append(row)
            continue

        # -- (3) fixture-plan semantic equivalence ----------------------------
        try:
            conv_sig = _plan_signature([conv_path], cdir / "conv", extra_sets=extra_sets)
            fix_sig = _plan_signature(fix_full, cdir / "fix")
            conv_onnx = _onnx_manifest([conv_path], cdir / "conv", extra_sets=extra_sets)
            fix_onnx = _onnx_manifest(fix_full, cdir / "fix")
        except Exception as err:  # noqa: BLE001 - signature build failure is a datum
            row["signature_error"] = f"{type(err).__name__}: {err}"
            conv_sig = fix_sig = {}
            conv_onnx = fix_onnx = None

        mode_match: dict[str, bool] = {}
        for mode in fix_sig:
            c, f = conv_sig.get(mode, {}), fix_sig[mode]
            if "error" in c or "error" in f:
                mode_match[mode] = False
            else:
                mode_match[mode] = (
                    c["classes"] == f["classes"]
                    and c["edges"] == f["edges"]
                    and c["sinks"] == f["sinks"]
                )
        row["mode_match"] = mode_match
        eval_match = _eval_manifest(conv_sig.get("TEST", {})) == _eval_manifest(
            fix_sig.get("TEST", {})
        )
        onnx_match = conv_onnx == fix_onnx
        row["eval_manifest_match"] = eval_match
        row["onnx_manifest_match"] = onnx_match
        row["conv_onnx_manifest"] = conv_onnx
        row["fix_onnx_manifest"] = fix_onnx
        fixture_match = bool(mode_match) and all(mode_match.values()) and eval_match and onnx_match
        row["fixture_match"] = fixture_match

        is_subset = bool(entry.get("fixture_subset"))  # the 2 INTENTIONAL keeps
        is_restored = bool(entry.get("restored"))  # the 6 F2-restored configs
        # the STRUCTURAL-EQUIVALENCE proxy (module-CLASS multiset equal per shared
        # mode) — the faithfulness invariant for the RESTORED configs that diverge
        # from the converter only by topology-neutral cosmetics (verbose-v1 vs
        # short fixture head names, the embed.tracks==seq.x single-stream edge, the
        # fixture's PadMaskWriter mask column). A dropped module would fail this.
        struct = _structural_equivalence(conv_sig, fix_sig)
        row["structural_equivalence"] = struct
        structurally_equivalent = bool(struct) and all(struct.values())
        row["structurally_equivalent"] = structurally_equivalent

        # the task-count faithfulness fact: the converter output reproduces the
        # FULL v1 source task count and carries AT LEAST as many tasks as the
        # fixture. For a RESTORED config the fixture ALSO carries the full set, so
        # conv == v1 == fixture; for a KEEP it is conv == v1 >= fixture. A converter
        # (or fixture) regression that dropped a v1 task FAILS this check.
        v1n, convn, fixn = row["v1_task_count"], row["conv_task_count"], row["fixture_task_count"]
        task_counts_ok = (
            v1n is not None
            and convn is not None
            and fixn is not None
            and convn == v1n
            and convn >= fixn
        )
        row["task_counts_faithful"] = task_counts_ok

        # a config "reproduces the full v1 capability" when the converter output
        # reproduces the full v1 task count AND it is either an exact fixture plan
        # match OR structurally equivalent to the (now-full-fidelity) fixture. This
        # is the post-F2 assertion for the 6 RESTORED configs: capability restored,
        # not subset-divergent.
        reproduces_full_v1 = task_counts_ok and (fixture_match or structurally_equivalent)
        row["reproduces_full_v1_capability"] = reproduces_full_v1
        # "faithful to v1" = an exact fixture match, OR a restored config that
        # reproduces the full v1 capability, OR a documented keep whose converter
        # still reproduces the full v1 task count (the divergence is the fixture's,
        # by design).
        row["converter_faithful_to_v1"] = (
            fixture_match
            or (is_restored and reproduces_full_v1)
            or (is_subset and task_counts_ok)
        )

        # the per-config GATING check (three cases):
        #  - RESTORED configs (the 6 F2 restores): MUST convert + validate AND
        #    reproduce the FULL v1 capability (full task count + structural
        #    equivalence to the now-full-fidelity fixture) — NO longer
        #    subset-divergent. (Exact byte-for-byte fixture_match is recorded but
        #    not required: verbose-v1 head names / the embed==seq edge / the mask
        #    column are topology-neutral cosmetics.)
        #  - KEEP configs (the 2 intentional fixture-subset keeps): MUST convert +
        #    validate AND reproduce the full v1 task count, AND stay fixture-
        #    DIVERGENT (the documented keep — the fixture is the curated artifact);
        #    each carries a keep_reason naming WHY (asserted present).
        #  - all other fixture-bearing configs: MUST convert + validate + exactly
        #    plan-match the fixture.
        if is_restored:
            checks[f"{name}:accept"] = (
                row["converted"] and all_modes_ok and reproduces_full_v1
            )
            checks[f"{name}:reproduces_full_v1_capability"] = reproduces_full_v1
            checks[f"{name}:structurally_equivalent"] = structurally_equivalent
        elif is_subset:
            checks[f"{name}:accept"] = row["converted"] and all_modes_ok and task_counts_ok
            checks[f"{name}:intentional_keep_divergent"] = not fixture_match
            checks[f"{name}:reproduces_full_v1_tasks"] = task_counts_ok
            checks[f"{name}:keep_reason_documented"] = bool(entry.get("keep_reason"))
        else:
            checks[f"{name}:accept"] = row["converted"] and all_modes_ok and fixture_match
        results.append(row)

    # the fixture-subset KEEP set is PINNED — assert it is exactly the 2 documented
    # intentional keeps (event_classifier + regression_multi_target). A fixture
    # silently starting/stopping to match changes CV1's accounting.
    observed_subset = {r["name"] for r in results if r["fixture_subset"]}
    checks["fixture_subset_set_pinned"] = (
        observed_subset == set(_CV1_FIXTURE_SUBSET) and corruption is None
    ) or (corruption is not None)
    # the F2-RESTORED set is PINNED too — assert it is exactly the 6 restored
    # configs (a restored fixture silently regressing back to a subset would
    # change which configs the gate treats as faithful).
    observed_restored = {r["name"] for r in results if r["restored"]}
    checks["restored_set_pinned"] = (
        observed_restored == set(_CV1_RESTORED) and corruption is None
    ) or (corruption is not None)

    passed = all(checks.values())
    n_total = len(configs)
    n_converted = sum(1 for r in results if r["converted"])
    n_matched = sum(1 for r in results if r["fixture_match"])
    n_subset = len(observed_subset)
    n_flagged = sorted(observed_subset)
    n_restored = len(observed_restored)
    restored_flagged = sorted(observed_restored)
    n_today = sum(1 for r in results if r["today"])
    n_fixture = n_total - n_today  # needs-M5 + needs-M6 (have a hand-written fixture)
    today_names = sorted(r["name"] for r in results if r["today"])
    # the EXPECTED-hard-error configs (tutorial: non-standard vertex label the v2
    # module cannot map -> converter hard-errors per FD §10). They are ACCEPTED
    # via the expected hard-error, not convert-success, so they count toward the
    # 39 denominator but not toward n_converted.
    n_expected_err = sum(1 for r in results if r.get("expect_convert_error") and corruption is None)
    expected_err_names = sorted(
        r["name"] for r in results if r.get("expect_convert_error") and corruption is None
    )
    n_accepted = sum(1 for r in results if checks.get(f"{r['name']}:accept"))

    criterion = (
        "the M7 W1 converter acceptance over the FULL plan 13 / matrix §4 non-dropped denominator "
        f"({n_total} configs = {n_fixture} needs-M5/M6 with a hand-written v2 fixture + {n_today} "
        "reproducible-today ✅ configs with no fixture, all embedded verbatim). For EVERY "
        "convertible config salt2 convert-config on the v1 SOURCE (a) converts with NO hard-error "
        "AND (b) validates + plan-compiles fit/test/onnx (rc == 0, data-free non-strict, the "
        f"M5/M6-CONV precedent). The {n_fixture} fixture-bearing configs ADDITIONALLY (c) are "
        "SEMANTICALLY EQUIVALENT to the hand-written fixture (same compiled plan: class-canonical "
        "module set, edges, sinks, eval column manifest, ONNX output manifest). The 10 ✅ configs "
        "(matrix §5.3: bake class_names + origin_label, use_class_dict->weight_source, "
        "num_inputs->truncate, input_map->groups, callbacks list->dict) have NO fixture and are "
        f"gated on convert+validate+plan-compile only (cond. 1-3; the closed 29-vs-39 denominator "
        f"gap). {n_restored} fixture configs ({', '.join(restored_flagged)}) were RESTORED in Wave "
        "F2 to full v1 fidelity (the fixture now reproduces the full v1 task/stream/capability "
        "set) — they are asserted to REPRODUCE THE FULL V1 CAPABILITY (CHECKED: conv_task_count == "
        "v1_task_count == fixture_task_count AND the converter is STRUCTURALLY EQUIVALENT to the "
        "fixture — module-class multiset equal in every shared mode), no longer subset-divergent; "
        "the residual converter-vs-fixture differences are topology-neutral cosmetics (verbose-v1 "
        "vs short head names, the embed.tracks==seq.x single-stream edge, the PadMaskWriter mask "
        f"column). {n_subset} fixture configs ({', '.join(n_flagged)}) remain INTENTIONAL KEEPS — "
        "the hand-written fixture is the faithful v2 artifact and is NOT a 1:1 converter "
        "reproduction by design (event_classifier: the converter wires FIT/VAL metric sinks the "
        "fixture omits — an export-omission correct by design; regression_multi_target: the fixture "
        "renames the v1 task and is the curated F1a MultiTarget artifact). Each keep is gated on "
        "convert+validate + the full-v1-task-count fact + stays fixture-DIVERGENT, with a "
        f"keep_reason naming WHY; the keep set is PINNED + audited. {n_expected_err} "
        f"config(s) ({', '.join(expected_err_names)}) are EXPECTED-hard-error (a non-standard "
        "vertex label the v2 VertexingTaskModule invariant cannot map -> the converter correctly "
        "HARD-ERRORS per FD §10; counted in the 39 denominator, gated on the EXPECTED hard-error "
        "not convert-success; W4 retires/rewrites its docs config against v2). "
        f"{n_accepted}/{n_total} accepted (converted+validated OR the expected hard-error). Dipz "
        "is the matrix §1 conditional drop (converter still runs on it). NOTE: matrix §4 condition "
        "4's numerical forward-parity spot-check + the v1==v2 weight-parity test are NOT in CV1 — "
        "they are explicitly DEFERRED to a later wave "
        "(see deferred_forward_parity below): the v2-vs-v1 forward parity for every CV1 family is "
        "already bitwise-gated by parity_gn2 + gates_m2/m4/m5/m6, and CV1 gates the converter via "
        "plan + manifest equivalence to those v2 fixtures."
    )
    report = _base_report(
        "cv1_converter_acceptance",
        passed,
        criterion,
        {
            "total_configs": n_total,
            "denominator": n_total,
            "fixture_bearing_configs": n_fixture,
            "reproducible_today_configs": n_today,
            "reproducible_today_names": today_names,
            "converted": n_converted,
            "accepted": n_accepted,
            "expected_hard_error_configs": n_expected_err,
            "expected_hard_error_names": expected_err_names,
            "fixture_plan_matched": n_matched,
            "fixture_subset_flagged": n_subset,
            "fixture_subset_configs": n_flagged,
            "restored_flagged": n_restored,
            "restored_configs": restored_flagged,
            "missing_config_files": missing,
            "strict": False,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["configs"] = results
    report["cv1_config_names"] = [e["name"] for e in _CV1_CONFIGS]
    report["no_strict_rationale"] = _NO_STRICT_RATIONALE
    report["v1_source_root"] = str(V1_CONFIG_DIR)
    report["fixture_root"] = str(CONFIG_DIR)
    report["scope_note"] = (
        "CV1 is the convert+validate+plan-compile (+fixture-plan-parity for the fixture-bearing "
        "configs) acceptance for the converter (salt.core.convert, M7 W1) over the FULL 39-config "
        f"plan 13 / matrix §4 non-dropped denominator ({n_total} configs: {n_fixture} needs-M5/M6 "
        f"with a hand-written v2 fixture + {n_today} reproducible-today ✅ configs, no fixture). "
        "The 10 ✅ configs (gn2v1-opendata, GN2, GN2_extended, GN2_charded_neutral_loose_aux, "
        "GN2X, GN2XTau, flow, tutorial, TPLTmu, PLITel) close the high-severity 29-vs-39 gap "
        "(the gate previously scoped CV1 to the 29 fixture-bearing configs only): they have no "
        "M5/M6 fixture row but still require the converter for the mechanical conversion details "
        "that recur everywhere (matrix §5.3) and are gated on convert+validate+plan-compile "
        "(cond. 1-3); the global flavour_label head's class_names are supplied via --class-names "
        "(v1 autodiscovers them from h5 attrs, cli.py:get_object_class_names — the converter "
        "hard-errors without data, so CV1 bakes the canonical FTAG names per cardinality; "
        "TPLTmu/PLITel declare class_names inline). tutorial.yaml is the one EXPECTED-hard-error "
        "✅ config: its const_vertexing uses the non-standard `truth_vertex_idx` label the v2 "
        "VertexingTaskModule cannot map, so the converter correctly hard-errors per FD §10 (gated "
        "on the expected hard-error; W4 retires/rewrites its docs config against v2). The "
        "semantic-equivalence comparison (fixture-bearing configs) is CLASS-CANONICAL (instance "
        "names normalised to module classes): the converter derives task-head instance names from "
        "v1 task names while the hand-written fixtures pick short names, so the GRAPH is compared, "
        "not the labels. Wave F2 RESTORED 6 previously-subset fixtures (regression, "
        "regression_gaussian, regression_weighted, nan_regression, GN3V01, GN2XE) to FULL v1 "
        "fidelity — for these CV1 asserts they REPRODUCE THE FULL V1 CAPABILITY: conv_task_count == "
        "v1_task_count == fixture_task_count AND the converter is STRUCTURALLY EQUIVALENT to the "
        "fixture (the module-class multiset matches in every shared mode). The residual converter-"
        "vs-fixture differences are topology-neutral cosmetics — verbose-v1 vs short head names "
        "(class-canonical sink keys), the embed.tracks==seq.x edge for a single-stream Concat, the "
        "fixture's hand-set PadMaskWriter mask column — so a byte-exact fixture_match is recorded "
        "but NOT required; a dropped module would break the structural-equivalence check. The 2 "
        "remaining INTENTIONAL KEEPS (event_classifier: the converter wires FIT/VAL metric sinks "
        "the fixture omits by design; regression_multi_target: the fixture renames the v1 task and "
        "is the curated F1a MultiTarget artifact) are gated on convert+validate + the full-v1-task-"
        "count fact + staying fixture-DIVERGENT, each carrying a keep_reason; the keep set + the "
        "restored set are both PINNED so the accounting is audited. GN2X_qcdsplit: the v1 labeller "
        "class qcdxx is absent from ftag "
        "v0.2.17 (matrix §2) — the converter FAITHFULLY emits qcdxx; CV1 substitutes qcdll at "
        "plan-compile (the same workaround the fixture makes) so the converter output compiles in "
        "this build (the converter output is NOT edited)."
    )
    report["deferred_forward_parity"] = (
        "DEFERRED (explicitly, plan 13 CV1 review + the medium-severity critic finding): matrix §4 "
        "condition 4's NUMERICAL forward-parity spot-check (fixed seed, torch-math, ≤1e-6) for one "
        "representative per family, and the v1-injected-weights == v2-materialised-weights parity "
        "test, are NOT implemented in CV1. CV1 instead asserts class-canonical PLAN equivalence "
        "(module-class set, edges, sinks, eval manifest, ONNX manifest) between the converter "
        "output and the hand-written fixture — a strong structural proxy — plus the v1-task-count "
        "faithfulness fact. The deferral is sound because (1) the v2-fixture-vs-independent-v1 "
        "forward parity is ALREADY bitwise-gated (torch.equal) for every CV1 family: parity_gn2 "
        "(GN2 ModelWrapper.forward), gates_m5 (regression/dips encoder-less pool/GLS/hybrid-norm/"
        "vconcat), gates_m6 (labeller/muP/edges) — so a converter that wires the right graph to "
        "the M5/M6 fixtures inherits their proven numerical parity; (2) a per-config converter "
        "forward harness needs a bound reader (a data source) and a bespoke synthetic batch per "
        "topology (multi-stream GN3, MaskFormer query bank, edge features, muP infshapes) — a "
        "substantial new harness that is a later-wave deliverable, not a W1 fix. The residual gap "
        "plan-equivalence cannot catch (right graph, wrong init arg — e.g. a transposed dense "
        "width that survives plan-compile) is the explicit deferral; not silently claimed covered."
    )

    # -- stdout table ----------------------------------------------------------
    print(
        f"{'config':<24}{'tier':<6}{'conv':>5}{'fit':>5}{'test':>6}{'onnx':>6}"
        f"{'plan==fix':>11}  flags"
    )
    for r in results:
        v = r["validate"]
        fit = "PASS" if v.get("fit") == 0 else ("-" if "fit" not in v else "FAIL")
        test = "PASS" if v.get("test") == 0 else ("-" if "test" not in v else "FAIL")
        onnx = "PASS" if v.get("onnx") == 0 else ("na" if "onnx" not in v else "FAIL")
        flags = []
        if r["today"]:
            flags.append("today")
        if r["restored"]:
            flags.append("RESTORED")
        if r["fixture_subset"]:
            flags.append("KEEP")
        if r["conditional_drop"]:
            flags.append("cond-drop")
        if r.get("labeller_override_applied"):
            flags.append("lab-ovr")
        if r.get("expect_convert_error"):
            flags.append("hard-err-OK")
        # expected-hard-error configs (tutorial) show HERR in the conv column;
        # ✅ today configs have NO fixture -> plan==fix is N/A (gated on cond. 1-3)
        if r.get("expect_convert_error"):
            conv_col = "HERR" if r.get("hard_errored_as_expected") else "FAIL"
        else:
            conv_col = "PASS" if r["converted"] else "FAIL"
        # plan==fix: n/a for ✅ today configs (no fixture); for a RESTORED/KEEP
        # config that is structurally equivalent (full capability) but not byte-
        # exact, show STRUCT (the topology-neutral-cosmetic divergence); PASS for
        # an exact match, DIFF for an unexpected (gating) divergence.
        if r["today"]:
            plan_col = "n/a"
        elif r["fixture_match"]:
            plan_col = "PASS"
        elif r.get("structurally_equivalent") and (r["restored"] or r["fixture_subset"]):
            plan_col = "STRUCT"
        else:
            plan_col = "DIFF"
        print(
            f"{r['name']:<24}{r['tier']:<6}"
            f"{conv_col:>5}"
            f"{fit:>5}{test:>6}{onnx:>6}"
            f"{plan_col:>11}  {','.join(flags)}"
        )
    print(
        f"\n{n_accepted}/{n_total} accepted (denominator {n_total} = {n_fixture} fixture-bearing + "
        f"{n_today} ✅ today; {n_converted} converted+validated, {n_expected_err} expected-hard-"
        f"error: {expected_err_names}), {n_matched}/{n_fixture} exact-plan-match the fixture, "
        f"{n_restored} F2-RESTORED to full v1 capability: {restored_flagged}, "
        f"{n_subset} intentional fixture keeps (divergent by design): {n_flagged}"
    )
    if missing:
        print(f"MISSING files: {missing}")
    _print_verdict("cv1", passed, criterion, _emit_report(report, outdir, "cv1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CV2 — the converter hard-error path (the bit-rotted identifier)
# ---------------------------------------------------------------------------

# The AUTHORITATIVE CV2 list: the §5.1 bit-rotted (5) + parked (1) v1 configs
# that MUST hard-error in salt2 convert-config (FD §10 / matrix §5.1), embedded
# verbatim. Each carries the marker substring the ConvertError message MUST
# contain (the removed v1 class / R21Xbb) — proving the converter identifies the
# config as dropped, never approximates.
_CV2_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "GN2_open_data",
        "v1": ("GN2/GN2_open_data.yaml",),
        "kind": "bit-rotted",
        "marker": "salt.models.TransformerEncoder",
        "note": "bit-rotted: removed TransformerEncoder/ScaledDotProductAttention (matrix §5.1)",
    },
    {
        "name": "GN2_tracks_neutral_SA_aux",
        "v1": ("GN2/GN2_tracks_neutral_SA_aux.yaml",),
        "kind": "bit-rotted",
        "marker": "salt.models.TransformerEncoder",
        "note": "bit-rotted: removed TransformerEncoder/ScaledDotProductAttention (matrix §5.1)",
    },
    {
        "name": "GN2_tracks_neutral_CA",
        "v1": ("GN2/GN2_tracks_neutral_CA.yaml",),
        "kind": "bit-rotted",
        "marker": "salt.models.TransformerCrossAttentionEncoder",
        "note": "bit-rotted: removed TransformerCrossAttentionEncoder (no v2 module, matrix §5.1)",
    },
    {
        "name": "GN2_dR",
        "v1": ("GN3_dev/GN2_dR.yaml",),
        "kind": "bit-rotted",
        "marker": "salt.models.TransformerEncoder",
        "note": "bit-rotted: removed TransformerEncoder (matrix §5.1)",
    },
    {
        "name": "SubjetXbb",
        "v1": ("legacy/SubjetXbb.yaml",),
        "kind": "bit-rotted",
        "marker": "salt.models.TransformerEncoder",
        "note": "bit-rotted: removed TransformerEncoder/ScaledDotProductAttention (matrix §5.1)",
    },
    {
        "name": "Baseline_Xbb",
        "v1": ("legacy/Baseline_Xbb.yaml",),
        "kind": "parked",
        "marker": "salt.models.R21Xbb",
        "note": "parked: R21Xbb MLP, no v2 module (design §10, matrix §5.1)",
    },
)

# a HEALTHY v1 config the corruption hook swaps in to prove CV2's "must hard-
# error" check is non-vacuous (it converts cleanly, so the gate goes red).
_CV2_HEALTHY_PROBE = ("GN3X.yaml",)


def run_cv2(
    outdir: Path | str,
    *,
    corruption: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """CV2: ``salt2 convert-config`` HARD-ERRORS on the bit-rotted/parked configs.

    For EVERY config in the AUTHORITATIVE ``_CV2_CONFIGS`` (the 5 bit-rotted + 1
    parked §5.1 configs, embedded verbatim), CV2 runs ``salt2 convert-config`` on
    the v1 source and asserts it raises a ``ConvertError`` whose message (a)
    NAMES the config as dropped (the config name appears) AND (b) contains the
    expected removed/parked v1-class marker (FD §10: the converter mechanically
    identifies the bit-rotted configs, never approximates). It also asserts the
    error is a ``ConvertError`` (the clean one-block form), not an arbitrary
    exception.

    Negative control (``test_gates_m7.py`` + the ``corruption`` hook): a dropped
    config that SILENTLY converted would FAIL the gate. The hook repoints a CV2
    entry's ``v1`` at a HEALTHY config (``GN3X.yaml``, which converts cleanly) —
    the "must hard-error" assertion for that entry then flips False and the gate
    goes red, proving the check has teeth.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("CV2 convert-config HARD-ERROR path — bit-rotted/parked configs (the dropped identifier)")
    print("=" * 96)

    configs: list[dict[str, Any]] = [dict(e) for e in _CV2_CONFIGS]
    if corruption is not None:
        configs = corruption(configs)

    checks: dict[str, bool] = {}
    results: list[dict[str, Any]] = []

    # -- every v1 source file exists on disk ----------------------------------
    missing = [
        f"{e['name']}:{v}"
        for e in configs
        for v in e.get("v1", ())
        if not (V1_CONFIG_DIR / v).is_file()
    ]
    checks["all_cv2_config_files_present"] = missing == []

    for entry in configs:
        name = entry["name"]
        v1_full = [V1_CONFIG_DIR / p for p in entry.get("v1", ())]
        row: dict[str, Any] = {
            "name": name,
            "kind": entry["kind"],
            "v1_stack": list(entry.get("v1", ())),
            "marker": entry["marker"],
            "hard_errored": False,
            "is_convert_error": False,
            "names_config": False,
            "has_marker": False,
            "error_message": None,
            "note": entry["note"],
        }
        if not all(p.is_file() for p in v1_full):
            checks[f"{name}:hard_errors"] = False
            results.append(row)
            continue
        try:
            with _quiet():
                convert.convert_config(v1_full)
            # NO exception -> the converter silently converted a dropped config
            row["error_message"] = "(no error — converter SILENTLY converted a dropped config)"
        except ConvertError as err:
            msg = str(err)
            row["hard_errored"] = True
            row["is_convert_error"] = True
            row["names_config"] = name in msg
            row["has_marker"] = entry["marker"] in msg
            row["error_message"] = msg
        except Exception as err:  # noqa: BLE001 - a non-ConvertError is NOT the named hard-error
            row["error_message"] = f"(unexpected {type(err).__name__}: {err})"

        ok = (
            row["hard_errored"]
            and row["is_convert_error"]
            and row["names_config"]
            and row["has_marker"]
        )
        checks[f"{name}:hard_errors"] = ok
        results.append(row)

    passed = all(checks.values())
    n_total = len(configs)
    n_hard = sum(1 for r in results if r["hard_errored"] and r["is_convert_error"])

    criterion = (
        "the M7 W1 converter hard-error path (plan 13 CV2; FD §10, matrix §5.1): EVERY bit-rotted "
        f"+ parked §5.1 v1 config ({n_total} configs, embedded verbatim) makes salt2 "
        "convert-config raise a ConvertError (never an approximation) whose message NAMES the "
        "config as dropped AND contains the expected removed/parked v1-class marker "
        "(TransformerEncoder / ScaledDotProductAttention / TransformerCrossAttentionEncoder / "
        "R21Xbb). A dropped config that silently converted FAILS the gate (the negative control)."
    )
    report = _base_report(
        "cv2_converter_hard_error",
        passed,
        criterion,
        {
            "total_configs": n_total,
            "hard_errored": n_hard,
            "missing_config_files": missing,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["configs"] = results
    report["cv2_config_names"] = [e["name"] for e in _CV2_CONFIGS]
    report["v1_source_root"] = str(V1_CONFIG_DIR)
    report["scope_note"] = (
        "CV2 proves the converter mechanically identifies the bit-rotted/parked configs (FD §10): "
        "it HARD-ERRORS with a ConvertError naming the config + the removed/parked v1 class, never "
        "approximating. The 5 bit-rotted configs reference v1 classes REMOVED from salt.models "
        "(they fail jsonargparse on v1 main today, so there is no working v1 behaviour to "
        "reproduce); Baseline_Xbb is the parked R21Xbb MLP (design §10). The negative control (a "
        "dropped config that silently converted -> gate red) is exercised by the corruption hook "
        "swapping a CV2 entry for a HEALTHY config (GN3X.yaml)."
    )

    # -- stdout table ----------------------------------------------------------
    print(f"{'config':<28}{'kind':<12}{'hard-err':>9}{'ConvErr':>9}{'names':>7}{'marker':>8}")
    for r in results:
        print(
            f"{r['name']:<28}{r['kind']:<12}"
            f"{'YES' if r['hard_errored'] else 'NO':>9}"
            f"{'YES' if r['is_convert_error'] else 'NO':>9}"
            f"{'YES' if r['names_config'] else 'NO':>7}"
            f"{'YES' if r['has_marker'] else 'NO':>8}"
        )
    print(f"\n{n_hard}/{n_total} hard-errored as a ConvertError")
    _print_verdict("cv2", passed, criterion, _emit_report(report, outdir, "cv2"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CVF — converter-output target-producer fidelity (the plan-compile MISS)
# ---------------------------------------------------------------------------

# The two CV1 configs whose hand-written v2 fixture is an INTENTIONAL keep that
# is NOT a 1:1 converter reproduction — they are excluded from any
# converter-vs-FIXTURE equality in CVF (they are documented fixture exceptions,
# NOT converter-fidelity failures). They are NAMED here explicitly (the gate
# pins them) and asserted to be the curated fixture exceptions, never silently
# dropped. CRUCIALLY: regression_multi_target stays on this list (its fixture
# renames the v1 task and is therefore not byte-equal to the converter) BUT it
# MUST still pass CVF check (i) — F1a now translates the v1 `multi_target` block
# into a concrete MultiTarget data module that PRODUCES the `pt_label_handle`
# custom target the regression head consumes, so the head's target is no longer
# a phantom satisfied only by the demand-driven `Labels` wildcard.
_CVF_FIXTURE_EXCEPTIONS: tuple[str, ...] = ("event_classifier", "regression_multi_target")

# v2 data-module processor classes that are CONCRETE (non-wildcard) producers of
# synthetic label/target handles: a `custom_target` created by MultiTarget, a
# per-object MaskFormer target produced by MaskFormerTargets, etc. A synthetic
# target produced by one of these (allow_wildcards == False) is GENUINELY served;
# a synthetic target served ONLY by the `Labels` wildcard (allow_wildcards ==
# True, which narrows `labels.**` to ANY demanded key — it never proves the
# column exists) is the plan-compile MISS this gate exists to catch.
_CVF_CONCRETE_LABEL_PRODUCERS: tuple[str, ...] = (
    "MultiTarget",
    "MaskFormerTargets",
    "MaskDecoder",
)


def _v1_synthetic_targets(v1_resolved: Mapping[str, Any]) -> set[str]:
    """The v1-declared SYNTHETIC label handles a converted task must NOT phantom.

    These are label keys that do NOT exist as a dataset column and therefore
    REQUIRE a concrete data-module producer in the converted config — the
    demand-driven `Labels` wildcard would silently "produce" them at plan-compile
    (narrowing ``labels.**`` to any demanded key) while no real column backs them,
    which is exactly the MISS that let the broken (pre-F1a) regression_multi_target
    pass CV1. Derived INDEPENDENTLY from the v1 source (not the converter output)
    so a converter that DROPS the producer is caught:

    - v1 ``data.multi_target`` rules' ``custom_target`` outputs — the
      ``pt_label_handle`` family (v1 datasets.py:648-693 NaN-placeholder columns,
      created on the fly, never read from disk).

    Returns
    -------
    set[str]
        Dotted ``labels.<stream>.<handle>`` keys that must be concretely produced.
    """
    out: set[str] = set()
    data = v1_resolved.get("data") or {}
    for rule in data.get("multi_target") or ():
        if not isinstance(rule, Mapping):
            continue
        custom = rule.get("custom_target")
        stream = rule.get("input_name")
        if custom is not None and stream is not None:
            out.add(f"labels.{stream}.{custom}")
    return out


def _v1_task_signature(v1_resolved: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The v1 resolved task-NAME set + the stream set its tasks read.

    Reads the inner ``SaltModel`` heads (the path ``convert_stack`` reads,
    convert.py) — the converter preserves the v1 task ``name`` verbatim as the v2
    instance name, so the two name sets must coincide for a faithful conversion.
    A ``mask_decoder`` block counts as the ``mask_decoder`` head (the converter's
    instance name for it).

    Returns
    -------
    tuple[list[str], list[str]]
        ``(sorted task names, sorted stream names)``.
    """
    inner = ((v1_resolved.get("model") or {}).get("model") or {}).get("init_args", {}) or {}
    tasks = (inner.get("tasks") or {}).get("init_args", {}).get("modules", []) or []
    names: list[str] = []
    streams: set[str] = set()
    for t in tasks:
        init = t.get("init_args") or {}
        if init.get("name") is not None:
            names.append(str(init["name"]))
        if init.get("input_name") is not None:
            streams.add(str(init["input_name"]))
    if inner.get("mask_decoder") is not None:
        names.append("mask_decoder")
        streams.add("objects")
    return sorted(names), sorted(streams)


def _conv_task_signature(cfg: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The converted v2 task-NAME set + the stream set its task modules read.

    The task instance names are the ``model.init_args.modules`` keys whose class
    is a v2 head (``*TaskModule`` / ``MaskDecoder``); the streams are their
    ``init_args.stream`` (the v1 ``input_name``).

    Returns
    -------
    tuple[list[str], list[str]]
        ``(sorted task instance names, sorted stream names)``.
    """
    mods = (cfg.get("model") or {}).get("init_args", {}).get("modules", {}) or {}
    names: list[str] = []
    streams: set[str] = set()
    for name, mod in mods.items():
        if not isinstance(mod, Mapping):
            continue
        if any(str(mod.get("class_path", "")).endswith(s) for s in _V2_TASK_CLASS_SUFFIXES):
            names.append(name)
            stream = (mod.get("init_args") or {}).get("stream")
            if stream is not None:
                streams.add(str(stream))
    return sorted(names), sorted(streams)


def _fit_target_producers(
    conv_path: Path, outdir: Path, *, extra_sets: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Per task target key, the set of producing modules in the converted FIT plan.

    Compiles the converted config's FIT plan (the SAME data-free machinery as
    `_plan_signature` / `_validate_rc` — norm overrides + muP shapes), then for
    every ``labels.*`` / ``objects.*`` / ``masks.*`` key consumed by a TASK head
    (``*TaskModule`` / ``MaskDecoder``) records each producer module's CLASS and
    whether that producer is a demand-driven WILDCARD producer (``allow_wildcards``
    — the `Labels` ``labels.**`` producer, which never proves a real column backs
    the narrowed key). A target served ONLY by a wildcard producer has NO concrete
    backing — the plan-compile MISS.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{target_key: {"producers": [class, ...], "wildcard_only": bool,
        "consumers": [class, ...]}}`` for every target a FIT task consumes, or a
        single ``{"__error__": "..."}`` entry on a compile failure.
    """
    sets, _ = _normaliser_overrides([conv_path], outdir)
    sets += _mup_shape_override([conv_path], sets, outdir)
    if extra_sets:
        sets += extra_sets
    bare = [s for s in sets if s != "--set"]
    try:
        with _quiet():
            gc = load_config([str(conv_path)], bare)
            fit_modes = [m for m in gc.sinks if m.name == "FIT"]
            if not fit_modes:
                return {"__error__": "converted config declares no FIT mode"}
            mode = fit_modes[0]
            plan = compile_plan(
                gc.modules,
                mode,
                gc.sources,
                schema=gc.schema,
                sinks=gc.sinks,
                sink_origins=gc.sink_origins.get(mode),
            )
    except Exception as err:  # noqa: BLE001 - a FIT compile failure is the per-config datum
        return {"__error__": f"{type(err).__name__}: {err}"}

    def _is_task(node: str) -> bool:
        mod = gc.modules.get(node)
        return mod is not None and any(
            type(mod).__name__.endswith(s) for s in _V2_TASK_CLASS_SUFFIXES
        )

    # the keys each task head consumes (its label/target demand)
    target_prefixes = ("labels.", "objects.", "masks.", "targets.")
    consumers: dict[str, set[str]] = {}
    for e in plan.edges:
        if e.key.startswith(target_prefixes) and e.consumer in gc.modules and _is_task(e.consumer):
            consumers.setdefault(e.key, set()).add(type(gc.modules[e.consumer]).__name__)
    out: dict[str, dict[str, Any]] = {}
    for key, cons in consumers.items():
        producers: list[str] = []
        wildcard_flags: list[bool] = []
        for e in plan.edges:
            if e.key != key:
                continue
            if e.producer in gc.modules:
                pmod = gc.modules[e.producer]
                producers.append(type(pmod).__name__)
                wildcard_flags.append(bool(getattr(pmod, "allow_wildcards", False)))
            else:  # a SOURCES sentinel — a framework-provided concrete leaf
                producers.append(e.producer)
                wildcard_flags.append(False)
        out[key] = {
            "producers": sorted(set(producers)),
            # a target is "wildcard_only" when it has ≥1 producer and EVERY
            # producer is a demand-driven wildcard (the Labels labels.** narrow) —
            # no concrete data module proves the key is real. No producer at all is
            # impossible here (compile_plan raises on a missing producer), so this
            # is the meaningful failure surface.
            "wildcard_only": bool(producers) and all(wildcard_flags),
            "consumers": sorted(cons),
        }
    return out


def run_cvf(
    outdir: Path | str,
    *,
    corruption: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """CVF: every converted task's target is PRODUCED by a data module (no phantoms).

    Over the SAME CV1 denominator (``_CV1_CONFIGS``, the convertible configs —
    the expected-hard-error ``tutorial`` is skipped, it has no converter output),
    CVF reuses CV1's converter-run + v1-resolution machinery and asserts:

    (i) **target producer fidelity** — for EVERY task in the converted FIT plan,
        every label/target it consumes has a producer, AND every v1-declared
        SYNTHETIC handle (a ``multi_target`` ``custom_target`` — a column that
        does NOT exist on disk) is produced by a CONCRETE (non-wildcard) data
        module (``MultiTarget`` / ``MaskFormerTargets`` / ...), NOT served only by
        the demand-driven ``Labels`` ``labels.**`` wildcard. This is the
        plan-compile + ``salt2 graph validate`` MISS that let the broken (pre-F1a)
        ``regression_multi_target`` pass CV1: the wildcard narrows ``labels.**`` to
        ANY demanded key, so a regression head consuming a never-produced
        ``pt_label_handle`` compiled clean while no real column backed it.

    (ii) **converter-v1 faithfulness** — for configs INTENDED to be converter-v1
        faithful (all CV1 configs except the named fixture exceptions), the
        converter's task-NAME set and stream set equal the v1 RESOLVED set (the
        converter preserves v1 task names verbatim; a dropped/renamed task is
        caught against the v1 source, independently of the fixture).

    (iii) **named fixture exceptions** — ``event_classifier`` and
        ``regression_multi_target`` are NAMED explicitly (``_CVF_FIXTURE_EXCEPTIONS``)
        and excluded from converter-vs-FIXTURE equality (they are fixture keeps,
        NOT converter-fidelity failures). BUT they are NOT excused from check (i):
        ``regression_multi_target`` MUST produce its ``pt_label_handle`` target
        concretely (F1a). The exception set is PINNED (== the configs flagged
        fixture-subset whose fixture renames/curates the converter output).

    Negative control (``test_gates_m7.py`` + the in-gate ``corruption`` hook,
    test-only, never on the CLI): a converter output with a task whose synthetic
    target has NO concrete producer (the hook DELETES the ``MultiTarget`` data
    module from one config's converted dict) — the target then falls back to the
    ``Labels`` wildcard (plan-compile still SUCCEEDS, the MISS), so its
    ``wildcard_only`` flag flips True and CVF goes red, proving check (i) is not
    vacuous.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("CVF converter-output target-producer fidelity — every task target has a real producer")
    print("=" * 96)

    checks: dict[str, bool] = {}
    results: list[dict[str, Any]] = []

    # the named fixture exceptions must be in the CV1 config set AND must be
    # EXACTLY the pinned CV1 intentional-keep (fixture_subset) set — a typo, a
    # fixture silently made faithful, or a keep silently dropped changes the
    # accounting. Post-Wave-F2 these two sets coincide: the 2 CV1 keeps
    # (event_classifier, regression_multi_target) ARE the CVF fixture exceptions.
    cv1_names = {e["name"] for e in _CV1_CONFIGS}
    checks["fixture_exceptions_pinned"] = (
        set(_CVF_FIXTURE_EXCEPTIONS) <= cv1_names
        and set(_CVF_FIXTURE_EXCEPTIONS) == set(_CV1_FIXTURE_SUBSET)
    )

    for entry in _CV1_CONFIGS:
        name = entry["name"]
        # the expected-hard-error config (tutorial) has NO converter output to
        # check targets on — it is correctly absent (CV1 gates its hard-error).
        if entry.get("expect_convert_error"):
            continue
        cdir = outdir / name
        cdir.mkdir(parents=True, exist_ok=True)
        v1_full = [V1_CONFIG_DIR / p for p in entry.get("v1", ())]
        fix_full = [CONFIG_DIR / p for p in entry.get("fix", ())]
        is_today = bool(entry.get("today"))
        is_exception = name in _CVF_FIXTURE_EXCEPTIONS
        row: dict[str, Any] = {
            "name": name,
            "tier": entry["tier"],
            "today": is_today,
            "fixture_exception": is_exception,
            "converted": False,
            "v1_synthetic_targets": [],
            "task_target_producers": {},
            "synthetic_targets_concretely_produced": None,
            "all_targets_have_producer": None,
            "converter_v1_task_match": None,
            "converter_v1_stream_match": None,
            "v1_tasks": [],
            "conv_tasks": [],
        }
        if not all(p.is_file() for p in v1_full):
            checks[f"{name}:targets_produced"] = False
            results.append(row)
            continue

        # -- convert (reuse CV1's class_names sourcing) -----------------------
        class_names = (
            dict(entry.get("class_names") or {}) if is_today else _fixture_class_names(fix_full)
        )
        try:
            with _quiet():
                cfg = convert.convert_config(v1_full, class_names=class_names or None)
                v1_resolved = convert.resolve_stack(v1_full)
        except ConvertError as err:
            row["convert_error"] = f"{type(err).__name__}: {err}"
            checks[f"{name}:targets_produced"] = False
            results.append(row)
            continue
        # the test-only corruption hook mutates the converted config dict (e.g.
        # strips the MultiTarget producer) to prove the gate has teeth.
        if corruption is not None:
            cfg = corruption(cfg) if name == "regression_multi_target" else cfg
        row["converted"] = True
        conv_path = cdir / f"{name}__converted.yaml"
        conv_path.write_text(convert._yaml_dump(cfg))  # noqa: SLF001 - same-package dumper

        # -- the GN2X_qcdsplit labeller substitution (same as CV1) ------------
        extra_sets: list[str] = []
        conv_lab = (
            cfg.get("data", {}).get("modules", {}).get("ftag_labeller", {}).get("init_args") or {}
        ).get("class_names")
        fix_lab = _fixture_labeller_classes(fix_full)
        if entry.get("labeller_override") and conv_lab and fix_lab and conv_lab != fix_lab:
            extra_sets = ["--set", f"data.modules.ftag_labeller.init_args.class_names={fix_lab}"]

        # -- (i) target producer fidelity ------------------------------------
        synth = sorted(_v1_synthetic_targets(v1_resolved))
        row["v1_synthetic_targets"] = synth
        producers = _fit_target_producers(conv_path, cdir, extra_sets=extra_sets)
        if "__error__" in producers:
            row["fit_compile_error"] = producers["__error__"]
            checks[f"{name}:targets_produced"] = False
            results.append(row)
            continue
        row["task_target_producers"] = producers
        # every consumed target has a producer (compile_plan guarantees this, but
        # record it as an explicit check so a future plan that admits an orphan
        # target — e.g. a writers-only sink with no producer — would be caught).
        all_have_producer = all(bool(p["producers"]) for p in producers.values())
        row["all_targets_have_producer"] = all_have_producer
        # every v1-declared synthetic handle a task consumes is produced by a
        # CONCRETE (non-wildcard) data module — NOT served only by the Labels
        # wildcard. (A synthetic handle no task consumes won't appear in the FIT
        # plan; the converter would have dropped both the producer and consumer,
        # which the check-(ii) task match catches.)
        synth_in_plan = [k for k in synth if k in producers]
        synth_concrete = all(
            not producers[k]["wildcard_only"]
            and any(c in _CVF_CONCRETE_LABEL_PRODUCERS for c in producers[k]["producers"])
            for k in synth_in_plan
        )
        # a synthetic target the v1 source declares MUST surface in the converted
        # FIT plan (a task consumes it) — if it vanished entirely, the converter
        # dropped the task that needed it. Require each declared synthetic key to
        # be present AND concretely produced.
        synth_all_present = set(synth_in_plan) == set(synth)
        synthetic_ok = synth_concrete and synth_all_present
        row["synthetic_targets_concretely_produced"] = synthetic_ok
        row["synthetic_targets_in_plan"] = synth_in_plan
        # no task target is served ONLY by the demand-driven wildcard among the
        # synthetic set (the phantom). Real dataset columns (flavour_label,
        # ftagTruthOriginLabel) are legitimately wildcard-served and NOT flagged.
        checks[f"{name}:targets_produced"] = all_have_producer and synthetic_ok

        # -- (ii) converter-v1 task/stream faithfulness ----------------------
        v1_names, v1_streams = _v1_task_signature(v1_resolved)
        conv_names, conv_streams = _conv_task_signature(cfg)
        row["v1_tasks"] = v1_names
        row["conv_tasks"] = conv_names
        row["v1_streams"] = v1_streams
        row["conv_streams"] = conv_streams
        task_match = conv_names == v1_names
        stream_match = conv_streams == v1_streams
        row["converter_v1_task_match"] = task_match
        row["converter_v1_stream_match"] = stream_match
        # the named fixture exceptions are excluded from converter-vs-FIXTURE
        # equality, NOT from converter-vs-V1 faithfulness (the converter is still
        # faithful to the v1 SOURCE; only its fixture diverges). So check (ii)
        # applies to every convertible config including the exceptions — it is
        # the v1 SOURCE comparison, the thing the fixture exception does not break.
        checks[f"{name}:faithful_to_v1"] = task_match and stream_match
        results.append(row)

    passed = all(checks.values())
    n_total = len(results)
    n_converted = sum(1 for r in results if r["converted"])
    n_synth_configs = sum(1 for r in results if r["v1_synthetic_targets"])
    exception_rows = [r for r in results if r["fixture_exception"]]

    criterion = (
        "the M7 W1.5 converter-output target-producer fidelity gate (the plan-compile + salt2 "
        "graph validate MISS): over the SAME CV1 denominator, for EVERY task in the converted FIT "
        "plan (i) every label/target it consumes has a producer AND every v1-declared SYNTHETIC "
        "handle (a multi_target custom_target — a column absent from disk) is produced by a "
        "CONCRETE (non-wildcard) data module (MultiTarget / MaskFormerTargets / ...), never served "
        "only by the demand-driven Labels labels.** wildcard (which narrows to ANY demanded key, "
        "proving no real column) — this is the MISS that let the broken pre-F1a "
        "regression_multi_target pass CV1 (its regression head consumed a never-produced "
        "pt_label_handle yet plan-compiled clean); (ii) for configs intended converter-v1 faithful "
        "the converter's task-NAME + stream set equals the v1 RESOLVED set; (iii) the named "
        "fixture exceptions "
        f"({', '.join(_CVF_FIXTURE_EXCEPTIONS)}) are excluded from converter-vs-FIXTURE equality "
        "(fixture keeps, not fidelity failures) — but regression_multi_target MUST still pass (i) "
        "now that F1a translates its multi_target block into a concrete MultiTarget producer. "
        "Negative control: a converter output with a synthetic target whose producer was deleted "
        "FAILS (the target falls back to the wildcard -> wildcard_only -> gate red)."
    )
    report = _base_report(
        "cvf_target_producer_fidelity",
        passed,
        criterion,
        {
            "total_configs": n_total,
            "converted": n_converted,
            "synthetic_target_configs": n_synth_configs,
            "fixture_exception_names": list(_CVF_FIXTURE_EXCEPTIONS),
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["configs"] = results
    report["cvf_fixture_exceptions"] = list(_CVF_FIXTURE_EXCEPTIONS)
    report["v1_source_root"] = str(V1_CONFIG_DIR)
    report["fixture_root"] = str(CONFIG_DIR)
    report["scope_note"] = (
        "CVF closes the validate/plan-compile MISS CV1 alone cannot catch: salt2 graph validate "
        "and compile_plan accept a config in which a task consumes a label key that the "
        "demand-driven Labels processor narrows out of its labels.** wildcard — the wildcard "
        "produces ANY demanded key, so a phantom target (a custom_target column that exists "
        "nowhere on disk) compiles clean and only fails at training-data-read time. CVF derives "
        "the SYNTHETIC handle set INDEPENDENTLY from the v1 source (the multi_target custom_target "
        "outputs) and asserts each is produced by a CONCRETE (allow_wildcards == False) data "
        "module in the converted FIT plan — for regression_multi_target that producer is the "
        "MultiTarget "
        "module F1a now emits, producing labels.jets.pt_label_handle for the regression head "
        "(MultiTarget --labels.jets.pt_label_handle--> RegressionTaskModule). Real dataset columns "
        "(flavour_label, ftagTruthOriginLabel) are legitimately served by the Labels wildcard and "
        "are NOT flagged — only the v1-declared synthetic handles must be concrete. Check (ii) "
        "compares the converter against the v1 RESOLVED source (task names + streams), which the "
        "fixture exceptions do not break (only their hand-written fixture diverges, by a rename/"
        "curation); check (iii) names event_classifier + regression_multi_target as the pinned "
        "fixture exceptions, excluded from fixture equality but NOT from check (i)."
    )

    # -- stdout table ----------------------------------------------------------
    print(
        f"{'config':<26}{'tier':<6}{'conv':>5}{'targets-ok':>12}{'v1-faithful':>13}"
        f"{'synth':>7}  flags"
    )
    for r in results:
        conv = "PASS" if r["converted"] else "FAIL"
        targ = (
            "PASS"
            if checks.get(f"{r['name']}:targets_produced")
            else ("-" if not r["converted"] else "FAIL")
        )
        faith = (
            "PASS"
            if checks.get(f"{r['name']}:faithful_to_v1")
            else ("-" if not r["converted"] else "FAIL")
        )
        nsynth = len(r["v1_synthetic_targets"])
        flags = []
        if r["today"]:
            flags.append("today")
        if r["fixture_exception"]:
            flags.append("fix-exception")
        if nsynth:
            flags.append("multi-target")
        print(
            f"{r['name']:<26}{r['tier']:<6}{conv:>5}{targ:>12}{faith:>13}"
            f"{nsynth:>7}  {','.join(flags)}"
        )
    print(
        f"\n{n_converted}/{n_total} converted; {n_synth_configs} config(s) declare synthetic "
        f"multi_target handles (must be concretely produced); fixture exceptions: "
        f"{[r['name'] for r in exception_rows]}"
    )
    _print_verdict("cvf", passed, criterion, _emit_report(report, outdir, "cvf"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# REN1 — the vector->global_object rename + StreamEmbed flag collapse (W1.5 wave R)
# ---------------------------------------------------------------------------

# The grep-pattern that catches a RESIDUAL ``vector`` FLAG use after the W1.5
# wave-R rename — a reader ``GroupConfig`` ``vector:`` field or a ``StreamEmbed``
# ``vector`` __init__ param / ``self.vector`` attribute. The rename DELETED both
# the coupled reader flag (renamed to ``global_object`` to align with
# ``Normaliser(global_object=)``) and the model-side StreamEmbed rank flag (its
# rank is now INFERRED from its bound input). What is ALLOWED to survive: the
# VectorConcat module + its gates_m5 helpers (explicitly out of scope, a
# different "vector" concept — the global-vector concat), and DESCRIPTIVE PROSE
# ("[B,F] vector", "global vector", "pooled vector", "feature vector",
# "vector head", "vector-stream", regression "vector norm_params") in
# docstrings/config comments, plus the 3 intentional historical references that
# NAME the removed flag to EXPLAIN the collapse (nn/modules.py, DL1.yaml,
# convert.py). REN1 greps salt/core for the FLAG-SHAPED patterns only — a
# ``vector:`` YAML key, a ``{vector:``/``"vector":`` dict literal, a ``vector=``
# kwarg, a ``self.vector`` attribute, or a ``gc["vector"]``/``embed["vector"]``
# subscript — each NOT on a ``global_object`` line. A non-empty hit is a
# residual flag that should have been renamed; the gate goes red.
# A flag-shaped ``vector`` token is one NOT preceded by a backtick (a real YAML
# key / Python param / attribute is never backtick-wrapped; ```vector:``` /
# `` `vector:` `` are PROSE that names the removed flag, not a live flag). The
# negative lookbehind ``(?<![A-Za-z_.`])`` therefore rejects (1) an identifier
# substring (``vector_mlp``, ``a.vector``) AND (2) a backtick-quoted prose
# mention — leaving only genuine flag uses.
_REN1_FLAG_PATTERNS: tuple[str, ...] = (
    r'(?<!`)"vector"\s*:',  # a {"vector": ...} dict literal (Python), not prose
    r"(?<![A-Za-z_.`])vector\s*:",  # a `vector:` YAML key (not global_object:, not prose)
    r"(?<![A-Za-z_.`])vector\s*=",  # a `vector=` kwarg / assignment (not prose)
    r"(?<!`)self\.vector\b",  # a `self.vector` attribute (not prose)
    r'gc\[["\']vector["\']\]',  # a `gc["vector"]` reader-config subscript
    r'embed\[["\']vector["\']\]',  # an `embed["vector"]` subscript
)

# the gate + test HARNESS files (not product code): they NAME the removed flag in
# their patterns/docstrings/criterion to DESCRIBE the rename, so they are scoped
# OUT of the product-code residual-flag grep. REN1 part (a) asserts the PRODUCT
# code surface -- reader, modules, bind, convert, processors, configs -- is
# flag-free, not that the gate harness avoids mentioning the flag it gates.
_REN1_HARNESS_GLOBS: tuple[str, ...] = ("gates_m", "parity_")

# the VectorConcat surface (the out-of-scope "vector" concept) — a hit on one of
# these tokens is NOT a residual flag (the global-vector concat module + its
# gates_m5 build helpers, ~16 hits, explicitly kept per the wave-R brief).
_REN1_VECTORCONCAT_TOKENS: tuple[str, ...] = (
    "VectorConcat",
    "vector_concat",
    "vectorconcat",
)

# the 3 INTENTIONAL historical references that NAME the removed ``vector:`` flag
# to EXPLAIN the collapse (a flag-shaped ``vector:`` substring may appear inside
# the explanatory prose). They are PINNED by (file-suffix, required-substring) so
# a NEW residual flag use in one of these files is still caught — only the exact
# documented sentence is exempted.
_REN1_HISTORICAL_REFS: tuple[tuple[str, str], ...] = (
    ("nn/modules.py", "supersedes"),  # "supersedes the M6-6 ``vector:`` flag"
    ("configs/DL1.yaml", "collapsed"),  # "collapsed the old coupled reader+model `vector:` pair"
    ("convert.py", "No model-side vector flag is emitted"),
)


def _ren1_grep_residual_flags() -> list[dict[str, str]]:
    """Grep ``salt/core`` for a RESIDUAL ``vector`` FLAG use after the rename.

    Walks every ``*.py`` and ``*.yaml`` under ``salt/core`` and flags any line
    matching a flag-shaped ``vector`` pattern (``_REN1_FLAG_PATTERNS``) that is
    NOT (a) on a ``global_object`` line (the rename target), (b) a VectorConcat
    surface line (the out-of-scope concat module), or (c) one of the 3 pinned
    intentional historical references that name the removed flag to explain the
    collapse. A non-empty result means a flag the rename should have removed
    still lives in the tree.

    Returns
    -------
    list[dict[str, str]]
        One ``{"file", "line", "text", "pattern"}`` record per residual hit
        (paths relative to ``salt/core``), sorted; empty when the rename is clean.
    """
    hits: list[dict[str, str]] = []
    compiled = [(p, re.compile(p)) for p in _REN1_FLAG_PATTERNS]
    for path in sorted(CORE_DIR.rglob("*")):
        if path.suffix not in {".py", ".yaml"} or not path.is_file():
            continue
        # the gate/parity HARNESS files name the removed flag to describe the
        # rename — scope them out (REN1 gates the PRODUCT code, not its harness).
        if any(path.name.startswith(g) for g in _REN1_HARNESS_GLOBS):
            continue
        rel = str(path.relative_to(CORE_DIR))
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            # skip the out-of-scope VectorConcat surface + the global_object
            # rename target outright (a `global_object:` line is the rename, not
            # a residual flag; `Normaliser(global_object=)` also carries `=`).
            low = raw.lower()
            if any(tok.lower() in low for tok in _REN1_VECTORCONCAT_TOKENS):
                continue
            if "global_object" in raw:
                continue
            # the 3 pinned historical references that NAME the removed flag
            if any(rel.endswith(suf) and sub in raw for suf, sub in _REN1_HISTORICAL_REFS):
                continue
            for pat, rx in compiled:
                if rx.search(raw):
                    hits.append(
                        {"file": rel, "line": str(lineno), "text": raw.strip(), "pattern": pat}
                    )
                    break
    return hits


# the shipped-config denominator for REN1 part (b): the AUTHORITATIVE M5-CONV +
# M6-CONV config lists (the standalone-validatable shipped v2 configs, with their
# overlay stacks / norm_global / muP / onnx handling already pinned by those
# gates). This is the set the converter + the model emit with the NEW
# ``global_object`` flag; REN1 re-validates every one fit/test/onnx (data-free)
# to prove the rename did not break config loading/plan-compile. The 10 GN3
# overlay FRAGMENTS (GN3_baseline overlays etc.) are NOT standalone-validatable
# on their own — the M5-CONV list already carries them as STACKED entries (base +
# overlay), so REN1 inherits the correct stacking for free.
_REN1_SHIPPED_CONFIGS: tuple[dict[str, Any], ...] = (*_M5_CONV_CONFIGS, *_M6_CONV_CONFIGS)

_REN1_NO_STRICT_RATIONALE = (
    "REN1 part (b) validates data-free, NOT --strict — the SAME M5/M6-CONV + CV1 precedent "
    "(_NO_STRICT_RATIONALE): a data-free validation cannot satisfy --strict, which promotes EVERY "
    "warning to an error (cli.py), and several warnings are inherent to data-free validation "
    "(no-schema, warning-level deadcode, multi-stream Normaliser missing-input-type preflight) — "
    "orthogonal to whether the global_object rename loads + plan-compiles. rc == 0 (no ERROR-level "
    "finding) IS the 'validates in fit/test/onnx with the new global_object flag' criterion. "
    "(--strict on these data-free configs is RED BY CONSTRUCTION, confirmed empirically: DL1 "
    "--strict fit/test/onnx all rc=1 — exactly the documented data-free-warning promotion, not a "
    "rename regression.)"
)

# REN1 part (c) — the rank-inference synthetic check fixture (DL1-shaped jets).
_REN1_JET_VARIABLES: tuple[str, ...] = ("pt_btagJes", "eta_btagJes")
_REN1_EMBED_DIM = 16
_REN1_B = 4  # batch size for the synthetic rank check
_REN1_T = 5  # token positions for the rank-3 sequence case


def _ren1_rank_inference(outdir: Path) -> dict[str, Any]:
    """StreamEmbed infers rank from its BOUND INPUT — rank-2 AND rank-3, no flag.

    The load-bearing wave-R evidence (part (c)): a SINGLE ``StreamEmbed`` (carrying
    NO rank flag — the ``vector`` param is DELETED) is driven through the REAL
    compiler / two-phase bind / executor twice, the ONLY difference being the
    rank its bound input carries (set by the producer ``Normaliser`` — rank-2 for
    a ``global_object`` stream, rank-3 for a sequence stream — exactly the single
    reader ``global_object:`` flag that now drives the rank end-to-end):

    - a ``global_object='jets'`` ``Normaliser`` -> ``normed.jets`` ``[B, F]`` ->
      the SAME embed produces ``embed.jets`` ``[B, D]`` (rank-2, no token axis);
    - a sequence (``global_object=None``) ``Normaliser`` -> ``normed.jets``
      ``[B, T, F]`` -> the SAME embed produces ``embed.jets`` ``[B, T, D]``
      (rank-3, token axis preserved).

    The embed's ``out_dim`` width ``D`` is contributed identically in BOTH cases
    via ``derived_widths`` (a ``shape=None`` produce carries no last dim); only
    the RANK differs, and it flows from the bound input — proving the collapse is
    behaviour-preserving (the rank-2 DL1 path and the rank-3 sequence path share
    one flag-free module).

    Returns
    -------
    dict[str, Any]
        ``{"checks": {...}, "rank2": {...}, "rank3": {...}}`` — per-case the
        forward output rank/shape and the static derived width.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    nd = outdir / "ren1_norm_dict.yaml"
    nd.write_text(
        yaml.safe_dump({
            "jets": {
                "pt_btagJes": {"mean": 1.5, "std": 2.0},
                "eta_btagJes": {"mean": -0.25, "std": 1.25},
            }
        })
    )

    def _build() -> StreamEmbed:
        # a SINGLE StreamEmbed — NO rank flag (the deleted `vector` param). The
        # ONLY thing varied between the two cases is the Normaliser's
        # global_object flag, which sets the bound-input rank.
        return StreamEmbed(
            stream="jets",
            out_dim=_REN1_EMBED_DIM,
            dense={"hidden_layers": [8], "activation": "ReLU"},
        )

    def _run(global_object: str | None, src_shape: tuple, x: torch.Tensor) -> dict[str, Any]:
        norm = Normaliser(norm_dict=nd, streams=["jets"], global_object=global_object)
        embed = _build()
        modules: dict[str, Any] = {"norm": norm, "jets_embed": embed}
        for name, module in modules.items():
            module.name = name
        sources = unflatten_spec({
            "inputs.jets": TensorSpec(
                shape=src_shape, dtype="float32", fields=tuple(_REN1_JET_VARIABLES)
            )
        })
        with _quiet():
            plan = compile_plan(modules, Mode.FIT, sources=sources, sinks=["embed.jets"])
            input_spec = plan.sources["inputs.jets"]
            bind_all(modules, resolve_bind_schema([plan]))
            materialise_all(modules)
            b = Bundle()
            b.set("inputs.jets", x)
            out = Executor(plan).run(b, debug=True)
        emb = out.get("embed.jets")
        return {
            "global_object": global_object,
            "input_rank": len(input_spec.shape),
            "embed_rank": int(emb.ndim),
            "embed_shape": list(emb.shape),
            "derived_width": embed.derived_widths({}).get("embed.jets"),
            "embed_has_no_vector_param": not hasattr(embed, "vector"),
        }

    nf = len(_REN1_JET_VARIABLES)
    rank2 = _run(
        "jets",
        ("B", nf),
        torch.randn(_REN1_B, nf, generator=torch.Generator().manual_seed(1)),
    )
    rank3 = _run(
        None,
        ("B", "T:jets", nf),
        torch.randn(_REN1_B, _REN1_T, nf, generator=torch.Generator().manual_seed(2)),
    )

    checks = {
        # rank-2 [B, F] bound input -> rank-2 [B, D] embed (no token axis)
        "rank2_input_is_rank2": rank2["input_rank"] == 2,
        "rank2_embed_is_rank2": rank2["embed_rank"] == 2,
        "rank2_embed_shape": rank2["embed_shape"] == [_REN1_B, _REN1_EMBED_DIM],
        # rank-3 [B, T, F] bound input -> rank-3 [B, T, D] embed (token axis kept)
        "rank3_input_is_rank3": rank3["input_rank"] == 3,
        "rank3_embed_is_rank3": rank3["embed_rank"] == 3,
        "rank3_embed_shape": rank3["embed_shape"] == [_REN1_B, _REN1_T, _REN1_EMBED_DIM],
        # the SAME out_dim width via derived_widths in BOTH cases (flag-free)
        "out_dim_via_derived_widths": (
            rank2["derived_width"] == rank3["derived_width"] == _REN1_EMBED_DIM
        ),
        # the StreamEmbed carries NO `vector` rank flag (the param is DELETED)
        "streamembed_has_no_vector_attr": rank2["embed_has_no_vector_param"]
        and rank3["embed_has_no_vector_param"],
    }
    return {"checks": checks, "rank2": rank2, "rank3": rank3, "norm_dict": str(nd)}


def run_ren1(
    outdir: Path | str,
    *,
    corruption: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """REN1: the vector->global_object rename + StreamEmbed flag-collapse gate.

    The W1.5 wave-R behaviour-preserving rename gate. It asserts THREE things:

    (a) **no residual ``vector`` FLAG** — a grep over ``salt/core`` finds NO
        flag-shaped ``vector`` use (a reader ``GroupConfig`` ``vector:`` field or
        a ``StreamEmbed`` ``vector`` param / ``self.vector`` attribute). Only the
        out-of-scope ``VectorConcat`` surface, descriptive prose, and the 3
        pinned intentional historical references that NAME the removed flag to
        explain the collapse are allowed (``_ren1_grep_residual_flags``);

    (b) **every shipped config validates with the new flag** — over the
        AUTHORITATIVE M5-CONV + M6-CONV shipped-config denominator (the
        standalone-validatable v2 configs, overlay stacks pinned), every config
        validates + plan-compiles fit/test/onnx (``salt2 graph validate`` rc == 0
        in every applicable mode; data-free, NOT ``--strict`` — the M5/M6-CONV
        precedent, ``_REN1_NO_STRICT_RATIONALE``) with the renamed
        ``global_object`` flag in place;

    (c) **StreamEmbed infers rank for rank-2 AND rank-3 inputs** — a synthetic
        check (``_ren1_rank_inference``) drives the SAME flag-free ``StreamEmbed``
        through the real compile/bind/executor twice: a ``global_object`` rank-2
        ``[B, F]`` bound input -> ``[B, D]`` embed (no token axis), a sequence
        rank-3 ``[B, T, F]`` bound input -> ``[B, T, D]`` embed (token axis
        preserved) — rank inferred from the bound input, ZERO model-side flag.

    Negative control (``test_gates_m7.py`` + the in-gate ``corruption`` hook,
    test-only, never on the CLI): the hook INJECTS a fake residual-flag hit into
    the part-(a) result, so the ``no_residual_vector_flag`` check flips False and
    the gate goes red — proving the grep check is not vacuous.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("REN1 vector->global_object rename + StreamEmbed flag-collapse (W1.5 wave R)")
    print("=" * 96)

    checks: dict[str, bool] = {}

    # -- (a) no residual `vector` FLAG use in salt/core ----------------------
    residual = _ren1_grep_residual_flags()
    if corruption is not None:
        residual = corruption({"residual": residual})["residual"]
    checks["no_residual_vector_flag"] = residual == []

    # -- (b) every shipped config validates fit/test/onnx with global_object --
    outdir.mkdir(parents=True, exist_ok=True)
    norm_dict = outdir / "ren1_conv_norm_dict.yaml"
    class_dict = outdir / "ren1_conv_class_dict.yaml"
    write_parity_norm_dict(norm_dict, class_dict)
    config_rows: list[dict[str, Any]] = []
    for entry in _REN1_SHIPPED_CONFIGS:
        name = entry["name"]
        cfg_paths = [CONFIG_DIR / c for c in entry["cfg"]]
        files_ok = all(p.is_file() for p in cfg_paths)
        row: dict[str, Any] = {
            "name": name,
            "cfg": list(entry["cfg"]),
            "family": entry.get("family"),
            "validate": {},
            "files_present": files_ok,
        }
        if not files_ok:
            checks[f"{name}:files_present"] = False
            config_rows.append(row)
            continue
        sets = list(_m5_conv_set_args(entry, norm_dict))
        # a muP config (M6-CONV GN2_muP) needs its infshapes generated data-free
        # first (the _mup_shape_override helper, the MU2 precedent).
        if entry.get("mup"):
            sets += _mup_shape_override(cfg_paths, sets, outdir / name)
        all_ok = True
        modes = ["fit", "test"] + (["onnx"] if entry.get("onnx") == "validate" else [])
        for mode in modes:
            rc = _validate_rc(cfg_paths, mode, sets)
            row["validate"][mode] = rc
            mode_ok = rc == 0
            checks[f"{name}:validate:{mode}"] = mode_ok
            all_ok = all_ok and mode_ok
        row["all_modes_ok"] = all_ok
        config_rows.append(row)

    # -- (c) StreamEmbed rank inference (rank-2 [B,F] + rank-3 [B,T,F]) -------
    rank = _ren1_rank_inference(outdir / "rank_inference")
    for cname, cval in rank["checks"].items():
        checks[f"rank:{cname}"] = cval

    passed = all(checks.values())
    n_configs = len(config_rows)
    n_cfg_ok = sum(1 for r in config_rows if r.get("all_modes_ok"))

    criterion = (
        "the M7 W1.5 wave-R vector->global_object rename + StreamEmbed flag collapse "
        "(behaviour-preserving): (a) a grep over salt/core finds NO residual `vector` FLAG use (a "
        "reader GroupConfig `vector:` field or a StreamEmbed `vector` param / self.vector "
        "attribute) "
        "— only the out-of-scope VectorConcat surface, descriptive prose, and the 3 pinned "
        "intentional historical references that NAME the removed flag are allowed; (b) over the "
        f"AUTHORITATIVE M5-CONV + M6-CONV shipped-config denominator ({n_configs} configs) every "
        "config validates + plan-compiles fit/test/onnx (salt2 graph validate rc == 0, data-free "
        "non-strict — the M5/M6-CONV precedent) with the renamed global_object flag in place; (c) "
        "the SAME flag-free StreamEmbed INFERS its rank from its bound input — a rank-2 [B, F] "
        "global-object input -> [B, D] embed (no token axis) AND a rank-3 [B, T, F] sequence input "
        "-> [B, T, D] embed (token axis preserved), zero model-side flag. Negative control: an "
        "injected residual-flag hit flips check (a) red, proving the grep is not vacuous."
    )
    report = _base_report(
        "ren1_vector_rename_collapse",
        passed,
        criterion,
        {
            "core_dir": str(CORE_DIR),
            "shipped_configs": n_configs,
            "shipped_configs_validated": n_cfg_ok,
            "residual_flag_hits": len(residual),
            "strict": False,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["residual_flag_hits"] = residual
    report["flag_patterns"] = list(_REN1_FLAG_PATTERNS)
    report["vectorconcat_tokens_allowed"] = list(_REN1_VECTORCONCAT_TOKENS)
    report["historical_refs_allowed"] = [
        {"file_suffix": suf, "substring": sub} for suf, sub in _REN1_HISTORICAL_REFS
    ]
    report["shipped_config_validation"] = config_rows
    report["shipped_config_names"] = [e["name"] for e in _REN1_SHIPPED_CONFIGS]
    report["rank_inference"] = rank
    report["no_strict_rationale"] = _REN1_NO_STRICT_RATIONALE
    report["scope_note"] = (
        "REN1 gates the W1.5 wave-R rename that DELETED two coupled flags: the reader's "
        "GroupConfig.vector field (renamed global_object, aligning with "
        "Normaliser(global_object=)) "
        "and the model-side StreamEmbed `vector` rank flag (collapsed — the embed now INFERS its "
        "rank from its bound input). The rename is BEHAVIOUR-PRESERVING: the rank originates at "
        "ONE "
        "place (the reader's per-group global_object: flag), which drives both the reader/Features "
        "boundary rank and the matching Normaliser global_object rank-2 handling; the StreamEmbed "
        "declares rank-AGNOSTIC specs (shape=None on both the normed.<s> require and the embed.<s> "
        "produce) so the producer sets the input rank and the consumer sets the output rank — they "
        "agree by construction (both derive from that one flag). Part (a) greps for the "
        "FLAG-SHAPED "
        "residual only (a `vector:` key, a `vector=` kwarg, self.vector, a gc/embed['vector'] "
        "subscript) NOT on a global_object line — the VectorConcat module (a DIFFERENT 'vector' "
        "concept, the global-vector concat) + descriptive prose + the 3 pinned historical "
        "references that name the removed flag to explain the collapse are exempt. Part (b) reuses "
        "the M5-CONV + M6-CONV shipped-config lists (the standalone-validatable v2 configs, "
        "overlay "
        "stacks/norm_global/muP already pinned) so it inherits the correct stacking for the 10 GN3 "
        "overlay fragments. Part (c) proves the rank inference end-to-end for BOTH ranks through "
        "the "
        "real compile/bind/executor with one flag-free StreamEmbed."
    )

    # -- stdout table ----------------------------------------------------------
    print("(a) residual vector FLAG hits in salt/core:")
    if residual:
        for h in residual:
            print(f"    {h['file']}:{h['line']}  [{h['pattern']}]  {h['text']}")
    else:
        print("    NONE (rename clean)")
    print(f"\n(b) shipped configs validated fit/test/onnx ({n_cfg_ok}/{n_configs}):")
    print(f"{'config':<28}{'fit':>5}{'test':>6}{'onnx':>6}  family")
    for r in config_rows:
        v = r["validate"]
        fit = "PASS" if v.get("fit") == 0 else ("-" if "fit" not in v else "FAIL")
        test = "PASS" if v.get("test") == 0 else ("-" if "test" not in v else "FAIL")
        onnx = "PASS" if v.get("onnx") == 0 else ("na" if "onnx" not in v else "FAIL")
        print(f"{r['name']:<28}{fit:>5}{test:>6}{onnx:>6}  {r.get('family')}")
    r2, r3 = rank["rank2"], rank["rank3"]
    print(
        f"\n(c) StreamEmbed rank inference: rank-2 [B,F] in (rank {r2['input_rank']}) -> embed "
        f"{r2['embed_shape']} (rank {r2['embed_rank']}); rank-3 [B,T,F] in "
        f"(rank {r3['input_rank']}) "
        f"-> embed {r3['embed_shape']} (rank {r3['embed_rank']}); out_dim via derived_widths "
        f"(no model-side flag)"
    )
    _print_verdict("ren1", passed, criterion, _emit_report(report, outdir, "ren1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# RS1 — residual v1-import scanner (the M7 W2 worklist + the W2c end-state gate)
# ---------------------------------------------------------------------------

# the v1 legacy top-level packages a MODULARISED salt.core must NOT live-import.
# A production salt.core file importing from any of these is a residual coupling
# to the v1 tree (the modularisation §2 goal: salt.core is self-contained, the
# v1 tree stays in-tree dormant as the GATE ORACLE). W2 relocates the genuinely-
# shared helpers into salt.core/ and repoints every consumer; RS1 is the worklist
# (the authoritative residual list) AND the W2c end-state gate (count == 0).
_RS1_V1_PACKAGES: frozenset[str] = frozenset({
    "models",
    "data",
    "utils",
    "onnx",
    "optim",
    "modelwrapper",
    "callbacks",
    "submit",
    "stypes",
})

# the GATE HARNESS + scaffolding files under salt/core that are NOT production
# code (they are the M7 gate machinery + v1-adapter shims). RS1 scopes them OUT:
#   - gates_m*.py / parity_gn2.py — the gate harnesses themselves (they import v1
#     deliberately, as the ORACLE side of every parity/conversion gate);
#   - from_v1.py — the v1->v2 weight/state adapter (it MUST import v1 to translate
#     a v1 checkpoint; that is its entire purpose);
#   - wrappers.py — the v1 ModelWrapper/Lightning bridge shim;
#   - demo_m1.py — the M1 spike demo script.
# RS1 asserts the PRODUCT surface (reader, processors, nn task/module heads,
# writers, onnx reduces, saltmodule, convert, ...) is v1-import-free, not that the
# gate machinery avoids importing the tree it gates.
_RS1_HARNESS_BASENAMES: frozenset[str] = frozenset({
    "parity_gn2.py",
    "from_v1.py",
    "wrappers.py",
    "demo_m1.py",
})
_RS1_HARNESS_PREFIXES: tuple[str, ...] = ("gates_m",)


def _rs1_is_harness(path: Path) -> bool:
    """Return whether ``path`` is a gate-harness / scaffolding file (scoped out).

    Returns
    -------
    bool
        True for the gate harnesses (``gates_m*.py``, ``parity_gn2.py``), the v1
        adapters (``from_v1.py``, ``wrappers.py``) and the demo (``demo_m1.py``)
        — the non-production files RS1 excludes from the residual scan.
    """
    name = path.name
    if name in _RS1_HARNESS_BASENAMES:
        return True
    return any(name.startswith(p) for p in _RS1_HARNESS_PREFIXES)


def _rs1_module_top(module: str | None, level: int) -> str | None:
    """Return the v1 sub-package a ``salt.<pkg>...`` import targets, else None.

    Resolves an ``Import`` / ``ImportFrom`` module string to the v1 top-level
    sub-package it references (``models`` / ``data`` / ``utils`` / ...), or None
    when the import is NOT a live ``salt.<v1-pkg>`` import. A ``salt.core.*``
    import is NEVER a residual (it is the modularised tree itself); a relative
    import (``level > 0``, no ``salt.`` prefix) is intra-package and never a v1
    residual.

    Returns
    -------
    str | None
        The matched v1 sub-package name (e.g. ``"utils"``) when ``module`` is a
        live ``salt.<pkg>`` / ``salt.<pkg>.<...>`` import for a flagged package,
        else None.
    """
    if not module or level:  # relative imports are intra-package, never v1
        return None
    parts = module.split(".")
    # need at least salt.<pkg>; salt.core.* is the modularised tree (not v1)
    if len(parts) < 2 or parts[0] != "salt" or parts[1] == "core":
        return None
    return parts[1] if parts[1] in _RS1_V1_PACKAGES else None


def _rs1_scan_residual_imports() -> list[dict[str, Any]]:
    """AST-walk every PRODUCTION ``salt/core`` file for a live v1-package import.

    Parses each ``*.py`` under ``salt/core`` (EXCLUDING the gate harnesses +
    v1-adapter shims, ``_rs1_is_harness``) with ``ast.parse`` and walks every
    ``ast.Import`` / ``ast.ImportFrom`` node, reporting any LIVE import from a
    flagged v1 sub-package (``_RS1_V1_PACKAGES``). Because the scan is AST-based,
    a v1-package name appearing in a COMMENT or a string literal (e.g. the
    ``convert.py`` :119 ``# ... salt.models ...`` comment and the :355 f-string
    naming ``salt.models``) does NOT count — only a real import node does (the
    text-grep false-positive the worklist warns about).

    Both ``import salt.utils.x`` (an ``ast.Import`` alias) and
    ``from salt.utils.x import y`` (an ``ast.ImportFrom``) are caught; lazy
    function-body imports (``from salt.utils import file_utils`` inside a method)
    are caught too because ``ast.walk`` descends into function bodies.

    Returns
    -------
    list[dict[str, Any]]
        One ``{"file", "lineno", "module", "pkg"}`` record per residual live
        import (``file`` relative to ``salt/core``), sorted by (file, lineno);
        empty when ``salt.core`` is fully decoupled from the v1 tree (the W2c
        end-state).
    """
    hits: list[dict[str, Any]] = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        if not path.is_file() or _rs1_is_harness(path):
            continue
        rel = str(path.relative_to(CORE_DIR))
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = _rs1_module_top(alias.name, 0)
                    if pkg is not None:
                        hits.append(
                            {
                                "file": rel,
                                "lineno": node.lineno,
                                "module": alias.name,
                                "pkg": pkg,
                            }
                        )
            elif isinstance(node, ast.ImportFrom):
                pkg = _rs1_module_top(node.module, node.level)
                if pkg is not None:
                    hits.append(
                        {
                            "file": rel,
                            "lineno": node.lineno,
                            "module": node.module,
                            "pkg": pkg,
                        }
                    )
    hits.sort(key=operator.itemgetter("file", "lineno"))
    return hits


def run_rs1(
    outdir: Path | str,
    *,
    corruption: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """RS1: the residual v1-import scanner (the M7 W2 worklist + W2c end-state gate).

    AST-walks every PRODUCTION file under ``salt/core`` (EXCLUDING the gate
    harnesses ``gates_m*.py`` / ``parity_gn2.py`` and the v1-adapter shims
    ``from_v1.py`` / ``wrappers.py`` / ``demo_m1.py``) and reports every LIVE
    import from a flagged v1 sub-package (``salt.{models,data,utils,onnx,optim,
    modelwrapper,callbacks,submit,stypes}``). The scan is AST-based
    (``ast.parse`` + ``ast.walk`` over ``Import`` / ``ImportFrom`` nodes) so a v1
    package name in a COMMENT or a string literal is NOT a hit — only a real
    import node — defusing the text-grep false-positive on the ``convert.py``
    :119 comment + :355 f-string that NAME ``salt.models`` in prose.

    The gate PASSES when the residual count is ZERO (the W2c modularisation
    end-state: ``salt.core`` is fully decoupled from the v1 tree). During W2a/W2b
    it RUNS and emits the AUTHORITATIVE residual production-import list — the W2
    worklist — with a non-zero rc until every relocation has landed.

    Negative control (``test_gates_m7.py`` + the in-gate ``corruption`` hook,
    test-only, never on the CLI): the hook INJECTS a synthetic
    ``salt.models``-import residual into the scan result, so the
    ``no_residual_v1_imports`` check flips False and the gate goes red — proving
    the scanner is not vacuous.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("RS1 residual v1-import scanner (M7 W2 worklist + W2c end-state gate)")
    print("=" * 96)

    checks: dict[str, bool] = {}

    residual = _rs1_scan_residual_imports()
    if corruption is not None:
        residual = corruption({"residual": residual})["residual"]
    checks["no_residual_v1_imports"] = residual == []

    passed = all(checks.values())
    n_residual = len(residual)
    by_pkg: dict[str, int] = {}
    for h in residual:
        by_pkg[h["pkg"]] = by_pkg.get(h["pkg"], 0) + 1

    criterion = (
        "the M7 W2c modularisation end-state: an AST walk (ast.parse + ast.walk over "
        "Import/ImportFrom) of every PRODUCTION salt/core file (EXCLUDING the gate harnesses "
        "gates_m*.py / parity_gn2.py and the v1-adapter shims from_v1.py / wrappers.py / "
        "demo_m1.py) finds NO live import from a flagged v1 sub-package "
        "(salt.{models,data,utils,onnx,optim,modelwrapper,callbacks,submit,stypes}). The scan is "
        "AST-based so a v1 package name in a comment or string literal (the convert.py:119 comment "
        "+ :355 f-string) does NOT count — only a real import node. The gate PASSES when the "
        "residual count is 0 (salt.core fully decoupled from v1); during W2a/W2b it emits the "
        "authoritative residual production-import list (the W2 worklist) and rc != 0 until every "
        "relocation lands. Negative control: an injected synthetic salt.models residual flips the "
        "check red, proving the scanner is not vacuous."
    )
    report = _base_report(
        "rs1_residual_v1_imports",
        passed,
        criterion,
        {
            "core_dir": str(CORE_DIR),
            "residual_import_count": n_residual,
            "residual_by_package": dict(sorted(by_pkg.items())),
            "flagged_packages": sorted(_RS1_V1_PACKAGES),
            "excluded_harness_basenames": sorted(_RS1_HARNESS_BASENAMES),
            "excluded_harness_prefixes": list(_RS1_HARNESS_PREFIXES),
            "ast_based": True,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["checks"] = checks
    report["residual_imports"] = residual
    report["scope_note"] = (
        "RS1 is the M7 W2 worklist + the W2c end-state gate. It walks the AST of every PRODUCTION "
        "salt/core *.py (gate harnesses gates_m*.py/parity_gn2.py + v1 adapters from_v1.py/"
        "wrappers.py/demo_m1.py scoped OUT — they import v1 deliberately as the ORACLE/adapter "
        "side) and flags every live Import/ImportFrom of salt.{models,data,utils,onnx,optim,"
        "modelwrapper,callbacks,submit,stypes}. AST-based (not a text grep) so comments + string "
        "literals naming a v1 package do NOT register — the convert.py:119 dropped-class comment "
        "and the :355 bit-rot f-string both NAME salt.models in prose but are not import nodes. "
        "salt.core.* imports are never residuals (that is the modularised tree). Lazy "
        "function-body imports are caught because ast.walk descends into function bodies. The "
        "gate passes at "
        "count == 0 (W2c); the non-empty list it emits now is the authoritative W2 relocation "
        "worklist."
    )

    # -- stdout table ----------------------------------------------------------
    print(f"flagged v1 packages: {sorted(_RS1_V1_PACKAGES)}")
    print(f"\nresidual live v1 imports in production salt/core ({n_residual}):")
    if residual:
        print(f"{'file':<32}{'line':>6}  module")
        for h in residual:
            print(f"{h['file']:<32}{h['lineno']:>6}  {h['module']}")
        print(f"\nresidual by package: {dict(sorted(by_pkg.items()))}")
    else:
        print("    NONE (salt.core fully decoupled from the v1 tree — W2c end-state)")
    _print_verdict("rs1", passed, criterion, _emit_report(report, outdir, "rs1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# FILM1 — v2-native FiLM + PositionalEncoder + parameterised forward parity
# ---------------------------------------------------------------------------
#
# FILM1 (M7 W-FILM) gates the v2-native FiLM/posenc port (plan 16). For EVERY
# W-FILM surface it builds an independent v1 instance (the FILM1 ORACLE — v1 stays
# in-tree, the v2 port COPIES the math, never imports v1 in production), transfers
# the v1 weights into the v2-native block, and asserts the v2 forward is BITWISE
# (torch.equal) vs v1 on synthetic fixtures:
#
#   1. FeaturewiseTransformation at layer=input / encoder / global — the FiLM
#      scale/bias math (featurewise.py:71-81), incl. the optional LayerNorm.
#   2. PositionalEncoder — the sin/cos encoding (posenc.py:36-85, the v1 print()
#      debug dropped in the port — the encoding math is the parity-bearing part).
#   3. A parameterised DATA-PATH forward — the v2 `Features` processor materialises
#      `inputs.parameters` [B, n_params] in the DECLARED column order from a
#      structured raw, and a FiLM-wired `StreamEmbed` consumes it bitwise-vs-v1.
#      Proves the order/normalisation/rank contract end-to-end, not just a
#      hand-built tensor: the declared parameter NAMES determine column ORDER, the
#      tensor stays rank-2 [B, n_params], and a missing/duplicate parameter fails
#      loudly (negative-control assertions).
#
# Negative control (test_gates_m7.py + the in-gate corruption hook): perturbing
# ONE v2 weight after the transfer makes the bitwise check go red — proving the
# comparison is not vacuous.

# small synthetic dims (the parity is shape-independent; small keeps it fast)
_FILM1_BATCH = 5
_FILM1_TOKENS = 7
_FILM1_NUM_PARAMS = 3
_FILM1_NUM_FEATURES = 8
_FILM1_OUT_DIM = 6
_FILM1_PARAM_NAMES = ("mu", "npv", "pt_bin")
"""The declared parameter NAMES — the data-path contract: column order == this
list order, rank-2 [B, len(names)] (v1 datasets.py:222-228 order check)."""


def _film1_dense_cfg() -> dict[str, Any]:
    """A representative FiLM net dense config (hidden layer + ReLU).

    Returns
    -------
    dict[str, Any]
        Kwargs the v2 FiLM forwards to `salt.core.nn.Dense` (no width keys —
        those are inferred at build/bind).
    """
    return {"hidden_layers": [16], "activation": "ReLU"}


def _film1_featurewise_parity(layer: str, *, corrupt: bool) -> dict[str, Any]:
    """Build v1 + v2 FiLM at ``layer``, transfer weights, compare bitwise.

    Constructs an independent v1 ``FeaturewiseTransformation`` (the oracle) with
    an EXPLICIT ``output_size = num_features`` (v1's user-set width) and a
    v2-native one with the SAME width INFERRED at build; loads the v1 state_dict
    into the v2 block (the absorbed Dense is byte-identical so the transfer is
    exact) and asserts ``torch.equal`` on a synthetic ``[B, T, num_features]``
    feature tensor. ``apply_norm`` is exercised on the input layer.

    Returns
    -------
    dict[str, Any]
        ``{layer, bitwise, max_abs_diff, ref_shape, got_shape, apply_norm}``.
    """
    from salt.models.featurewise import FeaturewiseTransformation as V1FW  # noqa: PLC0415

    from salt.core.nn import FeaturewiseTransformation as V2FW  # noqa: PLC0415

    torch.manual_seed(0)
    apply_norm = layer == "input"
    dc = _film1_dense_cfg()
    v1 = V1FW(
        layer=layer,
        variables={"parameters": list(_FILM1_PARAM_NAMES)},
        dense_config_scale={**dc, "output_size": _FILM1_NUM_FEATURES},
        dense_config_bias={**dc, "output_size": _FILM1_NUM_FEATURES},
        apply_norm=apply_norm,
    )
    v2 = V2FW(
        layer=layer,
        num_params=_FILM1_NUM_PARAMS,
        num_features=_FILM1_NUM_FEATURES,
        dense_config_scale=dict(dc),
        dense_config_bias=dict(dc),
        apply_norm=apply_norm,
    )
    v2.build()
    v2.load_state_dict(v1.state_dict())
    if corrupt:
        # negative control: nudge ONE v2 weight so the forward diverges
        with torch.no_grad():
            next(p for p in v2.parameters()).add_(1.0)
    params = torch.randn(_FILM1_BATCH, _FILM1_NUM_PARAMS)
    features = torch.randn(_FILM1_BATCH, _FILM1_TOKENS, _FILM1_NUM_FEATURES)
    v1.eval()
    v2.eval()
    with torch.no_grad():
        ref = v1({"parameters": params}, features)
        got = v2(params, features)
    bitwise = ref.shape == got.shape and torch.equal(ref, got)
    return {
        "layer": layer,
        "apply_norm": apply_norm,
        "bitwise": bool(bitwise),
        "max_abs_diff": 0.0 if bitwise else float((ref - got).abs().max().item()),
        "ref_shape": tuple(ref.shape),
        "got_shape": tuple(got.shape),
    }


def _film1_posenc_parity(*, corrupt: bool) -> dict[str, Any]:
    """Build v1 + v2 `PositionalEncoder`, compare bitwise (no weights — fixed fn).

    The encoder is parameter-free (a fixed sin/cos function under
    ``@torch.no_grad``), so there is nothing to weight-transfer — the v2 port must
    reproduce the v1 encoding BYTE-FOR-BYTE on the SAME coordinate tensor. The
    ``phi`` symmetric variable exercises the ``SYM_VARS`` branch (posenc.py:79-81).

    Returns
    -------
    dict[str, Any]
        ``{bitwise, max_abs_diff, ref_shape, got_shape, variables, dim}``.
    """
    from salt.models.posenc import PositionalEncoder as V1PE  # noqa: PLC0415

    from salt.core.nn import PositionalEncoder as V2PE  # noqa: PLC0415

    variables = ["phi", "eta"]  # phi -> SYM_VARS symmetric branch; eta -> default
    dim = _FILM1_OUT_DIM
    v1 = V1PE(variables=variables, dim=dim, alpha=100)
    v2 = V2PE(variables=variables, dim=dim, alpha=100)
    torch.manual_seed(1)
    coords = torch.randn(_FILM1_BATCH, _FILM1_TOKENS, len(variables))
    with torch.no_grad():
        ref = v1(coords)
        got = v2(coords)
    if corrupt:
        got = got + 1.0  # negative control: perturb the v2 output
    bitwise = ref.shape == got.shape and torch.equal(ref, got)
    return {
        "variables": variables,
        "dim": dim,
        "bitwise": bool(bitwise),
        "max_abs_diff": 0.0 if bitwise else float((ref - got).abs().max().item()),
        "ref_shape": tuple(ref.shape),
        "got_shape": tuple(got.shape),
    }


def _film1_param_contract() -> dict[str, Any]:
    """Prove the v2 ``parameters`` order/normalisation/rank contract end-to-end.

    Drives the REAL v2 `Features` processor (the design §6.2 ``raw.* -> inputs.*``
    boundary, the ONE place column order is defined) on a structured raw whose
    fields are DELIBERATELY out of declared order, and asserts:

    - the produced ``inputs.parameters`` is rank-2 ``[B, n_params]`` (a global
      stream, v1 datasets.py global_object path);
    - the columns are in the DECLARED ``_FILM1_PARAM_NAMES`` order (NOT the raw
      field order) — the parity-sensitive order contract (v1 datasets.py:222-228);
    - the produced spec carries ``fields`` == the declared names (so a downstream
      FiLM/`StreamEmbed` resolves columns by NAME);
    - a DUPLICATE parameter name fails loudly (`ConfigError`);
    - a Features stream demanding a parameter ABSENT from the raw fails loudly.

    Returns
    -------
    dict[str, Any]
        ``{rank2, declared_order, fields_match, duplicate_rejected,
        missing_rejected}``.
    """
    from salt.core.data.processors import Features  # noqa: PLC0415
    from salt.core.graph import Bundle, Mode  # noqa: PLC0415
    from salt.core.graph.spec import flatten_spec  # noqa: PLC0415

    result: dict[str, Any] = {}

    # the raw structured array with the parameter fields in a SCRAMBLED order
    # (npv, pt_bin, mu) — the declared order is (mu, npv, pt_bin); the Features
    # processor MUST reorder to the declared list, never the raw field order.
    scrambled = ("npv", "pt_bin", "mu")
    raw = np.zeros(
        _FILM1_BATCH, dtype=[(name, np.float32) for name in scrambled]
    )
    # distinct per-column sentinels so a wrong order is detectable
    sentinels = {"mu": 10.0, "npv": 20.0, "pt_bin": 30.0}
    for name in scrambled:
        raw[name] = sentinels[name]

    feats = Features(variables={"parameters": list(_FILM1_PARAM_NAMES)})
    out = feats.process(Bundle({"raw": {"parameters": raw}}), slice(None), Mode.FIT)
    arr = out["inputs.parameters"]
    result["rank2"] = arr.ndim == 2 and arr.shape == (_FILM1_BATCH, len(_FILM1_PARAM_NAMES))
    # column j must hold the DECLARED name's sentinel (declared order, not raw)
    declared_ok = all(
        float(arr[0, j]) == sentinels[name] for j, name in enumerate(_FILM1_PARAM_NAMES)
    )
    result["declared_order"] = bool(declared_ok)
    # the produced spec carries fields == declared names (column-by-name lookups)
    io = feats.declare_io(Mode.FIT)
    spec = flatten_spec(io.produces)["inputs.parameters"]
    result["fields_match"] = tuple(spec.fields or ()) == _FILM1_PARAM_NAMES

    # duplicate parameter name -> loud ConfigError
    try:
        Features(variables={"parameters": ["mu", "mu", "npv"]})
        result["duplicate_rejected"] = False
    except ConvertError:  # pragma: no cover - wrong error type
        result["duplicate_rejected"] = False
    except Exception as err:  # noqa: BLE001 - any ConfigError-family loud failure
        result["duplicate_rejected"] = "duplicate" in str(err).lower()

    # a declared parameter ABSENT from the raw -> loud failure at process()
    missing_feats = Features(variables={"parameters": ["mu", "npv", "absent_param"]})
    try:
        missing_feats.process(Bundle({"raw": {"parameters": raw}}), slice(None), Mode.FIT)
        result["missing_rejected"] = False
    except Exception:  # noqa: BLE001 - a numpy field KeyError / ValueError is loud
        result["missing_rejected"] = True

    return result


def _film1_datapath_forward(*, corrupt: bool) -> dict[str, Any]:
    """End-to-end: a FiLM-wired `StreamEmbed` consumes data-path ``inputs.parameters``.

    Builds a `StreamEmbed` with an INPUT-layer ``featurewise:`` block, binds it
    against a synthetic schema, drives its forward on a `Bundle` carrying the
    data-path-produced ``inputs.parameters`` (declared order, rank-2), and asserts
    the embed output is BITWISE vs an independent v1 ``InitNet`` + v1
    ``FeaturewiseTransformation`` (weight-transferred). This proves the FiLM
    consumes the contract-produced parameters tensor — not just a hand-built one.

    Returns
    -------
    dict[str, Any]
        ``{bitwise, max_abs_diff, ref_shape, got_shape}``.
    """
    from salt.models.dense import Dense as V1Dense  # noqa: PLC0415
    from salt.models.featurewise import FeaturewiseTransformation as V1FW  # noqa: PLC0415

    from salt.core.nn import StreamEmbed  # noqa: PLC0415
    from salt.core.nn import ResolvedSchema  # noqa: PLC0415
    from salt.core.graph import Bundle, Mode  # noqa: PLC0415

    torch.manual_seed(2)
    n_in = 5  # the tracks input width (the embed INPUT feature count)
    # -- v2 FiLM-wired StreamEmbed -----------------------------------------
    dc = _film1_dense_cfg()
    se = StreamEmbed(
        stream="tracks",
        out_dim=_FILM1_OUT_DIM,
        dense=dict(dc),
        featurewise={
            "dense_config_scale": dict(dc),
            "dense_config_bias": dict(dc),
        },
    )
    se.name = "track_embed"
    schema = ResolvedSchema(
        widths={"normed.tracks": n_in, "inputs.parameters": _FILM1_NUM_PARAMS},
        fields={"normed.tracks": tuple(f"v{i}" for i in range(n_in))},
    )
    se.bind(schema)
    se.eval()

    # -- v1 oracle: InitNet(featurewise=...) with NO attach_global ----------
    # (v1 applies the input FiLM to x BEFORE net(x); initnet.py:85-89). Build a
    # bare v1 Dense + v1 FiLM mirroring the StreamEmbed's net + featurewise.
    v1_fw = V1FW(
        layer="input",
        variables={"parameters": list(_FILM1_PARAM_NAMES)},
        dense_config_scale={**dc, "output_size": n_in},
        dense_config_bias={**dc, "output_size": n_in},
    )
    v1_net = V1Dense(input_size=n_in, output_size=_FILM1_OUT_DIM, **dc)
    # transfer the v2 weights into the v1 oracle (v2 is the system under test;
    # the v1 oracle gets v2's weights so a faithful port matches bit-for-bit).
    v1_fw.load_state_dict(se.featurewise.state_dict())
    v1_net.load_state_dict(se.net.state_dict())
    v1_fw.eval()
    v1_net.eval()

    # -- the data-path parameters tensor (declared order, rank-2) -----------
    params = torch.randn(_FILM1_BATCH, _FILM1_NUM_PARAMS)
    x = torch.randn(_FILM1_BATCH, _FILM1_TOKENS, n_in)
    b = Bundle({
        "normed": {"tracks": x},
        "inputs": {"parameters": params},
    })
    with torch.no_grad():
        got = se.forward(b, Mode.FIT)["embed.tracks"]
        # v1 reference: FiLM(x) then net (initnet.py:85-89)
        ref = v1_net(v1_fw({"parameters": params}, x))
    if corrupt:
        got = got + 1.0
    bitwise = ref.shape == got.shape and torch.equal(ref, got)
    return {
        "bitwise": bool(bitwise),
        "max_abs_diff": 0.0 if bitwise else float((ref - got).abs().max().item()),
        "ref_shape": tuple(ref.shape),
        "got_shape": tuple(got.shape),
    }


def run_film1(
    outdir: Path | str,
    *,
    corruption: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """FILM1: v2-native FiLM + PositionalEncoder + parameterised forward parity.

    For EVERY W-FILM surface (the FiLM at layer=input/encoder/global, the
    PositionalEncoder, and a parameterised data-path forward) the gate builds an
    independent v1 oracle, weight-transfers, and asserts the v2-native block's
    forward is BITWISE vs v1 on synthetic fixtures, PLUS proves the v2
    ``parameters`` order/normalisation/rank contract end-to-end via the real
    `Features` processor.

    Negative control (``test_gates_m7.py`` + the in-gate ``corruption`` hook,
    test-only, never on the CLI): the hook flips the ``corrupt`` flag on the FiLM
    + posenc + data-path sub-checks, so a perturbed v2 weight/output makes the
    bitwise assertions go red — proving the parity checks are not vacuous.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    print("=" * 96)
    print("FILM1 v2-native FiLM + PositionalEncoder + parameterised forward parity (M7 W-FILM)")
    print("=" * 96)

    corrupt = False
    if corruption is not None:
        corrupt = bool(corruption({"corrupt": False})["corrupt"])

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # 1. FeaturewiseTransformation at every layer
    fw_results = {
        layer: _film1_featurewise_parity(layer, corrupt=corrupt)
        for layer in ("input", "encoder", "global")
    }
    for layer, res in fw_results.items():
        checks[f"featurewise_{layer}_bitwise"] = res["bitwise"]
    details["featurewise"] = fw_results

    # 2. PositionalEncoder
    pe_res = _film1_posenc_parity(corrupt=corrupt)
    checks["posenc_bitwise"] = pe_res["bitwise"]
    details["posenc"] = pe_res

    # 3a. the parameters order/normalisation/rank contract (NOT corruptible —
    # it's a structural contract, always asserted)
    contract = _film1_param_contract()
    checks["param_rank2"] = bool(contract["rank2"])
    checks["param_declared_order"] = bool(contract["declared_order"])
    checks["param_fields_match"] = bool(contract["fields_match"])
    checks["param_duplicate_rejected"] = bool(contract["duplicate_rejected"])
    checks["param_missing_rejected"] = bool(contract["missing_rejected"])
    details["param_contract"] = contract

    # 3b. the data-path FiLM forward (FiLM consumes contract-produced parameters)
    dp_res = _film1_datapath_forward(corrupt=corrupt)
    checks["datapath_forward_bitwise"] = dp_res["bitwise"]
    details["datapath_forward"] = dp_res

    passed = all(checks.values())
    criterion = (
        "every v2-native W-FILM surface (FeaturewiseTransformation at layer=input/encoder/global, "
        "PositionalEncoder, and a parameterised data-path forward) is BITWISE (torch.equal) vs an "
        "independent v1 instance with weight-transfer on synthetic fixtures, AND the v2 parameters "
        "input enforces the order/normalisation/rank contract (declared NAMES == column order, "
        "rank-2 [B, n_params], duplicate/missing params fail loudly) via the real Features "
        "processor. The v1 tree is the FILM1 ORACLE (copied, never imported by the v2 port — RS1 "
        "stays 0). Negative control: a perturbed v2 weight/output flips the bitwise checks red."
    )
    report = _base_report("film1", passed, criterion, {"corrupted_by_test_hook": corrupt})
    report["checks"] = checks
    report["details"] = details

    # -- stdout table ----------------------------------------------------------
    print(f"{'surface':<40}{'bitwise/ok':>12}{'max|diff|':>14}")
    print("-" * 96)
    for layer, res in fw_results.items():
        print(f"{'FeaturewiseTransformation/' + layer:<40}{str(res['bitwise']):>12}"
              f"{res['max_abs_diff']:>14.3e}")
    print(f"{'PositionalEncoder':<40}{str(pe_res['bitwise']):>12}{pe_res['max_abs_diff']:>14.3e}")
    print(f"{'parameters/rank2':<40}{str(contract['rank2']):>12}")
    print(f"{'parameters/declared-order':<40}{str(contract['declared_order']):>12}")
    print(f"{'parameters/fields-match':<40}{str(contract['fields_match']):>12}")
    print(f"{'parameters/duplicate-rejected':<40}{str(contract['duplicate_rejected']):>12}")
    print(f"{'parameters/missing-rejected':<40}{str(contract['missing_rejected']):>12}")
    print(f"{'data-path FiLM forward':<40}{str(dp_res['bitwise']):>12}"
          f"{dp_res['max_abs_diff']:>14.3e}")
    _print_verdict("film1", passed, criterion, _emit_report(report, outdir, "film1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the M7 gate subcommand parser (cv1 / cv2 / cvf / ren1 / rs1).

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``cv1``, ``cv2``, ``cvf``, ``ren1`` and ``rs1``
        subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.gates_m7", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)
    helps = {
        "cv1": "Converter acceptance over the needs-M5+M6 denominator (convert+validate+plan==fix)",
        "cv2": "Converter hard-error path on the bit-rotted/parked configs (the dropped marker)",
        "cvf": "Converter target-producer fidelity (every task target has a real producer)",
        "ren1": "vector->global_object rename + StreamEmbed flag collapse (W1.5 wave R)",
        "rs1": "Residual v1-import scanner over production salt/core (W2 worklist + W2c gate)",
        "film1": "v2-native FiLM + PositionalEncoder + parameterised forward parity vs v1 (W-FILM)",
    }
    for gate, help_text in helps.items():
        p = sub.add_parser(gate, help=help_text)
        p.add_argument("--outdir", type=Path, required=True, help="report output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an M7 gate from the command line.

    Returns
    -------
    int
        0 if the gate passed, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    runner = {
        "cv1": run_cv1,
        "cv2": run_cv2,
        "cvf": run_cvf,
        "ren1": run_ren1,
        "rs1": run_rs1,
        "film1": run_film1,
    }[args.gate]
    code, _ = runner(args.outdir)
    return code


if __name__ == "__main__":
    sys.exit(main())
