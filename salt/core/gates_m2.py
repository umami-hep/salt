"""M2 gates harness — G1-G5 for the GN2v2 fit-parity milestone (plan 05, stage D; design §9.5).

Five standalone gates, each a subcommand of ``python -m salt.core.gates_m2``,
each writing a machine-readable ``<gate>_report.json`` into ``--outdir`` and a
human-readable table to stdout, exiting non-zero on failure. EVERY data path
is a CLI argument — no machine paths live in this file (design §5 placeholder
policy); the experiment supplies real paths, the test suite supplies tmp-dir
dummy files from the salt generators.

Gate criteria (each derived/justified in its ``run_g*`` docstring):

- **G1 dataset parity** — v1 ``SaltDataset`` vs the v2 pipeline on the same
  file and the same row slices: BITWISE-identical tensors on every leaf.
- **G2 fit smoke** — a short ``salt2 fit`` of the shipped GN2v2 config on
  CPU: per-step train losses all finite and decreasing front-to-back.
- **G3 loss-curve parity** — small v1 GN2 vs config-built v2 with TRANSFERRED
  weights, trained side-by-side on identical batch sequences with identical
  optimizers: per-step |loss diff| within a step-indexed budget
  (`g3_budget`), plus explicit task-weight and class-dict weighting parity.
- **G4 throughput** — v1 vs v2 train dataloader on the same file with the
  same batch size / workers / demanded columns: MEDIAN over interleaved
  timed repeats of (v2 jets/s / v1 jets/s) >= 0.97, with per-repeat numbers,
  spread, and an equal-work v2 control in the report.
- **G5 resume round-trip** — fit K, checkpoint, resume K == uninterrupted 2K
  (loss-curve continuity), plus a compile-interop smoke (in-place
  ``nn.Module.compile``; checkpoints interchangeable with uncompiled runs,
  state-dict keys stable).

Negative controls (the pytest suite, ``test_gates_m2.py``): G1 must FAIL on a
mis-mapped label key, G3 must FAIL on a perturbed transferred weight — the
hooks exist as Python-only keyword arguments (`run_g1` ``corruption``,
`run_g3` ``perturb``), never exposed on the CLI, mirroring the plan-04
``modules_hook`` pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from lightning import Callback, Trainer
from torch import Tensor, nn

from salt.core.data import Features, GraphDataModule, GraphDataset, H5StructuredReader, Labels
from salt.core.graph import Bundle, Executor, Mode
from salt.core.main import CONFIG_DIR
from salt.core.main import main as salt2_main
from salt.core.nn import bind_all, map_v1_state_dict, materialise_all, resolve_bind_schema
from salt.core.saltmodule import SaltModule
from salt.core.schema import dump_schema, save_schema
from salt.data.datamodules import SaltDataModule
from salt.data.datasets import SaltDataset
from salt.models.inputnorm import InputNorm
from salt.tests.core.gn2_fixture import JET_VARIABLES, TRACK_VARIABLES, build_test_gn2
from salt.tests.core.gn2v2_fixture import build_gn2v2_modules, compile_gn2v2

__all__ = [
    "G3_ATOL0",
    "G3_BUDGET_CAP",
    "G3_GROWTH",
    "G4_REPEATS",
    "G4_THRESHOLD",
    "G5_CONTINUITY_ATOL",
    "g3_budget",
    "main",
    "run_g1",
    "run_g2",
    "run_g3",
    "run_g4",
    "run_g5",
]

DEFAULT_JET_LABELS = ("flavour_label",)
DEFAULT_TRACK_LABELS = ("ftagTruthOriginLabel", "ftagTruthVertexIndex")

G3_ATOL0 = 1e-6
"""Step-0 absolute loss-diff ceiling — the A2 forward-equivalence bound.

The v2 production task path materialises per-stream slices via `Split` where
v1 boolean-indexes the full register-augmented sequence (design §3.3):
mathematically equal, not bitwise, measured <= 1e-6 absolute on every pred
and per-task loss (``test_v1_transfer.py``). Step 0 of G3 is exactly that
comparison, so it inherits exactly that bound.
"""

G3_GROWTH = 2.0
"""Per-step multiplicative budget growth for G3 (see `g3_budget`)."""

G3_BUDGET_CAP = 1e-3
"""Absolute ceiling on the G3 per-step budget (see `g3_budget`).

Without a cap the exponential envelope exceeds typical loss magnitudes
beyond step ~23, so a CLI run with a large ``--steps`` would silently
ungate the late steps (critic finding, plan 05 stage E). 1e-3 keeps every
step meaningfully gated at loss scale ~1-10 while staying far above the
measured drift (exactly 0.0 at the default K=10).
"""

G4_THRESHOLD = 0.97
"""G4 default PASS threshold: v2 throughput >= 0.97x v1 (plan 05, <= 3% loss)."""

G4_REPEATS = 3
"""G4 default timed repeats per side, interleaved v1/v2 (see `run_g4`).

A single timed sample cannot resolve a 3% threshold: back-to-back identical
runs on the real 1.1 GB file showed ~8% run-to-run spread on the v2 side
(critic finding, plan 05 stage E). The gate therefore times >= 3 interleaved
repeats per side (interleaving cancels slow machine drift), gates on the
MEDIAN per-repeat ratio, and reports the per-repeat numbers + spread so a
near-threshold verdict is interpretable.
"""

G5_CONTINUITY_ATOL = 1e-7
"""G5 resume-continuity ceiling on per-step |loss(uninterrupted) - loss(resumed)|.

With identical weights/optimizer/scheduler state (exact ``torch.save``
round-trips), an identical pinned batch sequence, and deterministic CPU
torch-math kernels, the resumed losses are expected BITWISE equal to the
uninterrupted run (observed 0.0); 1e-7 (under one float32 ulp at loss scale
~1-10) only allows for kernel scheduling noise without ever masking a real
state-restoration bug, which shows up orders of magnitude above this.
"""

_G5_LRS = {"initial": 1e-3, "max": 5e-3, "end": 1e-4, "pct_start": 0.1}


# ---------------------------------------------------------------------------
# shared helpers
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
    path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return path


def _print_verdict(gate: str, passed: bool, criterion: str, report_path: Path) -> None:
    """Print the closing verdict block every gate ends with."""
    print("=" * 96)
    print(f"GATE {gate.upper()}: {'PASS' if passed else 'FAIL'} — criterion: {criterion}")
    print(f"report: {report_path}")


def _base_report(gate: str, passed: bool, criterion: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the common report envelope shared by all five gates.

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
        "environment": {"torch": torch.__version__, "device": "cpu"},
    }


def _norm_dict_variables(
    norm_dict: Path | str, streams: Sequence[str] = ("jets", "tracks")
) -> dict[str, list[str]]:
    """Derive the per-stream variable lists from a norm-dict YAML.

    The norm dict is the one artifact that already names exactly the trained
    input columns per stream, so deriving the G1/G4 column set from it keeps
    both gates file-agnostic (dummy and real files alike) with no hardcoded
    variable lists.

    Returns
    -------
    dict[str, list[str]]
        ``{stream: [variable, ...]}`` for the requested streams.

    Raises
    ------
    ValueError
        If a requested stream is missing from the norm dict.
    """
    with open(norm_dict) as fh:
        nd = yaml.safe_load(fh)
    missing = [stream for stream in streams if stream not in nd]
    if missing:
        raise ValueError(f"streams {missing} not in norm dict {norm_dict} (has {sorted(nd)})")
    return {stream: list(nd[stream]) for stream in streams}


def _compare_tensors(key: str, ref: Tensor, got: Tensor) -> dict[str, Any]:
    """Compare one v2 tensor against its v1 reference (shape, dtype, bitwise).

    Returns
    -------
    dict[str, Any]
        Comparison record: shapes, dtypes, bitwise flag, max abs diff and
        pass verdict (PASS iff shapes, dtypes and every value match).
    """
    record: dict[str, Any] = {
        "key": key,
        "ref_shape": tuple(ref.shape),
        "got_shape": tuple(got.shape),
        "ref_dtype": str(ref.dtype),
        "got_dtype": str(got.dtype),
    }
    if ref.shape != got.shape or ref.dtype != got.dtype:
        record.update(bitwise=False, max_abs_diff=None, passed=False, note="shape/dtype mismatch")
        return record
    bitwise = torch.equal(ref, got)
    max_abs = 0.0 if bitwise else float((ref.double() - got.double()).abs().max().item())
    record.update(bitwise=bitwise, max_abs_diff=max_abs, passed=bitwise, note="")
    return record


def _fmt_cmp_row(record: dict[str, Any], extra: str = "") -> str:
    """Format one comparison record as a fixed-width table row.

    Returns
    -------
    str
        The formatted row.
    """
    if record["ref_shape"] == record["got_shape"]:
        shape = str(tuple(record["got_shape"]))
    else:
        shape = f"ref{tuple(record['ref_shape'])} != got{tuple(record['got_shape'])}"
    diff = "n/a" if record["max_abs_diff"] is None else f"{record['max_abs_diff']:.3e}"
    status = "PASS" if record["passed"] else "FAIL"
    return f"{extra}{record['key']:<38}{shape:<26}{diff:>12}{status:>7}"


# ---------------------------------------------------------------------------
# G1 — dataset parity (v1 SaltDataset vs v2 pipeline, bitwise)
# ---------------------------------------------------------------------------


def run_g1(
    file: Path | str,
    norm_dict: Path | str,
    outdir: Path | str,
    *,
    n_batches: int = 5,
    batch_size: int = 1000,
    jet_vars: Sequence[str] | None = None,
    track_vars: Sequence[str] | None = None,
    jet_labels: Sequence[str] = DEFAULT_JET_LABELS,
    track_labels: Sequence[str] = DEFAULT_TRACK_LABELS,
    corruption: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """G1: v1 ``SaltDataset`` vs the v2 pipeline — identical tensors, same slices.

    Both sides are configured with the SAME column set: variables default to
    the norm-dict per-stream lists (`_norm_dict_variables` — works for dummy
    and real files alike), labels to the GN2 training labels. The v2 reader
    gets the schema dumped live from ``file`` (design §2.6), so vector flags
    and field validation come from the file itself. Compared per slice:
    ``inputs.jets/tracks`` (f32), ``masks.tracks`` (bool, True = padded) and
    every ``labels.<stream>.<label>`` (int64) — the explicit v1<->v2 key map
    of ``test_data_pipeline.assert_batches_match``. Sequential slices from
    row 0 plus one odd-sized tail probe (off-by-one coverage on the last
    partial batch). PASS requires BITWISE equality on every leaf: the v2
    read path is a faithful port (design §6.1), so any diff is a wiring bug.

    Parameters
    ----------
    file : Path | str
        Input H5 file (dummy or real).
    norm_dict : Path | str
        Norm-dict YAML; defines the default per-stream variable lists.
    outdir : Path | str
        Report output directory.
    n_batches : int, optional
        Number of sequential slices to compare, by default 5.
    batch_size : int, optional
        Rows per slice, by default 1000.
    jet_vars, track_vars : Sequence[str] | None, optional
        Explicit variable lists overriding the norm-dict derivation.
    jet_labels, track_labels : Sequence[str], optional
        Labels demanded from both sides, by default the GN2 training labels.
    corruption : Callable | None, optional
        TEST-ONLY hook: applied to the v2 nested batch dict before
        comparison so the negative control can prove the gate fails on a
        mis-mapped label key. Never exposed on the CLI.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every compared leaf is bitwise.
    """
    outdir = Path(outdir)
    variables = _norm_dict_variables(norm_dict)
    if jet_vars is not None:
        variables["jets"] = list(jet_vars)
    if track_vars is not None:
        variables["tracks"] = list(track_vars)
    labels = {"jets": list(jet_labels), "tracks": list(track_labels)}

    v1 = SaltDataset(
        filename=file, norm_dict=norm_dict, variables=variables, stage="fit", labels=labels
    )
    reader = H5StructuredReader(
        groups={"jets": {}, "tracks": {}}, schema=dump_schema(file), filename=file
    )
    sinks = [
        "inputs.jets",
        "inputs.tracks",
        "masks.tracks",
        *[f"labels.jets.{label}" for label in labels["jets"]],
        *[f"labels.tracks.{label}" for label in labels["tracks"]],
    ]
    v2 = GraphDataset(
        {"reader": reader, "features": Features(variables=variables), "labels": Labels()},
        mode=Mode.FIT,
        sinks=sinks,
    )

    n_rows = len(v1)
    length_match = n_rows == len(v2)
    slices = [
        np.s_[start : min(start + batch_size, n_rows)]
        for start in range(0, min(n_batches * batch_size, n_rows), batch_size)
    ]
    tail = max(0, n_rows - max(1, batch_size // 3) - 1)
    slices.append(np.s_[tail:n_rows])  # odd-sized tail probe

    rows: list[dict[str, Any]] = []
    for rng in slices:
        v1_inputs, v1_masks, v1_labels = v1[rng]
        batch = v2[rng]
        if corruption is not None:
            batch = corruption(batch)
        span = f"[{rng.start}:{rng.stop}]"
        rows.extend(
            {"rows": span}
            | _compare_tensors(f"inputs.{stream}", v1_inputs[stream], batch["inputs"][stream])
            for stream in ("jets", "tracks")
        )
        rows.append(
            {"rows": span}
            | _compare_tensors("masks.tracks", v1_masks["tracks"], batch["masks"]["tracks"])
        )
        for stream, stream_labels in labels.items():
            rows.extend(
                {"rows": span}
                | _compare_tensors(
                    f"labels.{stream}.{label}",
                    v1_labels[stream][label],
                    batch["labels"][stream][label],
                )
                for label in stream_labels
            )

    passed = length_match and all(record["passed"] for record in rows)
    criterion = "bitwise torch.equal on every inputs/masks/labels leaf, every compared slice"
    report = _base_report(
        "g1_dataset_parity",
        passed,
        criterion,
        {
            "file": str(file),
            "norm_dict": str(norm_dict),
            "n_rows": n_rows,
            "length_match": length_match,
            "batch_size": batch_size,
            "slices": [[rng.start, rng.stop] for rng in slices],
            "variables": variables,
            "labels": labels,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report["comparisons"] = rows

    print("=" * 96)
    print("G1 dataset parity — v1 SaltDataset vs v2 GraphDataset, bitwise criterion")
    print("=" * 96)
    print(f"file: {file} ({n_rows:,} rows; length match: {length_match})")
    print(f"{'rows':<14}{'leaf':<38}{'shape':<26}{'max|diff|':>12}{'status':>7}")
    print("-" * 96)
    for record in rows:
        print(_fmt_cmp_row(record, extra=f"{record['rows']:<14}"))
    _print_verdict("g1", passed, criterion, _emit_report(report, outdir, "g1"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# G2 — fit smoke (salt2 fit on the shipped config, finite decreasing loss)
# ---------------------------------------------------------------------------


def run_g2(
    train_file: Path | str,
    val_file: Path | str,
    norm_dict: Path | str,
    outdir: Path | str,
    *,
    schema: Path | str | None = None,
    config: Path | str | None = None,
    steps: int = 30,
    batch_size: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """G2: short CPU ``salt2 fit`` of the shipped GN2v2 config — finite, decreasing loss.

    Runs the REAL CLI surface (``salt.core.main.main`` with the ``fit``
    subcommand and the gn2v2-dummy config's documented required overrides),
    capped at `steps` optimizer steps via ``--trainer.max_steps``, with a
    ``CSVLogger`` injected so per-step ``train/loss`` lands on disk.
    PASS criteria:

    - the CLI returns 0;
    - >= 90% of the expected per-step losses were logged (CSV flush slack);
    - every logged train/val loss is finite;
    - mean(train/loss over the last quartile of steps) < mean(first
      quartile) — random-init CE/BCE losses must move toward the label
      priors even on the random-label dummy file, and any NaN/explosion in
      the optimizer wiring fails the finiteness check first.

    Parameters
    ----------
    train_file, val_file : Path | str
        Training/validation H5 files.
    norm_dict : Path | str
        Norm dict for the model's `Normaliser`.
    outdir : Path | str
        Report + run-artifact output directory (trainer root dir, CSV logs).
    schema : Path | str | None, optional
        Schema artifact for the reader (recommended; the shipped config also
        runs without one), by default None.
    config : Path | str | None, optional
        Config YAML, by default the shipped ``gn2v2-dummy.yaml``.
    steps : int, optional
        Optimizer steps to run, by default 30.
    batch_size : int | None, optional
        ``--data.batch_size`` override, by default the config's value.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = Path(config) if config is not None else CONFIG_DIR / "gn2v2-dummy.yaml"
    csv_dir = outdir / "csv"
    logger_cfg = {
        "class_path": "lightning.pytorch.loggers.CSVLogger",
        "init_args": {"save_dir": str(csv_dir)},
    }
    argv = [
        "fit",
        "--config",
        str(config),
        f"--data.train_file={train_file}",
        f"--data.val_file={val_file}",
        f"--model.modules.norm.init_args.norm_dict={norm_dict}",
        f"--trainer.default_root_dir={outdir}",
        "--trainer.accelerator=cpu",
        "--trainer.max_epochs=-1",
        f"--trainer.max_steps={steps}",
        "--trainer.log_every_n_steps=1",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.enable_progress_bar=false",
        f"--trainer.logger={json.dumps(logger_cfg)}",
    ]
    if schema is not None:
        argv.append(f"--data.modules.reader.init_args.schema={schema}")
    if batch_size is not None:
        argv.append(f"--data.batch_size={batch_size}")

    error = ""
    try:
        rc = salt2_main(argv)
    except Exception as err:  # noqa: BLE001 - the gate reports any CLI failure as FAIL
        rc, error = 1, f"{type(err).__name__}: {err}"

    train_losses: list[float] = []
    val_losses: list[float] = []
    metrics_files = sorted(csv_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
    if metrics_files:
        with open(metrics_files[-1]) as fh:
            for row in csv.DictReader(fh):
                if row.get("train/loss"):
                    train_losses.append(float(row["train/loss"]))
                if row.get("val/loss"):
                    val_losses.append(float(row["val/loss"]))

    quart = max(1, len(train_losses) // 4)
    first = float(np.mean(train_losses[:quart])) if train_losses else float("nan")
    last = float(np.mean(train_losses[-quart:])) if train_losses else float("nan")
    checks = {
        "cli_returned_zero": rc == 0,
        "enough_steps_logged": len(train_losses) >= int(0.9 * steps),
        "all_train_losses_finite": bool(np.isfinite(train_losses).all()) if train_losses else False,
        "all_val_losses_finite": bool(np.isfinite(val_losses).all()) if val_losses else False,
        "loss_decreasing": bool(last < first) if train_losses else False,
    }
    passed = all(checks.values())
    criterion = (
        "salt2 fit exits 0; per-step train losses all finite; "
        "mean(last quartile) < mean(first quartile); val losses finite"
    )
    report = _base_report(
        "g2_fit_smoke",
        passed,
        criterion,
        {
            "config": str(config),
            "train_file": str(train_file),
            "val_file": str(val_file),
            "norm_dict": str(norm_dict),
            "schema": str(schema) if schema is not None else None,
            "steps": steps,
            "batch_size": batch_size,
            "argv": argv,
        },
    )
    report["checks"] = checks
    report["error"] = error
    report["first_quartile_mean"] = first
    report["last_quartile_mean"] = last
    report["train_losses"] = train_losses
    report["val_losses"] = val_losses

    print("=" * 96)
    print("G2 fit smoke — salt2 fit (gn2v2 config), CPU, finite decreasing loss")
    print("=" * 96)
    print(f"config: {config}")
    print(f"steps logged: {len(train_losses)}/{steps}  val points: {len(val_losses)}")
    if train_losses:
        print(
            f"train/loss: first={train_losses[0]:.5f} last={train_losses[-1]:.5f} "
            f"mean(first quartile)={first:.5f} mean(last quartile)={last:.5f}"
        )
    if error:
        print(f"CLI error: {error}")
    for name, ok in checks.items():
        print(f"  {name:<28} {'PASS' if ok else 'FAIL'}")
    _print_verdict("g2", passed, criterion, _emit_report(report, outdir, "g2"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# G3 — loss-curve parity (v1 vs weight-transferred v2, identical batches)
# ---------------------------------------------------------------------------


def g3_budget(step: int) -> float:
    """Per-step G3 loss-diff budget: ``min(G3_ATOL0 * G3_GROWTH**step, G3_BUDGET_CAP)``.

    Derivation (documented per plan 05 risk 1). At step 0 both sides hold
    IDENTICAL parameters (strict state-dict transfer), so the loss diff is
    pure forward-path reassociation — the v2 Split path materialises
    per-stream slices where v1 boolean-indexes the register-augmented
    sequence (design §3.3) — bounded by the measured A2 forward equivalence,
    `G3_ATOL0` = 1e-6 (``test_v1_transfer.py``). Each optimizer step then
    converts any gradient discrepancy of the same origin into a parameter
    discrepancy, which the next forward and AdamW's normalised update can
    compound; an envelope that DOUBLES the accumulated discrepancy per step
    (`G3_GROWTH` = 2) is conservative for this 16-dim fixture at lr <= 1e-3.
    Measured (dummy file, K=10, defaults): the diffs are exactly 0.0 at
    EVERY step — on a single-sequence-stream GN2 with valid-first batches,
    Split's contiguous slice happens to materialise bitwise-identically to
    v1's boolean index, so the whole budget is currently headroom. The
    nonzero budget stays because that bitwise coincidence is NOT a contract
    across BLAS paths / thread counts / multi-stream layouts (the
    ``test_v1_transfer.py`` docstring spells out why), and the report's
    ``drift_growth_per_step`` field recomputes the actual envelope on every
    run. The exponential form makes the gate meaningful only for short
    horizons (budget ~1e-3 by step 10, the default K) — exactly the regime
    where per-step comparison is informative; long-horizon agreement is the
    cluster training experiment's job (cosmetic loss-curve parity at scale,
    plan 05 follow-up). The envelope is capped at `G3_BUDGET_CAP` so a large
    ``--steps`` value cannot quietly ungate the late steps.

    Scope (documented per the stage-E critic): G3 measures loss parity of
    the raw module dicts under harness-built bare ``AdamW`` — the PRODUCTION
    optimizer/scheduler path (``SaltModule.configure_optimizers`` +
    OneCycleLR) is a line-for-line v1 port whose parity rests on that code
    identity plus G2/G5 (which exercise it end-to-end but not against v1);
    fp32-invisible optimizer differences (e.g. ``1 - lr*wd == 1.0``) are
    below this gate's resolution by construction. The cluster-scale training
    follow-up logs v1-vs-v2 curves through the full ``salt2 fit`` path.

    Returns
    -------
    float
        The absolute |loss_v1 - loss_v2| budget at ``step``.
    """
    return min(G3_ATOL0 * G3_GROWTH**step, G3_BUDGET_CAP)


def run_g3(
    file: Path | str,
    norm_dict: Path | str,
    class_dict: Path | str,
    outdir: Path | str,
    *,
    steps: int = 10,
    batch_size: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
    model_seed: int = 42,
    perturb: float = 0.0,
) -> tuple[int, dict[str, Any]]:
    """G3: per-step loss parity, v1 GN2 vs weight-transferred v2, identical batches.

    Construction: the small v1 GN2 (`build_test_gn2`, fixture widths) with
    its `InputNorm` rebuilt from the GIVEN ``--norm-dict`` and its
    ``track_origin`` CE loss given the class weights from the GIVEN
    ``--class-dict`` (exactly what the v1 CLI's ``use_class_dict`` glue
    injects, cli.py:446-491); the equivalent config-built v2 module dict
    (`build_gn2v2_modules`, the gn2v2-dummy.yaml shape, with
    ``weight_source: {from_class_dict: ...}``). Weights move v1->v2 via
    `map_v1_state_dict` (strict load — checkpoint-load path, no
    materialise). Both sides then train `steps` AdamW steps (identical lr /
    weight-decay / defaults; per-parameter updates are order-independent) on
    the IDENTICAL batch sequence: full batches of the file, order fixed by
    ``numpy.random.default_rng(seed).permutation``, every tensor cloned per
    side per step (v1 mutates its input dicts).

    PASS requires, at every step k, ``|total_v1 - total_v2| <= g3_budget(k)``
    and the same bound per task — plus the weighting-parity block: per-task
    scalar weights equal, the transferred CE class-weight buffer bitwise
    equal to v1's, and a separately-built v2 that MATERIALISES
    ``weight_source`` from the class dict reproducing the same buffer (the
    fresh-fit path equals the v1 CLI injection).

    Parameters
    ----------
    file : Path | str
        H5 file providing batches; must carry the fixture variable set
        (``JET_VARIABLES``/``TRACK_VARIABLES`` — the dummy-file generators)
        and the GN2 training labels.
    norm_dict : Path | str
        Norm dict used by BOTH sides' normalisers.
    class_dict : Path | str
        Class dict providing the ``track_origin`` CE class weights.
    outdir : Path | str
        Report output directory (also receives the v1 fixture scratch dir).
    steps : int, optional
        Optimizer steps per side, by default 10 (the budget's meaningful
        horizon — see `g3_budget`).
    batch_size : int, optional
        Rows per batch, by default 100.
    lr : float, optional
        AdamW learning rate (both sides), by default 1e-3.
    weight_decay : float, optional
        AdamW weight decay (both sides), by default 1e-5 (v1's default).
    seed : int, optional
        Batch-order seed, by default 42.
    model_seed : int, optional
        v1 parameter-init seed, by default 42.
    perturb : float, optional
        TEST-ONLY negative-control hook: added to one transferred v2 weight
        tensor so the suite can prove the gate FAILS on a real parameter
        difference. Never exposed on the CLI; by default 0.0.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — report carries the full per-step curves.

    Raises
    ------
    ValueError
        If the class dict lacks the 8-class ``tracks.ftagTruthOriginLabel``
        entry the fixture's origin head requires.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    variables = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}

    # -- v1 side: fixture GN2 + the GIVEN norm dict + class-dict CE weights
    fixture_dir = outdir / "v1_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_test_gn2(fixture_dir, seed=model_seed)
    wrapper.norm = InputNorm(
        norm_dict=Path(norm_dict),
        variables=variables,
        global_object="jets",
        input_map={stream: stream for stream in variables},
    )
    with open(class_dict) as fh:
        cd = yaml.safe_load(fh)
    class_weights = cd.get("tracks", {}).get("ftagTruthOriginLabel")
    if class_weights is None or len(class_weights) != 8:
        raise ValueError(
            f"class dict {class_dict} needs tracks.ftagTruthOriginLabel with 8 entries "
            f"(the fixture origin head), got {class_weights!r}"
        )
    cw = torch.tensor(class_weights, dtype=torch.float32)
    origin_task = wrapper.model.tasks[1]
    assert origin_task.name == "track_origin", "fixture task order changed"
    origin_task.loss = nn.CrossEntropyLoss(weight=cw.clone())
    origin_task.loss.ignore_index = -1  # v1 ClassificationTask contract (task.py:122-124)

    # -- v2 side: config-built modules, FIT plan, strict weight transfer
    modules = build_gn2v2_modules(norm_dict, class_dict=class_dict)
    plan = compile_gn2v2(modules, Mode.FIT)
    bind_all(modules, resolve_bind_schema(plan))
    holder = nn.ModuleDict(modules)
    holder.load_state_dict(map_v1_state_dict(wrapper.state_dict(), modules), strict=True)

    # -- weighting parity: task weights, transferred buffer, materialise path
    task_weights = {
        name: {
            "v1": float(wrapper.model.tasks[i].weight),
            "v2": float(modules[name].weight),
        }
        for i, name in enumerate(("jets_classification", "track_origin", "track_vertexing"))
    }
    fresh = build_gn2v2_modules(norm_dict, class_dict=class_dict)
    bind_all(fresh, resolve_bind_schema(compile_gn2v2(fresh, Mode.FIT)))
    materialise_all(fresh)
    weighting = {
        "class_weights_from_dict": [float(w) for w in class_weights],
        "task_weights": task_weights,
        "task_weights_match": all(w["v1"] == w["v2"] for w in task_weights.values()),
        "transferred_buffer_matches_v1": bool(
            torch.equal(modules["track_origin"].task.loss.weight, origin_task.loss.weight)
        ),
        "transferred_buffer_matches_dict": bool(
            torch.equal(modules["track_origin"].task.loss.weight, cw)
        ),
        "materialised_buffer_matches_dict": bool(
            torch.equal(fresh["track_origin"].task.loss.weight, cw)
        ),
    }

    if perturb:
        with torch.no_grad():
            next(iter(modules["track_embed"].parameters())).add_(perturb)

    # -- identical batch sequence (read once, cloned per side per step)
    ds = SaltDataset(
        filename=file,
        norm_dict=norm_dict,
        variables=variables,
        stage="fit",
        labels={"jets": ["flavour_label"], "tracks": list(DEFAULT_TRACK_LABELS)},
    )
    n_full = len(ds) // batch_size
    order = np.resize(np.random.default_rng(seed).permutation(n_full), steps)
    batches = [ds[np.s_[int(i) * batch_size : (int(i) + 1) * batch_size]] for i in order]

    wrapper.train()
    holder.train()
    opt_v1 = torch.optim.AdamW(wrapper.parameters(), lr=lr, weight_decay=weight_decay)
    opt_v2 = torch.optim.AdamW(holder.parameters(), lr=lr, weight_decay=weight_decay)
    executor = Executor(plan)
    task_names = ("jets_classification", "track_origin", "track_vertexing")

    curve: list[dict[str, Any]] = []
    for step, (inputs, masks, labels) in enumerate(batches):
        _, v1_losses = wrapper(
            {key: tensor.clone() for key, tensor in inputs.items()},
            {key: tensor.clone() for key, tensor in masks.items()},
            {s: {key: t.clone() for key, t in d.items()} for s, d in labels.items()},
        )
        v1_total = sum(v1_losses.values())
        opt_v1.zero_grad()
        v1_total.backward()
        opt_v1.step()

        bundle = Bundle()
        for stream, tensor in inputs.items():
            bundle.set(f"inputs.{stream}", tensor.clone())
        for stream, tensor in masks.items():
            bundle.set(f"masks.{stream}", tensor.clone())
        for stream, stream_labels in labels.items():
            for label, tensor in stream_labels.items():
                bundle.set(f"labels.{stream}.{label}", tensor.clone())
        executed = executor.run(bundle, debug=step == 0)
        v2_total = executed.get("loss.total")
        opt_v2.zero_grad()
        v2_total.backward()
        opt_v2.step()

        budget = g3_budget(step)
        per_task = {
            task: {
                "v1": float(v1_losses[task].item()),
                "v2": float(executed.get(f"losses.{task}").item()),
                "abs_diff": float((v1_losses[task] - executed.get(f"losses.{task}")).abs().item()),
            }
            for task in task_names
        }
        total_diff = float((v1_total - v2_total).abs().item())
        curve.append({
            "step": step,
            "rows": [int(order[step]) * batch_size, (int(order[step]) + 1) * batch_size],
            "v1_total": float(v1_total.item()),
            "v2_total": float(v2_total.item()),
            "abs_diff": total_diff,
            "budget": budget,
            "tasks": per_task,
            "passed": total_diff <= budget
            and all(t["abs_diff"] <= budget for t in per_task.values()),
        })

    diffs = [point["abs_diff"] for point in curve]
    growth = [
        diffs[i + 1] / diffs[i] for i in range(len(diffs) - 1) if diffs[i] > 0 and diffs[i + 1] > 0
    ]
    passed = (
        all(point["passed"] for point in curve)
        and weighting["task_weights_match"]
        and weighting["transferred_buffer_matches_v1"]
        and weighting["transferred_buffer_matches_dict"]
        and weighting["materialised_buffer_matches_dict"]
    )
    criterion = (
        f"per step k (total and per task): |loss_v1 - loss_v2| <= "
        f"min({G3_ATOL0:g}*{G3_GROWTH:g}**k, {G3_BUDGET_CAP:g}) (see g3_budget derivation); "
        "task weights and class-dict CE weights identical both sides"
    )
    report = _base_report(
        "g3_loss_curve_parity",
        passed,
        criterion,
        {
            "file": str(file),
            "norm_dict": str(norm_dict),
            "class_dict": str(class_dict),
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "optimizer": "AdamW (torch defaults otherwise; identical both sides)",
            "batch_seed": seed,
            "model_seed": model_seed,
            "perturbed_by_test_hook": bool(perturb),
            "atol0": G3_ATOL0,
            "growth": G3_GROWTH,
            "budget_cap": G3_BUDGET_CAP,
        },
    )
    report["scope"] = (
        "G3 measures loss parity of the raw module dicts under harness-built bare AdamW; "
        "the production optimizer/scheduler path (SaltModule.configure_optimizers + OneCycleLR) "
        "is a line-for-line v1 port (modelwrapper.py:340-381) whose parity rests on that code "
        "identity plus G2/G5 — see g3_budget docstring"
    )
    report["weighting_parity"] = weighting
    report["curve"] = curve
    report["drift_growth_per_step"] = {
        "max": max(growth) if growth else None,
        "mean": float(np.mean(growth)) if growth else None,
    }

    print("=" * 96)
    print("G3 loss-curve parity — v1 GN2 vs weight-transferred v2, identical batches/optimizer")
    print("=" * 96)
    print(
        f"steps={steps} batch={batch_size} lr={lr:g} wd={weight_decay:g} "
        f"batch_seed={seed} model_seed={model_seed}"
    )
    print(f"{'step':<6}{'v1 total':>12}{'v2 total':>12}{'|diff|':>12}{'budget':>12}{'status':>8}")
    print("-" * 96)
    for point in curve:
        print(
            f"{point['step']:<6}{point['v1_total']:>12.6f}{point['v2_total']:>12.6f}"
            f"{point['abs_diff']:>12.3e}{point['budget']:>12.3e}"
            f"{'PASS' if point['passed'] else 'FAIL':>8}"
        )
    drift = report["drift_growth_per_step"]
    if drift["max"] is not None:
        print(f"measured drift growth/step: mean={drift['mean']:.2f} max={drift['max']:.2f}")
    print("weighting parity:")
    for name, value in weighting.items():
        if isinstance(value, bool):
            print(f"  {name:<36} {'PASS' if value else 'FAIL'}")
    _print_verdict("g3", passed, criterion, _emit_report(report, outdir, "g3"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# G4 — dataloader throughput (v1 vs v2, jets/s)
# ---------------------------------------------------------------------------


def _warm_loader(loader, count_jets: Callable[[Any], int], epochs: int) -> None:
    """Iterate ``epochs`` untimed epochs (page cache + persistent-worker spin-up)."""
    for _ in range(epochs):
        for batch in loader:
            count_jets(batch)


def _time_pass(loader, count_jets: Callable[[Any], int], epochs: int) -> dict:
    """Time one pass of ``epochs`` full epochs over a dataloader.

    Returns
    -------
    dict
        ``{"jets": int, "seconds": float, "jets_per_s": float}`` for the pass.
    """
    jets = 0
    start = time.perf_counter()
    for _ in range(epochs):
        for batch in loader:
            jets += count_jets(batch)
    elapsed = time.perf_counter() - start
    return {"jets": jets, "seconds": elapsed, "jets_per_s": jets / elapsed if elapsed else 0.0}


def _timing_summary(passes: Sequence[dict]) -> dict:
    """Summarise repeated timed passes: medians plus per-repeat numbers + spread.

    Returns
    -------
    dict
        ``jets``/``seconds``/``jets_per_s`` are the per-pass jet count and
        the MEDIANS over repeats (back-compatible report shape);
        ``*_per_repeat`` carry the raw samples and ``spread_frac`` is
        ``(max - min) / median`` of the per-repeat jets/s.
    """
    jps = [p["jets_per_s"] for p in passes]
    med = float(np.median(jps))
    return {
        "jets": passes[0]["jets"],
        "seconds": float(np.median([p["seconds"] for p in passes])),
        "jets_per_s": med,
        "jets_per_s_per_repeat": jps,
        "seconds_per_repeat": [p["seconds"] for p in passes],
        "spread_frac": float((max(jps) - min(jps)) / med) if med else 0.0,
    }


def run_g4(
    file: Path | str,
    norm_dict: Path | str,
    outdir: Path | str,
    *,
    batch_size: int = 1000,
    num_workers: int = 2,
    warmup_epochs: int = 1,
    timed_epochs: int = 1,
    repeats: int = G4_REPEATS,
    threshold: float = G4_THRESHOLD,
    num: int = -1,
    jet_vars: Sequence[str] | None = None,
    track_vars: Sequence[str] | None = None,
    jet_labels: Sequence[str] = DEFAULT_JET_LABELS,
    track_labels: Sequence[str] = DEFAULT_TRACK_LABELS,
    equal_work_control: bool = True,
) -> tuple[int, dict[str, Any]]:
    """G4: v1 vs v2 train-dataloader throughput on the same file — v2 >= threshold x v1.

    Methodology (the fairness contract, plan 05 + stage-E critic fixes):

    - **Same demanded columns.** Both sides are configured with the SAME
      variables (norm-dict derivation, as G1) and the SAME labels; v2's
      sink demand is exactly those features + labels. v2 then reads exactly
      features + labels + ``valid`` per stream (demand narrowing, design
      §6.1) while v1 reads whatever its production reader reads for the same
      configuration — the gate deliberately compares the two PRODUCTION
      paths a user gets for one and the same training config, and reports
      the actual per-stream read columns of both sides so any read-set
      difference is visible in the JSON rather than hidden.
    - **Same loader mechanics.** Both sides go through their datamodule's
      train path: ``DataLoader(batch_size=None, sampler=RandomBatchSampler
      (shuffle, drop_last))`` with identical ``batch_size``,
      ``num_workers``, ``persistent_workers``, ``prefetch_factor`` and
      ``pin_memory=False`` (CPU benchmark; H2D transfer is out of scope).
    - **Warmup, then interleaved repeats.** Both sides run their warmup
      epoch(s) first (cold page cache absorbed once — the train file fits
      in RAM — and persistent workers spun up), then `repeats` timed passes
      of ``timed_epochs`` epochs each are taken INTERLEAVED
      (v1,v2,v1,v2,...) so slow machine drift cancels. The gate verdict is
      the MEDIAN per-repeat ratio; per-repeat jets/s and the spread are in
      the report (see `G4_REPEATS` — single-shot timing cannot resolve the
      3% threshold).
    - **Equal-work control.** The production-path comparison lets v2's
      demand narrowing read FEWER global-object columns than v1 (the
      disclosed jets-column amplification, ``datasets.py:395-396``). When
      the read sets differ, a second v2 timing with its features expanded
      to v1's actual read set (same labels, ``valid`` auto-appended) is
      recorded in the report's ``equal_work`` block — making the
      narrowing-free core-read-path ratio visible in the formal artifact.
      PASS stays on the production-path ratio (that is what a user gets).

    Parameters
    ----------
    file : Path | str
        Train H5 file (the real open-data file for the formal gate run).
    norm_dict : Path | str
        Norm dict (variable derivation + v1 dataset requirement).
    outdir : Path | str
        Report output directory.
    batch_size : int, optional
        Rows per batch, by default 1000.
    num_workers : int, optional
        Dataloader workers BOTH sides, by default 2.
    warmup_epochs : int, optional
        Untimed epochs per side, by default 1.
    timed_epochs : int, optional
        Epochs per timed pass, by default 1.
    repeats : int, optional
        Interleaved timed passes per side, by default `G4_REPEATS` (3).
    threshold : float, optional
        PASS bound on the median v2/v1 jets/s ratio, by default
        `G4_THRESHOLD` (0.97).
    num : int, optional
        Row cap (``-1`` = all rows), by default -1.
    jet_vars, track_vars : Sequence[str] | None, optional
        Explicit variable overrides (default: norm-dict derivation).
    jet_labels, track_labels : Sequence[str], optional
        Labels demanded both sides, by default the GN2 training labels.
    equal_work_control : bool, optional
        Run the equal-work v2 control when the read sets differ, by
        default True.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    variables = _norm_dict_variables(norm_dict)
    if jet_vars is not None:
        variables["jets"] = list(jet_vars)
    if track_vars is not None:
        variables["tracks"] = list(track_vars)
    labels = {"jets": list(jet_labels), "tracks": list(track_labels)}
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }

    # -- v1 production path: SaltDataModule -> train_dataloader
    v1_dm = SaltDataModule(
        train_file=file,
        val_file=file,
        num_train=num,
        num_val=num,
        num_test=0,
        norm_dict=norm_dict,
        variables=variables,
        labels=labels,
        **loader_kwargs,
    )
    # setup() dereferences self.trainer (datamodules.py:230) — stub the two
    # attributes it reads; no Trainer is involved in a pure-loader benchmark.
    v1_dm.trainer = SimpleNamespace(is_global_zero=True, fast_dev_run=False)
    v1_dm.setup("fit")
    v1_loader = v1_dm.train_dataloader()
    v1_dm.train_dset[np.s_[0:1]]  # touch once to materialise the read buffers
    v1_columns = {
        stream: sorted(array.dtype.names) for stream, array in v1_dm.train_dset.arrays.items()
    }

    # -- v2 production path: GraphDataModule -> train_dataloader
    sinks = [
        *[f"inputs.{stream}" for stream in variables],
        "masks.tracks",
        *[f"labels.{s}.{label}" for s, stream_labels in labels.items() for label in stream_labels],
    ]
    v2_reader = H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=dump_schema(file))
    v2_dm = GraphDataModule(
        modules={
            "reader": v2_reader,
            "features": Features(variables=variables),
            "labels": Labels(),
        },
        train_file=file,
        val_file=file,
        num_train=num,
        num_val=num,
        sinks={Mode.FIT: sinks, Mode.VAL: sinks},
        **loader_kwargs,
    )
    v2_dm.setup("fit")
    v2_loader = v2_dm.train_dataloader()
    v2_dm.train_dset[np.s_[0:1]]
    v2_columns = {
        stream: sorted(buffer.dtype.names)
        for stream, buffer in v2_dm.train_dset.reader._buffers.items()  # noqa: SLF001 - report-only read-set introspection
    }

    v1_count = lambda batch: batch[0]["jets"].shape[0]  # noqa: E731 - local counter
    v2_count = lambda batch: batch["inputs"]["jets"].shape[0]  # noqa: E731 - local counter

    # warmups first (cold cache absorbed once), then interleaved timed passes
    _warm_loader(v1_loader, v1_count, warmup_epochs)
    _warm_loader(v2_loader, v2_count, warmup_epochs)
    repeats = max(1, repeats)
    v1_passes: list[dict] = []
    v2_passes: list[dict] = []
    ratios: list[float] = []
    for _ in range(repeats):
        v1_passes.append(_time_pass(v1_loader, v1_count, timed_epochs))
        v2_passes.append(_time_pass(v2_loader, v2_count, timed_epochs))
        v1_jps = v1_passes[-1]["jets_per_s"]
        ratios.append(v2_passes[-1]["jets_per_s"] / v1_jps if v1_jps else 0.0)
    del v1_loader, v2_loader  # release persistent workers
    v1_timing = _timing_summary(v1_passes)
    v2_timing = _timing_summary(v2_passes)

    ratio = float(np.median(ratios))
    same_jets = all(p["jets"] == v1_passes[0]["jets"] for p in (*v1_passes, *v2_passes))

    # equal-work control: v2 with features expanded to v1's actual read set
    # (minus labels — demanded anyway — and the auto-appended 'valid')
    equal_work: dict[str, Any] = {"ran": False}
    if equal_work_control:
        control_vars = {
            stream: [
                column
                for column in v1_columns.get(stream, [])
                if column != "valid" and column not in set(labels.get(stream, []))
            ]
            for stream in variables
        }
        if any(sorted(control_vars[s]) != sorted(variables[s]) for s in variables):
            ctl_dm = GraphDataModule(
                modules={
                    "reader": H5StructuredReader(
                        groups={"jets": {}, "tracks": {}}, schema=dump_schema(file)
                    ),
                    # non_finite_to_num: the extra v1 columns may carry NaNs
                    # (truth columns); v2 additionally materialises them as
                    # f32 features where v1 only READS them — the control is
                    # deliberately conservative (extra work on the v2 side)
                    "features": Features(variables=control_vars, non_finite_to_num=True),
                    "labels": Labels(),
                },
                train_file=file,
                val_file=file,
                num_train=num,
                num_val=num,
                sinks={Mode.FIT: sinks, Mode.VAL: sinks},
                **loader_kwargs,
            )
            ctl_dm.setup("fit")
            ctl_loader = ctl_dm.train_dataloader()
            ctl_dm.train_dset[np.s_[0:1]]
            ctl_columns = {
                stream: sorted(buffer.dtype.names)
                for stream, buffer in ctl_dm.train_dset.reader._buffers.items()  # noqa: SLF001 - report-only read-set introspection
            }
            _warm_loader(ctl_loader, v2_count, warmup_epochs)
            ctl_passes = [_time_pass(ctl_loader, v2_count, timed_epochs) for _ in range(repeats)]
            del ctl_loader
            ctl_timing = _timing_summary(ctl_passes)
            equal_work = {
                "ran": True,
                "variables": control_vars,
                "read_columns": ctl_columns,
                "v2_equal_work": ctl_timing,
                "ratio_vs_v1_production_median": (
                    ctl_timing["jets_per_s"] / v1_timing["jets_per_s"]
                    if v1_timing["jets_per_s"]
                    else 0.0
                ),
                "note": (
                    "v2 timed with its read set expanded to v1's actual per-stream columns "
                    "(removing the demand-narrowing I/O advantage); informational — PASS is "
                    "judged on the production-path median ratio"
                ),
            }

    passed = ratio >= threshold and same_jets
    criterion = (
        f"median over {repeats} interleaved repeats of (v2 jets/s / v1 jets/s) >= {threshold:g}, "
        f"each repeat timing {timed_epochs} epoch(s) after {warmup_epochs} warmup epoch(s); "
        "identical jet counts every pass"
    )
    report = _base_report(
        "g4_throughput",
        passed,
        criterion,
        {
            "file": str(file),
            "norm_dict": str(norm_dict),
            "num": num,
            "variables": variables,
            "labels": labels,
            "threshold": threshold,
            "warmup_epochs": warmup_epochs,
            "timed_epochs": timed_epochs,
            "repeats": repeats,
            **loader_kwargs,
        },
    )
    report["v1"] = v1_timing | {"read_columns": v1_columns}
    report["v2"] = v2_timing | {"read_columns": v2_columns}
    report["ratios_per_repeat"] = ratios
    report["ratio_v2_over_v1"] = ratio
    report["same_jet_count"] = same_jets
    report["equal_work"] = equal_work
    report["methodology"] = (
        "same variables+labels both sides (norm-dict derived); both sides' production "
        "datamodule train path (batch_size=None + RandomBatchSampler, identical loader kwargs, "
        "pin_memory off — CPU benchmark); both sides warmed up first, then timed passes taken "
        "interleaved v1,v2,v1,v2,... and the verdict judged on the MEDIAN per-repeat ratio "
        "(single-shot timing cannot resolve the 3% threshold — see G4_REPEATS); actual "
        "per-stream read columns reported for both sides, plus an equal-work v2 control when "
        "the read sets differ"
    )

    print("=" * 96)
    print("G4 throughput — v1 SaltDataModule vs v2 GraphDataModule train loaders")
    print("=" * 96)
    print(
        f"file: {file}\nbatch={batch_size} workers={num_workers} warmup={warmup_epochs} "
        f"timed={timed_epochs}/pass x {repeats} repeats rows={'all' if num == -1 else num}"
    )
    print(f"{'side':<6}{'jets/pass':>12}{'median jets/s':>16}{'spread':>9}  per-repeat jets/s")
    print("-" * 96)
    for side, timing in (("v1", v1_timing), ("v2", v2_timing)):
        per_repeat = ", ".join(f"{jps:,.0f}" for jps in timing["jets_per_s_per_repeat"])
        print(
            f"{side:<6}{timing['jets']:>12,}{timing['jets_per_s']:>16,.0f}"
            f"{timing['spread_frac']:>8.1%}  [{per_repeat}]"
        )
    print(
        f"per-repeat ratios: [{', '.join(f'{r:.4f}' for r in ratios)}] -> median {ratio:.4f} "
        f"(threshold {threshold:g}); same jet count: {same_jets}"
    )
    if equal_work["ran"]:
        print(
            f"equal-work control: v2 at v1's read set -> "
            f"{equal_work['v2_equal_work']['jets_per_s']:,.0f} jets/s, ratio vs v1 "
            f"{equal_work['ratio_vs_v1_production_median']:.4f} (informational)"
        )
    _print_verdict("g4", passed, criterion, _emit_report(report, outdir, "g4"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# G5 — resume round-trip + compile-interop smoke
# ---------------------------------------------------------------------------


class _StepLossRecorder(Callback):
    """Record the per-train-step total loss as Python floats."""

    def __init__(self) -> None:
        super().__init__()
        self.losses: list[float] = []

    def on_train_batch_end(self, _trainer, _module, outputs, _batch, _batch_idx) -> None:
        """Append the step's detached total loss."""
        self.losses.append(float(outputs["loss"].detach().item()))


class _StopAfterEpoch(Callback):
    """Stop a fit at the end of epoch ``n_epochs - 1``, leaving room to resume.

    The OneCycleLR total step count is fixed by the FIRST trainer's
    ``estimated_stepping_batches`` and restored verbatim on resume (v1
    semantics) — stopping early keeps the resumed run inside the schedule,
    so all G5 trainers share one ``max_epochs``.

    Parameters
    ----------
    n_epochs : int
        Number of epochs to run before stopping.
    """

    def __init__(self, n_epochs: int) -> None:
        super().__init__()
        self.n_epochs = n_epochs

    def on_train_epoch_end(self, trainer, _module) -> None:
        """Raise the stop flag once ``n_epochs`` epochs have completed."""
        if trainer.current_epoch + 1 >= self.n_epochs:
            trainer.should_stop = True


def run_g5(
    file: Path | str,
    norm_dict: Path | str,
    outdir: Path | str,
    *,
    k: int = 4,
    batch_size: int = 100,
    seed: int = 42,
) -> tuple[int, dict[str, Any]]:
    """G5: resume round-trip — fit K, checkpoint, resume K == uninterrupted 2K.

    All runs share one pinned data sequence: ``num_train = batch_size`` gives
    exactly ONE train batch per epoch (so K epochs = K optimizer steps and
    the weak-shuffling ``randperm(1)`` is the identity) — batch-order
    reproducibility across a restart is NOT part of the v1 resume contract
    (the sampler draws from the global RNG), so the gate pins the sequence
    to isolate what IS contracted: exact restoration of model weights,
    optimizer moments, OneCycleLR state, and the materialise-skip on resume
    (design §2.3). Runs (fresh `SaltModule` over `build_gn2v2_modules`,
    fixture scale, identical ``torch.manual_seed`` before each build):

    1. **A** — uninterrupted ``max_epochs=2K`` fit; per-step losses recorded.
    2. **B** — same config, stopped after epoch K-1, checkpoint saved, then
       a FRESH model resumes from it to 2K. Precondition: B's first-K losses
       match A's bitwise (identical-run sanity — if this fails the harness,
       not the contract, is broken). Continuity: ``max_k |A[K+i] - B[i]| <=``
       `G5_CONTINUITY_ATOL` (expected 0.0 — see the constant's docstring).
    3. **Compile interop smoke** — a fresh model with the encoder's composed
       transformer compiled IN PLACE (``nn.Module.compile(backend="eager")``
       — dynamo without inductor codegen, so no compiler toolchain is
       assumed; checkpoint interop, not kernel speed, is what's gated) fits
       K epochs and checkpoints: the saved state-dict keys must equal the
       uncompiled run's (no ``_orig_mod`` rewriting); an uncompiled model
       must resume from the compiled checkpoint and a compiled model from
       the uncompiled checkpoint, both to 2K with finite losses.

    Parameters
    ----------
    file : Path | str
        H5 file carrying the fixture variable set (dummy-file generators).
    norm_dict : Path | str
        Norm dict for the model's `Normaliser` (parity-style constants).
    outdir : Path | str
        Report + checkpoint output directory.
    k : int, optional
        Steps (= epochs) per half, by default 4.
    batch_size : int, optional
        Rows per (single) batch, by default 100.
    seed : int, optional
        ``torch.manual_seed`` before every model build, by default 42.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    schema_path = outdir / "schema.yaml"
    save_schema(dump_schema(file), schema_path)

    def make_dm() -> GraphDataModule:
        return GraphDataModule(
            modules={
                "reader": H5StructuredReader(groups={"jets": {}, "tracks": {}}, schema=schema_path),
                "features": Features(
                    variables={"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
                ),
                "labels": Labels(),
            },
            train_file=file,
            val_file=file,
            batch_size=batch_size,
            num_train=batch_size,  # exactly one batch per epoch (see docstring)
            num_val=batch_size,
            num_workers=0,
            pin_memory=False,
        )

    def make_model() -> SaltModule:
        torch.manual_seed(seed)
        return SaltModule(build_gn2v2_modules(norm_dict), lrs_config=dict(_G5_LRS))

    def make_trainer(*callbacks: Callback) -> Trainer:
        return Trainer(
            max_epochs=2 * k,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=1,
            log_every_n_steps=1,
            callbacks=list(callbacks),
        )

    # 1. uninterrupted reference: 2K steps
    rec_a = _StepLossRecorder()
    model_a = make_model()
    make_trainer(rec_a).fit(model_a, make_dm())
    plain_keys = sorted(model_a.state_dict())

    # 2. interrupted + resumed: K steps, checkpoint, K more
    rec_b1 = _StepLossRecorder()
    model_b = make_model()
    trainer_b = make_trainer(rec_b1, _StopAfterEpoch(k))
    trainer_b.fit(model_b, make_dm())
    ckpt_plain = outdir / "g5_plain.ckpt"
    trainer_b.save_checkpoint(ckpt_plain)
    rec_b2 = _StepLossRecorder()
    make_trainer(rec_b2).fit(make_model(), make_dm(), ckpt_path=ckpt_plain)

    pre_resume_max = max(
        (abs(a - b) for a, b in zip(rec_a.losses[:k], rec_b1.losses, strict=True)), default=0.0
    )
    continuity_max = max(
        (abs(a - b) for a, b in zip(rec_a.losses[k:], rec_b2.losses, strict=True)), default=0.0
    )
    checks = {
        "step_counts": len(rec_a.losses) == 2 * k
        and len(rec_b1.losses) == k
        and len(rec_b2.losses) == k,
        "pre_resume_runs_identical": pre_resume_max <= G5_CONTINUITY_ATOL,
        "resumed_curve_continuous": continuity_max <= G5_CONTINUITY_ATOL,
    }

    # 3. compile interop smoke (in-place nn.Module.compile, eager backend)
    rec_c = _StepLossRecorder()
    model_c = make_model()
    model_c.net["encoder"].encoder.compile(backend="eager")
    trainer_c = make_trainer(rec_c, _StopAfterEpoch(k))
    trainer_c.fit(model_c, make_dm())
    ckpt_compiled = outdir / "g5_compiled.ckpt"
    trainer_c.save_checkpoint(ckpt_compiled)
    compiled_keys = sorted(torch.load(ckpt_compiled, weights_only=False)["state_dict"])
    checks["compiled_state_dict_keys_stable"] = compiled_keys == plain_keys and not any(
        "_orig_mod" in key for key in compiled_keys
    )

    rec_c2 = _StepLossRecorder()  # compiled ckpt -> uncompiled resume
    make_trainer(rec_c2).fit(make_model(), make_dm(), ckpt_path=ckpt_compiled)
    rec_d = _StepLossRecorder()  # uncompiled ckpt -> compiled resume
    model_d = make_model()
    model_d.net["encoder"].encoder.compile(backend="eager")
    make_trainer(rec_d).fit(model_d, make_dm(), ckpt_path=ckpt_plain)
    checks["compiled_to_uncompiled_resume"] = len(rec_c2.losses) == k and all(
        np.isfinite(rec_c2.losses)
    )
    checks["uncompiled_to_compiled_resume"] = len(rec_d.losses) == k and all(
        np.isfinite(rec_d.losses)
    )

    passed = all(checks.values())
    criterion = (
        f"resumed per-step losses match the uninterrupted run within {G5_CONTINUITY_ATOL:g} "
        "(pinned single-batch sequence); compiled/uncompiled checkpoints interchangeable with "
        "stable state-dict keys"
    )
    report = _base_report(
        "g5_resume_round_trip",
        passed,
        criterion,
        {
            "file": str(file),
            "norm_dict": str(norm_dict),
            "k": k,
            "batch_size": batch_size,
            "seed": seed,
            "lrs_config": dict(_G5_LRS),
            "compile": "nn.Module.compile(backend='eager') on net.encoder.encoder, in place",
        },
    )
    report["checks"] = checks
    report["pre_resume_max_abs_diff"] = pre_resume_max
    report["continuity_max_abs_diff"] = continuity_max
    report["losses"] = {
        "uninterrupted": rec_a.losses,
        "interrupted_first_half": rec_b1.losses,
        "resumed_second_half": rec_b2.losses,
        "compiled_first_half": rec_c.losses,
        "compiled_to_uncompiled_resume": rec_c2.losses,
        "uncompiled_to_compiled_resume": rec_d.losses,
    }

    print("=" * 96)
    print("G5 resume round-trip — fit K, checkpoint, resume K == uninterrupted 2K + compile smoke")
    print("=" * 96)
    print(f"k={k} batch={batch_size} seed={seed} (one batch per epoch — see report methodology)")
    print(f"{'step':<6}{'uninterrupted':>16}{'interrupted/resumed':>22}{'|diff|':>12}")
    print("-" * 96)
    halves = [*rec_b1.losses, *rec_b2.losses]
    for i, (a, b) in enumerate(zip(rec_a.losses, halves, strict=True)):
        marker = " <- resume" if i == k else ""
        print(f"{i:<6}{a:>16.8f}{b:>22.8f}{abs(a - b):>12.3e}{marker}")
    for name, ok in checks.items():
        print(f"  {name:<36} {'PASS' if ok else 'FAIL'}")
    _print_verdict("g5", passed, criterion, _emit_report(report, outdir, "g5"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every gate subcommand."""
    parser.add_argument("--outdir", type=Path, required=True, help="report output directory")


def _build_parser() -> argparse.ArgumentParser:
    """Build the gate subcommand parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``g1``-``g5`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.gates_m2", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)

    g1 = sub.add_parser("g1", help="dataset parity: v1 SaltDataset vs v2 pipeline (bitwise)")
    _add_common(g1)
    g1.add_argument("--file", type=Path, required=True, help="input H5 file")
    g1.add_argument("--norm-dict", type=Path, required=True, help="norm dict YAML")
    g1.add_argument("--n-batches", type=int, default=5, help="sequential slices to compare")
    g1.add_argument("--batch-size", type=int, default=1000, help="rows per slice")
    g1.add_argument("--jet-vars", type=str, default=None, help="comma-separated override")
    g1.add_argument("--track-vars", type=str, default=None, help="comma-separated override")
    g1.add_argument("--jet-labels", type=str, default=",".join(DEFAULT_JET_LABELS))
    g1.add_argument("--track-labels", type=str, default=",".join(DEFAULT_TRACK_LABELS))

    g2 = sub.add_parser("g2", help="fit smoke: short salt2 fit, finite decreasing loss")
    _add_common(g2)
    g2.add_argument("--train-file", type=Path, required=True)
    g2.add_argument("--val-file", type=Path, required=True)
    g2.add_argument("--norm-dict", type=Path, required=True)
    g2.add_argument("--schema", type=Path, default=None, help="schema artifact (optional)")
    g2.add_argument("--config", type=Path, default=None, help="default: shipped gn2v2-dummy.yaml")
    g2.add_argument("--steps", type=int, default=30, help="optimizer steps to run")
    g2.add_argument("--batch-size", type=int, default=None, help="--data.batch_size override")

    g3 = sub.add_parser("g3", help="loss-curve parity: v1 vs weight-transferred v2")
    _add_common(g3)
    g3.add_argument("--file", type=Path, required=True, help="H5 with the fixture variable set")
    g3.add_argument("--norm-dict", type=Path, required=True)
    g3.add_argument("--class-dict", type=Path, required=True)
    g3.add_argument("--steps", type=int, default=10, help="optimizer steps per side")
    g3.add_argument("--batch-size", type=int, default=100)
    g3.add_argument("--lr", type=float, default=1e-3, help="AdamW lr (both sides)")
    g3.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW wd (both sides)")
    g3.add_argument("--seed", type=int, default=42, help="batch-order seed")
    g3.add_argument("--model-seed", type=int, default=42, help="v1 parameter-init seed")

    g4 = sub.add_parser("g4", help="throughput: v1 vs v2 train dataloader jets/s")
    _add_common(g4)
    g4.add_argument("--file", type=Path, required=True)
    g4.add_argument("--norm-dict", type=Path, required=True)
    g4.add_argument("--batch-size", type=int, default=1000)
    g4.add_argument("--num-workers", type=int, default=2)
    g4.add_argument("--warmup-epochs", type=int, default=1)
    g4.add_argument("--timed-epochs", type=int, default=1, help="epochs per timed pass")
    g4.add_argument(
        "--repeats", type=int, default=G4_REPEATS, help="interleaved timed passes per side"
    )
    g4.add_argument("--threshold", type=float, default=G4_THRESHOLD)
    g4.add_argument("--num", type=int, default=-1, help="row cap (-1 = all)")
    g4.add_argument("--jet-vars", type=str, default=None, help="comma-separated override")
    g4.add_argument("--track-vars", type=str, default=None, help="comma-separated override")
    g4.add_argument("--jet-labels", type=str, default=",".join(DEFAULT_JET_LABELS))
    g4.add_argument("--track-labels", type=str, default=",".join(DEFAULT_TRACK_LABELS))

    g5 = sub.add_parser("g5", help="resume round-trip + compile-interop smoke")
    _add_common(g5)
    g5.add_argument("--file", type=Path, required=True, help="H5 with the fixture variable set")
    g5.add_argument("--norm-dict", type=Path, required=True)
    g5.add_argument("--k", type=int, default=4, help="steps per half (epochs; 1 batch/epoch)")
    g5.add_argument("--batch-size", type=int, default=100)
    g5.add_argument("--seed", type=int, default=42)
    return parser


def _csv_list(value: str | None) -> list[str] | None:
    """Split a comma-separated CLI value into a list (None passes through).

    Returns
    -------
    list[str] | None
        The split values, or None.
    """
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one M2 gate from the command line.

    Returns
    -------
    int
        0 if the gate passed, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    if args.gate == "g1":
        code, _ = run_g1(
            args.file,
            args.norm_dict,
            args.outdir,
            n_batches=args.n_batches,
            batch_size=args.batch_size,
            jet_vars=_csv_list(args.jet_vars),
            track_vars=_csv_list(args.track_vars),
            jet_labels=_csv_list(args.jet_labels) or [],
            track_labels=_csv_list(args.track_labels) or [],
        )
    elif args.gate == "g2":
        code, _ = run_g2(
            args.train_file,
            args.val_file,
            args.norm_dict,
            args.outdir,
            schema=args.schema,
            config=args.config,
            steps=args.steps,
            batch_size=args.batch_size,
        )
    elif args.gate == "g3":
        code, _ = run_g3(
            args.file,
            args.norm_dict,
            args.class_dict,
            args.outdir,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            model_seed=args.model_seed,
        )
    elif args.gate == "g4":
        code, _ = run_g4(
            args.file,
            args.norm_dict,
            args.outdir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup_epochs=args.warmup_epochs,
            timed_epochs=args.timed_epochs,
            repeats=args.repeats,
            threshold=args.threshold,
            num=args.num,
            jet_vars=_csv_list(args.jet_vars),
            track_vars=_csv_list(args.track_vars),
            jet_labels=_csv_list(args.jet_labels) or [],
            track_labels=_csv_list(args.track_labels) or [],
        )
    else:
        code, _ = run_g5(
            args.file,
            args.norm_dict,
            args.outdir,
            k=args.k,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
