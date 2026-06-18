"""M3 gates harness — W1-W5 for the prediction-writer milestone (plan 06, stage C; design §9.5 M3).

Five standalone gates, each a subcommand of ``python -m salt.core.gates_m3``,
each writing a machine-readable ``<gate>_report.json`` into ``--outdir`` and a
human-readable table to stdout, exiting non-zero on failure. EVERY machine
data path is a CLI argument (design §5 placeholder policy); the dummy-file
gates generate their fixture data into ``--outdir`` from the salt generators
when no ``--file`` is given, so nothing here names a machine path.

Gate criteria (each derived/justified in its ``run_w*`` docstring):

- **W1 writer parity (dummy)** — a weight-matched v1 eval (``ModelWrapper`` +
  v1 `PredictionWriter`) vs the v2 ``salt2 test`` writer path on the same
  test file: identical H5 byte-schema (dataset names, column names + formats
  + ORDER, shapes) AND values — bitwise for input copies / labels / masks /
  ``VertexIndex``; converted probability columns bitwise-first with a
  printed <= 1e-6 justification fallback (never silent).
- **W2 writer parity (real file)** — the same comparison on a real
  open-data validation file (``--file``/``--norm-dict`` CLI args).
- **W3 negative controls** — the machinery must FAIL when it should: the
  TEST dead-preds hard error fires for a writer set that consumes only a
  subset of the produced predictions, and a deliberately corrupted v2
  output (two probability columns value-swapped via a python-only hook)
  fails the W1 comparison.
- **W4 ergonomics journey** — a custom `Writer` subclass plus a FOUR-LINE
  YAML override config produces a new column end-to-end through the real
  ``salt2 test`` surface (design §8 user story), with the shipped columns
  surviving the deep merge.
- **W5 metrics parity** — the v2 `ConfusionMatrix` callback accumulates
  exactly the values the v1 ``ConfusionMatrixCallback`` accumulates on the
  same weight-matched eval batches (element-wise identical label lists,
  identical integer counts matrices), with >= 2 distinct predicted classes
  per task (`_decollapse_heads` non-degeneracy bar).

Negative-control hooks (the pytest suite, ``test_gates_m3.py``): `run_w1`
takes a python-only ``corruption`` keyword (applied to the loaded v2 arrays
before comparison) so W3/tests can prove the comparison fails on a real
difference — the gates_m2 ``corruption``/``perturb`` pattern, never exposed
on the CLI.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from ftag import Flavours
from lightning import Trainer
from numpy.lib.recfunctions import unstructured_to_structured as u2s
from torch import nn

from salt.callbacks.confusion_matrix import ConfusionMatrixCallback as V1ConfusionMatrix
from salt.callbacks.predictionwriter import PredictionWriter
from salt.core.callbacks import ConfusionMatrix
from salt.core.data import Features, GraphDataModule, H5StructuredReader, Labels
from salt.core.graph import Bundle, Executor, Mode
from salt.core.graph.spec import TensorSpec
from salt.core.main import CONFIG_DIR
from salt.core.main import main as salt2_main
from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
from salt.core.saltmodule import SaltModule
from salt.core.writers import Writer
from salt.data.datamodules import SaltDataModule
from salt.data.datasets import SaltDataset
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules, compile_gn2v2
from salt.utils.inputs import write_dummy_file

__all__ = [
    "PROB_ATOL",
    "ValidCountWriter",
    "main",
    "run_w1",
    "run_w2",
    "run_w3",
    "run_w4",
    "run_w5",
]

PROB_ATOL = 1e-6
"""Justified fallback bound for the converted probability columns.

Both sides softmax the SAME checkpoint's logits, but the v2 production task
path materialises per-stream slices via ``Split`` where v1 boolean-indexes
the register-augmented sequence (design §3.3): mathematically equal, not
contractually bitwise, measured <= 1e-6 absolute on every pred (the A2/G3
forward-equivalence evidence, ``gates_m2.G3_ATOL0``). On the single-stream
GN2 fixture with valid-first files the two paths happen to be bitwise
identical, so this bound is normally pure headroom — any non-bitwise
probability column is REPORTED with this justification, never passed
silently (the ``parity_gn2`` JUSTIFIED_NONBITWISE pattern). Integer columns
(``VertexIndex``), input copies, labels and masks are NEVER tolerant: the
vertexing op chain and the file re-reads are exact ports.
"""

PROB_JUSTIFICATION = (
    "v2 Split per-stream slices vs v1 full-sequence boolean indexing ahead of the softmax "
    "(design §3.3): mathematically equal, measured <= 1e-6 — the gates_m2.G3_ATOL0 evidence"
)

JET_CLASSES = ("bjets", "cjets", "ujets")
"""The fixture jet classes (gn2_fixture / gn2v2-dummy.yaml head order)."""

RUN_NAME = "salt"
"""Both sides' run name: the v1 ``ModelWrapper`` default (modelwrapper.py:87).

The v2 side passes ``--name=salt`` to ``salt2 test`` so the v1 column-prefix
contract (``task.py:140-151``) is met by the production naming path.
"""

V1_TRACK_LABELS = ("ftagTruthOriginLabel", "ftagTruthVertexIndex")

_LRS = {"initial": 1e-7, "max": 1e-3, "end": 1e-5, "pct_start": 0.01}

_DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"

W4_OVERRIDE_YAML = """# add one custom column to the eval file (design §8 user journey)
writers:
  modules:
    valid_count: {class_path: salt.core.gates_m3.ValidCountWriter}
"""
"""The W4 four-line override config (comment included — the full user diff)."""


# ---------------------------------------------------------------------------
# shared report helpers (the gates_m2 envelope, kept standalone per harness)
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


# ---------------------------------------------------------------------------
# fixtures: dummy eval data + the weight-matched v1/v2 eval paths
# ---------------------------------------------------------------------------


def _dummy_eval_data(workdir: Path) -> SimpleNamespace:
    """Generate the dummy W-gate fixture data into ``workdir``.

    Parity norm/class dicts (`write_parity_norm_dict` — distinct
    per-variable constants) plus a 1000-jet salt dummy file named with
    exactly four underscore parts, so the v1 sample heuristic yields
    ``ttbar`` (``predictionwriter.py:169``).

    Returns
    -------
    SimpleNamespace
        ``file`` / ``norm_dict`` / ``class_dict`` paths.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    nd_path = workdir / "norm_dict.yaml"
    cd_path = workdir / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = workdir / "pp_output_test_ttbar.h5"
    write_dummy_file(h5_path, nd_path)
    return SimpleNamespace(file=h5_path, norm_dict=nd_path, class_dict=cd_path)


def _fixture_variables() -> dict[str, list[str]]:
    """The default (dummy) per-stream variable lists.

    Returns
    -------
    dict[str, list[str]]
        Fresh copies of the fixture jet/track variable lists.
    """
    return {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}


def _opendata_variables() -> dict[str, list[str]]:
    """The open-data per-stream variable lists from the shipped config.

    Read from ``gn2v2-opendata.yaml`` (single source of truth for the
    ``lifetimeSigned*`` renames) so the W2 gate cannot drift from the
    shipped config.

    Returns
    -------
    dict[str, list[str]]
        ``{stream: [variable, ...]}``.
    """
    with open(CONFIG_DIR / "gn2v2-opendata.yaml") as fh:
        cfg = yaml.safe_load(fh)
    variables = cfg["data"]["modules"]["features"]["init_args"]["variables"]
    return {stream: list(names) for stream, names in variables.items()}


def _quiet_trainer(**kwargs: Any) -> Trainer:
    """Build a silent single-device CPU trainer for the harness eval runs.

    Returns
    -------
    Trainer
        The configured trainer.
    """
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def _v1_write_eval(
    workdir: Path,
    file: Path | str,
    norm_dict: Path | str,
    variables: dict[str, list[str]],
    *,
    batch_size: int,
    num_test: int,
):
    """Run the v1 eval path end to end and return ``(wrapper, output_h5)``.

    Two steps (the recipe's public-API-only construction):

    1. an attach-only ``Trainer.test`` (one batch) so
       ``trainer.save_checkpoint`` can write a real Lightning checkpoint —
       `PredictionWriter` derives its output path from ``trainer.ckpt_path``
       (``predictionwriter.py:164-171``), so a checkpoint must exist;
    2. the REAL v1 eval: a fresh trainer with
       ``PredictionWriter(write_tracks=True)`` testing from that checkpoint.

    Returns
    -------
    tuple
        ``(ModelWrapper, Path)`` — the built v1 model (weights source for
        the v2 transfer) and the written eval file.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    fixture_dir = workdir / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_test_gn2(fixture_dir, variables=variables, norm_dict=norm_dict)
    dm = SaltDataModule(
        train_file=file,
        val_file=file,
        test_file=file,
        batch_size=batch_size,
        num_workers=0,
        num_train=num_test,
        num_val=num_test,
        num_test=num_test,
        norm_dict=str(norm_dict),
        variables=variables,
        global_object="jets",
        pin_memory=False,
    )
    attach = _quiet_trainer(limit_test_batches=1)
    attach.test(wrapper, datamodule=dm)
    ckpt_path = workdir / "v1.ckpt"
    attach.save_checkpoint(ckpt_path)

    writer = PredictionWriter(write_tracks=True)
    trainer = _quiet_trainer(callbacks=[writer])
    trainer.test(wrapper, datamodule=dm, ckpt_path=str(ckpt_path))
    return wrapper, Path(writer.output_path)


def _v2_checkpoint(
    workdir: Path,
    file: Path | str,
    norm_dict: Path | str,
    variables: dict[str, list[str]],
    wrapper,
    *,
    batch_size: int,
) -> Path:
    """Build a v2 checkpoint weight-matched to ``wrapper``.

    The recipe flow: a one-batch programmatic ``Trainer.test`` compiles the
    TEST plan and runs the two-phase bind (``saltmodule.py:473-502``; no
    writers attached, so the M2 anchor-on-all-preds sinks apply), then the
    mapped v1 state dict is strict-loaded (`map_v1_state_dict` — Normaliser
    buffers arrive via the state dict, no materialise on the test path) and
    the trainer saves the checkpoint ``salt2 test`` will evaluate.

    No schema artifact (explicit global_object flags instead, matching the
    schema-less ``salt2 test`` invocation): a real open-data file's
    ``flavour_label`` class attr (4 classes) would otherwise trip
    `check_class_names` against the 3-class fixture head — the W gates
    compare WRITER outputs, and the heads stay v1-width-matched.

    Returns
    -------
    Path
        The saved ``v2.ckpt`` path (its parent is the v2 eval-output dir).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    dm = GraphDataModule(
        modules={
            "reader": H5StructuredReader(
                groups={"jets": {"global_object": True}, "tracks": {"global_object": False}}
            ),
            "features": Features(variables=variables),
            "labels": Labels(),
        },
        test_file=file,
        batch_size=batch_size,
        num_test=batch_size,  # the attach run never reads a batch
        num_workers=0,
        pin_memory=False,
    )
    modules = build_gn2v2_modules(norm_dict)
    model = SaltModule(modules, lrs=dict(_LRS), name=RUN_NAME)
    # zero batches: the setup hooks (compile + bind) are all this run is for —
    # a FRESH Normaliser refuses forward before materialise (design §2.3), and
    # the buffers arrive via the strict state-dict load below
    trainer = _quiet_trainer(limit_test_batches=0)
    trainer.test(model, datamodule=dm)
    mapped = map_v1_state_dict(wrapper.state_dict(), modules)
    model.load_state_dict({f"net.{key}": value for key, value in mapped.items()}, strict=True)
    ckpt_path = workdir / "v2.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


def _salt2_test_argv(
    file: Path | str,
    norm_dict: Path | str,
    ckpt_path: Path,
    run_dir: Path,
    *,
    batch_size: int,
    num_test: int,
    variables: dict[str, list[str]] | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Assemble the ``salt2 test`` argv for the v2 eval (the REAL CLI surface).

    Returns
    -------
    list[str]
        argv for `salt.core.main.main`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "test",
        "--config",
        str(_DUMMY_CFG),
        f"--data.test_file={file}",
        f"--data.num_test={num_test}",
        f"--data.batch_size={batch_size}",
        "--data.num_workers=0",
        f"--model.modules.norm.init_args.norm_dict={norm_dict}",
        f"--ckpt_path={ckpt_path}",
        f"--name={RUN_NAME}",  # the v1 column-prefix contract (see RUN_NAME)
        "--trainer.accelerator=cpu",
        "--trainer.devices=1",
        # null-delete the base2 ProgressBar (D2 default-on); the stock
        # enable_progress_bar=false cannot coexist with a configured bar
        "--callbacks.progress=null",
        # GraphArtifacts (default-on, design §4.4) writes into the trainer
        # log dir — point it at the gate outdir, not the invocation cwd
        f"--trainer.default_root_dir={run_dir}",
    ]
    if variables is not None:
        argv.append(f"--data.modules.features.init_args.variables={json.dumps(variables)}")
    argv.extend(extra)
    return argv


def _sample_name(file: Path | str) -> str:
    """The v1 test-sample stem heuristic (``predictionwriter.py:169``).

    Returns
    -------
    str
        ``stem.split('_')[3]`` iff the stem has exactly four underscore
        parts, else the whole stem.
    """
    stem = Path(file).stem
    split = stem.split("_")
    return split[3] if len(split) == 4 else stem


# ---------------------------------------------------------------------------
# H5 comparison (the W1/W2 core)
# ---------------------------------------------------------------------------


def _tolerant_columns(run_name: str = RUN_NAME) -> dict[str, set[str]]:
    """The converted probability columns eligible for the `PROB_ATOL` fallback.

    Derived from the fixture task heads exactly as the writers derive the
    names: jet classes through the ftag ``Flavours`` px (``salt_pb`` ...)
    and the 8 origin classes as ``salt_p<class>`` (``task.py:140-151``).

    Returns
    -------
    dict[str, set[str]]
        ``{group: {column, ...}}``.
    """
    return {
        "jets": {f"{run_name}_{Flavours[c].px}" for c in JET_CLASSES},
        "tracks": {f"{run_name}_p{c}" for c in ORIGIN_CLASSES},
    }


def _compare_column(
    group: str, column: str, ref: np.ndarray, got: np.ndarray, *, tolerant: bool, atol: float
) -> dict[str, Any]:
    """Compare one column of one output group (bitwise-first).

    Returns
    -------
    dict[str, Any]
        Comparison record; ``passed`` is True for a bitwise match, or — for
        `tolerant` floating columns only — a justified ``max|diff| <= atol``.
    """
    record: dict[str, Any] = {
        "group": group,
        "column": column,
        "dtype": str(ref.dtype),
        "tolerant": tolerant,
    }
    bitwise = ref.tobytes() == got.tobytes()
    max_abs: float | None = 0.0
    if not bitwise and np.issubdtype(ref.dtype, np.floating):
        a64, b64 = ref.astype("f8"), got.astype("f8")
        finite_match = np.array_equal(np.isfinite(a64), np.isfinite(b64))
        diff = np.abs(a64 - b64)[np.isfinite(a64) & np.isfinite(b64)]
        max_abs = float(diff.max()) if diff.size else 0.0
        if not finite_match:
            max_abs = float("inf")
    elif not bitwise:
        try:
            max_abs = float(np.abs(ref.astype("f8") - got.astype("f8")).max())
        except (TypeError, ValueError):
            max_abs = None
    justified = bool(
        not bitwise
        and tolerant
        and max_abs is not None
        and np.isfinite(max_abs)
        and max_abs <= atol
    )
    record.update(
        bitwise=bitwise,
        max_abs_diff=max_abs,
        justified=justified,
        passed=bitwise or justified,
        note=f"JUSTIFIED <= {atol:g}: {PROB_JUSTIFICATION}" if justified else "",
    )
    return record


def _compare_outputs(
    v1_path: Path,
    v2_path: Path,
    *,
    atol: float = PROB_ATOL,
    run_name: str = RUN_NAME,
    corruption: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Compare the two eval files: byte-schema AND values, column by column.

    Compared per shared group: ``dtype.descr`` equality (names + formats +
    ORDER — column order is part of the output byte-schema,
    ``array_utils.py:7-37``) and shape equality, then every column —
    bitwise via ``tobytes()``, with the `PROB_ATOL` justified fallback for
    the converted probability columns only. ``maxshape``/chunking are NEVER
    compared (v1 dynamic vs v2 fixed H5Writer modes differ legitimately);
    the ``writer_version`` root attr is report-only.

    Parameters
    ----------
    v1_path, v2_path : Path
        The two eval H5 files.
    atol : float, optional
        Probability-column fallback bound, by default `PROB_ATOL`.
    run_name : str, optional
        Column-prefix run name, by default `RUN_NAME`.
    corruption : Callable | None, optional
        TEST-ONLY hook applied to the loaded v2 ``{group: array}`` dict
        before comparison (the W3 negative control), by default None.

    Returns
    -------
    dict[str, Any]
        ``passed`` + per-group schema records + per-column records.
    """
    with h5py.File(v1_path) as f1, h5py.File(v2_path) as f2:
        v1_arrays = {name: f1[name][:] for name in f1}
        v2_arrays = {name: f2[name][:] for name in f2}
        attrs = {
            "v1_writer_version": str(f1.attrs.get("writer_version")),
            "v2_writer_version": str(f2.attrs.get("writer_version")),
        }
    if corruption is not None:
        v2_arrays = corruption(v2_arrays)

    tolerant = _tolerant_columns(run_name)
    datasets_match = set(v1_arrays) == set(v2_arrays)
    groups: dict[str, Any] = {}
    columns: list[dict[str, Any]] = []
    for group in sorted(set(v1_arrays) & set(v2_arrays)):
        ref, got = v1_arrays[group], v2_arrays[group]
        schema_match = ref.dtype.descr == got.dtype.descr and ref.shape == got.shape
        groups[group] = {
            "schema_match": schema_match,
            "v1_shape": list(ref.shape),
            "v2_shape": list(got.shape),
            "v1_columns": list(ref.dtype.names or ()),
            "v2_columns": list(got.dtype.names or ()),
        }
        ref_fields = dict(ref.dtype.fields or {})
        got_fields = dict(got.dtype.fields or {})
        for column in ref.dtype.names or ():
            if column not in got_fields or ref_fields[column][0] != got_fields[column][0]:
                columns.append({
                    "group": group,
                    "column": column,
                    "dtype": str(ref_fields[column][0]),
                    "tolerant": False,
                    "bitwise": False,
                    "max_abs_diff": None,
                    "justified": False,
                    "passed": False,
                    "note": "missing or dtype-mismatched in the v2 output",
                })
                continue
            columns.append(
                _compare_column(
                    group,
                    column,
                    ref[column],
                    got[column],
                    tolerant=column in tolerant.get(group, set()),
                    atol=atol,
                )
            )
        columns.extend(
            {
                "group": group,
                "column": column,
                "dtype": str(got_fields[column][0]),
                "tolerant": False,
                "bitwise": False,
                "max_abs_diff": None,
                "justified": False,
                "passed": False,
                "note": "extra column in the v2 output",
            }
            for column in got.dtype.names or ()
            if column not in ref_fields
        )

    passed = (
        datasets_match
        and all(record["schema_match"] for record in groups.values())
        and all(record["passed"] for record in columns)
    )
    return {
        "passed": passed,
        "datasets_match": datasets_match,
        "v1_datasets": sorted(v1_arrays),
        "v2_datasets": sorted(v2_arrays),
        "attrs": attrs,
        "groups": groups,
        "columns": columns,
        "n_columns": len(columns),
        "n_bitwise": sum(record["bitwise"] for record in columns),
        "n_justified": sum(record["justified"] for record in columns),
        "n_failed": sum(not record["passed"] for record in columns),
        "corrupted_by_test_hook": corruption is not None,
    }


def _print_comparison(comparison: dict[str, Any]) -> None:
    """Print the column-by-column comparison table + justification block."""
    print(
        f"datasets: v1={comparison['v1_datasets']} v2={comparison['v2_datasets']} "
        f"(match: {comparison['datasets_match']})"
    )
    for group, record in comparison["groups"].items():
        shape = (
            f"{tuple(record['v1_shape'])}"
            if record["v1_shape"] == record["v2_shape"]
            else f"v1{tuple(record['v1_shape'])} != v2{tuple(record['v2_shape'])}"
        )
        status = "PASS" if record["schema_match"] else "FAIL"
        print(f"group {group!r}: shape {shape}, dtype.descr+order match: {status}")
    print(f"{'group':<9}{'column':<44}{'dtype':<10}{'max|diff|':>12}{'status':>10}")
    print("-" * 96)
    for record in comparison["columns"]:
        diff = "n/a" if record["max_abs_diff"] is None else f"{record['max_abs_diff']:.3e}"
        if record["bitwise"]:
            status = "BITWISE"
        elif record["justified"]:
            status = "JUSTIFIED"
        else:
            status = "FAIL"
        print(
            f"{record['group']:<9}{record['column']:<44}{record['dtype']:<10}{diff:>12}{status:>10}"
        )
    justified = [record for record in comparison["columns"] if record["justified"]]
    if justified:
        print(f"justified non-bitwise columns ({len(justified)}; never silent):")
        for record in justified:
            print(f"  {record['group']}.{record['column']}: {record['note']}")
    print(
        f"columns: {comparison['n_columns']} compared, {comparison['n_bitwise']} bitwise, "
        f"{comparison['n_justified']} justified, {comparison['n_failed']} failed; "
        f"writer_version: v1={comparison['attrs']['v1_writer_version']} "
        f"v2={comparison['attrs']['v2_writer_version']} (report-only)"
    )


def _writer_parity(
    workdir: Path,
    file: Path | str,
    norm_dict: Path | str,
    variables: dict[str, list[str]],
    *,
    batch_size: int,
    num_test: int,
    corruption: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Run both eval paths on one file and compare the outputs.

    Returns
    -------
    dict[str, Any]
        ``checks`` + the `_compare_outputs` comparison + the artifact paths
        (including ``v2_ckpt`` for follow-up gates).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    wrapper, v1_out = _v1_write_eval(
        workdir / "v1", file, norm_dict, variables, batch_size=batch_size, num_test=num_test
    )
    v2_ckpt = _v2_checkpoint(
        workdir / "v2", file, norm_dict, variables, wrapper, batch_size=batch_size
    )
    is_dummy_vars = variables == _fixture_variables()
    argv = _salt2_test_argv(
        file,
        norm_dict,
        v2_ckpt,
        workdir / "v2_run",
        batch_size=batch_size,
        num_test=num_test,
        variables=None if is_dummy_vars else variables,
    )
    rc = salt2_main(argv)
    v2_out = v2_ckpt.parent / f"{v2_ckpt.stem}__test_{_sample_name(file)}.h5"

    checks = {
        "v2_cli_returned_zero": rc == 0,
        "v1_output_exists": v1_out.exists(),
        "v2_output_exists": v2_out.exists(),
        # the v1 test_pipeline bar: exactly ONE eval h5 next to each checkpoint
        "one_h5_per_ckpt_dir": len(list(v1_out.parent.glob("*.h5"))) == 1
        and len(list(v2_ckpt.parent.glob("*.h5"))) == 1,
    }
    comparison: dict[str, Any] = {"passed": False}
    rows = {}
    if checks["v1_output_exists"] and checks["v2_output_exists"]:
        comparison = _compare_outputs(v1_out, v2_out, corruption=corruption)
        with h5py.File(v1_out) as f1, h5py.File(v2_out) as f2:
            rows = {"v1": int(f1["jets"].shape[0]), "v2": int(f2["jets"].shape[0])}
        checks["row_counts_match_num_test"] = rows["v1"] == rows["v2"] == num_test
    return {
        "checks": checks,
        "comparison": comparison,
        "rows": rows,
        "v1_output": str(v1_out),
        "v2_output": str(v2_out),
        "v2_ckpt": str(v2_ckpt),
        "salt2_argv": argv,
    }


def _finish_parity_gate(
    gate: str,
    details: dict[str, Any],
    outdir: Path,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Assemble, print and emit a W1/W2-style parity report.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    comparison = details["comparison"]
    passed = all(details["checks"].values()) and comparison["passed"]
    criterion = (
        "identical H5 byte-schema (dataset names; column names+formats+ORDER; shapes) and "
        "values: bitwise for input copies/labels/masks/VertexIndex; converted probability "
        f"columns bitwise or justified <= {PROB_ATOL:g} (printed, never silent); exactly one "
        "eval h5 per checkpoint dir"
    )
    report = _base_report(f"{gate}_writer_parity", passed, criterion, config)
    report["checks"] = details["checks"]
    report["rows"] = details["rows"]
    report["artifacts"] = {
        "v1_output": details["v1_output"],
        "v2_output": details["v2_output"],
        "v2_ckpt": details["v2_ckpt"],
    }
    report["salt2_argv"] = details["salt2_argv"]
    report["prob_atol"] = PROB_ATOL
    report["prob_justification"] = PROB_JUSTIFICATION
    report["vertex_column_decision"] = (
        "vertexing column is BARE 'VertexIndex' (i8, no run-name prefix) by default for v1 "
        "byte parity; the design §8 run-name prefix is opt-in via TaskWriter "
        "prefix_vertex_column=true (M3 decision, salt/core/writers/modules.py)"
    )
    report["comparison"] = comparison

    print(f"v1 output: {details['v1_output']}")
    print(f"v2 output: {details['v2_output']}")
    if details["rows"]:
        print(f"rows: v1={details['rows']['v1']:,} v2={details['rows']['v2']:,}")
    if comparison.get("columns"):
        _print_comparison(comparison)
    for name, ok in details["checks"].items():
        print(f"  {name:<32} {'PASS' if ok else 'FAIL'}")
    print(f"  note: {report['vertex_column_decision']}")
    _print_verdict(gate, passed, criterion, _emit_report(report, outdir, gate))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# W1 — writer parity on the dummy fixture
# ---------------------------------------------------------------------------


def run_w1(
    outdir: Path | str,
    *,
    file: Path | str | None = None,
    norm_dict: Path | str | None = None,
    batch_size: int = 96,
    num_test: int = 1000,
    corruption: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """W1: weight-matched v1 vs v2 ``salt2 test`` eval files — identical H5.

    Construction: the small v1 GN2 (`build_test_gn2`, 16-dim fixture widths,
    torch-math attention) evaluated through the REAL v1 path
    (``Trainer.test`` + ``PredictionWriter(write_tracks=True)`` from a saved
    checkpoint); the equivalent config-built v2 (shipped
    ``gn2v2-dummy.yaml``) with the v1 weights strict-transferred
    (`map_v1_state_dict`) into a v2 checkpoint, evaluated through the REAL
    ``salt2 test`` surface (base2 writers ``inputs_copy -> tasks ->
    pad_mask`` — the v1 column order). Same test file, same ``batch_size`` /
    ``num_test``, sequential no-shuffle loaders, single CPU device, fp32.

    PASS requires every `_compare_outputs` criterion: identical dataset
    names, identical per-group ``dtype.descr`` (names + formats + ORDER) and
    shapes, bitwise values for input copies / labels / pad mask /
    ``VertexIndex``, probability columns bitwise or justified within
    `PROB_ATOL` (printed), plus exactly one eval h5 per checkpoint dir and
    row counts equal to ``num_test``.

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    file : Path | str | None, optional
        Test H5 carrying the fixture variable set; by default None — a salt
        dummy file is generated into ``outdir/data``.
    norm_dict : Path | str | None, optional
        Norm dict for both sides, by default None (generated parity dict;
        REQUIRED when `file` is given).
    batch_size : int, optional
        Eval batch size both sides, by default 96 — DELIBERATELY a
        non-divisor of the default ``num_test=1000`` so the shipped gate
        exercises the partial final batch (40 rows) on both sides'
        last-batch handling (v1's ``bhigh=min(...)`` clamp vs the v2
        ``meta.rows`` alignment; M3-review fix — every previous default
        divided evenly, leaving that path gate-untested).
    num_test : int, optional
        Rows evaluated both sides, by default 1000 (the dummy-file size).
    corruption : Callable | None, optional
        TEST-ONLY hook applied to the loaded v2 arrays before comparison
        (negative control — the gate must FAIL), never exposed on the CLI,
        by default None.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check and column passed.

    Raises
    ------
    ValueError
        If `file` is given without `norm_dict` (or vice versa).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if (file is None) != (norm_dict is None):
        raise ValueError("pass --file and --norm-dict together (or neither, for dummy data)")
    if file is None:
        data = _dummy_eval_data(outdir / "data")
        file, norm_dict = data.file, data.norm_dict

    print("=" * 96)
    print("W1 writer parity (dummy) — v1 PredictionWriter vs v2 salt2-test writers, same file")
    print("=" * 96)
    assert norm_dict is not None
    details = _writer_parity(
        outdir / "parity",
        file,
        norm_dict,
        _fixture_variables(),
        batch_size=batch_size,
        num_test=num_test,
        corruption=corruption,
    )
    config = {
        "file": str(file),
        "norm_dict": str(norm_dict),
        "batch_size": batch_size,
        "num_test": num_test,
        "variables": _fixture_variables(),
        "corrupted_by_test_hook": corruption is not None,
    }
    return _finish_parity_gate("w1", details, outdir, config)


# ---------------------------------------------------------------------------
# W2 — writer parity on a real open-data file
# ---------------------------------------------------------------------------


def run_w2(
    file: Path | str,
    norm_dict: Path | str,
    outdir: Path | str,
    *,
    class_dict: Path | str | None = None,
    batch_size: int = 1000,
    num_test: int = 5000,
    jet_vars: Sequence[str] | None = None,
    track_vars: Sequence[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """W2: the W1 comparison at scale, on a real open-data validation file.

    Identical machinery to `run_w1` with the per-stream variables defaulting
    to the shipped ``gn2v2-opendata.yaml`` Features lists (the
    ``lifetimeSigned*`` renames; same list lengths as the dummy fixture, so
    the 16-dim model is width-matched and `map_v1_state_dict` applies
    unchanged) and the v1 fixture's `InputNorm` built from the REAL norm
    dict. NO truncation on either side (the v2 invocation uses the
    no-truncate dummy config with overridden variables; v1 sets no
    ``num_inputs``), so the v1 ``maybe_pad`` zero-fill quirk stays vacuous —
    the recipe's W2 contract. Input copies cover ALL source dtype fields of
    both groups, exercising the real file's full label/truth column set
    bitwise.

    Parameters
    ----------
    file : Path | str
        Real test/validation H5 file (CLI arg — never hardcoded).
    norm_dict : Path | str
        The matching norm dict (covers the open-data variable names).
    outdir : Path | str
        Report + artifact output directory.
    class_dict : Path | str | None, optional
        Accepted for CLI symmetry and recorded in the report; UNUSED by the
        gate — TEST plans carry no losses, so class weights never enter the
        eval path (both sides are built unweighted), by default None.
    batch_size : int, optional
        Eval batch size both sides, by default 1000.
    num_test : int, optional
        Rows evaluated both sides, by default 5000 (CPU-budget cap; raise
        from the CLI for a fuller pass).
    jet_vars, track_vars : Sequence[str] | None, optional
        Explicit variable overrides, by default the opendata-config lists.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    variables = _opendata_variables()
    if jet_vars is not None:
        variables["jets"] = list(jet_vars)
    if track_vars is not None:
        variables["tracks"] = list(track_vars)

    print("=" * 96)
    print("W2 writer parity (real file) — v1 PredictionWriter vs v2 salt2-test writers")
    print("=" * 96)
    with h5py.File(file) as fh:
        seq_len = int(fh["tracks"].shape[1])
    print(f"file: {file} (tracks length {seq_len}; no truncation configured on either side)")
    details = _writer_parity(
        outdir / "parity",
        file,
        norm_dict,
        variables,
        batch_size=batch_size,
        num_test=num_test,
    )
    config = {
        "file": str(file),
        "norm_dict": str(norm_dict),
        "class_dict": str(class_dict) if class_dict is not None else None,
        "class_dict_note": "recorded only — TEST plans carry no losses, weights unused",
        "batch_size": batch_size,
        "num_test": num_test,
        "variables": variables,
        "file_track_length": seq_len,
        "truncation": "none on either side (recipe W2 contract)",
    }
    return _finish_parity_gate("w2", details, outdir, config)


# ---------------------------------------------------------------------------
# W3 — negative controls
# ---------------------------------------------------------------------------


def _swap_jet_probs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The W3 corruption hook: value-swap two v2 jet probability columns.

    Swapping VALUES (not names) keeps the byte-schema intact, so a pass of
    this corrupted comparison would mean the VALUE checks are vacuous — the
    sharper negative control.

    Returns
    -------
    dict[str, np.ndarray]
        The corrupted ``{group: array}`` dict.
    """
    jets = arrays["jets"].copy()
    pb, pc = jets[f"{RUN_NAME}_pb"].copy(), jets[f"{RUN_NAME}_pc"].copy()
    jets[f"{RUN_NAME}_pb"], jets[f"{RUN_NAME}_pc"] = pc, pb
    return {**arrays, "jets": jets}


def run_w3(
    outdir: Path | str,
    *,
    batch_size: int = 100,
    num_test: int = 300,
) -> tuple[int, dict[str, Any]]:
    """W3: both negative controls must FAIL the machinery they target.

    1. **Corrupted comparison fails W1** — `run_w1` re-runs on dummy data
       with the python-only ``corruption`` hook value-swapping the v2
       ``salt_pb``/``salt_pc`` columns: the W1 verdict must be FAIL, the v2
       CLI itself must have SUCCEEDED (the failure is the comparison, not a
       crash), and the failing columns must be exactly the two swapped ones
       (everything else still passes — locality of the detection).
    2. **TEST dead-preds hard error** — ``salt2 test`` with the TaskWriter
       narrowed to ``streams=["jets"]`` leaves ``track_origin`` and
       ``track_vertexing`` predictions consumed by no writer: the CLI must
       exit non-zero with the design §4.2/§8 error naming both dead keys,
       and no new eval file may appear. The sub-run's ``writers.output`` is
       pointed at a FRESH path and that file is asserted absent (M3-review
       fix: at the default path the corrupted W1 sub-run's eval file
       already existed, so the original count-based check could not have
       detected an overwrite); the checkpoint-dir count check is kept as a
       second guard.

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    batch_size : int, optional
        Eval batch size for the sub-runs, by default 100.
    num_test : int, optional
        Rows evaluated in the sub-runs, by default 300 (cheap — the
        controls only need the machinery to engage).

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if BOTH controls failed as
        designed.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = _dummy_eval_data(outdir / "data")

    print("=" * 96)
    print("W3 negative controls — the gate machinery must FAIL when it should")
    print("=" * 96)
    print("control (1): W1 with a value-swap corruption hook — the W1 verdict below MUST be FAIL")
    w1_code, w1_report = run_w1(
        outdir / "corrupted_w1",
        file=data.file,
        norm_dict=data.norm_dict,
        batch_size=batch_size,
        num_test=num_test,
        corruption=_swap_jet_probs,
    )
    failed_columns = sorted(
        f"{record['group']}.{record['column']}"
        for record in w1_report["comparison"].get("columns", [])
        if not record["passed"]
    )
    expected_failures = sorted(f"jets.{RUN_NAME}_pb jets.{RUN_NAME}_pc".split())
    control_corruption = {
        "w1_exit_code": w1_code,
        "w1_failed": w1_code != 0,
        "v2_cli_succeeded": bool(w1_report["checks"].get("v2_cli_returned_zero")),
        "failed_columns": failed_columns,
        "failed_columns_are_exactly_the_swapped_pair": failed_columns == expected_failures,
    }
    corruption_ok = (
        control_corruption["w1_failed"]
        and control_corruption["v2_cli_succeeded"]
        and control_corruption["failed_columns_are_exactly_the_swapped_pair"]
    )

    # control (2): dead-preds — reuse the corrupted run's v2 checkpoint, but
    # point the output at a FRESH path so file non-creation is decidable
    # (docstring; the default path already carried the W1 sub-run's file)
    v2_ckpt = Path(w1_report["artifacts"]["v2_ckpt"])
    h5_before = len(list(v2_ckpt.parent.glob("*.h5")))
    dead_out = outdir / "dead_preds_run" / "dead_preds_eval.h5"
    argv = _salt2_test_argv(
        data.file,
        data.norm_dict,
        v2_ckpt,
        outdir / "dead_preds_run",
        batch_size=batch_size,
        num_test=num_test,
        extra=[
            '--writers.modules.tasks.init_args.tasks=["jets_classification"]',
            f"--writers.output={dead_out}",
        ],
    )
    print(
        "control (2): salt2 test with TaskWriter narrowed to tasks=['jets_classification'] "
        "MUST hard-error"
    )
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        rc = salt2_main(argv)
    err = stderr.getvalue()
    print(err.strip() or "<no stderr>")
    control_dead_preds = {
        "exit_code": rc,
        "errored": rc != 0,
        "message_names_dead_preds": "consumed by NO writer" in err,
        "message_names_both_tasks": "track_origin" in err and "track_vertexing" in err,
        "no_new_eval_file": not dead_out.exists()
        and len(list(v2_ckpt.parent.glob("*.h5"))) == h5_before,
    }
    dead_preds_ok = all(control_dead_preds[key] for key in control_dead_preds if key != "exit_code")

    passed = corruption_ok and dead_preds_ok
    criterion = (
        "the corrupted-comparison control FAILS W1 on exactly the swapped columns (v2 CLI "
        "itself succeeding), AND the dead-preds control hard-errors naming both unconsumed "
        "prediction keys without writing an eval file"
    )
    report = _base_report(
        "w3_negative_controls",
        passed,
        criterion,
        {
            "file": str(data.file),
            "norm_dict": str(data.norm_dict),
            "batch_size": batch_size,
            "num_test": num_test,
            "dead_preds_argv": argv,
        },
    )
    report["control_corrupted_comparison"] = control_corruption
    report["control_dead_preds"] = control_dead_preds
    report["dead_preds_stderr"] = err

    print("-" * 96)
    for name, record in (
        ("corrupted comparison", control_corruption),
        ("dead-preds error", control_dead_preds),
    ):
        print(f"control: {name}")
        for key, value in record.items():
            mark = ""
            if isinstance(value, bool):
                mark = "PASS" if value else "FAIL"
            print(f"  {key:<44} {value!r:<28} {mark}")
    _print_verdict("w3", passed, criterion, _emit_report(report, outdir, "w3"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# W4 — custom-writer ergonomics journey
# ---------------------------------------------------------------------------


class ValidCountWriter(Writer):
    """The design §8 custom-writer journey (gate W4): one new jet column.

    The user story end to end: subclass `Writer`, declare the consumed
    bundle key, return one structured fragment per batch, and register the
    class with four YAML lines under ``writers.modules`` (`W4_OVERRIDE_YAML`)
    — the new ``n_valid_tracks`` column (valid = non-padded track count per
    jet, from the TEST pad mask) lands in the eval file next to the shipped
    columns.
    """

    COLUMN = "n_valid_tracks"

    def requires(self, ctx) -> dict[str, TensorSpec]:
        """Declare the consumed pad mask.

        Returns
        -------
        dict[str, TensorSpec]
            The ``masks.tracks`` spec.
        """
        del ctx
        return {"masks.tracks": TensorSpec(shape=None, dtype="bool", kind="pad_mask")}

    def columns(self, ctx) -> dict[str, np.dtype]:
        """Declare the single new jet column.

        Returns
        -------
        dict[str, np.dtype]
            ``{"jets": [("n_valid_tracks", "i4")]}``.
        """
        del ctx
        return {"jets": np.dtype([(self.COLUMN, "i4")])}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Count valid tracks per jet from the batch pad mask.

        Returns
        -------
        dict[str, np.ndarray]
            The per-jet count fragment.
        """
        del rows
        mask = bundle.get("masks.tracks").cpu().numpy()
        counts = (~mask).sum(-1, keepdims=True).astype("i4")
        return {"jets": u2s(counts, np.dtype([(self.COLUMN, "i4")]))}


def run_w4(
    outdir: Path | str,
    *,
    batch_size: int = 100,
    num_test: int = 300,
) -> tuple[int, dict[str, Any]]:
    """W4: custom writer + four-line override -> a new column, end to end.

    The journey a user takes for "I want a new column in my eval file"
    (design §8): `ValidCountWriter` (defined in this module — any importable
    class path works) plus the `W4_OVERRIDE_YAML` override file stacked as a
    second ``--config`` onto the shipped dummy config. PASS requires:

    - ``salt2 test`` exits 0 with the override stacked;
    - the eval file's ``jets`` group carries ``n_valid_tracks`` AND values
      equal the source file's per-jet valid-track counts (bitwise i4);
    - the shipped v1-layout columns survive the deep merge (input copies +
      probability columns + ``VertexIndex`` + ``mask`` all still present);
    - the override file is at most four lines (the ergonomics bar — printed
      verbatim in the report).

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    batch_size : int, optional
        Eval batch size, by default 100.
    num_test : int, optional
        Rows evaluated, by default 300.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = _dummy_eval_data(outdir / "data")
    variables = _fixture_variables()

    print("=" * 96)
    print("W4 ergonomics journey — custom Writer + 4-line override -> new column via salt2 test")
    print("=" * 96)
    fixture_dir = outdir / "v1_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_test_gn2(fixture_dir, norm_dict=data.norm_dict)
    v2_ckpt = _v2_checkpoint(
        outdir / "v2", data.file, data.norm_dict, variables, wrapper, batch_size=batch_size
    )
    override_path = outdir / "add_writer.yaml"
    override_path.write_text(W4_OVERRIDE_YAML)
    override_lines = len(W4_OVERRIDE_YAML.strip().splitlines())
    print(f"override config ({override_path}, {override_lines} lines):")
    print(W4_OVERRIDE_YAML.strip())

    argv = _salt2_test_argv(
        data.file,
        data.norm_dict,
        v2_ckpt,
        outdir / "run",
        batch_size=batch_size,
        num_test=num_test,
        extra=["--config", str(override_path)],
    )
    rc = salt2_main(argv)
    out_path = v2_ckpt.parent / f"{v2_ckpt.stem}__test_{_sample_name(data.file)}.h5"

    checks: dict[str, bool] = {
        "salt2_test_returned_zero": rc == 0,
        "output_exists": out_path.exists(),
        "override_is_at_most_4_lines": override_lines <= 4,
    }
    values_note = ""
    if out_path.exists():
        with h5py.File(out_path) as fh:
            jets = fh["jets"][:]
            track_columns = list(fh["tracks"].dtype.names)
        with h5py.File(data.file) as fh:
            expected = fh["tracks"]["valid"][:num_test].sum(-1).astype("i4")
            source_jet_columns = list(fh["jets"].dtype.names)
        jet_columns = list(jets.dtype.names)
        prob_columns = [f"{RUN_NAME}_{Flavours[c].px}" for c in JET_CLASSES]
        checks["new_column_present"] = ValidCountWriter.COLUMN in jet_columns
        checks["new_column_values_match_source"] = bool(
            checks["new_column_present"]
            and jets[ValidCountWriter.COLUMN].tobytes() == expected.tobytes()
        )
        checks["shipped_columns_survive_merge"] = (
            all(column in jet_columns for column in (*source_jet_columns, *prob_columns))
            and "VertexIndex" in track_columns
            and "mask" in track_columns
        )
        values_note = (
            f"jets columns: {jet_columns[-4:]} (tail); "
            f"n_valid_tracks[:5]={jets[ValidCountWriter.COLUMN][:5].tolist()}"
        )

    passed = all(checks.values())
    criterion = (
        "salt2 test with the <=4-line override exits 0; the new column is present with "
        "values bitwise-equal to the source valid-track counts; the shipped v1-layout "
        "columns survive the deep merge"
    )
    report = _base_report(
        "w4_ergonomics_journey",
        passed,
        criterion,
        {
            "file": str(data.file),
            "norm_dict": str(data.norm_dict),
            "batch_size": batch_size,
            "num_test": num_test,
            "override_path": str(override_path),
            "argv": argv,
        },
    )
    report["checks"] = checks
    report["override_yaml"] = W4_OVERRIDE_YAML
    report["override_lines"] = override_lines
    report["output"] = str(out_path)
    report["values_note"] = values_note

    if values_note:
        print(values_note)
    for name, ok in checks.items():
        print(f"  {name:<36} {'PASS' if ok else 'FAIL'}")
    _print_verdict("w4", passed, criterion, _emit_report(report, outdir, "w4"))
    return (0 if passed else 1), report


# ---------------------------------------------------------------------------
# W5 — metrics parity (ConfusionMatrix v1 vs v2)
# ---------------------------------------------------------------------------

_W5_TASKS = ("jets_classification", "track_origin")


def _decollapse_heads(wrapper, ds, batch_size: int) -> None:
    """Subtract each classification head's mean logit offset (probe batch).

    An UNTRAINED fixture model argmaxes (nearly) every input to one class —
    the per-class logit offsets from the random final layer dominate the
    per-sample variation, so the W5 prediction-side comparison had no
    discriminating power (M3-review finding: the jets matrix was a single
    non-zero column; a v2 bug that also collapses argmax would have
    passed). Centering each gated head's final-layer bias on its mean
    logits over one probe batch makes argmax follow the per-sample
    fluctuations, spreading predictions over the classes. Applied to the v1
    wrapper BEFORE the `map_v1_state_dict` transfer, so both sides stay
    weight-identical and the parity bar is unchanged; `run_w5` asserts >= 2
    distinct predicted classes per task afterwards (``pred_diversity_ok``).
    """
    inputs, masks, _labels = ds[np.s_[0:batch_size]]
    with torch.no_grad():
        preds, _ = wrapper(
            {key: tensor.clone() for key, tensor in inputs.items()},
            {key: tensor.clone() for key, tensor in masks.items()},
            None,
        )
        for task in wrapper.model.tasks:
            if getattr(task, "name", None) not in _W5_TASKS:
                continue
            logits = preds[task.input_name][task.name]
            offset = logits.reshape(-1, logits.shape[-1]).mean(0)
            final = [m for m in task.modules() if isinstance(m, nn.Linear)][-1]
            final.bias.sub_(offset)


def run_w5(
    outdir: Path | str,
    *,
    file: Path | str | None = None,
    norm_dict: Path | str | None = None,
    n_batches: int = 3,
    batch_size: int = 100,
) -> tuple[int, dict[str, Any]]:
    """W5: v2 `ConfusionMatrix` values == v1 callback values, weight-matched eval.

    Construction: the v1 fixture wrapper evaluates each batch through its
    REAL forward (raw VAL logits) and feeds the v1
    ``ConfusionMatrixCallback`` its native ``{"outputs": {preds, labels}}``
    contract; the weight-transferred v2 module dict executes the compiled
    VAL plan (raw logits per design §3.3 — argmax is conversion-invariant)
    on the SAME batches and feeds the v2 callback through its Lightning hook
    surface (``{"bundle": ...}``). Batches come from one `SaltDataset`
    (dataset parity is G1-established), cloned per side.

    PASS requires, for ``jets_classification`` AND ``track_origin``
    (per-jet and per-track shapes, the -1 padding reduction): element-wise
    ``torch.equal`` accumulated truth and prediction lists, and identical
    integer counts matrices + ignored counts under the same transparent
    reduction (`ConfusionMatrix.confusion_counts` applied to BOTH sides) —
    exact-count criterion, no tolerance anywhere — plus >= 2 distinct
    predicted classes per task (`_decollapse_heads`: an untrained fixture
    otherwise argmax-collapses, leaving the pred-side comparison without
    discriminating power; M3-review fix).

    Parameters
    ----------
    outdir : Path | str
        Report output directory.
    file : Path | str | None, optional
        H5 carrying the fixture variable set + GN2 labels, by default None
        (dummy data generated into ``outdir/data``).
    norm_dict : Path | str | None, optional
        Norm dict (REQUIRED with `file`), by default None.
    n_batches : int, optional
        Eval batches accumulated per callback, by default 3.
    batch_size : int, optional
        Rows per batch, by default 100.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)``.

    Raises
    ------
    ValueError
        If `file` is given without `norm_dict` (or vice versa).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if (file is None) != (norm_dict is None):
        raise ValueError("pass --file and --norm-dict together (or neither, for dummy data)")
    if file is None:
        data = _dummy_eval_data(outdir / "data")
        file, norm_dict = data.file, data.norm_dict
    assert norm_dict is not None
    variables = _fixture_variables()

    print("=" * 96)
    print("W5 metrics parity — v1 ConfusionMatrixCallback vs v2 ConfusionMatrix, same eval")
    print("=" * 96)

    # v1 side: fixture wrapper (eval mode) + the real v1 callback
    fixture_dir = outdir / "v1_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_test_gn2(fixture_dir, norm_dict=norm_dict)

    ds = SaltDataset(
        filename=file,
        norm_dict=norm_dict,
        variables=variables,
        stage="fit",
        labels={"jets": ["flavour_label"], "tracks": list(V1_TRACK_LABELS)},
    )
    # de-degenerate the untrained heads BEFORE the v2 weight transfer, so the
    # comparison has discriminating power on the prediction side too
    # (M3-review fix — see _decollapse_heads); both sides stay weight-matched
    _decollapse_heads(wrapper, ds, batch_size)

    # v2 side: weight-transferred module dict + compiled VAL plan (raw logits)
    modules = build_gn2v2_modules(norm_dict)
    plan = compile_gn2v2(modules, Mode.VAL)
    bind_all(modules, resolve_bind_schema(plan))
    holder = nn.ModuleDict(modules)
    holder.load_state_dict(map_v1_state_dict(wrapper.state_dict(), modules), strict=True)
    holder.eval()
    executor = Executor(plan)
    pl_stub = SimpleNamespace(_graph_modules=modules)

    tasks = (
        ("jets_classification", "jets", "flavour_label", list(JET_CLASSES)),
        ("track_origin", "tracks", "ftagTruthOriginLabel", list(ORIGIN_CLASSES)),
    )
    v1_cbs: dict[str, V1ConfusionMatrix] = {}
    v2_cbs: dict[str, ConfusionMatrix] = {}
    for task_name, *_ in tasks:
        v1_cb = V1ConfusionMatrix(task_name=task_name)
        v1_cb.setup(SimpleNamespace(model=wrapper), None, stage="fit")
        v2_cb = ConfusionMatrix(task_name=task_name)
        v2_cb.setup(None, pl_stub, stage="fit")
        v1_cbs[task_name], v2_cbs[task_name] = v1_cb, v2_cb

    n_rows = len(ds)
    for index in range(n_batches):
        start = index * batch_size
        stop = min(start + batch_size, n_rows)
        inputs, masks, labels = ds[np.s_[start:stop]]
        with torch.no_grad():
            v1_preds, _ = wrapper(
                {key: tensor.clone() for key, tensor in inputs.items()},
                {key: tensor.clone() for key, tensor in masks.items()},
                None,
            )
        v1_outputs = {"outputs": {"preds": v1_preds, "labels": labels}}
        bundle = Bundle()
        for stream, tensor in inputs.items():
            bundle.set(f"inputs.{stream}", tensor.clone())
        for stream, tensor in masks.items():
            bundle.set(f"masks.{stream}", tensor.clone())
        for stream, stream_labels in labels.items():
            for label, tensor in stream_labels.items():
                bundle.set(f"labels.{stream}.{label}", tensor.clone())
        with torch.no_grad():
            executed = executor.run(bundle)
        for task_name, *_ in tasks:
            v1_cbs[task_name].on_validation_batch_end(None, None, v1_outputs, None, index)
            v2_cbs[task_name].on_validation_batch_end(
                None, pl_stub, {"bundle": executed}, None, index
            )

    per_task: dict[str, Any] = {}
    for task_name, _stream, _label, class_names in tasks:
        v1_cb, v2_cb = v1_cbs[task_name], v2_cbs[task_name]
        # snapshot v1 lists BEFORE any epoch-end reset; drive the v2 epoch
        # end so the matrix/stash path runs (no logger attached)
        v1_truth, v1_pred = list(v1_cb.truth_labels), list(v1_cb.pred_labels)
        v2_cb.on_validation_epoch_end(SimpleNamespace(logger=None, current_epoch=0), pl_stub)
        lists_equal = (
            len(v1_truth) == len(v2_cb.last_truth_labels) > 0
            and all(
                torch.equal(torch.as_tensor(ours), torch.as_tensor(theirs))
                for ours, theirs in zip(v2_cb.last_truth_labels, v1_truth, strict=True)
            )
            and all(
                torch.equal(torch.as_tensor(ours), torch.as_tensor(theirs))
                for ours, theirs in zip(v2_cb.last_pred_labels, v1_pred, strict=True)
            )
        )
        v1_matrix, v1_ignored = ConfusionMatrix.confusion_counts(
            v1_truth, v1_pred, len(class_names)
        )
        assert v2_cb.last_matrix is not None
        # non-degeneracy bar (M3-review fix, _decollapse_heads): the
        # prediction side must spread over >= 2 classes or the pred-side
        # comparison proves nothing
        flat_preds = torch.cat([torch.as_tensor(x).flatten() for x in v2_cb.last_pred_labels])
        pred_classes = int(torch.unique(flat_preds).numel())
        per_task[task_name] = {
            "n_entries": len(v1_truth),
            "lists_equal": bool(lists_equal),
            "matrix_equal": bool(torch.equal(v1_matrix, v2_cb.last_matrix)),
            "ignored_equal": v1_ignored == v2_cb.last_ignored,
            "pred_classes": pred_classes,
            "pred_diversity_ok": pred_classes >= 2,
            "v1_ignored": v1_ignored,
            "v2_ignored": v2_cb.last_ignored,
            "v1_matrix": v1_matrix.tolist(),
            "v2_matrix": v2_cb.last_matrix.tolist(),
            "requires": list(v2_cb.requires),
        }

    passed = all(
        record["lists_equal"]
        and record["matrix_equal"]
        and record["ignored_equal"]
        and record["pred_diversity_ok"]
        for record in per_task.values()
    )
    criterion = (
        "per gated task (jets_classification, track_origin): accumulated truth/pred label "
        "lists element-wise torch.equal; integer counts matrices and ignored counts identical "
        "under the same out-of-range reduction (exact counts, no tolerance); >= 2 distinct "
        "predicted classes per task (non-degeneracy — _decollapse_heads)"
    )
    report = _base_report(
        "w5_metrics_parity",
        passed,
        criterion,
        {
            "file": str(file),
            "norm_dict": str(norm_dict),
            "n_batches": n_batches,
            "batch_size": batch_size,
            "val_preds_note": (
                "VAL publishes RAW logits both sides (design §3.3) — argmax is "
                "conversion-invariant, so the comparison is exact"
            ),
        },
    )
    report["tasks"] = per_task

    for task_name, record in per_task.items():
        print(
            f"{task_name}: entries={record['n_entries']} "
            f"lists_equal={record['lists_equal']} matrix_equal={record['matrix_equal']} "
            f"ignored v1={record['v1_ignored']} v2={record['v2_ignored']} "
            f"pred_classes={record['pred_classes']} (diversity_ok={record['pred_diversity_ok']})"
        )
        for row_v1, row_v2 in zip(record["v1_matrix"], record["v2_matrix"], strict=True):
            marker = "" if row_v1 == row_v2 else "   <- MISMATCH"
            print(f"  v1 {row_v1}  v2 {row_v2}{marker}")
    _print_verdict("w5", passed, criterion, _emit_report(report, outdir, "w5"))
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
        Parser with ``w1``-``w5`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.gates_m3", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)

    w1 = sub.add_parser("w1", help="writer parity (dummy): v1 vs v2 eval H5, byte-schema+values")
    _add_common(w1)
    w1.add_argument(
        "--file", type=Path, default=None, help="test H5 (default: generate dummy data)"
    )
    w1.add_argument("--norm-dict", type=Path, default=None, help="norm dict (required with --file)")
    w1.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="default 96: a non-divisor of --num-test=1000, so the partial final batch "
        "(40 rows) is exercised (run_w1 docstring)",
    )
    w1.add_argument("--num-test", type=int, default=1000)

    w2 = sub.add_parser("w2", help="writer parity (real file): same comparison at scale")
    _add_common(w2)
    w2.add_argument("--file", type=Path, required=True, help="real test/val H5 file")
    w2.add_argument("--norm-dict", type=Path, required=True)
    w2.add_argument(
        "--class-dict", type=Path, default=None, help="recorded only (TEST carries no losses)"
    )
    w2.add_argument("--batch-size", type=int, default=1000)
    w2.add_argument("--num-test", type=int, default=5000, help="row cap (CPU budget)")
    w2.add_argument("--jet-vars", type=str, default=None, help="comma-separated override")
    w2.add_argument("--track-vars", type=str, default=None, help="comma-separated override")

    w3 = sub.add_parser("w3", help="negative controls: dead-preds error + corrupted comparison")
    _add_common(w3)
    w3.add_argument("--batch-size", type=int, default=100)
    w3.add_argument("--num-test", type=int, default=300)

    w4 = sub.add_parser("w4", help="ergonomics: custom Writer + 4-line override -> new column")
    _add_common(w4)
    w4.add_argument("--batch-size", type=int, default=100)
    w4.add_argument("--num-test", type=int, default=300)

    w5 = sub.add_parser("w5", help="metrics parity: v1 vs v2 ConfusionMatrix values")
    _add_common(w5)
    w5.add_argument(
        "--file", type=Path, default=None, help="H5 with the fixture variables + GN2 labels"
    )
    w5.add_argument("--norm-dict", type=Path, default=None, help="norm dict (required with --file)")
    w5.add_argument("--n-batches", type=int, default=3)
    w5.add_argument("--batch-size", type=int, default=100)
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
    """Run one M3 gate from the command line.

    Returns
    -------
    int
        0 if the gate passed, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    if args.gate == "w1":
        code, _ = run_w1(
            args.outdir,
            file=args.file,
            norm_dict=args.norm_dict,
            batch_size=args.batch_size,
            num_test=args.num_test,
        )
    elif args.gate == "w2":
        code, _ = run_w2(
            args.file,
            args.norm_dict,
            args.outdir,
            class_dict=args.class_dict,
            batch_size=args.batch_size,
            num_test=args.num_test,
            jet_vars=_csv_list(args.jet_vars),
            track_vars=_csv_list(args.track_vars),
        )
    elif args.gate == "w3":
        code, _ = run_w3(args.outdir, batch_size=args.batch_size, num_test=args.num_test)
    elif args.gate == "w4":
        code, _ = run_w4(args.outdir, batch_size=args.batch_size, num_test=args.num_test)
    else:
        code, _ = run_w5(
            args.outdir,
            file=args.file,
            norm_dict=args.norm_dict,
            n_batches=args.n_batches,
            batch_size=args.batch_size,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
