"""M4.5 gate harness — U1 manifest coherence (plan 08, stage C; amendment-unified-writers.md).

One standalone gate, ``python -m salt.core.gates_m45 u1``, writing a
machine-readable ``u1_report.json`` into ``--outdir`` and a human-readable
table to stdout, exiting non-zero on failure (the gates_m3/m4 envelope). NO
machine data path appears anywhere: the fixture data is generated into
``--outdir`` from the in-repo generators (the W-gate dummy fixture).

**U1 — manifest coherence**: on a config with aux tasks (gn2v2-dummy:
``track_origin`` + ``track_vertexing``) PLUS custom writers in BOTH
narrowing directions (`ValidCountWriter` eval-only, `JetEchoOutput` blessed
export-only) PLUS ``export.combine`` post-processing, run the REAL
``salt2 test`` AND the REAL ``salt2 export`` from one checkpoint and
cross-check that the eval H5 columns and the ONNX output names/dtypes
derive from the SAME writer declarations under the amendment §5 naming
policy (one logical declaration, two prefixes):

- the exported graph's output list equals the writer-derived ordered
  manifest AND a PINNED literal reference (the full ordered list, not a
  set — amendment §7 risk 5 / U1(b) discipline), with the combined ``pbc``
  output inserted after the global entries and BEFORE the first per-token
  aux entry (the v1 rule, merge condition 5);
- global classification suffixes are equal modulo prefix between the eval
  file (``{run_name}_pb`` ...) and the graph (``{model_name}_pb`` ...) —
  U1(a), observed-vs-observed;
- the vertexing output is ONE shared constant
  (`salt.core.writers.names.VERTEX_INDEX`): bare eval column while
  ``prefix_vertex_column`` is down (v1 byte parity, recorded), prefixed
  ONNX output (amendment §5 rule 6);
- both mode-narrowing directions express: the eval-only column lands in
  the H5 and NOT in the graph; the export-only outputs land in the graph
  and NOT in the H5 (merge condition 3 — `ExportOnlyWriter` is the
  positive control, the in-gate `UnblessedStub` rejection the negative);
- ``onnx_tasks`` narrowing (merge condition 1) drops exactly the named
  task from ``salt2 export --manifest`` and nothing else;
- ``salt2 graph resolve --annotate`` writes the §4.4 comment block and the
  annotation MATCHES BOTH REAL manifests (every annotated ONNX name == the
  real graph's ordered output list; every annotated eval column exists in
  the real eval file), idempotently (merge condition 8);
- in-gate negative controls: the unblessed export-only stub shape is a
  hard `ConfigError` through the real CLI, and a config-declared
  ``export.outputs`` hits the M4.5 migration hard error (§4.1 bar).

Amendment U1 letters: (a)/(b)/(e)/(f) run here at gate scale; (c) the
``class_names``-permutation negative control and (d) the both-empty-writer
`ConfigError` are unit-gated in ``test_manifest.py`` (TestManifestDerivation
/ TestExportOnlyStory) — recorded, not duplicated.

Negative-control hook (the pytest suite, ``test_gates_m45.py``): `run_u1`
takes a python-only ``corruption`` keyword applied to the OBSERVED ONNX
output-name list before the coherence comparisons — simulating an exporter
output surface that diverged from the writer declarations; the gate must
FAIL, locally (surface/controls stay green). The gates_m2/m3/m4 pattern,
never exposed on the CLI.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.cli import MANIFEST_BEGIN, MANIFEST_END

# the W1 fixture path REUSED on purpose: U1 cross-checks the very eval
# surface the W gates byte-gate, so its checkpoint/data construction must be
# the gates_m3 one, not a re-implementation that could drift (plan 08)
from salt.core.gates_m3 import (
    _dummy_eval_data,
    _fixture_variables,
    _sample_name,
    _v2_checkpoint,
)
from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import TensorSpec
from salt.core.main import CONFIG_DIR, Salt2CLI
from salt.core.main import main as salt2_main
from salt.core.onnx import (
    ExportOutput,
    attach_manifest,
    make_session,
    ordered_output_names,
    resolve_export_config,
)
from salt.core.writers import VERTEX_INDEX, ExportOnlyWriter, Writer, WriterCallback
from salt.tests.core.gn2_fixture import build_test_gn2

__all__ = [
    "EXPECTED_NARROWED_NAMES",
    "EXPECTED_OUTPUT_ROWS",
    "MODEL_NAME",
    "RUN_NAME",
    "U1_OVERRIDE_YAML",
    "V1_TASK_OUTPUT_LIST",
    "JetEchoOutput",
    "UnblessedStub",
    "ValidCountWriter",
    "main",
    "run_u1",
]

_DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"

RUN_NAME = "GN2v2_dummy"
"""The fixture run ``name:`` (gn2v2-dummy.yaml) — the TEST column prefix
(amendment §5 rule 2). Carries an underscore on purpose: the eval and ONNX
prefixes must visibly differ for the modulo-prefix checks to have teeth."""

MODEL_NAME = "GN2v2dummy"
"""The fixture ``export.model_name`` — the ONNX output prefix (rule 3)."""

ECHO_PORT = "normed.jets"
ECHO_SUFFIXES = ("jetEcho0", "jetEcho1")
"""One suffix per gn2v2-dummy jet variable (pt_btagJes, eta_btagJes) —
`JetEchoOutput` echoes the [1, 2] normalised jet vector as split scalars."""

EXPECTED_OUTPUT_ROWS = (
    (f"{MODEL_NAME}_pb", "float32"),
    (f"{MODEL_NAME}_pc", "float32"),
    (f"{MODEL_NAME}_pu", "float32"),
    (f"{MODEL_NAME}_pbc", "float32"),  # combine: after globals, BEFORE aux (condition 5)
    (f"{MODEL_NAME}_TrackOrigin", "int8"),
    (f"{MODEL_NAME}_{VERTEX_INDEX}", "int8"),
    (f"{MODEL_NAME}_{ECHO_SUFFIXES[0]}", "float32"),
    (f"{MODEL_NAME}_{ECHO_SUFFIXES[1]}", "float32"),
)
"""The PINNED full ordered (name, dtype) reference for the exported graph.

Hand-derived, NOT computed from the machinery under gate (the non-circular
half of the U1(b) full-ordered-list bar): TaskWriter entries in the v1
output order (globals pb/pc/pu from ``class_names`` via Flavours, then the
sequence-aux ``TrackOrigin`` argmax and the shared-constant
``VertexIndex``), the ``pbc`` combine inserted before the FIRST per-token
entry (``combine_insertion_index``, v1 ``to_onnx.py:258-292``), then the
export-only echo scalars in writer config order — global split_scalars
entries declared AFTER the aux writers stay after them (manifest order =
writer order, amendment §5 rule 5; the insertion rule keys off the first
per-token entry, not "the end of the globals")."""

NARROWED_ONNX_TASKS = ("jets_classification", "track_vertexing")
"""The merge-condition-1 narrowing: drop ``track_origin`` ONLY — the v1
``tasks_to_output`` expressiveness ("export track_vertexing but not
track_origin", both on stream ``tracks``)."""

EXPECTED_NARROWED_NAMES = tuple(
    name for name, _ in EXPECTED_OUTPUT_ROWS if name != f"{MODEL_NAME}_TrackOrigin"
)
"""`EXPECTED_OUTPUT_ROWS` minus exactly the narrowed-out task's output."""

V1_TASK_OUTPUT_LIST = (
    ("preds.jets.jets_classification", ("pb", "pc", "pu"), "split_scalars", "float32"),
    ("preds.tracks.track_origin", ("TrackOrigin",), "argmax", "int8"),
    ("preds.tracks.track_vertexing", (VERTEX_INDEX,), "vertex_union_find", "int8"),
)
"""The v1 GN2 export-output list (``to_onnx.py:258-292``) — the U1(b)
reference for the TaskWriter-derived half of the manifest (the unit-scale
twin lives in ``test_manifest.V1_OUTPUT_LIST``)."""

_ORT_TYPE = {"float32": "tensor(float)", "int8": "tensor(int8)"}
"""Manifest dtype -> onnxruntime ``NodeArg.type`` string."""

U1_OVERRIDE_YAML = """\
# u1 fixture override (M4.5 plan 08): custom writers in BOTH narrowing
# directions + export-block post-processing, stacked onto gn2v2-dummy.yaml
writers:
  modules:
    valid_count: {class_path: salt.core.gates_m45.ValidCountWriter}
    jet_echo: {class_path: salt.core.gates_m45.JetEchoOutput}
export:
  model_name: GN2v2dummy
  inputs:
    - {port: inputs.jets, name: jet_features}
    - {port: inputs.tracks, name: track_features, sequence: true, dyn_axis: n_tracks}
  combine:
    - name: pbc
      inputs: {pb: 0.5, pc: 0.5}
"""
"""The U1 fixture override config (embedded in the report verbatim)."""

_STUB_OVERRIDE_YAML = """\
# negative control: the export-only stub shape WITHOUT the explicit flag
writers:
  modules:
    stub: {class_path: salt.core.gates_m45.UnblessedStub}
"""

_LEGACY_OUTPUTS_YAML = """\
# negative control: a pre-M4.5 config still declaring export.outputs
export:
  model_name: GN2v2dummy
  inputs:
    - {port: inputs.jets, name: jet_features}
    - {port: inputs.tracks, name: track_features, sequence: true, dyn_axis: n_tracks}
  outputs:
    - {port: preds.jets.jets_classification, names: [pb, pc, pu]}
"""

VERTEX_COLUMN_DECISION = (
    "vertexing column is BARE 'VertexIndex' (i8, no run-name prefix) in TEST while "
    "prefix_vertex_column=false (v1 byte parity, M3 decision) and "
    "'{model_name}_VertexIndex' in ONNX (v1 to_onnx.py:287) — ONE shared suffix constant "
    "(salt.core.writers.names.VERTEX_INDEX), one per-mode prefix rule, one compat flag "
    "(amendment §5 rule 6; the M7 adjudication flips the flag, study CLAUDE.md TODO)"
)
"""The recorded TEST/ONNX vertex prefix asymmetry U1 asserts (never silent)."""


# ---------------------------------------------------------------------------
# fixture writers (config-instantiable via class_path)
# ---------------------------------------------------------------------------


class ValidCountWriter(Writer):
    """The EVAL-ONLY direction of the U1 fixture: one custom H5 column.

    The design §8 / gate-W4 custom-writer journey re-used as the U1
    eval-only control: consumes the TEST pad mask, writes the per-jet
    valid-track count, and (deliberately) never overrides
    `Writer.onnx_outputs` — the default ``[]`` is the eval-only story
    (amendment merge condition 3). `column_manifest` is overridden so the
    §4.4 annotation block covers the custom column too.
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

    def column_manifest(self, ctx, run_name) -> dict[str, list[str]]:
        """The statically-derivable column (the §4.4 annotation surface).

        Returns
        -------
        dict[str, list[str]]
            ``{"jets": ["n_valid_tracks"]}`` (no run-name prefix — custom
            writers own their full column names, design §8).
        """
        del ctx, run_name
        return {"jets": [self.COLUMN]}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Count valid (non-padded) tracks per jet from the batch pad mask.

        Returns
        -------
        dict[str, np.ndarray]
            The per-jet count fragment.
        """
        del rows
        mask = bundle.get("masks.tracks").cpu().numpy()
        counts = (~mask).sum(-1, keepdims=True).astype("i4")
        return {"jets": u2s(counts, np.dtype([(self.COLUMN, "i4")]))}


class JetEchoOutput(ExportOnlyWriter):
    """The EXPORT-ONLY direction of the U1 fixture (the BLESSED pattern).

    Subclasses `ExportOnlyWriter` (amendment merge condition 3 — the
    explicit ``export_only`` flag is what blesses the shape) and declares
    one ``split_scalars`` entry echoing the two normalised jet features
    (`ECHO_PORT` — a non-``preds`` port is legal, amendment §2.1) as extra
    Athena outputs with NO eval analogue. The output math is the entry's
    REGISTERED reduce; there is no ``write()`` (amendment cost 1).
    """

    def onnx_outputs(self, ctx) -> list[ExportOutput]:
        """The export-only manifest entry — this writer's single role.

        Returns
        -------
        list[ExportOutput]
            One ``split_scalars`` entry on `ECHO_PORT` (`ECHO_SUFFIXES` —
            one suffix per fixture jet variable, validated against the
            port's declared width at plan compile).
        """
        del ctx
        return [ExportOutput(port=ECHO_PORT, names=list(ECHO_SUFFIXES))]


class UnblessedStub(Writer):
    """The cargo-cult stub shape (negative control): export entries, no flag.

    Identical role surface to `JetEchoOutput` but WITHOUT subclassing
    `ExportOnlyWriter` / setting ``export_only`` — writer-role validation
    must reject it with the merge-condition-3 error naming the blessed
    pattern (`WriterCallback._validate_writer_roles`).
    """

    def requires(self, ctx) -> dict[str, TensorSpec]:
        """No TEST demand (the stub shape).

        Returns
        -------
        dict[str, TensorSpec]
            Always empty.
        """
        del ctx
        return {}

    def columns(self, ctx) -> dict[str, np.dtype]:
        """No eval columns (the stub shape).

        Returns
        -------
        dict[str, np.dtype]
            Always empty.
        """
        del ctx
        return {}

    def write(self, bundle: Bundle, rows: slice) -> dict[str, np.ndarray]:
        """Nothing to write (never reached: the role validation rejects first).

        Returns
        -------
        dict[str, np.ndarray]
            Always empty.
        """
        del bundle, rows
        return {}

    def onnx_outputs(self, ctx) -> list[ExportOutput]:
        """A non-empty manifest WITHOUT the export_only flag — the violation.

        Returns
        -------
        list[ExportOutput]
            One entry (the rejected stub declaration).
        """
        del ctx
        return [ExportOutput(port="pooled.global", names=["stub0", "stub1", "stub2"])]


# ---------------------------------------------------------------------------
# shared report helpers (the gates_m2/m3/m4 envelope, kept standalone per harness)
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
    """Build the common report envelope shared with the W/O gates.

    Returns
    -------
    dict[str, Any]
        Envelope with gate name, timestamp, verdict, criterion, config and
        environment fields; gate-specific sections are added by the caller.
    """
    import onnx  # noqa: PLC0415 - heavy import, version recording only
    import onnxruntime  # noqa: PLC0415 - heavy import, version recording only

    return {
        "gate": gate,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "criterion": criterion,
        "config": config,
        "environment": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "device": "cpu",
        },
    }


def _print_checks(checks: dict[str, bool]) -> None:
    """Print a name/PASS-FAIL table for a checks mapping."""
    for name, ok in checks.items():
        print(f"  {name:<52} {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _run_free_cli(args: Sequence[str]) -> Any:
    """Parse a config stack through the REAL salt2 surface, run-free.

    Returns
    -------
    Salt2CLI
        The run-free CLI (modules/datamodule constructed, nothing run).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*args parameter is intended to run from within Python.*"
        )
        return Salt2CLI(args=list(args), run=False)


def _writer_callback(cli: Any) -> WriterCallback:
    """Assemble the parsed ``writers:`` block (the runtime assembly, static).

    Returns
    -------
    WriterCallback
        The callback whose manifest the exporter consumes.
    """
    modules = {
        name: writer
        for name, writer in (cli._get(cli.config_init, "writers.modules") or {}).items()  # noqa: SLF001 - the main.py/cli.py adapter precedent
        if writer is not None
    }
    return WriterCallback(modules=modules)


def _is_subsequence(sub: Sequence[str], seq: Sequence[str]) -> bool:
    """Check that `sub` appears in `seq` in order (not necessarily contiguous).

    Returns
    -------
    bool
        True when `sub` is an ordered subsequence of `seq`.
    """
    it = iter(seq)
    return all(item in it for item in sub)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two plain config dicts (override wins per key).

    Mirrors the DeepMergeParser dict-union semantics closely enough to
    produce the single merged config file the ``--annotate`` control uses
    (one self-contained annotated file; ``salt2 graph resolve`` also
    accepts a repeatable ``-c`` stack since the M4.5 fix stage, annotating
    the LAST file — the merged-file route keeps this gate's annotated
    artifact self-contained).

    Returns
    -------
    dict
        A new merged dict (inputs not mutated).
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _capture_main(argv: Sequence[str]) -> tuple[int, str, str]:
    """Run the in-process ``salt2`` entry point capturing stdout/stderr.

    Returns
    -------
    tuple[int, str, str]
        ``(exit_code, stdout, stderr)``.
    """
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = salt2_main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def _parse_annotation(path: Path) -> dict[str, Any]:
    """Parse the §4.4 manifest comment block out of an annotated config.

    Returns
    -------
    dict[str, Any]
        ``n_blocks``, ``eval`` (``{(writer, stream): [columns]}``),
        ``file_dependent`` (writer names), ``onnx_names`` (ordered list).
    """
    text = path.read_text()
    n_blocks = text.count(MANIFEST_BEGIN)
    eval_cols: dict[tuple[str, str], list[str]] = {}
    file_dependent: list[str] = []
    onnx_names: list[str] = []
    in_block = False
    in_onnx = False
    for line in text.splitlines():
        if line.startswith(MANIFEST_BEGIN):
            in_block, in_onnx = True, False
            continue
        if not in_block:
            continue
        if line.startswith(MANIFEST_END):
            break
        if line.startswith("# onnx outputs"):
            in_onnx = True
            continue
        if in_onnx:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "#":
                onnx_names.append(parts[1])
            continue
        if match := re.match(r"^#   \[(\w+)\] file-dependent", line):
            file_dependent.append(match[1])
            continue
        if match := re.match(r"^#   \[(\w+)\] (\w+): (.+)$", line):
            eval_cols[match[1], match[2]] = match[3].split()
    return {
        "n_blocks": n_blocks,
        "eval": eval_cols,
        "file_dependent": file_dependent,
        "onnx_names": onnx_names,
    }


# ---------------------------------------------------------------------------
# U1 — manifest coherence
# ---------------------------------------------------------------------------


def run_u1(
    outdir: Path | str,
    *,
    batch_size: int = 96,
    num_test: int = 1000,
    check_trials: int = 10,
    check_max_length: int = 40,
    corruption: Callable[[list[str]], list[str]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """U1: eval columns and ONNX outputs derive from ONE set of writer declarations.

    Construction: the W1 dummy fixture + weight-matched v2 checkpoint (the
    gates_m3 path, reused on purpose), the shipped ``gn2v2-dummy.yaml``
    stacked with `U1_OVERRIDE_YAML` (eval-only + export-only custom writers,
    ``export.combine`` pbc), evaluated through the REAL ``salt2 test`` and
    exported through the REAL ``salt2 export`` (its torch-vs-ONNX sweep
    checker kept ON at the 1e-6 gate bar). The observed eval H5 columns and
    ONNX graph outputs are then cross-checked against each other, against
    the statically-assembled writer manifest, and against PINNED literal
    references (`EXPECTED_OUTPUT_ROWS`, `V1_TASK_OUTPUT_LIST`) — see the
    module docstring for the full criterion list.

    Parameters
    ----------
    outdir : Path | str
        Report + artifact output directory.
    batch_size : int, optional
        ``salt2 test`` batch size, by default 96 (the W1 non-divisor
        default — the partial final batch stays exercised).
    num_test : int, optional
        Rows evaluated, by default 1000 (the dummy-file size).
    check_trials : int, optional
        ``salt2 export`` checker draws per length, by default 10 (v1
        ``check.py:177``).
    check_max_length : int, optional
        Checker sweep lengths ``0..N-1``, by default 40 (``check.py:176``).
    corruption : Callable | None, optional
        TEST-ONLY hook applied to the OBSERVED ONNX output-name list before
        the coherence comparisons (negative control — the gate must FAIL),
        never exposed on the CLI, by default None.

    Returns
    -------
    tuple[int, dict[str, Any]]
        ``(exit_code, report)`` — 0 only if every check passed.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("U1 manifest coherence — eval H5 columns vs ONNX outputs from ONE writer manifest")
    print("=" * 96)

    # -- fixture: dummy data + weight-matched v2 checkpoint (the W1 path) ----
    data = _dummy_eval_data(outdir / "data")
    variables = _fixture_variables()
    fixture_dir = outdir / "v1_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    wrapper = build_test_gn2(fixture_dir, norm_dict=data.norm_dict)
    v2_ckpt = _v2_checkpoint(
        outdir / "v2", data.file, data.norm_dict, variables, wrapper, batch_size=batch_size
    )
    override_path = outdir / "u1_override.yaml"
    override_path.write_text(U1_OVERRIDE_YAML)
    norm_override = f"--model.modules.norm.init_args.norm_dict={data.norm_dict}"
    print(f"override config ({override_path}):")
    print(U1_OVERRIDE_YAML.strip())

    # -- the REAL salt2 test run (eval H5) ------------------------------------
    test_argv = [
        "test",
        "--config",
        str(_DUMMY_CFG),
        "--config",
        str(override_path),
        f"--data.test_file={data.file}",
        f"--data.num_test={num_test}",
        f"--data.batch_size={batch_size}",
        "--data.num_workers=0",
        norm_override,
        f"--ckpt_path={v2_ckpt}",
        "--trainer.accelerator=cpu",
        "--trainer.devices=1",
        # null-delete the base2 ProgressBar (D2 default-on); the stock
        # enable_progress_bar=false cannot coexist with a configured bar
        "--callbacks.progress=null",
        f"--trainer.default_root_dir={outdir / 'test_run'}",
    ]
    rc_test = salt2_main(test_argv)
    eval_path = v2_ckpt.parent / f"{v2_ckpt.stem}__test_{_sample_name(data.file)}.h5"

    # -- the REAL salt2 export run (checked .onnx) -----------------------------
    onnx_path = outdir / "export" / "network.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_argv = [
        "export",
        "--ckpt_path",
        str(v2_ckpt),
        "-c",
        str(_DUMMY_CFG),
        "-c",
        str(override_path),
        "--set",
        f"model.modules.norm.init_args.norm_dict={data.norm_dict}",
        "--output",
        str(onnx_path),
        "--trials",
        str(check_trials),
        "--max-length",
        str(check_max_length),
        "--float-atol",
        "1e-6",
    ]
    rc_export = salt2_main(export_argv)

    surface = {
        "salt2_test_returned_zero": rc_test == 0,
        "salt2_export_returned_zero_with_checker": rc_export == 0,
        "eval_output_exists": eval_path.is_file(),
        "onnx_output_exists": onnx_path.is_file(),
    }
    if not (surface["eval_output_exists"] and surface["onnx_output_exists"]):
        return _finish_u1_early(outdir, surface, test_argv, export_argv)

    # -- observed artifacts -----------------------------------------------------
    with h5py.File(eval_path) as fh:
        eval_columns = {name: list(fh[name].dtype.names or ()) for name in fh}
        eval_formats = {
            name: {col: str(fh[name].dtype.fields[col][0]) for col in fh[name].dtype.names or ()}
            for name in fh
        }
    session = make_session(onnx_path)
    observed_names = [out.name for out in session.get_outputs()]
    observed_types = [out.type for out in session.get_outputs()]
    if corruption is not None:
        observed_names = corruption(list(observed_names))
    gnn_config = json.loads(session.get_modelmeta().custom_metadata_map["gnn_config"])

    # -- the statically-assembled writer manifest (the same source salt2
    # export consumed — asserted against the OBSERVED artifacts) -------------
    cli = _run_free_cli([
        "--config",
        str(_DUMMY_CFG),
        "--config",
        str(override_path),
        norm_override,
    ])
    writer_cb = _writer_callback(cli)
    model_modules = cli.model._graph_modules  # noqa: SLF001 - the cli.py adapter precedent
    reader = cli.datamodule.reader
    run_name = cli._get(cli.config_init, "name")  # noqa: SLF001 - the cli.py adapter precedent
    export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001 - the cli.py adapter precedent
    per_writer_manifest = writer_cb.per_writer_onnx_manifest(model_modules, reader)
    per_writer_demand = writer_cb.per_writer_demand(model_modules, reader)
    column_manifests = writer_cb.column_manifests(model_modules, reader, run_name)
    manifest = writer_cb.onnx_manifest(model_modules, reader)
    resolved = attach_manifest(resolve_export_config(export_cfg, run_name), manifest)
    manifest_rows = ordered_output_names(resolved)

    coherence = _coherence_checks(
        observed_names,
        observed_types,
        gnn_config,
        manifest_rows,
        resolved,
        eval_columns,
        eval_formats,
        column_manifests,
        per_writer_manifest,
        model_modules,
        run_name,
    )
    modes = _mode_split_checks(
        observed_names, eval_columns, per_writer_manifest, per_writer_demand, column_manifests
    )
    narrowing = _narrowing_checks(override_path, data.norm_dict)
    annotation = _annotation_checks(
        outdir, override_path, data.norm_dict, observed_names, eval_columns
    )
    controls = _control_checks(outdir, data.norm_dict)

    sections = {
        "surface": surface,
        "coherence": coherence,
        "modes": modes,
        "narrowing": narrowing,
        "annotation": annotation,
        "controls": controls,
    }
    passed = all(ok for checks in sections.values() for ok in checks.values())
    criterion = (
        "one writer manifest, two surfaces: the exported ONNX graph's full ordered "
        "output/dtype list equals the writer-derived manifest AND the pinned reference "
        "(combine pbc after globals, BEFORE the aux entries — the v1 rule); eval H5 columns "
        "match the static writer column manifests; global suffixes equal modulo prefix "
        f"(run name {RUN_NAME!r} vs model_name {MODEL_NAME!r}); VertexIndex is ONE shared "
        "constant (TEST bare, ONNX prefixed — recorded); eval-only and export-only writers "
        "express both narrowing directions; onnx_tasks narrowing drops exactly the named "
        "task; the --annotate block matches BOTH real manifests idempotently; the unblessed "
        "stub and a config-declared export.outputs hard-error through the real CLI"
    )
    report = _base_report(
        "u1_manifest_coherence",
        passed,
        criterion,
        {
            "batch_size": batch_size,
            "num_test": num_test,
            "check_trials": check_trials,
            "check_max_length": check_max_length,
            "run_name": run_name,
            "model_name": MODEL_NAME,
            "eval_output": str(eval_path),
            "onnx_output": str(onnx_path),
            "test_argv": test_argv,
            "export_argv": export_argv,
            "corrupted_by_test_hook": corruption is not None,
        },
    )
    report.update(sections)
    report["override_yaml"] = U1_OVERRIDE_YAML
    report["observed_onnx_outputs"] = list(zip(observed_names, observed_types, strict=True))
    report["expected_onnx_outputs"] = [list(row) for row in EXPECTED_OUTPUT_ROWS]
    report["manifest_rows"] = [list(row) for row in manifest_rows]
    report["eval_columns"] = eval_columns
    report["static_column_manifests"] = column_manifests
    report["per_writer_onnx_ports"] = {
        name: [entry.port for entry in entries] for name, entries in per_writer_manifest.items()
    }
    report["vertex_column_decision"] = VERTEX_COLUMN_DECISION
    report["unit_scale_hooks"] = (
        "U1(c) class_names-permutation and U1(d) both-empty-writer controls are unit-gated "
        "in salt/tests/core/test_manifest.py (TestManifestDerivation / TestExportOnlyStory)"
    )

    print(f"eval output: {eval_path}")
    print(f"onnx output: {onnx_path}")
    print(f"{'output':<28}{'dtype':<16}{'expected':<28}")
    print("-" * 96)
    for (name, dtype), (exp_name, exp_dtype) in zip(
        list(zip(observed_names, observed_types, strict=True)),
        EXPECTED_OUTPUT_ROWS,
        strict=False,
    ):
        marker = "" if (name, dtype) == (exp_name, _ORT_TYPE[exp_dtype]) else "   <- MISMATCH"
        print(f"{name:<28}{dtype:<16}{exp_name:<28}{marker}")
    for stream, columns in eval_columns.items():
        print(f"eval group {stream!r}: {len(columns)} columns (tail: {columns[-4:]})")
    print(f"note: {VERTEX_COLUMN_DECISION}")
    for title, checks in sections.items():
        print(f"checks: {title}")
        _print_checks(checks)
    _print_verdict("u1", passed, criterion, _emit_report(report, outdir, "u1"))
    return (0 if passed else 1), report


def _finish_u1_early(
    outdir: Path,
    surface: dict[str, bool],
    test_argv: list[str],
    export_argv: list[str],
) -> tuple[int, dict[str, Any]]:
    """Emit a FAIL report when the salt2 runs produced no artifacts.

    Returns
    -------
    tuple[int, dict[str, Any]]
        Always ``(1, report)``.
    """
    criterion = "salt2 test and salt2 export must both produce their artifacts"
    report = _base_report(
        "u1_manifest_coherence",
        False,
        criterion,
        {"test_argv": test_argv, "export_argv": export_argv},
    )
    report["surface"] = surface
    _print_checks(surface)
    _print_verdict("u1", False, criterion, _emit_report(report, outdir, "u1"))
    return 1, report


def _coherence_checks(
    observed_names: list[str],
    observed_types: list[str],
    gnn_config: dict[str, Any],
    manifest_rows: list[tuple[str, str, str]],
    resolved: Any,
    eval_columns: dict[str, list[str]],
    eval_formats: dict[str, dict[str, str]],
    column_manifests: dict[str, dict[str, list[str]]],
    per_writer_manifest: dict[str, list[ExportOutput]],
    model_modules: dict[str, Any],
    run_name: str,
) -> dict[str, bool]:
    """The core cross-checks: one manifest, two observed surfaces.

    Returns
    -------
    dict[str, bool]
        The coherence checks (see `run_u1`).
    """
    expected_names = [name for name, _ in EXPECTED_OUTPUT_ROWS]
    expected_types = [_ORT_TYPE[dtype] for _, dtype in EXPECTED_OUTPUT_ROWS]
    run_prefix = f"{run_name}_"
    model_prefix = f"{MODEL_NAME}_"

    # U1(a): observed-vs-observed global suffixes, modulo the two prefixes
    eval_global_suffixes = [
        column.removeprefix(run_prefix)
        for column in eval_columns.get("jets", ())
        if column.startswith(run_prefix)
    ]
    onnx_global_suffixes = [
        name.removeprefix(model_prefix) for name in observed_names[: len(eval_global_suffixes)]
    ]

    # single ownership of the aux family (amendment §3 honesty note): the
    # SAME writer + task instance feeds the eval per-class probs AND the
    # ONNX argmax entry — both representations co-located, one owner
    origin_entry = next(
        (entry for entry in per_writer_manifest.get("tasks", []) if entry.name == "TrackOrigin"),
        None,
    )
    origin_module = model_modules.get("track_origin")
    origin_eval_columns = [f"{run_name}_p{c}" for c in getattr(origin_module, "class_names", ())]
    aux_single_owner = (
        origin_entry is not None
        and origin_module is not None
        and origin_entry.port == origin_module.pred_key
        and _is_subsequence(origin_eval_columns, column_manifests["tasks"].get("tracks", []))
    )

    # the TaskWriter-derived half of the resolved manifest == the v1 list
    task_entries = tuple(
        (
            entry.port,
            tuple(entry.names) if entry.names is not None else (entry.name,),
            entry.reduce,
            entry.dtype,
        )
        for entry in resolved.outputs
        if entry.port.startswith("preds.")
    )

    static_eval_ok = all(
        _is_subsequence(columns, eval_columns.get(stream, []))
        for streams in column_manifests.values()
        for stream, columns in streams.items()
    )
    jets_formats = eval_formats.get("jets", {})
    tracks_formats = eval_formats.get("tracks", {})
    return {
        "onnx_graph_equals_manifest_ordered_list": observed_names
        == [name for name, _, _ in manifest_rows],
        "onnx_graph_equals_pinned_reference": observed_names == expected_names,
        "onnx_dtypes_match_manifest": observed_types == expected_types,
        "combine_inserted_after_globals_before_aux": (
            f"{MODEL_NAME}_pbc" in observed_names
            and observed_names.index(f"{MODEL_NAME}_pbc")
            == observed_names.index(f"{MODEL_NAME}_pu") + 1
            and observed_names.index(f"{MODEL_NAME}_pbc")
            < observed_names.index(f"{MODEL_NAME}_TrackOrigin")
        ),
        "gnn_config_output_names_match_graph": gnn_config.get("output_names") == observed_names,
        "gnn_config_combine_recorded_v1_format": gnn_config.get("combine_outputs")
        == [["pbc", [[0.5, "pb"], [0.5, "pc"]]]],
        "tasks_manifest_is_the_v1_output_list": task_entries == V1_TASK_OUTPUT_LIST,
        "global_suffixes_equal_modulo_prefix": eval_global_suffixes
        == onnx_global_suffixes
        == ["pb", "pc", "pu"],
        "vertex_suffix_is_one_shared_constant": (
            VERTEX_INDEX in eval_columns.get("tracks", ())
            and f"{run_name}_{VERTEX_INDEX}" not in eval_columns.get("tracks", ())
            and f"{MODEL_NAME}_{VERTEX_INDEX}" in observed_names
        ),
        "aux_family_single_owner": bool(aux_single_owner),
        "eval_columns_match_static_manifests": static_eval_ok,
        "eval_task_column_formats_match_v1": (
            all(jets_formats.get(f"{run_name}_{s}") == "float32" for s in ("pb", "pc", "pu"))
            and tracks_formats.get(VERTEX_INDEX) == "int64"
        ),
    }


def _mode_split_checks(
    observed_names: list[str],
    eval_columns: dict[str, list[str]],
    per_writer_manifest: dict[str, list[ExportOutput]],
    per_writer_demand: dict[str, list[str]],
    column_manifests: dict[str, dict[str, list[str]]],
) -> dict[str, bool]:
    """Both mode-narrowing directions, asserted on the REAL artifacts.

    Returns
    -------
    dict[str, bool]
        The eval-only / export-only checks (merge condition 3).
    """
    echo_names = {f"{MODEL_NAME}_{suffix}" for suffix in ECHO_SUFFIXES}
    all_eval_columns = {column for columns in eval_columns.values() for column in columns}
    return {
        "eval_only_column_in_eval_file": ValidCountWriter.COLUMN in eval_columns.get("jets", ()),
        "eval_only_writer_declares_no_onnx_outputs": per_writer_manifest.get("valid_count") == [],
        "eval_only_column_absent_from_onnx": all(
            ValidCountWriter.COLUMN not in name for name in observed_names
        ),
        "export_only_outputs_in_onnx_graph": echo_names <= set(observed_names),
        "export_only_writer_declares_no_test_demand": per_writer_demand.get("jet_echo") == [],
        "export_only_writer_declares_no_eval_columns": column_manifests.get("jet_echo") == {},
        "export_only_suffixes_absent_from_eval_file": not (set(ECHO_SUFFIXES) & all_eval_columns),
    }


def _narrowing_checks(override_path: Path, norm_dict: Path) -> dict[str, bool]:
    """``onnx_tasks`` narrowing through the REAL ``salt2 export --manifest``.

    Returns
    -------
    dict[str, bool]
        The merge-condition-1 checks: exactly `NARROWED_ONNX_TASKS` survive.
    """
    onnx_tasks = json.dumps(list(NARROWED_ONNX_TASKS))
    rc, out, _ = _capture_main([
        "export",
        "--manifest",
        "-c",
        str(_DUMMY_CFG),
        "-c",
        str(override_path),
        "--set",
        f"model.modules.norm.init_args.norm_dict={norm_dict}",
        "--set",
        f"writers.modules.tasks.init_args.onnx_tasks={onnx_tasks}",
    ])
    print(f"salt2 export --manifest with onnx_tasks={onnx_tasks}:")
    print(out.strip())
    rows = [line.split()[0] for line in out.splitlines()[1:] if line.strip()]
    return {
        "narrowed_manifest_cli_returned_zero": rc == 0,
        "narrowed_manifest_drops_exactly_track_origin": rows == list(EXPECTED_NARROWED_NAMES),
    }


def _annotation_checks(
    outdir: Path,
    override_path: Path,
    norm_dict: Path,
    observed_names: list[str],
    eval_columns: dict[str, list[str]],
) -> dict[str, bool]:
    """``salt2 graph resolve --annotate`` vs BOTH real manifests (condition 8).

    A single merged config file is produced (keeping the annotated gate
    artifact self-contained; ``-c`` stacking is unit-covered in
    ``test_manifest.py``) with the norm-dict path resolved, annotated TWICE
    (the refresh must be idempotent), and the parsed comment block compared
    against the REAL session output list and the REAL eval file columns.

    Returns
    -------
    dict[str, bool]
        The annotation checks.
    """
    merged = _deep_merge(
        yaml.safe_load(_DUMMY_CFG.read_text()), yaml.safe_load(override_path.read_text())
    )
    merged["model"]["init_args"]["modules"]["norm"]["init_args"]["norm_dict"] = str(norm_dict)
    annotated_path = outdir / "annotated.yaml"
    annotated_path.write_text(yaml.dump(merged, sort_keys=False))
    argv = ["graph", "resolve", "-c", str(annotated_path), "--annotate"]
    rc1, out1, _ = _capture_main(argv)
    print("salt2 graph resolve --annotate block:")
    print(out1.strip())
    rc2, _, _ = _capture_main(argv)  # refresh must replace, not duplicate
    parsed = _parse_annotation(annotated_path)
    eval_annotated_ok = bool(parsed["eval"]) and all(
        _is_subsequence(columns, eval_columns.get(stream, []))
        for (_writer, stream), columns in parsed["eval"].items()
    )
    return {
        "annotate_cli_returned_zero": rc1 == 0 and rc2 == 0,
        "annotation_onnx_table_matches_real_graph": parsed["onnx_names"] == observed_names,
        "annotation_eval_columns_exist_in_real_file": eval_annotated_ok,
        "annotation_covers_custom_eval_column": parsed["eval"].get(("valid_count", "jets"))
        == [ValidCountWriter.COLUMN],
        "annotation_marks_file_dependent_writer": "inputs_copy" in parsed["file_dependent"],
        "annotation_refresh_idempotent": parsed["n_blocks"] == 1,
    }


def _control_checks(outdir: Path, norm_dict: Path) -> dict[str, bool]:
    """In-gate negative controls through the REAL CLI (no python shortcuts).

    1. The unblessed export-only stub shape must be a hard `ConfigError`
       naming the blessed pattern (merge condition 3).
    2. A config-declared ``export.outputs`` must hit the M4.5 migration
       hard error pointing at the writers (§4.1 bar).

    Returns
    -------
    dict[str, bool]
        The control checks.
    """
    stub_path = outdir / "stub_override.yaml"
    stub_path.write_text(_STUB_OVERRIDE_YAML)
    rc_stub, _, err_stub = _capture_main([
        "export",
        "--manifest",
        "-c",
        str(_DUMMY_CFG),
        "-c",
        str(stub_path),
        "--set",
        f"model.modules.norm.init_args.norm_dict={norm_dict}",
    ])
    print("control (1): unblessed export-only stub through salt2 export --manifest")
    print(err_stub.strip() or "<no stderr>")

    legacy_path = outdir / "legacy_outputs.yaml"
    legacy_path.write_text(_LEGACY_OUTPUTS_YAML)
    rc_legacy, _, err_legacy = _capture_main([
        "export",
        "--manifest",
        "-c",
        str(_DUMMY_CFG),
        "-c",
        str(legacy_path),
        "--set",
        f"model.modules.norm.init_args.norm_dict={norm_dict}",
    ])
    print("control (2): config-declared export.outputs through salt2 export --manifest")
    print(err_legacy.strip() or "<no stderr>")
    return {
        "unblessed_stub_rejected": rc_stub != 0,
        "stub_error_names_the_blessed_pattern": "export-only stub shape" in err_stub
        and "ExportOnlyWriter" in err_stub,
        "legacy_export_outputs_rejected": rc_legacy != 0,
        "legacy_error_is_the_migration_error": "REMOVED by the M4.5" in err_legacy
        and "writers" in err_legacy,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the gate subcommand parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with the ``u1`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="python -m salt.core.gates_m45", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="gate", required=True)
    u1 = sub.add_parser(
        "u1", help="manifest coherence: eval H5 columns vs ONNX outputs from ONE writer manifest"
    )
    u1.add_argument("--outdir", type=Path, required=True, help="report output directory")
    u1.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="default 96: the W1 non-divisor of --num-test=1000 (partial final batch)",
    )
    u1.add_argument("--num-test", type=int, default=1000)
    u1.add_argument("--check-trials", type=int, default=10, help="export-checker draws per length")
    u1.add_argument(
        "--check-max-length", type=int, default=40, help="export-checker sweep L=0..N-1"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the U1 gate from the command line.

    Returns
    -------
    int
        0 if the gate passed, 1 otherwise.
    """
    args = _build_parser().parse_args(argv)
    code, _ = run_u1(
        args.outdir,
        batch_size=args.batch_size,
        num_test=args.num_test,
        check_trials=args.check_trials,
        check_max_length=args.check_max_length,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
