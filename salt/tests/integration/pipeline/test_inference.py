"""Post-pipeline gate: ``salt inference`` over every matrix row's own fit
artifacts (user ruling: "run inference using the model checkpoints from all
the models covered by pipeline").

Consumes the matrix's explicit artifact contract (``run_row``/``run_export``/
``MATRIX``/``row_by_name``/``LegFailedError``/``golden_path`` from
``test_pipeline``) rather than relying on pytest file/collection ordering:
``run_row`` is memoised module-level in ``test_pipeline``, so invoking this
file directly (``pytest salt/tests/integration/pipeline/test_inference.py``)
rebuilds whatever row artifacts it needs, and a producer row's fit failure
surfaces as an actionable skip naming that row (mirrors ``test_pipeline``'s
own ``LegFailedError`` handling).

Parametrised over every ``fit=True`` MATRIX row. A row skips, with a stated
reason, rather than running silently-wrong or erroring uninformatively, when:
it is ROOT-fed (no H5 file to build a labelled/label-stripped pair from), its
``outputs:`` section assembles no ONNX export selection (``salt inference``
is inexpressible for it — same contract ``salt inference`` itself enforces),
or its own fit (or a chained producer's) failed.

Three gates, generalised per row wherever the committed golden
(``golden_path``, the ``onnx`` section) supports it:

(a) ``TestExportSelectionColumns`` — the inference H5's task columns
    correspond 1:1 to the export-mode selection / ONNX tuple order, dtypes
    included;
(b) ``TestGn2v2OpendataValuesMatchOnnxRuntime`` — H5 values equal
    onnxruntime outputs on the SAME real per-jet features, at check_onnx
    tolerance;
(c) ``TestLabelStripped`` — a label-stripped copy of the row's own test file
    runs green with identical prediction columns.

Gate (b), and the input-copy/pad-mask structural check, are anchored to the
single row ``gn2v2_opendata`` (the flagship default) rather than generalised:
doing so for every do_onnx row would mean re-deriving each config's
global-vs-sequence ONNX input-port mapping (jets vs. tracks vs. flows vs.
per-object masks, ...) outside the model's own resolved export contract —
substantial duplication of ``salt.outputs.sinks.onnx.export`` internals
across ~10 structurally different configs for a check ``test_pipeline``'s
``test_export`` (check_onnx's random-input sweep) already partially covers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from numpy.lib.recfunctions import repack_fields

from salt.graph.errors import ConfigError
from salt.inference import run_inference
from salt.outputs.sinks.onnx import make_session
from salt.schema import dump_schema, save_schema

from .test_pipeline import (
    FEEDS,
    GPU_ROWS,
    MATRIX,
    Artifacts,
    LegFailedError,
    RootDepsMissingError,
    Row,
    _feed_context,  # noqa: PLC2701 - same-package fixture builder, see module docstring
    golden_path,
    row_by_name,
    run_export,
    run_row,
)

pytestmark = pytest.mark.cpu_always

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

_FIT_CAPABLE_ROWS = [row for row in MATRIX if row.fit]
_PARAMS = [
    pytest.param(row.test_name, marks=(pytest.mark.gpu,) if row.test_name in GPU_ROWS else ())
    for row in _FIT_CAPABLE_ROWS
]


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


def _load_golden(row: Row) -> dict | None:
    """The committed output golden for ``row``, or ``None`` if none is committed."""
    path = golden_path(row)
    return json.loads(path.read_text()) if path.is_file() else None


def _golden_onnx_or_skip(name: str) -> tuple[dict, str]:
    """``(onnx_section, run_name)`` from ``name``'s golden, or a skip.

    Skips (rather than raising) both when no golden is committed and when a
    committed golden predates this row's ONNX export block (``onnx`` key
    absent — a tracked drift some rows currently carry, not a bug here; see
    the study's golden-regeneration TODO).
    """
    row = row_by_name(name)
    golden = _load_golden(row)
    if golden is None or golden.get("onnx") is None:
        pytest.skip(
            f"{name}: no committed ONNX golden section at {golden_path(row)} "
            "(missing file, or a golden pending regeneration for this row)"
        )
    return golden["onnx"], golden["provenance"]["run_name"]


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

    Memoised per row, mirroring ``test_pipeline.run_row``. Re-raises the SAME
    exception on every subsequent call for a row that failed once:
    ``RootDepsMissingError`` (ROOT-fed row, no H5), ``LegFailedError`` (this
    row's or a producer's fit failed), or ``ConfigError`` (the row's
    ``outputs:`` section assembles no ONNX export selection — inference is
    inexpressible for it).
    """
    cached = _INFERENCE_CACHE.get(name)
    if isinstance(cached, Exception):
        raise cached
    if cached is not None:
        return cached
    row = row_by_name(name)
    try:
        artifacts = _build_inference_pair(row, tmp_path_factory)
    except (LegFailedError, RootDepsMissingError, ConfigError) as exc:
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
    """
    if _uses_input_samples(fit):
        return [
            f"data.modules.input_samples.init_args.files.test={path}",
            f"data.modules.input_samples.init_args.num.test={N_TEST}",
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
    except ConfigError as exc:
        pytest.skip(f"{name}: salt inference cannot run — {exc}")


# ---------------------------------------------------------------------------
# gate (a): task columns == the ONNX tuple, per row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _PARAMS)
class TestExportSelectionColumns:
    """Gate (a): inference H5 columns == the export-mode selection, 1:1 with
    the ONNX tuple, checked against each row's own committed golden.
    """

    def test_task_columns_are_the_onnx_tuple(self, name, tmp_path_factory):
        """Every golden ONNX output has exactly one H5 column, right stream/order,
        and no other run-prefixed task column exists.
        """
        onnx, run_name = _golden_onnx_or_skip(name)
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        model_name = onnx["model_name"]
        manifest = onnx["task_manifest_onnx"]
        global_names = [
            f"{model_name}_{m['resolved_onnx_name']}" for m in manifest if m["axis"] == "global"
        ]
        per_token_names = [
            f"{model_name}_{m['resolved_onnx_name']}" for m in manifest if m["axis"] == "per_token"
        ]
        assert global_names + per_token_names == onnx["output_names"], (
            f"{name}: golden self-consistency (globals-then-per-token tuple)"
        )
        expected: dict[str, list[str]] = {}
        for entry in manifest:
            stream = entry["leaf_key"].split(".")[1]
            col_name = (
                f"{run_name}_{entry['resolved_onnx_name']}"
                if entry["prefix"]
                else entry["resolved_onnx_name"]
            )
            expected.setdefault(stream, []).append(col_name)
        with h5py.File(artifacts.labelled_output) as f:
            groups = {g: list(f[g].dtype.names or ()) for g in f}
        for stream, cols in expected.items():
            assert stream in groups, f"{name}: golden expects H5 group {stream!r}, none in output"
            present = [c for c in groups[stream] if c in set(cols)]
            assert present == cols, (
                f"{name}/{stream}: export-selection columns {present} != golden tuple {cols}"
            )
            extras = [
                c
                for c in groups[stream]
                if c.startswith((f"{run_name}_", "target_")) and c not in set(cols)
            ]
            assert not extras, f"{name}/{stream}: columns beyond the export selection: {extras}"

    def test_dtypes_match_onnx_tuple_dtypes(self, name, tmp_path_factory):
        """f4 columns for float32 tuple entries, integer for int8."""
        onnx, run_name = _golden_onnx_or_skip(name)
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        with h5py.File(artifacts.labelled_output) as f:
            dtypes = {g: f[g].dtype for g in f}
        for entry in onnx["task_manifest_onnx"]:
            stream = entry["leaf_key"].split(".")[1]
            col_name = (
                f"{run_name}_{entry['resolved_onnx_name']}"
                if entry["prefix"]
                else entry["resolved_onnx_name"]
            )
            if entry["onnx_dtype"] == "int8":
                assert np.issubdtype(dtypes[stream][col_name], np.integer), f"{name}: {col_name}"
            else:
                assert np.issubdtype(dtypes[stream][col_name], np.floating), f"{name}: {col_name}"


# ---------------------------------------------------------------------------
# gate (b), anchored: real-data H5-vs-onnxruntime numeric parity
# ---------------------------------------------------------------------------


class TestGn2v2OpendataValuesMatchOnnxRuntime:
    """Gate (b), anchored to ``gn2v2_opendata`` (the flagship default row): H5
    values equal onnxruntime outputs on the SAME real per-jet inputs — not
    just check_onnx's random-input sweep (already covered per-row by
    ``test_pipeline.test_export``). See the module docstring for why this is
    not generalised across the whole matrix.
    """

    ROW = "gn2v2_opendata"
    N = N_TEST

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
        onnx, run_name = _golden_onnx_or_skip(self.ROW)
        model_name = onnx["model_name"]
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
        assert ort_names == onnx["output_names"], f"{self.ROW}: exported tuple != golden tuple"
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
            for entry in onnx["task_manifest_onnx"]:
                stream = entry["leaf_key"].split(".")[1]
                col_name = (
                    f"{run_name}_{entry['resolved_onnx_name']}"
                    if entry["prefix"]
                    else entry["resolved_onnx_name"]
                )
                ref = ort_out[f"{model_name}_{entry['resolved_onnx_name']}"]
                if entry["axis"] == "global":
                    assert stream == "jets", f"global export leaf on unexpected stream {stream}"
                    np.testing.assert_allclose(
                        np.float64(jets_out[col_name][i]),
                        np.ravel(ref)[0],
                        rtol=1e-4,
                        atol=1e-4,
                        err_msg=f"{col_name} jet {i}",
                    )
                else:
                    n_valid = int(valid[i].sum())
                    got_tokens = tracks_out[col_name][i]
                    if entry["onnx_dtype"] == "int8":
                        np.testing.assert_array_equal(
                            got_tokens[:n_valid], ref, err_msg=f"{col_name} jet {i}"
                        )
                    else:
                        np.testing.assert_allclose(
                            got_tokens[:n_valid],
                            ref,
                            rtol=1e-4,
                            atol=1e-4,
                            err_msg=f"{col_name} jet {i}",
                        )
                    assert (got_tokens[n_valid:] == 0).all(), f"{col_name} jet {i}: pad not zero"
                    n_mismatch_checked += 1
        assert n_mismatch_checked > 0, "no per-token comparison ran (degenerate fixture)"


class TestGn2v2OpendataStructuralAnchor:
    """Anchored structural checks that depend on per-config writer wiring
    (which streams get an ``InputCopyWriter``/``PadMaskWriter``) rather than
    anything the committed golden encodes — see the module docstring.
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


@pytest.mark.parametrize("name", _PARAMS)
class TestLabelStripped:
    """Gate (c): a label-stripped copy of the row's own test file runs green
    with identical prediction columns (H5-fed rows only — ROOT-fed rows skip
    upstream in ``_artifacts_or_skip``).
    """

    def test_stripped_input_lacks_label_fields(self, name, tmp_path_factory):
        """Fixture sanity: no LABEL_FIELDS column survives in the stripped INPUT."""
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        with h5py.File(artifacts.stripped_input) as f:
            for group in f:
                assert not set(f[group].dtype.names or ()) & LABEL_FIELDS, group

    def test_prediction_columns_identical(self, name, tmp_path_factory):
        """Every export-selection column (and any pad mask) is bit-identical
        between the labelled and stripped runs — labels contribute nothing.
        """
        onnx, run_name = _golden_onnx_or_skip(name)
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        with h5py.File(artifacts.labelled_output) as fa, h5py.File(artifacts.stripped_output) as fb:
            for entry in onnx["task_manifest_onnx"]:
                stream = entry["leaf_key"].split(".")[1]
                col_name = (
                    f"{run_name}_{entry['resolved_onnx_name']}"
                    if entry["prefix"]
                    else entry["resolved_onnx_name"]
                )
                a, b = fa[stream][col_name][:], fb[stream][col_name][:]
                assert np.array_equal(a, b), f"{name}: {stream}/{col_name} differs when stripped"
            for stream in fa:
                if "mask" in (fa[stream].dtype.names or ()):
                    assert np.array_equal(fa[stream]["mask"][:], fb[stream]["mask"][:]), (
                        f"{name}: {stream}/mask differs when stripped"
                    )

    def test_stripped_output_carries_no_label_columns(self, name, tmp_path_factory):
        """The stripped-run H5 carries no label copy and no target_* column."""
        artifacts = _artifacts_or_skip(name, tmp_path_factory)
        with h5py.File(artifacts.stripped_output) as f:
            for group in f:
                names = set(f[group].dtype.names or ())
                assert not names & LABEL_FIELDS, f"{name}: label columns leaked into {group}"
                assert not {n for n in names if n.startswith("target_")}, f"{name}: {group}"
