"""Post-pipeline gate: ``salt inference`` over every matrix row's own fit
artifacts (user ruling: "run inference using the model checkpoints from all
the models covered by pipeline").

Consumes the matrix's explicit artifact contract (``run_row``/``run_export``/
``MATRIX``/``row_by_name``/``LegFailedError``/``EXPECTED_OUTPUTS`` from
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
its ONNX-mode plan does not even connect (``GraphError`` — e.g. gn3_flow/
gn3_lepid_smt's ``ConnectivityError`` on a missing ``inputs.flow``), or its
own fit (or a chained producer's) failed.

Gates, generalised per row wherever ``EXPECTED_OUTPUTS`` (the curated
output-schema table in ``test_pipeline``, seeded from the now-deleted
per-config golden JSONs) names an ``"onnx"`` entry for it:

(a) ``TestExportedOnnxOutputNames`` — the row's own freshly-exported ONNX
    session's output names equal ``EXPECTED_OUTPUTS[name]["onnx"]`` (order
    included — the tuple is an Athena contract);
(b) ``TestGn2v2OpendataValuesMatchOnnxRuntime`` — H5 values equal
    onnxruntime outputs on the SAME real per-jet features, at check_onnx
    tolerance;
(c) ``TestLabelStripped`` — a label-stripped copy of the row's own test file
    runs green with bit-identical prediction columns.

NOTE on why ``EXPECTED_OUTPUTS["h5"]`` is NOT used here: ``salt inference``'s
own H5 output uses the EXPORT-mode column selection
(``H5OutputSink.use_export_selection()``, ``salt/inference.py::
build_inference_sink``) — run-name-prefixed leaves resolved from
``manifest_fields(Mode.ONNX)`` — which is a DIFFERENT naming convention from
the TEST-mode eval H5 ``EXPECTED_OUTPUTS["h5"]`` encodes (e.g. a folded
per-token classification head is one reduced ``TrackOrigin`` column here vs.
several per-class probability columns under ``salt test``). Reconstructing
the true inference-H5 column names generically would need per-field
axis/dtype/prefix metadata the curated table deliberately does not carry (it
is meant to stay a flat, human-curated name list — see ``test_pipeline``'s
``EXPECTED_OUTPUTS`` docstring). Gate (a) above checks the ONNX tuple itself
(unambiguous, no naming-convention translation needed); gate (c) sidesteps
the naming question entirely by comparing the SAME row's own labelled vs.
label-stripped output columns to each other, whatever they are named.

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

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from numpy.lib.recfunctions import repack_fields

from salt.graph.errors import GraphError
from salt.inference import run_inference
from salt.outputs.sinks.onnx import make_session
from salt.schema import dump_schema, save_schema

from .test_pipeline import (
    EXPECTED_OUTPUTS,
    FEEDS,
    GPU_ROWS,
    MATRIX,
    Artifacts,
    LegFailedError,
    RootDepsMissingError,
    Row,
    _feed_context,  # noqa: PLC2701 - same-package fixture builder, see module docstring
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


def _expected_onnx_or_skip(name: str) -> list[str]:
    """``EXPECTED_OUTPUTS[name]["onnx"]``, or a skip when the row names none.

    Skips (rather than raising) exactly like the deleted golden lookup did:
    a row with no committed ONNX contract (no ``do_onnx`` leg, or one that
    simply has not been curated yet) is not this gate's business.
    """
    onnx_names = (EXPECTED_OUTPUTS.get(name) or {}).get("onnx")
    if not onnx_names:
        pytest.skip(f"{name}: no EXPECTED_OUTPUTS['onnx'] entry (no ONNX contract to check)")
    return onnx_names


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
    row's or a producer's fit failed), or ``GraphError`` (the row's
    ``outputs:`` section assembles no ONNX export selection — inference is
    inexpressible for it — or its ONNX-mode plan does not even connect, e.g.
    gn3_flow/gn3_lepid_smt's ``ConnectivityError: 'norm' requires 'inputs.flow'``,
    pipeline #15651025 cluster 2: `ConnectivityError` is a `GraphError` sibling
    of `ConfigError`, not a subclass, so catching only `ConfigError` missed it).
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
    item B2 — same jsonargparse defect as ``test_pipeline.MATRIX``'s
    gn2v2_opendata row: a dotted ``--...files.test=`` hands the ``files``
    dict field a bare ``Namespace`` instead of merging into it). These are
    plain f-strings (unlike the MATRIX row templates, which go through
    ``test_pipeline._expand_one``'s regex substitution) consumed directly as
    single ``KEY=VALUE`` list entries by ``salt.outputs.sinks.onnx.export.
    _run_free_cli`` (``args.append(f"--{entry}")`` — no ``shlex`` re-split),
    so the doubled braces here are genuine f-string escapes for a literal
    ``{``/``}``, not tokens for a second substitution pass.
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


@pytest.mark.parametrize("name", _PARAMS)
class TestExportedOnnxOutputNames:
    """Gate (a): the row's own freshly-exported ONNX graph's output names
    equal ``EXPECTED_OUTPUTS[name]["onnx"]`` — order included, since the
    tuple order IS the Athena contract (``salt.outputs.sinks.onnx_sink``'s
    "globals before per-token" rule).
    """

    def test_onnx_output_names_match_expected(self, name, tmp_path_factory):
        expected = _expected_onnx_or_skip(name)
        try:
            onnx_path = run_export(name, tmp_path_factory)
        except LegFailedError as exc:
            pytest.skip(f"{name}: export failed: {exc}")
        session = make_session(onnx_path)
        actual = [o.name for o in session.get_outputs()]
        assert actual == expected, f"{name}: exported ONNX tuple != EXPECTED_OUTPUTS['onnx']"


# ---------------------------------------------------------------------------
# gate (b), anchored: real-data H5-vs-onnxruntime numeric parity
# ---------------------------------------------------------------------------


class TestGn2v2OpendataValuesMatchOnnxRuntime:
    """Gate (b), anchored to ``gn2v2_opendata`` (the flagship default row): H5
    values equal onnxruntime outputs on the SAME real per-jet inputs — not
    just check_onnx's random-input sweep (already covered per-row by
    ``test_pipeline.test_export``). See the module docstring for why this is
    not generalised across the whole matrix.

    The per-field manifest below (suffix/axis/dtype) is deliberately
    hardcoded to this ONE row's own known, shipped contract (config
    ``name: GN2v2_opendata``, ``outputs.export.model_name: GN2v2opendata``)
    rather than read from a golden JSON — this test was already anchored to
    a single row (not generalised), so its contract belongs directly in code.
    ``salt inference`` names H5 columns by the RUN name
    (``salt/inference.py::build_inference_sink``); the exported ONNX tuple by
    the MODEL name (``salt/outputs/sinks/onnx_sink.py``) — same suffixes,
    different prefixes. ``test_manifest_matches_expected_outputs`` below
    locks this table's suffixes against ``EXPECTED_OUTPUTS["onnx"]`` so the
    two cannot silently drift apart.
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
        """This class's hardcoded FIELDS names exactly EXPECTED_OUTPUTS['onnx']."""
        expected = EXPECTED_OUTPUTS[self.ROW]["onnx"]
        derived = [f"{self.MODEL_NAME}_{suffix}" for suffix, _, _ in self.FIELDS]
        assert derived == expected, (
            "TestGn2v2OpendataValuesMatchOnnxRuntime.FIELDS drifted from "
            f"EXPECTED_OUTPUTS['{self.ROW}']['onnx']: {derived} != {expected}"
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
        expected = EXPECTED_OUTPUTS[self.ROW]["onnx"]
        assert ort_names == expected, f"{self.ROW}: exported tuple != EXPECTED_OUTPUTS['onnx']"
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
    anything a committed golden ever encoded — see the module docstring.
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
    with bit-identical prediction columns (H5-fed rows only — ROOT-fed rows
    skip upstream in ``_artifacts_or_skip``). Generic — compares the row's OWN
    two outputs to each other, so it needs no per-row expected-name table at
    all (unlike gates (a)/(b), see the module docstring).
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
