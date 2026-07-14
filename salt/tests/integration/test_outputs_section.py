"""The outputs:-section + dumb-sink eval H5 — v2 self-consistency gate.

Historical note (DEL-1, plan 45): this file was the FULL-PAYLOAD H5 PARITY GATE
diffing the outputs:-section eval H5 against the legacy ``WriterCallback``
oracle. The legacy writers path is deleted; the byte-for-byte parity was proven
and CLOSED at git tag/hash 29c67a1 (parity-closure doctrine, salt/core/README.md).
What remains are the v2-only checks: the section stack runs end-to-end via the
real CLI, and the section H5's contents/order are asserted from first principles
(config + section manifest), not from a legacy oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from salt.core.graph.spec import Mode
from salt.core.main import CONFIG_DIR, main
from salt.core.outputs.input_copy_writer import InputCopyWriter
from salt.core.outputs.pad_mask_writer import PadMaskWriter
from salt.core.outputs.run_task_output import RunTaskOutput
from salt.core.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2v2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.core.testing.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
CUTOVER34_CFG = CONFIG_DIR / "gn2v2-dummy-cutover34.yaml"
GOLDEN = Path(__file__).resolve().parents[1] / "_fixtures/output_goldens/gn2v2-dummy-cutover34.json"
RUN_NAME = "GN2v2_dummy"
N_TEST = 300


def _golden_task_columns() -> dict[str, list[str]]:
    """Per-stream ordered flat TASK column names from the committed cutover34 golden."""
    golden = json.loads(GOLDEN.read_text())
    per_stream: dict[str, list[str]] = {}
    for col in golden["h5"]["columns"]:
        per_stream.setdefault(col["stream"], []).extend(col["column_names"])
    return per_stream


def _expected_full_columns(src_cols: dict[str, list[str]]) -> dict[str, list[str]]:
    """The FULL ordered per-stream H5 column contract for the Phase-B gate.

    Byte-identical columns = input-copy source columns FIRST (in source-file
    order; an empty golden ``copy_inputs`` means the v1 copy-ALL default, so
    every source field is copied), then task columns in golden order, then the
    trailing pad-mask column. Asserting H5 dtype.names EQUAL this (not merely
    contain it) enforces "no ADDED columns" — a Phase-C label-emission leak, or
    an un-deferred extra leaf from a bad expose merge, cannot slip past.
    """
    h5 = json.loads(GOLDEN.read_text())["h5"]
    tasks: dict[str, list[str]] = {}
    for col in h5["columns"]:
        tasks.setdefault(col["stream"], []).extend(col["column_names"])
    copy_cfg = h5["copy_inputs"]
    pad_streams = set(h5["write_pad_mask"])
    per_stream: dict[str, list[str]] = {}
    for stream, file_fields in src_cols.items():
        cols = list(copy_cfg.get(stream) or file_fields)
        cols += tasks.get(stream, [])
        if stream in pad_streams:
            cols.append("mask")
        per_stream[stream] = cols
    return per_stream

JET_SUFFIXES = ["pb", "pc", "pu"]
ORIGIN_SUFFIXES = [f"p{c}" for c in ORIGIN_CLASSES]
# VertexIndex is NOT deferred — the vertexing get_output fold mints it (H5
# integer column + ONNX int8 leaf). Nothing is deferred in this config.


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("w34_parity")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    h5_path = base / "pp_output_test_ttbar.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


def _overrides(data) -> list[str]:
    return [
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        "--trainer.accelerator=cpu",
        "--trainer.logger=false",
        "--callbacks.progress=null",
    ]


@pytest.fixture(scope="module")
def ckpt(data, tmp_path_factory) -> Path:
    fit_dir = tmp_path_factory.mktemp("w34_fit")
    rc = main([
        "fit",
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        *_overrides(data),
        f"--trainer.default_root_dir={fit_dir}",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.limit_val_batches=2",
        "--trainer.num_sanity_val_steps=0",
        "--trainer.log_every_n_steps=1",
    ])
    assert rc == 0
    ckpts = sorted(fit_dir.rglob("*.ckpt"))
    assert ckpts, f"no checkpoint under {fit_dir}"
    return ckpts[0]


@pytest.fixture(scope="module")
def section_h5(data, ckpt) -> Path:
    """Eval H5 from the outputs:-section + IMPLICIT sinks (via the real CLI).

    Plan 50 Phase B: no ``--callbacks.h5_output`` — the H5 sink is wired by the
    command over the section and writes the default-templated eval H5.
    """
    rc = main([
        "test",
        "--config",
        str(DUMMY_CFG),
        "--config",
        str(CUTOVER34_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        *_overrides(data),
    ])
    assert rc == 0, "salt2 test on the outputs:-section config must run end-to-end"
    evals = sorted(ckpt.parent.glob("*__test_*.h5"))
    assert evals, f"the implicit H5 sink wrote no eval H5 next to {ckpt}"
    return evals[-1]


@pytest.mark.cpu_always
class TestSectionH5SelfConsistency:
    """The outputs:-section eval H5, asserted from first principles.

    The legacy-WriterCallback parity oracle is retired (DEL-1; parity closed at
    29c67a1) — every expectation here derives from the section config + the
    section manifest alone.
    """

    def test_groups_are_the_reader_streams(self, section_h5):
        """The eval H5 carries exactly the reader streams as groups."""
        with h5py.File(section_h5) as f:
            assert set(f.keys()) == {"jets", "tracks"}

    def test_row_count_matches(self, section_h5):
        """The section eval H5 carries all N_TEST rows."""
        with h5py.File(section_h5) as f:
            assert f["jets"].shape[0] == N_TEST

    def test_task_columns_match_golden(self, data, section_h5):
        """The eval H5's columns EQUAL the committed cutover34 golden, per stream, in order.

        Exact-list equality (not membership) is the Phase-B gate: byte-identical
        columns with NO new label columns yet. Any ADDED column — a Phase-C
        label-emission leak, or an un-deferred extra leaf from a bad expose merge
        — fails here. Nothing is deferred in this config: the vertexing
        get_output fold mints the per-token VertexIndex integer column alongside
        the classification probs.
        """
        with h5py.File(data["h5"]) as src:
            src_cols = {"jets": list(src["jets"].dtype.names), "tracks": list(src["tracks"].dtype.names)}
        expected = _expected_full_columns(src_cols)
        with h5py.File(section_h5) as f:
            present = {"jets": list(f["jets"].dtype.names), "tracks": list(f["tracks"].dtype.names)}
            jets, tracks = f["jets"].dtype, f["tracks"].dtype
        assert present.keys() == expected.keys(), (
            f"H5 streams {sorted(present)} != golden streams {sorted(expected)}"
        )
        for stream, cols in expected.items():
            assert present[stream] == cols, (
                f"{stream} columns diverge from golden (added/removed/reordered): "
                f"got {present[stream]}, golden {cols}"
            )
        # dtypes: prob columns float, the bare VertexIndex column integer
        for s in JET_SUFFIXES:
            assert np.issubdtype(jets[f"{RUN_NAME}_{s}"], np.floating)
        for s in ORIGIN_SUFFIXES:
            assert np.issubdtype(tracks[f"{RUN_NAME}_{s}"], np.floating)
        assert np.issubdtype(tracks["VertexIndex"], np.integer)

    def test_probs_are_softmaxed_not_double_converted(self, section_h5):
        """The section prob columns are probabilities (sum ~1) — converted EXACTLY ONCE."""
        with h5py.File(section_h5) as f:
            jets = f["jets"][:]
            tracks = f["tracks"][:]
            valid = ~tracks["mask"]
        jet_cols = [f"{RUN_NAME}_{s}" for s in JET_SUFFIXES]
        prob_sum = sum(jets[c].astype("f8") for c in jet_cols)
        assert np.allclose(prob_sum, 1.0, atol=1e-3)
        origin_cols = [f"{RUN_NAME}_{s}" for s in ORIGIN_SUFFIXES]
        origin_sum = sum(tracks[c].astype("f8") for c in origin_cols)
        assert np.allclose(origin_sum[valid], 1.0, atol=1e-3)
        assert np.allclose(origin_sum[~valid], 0.0, atol=1e-6)


@pytest.mark.cpu_always
class TestColumnOrderDrivenBySection:
    """The H5 column ORDER is enforced by _merge_columns; the SECTION drives task-column order."""

    def test_pad_mask_is_last_column_in_tracks(self, section_h5):
        """The pad-mask 'mask' column is LAST in the tracks group (section order)."""
        with h5py.File(section_h5) as f:
            names = list(f["tracks"].dtype.names)
        assert names[-1] == "mask", f"pad mask must be the LAST tracks column, got order {names}"

    def test_input_copies_precede_task_columns(self, section_h5):
        """The input-copy source columns precede the run-name-prefixed task columns."""
        with h5py.File(section_h5) as f:
            names = list(f["tracks"].dtype.names)
        task_cols = [f"{RUN_NAME}_{s}" for s in ORIGIN_SUFFIXES]
        first_task = min(names.index(c) for c in task_cols)
        # every column before the first task column must be a copied source field
        # (NOT a run-name-prefixed task column and NOT the trailing mask)
        before = names[:first_task]
        assert before, "expected input-copy columns before the task columns"
        assert all(not c.startswith(f"{RUN_NAME}_") and c != "mask" for c in before), (
            f"columns before the task columns must be input copies, got {before}"
        )


class TestSectionOverlayConfigContent:
    """The gn2v2-dummy-cutover34.yaml overlay wires the full-family outputs: section.

    Plan 50 Phase B: the overlay replaces the base's mode-split jets_out/origin_out
    writers with ONE all-modes RunTaskOutput, and declares NO callbacks: sinks —
    the command wires the implicit H5 + ONNX sinks.
    """

    def test_config_content(self):
        cfg = yaml.safe_load(CUTOVER34_CFG.read_text())
        # track_vertexing is un-deferred (base defers via expose: [fit, val]) so
        # its get_output fold mints the VertexIndex eval/ONNX leaves.
        mods = cfg["model"]["modules"]
        assert mods["track_vertexing"]["init_args"]["expose"] is None
        # no legacy writers: block
        assert "writers" not in cfg
        # the outputs: section: null the base's mode-split writers, add run_tasks
        section = cfg["outputs"]
        assert section["jets_out"] is None
        assert section["origin_out"] is None
        assert section["run_tasks"]["class_path"] == "salt.core.outputs.RunTaskOutput"
        # track_vertexing JOINS the orchestrated tasks
        assert section["run_tasks"]["init_args"]["tasks"] == [
            "jets_classification",
            "track_origin",
            "track_vertexing",
        ]
        # the sinks are IMPLICIT — the overlay declares no h5_output/onnx_export
        assert "callbacks" not in cfg


@pytest.mark.cpu_always
class TestSectionWriterUnits:
    """Unit-level checks of the section writers' declare_io + manifest + ordering."""

    def _bound_run_task(self):
        # build the gn2v2 model modules (no file I/O for declare_io / manifest)
        import tempfile

        nd = Path(tempfile.mkdtemp()) / "nd.yaml"
        cd = nd.parent / "cd.yaml"
        write_parity_norm_dict(nd, cd)
        modules = build_gn2v2_modules(nd)
        # orchestrate the FULL gn2v2 family (incl track_vertexing)
        rt = RunTaskOutput(tasks=["jets_classification", "track_origin", "track_vertexing"])
        rt.name = "run_tasks"
        rt.bind_model_modules(modules)
        return rt

    def test_run_task_requires_preds_and_pad_mask(self):
        """RunTaskOutput.declare_io requires each task's preds + the seq head's pad mask."""
        from salt.core.graph.spec import flatten_spec

        rt = self._bound_run_task()
        req = flatten_spec(rt.declare_io(Mode.TEST).requires)
        assert "preds.jets.jets_classification" in req
        assert "preds.tracks.track_origin" in req
        # the vertexing head's raw edge-score leaf
        assert "preds.tracks.track_vertexing" in req
        # the seq head (track_origin) AND the vertexing head declare masks.tracks
        # via output_time_requires
        assert "masks.tracks" in req

    def test_run_task_produces_per_field_leaves_test(self):
        """In TEST, RunTaskOutput produces one outputs.*.<col> leaf PER class column."""
        from salt.core.graph.spec import flatten_spec

        rt = self._bound_run_task()
        prod = set(flatten_spec(rt.declare_io(Mode.TEST).produces))
        # global head: one leaf per jet class suffix
        for s in JET_SUFFIXES:
            assert f"outputs.jets.jets_classification.{s}" in prod
        # seq head: one leaf per origin class suffix (H5 probs)
        for s in ORIGIN_SUFFIXES:
            assert f"outputs.tracks.track_origin.{s}" in prod
        # vertexing head -> one i8 VertexIndex per-token leaf (H5)
        assert "outputs.tracks.track_vertexing.VertexIndex" in prod

    def test_run_task_produces_argmax_leaf_onnx(self):
        """In ONNX, the seq head produces a single argmax-index leaf (TrackOrigin)."""
        from salt.core.graph.spec import flatten_spec

        rt = self._bound_run_task()
        prod = set(flatten_spec(rt.declare_io(Mode.ONNX).produces))
        # global head: per-class scalar leaves
        for s in JET_SUFFIXES:
            assert f"outputs.jets.jets_classification.{s}" in prod
        # seq head ONNX: a single argmax index leaf under the pascal-case task name
        assert "outputs.tracks.track_origin.TrackOrigin" in prod
        # NOT the per-class probs leaves in ONNX (mode-keyed write-once)
        assert "outputs.tracks.track_origin.pPrimary" not in prod
        # vertexing head ONNX -> a single VertexIndex union-find leaf
        assert "outputs.tracks.track_vertexing.VertexIndex" in prod

    def test_manifest_fields_order_is_task_then_field(self):
        """manifest_fields orders fields task-then-field (the column-order authority)."""
        rt = self._bound_run_task()
        fields = rt.manifest_fields(Mode.TEST)
        cols = [f.h5_name for _, f in fields]
        # jets columns first (task order), then tracks origin columns, then the
        # vertexing VertexIndex column (task-then-field order)
        n_j, n_o = len(JET_SUFFIXES), len(ORIGIN_SUFFIXES)
        assert cols[:n_j] == JET_SUFFIXES
        assert cols[n_j : n_j + n_o] == ORIGIN_SUFFIXES
        assert cols[n_j + n_o :] == ["VertexIndex"]

    def test_input_copy_is_manifest_only(self):
        """InputCopyWriter is manifest-only (no graph leaf — the sink re-reads copies)."""
        icw = InputCopyWriter(streams=["jets", "tracks"])
        assert icw.is_manifest_only() is True
        assert icw.copy_spec()["streams"] == ["jets", "tracks"]

    def test_pad_mask_produces_mask_leaf(self):
        """PadMaskWriter produces outputs.<stream>.mask from masks.<stream>."""
        from salt.core.graph.spec import flatten_spec

        pmw = PadMaskWriter(streams=["tracks"])
        io = pmw.declare_io(Mode.TEST)
        assert "outputs.tracks.mask" in flatten_spec(io.produces)
        assert "masks.tracks" in flatten_spec(io.requires)

    def test_run_task_rejects_empty_and_duplicate(self):
        """RunTaskOutput rejects an empty tasks list and duplicate task names."""
        from salt.core.graph.errors import ConfigError

        with pytest.raises(ConfigError):
            RunTaskOutput(tasks=[])
        with pytest.raises(ConfigError):
            RunTaskOutput(tasks=["a", "a"])


# ONNX contract: the dumb OnnxExportSink names the section's get_output leaves.
# PROVES the LOCKED no-double-split decision: get_output squeezes the global
# per-class scalars, so the dumb sink ONLY names them. Expectations are
# hand-pinned literals (the /tmp golden apparatus is retired — closure at the
# v1 pin, see salt/core/README.md).

# the FULL gn2v2 contract — pb/pc/pu globals + the TrackOrigin per-token
# argmax + the VertexIndex per-token union-find (both int8).
SECTION_ONNX_NAMES = ["GN2v2_pb", "GN2v2_pc", "GN2v2_pu", "GN2v2_TrackOrigin", "GN2v2_VertexIndex"]
SECTION_ONNX_DTYPES = ["float32", "float32", "float32", "int8", "int8"]


@pytest.mark.cpu_always
class TestSectionOnnxContract:
    """The dumb OnnxExportSink names the section's get_output leaves (pinned contract)."""

    def _section_export(self, tmp_path):
        import torch

        from salt.core.nn import bind_all, resolve_bind_schema
        from salt.core.onnx import (
            ExportConfig,
            ExportInput,
            compile_onnx_plan,
            export_graph,
            resolve_export_config,
        )
        from salt.core.outputs import OnnxExportSink
        from salt.tests._fixtures.gn2v2_fixture import (
            JET_VARIABLES,
            TRACK_VARIABLES,
        )

        variables = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        export_cfg = ExportConfig(
            model_name="GN2v2",
            inputs=[
                ExportInput(port="inputs.jets", name="jet_features"),
                ExportInput(
                    port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
                ),
            ],
        )
        tmp_path.mkdir(parents=True, exist_ok=True)
        write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
        # deterministic non-trivial weights; the tests assert contract, not values
        torch.manual_seed(42)
        modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
        # the section: RunTaskOutput over the FULL gn2v2 family (classification
        # + vertexing) + the DUMB OnnxExportSink, both folded into the export
        # module dict.
        run_tasks = RunTaskOutput(
            tasks=["jets_classification", "track_origin", "track_vertexing"]
        )
        run_tasks.name = "run_tasks"
        run_tasks.bind_model_modules(modules)
        sink = OnnxExportSink()
        sink.name = "onnx_export"
        section = {"run_tasks": run_tasks}
        sink.bind_output_section(section)
        modules["run_tasks"] = run_tasks
        modules["onnx_export"] = sink
        resolved = resolve_export_config(export_cfg, "GN2_v2")
        plan = compile_onnx_plan(modules, resolved, variables)
        bind_all(modules, resolve_bind_schema([plan]))
        modules["norm"].materialise()
        return export_graph(
            modules, export_cfg, variables, tmp_path / "section.onnx", outputs=[], run_name="GN2_v2"
        )

    def test_onnx_contract_names_dtypes_axes_order(self, tmp_path):
        """The dumb-section ONNX names/dtypes/axes/ORDER == the pinned contract."""
        adapter = self._section_export(tmp_path).adapter
        assert adapter.output_names == SECTION_ONNX_NAMES, (
            f"section ONNX names {adapter.output_names} != pinned {SECTION_ONNX_NAMES}"
        )
        assert adapter.output_dtypes == SECTION_ONNX_DTYPES
        # the per-token int8 leaves (TrackOrigin argmax + VertexIndex union-find)
        # carry the n_tracks dynamic axis
        for per_token in ("GN2v2_TrackOrigin", "GN2v2_VertexIndex"):
            assert adapter.dynamic_axes.get(per_token) == {0: "n_tracks"}, (
                f"{per_token} dynamic axis {adapter.dynamic_axes.get(per_token)}"
            )
        # the global scalars carry NO dynamic axis (the no-double-split scalars)
        for g in ("GN2v2_pb", "GN2v2_pc", "GN2v2_pu"):
            assert g not in adapter.dynamic_axes

    def test_onnx_session_runs_no_double_split(self, tmp_path):
        """The exported ONNX adapter runs — 5 outputs, no re-split."""
        import torch

        result = self._section_export(tmp_path)
        adapter = result.adapter
        for length in (0, 5):
            example = adapter.example_inputs(sequence_length=length)
            with torch.no_grad():
                out = adapter(*example)
            assert len(out) == 5  # pb, pc, pu, TrackOrigin, VertexIndex

    def test_onnx_output_ranks_match_global_vs_per_token(self, tmp_path):
        """The exported ONNX graph's output RANKS: globals rank-0 [], per-token rank-1."""
        import onnx

        onnx_path = tmp_path / "section.onnx"
        self._section_export(tmp_path)
        model = onnx.load(str(onnx_path))
        ranks = {
            o.name: len(o.type.tensor_type.shape.dim) for o in model.graph.output
        }
        for g in ("GN2v2_pb", "GN2v2_pc", "GN2v2_pu"):
            assert ranks[g] == 0, f"{g} should be a rank-0 scalar, got rank {ranks[g]}"
        for per_token in ("GN2v2_TrackOrigin", "GN2v2_VertexIndex"):
            assert ranks[per_token] == 1, (
                f"{per_token} should be a rank-1 per-token vector, got rank {ranks[per_token]}"
            )
