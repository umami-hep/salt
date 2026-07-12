"""PLAN 34 W34.2 FULL-PAYLOAD H5 PARITY GATE — outputs:-section + dumb sinks vs WriterCallback."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from salt.core.graph.spec import Mode
from salt.core.main import CONFIG_DIR, main
from salt.core.outputs.writers import InputCopyWriter, PadMaskWriter, RunTaskOutput
from salt.core.schema import dump_schema, save_schema
from salt.tests._fixtures.gn2_fixture import write_parity_norm_dict
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
CUTOVER34_CFG = CONFIG_DIR / "gn2v2-dummy-cutover34.yaml"
RUN_NAME = "GN2v2_dummy"
N_TEST = 300
_FLOAT_TOL = 1e-6

JET_SUFFIXES = ["pb", "pc", "pu"]
ORIGIN_SUFFIXES = [f"p{c}" for c in ORIGIN_CLASSES]
# plan 34 W34.3: VertexIndex is NO LONGER deferred — the vertexing get_output fold
# mints it (H5 i8 column + ONNX int8 leaf), proven byte-for-byte here. Nothing is
# deferred in the gn2v2-dummy cutover any more.
DEFERRED_COLUMNS: dict[str, list[str]] = {}


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
def oracle_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from the M4.5 ``WriterCallback`` (the v1-parity ORACLE)."""
    out = tmp_path_factory.mktemp("w34_oracle") / "oracle.h5"
    rc = main([
        "test",
        "--config",
        str(DUMMY_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        f"--data.num_test={N_TEST}",
        f"--trainer.default_root_dir={data['dir']}",
        f"--writers.output={out}",
        *_overrides(data),
    ])
    assert rc == 0
    assert out.exists()
    return out


@pytest.fixture(scope="module")
def section_h5(data, ckpt, tmp_path_factory) -> Path:
    """Eval H5 from the PLAN 34 outputs:-section + dumb sinks (via the real CLI)."""
    out = tmp_path_factory.mktemp("w34_section") / "section.h5"
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
        f"--callbacks.h5_output.init_args.output={out}",
        *_overrides(data),
    ])
    assert rc == 0, "salt2 test on the cutover34 outputs:-section config must run end-to-end"
    assert out.exists()
    return out


def _drop_deferred(names: list[str], stream: str) -> list[str]:
    deferred = set(DEFERRED_COLUMNS.get(stream, ()))
    return [n for n in names if n not in deferred]


def _compare_column(group: str, col: str, want: np.ndarray, got: np.ndarray) -> str | None:
    if want.dtype != got.dtype:
        return f"{group}.{col}: dtype {want.dtype} (oracle) != {got.dtype} (section)"
    if want.shape != got.shape:
        return f"{group}.{col}: shape {want.shape} (oracle) != {got.shape} (section)"
    if np.issubdtype(want.dtype, np.floating):
        if not np.allclose(want, got, rtol=0.0, atol=_FLOAT_TOL, equal_nan=True):
            bad = int(np.argmax(np.abs(want.ravel() - got.ravel())))
            return (
                f"{group}.{col}: float values differ beyond atol={_FLOAT_TOL} — first worst "
                f"at flat idx {bad}: oracle={want.ravel()[bad]!r} section={got.ravel()[bad]!r}"
            )
    elif not np.array_equal(want, got):
        bad = int(np.argmax(want.ravel() != got.ravel()))
        return (
            f"{group}.{col}: int/bool values differ (exact required) — first at flat idx "
            f"{bad}: oracle={want.ravel()[bad]!r} section={got.ravel()[bad]!r}"
        )
    return None


@pytest.mark.cpu_always
class TestW34SectionH5Parity:
    """The plan-34 outputs:-section eval H5 == the legacy WriterCallback oracle."""

    def test_deferred_columns_present_in_oracle(self, oracle_h5):
        """Sanity: the DEFERRED (W34.3 vertexing) columns DO exist in the oracle."""
        with h5py.File(oracle_h5) as f:
            for stream, cols in DEFERRED_COLUMNS.items():
                names = set(f[stream].dtype.names)
                missing = [c for c in cols if c not in names]
                assert not missing, f"deferred columns {missing} absent from oracle {stream!r}"

    def test_groups_match(self, oracle_h5, section_h5):
        """Both eval paths write the same H5 groups."""
        with h5py.File(oracle_h5) as a, h5py.File(section_h5) as b:
            assert set(a.keys()) == set(b.keys())

    def test_row_count_matches(self, section_h5):
        """The section eval H5 carries all N_TEST rows."""
        with h5py.File(section_h5) as f:
            assert f["jets"].shape[0] == N_TEST

    def test_full_payload_h5_parity(self, oracle_h5, section_h5):
        """Per-column array equality + EXACT column NAMES + ORDER (deferred excluded)."""
        diffs: list[str] = []
        with h5py.File(oracle_h5) as a, h5py.File(section_h5) as b:
            for group in a:
                oracle = a[group][:]
                section = b[group][:]
                want_cols = _drop_deferred(list(oracle.dtype.names), group)
                got_cols = list(section.dtype.names)
                # the section file must carry EXACTLY the compared (non-deferred)
                # columns, in the SAME ORDER (column order IS part of the schema —
                # the v1 inputs_copy -> tasks -> pad_mask layout, section-order-driven)
                if want_cols != got_cols:
                    diffs.append(
                        f"{group}: column set/order mismatch\n  oracle (minus deferred): "
                        f"{want_cols}\n  section                : {got_cols}"
                    )
                    continue
                if oracle.shape != section.shape:
                    diffs.append(
                        f"{group}: shape {oracle.shape} (oracle) != {section.shape} (section)"
                    )
                    continue
                diffs.extend(
                    msg
                    for col in want_cols
                    if (msg := _compare_column(group, col, oracle[col], section[col])) is not None
                )
        assert not diffs, "W34.2 FULL-PAYLOAD H5 PARITY FAILED:\n" + "\n".join(diffs)

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
class TestW34ColumnOrderDrivenBySection:
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


class TestW34CutoverConfigContent:
    """The gn2v2-dummy-cutover34.yaml wires the outputs: section + dumb sinks."""

    def test_config_content(self):
        cfg = yaml.safe_load(CUTOVER34_CFG.read_text())
        # the plan-29/31 producers are nulled (the get_output fold replaces them)
        mods = cfg["model"]["modules"]
        assert mods["jet_probs"] is None
        assert mods["track_origin_probs"] is None
        assert mods["track_origin_index"] is None
        assert mods["track_vertex_index"] is None
        # plan 34 W34.3: track_vertexing is NO LONGER opted out of TEST/ONNX (no
        # expose key) — the vtx get_output fold mints its eval/ONNX leaves.
        assert "track_vertexing" not in mods or "expose" not in mods.get(
            "track_vertexing", {}
        ).get("init_args", {})
        # the M4.5 writers are nulled
        assert cfg["writers"]["modules"] == {
            "inputs_copy": None,
            "tasks": None,
            "pad_mask": None,
        }
        # the outputs: section in EXACT v1 column order
        section = cfg["outputs"]
        assert list(section.keys()) == ["inputs_copy", "run_tasks", "pad_mask"]
        assert section["inputs_copy"]["class_path"] == "salt.core.outputs.InputCopyWriter"
        assert section["run_tasks"]["class_path"] == "salt.core.outputs.RunTaskOutput"
        # plan 34 W34.3: track_vertexing JOINS the orchestrated tasks
        assert section["run_tasks"]["init_args"]["tasks"] == [
            "jets_classification",
            "track_origin",
            "track_vertexing",
        ]
        assert section["pad_mask"]["class_path"] == "salt.core.outputs.PadMaskWriter"
        # the DUMB sinks have no init args (they dump the section's outputs.* leaves)
        assert cfg["callbacks"]["h5_output"]["class_path"] == "salt.core.outputs.H5OutputSink"
        assert "init_args" not in cfg["callbacks"]["h5_output"]
        # plan 34 W34.3: the DUMB OnnxExportSink is re-enabled (no init args)
        assert cfg["callbacks"]["onnx_export"]["class_path"] == "salt.core.outputs.OnnxExportSink"
        assert "init_args" not in cfg["callbacks"]["onnx_export"]


@pytest.mark.cpu_always
class TestW34SectionWriterUnits:
    """Unit-level checks of the section writers' declare_io + manifest + ordering."""

    def _bound_run_task(self):
        from salt.tests._fixtures.gn2_fixture import write_parity_norm_dict as _wnd  # noqa: F401

        # build the gn2v2 model modules (no file I/O for declare_io / manifest)
        import tempfile

        nd = Path(tempfile.mkdtemp()) / "nd.yaml"
        cd = nd.parent / "cd.yaml"
        write_parity_norm_dict(nd, cd)
        modules = build_gn2v2_modules(nd)
        # plan 34 W34.3: orchestrate the FULL gn2v2 family (incl track_vertexing)
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
        # plan 34 W34.3: the vertexing head's raw edge-score leaf
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
        # plan 34 W34.3: vertexing head -> one i8 VertexIndex per-token leaf (H5)
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
        # plan 34 W34.3: vertexing head ONNX -> a single VertexIndex union-find leaf
        assert "outputs.tracks.track_vertexing.VertexIndex" in prod

    def test_manifest_fields_order_is_task_then_field(self):
        """manifest_fields orders fields task-then-field (the column-order authority)."""
        rt = self._bound_run_task()
        fields = rt.manifest_fields(Mode.TEST)
        cols = [f.h5_name for _, f in fields]
        # jets columns first (task order), then tracks origin columns, then the
        # vertexing VertexIndex column (plan 34 W34.3 task-then-field order)
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


# ONNX parity: the dumb OnnxExportSink names the section's get_output leaves
# byte-identically vs the /tmp/w4_oracle golden (classification subset — vtx is
# W34.3). PROVES the LOCKED no-double-split decision: get_output squeezes the
# global per-class scalars (W34.1), so the dumb sink ONLY names them.

W4_ORACLE = Path("/tmp/w4_oracle")


@pytest.mark.cpu_always
class TestW34SectionOnnxParity:
    """The dumb OnnxExportSink names the section's get_output leaves vs /tmp/w4_oracle."""

    def _section_export(self, tmp_path):
        from torch import nn

        from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
        from salt.core.onnx import (
            ExportConfig,
            ExportInput,
            compile_onnx_plan,
            export_graph,
            resolve_export_config,
        )
        from salt.core.outputs import OnnxExportSink
        from salt.tests._fixtures.gn2_fixture import (
            JET_VARIABLES,
            TRACK_VARIABLES,
            build_test_gn2,
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
        v1 = build_test_gn2(tmp_path)
        modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
        # the plan-34 W34.3 section: RunTaskOutput over the FULL gn2v2 family
        # (classification + vertexing) + the DUMB OnnxExportSink, both folded into
        # the export module dict.
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
        nn.ModuleDict(
            {k: v for k, v in modules.items() if isinstance(v, nn.Module)}
        ).load_state_dict(map_v1_state_dict(v1.state_dict(), modules), strict=False)
        return export_graph(
            modules, export_cfg, variables, tmp_path / "section.onnx", outputs=[], run_name="GN2_v2"
        )

    @pytest.mark.skipif(
        not (W4_ORACLE / "gn2v2.json").is_file(),
        reason="W4 oracle golden /tmp/w4_oracle/gn2v2.json not present",
    )
    def test_onnx_full_golden_matches_incl_vertex_index(self, tmp_path):
        """The dumb-section ONNX names/dtypes/axes/ORDER == the FULL golden (incl VertexIndex)."""
        import json

        golden = json.loads((W4_ORACLE / "gn2v2.json").read_text())
        adapter = self._section_export(tmp_path).adapter
        # plan 34 W34.3: the FULL gn2v2 golden — pb/pc/pu globals + the TrackOrigin
        # per-token argmax + the VertexIndex per-token union-find (all int8).
        want_names = ["GN2v2_pb", "GN2v2_pc", "GN2v2_pu", "GN2v2_TrackOrigin", "GN2v2_VertexIndex"]
        want_dtypes = ["float32", "float32", "float32", "int8", "int8"]
        # the golden lists these names in this exact order (FULL golden)
        assert golden["output_names"] == want_names
        assert golden["output_dtypes"] == want_dtypes
        # the dumb-section export reproduces the FULL golden byte-for-byte (ORDERED)
        assert adapter.output_names == want_names, (
            f"section ONNX names {adapter.output_names} != golden {want_names}"
        )
        assert adapter.output_dtypes == want_dtypes
        # the per-token int8 leaves (TrackOrigin argmax + VertexIndex union-find)
        # carry the n_tracks dynamic axis; json normalises int axis keys to strings.
        adapter_axes = json.loads(json.dumps(adapter.dynamic_axes))
        for per_token in ("GN2v2_TrackOrigin", "GN2v2_VertexIndex"):
            assert adapter_axes.get(per_token) == golden["dynamic_axes"][per_token], (
                f"{per_token} dynamic axis {adapter_axes.get(per_token)} != golden "
                f"{golden['dynamic_axes'][per_token]}"
            )
        # the global scalars carry NO dynamic axis (the no-double-split scalars)
        for g in ("GN2v2_pb", "GN2v2_pc", "GN2v2_pu"):
            assert g not in adapter.dynamic_axes

    @pytest.mark.skipif(
        not (W4_ORACLE / "gn2v2.json").is_file(),
        reason="W4 oracle golden /tmp/w4_oracle/gn2v2.json not present",
    )
    def test_onnx_session_runs_no_double_split(self, tmp_path):
        """The exported ONNX session runs (onnxruntime) — 5 outputs, no re-split."""
        import torch

        result = self._section_export(tmp_path)
        adapter = result.adapter
        for length in (0, 5):
            example = adapter.example_inputs(sequence_length=length)
            with torch.no_grad():
                out = adapter(*example)
            assert len(out) == 5  # pb, pc, pu, TrackOrigin, VertexIndex (W34.3)

    @pytest.mark.skipif(
        not (W4_ORACLE / "gn2v2.json").is_file(),
        reason="W4 oracle golden /tmp/w4_oracle/gn2v2.json not present",
    )
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
