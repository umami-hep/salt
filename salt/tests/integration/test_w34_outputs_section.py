"""PLAN 34 W34.2 FULL-PAYLOAD H5 PARITY GATE — outputs:-section + dumb sinks vs WriterCallback.

The W34.2 gate (plan 34 §6): on the SAME synthetic GN2-like model + the SAME
source H5 + the SAME trained checkpoint, run BOTH eval paths and assert the eval
H5 files match byte-for-byte at SEMANTIC parity (ints/bools EXACT, floats <=1e-6),
including column NAMES + ORDER + DTYPES + per-token pad re-expansion:

- **(a) the M4.5 ``WriterCallback``** (the v1-parity ORACLE, ``salt/core/writers``)
  via the proven ``salt2 test`` CLI surface (the gn2v2-dummy default), and
- **(b) the PLAN 34 ``outputs:`` section + dumb sinks** — `RunTaskOutput` (calling
  each task's ``get_output``) + `InputCopyWriter` + `PadMaskWriter`, dumped by the
  DUMB `H5OutputSink`, driven by the ``gn2v2-dummy-cutover34.yaml`` config through
  the real ``salt2 test`` CLI.

The plan-34 path REPLACES the plan-29/31 standalone conversion producers with the
get_output()-on-task fold, so the H5 columns come from ``RunTaskOutput`` reassembling
each task's ``get_output`` fields — proven here to be byte-identical to the legacy
``TaskWriter`` columns. Vertexing (``track_vertexing`` -> bare ``VertexIndex`` i8)
is a DEFERRED family (W34.3); EXCLUDED from the comparison and recorded.

If parity cannot be reached the assertion reports the exact column with expected
vs got — never weaken the tolerance to pass.
"""

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
DEFERRED_COLUMNS = {"tracks": ["VertexIndex"]}


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
        """Per-column array equality + EXACT column NAMES + ORDER (deferred excluded).

        This is the W34.2 full-payload gate: input-copies + classification/seq
        columns + pad-mask, with the column NAMES + ORDER + DTYPES + per-token pad
        re-expansion all matching the legacy WriterCallback byte schema.
        """
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


class TestW34ColumnOrderDrivenBySection:
    """The H5 column ORDER is driven by the outputs: SECTION field order, NOT topo order.

    Plan §4 W34.2 / §7 risk 4: the section writers in declaration order
    (InputCopyWriter -> RunTaskOutput task fields -> PadMaskWriter) drive the H5
    column order. A test that fails if executor topo order reshuffled the columns:
    the v1 contract is inputs_copy FIRST, then task columns, then the pad mask LAST.
    """

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
        assert mods["track_vertexing"]["init_args"]["expose"] == ["fit", "val"]
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
        assert section["run_tasks"]["init_args"]["tasks"] == [
            "jets_classification",
            "track_origin",
        ]
        assert section["pad_mask"]["class_path"] == "salt.core.outputs.PadMaskWriter"
        # the DUMB H5 sink has no init args (it dumps the section's outputs.* leaves)
        assert cfg["callbacks"]["h5_output"]["class_path"] == "salt.core.outputs.H5OutputSink"
        assert "init_args" not in cfg["callbacks"]["h5_output"]


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
        rt = RunTaskOutput(tasks=["jets_classification", "track_origin"])
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
        # the seq head (track_origin) declares its stream pad mask via
        # output_time_requires
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

    def test_manifest_fields_order_is_task_then_field(self):
        """manifest_fields orders fields task-then-field (the column-order authority)."""
        rt = self._bound_run_task()
        fields = rt.manifest_fields(Mode.TEST)
        cols = [f.h5_name for _, f in fields]
        # jets columns first (task order), then tracks columns
        assert cols[: len(JET_SUFFIXES)] == JET_SUFFIXES
        assert cols[len(JET_SUFFIXES) :] == ORIGIN_SUFFIXES

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


# ---------------------------------------------------------------------------
# ONNX parity: the dumb OnnxExportSink names the section's get_output leaves
# byte-identically vs the /tmp/w4_oracle golden (classification subset — vtx is
# W34.3). PROVES the LOCKED no-double-split decision: get_output squeezes the
# global per-class scalars (W34.1), so the dumb sink ONLY names them.
# ---------------------------------------------------------------------------

W4_ORACLE = Path("/tmp/w4_oracle")


@pytest.mark.cpu_always
class TestW34SectionOnnxParity:
    """The dumb OnnxExportSink names the section's get_output leaves vs /tmp/w4_oracle.

    W34.2 scope: classification family only (pb/pc/pu globals + TrackOrigin argmax)
    — vertexing's VertexIndex ONNX is W34.3. The classification SUBSET of the
    /tmp/w4_oracle/gn2v2.json golden (names/dtypes/axes/ORDER) must match the
    dumb-section export byte-for-byte. The no-double-split decision is proven: the
    global per-class scalars are NAMED directly (no torch.split).
    """

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
        # the plan-34 section: RunTaskOutput (classification only — vtx is W34.3) +
        # the DUMB OnnxExportSink, both folded into the export module dict.
        run_tasks = RunTaskOutput(tasks=["jets_classification", "track_origin"])
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
    def test_onnx_classification_subset_matches_golden(self, tmp_path):
        """The dumb-section ONNX names/dtypes/axes/ORDER == the golden's classification subset."""
        import json

        golden = json.loads((W4_ORACLE / "gn2v2.json").read_text())
        adapter = self._section_export(tmp_path).adapter
        # the W34.2 classification subset of the golden: pb/pc/pu globals + the
        # TrackOrigin per-token argmax (VertexIndex is W34.3).
        want_names = ["GN2v2_pb", "GN2v2_pc", "GN2v2_pu", "GN2v2_TrackOrigin"]
        want_dtypes = ["float32", "float32", "float32", "int8"]
        # the golden lists these names in this exact order (subset of golden)
        assert golden["output_names"][:4] == want_names
        assert golden["output_dtypes"][:4] == want_dtypes
        # the dumb-section export reproduces the subset byte-for-byte (ORDERED)
        assert adapter.output_names == want_names, (
            f"section ONNX names {adapter.output_names} != golden subset {want_names}"
        )
        assert adapter.output_dtypes == want_dtypes
        # the TrackOrigin dynamic axis matches the golden (n_tracks on axis 0).
        # json normalises int axis keys to strings, so round-trip the adapter's for
        # an apples-to-apples compare.
        adapter_axes = json.loads(json.dumps(adapter.dynamic_axes))
        assert adapter_axes.get("GN2v2_TrackOrigin") == golden["dynamic_axes"]["GN2v2_TrackOrigin"]
        # the global scalars carry NO dynamic axis (the no-double-split scalars)
        for g in ("GN2v2_pb", "GN2v2_pc", "GN2v2_pu"):
            assert g not in adapter.dynamic_axes

    @pytest.mark.skipif(
        not (W4_ORACLE / "gn2v2.json").is_file(),
        reason="W4 oracle golden /tmp/w4_oracle/gn2v2.json not present",
    )
    def test_onnx_session_runs_no_double_split(self, tmp_path):
        """The exported ONNX session runs (onnxruntime) — 4 classification outputs, no re-split.

        Proves the LOCKED no-double-split decision end-to-end: the dumb sink names
        the already-scalar per-class get_output values; if it had re-split a
        pre-split scalar the trace/run would be malformed.
        """
        import torch

        result = self._section_export(tmp_path)
        adapter = result.adapter
        for length in (0, 5):
            example = adapter.example_inputs(sequence_length=length)
            with torch.no_grad():
                out = adapter(*example)
            assert len(out) == 4  # pb, pc, pu, TrackOrigin (no VertexIndex in W34.2)
