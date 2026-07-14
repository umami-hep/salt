"""Unit tests for ``salt2 inference`` (plan 50 Phase D): parsing/dispatch, the
export-selection sink, label-demand-free ONNX plan compilation, and the
unlabelled-file dataset path.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.lib.recfunctions import repack_fields

from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import Mode
from salt.core.inference import (
    INFERENCE_OUTPUT,
    _parse_args,
    build_inference_sink,
    inference_demand,
)
from salt.core.main import CONFIG_DIR
from salt.core.onnx.config import ExportConfig, ExportInput
from salt.core.outputs import H5OutputSink, OnnxExportSink, PadMaskWriter
from salt.core.outputs.input_copy_writer import InputCopyWriter
from salt.core.outputs.run_task_output import RunTaskOutput
from salt.tests._fixtures.gn2v2_fixture import build_gn2v2_modules, write_parity_norm_dict

LABEL_FIELDS = {
    "flavour_label",
    "HadronConeExclTruthLabelID",
    "HadronGhostInitialTruthLabelPdgId",
    "ftagTruthOriginLabel",
    "ftagTruthTypeLabel",
    "ftagTruthVertexIndex",
    "ftagTruthParentBarcode",
}


class TestParseArgs:
    """The salt2 inference argparse surface."""

    def test_required_flags(self, capsys):
        """--ckpt_path and --data.test_file are both required."""
        with pytest.raises(SystemExit):
            _parse_args(["--ckpt_path", "x.ckpt"])
        with pytest.raises(SystemExit):
            _parse_args(["--data.test_file", "x.h5"])
        capsys.readouterr()

    def test_full_surface(self):
        """Config stacking, --set repeats, --output, --batch_size all land."""
        parsed = _parse_args([
            "--ckpt_path",
            "run/ckpt.ckpt",
            "-c",
            "a.yaml",
            "-c",
            "b.yaml",
            "--data.test_file",
            "test.h5",
            "--output",
            "out.h5",
            "--batch_size",
            "128",
            "--set",
            "data.num_test=10",
            "--set",
            "name=foo",
        ])
        assert parsed.ckpt_path == Path("run/ckpt.ckpt")
        assert parsed.config == [Path("a.yaml"), Path("b.yaml")]
        assert parsed.test_file == Path("test.h5")
        assert parsed.output == Path("out.h5")
        assert parsed.batch_size == 128
        assert parsed.set_overrides == ["data.num_test=10", "name=foo"]

    def test_defaults(self):
        """Output/config/batch_size default to None (template/inferred/datamodule)."""
        parsed = _parse_args(["--ckpt_path", "c.ckpt", "--data.test_file", "t.h5"])
        assert parsed.output is None
        assert parsed.config is None
        assert parsed.batch_size is None
        assert parsed.set_overrides == []


class TestDispatch:
    """``salt2 inference ...`` dispatches to salt.core.inference.main."""

    def test_main_dispatches(self, monkeypatch):
        """The main() entry hands argv (minus the command) to inference.main."""
        import salt.core.inference as inference_mod
        from salt.core.main import main

        seen: dict[str, list[str]] = {}

        def fake_main(argv):
            seen["argv"] = list(argv)
            return 0

        monkeypatch.setattr(inference_mod, "main", fake_main)
        rc = main(["inference", "--ckpt_path", "x.ckpt", "--data.test_file", "y.h5"])
        assert rc == 0
        assert seen["argv"] == ["--ckpt_path", "x.ckpt", "--data.test_file", "y.h5"]


class TestInferenceDemand:
    """The dataset demand of an inference run is the export input contract, label-free."""

    def test_ports_masks_and_rows_only(self):
        """Positional ports + sequence masks + meta.rows; alias entries consume nothing."""
        export = ExportConfig(
            model_name="M",
            inputs=[
                ExportInput(port="inputs.jets", name="jet_features"),
                ExportInput(port="inputs.tracks", name="track_features", sequence=True),
                ExportInput(port="inputs.global", alias="inputs.jets"),
            ],
        )
        demand = inference_demand(export)
        assert demand == ["inputs.jets", "inputs.tracks", "masks.tracks", "meta.rows"]
        assert not any(key.startswith("labels.") for key in demand)


def _section(tmp_path, copy_modes=None, mask_modes=None) -> dict:
    """A gn2v2 outputs: section (full task family) over real bound-free modules."""  # noqa: DOC201
    nd = tmp_path / "nd.yaml"
    write_parity_norm_dict(nd, tmp_path / "cd.yaml")
    modules = build_gn2v2_modules(nd)
    rt = RunTaskOutput(tasks=["jets_classification", "track_origin", "track_vertexing"])
    rt.name = "run_tasks"
    rt.bind_model_modules(modules)
    icw = InputCopyWriter(streams=["jets", "tracks"], modes=copy_modes)
    pmw = PadMaskWriter(streams=["tracks"], modes=mask_modes)
    return {"inputs_copy": icw, "run_tasks": rt, "pad_mask": pmw}


class TestExportSelectionSink:
    """build_inference_sink: the H5 columns ARE the export selection, 1:1 with the ONNX tuple."""

    def test_columns_are_one_to_one_with_onnx_leaves(self, tmp_path):
        """Same leaf keys, single suffixes == resolved ONNX names, same count/order rule."""
        section = _section(tmp_path)
        sink = build_inference_sink(section)
        onnx = OnnxExportSink()
        onnx.bind_output_section(section)
        onnx.model_name = "M"
        columns = sink._resolve_columns("run")  # noqa: SLF001 - resolution under test
        assert {c.key for c in columns} == {leaf.key for leaf in onnx.leaves}
        assert all(len(c.suffixes) == 1 for c in columns)
        by_key = {leaf.key: leaf for leaf in onnx.leaves}
        for col in columns:
            assert list(col.suffixes) == list(by_key[col.key].suffixes)
        # the gn2v2 export contract: 3 global probs + 2 per-token int8 leaves
        assert onnx.output_names() == ["M_pb", "M_pc", "M_pu", "M_TrackOrigin", "M_VertexIndex"]
        assert len(columns) == 5

    def test_no_label_columns_in_export_selection(self, tmp_path):
        """The export selection carries no target_* column and demands no label."""
        sink = build_inference_sink(_section(tmp_path))
        columns = sink._resolve_columns("run")  # noqa: SLF001 - resolution under test
        assert all("target_" not in s for c in columns for s in c.suffixes)

    def test_test_selection_differs(self, tmp_path):
        """Control: the default (TEST) sink over the SAME section resolves the eval
        schema — per-class seq probs + target_* label columns — not the export set.
        """
        section = _section(tmp_path)
        test_sink = H5OutputSink()
        test_sink.bind_output_section(section)
        suffixes = [s for c in test_sink._resolve_columns("run") for s in c.suffixes]  # noqa: SLF001
        assert "target_jets_classification" in suffixes  # Phase C label column
        assert "pPileup" in suffixes  # per-token per-class prob (TEST-only)
        assert "TrackOrigin" not in suffixes  # the argmax leaf is export-only

    def test_copy_and_mask_writers_gated_by_export_mode(self, tmp_path):
        """test-only copy/mask writers are SKIPPED under inference; default (both) run."""
        gated = build_inference_sink(_section(tmp_path, copy_modes=["test"], mask_modes=["test"]))
        assert gated.copy_inputs == {}
        assert gated.write_pad_mask is False
        default = build_inference_sink(_section(tmp_path))
        assert default.copy_inputs == {"jets": [], "tracks": []}
        assert default.write_pad_mask == ("tracks",)

    def test_test_sink_keeps_test_only_writers(self, tmp_path):
        """Control: the TEST sink keeps modes=[test] copy/mask writers (gating is
        mode-symmetric, not a blanket skip).
        """
        section = _section(tmp_path, copy_modes=["test"], mask_modes=["test"])
        test_sink = H5OutputSink()
        test_sink.bind_output_section(section)
        assert test_sink.copy_inputs == {"jets": [], "tracks": []}
        assert test_sink.write_pad_mask == ("tracks",)

    def test_empty_section_is_refused(self):
        """No outputs: section -> actionable ConfigError (the WHAT is missing)."""
        with pytest.raises(ConfigError, match="outputs"):
            build_inference_sink({})

    def test_default_output_template_is_inference_named(self, tmp_path):
        """The default output template never clobbers a salt2 test eval H5."""
        sink = build_inference_sink(_section(tmp_path))
        assert sink.output == INFERENCE_OUTPUT
        assert "__inference_" in sink.output


class TestInferenceCoreLoop:
    """The command core driven without Lightning: real adapter + dataset + sink.

    The trainer-facing wrapping (checkpoint load, CLI parse) is CI-gated in
    ``test_inference_e2e.py``; this covers the eager per-jet loop, the
    column/leaf 1:1 plan, and the duck-driven sink lifecycle locally.
    """

    N = 120

    def test_eager_loop_writes_export_selection_h5(self, tmp_path):
        """End-to-end core: eager per-jet adapter values land in the export-selection
        H5 columns (spot-checked against a direct adapter call), pads read 0.
        """
        import torch
        from types import SimpleNamespace

        from salt.core.data import Features, GraphDataset, H5StructuredReader, Labels
        from salt.core.graph.bundle import Bundle
        from salt.core.inference import _column_plan, _consume_batch, _jet_args
        from salt.core.nn import bind_all, resolve_bind_schema
        from salt.core.onnx.adapter import OnnxAdapter
        from salt.core.onnx.config import resolve_export_config
        from salt.core.onnx.export import compile_onnx_plan
        from salt.core.schema import dump_schema, save_schema
        from salt.core.testing.inputs import write_dummy_file
        from salt.tests._fixtures.gn2v2_fixture import JET_VARIABLES, TRACK_VARIABLES

        torch.manual_seed(42)
        nd = tmp_path / "nd.yaml"
        section = _section(tmp_path)
        modules = section["run_tasks"]._model_modules  # noqa: SLF001 - fixture reuse
        export_sink = OnnxExportSink()
        export_sink.name = "onnx_export"
        export_sink.model_name = "M"
        export_sink.bind_output_section(section)
        variables = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}
        export = ExportConfig(
            model_name="M",
            inputs=[
                ExportInput(port="inputs.jets", name="jet_features"),
                ExportInput(port="inputs.tracks", name="track_features", sequence=True),
            ],
        )
        resolved = resolve_export_config(export, "run")
        plan_modules = dict(modules)
        plan_modules["run_tasks"] = section["run_tasks"]
        plan_modules["onnx_export"] = export_sink
        plan = compile_onnx_plan(plan_modules, resolved, variables)
        bind_all(plan_modules, resolve_bind_schema([plan]))
        plan_modules["norm"].materialise()
        feature_fields = {f"inputs.{s}": tuple(v) for s, v in variables.items()}
        adapter = OnnxAdapter(plan, resolved, feature_fields)
        # the dummy file + the inference (label-free) dataset demand
        h5_path = tmp_path / "pp_output_test_core.h5"
        write_dummy_file(h5_path, nd)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5_path), schema)
        dset = GraphDataset(
            modules={
                "reader": H5StructuredReader(
                    groups={"jets": {}, "tracks": {}}, schema=schema,
                    filename=h5_path, num=self.N,
                ),
                "features": Features(variables=variables),
                "labels": Labels(),
            },
            mode=Mode.TEST,
            sinks=inference_demand(resolved),
        )
        sink = build_inference_sink(section, output=str(tmp_path / "inference.h5"))
        trainer = SimpleNamespace(
            lightning_module=SimpleNamespace(name="run"),
            datamodule=SimpleNamespace(test_dset=dset, batch_size=60, test_suff=None),
            ckpt_path="unused.ckpt",
            num_test_batches=None,
        )
        sink.open_schema(trainer)
        column_plan = _column_plan(sink, export_sink)
        assert {col.key for col, _, _ in column_plan} == set(export_sink.outputs)
        for start in range(0, self.N, 60):
            _consume_batch(sink, adapter, column_plan, dset[np.s_[start : start + 60]])
        sink.flush()
        with h5py.File(tmp_path / "inference.h5") as f:
            jets, tracks = f["jets"][:], f["tracks"][:]
        assert len(jets) == self.N
        # spot-check jet 0 against a direct per-jet adapter call
        batch = dset[np.s_[0:1]]
        named = dict(
            zip(
                adapter.output_names,
                adapter(*_jet_args(adapter, Bundle(dict(batch)), 0)),
                strict=True,
            )
        )
        for suffix in ("pb", "pc", "pu"):
            np.testing.assert_allclose(
                np.float64(jets[f"run_{suffix}"][0]),
                float(named[f"M_{suffix}"]),
                rtol=1e-6, atol=1e-6,
            )
        n_valid = int((~batch["masks"]["tracks"][0]).sum())
        for col, out in (("run_TrackOrigin", "M_TrackOrigin"), ("run_VertexIndex", "M_VertexIndex")):
            np.testing.assert_array_equal(tracks[col][0][:n_valid], named[out].numpy())
            assert (tracks[col][0][n_valid:] == 0).all()
        # pad-mask column (PadMaskWriter declares export implicitly by default):
        # True = padded, exactly the batch pad mask
        assert tracks["mask"].dtype == np.dtype(bool)
        assert np.array_equal(tracks["mask"][0], batch["masks"]["tracks"][0].numpy())


class TestLabelFreePlanCompile:
    """The Mode.ONNX plan of representative shipped configs demands no labels."""

    @pytest.mark.parametrize(
        ("config", "overrides"),
        [
            ("gn2v2-dummy.yaml", ["model.modules.norm.init_args.norm_dict=unused.yaml"]),
            ("gn2v2-opendata.yaml", []),
        ],
    )
    def test_onnx_plan_is_label_free(self, config, overrides):
        """Compile the ONNX plan through the real config surface: no labels.* key
        flows over any plan edge and the (narrowed) Labels step reads no field.
        """
        from salt.core.cli import load_config
        from salt.core.graph.planner import compile_plan

        gcfg = load_config([str(CONFIG_DIR / config)], overrides)
        assert Mode.ONNX not in gcfg.mode_errors, gcfg.mode_errors.get(Mode.ONNX)
        plan = compile_plan(
            gcfg.modules, Mode.ONNX, gcfg.sources, gcfg.schema, gcfg.sinks.get(Mode.ONNX)
        )
        assert plan.steps, "ONNX plan compiled empty"
        labelled_edges = [e.key for e in plan.edges if e.key.startswith("labels.")]
        assert not labelled_edges, f"{config}: ONNX plan consumes labels {labelled_edges}"
        # the Labels wildcard narrowed against the ONNX demand: the collected step
        # produces NO label key and reads NO field (no label dataset is ever
        # touched — the 50a Task 2 contract).
        for step in plan.steps:
            if step.name != "labels":
                continue
            produced = [key for key in step.produces if key.startswith("labels.")]
            assert not produced, f"{config}: Labels still produces {produced} in ONNX mode"
            read_fields = getattr(step.module, "read_fields", None)
            if callable(read_fields):
                assert not read_fields(step), f"{config}: Labels still reads fields"


class TestUnlabelledDatasetPath:
    """The inference dataset demand serves batches from a label-stripped file."""

    JET_VARS = ("pt_btagJes", "eta_btagJes")
    TRACK_VARS = ("d0", "z0SinTheta", "dphi", "deta")

    @pytest.fixture(scope="class")
    def stripped(self, tmp_path_factory) -> dict[str, Path]:
        from salt.core.schema import dump_schema, save_schema
        from salt.core.testing.inputs import write_dummy_file, write_dummy_norm_dict

        base = tmp_path_factory.mktemp("inference_stripped")
        nd = base / "norm_dict.yaml"
        write_dummy_norm_dict(nd, base / "class_dict.yaml")
        labelled = base / "labelled.h5"
        write_dummy_file(labelled, nd)
        stripped = base / "stripped.h5"
        with h5py.File(labelled) as fin, h5py.File(stripped, "w") as fout:
            for name, ds in fin.items():
                arr = ds[:]
                keep = [f for f in arr.dtype.names if f not in LABEL_FIELDS]
                out = fout.create_dataset(name, data=repack_fields(arr[keep]))
                for k, v in ds.attrs.items():
                    if k not in LABEL_FIELDS:
                        out.attrs[k] = v
        schema = base / "schema.yaml"
        save_schema(dump_schema(stripped), schema)
        return {"h5": stripped, "schema": schema}

    def test_datamodule_serves_label_free_batches(self, stripped):
        """GraphDataModule + the inference demand: the label-stripped file binds and
        serves batches with no labels leaf — the exact command data path.
        """
        from salt.core.data import Features, H5StructuredReader, Labels
        from salt.core.data.datamodule import GraphDataModule

        export = ExportConfig(
            model_name="M",
            inputs=[
                ExportInput(port="inputs.jets", name="jet_features"),
                ExportInput(port="inputs.tracks", name="track_features", sequence=True),
            ],
        )
        dm = GraphDataModule(
            modules={
                "reader": H5StructuredReader(
                    groups={"jets": {}, "tracks": {}}, schema=stripped["schema"]
                ),
                "features": Features(
                    variables={"jets": list(self.JET_VARS), "tracks": list(self.TRACK_VARS)}
                ),
                "labels": Labels(),
            },
            test_file=stripped["h5"],
            num_workers=0,
        )
        dm.set_sinks({Mode.TEST: inference_demand(export)})
        dm.setup("test")
        assert dm.test_dset is not None
        for fields in dm.test_dset.read_fields.values():
            assert not set(fields) & LABEL_FIELDS
        batch = dm.test_dset[np.s_[0:64]]
        assert "labels" not in batch
        assert batch["inputs"]["jets"].shape == (64, len(self.JET_VARS))
        assert batch["inputs"]["tracks"].shape[0] == 64
        rows = batch["meta"]["rows"]
        assert int(rows[0]) == 0
        assert int(rows[1]) == 64
