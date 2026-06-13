"""Tests for salt/core/writers (M3 stage A — design §2.7, §8).

Per-writer units check the v1 column-naming/dtype/value contract on the
GN2v2 fixture modules with fabricated bundles; the end-to-end class runs the
REAL ``salt2 fit`` -> ``salt2 test`` surface on a tmp dummy file and checks
the v1 output-file shape (one H5 next to the checkpoint, input copies +
prob columns + VertexIndex + mask, padded-position encodings); negative
tests cover the TEST dead-preds hard error, the writer-less eval refusal,
and the no-ckpt single-config contract. `JetCountWriter` is the design §8
custom-writer journey (W4 shape).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import TensorSpec
from salt.core.main import CONFIG_DIR, main
from salt.core.schema import dump_schema, save_schema
from salt.core.writers import (
    OBJECT_INDEX,
    InputCopyWriter,
    MaskFormerObjectWriter,
    PadMaskWriter,
    TaskWriter,
    WriteCtx,
    Writer,
    WriterCallback,
    WriterDeclareCtx,
)
from salt.tests.core.gn2_fixture import write_parity_norm_dict
from salt.tests.core.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
RUN_NAME = "GN2v2_dummy"  # the dummy config's `name:`
N_JETS, L_FILE = 1000, 40  # write_dummy_file geometry
PRED_KEYS = [
    "preds.jets.jets_classification",
    "preds.tracks.track_origin",
    "preds.tracks.track_vertexing",
]
JET_PROB_COLS = ["salt_pb", "salt_pc", "salt_pu"]
ORIGIN_PROB_COLS = [f"salt_p{c}" for c in ORIGIN_CLASSES]


class JetCountWriter(Writer):
    """The design §8 custom-writer journey: one new column, four YAML lines."""

    def requires(self, ctx):
        del ctx
        return {"masks.tracks": TensorSpec(shape=None, dtype="bool", kind="pad_mask")}

    def columns(self, ctx):
        del ctx
        return {"jets": np.dtype([("n_valid_tracks", "i4")])}

    def write(self, bundle, rows):
        del rows
        mask = bundle.get("masks.tracks").cpu().numpy()
        counts = (~mask).sum(-1, keepdims=True).astype("i4")
        return {"jets": u2s(counts, np.dtype([("n_valid_tracks", "i4")]))}


class NamedFeatureWriter(Writer):
    """Resolves a feature column BY NAME via ``ctx.feature_fields`` (M3-review fix)."""

    COLUMN = "eta_by_name"
    DTYPE = np.dtype([("eta_by_name", "f4")])

    def requires(self, ctx):
        del ctx
        return {"inputs.jets": TensorSpec(dtype="float32", kind="data")}

    def columns(self, ctx):
        assert "inputs.jets" in ctx.feature_fields  # the bind-time field names
        return {"jets": self.DTYPE}

    def write(self, bundle, rows):
        del rows
        # no index arithmetic off the config — resolve the column by name
        idx = self.ctx.feature_fields["inputs.jets"].index("eta_btagJes")
        vals = bundle.get("inputs.jets").cpu().numpy()[:, idx : idx + 1].astype("f4")
        return {"jets": u2s(vals, self.DTYPE)}


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("m3_writers")
    nd_path, cd_path = base / "norm_dict.yaml", base / "class_dict.yaml"
    write_parity_norm_dict(nd_path, cd_path)
    # exactly four underscore parts -> sample heuristic yields 'ttbar' (PW:169)
    h5_path = base / "pp_output_test_ttbar.h5"
    write_dummy_file(h5_path, nd_path)
    schema_path = base / "schema.yaml"
    save_schema(dump_schema(h5_path), schema_path)
    return {"dir": base, "h5": h5_path, "nd": nd_path, "schema": schema_path}


@pytest.fixture(scope="module")
def modules(data):
    # config-only module dict: requires/columns need attrs, not bound layers
    return build_gn2v2_modules(data["nd"])


def declare_ctx(modules) -> WriterDeclareCtx:
    return WriterDeclareCtx(
        model_modules=modules, streams=("jets", "tracks"), sequence_streams=("tracks",)
    )


def write_ctx(data, modules, out: Path | None = None, run_name: str = "salt") -> WriteCtx:
    return WriteCtx(
        output_path=out or (data["dir"] / "unit_out.h5"),
        total=N_JETS,
        run_name=run_name,
        source_path=data["h5"],
        streams=("jets", "tracks"),
        sequence_streams=("tracks",),
        group_datasets={"jets": "jets", "tracks": "tracks"},
        seq_lengths={"tracks": L_FILE},
        model_modules=modules,
        batch_size=100,
    )


def make_preds_bundle(b: int = 5, length: int = L_FILE, seed: int = 3) -> Bundle:
    gen = torch.Generator().manual_seed(seed)
    jets = torch.softmax(torch.randn(b, 3, generator=gen), -1)
    origin = torch.softmax(torch.randn(b, length, 8, generator=gen), -1)
    vertex = torch.randint(-1, 4, (b, length, 1), generator=gen).float()
    vertex[:, -2:] = float("-inf")  # padded rows (v1 mask_fill_flattened encoding)
    return Bundle({
        "preds": {
            "jets": {"jets_classification": jets},
            "tracks": {"track_origin": origin, "track_vertexing": vertex},
        }
    })


# ---------------------------------------------------------------------------
# TaskWriter: v1 column naming/dtypes/values (predictionwriter.py:249-259)
# ---------------------------------------------------------------------------


class TestTaskWriter:
    def test_requires_all_task_streams_by_default(self, modules):
        assert list(TaskWriter().requires(declare_ctx(modules))) == PRED_KEYS

    def test_requires_narrowed_by_streams(self, modules):
        tw = TaskWriter(streams=["jets"])
        assert list(tw.requires(declare_ctx(modules))) == [PRED_KEYS[0]]

    def test_unknown_stream_errors(self, modules):
        with pytest.raises(ConfigError, match="no configured task"):
            TaskWriter(streams=["muons"]).requires(declare_ctx(modules))

    def test_columns_v1_naming(self, data, modules):
        cols = TaskWriter().columns(write_ctx(data, modules))
        # jets: Flavours px naming with the run-name prefix (task.py:140-151)
        assert list(cols["jets"].names) == JET_PROB_COLS
        assert all(cols["jets"][n] == np.dtype("f4") for n in cols["jets"].names)
        # tracks: 8 origin prob columns + the BARE v1 VertexIndex i8
        assert list(cols["tracks"].names) == [*ORIGIN_PROB_COLS, "VertexIndex"]
        assert cols["tracks"]["VertexIndex"] == np.dtype("i8")

    def test_vertex_prefix_option(self, data, modules):
        cols = TaskWriter(prefix_vertex_column=True).columns(write_ctx(data, modules))
        assert "salt_VertexIndex" in cols["tracks"].names

    def test_write_values_bitwise(self, data, modules):
        tw = TaskWriter()
        tw.setup(write_ctx(data, modules))
        bundle = make_preds_bundle()
        out = tw.write(bundle, slice(0, 5))
        jets = bundle.get(PRED_KEYS[0]).numpy()
        for i, col in enumerate(JET_PROB_COLS):
            assert out["jets"][col].tobytes() == jets[:, i].astype("f4").tobytes()
        # vertexing: the exact v1 op chain .int() -> u2s i8 (task.py:1003)
        vertex = bundle.get(PRED_KEYS[2]).int().numpy()[..., 0].astype("i8")
        assert out["tracks"]["VertexIndex"].tobytes() == vertex.tobytes()
        assert (out["tracks"]["VertexIndex"][:, -2:] == np.int64(-2147483648)).all()

    def test_write_pads_to_file_length(self, data, modules):
        tw = TaskWriter()
        tw.setup(write_ctx(data, modules))
        out = tw.write(make_preds_bundle(length=30), slice(0, 5))
        assert out["tracks"].shape == (5, L_FILE)  # maybe_pad re-expansion
        assert (out["tracks"][ORIGIN_PROB_COLS[0]][:, 30:] == 0.0).all()
        assert (out["tracks"]["VertexIndex"][:, 30:] == 0).all()

    def test_unknown_task_family_errors(self, data, modules):
        class WeirdTask:
            pred_key = "preds.tracks.weird"
            stream = "tracks"

        bad = dict(modules) | {"weird": WeirdTask()}
        with pytest.raises(ConfigError, match="custom Writer"):
            TaskWriter().columns(write_ctx(data, bad))


# ---------------------------------------------------------------------------
# InputCopyWriter: source dtypes/order, meta.rows re-reads (PW:173-196)
# ---------------------------------------------------------------------------


class TestInputCopyWriter:
    def test_requires_meta_rows_only(self, modules):
        assert list(InputCopyWriter().requires(declare_ctx(modules))) == ["meta.rows"]

    def test_columns_full_source_dtypes(self, data, modules):
        icw = InputCopyWriter()
        icw.setup(write_ctx(data, modules))
        cols = icw.columns(write_ctx(data, modules))
        with h5py.File(data["h5"]) as f:
            assert list(cols["jets"].names) == list(f["jets"].dtype.names)
            assert list(cols["tracks"].names) == list(f["tracks"].dtype.names)
            # SOURCE dtypes ride along untouched (f.ex. int labels stay int)
            for name in cols["jets"].names:
                assert cols["jets"][name] == f["jets"].dtype[name]
        icw.finalize()

    def test_write_rereads_absolute_rows(self, data, modules):
        icw = InputCopyWriter()
        icw.setup(write_ctx(data, modules))
        out = icw.write(Bundle(), slice(100, 200))
        with h5py.File(data["h5"]) as f:
            names = list(f["jets"].dtype.names)
            expect = f["jets"].fields(names)[100:200]
        assert out["jets"].tobytes() == expect.tobytes()
        assert out["tracks"].shape == (100, L_FILE)
        icw.finalize()

    def test_variables_narrowing_and_validation(self, data, modules):
        icw = InputCopyWriter(variables={"jets": ["pt", "eta"]})
        icw.setup(write_ctx(data, modules))
        assert list(icw.columns(write_ctx(data, modules))["jets"].names) == ["pt", "eta"]
        icw.finalize()
        with pytest.raises(ConfigError, match="missing for stream"):
            InputCopyWriter(variables={"jets": ["not_a_var"]}).setup(write_ctx(data, modules))

    def test_unknown_stream_errors(self, modules):
        with pytest.raises(ConfigError, match="unknown streams"):
            InputCopyWriter(streams=["muons"])._selected_streams(declare_ctx(modules))


# ---------------------------------------------------------------------------
# PadMaskWriter: per-requested-stream mask column (design §8 fix of PW:263-265)
# ---------------------------------------------------------------------------


class TestPadMaskWriter:
    def test_default_streams_are_tasked_sequence_streams(self, modules):
        assert list(PadMaskWriter().requires(declare_ctx(modules))) == ["masks.tracks"]

    def test_vector_stream_rejected(self, modules):
        with pytest.raises(ConfigError, match="not sequence streams"):
            PadMaskWriter(streams=["jets"]).requires(declare_ctx(modules))

    def test_write_values_and_padding(self, data, modules):
        pmw = PadMaskWriter()
        pmw.setup(write_ctx(data, modules))
        mask = torch.rand(5, 30) > 0.5
        out = pmw.write(Bundle({"masks": {"tracks": mask}}), slice(0, 5))
        assert out["tracks"].dtype == np.dtype([("mask", "?")])
        assert out["tracks"].shape == (5, L_FILE)
        assert (out["tracks"]["mask"][:, :30] == mask.numpy()).all()
        # the preserved v1 maybe_pad quirk: truncated positions read False
        assert not out["tracks"]["mask"][:, 30:].any()


# ---------------------------------------------------------------------------
# WriterCallback internals: column merge collision check (design §8)
# ---------------------------------------------------------------------------


class TestWriterCallback:
    def test_empty_modules_rejected(self):
        with pytest.raises(ConfigError, match="non-empty writer dict"):
            WriterCallback(modules={})
        with pytest.raises(ConfigError, match="non-empty writer dict"):
            WriterCallback(modules={"tasks": None})

    def test_non_writer_rejected(self):
        with pytest.raises(ConfigError, match="not a"):
            WriterCallback(modules={"tasks": object()})  # type: ignore[dict-item]

    def test_column_collision_names_both_writers(self, data, modules):
        cb = WriterCallback(modules={"a": PadMaskWriter(), "b": PadMaskWriter()})
        ctx = write_ctx(data, modules)
        for writer in cb.writers.values():
            writer.setup(ctx)
        with pytest.raises(ConfigError, match="'a' AND 'b'"):
            cb._merge_columns(ctx)

    def test_merge_preserves_writer_order(self, data, modules):
        cb = WriterCallback(
            modules={"inputs_copy": InputCopyWriter(), "tasks": TaskWriter()},
        )
        ctx = write_ctx(data, modules)
        for writer in cb.writers.values():
            writer.setup(ctx)
        dtypes, shapes = cb._merge_columns(ctx)
        with h5py.File(data["h5"]) as f:
            source_jets = list(f["jets"].dtype.names)
        # v1 layout: input copies FIRST, then task columns
        assert list(dtypes["jets"].names) == source_jets + JET_PROB_COLS
        assert shapes["tracks"] == (N_JETS, L_FILE)
        cb.writers["inputs_copy"].finalize()

    def test_runtime_stub_backstop_in_merge_columns(self, data, modules):
        # M4.5 merge condition 3, the RUNTIME half (fix-stage regression):
        # a writer that slips past the static role check (it declares TEST
        # requires) but produces NO columns while declaring ONNX outputs
        # must hit the explicit stub error at column merge — never a silent
        # zero-contribution fall-through
        from salt.core.onnx.config import ExportOutput

        class RuntimeStub(Writer):
            def requires(self, ctx):
                del ctx
                return {"meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta")}

            def columns(self, ctx):
                del ctx
                return {}

            def write(self, bundle, rows):
                del bundle, rows
                return {}

            def onnx_outputs(self, ctx):
                del ctx
                return [ExportOutput(port="pooled.global", names=["s0", "s1"])]

        cb = WriterCallback(modules={"tasks": TaskWriter(), "stub": RuntimeStub()})
        ctx = write_ctx(data, modules)
        for writer in cb.writers.values():
            writer.setup(ctx)
        with pytest.raises(ConfigError, match="export-only stub shape"):
            cb._merge_columns(ctx)


# ---------------------------------------------------------------------------
# MaskFormerObjectWriter: TEST byte-parity vs v1 + the extra-group plumbing
# (predictionwriter.py:267-308; M5 sub-wave C, plan 10)
# ---------------------------------------------------------------------------


OBJECT_CLASSES = ["b", "c", "null"]


def mf_writer_modules(nd):
    from salt.tests.core.regression_fixture import build_maskformer_writer_modules

    return build_maskformer_writer_modules(nd)


def mf_write_ctx(data, modules, n_tracks: int = 10, total: int = 6) -> WriteCtx:
    return WriteCtx(
        output_path=data["dir"] / "mf.h5",
        total=total,
        run_name="MFrun",
        source_path=data["h5"],
        streams=("jets", "tracks"),
        sequence_streams=("tracks",),
        group_datasets={"jets": "jets", "tracks": "tracks"},
        seq_lengths={"tracks": n_tracks},
        model_modules=modules,
        batch_size=total,
    )


class TestMaskFormerObjectWriter:
    def _writer(self):
        w = MaskFormerObjectWriter(object_classes=OBJECT_CLASSES, regression_task="regression")
        w.name = "object_writer"
        return w

    def test_requires_decoder_preds_and_truth_labels(self, modules):
        # the truth requires keep MaskFormerTargets alive in the TEST plan (amendment §3)
        keys = sorted(self._writer().requires(declare_ctx(modules)))
        assert keys == [
            "labels.objects.masks",
            "labels.objects.object_class",
            "objects.class_probs",
            "objects.masks",
        ]

    def test_extra_groups_object_and_object_masks(self, data, modules):
        mods = mf_writer_modules(data["nd"])
        groups = self._writer().extra_groups(mf_write_ctx(data, mods, n_tracks=L_FILE))
        # objects [M], object_masks [M, T] (M from the decoder, T from the file length)
        assert groups == {"objects": (5,), "object_masks": (5, L_FILE)}

    def test_columns_v1_naming(self, data, modules):
        mods = mf_writer_modules(data["nd"])
        cols = self._writer().columns(mf_write_ctx(data, mods))
        # objects: per-class p{name} probs + the class_label truth (v1 :276-285)
        assert list(cols["objects"].names) == ["MFrun_pb", "MFrun_pc", "MFrun_pnull", "class_label"]
        # the constituent stream gains the MaskIndex column (the PINNED test suffix)
        assert list(cols["tracks"].names) == [f"MFrun_{OBJECT_INDEX.test}"]
        # object_masks: truth + logits (v1 :300-308)
        assert list(cols["object_masks"].names) == ["truth_mask", "mask_logits"]

    def test_write_byte_parity_vs_v1_opchain(self, data, modules):
        from salt.tests.core.regression_fixture import make_maskformer_writer_batch
        from salt.utils.mask_utils import indices_from_mask

        mods = mf_writer_modules(data["nd"])
        writer = self._writer()
        writer.setup(mf_write_ctx(data, mods))
        batch = make_maskformer_writer_batch(batch_size=6, n_tracks=10)
        bundle = Bundle()
        for key, value in batch.items():
            bundle.set(key, value)
        out = writer.write(bundle, slice(0, 6))

        cp, masks = batch["objects.class_probs"], batch["objects.masks"]
        oc, tm, pad = (
            batch["labels.objects.object_class"],
            batch["labels.objects.masks"],
            batch["masks.tracks"],
        )
        # objects group: probs + remapped truth class (v1 op chain, predictionwriter.py:276-285)
        v1_probs = u2s(cp.numpy(), np.dtype([(f"MFrun_p{c}", "f4") for c in OBJECT_CLASSES]))
        for n in v1_probs.dtype.names:
            assert out["objects"][n].tobytes() == v1_probs[n].tobytes()
        v1_class = u2s(oc.unsqueeze(-1).numpy(), np.dtype([("class_label", "i8")]))
        assert out["objects"]["class_label"].tobytes() == v1_class["class_label"].tobytes()
        # MaskIndex: indices_from_mask(sigmoid > 0.5) (-2 no object), -1 padded (v1 :287-297)
        v1_idx = indices_from_mask(masks.sigmoid() > 0.5).int().numpy()
        v1_idx = np.where(~pad.numpy(), v1_idx, -1)
        col = f"MFrun_{OBJECT_INDEX.test}"
        assert (
            out["tracks"][col].tobytes()
            == u2s(np.expand_dims(v1_idx, -1), np.dtype([(col, "i8")]))[col].tobytes()
        )
        assert (out["tracks"][col] == -1).any()  # padded sentinel
        assert (out["tracks"][col] == -2).any()  # no-object sentinel
        # object_masks group: truth mask + logits (v1 :300-308)
        assert (
            out["object_masks"]["truth_mask"].tobytes()
            == u2s(tm.unsqueeze(-1).numpy(), np.dtype([("truth_mask", "i8")]))[
                "truth_mask"
            ].tobytes()
        )
        assert (
            out["object_masks"]["mask_logits"].tobytes()
            == u2s(masks.float().unsqueeze(-1).numpy(), np.dtype([("mask_logits", "f4")]))[
                "mask_logits"
            ].tobytes()
        )

    def test_onnx_manifest_two_object_reduces(self, data):
        from salt.core.onnx.config import ExportOutput

        mods = mf_writer_modules(data["nd"])
        manifest = self._writer().onnx_outputs(declare_ctx(mods))
        assert all(type(e) is ExportOutput for e in manifest)
        # leading_object (5/3 split_scalars-style names) then object_index (HadronIndex)
        assert manifest[0].port == "preds.objects.regression"
        assert manifest[0].reduce == "leading_object"
        assert manifest[0].names == [
            "leading_objects_pt",
            "leading_objects_Lxy",
            "leading_objects_mass",
        ]
        assert manifest[1].port == "objects.masks"
        assert manifest[1].reduce == "object_index"
        assert manifest[1].name == OBJECT_INDEX.onnx  # HadronIndex (pinned divergence)

    def test_onnx_false_is_eval_only(self, data):
        mods = mf_writer_modules(data["nd"])
        w = MaskFormerObjectWriter(
            object_classes=OBJECT_CLASSES, regression_task="regression", onnx=False
        )
        w.name = "object_writer"
        assert w.onnx_outputs(declare_ctx(mods)) == []
        assert "objects.class_probs" in w.requires(declare_ctx(mods))  # TEST role intact

    def test_empty_object_classes_rejected(self):
        with pytest.raises(ConfigError, match="non-empty"):
            MaskFormerObjectWriter(object_classes=[])

    def test_object_index_imported_not_redeclared(self):
        # merge condition 4: the strings live ONLY in salt.core.writers.names
        import salt.core.writers.maskformer as src

        source = Path(src.__file__).read_text()
        assert "from salt.core.writers.names import OBJECT_INDEX" in source
        assert '"MaskIndex"' not in source and "'MaskIndex'" not in source
        assert '"HadronIndex"' not in source and "'HadronIndex'" not in source

    def test_extra_group_merges_through_writer_callback(self, data):
        # the extra-group plumbing: the writer's objects/object_masks groups flow
        # through WriterCallback._merge_columns with their writer-declared shapes
        mods = mf_writer_modules(data["nd"])
        cb = WriterCallback(modules={"object_writer": self._writer()})
        ctx = mf_write_ctx(data, mods)
        for writer in cb.writers.values():
            writer.setup(ctx)
        dtypes, shapes = cb._merge_columns(ctx)
        assert "objects" in dtypes and "object_masks" in dtypes
        assert shapes["objects"] == (6, 5)  # (total, M)
        assert shapes["object_masks"] == (6, 5, 10)  # (total, M, T=n_tracks of the ctx)
        # the reader-stream MaskIndex column rides on the tracks group
        assert f"MFrun_{OBJECT_INDEX.test}" in dtypes["tracks"].names


# ---------------------------------------------------------------------------
# end to end: salt2 fit -> salt2 test on the dummy config (design §9.5 M3)
# ---------------------------------------------------------------------------


def overrides(data) -> list[str]:
    return [
        f"--data.modules.reader.init_args.schema={data['schema']}",
        f"--model.modules.norm.init_args.norm_dict={data['nd']}",
        "--trainer.accelerator=cpu",
        "--trainer.enable_progress_bar=false",
    ]


@pytest.fixture(scope="module")
def ckpt(data, tmp_path_factory) -> Path:
    fit_dir = tmp_path_factory.mktemp("m3_fit")
    rc = main([
        "fit",
        "--config",
        str(DUMMY_CFG),
        f"--data.train_file={data['h5']}",
        f"--data.val_file={data['h5']}",
        *overrides(data),
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


def run_test_cli(data, ckpt: Path, extra: list[str] | None = None) -> int:
    return main([
        "test",
        "--config",
        str(DUMMY_CFG),
        f"--data.test_file={data['h5']}",
        f"--ckpt_path={ckpt}",
        "--data.num_test=300",
        # keep run artifacts (GraphArtifacts plan/graph files) out of the cwd
        f"--trainer.default_root_dir={data['dir']}",
        *overrides(data),
        *(extra or []),
    ])


class TestSalt2TestEndToEnd:
    def test_eval_file_v1_layout(self, data, ckpt):
        rc = run_test_cli(data, ckpt)
        assert rc == 0
        # exactly ONE h5 next to the checkpoint, v1 path contract (PW:164-171)
        outputs = list(ckpt.parent.glob("*.h5"))
        assert len(outputs) == 1
        expected = ckpt.parent / f"{ckpt.stem}__test_ttbar.h5"
        assert outputs == [expected]
        run_cols_jets = [f"{RUN_NAME}_p{c}" for c in "bcu"]
        run_cols_origin = [f"{RUN_NAME}_p{c}" for c in ORIGIN_CLASSES]
        with h5py.File(expected) as f:
            assert set(f.keys()) == {"jets", "tracks"}
            assert f.attrs["writer_version"]
            jets, tracks = f["jets"][:], f["tracks"][:]
        with h5py.File(data["h5"]) as f:
            src_jets = list(f["jets"].dtype.names)
            src_tracks = list(f["tracks"].dtype.names)
            valid = f["tracks"]["valid"][:300]
        assert len(jets) == 300
        assert tracks.shape == (300, L_FILE)
        # v1 per-group layout: input copies, task columns, mask last
        assert list(jets.dtype.names) == src_jets + run_cols_jets
        assert list(tracks.dtype.names) == src_tracks + run_cols_origin + ["VertexIndex", "mask"]
        # conversions ran: probabilities, not logits
        prob_sum = sum(jets[c].astype("f8") for c in run_cols_jets)
        assert np.allclose(prob_sum, 1.0, atol=1e-3)
        # padded-position encodings (design §8): probs 0.0, vertexing -inf-cast-int
        padded = tracks["mask"]
        assert (padded == ~valid).all()
        assert (tracks[run_cols_origin[0]][padded] == 0.0).all()
        assert (tracks["VertexIndex"][padded] == np.int64(-2147483648)).all()
        # float input copies downcast to f4, ints untouched (v1 'full' policy)
        assert jets.dtype["pt"] == np.dtype("f4")

    def test_half_precision_v1_flag(self, data, ckpt, tmp_path):
        # M3-review fix: the advertised v1 half_precision compat surface has
        # coverage — float columns (probs AND input copies) land as f2,
        # integer/bool columns untouched (the shared ftag H5Writer policy)
        out = tmp_path / "half.h5"
        rc = run_test_cli(
            data,
            ckpt,
            extra=["--writers.half_precision=true", f"--writers.output={out}"],
        )
        assert rc == 0
        with h5py.File(out) as f:
            jets_dtype, tracks_dtype = f["jets"].dtype, f["tracks"].dtype
        assert jets_dtype["pt"] == np.dtype("f2")  # input copy downcast
        assert jets_dtype[f"{RUN_NAME}_pb"] == np.dtype("f2")  # prob column
        assert tracks_dtype["VertexIndex"] == np.dtype("i8")  # ints untouched
        assert tracks_dtype["mask"] == np.dtype("?")  # bools untouched

    def test_artifacts_land_next_to_checkpoint(self, data, ckpt):
        # M3-review HIGH fix: on the test path the GraphArtifacts default is
        # the checkpoint dir (with the eval H5), not the cwd-derived log dir
        rc = run_test_cli(data, ckpt)
        assert rc == 0
        plan = ckpt.parent / "plan_test.txt"
        assert plan.exists()
        text = plan.read_text()
        # the writer-sinks table answers "which writer consumes preds.X"
        assert "# writer sinks (design §8)" in text
        assert "tasks (TaskWriter):" in text
        assert "preds.tracks.track_origin" in text
        assert (ckpt.parent / "resolved_io.yaml").exists()

    def test_no_ckpt_fallback_globs_v2_checkpoints(self, data, ckpt, tmp_path):
        # M3-review fix: base2.yaml names checkpoints 'epoch=NNN-loss=...'
        # under checkpoints/ and the fallback globs that dir next to the
        # saved config — salt2 test now works without --ckpt_path on a
        # v2-trained run dir (the old ckpts/-only 'loss=' glob NEVER matched)
        assert "loss=" in ckpt.name  # the base2.yaml filename contract
        config = ckpt.parent.parent / "config.yaml"
        assert config.is_file()  # the saved run config next to checkpoints/
        out = tmp_path / "fallback.h5"
        rc = main([
            "test",
            "--config",
            str(config),
            f"--data.test_file={data['h5']}",
            "--data.num_test=300",
            f"--writers.output={out}",
            "--trainer.enable_progress_bar=false",
        ])
        assert rc == 0
        assert out.exists()

    def test_custom_writer_journey(self, data, ckpt, tmp_path):
        # W4 shape: subclass + a few config lines -> a new column end to end
        out = tmp_path / "custom.h5"
        rc = run_test_cli(
            data,
            ckpt,
            extra=[
                '--writers.modules.n_tracks={"class_path": '
                '"salt.tests.core.test_writers.JetCountWriter"}',
                f"--writers.output={out}",
            ],
        )
        assert rc == 0
        with h5py.File(out) as f:
            jets = f["jets"][:]
        assert "n_valid_tracks" in jets.dtype.names
        with h5py.File(data["h5"]) as f:
            counts = f["tracks"]["valid"][:300].sum(-1).astype("i4")
        assert (jets["n_valid_tracks"] == counts).all()

    def test_custom_writer_resolves_features_by_name(self, data, ckpt, tmp_path):
        # M3-review ergonomics fix: WriteCtx.feature_fields lets a custom
        # writer resolve a feature column by NAME (no index arithmetic)
        out = tmp_path / "by_name.h5"
        rc = run_test_cli(
            data,
            ckpt,
            extra=[
                '--writers.modules.eta={"class_path": '
                '"salt.tests.core.test_writers.NamedFeatureWriter"}',
                f"--writers.output={out}",
            ],
        )
        assert rc == 0
        with h5py.File(out) as f:
            got = f["jets"][NamedFeatureWriter.COLUMN][:]
        with h5py.File(data["h5"]) as f:
            expected = f["jets"]["eta_btagJes"][:300].astype("f4")
        assert got.tobytes() == expected.tobytes()

    def test_dead_preds_hard_error(self, data, ckpt, capsys):
        # narrowing TaskWriter to jets leaves the two track preds unconsumed
        rc = run_test_cli(data, ckpt, extra=['--writers.modules.tasks.init_args.streams=["jets"]'])
        assert rc == 1
        err = capsys.readouterr().err
        assert "consumed by NO writer" in err
        assert "track_origin" in err
        assert "track_vertexing" in err
        # §4.2-exemplar attribution (M3-review fix): culprit writer address
        # with the excluded streams, per-task config addresses, and the
        # null-deletion workaround for train-only aux tasks
        assert "writers.modules.tasks.init_args.streams" in err
        assert "excludes" in err
        assert "model.modules.track_origin" in err
        assert "--model.modules.track_origin=null" in err

    def test_writerless_eval_refused(self, data, ckpt, capsys):
        rc = run_test_cli(
            data,
            ckpt,
            extra=[
                "--writers.modules.inputs_copy=null",
                "--writers.modules.tasks=null",
                "--writers.modules.pad_mask=null",
            ],
        )
        assert rc == 1
        assert "at least one writer" in capsys.readouterr().err

    def test_no_ckpt_needs_single_config_with_losses(self, data, capsys):
        # no --ckpt_path: the v1 best-epoch glob needs loss= checkpoint names
        rc = main([
            "test",
            "--config",
            str(DUMMY_CFG),
            f"--data.test_file={data['h5']}",
            *overrides(data),
        ])
        assert rc == 1
        assert "ckpt" in capsys.readouterr().err
