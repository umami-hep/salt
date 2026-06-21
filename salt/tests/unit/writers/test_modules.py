"""Tests for `salt.core.writers.modules` — TaskWriter / InputCopyWriter / PadMaskWriter.

Per-writer units check the v1 column-naming/dtype/value contract on the GN2v2
fixture modules with fabricated bundles. Also pins that `TaskWriter` is pure
orchestration carrying NO per-family knowledge (the M-modular refactor moved
all per-family rendering onto the task), and that the per-task `output_names` /
`get_h5` / `onnx_outputs` render their own output.

(Split out of the former monolithic ``test_writers.py``; shared fixtures /
toy-writers / constants come from ``salt.tests._fixtures.writers_common``.)
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.writers import (
    InputCopyWriter,
    PadMaskWriter,
    TaskWriter,
)
from salt.tests._fixtures.writers_common import (  # noqa: F401  (data/modules are fixtures)
    JET_PROB_COLS,
    L_FILE,
    ORIGIN_PROB_COLS,
    PRED_KEYS,
    bound_modules,
    data,
    declare_ctx,
    make_preds_bundle,
    modules,
    write_ctx,
)

# ---------------------------------------------------------------------------
# TaskWriter: v1 column naming/dtypes/values (predictionwriter.py:249-259)
# ---------------------------------------------------------------------------


class TestTaskWriter:
    def test_requires_all_task_streams_by_default(self, modules):
        assert list(TaskWriter().requires(declare_ctx(modules))) == PRED_KEYS

    def test_requires_narrowed_by_tasks(self, modules):
        tw = TaskWriter(tasks=["jets_classification"])
        assert list(tw.requires(declare_ctx(modules))) == [PRED_KEYS[0]]

    def test_unknown_task_errors(self, modules):
        with pytest.raises(ConfigError, match="no configured task"):
            TaskWriter(tasks=["muons"]).requires(declare_ctx(modules))

    def test_columns_v1_naming(self, data, modules):
        cols = TaskWriter().columns(write_ctx(data, modules))
        # jets: Flavours px naming with the run-name prefix (task.py:140-151)
        assert list(cols["jets"].names) == JET_PROB_COLS
        assert all(cols["jets"][n] == np.dtype("f4") for n in cols["jets"].names)
        # tracks: 8 origin prob columns + the BARE v1 VertexIndex i8
        assert list(cols["tracks"].names) == [*ORIGIN_PROB_COLS, "VertexIndex"]
        assert cols["tracks"]["VertexIndex"] == np.dtype("i8")

    def test_vertex_prefix_option(self, data, modules):
        # prefix_vertex_column moved onto the VertexingTaskModule (the task owns
        # its column naming); the writer is pure orchestration and never sees it
        import copy

        prefixed = dict(modules)
        prefixed["track_vertexing"] = copy.deepcopy(modules["track_vertexing"])
        prefixed["track_vertexing"].prefix_vertex_column = True
        cols = TaskWriter().columns(write_ctx(data, prefixed))
        assert "salt_VertexIndex" in cols["tracks"].names

    def test_write_values_bitwise(self, data, bound_modules):
        tw = TaskWriter()
        tw.setup(write_ctx(data, bound_modules))
        bundle = make_preds_bundle()
        out = tw.write(bundle, slice(0, 5))
        # the P1.5-flipped jets_classification.get_h5 softmaxes the RAW logits
        # before packing, so the column == softmax(raw logits) (the SAME value
        # the pre-flip forward used to publish — get_h5 just owns it now)
        jets = torch.softmax(bundle.get(PRED_KEYS[0]), -1).numpy()
        for i, col in enumerate(JET_PROB_COLS):
            assert out["jets"][col].tobytes() == jets[:, i].astype("f4").tobytes()
        # vertexing (NOT flipped): the exact v1 op chain .int() -> u2s i8 (task.py:1003)
        vertex = bundle.get(PRED_KEYS[2]).int().numpy()[..., 0].astype("i8")
        assert out["tracks"]["VertexIndex"].tobytes() == vertex.tobytes()
        assert (out["tracks"]["VertexIndex"][:, -2:] == np.int64(-2147483648)).all()

    def test_write_pads_to_file_length(self, data, bound_modules):
        tw = TaskWriter()
        tw.setup(write_ctx(data, bound_modules))
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
# The per-family rendering is OWNED BY THE TASK now (the M-modular refactor,
# mirroring v1's task.py placement). These units exercise `output_names` /
# `get_h5` / `onnx_outputs` DIRECTLY on the task instances — the writer's
# isinstance/class_names ladder is gone (proved separately below).
# ---------------------------------------------------------------------------


class TestTaskRendersItsOwnOutput:
    def test_global_classification_output_names(self, modules):
        # v1 ClassificationTask.output_names (task.py:140-151): Flavours px,
        # run-name-prefixed, f4 — rendered by the TASK, not the writer
        descr = modules["jets_classification"].output_names("salt")
        assert descr == [("salt_pb", "f4"), ("salt_pc", "f4"), ("salt_pu", "f4")]

    def test_global_classification_get_h5_is_v1_opchain(self, bound_modules):
        # the P1.5-flipped get_h5 run_inference-s the RAW logits (softmax) then
        # packs probs -> f4 u2s verbatim (task.py:266-283), dtype = output_names
        bundle = make_preds_bundle()
        arr = bound_modules["jets_classification"].get_h5(bundle, "salt")
        assert list(arr.dtype.names) == ["salt_pb", "salt_pc", "salt_pu"]
        jets = torch.softmax(bundle.get(PRED_KEYS[0]), -1).float().numpy()
        for i, col in enumerate(["salt_pb", "salt_pc", "salt_pu"]):
            assert arr[col].tobytes() == jets[:, i].astype("f4").tobytes()

    def test_global_classification_onnx_outputs_per_class_names(self, modules):
        # a global head renders a per-class `names` entry with BARE suffixes
        # (the exporter prepends {model_name}_); the `names` shape is what the
        # export-config resolution turns into split_scalars (test_manifest)
        (entry,) = modules["jets_classification"].onnx_outputs()
        assert entry.port == "preds.jets.jets_classification"
        assert entry.names == ["pb", "pc", "pu"]
        assert entry.name is None  # a names-list entry, not a single-name one

    def test_sequence_classification_onnx_outputs_argmax(self, modules):
        # a per-token head emits a single int8 argmax, Pascal-cased name
        (entry,) = modules["track_origin"].onnx_outputs()
        assert entry.name == "TrackOrigin"
        assert entry.reduce == "argmax"
        assert entry.dtype == "int8"
        assert entry.names is None  # single-name entry, not per-class

    def test_vertexing_output_names_bare_by_default(self, modules):
        # v1 byte parity: a single BARE ('VertexIndex', 'i8') column
        assert modules["track_vertexing"].output_names("salt") == [("VertexIndex", "i8")]

    def test_vertexing_output_names_prefix_flag_on_the_task(self, modules):
        # the prefix_vertex_column flag lives on the TASK now (design §8 opt-in)
        import copy

        vtx = copy.deepcopy(modules["track_vertexing"])
        vtx.prefix_vertex_column = True
        assert vtx.output_names("salt") == [("salt_VertexIndex", "i8")]

    def test_vertexing_get_h5_is_v1_int_opchain(self, modules):
        # the EXACT v1 op chain .int().cpu() -> u2s i8 (task.py:988-1005),
        # -inf padded rows read the int32 cast (-2147483648)
        bundle = make_preds_bundle()
        arr = modules["track_vertexing"].get_h5(bundle, "salt")
        assert arr.dtype.names == ("VertexIndex",)
        vertex = bundle.get(PRED_KEYS[2]).int().numpy()[..., 0].astype("i8")
        assert arr["VertexIndex"].tobytes() == vertex.tobytes()
        assert (arr["VertexIndex"][:, -2:] == np.int64(-2147483648)).all()

    def test_vertexing_onnx_outputs_union_find_shared_constant(self, modules):
        # the export suffix is the SAME VertexIndex constant the column uses
        from salt.core.writers import VERTEX_INDEX

        (entry,) = modules["track_vertexing"].onnx_outputs()
        assert entry.name == VERTEX_INDEX
        assert entry.reduce == "vertex_union_find"
        assert entry.dtype == "int8"

    def test_onnx_renameable_policy_lives_on_the_task(self, modules):
        # the writer reads this flag instead of branching on the task type:
        # classification renameable, vertexing not (single ownership §2.2)
        assert modules["jets_classification"].onnx_renameable is True
        assert modules["track_origin"].onnx_renameable is True
        assert modules["track_vertexing"].onnx_renameable is False

    def test_classification_sequence_invariant_matches_reader_streams(self, modules):
        # PINS the equivalence the refactor's ONNX shape selection relies on
        # (critic finding #1): the old writer chose split_scalars-vs-argmax via
        # `module.stream not in ctx.sequence_streams` (a reader-config fact); the
        # task now keys on `self.sequence` (a task-instance attr). For every
        # classification task `self.sequence == (its stream is a sequence
        # stream)`, so the two invariants are provably identical on shipped
        # configs and O2/O3 byte-parity is not an accident. A misconfigured head
        # (explicit `sequence:` override fighting its stream) is the only way to
        # break this pin — exactly the divergence the critic flagged.
        from salt.core.nn.tasks import ClassificationTaskModule

        ctx = declare_ctx(modules)
        cls_tasks = {
            n: m for n, m in modules.items() if isinstance(m, ClassificationTaskModule)
        }
        assert cls_tasks, "fixture must carry classification tasks to pin the invariant"
        for name, module in cls_tasks.items():
            assert module.sequence == (module.stream in ctx.sequence_streams), (
                f"{name}: self.sequence={module.sequence} disagrees with reader "
                f"sequence_streams membership (stream {module.stream!r}) — the ONNX "
                "shape-selection invariants the refactor relies on have diverged"
            )

    def test_base_task_raises_for_a_no_render_family(self, modules):
        # a _TaskModuleBase subclass that ships no rendering inherits the loud
        # unsupported-family ConfigError FROM THE TASK BASE (moved off the
        # writer) — same error regardless of which render method is asked
        from salt.core.nn.tasks import _TaskModuleBase

        bare = _TaskModuleBase.__new__(_TaskModuleBase)
        bare.name = "mystery"
        with pytest.raises(ConfigError, match="ships no TEST columns rendering"):
            bare.output_names("salt")
        with pytest.raises(ConfigError, match="ships no TEST values rendering"):
            bare.get_h5(make_preds_bundle(), "salt")
        with pytest.raises(ConfigError, match="ships no ONNX output rendering"):
            bare.onnx_outputs()


# ---------------------------------------------------------------------------
# TaskWriter is pure ORCHESTRATION: it must carry NO per-family knowledge —
# no isinstance(...TaskModule), no class_names/output_suffixes branching, no
# Flavours/VERTEX_INDEX/pascal_case suffix derivation. The refactor's whole
# point is that adding a task family touches the TASK, never this writer.
# ---------------------------------------------------------------------------


class TestTaskWriterHasNoFamilyKnowledge:
    def test_source_has_no_family_dispatch(self):
        # AST guard (not a naive grep — family names survive in docstrings as
        # legitimate prose): walk the EXECUTABLE nodes of TaskWriter and assert
        # there is no per-family branching left. Banned = isinstance against a
        # task-module class, references to a task-module class name, or to the
        # family-specific suffix machinery the old ladder used. The onnx_names
        # VALUE-shape `isinstance(value, list/str)` check is allowed (it
        # inspects a config value, never a task).
        import ast
        import inspect

        from salt.core.writers import modules as mod

        tree = ast.parse(inspect.getsource(mod.TaskWriter))
        banned_names = {
            "RegressionTaskModule",
            "VertexingTaskModule",
            "ClassificationTaskModule",
            "Flavours",  # the px naming table
            "pascal_case",  # the argmax-name derivation
            "VERTEX_INDEX",  # the vertexing suffix constant
        }
        banned_attrs = {
            "class_names",  # the classification duck-type probe
            "output_suffixes",  # the regression suffix list
            "class_suffixes",
        }
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_names:
                offenders.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in banned_attrs:
                offenders.append(f".{node.attr}")
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "isinstance":
                # allow isinstance(value, list|str) on a config value, ban any
                # isinstance whose first arg is a task `module`
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Name) and first.id == "module":
                    offenders.append("isinstance(module, ...)")
        assert not offenders, f"TaskWriter still carries family dispatch: {sorted(set(offenders))}"

    def test_writer_module_does_not_import_task_classes(self):
        # the import that powered the old isinstance ladder is gone
        import inspect

        from salt.core.writers import modules as mod

        file_src = inspect.getsource(mod)
        assert "from salt.core.nn.tasks import" not in file_src
        assert "from ftag import Flavours" not in file_src

    def test_new_duck_typed_family_works_with_zero_writer_edits(self, data, modules):
        # behavioural proof: a task family the writer has NEVER heard of
        # renders end-to-end purely via its OWN methods. No writer branch
        # could possibly know about `DuckClass` — yet columns/onnx_outputs
        # pick it up through output_names/onnx_outputs.
        from salt.core.onnx.config import ExportOutput

        class DuckTask:
            pred_key = "preds.tracks.duck"
            stream = "tracks"
            name = "duck"

            def output_names(self, run_name):
                return [(f"{run_name}_quack", "f4")]

            def onnx_outputs(self):
                return [ExportOutput(port=self.pred_key, names=["quack"])]

        only_duck = {"duck": DuckTask()}
        cols = TaskWriter().columns(write_ctx(data, only_duck))
        assert list(cols["tracks"].names) == ["salt_quack"]
        entries = TaskWriter().onnx_outputs(declare_ctx(only_duck))
        assert [e.names for e in entries] == [["quack"]]


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
