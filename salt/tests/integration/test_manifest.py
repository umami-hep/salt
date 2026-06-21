"""Unit tests for the M4.5 unified output manifest (amendment-unified-writers.md).

One test class per merge condition plus the derivation itself:

- derivation: the default TaskWriter manifest reproduces v1's hand-built
  output list (the U1(b) full-ordered-list bar at unit scale), entries ARE
  M4 ``ExportOutput`` objects (condition 2), order is global-first
  regardless of module declaration order, and permuting ``class_names``
  moves eval columns AND ONNX suffixes together (the U1(c) negative
  control: drift is impossible by construction).
- condition 1: ``onnx``/``onnx_streams``/``onnx_tasks`` narrowing.
- condition 3: the blessed `ExportOnlyWriter` pattern + the stub bans.
- condition 4: the shared/pinned suffix constants (`names.py`).
- condition 5: ``export.combine``/``export.rename`` with the v1
  combine-before-aux insertion order, end-to-end through a traced export
  (the combine+aux fixture).
- condition 6: ``onnx_names`` typing (str | list per task family).
- demand unification: ONNX sinks == manifest ports via `_model_sinks`.
- the §4.4 annotation: ``salt2 graph resolve [--annotate]``.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch
from torch import nn

from salt.core.cli import main as graph_main
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import Mode
from salt.core.main import CONFIG_DIR
from salt.core.nn import bind_all, map_v1_state_dict, resolve_bind_schema
from salt.core.onnx import (
    ExportCombine,
    ExportOutput,
    attach_manifest,
    check_onnx,
    combine_insertion_index,
    compile_onnx_plan,
    export_graph,
    make_session,
    ordered_output_names,
    resolve_export_config,
)
from salt.core.nn.tasks import RegressionTaskModule
from salt.core.onnx.config import KNOWN_REDUCES
from salt.core.onnx.metadata import build_gnn_config
from salt.core.schema import dump_schema, save_schema
from salt.core.writers import (
    OBJECT_INDEX,
    VERTEX_INDEX,
    ExportOnlyWriter,
    TaskWriter,
    Writer,
    WriterCallback,
    WriterDeclareCtx,
    pascal_case,
)
from salt.tests._fixtures.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
    write_parity_norm_dict,
)
from salt.tests._fixtures.gn2v2_fixture import ORIGIN_CLASSES, build_gn2v2_modules
from salt.tests.unit.onnx.test_adapter import VARIABLES, gn2_export_cfg
from salt.utils.inputs import write_dummy_file

DUMMY_CFG = CONFIG_DIR / "gn2v2-dummy.yaml"
RUN_NAME = "GN2_v2"

V1_OUTPUT_LIST = [
    ("preds.jets.jets_classification", ("pb", "pc", "pu"), "split_scalars", "float32"),
    ("preds.tracks.track_origin", ("TrackOrigin",), "argmax", "int8"),
    ("preds.tracks.track_vertexing", ("VertexIndex",), "vertex_union_find", "int8"),
]
"""The v1 GN2 export-output list (to_onnx.py:258-292) — the U1(b) reference."""


@pytest.fixture(scope="module")
def modules(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("manifest_fixture")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    return build_gn2v2_modules(tmp / "norm_dict.yaml")


def declare_ctx(modules) -> WriterDeclareCtx:
    return WriterDeclareCtx(
        model_modules=modules, streams=("jets", "tracks"), sequence_streams=("tracks",)
    )


def task_writer(**kwargs) -> TaskWriter:
    writer = TaskWriter(**kwargs)
    writer.name = "tasks"
    return writer


def normalised(entries) -> list[tuple]:
    out = []
    for entry in entries:
        resolved = attach_manifest(
            resolve_export_config(gn2_export_cfg(), RUN_NAME), [entry]
        ).outputs[0]
        suffixes = tuple(resolved.names) if resolved.names is not None else (resolved.name,)
        out.append((resolved.port, suffixes, resolved.reduce, resolved.dtype))
    return out


# ---------------------------------------------------------------------------
# the derivation (amendment §2.2; U1(b)/(c) at unit scale)
# ---------------------------------------------------------------------------


class TestManifestDerivation:
    def test_default_taskwriter_reproduces_the_v1_output_list(self, modules):
        # the FULL ordered list, not a set (amendment §7 risk 5)
        derived = task_writer().onnx_outputs(declare_ctx(modules))
        assert normalised(derived) == V1_OUTPUT_LIST

    def test_entries_are_m4_exportoutput_objects(self, modules):
        # merge condition 2: no parallel manifest type
        for entry in task_writer().onnx_outputs(declare_ctx(modules)):
            assert type(entry) is ExportOutput

    def test_global_entries_first_regardless_of_declaration_order(self, modules):
        # v1 output order (to_onnx.py:258-292): aux tasks declared FIRST in
        # the module dict must still emit after the global entries
        reordered = {
            name: modules[name]
            for name in [
                "track_origin",
                "track_vertexing",
                *[n for n in modules if n not in {"track_origin", "track_vertexing"}],
            ]
        }
        ctx = WriterDeclareCtx(
            model_modules=reordered, streams=("jets", "tracks"), sequence_streams=("tracks",)
        )
        derived = task_writer().onnx_outputs(ctx)
        assert [entry.port for entry in derived] == [port for port, _, _, _ in V1_OUTPUT_LIST]

    def test_class_names_permutation_moves_eval_and_onnx_together(self, modules):
        # the U1(c) negative control: ONE helper feeds both modes, so a
        # class_names permutation moves eval columns AND ONNX suffixes —
        # the v1 silent-mislabeling drift is unrepresentable
        import copy

        permuted = dict(modules)
        permuted["jets_classification"] = copy.deepcopy(modules["jets_classification"])
        permuted["jets_classification"].class_names = ["ujets", "bjets", "cjets"]
        writer = task_writer()
        ctx = WriterDeclareCtx(
            model_modules=permuted, streams=("jets", "tracks"), sequence_streams=("tracks",)
        )
        onnx_suffixes = writer.onnx_outputs(ctx)[0].names
        eval_columns = writer.column_manifest(ctx, "run")["jets"]
        assert onnx_suffixes == ["pu", "pb", "pc"]
        assert eval_columns[:3] == ["run_pu", "run_pb", "run_pc"]

    def test_column_manifest_matches_columns_names(self, modules):
        # the §4.4 annotation surface mirrors the binding TEST schema
        writer = task_writer()
        ctx = declare_ctx(modules)
        manifest = writer.column_manifest(ctx, "GN2v2_dummy")
        assert manifest["jets"] == [f"GN2v2_dummy_{s}" for s in ("pb", "pc", "pu")]
        assert manifest["tracks"] == [
            *[f"GN2v2_dummy_p{c}" for c in ORIGIN_CLASSES],
            VERTEX_INDEX,
        ]


# ---------------------------------------------------------------------------
# The single-ownership (amendment §2.2) is now realised ON THE TASK: the same
# task renders BOTH its TEST columns and its ONNX entry from one suffix list.
# These units assert that property directly on the task render methods — the
# writer's old `_class_suffixes`/`_onnx_class_suffixes` helpers are gone.
# ---------------------------------------------------------------------------


class TestTaskOwnsBothModes:
    def test_one_suffix_list_feeds_test_and_onnx(self, modules):
        # the cross-mode single owner is now the task: TEST column suffixes
        # (modulo the run-name prefix) == the ONNX entry suffixes, both from
        # `class_suffixes` — co-located on the task, not derived twice
        task = modules["jets_classification"]
        test_suffixes = [col.removeprefix("run_") for col, _ in task.output_names("run")]
        (onnx_entry,) = task.onnx_outputs()
        assert test_suffixes == onnx_entry.names == task.class_suffixes == ["pb", "pc", "pu"]

    def test_class_names_permutation_on_the_task_moves_both(self, modules):
        # the U1(c) drift control, proved on the TASK: permuting class_names
        # moves the TEST columns AND the ONNX suffixes in lockstep because one
        # list (`class_suffixes`) feeds both render methods
        import copy

        task = copy.deepcopy(modules["jets_classification"])
        task.class_names = ["ujets", "bjets", "cjets"]
        test_cols = [col for col, _ in task.output_names("run")]
        (onnx_entry,) = task.onnx_outputs()
        assert test_cols == ["run_pu", "run_pb", "run_pc"]
        assert onnx_entry.names == ["pu", "pb", "pc"]

    def test_regression_task_renders_its_own_split_scalars(self, modules):
        # the regression family's ONNX entry is rendered BY THE TASK from the
        # same `output_suffixes` (custom_output_names) its TEST columns use —
        # the writer no longer reaches into output_suffixes
        reg = _regression_modules(modules)["jets_regression"]
        test_suffixes = [col.removeprefix("run_") for col, _ in reg.output_names("run")]
        (onnx_entry,) = reg.onnx_outputs()
        assert test_suffixes == onnx_entry.names == ["truthMass", "truthPt"]
        assert onnx_entry.reduce == "split_scalars"


# ---------------------------------------------------------------------------
# condition 1: onnx / onnx_streams / onnx_tasks narrowing
# ---------------------------------------------------------------------------


class TestOnnxNarrowing:
    def test_onnx_false_is_eval_only(self, modules):
        assert task_writer(onnx=False).onnx_outputs(declare_ctx(modules)) == []

    def test_onnx_false_with_narrowing_is_contradictory(self):
        with pytest.raises(ConfigError, match="contradictory"):
            TaskWriter(onnx=False, onnx_streams=["tracks"])
        with pytest.raises(ConfigError, match="contradictory"):
            TaskWriter(onnx=False, onnx_tasks=["track_origin"])

    def test_onnx_streams_narrows_stream_grained(self, modules):
        derived = task_writer(onnx_streams=["jets"]).onnx_outputs(declare_ctx(modules))
        assert [entry.port for entry in derived] == ["preds.jets.jets_classification"]

    def test_onnx_tasks_narrows_task_grained(self, modules):
        # the v1 tasks_to_output expressiveness: export track_vertexing but
        # NOT track_origin — both on stream 'tracks' (merge condition 1)
        derived = task_writer(onnx_tasks=["jets_classification", "track_vertexing"]).onnx_outputs(
            declare_ctx(modules)
        )
        assert [entry.port for entry in derived] == [
            "preds.jets.jets_classification",
            "preds.tracks.track_vertexing",
        ]

    def test_unknown_onnx_stream_errors(self, modules):
        with pytest.raises(ConfigError, match="onnx_streams"):
            task_writer(onnx_streams=["muons"]).onnx_outputs(declare_ctx(modules))

    def test_unknown_onnx_task_errors(self, modules):
        with pytest.raises(ConfigError, match="onnx_tasks"):
            task_writer(onnx_tasks=["track_typo"]).onnx_outputs(declare_ctx(modules))

    def test_test_role_is_untouched_by_onnx_narrowing(self, modules):
        # ONNX narrowing must not change TEST demand (the modes split)
        writer = task_writer(onnx_tasks=["track_vertexing"])
        assert list(writer.requires(declare_ctx(modules))) == [
            "preds.jets.jets_classification",
            "preds.tracks.track_origin",
            "preds.tracks.track_vertexing",
        ]


# ---------------------------------------------------------------------------
# condition 6: onnx_names typing (str | list per task family)
# ---------------------------------------------------------------------------


class TestOnnxNames:
    def test_str_override_renames_an_aux_entry(self, modules):
        derived = task_writer(onnx_names={"track_origin": "TrackLabel"}).onnx_outputs(
            declare_ctx(modules)
        )
        assert derived[1].name == "TrackLabel"

    def test_list_override_replaces_class_suffixes(self, modules):
        derived = task_writer(
            onnx_names={"jets_classification": ["pb", "pcharm", "plight"]}
        ).onnx_outputs(declare_ctx(modules))
        assert derived[0].names == ["pb", "pcharm", "plight"]

    def test_list_length_validated_against_class_names(self, modules):
        with pytest.raises(ConfigError, match="3 classes"):
            task_writer(onnx_names={"jets_classification": ["pb", "pc"]}).onnx_outputs(
                declare_ctx(modules)
            )

    def test_str_for_classification_task_rejected(self, modules):
        with pytest.raises(ConfigError, match="LIST"):
            task_writer(onnx_names={"jets_classification": "probs"}).onnx_outputs(
                declare_ctx(modules)
            )

    def test_list_for_aux_task_rejected(self, modules):
        with pytest.raises(ConfigError, match="single"):
            task_writer(onnx_names={"track_origin": ["a", "b"]}).onnx_outputs(declare_ctx(modules))

    def test_vertexing_rename_rejected(self, modules):
        # the suffix is the shared cross-mode constant (amendment §2.2)
        with pytest.raises(ConfigError, match="VertexIndex"):
            task_writer(onnx_names={"track_vertexing": "SVIndex"}).onnx_outputs(
                declare_ctx(modules)
            )

    def test_unknown_task_rejected(self, modules):
        with pytest.raises(ConfigError, match="names no ONNX-selected task"):
            task_writer(onnx_names={"nope": "X"}).onnx_outputs(declare_ctx(modules))

    def test_pascal_case_default(self):
        assert pascal_case("track_origin") == "TrackOrigin"
        assert pascal_case("track_type") == "TrackType"
        assert pascal_case("electron_origin") == "ElectronOrigin"


# ---------------------------------------------------------------------------
# condition 4: the shared / pinned suffix constants
# ---------------------------------------------------------------------------


class TestSuffixConstants:
    def test_vertex_suffix_is_one_constant_in_both_modes(self, modules):
        # amendment §5 rule 6: one suffix constant; TEST bare while the
        # compat flag is down, prefixed when up; ONNX always prefixed by
        # the exporter — flipping the M7 flag aligns the names
        ctx = declare_ctx(modules)
        onnx_entry = task_writer().onnx_outputs(ctx)[2]
        assert onnx_entry.name == VERTEX_INDEX
        bare = task_writer().column_manifest(ctx, "run")["tracks"][-1]
        assert bare == VERTEX_INDEX
        # prefix_vertex_column moved onto the VertexingTaskModule (the task owns
        # its column naming); flip it on the task, not the writer
        import copy

        prefixed_modules = dict(modules)
        prefixed_modules["track_vertexing"] = copy.deepcopy(modules["track_vertexing"])
        prefixed_modules["track_vertexing"].prefix_vertex_column = True
        prefixed = task_writer().column_manifest(declare_ctx(prefixed_modules), "run")["tracks"][-1]
        assert prefixed == f"run_{VERTEX_INDEX}"

    def test_object_index_divergence_is_pinned(self):
        # merge condition 4: the M5 MaskFormer index pair lands NOW as ONE
        # explicit declaration (v1: MaskIndex eval, HadronIndex ONNX)
        assert OBJECT_INDEX.test == "MaskIndex"
        assert OBJECT_INDEX.onnx == "HadronIndex"
        assert "predictionwriter.py" in OBJECT_INDEX.why


# ---------------------------------------------------------------------------
# M5 sub-wave A: RegressionTaskModule is a supported TaskWriter family in
# BOTH modes (TEST f4 columns + split_scalars ONNX manifest) — the A1
# _validate_writer_roles adjudication consequence (AM dec 4)
# ---------------------------------------------------------------------------


def _regression_modules(modules):
    """Add a GLOBAL regression head (2 targets, custom names) to the module set."""
    reg = RegressionTaskModule(
        stream="jets",
        targets=["mass", "pt"],
        input="pooled.global",
        custom_output_names=["truthMass", "truthPt"],
    )
    reg.name = "jets_regression"
    return dict(modules) | {"jets_regression": reg}


class TestRegressionWriterFamily:
    def test_test_columns_are_f4_per_target_suffix(self, modules):
        # one {run}_{suffix} f4 column per target; suffix = custom name
        writer = task_writer()
        ctx = WriterDeclareCtx(
            model_modules=_regression_modules(modules),
            streams=("jets", "tracks"),
            sequence_streams=("tracks",),
        )
        cols = writer.column_manifest(ctx, "run")["jets"]
        assert cols == ["run_pb", "run_pc", "run_pu", "run_truthMass", "run_truthPt"]

    def test_onnx_manifest_uses_split_scalars_per_target(self, modules):
        # regression joins classification + vertexing as export-representable
        writer = task_writer()
        ctx = WriterDeclareCtx(
            model_modules=_regression_modules(modules),
            streams=("jets", "tracks"),
            sequence_streams=("tracks",),
        )
        entries = writer.onnx_outputs(ctx)
        reg = next(e for e in entries if e.port == "preds.jets.jets_regression")
        assert reg.names == ["truthMass", "truthPt"]
        assert reg.reduce == "split_scalars"
        # global entry emitted before the sequence-stream aux entries
        ports = [e.port for e in entries]
        assert ports.index("preds.jets.jets_regression") < ports.index("preds.tracks.track_origin")

    def test_onnx_false_makes_regression_eval_only(self, modules):
        # a regression task an author declines to export -> empty manifest,
        # non-empty TEST demand (the legal eval-only shape, A1 adjudication)
        writer = task_writer(onnx=False)
        ctx = WriterDeclareCtx(
            model_modules=_regression_modules(modules),
            streams=("jets", "tracks"),
            sequence_streams=("tracks",),
        )
        assert writer.onnx_outputs(ctx) == []
        assert "preds.jets.jets_regression" in writer.requires(ctx)

    def test_onnx_names_rename_rejected_for_regression(self, modules):
        # rename surface is the task's custom_output_names, not onnx_names
        writer = task_writer(onnx_names={"jets_regression": ["a", "b"]})
        ctx = WriterDeclareCtx(
            model_modules=_regression_modules(modules),
            streams=("jets", "tracks"),
            sequence_streams=("tracks",),
        )
        with pytest.raises(ConfigError, match="custom_output_names"):
            writer.onnx_outputs(ctx)


# ---------------------------------------------------------------------------
# condition 3: export-only / eval-only story
# ---------------------------------------------------------------------------


class _MasksOutput(ExportOnlyWriter):
    """The blessed export-only pattern under test (a non-preds port is legal)."""

    def onnx_outputs(self, ctx):
        del ctx
        return [ExportOutput(port="pooled.global", names=["x1", "x2", "x3"])]


class _UnblessedStub(Writer):
    """The cargo-cult stub shape: export entries without the explicit flag."""

    def requires(self, ctx):
        del ctx
        return {}

    def columns(self, ctx):
        del ctx
        return {}

    def write(self, bundle, rows):
        del bundle, rows
        return {}

    def onnx_outputs(self, ctx):
        del ctx
        return [ExportOutput(port="preds.jets.jets_classification", names=["y1", "y2", "y3"])]


class _DoesNothing(Writer):
    def requires(self, ctx):
        del ctx
        return {}

    def columns(self, ctx):
        del ctx
        return {}

    def write(self, bundle, rows):
        del bundle, rows
        return {}


def _reader():
    from salt.core.data import H5StructuredReader

    return H5StructuredReader(groups={"jets": {"global_object": True}, "tracks": {"global_object": False}})


class TestExportOnlyStory:
    def test_blessed_export_only_writer_joins_the_manifest(self, modules):
        cb = WriterCallback(modules={"tasks": TaskWriter(), "masks_out": _MasksOutput()})
        manifest = cb.onnx_manifest(dict(modules), _reader())
        suffixes = [s for e in manifest for s in (e.names or [e.name])]
        assert {"x1", "x2", "x3"} <= set(suffixes)

    def test_export_only_writer_adds_no_test_demand(self, modules):
        cb = WriterCallback(modules={"tasks": TaskWriter(), "masks_out": _MasksOutput()})
        assert cb.per_writer_demand(dict(modules), _reader())["masks_out"] == []

    def test_unblessed_stub_is_rejected(self, modules):
        cb = WriterCallback(modules={"stub": _UnblessedStub()})
        with pytest.raises(ConfigError, match="export-only stub shape"):
            cb.onnx_manifest(dict(modules), _reader())

    def test_does_nothing_writer_is_rejected(self, modules):
        # design principle 10 extended to writers (amendment §4)
        cb = WriterCallback(modules={"noop": _DoesNothing()})
        with pytest.raises(ConfigError, match="must do something"):
            cb.per_writer_demand(dict(modules), _reader())

    def test_export_only_with_test_demand_is_contradictory(self, modules):
        class Contradiction(_MasksOutput):
            def requires(self, ctx):
                del ctx
                return {"meta.rows": None}

        cb = WriterCallback(modules={"bad": Contradiction()})
        with pytest.raises(ConfigError, match="export_only=True but declares TEST"):
            cb.onnx_manifest(dict(modules), _reader())

    def test_export_only_without_manifest_is_rejected(self, modules):
        class Empty(ExportOnlyWriter):
            def onnx_outputs(self, ctx):
                del ctx
                return []

        cb = WriterCallback(modules={"empty": Empty()})
        with pytest.raises(ConfigError, match="no onnx_outputs"):
            cb.per_writer_demand(dict(modules), _reader())

    def test_docstring_example_is_runnable(self, modules):
        # the merge-condition-3 'documented' bar (fix-stage regression): the
        # blessed pattern's worked example must survive its first copy-paste
        # — the pre-fix docstring named a phantom 'registered' reduce that
        # failed manifest resolution. The example is EXEC'd verbatim out of
        # the docstring so doc drift fails here
        import inspect
        import textwrap

        doc = inspect.getdoc(ExportOnlyWriter)
        assert doc is not None
        body = doc.split(".. code-block:: python", 1)[1].splitlines()[1:]
        code_lines = []
        for line in body:
            if line.strip() and not line.startswith("    "):
                break
            code_lines.append(line)
        namespace: dict = {"ExportOnlyWriter": ExportOnlyWriter, "ExportOutput": ExportOutput}
        exec(textwrap.dedent("\n".join(code_lines)), namespace)  # noqa: S102 - the doc example
        echo_cls = namespace["JetEcho"]
        cb = WriterCallback(modules={"tasks": TaskWriter(), "echo": echo_cls()})
        manifest = cb.onnx_manifest(dict(modules), _reader())
        resolved = attach_manifest(resolve_export_config(gn2_export_cfg(), RUN_NAME), manifest)
        names = [name for name, _, _ in ordered_output_names(resolved)]
        assert "GN2v2_jetEcho0" in names
        assert "GN2v2_jetEcho1" in names

    def test_documented_reduce_keys_are_shipped(self):
        # no phantom reduce keys in the writer-base docs: every literal
        # reduce="..." mention must name a registered registry key. The public
        # register_reduce surface has LANDED (M5 sub-wave C/D), so base.py must
        # document it as available, NOT as an unlanded "M5 deliverable".
        import inspect
        import re

        import salt.core.writers.base as writer_base

        source = inspect.getsource(writer_base)
        documented = re.findall(r'reduce="(\w+)"', source)
        assert all(key in KNOWN_REDUCES for key in documented), documented
        # the honest re-scope: base.py points at the landed register_reduce
        # surface and no longer calls it an unlanded M5 deliverable
        assert "register_reduce" in source
        assert "M5 deliverable" not in source


# ---------------------------------------------------------------------------
# the flat-namespace collision check (amendment §5 rule 4)
# ---------------------------------------------------------------------------


class _CollidingSuffix(ExportOnlyWriter):
    def onnx_outputs(self, ctx):
        del ctx
        # a DISTINCT port whose suffix collides with the TaskWriter's 'pb'
        # (two streams may share H5 column names; the flat ONNX namespace
        # cannot — amendment §5 rule 4)
        return [ExportOutput(port="encoded.tracks", name="pb", reduce="argmax", dtype="int8")]


class _CollidingPort(ExportOnlyWriter):
    def onnx_outputs(self, ctx):
        del ctx
        return [ExportOutput(port="preds.jets.jets_classification", names=["q1", "q2", "q3"])]


class TestFlatNamespaceCollisions:
    def test_suffix_collision_names_both_writers_and_the_fix(self, modules):
        cb = WriterCallback(modules={"tasks": TaskWriter(), "other": _CollidingSuffix()})
        with pytest.raises(ConfigError) as excinfo:
            cb.onnx_manifest(dict(modules), _reader())
        message = str(excinfo.value)
        assert "'pb'" in message
        assert "'tasks'" in message and "'other'" in message
        assert "onnx_names" in message

    def test_port_collision_names_both_writers(self, modules):
        cb = WriterCallback(modules={"tasks": TaskWriter(), "other": _CollidingPort()})
        with pytest.raises(ConfigError, match="one writer owns one export port"):
            cb.onnx_manifest(dict(modules), _reader())


# ---------------------------------------------------------------------------
# condition 5: rename/combine post-processing + the v1 insertion order,
# end-to-end through a traced export (the combine+aux fixture)
# ---------------------------------------------------------------------------


class TestRenameCombine:
    def manifest(self, modules):
        return task_writer().onnx_outputs(declare_ctx(modules))

    def test_combine_inserts_before_aux_entries(self, modules):
        # the v1 rule (to_onnx.py:258-292): after the global entries,
        # BEFORE the first per-token aux entry — NOT appended at the end
        cfg = gn2_export_cfg(combine=[ExportCombine(name="pbc", inputs={"pb": 0.5, "pc": 0.5})])
        resolved = attach_manifest(resolve_export_config(cfg, RUN_NAME), self.manifest(modules))
        assert combine_insertion_index(resolved.outputs) == 1
        assert [name for name, _, _ in ordered_output_names(resolved)] == [
            "GN2v2_pb",
            "GN2v2_pc",
            "GN2v2_pu",
            "GN2v2_pbc",  # combine: after globals, before aux
            "GN2v2_TrackOrigin",
            "GN2v2_VertexIndex",
        ]

    def test_combine_without_aux_appends_at_the_end(self, modules):
        cfg = gn2_export_cfg(combine=[ExportCombine(name="pbc", inputs={"pb": 1.0})])
        manifest = [entry for entry in self.manifest(modules) if entry.names is not None]
        resolved = attach_manifest(resolve_export_config(cfg, RUN_NAME), manifest)
        names = [name for name, _, _ in ordered_output_names(resolved)]
        assert names[-1] == "GN2v2_pbc"

    def test_combine_inputs_must_be_global_suffixes(self, modules):
        cfg = gn2_export_cfg(combine=[ExportCombine(name="bad", inputs={"TrackOrigin": 1.0})])
        with pytest.raises(ConfigError, match="not global float outputs"):
            attach_manifest(resolve_export_config(cfg, RUN_NAME), self.manifest(modules))

    def test_combine_name_collision_rejected(self, modules):
        cfg = gn2_export_cfg(combine=[ExportCombine(name="pb", inputs={"pc": 1.0})])
        with pytest.raises(ConfigError, match="collides"):
            attach_manifest(resolve_export_config(cfg, RUN_NAME), self.manifest(modules))

    def test_rename_applies_before_combine(self, modules):
        # v1 order (to_onnx.py:263-279): renames first, combines reference
        # the renamed suffixes
        cfg = gn2_export_cfg(
            rename={"pu": "plight"},
            combine=[ExportCombine(name="pall", inputs={"plight": 1.0, "pb": 1.0})],
        )
        resolved = attach_manifest(resolve_export_config(cfg, RUN_NAME), self.manifest(modules))
        names = [name for name, _, _ in ordered_output_names(resolved)]
        assert "GN2v2_plight" in names and "GN2v2_pall" in names
        assert "GN2v2_pu" not in names

    def test_rename_of_missing_suffix_rejected(self, modules):
        cfg = gn2_export_cfg(rename={"nope": "new"})
        with pytest.raises(ConfigError, match="matches no manifest output"):
            attach_manifest(resolve_export_config(cfg, RUN_NAME), self.manifest(modules))

    def test_metadata_records_v1_combine_and_rename_format(self, modules):
        cfg = gn2_export_cfg(
            rename={"pu": "plight"},
            combine=[ExportCombine(name="pbc", inputs={"pb": 0.5, "pc": 0.5})],
        )
        resolved = attach_manifest(resolve_export_config(cfg, RUN_NAME), self.manifest(modules))
        meta = build_gnn_config(resolved, VARIABLES, [], {}, {}, None, "hash")
        # v1 (name, [(scale, input), ...]) tuples, JSON-identical as lists
        assert meta["combine_outputs"] == [["pbc", [[0.5, "pb"], [0.5, "pc"]]]]
        assert meta["rename_outputs"] == {"pu": "plight"}


@pytest.fixture(scope="module")
def combine_aux_export(tmp_path_factory, modules):
    """The condition-5 fixture: a traced export carrying combine AND aux outputs."""
    tmp = tmp_path_factory.mktemp("combine_aux")
    v1 = build_test_gn2(tmp)  # weight source (norm dict regenerated alongside)
    local_modules = build_gn2v2_modules(tmp / "norm_dict.yaml")
    manifest = TaskWriter().onnx_outputs(
        WriterDeclareCtx(
            model_modules=local_modules, streams=("jets", "tracks"), sequence_streams=("tracks",)
        )
    )
    cfg = gn2_export_cfg(combine=[ExportCombine(name="pbc", inputs={"pb": 0.5, "pc": 0.5})])
    resolved = attach_manifest(resolve_export_config(cfg, RUN_NAME), manifest)
    plan = compile_onnx_plan(local_modules, resolved, VARIABLES)
    bind_all(local_modules, resolve_bind_schema([plan]))
    nn.ModuleDict(local_modules).load_state_dict(map_v1_state_dict(v1.state_dict(), local_modules))
    result = export_graph(
        local_modules,
        cfg,
        VARIABLES,
        tmp / "combine_aux.onnx",
        outputs=manifest,
        run_name=RUN_NAME,
    )
    return result


class TestCombineAuxExport:
    def test_traced_output_order_is_combine_before_aux(self, combine_aux_export):
        session = make_session(combine_aux_export.onnx_path)
        assert [out.name for out in session.get_outputs()] == [
            "GN2v2_pb",
            "GN2v2_pc",
            "GN2v2_pu",
            "GN2v2_pbc",
            "GN2v2_TrackOrigin",
            "GN2v2_VertexIndex",
        ]
        assert session.get_outputs()[3].type == "tensor(float)"

    def test_combined_value_is_the_linear_combination(self, combine_aux_export):
        session = make_session(combine_aux_export.onnx_path)
        gen = torch.Generator().manual_seed(11)
        feed = {
            "jet_features": torch.rand(1, len(JET_VARIABLES), generator=gen).numpy(),
            "track_features": torch.rand(9, len(TRACK_VARIABLES), generator=gen).numpy(),
        }
        outputs = dict(
            zip([o.name for o in session.get_outputs()], session.run(None, feed), strict=True)
        )
        np.testing.assert_allclose(
            outputs["GN2v2_pbc"],
            0.5 * outputs["GN2v2_pb"] + 0.5 * outputs["GN2v2_pc"],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_torch_vs_onnx_agreement_with_combine(self, combine_aux_export):
        result = check_onnx(
            combine_aux_export.adapter,
            combine_aux_export.onnx_path,
            trials=2,
            float_rtol=1e-6,
            float_atol=1e-6,
            lengths_grid=[{"tracks": length} for length in (0, 3, 17)],
        )
        assert result.passed, result.failures

    def test_metadata_output_names_match_the_graph(self, combine_aux_export):
        import json

        session = make_session(combine_aux_export.onnx_path)
        info = json.loads(session.get_modelmeta().custom_metadata_map["gnn_config"])
        assert info["output_names"] == [out.name for out in session.get_outputs()]
        assert info["combine_outputs"] == [["pbc", [[0.5, "pb"], [0.5, "pc"]]]]


# ---------------------------------------------------------------------------
# demand unification: ONNX sinks == manifest ports (amendment §4)
# ---------------------------------------------------------------------------


class TestDemandUnification:
    def test_model_sinks_onnx_sources_from_the_manifest(self, modules):
        from salt.core.saltmodule import SaltModule

        model = SaltModule(
            dict(modules),
            lrs={"initial": 1e-7, "max": 1e-3, "end": 1e-5, "pct_start": 0.01},
            name=RUN_NAME,
        )
        cb = WriterCallback(modules={"tasks": TaskWriter(onnx_tasks=["jets_classification"])})
        sinks = model._model_sinks(Mode.ONNX, writers=cb, reader=_reader())
        assert sinks == ["preds.jets.jets_classification"]

    def test_onnx_demand_is_the_manifest_port_set(self, modules):
        cb = WriterCallback(modules={"tasks": TaskWriter()})
        manifest = cb.onnx_manifest(dict(modules), _reader())
        assert {entry.port for entry in manifest} == {
            "preds.jets.jets_classification",
            "preds.tracks.track_origin",
            "preds.tracks.track_vertexing",
        }


# ---------------------------------------------------------------------------
# the §4.4 annotation: salt2 graph resolve [--annotate]
# ---------------------------------------------------------------------------


SET_NORM = ["--set", "model.modules.norm.init_args.norm_dict=unused.yaml"]


class TestResolveAnnotation:
    def test_resolve_prints_eval_and_onnx_manifests(self, capsys):
        rc = graph_main(["graph", "resolve", "-c", str(DUMMY_CFG), *SET_NORM])
        assert rc == 0
        out = capsys.readouterr().out
        assert "eval columns" in out
        assert "onnx outputs" in out
        # one logical declaration, two prefixes (amendment §5)
        assert "GN2v2_dummy_pb" in out  # TEST: run-name prefix
        assert "GN2v2dummy_pb" in out  # ONNX: model_name prefix
        assert "GN2v2dummy_VertexIndex" in out
        assert "VertexIndex" in out  # bare TEST column (compat flag down)

    def test_annotate_writes_and_refreshes_the_block(self, tmp_path, capsys):
        target = tmp_path / "annotated.yaml"
        shutil.copy(DUMMY_CFG, target)
        assert graph_main(["graph", "resolve", "-c", str(target), "--annotate", *SET_NORM]) == 0
        text = target.read_text()
        assert text.count("# === salt2 output manifest") == 1
        assert "GN2v2dummy_TrackOrigin" in text
        # idempotent: a second --annotate run REPLACES the block
        capsys.readouterr()
        assert graph_main(["graph", "resolve", "-c", str(target), "--annotate", *SET_NORM]) == 0
        text2 = target.read_text()
        assert text2.count("# === salt2 output manifest") == 1
        # the annotated config still parses (comments are inert YAML)
        assert graph_main(["graph", "resolve", "-c", str(target), *SET_NORM]) == 0

    def test_resolve_rejects_toy_configs(self, tmp_path, capsys):
        toy = tmp_path / "toy.yaml"
        toy.write_text("modules: {m: {class_path: x.Y}}\n")
        rc = graph_main(["graph", "resolve", "-c", str(toy)])
        assert rc == 1
        assert "trainer config" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# config stacking on the static tooling (-c repeatable, fix stage) + the
# strict ONNX-narrowing contract (info-level deadcode, README §4.2 story)
# ---------------------------------------------------------------------------


AUX_TASK_OVERRIDE = """\
model:
  init_args:
    modules:
      track_type:
        class_path: salt.core.nn.tasks.ClassificationTaskModule
        init_args:
          stream: tracks
          context: pooled.global
          label: ftagTruthOriginLabel
          class_names: [a, b, c]
          dense: {hidden_layers: [16], activation: ReLU}
"""

NARROWED_WRITER_OVERRIDE = """\
writers:
  modules:
    tasks:
      class_path: salt.core.writers.TaskWriter
      init_args: {onnx_tasks: [jets_classification, track_vertexing]}
"""


class TestConfigStacking:
    def test_resolve_stacks_override_configs(self, tmp_path, capsys):
        # the README base+override journey through the static tooling
        # (fix-stage regression: -c was single-valued argparse, silently
        # last-wins, locking the recommended pattern out of --annotate)
        override = tmp_path / "my_aux_task.yaml"
        override.write_text(AUX_TASK_OVERRIDE)
        rc = graph_main(["graph", "resolve", "-c", str(DUMMY_CFG), "-c", str(override), *SET_NORM])
        assert rc == 0
        out = capsys.readouterr().out
        # deep-merge, not last-wins: the base manifest survives AND the new
        # aux task lands in eval + ONNX with zero extra writer config
        assert "GN2v2dummy_pb" in out
        assert "GN2v2dummy_TrackType" in out

    def test_annotate_with_stacked_configs_targets_the_last_file(self, tmp_path, capsys):
        base_copy = tmp_path / "base.yaml"
        shutil.copy(DUMMY_CFG, base_copy)
        override = tmp_path / "my_aux_task.yaml"
        override.write_text(AUX_TASK_OVERRIDE)
        argv = ["graph", "resolve", "-c", str(base_copy), "-c", str(override), *SET_NORM]
        assert graph_main([*argv, "--annotate"]) == 0
        assert "# === salt2 output manifest" not in base_copy.read_text()
        annotated = override.read_text()
        assert annotated.count("# === salt2 output manifest") == 1
        # the embedded refresh hint reproduces the FULL generating command
        # (every -c and every --set) so it re-runs verbatim (fix-stage
        # regression: the hint used to drop --set and fail on configs with
        # required init_args)
        assert "-c base.yaml -c my_aux_task.yaml" in annotated
        assert "--set model.modules.norm.init_args.norm_dict=unused.yaml" in annotated
        # the annotated override still parses and stacks (idempotent refresh)
        capsys.readouterr()
        assert graph_main([*argv, "--annotate"]) == 0
        assert override.read_text().count("# === salt2 output manifest") == 1

    def test_two_toy_configs_rejected(self, tmp_path, capsys):
        toy = tmp_path / "toy.yaml"
        toy.write_text("modules: {m: {class_path: x.Y}}\n")
        rc = graph_main(["graph", "validate", "-c", str(toy), "-c", str(toy)])
        assert rc == 1
        assert "trainer configs only" in capsys.readouterr().err


class TestNarrowedStrictValidate:
    def test_narrowed_config_passes_strict_onnx_validate(self, tmp_path, capsys):
        # the documented CI default (README static-tooling section): a
        # legitimately narrowed config (onnx_tasks — the exact shape every
        # M7-converted config with aux heads will carry) must pass
        # `validate --strict --mode onnx`, with the narrowing reported as an
        # info-level deadcode finding (fix-stage regression: ONNX
        # pruned-module findings were warning-level and --strict promoted
        # them). Real norm dict + schema artifact so no unrelated warnings.
        norm = tmp_path / "norm_dict.yaml"
        write_parity_norm_dict(norm, tmp_path / "class_dict.yaml")
        h5 = tmp_path / "dummy.h5"
        write_dummy_file(h5, norm)
        schema = tmp_path / "schema.yaml"
        save_schema(dump_schema(h5), schema)
        override = tmp_path / "narrowed.yaml"
        override.write_text(NARROWED_WRITER_OVERRIDE)
        rc = graph_main([
            "graph",
            "validate",
            "-c",
            str(DUMMY_CFG),
            "-c",
            str(override),
            "--mode",
            "onnx",
            "--strict",
            "--set",
            f"model.modules.norm.init_args.norm_dict={norm}",
            "--set",
            f"data.modules.reader.init_args.schema={schema}",
        ])
        captured = capsys.readouterr()
        assert rc == 0, f"stdout:\n{captured.out}\nstderr:\n{captured.err}"
        assert "info:" in captured.out
        assert "track_origin" in captured.out
