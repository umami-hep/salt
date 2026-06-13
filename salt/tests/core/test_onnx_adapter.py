"""Unit tests for the M4 export surface: config resolution, reduces, adapter (plan 07 stage A).

Covers the static pieces — `resolve_export_config` validation/defaulting
(incl. the export-only model-name rule, design §7 "Naming"), the reduce
registry (design §7.3) on hand-made bundles, the `OnnxAdapter` input
construction and alias gather, and the ONNX-plan purity guarantee (no
labels/losses/writers, design §7). Tracing/onnxruntime agreement lives in
``test_onnx_export.py``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from salt.core.graph import Bundle, Mode
from salt.core.graph.errors import ConfigError, ConnectivityError, ShapeError
from salt.core.graph.spec import TensorSpec
from salt.core.nn import (
    Concat,
    GlobalAttentionPooling,
    Normaliser,
    Split,
    StreamEmbed,
    TransformerEncoder,
    bind_all,
    map_v1_state_dict,
    resolve_bind_schema,
)
from salt.core.nn.tasks import ClassificationTaskModule
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    ExportOutput,
    OnnxAdapter,
    attach_manifest,
    compile_onnx_plan,
    derive_onnx_sources,
    resolve_export_config,
    sanitised_model_name,
    validate_model_name,
)
from salt.core.onnx.config import _resolve_output, default_athena_name
from salt.core.onnx.reduces import (
    BoundReduce,
    ReduceCtx,
    bind_reduce,
    per_token_reduces,
    reduce_dtype,
    register_reduce,
    registered_reduces,
)
from salt.models.task import mask_fill_flattened
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import build_gn2v2_modules
from salt.utils.union_find import get_node_assignment_jit

VARIABLES = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}


def gn2_export_cfg(**overrides) -> ExportConfig:
    # export-only half (M4.5) — outputs come from gn2_manifest()/the writers
    cfg = ExportConfig(
        model_name="GN2v2",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
        ],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def gn2_manifest() -> list[ExportOutput]:
    # the GN2 output manifest, as the default TaskWriter derives it (the
    # writer-derivation itself is covered in test_manifest.py — these unit
    # tests exercise the resolution/adapter mechanics on the same entries)
    return [
        ExportOutput(port="preds.jets.jets_classification", names=["pb", "pc", "pu"]),
        ExportOutput(
            port="preds.tracks.track_origin", name="TrackOrigin", dtype="int8", reduce="argmax"
        ),
        ExportOutput(
            port="preds.tracks.track_vertexing",
            name="VertexIndex",
            dtype="int8",
            reduce="vertex_union_find",
        ),
    ]


def gn2_resolved(run_name: str = "GN2_v2", manifest=None, **overrides) -> ExportConfig:
    return attach_manifest(
        resolve_export_config(gn2_export_cfg(**overrides), run_name),
        gn2_manifest() if manifest is None else manifest,
    )


# ---------------------------------------------------------------------------
# config resolution + validation (design §7 "Naming", §5.1)
# ---------------------------------------------------------------------------


class TestModelName:
    def test_sanitised_default_reproduces_v1(self):
        # to_onnx.py:687: config['name'].replace('_','').replace('-','')
        assert sanitised_model_name("GN2_v2") == "GN2v2"
        assert sanitised_model_name("GN3-pileup_study") == "GN3pileupstudy"

    def test_underscore_rejected(self):
        with pytest.raises(ConfigError, match="underscores or dashes"):
            validate_model_name("GN2_v2")

    def test_dash_rejected(self):
        with pytest.raises(ConfigError, match="underscores or dashes"):
            validate_model_name("GN2-v2")

    def test_empty_rejected(self):
        with pytest.raises(ConfigError, match="non-empty"):
            validate_model_name("")

    def test_default_from_run_name(self):
        resolved = resolve_export_config(gn2_export_cfg(model_name=None), "GN2_v2")
        assert resolved.model_name == "GN2v2"

    def test_run_name_itself_never_validated(self):
        # a run name with underscores is fine — only the export name is checked
        resolved = resolve_export_config(gn2_export_cfg(model_name="Clean"), "any_run-name")
        assert resolved.model_name == "Clean"


class TestResolveInputs:
    def test_defaults_filled(self):
        cfg = ExportConfig(
            model_name="M",
            inputs=[
                ExportInput(port="inputs.jets"),
                ExportInput(port="inputs.tracks", sequence=True),
            ],
        )
        resolved = resolve_export_config(cfg, "m")
        jets, tracks = resolved.inputs
        assert jets.name == "jet_features"  # v1 name_athena_out (to_onnx.py:508)
        assert jets.dyn_axis is None
        assert jets.athena_name == "jet_var"  # to_onnx.py:507
        assert tracks.name == "track_features"
        assert tracks.dyn_axis == "n_tracks"  # v1 athena_num_name (to_onnx.py:519)
        assert tracks.athena_name == "tracks_r22default_sd0sort"

    def test_athena_name_rules(self):
        # the v1 get_default_onnx_feature_map if-ladder (to_onnx.py:475-553)
        assert default_athena_name("jets", False, "r22default") == "jet_var"
        assert default_athena_name("tracks", True, "ip3d") == "tracks_ip3d_sd0sort"
        assert default_athena_name("tracks_loose", True, "r22loose") == "tracks_r22loose_sd0sort"
        assert default_athena_name("flows", True, "r22default") == "flows_r22default_sd0sort"
        assert default_athena_name("flow", True, "r22default") == "flows_r22default_sd0sort"
        assert default_athena_name("electrons", True, "r22default") == "electrons_r22default"
        assert default_athena_name("muons", True, "r22default") == "muons_var"

    def test_bad_port_namespace(self):
        cfg = gn2_export_cfg()
        cfg.inputs[0].port = "normed.jets"
        with pytest.raises(ConfigError, match="inputs.<stream>"):
            resolve_export_config(cfg, "m")

    def test_dyn_axis_on_global_rejected(self):
        cfg = gn2_export_cfg()
        cfg.inputs[0].dyn_axis = "n_jets"
        with pytest.raises(ConfigError, match="not a sequence"):
            resolve_export_config(cfg, "m")

    def test_alias_with_name_rejected(self):
        cfg = gn2_export_cfg()
        cfg.inputs.append(ExportInput(port="inputs.global", alias="inputs.jets", name="x"))
        with pytest.raises(ConfigError, match="alias entries"):
            resolve_export_config(cfg, "m")

    def test_sequence_alias_rejected(self):
        cfg = gn2_export_cfg()
        cfg.inputs.append(ExportInput(port="inputs.global", alias="inputs.tracks", sequence=True))
        with pytest.raises(ConfigError, match="sequence alias"):
            resolve_export_config(cfg, "m")

    def test_alias_source_must_be_declared(self):
        cfg = gn2_export_cfg()
        cfg.inputs.append(ExportInput(port="inputs.global", alias="inputs.muons"))
        with pytest.raises(ConfigError, match="not another export input"):
            resolve_export_config(cfg, "m")

    def test_duplicate_port_rejected(self):
        cfg = gn2_export_cfg()
        cfg.inputs.append(ExportInput(port="inputs.jets", name="other"))
        with pytest.raises(ConfigError, match="twice"):
            resolve_export_config(cfg, "m")

    def test_bad_track_selection(self):
        with pytest.raises(ConfigError, match="track selection"):
            resolve_export_config(gn2_export_cfg(track_selection="nope"), "m")

    def test_empty_inputs_rejected(self):
        with pytest.raises(ConfigError, match="export.inputs"):
            resolve_export_config(gn2_export_cfg(inputs=[]), "m")


class TestResolveOutputs:
    # since M4.5 outputs are writer-derived and resolved via attach_manifest
    # (the M4 per-entry rules unchanged)

    def test_config_declared_outputs_are_the_migration_hard_error(self):
        # the M4.5 §4.1-bar error: export.outputs is gone; the message must
        # point at the writers and the inspection tooling
        cfg = gn2_export_cfg(outputs=gn2_manifest())
        with pytest.raises(ConfigError, match="REMOVED by the M4.5") as excinfo:
            resolve_export_config(cfg, "m")
        message = str(excinfo.value)
        assert "writers" in message
        assert "salt2 export --manifest" in message
        assert "fix:" in message

    def test_name_and_names_exclusive(self):
        manifest = gn2_manifest()
        manifest[0].name = "both"
        with pytest.raises(ConfigError, match="exactly one of"):
            gn2_resolved(manifest=manifest)

    def test_single_name_needs_explicit_reduce(self):
        manifest = gn2_manifest()
        manifest[1].reduce = None
        with pytest.raises(ConfigError, match="explicit 'reduce'"):
            gn2_resolved(manifest=manifest)

    def test_names_imply_split_scalars(self):
        resolved = gn2_resolved()
        assert resolved.outputs[0].reduce == "split_scalars"
        assert resolved.outputs[0].dtype == "float32"
        manifest = gn2_manifest()
        manifest[0].reduce = "argmax"
        with pytest.raises(ConfigError, match="split_scalars"):
            gn2_resolved(manifest=manifest)

    def test_aux_dtype_is_int8(self):
        resolved = gn2_resolved()
        assert resolved.outputs[1].dtype == "int8"
        manifest = gn2_manifest()
        manifest[1].dtype = "float32"
        with pytest.raises(ConfigError, match="int8"):
            gn2_resolved(manifest=manifest)

    def test_unknown_reduce(self):
        manifest = gn2_manifest()
        manifest[1].reduce = "not_a_registered_reduce"  # never registered
        with pytest.raises(ConfigError, match="unknown reduce"):
            gn2_resolved(manifest=manifest)

    def test_duplicate_output_name_rejected(self):
        manifest = gn2_manifest()
        manifest[1].name = "pb"  # collides with the split_scalars suffix
        with pytest.raises(ConfigError, match="twice"):
            gn2_resolved(manifest=manifest)

    def test_empty_manifest_rejected(self):
        with pytest.raises(ConfigError, match="no ONNX outputs"):
            gn2_resolved(manifest=[])


# ---------------------------------------------------------------------------
# reduces (design §7.3) on hand-made bundles
# ---------------------------------------------------------------------------


def make_ctx(**produced) -> ReduceCtx:
    return ReduceCtx(
        model_name="M",
        seq_dyn_axis={"tracks": "n_tracks"},
        produced_specs={key: TensorSpec(shape=shape) for key, shape in produced.items()},
    )


class TestReduces:
    def test_split_scalars(self):
        out = ExportOutput(
            port="preds.jets.c", names=["pb", "pc", "pu"], reduce="split_scalars", dtype="float32"
        )
        bound = bind_reduce(out, make_ctx(**{"preds.jets.c": ("B", 3)}))
        assert bound.output_names == ("M_pb", "M_pc", "M_pu")
        assert bound.dtypes == ("float32",) * 3
        assert bound.dynamic_axes == {}
        b = Bundle()
        probs = torch.tensor([[0.5, 0.3, 0.2]])
        b.set("preds.jets.c", probs)
        scalars = bound.fn(b)
        assert len(scalars) == 3
        # v1 split+squeeze (task.py:301): 0-dim scalars
        assert all(s.dim() == 0 for s in scalars)
        assert torch.allclose(torch.stack(scalars), probs.squeeze(0))

    def test_split_scalars_class_count_mismatch(self):
        out = ExportOutput(
            port="preds.jets.c", names=["pb", "pc"], reduce="split_scalars", dtype="float32"
        )
        with pytest.raises(ConfigError, match="2 names"):
            bind_reduce(out, make_ctx(**{"preds.jets.c": ("B", 3)}))

    @pytest.mark.parametrize("length", [0, 1, 7])
    def test_argmax_matches_plain_argmax(self, length):
        out = ExportOutput(port="preds.tracks.o", name="TrackOrigin", reduce="argmax", dtype="int8")
        bound = bind_reduce(out, make_ctx(**{"preds.tracks.o": None}))
        assert bound.output_names == ("M_TrackOrigin",)
        assert bound.dynamic_axes == {"M_TrackOrigin": {0: "n_tracks"}}
        gen = torch.Generator().manual_seed(3)
        scores = torch.rand(1, length, 8, generator=gen)
        b = Bundle()
        b.set("preds.tracks.o", scores)
        (got,) = bound.fn(b)
        assert got.dtype == torch.int8
        assert got.shape == (length,)
        # the zero-row append/strip (to_onnx.py:415-423) never changes values
        expected = torch.argmax(scores, dim=-1).squeeze(0).char()
        assert torch.equal(got, expected.reshape(-1))

    def test_vertex_union_find_matches_v1_chain(self):
        out = ExportOutput(
            port="preds.tracks.v", name="VertexIndex", reduce="vertex_union_find", dtype="int8"
        )
        bound = bind_reduce(out, make_ctx(**{"preds.tracks.v": None}))
        length = 5
        gen = torch.Generator().manual_seed(4)
        scores = torch.randn(length * (length - 1), 1, generator=gen)
        mask = torch.zeros((1, length), dtype=torch.bool)
        b = Bundle()
        b.set("preds.tracks.v", scores)
        b.set("masks.tracks", mask)
        (got,) = bound.fn(b)
        expected = (
            mask_fill_flattened(get_node_assignment_jit(scores, mask), mask).reshape(-1).char()
        )
        assert got.dtype == torch.int8
        assert torch.equal(got, expected)

    def test_aux_reduce_needs_sequence_stream(self):
        out = ExportOutput(port="preds.jets.o", name="X", reduce="argmax", dtype="int8")
        with pytest.raises(ConfigError, match="no sequence entry"):
            bind_reduce(out, make_ctx(**{"preds.jets.o": None}))

    def test_aux_reduce_needs_preds_port(self):
        out = ExportOutput(port="pooled.global", name="X", reduce="argmax", dtype="int8")
        with pytest.raises(ConfigError, match="preds.<stream>.<task>"):
            bind_reduce(out, make_ctx(**{"pooled.global": None}))


# ---------------------------------------------------------------------------
# the MaskFormer object reduces (M5 sub-wave C) vs the v1 get_maskformer_outputs
# ---------------------------------------------------------------------------


def _mf_object_bundle(b: int = 1, m: int = 5, n_cls: int = 3, t: int = 8, seed: int = 9) -> Bundle:
    """A bundle carrying the decoder object keys the MaskFormer reduces read."""
    gen = torch.Generator().manual_seed(seed)
    bundle = Bundle()
    bundle.set("objects.class_probs", torch.randn(b, m, n_cls, generator=gen).softmax(-1))
    bundle.set("objects.masks", torch.randn(b, m, t, generator=gen))
    bundle.set("preds.objects.regression", torch.randn(b, m, 3, generator=gen))
    return bundle


class TestMaskFormerReduces:
    def _ctx(self):
        return ReduceCtx(
            model_name="MFv2",
            seq_dyn_axis={"tracks": "n_tracks"},
            produced_specs={"preds.objects.regression": TensorSpec(shape=("B", 5, 3))},
        )

    def test_leading_object_matches_v1_helper(self):
        from salt.models.maskformer import get_maskformer_outputs

        out = ExportOutput(
            port="preds.objects.regression",
            names=["lead_pt", "lead_Lxy", "lead_mass"],
            reduce="leading_object",
        )
        bound = bind_reduce(out, self._ctx())
        assert bound.output_names == ("MFv2_lead_pt", "MFv2_lead_Lxy", "MFv2_lead_mass")
        assert bound.dtypes == ("float32",) * 3
        assert bound.dynamic_axes == {}  # GLOBAL scalars
        b = _mf_object_bundle()
        scalars = bound.fn(b)
        # v1 op chain on a CLONE (the reduce must not mutate the bundle)
        objs = {
            "class_probs": b.get("objects.class_probs").clone(),
            "masks": b.get("objects.masks").clone(),
            "regression": b.get("preds.objects.regression").clone(),
        }
        v1_leading, _, _, _ = get_maskformer_outputs(objs, apply_reorder=True)
        for i, s in enumerate(scalars):
            assert s.dim() == 0  # 0-dim scalar per target (v1 leading_reg[0][i])
            assert torch.equal(s, v1_leading[0, i]) or (
                torch.isnan(s) and torch.isnan(v1_leading[0, i])
            )

    def test_leading_object_does_not_mutate_the_bundle(self):
        out = ExportOutput(
            port="preds.objects.regression", names=["a", "b", "c"], reduce="leading_object"
        )
        bound = bind_reduce(out, self._ctx())
        b = _mf_object_bundle()
        before = b.get("preds.objects.regression").clone()
        bound.fn(b)  # v1's get_maskformer_outputs mutates in place — the reduce clones
        assert torch.equal(b.get("preds.objects.regression"), before)

    def test_object_index_matches_v1_helper(self):
        from salt.models.maskformer import get_maskformer_outputs

        out = ExportOutput(
            port="objects.masks", name="HadronIndex", reduce="object_index", dtype="int8"
        )
        bound = bind_reduce(out, self._ctx())
        assert bound.output_names == ("MFv2_HadronIndex",)
        assert bound.dtypes == ("int8",)
        assert bound.dynamic_axes == {"MFv2_HadronIndex": {0: "n_tracks"}}  # PER-TOKEN
        b = _mf_object_bundle()
        (got,) = bound.fn(b)
        objs = {
            "class_probs": b.get("objects.class_probs").clone(),
            "masks": b.get("objects.masks").clone(),
            "regression": b.get("preds.objects.regression").clone(),
        }
        _, v1_indices, _, _ = get_maskformer_outputs(objs, apply_reorder=True)
        assert got.dtype == torch.int8
        assert torch.equal(got, v1_indices.reshape(-1).char())  # v1 to_onnx.py:469

    def test_object_index_needs_one_sequence_stream(self):
        out = ExportOutput(
            port="objects.masks", name="HadronIndex", reduce="object_index", dtype="int8"
        )
        ctx = ReduceCtx(model_name="MFv2", seq_dyn_axis={}, produced_specs={})
        with pytest.raises(ConfigError, match="EXACTLY one sequence"):
            bind_reduce(out, ctx)


# ---------------------------------------------------------------------------
# the LIVE register_reduce surface (M5 D-prereq; AM 555-567)
# ---------------------------------------------------------------------------


def _bind_passthrough_int8(out_cfg, ctx):
    """A toy single-output reduce: flatten the port to int8 (registration target).

    Returns
    -------
    BoundReduce
        The bound toy reduce.
    """
    name = f"{ctx.model_name}_{out_cfg.name}"

    def fn(b):
        return (b.get(out_cfg.port).reshape(-1).char(),)

    return BoundReduce(
        port=out_cfg.port, output_names=(name,), dtypes=("int8",), dynamic_axes={}, fn=fn
    )


@pytest.fixture
def fresh_reduce_name():
    """Yield a never-registered reduce name and unregister it on teardown.

    Keeps the global registry pristine across tests — the registration is the
    behaviour under test, but it must not leak into the rest of the suite.

    Yields
    ------
    str
        A reduce name guaranteed unregistered at entry, popped on teardown.
    """
    import salt.core.onnx.reduces as reduces_mod  # noqa: PLC0415 - registry mutation guard

    name = "test_passthrough_int8"
    assert name not in reduces_mod._REGISTRY, "fixture name already registered (leak)"  # noqa: SLF001
    yield name
    reduces_mod._REGISTRY.pop(name, None)  # noqa: SLF001


class TestRegisterReduce:
    """The public `register_reduce` live-registry surface."""

    def test_shipped_reduces_are_registered_with_declared_dtypes(self):
        # the three core reduces + the two MaskFormer object reduces register at
        # import via register_reduce, with the dtypes/per-token flags that replace
        # the frozen M4.5 tuples (leading_object/object_index land in sub-wave C)
        assert set(registered_reduces()) == {
            "split_scalars",
            "argmax",
            "vertex_union_find",
            "leading_object",
            "object_index",
        }
        assert reduce_dtype("split_scalars") == "float32"
        assert reduce_dtype("argmax") == "int8"
        assert reduce_dtype("vertex_union_find") == "int8"
        # leading_object is GLOBAL float32 scalars; object_index is PER-TOKEN int8
        assert reduce_dtype("leading_object") == "float32"
        assert reduce_dtype("object_index") == "int8"
        assert set(per_token_reduces()) == {"argmax", "vertex_union_find", "object_index"}

    def test_config_known_reduces_is_a_live_registry_view(self):
        # the config-level public names are LIVE views of the registry (PEP 562
        # __getattr__), not the M4.5 frozen tuples
        from salt.core.onnx import config as cfg  # noqa: PLC0415 - live-attr access under test

        assert set(cfg.KNOWN_REDUCES) == set(registered_reduces())
        assert set(cfg.PER_TOKEN_REDUCES) == set(per_token_reduces())

    def test_register_and_use_a_new_reduce(self, fresh_reduce_name):
        register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="int8", per_token=False)
        # live everywhere: registry, config view, dtype lookup
        assert fresh_reduce_name in registered_reduces()
        assert reduce_dtype(fresh_reduce_name) == "int8"
        from salt.core.onnx import config as cfg  # noqa: PLC0415 - live-attr access under test

        assert fresh_reduce_name in cfg.KNOWN_REDUCES
        # config validates a manifest entry naming it AND defaults its dtype from
        # the DECLARED dtype (no hard-coded per-reduce rule)
        out = ExportOutput(port="preds.objects.x", name="Lead", reduce=fresh_reduce_name)
        resolved = _resolve_output(out)
        assert resolved.reduce == fresh_reduce_name
        assert resolved.dtype == "int8"
        # and bind_reduce dispatches to the registered binder
        ctx = ReduceCtx(model_name="M", seq_dyn_axis={}, produced_specs={})
        bound = bind_reduce(replace(out, dtype="int8"), ctx)
        assert bound.output_names == ("M_Lead",)
        b = Bundle()
        b.set("preds.objects.x", torch.tensor([[1.0, 2.0]]))
        (got,) = bound.fn(b)
        assert got.dtype == torch.int8

    def test_new_reduce_dtype_mismatch_rejected(self, fresh_reduce_name):
        # the registry's DECLARED dtype is enforced — a contradicting entry dtype
        # is a ConfigError (negative control: with the live-dtype default removed
        # this would silently accept float32)
        register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="int8")
        with pytest.raises(ConfigError, match="emits int8"):
            _resolve_output(
                ExportOutput(
                    port="preds.objects.x", name="Lead", reduce=fresh_reduce_name, dtype="float32"
                )
            )

    def test_duplicate_registration_rejected(self):
        # re-registering a shipped name is a hard error (no silent override —
        # would mask a real collision otherwise)
        with pytest.raises(ConfigError, match="already registered"):
            register_reduce("argmax", _bind_passthrough_int8, dtype="int8")

    def test_bad_declared_dtype_rejected(self, fresh_reduce_name):
        with pytest.raises(ConfigError, match=r"float32.*int8|int8.*float32"):
            register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="float16")

    def test_empty_name_rejected(self):
        with pytest.raises(ConfigError, match="non-empty string"):
            register_reduce("", _bind_passthrough_int8, dtype="int8")

    def test_unregistered_reduce_rejected_by_config_and_bind(self):
        # a name that is NOT registered is rejected at config validation AND at bind
        out = ExportOutput(port="preds.objects.x", name="Lead", reduce="never_registered_reduce")
        with pytest.raises(ConfigError, match="unknown reduce"):
            _resolve_output(out)
        ctx = ReduceCtx(model_name="M", seq_dyn_axis={}, produced_specs={})
        with pytest.raises(ConfigError, match="unknown reduce"):
            bind_reduce(out, ctx)


# ---------------------------------------------------------------------------
# the ONNX plan: purity + sources (design §7, §7.1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gn2_modules(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("onnx_adapter_fixture")
    v1 = build_test_gn2(tmp)
    modules = build_gn2v2_modules(tmp / "norm_dict.yaml")
    resolved = gn2_resolved()
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    nn.ModuleDict(modules).load_state_dict(map_v1_state_dict(v1.state_dict(), modules))
    return modules, resolved, plan


class TestOnnxPlan:
    def test_plan_contains_no_labels_losses_or_writers(self, gn2_modules):
        _, _, plan = gn2_modules
        assert plan.mode is Mode.ONNX
        assert "loss" not in plan.module_names  # LossSum is TRAINING-only
        for step in plan.steps:
            assert not any(key.startswith("labels.") for key in step.requires), step.name
            assert not any(key.startswith("losses.") for key in step.produces), step.name
        assert not any(key.startswith("labels.") for key in plan.sources)

    def test_sources_are_config_derived(self):
        resolved = resolve_export_config(gn2_export_cfg(), "m")
        from salt.core.graph.spec import flatten_spec

        flat = flatten_spec(derive_onnx_sources(resolved, VARIABLES))
        assert set(flat) == {"inputs.jets", "inputs.tracks", "masks.tracks"}
        assert flat["inputs.jets"].fields == tuple(JET_VARIABLES)
        assert flat["inputs.tracks"].shape == ("B", "T:tracks", len(TRACK_VARIABLES))
        assert flat["masks.tracks"].kind == "pad_mask"

    def test_missing_output_producer_is_a_named_error(self, gn2_modules):
        modules, resolved, _ = gn2_modules
        manifest = gn2_manifest()
        manifest[0].port = "preds.jets.typo_task"
        bad = gn2_resolved(manifest=manifest)
        with pytest.raises(ConnectivityError, match="ONNX manifest output"):
            compile_onnx_plan(modules, bad, VARIABLES)

    def test_manifest_less_config_cannot_compile(self, gn2_modules):
        modules, _, _ = gn2_modules
        half = resolve_export_config(gn2_export_cfg(), "m")
        with pytest.raises(ConfigError, match="manifest-attached"):
            compile_onnx_plan(modules, half, VARIABLES)

    def test_missing_variables_stream(self):
        resolved = gn2_resolved(run_name="m")
        with pytest.raises(ConfigError, match="Features variable"):
            derive_onnx_sources(resolved, {"jets": JET_VARIABLES})

    def test_misflagged_sequence_error_names_export_inputs(self, gn2_modules):
        # M4-review fix (§4.1 quality bar): 'sequence: false' on a
        # variable-length stream used to fail with a fix-less planner
        # ShapeError ("producer '<sources>' ...") that never pointed at the
        # export block — the error must attribute the source to
        # export.inputs and state the concrete fix
        modules, _, _ = gn2_modules
        cfg = gn2_export_cfg()
        cfg.inputs[1].sequence = False
        cfg.inputs[1].dyn_axis = None
        bad = attach_manifest(resolve_export_config(cfg, "m"), gn2_manifest())
        with pytest.raises(ShapeError) as excinfo:
            compile_onnx_plan(modules, bad, VARIABLES)
        message = str(excinfo.value)
        assert "export.inputs[1]" in message
        assert "inputs.tracks" in message
        assert "'sequence: true'" in message


# ---------------------------------------------------------------------------
# the adapter (design §7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gn2_adapter(gn2_modules):
    _, resolved, plan = gn2_modules
    fields = {"inputs.jets": tuple(JET_VARIABLES), "inputs.tracks": tuple(TRACK_VARIABLES)}
    return OnnxAdapter(plan, resolved, fields)


class TestOnnxAdapter:
    def test_generated_names_and_axes(self, gn2_adapter):
        assert gn2_adapter.input_names == ["jet_features", "track_features"]
        assert gn2_adapter.output_names == [
            "GN2v2_pb",
            "GN2v2_pc",
            "GN2v2_pu",
            "GN2v2_TrackOrigin",
            "GN2v2_VertexIndex",
        ]
        assert gn2_adapter.dynamic_axes == {
            "track_features": {0: "n_tracks"},
            "GN2v2_TrackOrigin": {0: "n_tracks"},
            "GN2v2_VertexIndex": {0: "n_tracks"},
        }
        assert gn2_adapter.output_dtypes == ["float32"] * 3 + ["int8"] * 2

    def test_example_inputs_shapes(self, gn2_adapter):
        jets, tracks = gn2_adapter.example_inputs(sequence_length=40)
        assert jets.shape == (1, len(JET_VARIABLES))  # global keeps the batch dim
        assert tracks.shape == (40, len(TRACK_VARIABLES))  # sequence has none

    @pytest.mark.parametrize("length", [0, 1, 11])
    def test_forward_shapes(self, gn2_adapter, length):
        gen = torch.Generator().manual_seed(5)
        jets = torch.rand(1, 2, generator=gen)
        tracks = torch.rand(length, 19, generator=gen)
        with torch.no_grad():
            outputs = gn2_adapter(jets, tracks)
        assert len(outputs) == 5
        assert all(out.dim() == 0 for out in outputs[:3])
        assert outputs[3].shape == (length,)
        assert outputs[3].dtype == torch.int8
        assert outputs[4].shape == (length,)
        assert outputs[4].dtype == torch.int8

    def test_global_input_must_have_batch_dim(self, gn2_adapter):
        with pytest.raises(AssertionError, match="batch, features"):
            gn2_adapter(torch.rand(2), torch.rand(3, 19))

    def test_wrong_arity_rejected(self, gn2_adapter):
        with pytest.raises(AssertionError, match="positional inputs"):
            gn2_adapter(torch.rand(1, 2))

    def test_non_onnx_plan_rejected(self, gn2_modules):
        from salt.tests.core.gn2v2_fixture import compile_gn2v2

        modules, resolved, _ = gn2_modules
        test_plan = compile_gn2v2(modules, Mode.TEST)
        with pytest.raises(ConfigError, match="Mode.ONNX plan|ONNX plan"):
            OnnxAdapter(test_plan, resolved, {})


class TestAliasGather:
    def make_adapter(self, gn2_modules, alias_variables):
        modules, _, _ = gn2_modules
        cfg = gn2_export_cfg()
        cfg.inputs.append(ExportInput(port="inputs.global", alias="inputs.jets"))
        resolved = attach_manifest(resolve_export_config(cfg, "m"), gn2_manifest())
        variables = {**VARIABLES, "global": alias_variables}
        plan = compile_onnx_plan(modules, resolved, variables)
        fields = {f"inputs.{s}": tuple(names) for s, names in variables.items()}
        return OnnxAdapter(plan, resolved, fields)

    def test_identity_clone_when_fields_equal(self, gn2_modules):
        adapter = self.make_adapter(gn2_modules, list(JET_VARIABLES))
        assert adapter._alias_gathers == [None]
        # alias consumes no positional input
        assert adapter.input_names == ["jet_features", "track_features"]

    def test_column_gather_when_fields_differ(self, gn2_modules):
        adapter = self.make_adapter(gn2_modules, [JET_VARIABLES[1]])
        (index,) = adapter._alias_gathers
        assert index.tolist() == [1]

    def test_missing_column_is_a_named_error(self, gn2_modules):
        with pytest.raises(ConfigError, match="lacks columns"):
            self.make_adapter(gn2_modules, ["not_a_jet_var"])


# ---------------------------------------------------------------------------
# the export-mode protocol: torch-math forcing + construction-time guards
# (design §7.2; v1 modelwrapper.py:331-335, to_onnx.py:670,700)
# ---------------------------------------------------------------------------


def build_flash_gn2_modules(tmp_path) -> dict:
    """A GN2-shaped module dict whose encoder is BUILT with torch-flash.

    Every shipped fixture constructs torch-math, so without this the
    `set_export_mode` forcing path — the actual Athena-agreement requirement
    for flash-trained configs — would be dead code under the test suite
    (M4-review fix).
    """
    write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
    dense = {"hidden_layers": [16], "activation": "ReLU"}
    torch.manual_seed(0)
    modules = {
        "norm": Normaliser(
            norm_dict=tmp_path / "norm_dict.yaml", streams=["jets", "tracks"], global_object="jets"
        ),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=16, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=16,
            num_layers=2,
            out_dim=16,
            attention={"num_heads": 2, "attn_type": "torch-flash"},
            dense={"activation": "ReLU", "gated": False},
        ),
        "split": Split(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            dense=dense,
        ),
    }
    for name, module in modules.items():
        module.name = name
    return modules


def _flash_export_cfg() -> ExportConfig:
    return ExportConfig(
        model_name="Flash",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
        ],
    )


class TestExportModeProtocol:
    def _compiled(self, tmp_path, *, materialise: bool):
        modules = build_flash_gn2_modules(tmp_path)
        resolved = attach_manifest(
            resolve_export_config(_flash_export_cfg(), "flash_run"),
            [ExportOutput(port="preds.jets.jets_classification", names=["pb", "pc", "pu"])],
        )
        plan = compile_onnx_plan(modules, resolved, VARIABLES)
        bind_all(modules, resolve_bind_schema([plan]))
        if materialise:
            modules["norm"].materialise()
        fields = {"inputs.jets": tuple(JET_VARIABLES), "inputs.tracks": tuple(TRACK_VARIABLES)}
        return modules, resolved, plan, fields

    def test_flash_encoder_forced_to_torch_math_at_construction(self, tmp_path):
        modules, resolved, plan, fields = self._compiled(tmp_path, materialise=True)
        encoder = modules["encoder"]
        # the trained config really is flash BEFORE the adapter exists
        assert encoder.encoder.attn_type == "torch-flash"
        assert all(layer.attn.fn.attn_type == "torch-flash" for layer in encoder.encoder.layers)
        OnnxAdapter(plan, resolved, fields)
        # the Athena-agreement requirement: torch-math forced via the
        # set_export_mode protocol on the encoder AND every layer
        # (modelwrapper.py:331-335, to_onnx.py:670,700)
        assert encoder.encoder.attn_type == "torch-math"
        assert all(layer.attn.fn.attn_type == "torch-math" for layer in encoder.encoder.layers)

    def test_unmaterialised_normaliser_rejected_before_tracing(self, tmp_path):
        # Normaliser.forward skips its eager guard under tracing
        # (TracerWarning hygiene) — the adapter must therefore refuse to
        # build on unmaterialised buffers, or the trace would silently bake
        # un-materialised values (M4-review fix)
        modules, resolved, plan, fields = self._compiled(tmp_path, materialise=False)
        with pytest.raises(ConfigError, match="materialised=False"):
            OnnxAdapter(plan, resolved, fields)
