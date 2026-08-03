"""Unit tests for the export surface: config resolution, reduces, adapter."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from salt.graph import Bundle, Mode
from salt.graph.errors import ConfigError, ConnectivityError, ShapeError
from salt.model.modules import (
    Concat,
    GlobalAttentionPooling,
    Normaliser,
    Split,
    StreamEmbed,
    TransformerEncoder,
    bind_all,
    resolve_bind_schema,
)
from salt.model.modules.tasks import ClassificationTaskModule
from salt.outputs.sinks.onnx import (
    ExportConfig,
    ExportInput,
    OnnxAdapter,
    compile_onnx_plan,
    derive_onnx_sources,
    resolve_export_config,
    sanitised_model_name,
    validate_model_name,
)
from salt.outputs.sinks.onnx.config import ExportOutput, default_athena_name
from salt.outputs.sinks.onnx.reduces import (
    BoundReduce,
    ReduceCtx,
    bind_reduce,
    reduce_dtype,
    reduce_spec,
    register_reduce,
    registered_reduces,
)
from salt.outputs import (
    ClassProbs,
    OnnxExportLeaf,
    OnnxExportSink,
    OutputField,
    SeqClassIndex,
)
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    write_parity_norm_dict,
)

VARIABLES = {"jets": list(JET_VARIABLES), "tracks": list(TRACK_VARIABLES)}


def gn2_export_cfg(**overrides) -> ExportConfig:
    # export-only half — the outputs come from the folded OnnxExportSink
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


def _named(node, name):
    node.name = name
    return node


class _SplitStub:
    """An ONNX-only producer minting N global float fields under one leaf key."""

    def __init__(self, key, names):
        self.key = key
        self.names = list(names)

    def manifest_fields(self, mode):
        """The N global fields, ONNX-only."""
        if not (mode & Mode.ONNX):
            return []
        return [(self.key, OutputField(h5_name=None, onnx_name=n)) for n in self.names]


def gn2_folded_modules(tmp_path):
    """The GN2 module dict + the folded conversion nodes + OnnxExportSink."""
    modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
    modules.update({
        "jet_probs": _named(ClassProbs(task="jets_classification", stream="jets"), "jet_probs"),
        "track_origin_index": _named(
            SeqClassIndex(task="track_origin", stream="tracks"), "track_origin_index"
        ),
        "onnx_export": _named(OnnxExportSink(), "onnx_export"),
    })
    bind_producers(modules)
    return modules


def bind_producers(modules) -> None:
    """Bind the model modules to every producer/sink that resolves names from them."""
    for module in modules.values():
        if callable(getattr(module, "bind_model_modules", None)):
            module.bind_model_modules(modules)


def gn2_resolved(run_name: str = "GN2_v2", **overrides) -> ExportConfig:
    return resolve_export_config(gn2_export_cfg(**overrides), run_name)


# config resolution + validation


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


class TestExportSinkOutputs:
    # the ONNX output manifest is declared by the OnnxExportSink, whose
    # OnnxExportLeaf carries the per-output naming/dtype/per-token rules (the
    # conversion math itself is proven bitwise in
    # test_onnx_fold_classification/objects). These assert that surface.

    def test_name_and_names_exclusive(self):
        with pytest.raises(ConfigError, match="BOTH"):
            OnnxExportLeaf(key="outputs.jets.c", name="both", names=["pb", "pc"])

    def test_name_defaults_to_leaf_terminal_segment(self):
        """Single-source naming: omitting name/names defaults the suffix to the leaf terminal."""
        leaf = OnnxExportLeaf(key="outputs.tracks.HadronIndex", dtype="int8", per_token=True)
        assert leaf.name == "HadronIndex"
        assert leaf.suffixes == ("HadronIndex",)

    def test_names_default_to_split(self):
        leaf = OnnxExportLeaf(key="outputs.jets.c", names=["pb", "pc", "pu"])
        assert leaf.suffixes == ("pb", "pc", "pu")
        assert leaf.dtype == "float32"
        assert not leaf.per_token  # split_scalars leaves are GLOBAL float scalars

    def test_aux_dtype_is_int8(self):
        leaf = OnnxExportLeaf(
            key="outputs.tracks.track_origin", name="TrackOrigin", dtype="int8", per_token=True
        )
        assert leaf.dtype == "int8"
        assert leaf.per_token

    def test_bad_dtype_rejected(self):
        with pytest.raises(ConfigError, match="float32.*int8|int8.*float32"):
            OnnxExportLeaf(key="outputs.jets.c", name="x", dtype="float64")

    def test_split_names_with_per_token_rejected(self):
        with pytest.raises(ConfigError, match="per_token"):
            OnnxExportLeaf(key="outputs.tracks.c", names=["pb", "pc"], per_token=True)

    def test_non_outputs_key_rejected(self):
        with pytest.raises(ConfigError, match="outputs"):
            OnnxExportLeaf(key="preds.jets.c", names=["pb", "pc"])

    def test_duplicate_output_name_rejected(self):
        with pytest.raises(ConfigError, match="duplicate flat ONNX output name"):
            OnnxExportSink._validate_leaves([  # noqa: SLF001 - direct guard check
                OnnxExportLeaf(key="outputs.jets.a", names=["pb", "pc"]),
                OnnxExportLeaf(key="outputs.jets.b", name="pb", dtype="int8", per_token=True),
            ])

    def test_explicit_outputs_list_rejected(self):
        """The config-level leaf list is retired — producers name their own leaves."""
        with pytest.raises(ConfigError, match="no longer accepts an explicit"):
            OnnxExportSink(outputs=[OnnxExportLeaf(key="outputs.jets.a", names=["pb", "pc"])])

    def test_empty_sink_defers_to_its_manifest_sources(self):
        """An omitted `outputs` defers to the bound manifest sources."""
        sink = OnnxExportSink(outputs=[], model_name="M")
        with pytest.raises(ConfigError, match="collected no ONNX output"):
            sink.output_names()


# the SHIPPED reduces are RETIRED — folded into conversion nodes.
# The argmax/maskformer math is now proven BITWISE in
# test_onnx_fold_classification.py (SeqClassIndex/Combination) and test_onnx_fold_objects.py
# (MaskFormerObjects) against the same v1 chains these reduces composed; the union-find
# vertex index rides the live VertexingTaskModule.get_output path (tests/unit get_output).
# These tests pin the RETIREMENT (no shipped reduce registered).


class TestRetiredReduces:
    def test_no_shipped_reduces_registered(self):
        # the five shipped reduce REGISTRATIONS are gone (the conversion
        # nodes own the math); registered_reduces() carries no shipped name
        from salt.outputs.sinks.onnx.reduces import registered_reduces  # noqa: PLC0415

        shipped = {"split_scalars", "argmax", "vertex_union_find", "leading_object", "object_index"}
        assert shipped.isdisjoint(set(registered_reduces()))

    def test_split_named_split_helper_is_the_sink(self):
        # the split_scalars NAMING split now lives on the OnnxExportSink (v1
        # task.py:301 torch.split+squeeze) — pinned here on a hand-made bundle
        sink = OnnxExportSink(model_name="M")
        sink.bind_model_modules({"probs": _SplitStub("outputs.jets.c", ["pb", "pc", "pu"])})
        b = Bundle()
        probs = torch.tensor([[0.5, 0.3, 0.2]])
        b.set("outputs.jets.c", probs)
        named = sink.named_outputs(b)
        assert set(named) == {"M_pb", "M_pc", "M_pu"}
        assert all(named[k].dim() == 0 for k in named)  # v1 split+squeeze -> 0-dim scalars
        assert torch.allclose(
            torch.stack([named["M_pb"], named["M_pc"], named["M_pu"]]), probs.squeeze(0)
        )


# the LIVE register_reduce surface


def _bind_passthrough_int8(out_cfg, ctx):
    """A toy single-output reduce: flatten the port to int8 (registration target)."""
    name = f"{ctx.model_name}_{out_cfg.name}"

    def fn(b):
        return (b.get(out_cfg.port).reshape(-1).char(),)

    return BoundReduce(
        port=out_cfg.port, output_names=(name,), dtypes=("int8",), dynamic_axes={}, fn=fn
    )


@pytest.fixture
def fresh_reduce_name():
    """Yield a never-registered reduce name and unregister it on teardown."""
    import salt.outputs.sinks.onnx.reduces as reduces_mod  # noqa: PLC0415 - registry mutation guard

    name = "test_passthrough_int8"
    assert name not in reduces_mod._REGISTRY, "fixture name already registered (leak)"  # noqa: SLF001
    yield name
    reduces_mod._REGISTRY.pop(name, None)  # noqa: SLF001


class TestRegisterReduce:
    """The public `register_reduce` live-registry surface."""

    def test_no_shipped_reduces_registered_at_import(self):
        # the registry starts EMPTY of the retired shipped reduces
        assert {
            "split_scalars",
            "argmax",
            "vertex_union_find",
            "leading_object",
            "object_index",
        }.isdisjoint(set(registered_reduces()))

    def test_config_known_reduces_is_a_live_registry_view(self, fresh_reduce_name):
        # the config-level public names stay a LIVE view of the registry (PEP 562
        # __getattr__); a freshly-registered probe reduce shows up in the view
        from salt.outputs.sinks.onnx import config as cfg  # noqa: PLC0415 - live-attr access under test

        register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="int8")
        assert set(cfg.KNOWN_REDUCES) == set(registered_reduces())
        assert fresh_reduce_name in cfg.KNOWN_REDUCES

    def test_register_and_use_a_new_reduce(self, fresh_reduce_name):
        # (the retired `_resolve_output` manifest resolver is gone; the registry
        # spec is the dtype/arity authority now.)
        register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="int8", per_token=False)
        # live everywhere: registry, config view, dtype lookup
        assert fresh_reduce_name in registered_reduces()
        assert reduce_dtype(fresh_reduce_name) == "int8"
        from salt.outputs.sinks.onnx import config as cfg  # noqa: PLC0415 - live-attr access under test

        assert fresh_reduce_name in cfg.KNOWN_REDUCES
        # the registered spec carries the declared field knowledge
        spec = reduce_spec(fresh_reduce_name)
        assert spec.dtype == "int8"
        assert spec.per_token is False
        assert spec.expects_names is False
        # and bind_reduce dispatches to the registered binder
        out = ExportOutput(port="preds.objects.x", name="Lead", reduce=fresh_reduce_name)
        ctx = ReduceCtx(model_name="M", seq_dyn_axis={}, produced_specs={})
        bound = bind_reduce(replace(out, dtype="int8"), ctx)
        assert bound.output_names == ("M_Lead",)
        b = Bundle()
        b.set("preds.objects.x", torch.tensor([[1.0, 2.0]]))
        (got,) = bound.fn(b)
        assert got.dtype == torch.int8

    def test_duplicate_registration_rejected(self, fresh_reduce_name):
        # re-registering an already-registered name is a hard error (no silent
        # override — would mask a real collision otherwise)
        register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="int8")
        with pytest.raises(ConfigError, match="already registered"):
            register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="int8")

    def test_bad_declared_dtype_rejected(self, fresh_reduce_name):
        with pytest.raises(ConfigError, match=r"float32.*int8|int8.*float32"):
            register_reduce(fresh_reduce_name, _bind_passthrough_int8, dtype="float16")

    def test_empty_name_rejected(self):
        with pytest.raises(ConfigError, match="non-empty string"):
            register_reduce("", _bind_passthrough_int8, dtype="int8")

    def test_unregistered_reduce_rejected_at_bind(self):
        # a name that is NOT registered is rejected at bind (spec lookup)
        out = ExportOutput(port="preds.objects.x", name="Lead", reduce="never_registered_reduce")
        with pytest.raises(ConfigError, match="unknown reduce"):
            reduce_spec("never_registered_reduce")
        ctx = ReduceCtx(model_name="M", seq_dyn_axis={}, produced_specs={})
        with pytest.raises(ConfigError, match="unknown reduce"):
            bind_reduce(out, ctx)


# the ONNX plan: purity + sources


@pytest.fixture(scope="module")
def gn2_modules(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("onnx_adapter_fixture")
    write_parity_norm_dict(tmp / "norm_dict.yaml", tmp / "class_dict.yaml")
    torch.manual_seed(42)  # deterministic non-trivial weights (retired v1 transfer stand-in)
    modules = gn2_folded_modules(tmp)
    resolved = gn2_resolved()
    plan = compile_onnx_plan(modules, resolved, VARIABLES)
    bind_all(modules, resolve_bind_schema([plan]))
    modules["norm"].materialise()
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
        from salt.graph.spec import flatten_spec

        flat = flatten_spec(derive_onnx_sources(resolved, VARIABLES))
        assert set(flat) == {"inputs.jets", "inputs.tracks", "masks.tracks"}
        assert flat["inputs.jets"].fields == tuple(JET_VARIABLES)
        assert flat["inputs.tracks"].shape == ("B", "T:tracks", len(TRACK_VARIABLES))
        assert flat["masks.tracks"].kind == "pad_mask"

    def test_missing_output_producer_is_a_named_error(self, tmp_path):
        # a sink leaf naming a conversion leaf no node produces is a connectivity
        # error (the folded sink anchors the demand)
        write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
        modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
        bad_sink = OnnxExportSink()
        bad_sink.bind_model_modules({
            "typo": _SplitStub("outputs.jets.typo_task", ["pb", "pc", "pu"])
        })
        bad_sink.name = "onnx_export"
        modules["onnx_export"] = bad_sink
        with pytest.raises(ConnectivityError):
            compile_onnx_plan(modules, gn2_resolved(), VARIABLES)

    def test_sinkless_config_cannot_compile(self, tmp_path):
        # without a folded OnnxExportSink there is no ONNX-output demand source
        write_parity_norm_dict(tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml")
        modules = build_gn2v2_modules(tmp_path / "norm_dict.yaml")
        with pytest.raises(ConfigError, match="OnnxExportSink"):
            compile_onnx_plan(modules, gn2_resolved(), VARIABLES)

    def test_missing_variables_stream(self):
        resolved = gn2_resolved(run_name="m")
        with pytest.raises(ConfigError, match="Features variable"):
            derive_onnx_sources(resolved, {"jets": JET_VARIABLES})

    def test_misflagged_sequence_error_names_export_inputs(self, gn2_modules):
        # 'sequence: false' on a variable-length stream must not fail with a
        # fix-less planner ShapeError ("producer '<sources>' ...") that never
        # points at the export block — the error must attribute the source to
        # export.inputs and state the concrete fix
        modules, _, _ = gn2_modules
        cfg = gn2_export_cfg()
        cfg.inputs[1].sequence = False
        cfg.inputs[1].dyn_axis = None
        bad = resolve_export_config(cfg, "m")
        with pytest.raises(ShapeError) as excinfo:
            compile_onnx_plan(modules, bad, VARIABLES)
        message = str(excinfo.value)
        assert "export.inputs[1]" in message
        assert "inputs.tracks" in message
        assert "'sequence: true'" in message


# the adapter


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
        ]
        assert gn2_adapter.dynamic_axes == {
            "track_features": {0: "n_tracks"},
            "GN2v2_TrackOrigin": {0: "n_tracks"},
        }
        assert gn2_adapter.output_dtypes == ["float32"] * 3 + ["int8"]

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
        assert len(outputs) == 4
        assert all(out.dim() == 0 for out in outputs[:3])
        assert outputs[3].shape == (length,)
        assert outputs[3].dtype == torch.int8

    def test_global_input_must_have_batch_dim(self, gn2_adapter):
        with pytest.raises(AssertionError, match="batch, features"):
            gn2_adapter(torch.rand(2), torch.rand(3, 19))

    def test_wrong_arity_rejected(self, gn2_adapter):
        with pytest.raises(AssertionError, match="positional inputs"):
            gn2_adapter(torch.rand(1, 2))

    def test_non_onnx_plan_rejected(self, gn2_modules):
        from salt.tests._fixtures.gn2v2_fixture import compile_gn2v2

        modules, resolved, _ = gn2_modules
        test_plan = compile_gn2v2(modules, Mode.TEST)
        with pytest.raises(ConfigError, match="Mode.ONNX plan|ONNX plan"):
            OnnxAdapter(test_plan, resolved, {})


class TestAliasGather:
    def make_adapter(self, gn2_modules, alias_variables):
        modules, _, _ = gn2_modules
        cfg = gn2_export_cfg()
        cfg.inputs.append(ExportInput(port="inputs.global", alias="inputs.jets"))
        resolved = resolve_export_config(cfg, "m")
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


# the export-mode protocol: torch-math forcing + construction-time guards
# (v1 modelwrapper.py:331-335, to_onnx.py:670,700)


def build_flash_gn2_modules(tmp_path) -> dict:
    """A GN2-shaped module dict whose encoder is BUILT with torch-flash."""
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
        modules["jet_probs"] = _named(
            ClassProbs(task="jets_classification", stream="jets"), "jet_probs"
        )
        modules["onnx_export"] = _named(OnnxExportSink(), "onnx_export")
        bind_producers(modules)
        resolved = resolve_export_config(_flash_export_cfg(), "flash_run")
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
        # un-materialised values
        modules, resolved, plan, fields = self._compiled(tmp_path, materialise=False)
        with pytest.raises(ConfigError, match="materialised=False"):
            OnnxAdapter(plan, resolved, fields)
