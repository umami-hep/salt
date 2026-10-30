"""Unit tests for bind-schema resolution (mirror of salt/model/bind.py)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from salt.graph.bundle import Bundle
from salt.graph.executor import Executor
from salt.graph.planner import compile_plan
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
from salt.model.modules import (
    BindError,
    SaltModelModule,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_gn2v2_modules,
    compile_gn2v2,
    make_gn2_batch,
    make_gn2_labels,
    write_parity_norm_dict,
)

# bind schema resolution


class TestResolvedSchema:
    def test_widths_resolve_through_the_graph(self, gn2v2):
        """Symbolic widths bind from source/config declarations."""
        _, _, schema = gn2v2
        assert schema.width("inputs.jets") == len(JET_VARIABLES)
        assert schema.width("inputs.tracks") == len(TRACK_VARIABLES)
        # through the Normaliser's shared symbol
        assert schema.width("normed.tracks") == len(TRACK_VARIABLES)
        # config-concrete produces
        assert schema.width("embed.tracks") == 16
        assert schema.width("encoded.seq") == 16
        # through Split / Pooling instance symbols
        assert schema.width("encoded.tracks") == 16
        assert schema.width("pooled.global") == 16

    def test_fields_come_from_the_source_declaration(self, gn2v2):
        _, _, schema = gn2v2
        assert schema.fields_of("inputs.tracks") == tuple(TRACK_VARIABLES)
        with pytest.raises(BindError, match="no declared fields"):
            schema.fields_of("encoded.seq")

    def test_unknown_width_raises_with_suggestions(self, gn2v2):
        _, _, schema = gn2v2
        with pytest.raises(BindError, match="encoded.tracks"):
            schema.width("encoded.trakcs")

    def test_meta_and_scalar_keys_have_no_width(self, gn2v2):
        _, _, schema = gn2v2
        for key in ("seq.layout", "loss.total"):
            with pytest.raises(BindError):
                schema.width(key)

    def test_conflicting_widths_raise(self):
        from salt.graph.planner import DimTable

        dims = DimTable(error=BindError)
        dims.bind("F:x", 3, "here")
        with pytest.raises(BindError, match="conflicting sizes"):
            dims.bind("F:x", 4, "there")


# trainability after bind + materialise


class TestMaterialisedTrainability:
    def test_losses_require_grad_in_train_mode(self, tmp_path):
        """The FIT path is trainable after bind + materialise (grads reach every module)."""
        batch_size, n_tracks = 6, 10
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        modules = build_gn2v2_modules(nd)
        plan = compile_gn2v2(modules, Mode.FIT)
        bind_all(modules, resolve_bind_schema(plan))
        materialise_all(modules)
        nn.ModuleDict(modules).train()

        inputs, masks = make_gn2_batch(batch_size, n_tracks)
        labels = make_gn2_labels(batch_size, n_tracks)
        bundle = Bundle()
        for stream, x in inputs.items():
            bundle.set(f"inputs.{stream}", x.clone())
        for stream, m in masks.items():
            bundle.set(f"masks.{stream}", m.clone())
        for stream, fields in labels.items():
            for name, val in fields.items():
                bundle.set(f"labels.{stream}.{name}", val.clone())

        out = Executor(plan).run(bundle)
        loss = out.get("loss.total")
        assert torch.isfinite(loss)
        loss.backward()
        for name in ("track_embed", "encoder", "pool", "jets_classification"):
            grads = [p.grad for p in modules[name].parameters() if p.grad is not None]
            assert grads, f"no gradients reached module {name!r}"


# bind_all / materialise_all de-duck-typing: the PRODUCTION caller argument
# (SaltModule._graph_modules, saltmodule.py) is model-only by construction (never a terminal
# sink — see salt.model.bind.bind_all's docstring for the audit). But at least one TEST call
# site (salt.tests.unit.outputs.sinks.onnx.test_adapter's gn2_folded_modules fixture) calls bind_all directly
# on a per-mode LOCAL module dict with a terminal OnnxExportSink folded in, mirroring
# SaltModule.compile_mode's own fold — so bind_all/materialise_all keep an explicit
# isinstance(module, SaltModelModule) partition (not getattr/callable duck-typing) rather than
# calling .bind()/.materialise() unconditionally; a sink is silently skipped (sinks have no
# bind/materialise method).


class TestBindAllDirectCalls:
    def test_gn2v2_modules_are_all_salt_model_modules(self, tmp_path):
        """The common-case invariant: every module a production SaltModule sees is model-only."""
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        modules = build_gn2v2_modules(nd)
        assert all(isinstance(m, SaltModelModule) for m in modules.values())

    def test_bind_all_and_materialise_all_call_base_no_op_defaults_directly(self, tmp_path):
        """A bare `SaltModelModule` subclass (base no-op bind/materialise, no override) works
        fine through bind_all/materialise_all — proves the direct calls need no discovery.
        """

        class _Bare(SaltModelModule):
            def declare_io(self, mode):
                del mode
                return IO(requires={}, produces={})

        bare = _Bare()
        bare.name = "bare"
        modules = {"bare": bare}
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        # base bind()/materialise() are documented no-ops — must not raise
        bind_all(modules, resolve_bind_schema(compile_gn2v2(build_gn2v2_modules(nd), Mode.FIT)))
        materialise_all(modules)

    def test_bind_all_and_materialise_all_skip_a_folded_sink(self, tmp_path):
        """Regression: a mixed dict with a non-SaltModelModule 'sink' double is handled —
        bind_all/materialise_all call the real module and silently skip the sink (a sink
        has no bind/materialise).
        """

        class _FakeSink:
            """A minimal stand-in for a folded terminal sink (e.g. OnnxExportSink): no
            bind/materialise, so calling either unconditionally would raise AttributeError.
            """

            name = "fake_sink"

            def is_sink(self) -> bool:
                return True

        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_parity_norm_dict(nd, cd)
        modules = build_gn2v2_modules(nd)
        plan = compile_gn2v2(modules, Mode.FIT)
        mixed = {**modules, "fake_sink": _FakeSink()}
        # must not raise AttributeError on the sink double
        bind_all(mixed, resolve_bind_schema(plan))
        materialise_all(mixed)
        # the real modules were still bound/materialised (Normaliser flips its buffer)
        assert bool(modules["norm"].materialised)


# derived_fields: column names that are a function of resolved input fields


class _FieldToy(SaltModelModule):
    """A toy plan step with static IO and an optional derived-fields rule."""

    def __init__(self, name, requires, produces, derive=None):
        super().__init__()
        self.name = name
        self._io = IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))
        self._derive = derive

    def declare_io(self, mode):
        del mode
        return self._io

    def derived_fields(self, fields):
        return {} if self._derive is None else self._derive(fields)


_SRC_FIELDS = ("a", "b", "c")


def _field_plan(*modules, sink):
    sources = unflatten_spec({
        "inputs.x": TensorSpec(shape=("B", "F:x"), dtype="float32", fields=_SRC_FIELDS)
    })
    return compile_plan({m.name: m for m in modules}, Mode.TEST, sources=sources, sinks=[sink])


def _spec(last="F:out", fields=None):
    return TensorSpec(shape=("B", last), dtype="float32", fields=fields)


def _same_as(src, dst, prefix=""):
    def derive(fields):
        return {dst: tuple(f"{prefix}{n}" for n in fields[src])} if src in fields else {}

    return derive


def _scale(out_spec):
    derive = _same_as("inputs.x", "normed.x")
    return _FieldToy("scale", {"inputs.x": _spec("F:x")}, {"normed.x": out_spec}, derive)


class TestDerivedFields:
    def test_names_propagate_through_a_chain(self):
        """A chain of derivations resolves names and widths regardless of module order."""
        derive = _same_as("normed.x", "renamed.x", prefix="r_")
        rename = _FieldToy(
            "rename", {"normed.x": _spec("F:scale")}, {"renamed.x": _spec("F:rename")}, derive
        )
        plan = _field_plan(rename, _scale(_spec("F:scale")), sink="renamed.x")
        schema = resolve_bind_schema(plan)
        assert schema.fields_of("normed.x") == _SRC_FIELDS
        assert schema.width("normed.x") == len(_SRC_FIELDS)
        assert schema.fields_of("renamed.x") == ("r_a", "r_b", "r_c")
        assert schema.width("renamed.x") == len(_SRC_FIELDS)

    def test_conflicting_derived_tuple_raises(self):
        """A derived tuple disagreeing with the declared fields raises, naming module and key."""
        scale = _scale(_spec("F:scale", fields=("p", "q", "r")))
        with pytest.raises(BindError, match=r"'scale' derives fields .* for 'normed\.x'"):
            resolve_bind_schema(_field_plan(scale, sink="normed.x"))

    def test_width_vs_derived_fields_length_mismatch_raises(self):
        """A key whose resolved width differs from its derived column count raises."""
        with pytest.raises(BindError, match=r"'normed\.x' resolves to width 2 but carries 3"):
            resolve_bind_schema(_field_plan(_scale(_spec(2)), sink="normed.x"))

    def test_structured_key_declared_fields_do_not_constrain_width(self):
        """Declared struct fields on a ``[B, T]`` raw key do not override its observed width."""
        features = _FieldToy(
            "features",
            {"raw.x": TensorSpec(kind="data", fields=_SRC_FIELDS)},
            {"inputs.x": _spec("F:x", fields=_SRC_FIELDS)},
        )
        sources = unflatten_spec({"raw.x": TensorSpec(shape=("B", 40), kind="data")})
        plan = compile_plan({"features": features}, Mode.TEST, sources=sources, sinks=["inputs.x"])
        schema = resolve_bind_schema(plan)
        assert schema.width("raw.x") == 40
        assert schema.fields_of("raw.x") == _SRC_FIELDS
        assert schema.width("inputs.x") == len(_SRC_FIELDS)
