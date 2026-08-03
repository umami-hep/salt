"""Unit tests for bind-schema resolution (mirror of salt/model/bind.py)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from salt.graph import IO, Bundle, Executor, Mode
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
        from salt.model.bind import _DimBindings

        dims = _DimBindings()
        dims.bind("F:x", 3, "here")
        with pytest.raises(BindError, match="conflicting widths"):
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
# site (salt.tests.unit.onnx.test_adapter's gn2_folded_modules fixture) calls bind_all directly
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
