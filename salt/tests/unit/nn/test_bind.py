"""Unit tests for bind-schema resolution (mirror of salt/core/nn/bind.py)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from salt.core.graph import Bundle, Executor, Mode
from salt.core.nn import (
    BindError,
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
        """Symbolic widths bind from source/config declarations (design §2.3)."""
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
        from salt.core.nn.bind import _DimBindings

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
