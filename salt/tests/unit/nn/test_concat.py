"""Unit tests for Concat and VectorConcat (mirror of concat.py + vector_concat.py)."""

from __future__ import annotations

import pytest
import torch

from salt.core.graph import (
    Bundle,
    ConfigError,
    Mode,
    flatten_spec,
)
from salt.core.nn import (
    Concat,
    VectorConcat,
    resolve_bind_schema,
)
from salt.tests._fixtures.regression_fixture import (
    GLOBAL_VARIABLES,
    build_vector_concat_modules,
    compile_vector_concat,
    write_vector_concat_norm_dict,
)


class TestConcat:
    def test_registers_rejected_in_m2(self):
        with pytest.raises(ConfigError, match="registers are internal to TransformerEncoder"):
            Concat(streams=["tracks"], registers=8)

    def test_layout_and_order(self):
        concat = Concat(streams=["tracks", "electrons"])
        concat.name = "concat"
        b = Bundle()
        b.set("embed.tracks", torch.ones(2, 3, 4))
        b.set("embed.electrons", 2 * torch.ones(2, 2, 4))
        b.set("masks.tracks", torch.zeros(2, 3, dtype=torch.bool))
        b.set("masks.electrons", torch.ones(2, 2, dtype=torch.bool))
        out = concat(b, Mode.FIT)
        assert out["seq.x"].shape == (2, 5, 4)
        assert out["seq.layout"] == {"tracks": (0, 3), "electrons": (3, 5)}
        assert torch.equal(out["seq.x"][:, :3], b.get("embed.tracks"))
        assert out["seq.mask"].tolist() == [[False] * 3 + [True] * 2] * 2


class TestVectorConcat:
    def test_empty_inputs_rejected(self):
        with pytest.raises(ConfigError, match="non-empty sequence"):
            VectorConcat(inputs=[])

    def test_duplicate_inputs_rejected(self):
        with pytest.raises(ConfigError, match="duplicate inputs"):
            VectorConcat(inputs=["pooled.global", "pooled.global"])

    def test_self_feed_out_in_inputs_rejected(self):
        with pytest.raises(ConfigError, match="consume its own output"):
            VectorConcat(inputs=["pooled.global", "vconcat.global"], out="vconcat.global")

    def test_declare_io_per_input_symbols(self):
        vc = VectorConcat(inputs=["pooled.global", "normed.global"], out="vconcat.global")
        vc.name = "vconcat"
        io = vc.declare_io(Mode.FIT)
        req = flatten_spec(io.requires)
        # each input gets its OWN width symbol (genuinely different widths)
        assert req["pooled.global"].shape[-1] != req["normed.global"].shape[-1]
        produces = flatten_spec(io.produces)
        assert "vconcat.global" in produces
        assert produces["vconcat.global"].dtype == "float32"

    def test_forward_order_and_shape(self):
        vc = VectorConcat(inputs=["pooled.global", "normed.global"], out="vconcat.global")
        vc.name = "vconcat"
        b = Bundle()
        b.set("pooled.global", torch.ones(4, 16))
        b.set("normed.global", 9 * torch.ones(4, 2))
        out = vc(b, Mode.FIT)
        cat = out["vconcat.global"]
        assert cat.shape == (4, 18)
        # pooled FIRST, global features LAST (design §6.6 1390-1391)
        assert torch.equal(cat[:, :16], torch.ones(4, 16))
        assert torch.equal(cat[:, 16:], 9 * torch.ones(4, 2))

    def test_forward_is_mode_agnostic(self):
        # [B, D] vectors have no dynamic feature axis, so ONNX == FIT exactly
        vc = VectorConcat(inputs=["a", "b"], out="c")
        vc.name = "vc"
        b = Bundle()
        b.set("a", torch.randn(3, 5))
        b.set("b", torch.randn(3, 7))
        assert torch.equal(vc(b, Mode.FIT)["c"], vc(b, Mode.ONNX)["c"])

    def test_derived_widths_sum_when_resolved(self):
        vc = VectorConcat(inputs=["pooled.global", "normed.global"], out="vconcat.global")
        vc.name = "vconcat"
        assert vc.derived_widths({"pooled.global": 256, "normed.global": 2}) == {
            "vconcat.global": 258
        }
        # an unresolved input -> no contribution (order-insensitive across plans)
        assert vc.derived_widths({"pooled.global": 256}) == {}

    def test_dsum_resolves_through_the_schema(self, tmp_path):
        # the bind-time second pass resolves Dsum from the input-width sum
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_vector_concat_norm_dict(nd, cd)
        modules = build_vector_concat_modules(nd)
        fit = compile_vector_concat(modules, Mode.FIT)
        test = compile_vector_concat(modules, Mode.TEST)
        schema = resolve_bind_schema([fit, test])
        d_pool = schema.width("pooled.global")
        d_glob = schema.width("normed.global")
        assert d_glob == len(GLOBAL_VARIABLES)
        assert schema.width("vconcat.global") == d_pool + d_glob
