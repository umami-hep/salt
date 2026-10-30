"""Unit tests for Concat, VectorConcat and SelectFields (mirror of plumbing.py)."""

from __future__ import annotations

import pytest
import torch

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import Mode, flatten_spec
from salt.model.modules import (
    Concat,
    ResolvedSchema,
    VectorConcat,
    bind_all,
    resolve_bind_schema,
)
from salt.model.modules.plumbing import SelectFields
from salt.tests._fixtures.gn2v2_fixture import JET_VARIABLES
from salt.tests._fixtures.v2_builders import (
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
        # pooled FIRST, global features LAST
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

    @pytest.mark.parametrize(("fields", "expected"), [
        ({"a": ("pt", "eta"), "b": ("mass",)}, {"c": ("pt", "eta", "mass")}),  # concat order
        ({"b": ("pt", "eta")}, {}),  # a field-less input (e.g. pooled.global)
        ({"a": ("pt", "eta"), "b": ("eta",)}, {}),  # colliding names
    ])
    def test_derived_fields(self, fields, expected):
        """Names propagate in concat order only when every input carries them, uniquely."""
        vc = VectorConcat(inputs=["a", "b"], out="c")
        vc.name = "vc"
        assert vc.derived_fields(fields) == expected


_JET_SCHEMA = ResolvedSchema({"normed.jets": 4}, {"normed.jets": ("pt", "eta", "phi", "mass")})


def _select(fields, input="normed.jets", out="prior_kin.jets"):  # noqa: A002
    sel = SelectFields(input=input, fields=fields, out=out)
    sel.name = "sel"
    return sel


class TestSelectFields:
    @pytest.mark.parametrize(("fields", "out", "match"), [
        ([], "prior_kin.jets", "non-empty"),
        (["pt", "pt"], "prior_kin.jets", "duplicate fields"),
        ("pt", "prior_kin.jets", "bare string"),
        (["pt"], "normed.jets", "own output"),
    ])
    def test_bad_config_rejected(self, fields, out, match):
        """Empty/duplicate/bare-string fields and a self-feed (out == input) are config errors."""
        with pytest.raises(ConfigError, match=match):
            SelectFields(input="normed.jets", fields=fields, out=out)

    def test_declare_io_output_carries_fields(self):
        """The output is [B, len(fields)] and declares the selected names in order."""
        io = _select(["eta", "pt"]).declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"normed.jets"}
        out = flatten_spec(io.produces)["prior_kin.jets"]
        assert (out.shape, out.fields, out.dtype) == (("B", 2), ("eta", "pt"), "float32")

    def test_selects_columns_by_name_in_configured_order(self):
        """Forward gathers columns by name, in config order; no state_dict entries."""
        sel = _select(["eta", "pt"])
        sel.bind(_JET_SCHEMA)
        x = torch.randn(5, 4)
        b = Bundle()
        b.set("normed.jets", x)
        assert torch.equal(sel(b, Mode.FIT)["prior_kin.jets"], x[:, [1, 0]])
        assert len(sel.state_dict()) == 0
        assert sel._index.dtype == torch.int64  # noqa: SLF001

    @pytest.mark.parametrize(("fields", "schema", "match"), [
        (["pt", "energy"], _JET_SCHEMA, r"\['energy'\].*available: \['pt', 'eta', 'phi'"),
        (["pt"], ResolvedSchema({"normed.jets": 4}), "carries no column names"),
        (["pt"], ResolvedSchema({"normed.jets": 3}, _JET_SCHEMA.fields), "width 3"),
    ])
    def test_bind_errors(self, fields, schema, match):
        """Missing names (listing the available ones), fieldless input, width/field mismatch."""
        with pytest.raises(ConfigError, match=match):
            _select(fields).bind(schema)

    def test_select_then_concat_resolves_fields_through_the_schema(self, tmp_path):
        """SelectFields -> VectorConcat propagates column names through resolve_bind_schema."""
        nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
        write_vector_concat_norm_dict(nd, cd)
        modules = build_vector_concat_modules(nd)
        extra = {
            "sel_eta": SelectFields("normed.global", [GLOBAL_VARIABLES[1]], "sel_eta.global"),
            "sel_pt": SelectFields("normed.jets", [JET_VARIABLES[0]], "sel_pt.global"),
            "named": VectorConcat(inputs=["sel_eta.global", "sel_pt.global"], out="named.global"),
            "vconcat": VectorConcat(inputs=["pooled.global", "named.global"], out="vconcat.global"),
        }
        for name, module in extra.items():
            module.name = name
        modules.update(extra)
        plans = [compile_vector_concat(modules, mode) for mode in (Mode.FIT, Mode.TEST)]
        schema = resolve_bind_schema(plans)
        assert schema.fields_of("named.global") == (GLOBAL_VARIABLES[1], JET_VARIABLES[0])
        assert schema.width("named.global") == 2
        assert "vconcat.global" not in schema.fields  # pooled.global carries no names
        bind_all(modules, schema)
        assert modules["sel_eta"]._index.tolist() == [1]  # noqa: SLF001
        assert modules["sel_pt"]._index.tolist() == [0]  # noqa: SLF001
