"""Unit tests for the standalone config-constructed nn modules (plan 05, stage A2).

Covers, per module: construction from plain kwargs (no YAML anchors, no
``input_size`` arithmetic — design §2.3), `declare_io` correctness (keys,
kinds, mode gating), bind-time shape inference via `resolve_bind_schema`,
the Normaliser materialise lifecycle, the declarative class-weight source,
and full FIT/TEST executions through the M1 Executor in debug mode
(read-tracking + write-once + mutation detection on).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from salt.core.graph import (
    Bundle,
    ConfigError,
    Executor,
    Mode,
    PlanStep,
    compile_plan,
    flatten_spec,
)
from salt.core.nn import (
    BindError,
    Concat,
    GlobalAttentionPooling,
    LossGLS,
    LossSum,
    MaskDecoder,
    Normaliser,
    ResolvedSchema,
    Split,
    StreamEmbed,
    TransformerEncoder,
    VectorConcat,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.core.nn.tasks import (
    ClassificationTaskModule,
    RegressionTaskModule,
    VertexingTaskModule,
)
from salt.models import Dense as V1Dense
from salt.models import Transformer as V1Transformer
from salt.core.schema import GroupSchema, Schema
from salt.tests.core.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
    make_gn2_batch,
    write_parity_norm_dict,
)
from salt.tests.core.gn2v2_fixture import (
    ORIGIN_CLASSES,
    build_gn2v2_modules,
    compile_gn2v2,
    gn2v2_sources,
    make_gn2_labels,
)
from salt.tests.core.regression_fixture import (
    GLOBAL_VARIABLES,
    MASKFORMER_NUM_OBJECTS,
    build_independent_v1_mask_decoder,
    build_independent_v1_transformer_drop,
    build_maskformer_decoder_modules,
    build_regression_modules,
    build_vector_concat_modules,
    compile_maskformer_decoder,
    compile_regression,
    compile_vector_concat,
    make_regression_labels,
    write_vector_concat_norm_dict,
)

B, T = 6, 10


@pytest.fixture
def norm_paths(tmp_path):
    """Write the parity norm/class dicts; return (norm_dict, class_dict) paths."""
    nd, cd = tmp_path / "norm_dict.yaml", tmp_path / "class_dict.yaml"
    write_parity_norm_dict(nd, cd)
    return nd, cd


@pytest.fixture
def gn2v2(norm_paths):
    """Build, compile (FIT), bind, and materialise the small GN2v2."""
    modules = build_gn2v2_modules(norm_paths[0])
    plan = compile_gn2v2(modules, Mode.FIT)
    schema = resolve_bind_schema(plan)
    bind_all(modules, schema)
    materialise_all(modules)
    return modules, plan, schema


def fit_bundle(n_electrons: int = 0) -> Bundle:
    """Build a FIT-mode bundle from the deterministic v1 batch + labels."""
    inputs, masks = make_gn2_batch(B, T, n_electrons=n_electrons)
    labels = make_gn2_labels(B, T)
    b = Bundle()
    for stream, x in inputs.items():
        b.set(f"inputs.{stream}", x)
    for stream, m in masks.items():
        b.set(f"masks.{stream}", m)
    for stream, fields in labels.items():
        for name, val in fields.items():
            b.set(f"labels.{stream}.{name}", val)
    return b


# ---------------------------------------------------------------------------
# bind schema resolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# construction + declare_io per module
# ---------------------------------------------------------------------------


class TestNormaliser:
    def test_init_does_no_file_io(self, tmp_path):
        """__init__ records the path only — the file need not exist (design §2.3)."""
        norm = Normaliser(norm_dict=tmp_path / "absent.yaml", streams=["tracks"])
        io = norm.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"inputs.tracks"}
        assert set(flatten_spec(io.produces)) == {"normed.tracks"}

    def test_global_object_is_rank_two(self):
        norm = Normaliser(norm_dict="x.yaml", streams=["jets", "tracks"], global_object="jets")
        norm.name = "norm"
        flat = flatten_spec(norm.declare_io(Mode.FIT).requires)
        assert len(flat["inputs.jets"].shape) == 2
        assert len(flat["inputs.tracks"].shape) == 3

    def test_config_errors(self):
        with pytest.raises(ConfigError, match="non-empty"):
            Normaliser(norm_dict="x.yaml", streams=[])
        with pytest.raises(ConfigError, match="duplicate"):
            Normaliser(norm_dict="x.yaml", streams=["a", "a"])
        with pytest.raises(ConfigError, match="global_object"):
            Normaliser(norm_dict="x.yaml", streams=["a"], global_object="b")

    def test_materialise_fills_buffers_to_v1_values(self, norm_paths, gn2v2):
        """Buffer values equal v1 InputNorm's from the same norm dict."""
        modules, _, _ = gn2v2
        wrapper = build_test_gn2(norm_paths[0].parent)
        norm = modules["norm"]
        assert torch.equal(norm.means_tracks, wrapper.norm.tracks_means)
        assert torch.equal(norm.stds_tracks, wrapper.norm.tracks_stds)
        assert torch.equal(norm.means_jets, wrapper.norm.jets_means)
        assert bool(norm.materialised)

    def test_forward_before_materialise_raises(self, norm_paths):
        modules = build_gn2v2_modules(norm_paths[0])
        plan = compile_gn2v2(modules, Mode.FIT)
        bind_all(modules, resolve_bind_schema(plan))
        with pytest.raises(RuntimeError, match="materialise"):
            modules["norm"](fit_bundle(), Mode.FIT)

    def test_forward_produces_new_keys_and_never_mutates(self, gn2v2):
        modules, _, _ = gn2v2
        b = fit_bundle()
        before = b.get("inputs.tracks").clone()
        out = modules["norm"](b, Mode.FIT)
        assert set(out) == {"normed.jets", "normed.tracks"}
        assert torch.equal(b.get("inputs.tracks"), before)
        expected = (before - modules["norm"].means_tracks) / modules["norm"].stds_tracks
        assert torch.equal(out["normed.tracks"], expected)

    def test_materialise_missing_variable_raises(self, tmp_path, norm_paths):
        import yaml

        with open(norm_paths[0]) as fh:
            nd = yaml.safe_load(fh)
        del nd["tracks"]["d0"]
        bad = tmp_path / "bad_norm.yaml"
        with open(bad, "w") as fh:
            yaml.dump(nd, fh)
        modules = build_gn2v2_modules(bad)
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        with pytest.raises(ValueError, match="d0"):
            modules["norm"].materialise()


class TestStreamEmbed:
    def test_width_keys_rejected(self):
        with pytest.raises(ConfigError, match="input_size"):
            StreamEmbed(stream="tracks", out_dim=8, dense={"input_size": 21})

    def test_bind_infers_input_plus_context_width(self, gn2v2):
        modules, _, _ = gn2v2
        embed = modules["track_embed"]
        # 19 track vars + 2 jet context vars — initnet.py:54-59 inference
        assert embed.net.input_size == len(TRACK_VARIABLES) + len(JET_VARIABLES)
        assert embed.net.output_size == 16

    def test_declare_io(self):
        embed = StreamEmbed(stream="tracks", out_dim=8, context=["normed.jets"])
        embed.name = "track_embed"
        io = embed.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"normed.tracks", "normed.jets"}
        assert set(flatten_spec(io.produces)) == {"embed.tracks"}
        # rank-agnostic produce (M7 W1.5 wave R): shape=None, width via derived_widths
        assert flatten_spec(io.produces)["embed.tracks"].shape is None
        assert embed.derived_widths({}) == {"embed.tracks": 8}

    def test_context_prepended_in_v1_order(self, gn2v2):
        """Single GN2 context entry reproduces v1's [global, stream] layout."""
        modules, _, _ = gn2v2
        b = fit_bundle()
        normed = modules["norm"](b, Mode.FIT)
        for key, val in normed.items():
            b.set(key, val)
        out = modules["track_embed"](b, Mode.FIT)
        x = torch.cat(
            [
                b.get("normed.jets").unsqueeze(1).expand(B, T, -1),
                b.get("normed.tracks"),
            ],
            dim=-1,
        )
        assert torch.equal(out["embed.tracks"], modules["track_embed"].net(x))


class TestStreamEmbedVector:
    """Rank INFERRED from the bound input (M7 W1.5 wave R; collapses the M6-6 flag).

    `StreamEmbed` no longer carries a rank flag. It declares rank-AGNOSTIC specs
    (``shape=None`` on its ``normed.<s>`` require AND its ``embed.<s>`` produce),
    so the producer (Normaliser/reader, keyed on the single reader
    ``global_object:`` flag) sets the input rank and the consumer sets the output
    rank — a rank-2 ``[B, F]`` input yields a rank-2 ``[B, D]`` embed (DL1
    jets-only MLP), a rank-3 ``[B, T, F]`` input yields a rank-3 ``[B, T, D]``
    embed, all WITHOUT a model-side flag. The ``out_dim`` width is contributed via
    `derived_widths`. The forward math is identity to v1's no-context
    `InitNet`/`Dense` (initnet.py:72-89).
    """

    def test_declare_io_is_rank_agnostic(self):
        embed = StreamEmbed(stream="jets", out_dim=8)
        embed.name = "jet_embed"
        io = embed.declare_io(Mode.FIT)
        req = flatten_spec(io.requires)
        prod = flatten_spec(io.produces)
        assert set(req) == {"normed.jets"}
        assert set(prod) == {"embed.jets"}
        # rank-agnostic: no flag, the rank flows from producer/consumer
        assert req["normed.jets"].shape is None
        assert prod["embed.jets"].shape is None

    def test_derived_widths_contribute_out_dim(self):
        """The ``embed.<s>`` width comes from `derived_widths` (shape=None produce)."""
        embed = StreamEmbed(stream="jets", out_dim=8)
        embed.name = "jet_embed"
        assert embed.derived_widths({}) == {"embed.jets": 8}

    def test_bind_infers_vector_width(self):
        """A vector stream's Dense input width is the resolved [B, F] width."""
        embed = StreamEmbed(stream="jets", out_dim=16, dense={"hidden_layers": [32]})
        embed.name = "jet_embed"
        embed.bind(ResolvedSchema(widths={"normed.jets": 2}))
        assert embed.net.input_size == 2
        assert embed.net.output_size == 16

    def test_forward_rank_two_bitwise_vs_independent_v1(self):
        """[B, F] embed forward == an INDEPENDENT v1 no-context InitNet/Dense.

        The v2 `StreamEmbed.forward` on a ``[B, F]`` stream is ``net(x)`` with
        no context (the DL1 ``attach_global: false`` path, initnet.py:72-89);
        copying the bound Dense's weights into a fresh v1 `Dense` reference and
        running it on the SAME input must agree BITWISE (no float reordering).
        """
        torch.manual_seed(0)
        embed = StreamEmbed(stream="jets", out_dim=4, dense={"hidden_layers": [8, 8]})
        embed.name = "jet_embed"
        embed.bind(ResolvedSchema(widths={"normed.jets": 2}))
        # independent v1 reference Dense with the SAME hyper-params + weights
        ref = V1Dense(input_size=2, output_size=4, hidden_layers=[8, 8])
        ref.load_state_dict(embed.net.state_dict())
        x = torch.randn(B, 2)
        b = Bundle()
        b.set("normed.jets", x)
        out = embed(b, Mode.FIT)
        assert out["embed.jets"].shape == (B, 4)
        assert torch.equal(out["embed.jets"], ref(x))


class TestStreamEmbedMup:
    """The ``mup:`` flag on `StreamEmbed` (M6 sub-wave B, plan 12 muP arch port).

    The flag threads into the composed v1 `Dense(mup=True)` at bind, applying the
    muP weight init (``~N(0, 1/fan_out)`` weights, zeroed biases); the forward is
    unchanged. The flag must NOT be accepted inside ``dense`` (it is a module-level
    init_arg, not a Dense width key) — that path is the routing/config surface.
    """

    def test_default_is_not_mup(self):
        embed = StreamEmbed(stream="tracks", out_dim=8)
        assert embed.mup is False

    def test_mup_flag_threads_into_dense_at_bind(self):
        embed = StreamEmbed(stream="tracks", out_dim=8, mup=True)
        embed.name = "track_embed"
        embed.bind(ResolvedSchema(widths={"normed.tracks": 5}))
        assert embed.mup is True
        assert embed.net.mup is True

    def test_mup_in_dense_rejected(self):
        # mup is a module-level init_arg, not a dense width/option key
        with pytest.raises(ConfigError, match="set mup on the module"):
            StreamEmbed(stream="tracks", out_dim=8, dense={"mup": True})

    def test_mup_init_matches_independent_v1_dense_mup(self):
        """The bound mup Dense has the SAME parameter distribution as a v1 Dense(mup=True).

        Both run ``Dense.__init__``'s ``_reset_parameters`` (dense.py:96-102) under
        the same seed, so the initialised weights are BITWISE identical — the muP
        init is genuinely active (not the default torch init).
        """
        torch.manual_seed(0)
        embed = StreamEmbed(stream="tracks", out_dim=4, dense={"hidden_layers": [8]}, mup=True)
        embed.name = "track_embed"
        embed.bind(ResolvedSchema(widths={"normed.tracks": 5}))
        torch.manual_seed(0)
        ref = V1Dense(input_size=5, output_size=4, hidden_layers=[8], mup=True)
        for (k1, p1), (k2, p2) in zip(
            embed.net.state_dict().items(), ref.state_dict().items(), strict=True
        ):
            assert k1 == k2
            assert torch.equal(p1, p2), k1

    def test_mup_forward_is_unchanged(self):
        """muP affects init only — the forward math is the standard Dense forward."""
        torch.manual_seed(1)
        embed = StreamEmbed(stream="tracks", out_dim=4, dense={"hidden_layers": [8]}, mup=True)
        embed.name = "track_embed"
        embed.bind(ResolvedSchema(widths={"normed.tracks": 5}))
        ref = V1Dense(input_size=5, output_size=4, hidden_layers=[8], mup=True)
        ref.load_state_dict(embed.net.state_dict())
        x = torch.randn(B, T, 5)
        b = Bundle()
        b.set("normed.tracks", x)
        out = embed(b, Mode.FIT)
        assert torch.equal(out["embed.tracks"], ref(x))


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


class TestTransformerEncoder:
    def test_attention_config_required(self):
        with pytest.raises(ConfigError, match="num_heads"):
            TransformerEncoder(dim=16, num_layers=1, attention={})

    def test_declare_io_concrete_widths(self):
        enc = TransformerEncoder(dim=16, num_layers=1, out_dim=8, attention={"num_heads": 2})
        enc.name = "encoder"
        io = enc.declare_io(Mode.FIT)
        assert flatten_spec(io.requires)["seq.x"].shape[-1] == 16
        produces = flatten_spec(io.produces)
        assert produces["encoded.seq"].shape[-1] == 8
        assert produces["masks.registers"].shape == ("B", 1)
        assert produces["masks.registers"].kind == "pad_mask"

    def test_registers_internal_and_mask_published(self, gn2v2):
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        # register row appended after the T track positions
        assert b.get("encoded.seq").shape == (B, T + 1, 16)
        assert b.get("masks.registers").shape == (B, 1)
        assert not b.get("masks.registers").any()

    def test_norm_type_defaults_to_pre(self):
        enc = TransformerEncoder(dim=16, num_layers=2, attention={"num_heads": 2})
        assert enc.norm_type == "pre"
        for layer in enc.encoder.layers:
            assert layer.norm_type == "pre"
            # pre-norm does NOT force the qk/v attention norms
            assert not layer.attn.fn.do_qk_norm
            assert not layer.attn.fn.do_v_norm

    def test_norm_type_post_passthrough(self):
        enc = TransformerEncoder(dim=16, num_layers=2, attention={"num_heads": 2}, norm_type="post")
        assert enc.norm_type == "post"
        for layer in enc.encoder.layers:
            assert layer.norm_type == "post"
            assert layer.attn.norm_type == "post"

    def test_norm_type_hybrid_forces_qk_v_and_depth_split(self):
        # the GN3V01 flagship: hybrid forces do_qk_norm/do_v_norm on every layer,
        # depth 0 keeps a "pre" residual + Identity layer-norm, depth>0 uses a
        # "none" residual + a real layer-norm (v1 transformer.py:350-356,421)
        enc = TransformerEncoder(
            dim=16, num_layers=3, attention={"num_heads": 2}, norm_type="hybrid"
        )
        assert enc.norm_type == "hybrid"
        layers = enc.encoder.layers
        assert all(layer.norm_type == "hybrid" for layer in layers)
        assert all(layer.attn.fn.do_qk_norm and layer.attn.fn.do_v_norm for layer in layers)
        assert layers[0].attn.norm_type == "pre"
        assert isinstance(layers[0].norm, nn.Identity)
        for layer in layers[1:]:
            assert layer.attn.norm_type == "none"
            assert not isinstance(layer.norm, nn.Identity)

    def test_unknown_norm_type_rejected(self):
        # "none" is a v1 residual-only mode with no shipped v2 config — rejected
        with pytest.raises(ConfigError, match="norm_type must be one of"):
            TransformerEncoder(dim=16, num_layers=1, attention={"num_heads": 2}, norm_type="none")

    def test_drop_registers_declare_io_drops_register_mask(self):
        # drop_registers => no masks.registers produced (the v1 del pad_mask
        # ["REGISTERS"] shape, transformer.py:748)
        enc = TransformerEncoder(
            dim=16, num_layers=1, attention={"num_heads": 2}, num_registers=3, drop_registers=True
        )
        enc.name = "encoder"
        produces = flatten_spec(enc.declare_io(Mode.FIT).produces)
        assert "encoded.seq" in produces
        assert "masks.registers" not in produces
        # default (drop_registers=False) still publishes the mask
        enc2 = TransformerEncoder(dim=16, num_layers=1, attention={"num_heads": 2}, num_registers=3)
        enc2.name = "encoder"
        assert "masks.registers" in flatten_spec(enc2.declare_io(Mode.FIT).produces)

    def test_drop_registers_forward_strips_registers(self):
        enc = TransformerEncoder(
            dim=16, num_layers=2, attention={"num_heads": 2}, num_registers=3, drop_registers=True
        )
        enc.name = "encoder"
        b = Bundle()
        b.set("seq.x", torch.randn(B, T, 16))
        b.set("seq.mask", torch.zeros(B, T, dtype=torch.bool))
        out = enc(b, Mode.FIT)
        # registers stripped: encoded.seq is the T stream rows only, no +num_registers
        assert out["encoded.seq"].shape == (B, T, 16)
        assert "masks.registers" not in out


class TestTransformerEncoderMup:
    """The ``mup:`` flag on `TransformerEncoder` (M6 sub-wave B, plan 12 muP arch port).

    The flag threads into the composed v1 `Transformer(mup=True)`: the encoder
    out-proj becomes a ``mup.MuReadout`` (weight+bias zeroed at init), and
    ``mup.set_base_shapes(enc, enc, rescale_params=False)`` is applied so the
    MuReadout is forward-runnable (``width_mult == 1`` — the architectural default
    before the routing stage applies a real shape file). At export the MuReadout is
    folded to a plain `nn.Linear` (``set_export_mode``).
    """

    def test_default_is_not_mup(self):
        enc = TransformerEncoder(dim=16, num_layers=1, out_dim=8, attention={"num_heads": 2})
        assert enc.mup is False
        assert type(enc.encoder.out_proj) is nn.Linear

    def test_mup_requires_out_dim(self):
        with pytest.raises(ConfigError, match="mup requires an out_dim"):
            TransformerEncoder(dim=16, num_layers=1, attention={"num_heads": 2}, mup=True)

    def test_mup_swaps_out_proj_for_zeroed_mu_readout(self):
        from mup import MuReadout

        enc = TransformerEncoder(
            dim=16, num_layers=2, out_dim=8, attention={"num_heads": 2}, mup=True
        )
        assert enc.mup is True
        assert isinstance(enc.encoder.out_proj, MuReadout)
        # v1 zeroes both weight and bias of the readout at init (transformer.py:627-628)
        assert bool((enc.encoder.out_proj.weight == 0).all())
        assert bool((enc.encoder.out_proj.bias == 0).all())
        # set_base_shapes was applied at construction -> width_mult resolves to 1
        assert enc.encoder.out_proj.width_mult() == 1.0

    def test_mup_forward_runs_standalone(self):
        # without set_base_shapes the MuReadout forward would assert on infshape;
        # the construction-time set_base_shapes makes a standalone mup encoder
        # forward-runnable
        enc = TransformerEncoder(
            dim=16, num_layers=2, out_dim=8, attention={"num_heads": 2}, mup=True
        )
        enc.name = "encoder"
        b = Bundle()
        b.set("seq.x", torch.randn(B, T, 16))
        b.set("seq.mask", torch.zeros(B, T, dtype=torch.bool))
        out = enc(b, Mode.FIT)
        assert out["encoded.seq"].shape == (B, T + 1, 8)

    def test_mup_forward_bitwise_vs_independent_v1(self):
        """The v2 mup encoder forward == an INDEPENDENT v1 Transformer(mup=True).

        A separately built v1 ``Transformer(mup=True)``, weight-loaded from the v2
        encoder, run through v1 forward must agree BITWISE (same weights, same
        torch-math math — no reordering). This is the module-level analogue of the
        MU1 gate's forward-init parity.
        """
        torch.manual_seed(2)
        enc = TransformerEncoder(
            dim=16, num_layers=2, out_dim=8, attention={"num_heads": 2}, mup=True
        )
        enc.name = "encoder"
        # non-zero weights so the comparison is meaningful (zeroed readout -> all 0)
        with torch.no_grad():
            for p in enc.encoder.parameters():
                p.copy_(torch.randn(p.shape) * 0.1)
        from mup import set_base_shapes

        ref = V1Transformer(
            num_layers=2,
            embed_dim=16,
            out_dim=8,
            norm="LayerNorm",
            attn_type="torch-math",
            do_final_norm=True,
            num_registers=enc.num_registers,
            attn_kwargs={"num_heads": 2},
            dense_kwargs={"activation": "SiLU"},
            mup=True,
        )
        set_base_shapes(ref, ref, rescale_params=False)
        ref.load_state_dict(enc.encoder.state_dict())
        ref.eval()
        enc.encoder.eval()
        seq_x = torch.randn(B, T, 16)
        seq_mask = torch.zeros(B, T, dtype=torch.bool)
        b = Bundle()
        b.set("seq.x", seq_x)
        b.set("seq.mask", seq_mask)
        with torch.no_grad():
            v2 = enc(b, Mode.FIT)["encoded.seq"]
            v1, _ = ref({"seq": seq_x}, pad_mask={"seq": seq_mask})
        assert torch.equal(v2, v1)

    def test_set_export_mode_folds_mu_readout_to_plain_linear(self):
        """set_export_mode swaps the MuReadout for a plain Linear, forward unchanged.

        At the architectural default (``width_mult == 1``, ``output_mult == 1``) the
        fold is the identity multiplier, so the pre/post-fold forward is BITWISE
        identical. The out-proj becomes a plain ``nn.Linear`` (no MuReadout
        multiplier op left for the tracer). Idempotent.
        """
        torch.manual_seed(3)
        enc = TransformerEncoder(
            dim=16, num_layers=2, out_dim=8, attention={"num_heads": 2}, mup=True
        )
        enc.name = "encoder"
        # non-zero readout weights so the fold is a real comparison
        with torch.no_grad():
            enc.encoder.out_proj.weight.copy_(torch.randn(8, 16))
            enc.encoder.out_proj.bias.copy_(torch.randn(8))
        b = Bundle()
        b.set("seq.x", torch.randn(B, T, 16))
        b.set("seq.mask", torch.zeros(B, T, dtype=torch.bool))
        pre = enc(b, Mode.ONNX)["encoded.seq"].clone()
        enc.set_export_mode()
        assert type(enc.encoder.out_proj) is nn.Linear
        post = enc(b, Mode.ONNX)["encoded.seq"]
        assert torch.equal(pre, post)
        # idempotent: a second call leaves the plain Linear in place
        enc.set_export_mode()
        assert type(enc.encoder.out_proj) is nn.Linear

    def test_set_export_mode_no_mup_is_noop_on_out_proj(self):
        # a non-mup encoder keeps its plain Linear out-proj through set_export_mode
        enc = TransformerEncoder(dim=16, num_layers=1, out_dim=8, attention={"num_heads": 2})
        before = enc.encoder.out_proj
        enc.set_export_mode()
        assert enc.encoder.out_proj is before
        assert type(enc.encoder.out_proj) is nn.Linear


class TestMaskDecoder:
    def _build(self, norm_paths, **overrides):
        modules = build_maskformer_decoder_modules(norm_paths[0], **overrides)
        test_plan = compile_maskformer_decoder(modules, Mode.TEST)
        onnx_plan = compile_maskformer_decoder(modules, Mode.ONNX)
        bind_all(modules, resolve_bind_schema([test_plan, onnx_plan]))
        materialise_all(modules)
        return modules, test_plan, onnx_plan

    def test_missing_n_heads_rejected(self):
        with pytest.raises(ConfigError, match="n_heads"):
            MaskDecoder(embed_dim=16, num_objects=5, num_layers=2, class_net={"output_size": 3})

    def test_missing_class_output_size_rejected(self):
        with pytest.raises(ConfigError, match=r"class_net\.output_size is required"):
            MaskDecoder(embed_dim=16, num_objects=5, num_layers=2, class_net={}, md={"n_heads": 2})

    def test_class_net_width_key_rejected(self):
        with pytest.raises(ConfigError, match=r"class_net\.input_size"):
            MaskDecoder(
                embed_dim=16,
                num_objects=5,
                num_layers=2,
                class_net={"input_size": 16, "output_size": 3},
                md={"n_heads": 2},
            )

    def test_non_positive_dims_rejected(self):
        with pytest.raises(ConfigError, match="num_objects"):
            MaskDecoder(
                embed_dim=16,
                num_objects=0,
                num_layers=2,
                class_net={"output_size": 3},
                md={"n_heads": 2},
            )
        with pytest.raises(ConfigError, match="num_layers"):
            MaskDecoder(
                embed_dim=16,
                num_objects=5,
                num_layers=0,
                class_net={"output_size": 3},
                md={"n_heads": 2},
            )

    def test_declare_io_keys_and_shapes(self):
        md = MaskDecoder(
            embed_dim=16,
            num_objects=5,
            num_layers=2,
            class_net={"output_size": 3},
            md={"n_heads": 2, "mask_attention": True, "bidirectional_ca": True},
        )
        md.name = "mask_decoder"
        io = md.declare_io(Mode.FIT)
        req = flatten_spec(io.requires)
        assert set(req) == {"encoded.seq", "seq.mask"}
        produces = flatten_spec(io.produces)
        assert set(produces) == {
            "objects.embed",
            "objects.class_logits",
            "objects.class_probs",
            "objects.masks",
        }
        assert produces["objects.embed"].shape[1] == 5
        assert produces["objects.class_logits"].shape[-1] == 3
        # num_classes is C - 1 (last class = null)
        assert md.num_classes == 2

    def test_embed_dim_mismatch_at_bind_raises(self, norm_paths):
        # the fixture's encoder produces a 16-wide encoded.seq; a decoder declaring
        # embed_dim=32 must raise at bind (the queries cannot attend to a
        # width-mismatched sequence; v1 maskformer.py:48,172)
        modules = build_maskformer_decoder_modules(norm_paths[0])
        modules["mask_decoder"] = MaskDecoder(
            embed_dim=32,
            num_objects=5,
            num_layers=2,
            class_net={"output_size": 3},
            md={"n_heads": 2, "mask_attention": True, "bidirectional_ca": True},
        )
        modules["mask_decoder"].name = "mask_decoder"
        plan = compile_maskformer_decoder(modules, Mode.TEST)
        with pytest.raises(ConfigError, match="embed_dim"):
            bind_all(modules, resolve_bind_schema([plan]))

    def test_forward_through_executor_shapes(self, norm_paths):
        modules, test_plan, _ = self._build(norm_paths)
        inputs, masks = make_gn2_batch(B, T)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        out = Executor(test_plan).run(b, debug=True)
        d = modules["mask_decoder"]
        emb = out.get("objects.embed")
        assert emb.shape == (B, MASKFORMER_NUM_OBJECTS, d.embed_dim)
        # the dummy token is stripped: masks span the T constituents (not T + 1)
        assert out.get("objects.masks").shape == (B, MASKFORMER_NUM_OBJECTS, T)
        # class_probs is a proper distribution over C
        probs = out.get("objects.class_probs")
        assert torch.allclose(probs.sum(-1), torch.ones(B, MASKFORMER_NUM_OBJECTS), atol=1e-5)

    def test_forward_bitwise_vs_independent_v1(self, norm_paths):
        # the four objects.* outputs == an INDEPENDENT v1 MaskDecoder BITWISE; the
        # encoded.seq == an INDEPENDENT v1 Transformer(drop_registers=True) BITWISE
        modules, test_plan, _ = self._build(norm_paths)
        inputs, masks = make_gn2_batch(B, T)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        out = Executor(test_plan).run(b, debug=True)
        # drop_registers parity
        v1_enc = build_independent_v1_transformer_drop(modules["encoder"])
        with torch.no_grad():
            v1_encoded, v1_pad = v1_enc(
                {"seq": out.get("seq.x").clone()}, pad_mask={"seq": out.get("seq.mask").clone()}
            )
        assert torch.equal(out.get("encoded.seq"), v1_encoded)
        assert "REGISTERS" not in v1_pad
        # decoder parity
        v1_dec = build_independent_v1_mask_decoder(modules["mask_decoder"])
        with torch.no_grad():
            preds, _, _ = v1_dec(
                {"embed_xs": out.get("encoded.seq").clone()},
                tasks=[],
                pad_mask=out.get("seq.mask").clone(),
                labels=None,
            )
        obj = preds["objects"]
        assert torch.equal(out.get("objects.embed"), obj["embed"])
        assert torch.equal(out.get("objects.class_logits"), obj["class_logits"])
        assert torch.equal(out.get("objects.class_probs"), obj["class_probs"])
        assert torch.equal(out.get("objects.masks"), obj["masks"])

    def test_zero_constituent_jet_is_finite(self):
        # the dummy-token trick keeps a zero-length sequence from NaN-ing (ONNX)
        md = MaskDecoder(
            embed_dim=16,
            num_objects=5,
            num_layers=2,
            class_net={"output_size": 3},
            md={"n_heads": 2, "mask_attention": True, "bidirectional_ca": True},
        )
        md.name = "mask_decoder"
        b = Bundle()
        b.set("encoded.seq", torch.zeros(1, 0, 16))
        b.set("seq.mask", torch.zeros(1, 0, dtype=torch.bool))
        out = md(b, Mode.ONNX)
        assert torch.isfinite(out["objects.embed"]).all()
        assert out["objects.masks"].shape == (1, 5, 0)

    def test_binary_class_net_sigmoid_expands(self):
        # output_size == 1 reproduces v1's sigmoid -> 2-column class_probs
        # (maskformer.py:106-110)
        md = MaskDecoder(
            embed_dim=16,
            num_objects=5,
            num_layers=2,
            class_net={"output_size": 1},
            md={"n_heads": 2, "mask_attention": True, "bidirectional_ca": True},
        )
        md.name = "mask_decoder"
        b = Bundle()
        b.set("encoded.seq", torch.randn(4, 7, 16))
        b.set("seq.mask", torch.zeros(4, 7, dtype=torch.bool))
        out = md(b, Mode.TEST)
        probs = out["objects.class_probs"]
        assert probs.shape[-1] == 2  # 1 logit sigmoid-expanded to [1-p, p]
        assert torch.allclose(probs.sum(-1), torch.ones(4, 5), atol=1e-5)


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


class TestDerivedWidthsFixpoint:
    """`_apply_derived_widths` resolves a CHAIN of derived widths order-independently."""

    @staticmethod
    def _chained_plan(order):
        """A duck-typed plan whose steps hold two chained derived-width modules.

        Module ``a`` derives ``mid`` from a concrete ``src``; module ``b``
        derives ``out`` from ``mid`` (b's input is a's output). ``order`` picks
        the step order so the dependent module can be visited FIRST.
        """

        class _Mod:
            def __init__(self, name, in_key, out_key):
                self.name, self._in, self._out = name, in_key, out_key

            def derived_widths(self, widths):
                src = widths.get(self._in)
                return {self._out: src + 1} if src is not None else {}

        mods = {
            "a": _Mod("a", "src", "mid"),
            "b": _Mod("b", "mid", "out"),
        }
        steps = tuple(PlanStep(name=n, module=mods[n]) for n in order)
        return SimpleNamespace(steps=steps)

    def test_chain_resolves_when_dependent_visited_first(self):
        # b (depends on mid) BEFORE a (produces mid): a single pass would leave
        # 'out' unbound; the fixpoint loop re-runs b after a binds 'mid'.
        from salt.core.nn.bind import _apply_derived_widths

        widths = {"src": 10}
        _apply_derived_widths([self._chained_plan(["b", "a"])], widths)
        assert widths == {"src": 10, "mid": 11, "out": 12}

    def test_chain_resolves_for_either_order(self):
        from salt.core.nn.bind import _apply_derived_widths

        for order in (["a", "b"], ["b", "a"]):
            widths = {"src": 10}
            _apply_derived_widths([self._chained_plan(order)], widths)
            assert widths == {"src": 10, "mid": 11, "out": 12}, order

    def test_conflicting_derived_width_still_raises(self):
        from salt.core.nn.bind import _apply_derived_widths

        class _Mod:
            name = "clash"

            def derived_widths(self, widths):
                return {"out": 99}

        widths = {"out": 7}
        with pytest.raises(BindError, match="conflicting widths"):
            _apply_derived_widths(
                [SimpleNamespace(steps=(PlanStep(name="clash", module=_Mod()),))], widths
            )


class TestSplitAndPooling:
    def test_split_slices_by_layout(self, gn2v2):
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        assert b.get("encoded.tracks").shape == (B, T, 16)
        assert torch.equal(b.get("encoded.tracks"), b.get("encoded.seq")[:, :T])

    def test_pooling_binds_gate_width(self, gn2v2):
        modules, _, _ = gn2v2
        assert modules["pool"].pool_net.gate_nn.in_features == 16

    def test_pooled_shape(self, gn2v2):
        _, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        assert b.get("pooled.global").shape == (B, 16)


# ---------------------------------------------------------------------------
# encoder-less pooling (M5; DiPS/DeepSets — init_nets + pool_net, no encoder)
# ---------------------------------------------------------------------------


def _encoderless_modules(norm_dict):
    """A DiPS-shaped module dict: norm -> embed -> concat -> pool -> head -> loss.

    No `TransformerEncoder`, no `Split` (v1 saltmodel.py:90-93,155-156): the
    pool reads ``seq.x`` (the `Concat` output) and there is no
    ``masks.registers`` producer at all.
    """
    dense = {"hidden_layers": [16], "activation": "ReLU"}
    modules = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["jets", "tracks"], global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=16, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="seq.x", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            dense=dense,
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    modules["loss"].narrow(LossSum.collect_loss_keys(modules))
    return modules


class TestEncoderlessPooling:
    def test_registers_require_is_optional(self):
        """`masks.registers` is optional so an encoder-less config plan-compiles."""
        pool = GlobalAttentionPooling(input="seq.x")
        pool.name = "pool"
        req = flatten_spec(pool.declare_io(Mode.FIT).requires)
        assert req["masks.registers"].optional
        assert not req["seq.mask"].optional

    def test_encoderless_plan_compiles_fit_and_test(self, norm_paths):
        """No encoder -> no `masks.registers` producer; FIT and TEST still compile."""
        modules = _encoderless_modules(norm_paths[0])
        fit = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        test = compile_plan(
            modules,
            Mode.TEST,
            sources=gn2v2_sources(),
            sinks=["preds.jets.jets_classification"],
        )
        # the encoder is genuinely absent, but pooling still resolves
        assert "encoder" not in fit.module_names
        assert "pool" in fit.module_names and "pool" in test.module_names
        for plan in (fit, test):
            assert all(edge.key != "masks.registers" for edge in plan.edges), (
                "no register mask edge on the encoder-less path"
            )

    def test_encoderless_forward_runs(self, norm_paths):
        """Full FIT execution (debug: read-tracking + write-once) with no encoder."""
        modules = _encoderless_modules(norm_paths[0])
        plan = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        bind_all(modules, resolve_bind_schema(plan))
        materialise_all(modules)
        b = Executor(plan).run(fit_bundle(), debug=True)
        assert b.get("pooled.global").shape == (B, 16)
        assert torch.isfinite(b.get("loss.total"))

    def test_encoderless_pool_equals_v1_no_registers(self, norm_paths):
        """The pooled vector == a direct v1 GAP call with the {"seq": seq.mask}
        pad dict (the exact encoder-less semantics: no REGISTERS row)."""
        modules = _encoderless_modules(norm_paths[0])
        plan = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        bind_all(modules, resolve_bind_schema(plan))
        materialise_all(modules)
        # capture the seq.x / seq.mask the pool actually sees
        captured: dict[str, torch.Tensor] = {}
        executor = Executor(plan)
        b = executor.run(fit_bundle())
        captured["seq.x"] = b.get("seq.x")
        captured["seq.mask"] = b.get("seq.mask")
        v2_pooled = b.get("pooled.global")
        v1_pool = modules["pool"].pool_net  # the SAME composed v1 instance
        v1_pooled = v1_pool({"seq": captured["seq.x"]}, pad_mask={"seq": captured["seq.mask"]})
        assert torch.equal(v2_pooled, v1_pooled)


class TestLossSum:
    def test_unnarrowed_declare_raises(self):
        loss = LossSum()
        loss.name = "loss"
        with pytest.raises(ConfigError, match="framework"):
            loss.declare_io(Mode.FIT)

    def test_explicit_losses_config(self):
        loss = LossSum(losses=["jets_classification", "losses.track_origin"])
        loss.name = "loss"
        io = loss.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {
            "losses.jets_classification",
            "losses.track_origin",
        }
        assert set(flatten_spec(io.produces)) == {"loss.total"}

    def test_inactive_outside_training(self):
        loss = LossSum(losses=["a"])
        loss.name = "loss"
        io = loss.declare_io(Mode.TEST)
        assert not flatten_spec(io.requires) and not flatten_spec(io.produces)

    def test_unknown_weight_key_rejected(self):
        with pytest.raises(ConfigError, match="unknown loss keys"):
            LossSum(losses=["a"], weights={"b": 2.0})

    def test_weighted_sum(self):
        loss = LossSum(losses=["a", "b"], weights={"b": 2.0})
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(1.0))
        b.set("losses.b", torch.tensor(3.0))
        out = loss(b, Mode.FIT)
        assert out["loss.total"].item() == pytest.approx(7.0)

    def test_collect_loss_keys(self, gn2v2):
        modules, _, _ = gn2v2
        assert LossSum.collect_loss_keys(modules) == (
            "losses.jets_classification",
            "losses.track_origin",
            "losses.track_vertexing",
        )

    def test_double_narrow_rejected(self, gn2v2):
        modules, _, _ = gn2v2
        with pytest.raises(ConfigError, match="already fixed"):
            modules["loss"].narrow(["losses.x"])


class TestLossGLS:
    """`LossGLS` — geometric-mean combination + the all-weights==1.0 guard."""

    def test_is_a_losssum_subclass(self):
        # the SaltModule narrow loop keys off isinstance(_, LossSum), so a
        # LossGLS must be picked up for free (collect_loss_keys/narrow/declare_io)
        assert issubclass(LossGLS, LossSum)
        assert isinstance(LossGLS(losses=["a"]), LossSum)

    def test_geometric_mean_forward(self):
        loss = LossGLS(losses=["a", "b", "c"])
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(2.0))
        b.set("losses.b", torch.tensor(8.0))
        b.set("losses.c", torch.tensor(4.0))
        out = loss(b, Mode.FIT)
        # geometric mean of 2, 8, 4 is the cube root of 64, which is 4.0
        assert out["loss.total"].item() == pytest.approx(4.0)

    def test_two_task_geometric_mean_not_sum(self):
        # the smallest genuinely-combined case (the GN3 journey): >= 2 losses
        loss = LossGLS(losses=["a", "b"])
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(2.0))
        b.set("losses.b", torch.tensor(8.0))
        total = loss(b, Mode.FIT)["loss.total"]
        assert total.item() == pytest.approx(4.0)  # sqrt(16), NOT the sum 10
        assert total.item() != pytest.approx(10.0)

    def test_inherits_losssum_skeleton(self):
        # declare_io is the LossSum skeleton: losses.* -> loss.total (TRAINING),
        # empty outside TRAINING
        loss = LossGLS(losses=["a", "b"])
        loss.name = "loss"
        io = loss.declare_io(Mode.FIT)
        assert set(flatten_spec(io.requires)) == {"losses.a", "losses.b"}
        assert set(flatten_spec(io.produces)) == {"loss.total"}
        assert not flatten_spec(loss.declare_io(Mode.TEST).requires)

    def test_module_weight_not_one_rejected(self):
        # GLS does not utilise weights — the per-loss weights surface is guarded
        with pytest.raises(ConfigError, match="not utilised by the geometric mean"):
            LossGLS(losses=["a", "b"], weights={"b": 2.0})

    def test_module_weight_one_accepted(self):
        # an explicit weight of exactly 1.0 is a no-op, not an error
        loss = LossGLS(losses=["a", "b"], weights={"a": 1.0, "b": 1.0})
        loss.name = "loss"
        b = Bundle()
        b.set("losses.a", torch.tensor(4.0))
        b.set("losses.b", torch.tensor(9.0))
        assert loss(b, Mode.FIT)["loss.total"].item() == pytest.approx(6.0)  # sqrt(36)

    def test_check_task_weights_rejects_weighted_task(self):
        # the task-side weight: float surface — the v2 home of v1's ctor assert
        # (modelwrapper.py:139-142). check_task_weights inspects sibling tasks.
        weighted = ClassificationTaskModule(
            stream="jets", label="flav", class_names=["u", "b"], weight=2.0
        )
        weighted.name = "jets_classification"
        unit = RegressionTaskModule(stream="jets", targets="pt", input="pooled.global", weight=1.0)
        unit.name = "reg"
        gls = LossGLS()
        gls.name = "loss"
        modules = {"jets_classification": weighted, "reg": unit, "loss": gls}
        with pytest.raises(ConfigError, match="does not utilise task weights"):
            LossGLS.check_task_weights(modules)

    def test_check_task_weights_accepts_unit_weights(self):
        # all task weights 1.0 (the GLS validity domain) passes cleanly; the
        # loss module itself (a LossSum subclass) is skipped, not flagged
        t1 = ClassificationTaskModule(
            stream="jets", label="flav", class_names=["u", "b"], weight=1.0
        )
        t1.name = "jets_classification"
        t2 = RegressionTaskModule(stream="jets", targets="pt", input="pooled.global", weight=1.0)
        t2.name = "reg"
        gls = LossGLS()
        gls.name = "loss"
        LossGLS.check_task_weights({"jets_classification": t1, "reg": t2, "loss": gls})

    def test_check_task_weights_catches_int_weight(self):
        # hardening: the guard is duck-typed numeric (int OR float), so a future
        # task module storing an un-coerced int weight is still caught — int 2
        # raises, int 1 passes, a non-numeric weight is ignored (no numeric
        # weight to guard). The shipped task base float-coerces (tasks.py:88), so
        # this is forward-robustness, not a config-reachable path today.
        class _IntWeightTask:
            def __init__(self, w):
                self.weight = w

        with pytest.raises(ConfigError, match="does not utilise task weights"):
            LossGLS.check_task_weights({"t": _IntWeightTask(2)})  # int, not float
        # int 1 (== 1.0) and a non-numeric weight both pass cleanly
        LossGLS.check_task_weights({"t": _IntWeightTask(1)})
        LossGLS.check_task_weights({"t": _IntWeightTask("not-a-number")})


# ---------------------------------------------------------------------------
# task modules
# ---------------------------------------------------------------------------


class TestClassificationTaskModule:
    def test_class_names_required(self):
        with pytest.raises(ConfigError, match="class_names"):
            ClassificationTaskModule(stream="jets", label="flavour_label", class_names=[])

    def test_sequence_inference(self):
        seq_task = ClassificationTaskModule(stream="tracks", label="l", class_names=["a", "b"])
        glob_task = ClassificationTaskModule(
            stream="jets", label="l", class_names=["a", "b"], input="pooled.global"
        )
        assert seq_task.sequence and not glob_task.sequence

    def test_declare_io_mode_gating(self):
        task = ClassificationTaskModule(
            stream="tracks", label="lbl", class_names=["a", "b"], context="pooled.global"
        )
        task.name = "track_origin"
        io = task.declare_io(Mode.FIT)
        req = flatten_spec(io.requires)
        assert set(req) == {
            "encoded.tracks",
            "pooled.global",
            "masks.tracks",
            "labels.tracks.lbl",
        }
        assert req["labels.tracks.lbl"].kind == "label"
        assert req["labels.tracks.lbl"].modes == Mode.TRAINING
        assert req["masks.tracks"].kind == "pad_mask"
        produces = flatten_spec(io.produces)
        assert produces["preds.tracks.track_origin"].modes == Mode.ALL
        assert produces["losses.track_origin"].modes == Mode.TRAINING
        assert produces["losses.track_origin"].kind == "loss"

    def test_bind_builds_v1_head_with_inferred_widths(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_origin"].task
        assert head.net.input_size == 16
        assert head.net.context_size == 16
        assert head.net.output_size == len(ORIGIN_CLASSES)
        assert head.loss.ignore_index == -1  # v1 contract (task.py:123-124)

    def test_weight_source_literal_conflict(self):
        with pytest.raises(ConfigError, match="already specified"):
            ClassificationTaskModule(
                stream="tracks",
                label="l",
                class_names=["a", "b"],
                loss={"class_path": "torch.nn.CrossEntropyLoss", "init_args": {"weight": [1, 2]}},
                weight_source={"from_class_dict": "cd.yaml"},
            )

    def test_weight_source_materialise(self, norm_paths):
        """Class weights fill the CE buffer at materialise (design §3.3)."""
        modules = build_gn2v2_modules(norm_paths[0], class_dict=norm_paths[1])
        bind_all(modules, resolve_bind_schema(compile_gn2v2(modules, Mode.FIT)))
        head = modules["track_origin"].task
        assert torch.equal(head.loss.weight, torch.ones(8))  # bind: allocated, identity
        materialise_all(modules)
        expected = torch.tensor([4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3])
        assert torch.allclose(head.loss.weight, expected)
        # the buffer is in the state dict (resume inherits it, cli.py:253-267)
        assert "task.loss.weight" in modules["track_origin"].state_dict()

    def test_weight_source_length_mismatch_raises(self, norm_paths):
        """The fixture class dict has 4 flavour weights but 3 class names."""
        task = ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            weight_source={"from_class_dict": str(norm_paths[1])},
        )
        task.name = "jets_classification"
        task.bind(ResolvedSchema(widths={"pooled.global": 16}))
        with pytest.raises(ValueError, match="3 class_names"):
            task.materialise()


class TestVertexingTaskModule:
    def test_label_must_contain_vertex_index(self):
        with pytest.raises(ConfigError, match="VertexIndex"):
            VertexingTaskModule(stream="tracks", label="vtx", origin_label="o")

    def test_unknown_origin_weighting_key(self):
        with pytest.raises(ConfigError, match="origin_weighting"):
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="o",
                origin_weighting={"heavy": [3], "bogus": [1]},
            )

    def test_declares_origin_label_dependency(self):
        task = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="ftagTruthOriginLabel"
        )
        task.name = "track_vertexing"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert "labels.tracks.ftagTruthOriginLabel" in req
        assert "labels.tracks.ftagTruthVertexIndex" in req

    def test_bind_pair_widths_and_reduction_check(self, gn2v2):
        modules, _, _ = gn2v2
        head = modules["track_vertexing"].task
        assert head.net.input_size == 2 * 16  # pair concat (task.py:894-896)
        assert head.net.context_size == 16
        task = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="o",
            loss={"class_path": "torch.nn.BCEWithLogitsLoss", "init_args": {"reduction": "mean"}},
        )
        task.name = "v"
        with pytest.raises(ConfigError, match="reduction"):
            task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))

    def test_default_origin_weighting_matches_v1(self, gn2v2):
        """Configured default ids reproduce v1's hardcoded weights bitwise."""
        from salt.models.task import VertexingTask as V1

        modules, _, _ = gn2v2
        head = modules["track_vertexing"].task
        labels = torch.tensor([[0, 1, 3, 4, 5, 2]])
        n = labels.shape[1]
        adjmat = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
        v1_weights = V1.get_weights(head, labels, adjmat)
        v2_weights = head.get_weights(labels, adjmat)
        assert torch.equal(v1_weights, v2_weights)


# ---------------------------------------------------------------------------
# RegressionTaskModule (M5 sub-wave A: targets/denom/norm_params/scaler,
# custom_output_names, sequence, multi-output, mode-split de-scaling)
# ---------------------------------------------------------------------------


def _bind_reg_module(task: RegressionTaskModule, schema_widths: dict[str, int]) -> None:
    """Bind a standalone regression head against fixed widths (no fields)."""
    task.bind(ResolvedSchema(widths=schema_widths))


class TestRegressionTaskModule:
    def test_targets_required(self):
        with pytest.raises(ConfigError, match="targets is required"):
            RegressionTaskModule(stream="jets", targets=[])

    def test_single_scaling_method_guard(self):
        # v1 task.py:355 — at most one of denom/norm_params/scaler
        with pytest.raises(ConfigError, match="single scaling method"):
            RegressionTaskModule(
                stream="jets",
                targets="x",
                norm_params={"mean": 1.0, "std": 1.0},
                target_denominators="y",
            )

    def test_custom_output_name_count_mismatch(self):
        with pytest.raises(ConfigError, match="custom_output_names"):
            RegressionTaskModule(stream="jets", targets=["a", "b"], custom_output_names="only_one")

    def test_denominator_count_mismatch(self):
        with pytest.raises(ConfigError, match="target_denominators"):
            RegressionTaskModule(stream="jets", targets=["a", "b"], target_denominators="d")

    def test_norm_params_requires_mean_and_std(self):
        with pytest.raises(ConfigError, match="mean.*std|norm_params"):
            RegressionTaskModule(stream="jets", targets="x", norm_params={"mean": 1.0})

    def test_sequence_inference(self):
        seq = RegressionTaskModule(stream="tracks", targets="x")
        glob = RegressionTaskModule(stream="jets", targets="x", input="pooled.global")
        assert seq.sequence and not glob.sequence

    def test_output_suffixes_custom_override(self):
        task = RegressionTaskModule(
            stream="jets", targets=["mass", "pt"], custom_output_names=["truthMass", "truthPt"]
        )
        assert task.output_suffixes == ("truthMass", "truthPt")
        bare = RegressionTaskModule(stream="jets", targets=["mass", "pt"])
        assert bare.output_suffixes == ("mass", "pt")

    def test_declare_io_mode_gating_global(self):
        task = RegressionTaskModule(stream="jets", targets=["t1", "t2"], input="pooled.global")
        task.name = "regression"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert set(req) == {"pooled.global", "labels.jets.t1", "labels.jets.t2"}
        assert req["labels.jets.t1"].kind == "label"
        assert req["labels.jets.t1"].modes == Mode.TRAINING
        produces = flatten_spec(task.declare_io(Mode.FIT).produces)
        assert produces["preds.jets.regression"].modes == Mode.ALL
        assert produces["preds.jets.regression"].shape == ("B", 2)
        assert produces["losses.regression"].modes == Mode.TRAINING

    def test_declare_io_sequence_shapes(self):
        task = RegressionTaskModule(stream="tracks", targets=["a", "b", "c"])
        task.name = "regression"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert "masks.tracks" in req and req["masks.tracks"].kind == "pad_mask"
        pred = flatten_spec(task.declare_io(Mode.TEST).produces)["preds.tracks.regression"]
        assert pred.shape == ("B", "T:tracks", 3)

    def test_mode_split_denominator_dependency(self):
        """FIT|VAL|TEST demand labels.<denom>; ONNX adds the input Feature."""
        task = RegressionTaskModule(
            stream="jets",
            targets="HadronConeExclTruthLabelPt",
            input="pooled.global",
            target_denominators="pt_btagJes",
        )
        task.name = "regression"
        for mode in (Mode.FIT, Mode.TEST):
            req = flatten_spec(task.declare_io(mode).requires)
            assert "labels.jets.pt_btagJes" in req
            assert "inputs.jets" not in req
        onnx_req = flatten_spec(task.declare_io(Mode.ONNX).requires)
        assert "inputs.jets" in onnx_req
        # the denominator label leaf is excluded from ONNX (modes gate it)
        assert onnx_req["labels.jets.pt_btagJes"].modes == Mode.FIT | Mode.VAL | Mode.TEST

    def test_bind_infers_widths_and_output_size(self):
        task = RegressionTaskModule(
            stream="jets", targets=["a", "b"], input="pooled.global", context="ctx"
        )
        task.name = "regression"
        _bind_reg_module(task, {"pooled.global": 16, "ctx": 8})
        assert task.task.net.input_size == 16
        assert task.task.net.context_size == 8
        assert task.task.net.output_size == 2

    def test_bind_rejects_denominator_not_an_input_feature(self):
        """A ratio denominator absent from inputs.<stream> Features fails at bind."""
        task = RegressionTaskModule(
            stream="jets",
            targets="t",
            input="pooled.global",
            target_denominators="not_a_feature",
        )
        task.name = "regression"
        schema = ResolvedSchema(
            widths={"pooled.global": 16, "inputs.jets": len(JET_VARIABLES)},
            fields={"inputs.jets": tuple(JET_VARIABLES)},
        )
        with pytest.raises(ConfigError, match="ONNX export graph de-scales"):
            task.bind(schema)

    def test_fit_forward_parity_norm_params(self, norm_paths):
        """FIT preds are RAW/scaled and loss == v1 head, scaled space (norm_params)."""
        targets = ("R10TruthLabel_R22v1_TruthJetMass", "R10TruthLabel_R22v1_TruthJetPt")
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
            weight=0.5,
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        out = Executor(fit).run(b, debug=True)
        v2_pred = out.get("preds.jets.regression")
        v2_loss = out.get("losses.regression")
        # the composed v1 head IS task.task — call it directly for the reference
        pooled = out.get("pooled.global")
        tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
        ref_pred, ref_loss = task.task(pooled, tdict, None, context=None)
        assert torch.allclose(v2_pred, ref_pred, atol=1e-6)  # FIT preds are raw/scaled
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_test_descale_uses_label_denominator(self, norm_paths):
        """TEST de-scales the ratio target with the LABEL denominator (v1 get_h5)."""
        targets, denoms = ("HadronConeExclTruthLabelPt",), ("pt_btagJes",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            target_denominators=list(denoms),
            custom_output_names="pt",
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets, denoms)
        test = compile_regression(modules, Mode.TEST, targets, denoms)
        onnx = compile_regression(modules, Mode.ONNX, targets, denoms)
        bind_all(modules, resolve_bind_schema([fit, test, onnx]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets, denoms)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        with torch.no_grad():
            b = Executor(test).run(b)
        v2_test = b.get("preds.jets.regression")
        pooled = b.get("pooled.global")
        with torch.no_grad():
            raw, _ = task.task(pooled, {}, None, context=None)
            ref = task.task.run_inference(
                raw.clone(), labels={"jets": {"pt_btagJes": labels["labels.jets.pt_btagJes"]}}
            )
        assert torch.allclose(v2_test, ref, atol=1e-6)

    def test_onnx_descale_uses_input_feature_by_name(self, norm_paths):
        """ONNX de-scales with the denominator gathered BY NAME from inputs.<stream>.

        The export graph has no label group (to_onnx.py:381-398), so the ratio
        denominator must come from the input Feature tensor — and the result
        differs from the TEST (label-sourced) de-scaling.
        """
        targets, denoms = ("HadronConeExclTruthLabelPt",), ("pt_btagJes",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            target_denominators=list(denoms),
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets, denoms)
        test = compile_regression(modules, Mode.TEST, targets, denoms)
        onnx = compile_regression(modules, Mode.ONNX, targets, denoms)
        bind_all(modules, resolve_bind_schema([fit, test, onnx]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets, denoms)

        def _run(plan, with_labels):
            bb = Bundle()
            for stream, x in inputs.items():
                bb.set(f"inputs.{stream}", x)
            bb.set("masks.tracks", masks["tracks"])
            if with_labels:
                for key, val in labels.items():
                    bb.set(key, val)
            with torch.no_grad():
                return Executor(plan).run(bb)

        b_onnx = _run(onnx, with_labels=False)
        b_test = _run(test, with_labels=True)
        v2_onnx = b_onnx.get("preds.jets.regression")
        v2_test = b_test.get("preds.jets.regression")
        # v1 reference: denominator from the inputs.jets column named pt_btagJes
        col = JET_VARIABLES.index("pt_btagJes")
        denom_from_input = inputs["jets"][..., col]
        pooled = b_onnx.get("pooled.global")
        with torch.no_grad():
            raw, _ = task.task(pooled, {}, None, context=None)
            ref = task.task.run_inference(
                raw.clone(), labels={"jets": {"pt_btagJes": denom_from_input}}
            )
        assert torch.allclose(v2_onnx, ref, atol=1e-6)
        # different denominator source -> different de-scaled values
        assert not torch.allclose(v2_onnx, v2_test, atol=1e-6)

    def test_sequence_scaler_descale_parity_and_nan_padding(self):
        """Per-token (sequence) regression + functional scaler: forward + de-scale.

        Reproduces the MaskFormer ``objects``-style query regression: the v1
        scaler ``run_inference`` indexes the 3D ``[B, L, R]`` preds
        (task.py:594-596) and nan-pads masked positions.
        """
        d = 16
        scaler = {"pt": {"op": "log", "op_scale": 0.2}, "mass": {"op": "linear", "op_scale": 10}}
        task = RegressionTaskModule(stream="tracks", targets=["pt", "mass"], scaler=scaler)
        task.name = "regression"
        _bind_reg_module(task, {"encoded.tracks": d})
        assert task.task.scaler is not None
        x = torch.randn(B, T, d)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, T - 1] = True
        gen = torch.Generator().manual_seed(5)
        b = Bundle()
        b.set("encoded.tracks", x)
        b.set("masks.tracks", mask)
        b.set("labels.tracks.pt", 1.0 + torch.rand(B, T, generator=gen))
        b.set("labels.tracks.mass", 1.0 + torch.rand(B, T, generator=gen))
        fit_out = task.forward(b, Mode.FIT)
        assert fit_out["preds.tracks.regression"].shape == (B, T, 2)
        assert torch.isfinite(fit_out["losses.regression"])
        bt = Bundle()
        bt.set("encoded.tracks", x)
        bt.set("masks.tracks", mask)
        with torch.no_grad():
            test_out = task.forward(bt, Mode.TEST)
        pred = test_out["preds.tracks.regression"]
        assert pred.shape == (B, T, 2)
        assert torch.isnan(pred[:, T - 1]).all()  # masked positions nan-padded
        # parity vs a direct v1 forward + scaler run_inference
        with torch.no_grad():
            raw, _ = task.task(x, {}, {"tracks": mask}, context=None)
            ref = task.task.run_inference(raw.clone(), labels=None, pad_mask=mask)
        assert torch.equal(torch.nan_to_num(pred), torch.nan_to_num(ref))

    # -- Gaussian (mu/sigma, output==2R, NLL, stddev=sqrt) — A3 -----------------

    def test_gaussian_output_size_is_doubled(self):
        task = RegressionTaskModule(
            stream="jets",
            targets="pt",
            input="pooled.global",
            gaussian=True,
            norm_params={"mean": 1.0, "std": 2.0},
        )
        task.name = "gaussian_regression"
        _bind_reg_module(task, {"pooled.global": 16})
        assert task.task.net.output_size == 2  # 2 * 1 target
        assert task.output_suffixes == ("pt", "pt_stddev")
        produces = flatten_spec(task.declare_io(Mode.TEST).produces)
        assert produces["preds.jets.gaussian_regression"].shape == ("B", 2)

    def test_gaussian_rejects_functional_scaler(self):
        with pytest.raises(ConfigError, match="gaussian head cannot use a functional 'scaler'"):
            RegressionTaskModule(
                stream="jets",
                targets="pt",
                input="pooled.global",
                gaussian=True,
                scaler={"pt": {"op": "log", "op_scale": 0.2}},
            )

    def test_gaussian_requires_a_scaling_method_at_bind(self):
        task = RegressionTaskModule(
            stream="jets", targets="pt", input="pooled.global", gaussian=True
        )
        task.name = "gaussian_regression"
        with pytest.raises(ConfigError, match="gaussian head requires norm_params"):
            _bind_reg_module(task, {"pooled.global": 16})

    def test_gaussian_fit_forward_parity_and_nll_loss(self, norm_paths):
        """FIT preds are the RAW [B, 2R] head output; loss == v1 gaussian NLL."""
        targets = ("HadronConeExclTruthLabelPt",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            gaussian=True,
            norm_params={"mean": 1.0, "std": 1.0},
            weight=0.5,
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        out = Executor(fit).run(b, debug=True)
        v2_pred = out.get("preds.jets.regression")
        v2_loss = out.get("losses.regression")
        assert v2_pred.shape == (B, 2)  # means ‖ raw variances
        pooled = out.get("pooled.global")
        tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
        ref_pred, ref_loss = task.task(pooled, tdict, None, context=None)
        assert torch.allclose(v2_pred, ref_pred, atol=1e-6)
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_gaussian_test_descale_one_array_means_then_stddev(self, norm_paths):
        """TEST publishes ONE [B, 2R] array (means ‖ stddevs); parity vs v1 tuple."""
        targets = ("HadronConeExclTruthLabelPt",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            gaussian=True,
            norm_params={"mean": 2.0, "std": 3.0},
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        test = compile_regression(modules, Mode.TEST, targets)
        bind_all(modules, resolve_bind_schema([fit, test]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(test).run(b)
        v2_test = b.get("preds.jets.regression")
        assert v2_test.shape == (B, 2)
        pooled = b.get("pooled.global")
        with torch.no_grad():
            raw, _ = task.task(pooled, {}, None, context=None)
            ref_means, ref_stds = task.task.run_inference(raw.clone())
        # the published array is the v1 (means, stds) re-concatenated
        assert torch.allclose(v2_test, torch.cat([ref_means, ref_stds], dim=-1), atol=1e-6)
        # the second column IS sqrt(softplus(var)) * std (v1 task.py:762-764)
        assert (v2_test[:, 1] > 0).all()

    # -- sample_weight + NaN masking (inside composed v1 nan_loss) — A3 ---------

    def test_sample_weight_requires_reduction_none(self):
        with pytest.raises(ConfigError, match="sample_weight.*reduction"):
            RegressionTaskModule(
                stream="jets", targets="pt", input="pooled.global", sample_weight="w"
            )
        # explicit reduction: none is accepted
        RegressionTaskModule(
            stream="jets",
            targets="pt",
            input="pooled.global",
            sample_weight="w",
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )

    def test_sample_weight_declares_the_weight_label(self):
        task = RegressionTaskModule(
            stream="jets",
            targets=["a", "b"],
            input="pooled.global",
            sample_weight="w",
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )
        task.name = "regression"
        req = flatten_spec(task.declare_io(Mode.FIT).requires)
        assert "labels.jets.w" in req and req["labels.jets.w"].modes == Mode.TRAINING
        # the weight label leaf is TRAINING-gated (the planner drops it in
        # TEST/ONNX, same as the target labels — modes attribute, not absence)
        assert flatten_spec(task.declare_io(Mode.TEST).requires)["labels.jets.w"].modes == (
            Mode.TRAINING
        )

    @pytest.mark.parametrize("weights", [[1.0, 0.5, 2.0, 0.0, 1.5, 0.25], "zero"])
    def test_sample_weight_loss_parity_nonuniform_and_zero(self, norm_paths, weights):
        """Per-sample-weighted loss parity (nonuniform + all-zero) vs the v1 head.

        Exercises the fragile interactions the codex plan review flagged:
        zero/nonuniform weights, the unsqueeze+expand over two targets, and
        the mask→0 + nanmean ordering — all owned by the composed v1 nan_loss.
        """
        targets = ("R10TruthLabel_R22v1_TruthJetMass", "R10TruthLabel_R22v1_TruthJetPt")
        w = torch.zeros(B) if weights == "zero" else torch.tensor(weights)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            sample_weight="w",
            norm_params={"mean": [1.0, 2.0], "std": [3.0, 4.0]},
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets, weight="w")
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        b.set("labels.jets.w", w)
        out = Executor(fit).run(b, debug=True)
        v2_loss = out.get("losses.regression")
        # v1 reference: the same head, weight in the targets dict (task.py:430)
        pooled = out.get("pooled.global")
        tdict = {"jets": {t: labels[f"labels.jets.{t}"] for t in targets}}
        tdict["jets"]["w"] = w
        _, ref_loss = task.task(pooled, tdict, None, context=None)
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_nan_target_masking_loss_parity(self, norm_paths):
        """NaN targets are masked (→0) and reduced with nanmean; parity vs v1.

        Requires reduction='none' so the per-element loss survives to nanmean
        (v1 task.py:412-437, the nan_regression contract).
        """
        targets = ("HadronConeExclTruthLabelPt",)
        task = RegressionTaskModule(
            stream="jets",
            targets=list(targets),
            input="pooled.global",
            norm_params={"mean": 1.0, "std": 1.0},
            loss={"class_path": "torch.nn.MSELoss", "init_args": {"reduction": "none"}},
        )
        modules = build_regression_modules(norm_paths[0], task)
        fit = compile_regression(modules, Mode.FIT, targets)
        bind_all(modules, resolve_bind_schema([fit]))
        materialise_all(modules)
        inputs, masks = make_gn2_batch(B, T)
        labels = make_regression_labels(B, targets)
        # poison half the targets with NaN
        poisoned = labels[f"labels.jets.{targets[0]}"].clone()
        poisoned[::2] = torch.nan
        labels[f"labels.jets.{targets[0]}"] = poisoned
        b = Bundle()
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        for key, val in labels.items():
            b.set(key, val)
        out = Executor(fit).run(b, debug=True)
        v2_loss = out.get("losses.regression")
        assert torch.isfinite(v2_loss)  # NaN targets did not poison the loss
        pooled = out.get("pooled.global")
        tdict = {"jets": {targets[0]: poisoned}}
        _, ref_loss = task.task(pooled, tdict, None, context=None)
        assert torch.allclose(v2_loss, ref_loss, atol=1e-6)

    def test_declare_and_bind_are_file_free(self, monkeypatch):
        """The §2.3 no-I/O guard holds for the regression module too."""

        def _forbid(*args, **kwargs):
            raise AssertionError("file I/O during declare_io/bind (design §2.3)")

        monkeypatch.setattr("builtins.open", _forbid)
        task = RegressionTaskModule(
            stream="jets", targets=["a"], input="pooled.global", norm_params={"mean": 1, "std": 1}
        )
        task.name = "regression"
        for mode in (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX):
            task.declare_io(mode)
        _bind_reg_module(task, {"pooled.global": 16})


# ---------------------------------------------------------------------------
# full plan execution (FIT + TEST), debug mode on
# ---------------------------------------------------------------------------


class TestGn2V2Execution:
    def test_fit_plan_runs_in_debug_mode(self, gn2v2):
        """Debug execution: read-tracking, write-once, and mutation checks all pass."""
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle(), debug=True)
        total = b.get("loss.total")
        assert total.shape == ()
        assert torch.isfinite(total)
        expected = (
            b.get("losses.jets_classification")
            + b.get("losses.track_origin")
            + b.get("losses.track_vertexing")
        )
        assert torch.equal(total, expected)

    def test_fit_preds_are_raw_logits(self, gn2v2):
        modules, plan, _ = gn2v2
        b = Executor(plan).run(fit_bundle())
        logits = b.get("preds.jets.jets_classification")
        assert logits.shape == (B, 3)
        assert not torch.allclose(logits.sum(-1), torch.ones(B))  # not softmaxed

    def test_test_plan_converts_predictions(self, gn2v2):
        """TEST preds are converted physical values (design §3.3)."""
        modules, _, _ = gn2v2
        plan = compile_gn2v2(modules, Mode.TEST)
        assert "loss" not in plan.module_names  # LossSum inactive outside TRAINING
        b = Bundle()
        inputs, masks = make_gn2_batch(B, T)
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(plan).run(b, debug=True)
        probs = b.get("preds.jets.jets_classification")
        assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-6)
        track_probs = b.get("preds.tracks.track_origin")
        valid = ~masks["tracks"]
        assert torch.allclose(track_probs[valid].sum(-1), torch.ones(int(valid.sum())), atol=1e-6)
        # vertexing TEST output: per-node assignments unflattened to the
        # batch shape, padded positions -inf (task.py:985-986,1003-1035)
        assert b.get("preds.tracks.track_vertexing").shape == (B, T, 1)

    def test_onnx_plan_keeps_raw_vertexing_scores(self, gn2v2):
        """ONNX vertexing publishes raw edge scores for the export reduce (§3.3)."""
        modules, _, _ = gn2v2
        plan = compile_gn2v2(modules, Mode.ONNX)
        b = Bundle()
        inputs, masks = make_gn2_batch(B, T)
        for stream, x in inputs.items():
            b.set(f"inputs.{stream}", x)
        b.set("masks.tracks", masks["tracks"])
        with torch.no_grad():
            b = Executor(plan).run(b)
        scores = b.get("preds.tracks.track_vertexing")
        assert scores.ndim == 2
        assert scores.shape[1] == 1  # [E, 1] raw edge scores

    def test_all_modules_are_nn_modules(self, gn2v2):
        """The module dict must be nn.ModuleDict-compatible (SaltModule, design §3.4)."""
        modules, _, _ = gn2v2
        assert all(isinstance(m, nn.Module) for m in modules.values())
        nn.ModuleDict(modules)  # must not raise


def _origin_schema_reader(origin_label: str = "ftagTruthOriginLabel") -> SimpleNamespace:
    """A duck-typed reader exposing the tracks origin class names (design §2.6).

    Mirrors the umami-preprocessing convention the §2.6 class-names check
    consults: the stream's group attr named after the origin label holds the
    index-aligned class-name list (``ORIGIN_CLASSES``: Fake=1, FromB=3,
    FromBC=4, FromC=5 — the v1 default ids).
    """
    schema = Schema(groups={"tracks": GroupSchema(fields={}, attrs={origin_label: ORIGIN_CLASSES})})
    return SimpleNamespace(schema_group=schema.groups.get)


class TestOriginWeightingConfig:
    """Name-based origin_weighting (design §5.1, M5 sub-wave D): names resolve to ids."""

    def _name_based(self) -> VertexingTaskModule:
        task = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            origin_weighting={"heavy": ["FromB", "FromBC", "FromC"], "fake": ["Fake"]},
        )
        task.name = "track_vertexing"
        return task

    def test_names_pending_until_resolved(self):
        # __init__ captures names but does NOT resolve them (no reader yet)
        task = self._name_based()
        assert task._names_pending is True  # noqa: SLF001
        assert task.heavy_ids is None
        assert task.fake_ids is None

    def test_names_resolve_to_v1_default_ids(self):
        # the GN3 origin names map to exactly v1's hardcoded heavy 3,4,5 / fake 1
        task = self._name_based()
        resolved = task.resolve_origin_names(_origin_schema_reader())
        assert resolved is True
        assert task.heavy_ids == (3, 4, 5)
        assert task.fake_ids == (1,)
        assert task._names_pending is False  # noqa: SLF001

    def test_name_resolved_weights_match_independent_v1(self):
        # parity: name-resolved ids produce weights bit-identical to a FRESH,
        # independently-constructed v1 VertexingTask whose get_weights HARDCODES
        # (3,4,5)/1 — never the v2 module's own head (gate-quality rule)
        from salt.models.task import VertexingTask as V1VertexingTask

        task = self._name_based()
        task.resolve_origin_names(_origin_schema_reader())
        task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))
        # independent v1 reference: a fresh head built from the same dense kwargs
        indep_v1 = V1VertexingTask(
            name="track_vertexing",
            input_name="tracks",
            label="ftagTruthVertexIndex",
            loss=nn.BCEWithLogitsLoss(reduction="none"),
            dense_config={"input_size": 32, "output_size": 1},
        )
        labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        n = labels.shape[1]
        adjmat = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
        v1_weights = V1VertexingTask.get_weights(indep_v1, labels, adjmat)
        v2_weights = task.task.get_weights(labels, adjmat)
        assert torch.equal(v1_weights, v2_weights)

    def test_resolve_origin_weighting_module_helper(self):
        # the saltmodule helper resolves over a module dict, counting resolutions
        from salt.core.saltmodule import resolve_origin_weighting

        task = self._name_based()
        # an int-id sibling must NOT count (already resolved)
        intd = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="o",
            origin_weighting={"heavy": [3], "fake": [1]},
        )
        intd.name = "int_vtx"
        n = resolve_origin_weighting(
            {"track_vertexing": task, "int_vtx": intd}, _origin_schema_reader()
        )
        assert n == 1
        assert task.heavy_ids == (3, 4, 5)

    def test_unknown_name_raises_quality_error(self):
        task = VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            origin_weighting={"heavy": ["NotAClass"], "fake": ["Fake"]},
        )
        task.name = "vtx"
        with pytest.raises(ConfigError, match="NotAClass") as excinfo:
            task.resolve_origin_names(_origin_schema_reader())
        assert "ftagTruthOriginLabel" in str(excinfo.value)

    def test_no_schema_attr_raises(self):
        # a name-based config but the schema has no origin class-name attr
        task = self._name_based()
        empty = SimpleNamespace(schema_group=lambda s: GroupSchema(fields={}, attrs={}))
        with pytest.raises(ConfigError, match="no string-list attr"):
            task.resolve_origin_names(empty)

    def test_name_based_bind_without_resolution_fails_loudly(self):
        # binding a name-based task that never reached a schema is a loud error,
        # NOT a silent mis-weighting
        task = self._name_based()
        with pytest.raises(ConfigError, match="not resolved") as excinfo:
            task.bind(ResolvedSchema(widths={"encoded.tracks": 16}))
        assert "schema artifact" in str(excinfo.value)

    def test_no_schema_reader_resolves_nothing(self):
        # a reader without schema support leaves names pending (the bind error
        # is the loud surface, not this no-op)
        task = self._name_based()
        assert task.resolve_origin_names(SimpleNamespace()) is False
        assert task._names_pending is True  # noqa: SLF001

    def test_integer_ids_are_noop_for_resolution(self):
        task = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="o",
            origin_weighting={"heavy": [3, 4, 5], "fake": [1]},
        )
        task.name = "vtx"
        assert task.resolve_origin_names(_origin_schema_reader()) is False
        assert task.heavy_ids == (3, 4, 5)

    def test_mixed_ids_and_names_rejected(self):
        with pytest.raises(ConfigError, match="mixes integer ids with class names"):
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="o",
                origin_weighting={"heavy": [3, "FromBC", 5], "fake": [1]},
            )

    def test_non_integer_id_rejected(self):
        with pytest.raises(ConfigError, match="INTEGER origin ids or class NAMES"):
            VertexingTaskModule(
                stream="tracks",
                label="ftagTruthVertexIndex",
                origin_label="o",
                origin_weighting={"heavy": [3.5], "fake": [1]},
            )


class TestExposeOptOut:
    """Per-task ``expose: [fit, val]`` opt-out (design §4.2, M5 sub-wave D).

    A train-only aux task gates its ``preds.*`` port to the listed modes so the
    planner prunes it from the TEST/ONNX plans (silencing the dead-preds hard
    error) while it keeps training. The default (no ``expose``) publishes in
    every mode as before.
    """

    def test_parse_default_is_all_modes(self):
        task = ClassificationTaskModule(
            stream="jets", label="f", class_names=["a", "b"], input="pooled.global"
        )
        assert task.expose_modes == Mode.ALL

    def test_parse_fit_val_gates_pred_port(self):
        task = ClassificationTaskModule(
            stream="tracks", label="o", class_names=["a", "b"],
            context="pooled.global", expose=["fit", "val"],
        )
        task.name = "aux"
        assert task.expose_modes == (Mode.FIT | Mode.VAL)
        produced_fit = flatten_spec(task.declare_io(Mode.FIT).produces)
        # the pred port is active in FIT/VAL but NOT in TEST/ONNX
        pred = produced_fit[task.pred_key]
        assert pred.active_in(Mode.FIT) and pred.active_in(Mode.VAL)
        assert not pred.active_in(Mode.TEST)
        assert not pred.active_in(Mode.ONNX)
        # the loss is FIT|VAL regardless (the task still trains)
        assert produced_fit[task.loss_key].active_in(Mode.FIT)

    def test_parse_case_insensitive(self):
        task = ClassificationTaskModule(
            stream="jets", label="f", class_names=["a", "b"], input="pooled.global",
            expose=["FIT", "Val"],
        )
        assert task.expose_modes == (Mode.FIT | Mode.VAL)

    def test_empty_list_rejected(self):
        with pytest.raises(ConfigError, match="empty list"):
            ClassificationTaskModule(
                stream="jets", label="f", class_names=["a"], input="pooled.global", expose=[]
            )

    def test_string_value_rejected(self):
        with pytest.raises(ConfigError, match="list of mode names"):
            ClassificationTaskModule(
                stream="jets", label="f", class_names=["a"], input="pooled.global", expose="fit"
            )

    def test_unknown_mode_rejected(self):
        with pytest.raises(ConfigError, match="unknown expose mode"):
            ClassificationTaskModule(
                stream="jets", label="f", class_names=["a"], input="pooled.global",
                expose=["fit", "predict"],
            )

    def test_expose_on_regression_and_vertexing(self):
        # the opt-out lives on the shared base, so all task families carry it:
        # the pred port is declared but mode-inactive in TEST (planner-pruned)
        reg = RegressionTaskModule(stream="jets", targets="x", input="pooled.global",
                                   expose=["fit", "val"])
        reg.name = "reg"
        reg_pred = flatten_spec(reg.declare_io(Mode.TEST).produces)[reg.pred_key]
        assert reg_pred.active_in(Mode.FIT) and not reg_pred.active_in(Mode.TEST)
        vtx = VertexingTaskModule(
            stream="tracks", label="ftagTruthVertexIndex", origin_label="o",
            expose=["fit", "val"],
        )
        vtx.name = "vtx"
        vtx_pred = flatten_spec(vtx.declare_io(Mode.TEST).produces)[vtx.pred_key]
        assert vtx_pred.active_in(Mode.FIT) and not vtx_pred.active_in(Mode.TEST)

    def _modules_with_exposed_aux(self, norm_dict):
        modules = build_gn2v2_modules(norm_dict)
        # rebuild track_origin with expose: [fit, val] (a train-only aux task)
        modules["track_origin"] = ClassificationTaskModule(
            stream="tracks", label="ftagTruthOriginLabel",
            class_names=list(ORIGIN_CLASSES), context="pooled.global", weight=0.5,
            dense={"hidden_layers": [16], "activation": "ReLU"}, expose=["fit", "val"],
        )
        modules["track_origin"].name = "track_origin"
        # fresh LossSum re-narrowed over the rebuilt module set (the prior one
        # has already fixed its keys and cannot re-narrow)
        loss = LossSum()
        loss.name = "loss"
        loss.narrow(LossSum.collect_loss_keys(modules))
        modules["loss"] = loss
        return modules

    def test_exposed_task_kept_in_fit_plan(self, norm_paths):
        modules = self._modules_with_exposed_aux(norm_paths[0])
        plan = compile_plan(modules, Mode.FIT, sources=gn2v2_sources(), sinks=["loss.total"])
        assert "track_origin" in plan.module_names

    def test_exposed_task_pruned_from_test_plan(self, norm_paths):
        modules = self._modules_with_exposed_aux(norm_paths[0])
        # the exposed aux produces nothing in TEST → the planner prunes it from
        # the TEST plan entirely (compile with its full preds set as sinks: the
        # exposed pred is mode-inactive, so it cannot be demanded)
        plan = compile_plan(
            modules,
            Mode.TEST,
            sources=gn2v2_sources(),
            sinks=["preds.jets.jets_classification", "preds.tracks.track_vertexing"],
        )
        assert "track_origin" not in plan.module_names
        assert "jets_classification" in plan.module_names

    def test_default_aux_alive_in_test_when_demanded(self, norm_paths):
        # negative control: WITHOUT expose, the same task IS active in TEST — its
        # pred is a legitimate TEST sink the planner keeps (the column expose
        # would have removed). Proves the exposed/non-exposed difference is the
        # pred port's TEST activity, not some other pruning.
        modules = build_gn2v2_modules(norm_paths[0])
        plan = compile_plan(
            modules,
            Mode.TEST,
            sources=gn2v2_sources(),
            sinks=[
                "preds.jets.jets_classification",
                "preds.tracks.track_origin",
                "preds.tracks.track_vertexing",
            ],
        )
        assert "track_origin" in plan.module_names

    def test_exposed_pred_cannot_be_a_test_sink(self, norm_paths):
        # the exposed pred is mode-inactive in TEST, so demanding it as a TEST
        # sink is a no-producer error naming the FIT/VAL modes where it lives
        from salt.core.graph.errors import GraphError

        modules = self._modules_with_exposed_aux(norm_paths[0])
        with pytest.raises(GraphError, match="FIT/VAL"):
            compile_plan(
                modules,
                Mode.TEST,
                sources=gn2v2_sources(),
                sinks=["preds.tracks.track_origin"],
            )


class TestNoIOGuard:
    """Design §2.3 CI guard: declare_io and bind run under a no-I/O trap.

    The rule itself ("declare_io/bind touch no files") was only ever verified
    manually; this pins it in CI over the shipped GN2v2 module set (stage-E
    design-compliance fix). materialise() stays the ONLY file-touching hook —
    proven by releasing the trap and pointing it at a nonexistent norm dict.
    """

    def test_declare_and_bind_are_file_free(self, monkeypatch):
        # construction is config capture only — safe to build under the trap
        def _forbid(*args, **kwargs):
            raise AssertionError(f"file I/O during declare_io/bind (design §2.3): open({args!r})")

        monkeypatch.setattr("builtins.open", _forbid)
        monkeypatch.setattr("h5py.File", _forbid)
        monkeypatch.setattr("pathlib.Path.open", _forbid)
        modules = build_gn2v2_modules("/nonexistent/norm_dict.yaml")
        for mode in (Mode.FIT, Mode.VAL, Mode.TEST, Mode.ONNX):
            for module in modules.values():
                module.declare_io(mode)
        plans = [compile_gn2v2(modules, mode) for mode in (Mode.FIT, Mode.TEST)]
        bind_all(modules, resolve_bind_schema(plans))

        # release the trap: materialise IS the sanctioned file hook and must
        # be the first thing that touches the (nonexistent) norm dict
        monkeypatch.undo()
        with pytest.raises(FileNotFoundError):
            materialise_all(modules)
