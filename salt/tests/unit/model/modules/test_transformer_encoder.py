"""Unit tests for TransformerEncoder (mirror of salt/model/modules/transformer_encoder.py)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.executor import Executor
from salt.graph.planner import PlanStep
from salt.graph.spec import Mode, flatten_spec
from salt.model.modules import (
    BindError,
    TransformerEncoder,
)
from salt.model.nn.attention import Attention, EdgeAttention
from salt.model.nn.transformer import DecoderLayer, DropPath, NormResidual, Transformer
from salt.tests.unit.model.conftest import B, T, fit_bundle


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
    """The ``mup:`` flag on `TransformerEncoder`."""

    def test_default_is_not_mup(self):
        enc = TransformerEncoder(dim=16, num_layers=1, out_dim=8, attention={"num_heads": 2})
        assert enc.mup is False
        assert type(enc.encoder.out_proj) is nn.Linear

    def test_mup_requires_out_dim(self):
        with pytest.raises(ConfigError, match="mup requires an out_dim"):
            TransformerEncoder(dim=16, num_layers=1, attention={"num_heads": 2}, mup=True)

    def test_mup_swaps_out_proj_for_zeroed_mu_readout(self):
        pytest.importorskip("mup", reason="muP tests need `pip install 'salt[muP]'`")
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
        pytest.importorskip("mup", reason="muP tests need `pip install 'salt[muP]'`")
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

    def test_set_export_mode_folds_mu_readout_to_plain_linear(self):
        """set_export_mode swaps the MuReadout for a plain Linear, forward unchanged."""
        pytest.importorskip("mup", reason="muP tests need `pip install 'salt[muP]'`")
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


class TestDerivedWidthsFixpoint:
    """`_apply_derived_widths` resolves a CHAIN of derived widths order-independently."""

    @staticmethod
    def _chained_plan(order):
        """A duck-typed plan whose steps hold two chained derived-width modules."""

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
        from salt.model.bind import _apply_derived_widths

        widths = {"src": 10}
        _apply_derived_widths([self._chained_plan(["b", "a"])], widths)
        assert widths == {"src": 10, "mid": 11, "out": 12}

    def test_chain_resolves_for_either_order(self):
        from salt.model.bind import _apply_derived_widths

        for order in (["a", "b"], ["b", "a"]):
            widths = {"src": 10}
            _apply_derived_widths([self._chained_plan(order)], widths)
            assert widths == {"src": 10, "mid": 11, "out": 12}, order

    def test_conflicting_derived_width_still_raises(self):
        from salt.model.bind import _apply_derived_widths

        class _Mod:
            name = "clash"

            def derived_widths(self, widths):
                return {"out": 99}

        widths = {"out": 7}
        with pytest.raises(BindError, match="conflicting widths"):
            _apply_derived_widths(
                [SimpleNamespace(steps=(PlanStep(name="clash", module=_Mod()),))], widths
            )


class TestSetBackend:
    def test_edge_attention_set_backend_warns_and_returns(self):
        attn = EdgeAttention(embed_dim=16, edge_embed_dim=8, num_heads=2)
        with pytest.warns(UserWarning, match="EdgeAttention does not support different backends"):
            out = attn.set_backend("torch-flash")
        assert out == "torch-flash"

    def test_edge_transformer_set_backend_warns_not_raises(self):
        tf = Transformer(num_layers=2, embed_dim=16, edge_embed_dim=8, attn_kwargs={"num_heads": 2})
        assert all(isinstance(layer.attn.fn, EdgeAttention) for layer in tf.layers)
        with pytest.warns(UserWarning, match="EdgeAttention does not support different backends"):
            tf.set_backend("torch-math")
        assert tf.attn_type == "torch-math"

    def test_plain_transformer_set_backend_no_warning(self):
        tf = Transformer(num_layers=2, embed_dim=16, edge_embed_dim=0, attn_kwargs={"num_heads": 2})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tf.set_backend("torch-math")
        assert tf.attn_type == "torch-math"
        assert all(type(layer.attn.fn) is Attention for layer in tf.layers)


def _edge_inputs(bs, seq_len, dim, edge_dim, frac_pad=0.0):
    """Random tokens, pad mask (True = padded, first token always valid) and edges."""
    torch.manual_seed(0)
    x = torch.randn(bs, seq_len, dim)
    mask = torch.rand(bs, seq_len) < frac_pad
    mask[:, 0] = False
    return x, mask, torch.randn(bs, seq_len, seq_len, edge_dim)


class TestEdgeAttentionOptions:
    """``talking_heads`` and ``edge_gate`` on `EdgeAttention`."""

    @pytest.mark.parametrize("talking_heads", [True, False])
    @pytest.mark.parametrize("edge_gate", [True, False])
    @pytest.mark.parametrize("mup", [True, False])
    def test_talking_heads_gate(self, talking_heads, edge_gate, mup):
        attn = EdgeAttention(
            embed_dim=16,
            edge_embed_dim=8,
            num_heads=4,
            talking_heads=talking_heads,
            edge_gate=edge_gate,
            mup=mup,
        )
        # the gate/talking-heads submodules only exist when requested
        assert (attn.linear_g is not None) == edge_gate
        assert hasattr(attn, "talking_1") == talking_heads
        assert hasattr(attn, "talking_2") == talking_heads

        x, mask, edge_x = _edge_inputs(2, 10, 16, 8, 0.5)
        out, edge_out = attn(x, edge_x=edge_x, mask=mask)
        assert out.shape == (2, 10, 16)
        assert edge_out.shape == (2, 10, 10, 8)
        assert not out.isnan().any()
        assert not edge_out.isnan().any()

    def test_talking_heads_changes_output(self):
        x, mask, edge_x = _edge_inputs(2, 8, 16, 8, 0.5)
        torch.manual_seed(0)
        plain = EdgeAttention(16, 8, num_heads=4, edge_gate=False)
        torch.manual_seed(0)
        talk = EdgeAttention(16, 8, num_heads=4, edge_gate=False, talking_heads=True)
        # share the projections so only the talking-heads mixing differs
        talk.load_state_dict(plain.state_dict(), strict=False)
        out_plain, _ = plain(x, edge_x=edge_x, mask=mask)
        out_talk, _ = talk(x, edge_x=edge_x, mask=mask)
        assert not torch.allclose(out_plain, out_talk)

    def test_edge_gate_off_is_additive_bias_only(self):
        x, mask, edge_x = _edge_inputs(2, 8, 16, 8, 0.5)
        gated = EdgeAttention(16, 8, num_heads=4)
        plain = EdgeAttention(16, 8, num_heads=4, edge_gate=False)
        plain.load_state_dict({k: v for k, v in gated.state_dict().items() if "linear_g" not in k})
        # a gate of sigmoid(+inf) = 1 everywhere reproduces the ungated attention
        with torch.no_grad():
            gated.linear_g.weight.zero_()
            gated.linear_g.bias.fill_(float("inf"))
        torch.testing.assert_close(gated(x, edge_x, mask=mask)[0], plain(x, edge_x, mask=mask)[0])


class TestLayerScaleDropPath:
    """``ls_init``, ``drop_path`` and ``drop_path_schedule`` on the transformer stack."""

    @pytest.mark.parametrize("ls_init", [None, 0.1])
    @pytest.mark.parametrize("drop_path", [0.0, 0.9])
    @pytest.mark.parametrize("norm_type", ["pre", "post", "none"])
    def test_norm_residual(self, ls_init, drop_path, norm_type):
        x = torch.randn(10, 20, 16)
        block = nn.Linear(16, 16)
        nr = NormResidual(
            block, ls_init=ls_init, drop_path=drop_path, embed_dim=16, norm_type=norm_type
        )
        y = nr(x)
        assert y.shape == x.shape
        assert not y.isnan().any()

    def test_edge_norm_residual_layer_scale(self):
        attn = EdgeAttention(embed_dim=32, edge_embed_dim=33, num_heads=2)
        nr = NormResidual(attn, ls_init=0.1, embed_dim=32)
        x, mask, edge_x = _edge_inputs(2, 5, 32, 33, 0.5)
        out, edge_out = nr(x, edge_x=edge_x, mask=mask)
        assert out.shape == (2, 5, 32)
        assert not out.isnan().any()
        assert torch.equal(edge_x, edge_out)

    def test_layer_scale_scales_the_branch(self):
        torch.manual_seed(0)
        nr = NormResidual(nn.Linear(16, 16), ls_init=1e-3, embed_dim=16)
        x = torch.randn(4, 16)
        torch.testing.assert_close(nr(x) - x, 1e-3 * nr.fn(nr.norm(x)))

    def test_drop_path_is_identity_in_eval(self):
        drop = DropPath(0.9)
        x = torch.randn(64, 5, 16)
        assert torch.equal(drop.eval()(x), x)
        # in training, whole samples are either dropped or rescaled by 1 / keep_prob
        y = drop.train()(x)
        per_sample = (y / x).flatten(1)
        assert set(per_sample[:, 0].round(decimals=4).tolist()) <= {0.0, 10.0}
        assert torch.allclose(per_sample, per_sample[:, :1])

    @pytest.mark.parametrize("schedule", ["uniform", "linear"])
    def test_drop_path_schedule(self, schedule):
        num_layers, drop_path = 4, 0.2
        trans = Transformer(
            num_layers=num_layers,
            embed_dim=32,
            dense_kwargs={"activation": "SiLU"},
            attn_kwargs={"num_heads": 2},
            drop_path=drop_path,
            drop_path_schedule=schedule,
        )
        rates = [layer.attn.drop_path.drop_prob for layer in trans.layers]
        if schedule == "uniform":
            assert all(r == drop_path for r in rates)
        else:
            expected = [drop_path * 0.5 * (1 + d / (num_layers - 1)) for d in range(num_layers)]
            assert rates == pytest.approx(expected)
        x, mask, _ = _edge_inputs(5, 10, 32, 1, 0.5)
        out, _ = trans(x, pad_mask=mask)
        assert not out.isnan().any()

    def test_unknown_drop_path_schedule_rejected(self):
        with pytest.raises(AssertionError, match="drop_path_schedule"):
            Transformer(
                num_layers=2,
                embed_dim=16,
                attn_kwargs={"num_heads": 2},
                drop_path_schedule="cosine",
            )

    def test_depart_style_transformer(self):
        x = torch.rand(2, 6, 16)
        edge_x = torch.rand(2, 6, 6, 16)
        # additive-bias attention (no gate) + talking heads + gated FFN + LayerScale
        # + linear drop-path, composed through the Transformer
        net = Transformer(
            embed_dim=16,
            edge_embed_dim=16,
            num_layers=3,
            attn_kwargs={"num_heads": 2, "talking_heads": True, "edge_gate": False},
            dense_kwargs={"activation": "ReLU", "gated": True},
            ls_init=1e-5,
            drop_path=0.1,
            drop_path_schedule="linear",
            num_registers=2,
        )
        mask = torch.rand(2, 6) < 0.5
        out = net(x, edge_x=edge_x, pad_mask=mask)[0]
        assert out.shape == (2, 8, 16)
        assert not out.isnan().any()
        # fully padded jets must still be NaN-free (registers guard the softmax)
        out, _ = net(x, edge_x=edge_x, pad_mask=torch.ones(2, 6, dtype=torch.bool))
        assert not out.isnan().any()

    def test_options_reach_layers_from_encoder_config(self):
        enc = TransformerEncoder(
            dim=16,
            num_layers=3,
            attention={"num_heads": 2, "talking_heads": True, "edge_gate": False},
            edges="edges.tracks_emb",
            edge_embed_dim=8,
            ls_init=1e-5,
            drop_path=0.2,
            drop_path_schedule="linear",
        )
        layers = enc.encoder.layers
        rates = [layer.attn.drop_path.drop_prob for layer in layers]
        assert rates == pytest.approx([0.1, 0.15, 0.2])
        for layer in layers:
            for res in (layer.attn, layer.dense):
                torch.testing.assert_close(res.ls.gamma, torch.full((16,), 1e-5))
            assert layer.attn.fn.talking_heads
            assert layer.attn.fn.linear_g is None


class TestDefaultsUnchanged:
    """With every DeParT option off, the encoder's parameters are exactly as before."""

    _LAYER_KEYS = [
        "attn.fn.in_proj_bias",
        "attn.fn.in_proj_weight",
        "attn.fn.out_proj.bias",
        "attn.fn.out_proj.weight",
        "dense.fn.in_proj.bias",
        "dense.fn.in_proj.weight",
        "dense.fn.out_proj.bias",
        "dense.fn.out_proj.weight",
    ]
    _EDGE_KEYS = [
        "attn.fn.linear_e.bias",
        "attn.fn.linear_e.weight",
        "attn.fn.linear_g.bias",
        "attn.fn.linear_g.weight",
    ]

    @staticmethod
    def _layer_keys(encoder):
        keys = {k.split(".", 3)[3] for k in encoder.state_dict() if k.startswith("encoder.layers.")}
        return sorted(keys)

    def test_gn2v2_encoder_state_dict_keys(self, gn2v2):
        modules, _, _ = gn2v2
        enc = modules["encoder"]
        assert self._layer_keys(enc) == self._LAYER_KEYS
        assert sorted(k for k in enc.state_dict() if not k.startswith("encoder.layers.")) == [
            "encoder.out_proj.bias",
            "encoder.out_proj.weight",
            "encoder.register_mask",
            "encoder.registers",
        ]

    def test_edge_encoder_state_dict_keys(self):
        enc = TransformerEncoder(
            dim=16,
            num_layers=2,
            attention={"num_heads": 2},
            edges="edges.tracks_emb",
            edge_embed_dim=8,
        )
        assert self._layer_keys(enc) == sorted(self._LAYER_KEYS + self._EDGE_KEYS)


class TestDecoderLayer:
    @pytest.mark.parametrize("self_attn", [True, False])
    def test_self_attn_option(self, self_attn):
        layer = DecoderLayer(embed_dim=16, attn_kwargs={"num_heads": 2}, self_attn=self_attn)
        assert hasattr(layer, "self_attn") == self_attn
        x, kv = torch.randn(B, 3, 16), torch.randn(B, T, 16)
        kv_mask = torch.zeros(B, T, dtype=torch.bool)
        assert layer(x, kv=kv, kv_mask=kv_mask).shape == (B, 3, 16)

    def test_self_attn_ignores_padded_queries(self):
        torch.manual_seed(0)
        layer = DecoderLayer(embed_dim=16, attn_kwargs={"num_heads": 2}).eval()
        x, kv = torch.randn(B, 4, 16), torch.randn(B, T, 16)
        mask = torch.tensor([[False, False, True, True]]).expand(B, -1)  # True = padded
        kv_mask = torch.zeros(B, T, dtype=torch.bool)
        x2 = x.clone()
        x2[mask] = torch.randn_like(x2[mask])
        out = layer(x, kv=kv, mask=mask, kv_mask=kv_mask)
        out2 = layer(x2, kv=kv, mask=mask, kv_mask=kv_mask)
        torch.testing.assert_close(out[~mask], out2[~mask])
