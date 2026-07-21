"""Unit tests for TransformerEncoder (mirror of salt/model/modules/transformer_encoder.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from salt.graph import (
    Bundle,
    ConfigError,
    Executor,
    Mode,
    PlanStep,
    flatten_spec,
)
from salt.model.modules import (
    BindError,
    TransformerEncoder,
)
from salt.tests.unit.nn.conftest import B, T, fit_bundle


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
    """The ``mup:`` flag on `TransformerEncoder` (M6 sub-wave B, plan 12 muP arch port)."""

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

    # DEL-1: test_mup_forward_bitwise_vs_independent_v1 retired with the v1 tree
    # (parity-closure doctrine: v1 comparisons = git checkout 29c67a1).

    def test_set_export_mode_folds_mu_readout_to_plain_linear(self):
        """set_export_mode swaps the MuReadout for a plain Linear, forward unchanged."""
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
