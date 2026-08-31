"""Unit tests for MaskDecoder (mirror of salt/model/modules/maskdecoder.py)."""

from __future__ import annotations

import pytest
import torch

from salt.graph import (
    Bundle,
    ConfigError,
    Executor,
    Mode,
    flatten_spec,
)
from salt.model.modules import (
    MaskDecoder,
    bind_all,
    materialise_all,
    resolve_bind_schema,
)
from salt.model.nn.maskformer_loss import (
    dice_loss,
    dice_loss_eager,
    mask_ce_loss,
    mask_ce_loss_eager,
    sigmoid_focal_loss,
    sigmoid_focal_loss_eager,
)
from salt.model.nn.matcher import (
    batch_dice_cost,
    batch_dice_cost_eager,
    batch_mae_loss,
    batch_mae_loss_eager,
    batch_sigmoid_ce_cost,
    batch_sigmoid_ce_cost_eager,
    batch_sigmoid_focal_cost,
    batch_sigmoid_focal_cost_eager,
)
from salt.tests._fixtures.gn2v2_fixture import (
    make_gn2_batch,
)
from salt.tests._fixtures.v2_builders import (
    MASKFORMER_NUM_OBJECTS,
    build_maskformer_decoder_modules,
    compile_maskformer_decoder,
)
from salt.tests.unit.model.conftest import B, T


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
            MaskDecoder(embed_dim=16, num_queries=5, num_layers=2, class_net={"output_size": 3})

    def test_missing_class_output_size_rejected(self):
        with pytest.raises(ConfigError, match=r"class_net\.output_size is required"):
            MaskDecoder(embed_dim=16, num_queries=5, num_layers=2, class_net={}, md={"n_heads": 2})

    def test_class_net_width_key_rejected(self):
        with pytest.raises(ConfigError, match=r"class_net\.input_size"):
            MaskDecoder(
                embed_dim=16,
                num_queries=5,
                num_layers=2,
                class_net={"input_size": 16, "output_size": 3},
                md={"n_heads": 2},
            )

    def test_non_positive_dims_rejected(self):
        with pytest.raises(ConfigError, match="num_queries"):
            MaskDecoder(
                embed_dim=16,
                num_queries=0,
                num_layers=2,
                class_net={"output_size": 3},
                md={"n_heads": 2},
            )
        with pytest.raises(ConfigError, match="num_layers"):
            MaskDecoder(
                embed_dim=16,
                num_queries=5,
                num_layers=0,
                class_net={"output_size": 3},
                md={"n_heads": 2},
            )

    def test_declare_io_keys_and_shapes(self):
        md = MaskDecoder(
            embed_dim=16,
            num_queries=5,
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
            num_queries=5,
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

    def test_zero_constituent_jet_is_finite(self):
        # the dummy-token trick keeps a zero-length sequence from NaN-ing (ONNX)
        md = MaskDecoder(
            embed_dim=16,
            num_queries=5,
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
            num_queries=5,
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


# ===========================================================================
# maskformer_loss / matcher: eager-vs-scripted parity (test_mirror_convention.py
# designates this file as the mirror for both)
# ===========================================================================


def test_dice_loss_eager_vs_scripted_parity():
    # K=7 matched (non-null) objects, C=40 constituents — the [K, C] shape
    # loss_masks passes after boolean-indexing labels["masks"] by valid_idx.
    gen = torch.Generator().manual_seed(201)
    inputs = torch.randn(7, 40, generator=gen)
    targets = (torch.rand(7, 40, generator=gen) > 0.5).float()

    eager = dice_loss_eager(inputs.clone(), targets.clone())
    scripted = dice_loss(inputs.clone(), targets.clone())
    assert torch.equal(eager, scripted), f"eager={eager.item()} vs scripted={scripted.item()}"


def test_mask_ce_loss_eager_vs_scripted_parity():
    gen = torch.Generator().manual_seed(202)
    inputs = torch.randn(7, 40, generator=gen)
    targets = (torch.rand(7, 40, generator=gen) > 0.5).float()

    eager = mask_ce_loss_eager(inputs.clone(), targets.clone())
    scripted = mask_ce_loss(inputs.clone(), targets.clone())
    assert torch.equal(eager, scripted), f"eager={eager.item()} vs scripted={scripted.item()}"


def test_sigmoid_focal_loss_eager_vs_scripted_parity():
    gen = torch.Generator().manual_seed(203)
    inputs = torch.randn(7, 40, generator=gen)
    targets = (torch.rand(7, 40, generator=gen) > 0.5).float()

    eager_default = sigmoid_focal_loss_eager(inputs.clone(), targets.clone())
    scripted_default = sigmoid_focal_loss(inputs.clone(), targets.clone())
    assert torch.equal(eager_default, scripted_default), (
        f"alpha<0: eager={eager_default.item()} vs scripted={scripted_default.item()}"
    )

    eager_weighted = sigmoid_focal_loss_eager(inputs.clone(), targets.clone(), alpha=0.25)
    scripted_weighted = sigmoid_focal_loss(inputs.clone(), targets.clone(), alpha=0.25)
    assert torch.equal(eager_weighted, scripted_weighted), (
        f"alpha>=0: eager={eager_weighted.item()} vs scripted={scripted_weighted.item()}"
    )


def test_batch_dice_cost_eager_vs_scripted_parity():
    # square [B, N, C] inputs, matching get_batch_cost's call shape (B=2, N=5, C=40).
    gen = torch.Generator().manual_seed(301)
    inputs = torch.randn(2, 5, 40, generator=gen)
    targets = (torch.rand(2, 5, 40, generator=gen) > 0.5).float()

    eager = batch_dice_cost_eager(inputs.clone(), targets.clone())
    scripted = batch_dice_cost(inputs.clone(), targets.clone())
    assert torch.equal(eager, scripted), (
        f"eager vs scripted diverge at {torch.nonzero((eager != scripted).reshape(-1))}: "
        f"eager={eager.reshape(-1)} scripted={scripted.reshape(-1)}"
    )


def test_batch_sigmoid_ce_cost_eager_vs_scripted_parity():
    gen = torch.Generator().manual_seed(302)
    inputs = torch.randn(2, 5, 40, generator=gen)
    targets = (torch.rand(2, 5, 40, generator=gen) > 0.5).float()

    eager = batch_sigmoid_ce_cost_eager(inputs.clone(), targets.clone())
    scripted = batch_sigmoid_ce_cost(inputs.clone(), targets.clone())
    assert torch.equal(eager, scripted), (
        f"eager vs scripted diverge at {torch.nonzero((eager != scripted).reshape(-1))}: "
        f"eager={eager.reshape(-1)} scripted={scripted.reshape(-1)}"
    )


def test_batch_sigmoid_focal_cost_eager_vs_scripted_parity():
    gen = torch.Generator().manual_seed(303)
    inputs = torch.randn(2, 5, 40, generator=gen)
    targets = (torch.rand(2, 5, 40, generator=gen) > 0.5).float()

    eager_default = batch_sigmoid_focal_cost_eager(inputs.clone(), targets.clone())
    scripted_default = batch_sigmoid_focal_cost(inputs.clone(), targets.clone())
    assert torch.equal(eager_default, scripted_default), (
        f"alpha<0 diverge at {torch.nonzero((eager_default != scripted_default).reshape(-1))}: "
        f"eager={eager_default.reshape(-1)} scripted={scripted_default.reshape(-1)}"
    )

    eager_weighted = batch_sigmoid_focal_cost_eager(inputs.clone(), targets.clone(), alpha=0.25)
    scripted_weighted = batch_sigmoid_focal_cost(inputs.clone(), targets.clone(), alpha=0.25)
    assert torch.equal(eager_weighted, scripted_weighted), (
        f"alpha>=0 diverge at {torch.nonzero((eager_weighted != scripted_weighted).reshape(-1))}: "
        f"eager={eager_weighted.reshape(-1)} scripted={scripted_weighted.reshape(-1)}"
    )


def test_batch_mae_loss_eager_vs_scripted_parity():
    # preds [B, N, D] and targets [B, M, D] with B=2, N=M=5, D=2.
    gen = torch.Generator().manual_seed(304)
    inputs = torch.randn(2, 5, 2, generator=gen)
    targets = torch.randn(2, 5, 2, generator=gen)

    eager = batch_mae_loss_eager(inputs.clone(), targets.clone())
    scripted = batch_mae_loss(inputs.clone(), targets.clone())
    assert torch.equal(eager, scripted), (
        f"eager vs scripted diverge at {torch.nonzero((eager != scripted).reshape(-1))}: "
        f"eager={eager.reshape(-1)} scripted={scripted.reshape(-1)}"
    )
