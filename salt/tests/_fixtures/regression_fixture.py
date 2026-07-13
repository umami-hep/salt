"""Encoder-less regression fixtures for the M5 sub-wave A `RegressionTaskModule`.

DEL-1: every v1-free builder with live v2 consumers moved to ``v2_builders.py``
and is re-imported here (this file dies with the v1 tree; only the
``build_independent_v1_*`` comparison machinery and consumer-less legacy
builders remain defined below).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from salt.core.data import MaskFormerTargets
from salt.core.graph.planner import Plan, compile_plan
from salt.core.graph.spec import (
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
    flatten_spec,
    unflatten_spec,
)
from salt.core.nn import (
    Concat,
    GlobalAttentionPooling,
    LossGLS,
    LossSum,
    MaskDecoder,
    MaskFormerMatchedLoss,
    Normaliser,
    Split,
    StreamEmbed,
    TransformerEncoder,
)
from salt.core.nn.tasks import ClassificationTaskModule, RegressionTaskModule
from salt.core.onnx import (
    ExportConfig,
    ExportInput,
    compile_onnx_plan,
    resolve_export_config,
)
from salt.models import Dense as V1Dense
from salt.models import Transformer as V1Transformer
# Pinned to a vendored upstream snapshot, not the live salt.models copies (which
# diverge under ongoing absorption work).
from salt.tests._fixtures.upstream_mf_snapshot.maskformer import MaskDecoder as V1MaskDecoder
from salt.tests._fixtures.upstream_mf_snapshot.maskformer_loss import (
    MaskFormerLoss as V1MaskFormerLoss,
)
from salt.models.task import GaussianRegressionTask as V1GaussianRegressionTask
from salt.models.task import RegressionTask as V1RegressionTask
from salt.tests._fixtures.gn2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    build_test_gn2,
    write_parity_norm_dict,
)
from salt.tests._fixtures.v2_builders import (
    DIPS_CLASS_NAMES,
    GLOBAL_VARIABLES,
    MASKFORMER_ENC_DIM,
    MASKFORMER_LOSS_WEIGHTS,
    MASKFORMER_NULL_CLASS_WEIGHT,
    MASKFORMER_NUM_OBJECT_CLASSES,
    MASKFORMER_NUM_OBJECTS,
    MASKFORMER_NUM_REGISTERS,
    MASKFORMER_WRITER_REG_TARGETS,
    POOL_DIM,
    VCONCAT_DIM,
    build_maskformer_decoder_modules,
    build_maskformer_writer_modules,
    build_matched_loss_module,
    build_regression_modules,
    build_vector_concat_modules,
    compile_maskformer_decoder,
    compile_regression,
    compile_vector_concat,
    dips_sources,
    make_maskformer_writer_batch,
    make_regression_labels,
    regression_sources,
    vector_concat_sources,
    write_vector_concat_norm_dict,
)
from salt.utils.scalers import RegressionTargetScaler

__all__ = [
    "DIPS_CLASS_NAMES",
    "GLOBAL_VARIABLES",
    "HYBRID_ENC_DIM",
    "MASKFORMER_ENC_DIM",
    "MASKFORMER_NUM_OBJECTS",
    "MASKFORMER_NUM_OBJECT_CLASSES",
    "MASKFORMER_NUM_REGISTERS",
    "MASKFORMER_NUM_REG_TARGETS",
    "MASKFORMER_OBJECT_CLASS_MAP",
    "MASKFORMER_WRITER_REG_TARGETS",
    "POOL_DIM",
    "VCONCAT_DIM",
    "build_dips_modules",
    "build_gls_modules",
    "build_hybrid_encoder_modules",
    "build_independent_v1_global_concat",
    "build_independent_v1_gls_total_loss",
    "build_independent_v1_head",
    "build_independent_v1_mask_decoder",
    "build_independent_v1_matched_loss",
    "build_independent_v1_transformer",
    "build_independent_v1_transformer_drop",
    "build_maskformer_decoder_modules",
    "build_maskformer_targets",
    "build_maskformer_writer_modules",
    "build_matched_loss_module",
    "build_regression_modules",
    "build_vector_concat_modules",
    "compile_dips",
    "compile_gls",
    "compile_hybrid_encoder",
    "compile_maskformer_decoder",
    "compile_vector_concat",
    "compile_vector_concat_onnx",
    "dips_sources",
    "gls_sources",
    "make_dips_labels",
    "make_gls_labels",
    "make_maskformer_object_batch",
    "make_maskformer_targets_batch",
    "make_maskformer_writer_batch",
    "make_regression_labels",
    "regression_sources",
    "vector_concat_sources",
    "write_vector_concat_norm_dict",
]

def build_independent_v1_head(
    composed: V1RegressionTask | V1GaussianRegressionTask,
    scaler: RegressionTargetScaler | None = None,
) -> V1RegressionTask | V1GaussianRegressionTask:
    """A FRESH, separately-constructed v1 head with ``composed``'s weights copied in."""
    net = composed.net
    # node_list packs input_size+context_size, hidden layers, then output_size —
    # dense_config is recoverable from the trained net's geometry.
    dense_config = {
        "input_size": net.node_list[0] - net.context_size,
        "output_size": net.node_list[-1],
        "hidden_layers": list(net.node_list[1:-1]),
        "context_size": net.context_size,
        "activation": "ReLU",
    }
    norm_params = (
        {"mean": list(composed.norm_params["mean"]), "std": list(composed.norm_params["std"])}
        if composed.norm_params is not None
        else None
    )
    common: dict = {
        "name": composed.name,
        "input_name": composed.input_name,
        "targets": list(composed.targets),
        "target_denominators": (
            list(composed.target_denominators) if composed.target_denominators is not None else None
        ),
        "norm_params": norm_params,
        "custom_output_names": (
            list(composed.custom_output_names) if composed.custom_output_names is not None else None
        ),
        "sample_weight": composed.sample_weight,
        "loss": composed.loss,
        "weight": composed.weight,
        "dense_config": dense_config,
    }
    # composed is the v2-native absorbed head, so detect the gaussian case
    # STRUCTURALLY (output_size == 2 * len(targets)) instead of isinstance.
    fresh: V1RegressionTask | V1GaussianRegressionTask
    if composed.net.output_size == 2 * len(composed.targets):
        fresh = V1GaussianRegressionTask(**common)
    else:
        fresh = V1RegressionTask(scaler=scaler, **common)
    fresh.load_state_dict(composed.state_dict())
    fresh.eval()
    return fresh


# legacy/dips.yaml — the encoder-less CI smoke fixture


def build_dips_modules(
    norm_dict: Path | str,
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """The v2 port of ``legacy/dips.yaml`` — encoder-less DiPS, one jet classifier."""
    dense = {"hidden_layers": [POOL_DIM], "activation": "ReLU"}
    head_dense = {"hidden_layers": [POOL_DIM], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=list(streams), global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=POOL_DIM, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="seq.x", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="HadronConeExclTruthLabelID",
            class_names=list(DIPS_CLASS_NAMES),
            label_map={0: 0, 4: 1, 5: 2},
            input="pooled.global",
            dense=head_dense,
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    loss = modules["loss"]
    assert isinstance(loss, LossSum)
    loss.narrow(LossSum.collect_loss_keys(modules))
    return modules


def compile_dips(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the encoder-less dips plan for one mode."""
    sinks = ["loss.total"] if mode & Mode.TRAINING else ["preds.jets.jets_classification"]
    return compile_plan(modules, mode, sources=dips_sources(), sinks=sinks)


def make_dips_labels(batch_size: int, seed: int = 5) -> dict[str, Tensor]:
    """Deterministic flavour labels (raw ids 0/4/5) for the dips FIT plan."""
    gen = torch.Generator().manual_seed(seed)
    raw_ids = torch.tensor([0, 4, 5], dtype=torch.int64)
    pick = torch.randint(0, 3, (batch_size,), generator=gen)
    return {"labels.jets.HadronConeExclTruthLabelID": raw_ids[pick]}


# LossGLS — geometric-mean loss combination

GLS_TARGET = "R10TruthLabel_R22v1_TruthJetPt"
"""The single regression target of the GLS fixture's regression head."""


def build_gls_modules(
    norm_dict: Path | str,
    weights: dict[str, float] | None = None,
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """A TWO-task encoder-less DiPS plan ending in `LossGLS` (>= 2 task losses)."""
    weights = weights or {}
    dense = {"hidden_layers": [POOL_DIM], "activation": "ReLU"}
    head_dense = {"hidden_layers": [POOL_DIM], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=list(streams), global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=POOL_DIM, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="seq.x", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="HadronConeExclTruthLabelID",
            class_names=list(DIPS_CLASS_NAMES),
            label_map={0: 0, 4: 1, 5: 2},
            input="pooled.global",
            dense=head_dense,
            weight=weights.get("jets_classification", 1.0),
        ),
        "jets_regression": RegressionTaskModule(
            stream="jets",
            targets=[GLS_TARGET],
            input="pooled.global",
            norm_params={"mean": 1.0, "std": 1.0},
            dense=head_dense,
            weight=weights.get("jets_regression", 1.0),
        ),
        "loss": LossGLS(),
    }
    for name, module in modules.items():
        module.name = name
    loss = modules["loss"]
    assert isinstance(loss, LossGLS)
    loss.narrow(LossGLS.collect_loss_keys(modules))
    return modules


def gls_sources() -> NestedSpec:
    """Dataset-boundary source spec for the GLS plan (GN2 inputs + both labels)."""
    # add the regression target label leaf (TRAINING-gated, like the flavour one).
    merged = dict(flatten_spec(dips_sources()))
    merged[f"labels.jets.{GLS_TARGET}"] = TensorSpec(
        shape=("B",), dtype="float32", kind="label", modes=Mode.TRAINING
    )
    return unflatten_spec(merged)


def compile_gls(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the encoder-less two-task GLS plan for one mode."""
    sinks = (
        ["loss.total"]
        if mode & Mode.TRAINING
        else ["preds.jets.jets_classification", "preds.jets.jets_regression"]
    )
    return compile_plan(modules, mode, sources=gls_sources(), sinks=sinks)


def make_gls_labels(batch_size: int, seed: int = 5) -> dict[str, Tensor]:
    """Deterministic labels for the two-task GLS FIT plan (flavour + reg target)."""
    labels = make_dips_labels(batch_size, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    labels[f"labels.jets.{GLS_TARGET}"] = 1.0 + torch.rand(batch_size, generator=gen) * 3.0
    return labels


def build_independent_v1_gls_total_loss(norm_dir: Path | str):
    """A FRESH v1 ``ModelWrapper`` with ``loss_mode='GLS'``; return its bound ``total_loss``."""
    wrapper = build_test_gn2(norm_dir)
    wrapper.loss_mode = "GLS"
    return wrapper.total_loss


# norm_type: hybrid encoder block

HYBRID_ENC_DIM = 16
"""Encoder embed/out width of the hybrid-norm fixture (matches the gn2v2 width)."""

HYBRID_NUM_LAYERS = 3
"""Encoder depth of the hybrid fixture. >= 2 so the hybrid depth-0 vs depth>0
residual-norm split (``"pre"``+Identity at depth 0, ``"none"``+real norm after)
is genuinely exercised across layers."""

HYBRID_NUM_HEADS = 2
"""Attention heads of the hybrid fixture."""


def build_hybrid_encoder_modules(
    norm_dict: Path | str,
    norm_type: str = "hybrid",
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """A GN3V01-style ENCODER block: norm -> embed -> concat -> encoder -> split -> pool -> head."""
    dense = {"hidden_layers": [HYBRID_ENC_DIM], "activation": "ReLU"}
    head_dense = {"hidden_layers": [HYBRID_ENC_DIM], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=list(streams), global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=HYBRID_ENC_DIM, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=HYBRID_ENC_DIM,
            num_layers=HYBRID_NUM_LAYERS,
            out_dim=HYBRID_ENC_DIM,
            attention={"num_heads": HYBRID_NUM_HEADS, "attn_type": "torch-math"},
            dense={"activation": "ReLU", "gated": False},
            norm_type=norm_type,
        ),
        "split": Split(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="HadronConeExclTruthLabelID",
            class_names=list(DIPS_CLASS_NAMES),
            label_map={0: 0, 4: 1, 5: 2},
            input="pooled.global",
            dense=head_dense,
        ),
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    loss = modules["loss"]
    assert isinstance(loss, LossSum)
    loss.narrow(LossSum.collect_loss_keys(modules))
    return modules


def compile_hybrid_encoder(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the hybrid-encoder plan for one mode (GN2 inputs + flavour label)."""
    sinks = ["loss.total"] if mode & Mode.TRAINING else ["preds.jets.jets_classification"]
    return compile_plan(modules, mode, sources=dips_sources(), sinks=sinks)


def build_independent_v1_transformer(composed: TransformerEncoder) -> V1Transformer:
    """A FRESH, separately-constructed v1 `Transformer` with ``composed``'s weights copied in."""
    inner = composed.encoder
    n_heads = inner.layers[0].attn.fn.num_heads
    fresh = V1Transformer(
        num_layers=inner.num_layers,
        embed_dim=inner.embed_dim,
        out_dim=inner.out_dim if inner.do_out_proj else None,
        attn_type="torch-math",
        do_final_norm=inner.do_final_norm,
        num_registers=inner.num_registers,
        norm_type=composed.norm_type,
        attn_kwargs={"num_heads": n_heads},
        dense_kwargs={"activation": "ReLU", "gated": False},
    )
    fresh.load_state_dict(inner.state_dict())
    fresh.eval()
    return fresh


# VectorConcat + export.inputs alias — GN3 'global' stream past the encoder
# (builders moved to v2_builders.py; only the consumer-less ONNX compile stays)


def compile_vector_concat_onnx(
    modules: dict[str, GraphModule],
    *,
    alias_variables: list[str] | None = None,
):
    """Compile the Mode.ONNX plan + resolved export config for the alias path."""
    alias_cols = list(JET_VARIABLES) if alias_variables is None else list(alias_variables)
    export = ExportConfig(
        model_name="GN3V01",
        inputs=[
            ExportInput(port="inputs.jets", name="jet_features"),
            ExportInput(
                port="inputs.tracks", name="track_features", sequence=True, dyn_axis="n_tracks"
            ),
            # the GN3 alias: one Athena jet tensor cloned into the global port
            ExportInput(port="inputs.global", alias="inputs.jets"),
        ],
    )
    # a folded ClassProbs node named by an OnnxExportSink
    from salt.core.outputs import ClassProbs, OnnxExportLeaf, OnnxExportSink  # noqa: PLC0415

    cp = ClassProbs(task="jets_classification", stream="jets")
    cp.name = "jet_probs"
    sink = OnnxExportSink(outputs=[
        OnnxExportLeaf(key="outputs.jets.jets_classification", names=list(DIPS_CLASS_NAMES)),
    ])
    sink.name = "onnx_export"
    modules["jet_probs"] = cp
    modules["onnx_export"] = sink
    resolved = resolve_export_config(export, "GN3V01")
    variables = {
        "jets": list(JET_VARIABLES),
        "tracks": list(TRACK_VARIABLES),
        "global": alias_cols,
    }
    plan = compile_onnx_plan(modules, resolved, variables)
    fields = {f"inputs.{s}": tuple(names) for s, names in variables.items()}
    return plan, resolved, fields


def build_independent_v1_global_concat(pooled: Tensor, global_feats: Tensor) -> Tensor:
    """The v1 reference for the post-pooling global concat (saltmodel.py:175-177)."""
    return torch.cat([pooled, global_feats], dim=-1)


# MaskDecoder + encoder drop_registers — the MaskFormer object decoder over a
# register-dropping encoder (builders moved to v2_builders.py)


def build_independent_v1_transformer_drop(composed: TransformerEncoder) -> V1Transformer:
    """A FRESH v1 `Transformer` (``drop_registers=True``) with ``composed``'s weights."""
    inner = composed.encoder
    n_heads = inner.layers[0].attn.fn.num_heads
    fresh = V1Transformer(
        num_layers=inner.num_layers,
        embed_dim=inner.embed_dim,
        out_dim=inner.out_dim if inner.do_out_proj else None,
        attn_type="torch-math",
        do_final_norm=inner.do_final_norm,
        num_registers=inner.num_registers,
        drop_registers=True,
        attn_kwargs={"num_heads": n_heads},
        dense_kwargs={"activation": "ReLU", "gated": False},
    )
    fresh.load_state_dict(inner.state_dict())
    fresh.eval()
    return fresh


def build_independent_v1_mask_decoder(composed: MaskDecoder) -> V1MaskDecoder:
    """A FRESH v1 `MaskDecoder` with ``composed``'s weights copied in."""
    class_net = composed.class_net
    mask_net = composed.mask_net
    # rebuild fresh Dense heads with the SAME geometry (context_size is 0 here)
    fresh_class = V1Dense(
        input_size=class_net.node_list[0] - class_net.context_size,
        output_size=class_net.node_list[-1],
        hidden_layers=list(class_net.node_list[1:-1]),
    )
    fresh_mask = V1Dense(
        input_size=mask_net.node_list[0] - mask_net.context_size,
        output_size=mask_net.node_list[-1],
        hidden_layers=list(mask_net.node_list[1:-1]),
    )
    md_config = {
        "n_heads": composed.layers[0].q_ca.num_heads,
        "mask_attention": composed.layers[0].mask_attention,
        "bidirectional_ca": composed.layers[0].bidirectional_ca,
    }
    fresh = V1MaskDecoder(
        embed_dim=composed.embed_dim,
        num_layers=len(composed.layers),
        md_config=md_config,
        class_net=fresh_class,
        mask_net=fresh_mask,
        num_objects=composed.num_objects,
        # the HungarianMatcher asserts the loss-weight sum is positive — supply a
        # dummy positive weight; the loss is NEVER invoked (forward runs with labels=None).
        loss_config={
            "num_classes": composed.num_classes,
            "loss_weights": {"object_class_ce": 1.0},
        },
        aux_loss=False,
    )
    # the v1 decoder's own state_dict also carries mask_loss buffers with no v2
    # counterpart; strict=False loads only the shared structural weights.
    fresh.load_state_dict(composed.state_dict(), strict=False)
    fresh.eval()
    return fresh


# MaskFormerMatchedLoss + HungarianMatcher — the matched loss over the decoder's
# object predictions

MASKFORMER_NUM_REG_TARGETS = 3
"""Object-regression target count R of the matched-loss fixture (a small stand-in
for the shipped 5: pt/Lxy/deta/dphi/mass)."""


def build_independent_v1_matched_loss(
    composed: MaskFormerMatchedLoss,
) -> V1MaskFormerLoss:
    """A FRESH, separately-constructed v1 `MaskFormerLoss` for the matched-loss parity."""
    fresh = V1MaskFormerLoss(
        num_classes=composed.num_classes,
        num_objects=composed.num_objects,
        loss_weights=dict(composed.loss_weights),
        matcher_weights=dict(composed.matcher_weights),
        null_class_weight=composed.null_class_weight,
    )
    # the only state is the empty_weight buffer (config-deterministic); copy it
    # anyway so a future learnable term would still be matched.
    fresh.load_state_dict(composed.v1_loss.state_dict())
    fresh.eval()
    return fresh


def make_maskformer_object_batch(
    batch_size: int = 6,
    num_objects: int = MASKFORMER_NUM_OBJECTS,
    num_classes: int = MASKFORMER_NUM_OBJECT_CLASSES - 1,
    n_tracks: int = 10,
    n_reg: int = MASKFORMER_NUM_REG_TARGETS,
    seed: int = 23,
) -> dict[str, Tensor]:
    """Deterministic object predictions + truth labels for the matched-loss gate."""
    gen = torch.Generator().manual_seed(seed)
    n_cls = num_classes + 1  # incl. null
    class_logits = torch.randn(batch_size, num_objects, n_cls, generator=gen)
    class_probs = class_logits.softmax(-1)
    masks = torch.randn(batch_size, num_objects, n_tracks, generator=gen)
    embed = torch.randn(batch_size, num_objects, MASKFORMER_ENC_DIM, generator=gen)
    reg_pred = torch.randn(batch_size, num_objects, n_reg, generator=gen)
    reg_tgt = torch.randn(batch_size, num_objects, n_reg, generator=gen)

    # truth object classes: a varying number of valid (0..num_classes-1) objects per
    # row, the rest null (== num_classes). Valid objects are placed FIRST (the matcher
    # treats the first batch_obj_lengths columns as valid).
    object_class = torch.full((batch_size, num_objects), num_classes, dtype=torch.int64)
    for b in range(batch_size):
        n_valid = int(torch.randint(1, num_objects, (1,), generator=gen).item())
        object_class[b, :n_valid] = torch.randint(0, num_classes, (n_valid,), generator=gen)
    # truth masks: a random boolean target mask over the tracks for each object
    target_masks = torch.rand(batch_size, num_objects, n_tracks, generator=gen) > 0.5

    return {
        "objects.class_logits": class_logits,
        "objects.class_probs": class_probs,
        "objects.masks": masks,
        "objects.embed": embed,
        "preds.objects.regression": reg_pred,
        "targets.objects.regression": reg_tgt,
        "labels.objects.object_class": object_class,
        "labels.objects.masks": target_masks,
    }


# MaskFormerTargets processor — object class + truth masks

# the shipped object class map: b->0, c->1, null->2 (null LAST). Raw ids 5/4/-1
# (HadronConeExclTruthLabel-style flavour codes).
MASKFORMER_OBJECT_CLASS_MAP = {
    "b": {"raw": 5, "mapped": 0},
    "c": {"raw": 4, "mapped": 1},
    "null": {"raw": -1, "mapped": 2},
}
"""The shipped object class map (b/c/null, null LAST)."""


def build_maskformer_targets(
    object_stream: str = "truth_hadrons",
    constituent_stream: str = "tracks",
    regression_targets: Sequence[str] = ("pt", "Lxy", "mass"),
    num_objects: int | None = MASKFORMER_NUM_OBJECTS,
) -> MaskFormerTargets:
    """A `MaskFormerTargets` processor matching the shipped MaskFormer.yaml mf_config."""
    proc = MaskFormerTargets(
        object_class="flavour",
        object_id="barcode",
        constituent_id="ftagTruthParentBarcode",
        class_map=MASKFORMER_OBJECT_CLASS_MAP,
        object_stream=object_stream,
        constituent_stream=constituent_stream,
        regression_targets=list(regression_targets),
        num_objects=num_objects,
    )
    proc.name = "object_targets"
    return proc


def make_maskformer_targets_batch(
    batch_size: int = 6,
    num_objects: int = MASKFORMER_NUM_OBJECTS,
    n_tracks: int = 10,
    regression_targets: Sequence[str] = ("pt", "Lxy", "mass"),
    object_stream: str = "truth_hadrons",
    constituent_stream: str = "tracks",
    seed: int = 29,
) -> dict[str, np.ndarray]:
    """A raw numpy batch for the `MaskFormerTargets` processor (structured arrays)."""
    rng = np.random.default_rng(seed)
    # object barcodes: distinct positive ids for valid objects, -1 for invalid
    obj_flav = np.full((batch_size, num_objects), -1, dtype=np.int32)
    obj_barcode = np.full((batch_size, num_objects), -1, dtype=np.int64)
    track_parent = np.full((batch_size, n_tracks), -999, dtype=np.int64)
    for b in range(batch_size):
        n_valid = int(rng.integers(1, num_objects))
        # unique positive barcodes per valid object (offset by batch so cross-row
        # barcodes never collide — a row's tracks only match its own objects)
        barcodes = (b + 1) * 100 + np.arange(n_valid) + 1
        obj_barcode[b, :n_valid] = barcodes
        obj_flav[b, :n_valid] = rng.choice([5, 4], size=n_valid)  # raw b/c codes
        # assign each track to one of this row's valid objects (or leave -999 = unmatched)
        for t in range(n_tracks):
            if rng.random() < 0.7:
                track_parent[b, t] = barcodes[rng.integers(0, n_valid)]

    obj_dtype = [("barcode", "i8"), ("flavour", "i4")] + [
        (name, "f4") for name in regression_targets
    ]
    obj = np.zeros((batch_size, num_objects), dtype=obj_dtype)
    obj["barcode"] = obj_barcode
    obj["flavour"] = obj_flav
    for i, name in enumerate(regression_targets):
        obj[name] = rng.standard_normal((batch_size, num_objects)).astype(np.float32) * (i + 1)

    con = np.zeros((batch_size, n_tracks), dtype=[("ftagTruthParentBarcode", "i8")])
    con["ftagTruthParentBarcode"] = track_parent

    return {
        f"raw.{object_stream}": obj,
        f"raw.{constituent_stream}": con,
    }


# MaskFormerObjectWriter fixtures moved to v2_builders.py (re-imported above).
