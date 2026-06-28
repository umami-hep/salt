"""Encoder-less regression fixtures for the M5 sub-wave A `RegressionTaskModule`.

Every shipped regression config is encoder-less (``init_nets`` + ``pool_net``,
no ``encoder:`` block — v1 saltmodel.py:90-93). These fixtures mirror that
shape (norm -> embed -> concat -> pool -> regression head -> loss) so the
``RegressionTaskModule`` is exercised through the real plan compiler /
two-phase bind / executor, exactly as `gn2v2_fixture` does for the
classification family.

Each builder pairs a v2 ``RegressionTaskModule`` config with the construction
args of a directly-instantiated v1 ``RegressionTask`` head, so a parity test
can compare the composed v2 forward/de-scaling against a v1 reference built
from the SAME kwargs.

Everything is built from plain config kwargs — no ``input_size`` anywhere
(widths inferred at bind, design §2.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from salt.core.data import MaskFormerTargets
from salt.core.graph.planner import Plan, compile_plan
from salt.core.graph.spec import (
    IO,
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
    VectorConcat,
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
# MFU-0 step B: the MaskFormer parity oracle (MF1a/MF1c) is pinned to upstream
# 6570e85 via a vendored snapshot, NOT the live salt.models copies (which diverge
# under the MFU absorption work). See salt/tests/_fixtures/upstream_mf_snapshot/.
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

DIPS_CLASS_NAMES = ("ujets", "cjets", "bjets")
"""The legacy ``dips.yaml`` flavour classes (label_map 0/4/5 -> 0/1/2)."""

POOL_DIM = 16
"""The encoder-less pool/embed width (matches the gn2v2 fixture)."""


def build_regression_modules(
    norm_dict: Path | str,
    task: RegressionTaskModule,
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """A DiPS-shaped module dict ending in one `RegressionTaskModule`.

    ``norm -> track_embed -> concat -> pool[seq.x] -> <task> -> loss`` — no
    encoder, no `Split` (the encoder-less path the regression family always
    uses, v1 saltmodel.py:155-156). The pool publishes ``pooled.global``; a
    GLOBAL-stream task reads it via ``input="pooled.global"``, a per-token
    SEQUENCE task reads ``encoded.tracks`` — but the encoder-less path has no
    ``encoded.*``, so per-token regression fixtures read ``seq.x`` directly.

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules with `LossSum` already narrowed.
    """
    dense = {"hidden_layers": [POOL_DIM], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=list(streams), global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=POOL_DIM, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="seq.x", out="pooled.global"),
        "regression": task,
        "loss": LossSum(),
    }
    for name, module in modules.items():
        module.name = name
    loss = modules["loss"]
    assert isinstance(loss, LossSum)
    loss.narrow(LossSum.collect_loss_keys(modules))
    return modules


def regression_sources(
    targets: tuple[str, ...],
    denominators: tuple[str, ...] = (),
    weight: str | None = None,
    *,
    sequence: bool = False,
) -> NestedSpec:
    """Dataset-boundary source spec: the GN2 inputs + the regression labels.

    ``inputs.*`` carry their declared variables as `fields` (the bind-time
    column source for the ONNX denominator gather, design §2.2); each
    regression target (and ratio denominator) is a float TRAINING/TEST label
    leaf under ``labels.jets.*``. A ``weight`` adds the per-sample weight label
    (TRAINING-only, the sample_weight source). When ``sequence`` is set the
    target labels live under ``labels.tracks.*`` shaped ``(B, T:tracks)`` — the
    per-token regression path the functional ``scaler`` de-scales (v1
    ``run_inference`` ``preds[:, :, i]``, task.py:594-596).

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    label = {"dtype": "float32", "kind": "label"}
    sources: dict[str, TensorSpec] = {
        "inputs.jets": TensorSpec(
            shape=("B", len(JET_VARIABLES)), dtype="float32", fields=tuple(JET_VARIABLES)
        ),
        "inputs.tracks": TensorSpec(
            shape=("B", "T:tracks", len(TRACK_VARIABLES)),
            dtype="float32",
            fields=tuple(TRACK_VARIABLES),
        ),
        "masks.tracks": TensorSpec(shape=("B", "T:tracks"), dtype="bool", kind="pad_mask"),
    }
    label_stream = "tracks" if sequence else "jets"
    label_shape: tuple[str, ...] = ("B", "T:tracks") if sequence else ("B",)
    for target in targets:
        sources[f"labels.{label_stream}.{target}"] = TensorSpec(
            shape=label_shape, modes=Mode.TRAINING, **label
        )
    for denom in denominators:
        # FIT|VAL|TEST source for ratio denominators (ONNX reads them from
        # inputs.jets instead, FD §3.3); pt_btagJes is ALSO an input Feature.
        sources[f"labels.jets.{denom}"] = TensorSpec(
            shape=("B",), modes=Mode.FIT | Mode.VAL | Mode.TEST, **label
        )
    if weight is not None:
        sources[f"labels.jets.{weight}"] = TensorSpec(shape=("B",), modes=Mode.TRAINING, **label)
    return unflatten_spec(sources)


def compile_regression(
    modules: dict[str, GraphModule],
    mode: Mode,
    targets: tuple[str, ...],
    denominators: tuple[str, ...] = (),
    weight: str | None = None,
    *,
    sequence: bool = False,
) -> Plan:
    """Compile the encoder-less regression plan for one mode.

    Returns
    -------
    Plan
        FIT/VAL anchored on ``loss.total``; TEST/ONNX on the prediction leaf.
    """
    pred_stream = "tracks" if sequence else "jets"
    sinks = ["loss.total"] if mode & Mode.TRAINING else [f"preds.{pred_stream}.regression"]
    return compile_plan(
        modules,
        mode,
        sources=regression_sources(targets, denominators, weight, sequence=sequence),
        sinks=sinks,
    )


def make_regression_labels(
    batch_size: int,
    targets: tuple[str, ...],
    denominators: tuple[str, ...] = (),
    seed: int = 11,
    *,
    seq_len: int | None = None,
) -> dict[str, Tensor]:
    """Deterministic positive float labels for the targets + ratio denominators.

    Positive values keep ``log``-scaler de-scaling well-defined; denominators
    are bounded away from zero so the ratio targets are finite. When ``seq_len``
    is given the target labels are per-token (``labels.tracks.<name>`` shaped
    ``[B, seq_len]``) — the functional-scaler sequence path; denominators stay
    jet-level (the scaler path uses no denominators).

    Returns
    -------
    dict[str, Tensor]
        ``{labels.<stream>.<name>: tensor}`` flat keys for the v2 bundle.
    """
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, Tensor] = {}
    label_stream = "tracks" if seq_len is not None else "jets"
    shape = (batch_size, seq_len) if seq_len is not None else (batch_size,)
    for i, target in enumerate(targets):
        out[f"labels.{label_stream}.{target}"] = 1.0 + torch.rand(*shape, generator=gen) * (i + 2)
    for denom in denominators:
        out[f"labels.jets.{denom}"] = 2.0 + torch.rand(batch_size, generator=gen)
    return out


def build_independent_v1_head(
    composed: V1RegressionTask | V1GaussianRegressionTask,
    scaler: RegressionTargetScaler | None = None,
) -> V1RegressionTask | V1GaussianRegressionTask:
    """A FRESH, separately-constructed v1 head with ``composed``'s weights copied in.

    This is the regression analogue of R3's dips-pooling cross-impl pattern: the
    `RegressionTaskModule` composes a v1 head and the executor invokes THAT object,
    so re-calling ``module.task`` measures self-consistency, not independent
    v1-vs-v2 parity. Here a brand-new ``RegressionTask``/``GaussianRegressionTask``
    is built from scratch — its own `Dense` net, its own scaling state — with the
    SAME architecture (read off the composed net's ``node_list``) and the SAME
    scaling kwargs, then ``load_state_dict`` copies the trained weights in. Driving
    THIS object reproduces v1 from an independent construction path; comparing it to
    the executor's output is a genuine v1-vs-v2 cross-impl check (the v2 module's
    Split/pool/executor key-routing must agree with a head v2 never touched).

    Returns
    -------
    V1RegressionTask | V1GaussianRegressionTask
        A fresh head (``.eval()``) with ``composed``'s weights, ready to drive.
    """
    net = composed.net
    # Dense.node_list packs the widths as input_size+context_size, the hidden
    # layers, then output_size, so the original dense_config is recoverable from
    # the trained net's geometry (the fresh v1 head rebuilds an identical Dense).
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
    # M7 W2c-1: ``composed`` is now the v2-native absorbed head (RegressionTaskModule
    # composes salt.core.nn.tasks._AbsorbedRegressionTask / _AbsorbedGaussianRegressionTask),
    # so detect the gaussian case STRUCTURALLY (output_size == 2 * len(targets), the v1
    # GaussianRegressionTask invariant, task.py:664) instead of isinstance against the v1
    # class — the independent v1 ORACLE built here is unchanged.
    fresh: V1RegressionTask | V1GaussianRegressionTask
    if composed.net.output_size == 2 * len(composed.targets):
        fresh = V1GaussianRegressionTask(**common)
    else:
        fresh = V1RegressionTask(scaler=scaler, **common)
    fresh.load_state_dict(composed.state_dict())
    fresh.eval()
    return fresh


# ---------------------------------------------------------------------------
# legacy/dips.yaml — the encoder-less CI smoke fixture (R3 pooling parity)
# ---------------------------------------------------------------------------


def build_dips_modules(
    norm_dict: Path | str,
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """The v2 port of ``legacy/dips.yaml`` — encoder-less DiPS, one jet classifier.

    ``norm -> track_embed -> concat -> pool[seq.x] -> jets_classification ->
    loss`` — the exact shape of ``salt/core/configs/dips.yaml`` (the primary CI
    smoke fixture, test_pipeline.py:208,309,317): ``init_nets`` + ``pool_net``
    with NO encoder, so the `GlobalAttentionPooling` reads ``seq.x`` (the `Concat`
    output) and the optional ``masks.registers`` require is dropped at compile
    (v1 saltmodel.py:90-93,155-156). Distinct from `build_regression_modules`
    only in the head (a classifier, the legacy fixture's sole task) so R3 gates
    the encoder-less pooling path on the dips config it ships, not a regression
    stand-in.

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules with `LossSum` already narrowed.
    """
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


def dips_sources() -> NestedSpec:
    """Dataset-boundary source spec for the dips plan (GN2 inputs + flavour label).

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    return unflatten_spec({
        "inputs.jets": TensorSpec(
            shape=("B", len(JET_VARIABLES)), dtype="float32", fields=tuple(JET_VARIABLES)
        ),
        "inputs.tracks": TensorSpec(
            shape=("B", "T:tracks", len(TRACK_VARIABLES)),
            dtype="float32",
            fields=tuple(TRACK_VARIABLES),
        ),
        "masks.tracks": TensorSpec(shape=("B", "T:tracks"), dtype="bool", kind="pad_mask"),
        "labels.jets.HadronConeExclTruthLabelID": TensorSpec(
            shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
        ),
    })


def compile_dips(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the encoder-less dips plan for one mode.

    Returns
    -------
    Plan
        FIT/VAL anchored on ``loss.total``; TEST/ONNX on the prediction leaf.
    """
    sinks = ["loss.total"] if mode & Mode.TRAINING else ["preds.jets.jets_classification"]
    return compile_plan(modules, mode, sources=dips_sources(), sinks=sinks)


def make_dips_labels(batch_size: int, seed: int = 5) -> dict[str, Tensor]:
    """Deterministic flavour labels (raw ids 0/4/5) for the dips FIT plan.

    Returns
    -------
    dict[str, Tensor]
        ``{labels.jets.HadronConeExclTruthLabelID: tensor[B] int64}`` — raw
        v1 ids the head's ``label_map`` (0/4/5 -> 0/1/2) remaps.
    """
    gen = torch.Generator().manual_seed(seed)
    raw_ids = torch.tensor([0, 4, 5], dtype=torch.int64)
    pick = torch.randint(0, 3, (batch_size,), generator=gen)
    return {"labels.jets.HadronConeExclTruthLabelID": raw_ids[pick]}


# ---------------------------------------------------------------------------
# LossGLS — geometric-mean loss combination (M5 sub-wave B, L1 gate)
# ---------------------------------------------------------------------------

GLS_TARGET = "R10TruthLabel_R22v1_TruthJetPt"
"""The single regression target of the GLS fixture's regression head."""


def build_gls_modules(
    norm_dict: Path | str,
    weights: dict[str, float] | None = None,
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """A TWO-task encoder-less DiPS plan ending in `LossGLS` (>= 2 task losses).

    ``norm -> track_embed -> concat -> pool[seq.x] -> {jets_classification,
    jets_regression} -> LossGLS`` — the encoder-less shape every GN3/regression
    config uses (init_nets + pool_net, no encoder; v1 saltmodel.py:90-93,155-156),
    with TWO loss producers so the geometric mean ``(L_cls * L_reg)^(1/2)`` is
    genuinely combined (a one-task GLS is the identity and would not excite the
    combination math). All task weights default to 1.0 — the GLS validity domain
    (v1 modelwrapper.py:139-142). ``weights`` overrides per-task ``weight`` to
    exercise the all-weights==1.0 guard's negative path.

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules with `LossGLS` already narrowed.
    """
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
    """Dataset-boundary source spec for the GLS plan (GN2 inputs + both labels).

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    # dips_sources already carries GN2 inputs + the flavour label; add the
    # regression target label leaf (TRAINING-gated, like the flavour one).
    merged = dict(flatten_spec(dips_sources()))
    merged[f"labels.jets.{GLS_TARGET}"] = TensorSpec(
        shape=("B",), dtype="float32", kind="label", modes=Mode.TRAINING
    )
    return unflatten_spec(merged)


def compile_gls(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the encoder-less two-task GLS plan for one mode.

    Returns
    -------
    Plan
        FIT/VAL anchored on ``loss.total``; TEST/ONNX on the two prediction
        leaves.
    """
    sinks = (
        ["loss.total"]
        if mode & Mode.TRAINING
        else ["preds.jets.jets_classification", "preds.jets.jets_regression"]
    )
    return compile_plan(modules, mode, sources=gls_sources(), sinks=sinks)


def make_gls_labels(batch_size: int, seed: int = 5) -> dict[str, Tensor]:
    """Deterministic labels for the two-task GLS FIT plan (flavour + reg target).

    Returns
    -------
    dict[str, Tensor]
        ``{labels.jets.HadronConeExclTruthLabelID, labels.jets.<reg target>}``.
    """
    labels = make_dips_labels(batch_size, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    labels[f"labels.jets.{GLS_TARGET}"] = 1.0 + torch.rand(batch_size, generator=gen) * 3.0
    return labels


def build_independent_v1_gls_total_loss(norm_dir: Path | str):
    """A FRESH v1 ``ModelWrapper`` with ``loss_mode='GLS'``; return its bound ``total_loss``.

    The GLS analogue of `build_independent_v1_head`: the v2 `LossGLS.forward`
    and v1 ``ModelWrapper.total_loss`` (the ``loss_mode == 'GLS'`` branch,
    modelwrapper.py:194-196) are SEPARATE code paths. A genuine v1-vs-v2 parity
    check feeds BOTH the same per-task loss dict and compares — so the reference
    must be v1's ACTUAL method, called on a real, independently-constructed v1
    object, not a re-implementation of the formula in the gate. This builds a
    small v1 GN2 ``ModelWrapper``, flips ``loss_mode`` to GLS, and hands back its
    bound ``total_loss`` so the gate drives v1's own ``math.prod`` + ``torch.pow``.

    ``ModelWrapper.total_loss`` reads only ``self.loss_mode`` and the per-task
    loss dict it is handed (modelwrapper.py:181-197) — it does NOT touch task
    weights (those are applied INSIDE each task before the dict is built). So
    flipping ``loss_mode`` post-construction yields v1's exact GLS reduction
    regardless of the fixture's task weights; the all-weights==1.0 *guard* is the
    separate ctor-time concern the L1 gate checks directly via
    `LossGLS.check_task_weights`.

    Returns
    -------
    Callable[[dict[str, torch.Tensor]], torch.Tensor]
        The v1 ``wrapper.total_loss`` bound method (GLS mode).
    """
    wrapper = build_test_gn2(norm_dir)
    wrapper.loss_mode = "GLS"
    return wrapper.total_loss


# ---------------------------------------------------------------------------
# norm_type: hybrid encoder block (M5 sub-wave B, L2 gate)
# ---------------------------------------------------------------------------

HYBRID_ENC_DIM = 16
"""Encoder embed/out width of the hybrid-norm fixture (matches the gn2v2 width)."""

HYBRID_NUM_LAYERS = 3
"""Encoder depth of the hybrid fixture. >= 2 so the v1 hybrid depth-0 vs depth>0
residual-norm split (transformer.py:353-356: ``"pre"`` + Identity at depth 0,
``"none"`` + a real norm after) is genuinely exercised across layers."""

HYBRID_NUM_HEADS = 2
"""Attention heads of the hybrid fixture."""


def build_hybrid_encoder_modules(
    norm_dict: Path | str,
    norm_type: str = "hybrid",
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """A GN3V01-style ENCODER block: norm -> embed -> concat -> encoder -> split -> pool -> head.

    The GN3V01 flagship's encoder is ``norm_type: hybrid`` (plan 10 sub-wave B);
    this fixture wires a real `TransformerEncoder` with ``norm_type`` set so the
    L2 gate drives the hybrid passthrough end-to-end through the compiler / bind /
    executor (the encoder pre/post-norm placement, forced ``do_qk_norm``/
    ``do_v_norm``, depth-0 residual handling all live in the composed v1
    `EncoderLayer`, transformer.py:350-356,421). The head is a single jet
    classifier (the minimal valid tail) — the gate's parity claim is about the
    ENCODER output ``encoded.seq``, not the head, so one head keeps the plan
    small while still requiring a downstream consumer.

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules with `LossSum` already narrowed.
    """
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
    """Compile the hybrid-encoder plan for one mode (GN2 inputs + flavour label).

    Returns
    -------
    Plan
        FIT/VAL anchored on ``loss.total``; TEST/ONNX on the prediction leaf.
    """
    sinks = ["loss.total"] if mode & Mode.TRAINING else ["preds.jets.jets_classification"]
    return compile_plan(modules, mode, sources=dips_sources(), sinks=sinks)


def build_independent_v1_transformer(composed: TransformerEncoder) -> V1Transformer:
    """A FRESH, separately-constructed v1 `Transformer` with ``composed``'s weights copied in.

    The encoder analogue of `build_independent_v1_head`: the v2
    `TransformerEncoder` COMPOSES a v1 `Transformer` and the executor invokes
    THAT internal object, so re-calling ``composed.encoder`` would only measure
    self-consistency. Here a brand-new v1 ``Transformer`` is built from scratch
    with the SAME construction kwargs (depth, widths, ``norm_type``, attention /
    dense kwargs read back off the composed instance), then ``load_state_dict``
    copies the trained weights in. Feeding THIS object the executor's
    ``seq.x``/``seq.mask`` reproduces v1 from an independent construction path —
    a genuine v1-vs-v2 cross-impl parity reference for the hybrid passthrough.

    The fresh ``Transformer`` is built with ``attn_type="torch-math"`` (the
    deterministic SDPBackend.MATH kernel, the v2 wrapper's default and what the
    fixture's attention config selects); ``num_registers``, the optional out
    projection, and the final norm all live inside both encoders, so the
    comparison is over the FULL encoder forward, registers included.

    Returns
    -------
    V1Transformer
        A fresh ``Transformer`` (``.eval()``) with ``composed``'s weights, ready
        to drive on a ``{"seq": x}`` / ``{"seq": mask}`` dict pair.
    """
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


# ---------------------------------------------------------------------------
# VectorConcat + export.inputs alias (M5 sub-wave B, L3 gate) — GN3 'global'
# stream past the encoder (design §6.6; v1 saltmodel.py:175-177)
# ---------------------------------------------------------------------------

VCONCAT_DIM = 16
"""Encoder embed/out/pool width of the VectorConcat fixture (matches gn2v2)."""

GLOBAL_VARIABLES = list(JET_VARIABLES)
"""The GN3V01 ``global`` stream columns. GN3V01's exact case: ``global`` and
``jets`` BOTH declare ``[pt_btagJes, eta_btagJes]`` (design §6.6 1400-1401) — the
ONNX alias IDENTITY case (same Features -> identity clone, no column gather)."""


def write_vector_concat_norm_dict(nd_path: Path, cd_path: Path) -> None:
    """Write the parity norm/class dicts PLUS a ``global`` stream (= jets columns).

    The GN3V01 ``global`` stream is an ordinary 2-feature vector stream
    normalised on demand like any other (design §6.6 1393-1396); the Normaliser
    needs mean/std entries for it. `write_parity_norm_dict` covers jets/tracks/
    electrons with distinct per-variable constants; this augments the dict with
    a ``global`` block whose entries DIFFER from the jets block (distinct
    constants so a global/jets norm-wiring swap would be excitable, not silently
    identical).

    Parameters
    ----------
    nd_path : Path
        Output path for the normalisation dictionary YAML.
    cd_path : Path
        Output path for the class dictionary YAML.
    """
    write_parity_norm_dict(nd_path, cd_path)
    with open(nd_path) as fh:
        sd = yaml.safe_load(fh)
    # distinct constants from the jets block (jets used 0.1*(i+1) / 1.0+0.05*(i+1))
    sd["global"] = {
        v: {"mean": round(0.3 * (i + 1), 6), "std": round(1.0 + 0.07 * (i + 1), 6)}
        for i, v in enumerate(GLOBAL_VARIABLES)
    }
    with open(nd_path, "w") as fh:
        yaml.dump(sd, fh, sort_keys=False)


def build_vector_concat_modules(norm_dict: Path | str) -> dict[str, GraphModule]:
    """A GN3V01-style block whose pooled rep is concatenated with the ``global`` stream.

    ``{norm[jets,tracks], norm_global[global]} -> track_embed -> concat ->
    encoder -> split -> pool[encoded.seq] -> VectorConcat[pooled.global,
    normed.global] -> head`` — the design §6.6 GN3 flagship shape: the
    2-feature ``global`` stream is normalised but fed PAST the encoder, then
    concatenated onto the pooled representation (v1 saltmodel.py:175-177
    ``cat([global_rep, global_feats], dim=-1)``). The `VectorConcat` produces the
    augmented ``vconcat.global`` (256+2 -> 18 here) that the head consumes —
    exactly v1's ``&pooled_dim 258`` slot.

    The ``global`` stream is normalised by a SECOND `Normaliser` with
    ``global_object="global"`` (a ``[B, F]`` vector), because one Normaliser has
    a single ``global_object`` slot and the ``jets`` slot is already the encoder
    stream's global context. Both read the same augmented norm dict.

    The concat ORDER is ``[pooled.global, normed.global]`` (pooled FIRST, global
    features LAST — design §6.6 1390-1391), so the head first-layer weight layout
    lines up with v1's ``cat([global_rep, global_feats])``. The pool writes
    ``pooled.global`` (the v1 ``global_rep``); the VectorConcat output is a
    DISTINCT key ``vconcat.global`` so it never collides with a pool output.

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules with `LossSum` already narrowed.
    """
    dense = {"hidden_layers": [VCONCAT_DIM], "activation": "ReLU"}
    head_dense = {"hidden_layers": [VCONCAT_DIM], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["jets", "tracks"], global_object="jets"),
        "norm_global": Normaliser(norm_dict=norm_dict, streams=["global"], global_object="global"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=VCONCAT_DIM, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=VCONCAT_DIM,
            num_layers=2,
            out_dim=VCONCAT_DIM,
            attention={"num_heads": 2, "attn_type": "torch-math"},
            dense={"activation": "ReLU", "gated": False},
        ),
        "split": Split(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        # the design §6.6 module: pooled FIRST, global features LAST
        "vconcat": VectorConcat(inputs=["pooled.global", "normed.global"], out="vconcat.global"),
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="HadronConeExclTruthLabelID",
            class_names=list(DIPS_CLASS_NAMES),
            label_map={0: 0, 4: 1, 5: 2},
            input="vconcat.global",
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


def vector_concat_sources() -> NestedSpec:
    """Dataset-boundary source spec: GN2 inputs + the ``global`` vector + flavour label.

    The ``global`` stream is a ``[B, F]`` vector (no pad mask — v1's mask
    exclusion of ``'global'``, datasets.py:519-524) carrying `GLOBAL_VARIABLES`
    as `fields` (the bind-time column source for the ONNX alias gather).

    Returns
    -------
    NestedSpec
        The nested source spec for `compile_plan`.
    """
    base = dict(flatten_spec(dips_sources()))
    base["inputs.global"] = TensorSpec(
        shape=("B", len(GLOBAL_VARIABLES)), dtype="float32", fields=tuple(GLOBAL_VARIABLES)
    )
    return unflatten_spec(base)


def compile_vector_concat(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the VectorConcat (GN3 ``global``) plan for one mode.

    Returns
    -------
    Plan
        FIT/VAL anchored on ``loss.total``; TEST on the prediction leaf.
    """
    sinks = ["loss.total"] if mode & Mode.TRAINING else ["preds.jets.jets_classification"]
    return compile_plan(modules, mode, sources=vector_concat_sources(), sinks=sinks)


def compile_vector_concat_onnx(
    modules: dict[str, GraphModule],
    *,
    alias_variables: list[str] | None = None,
):
    """Compile the Mode.ONNX plan + resolved export config for the alias path.

    Athena feeds ONE jet tensor that v1 clones into ``global``
    (to_onnx.py:377-378); v2 reproduces it with ``export.inputs`` ``alias:`` —
    the ``inputs.global`` port is bound from ``inputs.jets`` at trace time
    (`OnnxAdapter`, design §6.6/§7). When ``alias_variables`` equals the jets
    columns the gather is the IDENTITY clone (GN3V01); when it is a renamed/
    re-ordered subset the gather resolves by NAME (the GN2emu-style case).

    Returns
    -------
    tuple[Plan, ExportConfig, dict[str, tuple[str, ...]]]
        ``(onnx_plan, resolved_export, feature_fields)`` — ready for
        `OnnxAdapter`.
    """
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
    # W4: a folded ClassProbs node named by an OnnxExportSink (the off-graph reduce
    # manifest is retired)
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
    """The v1 reference for the post-pooling global concat (saltmodel.py:175-177).

    v1 has NO VectorConcat class — it inlines ``global_rep = torch.cat(
    [global_rep, global_feats], dim=-1)`` in ``SaltModel.forward``
    (saltmodel.py:175-177). The L3 design-conformance anchor is the ORDER
    (pooled first, global features last) and the width (Dsum = sum). This helper
    reproduces v1's exact expression from the SAME pooled rep + global features
    the v2 plan used, so the gate compares the v2 module output against the
    literal v1 line, not a re-statement of the module's own formula.

    Returns
    -------
    Tensor
        ``cat([pooled, global_feats], dim=-1)`` — the v1 ``global_rep``.
    """
    return torch.cat([pooled, global_feats], dim=-1)


# ---------------------------------------------------------------------------
# MaskDecoder + encoder drop_registers (M5 sub-wave C, MF1a gate) — the
# MaskFormer object decoder over a register-dropping encoder (FD 1121-1141)
# ---------------------------------------------------------------------------

MASKFORMER_ENC_DIM = 16
"""Encoder/decoder embed width of the MaskFormer fixture (matches the gn2v2 width)."""

MASKFORMER_NUM_OBJECTS = 5
"""Object queries M of the MaskFormer fixture (shipped MaskFormer.yaml:36)."""

MASKFORMER_NUM_REGISTERS = 4
""">1 register so the drop slice ``x[:, :-num_registers]`` is genuinely exercised
(a single register would still slice, but >1 makes an off-by-one slice excitable)."""

MASKFORMER_NUM_OBJECT_CLASSES = 3
"""Object class count C (b, c, null — null LAST), shipped MaskFormer.yaml:49."""


def build_maskformer_decoder_modules(
    norm_dict: Path | str,
    num_layers: int = 3,
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """A MaskFormer encoder->decoder block: norm -> embed -> concat -> encoder(drop) -> decoder.

    The shipped MaskFormer.yaml shape for the decoder slice (MaskFormer.yaml:14-56):
    the track embedding feeds the `Concat`, the `TransformerEncoder` runs WITH
    ``drop_registers=True`` (registers visible to attention, stripped from
    ``encoded.seq`` — MaskFormer.yaml:31), and the `MaskDecoder` reads the
    register-free ``encoded.seq`` to produce the five ``objects.*`` keys. The
    object-regression / matched-loss / writer tail is OTHER sub-wave-C modules; this
    fixture stops at the decoder so MF1a gates exactly the decoder + drop_registers
    passthrough.

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules (no loss — the decoder produces no ``losses.*``; the
        plan sinks are the ``objects.*`` keys).
    """
    dense = {"hidden_layers": [MASKFORMER_ENC_DIM], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=list(streams), global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=MASKFORMER_ENC_DIM, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=MASKFORMER_ENC_DIM,
            num_layers=2,
            out_dim=MASKFORMER_ENC_DIM,
            attention={"num_heads": 2, "attn_type": "torch-math"},
            dense={"activation": "ReLU", "gated": False},
            num_registers=MASKFORMER_NUM_REGISTERS,
            drop_registers=True,
        ),
        "mask_decoder": MaskDecoder(
            embed_dim=MASKFORMER_ENC_DIM,
            num_objects=MASKFORMER_NUM_OBJECTS,
            num_layers=num_layers,
            class_net={"output_size": MASKFORMER_NUM_OBJECT_CLASSES},
            md={"n_heads": 2, "mask_attention": True, "bidirectional_ca": True},
        ),
    }
    for name, module in modules.items():
        module.name = name
    return modules


def compile_maskformer_decoder(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the MaskFormer encoder->decoder plan for one mode.

    The decoder produces in EVERY mode and owns no loss, so the sinks are the four
    ``objects.*`` keys directly (no ``loss.total`` anchor — there is no loss module
    in this decoder-only fixture).

    Returns
    -------
    Plan
        Anchored on the four ``objects.*`` decoder products.
    """
    sinks = [
        "objects.embed",
        "objects.class_logits",
        "objects.class_probs",
        "objects.masks",
    ]
    return compile_plan(modules, mode, sources=dips_sources(), sinks=sinks)


def build_independent_v1_transformer_drop(composed: TransformerEncoder) -> V1Transformer:
    """A FRESH v1 `Transformer` (``drop_registers=True``) with ``composed``'s weights.

    The drop-registers analogue of `build_independent_v1_transformer`: the v2
    `TransformerEncoder` COMPOSES a v1 `Transformer` (here built with
    ``drop_registers=True``), so re-calling ``composed.encoder`` measures only
    self-consistency. This builds a brand-new v1 ``Transformer`` from scratch with the
    SAME kwargs (depth, widths, ``num_registers``, ``drop_registers=True``, attn/dense
    kwargs read back off the composed instance) and ``load_state_dict``-copies the
    trained weights — a genuine independent v1 reference for the register-drop passthrough.

    Returns
    -------
    V1Transformer
        A fresh ``Transformer`` (``.eval()``, ``drop_registers=True``) with ``composed``'s
        weights, ready to drive on a ``{"seq": x}`` / ``{"seq": mask}`` dict pair.
    """
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
    """A FRESH v1 `MaskDecoder` with ``composed``'s weights copied in.

    The decoder analogue of `build_independent_v1_head`: the v2 `MaskDecoder` composes
    v1 `MaskDecoderLayer`s + v1 `Dense` heads and the executor invokes THOSE objects, so
    re-calling them only measures self-consistency. This builds a brand-new v1
    ``salt.models.maskformer.MaskDecoder`` from scratch with the SAME architecture
    (embed_dim, num_objects, num_layers, md_config, and fresh ``class_net``/``mask_net``
    `Dense`s read off the composed heads' geometry) and ``load_state_dict``-copies the
    trained weights — then driving its INFERENCE path (``forward`` with ``labels=None``,
    maskformer.py:166-203) reproduces v1 from an independent construction path. Comparing
    its ``preds["objects"]`` against the v2 executor's ``objects.*`` is a genuine
    v1-vs-v2 cross-impl check.

    The v1 decoder needs a ``loss_config`` to construct its (unused, never-invoked in the
    ``labels=None`` path) `MaskFormerLoss`; a minimal ``num_classes`` matching the v2
    decoder is supplied. ``aux_loss`` stays False (the v2 decoder never implements it).

    Returns
    -------
    V1MaskDecoder
        A fresh v1 ``MaskDecoder`` (``.eval()``) with ``composed``'s weights, ready to
        drive via ``decoder({"embed_xs": encoded}, tasks=[], pad_mask, labels=None)``.
    """
    class_net = composed.class_net
    mask_net = composed.mask_net
    # rebuild fresh Dense heads with the SAME geometry (Dense.node_list packs
    # input_size+context_size, hidden layers, output_size; context_size is 0 here)
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
        # the v1 ctor builds a MaskFormerLoss whose HungarianMatcher asserts the
        # loss-weight sum is positive (matcher.py:167) — supply a dummy positive
        # weight; the loss is NEVER invoked (forward runs with labels=None).
        loss_config={
            "num_classes": composed.num_classes,
            "loss_weights": {"object_class_ce": 1.0},
        },
        aux_loss=False,
    )
    # the v1 decoder's own state_dict carries the mask_loss buffers (empty_weight);
    # load only the structural weights the v2 decoder owns (queries, norms, layers,
    # heads), which is exactly composed.state_dict() — strict=False ignores the v1
    # mask_loss.* keys with no v2 counterpart.
    fresh.load_state_dict(composed.state_dict(), strict=False)
    fresh.eval()
    return fresh


# ---------------------------------------------------------------------------
# MaskFormerMatchedLoss + HungarianMatcher (M5 sub-wave C) — the matched loss
# over the decoder's object predictions (FD 1158-1170)
# ---------------------------------------------------------------------------

MASKFORMER_NUM_REG_TARGETS = 3
"""Object-regression target count R of the matched-loss fixture (a small stand-in
for the shipped 5: pt/Lxy/deta/dphi/mass, MaskFormer.yaml:89)."""

# the shipped MaskFormer.yaml loss weights (MaskFormer.yaml:59-63) augmented with a
# mask_focal term so BOTH mask cost paths (dice + focal) are exercised in the matcher.
MASKFORMER_LOSS_WEIGHTS = {
    "object_class_ce": 2.0,
    "mask_dice": 2.0,
    "mask_focal": 5.0,
    "regression": 2.0,
}
"""Loss/matcher weights for the matched-loss fixture (all four components active)."""

MASKFORMER_NULL_CLASS_WEIGHT = 0.5
"""Null-class CE balance weight (shipped default, maskformer_loss.py:130)."""


def build_matched_loss_module(
    num_classes: int = MASKFORMER_NUM_OBJECT_CLASSES - 1,
    num_objects: int = MASKFORMER_NUM_OBJECTS,
    loss_weights: dict[str, float] | None = None,
) -> MaskFormerMatchedLoss:
    """A bound `MaskFormerMatchedLoss` over the object stream, ready to drive.

    The matched loss has no schema-derived submodules (the matcher + ``empty_weight``
    buffer are width-free, the only bind check is the regression pred/target width
    agreement), so it is constructed + named here and driven directly with a `Bundle`
    in the gate — like the loss-combination fixtures. ``num_objects`` is the query
    count M (== the truth-object slot count), ``num_classes`` the non-null class count.

    Returns
    -------
    MaskFormerMatchedLoss
        The named module (``.eval()``), all four components active.
    """
    module = MaskFormerMatchedLoss(
        num_classes=num_classes,
        num_objects=num_objects,
        loss_weights=dict(loss_weights or MASKFORMER_LOSS_WEIGHTS),
        null_class_weight=MASKFORMER_NULL_CLASS_WEIGHT,
    )
    module.name = "mf_matched_loss"
    module.eval()
    return module


def build_independent_v1_matched_loss(
    composed: MaskFormerMatchedLoss,
) -> V1MaskFormerLoss:
    """A FRESH, separately-constructed v1 `MaskFormerLoss` for the matched-loss parity.

    The matched-loss analogue of `build_independent_v1_head`: the v2
    `MaskFormerMatchedLoss` COMPOSES a v1 ``MaskFormerLoss`` (its ``v1_loss``) and
    drives that instance's matcher + loss methods, so re-using ``composed.v1_loss``
    would only measure self-consistency. This builds a brand-new v1 ``MaskFormerLoss``
    from the SAME config (num_classes, num_objects, loss_weights, matcher_weights,
    null_class_weight) — its own ``HungarianMatcher``, its own ``empty_weight`` buffer
    — so driving ITS matcher + ``loss_labels``/``loss_masks`` reproduces v1 from an
    independent construction path. The ``empty_weight`` buffer is a deterministic
    function of config (maskformer_loss.py:135-142), so the fresh instance's buffer is
    bitwise-identical without a state copy; the matcher carries no learnable weights.

    Returns
    -------
    V1MaskFormerLoss
        A fresh v1 ``MaskFormerLoss`` (``.eval()``) with the same config — its matcher
        + loss methods are the independent reference the gate drives.
    """
    fresh = V1MaskFormerLoss(
        num_classes=composed.num_classes,
        num_objects=composed.num_objects,
        loss_weights=dict(composed.loss_weights),
        matcher_weights=dict(composed.matcher_weights),
        null_class_weight=composed.null_class_weight,
    )
    # the only state is the empty_weight buffer (config-deterministic); copy it
    # anyway so a future learnable term would still be matched, not silently skipped.
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
    """Deterministic object predictions + truth labels for the matched-loss gate.

    Produces the decoder-style object predictions (``class_logits``/``class_probs``/
    ``masks``/``embed``), the scaled object-regression prediction + target
    (``preds.objects.regression`` / ``targets.objects.regression``), and the truth
    labels (``labels.objects.{object_class,masks}``). The truth ``object_class`` mixes
    valid (< num_classes) and null (== num_classes) slots so the matcher's
    valid-object handling (matcher.py:209-246) and the matched-regression validity
    mask are genuinely exercised; the per-batch valid count VARIES so the LSAP runs on
    a ragged cost matrix.

    Returns
    -------
    dict[str, Tensor]
        Flat bundle keys for the matched-loss forward + the v1 reference call.
    """
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
    # treats the first batch_obj_lengths columns as valid, matcher.py:243-246).
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


# ---------------------------------------------------------------------------
# MaskFormerTargets processor (M5 sub-wave C) — object class + truth masks
# (FD 1090-1110; v1 datasets.py:549-553,636-644)
# ---------------------------------------------------------------------------

# the shipped MaskFormer.yaml object class map (MaskFormer.yaml:169-178): b->0, c->1,
# null->2 (null LAST). Raw ids 5/4/-1 (HadronConeExclTruthLabel-style flavour codes).
MASKFORMER_OBJECT_CLASS_MAP = {
    "b": {"raw": 5, "mapped": 0},
    "c": {"raw": 4, "mapped": 1},
    "null": {"raw": -1, "mapped": 2},
}
"""The shipped object class map (b/c/null, null LAST; MaskFormer.yaml:169-178)."""


def build_maskformer_targets(
    object_stream: str = "truth_hadrons",
    constituent_stream: str = "tracks",
    regression_targets: Sequence[str] = ("pt", "Lxy", "mass"),
    num_objects: int | None = MASKFORMER_NUM_OBJECTS,
) -> MaskFormerTargets:
    """A `MaskFormerTargets` processor matching the shipped MaskFormer.yaml mf_config.

    Mirrors MaskFormer.yaml:164-181: object group ``truth_hadrons`` keyed by
    ``barcode`` / ``flavour``, constituent group ``tracks`` keyed by
    ``ftagTruthParentBarcode``, the b/c/null class map (null LAST), and a small set of
    per-object regression labels.

    Returns
    -------
    MaskFormerTargets
        The named processor.
    """
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
    """A raw numpy batch for the `MaskFormerTargets` processor (structured arrays).

    Builds ``raw.<object_stream>`` (structured: ``barcode``, ``flavour``, regression
    fields) and ``raw.<constituent_stream>`` (structured: ``ftagTruthParentBarcode``)
    so the processor's ``process`` reads its declared raw fields. The barcodes are
    constructed so SOME tracks share an object's barcode (a non-empty truth mask) and
    some objects are invalid (barcode -1, flavour -1 -> null), exercising the
    sentinel-substitution path WITHOUT the v1 in-place mutation.

    Returns
    -------
    dict[str, np.ndarray]
        ``{raw.<object_stream>: structured [B, M], raw.<constituent_stream>:
        structured [B, T]}`` — flat bundle keys for the processor batch.
    """
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


# ---------------------------------------------------------------------------
# MaskFormerObjectWriter fixtures (M5 sub-wave C) — the writer's TEST byte
# parity + the ONNX leading_object / object_index reduce export (MF2 gate)
# ---------------------------------------------------------------------------

MASKFORMER_WRITER_REG_TARGETS = ("pt", "Lxy", "mass")
"""The object-regression targets of the writer fixture (the leading-object ONNX
scalar names + the MaskFormerTargets regression labels)."""


class _ObjectRegressionStub(torch.nn.Module):
    """A minimal per-object regression head: ``objects.embed`` -> ``preds.objects.regression``.

    The MF2 gate / writer tests need an object-regression task ONLY so the
    ``leading_object`` ONNX reduce has its ``preds.objects.regression`` port and
    the writer can derive its leading-object suffixes from ``output_suffixes``.
    The full `RegressionTaskModule` on the object stream is a sub-wave-A concern
    with its own (sequence-mask) wiring; this stub isolates the MF2 deliverable —
    the two MaskFormer object reduces' dtypes in a real ``Mode.ONNX`` plan — from
    that. It publishes DE-SCALED ``[B, M, R]`` predictions in ALL modes (a plain
    linear projection, no scaling), the contract the ``leading_object`` reduce
    consumes (the v2 RegressionTaskModule de-scales in ONNX, tasks.py:1063-1067).
    """

    def __init__(self, targets: Sequence[str] = MASKFORMER_WRITER_REG_TARGETS) -> None:
        super().__init__()
        self.name = "regression"
        self.stream = "objects"
        self.pred_key = "preds.objects.regression"
        self.num_objects = MASKFORMER_NUM_OBJECTS
        self.output_suffixes = tuple(targets)
        self.proj = torch.nn.Linear(MASKFORMER_ENC_DIM, len(self.output_suffixes))

    def declare_io(self, mode: Mode):
        del mode
        r = len(self.output_suffixes)
        return self._io(
            {"objects.embed": TensorSpec(shape=("B", self.num_objects, MASKFORMER_ENC_DIM))},
            {self.pred_key: TensorSpec(shape=("B", self.num_objects, r), dtype="float32")},
        )

    @staticmethod
    def _io(requires: dict[str, TensorSpec], produces: dict[str, TensorSpec]) -> IO:
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def bind(self, schema) -> None:  # noqa: ARG002 - width is fixed at construction
        return None

    def forward(self, b, mode):  # noqa: ARG002 - mode-agnostic stub
        return {self.pred_key: self.proj(b.get("objects.embed"))}


def build_maskformer_writer_modules(
    norm_dict: Path | str,
    targets: Sequence[str] = MASKFORMER_WRITER_REG_TARGETS,
) -> dict[str, GraphModule]:
    """A MaskFormer decoder block + an object-regression stub, for the writer/MF2 gate.

    Extends `build_maskformer_decoder_modules` (norm -> embed -> concat ->
    encoder(drop) -> decoder) with a per-object regression stub producing
    ``preds.objects.regression`` — the module set the `MaskFormerObjectWriter`
    declares its ONNX manifest against (``leading_object`` reads the regression
    port, ``object_index`` reads the decoder ``objects.masks``).

    Returns
    -------
    dict[str, GraphModule]
        Instance-named modules including ``regression`` (the object stub).
    """
    modules = build_maskformer_decoder_modules(norm_dict)
    modules["regression"] = _ObjectRegressionStub(targets)
    return modules


def make_maskformer_writer_batch(
    batch_size: int = 6,
    num_objects: int = MASKFORMER_NUM_OBJECTS,
    num_classes: int = MASKFORMER_NUM_OBJECT_CLASSES,
    n_tracks: int = 10,
    n_reg: int = len(MASKFORMER_WRITER_REG_TARGETS),
    seed: int = 31,
) -> dict[str, Tensor]:
    """A TEST bundle for the `MaskFormerObjectWriter`: decoder preds + truth labels + pad.

    Builds the four object-prediction keys the writer consumes
    (``objects.{class_probs,masks}``, ``preds.objects.regression``) plus the
    truth labels (``labels.objects.{object_class,masks}``) and the constituent
    pad mask (``masks.tracks``) — the inputs the writer's TEST ``write`` reads.
    ``object_class`` mixes valid + null slots and the pad mask marks some
    constituents padded, so the ``MaskIndex`` ``-1`` (padded) / ``-2`` (no
    object) encodings are both exercised.

    Returns
    -------
    dict[str, Tensor]
        Flat bundle keys for the writer's TEST forward.
    """
    gen = torch.Generator().manual_seed(seed)
    class_probs = torch.randn(batch_size, num_objects, num_classes, generator=gen).softmax(-1)
    masks = torch.randn(batch_size, num_objects, n_tracks, generator=gen)
    reg = torch.randn(batch_size, num_objects, n_reg, generator=gen)
    object_class = torch.randint(0, num_classes, (batch_size, num_objects), generator=gen)
    target_masks = torch.rand(batch_size, num_objects, n_tracks, generator=gen) > 0.5
    pad = torch.zeros(batch_size, n_tracks, dtype=torch.bool)
    pad[:, -3:] = True  # last 3 constituents padded (MaskIndex -> -1 there)
    return {
        "objects.class_probs": class_probs,
        "objects.masks": masks,
        "preds.objects.regression": reg,
        "labels.objects.object_class": object_class,
        "labels.objects.masks": target_masks,
        "masks.tracks": pad,
    }
