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

from pathlib import Path

import torch
import yaml
from torch import Tensor

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
    ExportOutput,
    attach_manifest,
    compile_onnx_plan,
    resolve_export_config,
)
from salt.models import Transformer as V1Transformer
from salt.models.task import GaussianRegressionTask as V1GaussianRegressionTask
from salt.models.task import RegressionTask as V1RegressionTask
from salt.tests.core.gn2_fixture import (
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
    "POOL_DIM",
    "VCONCAT_DIM",
    "build_dips_modules",
    "build_gls_modules",
    "build_hybrid_encoder_modules",
    "build_independent_v1_global_concat",
    "build_independent_v1_gls_total_loss",
    "build_independent_v1_head",
    "build_independent_v1_transformer",
    "build_regression_modules",
    "build_vector_concat_modules",
    "compile_dips",
    "compile_gls",
    "compile_hybrid_encoder",
    "compile_vector_concat",
    "compile_vector_concat_onnx",
    "dips_sources",
    "gls_sources",
    "make_dips_labels",
    "make_gls_labels",
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
    fresh: V1RegressionTask | V1GaussianRegressionTask
    if isinstance(composed, V1GaussianRegressionTask):
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
        "norm_global": Normaliser(
            norm_dict=norm_dict, streams=["global"], global_object="global"
        ),
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
    manifest = [ExportOutput(port="preds.jets.jets_classification", names=list(DIPS_CLASS_NAMES))]
    resolved = attach_manifest(resolve_export_config(export, "GN3V01"), manifest)
    variables = {
        "jets": list(JET_VARIABLES),
        "tracks": list(TRACK_VARIABLES),
        "global": alias_cols,
    }
    plan = compile_onnx_plan(modules, resolved, variables)
    fields = {f"inputs.{s}": tuple(names) for s, names in variables.items()}
    return plan, resolved, fields


def build_independent_v1_global_concat(
    pooled: Tensor, global_feats: Tensor
) -> Tensor:
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
