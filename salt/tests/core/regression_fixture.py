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
from torch import Tensor

from salt.core.graph.planner import Plan, compile_plan
from salt.core.graph.spec import GraphModule, Mode, NestedSpec, TensorSpec, unflatten_spec
from salt.core.nn import (
    Concat,
    GlobalAttentionPooling,
    LossSum,
    Normaliser,
    StreamEmbed,
)
from salt.core.nn.tasks import ClassificationTaskModule, RegressionTaskModule
from salt.models.task import GaussianRegressionTask as V1GaussianRegressionTask
from salt.models.task import RegressionTask as V1RegressionTask
from salt.tests.core.gn2_fixture import JET_VARIABLES, TRACK_VARIABLES
from salt.utils.scalers import RegressionTargetScaler

__all__ = [
    "DIPS_CLASS_NAMES",
    "POOL_DIM",
    "build_dips_modules",
    "build_independent_v1_head",
    "build_regression_modules",
    "compile_dips",
    "dips_sources",
    "make_dips_labels",
    "make_regression_labels",
    "regression_sources",
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
