"""Small config-constructed GN2v2 builder for the M2 nn-module tests (plan 05, stage A2)."""

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
    Split,
    StreamEmbed,
    TransformerEncoder,
)
from salt.core.nn.tasks import ClassificationTaskModule, VertexingTaskModule
from salt.tests._fixtures.gn2_fixture import JET_VARIABLES, TRACK_VARIABLES

__all__ = [
    "FIT_SINKS",
    "ORIGIN_CLASSES",
    "TEST_SINKS",
    "build_gn2v2_modules",
    "compile_gn2v2",
    "gn2v2_sources",
    "make_gn2_labels",
]

ORIGIN_CLASSES = [
    "Pileup",
    "Fake",
    "Primary",
    "FromB",
    "FromBC",
    "FromC",
    "FromTau",
    "OtherSecondary",
]

FIT_SINKS = ["loss.total"]
TEST_SINKS = [
    "preds.jets.jets_classification",
    "preds.tracks.track_origin",
    "preds.tracks.track_vertexing",
]


def build_gn2v2_modules(
    norm_dict: Path | str,
    embed_dim: int = 16,
    out_dim: int = 16,
    num_layers: int = 2,
    num_heads: int = 2,
    class_dict: Path | str | None = None,
) -> dict[str, GraphModule]:
    """Build the GN2v2 module dict from plain config kwargs (design §5.1 shape)."""
    dense = {"hidden_layers": [embed_dim], "activation": "ReLU"}
    head_dense = {"hidden_layers": [out_dim], "activation": "ReLU"}
    modules: dict[str, GraphModule] = {
        "norm": Normaliser(norm_dict=norm_dict, streams=["jets", "tracks"], global_object="jets"),
        "track_embed": StreamEmbed(
            stream="tracks", out_dim=embed_dim, dense=dense, context=["normed.jets"]
        ),
        "concat": Concat(streams=["tracks"]),
        "encoder": TransformerEncoder(
            dim=embed_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            attention={"num_heads": num_heads, "attn_type": "torch-math"},
            dense={"activation": "ReLU", "gated": False},
        ),
        "split": Split(streams=["tracks"]),
        "pool": GlobalAttentionPooling(input="encoded.seq", out="pooled.global"),
        # task order below mirrors the v1 fixture's model.tasks order — the
        # default index association in map_v1_state_dict.
        "jets_classification": ClassificationTaskModule(
            stream="jets",
            label="flavour_label",
            class_names=["bjets", "cjets", "ujets"],
            input="pooled.global",
            dense=head_dense,
        ),
        "track_origin": ClassificationTaskModule(
            stream="tracks",
            label="ftagTruthOriginLabel",
            class_names=list(ORIGIN_CLASSES),
            context="pooled.global",
            weight=0.5,
            dense=head_dense,
            weight_source=(
                {"from_class_dict": str(class_dict)} if class_dict is not None else None
            ),
        ),
        "track_vertexing": VertexingTaskModule(
            stream="tracks",
            label="ftagTruthVertexIndex",
            origin_label="ftagTruthOriginLabel",
            context="pooled.global",
            weight=1.5,
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


def gn2v2_sources() -> NestedSpec:
    """Build the dataset-boundary source spec for the GN2v2 plans."""
    label = {"dtype": "int64", "kind": "label", "modes": Mode.TRAINING}
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
        "labels.jets.flavour_label": TensorSpec(shape=("B",), **label),
        "labels.tracks.ftagTruthOriginLabel": TensorSpec(shape=("B", "T:tracks"), **label),
        "labels.tracks.ftagTruthVertexIndex": TensorSpec(shape=("B", "T:tracks"), **label),
    })


def compile_gn2v2(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the GN2v2 plan for one mode with the canonical sinks."""
    sinks = FIT_SINKS if mode & Mode.TRAINING else TEST_SINKS
    return compile_plan(modules, mode, sources=gn2v2_sources(), sinks=sinks)


def make_gn2_labels(batch_size: int, n_tracks: int, seed: int = 7) -> dict[str, dict[str, Tensor]]:
    """Build deterministic labels in the v1 ``labels_dict`` nesting."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "jets": {"flavour_label": torch.randint(0, 3, (batch_size,), generator=gen)},
        "tracks": {
            "ftagTruthOriginLabel": torch.randint(0, 8, (batch_size, n_tracks), generator=gen),
            "ftagTruthVertexIndex": torch.randint(-2, 4, (batch_size, n_tracks), generator=gen),
        },
    }
