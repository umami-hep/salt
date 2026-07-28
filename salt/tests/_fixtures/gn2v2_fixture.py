"""Small config-constructed GN2v2 builder for the nn-module tests.

This module owns the (v1-free) GN2 fixture constants and helpers — the
variable lists, the parity norm-dict writer and the deterministic batch
builder — so the tests never import a v1 fixture.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import Tensor

from salt.graph.planner import Plan, compile_plan
from salt.graph.spec import GraphModule, Mode, NestedSpec, TensorSpec, unflatten_spec
from salt.model.modules import (
    Concat,
    GlobalAttentionPooling,
    LossSum,
    Normaliser,
    Split,
    StreamEmbed,
    TransformerEncoder,
)
from salt.model.modules.tasks import ClassificationTaskModule, VertexingTaskModule

__all__ = [
    "ELECTRON_VARIABLES",
    "FIT_SINKS",
    "JET_VARIABLES",
    "ORIGIN_CLASSES",
    "TEST_SINKS",
    "TRACK_VARIABLES",
    "build_gn2v2_modules",
    "compile_gn2v2",
    "gn2v2_sources",
    "make_gn2_batch",
    "make_gn2_labels",
    "write_parity_norm_dict",
]

JET_VARIABLES = ["pt_btagJes", "eta_btagJes"]  # GN2.yaml:90-92

TRACK_VARIABLES = [  # the 19 GN2 track variables, GN2.yaml:93-112
    "d0",
    "z0SinTheta",
    "dphi",
    "deta",
    "qOverP",
    "IP3D_signed_d0_significance",
    "IP3D_signed_z0_significance",
    "phiUncertainty",
    "thetaUncertainty",
    "qOverPUncertainty",
    "numberOfPixelHits",
    "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits",
    "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits",
    "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits",
    "numberOfPixelSplitHits",
    "numberOfSCTSharedHits",
]

ELECTRON_VARIABLES = [  # GN2e-style second sequence stream (subset of the ELECTRON_VARS list)
    "pt",
    "ptfrac",
    "ptrel",
    "dr",
    "abs_eta",
]


def write_parity_norm_dict(nd_path: Path, cd_path: Path) -> None:
    """Write the parity norm/class dicts with DISTINCT per-variable constants."""
    sd = {
        stream: {
            v: {"mean": round(0.1 * (i + 1), 6), "std": round(1.0 + 0.05 * (i + 1), 6)}
            for i, v in enumerate(variables)
        }
        for stream, variables in (
            ("jets", JET_VARIABLES),
            ("tracks", TRACK_VARIABLES),
            ("electrons", ELECTRON_VARIABLES),
        )
    }
    with open(nd_path, "w") as file:
        yaml.dump(sd, file, sort_keys=False)

    cd = {
        "jets": {
            "HadronConeExclTruthLabelID": [1.0, 2.0, 2.0, 2.0],
            "flavour_label": [1.0, 2.0, 2.0, 2.0],
        },
        "tracks": {"ftagTruthOriginLabel": [4.2, 73.7, 1.0, 17.5, 12.3, 12.5, 141.7, 22.3]},
    }
    with open(cd_path, "w") as file:
        yaml.dump(cd, file, sort_keys=False)


def make_gn2_batch(
    batch_size: int = 6,
    n_tracks: int = 10,
    p_valid: float = 0.6,
    seed: int = 123,
    n_electrons: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build a deterministic GN2 batch with real padding."""
    gen = torch.Generator().manual_seed(seed)
    jets = torch.randn(batch_size, len(JET_VARIABLES), generator=gen)
    tracks = torch.randn(batch_size, n_tracks, len(TRACK_VARIABLES), generator=gen)
    mask = torch.rand(batch_size, n_tracks, generator=gen) >= p_valid  # True = padded
    mask = torch.sort(mask.to(torch.uint8), dim=-1).values.bool()  # valid first, like v1 dumps
    mask[:, 0] = False  # >=1 valid track per jet by default
    mask[0, 1:] = True  # jet 0: exactly one valid track
    if batch_size >= 2:
        mask[1, :] = True  # jet 1: ZERO valid tracks (production edge case)
    tracks[mask] = 0.0  # padded positions zeroed (v1 datasets.py:524 semantics)
    inputs = {"jets": jets, "tracks": tracks}
    pad_masks = {"tracks": mask}
    if n_electrons > 0:
        electrons = torch.randn(batch_size, n_electrons, len(ELECTRON_VARIABLES), generator=gen)
        emask = torch.rand(batch_size, n_electrons, generator=gen) >= p_valid  # True = padded
        emask = torch.sort(emask.to(torch.uint8), dim=-1).values.bool()  # valid first
        if batch_size >= 3:
            emask[2, :] = True  # jet 2: ZERO electrons (typical for real jets)
        electrons[emask] = 0.0  # padded positions zeroed
        inputs["electrons"] = electrons
        pad_masks["electrons"] = emask
    return inputs, pad_masks


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
    """Build the GN2v2 module dict from plain config kwargs."""
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
