"""Pure-v2 graph-fixture builders (extracted from ``regression_fixture.py`` at DEL-1).

``regression_fixture.py`` mixes these v1-free builders with v1-comparison
machinery (``build_independent_v1_*``) and dies with the v1 tree. Every builder
here has live v2 consumers and imports NOTHING from the retired v1 namespace.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from torch import Tensor

from salt.graph.planner import Plan, compile_plan
from salt.graph.spec import (
    IO,
    GraphModule,
    Mode,
    NestedSpec,
    TensorSpec,
    flatten_spec,
    unflatten_spec,
)
from salt.model.modules import (
    Concat,
    GlobalAttentionPooling,
    LossSum,
    MaskDecoder,
    MaskFormerMatchedLoss,
    Normaliser,
    Split,
    StreamEmbed,
    TransformerEncoder,
    VectorConcat,
)
from salt.model.modules.tasks import ClassificationTaskModule, RegressionTaskModule
from salt.tests._fixtures.gn2v2_fixture import (
    JET_VARIABLES,
    TRACK_VARIABLES,
    write_parity_norm_dict,
)

__all__ = [
    "DIPS_CLASS_NAMES",
    "GLOBAL_VARIABLES",
    "MASKFORMER_ENC_DIM",
    "MASKFORMER_LOSS_WEIGHTS",
    "MASKFORMER_NULL_CLASS_WEIGHT",
    "MASKFORMER_NUM_OBJECTS",
    "MASKFORMER_NUM_OBJECT_CLASSES",
    "MASKFORMER_NUM_REGISTERS",
    "MASKFORMER_WRITER_REG_TARGETS",
    "POOL_DIM",
    "VCONCAT_DIM",
    "build_maskformer_decoder_modules",
    "build_maskformer_writer_modules",
    "build_matched_loss_module",
    "build_regression_modules",
    "build_vector_concat_modules",
    "compile_maskformer_decoder",
    "compile_regression",
    "compile_vector_concat",
    "dips_sources",
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
    """A DiPS-shaped module dict ending in one `RegressionTaskModule`."""
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
    """Dataset-boundary source spec: the GN2 inputs + the regression labels."""
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
        # FIT|VAL|TEST source for ratio denominators; ONNX reads them from inputs.jets instead.
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
    """Compile the encoder-less regression plan for one mode."""
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
    """Deterministic positive float labels for the targets + ratio denominators."""
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, Tensor] = {}
    label_stream = "tracks" if seq_len is not None else "jets"
    shape = (batch_size, seq_len) if seq_len is not None else (batch_size,)
    for i, target in enumerate(targets):
        out[f"labels.{label_stream}.{target}"] = 1.0 + torch.rand(*shape, generator=gen) * (i + 2)
    for denom in denominators:
        out[f"labels.jets.{denom}"] = 2.0 + torch.rand(batch_size, generator=gen)
    return out


def dips_sources() -> NestedSpec:
    """Dataset-boundary source spec for the dips-shaped plans (GN2 inputs + flavour label)."""
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


# VectorConcat + export.inputs alias — GN3 'global' stream past the encoder

VCONCAT_DIM = 16
"""Encoder embed/out/pool width of the VectorConcat fixture (matches gn2v2)."""

GLOBAL_VARIABLES = list(JET_VARIABLES)
"""The GN3V01 ``global`` stream columns. GN3V01's exact case: ``global`` and
``jets`` BOTH declare ``[pt_btagJes, eta_btagJes]`` — the ONNX alias IDENTITY
case (same Features -> identity clone, no column gather)."""


def write_vector_concat_norm_dict(nd_path: Path, cd_path: Path) -> None:
    """Write the parity norm/class dicts PLUS a ``global`` stream (= jets columns)."""
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
    """A GN3V01-style block whose pooled rep is concatenated with the ``global`` stream."""
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
        # pooled FIRST, global features LAST
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
    """Dataset-boundary source spec: GN2 inputs + the ``global`` vector + flavour label."""
    base = dict(flatten_spec(dips_sources()))
    base["inputs.global"] = TensorSpec(
        shape=("B", len(GLOBAL_VARIABLES)), dtype="float32", fields=tuple(GLOBAL_VARIABLES)
    )
    return unflatten_spec(base)


def compile_vector_concat(modules: dict[str, GraphModule], mode: Mode) -> Plan:
    """Compile the VectorConcat (GN3 ``global``) plan for one mode."""
    sinks = ["loss.total"] if mode & Mode.TRAINING else ["preds.jets.jets_classification"]
    return compile_plan(modules, mode, sources=vector_concat_sources(), sinks=sinks)


# MaskDecoder + encoder drop_registers — the MaskFormer object decoder over a
# register-dropping encoder

MASKFORMER_ENC_DIM = 16
"""Encoder/decoder embed width of the MaskFormer fixture (matches the gn2v2 width)."""

MASKFORMER_NUM_OBJECTS = 5
"""Object queries M of the MaskFormer fixture."""

MASKFORMER_NUM_REGISTERS = 4
""">1 register so the drop slice ``x[:, :-num_registers]`` is genuinely exercised
(a single register would still slice, but >1 makes an off-by-one slice excitable)."""

MASKFORMER_NUM_OBJECT_CLASSES = 3
"""Object class count C (b, c, null — null LAST)."""


def build_maskformer_decoder_modules(
    norm_dict: Path | str,
    num_layers: int = 3,
    streams: tuple[str, ...] = ("jets", "tracks"),
) -> dict[str, GraphModule]:
    """A MaskFormer encoder->decoder block: norm -> embed -> concat -> encoder(drop) -> decoder."""
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
    """Compile the MaskFormer encoder->decoder plan for one mode."""
    sinks = [
        "objects.embed",
        "objects.class_logits",
        "objects.class_probs",
        "objects.masks",
    ]
    return compile_plan(modules, mode, sources=dips_sources(), sinks=sinks)


# MaskFormerMatchedLoss + HungarianMatcher — the matched loss over the decoder's
# object predictions

# augmented with a mask_focal term so BOTH mask cost paths (dice + focal) are
# exercised in the matcher.
MASKFORMER_LOSS_WEIGHTS = {
    "object_class_ce": 2.0,
    "mask_dice": 2.0,
    "mask_focal": 5.0,
    "regression": 2.0,
}
"""Loss/matcher weights for the matched-loss fixture (all four components active)."""

MASKFORMER_NULL_CLASS_WEIGHT = 0.5
"""Null-class CE balance weight (shipped default)."""


def build_matched_loss_module(
    num_classes: int = MASKFORMER_NUM_OBJECT_CLASSES - 1,
    num_objects: int = MASKFORMER_NUM_OBJECTS,
    loss_weights: dict[str, float] | None = None,
) -> MaskFormerMatchedLoss:
    """A bound `MaskFormerMatchedLoss` over the object stream, ready to drive."""
    module = MaskFormerMatchedLoss(
        num_classes=num_classes,
        num_objects=num_objects,
        loss_weights=dict(loss_weights or MASKFORMER_LOSS_WEIGHTS),
        null_class_weight=MASKFORMER_NULL_CLASS_WEIGHT,
    )
    module.name = "mf_matched_loss"
    module.eval()
    return module


# MaskFormer object fixtures — the eval-H5 TEST byte parity + the ONNX
# leading_object / object_index reduce export

MASKFORMER_WRITER_REG_TARGETS = ("pt", "Lxy", "mass")
"""The object-regression targets of the writer fixture (the leading-object ONNX
scalar names + the MaskFormerTargets regression labels)."""


class _ObjectRegressionStub(torch.nn.Module):
    """A minimal per-object regression head: ``objects.embed`` -> ``preds.objects.regression``."""

    def __init__(self, targets: Sequence[str] = MASKFORMER_WRITER_REG_TARGETS) -> None:
        super().__init__()
        self.name = "regression"
        self.stream = "objects"
        self.pred_key = "preds.objects.regression"
        self.num_objects = MASKFORMER_NUM_OBJECTS
        self.targets = tuple(targets)
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
    """A MaskFormer decoder block + an object-regression stub, for the writer parity gate."""
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
    """A TEST bundle for the MaskFormer eval-H5: decoder preds + truth labels + pad."""
    gen = torch.Generator().manual_seed(seed)
    class_probs = torch.randn(batch_size, num_objects, num_classes, generator=gen).softmax(-1)
    masks = torch.randn(batch_size, num_objects, n_tracks, generator=gen)
    reg = torch.randn(batch_size, num_objects, n_reg, generator=gen)
    object_class = torch.randint(0, num_classes, (batch_size, num_objects), generator=gen)
    target_masks = torch.rand(batch_size, num_objects, n_tracks, generator=gen) > 0.5
    pad = torch.zeros(batch_size, n_tracks, dtype=torch.bool)
    pad[:, -3:] = True  # last 3 constituents padded (HadronIndex -> -1 there)
    return {
        "objects.class_probs": class_probs,
        "objects.masks": masks,
        "preds.objects.regression": reg,
        "labels.objects.object_class": object_class,
        "labels.objects.masks": target_masks,
        "masks.tracks": pad,
    }
