"""Toy graph participants (no physics): kernel-API toys compiled directly with
``compile_plan(..., sources=...)`` and the trainer-surface toys wired by
``configs/toy_trainer.yaml``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from salt.data.base import Reader, WorkerCtx
from salt.graph.bundle import Bundle
from salt.graph.spec import IO, UNNAMED, Mode, TensorSpec, unflatten_spec
from salt.model.base import SaltModelModule
from salt.outputs.sinks.sink import RuntimeSink
from salt.schema import GroupSchema, Schema

__all__ = [
    "ToyAuxLoss",
    "ToyDead",
    "ToyEmbed",
    "ToyHead",
    "ToyReader",
    "ToySource",
    "ToyWildcardLabels",
    "ToyWriter",
]


# ---------------------------------------------------------------------------
# kernel-API toys (plain GraphModules; names set by the test)
# ---------------------------------------------------------------------------


class ToySource:
    """Reader stand-in: turns the framework source leaf into inputs + pad mask."""

    def __init__(self, n_features: int = 8) -> None:
        self.name = UNNAMED
        self.n_features = n_features

    def declare_io(self, mode: Mode) -> IO:
        """Declare raw.x -> inputs.x + masks.x with symbolic batch dim."""
        del mode
        return IO(
            requires=unflatten_spec({
                "raw.x": TensorSpec(shape=("B", self.n_features), dtype="float32"),
            }),
            produces=unflatten_spec({
                "inputs.x": TensorSpec(shape=("B", self.n_features), dtype="float32"),
                "masks.x": TensorSpec(shape=("B",), dtype="bool", kind="pad_mask"),
            }),
        )

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Produce inputs.x (a copy of raw.x) and an all-False pad mask."""
        del mode
        raw = b.get("raw.x")
        return {
            "inputs": {"x": raw.clone()},
            "masks": {"x": torch.zeros(raw.shape[0], dtype=torch.bool)},
        }


class ToyWildcardLabels:
    """Framework-style wildcard label provider over ``labels.<stream>.*``."""

    allow_wildcards: ClassVar[bool] = True  # framework wildcard capability

    def __init__(
        self, stream: str = "x", fields: tuple[str, ...] = ("y",), n_classes: int = 3
    ) -> None:
        self.name = UNNAMED
        self.stream = stream
        self.fields = tuple(fields)
        self.n_classes = n_classes

    def declare_io(self, mode: Mode) -> IO:
        """Declare raw.<stream> -> labels.<stream>.* (kind=label, fit/val only)."""
        del mode
        return IO(
            requires=unflatten_spec({
                f"raw.{self.stream}": TensorSpec(
                    shape=("B", "F:raw"), dtype="float32", modes=Mode.TRAINING
                ),
            }),
            produces=unflatten_spec({
                f"labels.{self.stream}.*": TensorSpec(
                    shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
                ),
            }),
        )

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Produce one integer label tensor per configured field (seeded)."""
        del mode
        n = b.get(f"raw.{self.stream}").shape[0]
        gen = torch.Generator().manual_seed(0)
        return {
            f"labels.{self.stream}.{field}": torch.randint(self.n_classes, (n,), generator=gen)
            for field in self.fields
        }


class ToyDead:
    """Produces a key nothing consumes in any mode."""

    def __init__(self) -> None:
        self.name = UNNAMED

    def declare_io(self, mode: Mode) -> IO:
        """Declare a produced leaf with no consumer anywhere."""
        del mode
        return IO(produces=unflatten_spec({"dead.x": TensorSpec(shape=("B",))}))

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Produce the dead leaf (never reached — the planner rejects the graph)."""
        del b, mode
        return {"dead.x": torch.zeros(1)}


# ---------------------------------------------------------------------------
# trainer-surface toys (class_path targets of configs/toy_trainer.yaml)
# ---------------------------------------------------------------------------


class ToyReader(Reader):
    """In-memory reader serving one global stream ``x`` (no file I/O).

    Each row of the structured ``raw.x`` carries `n_features` float32 columns
    ``f0..f{n-1}`` and an int32 class label ``y``; ``masks.x`` is all-valid and
    ``meta.rows`` is TEST-only. Like `H5StructuredReader`, ``raw.x`` declares no
    ``fields`` — each consumer names its own subset.

    Parameters
    ----------
    n_rows : int, optional
        Rows served, by default 64.
    n_features : int, optional
        Float feature columns, by default 8.
    n_classes : int, optional
        Label classes drawn for ``y``, by default 3.
    seed : int, optional
        RNG seed of the generated rows, by default 0.
    schema : bool, optional
        Attach a `Schema` (so `label_universe` validates ``labels.x.*``
        narrowing), by default True; False leaves it None — the no-schema path.
    """

    def __init__(
        self,
        n_rows: int = 64,
        n_features: int = 8,
        n_classes: int = 3,
        seed: int = 0,
        schema: bool = True,
    ) -> None:
        super().__init__()
        self.n_rows = int(n_rows)
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        fields = {f"f{i}": "float32" for i in range(self.n_features)} | {"y": "int32"}
        self.schema = Schema(groups={"x": GroupSchema(fields=fields)}) if schema else None
        self._rows: np.ndarray | None = None

    @property
    def streams(self) -> tuple[str, ...]:
        return ("x",)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            produces=unflatten_spec({
                "raw.x": TensorSpec(shape=("B",), kind="data"),
                "masks.x": TensorSpec(shape=("B",), dtype="bool", kind="pad_mask"),
                "meta.rows": TensorSpec(shape=(2,), dtype="int64", kind="meta", modes=Mode.TEST),
            })
        )

    def _data(self) -> np.ndarray:
        """The generated structured rows (built once, seeded)."""
        if self._rows is None:
            names = [f"f{i}" for i in range(self.n_features)]
            rows = np.empty(self.n_rows, dtype=np.dtype([(n, "f4") for n in names] + [("y", "i4")]))
            rng = np.random.default_rng(self.seed)
            for name in names:
                rows[name] = rng.standard_normal(self.n_rows)
            rows["y"] = rng.integers(0, self.n_classes, self.n_rows)
            self._rows = rows
        return self._rows

    def __len__(self) -> int:
        return self.n_rows

    def bind(self, ctx: WorkerCtx) -> None:
        del ctx
        self._data()

    def read(self, rows: slice, mode: Mode) -> dict[str, np.ndarray]:
        batch = self._data()[rows].copy()  # never hand out the reader's buffer
        out = {"raw.x": batch, "masks.x": np.zeros(len(batch), dtype=bool)}
        if mode == Mode.TEST:
            out["meta.rows"] = np.array([rows.start, rows.stop], dtype=np.int64)
        return out

    def schema_group(self, stream: str) -> GroupSchema | None:
        return None if self.schema is None else self.schema.groups.get(stream)

    def label_universe(self) -> tuple[str, ...] | None:
        if self.schema is None:
            return None
        return tuple(f"labels.x.{field}" for field in self.schema.groups["x"].fields)

    def with_source(self, filename, num: int = -1, stage=None) -> ToyReader:  # noqa: ANN001
        """In-memory: the source is ignored; `num` caps the rows."""
        del filename, stage
        n_rows = self.n_rows if num < 0 else min(num, self.n_rows)
        has_schema = self.schema is not None
        clone = ToyReader(n_rows, self.n_features, self.n_classes, self.seed, has_schema)
        clone.name = self.name
        return clone


class ToyEmbed(SaltModelModule):
    """Linear embedder ``inputs.x (B, in_dim) -> embed.x (B, out_dim)``."""

    def __init__(self, in_dim: int = 8, out_dim: int = 16) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            requires=unflatten_spec({
                "inputs.x": TensorSpec(shape=("B", self.in_dim), dtype="float32"),
                "masks.x": TensorSpec(
                    shape=("B",), dtype="bool", kind="pad_mask", optional=True
                ),
            }),
            produces=unflatten_spec({
                "embed.x": TensorSpec(shape=("B", self.out_dim), dtype="float32"),
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict:
        """Embed inputs.x, zeroing padded rows when the optional mask is present."""
        del mode
        x = b.get("inputs.x")
        if "masks.x" in b:  # optional-port probe idiom
            x = x * (~b.get("masks.x")).unsqueeze(-1).to(x.dtype)
        return {"embed.x": torch.relu(self.linear(x))}


class ToyHead(SaltModelModule):
    """Prediction head with mode-gated ports: ``preds.x`` always, ``labels.x.y -> losses.x``
    in FIT/VAL.

    `extra_preds` adds a second ``preds.<extra_preds>`` port (the same probabilities)
    that nothing consumes — the dead-TEST-preds fixture.
    """

    def __init__(
        self,
        in_dim: int = 16,
        n_classes: int = 3,
        embed_key: str = "embed.x",
        extra_preds: str | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.embed_key = embed_key
        self.extra_preds = extra_preds

    def declare_io(self, mode: Mode) -> IO:
        del mode
        preds = TensorSpec(shape=("B", self.n_classes), dtype="float32")
        produces = {
            "preds.x": preds,
            "losses.x": TensorSpec(shape=(), dtype="float32", kind="loss", modes=Mode.TRAINING),
        }
        if self.extra_preds is not None:
            produces[f"preds.{self.extra_preds}"] = preds
        return IO(
            requires=unflatten_spec({
                self.embed_key: TensorSpec(shape=("B", self.in_dim), dtype="float32"),
                "labels.x.y": TensorSpec(
                    shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
                ),
            }),
            produces=unflatten_spec(produces),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict:
        """Softmax preds always, plus the cross-entropy loss in FIT/VAL."""
        logits = b.get(self.embed_key)[:, : self.n_classes]
        probs = logits.softmax(dim=-1)
        out: dict = {"preds.x": probs}
        if self.extra_preds is not None:
            out[f"preds.{self.extra_preds}"] = probs
        if mode & Mode.TRAINING:
            out["losses.x"] = F.cross_entropy(logits, b.get("labels.x.y"))
        return out


class ToyAuxLoss(SaltModelModule):
    """Label-free auxiliary loss ``embed.x -> losses.aux`` (mean squared embedding).

    Declares the loss in EVERY mode on purpose: `LossSum` consumes it only in
    FIT/VAL, so TEST/ONNX drop the module by demand pruning, not mode gating.
    """

    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.in_dim = in_dim

    def declare_io(self, mode: Mode) -> IO:
        del mode
        return IO(
            requires=unflatten_spec({
                "embed.x": TensorSpec(shape=("B", self.in_dim), dtype="float32"),
            }),
            produces=unflatten_spec({
                "losses.aux": TensorSpec(shape=(), dtype="float32", kind="loss"),
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict:
        del mode
        return {"losses.aux": b.get("embed.x").pow(2).mean()}


class ToyWriter(RuntimeSink):
    """TEST persistence sink: consumes `key` (+ ``meta.rows``), collecting each batch's tensor.

    A terminal graph node (also usable on the kernel API): the executor never
    calls it — the driver hands it each executed bundle via `consume`.
    """

    def __init__(self, key: str = "preds.x", modes: Sequence[str] | None = None) -> None:
        super().__init__(modes=modes)
        self.name = UNNAMED
        self.key = key
        self.collected: list[torch.Tensor] = []

    def declare_io(self, mode: Mode) -> IO:
        del mode  # non-TEST modes never get here (`allowed_modes` = TEST; `Node` gates)
        return IO(
            requires=unflatten_spec({
                self.key: TensorSpec(modes=Mode.TEST),
                "meta.rows": TensorSpec(dtype="int64", kind="meta", modes=Mode.TEST),
            })
        )

    def consume(self, bundle: Bundle) -> None:
        self.collected.append(bundle.get(self.key).detach().clone())
