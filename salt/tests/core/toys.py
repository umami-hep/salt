"""Toy GraphModules for the M1 kernel integration tests (plan 03, stage F).

No physics (M1 scope): these fixtures exercise every kernel feature on a
five-module toy graph — symbolic dims and dtype unification (design §2.2),
kind typing (pad_mask/label/loss), mode-gated ports, the optional-port probe
idiom, wildcard narrowing (design §2.2 rules (a)-(d)), terminal consumers
(writers, design §3.1), and the all-modes-dead error (design §3.1
principle 10).

The toy graph wired by ``salt/tests/core/configs/toy.yaml``::

    <sources> --raw.x--> source --inputs.x,masks.x--> embed --embed.x--> head
    <sources> --raw.x--> labels --labels.x (fit/val, narrowed)----------^
    head --losses.total (fit/val)--> <sinks>
    head --preds.x (test)----------> writer (terminal) / <sinks>

Modules are referenced via ``class_path: salt.tests.core.toys.<Class>`` (the
M1 CLI loader, salt/core/cli.py) and by the ``salt.core.demo_m1`` demo. The
loader assigns ``instance.name`` from the config key after construction.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec

__all__ = [
    "ToyDead",
    "ToyEmbed",
    "ToyHead",
    "ToySource",
    "ToyWildcardLabels",
    "ToyWriter",
]

_UNNAMED = "unnamed"  # overwritten by the CLI loader (or the test) per design §2.2


class ToySource:
    """Reader stand-in: turns the framework source leaf into inputs + pad mask.

    Declares the sources-style keys ``inputs.x`` / ``masks.x`` (design §2.4
    analogue) from the dataset-boundary leaf ``raw.x``. The produced mask is
    all-False (nothing padded) with ``kind="pad_mask"`` so kind typing is
    exercised end-to-end.
    """

    def __init__(self, n_features: int = 8) -> None:
        self.name = _UNNAMED
        self.n_features = n_features

    def declare_io(self, mode: Mode) -> IO:
        """Declare raw.x -> inputs.x + masks.x with symbolic batch dim (design §2.2)."""
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


class ToyEmbed(nn.Module):
    """nn.Module embedder: the executor call lands in ``forward(b, mode)`` (design §2.5).

    Optionally consumes the pad mask via the ``key in b`` probe idiom
    (design §2.2 optional ports) — legal under debug execution even when the
    planner dropped the port.
    """

    def __init__(self, in_dim: int = 8, out_dim: int = 16) -> None:
        super().__init__()
        self.name = _UNNAMED
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def declare_io(self, mode: Mode) -> IO:
        """Declare inputs.x (+ optional masks.x) -> embed.x (design §2.2)."""
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
        if "masks.x" in b:  # optional-port probe idiom (design §2.2)
            x = x * (~b.get("masks.x")).unsqueeze(-1).to(x.dtype)
        return {"embed.x": torch.relu(self.linear(x))}


class ToyWildcardLabels:
    """Framework-style wildcard label provider (design §2.2 rules (a)-(d)).

    Declares the pattern ``labels.*`` (fit/val only); the planner narrows it
    against concrete demand and validates the narrowed keys against the
    schema. At runtime it must return exactly the narrowed key set — here the
    configured ``fields`` mirror what the toy dataset "has".
    """

    allow_wildcards: ClassVar[bool] = True  # framework wildcard capability (design §2.2)

    def __init__(self, fields: tuple[str, ...] = ("x",), n_classes: int = 3) -> None:
        self.name = _UNNAMED
        self.fields = tuple(fields)
        self.n_classes = n_classes

    def declare_io(self, mode: Mode) -> IO:
        """Declare raw.x -> labels.* (kind=label, fit/val only) (design §2.2)."""
        del mode
        return IO(
            requires=unflatten_spec({
                "raw.x": TensorSpec(shape=("B", "F:raw"), dtype="float32", modes=Mode.TRAINING),
            }),
            produces=unflatten_spec({
                "labels.*": TensorSpec(
                    shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
                ),
            }),
        )

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Produce one integer label tensor per configured field (seeded)."""
        del mode
        n = b.get("raw.x").shape[0]
        gen = torch.Generator().manual_seed(0)
        return {
            f"labels.{field}": torch.randint(self.n_classes, (n,), generator=gen)
            for field in self.fields
        }


class ToyHead:
    """Prediction head with mode-gated ports: preds always, labels->loss in fit/val.

    ``embed_key`` is configurable so a typo'd config (toy_broken.yaml) can
    exercise the §4.1-quality missing-producer error.
    """

    def __init__(
        self, in_dim: int = 16, n_classes: int = 3, embed_key: str = "embed.x"
    ) -> None:
        self.name = _UNNAMED
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.embed_key = embed_key

    def declare_io(self, mode: Mode) -> IO:
        """Declare embed (+ fit/val labels) -> preds (+ fit/val loss) (design §2.2)."""
        del mode
        return IO(
            requires=unflatten_spec({
                self.embed_key: TensorSpec(shape=("B", self.in_dim), dtype="float32"),
                "labels.x": TensorSpec(
                    shape=("B",), dtype="int64", kind="label", modes=Mode.TRAINING
                ),
            }),
            produces=unflatten_spec({
                "preds.x": TensorSpec(shape=("B", self.n_classes), dtype="float32"),
                "losses.total": TensorSpec(
                    shape=(), dtype="float32", kind="loss", modes=Mode.TRAINING
                ),
            }),
        )

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Produce softmax preds always, plus the cross-entropy loss in fit/val.

        Returns a mixture of nested (``preds``) and dotted (``losses.total``)
        spellings — both are canonicalised by the executor (design §2.5).
        """
        logits = b.get(self.embed_key)[:, : self.n_classes]
        out: dict = {"preds": {"x": logits.softmax(dim=-1)}}
        if mode & Mode.TRAINING:
            out["losses.total"] = F.cross_entropy(logits, b.get("labels.x"))
        return out


class ToyWriter:
    """Test-only sink: a terminal consumer (requires, no produces) (design §3.1).

    Terminal consumers anchor demand themselves; the collected predictions
    stand in for an output file (writers are M3).
    """

    def __init__(self, key: str = "preds.x") -> None:
        self.name = _UNNAMED
        self.key = key
        self.collected: list[torch.Tensor] = []

    def declare_io(self, mode: Mode) -> IO:
        """Declare the written key as a TEST-only require, producing nothing."""
        del mode
        return IO(requires=unflatten_spec({self.key: TensorSpec(modes=Mode.TEST)}))

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Collect the written tensor; produce no bundle keys."""
        del mode
        self.collected.append(b.get(self.key).detach().clone())
        return {}


class ToyDead:
    """Produces a key nothing consumes in any mode (design §3.1 principle 10).

    Adding this module to the toy graph makes ``compile_plan`` raise
    `AllModesDeadError` (deadness is provable: the toy config declares sinks
    for every primary mode); ``deadcode`` instead *reports* it as pruned.
    """

    def __init__(self) -> None:
        self.name = _UNNAMED

    def declare_io(self, mode: Mode) -> IO:
        """Declare a produced leaf with no consumer anywhere."""
        del mode
        return IO(produces=unflatten_spec({"dead.x": TensorSpec(shape=("B",))}))

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Produce the dead leaf (never reached — the planner rejects the graph)."""
        del b, mode
        return {"dead.x": torch.zeros(1)}
