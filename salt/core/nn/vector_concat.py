"""VectorConcat GraphModule (feature-dim concatenation)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import (
    _UNNAMED,
    IO,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)


class VectorConcat(nn.Module):
    """Ordered concatenation of ``[B, D_i]`` vectors into one ``[B, Dsum]`` key.

    GN3's 2-feature ``global`` stream is fed past the encoder and
    concatenated onto the pooled representation (``pooled_dim = 256 + 2``,
    the width every GN3 task consumes). Concat ORDER is the config list —
    for converted GN3 checkpoints, ``inputs: [pooled.global, normed.global]``
    (pooled first) matches the task's first-layer weight layout. Output
    width ``Dsum = sum(D_i)`` is resolved at bind via `derived_widths`. ONNX
    uses the ``export.inputs`` ``alias:`` mechanism (`OnnxAdapter`), not this
    module; VectorConcat itself is mode-agnostic.
    """

    def __init__(self, inputs: Sequence[str], out: str = "pooled.global") -> None:
        """Capture the explicit ordered input list and the output key.

        Parameters
        ----------
        inputs : Sequence[str]
            Dotted bundle keys to concatenate, in the EXACT order they appear
            in the output. Must be non-empty with no duplicates.
        out : str, optional
            The produced concatenated key, by default ``"pooled.global"``
            (the slot every GN3 task reads).

        Raises
        ------
        ConfigError
            If `inputs` is empty, contains duplicates, or names `out` itself
            (a self-feed).
        """
        super().__init__()
        self.name = _UNNAMED
        if not inputs:
            raise ConfigError("VectorConcat: inputs must be a non-empty sequence")
        if len(set(inputs)) != len(tuple(inputs)):
            raise ConfigError(f"VectorConcat: duplicate inputs in {tuple(inputs)}")
        if out in inputs:
            raise ConfigError(
                f"VectorConcat: out {out!r} appears in inputs {tuple(inputs)} — a module cannot "
                "consume its own output (design §2.1 write-once)"
            )
        self.inputs = tuple(inputs)
        self.out_key = out

    def declare_io(self, mode: Mode) -> IO:
        """Declare each ``[B, D_i]`` input -> the ``[B, Dsum]`` output.

        Each input gets its OWN instance-scoped width symbol (inputs are
        genuinely different widths, so they must NOT share a symbol); the
        output's ``Dsum`` is resolved at bind via `derived_widths`.
        """
        del mode
        requires: dict[str, TensorSpec] = {
            key: TensorSpec(shape=("B", sym_dim(f"D{i}", self.name)), dtype="float32")
            for i, key in enumerate(self.inputs)
        }
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", sym_dim("Dsum", self.name)), dtype="float32"),
            }),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute ``Dsum = sum(D_i)`` once every input width is resolved."""
        if all(key in widths for key in self.inputs):
            return {self.out_key: sum(widths[key] for key in self.inputs)}
        return {}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Concatenate the inputs along the feature dim, in the configured order."""
        del mode
        return {self.out_key: torch.cat([b.get(key) for key in self.inputs], dim=-1)}
