"""`Combination` — a linear-combination producer over an ``outputs.*`` leaf."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import _UNNAMED, IO, Mode, TensorSpec, split_key, unflatten_spec
from salt.core.outputs.output_field import OutputField


class Combination(nn.Module):
    """Linear-combination producer: a new ``outputs.*`` leaf from a source bundle leaf.

    Reads a source ``outputs.<stream>.<src>`` leaf a producer already minted
    (e.g. softmaxed class probs) and produces a new
    ``outputs.<stream>.<name>`` scalar leaf as a weighted sum over its
    last-dim channels: ``out = sum(scale * source[..., index])`` over
    `terms`. Because it reads a bundle leaf (not a renamed export name), both
    sinks consume/name the new leaf like any other ``outputs.*`` leaf.

    The output last dim collapses to a scalar (a GLOBAL float, no per-token
    axis).

    Parameters
    ----------
    source : str
        The source bundle leaf, an ``outputs.<stream>.<src>`` key. The new
        leaf is written under the SAME stream as the source.
    name : str
        The new output leaf's last component (``outputs.<stream>.<name>``).
    terms : Mapping[int, float]
        Source last-dim channel index -> scale, in combination order (e.g.
        ``{0: 1.0, 1: 1.0}`` for ``probs[..., 0] + probs[..., 1]``). At least
        one term; every index must be a non-negative int.

    Raises
    ------
    ConfigError
        For a non-``outputs`` source, a wildcard source, an empty `terms`, or
        a negative/non-int channel index.
    """

    def __init__(
        self,
        source: str,
        name: str,
        terms: Mapping[int, float],
    ) -> None:
        super().__init__()
        self.name = _UNNAMED
        parts = split_key(source)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"Combination source {source!r} contains a wildcard — conversion sources are "
                "concrete (design §2.2)"
            )
        if len(parts) < 2 or parts[0] != "outputs":
            raise ConfigError(
                f"Combination source {source!r} must be an 'outputs.<stream>.<name>' producer "
                "leaf — a combination reads a bundle prob/pred leaf, not a raw prediction or a "
                "renamed Athena output (design §6.2 / Q2)"
            )
        if not terms:
            raise ConfigError(
                f"Combination {name!r}: 'terms' must map at least one source channel index to a "
                "scale (e.g. {0: 1.0, 1: 1.0} for probs[..., 0] + probs[..., 1])"
            )
        self.source = source
        self.output_name = name
        self.stream = parts[1]
        # preserve declaration order (jsonargparse builds an ordered dict); the
        # sum order is load-bearing for the v1 float bitwise-equality
        self.terms: tuple[tuple[int, float], ...] = tuple(
            (self._checked_index(index, name), float(scale)) for index, scale in terms.items()
        )
        self.output_key = f"outputs.{self.stream}.{name}"

    @staticmethod
    def _checked_index(index: Any, name: str) -> int:
        """Validate a source channel index is a non-negative int.

        Raises
        ------
        ConfigError
            For a non-int or negative index.
        """
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ConfigError(
                f"Combination {name!r}: source channel index {index!r} must be a non-negative "
                "int (the source leaf's last-dim position to weight)"
            )
        return index

    def declare_io(self, mode: Mode) -> IO:
        """Declare the source ``outputs.*`` leaf -> the new ``outputs.<stream>.<name>`` leaf.

        Both ports are active in every mode (``modes=ALL``): gated by demand,
        not a hard mode flag, like the other conversion producers.
        """
        del mode
        requires = {self.source: TensorSpec(shape=None, dtype="float32", kind="data")}
        produces = {self.output_key: TensorSpec(shape=None, dtype="float32", kind="data")}
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """The combination collapses the source last dim to a single scalar column (width 1)."""
        del widths
        return {self.output_key: 1}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Compute ``sum(scale * source[..., index])`` over `terms`, in declared order."""
        del mode
        source = b.get(self.source)
        out = sum(scale * source[..., index] for index, scale in self.terms)
        return {self.output_key: out}

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """One float global field named after the combination; present in both H5 and ONNX."""
        del run_name, model_modules
        return [OutputField(h5_name=self.output_name, dtype="f4", axis="global", final=True)]
