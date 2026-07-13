"""`PadMaskWriter` — the ``outputs:`` section pad-mask writer."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.core.outputs.run_task_output import OutputSectionWriter


class PadMaskWriter(OutputSectionWriter):
    """The ``outputs:`` section pad-mask writer.

    Declares ``outputs.<stream>.mask`` and produces the bool pad-mask leaf (read
    verbatim from ``masks.<stream>`` — True = padded). The per-token file-length
    re-expansion (incl. the v1 ``mask=False`` truncation quirk) stays in the dumb
    H5 sink. Matches the legacy ``salt.core.writers.PadMaskWriter`` semantics.

    Unlike `InputCopyWriter`, the pad mask IS a bundle leaf (``masks.<stream>``),
    so this writer is a real graph node: it requires ``masks.<stream>`` and
    produces ``outputs.<stream>.mask`` so the leaf flows through the graph to
    the sink, which then packs it as a ``('mask', '?')`` column and pads to the
    file length.

    Parameters
    ----------
    streams : Sequence[str]
        The sequence streams to write a mask column for. Each must be a padded
        sequence stream (validated by the sink against the reader). Required and
        explicit (the v1 default — every sequence stream with a configured task —
        is resolved by the sink, which knows the reader).
    """

    name = "pad_mask"
    """The section instance name (overridable by the config dict key)."""

    def __init__(self, streams: Sequence[str]) -> None:
        super().__init__()
        self.name = type(self).name
        names = list(streams or [])
        if not names:
            raise ConfigError(
                "PadMaskWriter needs a non-empty 'streams' list — name the sequence streams whose "
                "boolean pad-mask column to write (plan 34 W34.2)"
            )
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ConfigError(
                f"PadMaskWriter: duplicate stream(s) {dup} — one entry per stream (plan 34 W34.2)"
            )
        self.streams = tuple(names)

    def output_key(self, stream: str) -> str:
        """The ``outputs.<stream>.mask`` leaf for a stream."""
        return f"outputs.{stream}.mask"

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``masks.<stream>`` -> ``outputs.<stream>.mask`` per stream.

        Demand-gated: FIT/VAL prune it (the mask column is eval-H5-only). The
        require is ``kind=pad_mask`` (unifies with the stream mask producer);
        the produced leaf is ``kind=data`` (a serialisation column).
        """
        del mode
        requires: dict[str, TensorSpec] = {}
        produces: dict[str, TensorSpec] = {}
        for stream in self.streams:
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", sym_dim("T", stream)), dtype="bool", kind="pad_mask"
            )
            produces[self.output_key(stream)] = TensorSpec(shape=None, dtype="bool", kind="data")
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pass each stream's bool pad mask through to ``outputs.<stream>.mask``.

        The sink owns the ``u2s`` pack + the file-length re-expansion. A fresh
        clone avoids aliasing the bundle's ``masks.*`` leaf (write-once).
        """
        del mode
        return {
            self.output_key(stream): b.get(f"masks.{stream}").clone()
            for stream in self.streams
        }

    def mask_streams(self) -> tuple[str, ...]:
        """The streams a mask column is written for (the sink reads this)."""
        return self.streams
