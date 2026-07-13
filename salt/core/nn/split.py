"""Split GraphModule (slice the encoded sequence back into streams)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import (
    IO,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.core.nn.featurewise import _UNNAMED
from salt.core.nn.stream_embed import _stream_len
from salt.core.nn.transformer_encoder import _ENC_LEN


class Split(nn.Module):
    """Per-stream slices of ``encoded.seq`` via the ``seq.layout`` meta leaf.

    Register rows sit AFTER every stream in the encoder output, so the
    pre-register layout offsets remain valid slices of ``encoded.seq``.

    ONNX export uses dynamic ``index_select`` slicing (via `Concat`'s
    ``seq.offsets`` tensor) rather than the eager branch's Python-int
    ``seq.layout`` offsets: with >=2 dynamic sequence axes, tracing bakes
    Python-int offsets as constants, which is silently WRONG once a second
    stream is present (verified empirically with >=2 dynamic axes producing
    mis-sliced output at every probed grid point).
    """

    def __init__(self, streams: Sequence[str]) -> None:
        """Capture the streams to slice out.

        Raises
        ------
        ConfigError
            If `streams` is empty or contains duplicates.
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ConfigError("Split: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Split: duplicate streams in {tuple(streams)}")
        self.streams = tuple(streams)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``encoded.seq`` + ``seq.layout`` -> ``encoded.<stream>`` per stream.

        In ONNX mode the `Concat` ``seq.offsets`` boundary tensor is
        additionally required (the trace-safe slicing path, see class docstring).
        """
        del mode
        width = sym_dim("D", self.name)
        return IO(
            requires=unflatten_spec({
                "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, width), dtype="float32"),
                "seq.layout": TensorSpec(kind="meta"),
                "seq.offsets": TensorSpec(shape=None, dtype="int64", modes=Mode.ONNX),
            }),
            produces=unflatten_spec({
                f"encoded.{stream}": TensorSpec(
                    shape=("B", _stream_len(stream), width), dtype="float32"
                )
                for stream in self.streams
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Slice each configured stream out of the encoded sequence.

        ONNX mode uses dynamic ``index_select`` slicing driven by
        ``seq.offsets`` (see class docstring); the stream's position in the
        offsets table is its position in the ``seq.layout`` dict, so a
        `Split` over a stream SUBSET stays correct without the full concat list.
        """
        encoded = b.get("encoded.seq")
        layout = b.get("seq.layout")
        out: dict[str, Tensor] = {}
        if mode & Mode.ONNX:
            offsets = b.get("seq.offsets")
            order = list(layout)
            for stream in self.streams:
                i = order.index(stream)
                idx = torch.arange(offsets[i + 1] - offsets[i]) + offsets[i]
                out[f"encoded.{stream}"] = encoded.index_select(1, idx)
            return out
        for stream in self.streams:
            start, stop = layout[stream]
            out[f"encoded.{stream}"] = encoded[:, start:stop]
        return out
