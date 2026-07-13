"""Concat GraphModule (stream concatenation into the shared sequence)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

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
from salt.core.nn.transformer_encoder import _SEQ_LEN


class Concat(nn.Module):
    """Concatenate embedded streams into one sequence.

    Produces ``seq.x`` / ``seq.mask`` / ``seq.layout``; the concat ORDER is
    the configured list and nothing else. ``seq.layout`` is a dict-valued
    meta leaf ``{stream: (start, stop)}`` over the pre-register sequence,
    consumed by `Split`.

    Registers live in `TransformerEncoder`, not here (the composed v1
    `Transformer` appends its own register tokens internally and requires
    ``num_registers >= 1``), so nonzero ``registers`` is rejected.
    """

    def __init__(self, streams: Sequence[str], registers: int = 0) -> None:
        """Capture the explicit stream order.

        Raises
        ------
        ConfigError
            If `streams` is empty or contains duplicates, or if `registers`
            is nonzero (registers live in `TransformerEncoder`, see class docstring).
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ConfigError("Concat: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Concat: duplicate streams in {tuple(streams)}")
        if registers != 0:
            raise ConfigError(
                "Concat: registers are internal to TransformerEncoder in M2 (the composed v1 "
                "Transformer appends them, transformer.py:679-681) — set "
                "encoder num_registers instead; Concat-owned registers land at M7 (design §5.1)"
            )
        self.streams = tuple(streams)
        self.registers = registers

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``embed.*``/``masks.*`` per stream -> seq keys.

        All streams share one instance-scoped embed-width symbol — equal
        widths are a genuine concat constraint. In ONNX mode an additional
        ``seq.offsets`` int64 tensor is produced (the trace-safe stream
        boundary table consumed by `Split`'s export branch).
        """
        del mode
        embed_dim = sym_dim("E", self.name)
        requires: dict[str, TensorSpec] = {}
        for stream in self.streams:
            requires[f"embed.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream), embed_dim), dtype="float32"
            )
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream)), dtype="bool", kind="pad_mask"
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                "seq.x": TensorSpec(shape=("B", _SEQ_LEN, embed_dim), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                "seq.layout": TensorSpec(kind="meta"),
                "seq.offsets": TensorSpec(
                    shape=(len(self.streams) + 1,), dtype="int64", modes=Mode.ONNX
                ),
            }),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute the ``seq.x`` last-dim width from the per-stream embed widths.

        `StreamEmbed` produces ``embed.<stream>`` with ``shape=None`` (rank
        inferred from the bound input), so there is no concrete shape for the
        dim table to bind ``embed_dim`` from directly — this hook forwards the
        resolved embed width to ``seq.x`` (needed on the encoderless-pool path,
        which has no encoder require to bind it otherwise). The per-stream
        embeds share one width, so the FIRST resolved input width is used.

        Returns
        -------
        dict[str, int]
            ``{"seq.x": embed_width}`` once an ``embed.<stream>`` width is
            resolved, otherwise ``{}``.
        """
        for stream in self.streams:
            width = widths.get(f"embed.{stream}")
            if width is not None:
                return {"seq.x": width}
        return {}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Any]:
        """Concatenate streams along the token dim and record the layout.

        In ONNX mode the per-stream token counts are ALSO published as the
        ``seq.offsets`` cumulative-boundary tensor, built with
        ``torch.onnx.operators.shape_as_tensor`` so the boundaries trace to
        Shape/Concat/CumSum nodes that stay symbolic under tracing.
        """
        xs = [b.get(f"embed.{stream}") for stream in self.streams]
        masks = [b.get(f"masks.{stream}") for stream in self.streams]
        layout: dict[str, tuple[int, int]] = {}
        start = 0
        for stream, x in zip(self.streams, xs, strict=True):
            layout[stream] = (start, start + x.shape[1])
            start += x.shape[1]
        out: dict[str, Any] = {
            "seq.x": torch.cat(xs, dim=1),
            "seq.mask": torch.cat(masks, dim=1),
            "seq.layout": layout,
        }
        if mode & Mode.ONNX:
            lengths = [torch.onnx.operators.shape_as_tensor(mask)[1:2] for mask in masks]
            zero = torch.zeros(1, dtype=torch.int64)
            out["seq.offsets"] = torch.cumsum(torch.cat([zero, *lengths]), dim=0)
        return out
