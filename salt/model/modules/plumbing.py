"""Parameter-free tensor-plumbing GraphModules: stream concatenation/splitting of
the shared sequence, and feature-dim vector concatenation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import (
    IO,
    Mode,
    TensorSpec,
    sym_dim,
    unflatten_spec,
)
from salt.model.base import SaltModelModule
from salt.model.modules.stream_embed import _stream_len
from salt.model.modules.transformer_encoder import _ENC_LEN, _SEQ_LEN


class Concat(SaltModelModule):
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
        """Declare ``embed.*``/``masks.*`` per stream -> seq keys (+ ONNX ``seq.offsets``)."""
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
        """Contribute the ``seq.x`` last-dim width from the first resolved per-stream embed width.

        Needed because `StreamEmbed`'s ``shape=None`` output gives the dim
        table nothing concrete to bind ``embed_dim`` from directly.
        """
        for stream in self.streams:
            width = widths.get(f"embed.{stream}")
            if width is not None:
                return {"seq.x": width}
        return {}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Any]:
        """Concatenate streams along the token dim + record the layout (+ ONNX ``seq.offsets``)."""
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


class Split(SaltModelModule):
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
        if not streams:
            raise ConfigError("Split: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Split: duplicate streams in {tuple(streams)}")
        self.streams = tuple(streams)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``encoded.seq``/``seq.layout`` -> per-stream ``encoded.*`` (+ ONNX offsets)."""
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
        """Slice each stream out of the encoded sequence (ONNX: dynamic ``index_select``)."""
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


class VectorConcat(SaltModelModule):
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
        """Declare each ``[B, D_i]`` input (own width symbol) -> the ``[B, Dsum]`` output."""
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
