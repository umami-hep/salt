"""GraphModules wrapping live v1 ``nn.Module`` instances for GN2 forward parity.

Reproduce, as explicit graph wiring, the glue that ``ModelWrapper.forward``
performs around v1 modules. Weights are shared (not copied), so any output
difference between v1 and an executed plan over these wrappers is a wiring
bug: concat order, context prepend, pad-mask polarity/order, or
padded-token handling.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.models import InitNet, InputNorm, Transformer
from salt.models.pooling import GlobalAttentionPooling
from salt.models.task import TaskBase, VertexingTask

__all__ = [
    "Concat",
    "ConstituentTask",
    "GlobalObjectTask",
    "Normaliser",
    "Pooling",
    "Split",
    "StreamEmbed",
    "TransformerEncoder",
]

_UNNAMED = "unnamed"
"""Placeholder instance name — `from_v1` (or a test) assigns the dict key."""

_EMBED_DIM = sym_dim("E", "embed")
_SEQ_LEN = sym_dim("S", "seq")
_ENC_LEN = sym_dim("L", "enc")
_OUT_DIM = sym_dim("D", "out")
_REG_LEN = sym_dim("R", "registers")


def _stream_len(stream: str) -> str:
    """Return the symbolic token-count dim for a sequence stream, e.g. ``"T:tracks"``."""
    return sym_dim("T", stream)


class Normaliser(nn.Module):
    """Wraps the v1 `InputNorm`, producing NEW ``normed.*`` keys.

    v1's ``InputNorm.forward`` rebinds the keys of the dict it is given, so
    the wrapper builds a FRESH dict from the bundle's ``inputs.*`` leaves
    before calling it — ``inputs.*`` is never mutated.
    """

    def __init__(self, norm: InputNorm) -> None:
        """Wrap a live v1 `InputNorm` instance (weights shared, not copied).

        Raises
        ------
        ValueError
            If the norm covers streams outside the parity scope: ``global``
            remapping, ``_``-prefixed streams, or NO_NORM streams such as
            ``parameters`` (those pass through `InputNorm` un-normalised,
            which would alias ``normed.*`` to ``inputs.*``).
        """
        super().__init__()
        self.name = _UNNAMED
        for stream in norm.variables:
            if stream in norm.NO_NORM or stream.startswith("_") or stream == "global":
                raise ValueError(
                    f"Normaliser: stream {stream!r} is outside the GN2 parity scope "
                    "(NO_NORM / underscore / 'global' streams are not supported)"
                )
        self.norm = norm
        self.streams = tuple(norm.variables)
        self.global_object = norm.global_object

    def _spec(self, stream: str) -> TensorSpec:
        """Build the (shared) spec for ``inputs.<stream>`` / ``normed.<stream>``.

        Returns
        -------
        TensorSpec
            ``("B", F)`` for the global object, ``("B", "T:<stream>", F)``
            for sequence streams, with the variable names as `fields`.
        """
        variables = tuple(self.norm.variables[stream])
        shape: tuple[int | str, ...] = (
            ("B", len(variables))
            if stream == self.global_object
            else ("B", _stream_len(stream), len(variables))
        )
        return TensorSpec(shape=shape, dtype="float32", fields=variables)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` -> ``normed.<stream>`` for every stream."""
        del mode
        return IO(
            requires=unflatten_spec({f"inputs.{s}": self._spec(s) for s in self.streams}),
            produces=unflatten_spec({f"normed.{s}": self._spec(s) for s in self.streams}),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Normalise every stream via the wrapped v1 `InputNorm`.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        del mode
        # fresh dict: InputNorm.forward rebinds its keys in place — bundle
        # leaves must never be re-bound.
        fresh = {stream: b.get(f"inputs.{stream}") for stream in self.streams}
        normed = self.norm(fresh)
        return {f"normed.{stream}": normed[stream] for stream in self.streams}


class StreamEmbed(nn.Module):
    """Wraps a v1 `InitNet`: per-stream initial embedding incl. global-context attach.

    The wrapped `InitNet.forward` reads ``inputs[input_name]`` and, when
    ``attach_global`` is set, PREPENDS the global object's features (feature
    order ``[global, stream]``) before the dense projection.
    """

    def __init__(self, init_net: InitNet) -> None:
        """Wrap a live v1 `InitNet` instance (weights shared, not copied).

        Raises
        ------
        ValueError
            If the init net uses positional encoding, featurewise transforms,
            or ``parameters`` variables — outside the GN2 parity scope.
        """
        super().__init__()
        self.name = _UNNAMED
        if init_net.pos_enc is not None:
            raise ValueError("StreamEmbed: pos_enc is outside the GN2 parity scope")
        if init_net.featurewise is not None:
            raise ValueError("StreamEmbed: featurewise is outside the GN2 parity scope")
        if "parameters" in init_net.variables:
            raise ValueError("StreamEmbed: 'parameters' variables are outside the parity scope")
        self.init_net = init_net
        self.stream = init_net.input_name
        self.context_stream = init_net.global_object if init_net.attach_global else None
        self.out_dim = init_net.net.output_size

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``normed.<stream>`` (+ global context) -> ``embed.<stream>``."""
        del mode
        variables = self.init_net.variables
        requires = {
            f"normed.{self.stream}": TensorSpec(
                shape=("B", _stream_len(self.stream), len(variables[self.stream])),
                dtype="float32",
            ),
        }
        if self.context_stream is not None:
            requires[f"normed.{self.context_stream}"] = TensorSpec(
                shape=("B", len(variables[self.context_stream])), dtype="float32"
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(
                    shape=("B", _stream_len(self.stream), self.out_dim), dtype="float32"
                ),
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Embed the stream via the wrapped v1 `InitNet` (context prepended inside).

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        del mode
        # fresh dict shaped like v1's post-norm `inputs`
        fresh = {self.stream: b.get(f"normed.{self.stream}")}
        if self.context_stream is not None:
            fresh[self.context_stream] = b.get(f"normed.{self.context_stream}")
        return {f"embed.{self.stream}": self.init_net(fresh)}


class Concat:
    """Concatenates embedded streams into one sequence: ``seq.x`` / ``seq.mask`` / ``seq.layout``.

    Stream order is THIS list and nothing else — it reproduces the v1
    concatenation order. Load-bearing only with >= 2 sequence streams; on
    the single-stream GN2 default any order is trivially the identity.

    Register tokens are NOT appended here — they stay internal to the
    wrapped v1 `Transformer`, which appends them after this concat exactly
    as v1 does. ``seq.layout`` is a dict-valued meta leaf
    ``{stream: (start, stop)}`` over the pre-register sequence, for `Split`.
    """

    def __init__(self, streams: Sequence[str]) -> None:
        """Configure the explicit stream order.

        Raises
        ------
        ValueError
            If `streams` is empty.
        """
        if not streams:
            raise ValueError("Concat: streams must be a non-empty sequence")
        self.name = _UNNAMED
        self.streams = tuple(streams)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``embed.*``/``masks.*`` per stream -> ``seq.x``/``seq.mask``/``seq.layout``."""
        del mode
        requires: dict[str, TensorSpec] = {}
        for stream in self.streams:
            requires[f"embed.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream), _EMBED_DIM), dtype="float32"
            )
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream)), dtype="bool", kind="pad_mask"
            )
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                "seq.x": TensorSpec(shape=("B", _SEQ_LEN, _EMBED_DIM), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                "seq.layout": TensorSpec(kind="meta"),
            }),
        )

    def __call__(self, b: Bundle, mode: Mode) -> dict:
        """Concatenate streams along the token dim and record the layout.

        Returns
        -------
        dict
            The newly produced keys only.
        """
        del mode
        xs = [b.get(f"embed.{stream}") for stream in self.streams]
        masks = [b.get(f"masks.{stream}") for stream in self.streams]
        layout: dict[str, tuple[int, int]] = {}
        start = 0
        for stream, x in zip(self.streams, xs, strict=True):
            layout[stream] = (start, start + x.shape[1])
            start += x.shape[1]
        # torch.cat allocates a new tensor even for one input — seq.x/seq.mask
        # never alias the embed.*/masks.* leaves.
        return {
            "seq.x": torch.cat(xs, dim=1),
            "seq.mask": torch.cat(masks, dim=1),
            "seq.layout": layout,
        }


class TransformerEncoder(nn.Module):
    """Wraps the v1 `Transformer` encoder, registers and all.

    The wrapped instance keeps its ``.registers`` parameter,
    ``.register_mask`` buffer, ``.out_proj`` and ``.out_norm`` internal.
    Register tokens are appended INSIDE the v1 forward and, with
    ``drop_registers=False``, the register row stays in ``encoded.seq`` —
    exactly the tensor v1 calls ``embed_xs``. The register pad mask is
    published as the new key ``masks.registers``.
    """

    def __init__(self, encoder: Transformer) -> None:
        """Wrap a live v1 `Transformer` instance (weights shared, not copied).

        Raises
        ------
        ValueError
            If the encoder uses edge features, featurewise transforms, or
            ``drop_registers`` — outside the GN2 parity scope.
        """
        super().__init__()
        self.name = _UNNAMED
        if encoder.edge_embed_dim != 0:
            raise ValueError("TransformerEncoder: edge features are outside the parity scope")
        if len(encoder.featurewise) > 0:
            raise ValueError(
                "TransformerEncoder: featurewise transforms are outside the parity scope "
                "(the wrapper does not forward v1's `inputs` argument)"
            )
        if encoder.drop_registers:
            raise ValueError("TransformerEncoder: drop_registers is outside the parity scope")
        self.encoder = encoder
        self.num_registers = encoder.num_registers
        self.embed_dim = encoder.embed_dim
        self.out_dim = encoder.out_dim

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``seq.x``/``seq.mask`` -> ``encoded.seq``/``masks.registers``."""
        del mode
        return IO(
            requires=unflatten_spec({
                "seq.x": TensorSpec(shape=("B", _SEQ_LEN, self.embed_dim), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
            }),
            produces=unflatten_spec({
                "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, self.out_dim), dtype="float32"),
                "masks.registers": TensorSpec(
                    shape=("B", self.num_registers), dtype="bool", kind="pad_mask"
                ),
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Encode the sequence via the wrapped v1 `Transformer`.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        del mode
        # fresh dicts: _add_registers INSERTS a "REGISTERS" key into both —
        # the insertion must never touch bundle leaves (write-once).
        xs: dict[str, Tensor] = {"seq": b.get("seq.x")}
        pad: dict[str, Tensor] = {"seq": b.get("seq.mask")}
        # v1's `inputs` arg is only read by featurewise transforms, which are
        # rejected in __init__, so it is omitted here.
        encoded, out_pad = self.encoder(xs, pad_mask=pad)
        return {"encoded.seq": encoded, "masks.registers": out_pad["REGISTERS"]}


class Split:
    """Per-stream slices of ``encoded.seq`` via the ``seq.layout`` meta leaf.

    NOT on the parity task path: the v1 task heads slice the full sequence
    internally via ``input_name_mask``, and pre-splitting changes GEMM
    shapes, which can break bitwise equality. `from_v1` therefore does not
    include a `Split`; it is provided standalone for other callers.
    """

    def __init__(self, streams: Sequence[str]) -> None:
        """Configure the streams to slice out.

        Raises
        ------
        ValueError
            If `streams` is empty.
        """
        if not streams:
            raise ValueError("Split: streams must be a non-empty sequence")
        self.name = _UNNAMED
        self.streams = tuple(streams)

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``encoded.seq`` + ``seq.layout`` -> ``encoded.<stream>`` per stream."""
        del mode
        return IO(
            requires=unflatten_spec({
                "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, _OUT_DIM), dtype="float32"),
                "seq.layout": TensorSpec(kind="meta"),
            }),
            produces=unflatten_spec({
                f"encoded.{stream}": TensorSpec(
                    shape=("B", _stream_len(stream), _OUT_DIM), dtype="float32"
                )
                for stream in self.streams
            }),
        )

    def __call__(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Slice each configured stream out of the encoded sequence.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        del mode
        encoded = b.get("encoded.seq")
        layout = b.get("seq.layout")
        out: dict[str, Tensor] = {}
        for stream in self.streams:
            start, stop = layout[stream]
            out[f"encoded.{stream}"] = encoded[:, start:stop]
        return out


class Pooling(nn.Module):
    """Wraps the v1 `GlobalAttentionPooling` over the full encoded sequence.

    v1 pools over a single-entry dict holding the encoded sequence, with the
    post-register pad-mask dict; the pooling concatenates mask values in
    dict order, so the reconstruction order streams-then-REGISTERS is
    bitwise-critical.
    """

    def __init__(self, pool_net: GlobalAttentionPooling) -> None:
        """Wrap a live v1 `GlobalAttentionPooling` instance (weights shared).

        Raises
        ------
        ValueError
            If `pool_net` is any other pooling flavour — e.g. `NodeQueryGAP`
            needs the maskformer ``objects`` stream, outside the parity scope.
        """
        super().__init__()
        self.name = _UNNAMED
        if not isinstance(pool_net, GlobalAttentionPooling):
            # scope guard, not a type-contract check
            raise ValueError(  # noqa: TRY004
                "Pooling: only GlobalAttentionPooling is in the GN2 parity scope, got "
                f"{type(pool_net).__name__}"
            )
        self.pool_net = pool_net

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``encoded.seq`` + masks -> ``pooled.global``."""
        del mode
        return IO(
            requires=unflatten_spec({
                "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, _OUT_DIM), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                "masks.registers": TensorSpec(shape=("B", _REG_LEN), dtype="bool", kind="pad_mask"),
            }),
            produces=unflatten_spec({
                "pooled.global": TensorSpec(shape=("B", _OUT_DIM), dtype="float32"),
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pool the encoded sequence via the wrapped v1 pooling net.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        del mode
        # v1 flattens the x dict by cat over its values — a single-entry
        # dict gives the same bits.
        x = {"seq": b.get("encoded.seq")}
        # mask dict order streams-then-REGISTERS matches v1's insertion
        # order; pooling cats mask values in dict order.
        pad = {"seq": b.get("seq.mask"), "REGISTERS": b.get("masks.registers")}
        return {"pooled.global": self.pool_net(x, pad_mask=pad)}


class GlobalObjectTask(nn.Module):
    """Wraps a v1 task head routed via the global representation.

    Reproduces the ``task.input_name == task.global_object`` branch of
    ``run_tasks``: ``task(pooled_global, labels, None, context=None)`` with
    ``labels=None``. Publishes the raw (unconverted) task output.
    """

    def __init__(self, task: TaskBase) -> None:
        """Wrap a live v1 task head instance (weights shared, not copied)."""
        super().__init__()
        self.name = _UNNAMED
        self.task = task
        self.stream = task.input_name

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``pooled.global`` -> ``preds.<stream>.<task>`` (raw logits)."""
        del mode
        return IO(
            requires=unflatten_spec({
                "pooled.global": TensorSpec(shape=("B", self.task.net.input_size), dtype="float32"),
            }),
            produces=unflatten_spec({
                f"preds.{self.stream}.{self.task.name}": TensorSpec(
                    shape=("B", self.task.net.output_size), dtype="float32"
                ),
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the wrapped v1 task head on the pooled global representation.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        del mode
        # labels=None here, so the returned loss is always None.
        preds, _loss = self.task(b.get("pooled.global"), None, None, context=None)
        return {f"preds.{self.stream}.{self.task.name}": preds}


class ConstituentTask(nn.Module):
    """Wraps a v1 constituent-level task head (classification or vertexing).

    Reproduces the default branch of ``run_tasks``:
    ``task(encoded_seq, labels, masks, context=pooled_global)`` with
    ``labels=None``. The wrapped task receives the FULL post-register
    sequence and slices its own stream internally via ``input_name_mask``
    — pre-splitting would change GEMM shapes and risk breaking bitwise
    parity. Publishes raw outputs (per-token logits incl. rows at padded
    positions, or ``[E, 1]`` edge scores).
    """

    def __init__(self, task: TaskBase, streams: Sequence[str]) -> None:
        """Wrap a live v1 task head; `streams` is the Concat layout order.

        Raises
        ------
        ValueError
            If `streams` is empty or does not contain the task's input stream.
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ValueError("ConstituentTask: streams must be a non-empty sequence")
        if task.input_name not in streams:
            raise ValueError(
                f"ConstituentTask: task {task.name!r} reads stream {task.input_name!r} "
                f"which is not in the sequence layout {tuple(streams)}"
            )
        self.task = task
        self.streams = tuple(streams)

    def _produced_spec(self) -> TensorSpec:
        """Build the spec of the task's raw output.

        Returns
        -------
        TensorSpec
            ``("E:<task>", 1)`` edge scores for `VertexingTask` (data-dependent
            edge count), else per-token logits ``("B", "T:<stream>", C)``.
        """
        if isinstance(self.task, VertexingTask):
            return TensorSpec(
                shape=(sym_dim("E", self.task.name), self.task.net.output_size),
                dtype="float32",
            )
        return TensorSpec(
            shape=("B", _stream_len(self.task.input_name), self.task.net.output_size),
            dtype="float32",
        )

    def declare_io(self, mode: Mode) -> IO:
        """Declare seq + per-stream masks + context -> ``preds.<stream>.<task>``."""
        del mode
        requires: dict[str, TensorSpec] = {
            "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, _OUT_DIM), dtype="float32"),
        }
        # EVERY layout stream's mask is required (not just the task's own):
        # input_name_mask needs each stream's token width.
        for stream in self.streams:
            requires[f"masks.{stream}"] = TensorSpec(
                shape=("B", _stream_len(stream)), dtype="bool", kind="pad_mask"
            )
        requires["masks.registers"] = TensorSpec(
            shape=("B", _REG_LEN), dtype="bool", kind="pad_mask"
        )
        requires["pooled.global"] = TensorSpec(shape=("B", _OUT_DIM), dtype="float32")
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                f"preds.{self.task.input_name}.{self.task.name}": self._produced_spec(),
            }),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the wrapped v1 task head on the full encoded sequence.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only.
        """
        del mode
        # Rebuild the post-encoder pad-mask dict: streams in concat order,
        # THEN "REGISTERS" — matches v1's insertion order. input_name_mask
        # concatenates per-stream widths in DICT ORDER, so this insertion
        # order is bitwise-critical: a reordered dict selects the WRONG
        # slice silently.
        masks: dict[str, Tensor] = {stream: b.get(f"masks.{stream}") for stream in self.streams}
        masks["REGISTERS"] = b.get("masks.registers")
        # full sequence tensor; the task slices its own stream internally.
        preds, _loss = self.task(b.get("encoded.seq"), None, masks, context=b.get("pooled.global"))
        return {f"preds.{self.task.input_name}.{self.task.name}": preds}
