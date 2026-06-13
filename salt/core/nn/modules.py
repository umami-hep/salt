"""Standalone, config-constructed GraphModules for the GN2v2 surface (design §5.1, §9.2).

M2 porting policy (plan 05): these modules are constructed from plain config
kwargs — no live v1 instances, no ``input_size`` arithmetic in YAML — but may
COMPOSE fresh v1 layer classes internally where that is faithful (full code
absorption is M7). What is new here is the lifecycle (design §2.3):

``__init__`` captures config only; ``declare_io`` is static; ``bind(schema)``
builds width-dependent layers from the `ResolvedSchema`; ``materialise()`` is
the only place file I/O happens (Normaliser values). Forward passes read
declared bundle keys and return ONLY newly produced keys — bundle values are
never mutated (write-once, design §2.1; the executor's debug mode enforces
it).

Key vocabulary follows design §5.1: ``inputs.<stream>`` -> ``normed.<stream>``
-> ``embed.<stream>`` -> ``seq.x``/``seq.mask``/``seq.layout`` ->
``encoded.seq`` (+ ``masks.registers``) -> ``encoded.<stream>`` (`Split`) /
``pooled.global``; task modules live in `salt.core.nn.tasks`.

Documented M2 deviations from design §5.1 (both honest, both encoder-driven):

- **Registers stay inside `TransformerEncoder`** (its composed v1
  `Transformer` appends them, transformer.py:679-681, and *requires*
  ``num_registers >= 1``). `Concat` accepts the design's ``registers:`` arg
  but rejects nonzero values until M7 absorbs the encoder.
- **`Normaliser` takes an explicit ``streams:`` list** instead of
  demand-driven ``normed.*`` narrowing: the M1 kernel supports wildcard
  *produces* for framework modules, but the matching ``inputs.<s>``
  *requires* cannot be narrowed (planner.py rejects wildcard requires).
  TODO(M3+): demand-driven narrowing once the kernel grows consumer-side
  framework wildcards (same gap as `LossSum.collect_loss_keys`).

Symbolic-dim convention: token-count dims are stream-scoped and shared
(``"T:tracks"``); feature-width dims are *instance-scoped*
(``"D:<instance>"``) so two modules' unknown widths never falsely unify —
`resolve_bind_schema` still resolves them through the specs observed on the
same key (bind.py).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import (
    IO,
    GraphModule,
    Mode,
    TensorSpec,
    flatten_spec,
    sym_dim,
    unflatten_spec,
)
from salt.core.nn.bind import ResolvedSchema

# composed v1 layers (M2 porting policy, plan 05 — absorbed at M7)
from salt.models import Dense as V1Dense
from salt.models import Transformer as V1Transformer
from salt.models.pooling import GlobalAttentionPooling as V1GlobalAttentionPooling
from salt.utils.tensor_utils import attach_context

__all__ = [
    "Concat",
    "GlobalAttentionPooling",
    "LossGLS",
    "LossSum",
    "Normaliser",
    "Split",
    "StreamEmbed",
    "TransformerEncoder",
    "VectorConcat",
]

_UNNAMED = "unnamed"
"""Placeholder instance name — the config dict key is assigned before compile (design §2.2)."""

_SEQ_LEN = sym_dim("S", "seq")
_ENC_LEN = sym_dim("L", "enc")


def _stream_len(stream: str) -> str:
    """Return the shared symbolic token-count dim for a sequence stream.

    Returns
    -------
    str
        The symbolic dim, e.g. ``"T:tracks"``.
    """
    return sym_dim("T", stream)


class Normaliser(nn.Module):
    """Config-constructed input normalisation (design §6.3, replaces v1 `InputNorm`).

    Lifecycle (design §2.3): ``__init__`` records the ``norm_dict`` *path*
    and the stream list — no file I/O. ``bind(schema)`` allocates per-stream
    buffers (``means_<stream>`` / ``stds_<stream>``, the design §6.3 names)
    sized from the resolved ``inputs.<stream>`` widths, and captures the
    declared variable names. ``materialise()`` is the ONLY file-touching
    hook: it loads the norm dict and fills the buffers — skipped on
    checkpoint load, where values arrive via the state_dict (the
    ``materialised`` flag buffer travels with them).

    Unlike v1's ``InputNorm.forward`` (which rebinds its input dict's keys
    in place, inputnorm.py:103-106), this module produces NEW
    ``normed.<stream>`` keys and never mutates ``inputs.*`` (design §2.1).
    Because ``normed.*`` and ``inputs.*`` are distinct keys, consumers of
    raw inputs (edge features) need no ordering hack (design §6.3).
    """

    def __init__(
        self,
        norm_dict: str | Path,
        streams: Sequence[str],
        global_object: str | None = None,
    ) -> None:
        """Capture config only (design §2.3 — no file I/O here).

        Parameters
        ----------
        norm_dict : str | Path
            Path to the normalisation dictionary YAML; read at
            `materialise`, never here.
        streams : Sequence[str]
            Streams to normalise (explicit M2 surface — see the module
            docstring for the demand-driven TODO).
        global_object : str | None, optional
            The stream that is a per-object vector (``[B, F]``) rather than
            a padded sequence (``[B, T, F]``), by default None.

        Raises
        ------
        ConfigError
            If `streams` is empty, contains duplicates, or `global_object`
            is not one of them.
        """
        super().__init__()
        self.name = _UNNAMED
        if not streams:
            raise ConfigError("Normaliser: streams must be a non-empty sequence")
        if len(set(streams)) != len(tuple(streams)):
            raise ConfigError(f"Normaliser: duplicate streams in {tuple(streams)}")
        if global_object is not None and global_object not in streams:
            raise ConfigError(
                f"Normaliser: global_object {global_object!r} is not in streams {tuple(streams)}"
            )
        self.norm_dict_path = Path(norm_dict)
        self.streams = tuple(streams)
        self.global_object = global_object
        self._fields: dict[str, tuple[str, ...]] = {}
        self._bound = False

    def _spec(self, stream: str) -> TensorSpec:
        """Build the shared spec for ``inputs.<stream>`` / ``normed.<stream>``.

        The last dim is the instance-scoped symbol ``F:<name>.<stream>`` on
        BOTH sides, so the concrete width declared by the dataset boundary
        propagates to ``normed.<stream>`` through unification (bind.py).

        Returns
        -------
        TensorSpec
            ``("B", F)`` for the global object, ``("B", "T:<stream>", F)``
            for sequence streams.
        """
        width = sym_dim("F", f"{self.name}.{stream}")
        shape: tuple[int | str, ...] = (
            ("B", width) if stream == self.global_object else ("B", _stream_len(stream), width)
        )
        return TensorSpec(shape=shape, dtype="float32")

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``inputs.<stream>`` -> ``normed.<stream>`` for every stream.

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        return IO(
            requires=unflatten_spec({f"inputs.{s}": self._spec(s) for s in self.streams}),
            produces=unflatten_spec({f"normed.{s}": self._spec(s) for s in self.streams}),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Allocate normalisation buffers from the resolved schema (config-only).

        Buffers are named ``means_<stream>`` / ``stds_<stream>`` (design
        §6.3 checkpoint layout) and initialised to identity (0/1); the
        boolean ``materialised`` buffer guards against silently training on
        un-normalised values. Field names are captured here for
        `materialise`'s per-variable lookup; a `BindError` propagates from
        the schema if a stream's width or fields are not statically
        resolved.

        Raises
        ------
        RuntimeError
            If called twice (rebinding would discard loaded values).
        """
        if self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: bind() called twice (design §2.3)")
        for stream in self.streams:
            key = f"inputs.{stream}"
            width = schema.width(key)
            self._fields[stream] = schema.fields_of(key)
            self.register_buffer(f"means_{stream}", torch.zeros(width))
            self.register_buffer(f"stds_{stream}", torch.ones(width))
        self.register_buffer("materialised", torch.tensor(False))
        self._bound = True

    def preflight(self) -> None:
        """Fail-fast, data-free norm-dict validation (M3 leftover, design §2.3).

        Without this check a wrong ``norm_dict`` path/content only surfaces
        at `materialise` — after dataset setup and plan compilation. The
        preflight reads ONLY the norm-dict YAML (config I/O in the design
        §2.6 sense — no H5/bind I/O): the file must exist, parse, and carry
        every configured stream; when the module is already bound (the
        `SaltModule.setup` call site) the per-variable mean/std entries are
        checked too, mirroring `materialise`'s validation. Called by
        `SaltModule.setup` on fresh fits (hard error) and by ``salt2 graph
        validate`` (warning — data-less machines stay supported).

        Raises
        ------
        ConfigError
            On a missing/unparsable norm dict, a missing stream, or (when
            bound) missing/non-finite/zero-std variable entries.
        """
        path = self.norm_dict_path
        prefix = f"Normaliser {self.name!r} preflight"
        fix = (
            f"  fix: point model.modules.{self.name}.init_args.norm_dict at the "
            "preprocessing norm_dict.yaml for this sample"
        )
        if not path.is_file():
            raise ConfigError(f"{prefix}: norm dict not found: {path}\n{fix}")
        try:
            with open(path) as fh:
                norm_dict = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ConfigError(
                f"{prefix}: norm dict {path} is not valid YAML: {err}\n{fix}"
            ) from err
        if not isinstance(norm_dict, dict):
            raise ConfigError(f"{prefix}: norm dict {path} must be a mapping\n{fix}")
        for stream in self.streams:
            if stream not in norm_dict:
                raise ConfigError(
                    f"{prefix}: missing input type {stream!r} in {path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            if not self._fields:
                continue  # unbound (the data-free `salt2 graph validate` path)
            variables = self._fields[stream]
            if missing := set(variables) - set(norm_dict[stream]):
                raise ConfigError(
                    f"{prefix}: missing variables {sorted(missing)} for {stream!r} in {path}. "
                    f"Choose from {sorted(norm_dict[stream])}.\n"
                    f"  fix: add mean/std entries for {sorted(missing)} to {path}, or remove "
                    f"them from the features variable list "
                    f"(config: data.modules.features.init_args.variables.{stream})"
                )
            for variable in variables:
                entry = norm_dict[stream][variable]
                try:
                    mean, std = float(entry["mean"]), float(entry["std"])
                except (KeyError, TypeError, ValueError):
                    raise ConfigError(
                        f"{prefix}: entry for {stream}.{variable} in {path} must be a "
                        f"{{mean, std}} mapping, got {entry!r}"
                    ) from None
                if not (torch.isfinite(torch.tensor(mean)) and torch.isfinite(torch.tensor(std))):
                    raise ConfigError(
                        f"{prefix}: non-finite normalisation parameters for "
                        f"{stream}.{variable} in {path}."
                    )
                if std == 0:
                    raise ConfigError(
                        f"{prefix}: zero standard deviation for {stream}.{variable} in {path}."
                    )

    def materialise(self) -> None:
        """Fill the buffers from the norm dict (the ONLY file I/O, design §2.3).

        Mirrors v1 `InputNorm`'s validation (inputnorm.py:56-88): missing
        streams/variables, non-finite values, and zero stds are errors.

        Raises
        ------
        RuntimeError
            If called before `bind`.
        ValueError
            If the norm dict is missing this module's streams or variables,
            or contains non-finite means/stds or zero stds.
        """
        if not self._bound:
            raise RuntimeError(f"Normaliser {self.name!r}: materialise() before bind()")
        with open(self.norm_dict_path) as fh:
            norm_dict = yaml.safe_load(fh)
        for stream in self.streams:
            if stream not in norm_dict:
                raise ValueError(
                    f"Missing input type {stream!r} in {self.norm_dict_path}. "
                    f"Choose from {sorted(norm_dict)}."
                )
            variables = self._fields[stream]
            if missing := set(variables) - set(norm_dict[stream]):
                raise ValueError(
                    f"Missing variables {sorted(missing)} for {stream!r} in "
                    f"{self.norm_dict_path}. Choose from {sorted(norm_dict[stream])}.\n"
                    f"  fix: add mean/std entries for {sorted(missing)} to "
                    f"{self.norm_dict_path}, or remove them from the features variable "
                    f"list (config: data.modules.features.init_args.variables.{stream})"
                )
            means = torch.as_tensor(
                [float(norm_dict[stream][v]["mean"]) for v in variables], dtype=torch.float32
            )
            stds = torch.as_tensor(
                [float(norm_dict[stream][v]["std"]) for v in variables], dtype=torch.float32
            )
            if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
                raise ValueError(
                    f"Non-finite normalisation parameters for {stream!r} in {self.norm_dict_path}."
                )
            if (stds == 0).any():
                raise ValueError(
                    f"Zero standard deviation for {stream!r} in {self.norm_dict_path}."
                )
            with torch.no_grad():
                getattr(self, f"means_{stream}").copy_(means)
                getattr(self, f"stds_{stream}").copy_(stds)
        self.materialised.fill_(True)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Produce ``normed.<stream> = (inputs.<stream> - means) / stds``.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5); inputs are never
            mutated.

        Raises
        ------
        RuntimeError
            If the buffers were never materialised (fresh fit without
            `materialise`) — identity values would silently train
            un-normalised.
        """
        del mode
        # skip under tracing: the tensor->bool read would emit a spurious
        # TracerWarning on every export. Tracing is still guarded —
        # OnnxAdapter rejects unmaterialised modules at construction, before
        # any trace (the eager path keeps this check).
        if not torch.jit.is_tracing() and not bool(self.materialised):
            raise RuntimeError(
                f"Normaliser {self.name!r}: forward before materialise() — on a fresh fit "
                "call materialise(); on checkpoint load the state_dict provides the values "
                "(design §2.3)"
            )
        return {
            f"normed.{s}": (b.get(f"inputs.{s}") - getattr(self, f"means_{s}"))
            / getattr(self, f"stds_{s}")
            for s in self.streams
        }


class StreamEmbed(nn.Module):
    """Config-constructed per-stream initial embedding (design §9.2, replaces v1 `InitNet`).

    Composes a fresh v1 `Dense` built at `bind` — the dense input width is
    inferred from the resolved input and context widths, never configured
    (design §2.3 kills ``input_size`` YAML arithmetic, initnet.py:54-59).

    Context entries are attached with v1's ``attach_context`` semantics
    (tensor_utils.py:240: ``cat([context, x])`` — context PREPENDED), applied
    in list order; the final feature layout is therefore
    ``[ctx[-1], ..., ctx[0], stream]``. With the single GN2 entry
    ``context: [normed.jets]`` this reproduces v1's ``[global, stream]``
    column order exactly (checkpoint layout, design §5.1).

    TODO(M6): ``mup:`` flag; TODO(M7): pos_enc / featurewise blocks
    (design §6.4) when `InitNet` is absorbed.
    """

    def __init__(
        self,
        stream: str,
        out_dim: int,
        dense: dict[str, Any] | None = None,
        context: Sequence[str] = (),
        input: str | None = None,  # noqa: A002 - design §5.1 YAML surface name
    ) -> None:
        """Capture config only (design §2.3).

        Parameters
        ----------
        stream : str
            The sequence stream to embed.
        out_dim : int
            Output embedding width (concrete, config-fixed).
        dense : dict[str, Any] | None, optional
            Extra kwargs for the internal v1 `Dense` (``hidden_layers``,
            ``activation``, ...); must not contain width keys, by default
            None.
        context : Sequence[str], optional
            Dotted bundle keys attached as context, in v1 prepend order (see
            class docstring), by default ``()``.
        input : str | None, optional
            Input key override, by default ``normed.<stream>``.

        Raises
        ------
        ConfigError
            If `dense` configures widths (``input_size`` etc. — inferred at
            bind, design §2.3) or `out_dim` is not positive.
        """
        super().__init__()
        self.name = _UNNAMED
        if out_dim < 1:
            raise ConfigError(f"StreamEmbed: out_dim must be >= 1, got {out_dim}")
        _reject_width_keys("StreamEmbed", dense, ("input_size", "output_size", "context_size"))
        self.stream = stream
        self.out_dim = out_dim
        self.dense_cfg = dict(dense or {})
        self.context = tuple(context)
        self.input_key = input if input is not None else f"normed.{stream}"
        self.net: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + context keys -> ``embed.<stream>``.

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        requires: dict[str, TensorSpec] = {
            self.input_key: TensorSpec(
                shape=("B", _stream_len(self.stream), sym_dim("F", self.name)), dtype="float32"
            ),
        }
        for key in self.context:
            # rank/width unconstrained here: context may be a [B, F] global
            # vector or broadcastable — widths resolve from the producer side
            requires[key] = TensorSpec(shape=None, dtype="float32")
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(
                    shape=("B", _stream_len(self.stream), self.out_dim), dtype="float32"
                ),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the internal `Dense` with the inferred input width (design §2.3).

        ``input_size = width(input) + sum(width(ctx))`` — exactly v1's
        inference (initnet.py:54-59) driven by the resolved schema instead
        of CLI variable injection.
        """
        input_size = schema.width(self.input_key) + sum(schema.width(key) for key in self.context)
        self.net = V1Dense(input_size=input_size, output_size=self.out_dim, **self.dense_cfg)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Attach context (prepended, v1 order) and project.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        del mode
        x = b.get(self.input_key)
        for key in self.context:
            x = attach_context(x, b.get(key))  # cat([context, x]) — tensor_utils.py:240
        assert self.net is not None, "forward before bind()"
        return {f"embed.{self.stream}": self.net(x)}


class Concat(nn.Module):
    """Concatenate embedded streams into one sequence (design §5.1).

    Produces ``seq.x`` / ``seq.mask`` / ``seq.layout``; the concat ORDER is
    the configured list and nothing else (v1 relied on init_nets dict
    insertion order, saltmodel.py:128-129 / transformer.py:684-686).
    ``seq.layout`` is a dict-valued meta leaf ``{stream: (start, stop)}``
    over the pre-register sequence, consumed by `Split`.

    M2 deviation (documented in the module docstring): the design's
    ``registers:`` belong here, but the composed v1 `Transformer` appends
    its own register tokens internally and requires ``num_registers >= 1``
    — so nonzero ``registers`` is rejected until M7 absorbs the encoder.
    """

    def __init__(self, streams: Sequence[str], registers: int = 0) -> None:
        """Capture the explicit stream order (design §5.1).

        Raises
        ------
        ConfigError
            If `streams` is empty or contains duplicates, or if `registers`
            is nonzero (M2: registers live in `TransformerEncoder`, see
            class docstring).
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
        boundary table consumed by `Split`'s export branch, design §7).

        Returns
        -------
        IO
            The declared requires/produces.
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

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Any]:
        """Concatenate streams along the token dim and record the layout.

        In ONNX mode the per-stream token counts are ALSO published as the
        ``seq.offsets`` cumulative-boundary tensor, built with
        ``torch.onnx.operators.shape_as_tensor`` on the per-stream pad masks
        so the boundaries trace to Shape/Concat/CumSum nodes that stay
        symbolic under ``dynamo=False`` tracing (design §7 multi-stream
        trace-safety; the mode branch itself is static Python). Eager
        FIT/VAL/TEST numerics are untouched.

        Returns
        -------
        dict[str, Any]
            The newly produced keys only (design §2.5). ``torch.cat``
            allocates fresh tensors even for one input, so the seq leaves
            never alias the per-stream leaves.
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


class VectorConcat(nn.Module):
    """Ordered concatenation of ``[B, D_i]`` vectors into one ``[B, Dsum]`` key (design §6.6).

    The v2 spelling of v1's post-pooling ``'global'`` magic key
    (saltmodel.py:175-177): ``global_rep = cat([global_rep, global_feats],
    dim=-1)``. v1 fed GN3's 2-feature ``global`` stream PAST the encoder and
    concatenated it onto the pooled representation, producing the
    ``&pooled_dim 258`` (256 + 2) every GN3 task consumes. The synthesis killed
    ``'global'`` as a magic key without a replacement — a silent feature-drop
    hazard (both design critics rated HIGH: all widths still unify after the
    drop, so no ``ShapeError`` fires). This module is the explicit replacement.

    There is **no v1 class** for this — v1 inlined the cat in
    ``SaltModel.forward``. The nearest v1 primitives are the ``merge_dict``
    stream merge (saltmodel.py:135-147, sequence-level, code-only) and
    ``InitNet.attach_global`` (initnet.py:77-82, context prepend); neither is a
    standalone post-pooling vector concat. The correctness anchors are therefore
    design-conformance, not v1-byte-parity (plan 10 sub-wave B, L3):

    1. **Concat ORDER is the config list.** For converted GN3 checkpoints the
       converter emits ``inputs: [pooled.global, normed.global]`` (pooled first,
       global features last) so the task first-layer weight layouts line up with
       v1's ``cat([global_rep, global_feats])`` (design §6.6 1390-1391). The
       order is the configured list and nothing else.
    2. **Output width ``Dsum = sum(D_i)``.** Resolved at bind via
       `derived_widths` (there is no concrete edge for the dim table to bind
       ``Dsum`` to — only the concat knows the sum; design §6.6 "Dsum unified at
       bind", bind.py second pass).
    3. **ONNX is the ``alias:`` mechanism, not this module.** Athena feeds ONE
       jet tensor that v1 clones into ``global`` (to_onnx.py:377-378); v2
       reproduces that with ``export.inputs`` ``alias:`` (`OnnxAdapter`,
       onnx/adapter.py, design §6.6/§7) — a name-resolved column gather binding
       the aliased port from the source tensor. VectorConcat itself is
       mode-agnostic: ``torch.cat`` over ``[B, D]`` vectors has no dynamic
       feature axis, so the forward is identical in every mode (unlike
       sequence-level `Concat`/`Split`, which need ``seq.offsets`` trace-safety).

    4 configs depend on it: GN2emu, GN3V01, GN3_SoftE, GN3EPCLV01.
    """

    def __init__(self, inputs: Sequence[str], out: str = "pooled.global") -> None:
        """Capture the explicit ordered input list and the output key (design §6.6).

        Parameters
        ----------
        inputs : Sequence[str]
            Dotted bundle keys to concatenate, in the EXACT order they appear
            in the output (pooled first for converted GN3 checkpoints, design
            §6.6 1390-1391). Must be non-empty with no duplicates.
        out : str, optional
            The produced concatenated key, by default ``"pooled.global"`` (the
            v1 ``global_rep`` slot every GN3 task reads).

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
        """Declare each ``[B, D_i]`` input -> the ``[B, Dsum]`` output (design §6.6).

        Each input gets its OWN instance-scoped width symbol (the inputs are
        genuinely different widths — pooled 256 vs global 2 — so they must NOT
        share a symbol). The output's ``Dsum`` symbol is resolved at bind from
        the sum of the resolved input widths (`derived_widths`); the dim table
        alone cannot bind it (no concrete edge), which is exactly why the
        bind-time second pass exists (design §6.6 "Dsum unified at bind").

        Returns
        -------
        IO
            The declared requires/produces.
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
        """Contribute ``Dsum = sum(D_i)`` once every input width is resolved (design §6.6).

        Called by `resolve_bind_schema`'s second pass with the widths resolved
        so far. Returns the output width when ALL inputs are known, else an
        empty dict (a mode where some input is absent — the hook is
        order-insensitive across plans). This is the ONLY place the concat sum
        is known; the union-find dim table cannot infer it from edges alone.

        Returns
        -------
        dict[str, int]
            ``{out: sum(widths[input])}`` when every input width is resolved,
            otherwise ``{}``.
        """
        if all(key in widths for key in self.inputs):
            return {self.out_key: sum(widths[key] for key in self.inputs)}
        return {}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Concatenate the inputs along the feature dim, in the configured order.

        Mode-agnostic: ``[B, D]`` vectors have no dynamic feature axis, so the
        same ``torch.cat`` traces correctly for ONNX (design §6.6). ``torch.cat``
        allocates a fresh tensor, so the output never aliases an input leaf
        (design §2.1 write-once).

        Returns
        -------
        dict[str, Tensor]
            The newly produced key only (design §2.5).
        """
        del mode
        return {self.out_key: torch.cat([b.get(key) for key in self.inputs], dim=-1)}


class TransformerEncoder(nn.Module):
    """Config-constructed transformer encoder (design §9.2, composes a fresh v1 `Transformer`).

    Everything width-relevant is config (``dim``, ``out_dim``), so the
    composed v1 instance is built in ``__init__`` (design §2.3 permits
    config-only layer construction there; `bind` is for schema-derived
    widths). Registers, packing, and the out projection stay INTERNAL
    (design §2.5 composite modules); the register pad mask is published as
    the NEW key ``masks.registers`` — the caller's mask dict is never
    mutated (v1's ``_add_registers`` inserts ``"REGISTERS"`` into it,
    transformer.py:777,785).

    ``drop_registers`` (the MaskFormer encoder passthrough, M5 sub-wave C;
    v1 transformer.py:560,584,745-750) makes the composed v1 `Transformer`
    strip the register rows from its output AFTER the layer stack runs: the
    registers are still appended internally and visible to every attention
    layer (they exist so a constituent-less jet has SOMETHING to attend to,
    transformer.py:569-574), but the returned ``encoded.seq`` is sliced back
    to the stream tokens only (``x[:, :-num_registers]``) and v1 drops the
    ``"REGISTERS"`` pad entry (``del pad_mask["REGISTERS"]``,
    transformer.py:745-750). So with ``drop_registers=True`` this module
    produces NO ``masks.registers`` key — there are no register rows left in
    ``encoded.seq`` for a downstream consumer to mask — exactly the v1
    MaskFormer encoder shape its `MaskDecoder` reads (``embed_xs`` with the
    registers already gone, maskformer.py:151-156,172). The shipped
    MaskFormer.yaml sets ``drop_registers: true`` (MaskFormer.yaml:31). The
    design's longer-term home for this is a `Concat`-owned ``registers:`` +
    ``drop_registers_after:`` (FD 1121-1125); in the M2 architecture
    registers live INSIDE this encoder (the composed v1 `Transformer`
    appends them and requires ``num_registers >= 1``), so the M5 passthrough
    is the faithful, byte-for-byte v1 mechanism — the Concat-owned form lands
    when M7 absorbs the encoder.

    ``norm_type`` (``"pre"`` default, or ``"post"`` / ``"hybrid"``) is
    forwarded verbatim to every composed v1 ``EncoderLayer`` (M5 sub-wave B;
    the GN3V01 flagship + GN3_Hybrid / GN3EPCLV01 are ``"hybrid"``). The
    placement logic — hybrid forcing ``do_qk_norm``/``do_v_norm``, the
    depth-0 residual-norm special case, and the pre-FFN norm — all lives in
    the composed v1 layer (transformer.py:350-356,421); the wrapper only
    threads the flag through the `Transformer` ``**kwargs`` passthrough.

    TODO(M6): ``mup:`` flag; TODO(M7): ``featurewise:`` / ``edges:`` ports
    (design §6.4, §6.7) when the v1 internals are absorbed.
    """

    _NORM_TYPES = ("pre", "post", "hybrid")
    """The encoder-layer norm placements the wrapper forwards (v1 EncoderLayer,
    transformer.py:307-308,350-356). ``"none"`` is a residual-only v1 mode with
    no shipped v2 config — rejected loudly here rather than silently passed."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        attention: dict[str, Any],
        out_dim: int | None = None,
        dense: dict[str, Any] | None = None,
        norm: str = "LayerNorm",
        num_registers: int = 1,
        norm_type: str = "pre",
        drop_registers: bool = False,
    ) -> None:
        """Build the composed v1 `Transformer` from config.

        Parameters
        ----------
        dim : int
            Embedding width of ``seq.x``.
        num_layers : int
            Number of encoder layers.
        attention : dict[str, Any]
            Attention config; MUST contain ``num_heads``. ``attn_type``
            (default ``"torch-math"``) selects the backend; the remaining
            keys are v1 ``attn_kwargs`` (REQUIRED by v1: transformer.py
            writes ``attn_type`` into them, transformer.py:600-601).
        out_dim : int | None, optional
            Output projection width, by default None (= `dim`, no
            projection).
        dense : dict[str, Any] | None, optional
            v1 ``dense_kwargs`` (``activation``, ``gated``, ...), by
            default None.
        norm : str, optional
            Normalisation layer name, by default ``"LayerNorm"``.
        num_registers : int, optional
            Learned register tokens appended INSIDE the encoder, by default
            1 (v1 minimum — transformer.py:558-559).
        norm_type : str, optional
            Per-layer norm placement, one of ``{"pre", "post", "hybrid"}``,
            by default ``"pre"``. Forwarded verbatim to every v1
            ``EncoderLayer`` via the `Transformer` ``**kwargs`` passthrough
            (transformer.py:611). ``"hybrid"`` (the GN3V01 flagship) makes the
            EncoderLayer force ``do_qk_norm``/``do_v_norm`` on its `Attention`,
            use a residual ``norm_type`` of ``"pre"`` at depth 0 / ``"none"``
            after, and apply a pre-FFN norm in ``forward`` (transformer.py:
            350-356,421). The wrapper only passes the flag through — all that
            placement logic lives in the composed v1 layer.
        drop_registers : bool, optional
            Strip the register rows from ``encoded.seq`` after the layer stack
            (the MaskFormer encoder passthrough; v1 transformer.py:745-750), by
            default False. Registers stay visible to every attention layer; only
            the OUTPUT sequence is sliced back to the stream tokens, and no
            ``masks.registers`` key is produced (see the class docstring).

        Raises
        ------
        ConfigError
            If `attention` is missing ``num_heads``, or `norm_type` is not one
            of ``{"pre", "post", "hybrid"}``.
        """
        super().__init__()
        self.name = _UNNAMED
        if not isinstance(attention, Mapping) or "num_heads" not in attention:
            raise ConfigError(
                "TransformerEncoder: attention config must be a mapping containing 'num_heads' "
                "(v1 Transformer requires attn_kwargs, transformer.py:600-601)"
            )
        if norm_type not in self._NORM_TYPES:
            raise ConfigError(
                f"TransformerEncoder: norm_type must be one of {self._NORM_TYPES}, got "
                f"{norm_type!r}"
            )
        attn_kwargs = dict(attention)
        attn_type = attn_kwargs.pop("attn_type", "torch-math")
        self.dim = dim
        self.norm_type = norm_type
        self.drop_registers = bool(drop_registers)
        self.encoder = V1Transformer(
            num_layers=num_layers,
            embed_dim=dim,
            out_dim=out_dim,
            norm=norm,
            attn_type=attn_type,
            do_final_norm=True,
            num_registers=num_registers,
            drop_registers=self.drop_registers,
            attn_kwargs=attn_kwargs,
            dense_kwargs=dict(dense) if dense is not None else None,
            norm_type=norm_type,
        )
        self.out_dim = self.encoder.out_dim
        self.num_registers = num_registers

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``seq.x``/``seq.mask`` -> ``encoded.seq`` (+ ``masks.registers``).

        ``masks.registers`` is produced ONLY when ``drop_registers`` is False:
        with the registers dropped from ``encoded.seq`` there are no register
        rows left to mask (the v1 ``del pad_mask["REGISTERS"]`` shape,
        transformer.py:745-750), and a downstream `GlobalAttentionPooling`
        treats ``masks.registers`` as OPTIONAL — so a drop-registers config
        plan-compiles exactly like the encoder-less path (modules.py:1065-1070).

        Returns
        -------
        IO
            The declared requires/produces (widths concrete from config).
        """
        del mode
        produces: dict[str, TensorSpec] = {
            "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, self.out_dim), dtype="float32"),
        }
        if not self.drop_registers:
            produces["masks.registers"] = TensorSpec(
                shape=("B", self.num_registers), dtype="bool", kind="pad_mask"
            )
        return IO(
            requires=unflatten_spec({
                "seq.x": TensorSpec(shape=("B", _SEQ_LEN, self.dim), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
            }),
            produces=unflatten_spec(produces),
        )

    def set_export_mode(self) -> None:
        """Force the deterministic torch-math backend (design §2.5, test/ONNX semantics)."""
        self.encoder.set_backend("torch-math")

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Encode the sequence; publish the register mask as a NEW key.

        With ``drop_registers`` the composed v1 `Transformer` strips the
        register rows from its output and deletes the ``"REGISTERS"`` pad entry
        (transformer.py:745-750), so only ``encoded.seq`` is produced — no
        ``masks.registers`` (the produce is gated out in `declare_io`).

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        del mode
        # FRESH dicts: v1's _add_registers INSERTS a "REGISTERS" key into
        # both (transformer.py:777,785) — never hand it bundle-owned dicts.
        xs: dict[str, Tensor] = {"seq": b.get("seq.x")}
        pad: dict[str, Tensor] = {"seq": b.get("seq.mask")}
        encoded, out_pad = self.encoder(xs, pad_mask=pad)
        if self.drop_registers:
            # registers stripped from encoded.seq; v1 also removed "REGISTERS"
            # from the pad dict, so there is no register mask to publish
            return {"encoded.seq": encoded}
        return {"encoded.seq": encoded, "masks.registers": out_pad["REGISTERS"]}


class Split(nn.Module):
    """Per-stream slices of ``encoded.seq`` via the ``seq.layout`` meta leaf.

    The PRODUCTION task path (design §3.3): tasks consume per-stream
    ``encoded.<stream>`` tensors instead of reconstructing slices from
    pad-mask dict order (v1 ``input_name_mask``, task.py:58-78). Register
    rows sit AFTER every stream in the encoder output, so the pre-register
    layout offsets remain valid slices of ``encoded.seq``.

    Export-mode implementation (design §7 / risk 7, adjudicated empirically
    in the M4 recipe spike, 2026-06-12): the eager branch slices with
    Python-int ``seq.layout`` offsets, which ``dynamo=False`` tracing bakes
    as constants. A single-stream probe (L=0..60) showed the JIT tracer
    keeps ``size()``-derived ints symbolic through single-stream slicing —
    silently CORRECT for one dynamic sequence axis — but a two-stream probe
    mis-sliced at ALL 15 (L_trk, L_el) grid points: with >=2 dynamic axes
    the baked offsets are provably wrong. The ONNX branch therefore slices
    with ``index_select`` over an index range built from the `Concat`
    ``seq.offsets`` tensor (``shape_as_tensor``-derived, design §7
    mechanism (A)) — proven correct on the full two-axis grid including
    zero-length streams in the same spike; the design's fallback
    (per-stream encoder outputs) was NOT needed. The mode branch is static
    Python; eager FIT/VAL/TEST numerics are bit-identical to the M2 port,
    and ``index_select`` over a contiguous range equals the eager narrow
    slicing exactly.
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
        additionally required (the trace-safe slicing path, see the class
        docstring).

        Returns
        -------
        IO
            The declared requires/produces (one shared instance-scoped
            width symbol — slicing preserves the feature dim).
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
        ``seq.offsets`` (see the class docstring for the risk-7 evidence);
        the stream's position in the offsets table is its position in the
        ``seq.layout`` dict (static Python — `Concat` insertion order), so
        a `Split` over a stream SUBSET stays correct without knowing the
        full concat list.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
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


class GlobalAttentionPooling(nn.Module):
    """Config-constructed global attention pooling (design §5.1, explicit input/out ports).

    Composes a fresh v1 `GlobalAttentionPooling` built at `bind` (gate width
    inferred from the resolved input). Pools the register-augmented
    sequence with the post-register mask order streams-then-REGISTERS —
    exactly where v1's ``_add_registers`` leaves the dict
    (transformer.py:777,785; pooling cats mask values in dict order,
    pooling.py:56). The zero-token ONNX pad stays inside the composed v1
    class (pooling.py:59-63).

    Two wirings, one class (design §5.1; v1 saltmodel.py:90-93,155-156,
    169-170):

    - **With an encoder** (the GN2 path): ``input`` is ``encoded.seq`` and
      `TransformerEncoder` publishes ``masks.registers`` — the register rows
      sit after every stream, so the pad dict is streams-then-REGISTERS.
    - **Encoder-less** (M5; the DiPS/DeepSets family, every regression
      config + ``legacy/dips.yaml``): ``init_nets`` + ``pool_net`` with NO
      ``encoder:`` block, so nothing produces ``masks.registers`` (v1
      saltmodel.py:155-156 pools ``flatten_tensor_dict(xs)`` directly). The
      config points ``input`` at ``seq.x`` (the `Concat` output) and
      ``masks.registers`` is declared OPTIONAL — absent from the plan when no
      producer exists (planner `_collect_demand` skips optional requires,
      `_build_edges` binds no edge), so the config plan-compiles and the pad
      dict is just ``{"seq": seq.mask}``. v1's pooling cats mask values in
      dict order, so dropping the REGISTERS entry is the exact v1
      encoder-less semantics — NOT an approximation. The WITH-encoder path is
      untouched: when the encoder produces ``masks.registers`` the optional
      require still binds and the REGISTERS pad row is still consumed.
    """

    def __init__(self, input: str = "encoded.seq", out: str = "pooled.global") -> None:  # noqa: A002 - design §5.1 YAML surface name
        """Capture the explicit input/output ports (design §5.1)."""
        super().__init__()
        self.name = _UNNAMED
        self.input_key = input
        self.out_key = out
        self.pool_net: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + masks -> the pooled vector.

        The input's width symbol is shared with the produced key, so the
        pooled width resolves from the producing module's declaration.

        Returns
        -------
        IO
            The declared requires/produces.
        """
        del mode
        width = sym_dim("D", self.name)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", sym_dim("L", self.name), width), dtype="float32"
                ),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
                # OPTIONAL: produced by `TransformerEncoder` on the WITH-encoder
                # path, ABSENT on the encoder-less path (init_nets+pool_net, no
                # encoder — v1 saltmodel.py:90-93,155-156). Optional means the
                # planner drops it when no module produces it, so encoder-less
                # configs plan-compile (design §2.2; M5 encoder-less pooling).
                "masks.registers": TensorSpec(
                    shape=("B", sym_dim("R", "registers")),
                    dtype="bool",
                    kind="pad_mask",
                    optional=True,
                ),
            }),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", width), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the composed v1 pooling with the inferred gate width (design §2.3)."""
        self.pool_net = V1GlobalAttentionPooling(input_size=schema.width(self.input_key))

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Pool the sequence with the (optionally register-augmented) mask dict.

        Returns
        -------
        dict[str, Tensor]
            The newly produced keys only (design §2.5).
        """
        del mode
        assert self.pool_net is not None, "forward before bind()"
        x = {"seq": b.get(self.input_key)}
        # streams-then-REGISTERS dict order is numerics-critical: pooling
        # cats mask values in dict order (pooling.py:56). The encoder-less
        # path has no register row (no encoder produced one), so the pad dict
        # is just {"seq": seq.mask} — the exact v1 saltmodel.py:155-156,170
        # encoder-less semantics; absent optional ports are probed, not read
        # (executor optional-port contract).
        pad = {"seq": b.get("seq.mask")}
        if "masks.registers" in b:
            pad["REGISTERS"] = b.get("masks.registers")
        return {self.out_key: self.pool_net(x, pad_mask=pad)}


class LossSum(nn.Module):
    """Weighted sum of per-task losses -> ``loss.total`` (design §3.3).

    Owns loss combination outright, replacing the smeared v1 ownership
    (``ModelWrapper.total_loss`` wsum branch, modelwrapper.py:195-197 —
    per-task weights are applied INSIDE the tasks, as in v1, so the default
    here is a plain sum; `weights` is an extra per-loss-key multiplier).

    The ``losses.**`` auto-collection is a FRAMEWORK wildcard (design §3.3):
    the M1 kernel rejects wildcard *requires* (planner.py), so the
    narrowing happens framework-side — `collect_loss_keys` scans sibling
    modules' declared produces and `narrow` fixes the concrete key list
    before plan compilation (`SaltModule` calls both; tests may too). An
    explicit ``losses:`` config list skips collection entirely.
    """

    def __init__(
        self,
        losses: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Capture config; ``losses=None`` defers to framework narrowing.

        Parameters
        ----------
        losses : Sequence[str] | None, optional
            Explicit loss keys (``"losses.<task>"`` or bare task names), by
            default None (auto-collected via `collect_loss_keys`/`narrow`).
        weights : Mapping[str, float] | None, optional
            Per-loss multipliers keyed like `losses`, default 1.0 each.
        """
        super().__init__()
        self.name = _UNNAMED
        self._loss_keys: tuple[str, ...] | None = (
            tuple(_loss_key(k) for k in losses) if losses is not None else None
        )
        self.weights = {_loss_key(k): float(v) for k, v in (weights or {}).items()}
        if self._loss_keys is not None:
            self._check_weight_keys()

    def _check_weight_keys(self) -> None:
        """Reject weight entries that name no summed loss key.

        Raises
        ------
        ConfigError
            Naming the unknown weight keys and the known loss keys.
        """
        assert self._loss_keys is not None
        if unknown := sorted(set(self.weights) - set(self._loss_keys)):
            raise ConfigError(
                f"LossSum: weights for unknown loss keys {unknown} — summed keys are "
                f"{list(self._loss_keys)}"
            )

    @property
    def narrowed(self) -> bool:
        """Whether the loss-key list is fixed (explicit config or `narrow`).

        Returns
        -------
        bool
            True once the concrete loss keys are known — the framework
            (`SaltModule`) narrows un-fixed instances before compile
            (design §3.3).
        """
        return self._loss_keys is not None

    @staticmethod
    def collect_loss_keys(
        modules: Mapping[str, GraphModule], mode: Mode = Mode.FIT
    ) -> tuple[str, ...]:
        """Scan sibling modules for declared ``losses.*`` produces (framework narrowing).

        Parameters
        ----------
        modules : Mapping[str, GraphModule]
            The full module dict (LossSum instances are skipped).
        mode : Mode, optional
            The mode whose declarations are scanned, by default `Mode.FIT`.

        Returns
        -------
        tuple[str, ...]
            All declared loss keys, in module-dict declaration order.
        """
        keys: list[str] = []
        for module in modules.values():
            if isinstance(module, LossSum):
                continue
            for key, spec in flatten_spec(module.declare_io(mode).produces).items():
                if key.startswith("losses.") and spec.active_in(mode):
                    keys.append(key)
        return tuple(keys)

    def narrow(self, loss_keys: Iterable[str]) -> None:
        """Fix the concrete loss-key list (framework-side ``losses.**`` narrowing).

        Raises
        ------
        ConfigError
            If explicit ``losses:`` config already fixed the keys, the list
            is empty, or a configured weight names no key.
        """
        if self._loss_keys is not None:
            raise ConfigError(
                f"LossSum {self.name!r}: loss keys already fixed to {list(self._loss_keys)}"
            )
        keys = tuple(_loss_key(k) for k in loss_keys)
        if not keys:
            raise ConfigError(
                f"LossSum {self.name!r}: narrowed to an empty loss-key list — no module "
                "declares a losses.* produce (design §3.3)"
            )
        self._loss_keys = keys
        self._check_weight_keys()

    def declare_io(self, mode: Mode) -> IO:
        """Declare the narrowed loss keys -> ``loss.total`` (TRAINING only).

        Returns
        -------
        IO
            Empty in TEST/ONNX (the module is mode-inactive there).

        Raises
        ------
        ConfigError
            If the loss keys were never fixed (no ``losses:`` config and no
            `narrow` call).
        """
        if not (mode & Mode.TRAINING):
            return IO(requires={}, produces={})
        if self._loss_keys is None:
            raise ConfigError(
                f"LossSum {self.name!r}: loss keys not fixed — pass losses: in config or let "
                "the framework narrow via collect_loss_keys()/narrow() before compile "
                "(losses.** is a framework wildcard, design §3.3)"
            )
        loss_spec = TensorSpec(shape=(), kind="loss", modes=Mode.TRAINING)
        return IO(
            requires=unflatten_spec(dict.fromkeys(self._loss_keys, loss_spec)),
            produces=unflatten_spec({"loss.total": loss_spec}),
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Sum the (optionally weighted) loss leaves.

        Returns
        -------
        dict[str, Tensor]
            ``{"loss.total": scalar}``.
        """
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        total = sum(self.weights.get(key, 1.0) * b.get(key) for key in self._loss_keys)
        return {"loss.total": total}


class LossGLS(LossSum):
    """Geometric-mean (GLS) combination of per-task losses -> ``loss.total`` (design §3.3).

    A NetModule SIBLING of `LossSum` (it subclasses it to share the
    ``losses.**`` framework wildcard, `declare_io`, and the
    `collect_loss_keys`/`narrow` integration — `SaltModule`'s narrow loop
    already keys off ``isinstance(_, LossSum)``, so a `LossGLS` is narrowed
    for free, saltmodule.py:157-158). The ONLY behavioural change is the
    combination rule (`forward`): the n-task GEOMETRIC MEAN
    ``(∏ losses)^(1/n)`` (GLS = geometric loss strategy), reproducing the v1
    ``loss_mode == "GLS"`` branch (modelwrapper.py:194-196) which `LossSum`'s
    weighted sum replaced for the default ``wsum`` mode.

    GLS does NOT utilise loss weights. v1 enforces this with a loud
    construction-time guard — ``assert all(task.weight == 1.0 for task in
    self.model.tasks)`` (modelwrapper.py:139-142) — because the geometric mean
    of per-task losses is only meaningful when no task is pre-scaled (a
    weighted task loss ``w*L`` contributes ``w^(1/n)`` to the product, an
    arbitrary rescale of the mean — silent divergence, plan 10 risk
    "LossGLS gates 14 configs"). v2 has TWO weight surfaces, both guarded to
    1.0:

    - This module's per-loss ``weights`` multiplier (the `LossSum` extra) —
      rejected in ``__init__`` so a GLS config can never carry one.
    - The task-side ``weight: float`` applied INSIDE each composed v1 head
      before it publishes ``losses.<task>`` (tasks.py:88, task.py:243) — the
      EXACT v1 ``task.weight`` surface. This is sibling state the loss module
      cannot see at construction, so the framework calls
      `check_task_weights(modules)` from `SaltModule.__init__`'s narrow loop
      (the v1 ctor guard's v2 home), failing loudly before any plan compiles.
    """

    def __init__(
        self,
        losses: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Capture config; reject any per-loss weight (GLS ignores weights).

        Parameters
        ----------
        losses : Sequence[str] | None, optional
            Explicit loss keys (``"losses.<task>"`` or bare task names), by
            default None (auto-collected via `collect_loss_keys`/`narrow`,
            as for `LossSum`).
        weights : Mapping[str, float] | None, optional
            Accepted only for parity with the `LossSum` signature: GLS does
            NOT utilise weights, so any entry != 1.0 is rejected (v1
            modelwrapper.py:139-142).

        Raises
        ------
        ConfigError
            If any configured weight is not 1.0 (GLS ignores weights — set
            them to 1, or use `LossSum` for a weighted sum).
        """
        super().__init__(losses=losses, weights=weights)
        # exact == 1.0 is the faithful v1 semantic (modelwrapper.py:140 asserts
        # task.weight == 1.0); weights are config literals, never computed values
        if bad := {k: v for k, v in self.weights.items() if v != 1.0}:  # noqa: RUF069
            raise ConfigError(
                f"LossGLS: per-loss weights are not utilised by the geometric mean — got "
                f"{bad}; set all weights to 1.0, or use LossSum for a weighted sum "
                "(v1 modelwrapper.py:139-142)"
            )

    @staticmethod
    def check_task_weights(modules: Mapping[str, GraphModule]) -> None:
        """Assert every loss-producing task carries ``weight == 1.0`` (the v1 guard).

        The v2 home of v1's ``ModelWrapper.__init__`` GLS assertion
        (``all(task.weight == 1.0 for task in self.model.tasks)``,
        modelwrapper.py:139-142). Called by `SaltModule.__init__` when a
        `LossGLS` is present, BEFORE any `declare_io`/compile, so a weighted
        task under GLS fails loudly at assembly rather than silently
        rescaling the geometric mean (plan 10 risk). Inspects the public
        numeric ``weight`` every task module exposes (tasks.py:88 coerces it
        to ``float``; the guard accepts ``int`` too so a future un-coerced
        weight is still caught); modules without a numeric ``weight`` attribute
        (`Normaliser`, `Concat`, `LossSum`/`LossGLS`, ...) are ignored — only
        the loss producers carry it.

        Parameters
        ----------
        modules : Mapping[str, GraphModule]
            The full configured module dict.

        Raises
        ------
        ConfigError
            Naming each task whose ``weight`` is not 1.0.
        """
        offenders = {
            name: float(module.weight)
            for name, module in modules.items()
            if not isinstance(module, LossSum)
            # duck-typed numeric check (int OR float): the v2 task base coerces
            # ``self.weight = float(weight)`` (tasks.py:88) so a YAML ``weight: 2``
            # already arrives as 2.0 and is caught, but guarding ``(int, float)``
            # keeps a future task module that stored an un-coerced int weight from
            # silently slipping past the GLS guard. (LossSum carries ``weights`` —
            # a dict — not ``weight``, and is excluded above regardless.)
            and isinstance(getattr(module, "weight", None), (int, float))
            and float(module.weight) != 1.0  # noqa: RUF069 - exact, the v1 semantic
        }
        if offenders:
            raise ConfigError(
                f"LossGLS: GLS does not utilise task weights — set all task weights to 1.0, "
                f"got {offenders} (v1 modelwrapper.py:139-142)"
            )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Combine the loss leaves by their geometric mean.

        ``(∏_k losses[k])^(1/n)`` over the narrowed loss keys — the v1
        ``loss_mode == "GLS"`` reduction (``math.prod`` then ``pow(·, 1/n)``,
        modelwrapper.py:194-196). Weights are guaranteed 1.0 by ``__init__``
        and `check_task_weights`, so none appear here (a weighted product
        would be the divergence v1's guard forbids).

        Returns
        -------
        dict[str, Tensor]
            ``{"loss.total": (∏ losses)^(1/n)}``.
        """
        del mode
        assert self._loss_keys is not None, "forward before declare_io narrowing"
        product = math.prod(b.get(key) for key in self._loss_keys)
        return {"loss.total": torch.pow(product, 1.0 / len(self._loss_keys))}


def _loss_key(key: str) -> str:
    """Normalise a configured loss reference to a dotted ``losses.`` key.

    Returns
    -------
    str
        ``"losses.<name>"`` for bare task names; dotted keys unchanged.
    """
    return key if key.startswith("losses.") else f"losses.{key}"


def _reject_width_keys(who: str, cfg: Mapping[str, Any] | None, banned: tuple[str, ...]) -> None:
    """Reject configured width keys — widths are inferred at bind (design §2.3).

    Raises
    ------
    ConfigError
        Naming the offending keys.
    """
    if cfg and (bad := sorted(set(cfg) & set(banned))):
        raise ConfigError(
            f"{who}: dense config must not set {bad} — widths are inferred at bind from the "
            "resolved schema (design §2.3 kills YAML width arithmetic)"
        )
