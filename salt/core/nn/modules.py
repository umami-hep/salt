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
    "LossSum",
    "Normaliser",
    "Split",
    "StreamEmbed",
    "TransformerEncoder",
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

    TODO(M6): ``mup:`` flag; TODO(M7): ``featurewise:`` / ``edges:`` ports
    (design §6.4, §6.7) when the v1 internals are absorbed.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        attention: dict[str, Any],
        out_dim: int | None = None,
        dense: dict[str, Any] | None = None,
        norm: str = "LayerNorm",
        num_registers: int = 1,
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

        Raises
        ------
        ConfigError
            If `attention` is missing ``num_heads``.
        """
        super().__init__()
        self.name = _UNNAMED
        if not isinstance(attention, Mapping) or "num_heads" not in attention:
            raise ConfigError(
                "TransformerEncoder: attention config must be a mapping containing 'num_heads' "
                "(v1 Transformer requires attn_kwargs, transformer.py:600-601)"
            )
        attn_kwargs = dict(attention)
        attn_type = attn_kwargs.pop("attn_type", "torch-math")
        self.dim = dim
        self.encoder = V1Transformer(
            num_layers=num_layers,
            embed_dim=dim,
            out_dim=out_dim,
            norm=norm,
            attn_type=attn_type,
            do_final_norm=True,
            num_registers=num_registers,
            attn_kwargs=attn_kwargs,
            dense_kwargs=dict(dense) if dense is not None else None,
        )
        self.out_dim = self.encoder.out_dim
        self.num_registers = num_registers

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``seq.x``/``seq.mask`` -> ``encoded.seq``/``masks.registers``.

        Returns
        -------
        IO
            The declared requires/produces (widths concrete from config).
        """
        del mode
        return IO(
            requires=unflatten_spec({
                "seq.x": TensorSpec(shape=("B", _SEQ_LEN, self.dim), dtype="float32"),
                "seq.mask": TensorSpec(shape=("B", _SEQ_LEN), dtype="bool", kind="pad_mask"),
            }),
            produces=unflatten_spec({
                "encoded.seq": TensorSpec(shape=("B", _ENC_LEN, self.out_dim), dtype="float32"),
                "masks.registers": TensorSpec(
                    shape=("B", self.num_registers), dtype="bool", kind="pad_mask"
                ),
            }),
        )

    def set_export_mode(self) -> None:
        """Force the deterministic torch-math backend (design §2.5, test/ONNX semantics)."""
        self.encoder.set_backend("torch-math")

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Encode the sequence; publish the register mask as a NEW key.

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
