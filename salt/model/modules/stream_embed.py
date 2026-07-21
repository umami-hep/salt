"""StreamEmbed GraphModule (per-stream embedding front-end)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
from salt.core.nn.base import SaltModelModule
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.dense import Dense, _reject_width_keys
from salt.core.nn.featurewise import FeaturewiseTransformation
from salt.core.nn.posenc import PositionalEncoder
from salt.core.utils.tensor_utils import (
    attach_context,
)


def _stream_len(stream: str) -> str:
    """Return the shared symbolic token-count dim for a sequence stream, e.g. ``"T:tracks"``."""
    return sym_dim("T", stream)


class StreamEmbed(SaltModelModule):
    """Config-constructed per-stream initial embedding.

    Composes a `Dense` built at `bind`, with input width inferred
    from the resolved input and context widths. Context entries are
    PREPENDED in list order (``cat([context, x])``), so the feature layout is
    ``[ctx[-1], ..., ctx[0], stream]``.

    Stream rank is INFERRED from the bound input, not configured: a
    padded-sequence input ``[B, T, F]`` -> ``embed.<s> [B, T, D]``; a per-jet
    global input ``[B, F]`` -> ``embed.<s> [B, D]`` with no token axis. This is
    what makes DL1 a jets-only MLP (no encoder, no pooling).

    ``mup: true`` builds the internal `Dense` with ``mup=True`` at `bind`,
    applying the muP weight init; the forward math is otherwise unchanged.

    Optional, both OFF by default (unset => today's behaviour):

    - ``featurewise:`` — an input-layer `FeaturewiseTransformation` (FiLM):
      reads ``inputs.parameters`` and applies ``scale * x + bias`` to the
      embed INPUT before the `Dense` projection. When active, ``parameters``
      is NOT also concatenated as context.
    - ``pos_enc:`` — a `PositionalEncoder` ADDED to the embed OUTPUT (after
      projection), over the configured coordinate variables.
    """

    MUP_WIDTH_ARG = "out_dim"
    """The init_arg the muP shape-generation tooling (``salt mup-shapes``)
    sweeps for this module to produce the infshapes."""

    def __init__(
        self,
        stream: str,
        out_dim: int,
        dense: dict[str, Any] | None = None,
        context: Sequence[str] = (),
        input: str | None = None,  # noqa: A002 - YAML surface name
        mup: bool = False,
        featurewise: dict[str, Any] | None = None,
        pos_enc: dict[str, Any] | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        stream : str
            The stream to embed.
        out_dim : int
            Output embedding width (concrete, config-fixed).
        dense : dict[str, Any] | None, optional
            Extra kwargs for the internal v1 `Dense`; must not contain width keys.
        context : Sequence[str], optional
            Dotted bundle keys attached as context, in prepend order (see
            class docstring), by default ``()``.
        input : str | None, optional
            Input key override, by default ``normed.<stream>``. The embed's
            rank is INFERRED from this bound input.
        mup : bool, optional
            Whether to use the muP parametrisation for the embed's internal
            `Dense`, by default False.
        featurewise : dict[str, Any] | None, optional
            Optional input-layer FiLM config (see class docstring): kwargs for
            `FeaturewiseTransformation` minus the width args. By default None.
        pos_enc : dict[str, Any] | None, optional
            Optional positional-encoding config (see class docstring): kwargs
            for `PositionalEncoder`. By default None.

        Raises
        ------
        ConfigError
            If `dense` configures widths (inferred at bind) or `out_dim` is not positive.
        """
        super().__init__()
        if out_dim < 1:
            raise ConfigError(f"StreamEmbed: out_dim must be >= 1, got {out_dim}")
        _reject_width_keys("StreamEmbed", dense, ("input_size", "output_size", "context_size"))
        if "mup" in (dense or {}):
            raise ConfigError(
                "StreamEmbed: set mup on the module (init_args.mup), not inside dense — the flag "
                "is threaded into the composed v1 Dense at bind (design §3.4 muP architectural "
                "port)"
            )
        self.stream = stream
        self.out_dim = out_dim
        self.dense_cfg = dict(dense or {})
        self.context = tuple(context)
        self.input_key = input if input is not None else f"normed.{stream}"
        self.mup = bool(mup)
        self.net: nn.Module | None = None
        # -- optional input-layer FiLM ---------------------------------------
        self.featurewise_cfg = dict(featurewise) if featurewise is not None else None
        self.params_key = "inputs.parameters"
        self.featurewise: FeaturewiseTransformation | None = None
        if self.featurewise_cfg is not None:
            self.params_key = self.featurewise_cfg.pop("parameters", self.params_key)
            if self.featurewise_cfg.get("layer", "input") != "input":
                raise ConfigError(
                    "StreamEmbed featurewise: layer must be 'input' (the embed is the input-layer "
                    "FiLM site; use TransformerEncoder for encoder/global layers)"
                )
            self.featurewise_cfg["layer"] = "input"
        # -- optional positional encoding ------------------------------------
        self.pos_enc_cfg = dict(pos_enc) if pos_enc is not None else None
        self.pos_enc: PositionalEncoder | None = None
        self.pos_enc_indices: tuple[int, ...] = ()

    def declare_io(self, mode: Mode) -> IO:
        """Declare input + context keys -> ``embed.<stream>`` (rank-agnostic, ``shape=None``)."""
        del mode
        requires: dict[str, TensorSpec] = {
            self.input_key: TensorSpec(shape=None, dtype="float32"),
        }
        for key in self.context:
            # rank/width unconstrained: context may be a [B, F] global vector
            # or broadcastable — widths resolve from the producer side
            requires[key] = TensorSpec(shape=None, dtype="float32")
        if self.featurewise_cfg is not None:
            # the per-event conditioning parameters: a rank-2 [B, n_params] global stream
            requires[self.params_key] = TensorSpec(shape=("B", sym_dim("P", self.name)), dtype="float32")
        return IO(
            requires=unflatten_spec(requires),
            produces=unflatten_spec({
                f"embed.{self.stream}": TensorSpec(shape=None, dtype="float32"),
            }),
        )

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Contribute the ``embed.<stream>`` last-dim width (``out_dim``, a config constant)."""
        del widths
        return {f"embed.{self.stream}": self.out_dim}

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the internal `Dense` + optional input-FiLM + optional pos-enc column indices."""
        input_size = schema.width(self.input_key) + sum(schema.width(key) for key in self.context)
        self.net = Dense(
            input_size=input_size, output_size=self.out_dim, mup=self.mup, **self.dense_cfg
        )
        if self.featurewise_cfg is not None:
            self.featurewise = FeaturewiseTransformation(
                num_params=schema.width(self.params_key),
                num_features=input_size,
                **self.featurewise_cfg,
            )
            self.featurewise.name = self.name
            self.featurewise.build()
        if self.pos_enc_cfg is not None:
            cfg = dict(self.pos_enc_cfg)
            cfg.setdefault("dim", self.out_dim)
            self.pos_enc = PositionalEncoder(**cfg)
            self.pos_enc.name = self.name
            # resolve the coordinate column indices by NAME from the input fields
            fields = schema.fields_of(self.input_key)
            try:
                self.pos_enc_indices = tuple(fields.index(v) for v in self.pos_enc.variables)
            except ValueError as err:
                raise ConfigError(
                    f"StreamEmbed {self.name!r} pos_enc: variable not found in {self.input_key!r} "
                    f"fields {list(fields)}: {err}"
                ) from None

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Attach context (prepended) and project; optional FiLM/pos-enc are no-ops when unset."""
        del mode
        x = b.get(self.input_key)
        raw = x  # the raw input columns (pos_enc reads these at the variable idx)
        for key in self.context:
            x = attach_context(x, b.get(key))  # cat([context, x])
        if self.featurewise is not None:
            # FiLM on the embed INPUT, before the projection
            x = self.featurewise(b.get(self.params_key), x)
        assert self.net is not None, "forward before bind()"
        out = self.net(x)
        if self.pos_enc is not None:
            # ADD the positional encoding to the embed OUTPUT
            out = out + self.pos_enc(raw[..., self.pos_enc_indices])
        return {f"embed.{self.stream}": out}
