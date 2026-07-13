"""EdgeFeatures and EdgeEmbed GraphModules (edge-stream embedding)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.dense import Dense, _reject_width_keys
from salt.core.nn.edge_features import calculate_edge_features, check_edge_config
from salt.core.nn.stream_embed import _stream_len

_EDGE_FEATURES = ("dR", "z", "kt", "subjetIndex", "isSelfLoop", "mass")
"""Recognised edge-feature names. EdgeFeatures rejects anything outside this
set at config time."""


class EdgeFeatures(nn.Module):
    """Config-constructed pairwise edge-feature builder.

    Requires the RAW (un-normalised) ``inputs.<stream>`` ``[B, T, F]`` plus
    ``masks.<stream>``, and produces ``edges.<stream>`` ``[B, T, T, E]`` where
    ``E = len(features)``. Produces a NEW key; never mutates ``inputs.*``.

    The per-element math (dR/kt/z/subjetIndex/isSelfLoop/mass) is byte-faithful
    to v1's inlined `calculate_edge_features`/`check_edge_config`. The
    ``indices_map`` (variable name -> column index) is resolved at `bind` from
    the resolved schema's declared fields — column lookups resolve by NAME,
    never by YAML list position.

    ONNX: the produced tensor carries the SAME ``T:<stream>`` symbol on both
    token axes, so both trace as dynamic; the math is all
    unsqueeze/expand/elementwise ops driven by the dynamic token count.
    """

    def __init__(
        self,
        stream: str,
        features: Sequence[str],
        out: str | None = None,
        input: str | None = None,  # noqa: A002 - YAML surface name
    ) -> None:
        """Capture config only (no schema/data access here).

        Parameters
        ----------
        stream : str
            The stream whose pairwise edges are built (``inputs.<stream>``).
        features : Sequence[str]
            Edge feature names, in produced column order — any of
            ``{"dR", "z", "kt", "subjetIndex", "isSelfLoop", "mass"}``.
            Required input variables are checked at `bind` against the
            resolved fields.
        out : str | None, optional
            Produced edge key, by default ``edges.<stream>``.
        input : str | None, optional
            Raw input key override, by default ``inputs.<stream>``.

        Raises
        ------
        ConfigError
            If `features` is empty, has duplicates, or names an unrecognised
            edge feature.
        """
        super().__init__()
        self.name = _UNNAMED
        if not features:
            raise ConfigError("EdgeFeatures: features must be a non-empty sequence (design §6.7)")
        if len(set(features)) != len(tuple(features)):
            raise ConfigError(f"EdgeFeatures: duplicate features in {tuple(features)}")
        unknown = [f for f in features if f not in _EDGE_FEATURES]
        if unknown:
            raise ConfigError(
                f"EdgeFeatures: unrecognised edge feature(s) {unknown} — choose from "
                f"{sorted(_EDGE_FEATURES)} (v1 check_edge_config, edge_features.py:30-43)"
            )
        self.stream = stream
        self.features = tuple(features)
        self.out_key = out if out is not None else f"edges.{stream}"
        self.input_key = input if input is not None else f"inputs.{stream}"
        self.indices_map: dict[str, int] | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare raw ``inputs.<stream>`` + ``masks.<stream>`` -> ``edges.<stream>``.

        The produced edge tensor is ``[B, T, T, E]`` with the SAME
        ``T:<stream>`` symbol on both token axes (a square pairwise matrix).
        """
        del mode
        tlen = _stream_len(self.stream)
        edge_dim = len(self.features)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", tlen, sym_dim("F", self.name)), dtype="float32"
                ),
                f"masks.{self.stream}": TensorSpec(
                    shape=("B", tlen), dtype="bool", kind="pad_mask"
                ),
            }),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", tlen, tlen, edge_dim), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Resolve the variable-name -> column-index map and validate features.

        ``check_edge_config`` validates that every feature's required
        variables are present (e.g. ``dR`` needs ``eta``/``phi``).
        """
        fields = schema.fields_of(self.input_key)
        check_edge_config(list(self.features), list(fields))
        self.indices_map = {name: i for i, name in enumerate(fields)}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Compute the pairwise edge features on the RAW input.

        Returns a FRESH tensor; never mutates ``inputs.*``.
        """
        del mode
        assert self.indices_map is not None, "forward before bind()"
        x = b.get(self.input_key)
        edges = calculate_edge_features(x, self.indices_map, list(self.features))
        return {self.out_key: edges}


class EdgeEmbed(nn.Module):
    """Config-constructed edge-feature embedding.

    An edge-typed `StreamEmbed`: maps ``edges.<stream>`` ``[B, T, T, E]`` ->
    ``edges.<stream>_emb`` ``[B, T, T, D_e]`` with an internal v1 `Dense` (an
    ``nn.Linear`` stack over the last dim, so it embeds each pairwise edge
    independently). No context, no muP, no per-stream rank inference — edges
    are always the rank-4 pairwise matrix.

    ONNX: both ``T:<stream>`` token axes flow through unchanged (dynamic).

    Carries no FiLM/positional-encoding, faithfully to v1: those live on
    `StreamEmbed` for the constituent streams and on `TransformerEncoder` for
    the encoder/global FiLM.
    """

    def __init__(
        self,
        stream: str,
        out_dim: int,
        dense: dict[str, Any] | None = None,
        input: str | None = None,  # noqa: A002 - YAML surface name
        out: str | None = None,
    ) -> None:
        """Capture config only.

        Parameters
        ----------
        stream : str
            The stream whose edge tensor is embedded.
        out_dim : int
            Output edge-embedding width ``D_e`` (concrete, config-fixed).
        dense : dict[str, Any] | None, optional
            Extra kwargs for the internal v1 `Dense`; must not contain width keys.
        input : str | None, optional
            Edge input key override, by default ``edges.<stream>``.
        out : str | None, optional
            Produced embedded edge key, by default ``edges.<stream>_emb``.

        Raises
        ------
        ConfigError
            If `dense` configures widths (inferred at bind) or `out_dim` is not positive.
        """
        super().__init__()
        self.name = _UNNAMED
        if out_dim < 1:
            raise ConfigError(f"EdgeEmbed: out_dim must be >= 1, got {out_dim}")
        _reject_width_keys("EdgeEmbed", dense, ("input_size", "output_size", "context_size"))
        self.stream = stream
        self.out_dim = out_dim
        self.dense_cfg = dict(dense or {})
        self.input_key = input if input is not None else f"edges.{stream}"
        self.out_key = out if out is not None else f"edges.{stream}_emb"
        self.net: nn.Module | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare ``edges.<stream>`` -> ``edges.<stream>_emb`` (``[B,T,T,E] -> [B,T,T,D_e]``)."""
        del mode
        tlen = _stream_len(self.stream)
        return IO(
            requires=unflatten_spec({
                self.input_key: TensorSpec(
                    shape=("B", tlen, tlen, sym_dim("E", self.name)), dtype="float32"
                ),
            }),
            produces=unflatten_spec({
                self.out_key: TensorSpec(shape=("B", tlen, tlen, self.out_dim), dtype="float32"),
            }),
        )

    def bind(self, schema: ResolvedSchema) -> None:
        """Build the internal `Dense` with ``input_size = width(edges.<stream>) = E``."""
        self.net = Dense(
            input_size=schema.width(self.input_key), output_size=self.out_dim, **self.dense_cfg
        )

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Project the edge tensor through the internal `Dense` ([B,T,T,E] -> [B,T,T,D_e])."""
        del mode
        assert self.net is not None, "forward before bind()"
        return {self.out_key: self.net(b.get(self.input_key))}
