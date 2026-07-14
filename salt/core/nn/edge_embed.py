"""EdgeFeatures and EdgeEmbed GraphModules (edge-stream embedding).

Also holds the edge-feature computation helpers (`check_edge_config`,
`calculate_edge_features`) consumed only by `EdgeFeatures`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

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
from salt.core.nn.base import SaltModelModule
from salt.core.nn.bind import ResolvedSchema
from salt.core.nn.dense import Dense, _reject_width_keys
from salt.core.nn.stream_embed import _stream_len

_EDGE_FEATURES = ("dR", "z", "kt", "subjetIndex", "isSelfLoop", "mass")
"""Recognised edge-feature names. EdgeFeatures rejects anything outside this
set at config time."""


def check_edge_config(
    edge_features: list[str],
    available_vars: list[str],
) -> None:
    """Check the requested edge features are recognized and have the required input variables."""
    req_vars: list[str] = []
    for variable in edge_features:
        if variable == "dR":
            req_vars.extend(["eta", "phi"])
        elif variable == "z":
            req_vars.extend(["pt"])
        elif variable == "kt":
            req_vars.extend(["eta", "phi", "pt"])
        elif variable == "isSelfLoop":
            continue
        elif variable == "subjetIndex":
            req_vars.extend(["subjetIndex"])
        elif variable == "mass":
            req_vars.extend(["pt", "eta", "phi", "energy"])
        else:
            raise ValueError(f"Edge feature {variable} not recognized")

    missing = set(req_vars) - set(available_vars)
    if missing:
        raise ValueError(
            f"Indices of {missing} required for edge features calculation were not specified."
        )


def calculate_edge_features(
    batch: Tensor,
    indices_map: dict[str, int],
    variables: list[str],
) -> Tensor:
    """Compute pairwise dR/kt/z/subjetIndex/isSelfLoop/mass edge features -> ``[B, N, N, F]``."""
    ebatch = torch.zeros(
        (batch.shape[0], batch.shape[1], batch.shape[1], len(variables)),
        dtype=batch.dtype,
        device=batch.device,
    )

    # intermediate quantities
    if "dR" in variables or "kt" in variables:
        dphi = batch[:, :, indices_map["phi"]].unsqueeze(1).expand(-1, batch.shape[1], -1) - batch[
            :, :, indices_map["phi"]
        ].unsqueeze(2).expand(-1, -1, batch.shape[1])
        dphi -= (dphi > math.pi).type_as(dphi) * 2 * math.pi
        deta = batch[:, :, indices_map["eta"]].unsqueeze(1).expand(-1, batch.shape[1], -1) - batch[
            :, :, indices_map["eta"]
        ].unsqueeze(2).expand(-1, -1, batch.shape[1])
    if "kt" in variables or "z" in variables:
        pt_min = torch.minimum(
            batch[:, :, indices_map["pt"]].unsqueeze(1).expand(-1, batch.shape[1], -1),
            batch[:, :, indices_map["pt"]].unsqueeze(2).expand(-1, -1, batch.shape[1]),
        )
    if "mass" in variables:
        pt = batch[:, :, indices_map["pt"]]
        eta = batch[:, :, indices_map["eta"]]
        phi = batch[:, :, indices_map["phi"]]
        energy = batch[:, :, indices_map["energy"]]
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * (torch.exp(eta) - torch.exp(-eta)) / 2

    # fill edge features
    for i, variable in enumerate(variables):
        if variable == "dR":
            ebatch[:, :, :, i] = torch.log(torch.sqrt(torch.square(deta) + torch.square(dphi)))
        elif variable == "kt":
            ebatch[:, :, :, i] = torch.log(
                pt_min * torch.sqrt(torch.square(deta) + torch.square(dphi))
            )
        elif variable == "z":
            pt_sum = batch[:, :, indices_map["pt"]].unsqueeze(1).expand(
                -1, batch.shape[1], -1
            ) + batch[:, :, indices_map["pt"]].unsqueeze(2).expand(-1, -1, batch.shape[1])
            ebatch[:, :, :, i] = torch.log(pt_min / pt_sum)
        elif variable == "isSelfLoop":
            ebatch[:, :, :, i] = (
                torch.eye(batch.shape[1], dtype=ebatch.dtype, device=batch.device)
                .unsqueeze(0)
                .expand(batch.shape[0], -1, -1)
            )
        elif variable == "subjetIndex":
            sji1 = (
                batch[:, :, indices_map["subjetIndex"]].unsqueeze(1).expand(-1, batch.shape[1], -1)
            )
            sji2 = (
                batch[:, :, indices_map["subjetIndex"]].unsqueeze(2).expand(-1, -1, batch.shape[1])
            )
            ebatch[:, :, :, i] = torch.logical_and(torch.eq(sji1, sji2), sji1 >= 0)
        elif variable == "mass":
            e1 = energy.unsqueeze(1).expand(-1, batch.shape[1], -1)
            e2 = energy.unsqueeze(2).expand(-1, -1, batch.shape[1])
            px1 = px.unsqueeze(1).expand(-1, batch.shape[1], -1)
            px2 = px.unsqueeze(2).expand(-1, -1, batch.shape[1])
            py1 = py.unsqueeze(1).expand(-1, batch.shape[1], -1)
            py2 = py.unsqueeze(2).expand(-1, -1, batch.shape[1])
            pz1 = pz.unsqueeze(1).expand(-1, batch.shape[1], -1)
            pz2 = pz.unsqueeze(2).expand(-1, -1, batch.shape[1])
            e_sum = e1 + e2
            px_sum = px1 + px2
            py_sum = py1 + py2
            pz_sum = pz1 + pz2
            mass2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
            mass2 = torch.clamp_min(mass2, 1e-8)
            ebatch[:, :, :, i] = 0.5 * torch.log(mass2)

    return torch.nan_to_num(ebatch, nan=0.0, posinf=0.0, neginf=0.0)


class EdgeFeatures(SaltModelModule):
    """Config-constructed pairwise edge-feature builder.

    Requires the RAW (un-normalised) ``inputs.<stream>`` ``[B, T, F]`` plus
    ``masks.<stream>``, and produces ``edges.<stream>`` ``[B, T, T, E]`` where
    ``E = len(features)``. Produces a NEW key; never mutates ``inputs.*``.

    The ``indices_map`` (variable name -> column index) is resolved at `bind`
    from the resolved schema's declared fields — column lookups resolve by
    NAME, never by YAML list position.

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
        if not features:
            raise ConfigError("EdgeFeatures: features must be a non-empty sequence (design §6.7)")
        if len(set(features)) != len(tuple(features)):
            raise ConfigError(f"EdgeFeatures: duplicate features in {tuple(features)}")
        unknown = [f for f in features if f not in _EDGE_FEATURES]
        if unknown:
            raise ConfigError(
                f"EdgeFeatures: unrecognised edge feature(s) {unknown} — choose from "
                f"{sorted(_EDGE_FEATURES)}"
            )
        self.stream = stream
        self.features = tuple(features)
        self.out_key = out if out is not None else f"edges.{stream}"
        self.input_key = input if input is not None else f"inputs.{stream}"
        self.indices_map: dict[str, int] | None = None

    def declare_io(self, mode: Mode) -> IO:
        """Declare raw ``inputs``/``masks.<stream>`` -> ``edges.<stream>`` ``[B,T,T,E]``."""
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
        """Resolve the variable-name -> column-index map and validate the requested features."""
        fields = schema.fields_of(self.input_key)
        check_edge_config(list(self.features), list(fields))
        self.indices_map = {name: i for i, name in enumerate(fields)}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Compute the pairwise edge features on the RAW input; returns a FRESH tensor."""
        del mode
        assert self.indices_map is not None, "forward before bind()"
        x = b.get(self.input_key)
        edges = calculate_edge_features(x, self.indices_map, list(self.features))
        return {self.out_key: edges}


class EdgeEmbed(SaltModelModule):
    """Config-constructed edge-feature embedding.

    An edge-typed `StreamEmbed`: maps ``edges.<stream>`` ``[B, T, T, E]`` ->
    ``edges.<stream>_emb`` ``[B, T, T, D_e]`` with an internal `Dense` (an
    ``nn.Linear`` stack over the last dim, so it embeds each pairwise edge
    independently). No context, no muP, no per-stream rank inference — edges
    are always the rank-4 pairwise matrix.

    ONNX: both ``T:<stream>`` token axes flow through unchanged (dynamic).

    Carries no FiLM/positional-encoding — those live on `StreamEmbed` for the
    constituent streams and on `TransformerEncoder` for the encoder/global FiLM.
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
            Extra kwargs for the internal `Dense`; must not contain width keys.
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
