"""`MFLeadVertexDecorator` — the MaskFormer lead-vertex jet-level decorator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, split_key, unflatten_spec
from salt.core.nn.base import SaltModelModule
from salt.core.outputs.output_field import OutputField


class MFLeadVertexDecorator(SaltModelModule):
    """MaskFormer lead-vertex jet-level decorator (the decoration half of the two-node split).

    A thin selector that reads `MaskFormerObjects`'s exposed reordered
    per-vertex leaves (``vertices_class_probs [B, M, C]`` +
    ``vertices_regression [B, M, R]``) and emits jet-level global scalar
    leaves (e.g. ``jet.lead_vertex_pt``). Not a parity fold of any legacy
    reduce — a new capability with no legacy oracle.

    The lead vertex is selected per-jet as the highest-pT vertex (by
    ``vertices_regression[..., pt_index]``) among the vertices that are ALL of:

    - not null: ``vertices_class_probs[..., null_index] < pnull_threshold``;
    - not the primary vertex: ``argmax(vertices_class_probs) != pv_class_index``;
    - a real vertex class: ``argmax(vertices_class_probs) != null_index``
      (needed because with >=3 classes a vertex can have ``argmax==null`` yet
      ``pnull < threshold``).

    For each configured output ``{name: reg_index}`` it pulls the selected
    vertex's ``vertices_regression[..., reg_index]``.

    NOTE: this selection differs from the legacy ``leading_object`` reduce
    (pT-only, no PV/null exclusion on the decorator side) — it is a NEW
    output, not a relocation; both stay untouched (additive).

    When no vertex qualifies (all-null jet, all-PV jet, or empty inputs), the
    jet-level scalars are filled with NaN deterministically via a
    masked-argmax over an all-``-inf`` pT column, so the trace stays valid
    for every batch shape (no data-dependent control flow).

    Note also: when `MaskFormerObjects`'s ``get_maskformer_outputs`` hits its
    "no object exceeds the null threshold" dummy path, it returns an all-NaN
    ``vertices_regression`` while ``vertices_class_probs`` flows through
    real — so the decorator's qualify mask can pass vertices whose
    regression is undefined, and the jet-level scalars end up NaN.

    Parameters
    ----------
    source : str
        The `MaskFormerObjects` exposed per-vertex class-probs leaf. The
        regression source defaults to the same object stream's
        ``vertices_regression`` leaf unless `regression_source` overrides it.
    outputs : Mapping[str, int]
        ``{output_name: reg_index}`` — each jet-level scalar leaf pulls the
        lead vertex's ``vertices_regression[..., reg_index]``.
    pt_index : int
        The ``vertices_regression`` channel that is the vertex pT (the
        selection key).
    pv_class_index : int
        The vertex class index that marks the primary vertex (excluded from
        selection).
    pnull_threshold : float, optional
        The null-probability cut, by default 0.5.
    null_index : int | None, optional
        The class index of the null class, by default None = the last class.
    jet_stream : str, optional
        The jet-level output stream, by default ``"jet"``.
    regression_source : str | None, optional
        Override for the per-vertex regression leaf, by default None.

    Raises
    ------
    ConfigError
        For a non-``outputs`` / wildcard source, an empty `outputs` map, a
        negative/non-int reg index or pt_index, or a missing pt selection
        channel.
    """

    def __init__(
        self,
        source: str,
        outputs: Mapping[str, int],
        pt_index: int,
        pv_class_index: int,
        pnull_threshold: float = 0.5,
        null_index: int | None = None,
        jet_stream: str = "jet",
        regression_source: str | None = None,
    ) -> None:
        super().__init__()
        parts = split_key(source)
        if any(part in {"*", "**"} for part in parts):
            raise ConfigError(
                f"MFLeadVertexDecorator source {source!r} contains a wildcard — conversion "
                "sources are concrete (design §2.2)"
            )
        if len(parts) < 2 or parts[0] != "outputs":
            raise ConfigError(
                f"MFLeadVertexDecorator source {source!r} must be a "
                "'outputs.<object_stream>.<vertices_class_probs>' leaf the MaskFormerObjects "
                "node exposes (the per-vertex class probs) — it reads a bundle leaf, not a raw "
                "prediction (USER DESIGN 2026-06-22)"
            )
        if not outputs:
            raise ConfigError(
                "MFLeadVertexDecorator: 'outputs' must map at least one jet-level scalar name to "
                "the lead vertex's regression channel index (e.g. {lead_vertex_pt: 0})"
            )
        self.source = source
        self.object_stream = parts[1]
        # the regression leaf defaults to the same object stream's vertices_regression
        self.regression_source = (
            regression_source
            if regression_source is not None
            else f"outputs.{self.object_stream}.vertices_regression"
        )
        reg_parts = split_key(self.regression_source)
        if len(reg_parts) < 2 or reg_parts[0] != "outputs":
            raise ConfigError(
                f"MFLeadVertexDecorator regression_source {self.regression_source!r} must be a "
                "'outputs.<object_stream>.<vertices_regression>' leaf (the per-vertex regression)"
            )
        self.jet_stream = jet_stream
        self.pt_index = self._checked_index(pt_index, "pt_index")
        self.pv_class_index = self._checked_index(pv_class_index, "pv_class_index")
        self.pnull_threshold = float(pnull_threshold)
        self.null_index = null_index if null_index is None else self._checked_index(
            null_index, "null_index"
        )
        # preserve declaration order (jsonargparse builds an ordered dict)
        self.outputs_map: tuple[tuple[str, int], ...] = tuple(
            (name, self._checked_index(idx, f"outputs[{name!r}]")) for name, idx in outputs.items()
        )
        self.output_keys: tuple[str, ...] = tuple(
            f"outputs.{jet_stream}.{name}" for name, _ in self.outputs_map
        )

    @staticmethod
    def _checked_index(index: Any, what: str) -> int:
        """Validate an index is a non-negative int; raises `ConfigError` otherwise."""
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ConfigError(
                f"MFLeadVertexDecorator: {what} index {index!r} must be a non-negative int"
            )
        return index

    def declare_io(self, mode: Mode) -> IO:
        """Requires the two `MaskFormerObjects` per-vertex leaves; produces jet-level scalars."""
        del mode
        requires = {
            self.source: TensorSpec(shape=None, dtype="float32", kind="data"),
            self.regression_source: TensorSpec(shape=None, dtype="float32", kind="data"),
        }
        produces = {
            key: TensorSpec(shape=None, dtype="float32", kind="data") for key in self.output_keys
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Every jet-level output is a single scalar column (width 1)."""
        del widths
        return dict.fromkeys(self.output_keys, 1)

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Select the lead vertex (highest-pT, non-null, non-PV) and emit jet-level scalars."""
        del mode
        class_probs = b.get(self.source)  # [B, M, C]
        regression = b.get(self.regression_source)  # [B, M, R]
        n_obj = class_probs.shape[1]
        batch = class_probs.shape[0]
        if n_obj == 0:
            # no object queries -> no vertex can qualify; every output is all-NaN.
            # M (the object-query axis) is fixed by the decoder and never a
            # declared dynamic export input, so this frozen branch is harmless.
            nan = torch.full((batch,), torch.nan, dtype=torch.float32)
            return dict.fromkeys(self.output_keys, nan)
        null_idx = self.null_index if self.null_index is not None else class_probs.shape[-1] - 1
        pnull = class_probs[..., null_idx]  # [B, M]
        pred_class = torch.argmax(class_probs, dim=-1)  # [B, M]
        # qualify = not-null AND not-PV AND argmax-is-a-real-vertex-class. The
        # third condition is needed because with >=3 classes a vertex can have
        # argmax==null yet pnull<threshold (thin-spread probs).
        qualify = (
            (pnull < self.pnull_threshold)
            & (pred_class != self.pv_class_index)
            & (pred_class != null_idx)
        )  # [B, M]
        pt = regression[..., self.pt_index]  # [B, M]
        # mask non-qualifying vertices to -inf so the argmax never picks them
        masked_pt = torch.where(qualify, pt, torch.full_like(pt, -torch.inf))  # [B, M]
        lead = torch.argmax(masked_pt, dim=-1)  # [B] index of the lead vertex per jet
        any_qualify = qualify.any(dim=-1)  # [B] does this jet have ANY lead vertex?
        out: dict[str, Tensor] = {}
        lead_exp = lead.unsqueeze(-1)  # [B, 1] for gather along the object axis
        for (key, (_name, reg_index)) in zip(self.output_keys, self.outputs_map, strict=True):
            col = regression[..., reg_index]  # [B, M]
            value = torch.gather(col, 1, lead_exp).squeeze(1)  # [B] lead-vertex value
            # deterministic NaN fill where no vertex qualifies (empty/all-null/all-PV)
            value = torch.where(any_qualify, value, torch.full_like(value, torch.nan))
            out[key] = value.float()
        return out

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """One float global jet-level field per configured lead-vertex scalar (no legacy oracle)."""
        del run_name, model_modules
        return [
            OutputField(h5_name=name, dtype="f4", axis="global", final=True)
            for name, _ in self.outputs_map
        ]
