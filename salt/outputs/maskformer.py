"""MaskFormer output nodes: `MaskFormerObjects` reconstruction node (+ deprecated
alias) and `MFLeadVertexDecorator` lead-vertex jet-level decorator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, split_key, unflatten_spec
from salt.model.base import SaltModelModule

# The MaskFormer export math is inlined verbatim in salt.onnx.reduces (the
# shared math seam); this node reuses that exact copy so the two can never drift.
from salt.onnx.reduces import get_maskformer_outputs
from salt.utils.mask_utils import indices_from_mask


class MaskFormerObjects(SaltModelModule):
    """MaskFormer object reconstruction node — the single reconstruction path for both modes.

    In **ONNX** it runs `get_maskformer_outputs` once inside ``forward`` (the
    null-suppression + pT reorder + index math) and exposes its products so a
    downstream `MFLeadVertexDecorator` can read the reordered per-vertex
    outputs without redoing any of the heavy lifting. Produces:

    - **object_index** (``outputs.<constituent_stream>.<index_name>``, int8
      per-token): the per-constituent owning-object index.
    - **leading_object** (``outputs.<stream>.<leading_name>``, float32
      global): the leading object's R de-scaled regression scalars.
    - **vertices_class_probs** / **vertices_regression**
      (``outputs.<stream>.<...>``, float32 ``[B, M, C]`` / ``[B, M, R]``):
      the reordered (null-suppressed + pT-ordered) per-vertex tensors,
      exposed as intermediate leaves for `MFLeadVertexDecorator`.

    `get_maskformer_outputs` is reused verbatim from ``reduces.py`` (one
    source of truth). The node declares all three cross-node reads it needs
    (``objects.class_probs`` / ``objects.masks`` /
    ``preds.<stream>.<reg_task>``) so the demand-closure keeps the upstream
    decoder + regression task alive in the ONNX plan; all three tensors are
    cloned before `get_maskformer_outputs` (which mutates ``masks``/
    ``regression`` in place for null-suppression + pT reorder), so the
    write-once bundle is never mutated.

    In **TEST** it publishes the same ``object_index`` leaf, computed from the
    RAW decoder masks (``indices_from_mask(masks.sigmoid() > 0.5)``, padded
    constituents set to -1) — the eval-H5 ``HadronIndex`` semantics. The node is
    the ONE reconstruction path for both modes: the sink is a dumb terminal
    that only packs the leaf. Demand-gated (a TEST plan pulls it in only when
    an object-group field sources ``outputs.<constituent>.<index_name>``); FIT
    and VAL declare nothing, so the node never enters those plans.

    Parameters
    ----------
    regression_task : str
        The object-regression task's instance name — the node reads the
        de-scaled ``preds.<stream>.<regression_task>`` ``[B, M, R]``
        predictions.

        The folded ``object_index`` / ``leading_object`` leaves are
        bitwise-equal to the legacy reduces ONLY when
        ``regression_task == "regression"`` (the legacy default the export
        path asserts). A non-default `regression_task` reorders by a
        different regression tensor than the legacy reduce and has no parity
        oracle; keep the default for any config that must match the legacy
        ONNX path.
    stream : str, optional
        The object stream the decoder publishes under, by default
        ``"objects"``. Also the object output stream.
    leading_name : str, optional
        The leading-object regression leaf name, by default
        ``"leading_object"``.
    index_name : str, optional
        The per-constituent object-index leaf name, by default
        ``"object_index"``.
    n_reg : int
        The object-regression target count R (the leading-regression output
        width). The leading leaf is sliced to ``leading_reg[:, :n_reg]`` to
        reproduce the legacy reduce exactly, including the no-objects/
        empty-track dummy path. Must match the configured leading-object
        ``names`` count.
    constituent_stream : str, optional
        The constituent stream the per-token index leaf is written under and
        whose dynamic axis the index carries, by default ``"tracks"``.
    vertices_class_probs_name : str, optional
        The exposed reordered per-vertex class-probs leaf name, by default
        ``"vertices_class_probs"``.
    vertices_regression_name : str, optional
        The exposed reordered per-vertex regression leaf name, by default
        ``"vertices_regression"``.
    """

    def __init__(
        self,
        n_reg: int,
        regression_task: str = "regression",
        stream: str = "objects",
        leading_name: str = "leading_object",
        index_name: str = "object_index",
        constituent_stream: str = "tracks",
        vertices_class_probs_name: str = "vertices_class_probs",
        vertices_regression_name: str = "vertices_regression",
    ) -> None:
        super().__init__()
        if not isinstance(n_reg, int) or isinstance(n_reg, bool) or n_reg < 1:
            raise ConfigError(
                f"MaskFormerObjects: n_reg must be a positive int (the leading-object regression "
                f"target count R, matching the export leading names), got {n_reg!r}"
            )
        self.stream = stream
        self.constituent_stream = constituent_stream
        self.regression_task = regression_task
        self.leading_name = leading_name
        self.index_name = index_name
        self.n_reg = n_reg
        self.class_probs_key = f"{stream}.class_probs"
        self.masks_key = f"{stream}.masks"
        self.reg_key = f"preds.{stream}.{regression_task}"
        # the constituent pad mask the TEST index reconstruction reads (padded
        # constituents -> -1), the same demand the deleted sink used to fold.
        self.pad_key = f"masks.{constituent_stream}"
        # the GLOBAL leading-regression leaf is written under the OBJECT stream; the
        # PER-TOKEN index leaf under the CONSTITUENT stream (its dynamic axis source)
        self.leading_key = f"outputs.{stream}.{leading_name}"
        self.index_key = f"outputs.{constituent_stream}.{index_name}"
        # the exposed reordered per-vertex outputs (object stream) the decorator reads
        self.vertices_class_probs_key = f"outputs.{stream}.{vertices_class_probs_name}"
        self.vertices_regression_key = f"outputs.{stream}.{vertices_regression_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Mode-branched: TEST -> raw-mask ``object_index``; ONNX -> the reorder leaves; else empty.

        FIT/VAL declare nothing (the node never enters those plans, so their
        plan hashes are unaffected); the ONNX ports are byte-unchanged.
        """
        if mode & Mode.TEST:
            return self._test_io()
        if mode & Mode.ONNX:
            return self._onnx_io()
        return IO(requires={}, produces={})

    def _test_io(self) -> IO:
        """TEST: require the RAW masks + constituent pad mask; produce ``object_index`` [B, T]."""
        requires = {
            self.masks_key: TensorSpec(shape=None, dtype="float32", kind="data", modes=Mode.TEST),
            self.pad_key: TensorSpec(
                shape=None, dtype="bool", kind="pad_mask", modes=Mode.TEST
            ),
        }
        produces = {
            self.index_key: TensorSpec(shape=None, dtype="int64", kind="data", modes=Mode.TEST),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def _onnx_io(self) -> IO:
        """ONNX ports (Mode.ONNX gate): the null-suppression + pT-reorder
        chain is shaped for the traced export batch, so this node is inactive in FIT/VAL/TEST
        for these leaves — a config that wires it for ONNX export alongside an object-regression
        head opted out of TEST eval does not trip the planner's pre-prune connectivity check.
        """
        requires = {
            self.class_probs_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
            self.masks_key: TensorSpec(shape=None, dtype="float32", kind="data", modes=Mode.ONNX),
            self.reg_key: TensorSpec(shape=None, dtype="float32", kind="data", modes=Mode.ONNX),
        }
        produces = {
            self.leading_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
            self.index_key: TensorSpec(shape=None, dtype="int8", kind="data", modes=Mode.ONNX),
            # the exposed reordered per-vertex outputs the MFLeadVertexDecorator reads
            # (a node->node edge — the decorator's demand keeps this node alive)
            self.vertices_class_probs_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
            self.vertices_regression_key: TensorSpec(
                shape=None, dtype="float32", kind="data", modes=Mode.ONNX
            ),
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """Width-resolve the leaves: index collapses to 1, leading + vertex regression follow n_reg."""
        del widths
        return {
            self.index_key: 1,
            self.leading_key: self.n_reg,
            self.vertices_regression_key: self.n_reg,
        }

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """TEST -> raw-mask ``object_index``; ONNX -> ``get_maskformer_outputs`` reorder leaves."""
        if mode & Mode.TEST:
            return self._forward_test(b)
        objects = {
            "class_probs": b.get(self.class_probs_key).clone(),
            "masks": b.get(self.masks_key).clone(),
            "regression": b.get(self.reg_key).clone(),
        }
        leading_reg, indices, vertices_class_probs, vertices_regression = get_maskformer_outputs(
            objects, apply_reorder=True
        )
        # get_maskformer_outputs returns indices=None at L == 0 (n_tracks == 0).
        # This never happens on the export path (torch.onnx.export always traces
        # at a fixed L > 0), so this guard folds to a constant-False branch that
        # emits no ops; it only removes a latent AttributeError for a future
        # eager TEST-mode wiring of this node.
        if indices is None:
            empty_index = torch.zeros(0, dtype=torch.int8)
        else:
            empty_index = indices.reshape(-1).char()
        # leading_reg is [B, R] normally but [1, n_obj] in the no-objects/
        # empty-track dummy path; slice to [:, :n_reg] to reproduce both paths
        # exactly (matches the legacy reduce's leading_reg[0, i] for i in range(R)).
        return {
            self.leading_key: leading_reg[:, : self.n_reg],
            self.index_key: empty_index,
            # the reordered (null-suppressed + pT-ordered) per-vertex outputs — passed
            # through verbatim as get_maskformer_outputs returns them (the decorator
            # does the lead-vertex selection; all heavy lifting is here, cloned above)
            self.vertices_class_probs_key: vertices_class_probs,
            self.vertices_regression_key: vertices_regression,
        }

    def _forward_test(self, b: Bundle) -> dict[str, Tensor]:
        """The eval-H5 ``HadronIndex`` reconstruction.

        Per-constituent owning-object index from the RAW decoder masks
        (``indices_from_mask(sigmoid > 0.5)`` -> -2 where no object claims a
        constituent), with padded constituents forced to -1. No reordering /
        null-suppression (that is the ONNX path) — the eval-H5 index is a
        function of the raw masks.
        """
        masks = b.get(self.masks_key)  # [B, M, T] raw logits
        pad = b.get(self.pad_key)  # [B, T] bool, True = padded
        idx = indices_from_mask(masks.sigmoid() > 0.5)  # [B, T] int64, -2 = no object
        idx = torch.where(pad, torch.full_like(idx, -1), idx)  # padded constituents -> -1
        return {self.index_key: idx}


# One-window alias: `MaskFormerObject` was renamed `MaskFormerObjects`. The
# promoted node is a strict superset, so an existing `MaskFormerObject`
# config keeps working via this alias. Remove after the migration window.
MaskFormerObject = MaskFormerObjects
"""Deprecated alias for `MaskFormerObjects`."""


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
