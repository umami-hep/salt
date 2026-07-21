"""`MaskFormerObjects` — the MaskFormer object reconstruction node (+ deprecated alias)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.errors import ConfigError
from salt.graph.spec import IO, Mode, TensorSpec, unflatten_spec
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
    constituents set to -1) — the eval-H5 ``MaskIndex`` semantics. The node is
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
        """ONNX ports (same gate as `VertexUnionFind`): the null-suppression + pT-reorder
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
        """The eval-H5 ``MaskIndex`` reconstruction.

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
