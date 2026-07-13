"""`MaskFormerObjects` — the MaskFormer object reconstruction node (+ deprecated alias)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from salt.core.graph.bundle import Bundle
from salt.core.graph.errors import ConfigError
from salt.core.graph.spec import IO, Mode, TensorSpec, unflatten_spec

# The MaskFormer export math is inlined verbatim in salt.core.onnx.reduces (the
# legacy reduce path); this node reuses that exact copy so the folded node and
# the legacy reduce can never drift.
from salt.core.onnx.reduces import get_maskformer_outputs
from salt.core.outputs.output_field import OutputField
from salt.core.outputs.task_output import _UNNAMED


class MaskFormerObjects(nn.Module):
    """MaskFormer object reconstruction node (the "writer" half of the two-node MaskFormer split).

    Runs `get_maskformer_outputs` once inside ``forward(b, Mode.ONNX)`` (the
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

    ONNX-only by construction (like `VertexUnionFind`): never wired into a
    TEST H5 config.

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
        self.name = _UNNAMED
        self.stream = stream
        self.constituent_stream = constituent_stream
        self.regression_task = regression_task
        self.leading_name = leading_name
        self.index_name = index_name
        self.n_reg = n_reg
        self.class_probs_key = f"{stream}.class_probs"
        self.masks_key = f"{stream}.masks"
        self.reg_key = f"preds.{stream}.{regression_task}"
        # the GLOBAL leading-regression leaf is written under the OBJECT stream; the
        # PER-TOKEN index leaf under the CONSTITUENT stream (its dynamic axis source)
        self.leading_key = f"outputs.{stream}.{leading_name}"
        self.index_key = f"outputs.{constituent_stream}.{index_name}"
        # the exposed reordered per-vertex outputs (object stream) the decorator reads
        self.vertices_class_probs_key = f"outputs.{stream}.{vertices_class_probs_name}"
        self.vertices_regression_key = f"outputs.{stream}.{vertices_regression_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare all three maskformer reads -> the leading-regression + object-index leaves.

        The index width has no recoverable last dim in a bind (width 1); the
        leading width follows the regression port (or `n_reg`) — both
        re-emitted via `derived_widths`.
        """
        del mode
        # ONNX-only ports (same gate as `VertexUnionFind`): the null-suppression
        # + pT-reorder chain is shaped for the traced export batch, so this node
        # is inactive in FIT/VAL/TEST — a config that wires it for ONNX export
        # alongside an object-regression head opted out of TEST eval does not
        # trip the planner's pre-prune connectivity check on this require.
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
        """Run ``get_maskformer_outputs`` once -> object_index + leading + the per-vertex leaves."""
        del mode
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

    def output_columns(
        self, run_name: str, model_modules: Mapping[str, Any]
    ) -> list[OutputField]:
        """The MaskFormer object leaves' field manifest — index + leading + intermediates.

        The per-vertex ``vertices_class_probs`` / ``vertices_regression``
        are exposed intermediates (``final=False``) that sinks must not
        auto-collect.
        """
        del run_name, model_modules
        return [
            OutputField(
                h5_name=None, onnx_name=self.index_name, dtype="int8",
                axis="per_token", final=True,
            ),
            OutputField(h5_name=self.leading_name, dtype="f4", axis="global", final=True),
            OutputField(
                h5_name="vertices_class_probs", onnx_name=None, dtype="f4",
                axis="per_token", final=False,
            ),
            OutputField(
                h5_name="vertices_regression", onnx_name=None, dtype="f4",
                axis="per_token", final=False,
            ),
        ]


# One-window alias: `MaskFormerObject` was renamed `MaskFormerObjects`. The
# promoted node is a strict superset, so an existing `MaskFormerObject`
# config keeps working via this alias. Remove after the migration window.
MaskFormerObject = MaskFormerObjects
"""Deprecated alias for `MaskFormerObjects`."""
