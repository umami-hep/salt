"""`VertexUnionFind` — the in-graph union-find conversion node."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from salt.graph.bundle import Bundle
from salt.graph.spec import IO, Mode, TensorSpec, sym_dim, unflatten_spec
from salt.model.base import SaltModelModule

# The two scripted union-find helpers + the MaskFormer export math are inlined
# verbatim in salt.onnx.reduces (the legacy reduce path); the conversion
# nodes below reuse those exact copies so the folded node and the legacy
# reduce can never drift.
from salt.onnx.reduces import mask_fill_flattened
from salt.utils.union_find import get_node_assignment_jit


class VertexUnionFind(SaltModelModule):
    """In-graph union-find conversion node (folds the legacy union-find export reduce).

    Reads the RAW ``preds.<stream>.<task>`` ``[E, 1]`` edge scores the
    vertexing task publishes in ONNX mode plus the stream's
    ``masks.<stream>`` pad mask, and runs the ``@torch.jit.script``
    union-find inside ``forward(b, Mode.ONNX)``:

        ``get_node_assignment_jit`` -> ``mask_fill_flattened`` ->
        ``.reshape(-1).char()``

    producing the same int8 ``[L]`` per-token leaf the legacy chain emits.
    The two scripted helpers are reused verbatim from ``reduces.py`` (one
    source of truth, no drift).

    Because it declares the SAME ``preds.*`` port the legacy reduce reads,
    the demand-closure pulls the identical vertexing task node into the ONNX
    plan. The ``reshape(-1)`` collapse has no recoverable last dim in a bind,
    so `derived_widths` re-emits width 1.

    This node is ONNX-only by construction: the union-find chain is shaped
    for the traced export batch; it is never wired into a TEST H5 config
    (the eval vertex columns come from a separate per-token path).

    Parameters
    ----------
    task : str
        The source vertexing task's instance name — the node reads the RAW
        ``preds.<stream>.<task>`` ``[E, 1]`` edge scores.
    stream : str
        The constituent stream the head publishes under (also the stream the
        output is written under).
    name : str, optional
        The output leaf name; defaults to `task`.
    """

    def __init__(
        self,
        task: str,
        stream: str,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.stream = stream
        self.output_name = name if name is not None else task
        self.pred_key = f"preds.{stream}.{task}"
        self.mask_key = f"masks.{stream}"
        self.output_key = f"outputs.{stream}.{self.output_name}"

    def declare_io(self, mode: Mode) -> IO:
        """Declare the RAW ``preds.*`` edge scores + ``masks.*`` -> the int8 ``outputs.*`` leaf."""
        del mode
        # ONNX-only ports: this node is ONNX-only by construction (class
        # docstring), so gating to Mode.ONNX (rather than demand-gating like
        # the shared softmax producers) keeps it inactive in FIT/VAL/TEST —
        # a config that opts the vertexing head out of TEST does not trip
        # the planner's pre-prune connectivity check on this node's require.
        requires = {
            self.pred_key: TensorSpec(shape=None, dtype="float32", kind="data", modes=Mode.ONNX),
            self.mask_key: TensorSpec(
                shape=("B", sym_dim("T", self.stream)), dtype="bool", kind="pad_mask",
                modes=Mode.ONNX,
            ),
        }
        produces = {
            self.output_key: TensorSpec(shape=None, dtype="int8", kind="data", modes=Mode.ONNX)
        }
        return IO(requires=unflatten_spec(requires), produces=unflatten_spec(produces))

    def derived_widths(self, widths: Mapping[str, int]) -> dict[str, int]:
        """The union-find ``reshape(-1)`` collapses to a single per-token index column (width 1)."""
        del widths
        return {self.output_key: 1}

    def forward(self, b: Bundle, mode: Mode) -> dict[str, Tensor]:
        """Run the ``@torch.jit.script`` union-find chain (verbatim, see class docstring)."""
        del mode
        edge_scores = b.get(self.pred_key)  # RAW [E, 1] scores
        pad_mask = b.get(self.mask_key)  # the all-valid pad mask
        vertex_indices = get_node_assignment_jit(edge_scores, pad_mask)
        vertex_list = mask_fill_flattened(vertex_indices, pad_mask)
        return {self.output_key: vertex_list.reshape(-1).char()}
