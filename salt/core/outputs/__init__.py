"""salt.core.outputs — the two-layer output-writing system.

A clean split mirroring the model's dataflow:

- **producers** (`salt.core.outputs.producers`) — in-graph `GraphModule`s that
  read raw ``preds.<stream>.<task>`` and write eval/export-ready
  ``outputs.<stream>.<name>`` leaves. They live in ``model.modules`` and are
  carried by the existing planner / executor / bind machinery. Active in
  TEST/ONNX only, so the planner demand-prunes them from FIT/VAL and the
  training path is unperturbed.
- **sinks** (`salt.core.outputs.sinks`) — terminal Lightning callbacks that
  consume the producers' ``outputs.*`` leaves. Each is told which output names
  it wants; its declared demand anchors the TEST plan's sinks, keeping the
  producers alive in TEST.

The ``H5OutputWriter`` sink reproduces the v1
``inputs_copy -> tasks -> pad_mask`` eval H5 with semantic array parity,
owning the serialisation concerns (`u2s` packing, per-token pad
re-expansion, input copies, pad-mask columns) the producers must not know
about (`OutputColumn` is its declarative per-leaf column schema).
"""

from __future__ import annotations

from salt.core.outputs.collect_outputs import CollectOutputs
from salt.core.outputs.combination import Combination
from salt.core.outputs.conversion_ops import (
    ClassProbsOp,
    ConversionOp,
    IdentityOp,
    SeqClassIndexOp,
    SeqClassProbsOp,
)
from salt.core.outputs.h5_sink import H5OutputSink, H5OutputWriter
from salt.core.outputs.input_copy_writer import InputCopyWriter
from salt.core.outputs.maskformer_objects import MaskFormerObject, MaskFormerObjects
from salt.core.outputs.maskformer_objects_sink import MaskFormerObjectsSink
from salt.core.outputs.mf_lead_vertex_decorator import MFLeadVertexDecorator
from salt.core.outputs.onnx_sink import OnnxExportLeaf, OnnxExportSink
from salt.core.outputs.output_column import OutputColumn
from salt.core.outputs.output_field import OutputField
from salt.core.outputs.pad_mask_writer import PadMaskWriter
from salt.core.outputs.regression_descale_op import RegressionDescaleOp
from salt.core.outputs.run_task_output import OutputSectionWriter, RunTaskOutput
from salt.core.outputs.task_output import (
    ClassProbs,
    Regression,
    SeqClassIndex,
    SeqClassProbs,
    TaskOutput,
)
from salt.core.outputs.vertex_union_find import VertexUnionFind

__all__ = [
    "ClassProbs",
    "ClassProbsOp",
    "CollectOutputs",
    "Combination",
    "ConversionOp",
    "H5OutputSink",
    "H5OutputWriter",
    "IdentityOp",
    "InputCopyWriter",
    "MFLeadVertexDecorator",
    "MaskFormerObject",
    "MaskFormerObjects",
    "MaskFormerObjectsSink",
    "OnnxExportLeaf",
    "OnnxExportSink",
    "OutputColumn",
    "OutputField",
    "OutputSectionWriter",
    "PadMaskWriter",
    "Regression",
    "RegressionDescaleOp",
    "RunTaskOutput",
    "SeqClassIndex",
    "SeqClassIndexOp",
    "SeqClassProbs",
    "SeqClassProbsOp",
    "TaskOutput",
    "VertexUnionFind",
]
