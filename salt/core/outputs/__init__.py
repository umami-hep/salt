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

from salt.core.outputs.producers import (
    ClassProbs,
    ClassProbsOp,
    Combination,
    ConversionOp,
    IdentityOp,
    MaskFormerObject,
    MaskFormerObjects,
    MFLeadVertexDecorator,
    OutputField,
    Regression,
    RegressionDescaleOp,
    SeqClassIndex,
    SeqClassIndexOp,
    SeqClassProbs,
    SeqClassProbsOp,
    TaskOutput,
    VertexUnionFind,
)
from salt.core.outputs.sinks import (
    CollectOutputs,
    H5OutputSink,
    H5OutputWriter,
    OnnxExportLeaf,
    OnnxExportSink,
    OutputColumn,
)
from salt.core.outputs.writers import (
    InputCopyWriter,
    MaskFormerObjectsSink,
    OutputSectionWriter,
    PadMaskWriter,
    RunTaskOutput,
)

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
