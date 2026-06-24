"""salt.core.outputs — the two-layer output-writing system (design §2).

Replaces the M4.5 single-manifest writers (``salt/core/writers/``) with a
clean split mirroring the model's dataflow:

- **producers** (`salt.core.outputs.producers`) — in-graph `GraphModule`s that
  read raw ``preds.<stream>.<task>`` and write eval/export-ready
  ``outputs.<stream>.<name>`` leaves. They live in ``model.modules`` and are
  carried by the existing planner / executor / bind machinery. Active in
  TEST/ONNX only, so the planner demand-prunes them from FIT/VAL (the §4
  risk 4 keystone) and the training path is unperturbed.
- **sinks** (`salt.core.outputs.sinks`) — terminal Lightning callbacks that
  consume the producers' ``outputs.*`` leaves. Each is told which output names
  it wants; its declared demand anchors the TEST plan's sinks, keeping the
  producers alive in TEST.

P0 ships the generic ``TaskOutput`` producer (identity copy) and the minimal
``CollectOutputs`` no-op sink — the smallest slice proving
producer -> ``outputs.*`` -> sink with demand-gating. P1 adds the
classification + regression conversion producers (`ClassProbs`,
`SeqClassProbs`, `SeqClassIndex`, `Regression`, and the `ConversionOp`
strategies behind the generic ``TaskOutput(op=...)``) and the ``H5OutputWriter``
sink — the real H5 serialiser that reproduces the v1
``inputs_copy -> tasks -> pad_mask`` eval H5 with semantic array parity, owning
the serialisation concerns (`u2s` packing, per-token pad re-expansion, input
copies, pad-mask columns) the producers must not know about (`OutputColumn` is
its declarative per-leaf column schema).
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

__all__ = [
    "ClassProbs",
    "ClassProbsOp",
    "CollectOutputs",
    "Combination",
    "ConversionOp",
    "H5OutputSink",
    "H5OutputWriter",
    "IdentityOp",
    "MFLeadVertexDecorator",
    "MaskFormerObject",
    "MaskFormerObjects",
    "OnnxExportLeaf",
    "OnnxExportSink",
    "OutputColumn",
    "OutputField",
    "Regression",
    "RegressionDescaleOp",
    "SeqClassIndex",
    "SeqClassIndexOp",
    "SeqClassProbs",
    "SeqClassProbsOp",
    "TaskOutput",
    "VertexUnionFind",
]
