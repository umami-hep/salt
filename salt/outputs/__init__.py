"""salt.outputs — the two-layer output-writing system: in-graph producers
(``preds.*`` -> ``outputs.*``) and the terminal sink nodes that consume them.
"""

from __future__ import annotations

from salt.outputs.combination import Combination
from salt.outputs.conversion_ops import (
    ClassProbsOp,
    ConversionOp,
    IdentityOp,
    SeqClassIndexOp,
    SeqClassProbsOp,
)
from salt.outputs.input_copy_writer import InputCopyWriter
from salt.outputs.maskformer import (
    MaskFormerObjects,
    MFLeadVertexDecorator,
)
from salt.outputs.output_schema import (
    ObjectGroup,
    ObjectGroupField,
    OutputColumn,
    OutputField,
)
from salt.outputs.pad_mask_writer import PadMaskWriter
from salt.outputs.run_task_output import OutputSectionWriter, RunTaskOutput
from salt.outputs.sinks.h5_sink import H5OutputSink
from salt.outputs.sinks.jsonl_sink import JSONLOutputSink
from salt.outputs.sinks.onnx_sink import OnnxExportLeaf, OnnxExportSink
from salt.outputs.sinks.registry import iter_sinks, register_sink, sink_registry
from salt.outputs.sinks.sink import (
    Node,
    RuntimeSink,
    SinkContext,
    is_test_persistence_sink,
)
from salt.outputs.task_output import (
    ClassProbs,
    SeqClassIndex,
    SeqClassProbs,
    TaskOutput,
)

__all__ = [
    "ClassProbs",
    "ClassProbsOp",
    "Combination",
    "ConversionOp",
    "H5OutputSink",
    "IdentityOp",
    "InputCopyWriter",
    "JSONLOutputSink",
    "MFLeadVertexDecorator",
    "MaskFormerObjects",
    "Node",
    "ObjectGroup",
    "ObjectGroupField",
    "OnnxExportLeaf",
    "OnnxExportSink",
    "OutputColumn",
    "OutputField",
    "OutputSectionWriter",
    "PadMaskWriter",
    "RunTaskOutput",
    "RuntimeSink",
    "SeqClassIndex",
    "SeqClassIndexOp",
    "SeqClassProbs",
    "SeqClassProbsOp",
    "SinkContext",
    "TaskOutput",
    "is_test_persistence_sink",
    "iter_sinks",
    "register_sink",
    "sink_registry",
]
