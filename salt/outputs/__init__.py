"""salt.outputs — the two-layer output-writing system: in-graph producers
(``preds.*`` -> ``outputs.*``) and terminal sink callbacks that consume them.
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
from salt.outputs.h5_sink import H5OutputSink, H5OutputWriter
from salt.outputs.input_copy_writer import InputCopyWriter
from salt.outputs.maskformer import (
    MaskFormerObject,
    MaskFormerObjects,
    MFLeadVertexDecorator,
)
from salt.outputs.onnx_sink import OnnxExportLeaf, OnnxExportSink
from salt.outputs.output_schema import (
    ObjectGroup,
    ObjectGroupField,
    OutputColumn,
    OutputField,
)
from salt.outputs.pad_mask_writer import PadMaskWriter
from salt.outputs.run_task_output import OutputSectionWriter, RunTaskOutput
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
    "H5OutputWriter",
    "IdentityOp",
    "InputCopyWriter",
    "MFLeadVertexDecorator",
    "MaskFormerObject",
    "MaskFormerObjects",
    "ObjectGroup",
    "ObjectGroupField",
    "OnnxExportLeaf",
    "OnnxExportSink",
    "OutputColumn",
    "OutputField",
    "OutputSectionWriter",
    "PadMaskWriter",
    "RunTaskOutput",
    "SeqClassIndex",
    "SeqClassIndexOp",
    "SeqClassProbs",
    "SeqClassProbsOp",
    "TaskOutput",
]
