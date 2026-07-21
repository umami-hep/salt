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
from salt.outputs.maskformer_objects import MaskFormerObject, MaskFormerObjects
from salt.outputs.maskformer_objects_sink import MaskFormerObjectsSink
from salt.outputs.mf_lead_vertex_decorator import MFLeadVertexDecorator
from salt.outputs.onnx_sink import OnnxExportLeaf, OnnxExportSink
from salt.outputs.output_column import OutputColumn
from salt.outputs.output_field import OutputField
from salt.outputs.pad_mask_writer import PadMaskWriter
from salt.outputs.run_task_output import OutputSectionWriter, RunTaskOutput
from salt.outputs.task_output import (
    ClassProbs,
    SeqClassIndex,
    SeqClassProbs,
    TaskOutput,
)
from salt.outputs.vertex_union_find import VertexUnionFind

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
    "MaskFormerObjectsSink",
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
    "VertexUnionFind",
]
