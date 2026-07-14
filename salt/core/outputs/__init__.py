"""salt.core.outputs — the two-layer output-writing system: in-graph producers
(``preds.*`` -> ``outputs.*``) and terminal sink callbacks that consume them.
"""

from __future__ import annotations

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
