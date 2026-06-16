"""salt v2 prediction writers — the single output manifest (design §2.7, §8; M4.5).

The top-level ``writers:`` config block (design §5.1) is assembled by
`Salt2CLI` into one `WriterCallback` owning a single ftag ``H5Writer`` sink;
the shipped modules reproduce the v1 `PredictionWriter` output contract
(``base2.yaml`` order ``inputs_copy -> tasks -> pad_mask``). Writers declare
their consumed bundle keys, making them first-class TEST-graph sinks
(demand-gating + the dead-preds hard error, design §4.2).

Since the M4.5 unified-manifest amendment, writers are ALSO the ONNX output
manifest: `Writer.onnx_outputs` declares export entries (M4
`ExportOutput`s) that `WriterCallback.onnx_manifest` assembles for the
exporter — TEST executes, ONNX declares; eval columns and Athena outputs
derive from one set of declarations. `ExportOnlyWriter` is the blessed
export-only pattern; `salt.core.writers.names` owns the cross-mode suffix
constants.
"""

from salt.core.writers.base import (
    ExportOnlyWriter,
    WriteCtx,
    Writer,
    WriterDeclareCtx,
    task_modules,
)
from salt.core.writers.callback import DEFAULT_OUTPUT, WriterCallback
from salt.core.writers.integrated_gradients import (
    IntegratedGradientWriter,
    integrated_gradients,
)
from salt.core.writers.maskformer import MaskFormerObjectWriter
from salt.core.writers.modules import InputCopyWriter, PadMaskWriter, TaskWriter
from salt.core.writers.names import OBJECT_INDEX, VERTEX_INDEX, ModeSplitSuffix, pascal_case

__all__ = [
    "DEFAULT_OUTPUT",
    "OBJECT_INDEX",
    "VERTEX_INDEX",
    "ExportOnlyWriter",
    "InputCopyWriter",
    "IntegratedGradientWriter",
    "MaskFormerObjectWriter",
    "ModeSplitSuffix",
    "PadMaskWriter",
    "TaskWriter",
    "WriteCtx",
    "Writer",
    "WriterCallback",
    "WriterDeclareCtx",
    "integrated_gradients",
    "pascal_case",
    "task_modules",
]
