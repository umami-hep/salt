"""salt v2 prediction writers — `WriterCallback` + writer modules (design §2.7, §8).

The top-level ``writers:`` config block (design §5.1) is assembled by
`Salt2CLI` into one `WriterCallback` owning a single ftag ``H5Writer`` sink;
the shipped modules reproduce the v1 `PredictionWriter` output contract
(``base2.yaml`` order ``inputs_copy -> tasks -> pad_mask``). Writers declare
their consumed bundle keys, making them first-class TEST-graph sinks
(demand-gating + the dead-preds hard error, design §4.2).
"""

from salt.core.writers.base import WriteCtx, Writer, WriterDeclareCtx, task_modules
from salt.core.writers.callback import DEFAULT_OUTPUT, WriterCallback
from salt.core.writers.modules import InputCopyWriter, PadMaskWriter, TaskWriter

__all__ = [
    "DEFAULT_OUTPUT",
    "InputCopyWriter",
    "PadMaskWriter",
    "TaskWriter",
    "WriteCtx",
    "Writer",
    "WriterCallback",
    "WriterDeclareCtx",
    "task_modules",
]
