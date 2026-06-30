"""salt v2 writer base types — re-exports from ``salt.core.outputs`` (design §2.7).

The top-level ``writers:`` config block (``WriterCallback`` + ``TaskWriter`` /
``InputCopyWriter`` / ``PadMaskWriter``) was removed in W6c; migrate to the
``outputs:`` section + a ``callbacks:`` persistence sink — see
``gn2v2-dummy.yaml`` for the canonical cutover pattern.

This module re-exports the REMAINING stable surface from ``salt.core.outputs``:
the `Writer` ABC, the context types, `ExportOnlyWriter`, `MaskFormerObjectWriter`,
and the cross-mode suffix constants (``VERTEX_INDEX``, ``OBJECT_INDEX``, etc.).
"""

from salt.core.outputs.writer_base import (
    ExportOnlyWriter,
    WriteCtx,
    Writer,
    WriterDeclareCtx,
    task_modules,
)
from salt.core.outputs.maskformer import MaskFormerObjectWriter
from salt.core.outputs.names import OBJECT_INDEX, VERTEX_INDEX, ModeSplitSuffix, pascal_case

__all__ = [
    "OBJECT_INDEX",
    "VERTEX_INDEX",
    "ExportOnlyWriter",
    "MaskFormerObjectWriter",
    "ModeSplitSuffix",
    "WriteCtx",
    "Writer",
    "WriterDeclareCtx",
    "pascal_case",
    "task_modules",
]
