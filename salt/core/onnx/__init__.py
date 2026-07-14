"""salt v2 ONNX export: folded export sink + export block -> ``.onnx``
(config dataclasses, export core, adapter, checker, CLI).
"""

from salt.core.onnx.adapter import OnnxAdapter
from salt.core.onnx.check import CheckResult, check_onnx, compare_once, make_session
from salt.core.onnx.config import (
    ExportCombine,
    ExportConfig,
    ExportInput,
    ExportOutput,
    resolve_export_config,
    sanitised_model_name,
    validate_model_name,
)
from salt.core.onnx.export import (
    ExportResult,
    compile_onnx_plan,
    derive_onnx_sources,
    export_graph,
)
from salt.core.onnx.metadata import build_gnn_config, load_run_metadata, write_metadata

__all__ = [
    "CheckResult",
    "ExportCombine",
    "ExportConfig",
    "ExportInput",
    "ExportOutput",
    "ExportResult",
    "OnnxAdapter",
    "build_gnn_config",
    "check_onnx",
    "compare_once",
    "compile_onnx_plan",
    "derive_onnx_sources",
    "export_graph",
    "load_run_metadata",
    "make_session",
    "resolve_export_config",
    "sanitised_model_name",
    "validate_model_name",
    "write_metadata",
]
