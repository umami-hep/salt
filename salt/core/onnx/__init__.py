"""salt v2 ONNX export: writer manifest + export block -> ``.onnx``.

Public surface: `ExportConfig`/`ExportInput`/`ExportOutput`/`ExportCombine` (the
parsed ``export:`` block + output manifest), `resolve_export_config`/
`attach_manifest` (validation + manifest resolution), `compile_onnx_plan`/
`export_graph` (the programmatic export core), `OnnxAdapter` (the traced
wrapper handed to ``torch.onnx.export``), `check_onnx` (the torch-vs-onnxruntime
sweep checker), and the ``salt2 export`` CLI (`salt.core.onnx.export.main`).
"""

from salt.core.onnx.adapter import OnnxAdapter
from salt.core.onnx.check import CheckResult, check_onnx, compare_once, make_session
from salt.core.onnx.config import (
    ExportCombine,
    ExportConfig,
    ExportInput,
    ExportOutput,
    attach_manifest,
    combine_insertion_index,
    manifest_table,
    ordered_output_names,
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
    "attach_manifest",
    "build_gnn_config",
    "check_onnx",
    "combine_insertion_index",
    "compare_once",
    "compile_onnx_plan",
    "derive_onnx_sources",
    "export_graph",
    "load_run_metadata",
    "make_session",
    "manifest_table",
    "ordered_output_names",
    "resolve_export_config",
    "sanitised_model_name",
    "validate_model_name",
    "write_metadata",
]
