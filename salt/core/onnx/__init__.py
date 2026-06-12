"""salt v2 ONNX export (design §7): declarative ``export:`` block -> Athena-ready ``.onnx``.

Public surface:

- `ExportConfig` / `ExportInput` / `ExportOutput` — the parsed ``export:``
  block (registered on the `Salt2CLI` parser, design §5.1).
- `resolve_export_config` — validation + defaulting (the ONLY place
  ``model_name`` rules apply, design §7 "Naming").
- `compile_onnx_plan` / `export_graph` — the programmatic export core
  (fixture/gate-friendly: bound modules in, checked ``.onnx`` out).
- `OnnxAdapter` — the traceable wrapper handed to ``torch.onnx.export``.
- `check_onnx` — the torch-vs-onnxruntime sweep checker (design §7.6).
- ``salt2 export`` — the CLI (`salt.core.onnx.export.main`), dispatched
  from `salt.core.main`.
"""

from salt.core.onnx.adapter import OnnxAdapter
from salt.core.onnx.check import CheckResult, check_onnx, compare_once, make_session
from salt.core.onnx.config import (
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
