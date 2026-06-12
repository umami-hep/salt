"""salt v2 ONNX export (design §7; M4.5): writer manifest + export block -> ``.onnx``.

Public surface:

- `ExportConfig` / `ExportInput` / `ExportCombine` — the parsed ``export:``
  block (registered on the `Salt2CLI` parser, design §5.1) — the
  EXPORT-ONLY half since M4.5: inputs, model name, rename/combine.
- `ExportOutput` — one output-manifest entry. Writer-declared
  (``Writer.onnx_outputs``, amendment merge condition 2), assembled by
  ``WriterCallback.onnx_manifest`` — never config-parsed.
- `resolve_export_config` — export-half validation + defaulting (the ONLY
  place ``model_name`` rules apply, design §7 "Naming"; hard-errors on a
  config-declared ``export.outputs``).
- `attach_manifest` / `ordered_output_names` / `manifest_table` /
  `combine_insertion_index` — manifest resolution, the single output
  ordering authority (combines insert before per-token aux entries, the v1
  rule), and the human-readable rendering.
- `compile_onnx_plan` / `export_graph` — the programmatic export core
  (fixture/gate-friendly: bound modules + manifest in, checked ``.onnx``
  out).
- `OnnxAdapter` — the traceable wrapper handed to ``torch.onnx.export``.
- `check_onnx` — the torch-vs-onnxruntime sweep checker (design §7.6).
- ``salt2 export`` — the CLI (`salt.core.onnx.export.main`), dispatched
  from `salt.core.main`; ``--manifest`` prints the assembled manifest.
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
