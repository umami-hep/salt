"""salt.core.utils — core-local copies of the genuinely-shared v1 utilities.

M7 W2a relocations: the v1 ``salt.utils`` helpers that production ``salt.core``
code depends on are COPIED here BYTE-FAITHFULLY (identical logic) so the core
package no longer imports the v1 tree. The v1 originals stay in place as the
RS1 gate oracle (the gates still import v1); production imports the copy here.

Modules:

- ``array_utils`` — ``maybe_copy`` / ``listify`` / ``join_structured_arrays``
  (+ ``maybe_pad``, carried for a byte-faithful file copy).
- ``scalers`` — ``RegressionTargetScaler`` (functional regression-target scaler).
- ``union_find`` — ``get_node_assignment_jit`` (the ``@torch.jit.script`` ONNX
  union-find path) + its helpers.
- ``file_utils`` — temp-file / S3 staging helpers (the opt-in staging path).
"""
