"""salt.core.utils — core-local copies of the genuinely-shared v1 utilities.

Byte-faithful copies of the v1 ``salt.utils`` helpers that production
``salt.core`` code depends on, so the core package no longer imports the v1
tree. Modules: ``array_utils`` (array helpers), ``scalers``
(``RegressionTargetScaler``), ``union_find`` (the ONNX union-find path),
``file_utils`` (temp-file / S3 staging), ``mask_utils`` (MaskFormer mask/index
helpers + reconstruction metrics), ``tensor_utils`` (tensor-shape helpers).
"""
