"""``gnn_config`` ONNX metadata, bit-compatible with v1 on equivalent config
(v1 key order preserved; one additive ``plan_hash`` key appended).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from salt.core.onnx.config import ExportConfig, stream_of_input_port

__all__ = ["ONNX_MODEL_VERSION", "build_gnn_config", "load_run_metadata", "write_metadata"]

ONNX_MODEL_VERSION = "v1"
"""Athena metadata version — the default export is v1-content-identical."""


def build_gnn_config(
    export: ExportConfig,
    variables: Mapping[str, Sequence[str]],
    output_names: Sequence[str],
    config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    ckpt_path: str | Path | None,
    plan_hash: str | None,
) -> dict[str, Any]:
    """Build the ``gnn_config`` payload in the v1 key set/order.

    Parameters
    ----------
    export : ExportConfig
        The RESOLVED export config.
    variables : Mapping[str, Sequence[str]]
        Per-stream `Features` variable lists (the metadata variable names).
    output_names : Sequence[str]
        The exporter's flat output-name list.
    config : Mapping[str, Any]
        The resolved run config embedded under ``config.yaml``.
    run_metadata : Mapping[str, Any]
        The run-dir ``metadata.yaml`` content (``{}`` when the run has none —
        documented fallback).
    ckpt_path : str | Path | None
        The exported checkpoint (``""`` for checkpoint-free fixture
        exports — the key is always present).
    plan_hash : str | None
        The ONNX plan hash, recorded under the ADDITIVE trailing
        ``plan_hash`` key.

    Returns
    -------
    dict[str, Any]
        The payload, insertion-ordered exactly as v1 (+ trailing
        ``plan_hash``).
    """
    metadata: dict[str, Any] = {
        "ckpt_path": str(Path(ckpt_path).resolve()) if ckpt_path else "",
        "layers": [],
        "nodes": [],
    }
    metadata["config.yaml"] = dict(config)
    metadata["metadata.yaml"] = dict(run_metadata)
    metadata["salt_export_hash"] = _export_hash()
    metadata["onnx_model_version"] = ONNX_MODEL_VERSION
    metadata["output_names"] = list(output_names)
    metadata["model_name"] = export.model_name
    metadata["inputs"] = []
    metadata["input_sequences"] = []
    for entry in export.inputs:
        if entry.alias is not None:
            continue  # alias pseudo-inputs have no Athena tensor
        stream = stream_of_input_port(entry.port)
        if entry.sequence:
            metadata["input_sequences"].append({
                "name": entry.athena_name,
                "variables": [
                    {"name": name, "offset": 0.0, "scale": 1.0} for name in variables[stream]
                ],
            })
        else:
            # offsets/scales are informational placeholders (normalisation lives
            # inside the graph); '_btagJes' is stripped on GLOBAL variables only
            metadata["inputs"].append({
                "name": entry.athena_name,
                "variables": [
                    {"name": name.removesuffix("_btagJes"), "offset": 0.0, "scale": 1.0}
                    for name in variables[stream]
                ],
            })
    # combines are recorded as (name, [(scale, suffix), ...]) tuples — JSON-encoded
    # to nested lists — and renames as the raw old->new dict
    metadata["combine_outputs"] = [
        [entry.name, [[scale, suffix] for suffix, scale in entry.inputs.items()]]
        for entry in export.combine
    ]
    metadata["rename_outputs"] = dict(export.rename)
    # additive key, appended after the complete v1 set
    metadata["plan_hash"] = plan_hash
    return metadata


def write_metadata(onnx_path: str | Path, gnn_config: Mapping[str, Any], model_name: str) -> None:
    """Validate the graph and store ``gnn_config`` + the doc string.

    Loads the model, runs ``onnx.checker.check_model``, JSON-encodes the
    payload under the single ``gnn_config`` metadata key, sets
    ``doc_string = model_name``, and saves in place.
    """
    import onnx  # noqa: PLC0415 - heavy import, export-path only (keeps salt2 startup lean)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    meta = onnx_model.metadata_props.add()
    meta.key = "gnn_config"
    meta.value = json.dumps(gnn_config)
    onnx_model.doc_string = model_name
    onnx.save(onnx_model, str(onnx_path))


def load_run_metadata(config_path: str | Path | None) -> dict[str, Any]:
    """Read the run-dir ``metadata.yaml`` next to the config, ``{}`` when absent.

    Checkpoint-free exports legitimately have none — the documented fallback is
    an empty mapping.
    """
    if config_path is None:
        return {}
    path = Path(config_path).parent / "metadata.yaml"
    if not path.is_file():
        return {}
    with open(path) as fh:
        loaded = yaml.safe_load(fh)
    return dict(loaded) if isinstance(loaded, dict) else {}


def _export_hash() -> str | None:
    """The salt git hash recorded as ``salt_export_hash``.

    Tolerates a missing/unreadable git context (e.g. a container binding
    only the worktree, whose ``.git`` file points outside the bind) with a
    warning instead of failing the export. Returns the hash, or None when
    git state is unavailable.
    """
    try:
        from ftag.git_check import get_git_hash  # noqa: PLC0415 - optional, env-dependent

        return get_git_hash(Path(__file__).parent)
    except Exception as err:  # noqa: BLE001 - any git/env failure degrades to None
        warnings.warn(f"could not determine salt_export_hash: {err}", stacklevel=2)
        return None
