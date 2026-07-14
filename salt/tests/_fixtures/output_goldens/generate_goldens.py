#!/usr/bin/env python3
"""Plan 50 Phase A — golden H5/ONNX output-schema capture.

For every shipped config that declares an ``outputs:`` section (or wires an
explicit-sink table, the 4 ``gn2v2-dummy*`` configs), statically compiles the
TEST-mode and ONNX-mode plans through the real ``salt2`` surface (run-free,
no data touched) and dumps the literal H5 column table + ONNX output tuple to
one JSON file. These literals become the Phase E re-anchoring oracle — do
NOT regenerate against a later HEAD without updating the provenance sha.

Usage (inside the salt-py314 container, from the worktree root)::

    python salt/tests/_fixtures/output_goldens/generate_goldens.py [--only NAME ...]

Writes ``salt/tests/_fixtures/output_goldens/<config>.json`` (one per entry
in ``CONFIGS``) plus a ``_summary.json`` index. Exits non-zero if any config
failed to capture.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HEAD_SHA = "d848b61+phaseC"
"""Worktree state this golden set is anchored to (feature/one-class-per-file):
the plan-50 Phase-C commit on top of d848b61.

vs a regen at the Phase-B tip (d848b61) the diff is STRICTLY the per-task
TEST-mode target-label additions (``target_{task}`` / ``target_{task}_{target}``
column + TEST-manifest entries, plus TEST plan hashes where labels are newly
demanded); every ``onnx`` block and ONNX plan hash is byte-identical (verified
programmatically 2026-07-14, all 30 capturable configs; GN2_muP exempt — mup
not in salt-py314.sif).

vs the committed Phase-A capture (3031dfb @ 96d88d8) the diff is NOT
labels-only: Phase B (95d2934) rewrote the four gn2v2-dummy* configs (explicit
sinks -> implicit dumb-section) WITHOUT regenerating goldens, so this regen
also folds in that Phase-B re-anchoring — gn2v2-dummy{,-cutover,-onnx-fold}
flip ``is_dumb_section`` false->true and re-key grouped OutputColumns to
per-leaf entries (2->13 columns), and gn2v2-dummy's ONNX ``plan_hash`` changed
(63a88656... -> 2e4171e3...) with the exported ONNX output tuple
(names/dtypes/dynamic_axes) and flat H5 column names unchanged. To audit the
Phase-C label delta in isolation, regen at d848b61 (``git archive d848b61 |
tar -x`` into a scratch dir, run this script there with PYTHONPATH pointing at
the extraction) and diff those files against these."""

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../worktrees/one-class-per-file/salt
CONFIG_DIR = REPO_ROOT / "core" / "configs"
OUT_DIR = Path(__file__).resolve().parent


@dataclass
class ConfigSpec:
    """One golden-capture entry: the output name + the config stack to load."""

    name: str
    stack: list[str]
    note: str = ""


# -- config inventory -------------------------------------------------------
# Standalone configs: own `outputs:` section (or, for the gn2v2-dummy base,
# an explicit-sink table) + own `data:` block — compile as a single -c.
_STANDALONE = [
    "dips",
    "Dipz",
    "DL1",
    "easyjet_flavour",
    "easyjet_hh4b_ttbar",
    "event_classifier",
    "ftag1lite_empflow",
    "GN2emu",
    "GN2_muP",
    "gn2v2-opendata",
    "GN2XE",
    "GN2X_qcdsplit",
    "GN3_baseline",
    "GN3EPCLV01",
    "GN3_v00",
    "GN3V00",
    "gn3v01",
    "GN3X",
    "hitz",
    "MaskFormer",
    "nan_regression",
    "regression_gaussian",
    "regression_multi_target",
    "regression_weighted",
    "regression",
    "gn2v2-dummy",
]

# Overlay configs: no own `data:` block (or a list-replace outputs: override)
# — must be stacked on their base per the header comment's documented order.
_STACKED = [
    ConfigSpec(
        "gn2v2-dummy-cutover",
        ["gn2v2-dummy.yaml", "gn2v2-dummy-cutover.yaml"],
        "explicit-sink table (H5OutputWriter alias), two-layer cutover demo",
    ),
    ConfigSpec(
        "gn2v2-dummy-cutover34",
        ["gn2v2-dummy.yaml", "gn2v2-dummy-cutover34.yaml"],
        "dumb-section cutover demo (outputs: null base tables)",
    ),
    ConfigSpec(
        "gn2v2-dummy-onnx-fold",
        ["gn2v2-dummy.yaml", "gn2v2-dummy-onnx-fold.yaml"],
        "ONNX-fold demo; H5 side inherited unchanged from gn2v2-dummy.yaml",
    ),
    ConfigSpec(
        "GN3_Charge",
        ["GN3V00.yaml", "GN3_Charge.yaml"],
        "overlay on GN3V00 per header comment (run_tasks list replace)",
    ),
    ConfigSpec(
        "GN3_tracklabel",
        [
            "GN3_baseline.yaml",
            "GN3_baseline_loose.yaml",
            "GN3_flow.yaml",
            "GN3_LepID_SMT.yaml",
            "GN3_tracklabel.yaml",
        ],
        "5-file overlay chain per header comment",
    ),
]

CONFIGS: list[ConfigSpec] = [ConfigSpec(n, [f"{n}.yaml"]) for n in _STANDALONE] + _STACKED


# -- deep-merge (mirrors the jsonargparse cross-config semantics documented
# in the shipped configs: dict sections merge key-by-key, a null entry
# deletes, list-typed fields REPLACE) — used ONLY to statically detect which
# Normaliser modules need a data-free norm_dict override before parsing.
def _deep_merge(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for k, v in override.items():
        if v is None and k in out:
            del out[k]
        elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    import yaml  # noqa: PLC0415 - only needed for the static pre-scan

    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return raw if isinstance(raw, dict) else {}


def _norm_dict_overrides(stack_paths: list[Path]) -> list[str]:
    """``--set model.modules.<name>.init_args.norm_dict=unused.yaml`` for every
    Normaliser module in the merged stack that ships no norm_dict (data-free
    parse; the established override pattern, see log.md 2026-07-14 08:20).
    """
    merged: dict = {}
    for path in stack_paths:
        merged = _deep_merge(merged, _load_yaml(path))
    modules = (
        (merged.get("model") or {}).get("init_args", {}).get("modules")
        if isinstance(merged.get("model"), dict)
        else None
    ) or {}
    overrides = []
    for name, node in modules.items():
        if not isinstance(node, dict):
            continue
        if "Normaliser" not in str(node.get("class_path", "")):
            continue
        init_args = node.get("init_args") or {}
        if "norm_dict" not in init_args:
            overrides.append(f"model.modules.{name}.init_args.norm_dict=unused.yaml")
    return overrides


# -- golden extraction --------------------------------------------------------


def _field_dict(field_obj: Any) -> dict[str, Any]:
    return {
        "h5_name": field_obj.h5_name,
        "onnx_name": field_obj.onnx_name,
        "resolved_onnx_name": field_obj.resolved_onnx_name,
        "dtype": field_obj.dtype,
        "onnx_dtype": field_obj.onnx_dtype,
        "axis": field_obj.axis,
        "final": field_obj.final,
        "prefix": field_obj.prefix,
    }


def _task_manifest(sink: Any, mode: Any) -> list[dict[str, Any]]:
    """Task-level (leaf_key, OutputField) manifest for a bound outputs: section.

    Empty for an explicit-sink config (no RunTaskOutput section bound) — the
    resolved column/leaf tables below still carry the full schema in that case.
    """
    from salt.core.outputs.run_task_output import RunTaskOutput  # noqa: PLC0415

    out: list[dict[str, Any]] = []
    section = getattr(sink, "_output_section", None) or {}
    for name, writer in section.items():
        if not isinstance(writer, RunTaskOutput):
            continue
        for leaf_key, field_obj in writer.manifest_fields(mode):
            out.append({"section_writer": name, "leaf_key": leaf_key, **_field_dict(field_obj)})
    return out


def _capture_one(spec: ConfigSpec) -> dict[str, Any]:
    from salt.core.cli import (  # noqa: PLC0415
        _as_sink_node,
        _static_export_model_name,
        _static_onnx_export_sink,
        _static_writer_sink_callback,
    )
    from salt.core.graph.errors import ConfigError  # noqa: PLC0415
    from salt.core.graph.spec import Mode  # noqa: PLC0415
    from salt.core.onnx.export import _run_free_cli  # noqa: PLC0415
    from salt.core.outputs import H5OutputSink  # noqa: PLC0415

    stack_paths = [CONFIG_DIR / c for c in spec.stack]
    missing = [str(p) for p in stack_paths if not p.is_file()]
    if missing:
        return {"captured": False, "error": f"missing config file(s): {missing}"}

    overrides = _norm_dict_overrides(stack_paths)
    provenance: dict[str, Any] = {
        "head_sha": HEAD_SHA,
        "generator": "salt/tests/_fixtures/output_goldens/generate_goldens.py",
        "config_name": spec.name,
        "config_stack": spec.stack,
        "note": spec.note,
        "set_overrides": overrides,
    }
    try:
        cli = _run_free_cli(stack_paths, overrides)
    except ConfigError as err:
        return {"captured": False, "error": str(err), "provenance": provenance}

    run_name = cli._get(cli.config_init, "name") or "salt"  # noqa: SLF001
    provenance["run_name"] = run_name

    h5_sink = _as_sink_node(_static_writer_sink_callback(cli))
    onnx_sink = _static_onnx_export_sink(cli)

    result: dict[str, Any] = {"captured": True, "provenance": provenance, "h5": None, "onnx": None}

    if h5_sink is not None and isinstance(h5_sink, H5OutputSink):
        try:
            columns = h5_sink._resolve_columns(run_name)  # noqa: SLF001
            result["h5"] = {
                "is_dumb_section": h5_sink._is_dumb_section(),  # noqa: SLF001
                "columns": [
                    {
                        "key": col.key,
                        "stream": col.stream,
                        "suffixes": list(col.suffixes),
                        "column_names": col.column_names(run_name),
                        "dtype": col.dtype,
                        "prefix": col.prefix,
                    }
                    for col in columns
                ],
                "task_manifest_test": _task_manifest(h5_sink, Mode.TEST),
                "copy_inputs": h5_sink.copy_inputs,
                "write_pad_mask": h5_sink.write_pad_mask
                if isinstance(h5_sink.write_pad_mask, (bool, list, tuple))
                else list(h5_sink.write_pad_mask),
                "extra_group_names": list(h5_sink._extra_group_names),  # noqa: SLF001
            }
            if h5_sink._extra_group_names:  # noqa: SLF001
                # extra_groups (e.g. MaskFormerObjectsSink's objects/object_masks)
                # size their columns from a live reader at open_schema (seq
                # lengths / shapes) — NOT resolvable statically/data-free. The
                # `columns` list above therefore covers the RunTaskOutput task
                # columns only; the extra-group schema is a documented gap.
                result["h5"]["static_capture_gap"] = (
                    "extra_groups "
                    f"{list(h5_sink._extra_group_names)} columns NOT captured — "  # noqa: SLF001
                    "their schema is resolved from a live reader at open_schema "
                    "(sequence lengths / shapes), not data-free"
                )
        except ConfigError as err:
            result["h5"] = {"error": str(err)}

    if onnx_sink is not None:
        try:
            if onnx_sink.model_name is None:
                export_cfg = cli._get(cli.config_init, "export")  # noqa: SLF001
                onnx_sink.model_name = _static_export_model_name(export_cfg, run_name)
            result["onnx"] = {
                "is_dumb_section": onnx_sink._is_dumb_section(),  # noqa: SLF001
                "model_name": onnx_sink.model_name,
                "output_names": onnx_sink.output_names(),
                "output_dtypes": onnx_sink.output_dtypes(),
                "dynamic_axes": onnx_sink.dynamic_axes(),
                "task_manifest_onnx": _task_manifest(onnx_sink, Mode.ONNX),
            }
        except ConfigError as err:
            result["onnx"] = {"error": str(err)}

    # static plan-compile check (Mode.TEST / Mode.ONNX) through the real
    # planner — "the same machinery as salt2 graph validate" per plan 50 §A.
    try:
        from salt.core.cli import load_config  # noqa: PLC0415
        from salt.core.graph.planner import compile_plan  # noqa: PLC0415

        gcfg = load_config([str(p) for p in stack_paths], overrides)
        compile_info: dict[str, Any] = {}
        for mode in (Mode.TEST, Mode.ONNX):
            if mode in gcfg.mode_errors:
                compile_info[mode.name] = {"error": gcfg.mode_errors[mode]}
                continue
            try:
                plan = compile_plan(
                    gcfg.modules, mode, gcfg.sources, gcfg.schema, gcfg.sinks.get(mode)
                )
                compile_info[mode.name] = {
                    "plan_hash": plan.plan_hash,
                    "n_steps": len(plan.steps),
                }
            except Exception as err:  # noqa: BLE001 - report, don't abort golden capture
                compile_info[mode.name] = {"error": f"{type(err).__name__}: {err}"}
        result["plan_compile"] = compile_info
    except Exception as err:  # noqa: BLE001 - best-effort secondary validation
        result["plan_compile"] = {"error": f"{type(err).__name__}: {err}"}

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="capture only these config names")
    args = parser.parse_args(argv)

    generator_command = "apptainer exec --bind /home/npond/Documents/CCRA --bind /tmp " \
        "$CCRA_CONTAINERS_DIR/salt-py314.sif python " \
        "salt/tests/_fixtures/output_goldens/generate_goldens.py"

    targets = [c for c in CONFIGS if args.only is None or c.name in args.only]
    summary: dict[str, Any] = {
        "head_sha": HEAD_SHA,
        "generator_command": generator_command,
        "n_configs": len(targets),
        "results": {},
    }
    n_failed = 0
    for spec in targets:
        print(f"capturing {spec.name} ...", file=sys.stderr)
        try:
            golden = _capture_one(spec)
        except Exception as err:  # noqa: BLE001 - never let one config kill the run
            golden = {"captured": False, "error": f"UNEXPECTED {type(err).__name__}: {err}"}
        golden.setdefault("provenance", {}).setdefault("generator_command", generator_command)
        out_path = OUT_DIR / f"{spec.name}.json"
        with open(out_path, "w") as fh:
            json.dump(golden, fh, separators=(",", ":"), sort_keys=True); fh.write("\n")
            fh.write("\n")
        ok = golden.get("captured", False)
        summary["results"][spec.name] = {
            "captured": ok,
            "error": golden.get("error"),
            "h5_columns": len((golden.get("h5") or {}).get("columns", []) or [])
            if golden.get("h5")
            else 0,
            "onnx_outputs": len((golden.get("onnx") or {}).get("output_names", []) or [])
            if golden.get("onnx")
            else 0,
        }
        if not ok:
            n_failed += 1
            print(f"  FAILED: {golden.get('error')}", file=sys.stderr)
        else:
            print(
                f"  ok (h5_columns={summary['results'][spec.name]['h5_columns']}, "
                f"onnx_outputs={summary['results'][spec.name]['onnx_outputs']})",
                file=sys.stderr,
            )

    with open(OUT_DIR / "_summary.json", "w") as fh:
        json.dump(summary, fh, separators=(",", ":"), sort_keys=True); fh.write("\n")
        fh.write("\n")

    print(f"\n{len(targets) - n_failed}/{len(targets)} configs captured", file=sys.stderr)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
