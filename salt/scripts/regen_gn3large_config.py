"""Regenerate/verify the public GN3Large bundle's stale ``config_v2.yaml``.
Rewrites dead ``salt.core.*`` class paths, folds ``export:`` into
``outputs.onnx_export``, drops ``callbacks:``, and relativizes ``norm_dict``.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

#: Env var naming the stale bundle config on this machine, for `--check`.
STALE_SOURCE_ENV = "SALT_GN3LARGE_STALE_CONFIG"

#: Fallback stale-source path when neither `--source` nor `STALE_SOURCE_ENV` is set.
DEFAULT_STALE_SOURCE = Path(
    "/data/ccra-data/projects/salt-improvements/studies/2026_06_11_modularise-salt"
    "/experiments/23_gn3large_v1_to_v2_conversion/outputs/converted/config_v2.yaml"
)

#: The 20 dead `salt.core.*` class paths in the stale bundle config, mapped to
#: their current-core replacement (each re-exported from its package `__init__.py`).
CLASS_PATH_MAP: dict[str, str] = {
    "salt.core.data.InputSamples": "salt.data.InputSamples",
    "salt.core.data.H5StructuredReader": "salt.data.H5StructuredReader",
    "salt.core.data.Features": "salt.data.Features",
    "salt.core.data.Labels": "salt.data.Labels",
    "salt.core.SaltModule": "salt.model.SaltModule",
    "salt.core.nn.Normaliser": "salt.model.modules.Normaliser",
    "salt.core.nn.StreamEmbed": "salt.model.modules.StreamEmbed",
    "salt.core.nn.Concat": "salt.model.modules.Concat",
    "salt.core.nn.TransformerEncoder": "salt.model.modules.TransformerEncoder",
    "salt.core.nn.Split": "salt.model.modules.Split",
    "salt.core.nn.GlobalAttentionPooling": "salt.model.modules.GlobalAttentionPooling",
    "salt.core.nn.LossSum": "salt.model.modules.LossSum",
    "salt.core.nn.tasks.ClassificationTaskModule": (
        "salt.model.modules.tasks.ClassificationTaskModule"
    ),
    "salt.core.nn.tasks.VertexingTaskModule": "salt.model.modules.tasks.VertexingTaskModule",
    "salt.core.nn.tasks.RegressionTaskModule": "salt.model.modules.tasks.RegressionTaskModule",
    "salt.core.outputs.InputCopyWriter": "salt.outputs.InputCopyWriter",
    "salt.core.outputs.RunTaskOutput": "salt.outputs.RunTaskOutput",
    "salt.core.outputs.PadMaskWriter": "salt.outputs.PadMaskWriter",
    "salt.core.outputs.H5OutputSink": "salt.outputs.H5OutputSink",
    "salt.core.outputs.OnnxExportSink": "salt.outputs.OnnxExportSink",
}

#: The exact 5 lines this script deletes: sinks are not Lightning `Callback`s,
#: and the h5 sink is wired implicitly by `salt test` with no replacement.
_CALLBACKS_BLOCK = [
    "callbacks:\n",
    "  h5_output:\n",
    "    class_path: salt.outputs.H5OutputSink\n",
    "  onnx_export:\n",
    "    class_path: salt.outputs.OnnxExportSink\n",
]

_HEADER_BODY = """\
# Mirror of the public GN3Large checkpoint bundle's ``config_v2.yaml``
# (CERNBox https://cernbox.cern.ch/s/39Qui940MGt4jyM).
#
# DO NOT EDIT BY HAND -- regenerate with:
#   python -m salt.scripts.regen_gn3large_config \\
#       --source <bundle>/config_v2.yaml \\
#       --output docs/tutorials/configs/finetuning/gn3large_base.yaml
#
# Source bundle sha256 (as shipped on CERNBox, before this rewrite):
#   config_v2.yaml:  {source_sha256}
#   converted.ckpt:  {ckpt_sha256}
#   norm_dict_v2.yaml: {norm_dict_sha256}
#
# Rewrites applied to the stale bundle file (see regen_gn3large_config.py):
#   1. 20 dead class paths under the removed `salt.core` package rewritten
#      to their current-core equivalent (CLASS_PATH_MAP, see dc97af9).
#   2. The top-level `export:` alias (retired by 9c7e807) moved under
#      `outputs:` as a new `onnx_export` entry, appended after `pad_mask`.
#   3. The `callbacks:` block deleted -- output sinks are not Lightning
#      Callbacks; the h5 sink is wired implicitly by `salt test`.
#   4. `norm_dict` rewritten from a machine-local absolute path to the
#      relative sibling `norm_dict_v2.yaml`.
#
# The bundle's own `config_v2.yaml` (the file re-uploaded to CERNBox/EOS) is
# this file minus this header -- everything from `name: GN4_big` onward.
#
# body-sha256: {body_sha256}
#
# `norm_dict` above is a relative path: run `salt fit`/`salt test` from the
# bundle directory, or override it explicitly, e.g.
#   --model.init_args.modules.norm.init_args.norm_dict=$GN3LARGE/norm_dict_v2.yaml
#
# `trainer.precision: 32-true` is kept verbatim, matching the original
# training run. The shipped fine-tuning overlays set `16-mixed` instead:
# GN3Large's encoder uses `attn_type: flash-varlen`, which only runs in
# fp16/bf16 on GPU.
#
# A config whose class paths still start with `salt.core` is the pre-2026-09
# bundle file and must be replaced with this one (or a fresh
# `--check`-passing regeneration).
"""


def _emit(line: str) -> None:
    """Write one line to stdout (T201 forbids `print`; this module skips `salt`)."""
    sys.stdout.write(line + "\n")


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex sha256 digest of the file at `path`."""
    return sha256_bytes(path.read_bytes())


def _rewrite_class_paths(lines: list[str]) -> list[str]:
    """Rewrite every mapped `class_path:` line; exit naming the line if a
    `salt.core.` value has no entry in `CLASS_PATH_MAP`.
    """
    out: list[str] = []
    for line in lines:
        if "class_path:" in line:
            indent, _, value = line.partition("class_path:")
            stripped = value.strip()
            mapped = CLASS_PATH_MAP.get(stripped)
            if mapped is not None:
                out.append(f"{indent}class_path: {mapped}\n")
                continue
            if "salt.core." in stripped:
                sys.exit(f"regen_gn3large_config: unmapped salt.core class path: {line!r}")
        out.append(line)
    return out


def _restructure_outputs(lines: list[str]) -> list[str]:
    """Delete `callbacks:`/`export:`, fold the export body into a new
    `outputs.onnx_export` entry. Each anchor must appear exactly once, in
    order, and the deleted blocks must match exactly, else exit non-zero.
    """

    def _find(marker: str) -> int:
        hits = [i for i, line in enumerate(lines) if line == marker]
        if len(hits) != 1:
            sys.exit(
                f"regen_gn3large_config: expected exactly one {marker!r} line, found {len(hits)}"
            )
        return hits[0]

    i_callbacks = _find("callbacks:\n")
    i_export = _find("export:\n")
    i_trainer = _find("trainer:\n")
    if not (i_callbacks < i_export < i_trainer):
        sys.exit(
            "regen_gn3large_config: expected callbacks: before export: before trainer:, "
            f"found at lines {i_callbacks}, {i_export}, {i_trainer}"
        )

    callbacks_block = lines[i_callbacks:i_export]
    if callbacks_block != _CALLBACKS_BLOCK:
        sys.exit(
            "regen_gn3large_config: the callbacks: block does not match the expected "
            f"5-line stale form; got:\n{''.join(callbacks_block)}"
        )

    export_body = lines[i_export + 1 : i_trainer]
    if not export_body:
        sys.exit("regen_gn3large_config: export: block is empty")

    onnx_export_block = [
        "  onnx_export:\n",
        "    class_path: salt.outputs.OnnxExportSink\n",
        "    init_args:\n",
        *(f"    {body_line}" for body_line in export_body),
    ]
    return lines[:i_callbacks] + onnx_export_block + lines[i_trainer:]


def _fix_norm_dict(lines: list[str]) -> list[str]:
    """Rewrite the one absolute `norm_dict:` path to its relative basename;
    exit non-zero if zero or more than one such line is found.
    """
    hits = [
        i
        for i, line in enumerate(lines)
        if line.lstrip().startswith("norm_dict:") and "/" in line.split("norm_dict:", 1)[1]
    ]
    if len(hits) != 1:
        sys.exit(
            "regen_gn3large_config: expected exactly one absolute `norm_dict:` line, "
            f"found {len(hits)}"
        )
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    value = lines[i].split("norm_dict:", 1)[1].strip()
    lines[i] = f"{indent}norm_dict: {Path(value).name}\n"
    return lines


def regenerate_body(source_text: str) -> str:
    """Apply the four textual rewrites to `source_text`, returning the BODY
    (the fixed config with no header -- what the bundle re-upload ships).
    """
    lines = source_text.splitlines(keepends=True)
    lines = _rewrite_class_paths(lines)
    lines = _restructure_outputs(lines)
    lines = _fix_norm_dict(lines)
    body = "".join(lines)
    if "salt.core." in body:
        sys.exit("regen_gn3large_config: salt.core. survives after rewriting (internal bug)")
    return body


def build_header(
    source_sha256: str, ckpt_sha256: str, norm_dict_sha256: str, body_sha256: str
) -> str:
    """Render the provenance header prepended to the committed repo mirror."""
    return _HEADER_BODY.format(
        source_sha256=source_sha256,
        ckpt_sha256=ckpt_sha256,
        norm_dict_sha256=norm_dict_sha256,
        body_sha256=body_sha256,
    )


def _split_header(text: str) -> tuple[str, str]:
    """Split into `(header_text, body_text)` at the first non-`#` line, which
    must be `name: GN4_big`; exit non-zero if there is no header there.
    """
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    if i == 0:
        sys.exit("regen_gn3large_config: no header (leading `#` lines) found")
    if i >= len(lines) or not lines[i].startswith("name: GN4_big"):
        sys.exit("regen_gn3large_config: header must be followed immediately by `name: GN4_big`")
    return "".join(lines[:i]), "".join(lines[i:])


def write_bundle_dir(bundle_dir: Path, body: str, diff_text: str, source_path: Path) -> None:
    """Write the re-uploadable bundle dir: header-less config, diff, sibling
    ckpt/norm_dict copies (skipped when already present at the same size),
    and a `SHA256SUMS` manifest.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    config_path = bundle_dir / "config_v2.yaml"
    config_path.write_text(body)
    (bundle_dir / "config_v2.diff").write_text(diff_text)

    source_dir = source_path.parent
    for name in ("converted.ckpt", "norm_dict_v2.yaml"):
        src = source_dir / name
        dst = bundle_dir / name
        if not src.is_file():
            continue
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)

    sums_lines = [f"{sha256_file(config_path)}  config_v2.yaml\n"]
    sums_lines.extend(
        f"{sha256_file(bundle_dir / name)}  {name}\n"
        for name in ("converted.ckpt", "norm_dict_v2.yaml")
        if (bundle_dir / name).is_file()
    )
    if source_path.is_file():
        sums_lines.append(f"{sha256_file(source_path)}  config_v2.yaml.stale\n")
    (bundle_dir / "SHA256SUMS").write_text("".join(sums_lines))


def _resolve_check_source(explicit: Path | None) -> Path | None:
    """Resolve the stale source for `--check`: `explicit`, else
    `STALE_SOURCE_ENV`, else `DEFAULT_STALE_SOURCE`; `None` if none readable.
    """
    candidate = explicit
    if candidate is None and (env_value := os.environ.get(STALE_SOURCE_ENV)):
        candidate = Path(env_value)
    if candidate is None:
        candidate = DEFAULT_STALE_SOURCE
    if candidate.is_file() and os.access(candidate, os.R_OK):
        return candidate
    return None


def check_output(repo_path: Path, bundle_dir: Path | None, source_override: Path | None) -> None:
    """Verify `repo_path` is exactly what this script would (re)generate.

    Self-contained checks always run; the source-equality check (regenerating
    from the real stale bundle and comparing byte-for-byte) runs only when a
    readable source is found, else prints a `NOTE` and returns. Exits
    non-zero on any failure.
    """
    import yaml  # local: this module must stay importable without PyYAML unless --check runs

    text = repo_path.read_text()
    header_text, body_text = _split_header(text)

    if "salt.core." in body_text:
        sys.exit(f"regen_gn3large_config --check: salt.core. still present in {repo_path}")
    if "/data/" in body_text:
        sys.exit(f"regen_gn3large_config --check: a /data/ literal still present in {repo_path}")

    parsed = yaml.safe_load(body_text)
    if "export" in parsed or "callbacks" in parsed:
        sys.exit(
            "regen_gn3large_config --check: top-level `export:`/`callbacks:` key still present"
        )

    norm_dict_value = parsed["model"]["init_args"]["modules"]["norm"]["init_args"]["norm_dict"]
    if norm_dict_value != "norm_dict_v2.yaml":
        sys.exit(
            "regen_gn3large_config --check: norm_dict is "
            f"{norm_dict_value!r}, expected 'norm_dict_v2.yaml'"
        )

    onnx_export = parsed.get("outputs", {}).get("onnx_export", {})
    if onnx_export.get("class_path") != "salt.outputs.OnnxExportSink":
        sys.exit("regen_gn3large_config --check: outputs.onnx_export.class_path is wrong")
    onnx_init_args = onnx_export.get("init_args", {})
    if onnx_init_args.get("model_name") != "GN4big":
        sys.exit("regen_gn3large_config --check: outputs.onnx_export.init_args.model_name is wrong")
    if len(onnx_init_args.get("inputs", [])) != 4:
        sys.exit("regen_gn3large_config --check: outputs.onnx_export.init_args.inputs is not len 4")

    expected_body_sha = sha256_bytes(body_text.encode())
    header_sha_lines = [
        stripped_line
        for line in header_text.splitlines()
        if (stripped_line := line.removeprefix("# body-sha256:").strip()) != line.strip()
    ]
    if len(header_sha_lines) != 1:
        sys.exit("regen_gn3large_config --check: header is missing a `# body-sha256:` line")
    if header_sha_lines[0] != expected_body_sha:
        sys.exit(
            "regen_gn3large_config --check: header body-sha256 "
            f"{header_sha_lines[0]} != actual {expected_body_sha}"
        )

    source_path = _resolve_check_source(source_override)
    if source_path is None:
        _emit("NOTE: stale source not visible; source-equality check skipped")
        return

    source_text = source_path.read_text()
    regenerated = regenerate_body(source_text)
    if regenerated != body_text:
        diff = "".join(
            difflib.unified_diff(
                regenerated.splitlines(keepends=True),
                body_text.splitlines(keepends=True),
                fromfile="regenerated-from-source",
                tofile=str(repo_path),
            )
        )
        sys.exit(
            f"regen_gn3large_config --check: {repo_path} does not match a fresh regeneration "
            f"of {source_path}:\n{diff}"
        )

    if bundle_dir is not None:
        diff_text = "".join(
            difflib.unified_diff(
                source_text.splitlines(keepends=True),
                regenerated.splitlines(keepends=True),
                fromfile=str(source_path),
                tofile="config_v2.yaml (regenerated)",
            )
        )
        write_bundle_dir(bundle_dir, regenerated, diff_text, source_path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: regenerate by default, or verify with `--check`.

    Parameters
    ----------
    argv : Sequence[str] | None
        Argument vector, or ``None`` to use ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--source", type=Path, default=None, help="stale bundle config_v2.yaml")
    parser.add_argument("--output", type=Path, default=None, help="repo mirror path to write")
    parser.add_argument(
        "--check", type=Path, default=None, metavar="REPO_YAML", help="verify this file instead"
    )
    parser.add_argument(
        "--bundle-dir", type=Path, default=None, help="also write a re-uploadable bundle dir"
    )
    args = parser.parse_args(argv)

    if args.check is not None:
        check_output(args.check, bundle_dir=args.bundle_dir, source_override=args.source)
        _emit(f"OK: {args.check} matches a fresh regeneration")
        return 0

    if args.source is None or args.output is None:
        parser.error("generate mode requires --source and --output")

    source_text = args.source.read_text()
    body = regenerate_body(source_text)
    source_sha256 = sha256_bytes(source_text.encode())
    body_sha256 = sha256_bytes(body.encode())

    ckpt_path = args.source.parent / "converted.ckpt"
    norm_dict_path = args.source.parent / "norm_dict_v2.yaml"
    ckpt_sha256 = sha256_file(ckpt_path) if ckpt_path.is_file() else "unavailable"
    norm_dict_sha256 = sha256_file(norm_dict_path) if norm_dict_path.is_file() else "unavailable"

    header = build_header(source_sha256, ckpt_sha256, norm_dict_sha256, body_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + body)

    diff_text = "".join(
        difflib.unified_diff(
            source_text.splitlines(keepends=True),
            body.splitlines(keepends=True),
            fromfile=str(args.source),
            tofile="config_v2.yaml (regenerated)",
        )
    )

    _emit(f"source sha256: {source_sha256}")
    _emit(f"body sha256:   {body_sha256}")
    _emit(diff_text or "(no diff)")

    if args.bundle_dir is not None:
        write_bundle_dir(args.bundle_dir, body, diff_text, args.source)

    return 0


if __name__ == "__main__":
    sys.exit(main())
