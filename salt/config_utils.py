"""Config utilities for run-free parsing (tests, tooling, CLI subcommands)."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import yaml


def disable_logger_in_config(config_path: str) -> str:
    """Load a config, disable trainer.logger, and write to /tmp.

    For keyless environments (no COMET_API_KEY), the default-ON CometLogger
    in base.yaml fails during instantiate_classes with "Comet.ml requires
    an API key".

    The cache key covers the config's PATH **and its CONTENT**, so parallel
    runs still share one copy while an edited config always re-derives. Keying
    on the path alone silently serves a stale copy to every later run in the
    same ``TMPDIR`` — the config you edited is not the config that is parsed,
    and the failure surfaces far from its cause.
    """
    raw = Path(config_path).read_bytes()
    digest = hashlib.md5(config_path.encode() + b"\0" + raw, usedforsecurity=False)
    cache_key = digest.hexdigest()[:12]
    cached_path = Path(tempfile.gettempdir()) / f"salt_config_no_logger_{cache_key}.yaml"
    if cached_path.exists():
        return str(cached_path)

    cfg = yaml.safe_load(raw)
    if cfg is None:
        cfg = {}
    if "trainer" not in cfg:
        cfg["trainer"] = {}
    cfg["trainer"]["logger"] = False

    # Drop fit/test-subcommand-only load keys: a saved run config.yaml carries
    # `ckpt_path` and (lightning 2.6.5+) `weights_only` at the top level, which
    # the run-free top-level parser does not register and rejects with NSKeyError
    # ("Option 'weights_only' is not accepted"). The graph tooling never loads a
    # checkpoint, so these are irrelevant for run-free parsing.
    for fit_only_key in ("ckpt_path", "weights_only"):
        cfg.pop(fit_only_key, None)

    cached_path.parent.mkdir(exist_ok=True)
    with open(cached_path, "w") as f:
        # sort_keys=False: preserve the config's declared order — several tests
        # assert ordered structures (e.g. writers: inputs_copy -> tasks ->
        # pad_mask), which a default alphabetising dump would mangle.
        yaml.dump(cfg, f, sort_keys=False)
    return str(cached_path)


INCLUDE_KEY = "include"
"""Top-level key listing configs to merge underneath this one."""


class IncludeError(ValueError):
    """A config's ``include:`` could not be resolved."""


def _resolve_include(ref: str, source: Path, config_dir: Path) -> Path:
    """Locate one ``include:`` entry.

    Absolute paths are used as given. A relative path is tried against the
    including config's own directory first, then the shipped ``salt/configs``
    root, so a config can name a sibling without knowing where it sits.
    """
    candidate = Path(ref)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise IncludeError(f"{source}: include '{ref}' does not exist")
        return candidate

    tried = [source.parent / ref, config_dir / ref]
    for path in tried:
        if path.is_file():
            return path
    listed = "\n  ".join(str(p) for p in tried)
    raise IncludeError(f"{source}: include '{ref}' not found. Tried:\n  {listed}")


def _merge(base: Any, over: Any) -> Any:
    """``DeepMergeParser`` semantics: dicts union key-by-key, lists replace.

    ``None`` markers are KEPT, not treated as deletions here — deletion happens
    at assembly time, where ``SaltModule``/``GraphDataModule``/``SaltCLI``
    filter their module dicts. Resolving them earlier would let a later config
    resurrect an entry an earlier one deleted.
    """
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if key in out else value
    return out


def _expand(path: Path, config_dir: Path, stack: tuple[Path, ...]) -> tuple[dict, list[Path]]:
    """Recursively merge a config's includes underneath it.

    Returns the merged config plus every file that fed it, so a caller can key
    a cache on the whole chain's content rather than the entry point alone.
    """
    resolved = path.resolve()
    if resolved in stack:
        cycle = " -> ".join(p.name for p in (*stack, resolved))
        raise IncludeError(f"include cycle: {cycle}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise IncludeError(f"{path}: config is not a mapping")

    refs = raw.pop(INCLUDE_KEY, None) or []
    if isinstance(refs, str):
        refs = [refs]

    merged: dict = {}
    sources: list[Path] = []
    for ref in refs:
        target = _resolve_include(ref, path, config_dir)
        sub, sub_sources = _expand(target, config_dir, (*stack, resolved))
        merged = _merge(merged, sub)
        sources.extend(sub_sources)

    # the including config wins over everything it includes
    return _merge(merged, raw), [*sources, resolved]


def expand_includes(config_path: str, config_dir: Path | None = None) -> str:
    """Expand a config's ``include:`` chain, returning a path to the result.

    A config with no includes anywhere in its chain is returned unchanged, so
    this is free for the common case. Otherwise the merged config is written to
    ``$TMPDIR`` and that path returned — ``include:`` never reaches jsonargparse.

    The cache key covers the content of EVERY file in the chain, so editing an
    included config re-derives rather than serving a stale merge.
    """
    from salt.main import CONFIG_DIR  # local import: avoids a config_utils<->main cycle

    root = Path(config_path)
    config_dir = config_dir or CONFIG_DIR
    merged, sources = _expand(root, config_dir, ())
    if len(sources) == 1:
        return config_path  # nothing was included

    digest = hashlib.md5(usedforsecurity=False)
    for source in sources:
        digest.update(str(source).encode() + b"\0" + source.read_bytes())
    cached = Path(tempfile.gettempdir()) / f"salt_config_included_{digest.hexdigest()[:12]}.yaml"
    if not cached.exists():
        with open(cached, "w") as f:
            # sort_keys=False: several tests assert declared section order
            yaml.dump(merged, f, sort_keys=False)
    return str(cached)
