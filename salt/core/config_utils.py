"""Config utilities for run-free parsing (tests, tooling, CLI subcommands)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import yaml


def disable_logger_in_config(config_path: str) -> str:
    """Load a config, disable trainer.logger, and write to /tmp.

    For keyless environments (no COMET_API_KEY), the default-ON CometLogger
    in base2.yaml fails during instantiate_classes with "Comet.ml requires an
    API key". Tests use run=False or non-fit subcommands which skip the
    logger-wiring stage that would force it offline.

    This helper loads the config, disables the logger, and returns a /tmp path.
    Cache is per config-file (hash of input path) so parallel runs reuse it.
    """
    # Cache key: hash of the input config path (so runs sharing a config reuse it)
    cache_key = hashlib.md5(config_path.encode()).hexdigest()[:8]
    cached_path = Path(tempfile.gettempdir()) / f"salt_config_no_logger_{cache_key}.yaml"
    if cached_path.exists():
        return str(cached_path)

    # Load, disable logger, write
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if "trainer" not in cfg:
        cfg["trainer"] = {}
    cfg["trainer"]["logger"] = False

    # Write to /tmp with deterministic name
    cached_path.parent.mkdir(exist_ok=True)
    with open(cached_path, "w") as f:
        yaml.dump(cfg, f)
    return str(cached_path)
