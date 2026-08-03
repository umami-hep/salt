"""Config utilities for run-free parsing (tests, tooling, CLI subcommands)."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import yaml


def disable_logger_in_config(config_path: str) -> str:
    """Load a config, disable trainer.logger, and write to /tmp.

    For keyless environments (no COMET_API_KEY), the default-ON CometLogger
    in base2.yaml fails during instantiate_classes with "Comet.ml requires
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
