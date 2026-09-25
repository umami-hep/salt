"""Warm-start (`--init_from`) checkpoint loading with per-module coverage
accounting, plus the v1-reject / `_orig_mod.`-strip guard shared with the resume
path. Owns no checkpoint FORMAT (payload key + plan hashes stay on `SaltModule`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import Tensor, nn

from salt.graph.errors import ConfigError
from salt.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from salt.graph.planner import Plan
    from salt.graph.spec import Mode

_LOG = get_logger(__name__)

__all__ = [
    "WarmStartResult",
    "apply_warm_start",
    "coverage_mismatch",
    "partition_by_module",
    "reject_v1_and_strip_orig_mod",
    "warm_start_from_checkpoint",
    "warm_start_summary",
]


def reject_v1_and_strip_orig_mod(
    state_dict: Mapping[str, Tensor] | None,
) -> Mapping[str, Tensor] | None:
    """Reject a v1 (``ModelWrapper``) state-dict; strip a ``--compile``-added
    ``_orig_mod.`` prefix. Shared by the resume and warm-start paths. Returns the
    input unchanged when no prefix is present (so callers can detect a no-op by
    identity); raises `ConfigError` on a v1 layout.
    """
    if state_dict and any(k.startswith("model.pool_net.") for k in state_dict):
        raise ConfigError(
            "this checkpoint has the v1 (ModelWrapper) state-dict layout "
            "('model.pool_net.*' keys) — v1 checkpoints are not supported by "
            "salt. Use them at the v1 pin 29c67a1 (git checkout 29c67a1) "
            "or convert the weights offline (see the parity-closure section "
            "of docs/architecture.md)."
        )
    if state_dict and any("_orig_mod." in k for k in state_dict):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


class WarmStartResult(NamedTuple):
    """The per-module classification of a `--init_from` warm-start load."""

    loaded: list[str]
    new: list[str]
    dropped: list[str]


def warm_start_from_checkpoint(
    model: nn.Module,
    path: str,
    *,
    config_modules: Mapping[str, Any],
    plans: Mapping[Mode, Plan],
    bound: bool,
    payload_key: str,
) -> WarmStartResult:
    """Warm-start weights from `path` into the (already-bound) `model` with strict
    per-module accounting — the ``--init_from`` load path.

    Unlike a resume `ckpt_path`: trainer state stays fresh, the FIT plan-hash check
    is logged but NOT enforced (the architecture may have been surgically changed),
    and the load is prefix-filtered by module name rather than strict.

    Per config-module (``net.<name>.*``): **loaded** — in both checkpoint and config
    and fully covered (same key set, shapes, dtypes); **new** — config-only, left at
    fresh init for `on_fit_start` to materialise; **dropped** — checkpoint-only.
    A retained module that is only PARTIALLY covered is a hard `ConfigError`: that is
    an architecture swap — rename it so it drops+adds cleanly. Also raises when
    unbound, on a missing ``state_dict``, or a v1 layout.
    """
    if not bound:
        raise ConfigError(
            f"--init_from {path!r}: warm start before bind — the model must compile its "
            "plans and bind first. This is an internal ordering error."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not raw_state:
        raise ConfigError(
            f"--init_from {path!r}: the checkpoint carries no 'state_dict' — it is not a "
            "salt/Lightning training checkpoint."
        )
    ckpt_state = reject_v1_and_strip_orig_mod(raw_state)
    assert ckpt_state is not None  # non-empty raw_state → non-None

    payload = checkpoint.get(payload_key) if isinstance(checkpoint, Mapping) else None
    if payload:
        for mode, plan in plans.items():
            stored = (payload.get("plan_hashes") or {}).get(mode.name)
            if stored and stored != plan.plan_hash:
                _LOG.info(
                    "--init_from: %s plan hash differs (checkpoint %s…, current %s…) — "
                    "not enforced on a weights-only warm start.",
                    mode.name,
                    stored[:16],
                    plan.plan_hash[:16],
                )

    result = apply_warm_start(model, ckpt_state, config_modules, path)
    _LOG.info(
        "--init_from %s: %d module(s) loaded, %d new (fresh init + materialise), %d dropped.\n%s",
        path,
        len(result.loaded),
        len(result.new),
        len(result.dropped),
        warm_start_summary(*result),
    )
    return result


def apply_warm_start(
    model: nn.Module,
    ckpt_state: Mapping[str, Tensor],
    config_modules: Mapping[str, Any],
    path: str,
) -> WarmStartResult:
    """Classify + load `ckpt_state` by module into `model`; returns the
    ``(loaded, new, dropped)`` module-name partition. Raises `ConfigError` on
    partial coverage of a retained module (see `warm_start_from_checkpoint`).
    """
    current_state = model.state_dict()
    current_by_mod = partition_by_module(current_state)
    ckpt_by_mod = partition_by_module(ckpt_state)
    config_names = list(config_modules)

    loaded: list[str] = []
    new: list[str] = []
    partial: list[str] = []
    to_load: dict[str, Tensor] = {}
    for name in config_names:
        cur = current_by_mod.get(name, {})
        ckpt = ckpt_by_mod.get(name, {})
        if not ckpt:
            # params-free modules (Concat/Split) have nothing to load or
            # materialise, so they are not reported as new.
            if cur:
                new.append(name)
            continue
        mismatch = coverage_mismatch(cur, ckpt)
        if mismatch is not None:
            partial.append(f"  - {name}: {mismatch}")
            continue
        loaded.append(name)
        to_load.update({key: ckpt[key] for key in cur})
    if partial:
        raise ConfigError(
            f"--init_from {path!r}: {len(partial)} retained module(s) are only PARTIALLY "
            "covered by the checkpoint — their internal architecture changed. That is a "
            "swap, not a warm start: rename the module so it drops the old weights and "
            "fresh-inits the new ones (rename-with-weights is out of scope). "
            "Offenders:\n" + "\n".join(partial)
        )
    dropped = sorted(
        name for name in ckpt_by_mod if name is not None and name not in config_modules
    )
    # strict=False tolerates the missing new-module keys; shape errors cannot
    # reach here (the coverage preflight above already rejected them).
    model.load_state_dict(to_load, strict=False)
    return WarmStartResult(loaded, new, dropped)


def partition_by_module(state: Mapping[str, Tensor]) -> dict[str | None, dict[str, Tensor]]:
    """Group a ``net.<name>.*`` state dict by module name. Keys outside that layout
    land under the ``None`` bucket (never a model module — informational only).
    """
    grouped: dict[str | None, dict[str, Tensor]] = {}
    for key, value in state.items():
        parts = key.split(".", 2)
        name = parts[1] if len(parts) >= 3 and parts[0] == "net" else None
        grouped.setdefault(name, {})[key] = value
    return grouped


def coverage_mismatch(current: Mapping[str, Tensor], ckpt: Mapping[str, Tensor]) -> str | None:
    """One-line description of why `ckpt` does not fully cover `current`, or ``None``
    when coverage is exact. Runs BEFORE `load_state_dict` because PyTorch raises on a
    shape mismatch even under ``strict=False``.
    """
    cur_keys, ckpt_keys = set(current), set(ckpt)
    if missing := cur_keys - ckpt_keys:
        return f"{len(missing)} key(s) missing from checkpoint (e.g. {min(missing)})"
    if unexpected := ckpt_keys - cur_keys:
        return f"{len(unexpected)} extra key(s) in checkpoint (e.g. {min(unexpected)})"
    for key in sorted(cur_keys):
        cval, kval = current[key], ckpt[key]
        if tuple(cval.shape) != tuple(kval.shape):
            return f"shape mismatch at {key}: model {tuple(cval.shape)} vs ckpt {tuple(kval.shape)}"
        if cval.dtype != kval.dtype:
            return f"dtype mismatch at {key}: model {cval.dtype} vs ckpt {kval.dtype}"
    return None


def warm_start_summary(loaded: list[str], new: list[str], dropped: list[str]) -> str:
    """A per-module warm-start summary table (loaded / new / dropped)."""
    rows = [
        *(f"  loaded   {name}" for name in loaded),
        *(f"  new      {name}  (fresh init + materialise)" for name in new),
        *(f"  dropped  {name}  (in checkpoint, not in config)" for name in dropped),
    ]
    return "\n".join(rows) if rows else "  (no module-level weights)"
