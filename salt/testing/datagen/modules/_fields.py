"""Resolve a ``fields:`` init-arg that is either an inline list or a YAML path
(relative to ``recipes/feature_lists/``, then recipe dir / cwd).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_FEATURE_LISTS_DIR = (
    Path(__file__).resolve().parent.parent / "recipes" / "feature_lists"
)


def resolve_fields(fields) -> list[dict]:
    """Return a list of field-spec dicts from an inline list or a YAML path."""
    if isinstance(fields, (str, Path)):
        p = Path(fields)
        candidates = [p, _FEATURE_LISTS_DIR / p.name, Path.cwd() / p]
        for cand in candidates:
            if cand.exists():
                doc = yaml.safe_load(cand.read_text())
                if not isinstance(doc, list):
                    raise ValueError(
                        f"feature list {cand} must be a YAML list of field dicts"
                    )
                return list(doc)
        raise FileNotFoundError(
            f"feature list {fields!r} not found (tried {[str(c) for c in candidates]})"
        )
    return list(fields)
