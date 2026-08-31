"""``--pipeline-row`` selector for the integration-gpu CI matrix.

Study ruling (plan 04, 2026-08-31): GPU row selection MUST be an exact row-id
match, never a bare ``pytest -k`` substring selection. ``-k`` can silently pick
the wrong subset while staying green — e.g. ``-k gn3_v00`` also matches
``gn3v00_base`` AND anything else containing that substring, so a typo'd or
overly-broad ``-k`` expression reports success while running extra or fewer
rows than intended. The selector below matches on the exact parametrize id
instead (``item.callspec.params["name"]``), and raises immediately on an
unknown row id rather than silently collecting zero items.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_LEGS = {"test_fit", "test_eval", "test_export"}


def _valid_row_ids() -> list[str]:
    """Row ids = fixture stems whose YAML has no top-level ``fragment`` key."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    row_ids = []
    for path in sorted(fixtures_dir.glob("*.yaml")):
        with path.open() as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "fragment" not in data:
            row_ids.append(path.stem)
    return row_ids


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    row = config.getoption("--pipeline-row")
    if row is None:
        return

    valid = _valid_row_ids()
    if row not in valid:
        raise pytest.UsageError(
            f"--pipeline-row={row!r} is not a known pipeline row id. "
            f"Valid row ids: {', '.join(valid)}"
        )

    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        name = getattr(item, "originalname", None) or item.name.split("[")[0]
        if name not in _LEGS:
            deselected.append(item)
            continue
        callspec = getattr(item, "callspec", None)
        matches_row = callspec is not None and callspec.params.get("name") == row
        (kept if matches_row else deselected).append(item)

    if not kept:
        raise pytest.UsageError(
            f"--pipeline-row={row!r} matched a known row id but selected zero test "
            "items — harness bug (lifecycle leg collection or the callspec 'name' "
            "param may have changed shape)."
        )

    config.hook.pytest_deselected(items=deselected)
    items[:] = kept
