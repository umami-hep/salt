"""``--pipeline-row``/``--pipeline-row-scope`` selectors for the per-config CI matrix.

Study ruling (plan 04, 2026-08-31): GPU row selection MUST be an exact row-id
match, never a bare ``pytest -k`` substring selection. ``-k`` can silently pick
the wrong subset while staying green — e.g. ``-k gn3_v00`` also matches
``gn3v00_base`` AND anything else containing that substring, so a typo'd or
overly-broad ``-k`` expression reports success while running extra or fewer
rows than intended. The selectors below match on the exact parametrize id
instead (``item.callspec.params["name"]``), and raise immediately on an
unknown row id rather than silently collecting zero items.

Plan 06 extends this to three named scopes (``--pipeline-row-scope``, defined
in the root ``salt/tests/conftest.py`` — see its help text for the full
contract), so the three CI job families can each ask for exactly the tests
they own, with an explicit accounting rather than "the rest":

- ``all`` (default, with ``--pipeline-row``): every test parametrized with
  that row id — lifecycle legs, the compile_plot floor, and the
  inference/name-check tests. The per-config CPU CI jobs (one job per
  runnable fixture, running ALL of that row's tests).
- ``legs`` (with ``--pipeline-row``): lifecycle legs only
  (test_fit/test_eval/test_export), byte-for-byte the original selector
  behaviour. The manual integration-gpu jobs.
- ``misc`` (``--pipeline-row`` must be unset): only tests that are NOT
  parametrized with any row id at all — completeness/consistency checks,
  fragment tests, regression/gaussian sink-mechanics classes, finetune
  residual assertions, multistage_training/, etc. The integration-misc CPU
  job. Every "name"-parametrized item is accounted for explicitly: a value
  that IS a row id is deselected (it runs in exactly its per-config row job
  elsewhere); a value that is NOT a row id raises loudly instead of being
  silently orphaned.
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


def _deselect(config: pytest.Config, items: list[pytest.Item], kept: list[pytest.Item],
              deselected: list[pytest.Item], zero_kept_msg: str) -> None:
    if not kept:
        raise pytest.UsageError(zero_kept_msg)
    config.hook.pytest_deselected(items=deselected)
    items[:] = kept


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    row = config.getoption("--pipeline-row")
    scope = config.getoption("--pipeline-row-scope")

    if scope == "misc":
        if row is not None:
            raise pytest.UsageError(
                "--pipeline-row-scope=misc cannot be combined with --pipeline-row"
            )
        valid = set(_valid_row_ids())
        kept: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            callspec = getattr(item, "callspec", None)
            if callspec is not None and "name" in callspec.params:
                value = callspec.params["name"]
                if value in valid:
                    # Runs in exactly its per-config row job — not here.
                    deselected.append(item)
                    continue
                raise pytest.UsageError(
                    f"--pipeline-row-scope=misc: item {item.nodeid!r} is parametrized "
                    f"with name={value!r}, which is not a known pipeline row id "
                    f"({', '.join(sorted(valid))}). A 'name'-parametrized integration "
                    "test whose values are not row ids would be silently orphaned by "
                    "the misc job's row-id deselect — fix the fixture/parametrize "
                    "values or extend this accounting explicitly."
                )
            kept.append(item)
        _deselect(
            config,
            items,
            kept,
            deselected,
            "--pipeline-row-scope=misc selected zero test items — harness bug "
            "(every 'name'-parametrized item was a known row id, and nothing "
            "unparametrized was collected either).",
        )
        return

    if row is None:
        if scope == "legs":
            raise pytest.UsageError("--pipeline-row-scope=legs requires --pipeline-row")
        return  # scope == "all", row is None: plain local runs, unchanged.

    valid = _valid_row_ids()
    if row not in valid:
        raise pytest.UsageError(
            f"--pipeline-row={row!r} is not a known pipeline row id. "
            f"Valid row ids: {', '.join(valid)}"
        )

    kept = []
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        matches_row = callspec is not None and callspec.params.get("name") == row
        if not matches_row:
            deselected.append(item)
            continue
        if scope == "legs":
            name = getattr(item, "originalname", None) or item.name.split("[")[0]
            if name not in _LEGS:
                deselected.append(item)
                continue
        kept.append(item)

    _deselect(
        config,
        items,
        kept,
        deselected,
        f"--pipeline-row={row!r} --pipeline-row-scope={scope!r} matched a known row id "
        "but selected zero test items — harness bug (lifecycle leg collection or the "
        "callspec 'name' param may have changed shape).",
    )
