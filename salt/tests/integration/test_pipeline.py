"""The config-lifecycle matrix, driven — parametrisation, xfails, completeness.

The matrix and its mechanism (feeders, ``run_row``, the leg implementations)
live in ``pipeline.py``. This module is the pytest surface: three
parametrized legs per row (fit always runs; eval/export skip when the row
declares ``do_eval=False``/``do_onnx=False``), the xfail table lookup, the
matrix<->discovery completeness checks (§4.3), and the two residual
finetune-template assertions that survive the fold of
``test_finetune_templates.py`` (§3) — claims about the chained artifacts that
a matrix row cannot itself express.
"""

from __future__ import annotations

import pytest

from salt.main import CONFIG_DIR
from salt.tests.integration.pipeline import (
    FEEDS,
    GPU_ROWS,
    KNOWN_FAILURES,
    MATRIX,
    NO_GOLDEN,
    LegFailedError,
    RootDepsMissingError,
    dependencies_of,
    golden_path,
    production_configs,
    row_by_name,
    run_eval,
    run_export,
    run_row,
)

# cpu_always: every row uses --trainer.accelerator=auto on synthetic data, so
# the whole matrix is CPU-safe and must run on every CI invocation (the
# tests/integration/ GPU-skip in conftest.py would otherwise hide it on a
# CPU box). pipeline: the marker the CI split (§5.1) selects on — its own
# job, parallel with the rest of tests/integration/.
pytestmark = [pytest.mark.pipeline, pytest.mark.cpu_always]

_PARAMS = [
    pytest.param(row.test_name, marks=(pytest.mark.gpu,) if row.test_name in GPU_ROWS else ())
    for row in MATRIX
]


def _apply_known_xfail(request: pytest.FixtureRequest, name: str, leg: str) -> None:
    reason = KNOWN_FAILURES.get((name, leg))
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=reason))


@pytest.mark.parametrize("name", _PARAMS)
def test_fit(name, tmp_path_factory, request):
    """Leg 1 — every row: ``salt fit`` produces a checkpoint + saved config."""
    _apply_known_xfail(request, name, "fit")
    try:
        run_row(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.producer != name:
            pytest.skip(f"producer row {exc.producer} failed")
        raise


@pytest.mark.parametrize("name", _PARAMS)
def test_eval(name, tmp_path_factory, request):
    """Leg 2 — ``do_eval=True`` rows: ``salt test`` + the H5-schema-vs-golden assertion."""
    row = row_by_name(name)
    if not row.do_eval:
        pytest.skip("row declares do_eval=False")
    _apply_known_xfail(request, name, "eval")
    try:
        run_eval(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.leg == "fit":
            pytest.skip(f"fit failed for {name}")
        raise


@pytest.mark.parametrize("name", _PARAMS)
def test_export(name, tmp_path_factory, request):
    """Leg 3 — ``do_onnx=True`` rows: ``salt export``, checked (torch<->ONNX parity)."""
    row = row_by_name(name)
    if not row.do_onnx:
        pytest.skip("row declares do_onnx=False")
    _apply_known_xfail(request, name, "export")
    try:
        run_export(name, tmp_path_factory)
    except RootDepsMissingError as exc:
        pytest.skip(str(exc))
    except LegFailedError as exc:
        if exc.leg == "fit":
            pytest.skip(f"fit failed for {name}")
        raise


# --------------------------------------------------------- completeness (§4.3)


def test_every_production_config_has_a_matrix_row():
    """A new production flagship cannot slip in without a matrix row."""
    matrix_configs = {r.config for r in MATRIX}
    missing = sorted(set(production_configs()) - matrix_configs)
    assert not missing, (
        f"production configs with no matrix row: {missing}. Add a row to MATRIX "
        "(pipeline.py) and a FEEDS entry naming its data source."
    )


def test_matrix_rows_name_shipped_configs():
    """The reverse direction of the same fence: every row names a real config."""
    missing = sorted(r.config for r in MATRIX if not (CONFIG_DIR / f"{r.config}.yaml").is_file())
    assert not missing, f"MATRIX rows name configs that do not exist: {missing}"


def test_matrix_test_names_are_unique():
    """``test_name`` is the artifact-cache key and the junit id."""
    names = [r.test_name for r in MATRIX]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"MATRIX test_names are not unique: {dupes}"


def test_matrix_dependencies_are_defined_and_acyclic():
    """Every ``{ckpt:NAME}``/``{config:NAME}`` names a row; the DAG has no cycle."""
    names = {r.test_name for r in MATRIX}
    deps = {r.test_name: dependencies_of(r) for r in MATRIX}
    for name, dep_names in deps.items():
        unknown = sorted(dep_names - names)
        assert not unknown, f"{name}: train_args names unknown row(s) {unknown}"

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in visited:
            return
        if name in visiting:
            raise AssertionError(f"MATRIX dependency cycle: {' -> '.join((*stack, name))}")
        visiting.add(name)
        for dep in deps[name]:
            visit(dep, (*stack, name))
        visiting.discard(name)
        visited.add(name)

    for name in names:
        visit(name, ())


def test_every_matrix_config_has_a_feed():
    """No row can reach the fit leg without a declared data source."""
    missing = sorted({r.config for r in MATRIX} - set(FEEDS))
    assert not missing, f"MATRIX rows with no FEEDS entry: {missing}"


def test_known_failures_name_real_rows_and_legs():
    """Stale KNOWN_FAILURES entries fail loudly instead of silently protecting nothing."""
    names = {r.test_name for r in MATRIX}
    for name, leg in KNOWN_FAILURES:
        assert name in names, f"KNOWN_FAILURES names an unknown row: {name!r}"
        assert leg in {"fit", "eval", "export"}, (
            f"KNOWN_FAILURES names an unknown leg {leg!r} for {name!r}"
        )


def test_no_golden_table_is_honest():
    """Every do_eval=True row has a committed golden, or a reason in NO_GOLDEN — never both."""
    eval_rows = {r.test_name: r for r in MATRIX if r.do_eval}
    for name in NO_GOLDEN:
        assert name in eval_rows, f"NO_GOLDEN names a row that does not do_eval: {name!r}"
    for name, row in eval_rows.items():
        golden = golden_path(row)
        if golden.is_file():
            assert name not in NO_GOLDEN, (
                f"{name} is in NO_GOLDEN but a golden exists at {golden} — the table is stale"
            )
        else:
            assert name in NO_GOLDEN, (
                f"{name} has do_eval=True and no golden at {golden} — add it to "
                "NO_GOLDEN with a reason, or commit a golden"
            )


# ----------------------------------------------- residual finetune assertions (§3)
# What did NOT fold from test_finetune_templates.py: claims about the CHAINED
# artifacts (rows 13-15), not observable from rc == 0 on a fit.


def test_base_run_carries_the_modules_the_templates_freeze(tmp_path_factory):
    """The saved base config names the head both finetune templates warm up."""
    import yaml

    try:
        artifacts = run_row("gn3v00_base", tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"gn3v00_base fit failed: {exc}")
    modules = yaml.safe_load(artifacts.saved_config.read_text())["model"]["init_args"]["modules"]
    assert "jets_classification" in modules, (
        "finetune_gn3large.yaml warms up `jets_classification`; the base config "
        "no longer defines it"
    )


def test_added_head_is_absent_from_the_pretrained_checkpoint(tmp_path_factory):
    """The new head really is new — the warm start cannot be a no-op."""
    import torch

    try:
        artifacts = run_row("gn3v00_base", tmp_path_factory)
    except LegFailedError as exc:
        pytest.skip(f"gn3v00_base fit failed: {exc}")
    state = torch.load(artifacts.ckpt, map_location="cpu", weights_only=False)
    keys = state.get("state_dict", state)
    assert not any("large_r_jet_classification" in k for k in keys), (
        "the base checkpoint already carries the head the template adds, so this "
        "test would no longer prove the new-head path works"
    )
