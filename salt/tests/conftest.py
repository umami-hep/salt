"""Pytest configuration for the salt test suite.

Integration tests — the gate harnesses plus anything that builds/runs a full
model or reads a real data file — live under ``salt/tests/integration/`` and/or
carry the ``@pytest.mark.integration`` marker. They are **GPU-only by default**:

* a plain ``pytest`` on a CPU box runs the fast unit suite and *skips* the
  integration tests;
* on a machine with a CUDA device they run automatically;
* ``pytest --run-integration`` forces them even on CPU.

So day-to-day ``pytest`` (or ``pytest salt/tests/unit``) stays fast, and the
heavy parity / ONNX / training gates only fire where they belong.
"""

from __future__ import annotations

import pytest

from salt.core.config_utils import disable_logger_in_config  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def _warm_salt2cli_model_resolution():
    """Absorb the first run-free ``Salt2CLI`` parse's model-resolution failure.

    Empirically (see the plan-35 investigation), the FIRST run-free
    ``Salt2CLI(... run=False)`` parse in a test process fails to validate the
    ``model:`` block ("model does not validate against any of the Union
    subtypes"), but the attempt itself "warms" jsonargparse's first-time
    resolution of the ``salt.core.SaltModule`` subclass typehint, so EVERY
    subsequent parse in the process succeeds. It only triggers once the test
    collection has imported the sibling test modules (``test_main`` alone never
    trips it), and which test pays the cost depends on the pytest-randomly
    schedule — hence the intermittent CI flake on whichever ``make_cli``-based
    test happens to run first. Production never imports the test modules and
    parses once per fresh process, so it is unaffected (``salt2 graph validate``
    and the GPU integration suite all pass). Do one throwaway parse here, at
    session start, so the real tests always run warm. ``Salt2CLI(run=False)``
    raises ``SystemExit`` on a parse error, so catch that too — the resolution
    is warmed regardless of how the throwaway parse ends.
    """
    try:
        from salt.core.main import CONFIG_DIR, Salt2CLI

        cfg = disable_logger_in_config(str(CONFIG_DIR / "gn2v2-dummy.yaml"))
        Salt2CLI(args=["--config", cfg], run=False)
    except (Exception, SystemExit):  # noqa: BLE001 - warm-up only; never fail the session
        pass
    yield


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run the GPU/integration tests (gates, full-model fits, real-data reads) "
        "even without a CUDA device",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: slow/GPU test (gate harness, full-model fit, or real-data read). "
        "Skipped on a CPU box unless --run-integration is passed; runs automatically "
        "when a CUDA device is present.",
    )
    config.addinivalue_line(
        "markers",
        "cpu_always: a pure-CPU test that lives under tests/integration/ but does NOT "
        "need a GPU (e.g. the plan-29 W2 ONNX-trace fold gates). It runs on EVERY "
        "pytest invocation — the integration/GPU skip never applies — so these CI-load "
        "bearing CPU gates are never silently skipped.",
    )


def _is_integration(item: pytest.Item) -> bool:
    """An item is integration if it lives under tests/integration/ or is marked."""
    return "/integration/" in str(getattr(item, "path", "")).replace("\\", "/") or (
        item.get_closest_marker("integration") is not None
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # GPU box or explicit opt-in → run everything.
    if config.getoption("--run-integration"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            return
    except Exception:  # torch missing / driver error → treat as no-GPU, be safe
        pass

    skip = pytest.mark.skip(
        reason="integration test: GPU-only (run on a CUDA box, or pass --run-integration)"
    )
    for item in items:
        # `cpu_always` tests are pure-CPU and must run even on a CPU box without
        # --run-integration (plan 29 W2 B3: the ONNX-trace fold gates are CI-load
        # bearing). They opt OUT of the integration/GPU skip.
        if item.get_closest_marker("cpu_always") is not None:
            continue
        if _is_integration(item):
            item.add_marker(skip)
