"""Pytest configuration for the salt test suite."""

from __future__ import annotations

import pytest

from salt.core.config_utils import disable_logger_in_config  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def _warm_salt2cli_model_resolution():
    """Absorb the first run-free ``Salt2CLI`` parse's model-resolution failure."""
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
