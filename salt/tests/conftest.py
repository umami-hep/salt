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
        if _is_integration(item):
            item.add_marker(skip)
