"""Pytest configuration for the salt test suite."""

from __future__ import annotations

import pytest

from salt.config_utils import disable_logger_in_config  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def _warm_saltcli_model_resolution():
    """Absorb the first run-free ``SaltCLI`` parse's model-resolution failure.

    jsonargparse's first-time subclass typehint resolution is fragile to
    import/collection order (pytest-randomly reshuffles it). The first parse of
    each distinct resolution path warms jsonargparse's global type cache so the
    per-entry ``dict[str, Subclass]`` merge (``DeepMergeParser.merge_config``)
    is recognised instead of degrading to an atomic dict replace. A single
    single-config parse only warms one path; the stacked-config + dotted
    subclass-dict-override parse warms the deep-merge path every failing test
    (onnx-fold / mup / deep-merge / null-deletion / spike) actually exercises.
    """
    try:
        from salt.main import CONFIG_DIR, SaltCLI

        base = disable_logger_in_config(str(CONFIG_DIR / "gn2v2-dummy.yaml"))
        # 1. single-config parse — the original warm.
        SaltCLI(args=["--config", base], run=False)
        # 2. stacked config files + a dotted override into a subclass-dict value
        #    (``model.modules.norm`` -> Normaliser.norm_dict) — the cross-config
        #    deep-merge / per-entry-dict-value resolution path. Warming this once
        #    at session start keeps sibling module keys from being dropped when a
        #    later config or CLI override touches one entry.
        fold = disable_logger_in_config(str(CONFIG_DIR / "gn2v2-dummy-onnx-fold.yaml"))
        SaltCLI(
            args=[
                "--config",
                base,
                "--config",
                fold,
                "--model.modules.norm.init_args.norm_dict=unused.yaml",
            ],
            run=False,
        )
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
