from __future__ import annotations

import io
import logging
import sys

import pytest

import salt.logging as salt_logging
from salt.logging import DEFAULT_LEVEL, LEVEL_ENV_VAR, ROOT_NAME, console, get_logger, set_level


@pytest.fixture(autouse=True)
def _isolate_salt_logging(monkeypatch):
    """Snapshot and restore the `salt` logger + module globals around each test."""
    root = logging.getLogger(ROOT_NAME)
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_propagate = root.propagate
    saved_configured = salt_logging._CONFIGURED
    saved_explicit_level = salt_logging._EXPLICIT_LEVEL
    saved_style = salt_logging._style

    monkeypatch.delenv(LEVEL_ENV_VAR, raising=False)

    root.handlers = []
    salt_logging._CONFIGURED = False
    salt_logging._EXPLICIT_LEVEL = None

    yield

    root.handlers = saved_handlers
    root.setLevel(saved_level)
    root.propagate = saved_propagate
    salt_logging._CONFIGURED = saved_configured
    salt_logging._EXPLICIT_LEVEL = saved_explicit_level
    salt_logging._style = saved_style


def test_default_level_is_info():
    logger = get_logger()
    assert logger.level == logging.INFO
    assert logging.getLevelName(logging.INFO) == DEFAULT_LEVEL


def test_env_var_debug_is_honoured(monkeypatch):
    monkeypatch.setenv(LEVEL_ENV_VAR, "DEBUG")
    salt_logging._CONFIGURED = False
    salt_logging._EXPLICIT_LEVEL = None

    logger = get_logger()

    assert logger.level == logging.DEBUG


def test_env_var_invalid_falls_back_to_default_without_raising(monkeypatch):
    monkeypatch.setenv(LEVEL_ENV_VAR, "NOT_A_REAL_LEVEL")
    salt_logging._CONFIGURED = False
    salt_logging._EXPLICIT_LEVEL = None

    logger = get_logger()

    assert logger.level == logging.getLevelName(DEFAULT_LEVEL)


def test_set_level_wins_over_env_var(monkeypatch):
    monkeypatch.setenv(LEVEL_ENV_VAR, "DEBUG")

    set_level("WARNING")

    assert logging.getLogger(ROOT_NAME).level == logging.WARNING


def test_salt_logger_keeps_stdlib_propagation():
    get_logger()
    assert logging.getLogger(ROOT_NAME).propagate is True


def test_handler_installation_is_idempotent():
    for _ in range(5):
        get_logger()

    root = logging.getLogger(ROOT_NAME)
    marked = [h for h in root.handlers if getattr(h, "_salt_handler", False)]
    assert len(marked) == 1


def test_get_logger_with_salt_prefixed_name_used_verbatim():
    logger = get_logger("salt.foo")
    assert logger.name == "salt.foo"


def test_get_logger_with_bare_name_gets_salt_prefix():
    logger = get_logger("foo")
    assert logger.name == "salt.foo"


def test_console_defaults_to_stdout(capsys):
    console("x")
    captured = capsys.readouterr()
    assert captured.out == "x\n"
    assert captured.err == ""


def test_console_with_explicit_file_goes_to_stderr(capsys):
    console("x", file=sys.stderr)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "x\n"


def test_console_respects_sep_and_end():
    buf = io.StringIO()
    console("a", "b", sep="-", end="", file=buf)
    assert buf.getvalue() == "a-b"


def test_console_unknown_kwarg_raises_type_error():
    with pytest.raises(TypeError):
        console("x", nope=True)


def test_console_not_suppressed_at_critical_level():
    set_level("CRITICAL")

    buf = io.StringIO()
    console("still here", file=buf)

    assert buf.getvalue() == "still here\n"


def test_style_is_identity_today():
    assert salt_logging._style("INFO", "hello") == "hello"


def test_style_is_a_single_seam_for_both_surfaces(monkeypatch, capsys):
    monkeypatch.setattr(salt_logging, "_style", lambda level, text: f"[{level}] {text}")

    logger = get_logger("salt.seam_test")
    logger.warning("from logger")
    console("from console")

    captured = capsys.readouterr()
    assert "[WARNING] WARNING salt.seam_test: from logger" in captured.err
    assert captured.out == "[CONSOLE] from console\n"
