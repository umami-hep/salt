"""Salt's logging facility: level-filtered diagnostics plus an unconditional console.

Two surfaces: `get_logger` for stderr diagnostics filtered by level, `console`
for user-facing results that always reach stdout.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

ROOT_NAME = "salt"
DEFAULT_LEVEL = "INFO"
LEVEL_ENV_VAR = "SALT_LOG_LEVEL"

__all__ = ["DEFAULT_LEVEL", "LEVEL_ENV_VAR", "ROOT_NAME", "console", "get_logger", "set_level"]

_CONFIGURED = False
_EXPLICIT_LEVEL: int | str | None = None

_FORMAT = "%(levelname)s %(name)s: %(message)s"

_CONSOLE_KWARGS = {"sep", "end", "file", "flush"}


def _style(level: str, text: str) -> str:  # noqa: ARG001 - the two-arg signature is the seam's contract; a colour implementation branches on `level`
    """Colour hook. Identity today; the single seam to add colour later.

    Called by both surfaces: the formatter passes `record.levelname`,
    `console` passes the literal string `"CONSOLE"`.
    """
    return text


class _SaltFormatter(logging.Formatter):
    """Formats a record then runs the result through the `_style` seam."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return _style(record.levelname, formatted)


def _resolve_level(level: int | str) -> int:
    """Resolve an int or a level name (case-insensitive) to a numeric level."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"invalid log level: {level!r}")


def _env_level() -> int:
    """Resolve `SALT_LOG_LEVEL`, falling back to `DEFAULT_LEVEL` on any bad value."""
    raw = os.environ.get(LEVEL_ENV_VAR)
    if raw is None:
        return _resolve_level(DEFAULT_LEVEL)
    try:
        return _resolve_level(raw)
    except ValueError:
        return _resolve_level(DEFAULT_LEVEL)


class _StderrHandler(logging.StreamHandler):
    """StreamHandler resolving `sys.stderr` at emit time, not at construction."""

    _salt_handler = True  # marks salt's own handler, for the idempotency check below

    def __init__(self, level: int = logging.NOTSET) -> None:
        logging.Handler.__init__(self, level)

    @property
    def stream(self):
        return sys.stderr


def _ensure_configured() -> None:
    """Idempotently install a single marked handler on the `salt` logger."""
    global _CONFIGURED  # noqa: PLW0603 - module-level singleton state for a process-wide logging facility
    if _CONFIGURED:
        return
    root = logging.getLogger(ROOT_NAME)
    if not any(getattr(h, "_salt_handler", False) for h in root.handlers):
        handler = _StderrHandler()
        handler.setFormatter(_SaltFormatter(_FORMAT))
        root.addHandler(handler)
    level = _EXPLICIT_LEVEL if _EXPLICIT_LEVEL is not None else _env_level()
    root.setLevel(_resolve_level(level))
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the `salt` hierarchy, configuring it if needed.

    Parameters
    ----------
    name:
        `None` or `ROOT_NAME` returns the `salt` root logger itself. A name
        already prefixed with `"salt."` (the normal path — pass `__name__`)
        is used verbatim. Any other name is nested under `salt.`.

    Returns
    -------
    logging.Logger
        A standard library logger, configured with salt's handler and level.
    """
    _ensure_configured()
    if name is None or name == ROOT_NAME:
        return logging.getLogger(ROOT_NAME)
    if name.startswith(f"{ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def set_level(level: int | str) -> None:
    """Set the `salt` logger's level explicitly, overriding the env var.

    Parameters
    ----------
    level:
        A `logging` numeric level (e.g. `logging.DEBUG`) or a level name
        (e.g. `"DEBUG"`, case-insensitive).
    """
    global _EXPLICIT_LEVEL  # noqa: PLW0603 - module-level singleton state for a process-wide logging facility
    _EXPLICIT_LEVEL = level
    _ensure_configured()
    logging.getLogger(ROOT_NAME).setLevel(_resolve_level(level))


def console(*args: object, **kwargs: object) -> None:
    r"""Write user-facing results to stdout, unconditionally.

    Not a log record: no level, no filtering. This is the destination for
    CLI results and reports the user explicitly asked for.

    Parameters
    ----------
    *args:
        Values to print, stringified and joined by `sep`.
    **kwargs:
        `sep` (default `" "`), `end` (default `"\n"`), `file` (default
        `None`, resolved to `sys.stdout` at call time), `flush` (default
        `False`). Any other keyword raises `TypeError`.
    """
    unknown = set(kwargs) - _CONSOLE_KWARGS
    if unknown:
        raise TypeError(f"console() got unexpected keyword argument(s): {sorted(unknown)}")
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file: TextIO | None = kwargs.get("file")  # type: ignore[assignment]
    flush = kwargs.get("flush", False)

    if file is None:
        file = sys.stdout

    text = sep.join(str(a) for a in args)
    text = _style("CONSOLE", text)
    file.write(text + end)
    if flush:
        file.flush()
