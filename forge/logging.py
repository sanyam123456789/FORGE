"""
forge.logging — Structured logging foundation for FORGE.

Provides a get_logger() factory that returns standard Python loggers
pre-configured for either human-readable text output or newline-delimited
JSON (for later programmatic analysis of experiment runs).

Design rationale
----------------
- All FORGE modules obtain their logger via get_logger(__name__).
- A single call to setup_logging() at process startup applies the
  chosen format and level globally for the forge.* namespace.
- JSON format is designed for eventual research log parsing; each line
  is a self-contained JSON object.
- No third-party logging libraries are required (structlog etc. are
  explicitly avoided to keep dependencies minimal at Step 1).

Future extensions
-----------------
- RunLogger (in forge.observability) will wrap this to attach per-run
  metadata (run_id, task_id) and emit structured events for research
  instrumentation (LLM call counts, token usage, latency, etc.).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # only for type annotations

# The root logger name for the entire FORGE package.
FORGE_LOGGER_NAME = "forge"

# Track whether setup_logging() has been called.
_logging_configured = False


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields emitted
    --------------
    ts          ISO-8601 timestamp (UTC)
    level       Log level name
    logger      Logger name (e.g. forge.config)
    msg         Formatted message
    exc         Exception info, if present (optional)
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


class TextFormatter(logging.Formatter):
    """Human-readable text log format.

    Example output:
        2026-09-11T00:00:00+00:00  INFO     forge.config  Settings loaded
    """

    _FMT = "%(asctime)s  %(levelname)-8s %(name)-30s %(message)s"
    _DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE_FMT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    fmt: str = "text",
    log_dir: Path | None = None,
) -> None:
    """Configure the forge.* logging hierarchy.

    Should be called once at process startup (e.g. in the CLI entry point).
    Calling it more than once is safe — subsequent calls are no-ops.

    Parameters
    ----------
    level:
        One of DEBUG | INFO | WARNING | ERROR.
    fmt:
        'text' for human-readable output, 'json' for newline-delimited JSON.
    log_dir:
        If provided, FORGE also writes logs to a file in this directory.
        The file is named ``forge_<date>.log``.
    """
    global _logging_configured
    if _logging_configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter: logging.Formatter = JsonFormatter() if fmt == "json" else TextFormatter()

    # Configure the root forge logger
    root = logging.getLogger(FORGE_LOGGER_NAME)
    root.setLevel(numeric_level)
    root.propagate = False  # don't bubble up to the root Python logger

    # Always add a stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # Optionally add a file handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        log_file = log_dir / f"forge_{date_str}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _logging_configured = True

    logger = logging.getLogger(FORGE_LOGGER_NAME)
    logger.debug("FORGE logging initialised (level=%s, format=%s)", level, fmt)


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped under the forge.* namespace.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module, e.g. 'forge.config'.
        If the name does not already start with 'forge.', it is prefixed
        automatically so all FORGE loggers share a common hierarchy.
    """
    if not name.startswith(FORGE_LOGGER_NAME):
        name = f"{FORGE_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def reset_logging() -> None:
    """Reset the logging configuration (intended for use in tests only)."""
    global _logging_configured
    root = logging.getLogger(FORGE_LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    _logging_configured = False
