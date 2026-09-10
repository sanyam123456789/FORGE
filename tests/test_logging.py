"""
Tests for forge.logging — logging initialisation and formatters.
"""

from __future__ import annotations

import json
import logging

import pytest

from forge.logging import (
    JsonFormatter,
    TextFormatter,
    get_logger,
    reset_logging,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_forge_logging():
    """Ensure logging is reset before and after each test."""
    reset_logging()
    yield
    reset_logging()


class TestSetupLogging:
    def test_setup_configures_forge_logger(self):
        setup_logging(level="DEBUG", fmt="text")
        logger = logging.getLogger("forge")
        assert logger.level == logging.DEBUG

    def test_setup_idempotent(self):
        setup_logging(level="INFO", fmt="text")
        setup_logging(level="DEBUG", fmt="json")  # second call should be no-op
        logger = logging.getLogger("forge")
        # Level should still be INFO (first call wins)
        assert logger.level == logging.INFO

    def test_setup_adds_handler(self):
        setup_logging(level="INFO", fmt="text")
        logger = logging.getLogger("forge")
        assert len(logger.handlers) >= 1

    def test_file_handler_created(self, tmp_path):
        setup_logging(level="INFO", fmt="text", log_dir=tmp_path)
        log_files = list(tmp_path.glob("forge_*.log"))
        assert len(log_files) == 1

    def test_log_dir_created_if_missing(self, tmp_path):
        log_dir = tmp_path / "nested" / "logs"
        setup_logging(level="INFO", fmt="text", log_dir=log_dir)
        assert log_dir.exists()


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("forge.test_module")
        assert isinstance(logger, logging.Logger)

    def test_name_prefixed(self):
        logger = get_logger("my_module")
        assert logger.name.startswith("forge.")

    def test_already_prefixed_not_doubled(self):
        logger = get_logger("forge.config")
        assert logger.name == "forge.config"
        assert "forge.forge" not in logger.name


class TestJsonFormatter:
    def test_outputs_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="forge.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["msg"] == "hello world"
        assert data["level"] == "INFO"
        assert "ts" in data
        assert "logger" in data

    def test_exception_included(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="forge.test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="boom",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exc" in data
        assert "ValueError" in data["exc"]


class TestTextFormatter:
    def test_formats_without_error(self):
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="forge.config",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "WARNING" in output
        assert "test message" in output
        assert "forge.config" in output
