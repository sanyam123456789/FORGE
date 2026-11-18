"""
Shared pytest fixtures for the FORGE test suite.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------

# All FORGE env vars that tests might set — cleared before each test that
# uses the clean_env fixture so tests are fully isolated from each other
# and from whatever the developer has set in their shell.
_FORGE_ENV_KEYS = [
    "FORGE_LLM_PROVIDER",
    "FORGE_LLM_MODEL",
    "FORGE_LLM_API_KEY",
    "FORGE_LLM_BASE_URL",
    "FORGE_PROVIDER_ORDER",
    "FORGE_PROVIDER_RETRY_MAX",
    "FORGE_LLM_REQUEST_TIMEOUT_S",
    "FORGE_LLM_TEMPERATURE",
    "FORGE_LLM_MAX_OUTPUT_TOKENS",
    "FORGE_MAX_TOOL_CALLS",
    "FORGE_MAX_LLM_TURNS",
    "FORGE_WORKSPACE_ROOT",
    "FORGE_LOG_LEVEL",
    "FORGE_LOG_FORMAT",
    "FORGE_LOG_DIR",
    "FORGE_SHELL_ENV_ALLOW",
    "FORGE_REPO_MAP",
    "FORGE_PROJECT_RULES",
    "FORGE_APPROVE",
    "FORGE_STREAM",
    # Not FORGE_-prefixed, but read by the opencode_go provider preset.
    "OPENCODE_API_KEY",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all FORGE_* environment variables for the duration of a test.

    This ensures that tests run with pure defaults and are not affected by
    values the developer may have set in their local .env file or shell.
    """
    for key in _FORGE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_forge_logging():
    """Undo setup_logging() after each test.

    CLI tests call setup_logging(), which attaches a handler to the
    (pytest-captured, later closed) stderr and stops forge.* records from
    reaching the root logger, so later caplog-based tests would see nothing.
    """
    yield
    import logging

    import forge.logging as forge_logging

    root = logging.getLogger(forge_logging.FORGE_LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.propagate = True
    root.setLevel(logging.NOTSET)
    forge_logging._logging_configured = False


# ---------------------------------------------------------------------------
# Network guard
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_llm_network(monkeypatch):
    """Backstop: no test may reach a real LLM endpoint.

    The provider's default transport is the only code path that opens a
    network connection. If a test ever ends up on a real provider (for
    example through a missed monkeypatch), it gets an error instead of
    sending the developer's API key over the network. Tests that exercise
    the transport itself patch ``urlopen`` again inside the test.
    """

    def _blocked(*args, **kwargs):
        raise RuntimeError("real network access is disabled in FORGE tests")

    monkeypatch.setattr("forge.providers.openai_compat.urllib.request.urlopen", _blocked)
