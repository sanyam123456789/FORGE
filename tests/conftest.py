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
    "FORGE_MAX_TOOL_CALLS",
    "FORGE_MAX_LLM_TURNS",
    "FORGE_WORKSPACE_ROOT",
    "FORGE_LOG_LEVEL",
    "FORGE_LOG_FORMAT",
    "FORGE_LOG_DIR",
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
