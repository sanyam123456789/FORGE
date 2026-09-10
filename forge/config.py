"""
forge.config — Configuration loading and validation.

All runtime settings are sourced from environment variables (with .env
file support via python-dotenv).  No secrets are ever hard-coded or stored
in the source tree.

Usage
-----
    from forge.config import settings

    print(settings.llm_provider)
    print(settings.llm_model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# python-dotenv is an optional convenience — we fall back gracefully if absent.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # dotenv not installed; rely on real environment variables


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LLMProvider = Literal["openai", "anthropic", "google", "openai_compatible"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["text", "json"]


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgeSettings:
    """Immutable snapshot of all FORGE runtime settings.

    All values are read from environment variables at construction time.
    Secrets (api_key) are never logged or exposed in __repr__.
    """

    # --- LLM provider ---
    llm_provider: LLMProvider = field(default="openai")
    llm_model: str = field(default="gpt-4o")
    llm_api_key: str = field(default="")
    llm_base_url: str = field(default="")

    # --- Runtime ---
    max_tool_calls: int = field(default=50)
    max_llm_turns: int = field(default=30)
    workspace_root: Path = field(default_factory=Path.cwd)

    # --- Logging ---
    log_level: LogLevel = field(default="INFO")
    log_format: LogFormat = field(default="text")
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def __repr__(self) -> str:  # pragma: no cover
        """Redact secret fields from string representation."""
        key_hint = "***" if self.llm_api_key else "<not set>"
        return (
            f"ForgeSettings("
            f"llm_provider={self.llm_provider!r}, "
            f"llm_model={self.llm_model!r}, "
            f"llm_api_key={key_hint}, "
            f"max_tool_calls={self.max_tool_calls}, "
            f"max_llm_turns={self.max_llm_turns}, "
            f"workspace_root={self.workspace_root!r}, "
            f"log_level={self.log_level!r}, "
            f"log_format={self.log_format!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"FORGE configuration error: {key}={raw!r} is not a valid integer."
        ) from exc


def _get_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key, "").strip()
    return Path(raw) if raw else default


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_settings() -> ForgeSettings:
    """Read all environment variables and return a validated ForgeSettings instance.

    This function is called once at import time to populate the module-level
    ``settings`` singleton, but it can also be called again in tests to
    produce fresh instances with overridden environment variables.
    """
    provider_raw = _get_str("FORGE_LLM_PROVIDER", "openai")
    valid_providers = {"openai", "anthropic", "google", "openai_compatible"}
    if provider_raw not in valid_providers:
        raise ValueError(
            f"FORGE_LLM_PROVIDER={provider_raw!r} is not supported. "
            f"Choose one of: {sorted(valid_providers)}"
        )

    log_level_raw = _get_str("FORGE_LOG_LEVEL", "INFO").upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
    if log_level_raw not in valid_levels:
        raise ValueError(
            f"FORGE_LOG_LEVEL={log_level_raw!r} is invalid. "
            f"Choose one of: {sorted(valid_levels)}"
        )

    log_format_raw = _get_str("FORGE_LOG_FORMAT", "text").lower()
    valid_formats = {"text", "json"}
    if log_format_raw not in valid_formats:
        raise ValueError(
            f"FORGE_LOG_FORMAT={log_format_raw!r} is invalid. "
            f"Choose one of: {sorted(valid_formats)}"
        )

    return ForgeSettings(
        llm_provider=provider_raw,  # type: ignore[arg-type]
        llm_model=_get_str("FORGE_LLM_MODEL", "gpt-4o"),
        llm_api_key=_get_str("FORGE_LLM_API_KEY", ""),
        llm_base_url=_get_str("FORGE_LLM_BASE_URL", ""),
        max_tool_calls=_get_int("FORGE_MAX_TOOL_CALLS", 50),
        max_llm_turns=_get_int("FORGE_MAX_LLM_TURNS", 30),
        workspace_root=_get_path("FORGE_WORKSPACE_ROOT", Path.cwd()),
        log_level=log_level_raw,  # type: ignore[arg-type]
        log_format=log_format_raw,  # type: ignore[arg-type]
        log_dir=_get_path("FORGE_LOG_DIR", Path("logs")),
    )


# ---------------------------------------------------------------------------
# Module-level singleton — import this in other modules
# ---------------------------------------------------------------------------

settings: ForgeSettings = load_settings()
