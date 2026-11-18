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

LLMProvider = Literal["opencode_go", "openai_compatible"]

#: Provider names accepted by FORGE_LLM_PROVIDER and FORGE_PROVIDER_ORDER.
SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"opencode_go", "openai_compatible"})
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
    llm_provider: LLMProvider = field(default="opencode_go")
    llm_model: str = field(default="deepseek-v4.1-flash")
    llm_api_key: str = field(default="")
    llm_base_url: str = field(default="")

    # --- Multi-provider fallback (opt-in) ---
    # Empty tuple (FORGE_PROVIDER_ORDER unset) preserves the exact single-
    # provider behavior of every prior FORGE step: get_provider() returns
    # one adapter for llm_provider, no fallback. Set FORGE_PROVIDER_ORDER
    # (e.g. "opencode_go,openai_compatible") to opt into FallbackProvider.
    provider_order: tuple[str, ...] = field(default_factory=tuple)
    # Bounded retry count for the adapter's own transient-error backoff.
    provider_retry_max: int = field(default=3)
    # None = use the provider adapter's default timeout (120 s).
    llm_request_timeout_s: float | None = field(default=None)

    # --- LLM generation ---
    llm_temperature: float = field(default=0.0)
    # 0 = no per-call output cap (max_tokens is not sent to the provider).
    llm_max_output_tokens: int = field(default=0)

    # --- Runtime ---
    # Loop guards. 0 = unlimited.
    max_tool_calls: int = field(default=50)
    max_llm_turns: int = field(default=30)
    workspace_root: Path = field(default_factory=Path.cwd)

    # --- Logging ---
    log_level: LogLevel = field(default="INFO")
    log_format: LogFormat = field(default="text")
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # --- Shell security ---
    # Extra env var names passed through to run_shell children even if they
    # look like secrets (FORGE_SHELL_ENV_ALLOW). FORGE_* is never passed.
    shell_env_allow: tuple[str, ...] = field(default_factory=tuple)

    # --- Token savers ---
    # Prepend a compact list of workspace files to the task (FORGE_REPO_MAP),
    # so the agent does not spend turns exploring with list_directory.
    repo_map: bool = field(default=True)

    def __repr__(self) -> str:  # pragma: no cover
        """Redact secret fields from string representation."""
        key_hint = "***" if self.llm_api_key else "<not set>"
        return (
            f"ForgeSettings("
            f"llm_provider={self.llm_provider!r}, "
            f"llm_model={self.llm_model!r}, "
            f"llm_api_key={key_hint}, "
            f"llm_base_url={self.llm_base_url!r}, "
            f"provider_order={self.provider_order!r}, "
            f"provider_retry_max={self.provider_retry_max}, "
            f"llm_request_timeout_s={self.llm_request_timeout_s}, "
            f"llm_temperature={self.llm_temperature}, "
            f"llm_max_output_tokens={self.llm_max_output_tokens}, "
            f"max_tool_calls={self.max_tool_calls}, "
            f"max_llm_turns={self.max_llm_turns}, "
            f"workspace_root={self.workspace_root!r}, "
            f"log_level={self.log_level!r}, "
            f"log_format={self.log_format!r}, "
            f"shell_env_allow={self.shell_env_allow!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int, *, min_value: int | None = None) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"FORGE configuration error: {key}={raw!r} is not a valid integer."
        ) from exc
    if min_value is not None and value < min_value:
        raise ValueError(
            f"FORGE configuration error: {key}={value} is below the minimum "
            f"allowed value ({min_value})."
        )
    return value


def _get_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"FORGE configuration error: {key}={raw!r} is not a boolean "
        "(use true/false)."
    )


def _get_float(
    key: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"FORGE configuration error: {key}={raw!r} is not a valid number."
        ) from exc
    if min_value is not None and value < min_value:
        raise ValueError(
            f"FORGE configuration error: {key}={value} is below the minimum "
            f"allowed value ({min_value})."
        )
    if max_value is not None and value > max_value:
        raise ValueError(
            f"FORGE configuration error: {key}={value} is above the maximum "
            f"allowed value ({max_value})."
        )
    return value


def _get_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key, "").strip()
    return Path(raw) if raw else default


def _get_list(key: str) -> tuple[str, ...]:
    """Parse a comma-separated env var into a lowercase, whitespace-trimmed
    tuple. Empty/unset returns an empty tuple (the opt-in default)."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return ()
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _get_optional_float(
    key: str, *, min_value: float | None = None
) -> float | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"FORGE configuration error: {key}={raw!r} is not a valid number."
        ) from exc
    if min_value is not None and value < min_value:
        raise ValueError(
            f"FORGE configuration error: {key}={value} is below the minimum "
            f"allowed value ({min_value})."
        )
    return value


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_settings() -> ForgeSettings:
    """Read all environment variables and return a validated ForgeSettings instance.

    This function is called once at import time to populate the module-level
    ``settings`` singleton, but it can also be called again in tests to
    produce fresh instances with overridden environment variables.
    """
    removed_hint = (
        " (gemini, groq and cerebras were removed; see .env.example)"
    )
    provider_raw = _get_str("FORGE_LLM_PROVIDER", "opencode_go")
    if provider_raw not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"FORGE_LLM_PROVIDER={provider_raw!r} is not supported. "
            f"Choose one of: {sorted(SUPPORTED_PROVIDERS)}{removed_hint}"
        )

    provider_order = _get_list("FORGE_PROVIDER_ORDER")
    unknown = [p for p in provider_order if p not in SUPPORTED_PROVIDERS]
    if unknown:
        raise ValueError(
            f"FORGE_PROVIDER_ORDER contains unsupported provider(s) {unknown}. "
            f"Implemented adapters: {sorted(SUPPORTED_PROVIDERS)}{removed_hint}"
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
        llm_model=_get_str("FORGE_LLM_MODEL", "deepseek-v4.1-flash"),
        llm_api_key=_get_str("FORGE_LLM_API_KEY", ""),
        llm_base_url=_get_str("FORGE_LLM_BASE_URL", ""),
        provider_order=provider_order,
        provider_retry_max=_get_int("FORGE_PROVIDER_RETRY_MAX", 3, min_value=0),
        llm_request_timeout_s=_get_optional_float(
            "FORGE_LLM_REQUEST_TIMEOUT_S", min_value=0.0
        ),
        llm_temperature=_get_float(
            "FORGE_LLM_TEMPERATURE", 0.0, min_value=0.0, max_value=2.0
        ),
        llm_max_output_tokens=_get_int(
            "FORGE_LLM_MAX_OUTPUT_TOKENS", 0, min_value=0
        ),
        max_tool_calls=_get_int("FORGE_MAX_TOOL_CALLS", 50, min_value=0),
        max_llm_turns=_get_int("FORGE_MAX_LLM_TURNS", 30, min_value=0),
        workspace_root=_get_path("FORGE_WORKSPACE_ROOT", Path.cwd()),
        log_level=log_level_raw,  # type: ignore[arg-type]
        log_format=log_format_raw,  # type: ignore[arg-type]
        log_dir=_get_path("FORGE_LOG_DIR", Path("logs")),
        shell_env_allow=_get_list("FORGE_SHELL_ENV_ALLOW"),
        repo_map=_get_bool("FORGE_REPO_MAP", True),
    )


# ---------------------------------------------------------------------------
# Module-level singleton — import this in other modules
# ---------------------------------------------------------------------------

settings: ForgeSettings = load_settings()
