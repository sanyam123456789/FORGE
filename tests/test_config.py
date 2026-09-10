"""
Tests for forge.config — configuration loading and validation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from forge.config import ForgeSettings, load_settings


class TestDefaultSettings:
    """load_settings() with no environment overrides returns sensible defaults."""

    def test_default_provider(self, clean_env):
        s = load_settings()
        assert s.llm_provider == "gemini"

    def test_default_model(self, clean_env):
        s = load_settings()
        assert s.llm_model == "gemini-2.0-flash"

    def test_default_temperature(self, clean_env):
        s = load_settings()
        assert s.llm_temperature == 0.0

    def test_default_max_output_tokens(self, clean_env):
        s = load_settings()
        assert s.llm_max_output_tokens == 4096

    def test_api_key_default_empty(self, clean_env):
        s = load_settings()
        assert s.llm_api_key == ""

    def test_default_max_tool_calls(self, clean_env):
        s = load_settings()
        assert s.max_tool_calls == 50

    def test_default_max_llm_turns(self, clean_env):
        s = load_settings()
        assert s.max_llm_turns == 30

    def test_default_log_level(self, clean_env):
        s = load_settings()
        assert s.log_level == "INFO"

    def test_default_log_format(self, clean_env):
        s = load_settings()
        assert s.log_format == "text"

    def test_workspace_root_is_path(self, clean_env):
        s = load_settings()
        assert isinstance(s.workspace_root, Path)


class TestEnvOverrides:
    """load_settings() respects environment variable overrides."""

    def test_provider_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_PROVIDER", "anthropic")
        s = load_settings()
        assert s.llm_provider == "anthropic"

    def test_model_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_MODEL", "claude-3-5-sonnet-20241022")
        s = load_settings()
        assert s.llm_model == "claude-3-5-sonnet-20241022"

    def test_api_key_loaded(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_API_KEY", "sk-test-12345")
        s = load_settings()
        assert s.llm_api_key == "sk-test-12345"

    def test_max_tool_calls_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_MAX_TOOL_CALLS", "10")
        s = load_settings()
        assert s.max_tool_calls == 10

    def test_log_level_debug(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LOG_LEVEL", "DEBUG")
        s = load_settings()
        assert s.log_level == "DEBUG"

    def test_log_format_json(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LOG_FORMAT", "json")
        s = load_settings()
        assert s.log_format == "json"

    def test_workspace_root_custom(self, clean_env, monkeypatch, tmp_path):
        monkeypatch.setenv("FORGE_WORKSPACE_ROOT", str(tmp_path))
        s = load_settings()
        assert s.workspace_root == tmp_path

    def test_log_dir_custom(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LOG_DIR", "my_logs")
        s = load_settings()
        assert s.log_dir == Path("my_logs")


class TestValidation:
    """load_settings() raises ValueError for invalid values."""

    def test_invalid_provider(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_PROVIDER", "unknown_provider")
        with pytest.raises(ValueError, match="FORGE_LLM_PROVIDER"):
            load_settings()

    def test_invalid_log_level(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValueError, match="FORGE_LOG_LEVEL"):
            load_settings()

    def test_invalid_log_format(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LOG_FORMAT", "yaml")
        with pytest.raises(ValueError, match="FORGE_LOG_FORMAT"):
            load_settings()

    def test_invalid_max_tool_calls_type(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_MAX_TOOL_CALLS", "not_a_number")
        with pytest.raises(ValueError, match="FORGE_MAX_TOOL_CALLS"):
            load_settings()


class TestNumericBounds:
    """Count / limit settings reject invalid non-positive or out-of-range values."""

    def test_negative_max_tool_calls_rejected(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_MAX_TOOL_CALLS", "-5")
        with pytest.raises(ValueError, match="minimum"):
            load_settings()

    def test_zero_max_llm_turns_rejected(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_MAX_LLM_TURNS", "0")
        with pytest.raises(ValueError, match="minimum"):
            load_settings()

    def test_negative_max_output_tokens_rejected(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_MAX_OUTPUT_TOKENS", "-1")
        with pytest.raises(ValueError, match="minimum"):
            load_settings()

    def test_temperature_out_of_range_rejected(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_TEMPERATURE", "3.5")
        with pytest.raises(ValueError, match="maximum"):
            load_settings()

    def test_temperature_negative_rejected(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_TEMPERATURE", "-0.1")
        with pytest.raises(ValueError, match="minimum"):
            load_settings()

    def test_temperature_non_numeric_rejected(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_TEMPERATURE", "hot")
        with pytest.raises(ValueError, match="FORGE_LLM_TEMPERATURE"):
            load_settings()

    def test_valid_bounds_accepted(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_MAX_TOOL_CALLS", "1")
        monkeypatch.setenv("FORGE_LLM_TEMPERATURE", "2.0")
        s = load_settings()
        assert s.max_tool_calls == 1
        assert s.llm_temperature == 2.0


class TestSecretRedaction:
    """API keys must not appear in repr()."""

    def test_repr_redacts_api_key(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORGE_LLM_API_KEY", "sk-super-secret-key")
        s = load_settings()
        representation = repr(s)
        assert "sk-super-secret-key" not in representation
        assert "***" in representation

    def test_repr_no_key_shows_not_set(self, clean_env):
        s = load_settings()
        assert "<not set>" in repr(s)


class TestSettingsImmutability:
    """ForgeSettings is a frozen dataclass."""

    def test_cannot_mutate_settings(self, clean_env):
        s = load_settings()
        with pytest.raises((AttributeError, TypeError)):
            s.llm_model = "modified"  # type: ignore[misc]
