"""
Tests for forge.llm abstractions and the get_provider() factory.

Provider adapter behaviour (request/response translation, retries, errors)
is covered in tests/test_providers_openai_compat.py. No network calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.llm import (
    LLMResponse,
    ToolCall,
    get_provider,
)
from forge.providers.openai_compat import OpenAICompatibleProvider

# ---------------------------------------------------------------------------
# LLMResponse / vocabulary
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_defaults_are_empty_not_zero(self):
        r = LLMResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.input_tokens is None
        assert r.output_tokens is None
        assert r.has_tool_calls is False

    def test_resolved_total_prefers_provider_total(self):
        r = LLMResponse(input_tokens=10, output_tokens=5, total_tokens=42)
        assert r.resolved_total_tokens == 42

    def test_resolved_total_falls_back_to_sum(self):
        r = LLMResponse(input_tokens=10, output_tokens=5)
        assert r.resolved_total_tokens == 15

    def test_resolved_total_none_when_incomplete(self):
        assert LLMResponse(input_tokens=10).resolved_total_tokens is None
        assert LLMResponse().resolved_total_tokens is None

    def test_has_tool_calls(self):
        r = LLMResponse(tool_calls=[ToolCall(id="1", name="read_file", arguments={})])
        assert r.has_tool_calls is True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetProvider:
    def test_returns_opencode_go_provider(self):
        settings = SimpleNamespace(
            llm_provider="opencode_go", llm_model="deepseek-v4.1-flash", llm_api_key="k"
        )
        provider = get_provider(settings_override=settings)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider_name == "opencode_go"
        assert provider.model_name == "deepseek-v4.1-flash"

    def test_unimplemented_provider_raises(self):
        settings = SimpleNamespace(
            llm_provider="gemini", llm_model="gemini-3.6-flash", llm_api_key="k"
        )
        with pytest.raises(NotImplementedError, match="gemini"):
            get_provider(settings_override=settings)
