"""
Tests for forge.llm abstractions and the Gemini provider adapter.

No real Gemini API call is made: a fake client is injected (or
``google.genai.Client`` is monkeypatched).  The adapter still builds real
``google.genai.types`` request objects, so the translation code is exercised
for real.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.llm import (
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    get_provider,
)
from forge.providers.gemini import GeminiProvider


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
# Fakes for the Gemini SDK response shape (duck-typed; adapter uses getattr)
# ---------------------------------------------------------------------------


def _finish(name="STOP"):
    return SimpleNamespace(name=name)


def _text_part(text):
    return SimpleNamespace(text=text, function_call=None)


def _fc_part(name, args, call_id="call_abc"):
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(id=call_id, name=name, args=args),
    )


def _usage(**kw):
    fields = {
        "prompt_token_count": None,
        "candidates_token_count": None,
        "total_token_count": None,
        "cached_content_token_count": None,
        "thoughts_token_count": None,
    }
    fields.update(kw)
    ns = SimpleNamespace(**fields)
    ns.model_dump = lambda exclude_none=True: {  # noqa: ARG005
        k: v for k, v in fields.items() if v is not None or not exclude_none
    }
    return ns


def _response(parts, *, finish="STOP", usage=None, model_version="gemini-3.6-flash"):
    candidate = SimpleNamespace(
        finish_reason=_finish(finish),
        content=SimpleNamespace(parts=parts),
    )
    return SimpleNamespace(
        candidates=[candidate],
        usage_metadata=usage,
        model_version=model_version,
    )


class FakeModels:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._raises is not None:
            raise self._raises
        return self._response


class FakeClient:
    def __init__(self, response=None, raises=None):
        self.models = FakeModels(response=response, raises=raises)


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


class TestRequestTranslation:
    def test_system_and_user_messages_translated(self):
        client = FakeClient(response=_response([_text_part("hi")]))
        provider = GeminiProvider(api_key="x", client=client, model="gemini-3.6-flash")

        provider.complete(
            [
                Message(role="system", content="You are FORGE."),
                Message(role="user", content="Say hi."),
            ]
        )

        call = client.models.calls[0]
        assert call["model"] == "gemini-3.6-flash"
        # system prompt goes into config.system_instruction, not contents
        assert call["config"].system_instruction == "You are FORGE."
        assert len(call["contents"]) == 1
        assert call["contents"][0].role == "user"

    def test_temperature_and_max_tokens_passed_through(self):
        client = FakeClient(response=_response([_text_part("ok")]))
        provider = GeminiProvider(api_key="x", client=client)
        provider.complete(
            [Message(role="user", content="hi")],
            max_tokens=256,
            temperature=0.7,
        )
        cfg = client.models.calls[0]["config"]
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 256

    def test_tool_schemas_translated_to_gemini_tool(self):
        client = FakeClient(response=_response([_text_part("ok")]))
        provider = GeminiProvider(api_key="x", client=client)
        schema = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        provider.complete([Message(role="user", content="read x")], tools=[schema])
        cfg = client.models.calls[0]["config"]
        assert cfg.tools is not None and len(cfg.tools) == 1
        decls = cfg.tools[0].function_declarations
        assert decls[0].name == "read_file"

    def test_assistant_tool_call_and_tool_result_roundtrip(self):
        client = FakeClient(response=_response([_text_part("done")]))
        provider = GeminiProvider(api_key="x", client=client)
        provider.complete(
            [
                Message(role="user", content="do it"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a"})],
                ),
                Message(role="tool", content="file contents", tool_call_id="c1", name="read_file"),
            ]
        )
        contents = client.models.calls[0]["contents"]
        assert [c.role for c in contents] == ["user", "model", "user"]


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


class TestResponseTranslation:
    def test_plain_text_response(self):
        client = FakeClient(response=_response([_text_part("hello world")]))
        provider = GeminiProvider(api_key="x", client=client)
        resp = provider.complete([Message(role="user", content="hi")])
        assert isinstance(resp, LLMResponse)
        assert resp.content == "hello world"
        assert resp.stop_reason == "stop"
        assert resp.has_tool_calls is False
        assert resp.model == "gemini-3.6-flash"

    def test_function_call_response(self):
        client = FakeClient(
            response=_response(
                [_fc_part("write_file", {"path": "a.py", "content": "x"}, "call_1")],
                finish="STOP",
            )
        )
        provider = GeminiProvider(api_key="x", client=client)
        resp = provider.complete([Message(role="user", content="make a.py")])
        assert resp.has_tool_calls is True
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "write_file"
        assert tc.arguments == {"path": "a.py", "content": "x"}
        assert tc.id == "call_1"
        assert resp.stop_reason == "tool_use"

    def test_mixed_text_and_function_call(self):
        client = FakeClient(
            response=_response(
                [_text_part("I will create it."), _fc_part("write_file", {"path": "a"})]
            )
        )
        provider = GeminiProvider(api_key="x", client=client)
        resp = provider.complete([Message(role="user", content="go")])
        assert resp.content == "I will create it."
        assert resp.tool_calls[0].name == "write_file"

    def test_usage_metadata_mapped(self):
        usage = _usage(
            prompt_token_count=1200,
            candidates_token_count=340,
            total_token_count=1540,
            cached_content_token_count=800,
            thoughts_token_count=50,
        )
        client = FakeClient(response=_response([_text_part("ok")], usage=usage))
        provider = GeminiProvider(api_key="x", client=client)
        resp = provider.complete([Message(role="user", content="hi")])
        assert resp.input_tokens == 1200
        assert resp.output_tokens == 340
        assert resp.total_tokens == 1540
        assert resp.cached_input_tokens == 800
        assert resp.reasoning_tokens == 50
        assert resp.raw_usage["prompt_token_count"] == 1200

    def test_missing_usage_metadata_is_none_not_zero(self):
        client = FakeClient(response=_response([_text_part("ok")], usage=None))
        provider = GeminiProvider(api_key="x", client=client)
        resp = provider.complete([Message(role="user", content="hi")])
        assert resp.input_tokens is None
        assert resp.output_tokens is None
        assert resp.total_tokens is None
        assert resp.cached_input_tokens is None
        assert resp.reasoning_tokens is None
        assert resp.resolved_total_tokens is None

    def test_partial_usage_metadata(self):
        # Provider reports prompt tokens only.
        usage = _usage(prompt_token_count=500)
        client = FakeClient(response=_response([_text_part("ok")], usage=usage))
        provider = GeminiProvider(api_key="x", client=client)
        resp = provider.complete([Message(role="user", content="hi")])
        assert resp.input_tokens == 500
        assert resp.output_tokens is None
        assert resp.resolved_total_tokens is None

    def test_max_tokens_finish_reason_maps_to_length(self):
        client = FakeClient(response=_response([_text_part("truncated")], finish="MAX_TOKENS"))
        provider = GeminiProvider(api_key="x", client=client)
        resp = provider.complete([Message(role="user", content="hi")])
        assert resp.stop_reason == "length"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_provider_exception_becomes_llm_error(self):
        client = FakeClient(raises=RuntimeError("connection reset"))
        provider = GeminiProvider(api_key="x", client=client)
        with pytest.raises(LLMError, match="Gemini request failed"):
            provider.complete([Message(role="user", content="hi")])

    def test_api_error_becomes_llm_error(self):
        from google.genai import errors as genai_errors

        err = genai_errors.APIError.__new__(genai_errors.APIError)
        Exception.__init__(err, "429 rate limited")
        client = FakeClient(raises=err)
        provider = GeminiProvider(api_key="x", client=client)
        with pytest.raises(LLMError, match="Gemini API error"):
            provider.complete([Message(role="user", content="hi")])


# ---------------------------------------------------------------------------
# API key / configuration handling
# ---------------------------------------------------------------------------


class TestApiKeyHandling:
    def test_missing_api_key_raises_valueerror(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        provider = GeminiProvider(api_key=None)  # no injected client
        with pytest.raises(ValueError, match="No Gemini API key"):
            provider.complete([Message(role="user", content="hi")])

    def test_api_key_from_env_builds_client(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "env-key-123")
        built = {}

        class _FakeGenaiClient:
            def __init__(self, *, api_key):
                built["api_key"] = api_key
                self.models = FakeModels(response=_response([_text_part("hi")]))

        monkeypatch.setattr("google.genai.Client", _FakeGenaiClient)
        provider = GeminiProvider(api_key=None)
        resp = provider.complete([Message(role="user", content="hi")])
        assert built["api_key"] == "env-key-123"
        assert resp.content == "hi"

    def test_explicit_api_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        built = {}

        class _FakeGenaiClient:
            def __init__(self, *, api_key):
                built["api_key"] = api_key
                self.models = FakeModels(response=_response([_text_part("hi")]))

        monkeypatch.setattr("google.genai.Client", _FakeGenaiClient)
        provider = GeminiProvider(api_key="explicit-key")
        provider.complete([Message(role="user", content="hi")])
        assert built["api_key"] == "explicit-key"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetProvider:
    def test_returns_gemini_provider(self):
        settings = SimpleNamespace(
            llm_provider="gemini", llm_model="gemini-3.6-flash", llm_api_key="k"
        )
        provider = get_provider(settings_override=settings)
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"
        assert provider.model_name == "gemini-3.6-flash"

    def test_unimplemented_provider_raises(self):
        settings = SimpleNamespace(
            llm_provider="openai", llm_model="gpt-4o", llm_api_key="k"
        )
        with pytest.raises(NotImplementedError, match="openai"):
            get_provider(settings_override=settings)
