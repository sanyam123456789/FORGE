"""
Tests for forge.providers.openai_compat.OpenAICompatibleProvider and the
opencode_go / openai_compatible presets built by forge.llm.get_provider().

Every test uses a fake transport (or a monkeypatched urlopen): no network,
no real API key, and time.sleep is always mocked.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import re
import urllib.error
from types import SimpleNamespace

import pytest

from forge.context import ConversationContext
from forge.llm import LLMError, Message, ToolCall, get_provider
from forge.providers import openai_compat
from forge.providers.openai_compat import (
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_DEFAULT_MODEL,
    OpenAICompatibleProvider,
)

SECRET = "sk-test-secret-key-0123456789"


class FakeTransport:
    """Records each request and replays scripted (status, headers, body)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": copy.deepcopy(payload),
                "timeout": timeout,
            }
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(message, *, usage=None, finish="stop", model="deepseek-v4.1-flash"):
    body = {"choices": [{"message": message, "finish_reason": finish}], "model": model}
    if usage is not None:
        body["usage"] = usage
    return 200, {"Content-Type": "application/json"}, body


def _err(status, message="boom", headers=None):
    return status, headers or {}, {"error": {"message": message}}


def _provider(transport, **kwargs):
    params = {
        "base_url": "https://api.example.test/v1/",
        "api_key": SECRET,
        "model": "deepseek-v4.1-flash",
        "transport": transport,
    }
    params.update(kwargs)
    return OpenAICompatibleProvider(**params)


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(openai_compat.time, "sleep", lambda s: recorded.append(s))
    return recorded


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequest:
    def test_request_body_shape(self, sleeps):
        transport = FakeTransport([_ok({"role": "assistant", "content": "done"})])
        provider = _provider(transport)
        history = [
            Message(role="system", content="You are FORGE."),
            Message(role="user", content="Fix the bug."),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_a", name="read_file", arguments={"path": "a.py"})],
            ),
            Message(role="tool", content="print(1)\n", tool_call_id="call_a", name="read_file"),
        ]
        provider.complete(history, tools=TOOLS, max_tokens=512, temperature=0.3)

        call = transport.calls[0]
        assert call["url"] == "https://api.example.test/v1/chat/completions"
        assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
        assert call["headers"]["Content-Type"] == "application/json"
        assert call["timeout"] == 120.0
        payload = call["payload"]
        assert payload["model"] == "deepseek-v4.1-flash"
        assert payload["tools"] == TOOLS
        assert payload["tool_choice"] == "auto"
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 512
        assert payload["messages"] == [
            {"role": "system", "content": "You are FORGE."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "print(1)\n"},
        ]

    def test_no_tools_omits_tools_and_tool_choice(self, sleeps):
        transport = FakeTransport([_ok({"content": "hi"})])
        _provider(transport).complete([Message(role="user", content="hi")])
        payload = transport.calls[0]["payload"]
        assert "tools" not in payload and "tool_choice" not in payload

    def test_assistant_without_tool_calls_omits_key(self, sleeps):
        transport = FakeTransport([_ok({"content": "ok"})])
        _provider(transport).complete(
            [Message(role="user", content="q"), Message(role="assistant", content="a")]
        )
        assert transport.calls[0]["payload"]["messages"][1] == {
            "role": "assistant",
            "content": "a",
        }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponse:
    def test_plain_text(self, sleeps):
        transport = FakeTransport([_ok({"role": "assistant", "content": "All done."})])
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert resp.content == "All done."
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"
        assert resp.model == "deepseek-v4.1-flash"

    def test_tool_call_round_trip(self, sleeps):
        first = _ok(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "app.py"}'},
                    }
                ],
            },
            finish="tool_calls",
        )
        transport = FakeTransport([first, _ok({"content": "done"})])
        provider = _provider(transport)
        history = [Message(role="user", content="read app.py")]
        resp = provider.complete(history, tools=TOOLS)

        assert resp.stop_reason == "tool_use"
        assert resp.content == ""
        assert [(c.id, c.name, c.arguments) for c in resp.tool_calls] == [
            ("call_x1", "read_file", {"path": "app.py"})
        ]

        history += [
            Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls),
            Message(role="tool", content="x = 1\n", tool_call_id="call_x1", name="read_file"),
        ]
        provider.complete(history, tools=TOOLS)
        sent = transport.calls[1]["payload"]["messages"]
        assert sent[1]["tool_calls"][0]["id"] == "call_x1"
        assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {"path": "app.py"}
        assert sent[2] == {"role": "tool", "tool_call_id": "call_x1", "content": "x = 1\n"}

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ('{"path": "a.py"', '{"path": "a.py"'),  # truncated JSON -> raw kept
            ("[1, 2]", "[1, 2]"),  # valid JSON but not an object -> raw kept
            ("", {}),
            (None, {}),
        ],
    )
    def test_malformed_arguments_kept_raw(self, sleeps, raw, expected):
        msg = {"tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": raw}}]}
        transport = FakeTransport([_ok(msg, finish="tool_calls")])
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert resp.tool_calls[0].arguments == expected

    def test_raw_arguments_echoed_back_unchanged(self, sleeps):
        transport = FakeTransport([_ok({"content": "ok"})])
        call = ToolCall(id="c1", name="read_file", arguments='{"path": ')  # type: ignore[arg-type]
        _provider(transport).complete(
            [Message(role="user", content="q"), Message(role="assistant", tool_calls=[call])]
        )
        sent = transport.calls[0]["payload"]["messages"][1]["tool_calls"][0]
        assert sent["function"]["arguments"] == '{"path": '

    def test_missing_ids_get_unique_synthesized_ids(self, sleeps):
        no_id = {"tool_calls": [{"function": {"name": "list_directory", "arguments": "{}"}}]}
        transport = FakeTransport([_ok(no_id, finish="tool_calls"), _ok(no_id, finish="tool_calls")])
        provider = _provider(transport)
        ids = [provider.complete([Message(role="user", content="q")]).tool_calls[0].id for _ in range(2)]
        assert all(re.fullmatch(r"call_[0-9a-f]{8}", i) for i in ids)
        assert ids[0] != ids[1]
        assert "call_0" not in ids

    @pytest.mark.parametrize(
        "finish, expected",
        [("stop", "stop"), ("length", "length"), ("tool_calls", "tool_use"),
         ("content_filter", "safety"), ("WEIRD_Reason", "weird_reason")],
    )
    def test_finish_reason_mapping(self, sleeps, finish, expected):
        transport = FakeTransport([_ok({"content": "x"}, finish=finish)])
        assert _provider(transport).complete([Message(role="user", content="q")]).stop_reason == expected

    def test_no_choices_is_non_retryable_error(self, sleeps):
        transport = FakeTransport([(200, {}, {"choices": []})])
        with pytest.raises(LLMError, match="no choices") as info:
            _provider(transport).complete([Message(role="user", content="q")])
        assert info.value.retryable is False
        assert sleeps == []

    def test_content_parts_list_joined(self, sleeps):
        msg = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        transport = FakeTransport([_ok(msg)])
        assert _provider(transport).complete([Message(role="user", content="q")]).content == "ab"


# ---------------------------------------------------------------------------
# reasoning_content (DeepSeek)
# ---------------------------------------------------------------------------


class TestReasoningContent:
    def _tool_msg(self):
        return {
            "content": None,
            "reasoning_content": "I should read app.py first.",
            "tool_calls": [
                {"id": "call_r", "function": {"name": "read_file", "arguments": '{"path": "app.py"}'}}
            ],
        }

    def test_stored_and_echoed_back(self, sleeps):
        transport = FakeTransport([_ok(self._tool_msg(), finish="tool_calls"), _ok({"content": "done"})])
        provider = _provider(transport)
        history = [Message(role="user", content="q")]
        resp = provider.complete(history, tools=TOOLS)
        assert resp.content == ""  # never the final answer
        assert resp.tool_calls[0].provider_signature == {
            "reasoning_content": "I should read app.py first."
        }

        # AgentRuntime builds the assistant message exactly like this.
        assistant = Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls)
        history += [assistant, Message(role="tool", content="x", tool_call_id="call_r", name="read_file")]
        provider.complete(history, tools=TOOLS)
        sent_assistant = transport.calls[1]["payload"]["messages"][1]
        assert sent_assistant["reasoning_content"] == "I should read app.py first."

        ctx = ConversationContext()
        ctx.add_message(assistant)
        assert ctx.approximate_char_count == 0  # reasoning not counted

    def test_reasoning_from_message_metadata(self, sleeps):
        transport = FakeTransport([_ok({"content": "ok"})])
        assistant = Message(
            role="assistant",
            tool_calls=[ToolCall(id="c", name="read_file", arguments={})],
            metadata={"reasoning_content": "from metadata"},
        )
        _provider(transport).complete([Message(role="user", content="q"), assistant])
        assert transport.calls[0]["payload"]["messages"][1]["reasoning_content"] == "from metadata"

    def test_no_reasoning_key_when_absent(self, sleeps):
        transport = FakeTransport([_ok({"content": "ok"})])
        assistant = Message(role="assistant", tool_calls=[ToolCall(id="c", name="read_file", arguments={})])
        _provider(transport).complete([Message(role="user", content="q"), assistant])
        assert "reasoning_content" not in transport.calls[0]["payload"]["messages"][1]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class TestUsage:
    def test_full_usage_mapping(self, sleeps):
        usage = {
            "prompt_tokens": 1200,
            "completion_tokens": 80,
            "total_tokens": 1280,
            "prompt_tokens_details": {"cached_tokens": 1024},
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
        transport = FakeTransport([_ok({"content": "x"}, usage=usage, model="deepseek-v4.1-flash-0901")])
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert (resp.input_tokens, resp.output_tokens, resp.total_tokens) == (1200, 80, 1280)
        assert resp.cached_input_tokens == 1024
        assert resp.reasoning_tokens == 30
        assert resp.raw_usage == usage
        assert resp.model == "deepseek-v4.1-flash-0901"

    def test_prompt_cache_hit_tokens_accepted(self, sleeps):
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 8}
        transport = FakeTransport([_ok({"content": "x"}, usage=usage)])
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert resp.cached_input_tokens == 8

    def test_missing_fields_are_none_not_zero(self, sleeps):
        transport = FakeTransport([_ok({"content": "x"}, usage={"prompt_tokens": 5})])
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert resp.input_tokens == 5
        assert resp.output_tokens is None
        assert resp.total_tokens is None
        assert resp.cached_input_tokens is None
        assert resp.reasoning_tokens is None

    def test_no_usage_at_all(self, sleeps):
        body = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        transport = FakeTransport([(200, {}, body)])
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert resp.input_tokens is None and resp.output_tokens is None
        assert resp.raw_usage is None
        assert resp.model == "deepseek-v4.1-flash"  # falls back to configured model


# ---------------------------------------------------------------------------
# Errors and retries
# ---------------------------------------------------------------------------


class TestErrors:
    def test_429_with_retry_after_is_retried_and_honored(self, sleeps):
        transport = FakeTransport(
            [_err(429, "slow down", headers={"Retry-After": "7"}), _ok({"content": "ok"})]
        )
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert resp.content == "ok"
        assert sleeps == [7.0]

    def test_503_retried_with_exponential_backoff(self, sleeps):
        transport = FakeTransport([_err(503), _err(503), _ok({"content": "ok"})])
        resp = _provider(transport).complete([Message(role="user", content="q")])
        assert resp.content == "ok"
        assert sleeps == [2.0, 4.0]

    def test_retries_exhausted_raises_retryable(self, sleeps):
        transport = FakeTransport([_err(503)] * 3)
        with pytest.raises(LLMError) as info:
            _provider(transport, retry_max=2).complete([Message(role="user", content="q")])
        assert info.value.retryable is True
        assert len(transport.calls) == 3
        assert sleeps == [2.0, 4.0]

    @pytest.mark.parametrize("status", [408, 500, 502, 599])
    def test_other_transient_statuses_are_retryable(self, sleeps, status):
        transport = FakeTransport([_err(status)])
        with pytest.raises(LLMError) as info:
            _provider(transport, retry_max=0).complete([Message(role="user", content="q")])
        assert info.value.retryable is True

    @pytest.mark.parametrize(
        "exc", [urllib.error.URLError("connection refused"), TimeoutError("timed out")]
    )
    def test_connection_errors_and_timeouts_are_retryable(self, sleeps, exc):
        transport = FakeTransport([exc, _ok({"content": "ok"})])
        assert _provider(transport).complete([Message(role="user", content="q")]).content == "ok"
        assert sleeps == [2.0]

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_errors_not_retryable(self, sleeps, status):
        transport = FakeTransport([_err(status, "invalid key")])
        with pytest.raises(LLMError) as info:
            _provider(transport).complete([Message(role="user", content="q")])
        assert info.value.retryable is False
        assert "check FORGE_LLM_API_KEY / OPENCODE_API_KEY" in str(info.value)
        assert len(transport.calls) == 1 and sleeps == []

    def test_404_names_base_url_and_model(self, sleeps):
        transport = FakeTransport([_err(404, "model not found")])
        with pytest.raises(LLMError) as info:
            _provider(transport).complete([Message(role="user", content="q")])
        assert info.value.retryable is False
        assert "endpoint or model not found — check FORGE_LLM_BASE_URL and FORGE_LLM_MODEL" in str(info.value)

    def test_other_4xx_includes_provider_text_with_key_redacted(self, sleeps):
        transport = FakeTransport([_err(400, f"bad request for key {SECRET}")])
        with pytest.raises(LLMError) as info:
            _provider(transport).complete([Message(role="user", content="q")])
        assert info.value.retryable is False
        assert "HTTP 400" in str(info.value)
        assert "bad request for key ***" in str(info.value)
        assert SECRET not in str(info.value)
        assert sleeps == []


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_missing_base_url(self):
        with pytest.raises(ValueError, match="FORGE_LLM_BASE_URL"):
            OpenAICompatibleProvider(base_url="", api_key="k", model="m").preflight()

    def test_missing_model(self):
        with pytest.raises(ValueError, match="FORGE_LLM_MODEL"):
            OpenAICompatibleProvider(base_url="http://x/v1", api_key="k", model="").preflight()

    def test_missing_api_key(self):
        with pytest.raises(ValueError, match="FORGE_LLM_API_KEY"):
            OpenAICompatibleProvider(base_url="http://x/v1", api_key="", model="m").preflight()

    def test_complete_validates_before_any_request(self):
        transport = FakeTransport([])
        provider = OpenAICompatibleProvider(base_url="http://x/v1", api_key="", model="m", transport=transport)
        with pytest.raises(ValueError):
            provider.complete([Message(role="user", content="q")])
        assert transport.calls == []

    def test_valid_config_passes(self):
        OpenAICompatibleProvider(base_url="http://x/v1", api_key="k", model="m").preflight()

    def test_names(self):
        p = OpenAICompatibleProvider(base_url="http://x/v1", api_key="k", model="m", provider_label="lbl")
        assert p.provider_name == "lbl"
        assert p.model_name == "m"


# ---------------------------------------------------------------------------
# Presets via get_provider()
# ---------------------------------------------------------------------------


def _settings(**kwargs):
    base = {
        "llm_provider": "opencode_go",
        "llm_model": "",
        "llm_api_key": "",
        "llm_base_url": "",
        "provider_order": (),
        "provider_retry_max": 3,
        "llm_request_timeout_s": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestPresets:
    def test_opencode_go_defaults(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        provider = get_provider(settings_override=_settings(llm_api_key="k"))
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider_name == "opencode_go"
        assert provider.model_name == OPENCODE_GO_DEFAULT_MODEL == "deepseek-v4.1-flash"
        assert provider._base_url == OPENCODE_GO_BASE_URL == "https://opencode.ai/zen/go/v1"
        assert provider._timeout == 120.0
        provider.preflight()

    def test_opencode_go_falls_back_to_opencode_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "oc-key-from-env")
        provider = get_provider(settings_override=_settings())
        assert provider._api_key == "oc-key-from-env"
        provider.preflight()

    def test_forge_llm_api_key_preferred(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "oc-key-from-env")
        provider = get_provider(settings_override=_settings(llm_api_key="forge-key"))
        assert provider._api_key == "forge-key"

    def test_opencode_go_missing_key_names_both_variables(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        provider = get_provider(settings_override=_settings())
        with pytest.raises(ValueError, match="FORGE_LLM_API_KEY.*OPENCODE_API_KEY"):
            provider.preflight()

    def test_opencode_go_overrides_and_runtime_settings(self, monkeypatch):
        provider = get_provider(
            settings_override=_settings(
                llm_api_key="k",
                llm_model="other-model",
                llm_base_url="https://proxy.example/v1",
                llm_request_timeout_s=30.0,
                provider_retry_max=5,
            )
        )
        assert provider.model_name == "other-model"
        assert provider._base_url == "https://proxy.example/v1"
        assert provider._timeout == 30.0
        assert provider._retry_max == 5

    def test_openai_compatible_requires_base_url(self):
        provider = get_provider(
            settings_override=_settings(llm_provider="openai_compatible", llm_api_key="k", llm_model="m")
        )
        assert provider.provider_name == "openai_compatible"
        with pytest.raises(ValueError, match="FORGE_LLM_BASE_URL"):
            provider.preflight()

    def test_openai_compatible_with_base_url(self):
        provider = get_provider(
            settings_override=_settings(
                llm_provider="openai_compatible",
                llm_api_key="ollama",
                llm_model="qwen2.5-coder:7b",
                llm_base_url="http://localhost:11434/v1",
            )
        )
        provider.preflight()
        assert provider._base_url == "http://localhost:11434/v1"

    def test_config_defaults(self, clean_env):
        from forge.config import load_settings

        s = load_settings()
        assert s.llm_provider == "opencode_go"
        assert s.llm_model == "deepseek-v4.1-flash"

    @pytest.mark.parametrize("name", ["gemini", "groq", "cerebras", "openai", "anthropic", "google"])
    def test_removed_providers_rejected(self, clean_env, monkeypatch, name):
        from forge.config import load_settings

        monkeypatch.setenv("FORGE_LLM_PROVIDER", name)
        with pytest.raises(ValueError, match="FORGE_LLM_PROVIDER"):
            load_settings()


# ---------------------------------------------------------------------------
# Default urllib transport (urlopen monkeypatched: still offline)
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, status, body, headers):
        self.status = status
        self._body = body
        self.headers = SimpleNamespace(items=lambda: list(headers.items()))

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestUrllibTransport:
    def test_success_parses_json(self, monkeypatch):
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            seen["body"] = json.loads(request.data)
            seen["timeout"] = timeout
            return _FakeHTTPResponse(200, b'{"choices": []}', {"X-A": "1"})

        monkeypatch.setattr(openai_compat.urllib.request, "urlopen", fake_urlopen)
        status, headers, body = openai_compat.urllib_transport(
            "https://x/v1/chat/completions", {"Authorization": "Bearer k"}, {"a": 1}, 9.0
        )
        assert (status, headers, body) == (200, {"X-A": "1"}, {"choices": []})
        assert seen == {
            "url": "https://x/v1/chat/completions",
            "auth": "Bearer k",
            "body": {"a": 1},
            "timeout": 9.0,
        }

    def test_http_error_returns_status_and_body(self, monkeypatch):
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many", {"Retry-After": "3"},
                io.BytesIO(b'{"error": {"message": "slow"}}'),
            )

        monkeypatch.setattr(openai_compat.urllib.request, "urlopen", fake_urlopen)
        status, headers, body = openai_compat.urllib_transport("https://x", {}, {}, 1.0)
        assert status == 429
        assert headers["Retry-After"] == "3"
        assert body == {"error": {"message": "slow"}}


# ---------------------------------------------------------------------------
# The API key never leaks
# ---------------------------------------------------------------------------


def test_api_key_never_in_errors_repr_or_logs(sleeps, caplog):
    provider_transport = FakeTransport(
        [
            urllib.error.URLError(f"proxy rejected {SECRET}"),
            _err(503, f"overloaded {SECRET}"),
            _err(400, f"bad key {SECRET}"),
        ]
    )
    provider = _provider(provider_transport, retry_max=3)
    messages = []
    with caplog.at_level(logging.DEBUG, logger="forge"):
        with pytest.raises(LLMError) as info:
            provider.complete([Message(role="user", content="q")])
        messages.append(str(info.value))
        for status in (401, 404):
            p = _provider(FakeTransport([_err(status, f"nope {SECRET}")]))
            with pytest.raises(LLMError) as info2:
                p.complete([Message(role="user", content="q")])
            messages.append(str(info2.value))
            messages.append(repr(info2.value))

    assert len(provider_transport.calls) == 3
    assert all(SECRET not in m for m in messages)
    assert SECRET not in repr(provider)
    assert "api_key=***" in repr(provider)
    assert SECRET not in caplog.text
    assert "retrying" in caplog.text  # the retry path did log
