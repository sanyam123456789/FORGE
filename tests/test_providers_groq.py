"""
Tests for forge.providers.groq.

No real Groq API call is made: a fake client mimicking the groq SDK's
response shape is injected. Real groq SDK exception classes are used
(instantiated directly) so the adapter's classification of retryable vs.
non-retryable errors is exercised against the real exception hierarchy.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from forge.llm import LLMError, Message, ToolCall
from forge.providers.groq import GroqProvider


# ---------------------------------------------------------------------------
# Fakes for the Groq SDK response shape (duck-typed; adapter uses getattr)
# ---------------------------------------------------------------------------


def _tool_call(name, args, call_id="call_abc"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _usage(**kw):
    fields = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    fields.update(kw)
    ns = SimpleNamespace(**fields)
    ns.model_dump = lambda exclude_none=True: {  # noqa: ARG005
        k: v for k, v in fields.items() if v is not None or not exclude_none
    }
    return ns


def _response(message, *, finish_reason="stop", usage=None, model="llama-x"):
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


class FakeCompletions:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises_seq = list(raises) if isinstance(raises, list) else (
            [raises] if raises is not None else None
        )
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises_seq:
            raise self._raises_seq.pop(0)
        return self._response


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, response=None, raises=None):
        self.completions = FakeCompletions(response=response, raises=raises)
        self.chat = FakeChat(self.completions)


def _http_error(cls, status, *, retry_after=None, message="err"):
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(
        status, request=httpx.Request("POST", "https://api.groq.com/x"), headers=headers
    )
    return cls(message, response=resp, body={"error": {"message": message}})


# ---------------------------------------------------------------------------
# Request / response translation
# ---------------------------------------------------------------------------


class TestTranslation:
    def test_text_response(self):
        client = FakeClient(response=_response(_message("hello")))
        provider = GroqProvider(api_key="x", model="llama-3.3-70b-versatile", client=client)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.content == "hello"
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"
        assert resp.model == "llama-x"

    def test_tool_call_response_normalised(self):
        client = FakeClient(
            response=_response(
                _message(tool_calls=[_tool_call("write_file", {"path": "a.py"})]),
                finish_reason="tool_calls",
            )
        )
        provider = GroqProvider(api_key="x", model="m", client=client)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.stop_reason == "tool_use"
        assert len(resp.tool_calls) == 1
        call = resp.tool_calls[0]
        assert call.id == "call_abc"
        assert call.name == "write_file"
        assert call.arguments == {"path": "a.py"}

    def test_system_user_assistant_tool_roles_translated(self):
        client = FakeClient(response=_response(_message("done")))
        provider = GroqProvider(api_key="x", model="m", client=client)

        provider.complete(
            [
                Message(role="system", content="sys"),
                Message(role="user", content="do it"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "a"})],
                ),
                Message(role="tool", content="ok", tool_call_id="c1", name="write_file"),
            ]
        )
        sent = client.completions.calls[0]["messages"]
        assert sent[0] == {"role": "system", "content": "sys"}
        assert sent[1] == {"role": "user", "content": "do it"}
        assert sent[2]["role"] == "assistant"
        assert sent[2]["tool_calls"][0]["id"] == "c1"
        assert sent[2]["tool_calls"][0]["function"]["name"] == "write_file"
        assert json.loads(sent[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
        assert sent[3] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}

    def test_usage_mapped_when_present(self):
        usage = _usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        client = FakeClient(response=_response(_message("ok"), usage=usage))
        provider = GroqProvider(api_key="x", model="m", client=client)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.total_tokens == 15
        assert resp.cached_input_tokens is None
        assert resp.reasoning_tokens is None

    def test_usage_none_when_absent(self):
        client = FakeClient(response=_response(_message("ok"), usage=None))
        provider = GroqProvider(api_key="x", model="m", client=client)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.input_tokens is None
        assert resp.total_tokens is None

    def test_malformed_tool_arguments_become_empty_dict(self):
        client = FakeClient(
            response=_response(
                _message(tool_calls=[_tool_call("x", {}, call_id="c1")]),
            )
        )
        # Corrupt the arguments after construction to simulate bad JSON.
        client.completions._response.choices[0].message.tool_calls[0].function.arguments = "{not json"
        provider = GroqProvider(api_key="x", model="m", client=client)

        resp = provider.complete([Message(role="user", content="hi")])
        assert resp.tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# Retry / error classification
# ---------------------------------------------------------------------------


class TestRetryAndClassification:
    def test_rate_limit_retries_then_succeeds(self, monkeypatch):
        import groq

        sleeps = []
        monkeypatch.setattr(
            "forge.providers.groq.time.sleep", lambda s: sleeps.append(s)
        )
        err = _http_error(groq.RateLimitError, 429, retry_after="2")
        client = FakeClient(raises=[err], response=_response(_message("ok")))
        provider = GroqProvider(api_key="x", model="m", client=client, retry_max=3)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.content == "ok"
        assert sleeps == [2.0]
        assert len(client.completions.calls) == 2

    def test_rate_limit_exhausts_retries_then_raises_retryable(self, monkeypatch):
        import groq

        monkeypatch.setattr("forge.providers.groq.time.sleep", lambda s: None)
        errs = [_http_error(groq.RateLimitError, 429) for _ in range(5)]
        client = FakeClient(raises=errs)
        provider = GroqProvider(api_key="x", model="m", client=client, retry_max=3)

        with pytest.raises(LLMError, match="Groq API error") as exc_info:
            provider.complete([Message(role="user", content="hi")])

        assert exc_info.value.retryable is True
        assert len(client.completions.calls) == 4  # 1 + 3 retries

    def test_authentication_error_is_not_retryable_and_does_not_retry(self, monkeypatch):
        import groq

        sleeps = []
        monkeypatch.setattr(
            "forge.providers.groq.time.sleep", lambda s: sleeps.append(s)
        )
        err = _http_error(groq.AuthenticationError, 401, message="invalid api key")
        client = FakeClient(raises=err)
        provider = GroqProvider(api_key="x", model="m", client=client)

        with pytest.raises(LLMError, match="Groq API error") as exc_info:
            provider.complete([Message(role="user", content="hi")])

        assert exc_info.value.retryable is False
        assert sleeps == []
        assert len(client.completions.calls) == 1

    def test_bad_request_error_is_not_retryable(self):
        import groq

        err = _http_error(groq.BadRequestError, 400, message="malformed request")
        client = FakeClient(raises=err)
        provider = GroqProvider(api_key="x", model="m", client=client)

        with pytest.raises(LLMError) as exc_info:
            provider.complete([Message(role="user", content="hi")])
        assert exc_info.value.retryable is False

    def test_connection_error_is_retryable(self, monkeypatch):
        import groq

        monkeypatch.setattr("forge.providers.groq.time.sleep", lambda s: None)
        req = httpx.Request("POST", "https://api.groq.com/x")
        errs = [groq.APIConnectionError(request=req) for _ in range(5)]
        client = FakeClient(raises=errs)
        provider = GroqProvider(api_key="x", model="m", client=client, retry_max=2)

        with pytest.raises(LLMError) as exc_info:
            provider.complete([Message(role="user", content="hi")])
        assert exc_info.value.retryable is True

    def test_generic_exception_becomes_retryable_llm_error(self):
        client = FakeClient(raises=RuntimeError("boom"))
        provider = GroqProvider(api_key="x", model="m", client=client)

        with pytest.raises(LLMError, match="Groq request failed") as exc_info:
            provider.complete([Message(role="user", content="hi")])
        assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# Configuration / preflight
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_missing_model_raises_clear_configuration_error(self):
        provider = GroqProvider(api_key="x", model="")
        with pytest.raises(ValueError, match="No Groq model configured"):
            provider.preflight()

    def test_missing_api_key_raises_clear_configuration_error(self, monkeypatch):
        monkeypatch.delenv("FORGE_GROQ_API_KEY", raising=False)
        provider = GroqProvider(api_key=None, model="llama-3.3-70b-versatile")
        with pytest.raises(ValueError, match="No Groq API key configured"):
            provider.preflight()

    def test_preflight_passes_with_key_and_model(self, monkeypatch):
        monkeypatch.setenv("FORGE_GROQ_API_KEY", "k")
        provider = GroqProvider(api_key=None, model="llama-3.3-70b-versatile")
        provider.preflight()  # must not raise

    def test_injected_client_skips_key_check(self):
        provider = GroqProvider(
            api_key=None, model="m", client=FakeClient(response=_response(_message("ok")))
        )
        provider.preflight()  # must not raise: client already present

    def test_provider_and_model_name(self):
        provider = GroqProvider(api_key="x", model="llama-3.3-70b-versatile")
        assert provider.provider_name == "groq"
        assert provider.model_name == "llama-3.3-70b-versatile"

    def test_api_key_never_appears_in_error_text(self, monkeypatch):
        monkeypatch.delenv("FORGE_GROQ_API_KEY", raising=False)
        provider = GroqProvider(api_key="super-secret-key-value", model="")
        try:
            provider.preflight()
        except ValueError as exc:
            assert "super-secret-key-value" not in str(exc)
