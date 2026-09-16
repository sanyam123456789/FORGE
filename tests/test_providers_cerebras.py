"""
Tests for forge.providers.cerebras.

No real Cerebras API call is made: a fake client mimicking the
cerebras_cloud_sdk response shape is injected. Real SDK exception classes
are used (instantiated directly) so the adapter's classification of
retryable vs. non-retryable errors is exercised against the real exception
hierarchy.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from forge.llm import LLMError, Message, ToolCall
from forge.providers.cerebras import CerebrasProvider


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
        status,
        request=httpx.Request("POST", "https://api.cerebras.ai/x"),
        headers=headers,
    )
    return cls(message, response=resp, body={"error": {"message": message}})


class TestTranslation:
    def test_text_response(self):
        client = FakeClient(response=_response(_message("hello")))
        provider = CerebrasProvider(api_key="x", model="llama3.3-70b", client=client)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.content == "hello"
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"

    def test_tool_call_response_normalised(self):
        client = FakeClient(
            response=_response(
                _message(tool_calls=[_tool_call("write_file", {"path": "a.py"})]),
                finish_reason="tool_calls",
            )
        )
        provider = CerebrasProvider(api_key="x", model="m", client=client)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.stop_reason == "tool_use"
        call = resp.tool_calls[0]
        assert call.id == "call_abc"
        assert call.name == "write_file"
        assert call.arguments == {"path": "a.py"}

    def test_tool_result_message_preserved(self):
        client = FakeClient(response=_response(_message("done")))
        provider = CerebrasProvider(api_key="x", model="m", client=client)

        provider.complete(
            [
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
        assert sent[-1] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}

    def test_usage_mapped_when_present(self):
        usage = _usage(prompt_tokens=8, completion_tokens=4, total_tokens=12)
        client = FakeClient(response=_response(_message("ok"), usage=usage))
        provider = CerebrasProvider(api_key="x", model="m", client=client)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.input_tokens == 8
        assert resp.output_tokens == 4
        assert resp.total_tokens == 12


class TestRetryAndClassification:
    def test_rate_limit_retries_then_succeeds(self, monkeypatch):
        import cerebras.cloud.sdk as cb

        sleeps = []
        monkeypatch.setattr(
            "forge.providers.cerebras.time.sleep", lambda s: sleeps.append(s)
        )
        err = _http_error(cb.RateLimitError, 429, retry_after="3")
        client = FakeClient(raises=[err], response=_response(_message("ok")))
        provider = CerebrasProvider(api_key="x", model="m", client=client, retry_max=3)

        resp = provider.complete([Message(role="user", content="hi")])

        assert resp.content == "ok"
        assert sleeps == [3.0]

    def test_rate_limit_exhausts_retries_then_raises_retryable(self, monkeypatch):
        import cerebras.cloud.sdk as cb

        monkeypatch.setattr("forge.providers.cerebras.time.sleep", lambda s: None)
        errs = [_http_error(cb.RateLimitError, 429) for _ in range(5)]
        client = FakeClient(raises=errs)
        provider = CerebrasProvider(api_key="x", model="m", client=client, retry_max=3)

        with pytest.raises(LLMError, match="Cerebras API error") as exc_info:
            provider.complete([Message(role="user", content="hi")])

        assert exc_info.value.retryable is True
        assert len(client.completions.calls) == 4

    def test_authentication_error_is_not_retryable(self, monkeypatch):
        import cerebras.cloud.sdk as cb

        sleeps = []
        monkeypatch.setattr(
            "forge.providers.cerebras.time.sleep", lambda s: sleeps.append(s)
        )
        err = _http_error(cb.AuthenticationError, 401, message="invalid api key")
        client = FakeClient(raises=err)
        provider = CerebrasProvider(api_key="x", model="m", client=client)

        with pytest.raises(LLMError) as exc_info:
            provider.complete([Message(role="user", content="hi")])

        assert exc_info.value.retryable is False
        assert sleeps == []

    def test_generic_exception_becomes_retryable_llm_error(self):
        client = FakeClient(raises=RuntimeError("boom"))
        provider = CerebrasProvider(api_key="x", model="m", client=client)

        with pytest.raises(LLMError, match="Cerebras request failed") as exc_info:
            provider.complete([Message(role="user", content="hi")])
        assert exc_info.value.retryable is True


class TestConfiguration:
    def test_missing_model_raises_clear_configuration_error(self):
        provider = CerebrasProvider(api_key="x", model="")
        with pytest.raises(ValueError, match="No Cerebras model configured"):
            provider.preflight()

    def test_missing_api_key_raises_clear_configuration_error(self, monkeypatch):
        monkeypatch.delenv("FORGE_CEREBRAS_API_KEY", raising=False)
        provider = CerebrasProvider(api_key=None, model="llama3.3-70b")
        with pytest.raises(ValueError, match="No Cerebras API key configured"):
            provider.preflight()

    def test_preflight_passes_with_key_and_model(self, monkeypatch):
        monkeypatch.setenv("FORGE_CEREBRAS_API_KEY", "k")
        provider = CerebrasProvider(api_key=None, model="llama3.3-70b")
        provider.preflight()

    def test_provider_and_model_name(self):
        provider = CerebrasProvider(api_key="x", model="llama3.3-70b")
        assert provider.provider_name == "cerebras"
        assert provider.model_name == "llama3.3-70b"

    def test_api_key_never_appears_in_error_text(self, monkeypatch):
        monkeypatch.delenv("FORGE_CEREBRAS_API_KEY", raising=False)
        provider = CerebrasProvider(api_key="super-secret-key-value", model="")
        try:
            provider.preflight()
        except ValueError as exc:
            assert "super-secret-key-value" not in str(exc)
