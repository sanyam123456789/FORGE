"""
Tests for forge.providers.fallback.FallbackProvider and its wiring into
forge.llm.get_provider().

All providers here are fakes (LLMProvider subclasses) — no real network
calls, no real SDKs involved. This exercises FallbackProvider's own
decision logic (skip unconfigured, fall back on retryable errors, stop on
non-retryable errors, bounded single pass) independent of any one adapter's
translation code (covered separately in test_providers_groq.py /
test_providers_cerebras.py / test_llm.py's Gemini tests).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.llm import LLMError, LLMProvider, LLMResponse, Message, ToolCall, get_provider
from forge.providers.fallback import FallbackProvider


# ---------------------------------------------------------------------------
# Fake provider (mirrors tests/test_agent.py's ScriptedProvider pattern)
# ---------------------------------------------------------------------------


class FakeProvider(LLMProvider):
    """A scripted LLMProvider: raises `preflight_error` on preflight() if
    set, otherwise returns `response` or raises `complete_error` on
    complete()."""

    def __init__(
        self,
        name,
        *,
        model="fake-model",
        response=None,
        complete_error=None,
        preflight_error=None,
    ):
        self._name = name
        self._model = model
        self._response = response
        self._complete_error = complete_error
        self._preflight_error = preflight_error
        self.complete_calls = 0
        self.preflight_calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    def preflight(self) -> None:
        self.preflight_calls += 1
        if self._preflight_error is not None:
            raise self._preflight_error

    def complete(self, messages, *, tools=None, max_tokens=4096, temperature=0.0):
        self.complete_calls += 1
        if self._complete_error is not None:
            raise self._complete_error
        return self._response


def _ok_response(text="answer"):
    return LLMResponse(content=text, stop_reason="stop")


def _not_configured(provider_label):
    return ValueError(f"No {provider_label} API key configured.")


# ---------------------------------------------------------------------------
# 1. Primary succeeds without fallback
# ---------------------------------------------------------------------------


def test_primary_succeeds_without_fallback():
    gemini = FakeProvider("gemini", response=_ok_response("from gemini"))
    groq = FakeProvider("groq", response=_ok_response("from groq"))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

    resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from gemini"
    assert gemini.complete_calls == 1
    assert groq.complete_calls == 0  # never touched
    assert fb.provider_name == "gemini"


# ---------------------------------------------------------------------------
# 2 & 3. Gemini transient 429 / quota error -> Groq succeeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_message",
    ["429 RESOURCE_EXHAUSTED rate limit", "quota exceeded for today"],
)
def test_gemini_transient_error_falls_back_to_groq(error_message):
    gemini = FakeProvider(
        "gemini", complete_error=LLMError(error_message, retryable=True)
    )
    groq = FakeProvider("groq", response=_ok_response("from groq"))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

    resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from groq"
    assert gemini.complete_calls == 1
    assert groq.complete_calls == 1
    assert fb.provider_name == "groq"


# ---------------------------------------------------------------------------
# 4. Gemini fails -> Groq fails -> Cerebras succeeds
# ---------------------------------------------------------------------------


def test_full_chain_falls_back_to_cerebras():
    gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
    groq = FakeProvider("groq", complete_error=LLMError("503 unavailable", retryable=True))
    cerebras = FakeProvider("cerebras", response=_ok_response("from cerebras"))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq), ("cerebras", cerebras)])

    resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from cerebras"
    assert gemini.complete_calls == 1
    assert groq.complete_calls == 1
    assert cerebras.complete_calls == 1


# ---------------------------------------------------------------------------
# 5. All providers fail -> clear final error
# ---------------------------------------------------------------------------


def test_all_providers_fail_raises_clear_final_error():
    gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
    groq = FakeProvider("groq", complete_error=LLMError("503", retryable=True))
    cerebras = FakeProvider("cerebras", complete_error=LLMError("429", retryable=True))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq), ("cerebras", cerebras)])

    with pytest.raises(LLMError, match="All configured providers failed") as exc_info:
        fb.complete([Message(role="user", content="hi")])

    msg = str(exc_info.value)
    assert "gemini" in msg and "groq" in msg and "cerebras" in msg
    assert exc_info.value.retryable is False  # nothing further to fall back to


# ---------------------------------------------------------------------------
# 6 & 7. Missing key on a non-primary candidate is skipped, clearly logged
# ---------------------------------------------------------------------------


def test_missing_groq_key_skips_to_cerebras(caplog):
    gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
    groq = FakeProvider("groq", preflight_error=_not_configured("Groq"))
    cerebras = FakeProvider("cerebras", response=_ok_response("from cerebras"))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq), ("cerebras", cerebras)])

    with caplog.at_level("WARNING"):
        resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from cerebras"
    assert groq.complete_calls == 0  # never attempted: skipped at preflight
    assert any("groq" in r.message.lower() and "unavailable" in r.message.lower() for r in caplog.records)


def test_missing_cerebras_key_is_final_skip_reason():
    gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
    groq = FakeProvider("groq", complete_error=LLMError("503", retryable=True))
    cerebras = FakeProvider("cerebras", preflight_error=_not_configured("Cerebras"))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq), ("cerebras", cerebras)])

    with pytest.raises(LLMError, match="cerebras.*not configured|not configured.*cerebras"):
        fb.complete([Message(role="user", content="hi")])


# ---------------------------------------------------------------------------
# 8. Authentication/configuration errors at call time do not silently fallback
# ---------------------------------------------------------------------------


def test_non_retryable_error_does_not_fallback():
    gemini = FakeProvider(
        "gemini", complete_error=LLMError("401 invalid api key", retryable=False)
    )
    groq = FakeProvider("groq", response=_ok_response("from groq"))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

    with pytest.raises(LLMError, match="401 invalid api key"):
        fb.complete([Message(role="user", content="hi")])

    assert gemini.complete_calls == 1
    assert groq.complete_calls == 0  # never reached: no fallback for auth errors


def test_missing_key_on_sole_configured_provider_raises_clearly():
    """The only candidate in the chain (e.g. FORGE_PROVIDER_ORDER=gemini
    alone) must not silently no-op when its key is missing."""
    gemini = FakeProvider("gemini", preflight_error=_not_configured("Gemini"))
    fb = FallbackProvider([("gemini", gemini)])

    with pytest.raises(ValueError, match="No configured provider is usable"):
        fb.preflight()

    with pytest.raises(LLMError, match="All configured providers failed"):
        fb.complete([Message(role="user", content="hi")])


# ---------------------------------------------------------------------------
# 9, 10, 11. Tool calls / arguments / tool results preserved through fallback
# ---------------------------------------------------------------------------


def test_tool_call_response_passes_through_unmodified():
    tool_resp = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "a.py"})],
        stop_reason="tool_use",
    )
    gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
    groq = FakeProvider("groq", response=tool_resp)
    fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

    resp = fb.complete([Message(role="user", content="hi")])

    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "write_file"
    assert resp.tool_calls[0].arguments == {"path": "a.py"}  # unchanged dict


def test_full_message_history_forwarded_to_whichever_provider_serves_it():
    history = [
        Message(role="system", content="sys"),
        Message(role="user", content="do it"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "a"})],
        ),
        Message(role="tool", content="ok", tool_call_id="c1", name="write_file"),
    ]
    seen = {}

    class RecordingProvider(FakeProvider):
        def complete(self, messages, *, tools=None, max_tokens=4096, temperature=0.0):
            seen["messages"] = messages
            return super().complete(messages, tools=tools)

    gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
    groq = RecordingProvider("groq", response=_ok_response("ok"))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

    fb.complete(history)

    assert seen["messages"] == history  # identical, not reconstructed/mutated


# ---------------------------------------------------------------------------
# 12. Provider order is respected
# ---------------------------------------------------------------------------


def test_provider_order_is_respected():
    calls = []

    class OrderTrackingProvider(FakeProvider):
        def complete(self, messages, *, tools=None, max_tokens=4096, temperature=0.0):
            calls.append(self._name)
            return super().complete(messages)

    cerebras = OrderTrackingProvider(
        "cerebras", complete_error=LLMError("x", retryable=True)
    )
    gemini = OrderTrackingProvider("gemini", complete_error=LLMError("x", retryable=True))
    groq = OrderTrackingProvider("groq", response=_ok_response("ok"))
    # Constructed in a deliberately different order than passed to
    # FallbackProvider, to prove list order (not construction order) wins.
    fb = FallbackProvider([("gemini", gemini), ("cerebras", cerebras), ("groq", groq)])

    fb.complete([Message(role="user", content="hi")])

    assert calls == ["gemini", "cerebras", "groq"]


# ---------------------------------------------------------------------------
# 13. Retry limits are bounded (each candidate tried at most once per call)
# ---------------------------------------------------------------------------


def test_each_candidate_tried_at_most_once_per_call():
    gemini = FakeProvider("gemini", complete_error=LLMError("x", retryable=True))
    groq = FakeProvider("groq", complete_error=LLMError("x", retryable=True))
    fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

    with pytest.raises(LLMError):
        fb.complete([Message(role="user", content="hi")])

    assert gemini.complete_calls == 1
    assert groq.complete_calls == 1


def test_fallback_provider_requires_at_least_one_candidate():
    with pytest.raises(ValueError, match="requires at least one provider"):
        FallbackProvider([])


# ---------------------------------------------------------------------------
# 14. API keys never appear in FallbackProvider's own log/error output
# ---------------------------------------------------------------------------


def test_api_key_never_appears_in_fallback_error_output(caplog):
    """FallbackProvider only ever forwards a candidate's own preflight()
    error text — it never introspects or logs a candidate's constructor
    arguments. Using the real GroqProvider (whose own preflight() message
    is verified elsewhere to never include the key) end-to-end confirms the
    secret passed at construction cannot leak through FallbackProvider's
    logs even though the candidate is otherwise unusable (no model set)."""
    from forge.providers.groq import GroqProvider

    secret = "sk-super-secret-value-12345"
    groq = GroqProvider(api_key=secret, model="")  # missing model -> preflight fails
    fb = FallbackProvider([("groq", groq)])

    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError):
            fb.preflight()

    assert secret not in caplog.text


# ---------------------------------------------------------------------------
# get_provider() wiring: opt-in, legacy behavior preserved
# ---------------------------------------------------------------------------


class TestGetProviderFallbackWiring:
    def test_empty_provider_order_preserves_legacy_single_provider(self):
        settings = SimpleNamespace(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_api_key="k",
            provider_order=(),
        )
        provider = get_provider(settings_override=settings)
        assert provider.provider_name == "gemini"
        assert not isinstance(provider, FallbackProvider)

    def test_provider_order_constructs_fallback_provider_in_order(self):
        settings = SimpleNamespace(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_api_key="k",
            provider_order=("gemini", "groq", "cerebras"),
            groq_api_key="",
            groq_model="",
            cerebras_api_key="",
            cerebras_model="",
            provider_retry_max=3,
            llm_request_timeout_s=None,
        )
        provider = get_provider(settings_override=settings)
        assert isinstance(provider, FallbackProvider)
        assert [c.name for c in provider._candidates] == ["gemini", "groq", "cerebras"]

    def test_unconfigured_secondary_providers_do_not_block_construction(self):
        """Groq/Cerebras keys not set yet (user adds them later) must not
        prevent building the chain — only using those candidates."""
        settings = SimpleNamespace(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_api_key="k",
            provider_order=("gemini", "groq", "cerebras"),
            groq_api_key="",
            groq_model="",
            cerebras_api_key="",
            cerebras_model="",
            provider_retry_max=3,
            llm_request_timeout_s=None,
        )
        provider = get_provider(settings_override=settings)  # must not raise
        assert isinstance(provider, FallbackProvider)


# ---------------------------------------------------------------------------
# Run-scoped stickiness: a provider that fails retryably is not re-attempted
# on later complete() calls within the same FallbackProvider instance (one
# agent run). See forge/providers/fallback.py module docstring "Run-scoped
# stickiness" for the design rationale (this fixes the repeated ~120s Gemini
# retry-exhaustion cost observed once its daily quota was hit mid-run).
# ---------------------------------------------------------------------------


class TestRunScopedStickiness:
    def test_second_call_skips_provider_that_failed_retryably_on_first_call(self):
        gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
        groq = FakeProvider("groq", response=_ok_response("from groq"))
        fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

        resp1 = fb.complete([Message(role="user", content="turn 1")])
        resp2 = fb.complete([Message(role="user", content="turn 2")])

        assert resp1.content == "from groq"
        assert resp2.content == "from groq"
        # Gemini was attempted exactly once (the first call) — the second
        # call skipped it entirely, no repeated preflight or complete.
        assert gemini.preflight_calls == 1
        assert gemini.complete_calls == 1
        assert groq.complete_calls == 2

    def test_skipped_down_provider_does_not_appear_as_active(self):
        gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
        groq = FakeProvider("groq", response=_ok_response("from groq"))
        fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

        fb.complete([Message(role="user", content="turn 1")])
        fb.complete([Message(role="user", content="turn 2")])

        assert fb.provider_name == "groq"
        assert fb.model_name == groq.model_name

    def test_unconfigured_provider_is_also_marked_down_after_first_skip(self):
        groq = FakeProvider("groq", preflight_error=_not_configured("Groq"))
        cerebras = FakeProvider("cerebras", response=_ok_response("from cerebras"))
        fb = FallbackProvider([("groq", groq), ("cerebras", cerebras)])

        fb.complete([Message(role="user", content="turn 1")])
        fb.complete([Message(role="user", content="turn 2")])

        # preflight() on the unconfigured candidate is only ever checked
        # once; the second turn skips straight past it.
        assert groq.preflight_calls == 1
        assert cerebras.complete_calls == 2

    def test_third_provider_still_reached_after_first_two_marked_down(self):
        gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
        groq = FakeProvider("groq", complete_error=LLMError("503", retryable=True))
        cerebras = FakeProvider("cerebras", response=_ok_response("from cerebras"))
        fb = FallbackProvider([("gemini", gemini), ("groq", groq), ("cerebras", cerebras)])

        fb.complete([Message(role="user", content="turn 1")])
        resp2 = fb.complete([Message(role="user", content="turn 2")])

        assert resp2.content == "from cerebras"
        assert gemini.complete_calls == 1
        assert groq.complete_calls == 1
        assert cerebras.complete_calls == 2

    def test_non_retryable_failure_does_not_mark_provider_down(self):
        """A non-retryable error stops the chain immediately (unchanged
        behavior) and must NOT mark the provider down, since a future turn
        with a different (valid) request could still succeed against it."""
        gemini = FakeProvider(
            "gemini", complete_error=LLMError("401 invalid key", retryable=False)
        )
        fb = FallbackProvider([("gemini", gemini)])

        with pytest.raises(LLMError):
            fb.complete([Message(role="user", content="turn 1")])

        assert "gemini" not in fb._down

    def test_fresh_fallback_provider_instance_retries_everything_from_scratch(self):
        """A new run gets a new FallbackProvider — down-marking must never
        leak across instances/runs."""
        gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
        groq = FakeProvider("groq", response=_ok_response("from groq"))
        fb_run1 = FallbackProvider([("gemini", gemini), ("groq", groq)])
        fb_run1.complete([Message(role="user", content="hi")])
        assert "gemini" in fb_run1._down

        gemini2 = FakeProvider("gemini", response=_ok_response("from gemini"))
        groq2 = FakeProvider("groq", response=_ok_response("from groq"))
        fb_run2 = FallbackProvider([("gemini", gemini2), ("groq", groq2)])
        resp = fb_run2.complete([Message(role="user", content="hi")])

        assert resp.content == "from gemini"  # fresh instance tries Gemini first
        assert gemini2.complete_calls == 1
        assert groq2.complete_calls == 0

    def test_all_providers_marked_down_still_raises_clear_final_error(self):
        gemini = FakeProvider("gemini", complete_error=LLMError("quota", retryable=True))
        groq = FakeProvider("groq", complete_error=LLMError("503", retryable=True))
        fb = FallbackProvider([("gemini", gemini), ("groq", groq)])

        with pytest.raises(LLMError, match="All configured providers failed"):
            fb.complete([Message(role="user", content="turn 1")])
        # Second call: both already down, must still raise cleanly (no
        # exception from an empty loop, no infinite loop, bounded).
        with pytest.raises(LLMError, match="All configured providers failed"):
            fb.complete([Message(role="user", content="turn 2")])
