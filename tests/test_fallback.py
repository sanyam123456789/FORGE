"""
Tests for forge.providers.fallback.FallbackProvider and its wiring into
forge.llm.get_provider().

All providers here are fakes (LLMProvider subclasses) — no real network
calls, no real SDKs involved. This exercises FallbackProvider's own
decision logic (skip unconfigured, fall back on retryable errors, stop on
non-retryable errors, bounded single pass) independent of any one adapter's
translation code (covered separately in
test_providers_openai_compat.py).
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
    primary = FakeProvider("primary", response=_ok_response("from primary"))
    secondary = FakeProvider("secondary", response=_ok_response("from secondary"))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

    resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from primary"
    assert primary.complete_calls == 1
    assert secondary.complete_calls == 0  # never touched
    assert fb.provider_name == "primary"


# ---------------------------------------------------------------------------
# 2 & 3. Primary transient 429 / quota error -> Secondary succeeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_message",
    ["429 RESOURCE_EXHAUSTED rate limit", "quota exceeded for today"],
)
def test_primary_transient_error_falls_back_to_secondary(error_message):
    primary = FakeProvider(
        "primary", complete_error=LLMError(error_message, retryable=True)
    )
    secondary = FakeProvider("secondary", response=_ok_response("from secondary"))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

    resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from secondary"
    assert primary.complete_calls == 1
    assert secondary.complete_calls == 1
    assert fb.provider_name == "secondary"


# ---------------------------------------------------------------------------
# 4. Primary fails -> Secondary fails -> Tertiary succeeds
# ---------------------------------------------------------------------------


def test_full_chain_falls_back_to_tertiary():
    primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
    secondary = FakeProvider("secondary", complete_error=LLMError("503 unavailable", retryable=True))
    tertiary = FakeProvider("tertiary", response=_ok_response("from tertiary"))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary), ("tertiary", tertiary)])

    resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from tertiary"
    assert primary.complete_calls == 1
    assert secondary.complete_calls == 1
    assert tertiary.complete_calls == 1


# ---------------------------------------------------------------------------
# 5. All providers fail -> clear final error
# ---------------------------------------------------------------------------


def test_all_providers_fail_raises_clear_final_error():
    primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
    secondary = FakeProvider("secondary", complete_error=LLMError("503", retryable=True))
    tertiary = FakeProvider("tertiary", complete_error=LLMError("429", retryable=True))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary), ("tertiary", tertiary)])

    with pytest.raises(LLMError, match="All configured providers failed") as exc_info:
        fb.complete([Message(role="user", content="hi")])

    msg = str(exc_info.value)
    assert "primary" in msg and "secondary" in msg and "tertiary" in msg
    assert exc_info.value.retryable is False  # nothing further to fall back to


# ---------------------------------------------------------------------------
# 6 & 7. Missing key on a non-primary candidate is skipped, clearly logged
# ---------------------------------------------------------------------------


def test_missing_secondary_key_skips_to_tertiary(caplog):
    primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
    secondary = FakeProvider("secondary", preflight_error=_not_configured("Secondary"))
    tertiary = FakeProvider("tertiary", response=_ok_response("from tertiary"))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary), ("tertiary", tertiary)])

    with caplog.at_level("WARNING"):
        resp = fb.complete([Message(role="user", content="hi")])

    assert resp.content == "from tertiary"
    assert secondary.complete_calls == 0  # never attempted: skipped at preflight
    assert any("secondary" in r.message.lower() and "unavailable" in r.message.lower() for r in caplog.records)


def test_missing_tertiary_key_is_final_skip_reason():
    primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
    secondary = FakeProvider("secondary", complete_error=LLMError("503", retryable=True))
    tertiary = FakeProvider("tertiary", preflight_error=_not_configured("Tertiary"))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary), ("tertiary", tertiary)])

    with pytest.raises(LLMError, match="tertiary.*not configured|not configured.*tertiary"):
        fb.complete([Message(role="user", content="hi")])


# ---------------------------------------------------------------------------
# 8. Authentication/configuration errors at call time do not silently fallback
# ---------------------------------------------------------------------------


def test_non_retryable_error_does_not_fallback():
    primary = FakeProvider(
        "primary", complete_error=LLMError("401 invalid api key", retryable=False)
    )
    secondary = FakeProvider("secondary", response=_ok_response("from secondary"))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

    with pytest.raises(LLMError, match="401 invalid api key"):
        fb.complete([Message(role="user", content="hi")])

    assert primary.complete_calls == 1
    assert secondary.complete_calls == 0  # never reached: no fallback for auth errors


def test_missing_key_on_sole_configured_provider_raises_clearly():
    """The only candidate in the chain (e.g. FORGE_PROVIDER_ORDER=primary
    alone) must not silently no-op when its key is missing."""
    primary = FakeProvider("primary", preflight_error=_not_configured("Primary"))
    fb = FallbackProvider([("primary", primary)])

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
    primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
    secondary = FakeProvider("secondary", response=tool_resp)
    fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

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

    primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
    secondary = RecordingProvider("secondary", response=_ok_response("ok"))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

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

    tertiary = OrderTrackingProvider(
        "tertiary", complete_error=LLMError("x", retryable=True)
    )
    primary = OrderTrackingProvider("primary", complete_error=LLMError("x", retryable=True))
    secondary = OrderTrackingProvider("secondary", response=_ok_response("ok"))
    # Constructed in a deliberately different order than passed to
    # FallbackProvider, to prove list order (not construction order) wins.
    fb = FallbackProvider([("primary", primary), ("tertiary", tertiary), ("secondary", secondary)])

    fb.complete([Message(role="user", content="hi")])

    assert calls == ["primary", "tertiary", "secondary"]


# ---------------------------------------------------------------------------
# 13. Retry limits are bounded (each candidate tried at most once per call)
# ---------------------------------------------------------------------------


def test_each_candidate_tried_at_most_once_per_call():
    primary = FakeProvider("primary", complete_error=LLMError("x", retryable=True))
    secondary = FakeProvider("secondary", complete_error=LLMError("x", retryable=True))
    fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

    with pytest.raises(LLMError):
        fb.complete([Message(role="user", content="hi")])

    assert primary.complete_calls == 1
    assert secondary.complete_calls == 1


def test_fallback_provider_requires_at_least_one_candidate():
    with pytest.raises(ValueError, match="requires at least one provider"):
        FallbackProvider([])


# ---------------------------------------------------------------------------
# 14. API keys never appear in FallbackProvider's own log/error output
# ---------------------------------------------------------------------------


def test_api_key_never_appears_in_fallback_error_output(caplog):
    """FallbackProvider only ever forwards a candidate's own preflight()
    error text — it never introspects or logs a candidate's constructor
    arguments. Using the real OpenAICompatibleProvider (whose own
    preflight() message is verified elsewhere to never include the key)
    end-to-end confirms the secret passed at construction cannot leak
    through FallbackProvider's logs even though the candidate is otherwise
    unusable (no model set)."""
    from forge.providers.openai_compat import OpenAICompatibleProvider

    secret = "sk-super-secret-value-12345"
    real = OpenAICompatibleProvider(
        base_url="https://example.invalid/v1", api_key=secret, model=""
    )  # missing model -> preflight fails
    fb = FallbackProvider([("openai_compatible", real)])

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
            llm_provider="opencode_go",
            llm_model="deepseek-v4.1-flash",
            llm_api_key="k",
            provider_order=(),
        )
        provider = get_provider(settings_override=settings)
        assert provider.provider_name == "opencode_go"
        assert not isinstance(provider, FallbackProvider)

    def test_provider_order_constructs_fallback_provider_in_order(self):
        settings = SimpleNamespace(
            llm_provider="opencode_go",
            llm_model="deepseek-v4.1-flash",
            llm_api_key="k",
            llm_base_url="",
            provider_order=("opencode_go", "openai_compatible"),
            provider_retry_max=3,
            llm_request_timeout_s=None,
        )
        provider = get_provider(settings_override=settings)
        assert isinstance(provider, FallbackProvider)
        assert [c.name for c in provider._candidates] == ["opencode_go", "openai_compatible"]

    def test_unconfigured_secondary_providers_do_not_block_construction(self):
        """A secondary provider that is not configured yet (openai_compatible
        without FORGE_LLM_BASE_URL) must not prevent building the chain —
        only using that candidate."""
        settings = SimpleNamespace(
            llm_provider="opencode_go",
            llm_model="deepseek-v4.1-flash",
            llm_api_key="k",
            llm_base_url="",
            provider_order=("opencode_go", "openai_compatible"),
            provider_retry_max=3,
            llm_request_timeout_s=None,
        )
        provider = get_provider(settings_override=settings)  # must not raise
        assert isinstance(provider, FallbackProvider)


# ---------------------------------------------------------------------------
# Run-scoped stickiness: a provider that fails retryably is not re-attempted
# on later complete() calls within the same FallbackProvider instance (one
# agent run). See forge/providers/fallback.py module docstring "Run-scoped
# stickiness" for the design rationale (this fixes the repeated ~120s Primary
# retry-exhaustion cost observed once its daily quota was hit mid-run).
# ---------------------------------------------------------------------------


class TestRunScopedStickiness:
    def test_second_call_skips_provider_that_failed_retryably_on_first_call(self):
        primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
        secondary = FakeProvider("secondary", response=_ok_response("from secondary"))
        fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

        resp1 = fb.complete([Message(role="user", content="turn 1")])
        resp2 = fb.complete([Message(role="user", content="turn 2")])

        assert resp1.content == "from secondary"
        assert resp2.content == "from secondary"
        # Primary was attempted exactly once (the first call) — the second
        # call skipped it entirely, no repeated preflight or complete.
        assert primary.preflight_calls == 1
        assert primary.complete_calls == 1
        assert secondary.complete_calls == 2

    def test_skipped_down_provider_does_not_appear_as_active(self):
        primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
        secondary = FakeProvider("secondary", response=_ok_response("from secondary"))
        fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

        fb.complete([Message(role="user", content="turn 1")])
        fb.complete([Message(role="user", content="turn 2")])

        assert fb.provider_name == "secondary"
        assert fb.model_name == secondary.model_name

    def test_unconfigured_provider_is_also_marked_down_after_first_skip(self):
        secondary = FakeProvider("secondary", preflight_error=_not_configured("Secondary"))
        tertiary = FakeProvider("tertiary", response=_ok_response("from tertiary"))
        fb = FallbackProvider([("secondary", secondary), ("tertiary", tertiary)])

        fb.complete([Message(role="user", content="turn 1")])
        fb.complete([Message(role="user", content="turn 2")])

        # preflight() on the unconfigured candidate is only ever checked
        # once; the second turn skips straight past it.
        assert secondary.preflight_calls == 1
        assert tertiary.complete_calls == 2

    def test_third_provider_still_reached_after_first_two_marked_down(self):
        primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
        secondary = FakeProvider("secondary", complete_error=LLMError("503", retryable=True))
        tertiary = FakeProvider("tertiary", response=_ok_response("from tertiary"))
        fb = FallbackProvider([("primary", primary), ("secondary", secondary), ("tertiary", tertiary)])

        fb.complete([Message(role="user", content="turn 1")])
        resp2 = fb.complete([Message(role="user", content="turn 2")])

        assert resp2.content == "from tertiary"
        assert primary.complete_calls == 1
        assert secondary.complete_calls == 1
        assert tertiary.complete_calls == 2

    def test_non_retryable_failure_does_not_mark_provider_down(self):
        """A non-retryable error stops the chain immediately (unchanged
        behavior) and must NOT mark the provider down, since a future turn
        with a different (valid) request could still succeed against it."""
        primary = FakeProvider(
            "primary", complete_error=LLMError("401 invalid key", retryable=False)
        )
        fb = FallbackProvider([("primary", primary)])

        with pytest.raises(LLMError):
            fb.complete([Message(role="user", content="turn 1")])

        assert "primary" not in fb._down

    def test_fresh_fallback_provider_instance_retries_everything_from_scratch(self):
        """A new run gets a new FallbackProvider — down-marking must never
        leak across instances/runs."""
        primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
        secondary = FakeProvider("secondary", response=_ok_response("from secondary"))
        fb_run1 = FallbackProvider([("primary", primary), ("secondary", secondary)])
        fb_run1.complete([Message(role="user", content="hi")])
        assert "primary" in fb_run1._down

        primary2 = FakeProvider("primary", response=_ok_response("from primary"))
        secondary2 = FakeProvider("secondary", response=_ok_response("from secondary"))
        fb_run2 = FallbackProvider([("primary", primary2), ("secondary", secondary2)])
        resp = fb_run2.complete([Message(role="user", content="hi")])

        assert resp.content == "from primary"  # fresh instance tries Primary first
        assert primary2.complete_calls == 1
        assert secondary2.complete_calls == 0

    def test_all_providers_marked_down_still_raises_clear_final_error(self):
        primary = FakeProvider("primary", complete_error=LLMError("quota", retryable=True))
        secondary = FakeProvider("secondary", complete_error=LLMError("503", retryable=True))
        fb = FallbackProvider([("primary", primary), ("secondary", secondary)])

        with pytest.raises(LLMError, match="All configured providers failed"):
            fb.complete([Message(role="user", content="turn 1")])
        # Second call: both already down, must still raise cleanly (no
        # exception from an empty loop, no infinite loop, bounded).
        with pytest.raises(LLMError, match="All configured providers failed"):
            fb.complete([Message(role="user", content="turn 2")])
