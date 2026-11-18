"""
forge.llm — LLM provider interface layer.

This module defines the provider-agnostic contract that every LLM provider
adapter must satisfy.  The rest of FORGE (agent runtime, context, tools,
observability) speaks only the vocabulary defined here — ``Message``,
``ToolCall``, ``LLMResponse`` — and never touches a provider SDK object.

Design Notes
------------
- The interface is deliberately provider-agnostic.
- Concrete adapters live in ``forge/providers/`` (one module per provider).
- ``get_provider()`` constructs the right adapter from ``settings.llm_provider``.
- Token usage is returned as part of every ``LLMResponse`` so the
  observability layer can record it without providers needing to know
  anything about the research instrumentation.
- Unavailable usage metrics are ``None`` — never fabricated / never ``0``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tool-call vocabulary
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool/function invocation requested by the model.

    id:
        Provider-supplied identifier for the call (may be an empty string if
        the provider does not supply one; the agent will synthesise one).
    name:
        Registered tool name to invoke.
    arguments:
        Parsed keyword arguments for the tool (always a ``dict``; the adapter
        is responsible for parsing provider-native argument payloads).
    provider_signature:
        Optional opaque token some providers attach to a tool-call and require
        to be echoed back verbatim on the following request (e.g. DeepSeek
        ``reasoning_content`` on OpenAI-compatible endpoints).  FORGE never inspects, decodes or mutates it —
        it is carried through the conversation history untouched and handed
        back to the same provider adapter.  ``None`` when the provider does
        not use one.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    provider_signature: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Message vocabulary
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a conversation thread.

    role:
        One of 'system' | 'user' | 'assistant' | 'tool'.
    content:
        Plain-text content.  For an assistant message that only requests
        tool calls this may be an empty string.
    tool_calls:
        For assistant messages: the tool calls the model requested this turn.
    tool_call_id:
        For tool messages: the id of the ToolCall this message answers.
    name:
        For tool messages: the tool name this message answers.
    metadata:
        Arbitrary key-value pairs for future extensibility.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider.

    content:
        The text returned by the model (may be empty when only tool calls
        were emitted).
    tool_calls:
        Structured tool calls requested by the model this turn.
    input_tokens / output_tokens / total_tokens:
        Provider-reported usage.  ``None`` means the provider did not report
        that metric — it is never fabricated and never coerced to ``0``.
        Input and output availability are independent.
    cached_input_tokens:
        Provider-reported count of input tokens served from cache, if any.
    reasoning_tokens:
        Provider-reported "thinking"/reasoning tokens, if the provider
        exposes them separately.
    stop_reason:
        Normalised stop reason string (e.g. 'stop', 'length', 'tool_use').
    model:
        Model identifier the provider reported handling the request.
    raw_usage:
        The provider's raw usage object as a plain dict, for debugging and
        for later research that needs a metric not surfaced above.
    raw:
        The raw provider response object (never serialised into traces).
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    stop_reason: str = "stop"
    model: str | None = None
    raw_usage: dict[str, Any] | None = None
    raw: Any = field(default=None, repr=False)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def resolved_total_tokens(self) -> int | None:
        """Total tokens: provider-reported if available, else input+output.

        Returns ``None`` when neither the provider total nor both components
        are available.
        """
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised by a provider adapter when the underlying API call fails.

    Adapters translate provider-native exceptions into this type so the
    agent runtime can handle provider failures without importing any
    provider SDK.

    retryable:
        ``True`` when the adapter has classified this failure as transient /
        provider-availability related (rate limit, quota exhaustion,
        temporary server error, connection failure) — i.e. a failure where
        trying a *different* provider is a reasonable response.  ``False``
        (the default) for anything that would fail identically on any
        provider: authentication failure, malformed request, invalid tool
        schema, programming error, invalid configuration. Only
        ``forge.providers.fallback.FallbackProvider`` inspects this flag;
        every other caller (``agent.py`` included) treats ``LLMError``
        exactly as before this field was added.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(ABC):
    """Abstract base class for all LLM provider adapters.

    Each concrete adapter wraps a single provider SDK and translates between
    FORGE's Message/ToolCall/LLMResponse types and the provider's native API.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send *messages* to the LLM and return a normalised response.

        Parameters
        ----------
        messages:
            Conversation history including any system prompt.
        tools:
            Optional list of tool schemas in FORGE's canonical (OpenAI-style
            function) format, as produced by ``Tool.to_schema()``.  ``None``
            or empty means the model is called without tool-use capability.
        max_tokens:
            Maximum number of output tokens to generate. ``None`` = no cap
            (the provider's own default applies).
        temperature:
            Sampling temperature (0.0 = deterministic/greedy).

        Returns
        -------
        LLMResponse
            Normalised response including token usage when available.

        Raises
        ------
        LLMError
            If the provider API call fails.
        """

    def preflight(self) -> None:
        """Cheap readiness check run before a run starts.

        Default: no-op.  Adapters override this to fail fast on obvious
        misconfiguration (e.g. a missing API key) *without* making a network
        call, so callers can report a configuration error before any run
        state or trace is created.

        Raises
        ------
        ValueError
            If the adapter is not ready to make requests.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'opencode_go')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier as sent to the provider (e.g. 'deepseek-v4.1-flash')."""


# ---------------------------------------------------------------------------
# Provider registry / factory
# ---------------------------------------------------------------------------


def _build_named_provider(name: str, settings: Any) -> LLMProvider:
    """Construct one concrete adapter by provider name.

    Construction never touches the network and never validates the API key
    — that happens in the adapter's own ``preflight()`` (or on the first
    real ``complete()`` call). This lets ``FallbackProvider`` build every
    candidate in a fallback chain up front, even ones that are not yet
    configured, and only find out they are unusable when actually reached.
    """
    if name in ("opencode_go", "openai_compatible"):
        import os

        from forge.providers.openai_compat import (
            DEFAULT_TIMEOUT_S,
            OPENCODE_GO_BASE_URL,
            OPENCODE_GO_DEFAULT_MODEL,
            OpenAICompatibleProvider,
        )

        timeout = getattr(settings, "llm_request_timeout_s", None)
        common = {
            "timeout": DEFAULT_TIMEOUT_S if timeout is None else timeout,
            "retry_max": getattr(settings, "provider_retry_max", 3),
        }
        base_url = getattr(settings, "llm_base_url", "") or ""
        api_key = getattr(settings, "llm_api_key", "") or ""
        model = getattr(settings, "llm_model", "") or ""

        if name == "opencode_go":
            return OpenAICompatibleProvider(
                base_url=base_url or OPENCODE_GO_BASE_URL,
                api_key=api_key or os.environ.get("OPENCODE_API_KEY", "").strip(),
                model=model or OPENCODE_GO_DEFAULT_MODEL,
                provider_label="opencode_go",
                api_key_env="FORGE_LLM_API_KEY (or OPENCODE_API_KEY)",
                **common,
            )
        return OpenAICompatibleProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            provider_label="openai_compatible",
            **common,
        )

    raise NotImplementedError(
        f"LLM provider adapter for '{name}' is not implemented. "
        "Implemented adapters: opencode_go, openai_compatible. Additional "
        "adapters can be added under forge/providers/ behind the same "
        "LLMProvider interface."
    )


def get_provider(settings_override: Any = None) -> LLMProvider:
    """Construct and return the configured LLM provider adapter.

    Reads ``settings.llm_provider`` / ``settings.llm_model`` (or the provided
    *settings_override*) to select and configure the appropriate adapter.

    If ``settings.provider_order`` is non-empty (``FORGE_PROVIDER_ORDER`` is
    set), this instead returns a ``FallbackProvider`` that tries each named
    provider in order, falling back only on clearly-recoverable errors. This
    is opt-in: when ``FORGE_PROVIDER_ORDER`` is unset (the default), behavior
    is identical to every prior FORGE step — a single adapter for
    ``settings.llm_provider``, no fallback.

    Raises
    ------
    NotImplementedError
        If a configured provider name has no concrete adapter.
    ImportError
        If a required provider SDK is not installed.
    ValueError
        If the configuration is invalid (e.g. missing API key, empty
        provider order).
    """
    if settings_override is not None:
        settings = settings_override
    else:
        from forge.config import settings  # local import avoids circular deps

    provider_order = getattr(settings, "provider_order", ())
    if provider_order:
        from forge.providers.fallback import FallbackProvider

        candidates = [
            (name, _build_named_provider(name, settings)) for name in provider_order
        ]
        return FallbackProvider(candidates)

    return _build_named_provider(settings.llm_provider, settings)
