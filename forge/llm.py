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
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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
    """


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
        max_tokens: int = 4096,
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
            Maximum number of output tokens to generate.
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
        """Human-readable provider identifier (e.g. 'gemini')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier as sent to the provider (e.g. 'gemini-3.6-flash')."""


# ---------------------------------------------------------------------------
# Provider registry / factory
# ---------------------------------------------------------------------------


def get_provider(settings_override: Any = None) -> LLMProvider:
    """Construct and return the configured LLM provider adapter.

    Reads ``settings.llm_provider`` / ``settings.llm_model`` (or the provided
    *settings_override*) to select and configure the appropriate adapter.

    Raises
    ------
    NotImplementedError
        If the configured provider has no concrete adapter yet.
    ImportError
        If the required provider SDK is not installed.
    ValueError
        If the configuration is invalid (e.g. missing API key).
    """
    if settings_override is not None:
        settings = settings_override
    else:
        from forge.config import settings  # local import avoids circular deps

    provider = settings.llm_provider

    if provider == "gemini":
        from forge.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.llm_api_key or None,
            model=settings.llm_model,
        )

    raise NotImplementedError(
        f"LLM provider adapter for '{provider}' is not implemented. "
        "Step 2 ships the Gemini adapter only; set FORGE_LLM_PROVIDER=gemini. "
        "Additional adapters (openai, anthropic, ...) can be added under "
        "forge/providers/ behind the same LLMProvider interface."
    )
