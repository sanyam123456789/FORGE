"""
forge.llm — LLM provider interface layer.

This module defines the abstract contract that every LLM provider adapter
must satisfy.  Step 1 only defines the interface; concrete implementations
(OpenAI, Anthropic, Google) will be added in Step 2 when the agent loop
is built.

Design Notes
------------
- The interface is deliberately provider-agnostic.
- All providers speak the same Message / Response vocabulary.
- Concrete adapters live in forge/llm/ (one file per provider).
- The factory function get_provider() constructs the right adapter based
  on ``settings.llm_provider``.
- Token usage is returned as part of every LLMResponse so that the
  observability layer can log it without providers needing to know about
  the research instrumentation system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Message vocabulary
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a conversation thread.

    role:
        One of 'system' | 'user' | 'assistant' | 'tool'.
    content:
        Plain-text content (tool call payloads use structured content).
    metadata:
        Arbitrary key-value pairs for future extensibility (e.g. tool call IDs).
    """

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider.

    content:
        The text/content returned by the model.
    input_tokens:
        Number of input tokens consumed (None if not reported by provider).
    output_tokens:
        Number of output tokens generated (None if not reported by provider).
    stop_reason:
        Provider-specific stop reason string (e.g. 'stop', 'length', 'tool_use').
    raw:
        The raw provider response object for debugging / future features.
    """

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str = "stop"
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int | None:
        """Return total token count, or None if either component is unavailable."""
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base class for all LLM provider adapters.

    Each concrete adapter wraps a single provider SDK (e.g. openai, anthropic)
    and translates between FORGE's Message/LLMResponse types and the provider's
    native API objects.
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
            Optional list of tool schemas in the provider's expected format.
            When None, the model is called without tool-use capability.
        max_tokens:
            Maximum number of output tokens to generate.
        temperature:
            Sampling temperature (0.0 = deterministic/greedy).

        Returns
        -------
        LLMResponse
            Normalised response including token usage when available.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'openai')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier as sent to the provider (e.g. 'gpt-4o')."""


# ---------------------------------------------------------------------------
# Provider registry / factory
# ---------------------------------------------------------------------------


def get_provider() -> LLMProvider:
    """Construct and return the configured LLM provider adapter.

    Reads ``settings.llm_provider`` and ``settings.llm_model`` to select
    and configure the appropriate adapter.

    Raises
    ------
    NotImplementedError
        If the configured provider has no concrete adapter yet (Step 1 state).
    ImportError
        If the required provider SDK is not installed.
    ValueError
        If the configuration is invalid.
    """
    from forge.config import settings  # local import avoids circular deps

    provider = settings.llm_provider

    # Step 2 will fill these in with real adapter modules.
    raise NotImplementedError(
        f"LLM provider adapter for '{provider}' is not yet implemented. "
        "Concrete adapters will be added in Step 2 (forge/llm/openai_adapter.py etc.)."
    )
