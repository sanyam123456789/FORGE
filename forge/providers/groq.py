"""
forge.providers.groq — Groq adapter (first fallback provider).

Wraps the official ``groq`` SDK (OpenAI-compatible chat-completions API) and
implements ``forge.llm.LLMProvider``. All Groq-specific code is confined to
this file. The adapter:

- translates FORGE ``Message`` / tool schemas into Groq's chat-completions
  request format,
- translates the Groq response into a FORGE ``LLMResponse``,
- maps Groq ``usage`` onto FORGE's normalised usage fields (unavailable
  metrics stay ``None`` — never fabricated),
- raises ``forge.llm.LLMError`` for any provider/API failure, classified as
  ``retryable=True`` only for clearly transient/provider-availability
  errors (rate limit, quota, 5xx, connection/timeout) so
  ``forge.providers.fallback.FallbackProvider`` knows when moving to the
  next provider is appropriate. Authentication failures, malformed
  requests, and any other 4xx are never retryable.

Credentials come only from ``FORGE_GROQ_API_KEY`` (via configuration or,
as a fallback, the constructor's explicit ``api_key``). An API key is never
hard-coded and never logged.

The model must be explicitly configured (``FORGE_GROQ_MODEL``) — this
adapter does not guess a default, since which Groq models support
tool/function calling changes over time and must be checked against
Groq's own docs (https://console.groq.com/docs/models).
"""

from __future__ import annotations

import os
import time
from typing import Any

from forge.llm import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from forge.logging import get_logger

logger = get_logger(__name__)

_STOP_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "safety",
}

# Bounded retry for transient errors only (rate limit, quota, 5xx,
# connection/timeout). Mirrors forge/providers/gemini.py's pattern; kept as
# an independent loop rather than a shared helper since each provider's
# exception taxonomy differs.
_DEFAULT_RETRY_MAX = 3
_DEFAULT_BACKOFF_S = 5.0


def _import_sdk():
    """Import the groq SDK, raising a clear error if it is missing."""
    try:
        import groq
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "The 'groq' package is required for the Groq provider. "
            "Install it with:  pip install groq"
        ) from exc
    return groq


def _retry_delay_s(exc: Exception) -> float | None:
    """Extract a Retry-After header (seconds) from a Groq API error, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class GroqProvider(LLMProvider):
    """LLM provider adapter for Groq."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "",
        client: Any = None,
        retry_max: int = _DEFAULT_RETRY_MAX,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        self._explicit_api_key = api_key
        self._client = client  # injected client (tests) or lazily built
        self._retry_max = retry_max
        self._timeout = timeout

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def preflight(self) -> None:
        """Fail fast if no API key or no model is configured (no network call)."""
        if not self._model:
            raise ValueError(
                "No Groq model configured. Set FORGE_GROQ_MODEL to a Groq "
                "model that supports tool/function calling — check "
                "https://console.groq.com/docs/models for the current list."
            )
        if self._client is not None:
            return
        if not self._resolve_api_key():
            raise ValueError(
                "No Groq API key configured. Set FORGE_GROQ_API_KEY in the "
                "environment or .env file."
            )

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # preflight() already fails fast on missing model/key when called
        # by the CLI or FallbackProvider; complete() re-validates the model
        # here too since it can be invoked directly (e.g. tests, other
        # callers) without preflight().
        if not self._model:
            raise ValueError(
                "No Groq model configured. Set FORGE_GROQ_MODEL to a Groq "
                "model that supports tool/function calling."
            )

        groq_sdk = _import_sdk()
        client = self._get_client(groq_sdk)

        request_messages = self._to_groq_messages(messages)
        request_tools = self._to_groq_tools(tools) if tools else None

        attempt = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=request_messages,
                    tools=request_tools,
                    tool_choice="auto" if request_tools else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=self._timeout if self._timeout is not None else groq_sdk.NOT_GIVEN,
                )
                break
            except (
                groq_sdk.RateLimitError,
                groq_sdk.InternalServerError,
                groq_sdk.APIConnectionError,
                groq_sdk.APITimeoutError,
            ) as exc:
                if attempt >= self._retry_max:
                    raise LLMError(f"Groq API error: {exc}", retryable=True) from exc
                delay = _retry_delay_s(exc)
                if delay is None:
                    delay = _DEFAULT_BACKOFF_S * (2**attempt)
                attempt += 1
                logger.warning(
                    "Groq transient error (%s); retrying in %.1fs "
                    "(attempt %d/%d)",
                    type(exc).__name__,
                    delay,
                    attempt,
                    self._retry_max,
                )
                time.sleep(delay)
            except groq_sdk.GroqError as exc:
                # Authentication, permission, bad-request, not-found,
                # unprocessable-entity, conflict — none of these are fixed
                # by trying a different provider.
                raise LLMError(f"Groq API error: {exc}", retryable=False) from exc
            except Exception as exc:  # noqa: BLE001 - normalise everything to LLMError
                raise LLMError(f"Groq request failed: {exc!r}", retryable=True) from exc

        return self._to_llm_response(response)

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        return self._explicit_api_key or os.environ.get("FORGE_GROQ_API_KEY")

    def _get_client(self, groq_sdk) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "No Groq API key configured. Set FORGE_GROQ_API_KEY in the "
                "environment or .env file."
            )
        # max_retries=0: FORGE owns retry/backoff (above) so it is never
        # duplicated across the SDK's own retry layer and ours.
        self._client = groq_sdk.Groq(api_key=api_key, max_retries=0)
        return self._client

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_groq_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role in ("system", "user"):
                out.append({"role": msg.role, "content": msg.content or ""})
                continue

            if msg.role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                }
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": _dump_args(call.arguments),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                out.append(entry)
                continue

            if msg.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }
                )
                continue

            # Unknown role: fall back to a plain user turn so nothing is lost.
            out.append({"role": "user", "content": msg.content or ""})
        return out

    @staticmethod
    def _to_groq_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """FORGE's canonical tool schema is already OpenAI-function-format,
        which is exactly what Groq expects — pass through unchanged."""
        return list(tools)

    # ------------------------------------------------------------------
    # Response translation
    # ------------------------------------------------------------------

    def _to_llm_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        finish_name = (choice.finish_reason or "stop").lower()
        stop_reason = _STOP_REASON_MAP.get(finish_name, finish_name)

        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            try:
                arguments = _load_args(tc.function.arguments)
            except ValueError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
            )
        if tool_calls:
            stop_reason = "tool_use"

        usage = self._extract_usage(getattr(response, "usage", None))

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            stop_reason=stop_reason,
            model=getattr(response, "model", None) or self._model,
            raw_usage=usage["raw"],
            raw=response,
        )

    @staticmethod
    def _extract_usage(usage: Any) -> dict[str, Any]:
        empty = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "raw": None,
        }
        if usage is None:
            return empty

        def _int(value: Any) -> int | None:
            return int(value) if isinstance(value, int) else None

        raw: dict[str, Any] | None = None
        try:
            dumped = usage.model_dump(exclude_none=True)  # pydantic model
            raw = {k: v for k, v in dumped.items() if isinstance(v, (int, float, str))}
        except Exception:  # noqa: BLE001 - raw usage is best-effort only
            raw = None

        return {
            "input_tokens": _int(getattr(usage, "prompt_tokens", None)),
            "output_tokens": _int(getattr(usage, "completion_tokens", None)),
            "total_tokens": _int(getattr(usage, "total_tokens", None)),
            "cached_input_tokens": None,  # not exposed by Groq's usage object
            "reasoning_tokens": None,  # not exposed by Groq's usage object
            "raw": raw,
        }


def _dump_args(arguments: dict[str, Any]) -> str:
    import json

    return json.dumps(arguments or {})


def _load_args(raw: str) -> dict[str, Any]:
    import json

    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise ValueError("tool call arguments did not decode to a JSON object")
    return parsed
