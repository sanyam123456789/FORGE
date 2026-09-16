"""
forge.providers.cerebras — Cerebras adapter (second fallback provider).

Wraps the official ``cerebras_cloud_sdk`` SDK (OpenAI-compatible
chat-completions API) and implements ``forge.llm.LLMProvider``. Structurally
mirrors ``forge/providers/groq.py`` — see that module's docstring for the
shared design rationale (normalised response, ``retryable`` classification,
provider-owned bounded retry with the SDK's own retry disabled).

Credentials come only from ``FORGE_CEREBRAS_API_KEY``. An API key is never
hard-coded and never logged.

The model must be explicitly configured (``FORGE_CEREBRAS_MODEL``) — this
adapter does not guess a default. Unlike Groq's SDK, the installed Cerebras
SDK does not ship a ``Literal`` enum of model names to draw from, so there is
no adapter-known model list at all; check Cerebras's own docs
(https://inference-docs.cerebras.ai/models) for models that support
tool/function calling before setting ``FORGE_CEREBRAS_MODEL``.
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

_DEFAULT_RETRY_MAX = 3
_DEFAULT_BACKOFF_S = 5.0


def _import_sdk():
    """Import the cerebras_cloud_sdk SDK, raising a clear error if missing."""
    try:
        import cerebras.cloud.sdk as cerebras_sdk
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "The 'cerebras_cloud_sdk' package is required for the Cerebras "
            "provider. Install it with:  pip install cerebras_cloud_sdk"
        ) from exc
    return cerebras_sdk


def _retry_delay_s(exc: Exception) -> float | None:
    """Extract a Retry-After header (seconds) from a Cerebras API error, if present."""
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


class CerebrasProvider(LLMProvider):
    """LLM provider adapter for Cerebras."""

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
        return "cerebras"

    @property
    def model_name(self) -> str:
        return self._model

    def preflight(self) -> None:
        """Fail fast if no API key or no model is configured (no network call)."""
        if not self._model:
            raise ValueError(
                "No Cerebras model configured. Set FORGE_CEREBRAS_MODEL to a "
                "Cerebras model that supports tool/function calling — check "
                "https://inference-docs.cerebras.ai/models for the current "
                "list."
            )
        if self._client is not None:
            return
        if not self._resolve_api_key():
            raise ValueError(
                "No Cerebras API key configured. Set FORGE_CEREBRAS_API_KEY "
                "in the environment or .env file."
            )

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self._model:
            raise ValueError(
                "No Cerebras model configured. Set FORGE_CEREBRAS_MODEL to a "
                "Cerebras model that supports tool/function calling."
            )

        cerebras_sdk = _import_sdk()
        client = self._get_client(cerebras_sdk)

        request_messages = self._to_cerebras_messages(messages)
        request_tools = self._to_cerebras_tools(tools) if tools else None

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
                    timeout=(
                        self._timeout
                        if self._timeout is not None
                        else cerebras_sdk.NOT_GIVEN
                    ),
                )
                break
            except (
                cerebras_sdk.RateLimitError,
                cerebras_sdk.InternalServerError,
                cerebras_sdk.APIConnectionError,
                cerebras_sdk.APITimeoutError,
            ) as exc:
                if attempt >= self._retry_max:
                    raise LLMError(
                        f"Cerebras API error: {exc}", retryable=True
                    ) from exc
                delay = _retry_delay_s(exc)
                if delay is None:
                    delay = _DEFAULT_BACKOFF_S * (2**attempt)
                attempt += 1
                logger.warning(
                    "Cerebras transient error (%s); retrying in %.1fs "
                    "(attempt %d/%d)",
                    type(exc).__name__,
                    delay,
                    attempt,
                    self._retry_max,
                )
                time.sleep(delay)
            except cerebras_sdk.CerebrasError as exc:
                raise LLMError(f"Cerebras API error: {exc}", retryable=False) from exc
            except Exception as exc:  # noqa: BLE001 - normalise everything to LLMError
                raise LLMError(
                    f"Cerebras request failed: {exc!r}", retryable=True
                ) from exc

        return self._to_llm_response(response)

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        return self._explicit_api_key or os.environ.get("FORGE_CEREBRAS_API_KEY")

    def _get_client(self, cerebras_sdk) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "No Cerebras API key configured. Set FORGE_CEREBRAS_API_KEY "
                "in the environment or .env file."
            )
        # max_retries=0: FORGE owns retry/backoff (above), never duplicated
        # across the SDK's own retry layer and ours.
        self._client = cerebras_sdk.Cerebras(api_key=api_key, max_retries=0)
        return self._client

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_cerebras_messages(messages: list[Message]) -> list[dict[str, Any]]:
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

            out.append({"role": "user", "content": msg.content or ""})
        return out

    @staticmethod
    def _to_cerebras_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """FORGE's canonical tool schema is already OpenAI-function-format,
        which Cerebras's OpenAI-compatible API expects — pass through
        unchanged."""
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
            "cached_input_tokens": None,  # not exposed by Cerebras's usage object
            "reasoning_tokens": None,  # not exposed by Cerebras's usage object
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
