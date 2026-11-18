"""
forge.providers.openai_compat — OpenAI-compatible chat-completions adapter.

One adapter for any endpoint that speaks the OpenAI ``/chat/completions``
API: OpenCode Go (the default preset, ``opencode_go``), OpenAI, OpenRouter,
a local Ollama server, ... It uses only the standard library
(``urllib.request``); no provider SDK is required.

The adapter:

- translates FORGE ``Message`` / tool schemas into a chat-completions
  request (FORGE tool schemas are already OpenAI function format),
- translates the response into a FORGE ``LLMResponse`` (usage metrics the
  provider does not report stay ``None`` — never fabricated),
- carries DeepSeek-style ``reasoning_content`` through the conversation so
  it can be echoed back on later requests (see ``_REASONING_KEY``),
- raises ``forge.llm.LLMError`` for every API failure, ``retryable=True``
  only for transient errors (429, 408, 5xx, connection errors, timeouts),
  after its own bounded exponential-backoff retry.

The API key is sent only in the Authorization header. It is never logged,
never included in an exception message and never shown in ``repr()``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from forge.llm import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from forge.logging import get_logger

logger = get_logger(__name__)

#: OpenCode Go's OpenAI-compatible endpoint (the ``opencode_go`` preset).
OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
#: Default model for the ``opencode_go`` preset.
OPENCODE_GO_DEFAULT_MODEL = "deepseek-v4.1-flash"

#: OpenCode Go asks clients to identify themselves with their own user agent
#: (a generic HTTP-library name such as "Python-urllib" is blocked).
USER_AGENT = "forge-agent/1.0"
#: Header carrying a stable per-conversation id. OpenCode Go requires it and
#: uses it for routing and prompt caching; other endpoints ignore it.
SESSION_HEADER = "x-opencode-session"

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RETRY_MAX = 3
_BACKOFF_BASE_S = 2.0
#: Upper bound on a single Retry-After sleep, so one header cannot stall a
#: run for an hour (a daily-quota 429 should fail, not wait).
_MAX_RETRY_AFTER_S = 60.0
_RETRYABLE_STATUS = frozenset({408, 429})

#: Key used for DeepSeek-style reasoning text. AgentRuntime builds the
#: assistant Message itself (without metadata), so the reasoning travels on
#: the first ToolCall's ``provider_signature`` — the field FORGE already
#: reserves for opaque per-call data a provider needs echoed back. It is
#: also read from ``Message.metadata`` if present. It is never part of
#: ``Message.content``, so it is never the final answer and never counted
#: in context char totals.
_REASONING_KEY = "reasoning_content"

_STOP_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "content_filter": "safety",
}

#: transport(url, headers, payload, timeout) -> (status, headers, body).
#: ``body`` is the parsed JSON (or the raw text if it is not JSON).
#: Connection failures / timeouts are raised as exceptions.
Transport = Callable[
    [str, Mapping[str, str], dict[str, Any], float],
    tuple[int, Mapping[str, str], Any],
]


def urllib_transport(
    url: str, headers: Mapping[str, str], payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, str], Any]:
    """Default transport: a JSON POST with ``urllib.request``."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = response.status
            resp_headers = dict(response.headers.items())
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        resp_headers = dict(exc.headers.items()) if exc.headers else {}
        raw = exc.read() or b""
    text = raw.decode("utf-8", errors="replace")
    try:
        body: Any = json.loads(text) if text else None
    except ValueError:
        body = text
    return status, resp_headers, body


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider adapter for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        retry_max: int = DEFAULT_RETRY_MAX,
        provider_label: str = "openai_compatible",
        transport: Transport | None = None,
        api_key_env: str = "FORGE_LLM_API_KEY",
    ) -> None:
        self._base_url = (base_url or "").strip().rstrip("/")
        self._api_key = (api_key or "").strip()
        self._model = (model or "").strip()
        self._timeout = float(timeout)
        self._retry_max = max(0, int(retry_max))
        self._label = provider_label
        self._transport: Transport = transport or urllib_transport
        self._api_key_env = api_key_env
        # One id per provider instance = one id per conversation. A run builds
        # its own provider; `forge chat` keeps one for the whole session, so
        # every request of a conversation shares the id (needed for caching).
        self._session_id = uuid.uuid4().hex

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self) -> None:
        """Start a new conversation id (e.g. after `/clear` in chat)."""
        self._session_id = uuid.uuid4().hex

    def __repr__(self) -> str:
        key_hint = "***" if self._api_key else "<not set>"
        return (
            f"OpenAICompatibleProvider(provider_label={self._label!r}, "
            f"base_url={self._base_url!r}, model={self._model!r}, "
            f"api_key={key_hint})"
        )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self._label

    @property
    def model_name(self) -> str:
        return self._model

    def preflight(self) -> None:
        """Fail fast on missing configuration (no network call)."""
        if not self._base_url:
            raise ValueError(
                f"No base URL configured for provider '{self._label}'. "
                "Set FORGE_LLM_BASE_URL (e.g. http://localhost:11434/v1)."
            )
        if not self._model:
            raise ValueError(
                f"No model configured for provider '{self._label}'. "
                "Set FORGE_LLM_MODEL."
            )
        if not self._api_key:
            raise ValueError(
                f"No API key configured for provider '{self._label}'. "
                f"Set {self._api_key_env} in the environment or .env file."
            )

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.preflight()
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            SESSION_HEADER: self._session_id,
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_api_messages(messages),
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        attempt = 0
        while True:
            retry_after: float | None = None
            try:
                status, resp_headers, body = self._transport(
                    url, headers, payload, self._timeout
                )
            except Exception as exc:  # noqa: BLE001 - connection errors / timeouts
                error = LLMError(
                    f"{self._label} request failed: {type(exc).__name__}: "
                    f"{self._redact(str(exc))}",
                    retryable=True,
                )
            else:
                if 200 <= status < 300:
                    return self._to_llm_response(body)
                error = self._http_error(status, body)
                retry_after = _retry_after_s(resp_headers)

            if not error.retryable or attempt >= self._retry_max:
                raise error
            delay = (
                min(retry_after, _MAX_RETRY_AFTER_S)
                if retry_after is not None
                else _BACKOFF_BASE_S * (2**attempt)
            )
            attempt += 1
            logger.warning(
                "%s transient error (%s); retrying in %.1fs (attempt %d/%d)",
                self._label,
                error,
                delay,
                attempt,
                self._retry_max,
            )
            time.sleep(delay)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def _redact(self, text: str) -> str:
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "***")
        return text

    def _http_error(self, status: int, body: Any) -> LLMError:
        detail = self._redact(_error_detail(body))
        if status in (401, 403):
            suffix = f" — server said: {detail}" if detail else ""
            return LLMError(
                f"{self._label}: request rejected (HTTP {status}) — check "
                f"FORGE_LLM_API_KEY / OPENCODE_API_KEY{suffix}",
                retryable=False,
            )
        if status == 404:
            return LLMError(
                f"{self._label}: endpoint or model not found — check "
                f"FORGE_LLM_BASE_URL and FORGE_LLM_MODEL (HTTP 404, "
                f"model '{self._model}', base URL '{self._base_url}')",
                retryable=False,
            )
        retryable = status in _RETRYABLE_STATUS or 500 <= status <= 599
        suffix = f": {detail}" if detail else ""
        return LLMError(f"{self._label}: HTTP {status}{suffix}", retryable=retryable)

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role in ("system", "user"):
                out.append({"role": msg.role, "content": msg.content or ""})
            elif msg.role == "assistant":
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
                reasoning = _reasoning_of(msg)
                if reasoning is not None:
                    entry[_REASONING_KEY] = reasoning
                out.append(entry)
            elif msg.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }
                )
            else:
                # Unknown role: send as a plain user turn so nothing is lost.
                out.append({"role": "user", "content": msg.content or ""})
        return out

    # ------------------------------------------------------------------
    # Response translation
    # ------------------------------------------------------------------

    def _to_llm_response(self, body: Any) -> LLMResponse:
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices or not isinstance(choices, list) or not isinstance(choices[0], dict):
            raise LLMError(
                f"{self._label}: response contained no choices "
                f"(model '{self._model}')",
                retryable=False,
            )
        choice = choices[0]
        message = choice.get("message") or {}

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name") or "",
                    arguments=_load_args(fn.get("arguments")),
                )
            )

        reasoning = message.get(_REASONING_KEY)
        if isinstance(reasoning, str) and reasoning and tool_calls:
            tool_calls[0].provider_signature = {_REASONING_KEY: reasoning}

        finish = str(choice.get("finish_reason") or "stop").lower()
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
        prompt_details = (usage or {}).get("prompt_tokens_details") or {}
        completion_details = (usage or {}).get("completion_tokens_details") or {}
        cached = _int(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None
        if cached is None and usage:
            cached = _int(usage.get("prompt_cache_hit_tokens"))

        return LLMResponse(
            content=_text_content(message.get("content")),
            tool_calls=tool_calls,
            input_tokens=_int((usage or {}).get("prompt_tokens")),
            output_tokens=_int((usage or {}).get("completion_tokens")),
            total_tokens=_int((usage or {}).get("total_tokens")),
            cached_input_tokens=cached,
            reasoning_tokens=(
                _int(completion_details.get("reasoning_tokens"))
                if isinstance(completion_details, dict)
                else None
            ),
            stop_reason=_STOP_REASON_MAP.get(finish, finish),
            model=body.get("model") or self._model,
            raw_usage=usage,
            raw=body,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts: [{"type": "text", "text": ...}]
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _dump_args(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments  # raw (unparseable) arguments: send back as-is
    return json.dumps(arguments or {})


def _load_args(raw: Any) -> dict[str, Any] | str:
    """Parse tool arguments; keep the raw string if they are not a JSON object.

    AgentRuntime answers a non-dict ``arguments`` with a structured
    "tool arguments must be a JSON object" error the model can react to.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        return str(raw)
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    return parsed if isinstance(parsed, dict) else raw


def _reasoning_of(msg: Message) -> str | None:
    value = msg.metadata.get(_REASONING_KEY) if msg.metadata else None
    if isinstance(value, str) and value:
        return value
    for call in msg.tool_calls:
        sig = call.provider_signature
        if isinstance(sig, dict) and isinstance(sig.get(_REASONING_KEY), str):
            return sig[_REASONING_KEY]
    return None


def _retry_after_s(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


def _error_detail(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:500]
        if isinstance(err, str):
            return err[:500]
        if body.get("message"):
            return str(body["message"])[:500]
        return json.dumps(body)[:500]
    if body:
        return str(body)[:500]
    return ""
