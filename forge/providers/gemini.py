"""
forge.providers.gemini — Google Gemini adapter.

Wraps the ``google-genai`` SDK and implements ``forge.llm.LLMProvider``.
All Gemini-specific code is contained in this file.  The adapter:

- translates FORGE ``Message`` / tool schemas into Gemini request objects,
- translates the Gemini response into a FORGE ``LLMResponse``,
- maps Gemini ``usage_metadata`` onto FORGE's normalised usage fields
  (unavailable metrics stay ``None`` — never fabricated),
- raises ``forge.llm.LLMError`` for any provider/API failure.

Credentials come from configuration (``settings.llm_api_key``) or, as a
fallback, the ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` environment variables.
An API key is never hard-coded and never logged.
"""

from __future__ import annotations

import os
import time
from typing import Any

from forge.llm import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from forge.logging import get_logger

logger = get_logger(__name__)

# Bounded retry for transient rate-limit (429 RESOURCE_EXHAUSTED) responses
# only. Any other APIError still raises immediately, unchanged.
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_DEFAULT_BACKOFF_S = 5.0


def _rate_limit_retry_delay_s(exc: Exception) -> float | None:
    """Extract the API-suggested retry delay (seconds) from a 429 APIError, if present."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    error = details.get("error", {})
    if not isinstance(error, dict):
        return None
    for item in error.get("details") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("@type", "")).endswith("RetryInfo"):
            delay = item.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    return None
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or getattr(exc, "status", None) == (
        "RESOURCE_EXHAUSTED"
    )


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """True when a 429's QuotaFailure detail names a *daily* quota.

    Google's API reports two distinct free-tier quota buckets under the
    same 429 RESOURCE_EXHAUSTED status: a per-minute one (worth retrying —
    it clears within seconds) and a per-day one (``quotaId`` containing
    "PerDay", e.g. ``GenerateRequestsPerDayPerProjectPerModel-FreeTier``),
    which will not clear within any retry window attempted here. The API
    still returns a short ``retryDelay`` for the daily case regardless,
    which is why retrying it wastes real time for zero chance of success.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return False
    error = details.get("error", {})
    if not isinstance(error, dict):
        return False
    for item in error.get("details") or []:
        if not isinstance(item, dict):
            continue
        if not str(item.get("@type", "")).endswith("QuotaFailure"):
            continue
        for violation in item.get("violations") or []:
            if not isinstance(violation, dict):
                continue
            if "perday" in str(violation.get("quotaId", "")).lower():
                return True
    return False

# Gemini finish-reason string -> FORGE normalised stop reason.
_STOP_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "safety",
    "RECITATION": "recitation",
    "BLOCKLIST": "safety",
    "PROHIBITED_CONTENT": "safety",
    "SPII": "safety",
    "MALFORMED_FUNCTION_CALL": "error",
    "OTHER": "other",
}


def _import_sdk():
    """Import the google-genai SDK, raising a clear error if it is missing."""
    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "The 'google-genai' package is required for the Gemini provider. "
            "Install it with:  pip install google-genai"
        ) from exc
    return genai, types, genai_errors


class GeminiProvider(LLMProvider):
    """LLM provider adapter for Google Gemini models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-3.6-flash",
        client: Any = None,
    ) -> None:
        self._model = model
        self._explicit_api_key = api_key
        self._client = client  # injected client (tests) or lazily built

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def preflight(self) -> None:
        """Fail fast if no API key is resolvable (no network call)."""
        if self._client is not None:
            return
        if not self._resolve_api_key():
            raise ValueError(
                "No Gemini API key configured. Set FORGE_LLM_API_KEY (or "
                "GEMINI_API_KEY / GOOGLE_API_KEY) in the environment or .env file."
            )

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        genai, types, genai_errors = _import_sdk()
        client = self._get_client(genai)

        system_instruction, contents = self._to_gemini_contents(messages, types)
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [self._to_gemini_tools(tools, types)]

        attempt = 0
        while True:
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                break
            except genai_errors.APIError as exc:
                if not _is_rate_limit_error(exc):
                    # Anything that isn't a rate limit (auth, malformed
                    # request, unsupported schema, ...) would fail
                    # identically on any provider — not fallback-eligible.
                    raise LLMError(
                        f"Gemini API error: {exc}", retryable=False
                    ) from exc
                if _is_daily_quota_exhausted(exc):
                    # A daily quota will not clear within any retry window
                    # we could reasonably wait — retrying here would only
                    # waste ~130s+ (3 attempts x the API's own, misleading
                    # short retryDelay) for zero chance of success. Fail
                    # fast, with zero sleeps, so FallbackProvider moves to
                    # the next configured provider immediately.
                    logger.warning(
                        "Gemini daily quota exhausted (not a transient "
                        "per-minute rate limit); skipping retries and "
                        "falling back immediately."
                    )
                    raise LLMError(
                        f"Gemini API error: {exc}", retryable=True
                    ) from exc
                if attempt >= _RATE_LIMIT_MAX_RETRIES:
                    # Rate limit persisted past our internal retry budget —
                    # still a transient/provider-availability condition, so
                    # a different provider is a reasonable next step.
                    raise LLMError(
                        f"Gemini API error: {exc}", retryable=True
                    ) from exc
                delay = _rate_limit_retry_delay_s(exc)
                if delay is None:
                    delay = _RATE_LIMIT_DEFAULT_BACKOFF_S * (2**attempt)
                attempt += 1
                logger.warning(
                    "Gemini rate limited (429); retrying in %.1fs "
                    "(attempt %d/%d)",
                    delay,
                    attempt,
                    _RATE_LIMIT_MAX_RETRIES,
                )
                time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - normalise everything to LLMError
                # Unclassified failure talking to the SDK/transport (e.g. a
                # connection error) — treated as provider-availability, so
                # fallback to a different provider is reasonable.
                raise LLMError(
                    f"Gemini request failed: {exc!r}", retryable=True
                ) from exc

        return self._to_llm_response(response)

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        return (
            self._explicit_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    def _get_client(self, genai) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError(
                "No Gemini API key configured. Set FORGE_LLM_API_KEY (or "
                "GEMINI_API_KEY / GOOGLE_API_KEY) in the environment or .env file."
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Request translation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gemini_contents(messages: list[Message], types) -> tuple[str, list[Any]]:
        """Return (system_instruction, contents) for the Gemini request."""
        system_parts: list[str] = []
        contents: list[Any] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
                continue

            if msg.role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=msg.content or "")],
                    )
                )
                continue

            if msg.role == "assistant":
                parts: list[Any] = []
                if msg.content:
                    parts.append(types.Part(text=msg.content))
                for call in msg.tool_calls:
                    part_kwargs: dict[str, Any] = {
                        "function_call": types.FunctionCall(
                            name=call.name,
                            args=call.arguments or {},
                        )
                    }
                    # Echo Gemini 3.x's opaque thought_signature back verbatim
                    # on the matching function_call Part when we captured one.
                    signature = getattr(call, "provider_signature", None)
                    if signature is not None:
                        part_kwargs["thought_signature"] = signature
                    parts.append(types.Part(**part_kwargs))
                if not parts:
                    parts.append(types.Part(text=""))
                contents.append(types.Content(role="model", parts=parts))
                continue

            if msg.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=msg.name or "tool",
                                    response={"result": msg.content},
                                )
                            )
                        ],
                    )
                )
                continue

            # Unknown role: fall back to a plain user turn so nothing is lost.
            contents.append(
                types.Content(role="user", parts=[types.Part(text=msg.content or "")])
            )

        return "\n\n".join(system_parts), contents

    @staticmethod
    def _to_gemini_tools(tools: list[dict[str, Any]], types) -> Any:
        """Translate FORGE canonical tool schemas into a Gemini Tool."""
        declarations = []
        for schema in tools:
            fn = schema.get("function", schema)
            declarations.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters_json_schema=fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                )
            )
        return types.Tool(function_declarations=declarations)

    # ------------------------------------------------------------------
    # Response translation
    # ------------------------------------------------------------------

    def _to_llm_response(self, response: Any) -> LLMResponse:
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason = "stop"

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            candidate = candidates[0]
            finish = getattr(candidate, "finish_reason", None)
            finish_name = getattr(finish, "name", None) or (
                str(finish) if finish is not None else None
            )
            if finish_name:
                stop_reason = _STOP_REASON_MAP.get(finish_name, finish_name.lower())

            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for idx, part in enumerate(parts):
                part_text = getattr(part, "text", None)
                if part_text:
                    text_chunks.append(part_text)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    call_id = getattr(fc, "id", None) or f"call_{len(tool_calls)}"
                    raw_args = getattr(fc, "args", None) or {}
                    # Gemini 3.x attaches an opaque ``thought_signature`` to the
                    # Part carrying a function_call and rejects the next request
                    # (400 INVALID_ARGUMENT) unless it is sent back verbatim.
                    # Preserve it untouched on the ToolCall; may be absent.
                    tool_calls.append(
                        ToolCall(
                            id=str(call_id),
                            name=getattr(fc, "name", "") or "",
                            arguments=dict(raw_args),
                            provider_signature=getattr(
                                part, "thought_signature", None
                            ),
                        )
                    )

        if tool_calls:
            stop_reason = "tool_use"

        usage = self._extract_usage(getattr(response, "usage_metadata", None))
        model = getattr(response, "model_version", None) or self._model

        return LLMResponse(
            content="".join(text_chunks),
            tool_calls=tool_calls,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            stop_reason=stop_reason,
            model=model,
            raw_usage=usage["raw"],
            raw=response,
        )

    @staticmethod
    def _extract_usage(usage_metadata: Any) -> dict[str, Any]:
        """Map Gemini usage_metadata onto FORGE's normalised usage fields.

        Any field the provider does not report stays ``None``.
        """
        empty = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "raw": None,
        }
        if usage_metadata is None:
            return empty

        def _int(value: Any) -> int | None:
            return int(value) if isinstance(value, int) else None

        raw: dict[str, Any] | None = None
        try:
            dumped = usage_metadata.model_dump(exclude_none=True)  # pydantic model
            raw = {k: v for k, v in dumped.items() if isinstance(v, (int, float, str))}
        except Exception:  # noqa: BLE001 - raw usage is best-effort only
            raw = None

        return {
            "input_tokens": _int(getattr(usage_metadata, "prompt_token_count", None)),
            "output_tokens": _int(
                getattr(usage_metadata, "candidates_token_count", None)
            ),
            "total_tokens": _int(getattr(usage_metadata, "total_token_count", None)),
            "cached_input_tokens": _int(
                getattr(usage_metadata, "cached_content_token_count", None)
            ),
            "reasoning_tokens": _int(
                getattr(usage_metadata, "thoughts_token_count", None)
            ),
            "raw": raw,
        }
