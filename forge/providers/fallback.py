"""
forge.providers.fallback — provider-agnostic ordered fallback.

``FallbackProvider`` composes an ordered list of concrete ``LLMProvider``
adapters (e.g. opencode_go -> openai_compatible) and itself implements
``LLMProvider``, so nothing outside ``forge/providers/`` needs to change:
``AgentRuntime`` calls ``provider.complete()`` exactly as it always has and
has no idea whether a fallback happened underneath.

Fallback rules (see docs on ``complete()`` for the full decision table):

- A candidate that is simply *not configured* (its own ``preflight()``
  raises ``ValueError`` — missing API key or missing model) is logged and
  skipped, never silently. If every candidate in the order is unconfigured,
  the final error clearly lists each one and why.
- A candidate that raises ``LLMError(retryable=True)`` (rate limit, quota
  exhaustion, temporary server error, connection/timeout failure) is logged
  and the next candidate is tried.
- A candidate that raises ``LLMError(retryable=False)`` (authentication
  failure, malformed request, invalid tool schema, any other non-transient
  API error) is **not** a fallback trigger: it is re-raised immediately and
  the chain stops there. This is deliberate — those failures would occur
  identically on any provider, so silently trying the next one would only
  hide a real, actionable problem (a bad key, a bug in our own request
  building, ...).
- Each candidate is tried at most once per ``complete()`` call: bounded,
  no cross-provider retry loops. (Each adapter still does its own small
  internal retry for transient errors before giving up and letting
  FallbackProvider move on — see forge/providers/openai_compat.py.
  This is intentionally the only retry layer; FallbackProvider
  itself never retries a candidate.)

Run-scoped stickiness
---------------------
``AgentRuntime.run()`` constructs one provider (via ``get_provider()``) per
run and reuses that same instance for every turn (see forge/agent.py). Once
a candidate fails here with a retryable error, or is found unconfigured,
it is marked "down for the rest of this ``FallbackProvider`` instance's
life" — i.e. for the rest of this one agent run. Subsequent ``complete()``
calls within the same run skip it immediately, without re-attempting its
own internal retry/backoff. This is what prevents a provider whose quota
is exhausted for the day from re-paying its full retry-backoff cost on
every single LLM turn of a multi-turn run.

This is deliberately *not* a time-based cooldown (no timer, no expiry, no
extra sleep): a fresh run builds a fresh ``FallbackProvider`` via
``get_provider()`` and always starts clean, retrying every candidate from
the top of the order again. Down-marking never crosses run boundaries and
never affects a different run's provider choice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from forge.llm import LLMError, LLMProvider, LLMResponse, Message
from forge.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _Candidate:
    name: str
    provider: LLMProvider


class FallbackProvider(LLMProvider):
    """Tries an ordered list of provider adapters; see module docstring."""

    def __init__(self, providers: list[tuple[str, LLMProvider]]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider.")
        self._candidates = [_Candidate(name, p) for name, p in providers]
        # Reported by provider_name/model_name before any call is made, and
        # updated to whichever candidate actually served the most recent
        # complete() call.
        self._active = self._candidates[0]
        # Names marked unusable for the rest of THIS instance's life (i.e.
        # the rest of one agent run — see module docstring "Run-scoped
        # stickiness"). Never reset within a run; a new run gets a new
        # FallbackProvider (and therefore a fresh, empty set) via
        # get_provider().
        self._down: set[str] = set()

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self._active.name

    @property
    def model_name(self) -> str:
        return self._active.provider.model_name

    def preflight(self) -> None:
        """Fail fast only if EVERY candidate is unusable.

        Unlike a single adapter's preflight(), this does not require the
        *first* candidate to be configured — only that the chain has at
        least one usable provider. If the chain has exactly one candidate
        (a single-provider FORGE_PROVIDER_ORDER) this reduces to the same
        "fail loud if its key/model is missing" behavior as today.
        """
        reasons: list[str] = []
        for cand in self._candidates:
            try:
                cand.provider.preflight()
            except ValueError as exc:
                reasons.append(f"{cand.name}: {exc}")
                logger.warning(
                    "Provider '%s' is not configured (%s); it will be "
                    "skipped if reached during fallback.",
                    cand.name,
                    exc,
                )
                continue
            return  # at least one candidate is usable
        raise ValueError(
            "No configured provider is usable in FORGE_PROVIDER_ORDER "
            f"{[c.name for c in self._candidates]}. " + " | ".join(reasons)
        )

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        failures: list[str] = []

        for idx, cand in enumerate(self._candidates):
            if cand.name in self._down:
                logger.debug(
                    "Provider '%s' is marked down for the rest of this run; "
                    "skipping without retrying it.",
                    cand.name,
                )
                failures.append(f"{cand.name}: down for this run (see earlier log)")
                continue

            try:
                cand.provider.preflight()
            except ValueError as exc:
                logger.warning(
                    "Provider '%s' unavailable (%s); skipping to the next "
                    "provider in the fallback order and marking it down "
                    "for the rest of this run.",
                    cand.name,
                    exc,
                )
                self._down.add(cand.name)
                failures.append(f"{cand.name}: not configured ({exc})")
                continue

            t0 = time.monotonic()
            try:
                response = cand.provider.complete(
                    messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except LLMError as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                if not exc.retryable:
                    logger.error(
                        "Provider '%s' failed with a non-recoverable error "
                        "after %.0fms; not falling back (this would fail "
                        "identically on any provider). reason=%s",
                        cand.name,
                        elapsed_ms,
                        exc,
                    )
                    raise
                logger.warning(
                    "Provider '%s' failed with a recoverable error after "
                    "%.0fms; falling back to the next provider in the "
                    "order and marking it down for the rest of this run "
                    "(subsequent turns will not re-pay its retry cost). "
                    "reason=%s",
                    cand.name,
                    elapsed_ms,
                    exc,
                )
                self._down.add(cand.name)
                failures.append(f"{cand.name}: {exc}")
                continue

            elapsed_ms = (time.monotonic() - t0) * 1000
            self._active = cand
            if idx > 0:
                logger.info(
                    "Fallback succeeded: provider '%s' served this request "
                    "in %.0fms after %d earlier provider(s) failed or were "
                    "unavailable.",
                    cand.name,
                    elapsed_ms,
                    idx,
                )
            return response

        raise LLMError(
            "All configured providers failed or were unavailable: "
            + " | ".join(failures),
            retryable=False,
        )
