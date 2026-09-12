"""
forge.agent — the FORGE agent runtime.

``AgentRuntime.run()`` is the working agent loop:

    build context (system prompt + task)
      -> LLM call (with the currently-exposed tool schemas)
        -> if the model requested tool calls: validate + execute each,
           append results to context, loop
        -> else: the model's text is the final answer, stop

The loop enforces hard ceilings (``max_llm_turns``, ``max_tool_calls``) and
turns every failure mode — malformed tool call, unknown tool, bad arguments,
tool error, provider error — into a recorded event rather than a crash.

Research seams (NOT exercised in Step 2, but the loop already depends only on
the abstractions):
- ``AgentConfig.tool_exposure``  : a ``ToolExposureStrategy`` (Step 2 = Fixed).
- ``AgentConfig.context_strategy``: a ``ContextStrategy``      (Step 2 = Raw).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.config import settings
from forge.context import ContextStrategy, ConversationContext, RawContextStrategy
from forge.llm import LLMError, LLMProvider, Message, ToolCall, get_provider
from forge.observability import RunTracer, _add
from forge.prompts import FORGE_SYSTEM_PROMPT, system_prompt_id
from forge.tools import FixedToolExposure, ToolExposureStrategy, ToolRegistry, ToolResult

# Terminal run statuses.
STATUS_COMPLETED = "completed"
STATUS_MAX_TURNS = "max_turns_exceeded"
STATUS_MAX_TOOL_CALLS = "max_tool_calls_exceeded"
STATUS_PROVIDER_ERROR = "provider_error"
STATUS_FAILED = "failed"


@dataclass
class AgentConfig:
    """Configuration for a single agent run.

    task:
        The user's task description (natural language).
    system_prompt:
        Override for the system prompt.  ``None`` uses the canonical
        ``FORGE_SYSTEM_PROMPT``.
    tool_exposure:
        Strategy deciding which registered tools are exposed each turn.
        Step 2 default: ``FixedToolExposure`` (all tools, every turn).
    context_strategy:
        Strategy deciding which messages are sent each turn.
        Step 2 default: ``RawContextStrategy`` (full history, unmodified).
    tool_subset:
        Optional static list of tool names to pin (still "fixed" exposure —
        no ranking / no per-turn variation).  Convenience for experiments.
    max_tool_calls / max_llm_turns:
        Per-run ceilings.  ``None`` falls back to the global settings.
    temperature / max_output_tokens:
        Generation params.  ``None`` falls back to the global settings.
    """

    task: str
    system_prompt: str | None = None
    tool_exposure: ToolExposureStrategy = field(default_factory=FixedToolExposure)
    context_strategy: ContextStrategy = field(default_factory=RawContextStrategy)
    tool_subset: list[str] | None = None
    max_tool_calls: int | None = None
    max_llm_turns: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    """Record of a completed (or failed) agent run."""

    config: AgentConfig
    run_id: str
    status: str = "not_started"
    final_answer: str = ""
    error: str | None = None
    stop_reason: str | None = None

    # Counters.
    llm_calls: int = 0
    tool_calls: int = 0
    turns: int = 0

    # Provider-reported usage — None means "not reported", never fabricated.
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    # Context size proxies.
    context_messages: int = 0
    context_char_count: int = 0

    # Managed-context stats (Step 6) — cumulative across turns, from the
    # context strategy's own ContextReport. None when the strategy never
    # reported one (see RunTracer.context_items_dropped_total).
    context_items_dropped: int | None = None
    context_items_compressed: int | None = None
    context_chars_saved: int | None = None

    # Timing / artefacts.
    llm_latency_ms: float | None = None
    trace_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETED


class AgentRuntime:
    """Coordinates a single FORGE agent run.  Instantiate once per run."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        provider: LLMProvider | None = None,
        registry: ToolRegistry | None = None,
        tracer: RunTracer | None = None,
        workspace_root: str | Path | None = None,
        write_trace: bool = True,
        log_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self._provider = provider
        self._registry = registry
        self._tracer = tracer
        self._workspace_root = workspace_root
        self._write_trace = write_trace
        self._log_dir = Path(log_dir) if log_dir is not None else None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> AgentRun:
        cfg = self.config
        run_id = _new_run_id()

        provider = self._provider or get_provider()
        registry = self._registry or _default_registry(self._workspace_root)
        tracer = self._tracer or RunTracer(run_id, cfg.task)

        max_turns = cfg.max_llm_turns or settings.max_llm_turns
        max_tool_calls = cfg.max_tool_calls or settings.max_tool_calls
        temperature = (
            cfg.temperature if cfg.temperature is not None else settings.llm_temperature
        )
        max_output_tokens = cfg.max_output_tokens or settings.llm_max_output_tokens

        # ``tool_subset`` on the config is a convenience: if the user pinned a
        # static list but left the default FixedToolExposure, wire it in. This
        # is still fixed exposure (no ranking, no per-turn variation).
        tool_exposure = cfg.tool_exposure
        if (
            cfg.tool_subset is not None
            and isinstance(tool_exposure, FixedToolExposure)
            and tool_exposure.pinned is None
        ):
            tool_exposure = FixedToolExposure(pinned=cfg.tool_subset)

        system_prompt = cfg.system_prompt or FORGE_SYSTEM_PROMPT
        ctx = ConversationContext()
        ctx.add_message(Message(role="system", content=system_prompt))
        ctx.add_message(Message(role="user", content=cfg.task))

        run = AgentRun(config=cfg, run_id=run_id, status="running")

        initial_subset = tool_exposure.select(registry, ctx, cfg.task)
        tracer.record_run_start(
            provider=provider.provider_name,
            model=provider.model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_prompt_id=system_prompt_id(system_prompt),
            tool_exposure_strategy=tool_exposure.name,
            tool_subset=initial_subset,
            context_strategy=cfg.context_strategy.name,
            max_llm_turns=max_turns,
            max_tool_calls=max_tool_calls,
        )

        tool_calls_made = 0
        try:
            turn = 0
            while True:
                if turn >= max_turns:
                    run.status = STATUS_MAX_TURNS
                    break
                turn += 1
                run.turns = turn

                subset = tool_exposure.select(registry, ctx, cfg.task)
                exposed_names = (
                    subset if subset is not None else registry.list_names()
                )
                schemas = registry.get_schemas(subset)
                prepared = cfg.context_strategy.prepare(ctx)
                context_report = cfg.context_strategy.last_report()

                tracer.record_llm_request(
                    turn=turn,
                    message_count=len(prepared),
                    exposed_tools=exposed_names,
                    context_items_before=(
                        context_report.items_before if context_report else None
                    ),
                    context_items_after=(
                        context_report.items_after if context_report else None
                    ),
                    context_chars_before=(
                        context_report.chars_before if context_report else None
                    ),
                    context_chars_after=(
                        context_report.chars_after if context_report else None
                    ),
                    context_items_dropped=(
                        context_report.items_dropped if context_report else None
                    ),
                    context_items_compressed=(
                        context_report.items_compressed if context_report else None
                    ),
                )

                try:
                    t0 = time.monotonic()
                    response = provider.complete(
                        prepared,
                        tools=schemas or None,
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                    latency_ms = (time.monotonic() - t0) * 1000
                except (LLMError, ValueError) as exc:
                    tracer.record_error(f"provider error: {exc}")
                    run.status = STATUS_PROVIDER_ERROR
                    run.error = str(exc)
                    break

                tracer.record_llm_response(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    cached_input_tokens=response.cached_input_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                    latency_ms=latency_ms,
                    stop_reason=response.stop_reason,
                    tool_call_count=len(response.tool_calls),
                )
                run.llm_calls += 1
                run.input_tokens = _add(run.input_tokens, response.input_tokens)
                run.output_tokens = _add(run.output_tokens, response.output_tokens)
                run.total_tokens = _add(run.total_tokens, response.total_tokens)
                run.cached_input_tokens = _add(
                    run.cached_input_tokens, response.cached_input_tokens
                )
                run.reasoning_tokens = _add(
                    run.reasoning_tokens, response.reasoning_tokens
                )
                run.llm_latency_ms = (run.llm_latency_ms or 0.0) + latency_ms

                if not response.has_tool_calls:
                    ctx.add_message(
                        Message(role="assistant", content=response.content)
                    )
                    run.final_answer = response.content
                    run.stop_reason = response.stop_reason
                    run.status = STATUS_COMPLETED
                    break

                ctx.add_message(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                hit_tool_limit = False
                for call in response.tool_calls:
                    if tool_calls_made >= max_tool_calls:
                        hit_tool_limit = True
                        break
                    tool_calls_made += 1
                    run.tool_calls += 1
                    tracer.record_tool_call(call.name, args=_safe_args(call))
                    result = self._dispatch_tool(registry, call)
                    tracer.record_tool_result(
                        call.name,
                        success=result.success,
                        error=result.error,
                        metadata=result.metadata,
                    )
                    ctx.add_message(
                        Message(
                            role="tool",
                            content=result.output,
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )

                if hit_tool_limit:
                    run.status = STATUS_MAX_TOOL_CALLS
                    break

        except Exception as exc:  # noqa: BLE001 - the loop must always finish cleanly
            run.status = STATUS_FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            tracer.record_error(run.error)
        finally:
            run.context_messages = ctx.message_count
            run.context_char_count = ctx.approximate_char_count
            run.context_items_dropped = tracer.context_items_dropped_total
            run.context_items_compressed = tracer.context_items_compressed_total
            run.context_chars_saved = tracer.context_chars_saved_total
            tracer.record_run_end(
                status=run.status,
                final_answer=run.final_answer,
                error=run.error,
            )
            if self._write_trace:
                run.trace_path = tracer.flush(self._log_dir or settings.log_dir)

        return run

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _dispatch_tool(registry: ToolRegistry, call: ToolCall) -> ToolResult:
        if not isinstance(call.arguments, dict):
            return ToolResult.from_error(
                f"tool arguments must be a JSON object, got "
                f"{type(call.arguments).__name__}"
            )
        if not call.name:
            return ToolResult.from_error("tool call is missing a tool name")
        try:
            tool = registry.get(call.name)
        except KeyError:
            return ToolResult.from_error(
                f"unknown tool '{call.name}'. Available tools: {registry.list_names()}"
            )
        try:
            return tool.execute(**call.arguments)
        except TypeError as exc:
            return ToolResult.from_error(
                f"invalid arguments for '{call.name}': {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - defend against 3rd-party tools
            return ToolResult.from_error(
                f"tool '{call.name}' raised {type(exc).__name__}: {exc}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_run_id() -> str:
    return "run_" + datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%f")


def _safe_args(call: ToolCall) -> dict[str, Any]:
    return call.arguments if isinstance(call.arguments, dict) else {}


def _default_registry(workspace_root: str | Path | None) -> ToolRegistry:
    # Imported here to keep forge.builtin_tools out of the import graph for
    # code that only needs the runtime types.
    from forge.builtin_tools import build_default_registry

    return build_default_registry(workspace_root=workspace_root)
