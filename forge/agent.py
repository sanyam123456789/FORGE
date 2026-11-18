"""
forge.agent — the FORGE agent runtime.

``AgentRuntime.run()`` is the working agent loop:

    build context (system prompt + task [+ workspace file list])
      -> LLM call (with the currently-exposed tool schemas)
        -> if the model requested tool calls: validate + execute each,
           append results to context, loop
        -> else: the model's text is the final answer, stop

The loop enforces optional ceilings (``max_llm_turns``, ``max_tool_calls``;
0 = unlimited) and turns every failure mode — malformed tool call, unknown
tool, bad arguments, tool error, provider error — into a recorded event
rather than a crash.

Research seams (the loop depends only on the abstractions):
- ``AgentConfig.tool_exposure``  : a ``ToolExposureStrategy``.
- ``AgentConfig.context_strategy``: a ``ContextStrategy``.

Token savers built into the loop (identical for every strategy, so they do
not confound strategy comparisons):
- **Tool exposure is enforced, with a way to grow.** When a strategy hides
  tools, the request also carries a tiny ``enable_tool`` tool listing them;
  calling it offers that tool from the next turn on, so a strategy's wrong
  guess never blocks a task. A direct call to a hidden tool is refused (and
  the tool is enabled), never silently executed.
- **Stalls are flagged.** After several turns in a row that brought back
  nothing new, the model is told to finish or change approach.
- **Repeated identical reads collapse.** When a read-only call
  (read_file / list_directory / search_files) returns exactly what the same
  call returned earlier in the run, the model gets a one-line pointer to the
  earlier result instead of the full text again (once per call; a further
  repeat gets the full text, so nothing can be lost for good).
- **Repeated identical failures are flagged** so the model changes approach.
- **Cut-off replies are continued.** A reply stopped by the output-length
  limit is not treated as the final answer; the model is asked to continue.
- **Workspace file list.** The first message lists the workspace's files
  (``FORGE_REPO_MAP``), which saves exploratory rounds.

``AgentRuntime`` can also continue an existing ``ConversationContext``
(``context=``): that is how ``forge chat`` keeps one conversation across
several user messages.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from forge.config import settings
from forge.context import (
    DUPLICATE_RESULT_FLAG,
    ContextReport,
    ContextStrategy,
    ConversationContext,
    RawContextStrategy,
)
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

#: Read-only tools whose identical repeated results are collapsed.
_READ_ONLY_TOOLS = frozenset({"read_file", "list_directory", "search_files"})
#: An identical call failing this many times gets a "change approach" note.
_REPEATED_FAILURE_LIMIT = 3
#: Consecutive cut-off replies continued before giving up.
_MAX_LENGTH_CONTINUATIONS = 3
#: Consecutive turns with nothing new before the model gets a nudge.
_STALL_TURNS = 3

ENABLE_TOOL = "enable_tool"

_STALL_NOTE = (
    "[FORGE: your last {n} steps returned nothing you had not already seen. "
    "If the task is done, reply with your summary now. Otherwise take a "
    "different action (enable_tool can turn on a tool you need, if offered).]"
)


def enable_tool_schema(hidden: list[str]) -> dict[str, Any]:
    """Schema of the meta-tool that turns on a tool the strategy hid."""
    return {
        "type": "function",
        "function": {
            "name": ENABLE_TOOL,
            "description": (
                "Turn on a tool that is not offered for this task yet. "
                "Available to enable: " + ", ".join(hidden) + "."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(hidden)},
                },
                "required": ["name"],
            },
        },
    }

_CUT_OFF_NOTE = (
    "[FORGE: your last reply was cut off by the output length limit. "
    "Continue exactly where you stopped. If you were writing a large file, "
    "split it into smaller write_file / edit_file steps.]"
)


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
        Default: ``FixedToolExposure`` (all tools, every turn).
    context_strategy:
        Strategy deciding which messages are sent each turn.
        Default: ``RawContextStrategy`` (full history, unmodified).
    tool_subset:
        Optional static list of tool names to pin (still "fixed" exposure —
        no ranking / no per-turn variation).  Convenience for experiments.
    max_tool_calls / max_llm_turns:
        Per-run ceilings.  ``None`` falls back to the global settings; 0
        means unlimited.
    temperature / max_output_tokens:
        Generation params.  ``None`` falls back to the global settings.
    repo_map:
        Prepend the workspace file list to the first task message.
        ``None`` falls back to ``settings.repo_map``.
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
    repo_map: bool | None = None
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

    # Token-saver counters (Step 11).
    duplicate_reads_collapsed: int = 0
    hidden_tool_calls_refused: int = 0
    tools_enabled_on_request: int = 0
    stall_nudges: int = 0
    length_continuations: int = 0

    # Last request's state, for `forge chat` (/context, /tools).
    last_context_report: ContextReport | None = None
    last_exposed_tools: list[str] = field(default_factory=list)

    # Timing / artefacts.
    llm_latency_ms: float | None = None
    trace_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETED

    @property
    def cache_hit_rate(self) -> float | None:
        """Share of input tokens the provider served from its prompt cache."""
        if not self.input_tokens or self.cached_input_tokens is None:
            return None
        return self.cached_input_tokens / self.input_tokens


class AgentRuntime:
    """Coordinates one FORGE agent run.  Instantiate once per run.

    Pass ``context=`` to continue an existing conversation (``forge chat``):
    the task is then appended as a new user message instead of starting a
    fresh system-prompt + task context.
    """

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
        context: ConversationContext | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self._provider = provider
        self._registry = registry
        self._tracer = tracer
        self._workspace_root = workspace_root
        self._write_trace = write_trace
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._context = context
        self._on_tool_call = on_tool_call
        self._enabled_count = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> AgentRun:
        cfg = self.config
        run_id = _new_run_id()

        provider = self._provider or get_provider()
        registry = self._registry or _default_registry(self._workspace_root)
        tracer = self._tracer or RunTracer(run_id, cfg.task)

        max_turns = (
            cfg.max_llm_turns if cfg.max_llm_turns is not None else settings.max_llm_turns
        )
        max_tool_calls = (
            cfg.max_tool_calls if cfg.max_tool_calls is not None else settings.max_tool_calls
        )
        temperature = (
            cfg.temperature if cfg.temperature is not None else settings.llm_temperature
        )
        # 0 / None = no output cap: max_tokens is left out of the request.
        max_output_tokens = (cfg.max_output_tokens or settings.llm_max_output_tokens) or None

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
        workspace = Path(self._workspace_root or settings.workspace_root)
        if self._workspace_root is not None:
            # Create an explicitly requested workspace up front, so the first
            # list_directory / run_shell does not fail on a missing folder.
            workspace.mkdir(parents=True, exist_ok=True)

        if self._context is not None and self._context.message_count:
            ctx = self._context
            ctx.add_message(Message(role="user", content=cfg.task))
        else:
            ctx = self._context if self._context is not None else ConversationContext()
            ctx.add_message(Message(role="system", content=system_prompt))
            ctx.add_message(
                Message(role="user", content=self._first_message(cfg, workspace))
            )

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
        enabled_extra: list[str] = []  # hidden tools the model asked for
        read_results: dict[str, tuple[int, str]] = {}  # call key -> (turn, digest)
        collapsed_keys: set[str] = set()
        failures: dict[str, int] = {}
        seen_outputs: set[str] = set()
        stall_turns = 0
        answer_parts: list[str] = []
        try:
            turn = 0
            while True:
                if max_turns and turn >= max_turns:
                    run.status = STATUS_MAX_TURNS
                    break
                turn += 1
                run.turns = turn

                subset = tool_exposure.select(registry, ctx, cfg.task)
                if subset is not None and enabled_extra:
                    subset = list(subset) + [n for n in enabled_extra if n not in subset]
                exposed_names = (
                    list(subset) if subset is not None else registry.list_names()
                )
                schemas = registry.get_schemas(subset)
                hidden = [n for n in registry.list_names() if n not in exposed_names]
                offered = list(exposed_names)
                growth = tool_exposure.allows_growth
                if subset is not None and hidden and growth:
                    schemas = list(schemas) + [enable_tool_schema(hidden)]
                    offered.append(ENABLE_TOOL)
                prepared = cfg.context_strategy.prepare(ctx)
                context_report = cfg.context_strategy.last_report()
                run.last_context_report = context_report
                run.last_exposed_tools = offered

                tracer.record_llm_request(
                    turn=turn,
                    message_count=len(prepared),
                    exposed_tools=offered,
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

                cut_off = response.stop_reason == "length"

                if not response.has_tool_calls:
                    ctx.add_message(
                        Message(role="assistant", content=response.content)
                    )
                    answer_parts.append(response.content)
                    if cut_off and run.length_continuations < _MAX_LENGTH_CONTINUATIONS:
                        run.length_continuations += 1
                        ctx.add_message(Message(role="user", content=_CUT_OFF_NOTE))
                        continue
                    run.final_answer = "".join(answer_parts)
                    run.stop_reason = response.stop_reason
                    run.status = STATUS_COMPLETED
                    break
                answer_parts = []

                ctx.add_message(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                hit_tool_limit = False
                new_info = False
                for call in response.tool_calls:
                    if max_tool_calls and tool_calls_made >= max_tool_calls:
                        hit_tool_limit = True
                        break
                    tool_calls_made += 1
                    run.tool_calls += 1
                    tracer.record_tool_call(call.name, args=_safe_args(call))
                    if self._on_tool_call is not None:
                        try:
                            self._on_tool_call(call.name, _safe_args(call))
                        except Exception:  # noqa: BLE001 - display only
                            pass

                    metadata: dict[str, Any] = {}
                    if call.name == ENABLE_TOOL and ENABLE_TOOL not in registry:
                        result = self._enable_tool(
                            call, hidden if growth else [], enabled_extra
                        )
                        new_info = new_info or result.success
                    elif (
                        call.name
                        and call.name in registry
                        and call.name not in exposed_names
                    ):
                        # Enforce exposure: a hidden tool is never executed.
                        # Under a strategy that allows growth, offer it from
                        # the next turn on, so a wrong guess never makes a
                        # task impossible. A pinned list stays as pinned.
                        run.hidden_tool_calls_refused += 1
                        if growth:
                            enabled_extra.append(call.name)
                            result = ToolResult.from_error(
                                f"tool '{call.name}' was not offered for this task, "
                                "so it was not run. It is enabled from your next "
                                "step on; call it again if you still need it."
                            )
                        else:
                            result = ToolResult.from_error(
                                f"tool '{call.name}' is not available in this run. "
                                f"Available tools: {', '.join(exposed_names)}"
                            )
                    else:
                        result = self._dispatch_tool(registry, call)

                    output = result.output
                    key = _call_key(call)
                    if call.name in _READ_ONLY_TOOLS and result.success:
                        fingerprint = hashlib.sha256(output.encode("utf-8")).hexdigest()
                        if fingerprint not in seen_outputs:
                            seen_outputs.add(fingerprint)
                            new_info = True
                    elif call.name != ENABLE_TOOL and (
                        result.success or failures.get(key or "", 0) == 0
                    ):
                        # A change, a command, or a first-time failure is news.
                        new_info = True
                    if result.success and call.name in _READ_ONLY_TOOLS and key:
                        digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
                        earlier = read_results.get(key)
                        if (
                            earlier is not None
                            and earlier[1] == digest
                            and key not in collapsed_keys
                        ):
                            collapsed_keys.add(key)
                            run.duplicate_reads_collapsed += 1
                            output = (
                                f"[FORGE: identical to the result of this same "
                                f"{call.name} call in turn {earlier[0]} — nothing "
                                "changed since. Use that earlier result.]"
                            )
                            metadata[DUPLICATE_RESULT_FLAG] = True
                            result.metadata["duplicate_of_turn"] = earlier[0]
                        else:
                            read_results[key] = (turn, digest)
                    elif not result.success and key:
                        failures[key] = failures.get(key, 0) + 1
                        if failures[key] == _REPEATED_FAILURE_LIMIT:
                            output += (
                                f"\n[FORGE: this exact call has now failed "
                                f"{_REPEATED_FAILURE_LIMIT} times. Try a different "
                                "approach instead of repeating it.]"
                            )

                    tracer.record_tool_result(
                        call.name,
                        success=result.success,
                        error=result.error,
                        metadata=result.metadata,
                    )
                    ctx.add_message(
                        Message(
                            role="tool",
                            content=output,
                            tool_call_id=call.id,
                            name=call.name,
                            metadata=metadata,
                        )
                    )

                if hit_tool_limit:
                    run.status = STATUS_MAX_TOOL_CALLS
                    break
                stall_turns = 0 if new_info else stall_turns + 1
                if stall_turns and stall_turns % _STALL_TURNS == 0:
                    run.stall_nudges += 1
                    ctx.add_message(
                        Message(role="user", content=_STALL_NOTE.format(n=stall_turns))
                    )
                if cut_off:
                    # Tool calls in a cut-off reply may be incomplete.
                    run.length_continuations += 1
                    ctx.add_message(Message(role="user", content=_CUT_OFF_NOTE))

        except Exception as exc:  # noqa: BLE001 - the loop must always finish cleanly
            run.status = STATUS_FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            tracer.record_error(run.error)
        finally:
            run.tools_enabled_on_request = self._enabled_count
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
    # Helpers
    # ------------------------------------------------------------------

    def _enable_tool(
        self, call: ToolCall, hidden: list[str], enabled_extra: list[str]
    ) -> ToolResult:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        name = args.get("name")
        if not isinstance(name, str) or name not in hidden:
            return ToolResult.from_error(
                f"enable_tool: '{name}' is not a tool that can be enabled. "
                f"Choose one of: {', '.join(hidden) or '(none)'}"
            )
        enabled_extra.append(name)
        self._enabled_count += 1
        return ToolResult(
            output=f"Enabled {name}. It is offered from your next step on.",
            metadata={"enabled": name},
        )

    @staticmethod
    def _first_message(cfg: AgentConfig, workspace: Path) -> str:
        use_map = cfg.repo_map if cfg.repo_map is not None else settings.repo_map
        if not use_map:
            return cfg.task
        from forge.workspace_map import build_workspace_map

        overview = build_workspace_map(workspace)
        return f"{cfg.task}\n\n{overview}" if overview else cfg.task

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


def _call_key(call: ToolCall) -> str | None:
    """Stable identity of a tool call (name + arguments), or None."""
    if not call.name or not isinstance(call.arguments, dict):
        return None
    try:
        return call.name + ":" + json.dumps(call.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


def _default_registry(workspace_root: str | Path | None) -> ToolRegistry:
    # Imported here to keep forge.builtin_tools out of the import graph for
    # code that only needs the runtime types.
    from forge.builtin_tools import build_default_registry

    return build_default_registry(workspace_root=workspace_root)
