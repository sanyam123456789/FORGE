"""
forge.observability — run-level instrumentation for research experiments.

``RunTracer`` accumulates typed events for a single agent run and writes them
to a JSONL trace file.  It is the data source for the project's research
question (token / cost / latency effects of tool-exposure and context
strategies), but it does no analysis itself and never fabricates a metric:
a value the provider did not report is recorded as ``None``, not ``0``.

Event vocabulary
----------------
run_start | llm_call | llm_response | tool_call | tool_result |
safety_violation | error | run_end
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

RUN_START = "run_start"
LLM_CALL = "llm_call"          # a request was sent to the provider
LLM_RESPONSE = "llm_response"  # a response (and its usage) came back
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
SAFETY_VIOLATION = "safety_violation"
ERROR = "error"
RUN_END = "run_end"


def _add(acc: int | None, value: int | None) -> int | None:
    """Sum two optional counters, preserving 'unknown' (None) semantics.

    None + None -> None ; None + 5 -> 5 ; 5 + None -> 5 ; 5 + 3 -> 8
    """
    if value is None:
        return acc
    return (acc or 0) + value


@dataclass
class RunEvent:
    """A single timestamped event within a FORGE agent run."""

    event_type: str
    ts: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    """Aggregated statistics for one completed agent run."""

    run_id: str
    task: str
    status: str
    provider: str | None
    model: str | None
    llm_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    total_latency_ms: float | None
    tool_names_used: list[str]
    tool_exposure_strategy: str | None
    tool_subset: list[str] | None
    context_strategy: str | None
    system_prompt_id: str | None
    error: str | None


# ---------------------------------------------------------------------------
# Run tracer
# ---------------------------------------------------------------------------


class RunTracer:
    """Instruments a single FORGE agent run.

    Usage
    -----
        tracer = RunTracer(run_id="run_001", task="...")
        tracer.record_run_start(provider="gemini", model="gemini-2.0-flash", ...)
        tracer.record_llm_request(turn=1, message_count=2, exposed_tools=[...])
        tracer.record_llm_response(input_tokens=500, output_tokens=200, ...)
        tracer.record_tool_call("read_file", args={"path": "foo.py"})
        tracer.record_tool_result("read_file", success=True, metadata={...})
        tracer.record_run_end(status="completed", final_answer="Done.")
        tracer.flush(log_dir=Path("logs"))
    """

    def __init__(self, run_id: str, task: str) -> None:
        self.run_id = run_id
        self.task = task
        self._events: list[RunEvent] = []
        self._start_time: float | None = None

        # Experiment metadata (set by record_run_start).
        self._meta: dict[str, Any] = {}

        # Running counters.
        self._llm_calls: int = 0
        self._tool_calls: int = 0
        self._tool_names_used: list[str] = []

        # Token accumulators — each stays None until the provider reports it,
        # tracked independently so a provider that reports input but not output
        # does not make output look like a real zero.
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._total_tokens: int | None = None
        self._cached_input_tokens: int | None = None
        self._reasoning_tokens: int | None = None

        # Latency accumulator (only counts turns that reported a latency).
        self._latency_ms: float | None = None

    # ------------------------------------------------------------------
    # Event recorders
    # ------------------------------------------------------------------

    def record_run_start(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        system_prompt_id: str | None = None,
        tool_exposure_strategy: str | None = None,
        tool_subset: list[str] | None = None,
        context_strategy: str | None = None,
        max_llm_turns: int | None = None,
        max_tool_calls: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._start_time = time.monotonic()
        self._meta = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "system_prompt_id": system_prompt_id,
            "tool_exposure_strategy": tool_exposure_strategy,
            "tool_subset": tool_subset,
            "context_strategy": context_strategy,
            "max_llm_turns": max_llm_turns,
            "max_tool_calls": max_tool_calls,
        }
        if extra:
            self._meta.update(extra)
        self._events.append(
            RunEvent(
                event_type=RUN_START,
                data={
                    "run_id": self.run_id,
                    "task_preview": self.task[:200],
                    **self._meta,
                },
            )
        )
        logger.info("Run started: %s (provider=%s model=%s)", self.run_id, provider, model)

    def record_llm_request(
        self,
        *,
        turn: int,
        message_count: int,
        exposed_tools: list[str] | None = None,
    ) -> None:
        self._events.append(
            RunEvent(
                event_type=LLM_CALL,
                data={
                    "turn": turn,
                    "message_count": message_count,
                    "exposed_tool_count": len(exposed_tools) if exposed_tools is not None else None,
                    "exposed_tools": exposed_tools,
                },
            )
        )

    def record_llm_response(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        latency_ms: float | None = None,
        stop_reason: str = "stop",
        tool_call_count: int = 0,
    ) -> None:
        self._llm_calls += 1
        self._input_tokens = _add(self._input_tokens, input_tokens)
        self._output_tokens = _add(self._output_tokens, output_tokens)
        self._total_tokens = _add(self._total_tokens, total_tokens)
        self._cached_input_tokens = _add(self._cached_input_tokens, cached_input_tokens)
        self._reasoning_tokens = _add(self._reasoning_tokens, reasoning_tokens)
        if latency_ms is not None:
            self._latency_ms = (self._latency_ms or 0.0) + latency_ms
        self._events.append(
            RunEvent(
                event_type=LLM_RESPONSE,
                data={
                    "call_number": self._llm_calls,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "latency_ms": latency_ms,
                    "stop_reason": stop_reason,
                    "tool_call_count": tool_call_count,
                },
            )
        )

    def record_tool_call(self, tool_name: str, args: dict[str, Any] | None = None) -> None:
        self._tool_calls += 1
        self._tool_names_used.append(tool_name)
        self._events.append(
            RunEvent(
                event_type=TOOL_CALL,
                data={
                    "tool_name": tool_name,
                    "call_number": self._tool_calls,
                    "arg_keys": sorted(args.keys()) if args else [],
                },
            )
        )

    def record_tool_result(
        self,
        tool_name: str,
        *,
        success: bool,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            RunEvent(
                event_type=TOOL_RESULT,
                data={
                    "tool_name": tool_name,
                    "success": success,
                    "error": error,
                    "metadata": metadata or {},
                },
            )
        )

    def record_safety_violation(self, message: str) -> None:
        self._events.append(
            RunEvent(event_type=SAFETY_VIOLATION, data={"message": message})
        )
        logger.warning("Safety violation during run %s: %s", self.run_id, message)

    def record_error(self, message: str) -> None:
        self._events.append(RunEvent(event_type=ERROR, data={"message": message}))
        logger.error("Run error %s: %s", self.run_id, message)

    def record_run_end(
        self, *, status: str, final_answer: str = "", error: str | None = None
    ) -> None:
        elapsed_ms = (
            (time.monotonic() - self._start_time) * 1000
            if self._start_time is not None
            else None
        )
        self._events.append(
            RunEvent(
                event_type=RUN_END,
                data={
                    "run_id": self.run_id,
                    "status": status,
                    "elapsed_ms": elapsed_ms,
                    "llm_calls": self._llm_calls,
                    "tool_calls": self._tool_calls,
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                    "total_tokens": self.resolved_total_tokens,
                    "cached_input_tokens": self._cached_input_tokens,
                    "reasoning_tokens": self._reasoning_tokens,
                    "llm_latency_ms": self._latency_ms,
                    "final_answer_preview": final_answer[:200],
                    "error": error,
                },
            )
        )
        logger.info(
            "Run ended: %s status=%s llm_calls=%d tool_calls=%d",
            self.run_id, status, self._llm_calls, self._tool_calls,
        )

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------

    @property
    def resolved_total_tokens(self) -> int | None:
        """Provider-reported total if seen, else input+output, else None."""
        if self._total_tokens is not None:
            return self._total_tokens
        if self._input_tokens is not None and self._output_tokens is not None:
            return self._input_tokens + self._output_tokens
        return None

    # ------------------------------------------------------------------
    # Summary and persistence
    # ------------------------------------------------------------------

    def summary(self, *, status: str = "unknown", error: str | None = None) -> RunSummary:
        return RunSummary(
            run_id=self.run_id,
            task=self.task,
            status=status,
            provider=self._meta.get("provider"),
            model=self._meta.get("model"),
            llm_calls=self._llm_calls,
            tool_calls=self._tool_calls,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self.resolved_total_tokens,
            cached_input_tokens=self._cached_input_tokens,
            reasoning_tokens=self._reasoning_tokens,
            total_latency_ms=self._latency_ms,
            tool_names_used=list(self._tool_names_used),
            tool_exposure_strategy=self._meta.get("tool_exposure_strategy"),
            tool_subset=self._meta.get("tool_subset"),
            context_strategy=self._meta.get("context_strategy"),
            system_prompt_id=self._meta.get("system_prompt_id"),
            error=error,
        )

    def flush(self, log_dir: Path) -> Path:
        """Write all events to ``<log_dir>/<run_id>.jsonl`` and return the path."""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        trace_file = log_dir / f"{self.run_id}.jsonl"
        with trace_file.open("w", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        logger.info("Run trace written: %s (%d events)", trace_file, len(self._events))
        return trace_file
