"""
forge.observability — Run-level instrumentation for research experiments.

This module is the home for all FORGE measurement infrastructure.
It records the data that answers the research question:

  "Can adaptive tool exposure + managed context reduce token usage,
   inference cost, latency, and context consumption without sacrificing
   task success?"

Step 1 Status
-------------
- ``RunEvent`` dataclass: typed event vocabulary.
- ``RunTracer`` class: accumulates events for one agent run and writes
  them to a JSONL trace file at run end.
- Token counts and latency are NOT fabricated; fields are None until
  a real provider populates them.

Future Extensions (Step 3+)
---------------------------
- Aggregated experiment reports across multiple runs.
- Statistical comparison between adaptive vs fixed tool exposure.
- SWE-bench task success rate calculation.
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

LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
RUN_START = "run_start"
RUN_END = "run_end"
SAFETY_VIOLATION = "safety_violation"
ERROR = "error"


@dataclass
class RunEvent:
    """A single timestamped event within a FORGE agent run.

    Attributes
    ----------
    event_type:
        One of the event-type constants above.
    ts:
        UTC ISO-8601 timestamp.
    data:
        Arbitrary key-value payload (tool name, token counts, latency, etc.).
    """

    event_type: str
    ts: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run tracer
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Aggregated statistics for one completed agent run."""

    run_id: str
    task: str
    status: str
    llm_calls: int
    tool_calls: int
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_latency_ms: float | None
    tool_names_used: list[str]
    context_strategy: str
    tool_subset: list[str] | None
    error: str | None


class RunTracer:
    """Instruments a single FORGE agent run.

    Usage
    -----
        tracer = RunTracer(run_id="run_001", task="Fix the bug in foo.py")
        tracer.record_run_start(config)

        # ... inside agent loop ...
        tracer.record_llm_call(input_tokens=500, output_tokens=200, latency_ms=1200)
        tracer.record_tool_call("read_file", args={"path": "foo.py"})
        tracer.record_tool_result("read_file", success=True)

        tracer.record_run_end(status="completed", final_answer="Done.")
        tracer.flush(log_dir=Path("logs"))
    """

    def __init__(self, run_id: str, task: str) -> None:
        self.run_id = run_id
        self.task = task
        self._events: list[RunEvent] = []
        self._start_time: float | None = None

        # Running counters
        self._llm_calls: int = 0
        self._tool_calls: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._tool_names_used: list[str] = []
        self._has_token_data: bool = False

    # ------------------------------------------------------------------
    # Event recorders
    # ------------------------------------------------------------------

    def record_run_start(self, config: Any) -> None:
        self._start_time = time.monotonic()
        self._events.append(RunEvent(
            event_type=RUN_START,
            data={
                "run_id": self.run_id,
                "task_preview": self.task[:200],
                "tool_subset": getattr(config, "tool_subset", None),
                "context_strategy": getattr(config, "context_strategy", "raw"),
            },
        ))
        logger.info("Run started: %s", self.run_id)

    def record_llm_call(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
        stop_reason: str = "stop",
    ) -> None:
        self._llm_calls += 1
        if input_tokens is not None:
            self._input_tokens += input_tokens
            self._has_token_data = True
        if output_tokens is not None:
            self._output_tokens += output_tokens
        self._events.append(RunEvent(
            event_type=LLM_CALL,
            data={
                "call_number": self._llm_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "stop_reason": stop_reason,
            },
        ))

    def record_tool_call(self, tool_name: str, args: dict[str, Any] | None = None) -> None:
        self._tool_calls += 1
        self._tool_names_used.append(tool_name)
        self._events.append(RunEvent(
            event_type=TOOL_CALL,
            data={
                "tool_name": tool_name,
                "call_number": self._tool_calls,
                "args_keys": list(args.keys()) if args else [],
            },
        ))

    def record_tool_result(
        self,
        tool_name: str,
        *,
        success: bool,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(RunEvent(
            event_type=TOOL_RESULT,
            data={
                "tool_name": tool_name,
                "success": success,
                "error": error,
                **(metadata or {}),
            },
        ))

    def record_safety_violation(self, message: str) -> None:
        self._events.append(RunEvent(
            event_type=SAFETY_VIOLATION,
            data={"message": message},
        ))
        logger.warning("Safety violation during run %s: %s", self.run_id, message)

    def record_error(self, message: str) -> None:
        self._events.append(RunEvent(
            event_type=ERROR,
            data={"message": message},
        ))
        logger.error("Run error %s: %s", self.run_id, message)

    def record_run_end(self, *, status: str, final_answer: str = "", error: str | None = None) -> None:
        elapsed_ms = (
            (time.monotonic() - self._start_time) * 1000
            if self._start_time is not None
            else None
        )
        self._events.append(RunEvent(
            event_type=RUN_END,
            data={
                "run_id": self.run_id,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "llm_calls": self._llm_calls,
                "tool_calls": self._tool_calls,
                "input_tokens": self._input_tokens if self._has_token_data else None,
                "output_tokens": self._output_tokens if self._has_token_data else None,
                "error": error,
            },
        ))
        logger.info(
            "Run ended: %s status=%s llm_calls=%d tool_calls=%d",
            self.run_id, status, self._llm_calls, self._tool_calls,
        )

    # ------------------------------------------------------------------
    # Summary and persistence
    # ------------------------------------------------------------------

    def summary(self, *, status: str = "unknown", context_strategy: str = "raw",
                tool_subset: list[str] | None = None, error: str | None = None) -> RunSummary:
        """Return a structured summary of this run's measurements."""
        return RunSummary(
            run_id=self.run_id,
            task=self.task,
            status=status,
            llm_calls=self._llm_calls,
            tool_calls=self._tool_calls,
            total_input_tokens=self._input_tokens if self._has_token_data else None,
            total_output_tokens=self._output_tokens if self._has_token_data else None,
            total_latency_ms=None,  # computed from events when needed
            tool_names_used=list(self._tool_names_used),
            context_strategy=context_strategy,
            tool_subset=tool_subset,
            error=error,
        )

    def flush(self, log_dir: Path) -> Path:
        """Write all events to a JSONL trace file and return its path.

        Each line in the file is one RunEvent serialised as JSON.
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        trace_file = log_dir / f"{self.run_id}.jsonl"

        with trace_file.open("w", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        logger.info("Run trace written: %s (%d events)", trace_file, len(self._events))
        return trace_file
