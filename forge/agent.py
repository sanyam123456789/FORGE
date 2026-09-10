"""
forge.agent — Agent runtime controller (Step 1: structural stub).

The agent runtime is the central coordinator of a FORGE agent run.
It owns the agent loop, orchestrates LLM calls, dispatches tool calls,
manages conversation context, and feeds the observability layer.

Step 1 Status
-------------
Only the ``AgentRun`` dataclass and ``AgentConfig`` are defined here.
The actual ``run()`` loop is NOT implemented — it will be built in Step 2
once the LLM provider adapters and concrete tools exist.

Architecture
------------
The agent loop (to be implemented) will follow this pattern:

    1. Build initial context (system prompt + user task).
    2. Call LLM with current context + active tool schemas.
    3. If the LLM emits a tool call:
       a. Safety-check the call.
       b. Execute via ToolRegistry.
       c. Append result to context.
       d. Increment tool call counter; enforce max_tool_calls ceiling.
       e. Go to step 2.
    4. If the LLM emits a final answer (stop_reason='stop'):
       a. Log the completed run.
       b. Return the AgentRun result.

Research Hooks
--------------
- Step 2 will add the ``tool_subset`` parameter for adaptive tool exposure.
- Step 3 will add the ``context_strategy`` parameter for managed context.
- The observability layer (forge.observability) will be called at each
  step to record timings, token usage, and tool invocation traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for a single agent run.

    Attributes
    ----------
    task:
        The user's task description (natural language).
    tool_subset:
        If set, only these tool names are exposed to the LLM.
        None means all registered tools are available.
        (Research Arm A: adaptive tool exposure.)
    context_strategy:
        Identifier for the context management strategy to apply.
        'raw' means the full interaction history is kept unchanged.
        Other values will select pruning/summarisation strategies.
        (Research Arm B: managed context.)
    max_tool_calls:
        Override for the global max_tool_calls setting.
    max_llm_turns:
        Override for the global max_llm_turns setting.
    """

    task: str
    tool_subset: list[str] | None = None
    context_strategy: str = "raw"
    max_tool_calls: int | None = None
    max_llm_turns: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    """Record of a completed (or failed) agent run.

    Populated by the agent runtime and returned to the caller.
    Also consumed by the observability layer for logging.
    """

    config: AgentConfig
    status: str = "not_started"        # not_started | running | completed | failed
    final_answer: str = ""
    error: str | None = None

    # Observability counters (populated during the run)
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_messages: int = 0

    # Future: wall-clock timings, per-tool call traces, etc.


class AgentRuntime:
    """Coordinates a single FORGE agent run.

    Instantiate once per run; not intended to be reused across runs.

    Step 1: constructor only — run() raises NotImplementedError.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def run(self) -> AgentRun:
        """Execute the agent loop.

        Not yet implemented.  Will be built in Step 2.
        """
        raise NotImplementedError(
            "AgentRuntime.run() is not implemented in Step 1. "
            "The agent loop will be added in Step 2 once LLM provider "
            "adapters and concrete tools are available."
        )
