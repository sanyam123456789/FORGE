# FORGE Architecture

## What is FORGE?

FORGE is a deliberately lightweight, modular coding-agent harness built for a
final-year engineering research project.  It allows us to instrument, modify,
and experimentally evaluate an LLM-based software engineering agent.

The key distinction from existing agents (SWE-agent, OpenHands, etc.) is that
FORGE is **built to be understood and measured**, not to be the most capable
agent in production use.

---

## Research Question

> Can a coding agent achieve comparable software-engineering task performance
> with lower token usage, inference cost, latency, and context consumption
> through **adaptive tool exposure** and **interaction-context management**?

---

## Agent Loop (target, Step 2+)

```
┌─────────────────────────────────────────────────────────────┐
│                        User Task                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Agent Runtime (forge.agent)                 │
│  Coordinates: context ↔ LLM ↔ tools ↔ observability        │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Context Manager (forge.context)                │
│  Research Arm B:                                            │
│   [raw] full history  vs  [managed] pruned/summarised       │
└──────┬──────────────────────────────────────────────────────┘
       │ messages
       ▼
┌─────────────────────────────────────────────────────────────┐
│             LLM Provider Interface (forge.llm)              │
│  Normalised: Message → LLMResponse + token counts          │
│  Concrete adapters: OpenAI | Anthropic | Google             │
└──────┬──────────────────────────────────────────────────────┘
       │ tool schemas
       │ ┌──────────────────────────────────────────────────────┐
       │ │        Tool Registry (forge.tools)                   │
       │ │  Research Arm A:                                     │
       │ │   [fixed] all tools  vs  [adaptive] subset selected  │
       │ └──────────────────────┬───────────────────────────────┘
       │                        │ tool call
       ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Tool Execution (forge/tools/*.py)              │
│  read_file | write_file | list_dir | run_shell | …          │
└──────┬──────────────────────────────────────────────────────┘
       │ every operation calls ↓
┌─────────────────────────────────────────────────────────────┐
│               Safety Layer (forge.safety)                   │
│  Workspace boundary | overwrite guard | command blocklist   │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│           Observability Layer (forge.observability)         │
│  RunTracer: LLM calls | tool calls | tokens | latency       │
│  Output: per-run JSONL trace for research analysis          │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `forge.config` | Load and validate all settings from environment variables. No hard-coded secrets ever. |
| `forge.logging` | Initialise the forge.* logging hierarchy. Text and JSON formatters. |
| `forge.safety` | Enforce workspace boundaries and command restrictions before any file/shell operation. |
| `forge.llm` | Define the abstract LLM interface. Concrete adapters sit in `forge/llm/`. |
| `forge.tools` | Define the `Tool` ABC and `ToolRegistry`. The registry's `get_schemas(subset)` is the adaptive exposure hook. |
| `forge.context` | Hold the live conversation history for a run. Future: apply pruning/summarisation strategies. |
| `forge.agent` | Coordinate the agent loop (Step 2). Own the run lifecycle and enforce ceilings. |
| `forge.observability` | Record typed events per run. Flush to JSONL for analysis. Never fabricate measurements. |

---

## Research Arms

### Arm A — Tool Exposure Strategy

**Variable:** which tool schemas are passed to the LLM on each turn.

| Condition | Description |
|---|---|
| Fixed | All registered tools always exposed (baseline) |
| Adaptive | Only tools relevant to current task phase exposed |

**Implementation hook:** `ToolRegistry.get_schemas(subset=None | [names])` in `forge/tools.py`.

### Arm B — Context Management Strategy

**Variable:** how the interaction history is prepared before each LLM call.

| Condition | Description |
|---|---|
| Raw | Full history passed unchanged |
| Managed | History pruned or summarised by a strategy |

**Implementation hook:** `ConversationContext.get_messages(strategy=...)` in `forge/context.py`.

### Metrics Collected

| Metric | Source |
|---|---|
| Task success | Human evaluation / test suite pass/fail |
| Input tokens | LLMResponse.input_tokens |
| Output tokens | LLMResponse.output_tokens |
| Inference cost | Derived from token counts + provider pricing |
| Latency | RunTracer.record_llm_call(latency_ms=...) |
| Context size | ConversationContext.approximate_char_count |
| LLM calls | RunTracer._llm_calls |
| Tool calls | RunTracer._tool_calls |

---

## External Interfaces (Future)

### WhatsApp / Telegram

A future demonstration will allow users to interact with FORGE via WhatsApp or Telegram.
The integration must remain a **thin adapter** that:

1. Receives a user message from the messaging platform.
2. Calls `AgentRuntime(AgentConfig(task=message)).run()`.
3. Returns the `AgentRun.final_answer` to the platform.

It must NOT be wired into the core runtime. The core runtime must remain usable
from the CLI, a Python script, and a test suite without any messaging platform present.

### Pi (External Baseline)

Pi may be used as an external evaluation baseline in later experiments.
It will be evaluated by running it against the same task sets as FORGE,
collecting equivalent metrics externally.  It will never be a FORGE dependency.

---

## Safety Design

All tool implementations must call `forge.safety` guards before touching the filesystem or shell.
The current boundaries (Step 1):

1. **Workspace root** — all paths resolved and checked via `assert_within_workspace()`.
2. **Overwrite guard** — existing files require `allow_overwrite=True`.
3. **Command blocklist** — `rm`, `sudo`, `curl`, `dd`, etc. blocked by `assert_safe_command()`.

Not yet enforced: subprocess sandboxing, network restrictions, read-only zones.

---

## Architecture Decision Log

| Decision | Rationale |
|---|---|
| Python 3.11+ | Async support, typed `dataclass`, modern stdlib. Widely available on research machines. |
| No agent framework (LangChain, AutoGen, etc.) | We must instrument and modify every component. Black-box frameworks prevent this. |
| `python-dotenv` only at Step 1 | Zero transitive dependencies. Removes the need to export env vars manually. |
| Abstract `LLMProvider` interface | Provider adapters are swappable without changing agent logic. |
| JSONL trace files | Human-readable, easily parsed with `jq` or pandas. No database required. |
| Frozen `ForgeSettings` dataclass | Prevents accidental mutation of config after startup. |
| No subprocess sandbox yet | Step 1 scope: establish the safety contract. Actual sandboxing deferred to Step 2/3. |
