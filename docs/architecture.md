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

## Agent Loop (implemented in Step 2)

`AgentRuntime.run()` (`forge/agent.py`) executes:

1. Build `ConversationContext` = system prompt + user task.
2. `ToolExposureStrategy.select()` → tool subset; `ToolRegistry.get_schemas()`.
3. `ContextStrategy.prepare(context)` → message list.
4. `LLMProvider.complete(messages, tools=...)` → normalised `LLMResponse`.
5. If the response has tool calls: validate each (`_dispatch_tool`), execute
   through the registry, append a `tool` message per result, loop to step 2.
6. If the response has no tool calls: its text is the final answer; stop.
7. Stop early with a terminal status if `max_llm_turns` or `max_tool_calls`
   is reached, on a provider error, or on any unexpected exception — the run
   always records a trace and returns an `AgentRun`.

The diagram below shows the same flow with the research seams marked.


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
│  Normalised: Message / ToolCall → LLMResponse + usage      │
│  Adapters (forge.providers): gemini  [openai | anthropic …] │
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
| `forge.llm` | Provider-agnostic vocabulary (`Message`, `ToolCall`, `LLMResponse`, `LLMError`), the `LLMProvider` ABC, and `get_provider()`. |
| `forge.providers` | Concrete adapters. `forge.providers.gemini` is the only one so far; nothing else imports a provider SDK. |
| `forge.tools` | `Tool` ABC, `ToolRegistry` (`get_schemas(subset)` = exposure hook), and the `ToolExposureStrategy` seam (`FixedToolExposure`, `AdaptiveToolExposure` — Step 5). |
| `forge.builtin_tools` | The five concrete tools + `build_default_registry()`. |
| `forge.context` | `ConversationContext` history container + the `ContextStrategy` seam (`RawContextStrategy`, `ManagedContextStrategy` — Step 6) and `ContextReport` measurement. |
| `forge.prompts` | Canonical `FORGE_SYSTEM_PROMPT` and its stable `sha256` identifier. |
| `forge.agent` | The agent loop. Owns the run lifecycle, hard limits, tool dispatch, and error handling. |
| `forge.observability` | Record typed events per run. Flush to JSONL. Never fabricate measurements (`None` ≠ `0`). |
| `forge.evaluation` | Step 3 measurement layer *above* `AgentRuntime`: `EvalTask`, `ExperimentConfig`, `EvaluationRunner`, `EvalResult` (JSONL), `aggregate_results`. `fixed`/`adaptive` tool strategies (Step 5) and `raw`/`managed` context strategies (Step 6) are all implemented and freely combinable. See `docs/step-03-evaluation.md`, `docs/step-05-adaptive-tool-exposure.md`, `docs/step-06-managed-context.md`. |
| `forge.evaluation.suite` | Step 4 loader for the version-controlled baseline task suite in `experiments/tasks/` (→ ordinary `EvalTask`s, with fixtures provisioned into the run workspace). See `docs/step-04-baseline-tasks.md`. |
| `forge.cli` | `forge run --task "..."`, `forge tasks`, `forge evaluate (--task / --task-file / --suite-task-id)`. |

---

## Research Arms

### Arm A — Tool Exposure Strategy

**Variable:** which tool schemas are passed to the LLM on each turn.

| Condition | Description | Status |
|---|---|---|
| Fixed | All registered tools always exposed (baseline) | **implemented** (`FixedToolExposure`) |
| Adaptive | Only tools relevant to the current task exposed, via deterministic keyword classification | **implemented** (`AdaptiveToolExposure`, Step 5) |

**Seam:** `ToolExposureStrategy.select(registry, context, task) -> list[str] | None`
in `forge/tools.py`, consumed by `AgentRuntime` and fed to
`ToolRegistry.get_schemas(subset)`. The agent never branches on an "adaptive"
flag — it only calls `select()`. See `docs/step-05-adaptive-tool-exposure.md`
for the classification policy, its category → tool mapping, and its
limitations.

### Arm B — Context Management Strategy

**Variable:** how the interaction history is prepared before each LLM call.

| Condition | Description | Status |
|---|---|---|
| Raw | Full history passed unchanged | **implemented** (`RawContextStrategy`) |
| Managed | Deterministic, rule-based pruning/compression: always keep the task, the recent window, and any turn with a failed tool result; drop or compress older redundant successful tool output; enforce a hard message/char budget | **implemented** (`ManagedContextStrategy`, Step 6) |

**Seam:** `ContextStrategy.prepare(context) -> list[Message]` in
`forge/context.py`. `ConversationContext.get_messages()` always returns the
raw, unmodified history; any transformation lives in a strategy, not in the
container and not in the agent loop. A strategy may also implement
`last_report() -> ContextReport | None` to expose what it did (sizes
before/after, items dropped/compressed) for measurement — this never
affects what is sent to the LLM. See
`docs/step-06-managed-context.md` for the full policy, its determinism
argument, and its limitations.

### Metrics Collected

| Metric | Source |
|---|---|
| Task success | Test suite pass/fail on the produced changes |
| Input / output / total tokens | `LLMResponse` usage → `RunTracer` (`None` when unreported) |
| Cached input / reasoning tokens | `LLMResponse.cached_input_tokens` / `.reasoning_tokens` |
| Inference cost | Derived later from token counts + provider pricing |
| Latency | `RunTracer.record_llm_response(latency_ms=...)` |
| Context size | `ConversationContext.approximate_char_count` (char proxy) |
| Context size before/after management, items dropped/compressed | `ContextStrategy.last_report()` (`ContextReport`) → `RunTracer` per-turn `llm_call` events + cumulative `AgentRun`/`EvalResult` fields (Step 6) |
| LLM calls | `RunTracer._llm_calls` |
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

All tool implementations call `forge.safety` guards before touching the
filesystem or shell. The boundaries:

1. **Workspace root** — all paths resolved (symlinks included) and checked via
   `assert_within_workspace()`.
2. **Overwrite guard** — existing files require `allow_overwrite=True`
   (`assert_safe_write()`).
3. **Command blocklist** — `assert_safe_command()` splits the command
   (Windows-safe), inspects only `argv[0]` against a static blocklist (`rm`,
   `del`, `mkfs`, `dd`, `curl`, `wget`, `sudo`, …), and `run_shell` executes
   with `shell=False` from the workspace root.

This is a **heuristic, not a sandbox**. Not enforced: shell-wrapper / one-liner
evasion (`sh -c "rm ..."`), output redirection, network egress from an allowed
program, subprocess isolation, read-only zones, rate limiting. Run FORGE
against disposable workspaces.

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
| No subprocess sandbox yet | The safety layer is a documented heuristic; a real sandbox is out of scope for the research questions. |
| Gemini as the first provider (Step 2) | Current, non-permanent choice. Confined to `forge/providers/gemini.py`; `google-genai` is lazy-imported. The rest of FORGE only sees `forge.llm` types. |
| Strategy *objects* for the research arms | `ToolExposureStrategy` / `ContextStrategy` are injected into `AgentRuntime`. Swapping Fixed→Adaptive or Raw→Managed later needs no change to the loop, and the strategy id is recorded in every trace. |
| `None` ≠ `0` for usage | A metric a provider does not report is recorded as `None`; input and output availability are tracked independently so a real zero is never invented. |
| Keyword-based classifier for `AdaptiveToolExposure` (Step 5) | Deterministic, explainable, and dependency-free — no embeddings/ML model needed to compare Fixed vs. Adaptive. Deliberately a placeholder: a learned or per-turn selector can replace it later without touching `AgentRuntime`, `ExperimentConfig`, or the tracer. |
| Turn-based, rule-based `ManagedContextStrategy` (Step 6) | Deterministic and explainable, like Step 5's classifier — no summarisation model, no extra LLM call. Turns (assistant + its tool results) are always kept or dropped atomically so a tool call and its result are never separated. A recent window and any turn containing a failed tool result are always preserved in full; older redundant successful tool output is compressed in place (message kept, content shortened) rather than deleted, so provider-side pairing (by id or by name) never breaks. |
