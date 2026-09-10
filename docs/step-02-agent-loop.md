# Step 2 — The FORGE Agent Loop

## 1. Objective

Turn the Step 1 foundation into the **first working FORGE coding agent**: a
real LLM provider, a minimal tool set, a bounded agent loop, basic context
flow, observability wired to the loop, provider-reported token accounting, and
a small CLI — all under **fixed tool exposure** and **raw context** so the
later research interventions can be layered on without rewriting the runtime.

## 2. What was implemented

### Gemini provider (`forge/providers/gemini.py`)
- `GeminiProvider(LLMProvider)` wrapping the `google-genai` SDK.
- Translates FORGE `Message` / `ToolCall` / tool schemas → Gemini request
  objects (system messages → `system_instruction`; tool schemas →
  `FunctionDeclaration`s; tool results → `function_response` parts).
- Translates the Gemini response → `LLMResponse`: text, structured
  `tool_calls`, normalised `stop_reason`, and usage mapped from
  `usage_metadata` (`prompt_token_count`, `candidates_token_count`,
  `total_token_count`, `cached_content_token_count`, `thoughts_token_count`).
  Anything the provider omits stays `None`.
- API key from `FORGE_LLM_API_KEY` (or `GEMINI_API_KEY` / `GOOGLE_API_KEY`);
  never hard-coded or logged. `preflight()` fails fast with no network call.
- Provider/API exceptions are normalised to `forge.llm.LLMError`.

### LLM vocabulary (`forge/llm.py`)
- New `ToolCall` (`id`, `name`, `arguments`).
- `Message` gains `tool_calls`, `tool_call_id`, `name`.
- `LLMResponse` gains `tool_calls`, `total_tokens`, `cached_input_tokens`,
  `reasoning_tokens`, `model`, `raw_usage`, plus `has_tool_calls` and
  `resolved_total_tokens` helpers.
- New `LLMError`; `get_provider()` wired for `gemini`, accepts a settings
  override for testing.

### Tools (`forge/builtin_tools.py`)
`read_file`, `write_file`, `edit_file`, `list_directory`, `run_shell` — each a
`Tool` subclass with a JSON-Schema, input validation, a never-raising
`execute()` that returns a structured `ToolResult`, workspace-safety routing,
and useful metadata (`bytes`, `replacements`, `exit_code`, …).
`build_default_registry(workspace_root=...)` produces a populated registry.

### Strategy seams
- `ToolExposureStrategy` + `FixedToolExposure` (`forge/tools.py`).
- `ContextStrategy` + `RawContextStrategy` (`forge/context.py`).
- `AgentRuntime` depends only on these abstractions.

### Agent loop (`forge/agent.py`)
`AgentRuntime.run()` → `AgentRun`. Builds context (system prompt + task),
then per turn: `select` tools → `prepare` messages → `provider.complete` →
record usage → if tool calls, validate + execute each and append results →
else finish. Hard limits `max_llm_turns` / `max_tool_calls`; terminal
statuses `completed`, `max_turns_exceeded`, `max_tool_calls_exceeded`,
`provider_error`, `failed`. Malformed / unknown / mis-argumented / failing
tool calls become tool messages fed back to the model, never crashes.

### System prompt (`forge/prompts.py`)
`FORGE_SYSTEM_PROMPT` (concise: identity, workspace bounds, tool list, "use
tools then finish with plain text") + `FORGE_SYSTEM_PROMPT_ID` (`sha256:` of
the text) recorded in every run trace.

### Observability (`forge/observability.py`)
`RunTracer` now records `run_start` with experiment metadata (provider,
model, temperature, max output tokens, system-prompt id, tool-exposure
strategy, tool subset, context strategy, limits), a split
`llm_call` / `llm_response` pair, `tool_call` / `tool_result`, `error`, and
`run_end` with token totals. Token accumulators are `None` until a provider
reports them; input and output availability are independent (`_add` helper).
`RunSummary` widened to match. Trace flushed to `logs/<run_id>.jsonl`.

### CLI (`forge/cli.py`)
`forge run --task "..."` with `--workspace / --model / --max-turns /
--max-tool-calls / --temperature / --no-trace / --json / --quiet`. Exit
`0` completed, `1` stopped without completing, `2` config problem.

### Step 1 audit fixes folded in
- `forge/safety.py`: `.rstrip(".exe")` → real suffix removal; new
  `split_command()` keeps Windows backslash paths and quoted args.
- `forge/config.py`: `_get_int` gains `min_value`; new `_get_float` with
  range checks; `max_tool_calls` / `max_llm_turns` / `max_output_tokens`
  reject `< 1`; temperature constrained to `0.0–2.0`.
- Token fields default `None`, not `0`.
- `docs/architecture.md` corrected (`get_messages()` takes no `strategy=`).

## 3. Architecture

```
forge.cli
  └─ AgentRuntime(config, provider, registry, tracer)         forge.agent
       ├─ ContextStrategy.prepare(ConversationContext)         forge.context  (Raw)
       ├─ ToolExposureStrategy.select(ToolRegistry)            forge.tools    (Fixed)
       ├─ LLMProvider.complete(messages, tools)                forge.llm
       │     └─ GeminiProvider                                 forge.providers.gemini → google-genai
       ├─ ToolRegistry.get(name).execute(**args)               forge.builtin_tools
       │     └─ assert_within_workspace / assert_safe_command  forge.safety
       └─ RunTracer.record_* / flush                           forge.observability → logs/<run_id>.jsonl
```

Import direction is one-way: `agent` → everything; `builtin_tools` → `tools`,
`safety`; `providers.gemini` → `llm`; nothing imports `agent` or `cli`. Only
`forge/providers/` imports a provider SDK.

## 4. Important design decisions

- **Strategy objects, not flags.** The two research arms are injected
  objects; the loop calls `select()` / `prepare()` and records their `name`.
  Adaptive/managed variants slot in later with no loop change. A test asserts
  the word "adaptive" does not appear in `forge/agent.py`.
- **Provider isolation.** `AgentRuntime` speaks `Message` / `LLMResponse`
  only. Swapping Gemini for another provider is a new file under
  `forge/providers/` plus one line in `get_provider()`.
- **Failures are data.** Every tool/provider failure is caught and either fed
  back to the model (tool errors) or ends the run with a recorded terminal
  status (provider errors, limits, unexpected exceptions). The loop always
  writes a trace.
- **`None` ≠ `0`.** Unreported usage is `None` end to end
  (`LLMResponse` → `AgentRun` → `RunTracer` → JSONL).
- **`run_shell` is `shell=False` + argv blocklist**, documented as a
  heuristic, not a sandbox.
- **`tool_subset` convenience.** `AgentConfig(tool_subset=[...])` is wired
  into `FixedToolExposure(pinned=...)` and recorded — still fixed exposure
  (static list, no ranking).

## 5. Testing / verification

`pytest` → **214 passed**, no API key required.

| Area | File |
|---|---|
| Gemini adapter: request/response translation, usage mapping, missing/partial usage, errors, API-key handling, factory | `tests/test_llm.py` (23) |
| The five tools: happy paths, validation errors, workspace-escape, safety errors, registry helpers | `tests/test_builtin_tools.py` (34) |
| Command blocklist + `.exe` suffix regression + `split_command` | `tests/test_safety.py` (38) |
| Config bounds (negative / out-of-range / non-numeric) | `tests/test_config.py` (32) |
| Context container + `RawContextStrategy` | `tests/test_context.py` (11) |
| `RunTracer` events, `None`-safe token accounting, `_add` | `tests/test_observability.py` (22) |
| Agent loop: happy path, **mocked end-to-end two-step edit**, unknown/malformed/invalid/failing tool calls, provider error, unexpected exception, `max_llm_turns`, `max_tool_calls`, token accounting, trace contents, fixed exposure | `tests/test_agent.py` (23) |
| CLI: help, run→file, limit→exit 1, `--json`, unimplemented provider→exit 2 | `tests/test_cli.py` (6) |

**Real Gemini API demo: not run** — no API key is configured in this
environment. No performance or token numbers are reported. The loop is
verified end to end against a scripted provider with the real built-in tools
writing real files in a temp workspace.

## 6. Known limitations

- No real-API run yet (pending credentials).
- Safety is a heuristic (see §4 and `docs/architecture.md`).
- Context size is a character-count proxy; no tokenizer.
- `run_shell` uses `shlex`-style splitting, not a real shell parser.
- Gemini is the only provider; `openai` / `anthropic` raise
  `NotImplementedError`.
- One `RawContextStrategy` only — no managed context.
- `_add` is imported from `forge.observability` into `forge.agent`
  (package-internal helper).

## 7. Intentionally deferred

Adaptive tool selection · context pruning/summarisation · additional
providers · Pi integration · WhatsApp/Telegram · SWE-bench / benchmark
experiments · cost dashboards · Pyright/AutoTool/multi-agent/RAG/web-UI/cloud.

## 8. Next step (Step 3)

1. One real Gemini run once a key exists (smoke, not a benchmark).
2. **Evaluation harness**: a task runner that executes the same task under
   different strategy configurations and aggregates the per-run JSONL traces
   into a comparison table (success, tokens, cost, latency, LLM/tool calls,
   context size). Measurement only — no new agent capability.
