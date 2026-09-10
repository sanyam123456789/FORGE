# FORGE

**A lightweight, modular coding-agent harness for LLM-based software engineering.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Status: Step 2 — Agent Loop](https://img.shields.io/badge/status-Step%202%20Agent%20Loop-green.svg)]()

---

## What FORGE is

FORGE is a small, instrumented coding-agent runtime built for a research
project. It exists to answer one question:

> Can a coding agent reach comparable software-engineering performance with
> **lower token usage, cost, latency, and context consumption** through
> **adaptive tool exposure** and **interaction-context management**?

To study that, FORGE is built so the *harness strategy* can change while the
model, task, environment, and prompt are held constant. It is **not** a
Claude Code clone, and it deliberately avoids agent frameworks, multi-agent
systems, RAG/vector stores, and cloud infrastructure.

## Current status — Step 2

Step 2 delivers the **first working agent loop**:

- **Gemini provider adapter** (`google-genai`) behind a provider-agnostic
  `LLMProvider` interface.
- **Five built-in tools**: `read_file`, `write_file`, `edit_file`,
  `list_directory`, `run_shell` — all routed through the workspace-safety
  guards and returning structured results.
- **Agent loop** (`AgentRuntime.run()`): build context → LLM call → detect &
  validate tool calls → execute → feed results back → repeat → stop on final
  answer or a hard limit.
- **Fixed tool exposure** and **raw context** — the Step 2 baselines, each
  behind a strategy interface so the adaptive / managed variants can be added
  later without touching the loop.
- **Observability**: every run emits a JSONL trace (`run_start`, `llm_call`,
  `llm_response`, `tool_call`, `tool_result`, `error`, `run_end`) with
  provider-reported token usage and the experiment metadata needed to
  compare runs.
- **CLI**: `forge run --task "..."`.

The real Gemini API demo is **pending API-key setup** (see below); the loop is
fully covered by mocked tests including a mocked end-to-end run.

## Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/sanyam123456789/FORGE.git
cd FORGE
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
cp .env.example .env          # then edit .env
```

## Gemini configuration

FORGE reads all configuration from the environment (`.env` is loaded
automatically).

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_LLM_PROVIDER` | `gemini` | Provider id. Only `gemini` has an adapter today. |
| `FORGE_LLM_MODEL` | `gemini-3.6-flash` | Model name passed to the Gemini API. |
| `FORGE_LLM_API_KEY` | — | Gemini API key. Falls back to `GEMINI_API_KEY` / `GOOGLE_API_KEY`. |
| `FORGE_LLM_TEMPERATURE` | `0.0` | Sampling temperature (0.0–2.0). |
| `FORGE_LLM_MAX_OUTPUT_TOKENS` | `4096` | Max output tokens per call. |
| `FORGE_MAX_LLM_TURNS` | `30` | Hard ceiling on LLM turns per run. |
| `FORGE_MAX_TOOL_CALLS` | `50` | Hard ceiling on tool calls per run. |
| `FORGE_WORKSPACE_ROOT` | cwd | All file/shell operations are confined here. |
| `FORGE_LOG_DIR` | `logs` | Where run traces (`<run_id>.jsonl`) are written. |

The API key is **never** hard-coded, logged, or written to disk.

## CLI usage

```bash
forge run --task "Create add.py with an add(a, b) function, then add multiply(a, b)"
```

Options: `--workspace PATH`, `--model NAME`, `--max-turns N`,
`--max-tool-calls N`, `--temperature FLOAT`, `--no-trace`, `--json`,
`--quiet`.

Exit codes: `0` completed · `1` stopped without completing (limit / provider
error) · `2` configuration problem (e.g. no API key).

## Available tools

| Tool | Purpose | Key safety |
|---|---|---|
| `read_file` | Read a UTF-8 text file | workspace boundary |
| `write_file` | Create / overwrite a file (`overwrite` flag) | boundary + overwrite guard |
| `edit_file` | Replace an exact substring (`replace_all` flag) | boundary; must read-then-edit |
| `list_directory` | List a directory's entries | workspace boundary |
| `run_shell` | Run a non-interactive command from the workspace root | boundary + command blocklist |

## Architecture (overview)

```
User task
  → AgentRuntime (forge.agent)            owns the loop + hard limits
      → ContextStrategy (forge.context)   Step 2: RawContextStrategy
      → LLMProvider (forge.llm)           → GeminiProvider (forge.providers.gemini)
      → ToolExposureStrategy (forge.tools) Step 2: FixedToolExposure
      → ToolRegistry → built-in tools (forge.builtin_tools)
      → safety guards (forge.safety)
      → RunTracer (forge.observability)   → logs/<run_id>.jsonl
  → AgentRun result
```

Nothing outside `forge/providers/` imports a provider SDK. The agent depends
on the `LLMProvider`, `ContextStrategy`, and `ToolExposureStrategy`
abstractions, never on Gemini or on adaptive behaviour. See
[`docs/architecture.md`](docs/architecture.md) and
[`docs/step-02-agent-loop.md`](docs/step-02-agent-loop.md).

## Safety limitations

The safety layer is a **heuristic, not a sandbox**:

- File operations are confined to the workspace root (symlinks resolved).
- Overwriting an existing file requires an explicit flag.
- `run_shell` splits the command to argv, runs it with `shell=False`, and
  rejects an `argv[0]` on a static blocklist (`rm`, `del`, `mkfs`, `dd`,
  `curl`, `wget`, `sudo`, …).
- It does **not** stop shell wrappers (`sh -c "rm ..."`), interpreter
  one-liners, output redirection, or network access from within an allowed
  program. Run FORGE against disposable workspaces.

## Tests

```bash
pytest            # 214 tests, no API key required
```

Coverage: Gemini adapter (mocked SDK), the five tools + their safety and
error paths, tool registry, config validation, logging, context + strategy,
observability + token accounting, and the agent loop — malformed / unknown /
failing tool calls, provider errors, `max_llm_turns` / `max_tool_calls`
ceilings, and a mocked end-to-end two-step coding task.

## Step 2 changelog

| Area | Change |
|---|---|
| Provider | New `forge/providers/gemini.py` (`google-genai`); `get_provider()` wired for `gemini`. |
| LLM types | `LLMResponse` gains `tool_calls`, `cached_input_tokens`, `reasoning_tokens`, `total_tokens`, `model`, `raw_usage`; new `ToolCall`; `Message` gains tool fields; new `LLMError`. |
| Tools | New `forge/builtin_tools.py` with the five tools + `build_default_registry()`. |
| Strategies | `ContextStrategy`/`RawContextStrategy` (context.py); `ToolExposureStrategy`/`FixedToolExposure` (tools.py). |
| Agent | `AgentRuntime.run()` implemented with hard limits and full error handling. |
| Prompt | New `forge/prompts.py` — `FORGE_SYSTEM_PROMPT` + stable `sha256` id. |
| CLI | New `forge/cli.py`; `forge` console script. |
| Observability | `RunTracer` records experiment metadata + `llm_call`/`llm_response` split; token accounting is `None`-safe with independent input/output availability. |
| Audit fixes | `.rstrip(".exe")` → real suffix strip; Windows-safe command splitting; negative/out-of-range config values rejected; token fields default `None` not `0`; architecture doc corrected. |
| Deps | Added `google-genai` (lazy-imported, confined to the adapter). |

## Intentionally not implemented

- Adaptive tool selection / any tool ranking
- Context pruning or summarisation
- Additional providers (OpenAI, Anthropic, …) — interface is ready
- Pi integration (external baseline only, considered later)
- WhatsApp / Telegram interface (committed future demo, after evaluation)
- SWE-bench / benchmark harness, cost dashboards
- Web UI, databases, Docker, cloud infrastructure

## Next step (Step 3)

Run the first real Gemini task once a key is configured, then start the
**evaluation harness**: a task runner that executes the same task under
different harness strategies and aggregates the per-run JSONL traces into a
comparison table (task success, tokens, cost, latency, LLM/tool calls,
context size). No new agent capability — just measurement on top of the
Step 2 loop.
