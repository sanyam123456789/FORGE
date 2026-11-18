# FORGE

**A lightweight coding agent that spends fewer tokens for the same result.**

FORGE is a small, instrumented coding-agent runtime built for a research
question:

> Can a coding agent reach the same software-engineering results with
> **lower token usage, cost and context size**, through **adaptive tool
> exposure** and **careful context management**?

It is not a Claude Code clone. It uses no agent framework, no RAG or vector
store and no cloud services. It is plain Python with one standard-library
provider adapter. Every strategy can be swapped while the model, task and
prompt stay fixed, so the savings can be measured.

## Results

Benchmark run on 2026-11-18: the 10-task suite under six strategy
combinations, 60 real runs on OpenCode Go (`deepseek-v4.1-flash`).

| Arm | Solved | Avg tokens | Cache hit | Avg LLM calls |
|---|---:|---:|---:|---:|
| fixed_raw (baseline) | 10/10 | 7,144 | 76% | 3.8 |
| adaptive_raw (lean tools) | 10/10 | **5,594** | 73% | 3.4 |
| fixed_managed | 10/10 | 7,230 | 76% | 3.9 |
| adaptive_managed | 10/10 | 5,802 | 71% | 3.5 |
| fixed_stable | 10/10 | 6,988 | 75% | 3.7 |
| adaptive_stable (`smart`) | 10/10 | 7,252 | 77% | 4.2 |

- **Same results:** every arm solved every task.
- **Lean tools: 22% fewer tokens** than the baseline (lower on 9 of 10 tasks).
- **Context strategies: not tested by this suite.** The largest context in
  any run was about 1k tokens, below every pruning threshold, so raw,
  managed and stable sent identical requests. Differences between them here
  are run-to-run noise, which reached 3× on a single task. Measuring them
  needs longer tasks and repeated runs.
- **Earlier manual benchmark** (other providers): FORGE used 4–25× fewer
  tokens than an external agent running the same model, with the same
  correct output
  ([details](docs/benchmark-forge-vs-external-agents.md)).

Full tables and caveats: [`docs/step-11-benchmark-results.md`](docs/step-11-benchmark-results.md).
Reproduce with `forge bench`, then `forge report`.

## Quick start

**Requirements:** Python 3.11+ and an [OpenCode Go](https://opencode.ai) API
key (or any OpenAI-compatible endpoint).

```bash
git clone https://github.com/sanyam123456789/FORGE.git
cd FORGE
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
cp .env.example .env          # then put your key in FORGE_LLM_API_KEY
```

```bash
forge "fix the bug in average() in stats.py"     # one task
forge chat                                         # a conversation
```

A run ends with one line, for example:

```
✓ Done · 4 turns · 6.8k tokens · 61% cached · 15.4s
```

## Using FORGE

### One task

```bash
forge "add a median() function to stats.py and check it with python"
forge "rename tally() to count_items()" --workspace path/to/project
forge "..." --mode baseline --verbose
```

Options: `--mode`, `--workspace`, `--model`, `--max-turns`,
`--max-tool-calls`, `--temperature`, `--json`, `--quiet`, `--verbose`,
`--no-trace`. `forge run "..."` and `forge run --task "..."` also work.

### Chat

`forge chat` keeps one conversation, so follow-ups build on what FORGE
already knows. Type tasks in plain words; everything else is a slash command:

| Command | What it does |
|---|---|
| `/help` | List commands |
| `/mode [name]` | Show or switch mode |
| `/model [name]` | Show or switch model |
| `/tokens` | Tokens used this session, cache hit rate, uncached input |
| `/context` | What is in the context now: size, compressed / dropped messages, compactions |
| `/tools` | Tools offered on the last request |
| `/compact` | Compact the context on the next request |
| `/diff [all]` | File changes from the last task (or the whole session) |
| `/undo` | Revert the file changes of the last task |
| `/clear` | Start a fresh conversation |
| `/workspace [path]` | Show or change the folder FORGE works in |
| `/exit` | Quit |

### Modes

| Mode | Tools | Context | |
|---|---|---|---|
| `baseline` | all, every turn | full history every turn | the reference point |
| `lean-tools` | only what the task needs | full history | |
| `lean-context` | all | managed (pruned, compressed) | |
| `smart` | only what the task needs | managed **and** cache-friendly | **default** |

### Benchmark and report

```bash
forge bench                      # every suite task under all six arms, then the report
forge bench --task-id fix-inclusive-sum --arm fixed_raw --arm adaptive_stable
forge report                     # table for the newest results (or pass a results.jsonl)
```

The research commands are unchanged: `forge tasks`, `forge evaluate`,
`forge experiment` (the formal 2×2), `forge demo` and `forge analyze`
(statistics).

## How FORGE saves tokens

**Cache-friendly context (`smart` mode).** Providers bill cached input
tokens at a fraction of the price, but only for an *unchanged prefix* of
the request. FORGE's managed context (pruning stale tool output, keeping
errors and recent turns) is applied in occasional batches. Between batches
every request is the previous one plus new messages, so the cache keeps
working. See [`docs/step-11-token-savers.md`](docs/step-11-token-savers.md).

**Lean tool exposure.** Only the tools a task needs are sent. If FORGE
guessed wrong, the model calls a tiny `enable_tool` tool to turn one on, so
a task never gets stuck.

**Fewer wasted rounds.**
- The first message lists the workspace files.
- `search_files` finds code, and `read_file` reads just a line range.
- An identical repeated read gets a one-line pointer instead of the same
  text again.
- Repeated failures and stalls get a nudge to change approach.
- Cut-off replies are continued instead of being taken as the answer.

**Measured, not guessed.** Every run writes a JSONL trace with the
provider's own token counts, including cached tokens. Context budgets use
a built-in token estimate (within 0.5% of the provider's count on a real
request). Reported numbers always come from the provider.

## Tools

| Tool | Purpose |
|---|---|
| `read_file` | Read a text file, or a line range of it |
| `write_file` | Create or (with `overwrite`) replace a file |
| `edit_file` | Replace an exact piece of text in a file |
| `list_directory` | List a directory |
| `search_files` | Regex search across the workspace, returns `path:line: text` |
| `run_shell` | Run a non-interactive command in the workspace |

File writes keep each file's own line endings (LF stays LF on Windows).

## Configuration

FORGE reads settings from the environment; `.env` is loaded automatically.

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_LLM_PROVIDER` | `opencode_go` | `opencode_go`, or `openai_compatible` for any OpenAI-compatible endpoint (e.g. Ollama at `http://localhost:11434/v1`) |
| `FORGE_LLM_MODEL` | `deepseek-v4.1-flash` | Model name |
| `FORGE_LLM_API_KEY` | — | API key (`OPENCODE_API_KEY` also works for OpenCode Go) |
| `FORGE_LLM_BASE_URL` | OpenCode Go endpoint | Required for `openai_compatible` |
| `FORGE_LLM_MAX_OUTPUT_TOKENS` | `0` | Output cap per call; `0` = no cap |
| `FORGE_MAX_LLM_TURNS` | `30` | Turn limit per task; `0` = unlimited |
| `FORGE_MAX_TOOL_CALLS` | `50` | Tool-call limit per task; `0` = unlimited |
| `FORGE_REPO_MAP` | `true` | Put the workspace file list in the first message |
| `FORGE_LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `FORGE_LLM_REQUEST_TIMEOUT_S` | `120` | Per-request timeout |
| `FORGE_PROVIDER_RETRY_MAX` | `3` | Retries for 429 / 5xx / connection errors |
| `FORGE_WORKSPACE_ROOT` | current folder | All file and shell operations stay inside it |
| `FORGE_LOG_DIR` | `logs` | Where run traces are written |
| `FORGE_SHELL_ENV_ALLOW` | — | Extra env vars passed to `run_shell` (`FORGE_*` never is) |

The API key is never hard-coded, logged or written to disk.

## Architecture

```
task / chat message
  → AgentRuntime (forge/agent.py)            the loop, limits and token savers
      → ContextStrategy (forge/context.py)   raw | managed | stable
      → ToolExposureStrategy (forge/tools.py) fixed | adaptive (+ enable_tool)
      → LLMProvider (forge/providers/openai_compat.py)
      → built-in tools (forge/builtin_tools.py) → safety guards (forge/safety.py)
      → RunTracer (forge/observability.py)   → logs/<run_id>.jsonl
  → AgentRun (answer, tokens, cache hits, context stats)

forge/chat.py      conversation, slash commands, /diff and /undo
forge/modes.py     baseline / lean-tools / lean-context / smart
forge/evaluation/  tasks, runner, 2×2 matrix, analysis, report
```

More detail: [`docs/architecture.md`](docs/architecture.md) and the step
documents in [`docs/`](docs/).

## Safety

The safety layer is a **heuristic, not a sandbox**:

- File operations stay inside the workspace (symlinks resolved).
  Overwriting needs an explicit flag.
- `.env`, `*.pem`, `*.key` and SSH key files are refused, and are left out
  of the workspace list and search results.
- `run_shell` runs argv with `shell=False` and blocks destructive and
  network commands (`rm`, `del`, `mkfs`, `curl`, `wget`, `sudo`, …). It
  gets a minimal environment without API keys, and secrets are redacted
  from output.
- It does **not** stop interpreter one-liners or network access from
  inside an allowed program. Use FORGE on workspaces you can afford to
  lose, or under version control.

## Tests

```bash
pytest            # 659 tests, no API key or network needed
```

## Project history

| Step | What it added | Doc |
|---|---|---|
| 2 | Agent loop, tools, provider, tracing | [step-02](docs/step-02-agent-loop.md) |
| 3–4 | Evaluation harness, 10-task baseline suite | [step-03](docs/step-03-evaluation.md), [step-04](docs/step-04-baseline-tasks.md) |
| 5 | Adaptive tool exposure | [step-05](docs/step-05-adaptive-tool-exposure.md) |
| 6 | Managed context | [step-06](docs/step-06-managed-context.md) |
| 7–8 | Formal 2×2 experiment, statistical analysis | [step-07](docs/step-07-controlled-2x2-experiment.md), [step-08](docs/step-08-statistical-analysis.md) |
| 9 | Real-provider pilots, external-agent benchmark | [step-09](docs/step-09-research-multitask-analysis.md), [benchmark](docs/benchmark-forge-vs-external-agents.md) |
| 11 | Cache-friendly context, token savers, chat, modes, bench/report | [step-11](docs/step-11-token-savers.md), [results](docs/step-11-benchmark-results.md) |

## Not included (by design)

Multi-agent systems, RAG or vector stores, web UI, databases, Docker and
cloud infrastructure. A WhatsApp interface is planned as a separate,
optional add-on.
