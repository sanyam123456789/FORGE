# FORGE

**A lightweight, modular coding-agent harness for LLM-based software engineering.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Status: Step 1 — Foundation](https://img.shields.io/badge/status-Step%201%20Foundation-yellow.svg)]()

---

## Problem

Modern LLM-based coding agents (e.g. Devin, SWE-agent, OpenHands) consume large amounts of tokens,
incur high inference costs, and accumulate bloated context windows even for simple tasks.
Two main inefficiencies drive this:

1. **Tool over-exposure** — the LLM is given the full tool catalogue on every turn, even when
   most tools are irrelevant to the current task.
2. **Context accumulation** — the raw interaction history grows unboundedly, forcing the model
   to attend to earlier, irrelevant turns on every subsequent call.

## Proposed Approach

FORGE investigates whether **adaptive tool exposure** and **interaction-context management**
can reduce token usage, inference cost, latency, and context size without sacrificing task success.

- **Adaptive tool exposure** — expose only the tools that are likely relevant to the current
  task phase, rather than the full toolkit on every LLM call.
- **Context management** — selectively prune or summarise the interaction history before
  passing it to the LLM on each turn.

Both dimensions are treated as independent variables in controlled experiments.

## Current Status

> **Step 1 — Foundation only.**
> The coding agent itself is not yet implemented.
> This step establishes the project structure, configuration, safety boundaries,
> logging, and architecture required for the actual agent implementation in Step 2.

## Architecture (Overview)

```
User Task
    │
    ▼
Agent Runtime / Controller          ← forge.agent
    │
    ▼
Context Manager                     ← forge.context
(raw history or managed)
    │
    ▼
LLM Provider Interface              ← forge.llm
    │
    ▼
Tool Selection (adaptive exposure)  ← forge.tools (ToolRegistry.get_schemas)
    │
    ▼
Tool Execution                      ← forge/tools/*.py  (Step 2+)
    │
    ▼
Safety Enforcement                  ← forge.safety
    │
    ▼
Observability / Instrumentation     ← forge.observability
    │
    ▼
Final Result
```

External interfaces (WhatsApp/Telegram, future baselines) connect to the
Agent Runtime layer only — they never touch internals.

## Repository Structure

```
FORGE/
├── forge/                  # Core runtime package
│   ├── __init__.py         # Package metadata, version
│   ├── agent.py            # AgentRuntime, AgentConfig, AgentRun (Step 1: stub)
│   ├── config.py           # Configuration loading from environment variables
│   ├── context.py          # ConversationContext — interaction-history container
│   ├── llm.py              # Abstract LLM provider interface + Message/LLMResponse types
│   ├── logging.py          # Logging foundation (text + JSON formatters)
│   ├── observability.py    # RunTracer — per-run instrumentation and JSONL trace output
│   ├── safety.py           # Workspace boundary enforcement + command blocklist
│   └── tools.py            # Tool base class, ToolResult, ToolRegistry
│
├── tests/                  # Test suite
│   ├── conftest.py         # Shared fixtures (environment isolation)
│   ├── test_config.py      # Configuration loading tests
│   ├── test_logging.py     # Logging initialisation and formatter tests
│   ├── test_observability.py # RunTracer tests
│   ├── test_safety.py      # Safety boundary tests
│   └── test_tools.py       # ToolRegistry and Tool base class tests
│
├── docs/                   # Documentation
│   └── architecture.md     # Detailed architecture document
│
├── .env.example            # Environment variable template (copy to .env)
├── .gitignore              # Ignores secrets, venvs, caches
├── pyproject.toml          # Build config and dependency declarations
└── README.md               # This file
```

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Clone the repository
git clone https://github.com/sanyam123456789/FORGE.git
cd FORGE

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Copy the environment template and fill in your values
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux

# Edit .env — at minimum set FORGE_LLM_API_KEY
```

## Verification

Run the test suite to verify Step 1 is working correctly:

```bash
pytest
```

Expected: all tests pass. No real API key is required.

To see verbose output:

```bash
pytest -v
```

To run a single test module:

```bash
pytest tests/test_safety.py -v
```

## Step 1 Changelog

| Area | What was done |
|---|---|
| Repository structure | Established `forge/` package with clear module responsibilities |
| Configuration | `forge/config.py` — reads all settings from env vars; validates; redacts secrets from repr |
| Logging | `forge/logging.py` — text and JSON formatters, file handler, idempotent setup |
| Safety | `forge/safety.py` — workspace boundary predicate, overwrite guard, command blocklist |
| LLM interface | `forge/llm.py` — abstract `LLMProvider`, `Message`, `LLMResponse` types |
| Tools foundation | `forge/tools.py` — `Tool` ABC, `ToolResult`, `ToolRegistry` with adaptive-exposure hook |
| Context stub | `forge/context.py` — `ConversationContext` container with future pruning hook |
| Agent stub | `forge/agent.py` — `AgentConfig`, `AgentRun`, `AgentRuntime` (loop not yet implemented) |
| Observability | `forge/observability.py` — `RunTracer` with typed events, JSONL trace output |
| Tests | 4 test modules covering config, safety, logging, tools, observability (all passing) |
| Documentation | `README.md`, `docs/architecture.md` |
| Project config | `pyproject.toml`, `.gitignore`, `.env.example` |

## Deliberately Not Implemented Yet

- **Agent loop** — `AgentRuntime.run()` raises `NotImplementedError`
- **Adaptive tool selection** — hook exists in `ToolRegistry.get_schemas(subset=...)`; logic not built
- **Context pruning / summarisation** — hook exists in `ConversationContext`; not implemented
- **Concrete tools** — `read_file`, `write_file`, `shell` etc. (Step 2)
- **LLM provider adapters** — OpenAI, Anthropic adapters (Step 2)
- **CLI entry point** — `forge` command (Step 2)
- **Benchmarking / SWE-bench** — (Step 3+)
- **Pi baseline** — treated as a separate external baseline; never a core dependency
- **WhatsApp / Telegram integration** — committed future demonstration; not in Step 1
- **Web UI, databases, Docker, cloud infrastructure** — not needed

## Next Step (Step 2)

Step 2 should implement:

1. **LLM provider adapters** — at minimum an OpenAI adapter (`forge/llm/openai_adapter.py`)
2. **Concrete tools** — `read_file`, `write_file`, `list_directory`, `run_shell` with full safety checks
3. **Agent loop** — implement `AgentRuntime.run()` connecting context → LLM → tool dispatch → repeat
4. **CLI entry point** — `forge run --task "..."` that exercises the full loop
5. **Integration test** — an end-to-end test on a trivial task using a real (or mocked) LLM
