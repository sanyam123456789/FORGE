# Step 3 — Evaluation & Measurement Infrastructure

## 1. Purpose

Step 3 builds the **measurement layer** that will let us run the *same* coding
tasks under different future FORGE strategies and compare the outcomes
fairly.

It is infrastructure only. It does **not** add any agent capability:

- no adaptive tool exposure
- no managed / summarised / compressed context
- no new providers, no benchmark integration, no research run

The research question is unchanged:

> Can a coding agent achieve comparable software-engineering task performance
> with lower token usage, inference cost, latency, and context consumption
> through adaptive tool exposure and interaction-context management?

## 2. What was added

A new package `forge/evaluation/` that sits **above** the existing
`AgentRuntime`. There is still exactly one agent loop; the runner calls it.

```
EvalTask ──► ExperimentConfig ──► EvaluationRunner ──► AgentRuntime (unchanged)
                                        │                     │
                                        │                JSONL trace + AgentRun
                                        ▼
                                   EvalResult ──► write/read JSONL ──► aggregate_results / group_results
```

| File | Responsibility |
|---|---|
| `forge/evaluation/task.py` | `EvalTask` (stable `task_id` + prompt + optional `ArtifactCheck`s), `CheckOutcome`. |
| `forge/evaluation/experiment.py` | `ExperimentConfig` — the one controlled configuration; strategy id constants; `resolve_strategies()`. |
| `forge/evaluation/result.py` | `EvalResult` (metrics row), `TokenPricing` (caller-supplied), JSONL read/write. |
| `forge/evaluation/runner.py` | `EvaluationRunner` — isolates a workspace, invokes `AgentRuntime`, runs checks, builds `EvalResult`. |
| `forge/evaluation/aggregate.py` | `aggregate_results`, `group_results`, `AggregateStats` — None-safe roll-ups. |
| `forge/cli.py` | new `forge evaluate` subcommand. |

## 3. Task representation

`EvalTask(task_id, prompt, checks=(), metadata={})`

- **`task_id`** is the *only* identity. The same `EvalTask` is meant to be run
  many times under different `ExperimentConfig`s; nothing about the task
  changes between those runs, so results stay comparable.
- **`checks`** are optional `ArtifactCheck(path, must_exist=True, must_contain=())`
  — simple existence / substring assertions used to decide task success.
  No checks ⇒ success is "the run reached a completed status".
- **`metadata`** is free-form and never used as a comparison key.
- `to_dict` / `from_dict` — tasks can be stored as JSON and loaded with
  `forge evaluate --task-file`.

## 4. Experiment configuration

`ExperimentConfig` pins everything held constant plus the two research
variables:

| Field | Meaning |
|---|---|
| `provider`, `model` | from `ForgeSettings` via `ExperimentConfig.from_settings(...)` — not hard-coded |
| `tool_strategy` | `"fixed"` (only implemented) — future: `"adaptive"` |
| `context_strategy` | `"raw"` (only implemented) — future: `"managed"` |
| `temperature`, `max_output_tokens` | from settings |
| `max_llm_turns`, `max_tool_calls` | from settings unless overridden; `None` ⇒ defer to runtime |
| `task_id`, `run_id`, `label` | identity / batch metadata |

- `resolve_strategies()` returns `(FixedToolExposure(), RawContextStrategy())`
  for `fixed` / `raw`, and raises **`NotImplementedError`** for the
  recognised-but-unbuilt `adaptive` / `managed` names. The runner never
  silently downgrades.
- `is_implemented`, `strategy_key` (`"fixed+raw"`) support the future 2×2.

**Current active configuration resolves to: `fixed` + `raw`.**

## 5. Metrics collected (`EvalResult`)

Rule (same as `forge.observability`): a value the provider/runtime did not
report is `None` — never `0`, never invented.

Primary: `success`, `input_tokens`, `output_tokens`, `total_tokens`,
`latency_ms`.

Secondary: `llm_calls`, `tool_calls`, `turns`, `reasoning_tokens`,
`cached_input_tokens`, `cost_usd` / `cost_available`,
`context_messages`, `context_char_count`.

> **`context_char_count` is a character-count proxy, not a tokenizer count.**
> It is carried straight through from `ConversationContext.approximate_char_count`.
> Step 3 does not change the context subsystem to get an exact number.

Reproducibility metadata on every row: `task_id`, `run_id`, `provider`,
`model`, `tool_strategy`, `context_strategy`, `temperature`,
`max_output_tokens`, `max_llm_turns`, `max_tool_calls`, `workspace`,
`started_at`, `finished_at`, `duration_s`, `trace_path`, `status`,
`checks_run` / `checks_passed` / `checks_detail`.

**Cost:** FORGE ships no price table. `cost_usd` stays `None` and
`cost_available` `False` unless the caller passes a real `TokenPricing`
(USD per 1M tokens, with a `source` note). Determinism is *not* claimed —
LLM inference is not deterministic; the goal is controlled configuration,
recorded on every row.

## 6. Serialization

JSONL, one `EvalResult` object per line (matches the existing trace format):

- `write_results_jsonl(results, path, append=False)`
- `append_result_jsonl(result, path)` — used by the CLI
- `read_results_jsonl(path)` — blank lines skipped; a malformed line raises
  `ValueError` with the line number rather than dropping data.

No database. Results collect into one file for later analysis.

## 7. Aggregation

`aggregate_results(results) -> AggregateStats` and
`group_results(results) -> {strategy_key: AggregateStats}`.

None-safe behaviour:

- **Sum** of a metric: `None` if *every* row is `None` for it; otherwise the
  known values are summed and unknowns contribute nothing.
- **Average** of a metric: mean over the rows that reported it; `None` if none
  did.
- `success_rate`: `None` when there are no results (zero denominator).
- `cost_per_successful_task`: `None` when cost is unavailable or no run
  succeeded.

`AggregateStats` also carries the homogeneous `provider` / `model` /
`tool_strategy` / `context_strategy` for the group (or `None` if mixed).

## 8. How to run a small evaluation

```bash
# inline task, temp workspace (auto-cleaned), append one row to runs/eval_results.jsonl
forge evaluate --task "Create calculator.py with add(a, b) and multiply(a, b)." --task-id calc

# task file with artefact checks, explicit workspace, explicit results file
forge evaluate --task-file tasks/calc.json \
               --workspace /tmp/forge_eval_ws \
               --results-file runs/eval_results.jsonl
```

Task file shape:

```json
{
  "task_id": "calc-add-multiply",
  "prompt": "Create a Python file named calculator.py ...",
  "checks": [
    {"path": "calculator.py", "must_contain": ["def add(a, b)", "def multiply(a, b)"]}
  ],
  "metadata": {"source": "pilot"}
}
```

`forge evaluate` exit codes match `forge run`: `0` success, `1` run did not
succeed, `2` configuration problem (missing key, or an unimplemented strategy
such as `--tool-strategy adaptive`).

Programmatic use:

```python
from forge.config import settings
from forge.evaluation import EvalTask, ExperimentConfig, EvaluationRunner

task = EvalTask(task_id="calc", prompt="Create calculator.py ...")
exp = ExperimentConfig.from_settings(settings, task_id=task.task_id)  # fixed + raw
result = EvaluationRunner().run(task, exp)          # temp workspace, cleaned
```

## 9. Currently implemented vs. not

**Implemented (this step):**

- `EvalTask` + `ArtifactCheck`
- `ExperimentConfig` (+ `from_settings`, `resolve_strategies`)
- `EvaluationRunner` (reuses `AgentRuntime`; isolated workspace; artefact checks)
- `EvalResult` + JSONL read/write
- `aggregate_results` / `group_results` / `AggregateStats`
- `forge evaluate` CLI
- unit tests + one real Gemini verification run

**Current strategy:** `Fixed + Raw`

**Future strategies — NOT implemented by this step:**

- `Adaptive + Raw`
- `Fixed + Managed`
- `Adaptive + Managed`

These names are recognised so future result files are unambiguous, but
`resolve_strategies()` raises `NotImplementedError` for them.

## 10. How this supports the future 2×2 experiment

Every `EvalResult` records `tool_strategy` and `context_strategy` alongside
`provider`, `model` and the generation parameters. `group_results()` already
buckets by `"<tool>+<context>"`. When `AdaptiveToolExposure` and
`ManagedContextStrategy` are implemented, the *only* change needed here is to
add their names to `IMPLEMENTED_TOOL_STRATEGIES` /
`IMPLEMENTED_CONTEXT_STRATEGIES` and return them from `resolve_strategies()`.
The runner, result schema, serialization and aggregation are already able to
run and compare all four conditions:

| | Raw context | Managed context |
|---|---|---|
| **Fixed tools** | implemented | future |
| **Adaptive tools** | future | future |

## 11. Notes / limitations

- Task success for check-less tasks == "run reached `completed`". Artefact
  checks are string/existence assertions, not a test runner.
- `context_char_count` is a proxy (see §5).
- Cost is only ever computed from caller-supplied pricing.
- The single real verification run is **not** research data and not a
  benchmark result — it only confirms the plumbing works end to end.
