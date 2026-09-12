# Step 7 — Formal 2×2 Controlled Experiment

## 1. Purpose / research question

Steps 5 and 6 each shipped one independent intervention:

* **Arm A (tool exposure):** Fixed (every registered tool, every turn) vs.
  Adaptive (a deterministic, keyword-classified subset).
* **Arm B (context management):** Raw (full history, unchanged) vs. Managed
  (deterministic, rule-based pruning/compression — see
  `docs/step-06-managed-context.md`).

Both were individually selectable through `ExperimentConfig`/
`EvaluationRunner`, but nothing had run the four combinations together,
under the *same* tasks and the *same* held-constant conditions, and
collected them into one comparable, reloadable output. That is what Step 7
adds: `forge.evaluation.matrix`.

> Research question: holding the task, provider, model, and every runtime
> parameter constant, how do the four combinations of {tool exposure} ×
> {context management} compare on task success and resource usage
> (tokens, tool calls, LLM calls, latency, context size)?

Step 7 produces the *data* to answer that question — one structured,
reloadable JSONL row per (task, arm) execution. It deliberately does **not**
compute significance, effect sizes, or any other statistic over those rows;
that is Step 8 (see §10).

## 2. The four arms

| Arm id | Tool exposure | Context | Notes |
|---|---|---|---|
| `fixed_raw` | Fixed | Raw | The baseline — `ARMS[0]`, and still the default for `AgentConfig`/`ExperimentConfig` everywhere else in FORGE. |
| `fixed_managed` | Fixed | Managed | |
| `adaptive_raw` | Adaptive | Raw | |
| `adaptive_managed` | Adaptive | Managed | |

These are stable, machine-readable identifiers
(`forge.evaluation.matrix.ARM_FIXED_RAW`, etc., or just the string), declared
once as `ARMS: tuple[ExperimentArm, ...]` in `forge/evaluation/matrix.py`.
`ExperimentArm` is a tiny frozen dataclass: `arm_id`, `tool_strategy`,
`context_strategy` — nothing else. `MatrixRunner.run()` always iterates arms
in this exact declared order (filtered to whatever subset was requested —
see `arms_by_ids()`, which returns the requested ids in *canonical* `ARMS`
order regardless of the order they were requested in). Reordering `ARMS`
requires updating `tests/test_matrix.py::TestArms` and this document.

## 3. Fairness controls

For a comparison across arms to mean anything, every arm must see the same
task under the same conditions, differing *only* in the two strategies being
studied. `MatrixRunner` enforces this structurally, not by convention:

| Held constant across all four arms | How |
|---|---|
| Task prompt, task_id, checks, fixtures | The exact same `EvalTask` object is passed to every arm's `EvaluationRunner.run()` call — no per-arm copies, no per-arm mutation. |
| Provider, model, temperature, max_output_tokens, max_llm_turns, max_tool_calls | All four come from one `ForgeSettings`-like object passed once to `MatrixRunner(settings=...)`; every arm's `ExperimentConfig` is built via `ExperimentConfig.from_settings(settings, ...)` with no per-arm overrides. |
| Tool registry | Every arm's `AgentRuntime` builds the identical five-tool `build_default_registry()` registry (see `forge/builtin_tools.py`) — only the *exposed subset* of that registry differs per arm, which is exactly the studied variable, not a fairness violation. |
| Task ordering / arm ordering | Deterministic — see §2 and §4. |

| Never shared between arms or tasks | How |
|---|---|
| Workspace | Every (task, arm) pair gets its own directory: `<output_dir>/<experiment_id>/workspaces/<task_id>/<arm_id>/`, created with `mkdir(parents=True, exist_ok=False)` — a collision is a hard error, never a silent merge. Fixtures are provisioned fresh into each one independently by `EvalTask.provision()`. |
| Provider instance | `MatrixRunner` takes a `provider_factory: () -> LLMProvider`, called once *per task-arm execution* — never a single shared instance — so nothing (a scripted response queue, an SDK client's internal state) leaks between arms. The real (Gemini) default factory is `lambda: get_provider(settings_override=settings)`. |
| Conversation / agent state | Each execution is a fresh `AgentRuntime(...).run()` call with its own `ConversationContext` — the existing Step 2 guarantee, unchanged. |

This is why `MatrixRunner` never reuses a single `EvaluationRunner` instance
across arms either — a fresh `EvaluationRunner` (fresh provider, fresh trace
dir target, same settings) is constructed for each (task, arm) pair.

## 4. Deterministic ordering (no randomness anywhere)

FORGE has no randomness to seed (temperature 0.0 by default, every strategy
in Steps 5/6 is a fixed rule, not a sampled one). "Deterministic ordering"
therefore means: the same inputs always produce results in the same order.

* **Tasks** are visited in the order the caller's `tasks` list gives them —
  normally the suite loader's filename-sorted order (`forge.evaluation.suite.load_suite`).
* **Arms** are visited in the fixed, declared `ARMS` order for every task,
  regardless of what order a `--arm`/`arms=` subset was requested in
  (`arms_by_ids()` normalises this).
* The outer loop is task-major, arm-minor: `for task in tasks: for arm in arms: ...`.

`tests/test_matrix.py::test_deterministic_task_and_arm_ordering` pins this
exact interleaving.

## 5. Experiment runner design

`MatrixRunner` (`forge/evaluation/matrix.py`) is an orchestration layer
*above* the existing `EvaluationRunner` — **it contains no second agent
loop**. Every single (task, arm) execution is exactly one
`EvaluationRunner.run()` call, which is itself exactly one
`forge.agent.AgentRuntime.run()` call (unchanged since Step 2). `MatrixRunner`
only adds:

1. The four-arm × N-task loop (§4).
2. A fresh, isolated workspace path per (task, arm) pair (§3).
3. A fresh provider instance per (task, arm) pair (§3).
4. Decorating the returned `EvalResult` with experiment/arm metadata that
   `EvaluationRunner` itself has no reason to know about (`experiment_id`,
   `arm_id`, `task_category`, `exposed_tools` — see §7).
5. Catching any exception that escapes step 2/3/`EvaluationRunner.run()`
   itself (not a provider or in-loop error — both already become a normal,
   non-raising `EvalResult`, see §6) and turning it into a distinguishable
   `runner_error` result instead of aborting the whole batch (§6).
6. Writing one JSONL results file plus one metadata file per experiment (§8).

`forge/agent.py` was **not modified** by this step. No new agent-loop,
provider-call, or tool-dispatch code was introduced anywhere in
`forge/evaluation/matrix.py` — this is asserted directly by
`tests/test_matrix.py::test_matrix_module_introduces_no_second_agent_loop`.

### `exposed_tools`: informational, not a second source of truth

`MatrixRunner` also records which tool names an arm *would* expose for a
task, by calling `ToolExposureStrategy.select(registry, context=None,
task=task.prompt)` directly. This is a read-only, side-effect-free
computation (see `forge/tools.py`: both `FixedToolExposure.select()` and
`AdaptiveToolExposure.select()` are pure functions of the registry and the
task's prompt text, never of `context`) run against a registry built for
the same workspace — it cannot desync from what the real
`AgentRuntime` run actually used, because it uses the identical strategy
object and the identical classification rule.

## 6. Failure handling

Six categories are distinguished — never collapsed into a single boolean —
via `EvalResult.status` / `.success` / `.error`, and summarised by the new
`outcome_category(result) -> str` helper:

| Category | `EvalResult.status` | Meaning |
|---|---|---|
| `success` | `"completed"`, `success=True` | The run finished and its checks (if any) passed. |
| `completed_incorrect` | `"completed"`, `success=False` | The run finished, but the produced artefact failed its checks — a wrong answer, not an error. `error` is `None`. |
| `runtime_error` | `"failed"` | An unexpected exception occurred *inside* the agent loop (already caught by `AgentRuntime.run()` — see Step 2). |
| `provider_error` | `"provider_error"` | The LLM provider call itself failed (including a quota error) — caught inside the loop, exactly like Step 2/3. Never treated as "the model got it wrong". |
| `limit_reached` | `"max_turns_exceeded"` / `"max_tool_calls_exceeded"` | A configured ceiling was hit. The closest existing FORGE concept to "timeout/interruption" — see Limitations (§11): there is no wall-clock per-run timeout. |
| `runner_error` | `"runner_error"` (new, Step 7) | The *orchestration* layer failed before a normal `AgentRun` could even be produced — workspace creation, provider construction, or similar. Distinct from all of the above; see below. |

A `runner_error` result is still a complete, well-formed `EvalResult` row:
`task_id`, `arm_id`, `experiment_id`, `tool_strategy`, `context_strategy`,
`task_category` and `error` (the exception's `type: message`) are always
populated, but the run-metric fields (tokens, latency, calls, checks) stay
at their unavailable defaults — they are never fabricated as `0`.

Any task-arm execution that raises is caught in
`MatrixRunner._run_one()`; the batch continues with the next (task, arm)
pair. A `KeyboardInterrupt` is **not** caught (it is not an `Exception`
subclass), so a user can still abort an in-progress experiment; whatever
results were already appended remain on disk (see §8).

## 7. Result schema

Every row is an `EvalResult` (`forge/evaluation/result.py`) — the same
structure Steps 3/5/6 already use, extended (this step) with four new,
optional, `None`-safe fields:

| New field | Meaning |
|---|---|
| `experiment_id` | Which `MatrixRunner.run()` batch this row belongs to. |
| `arm_id` | `"fixed_raw"` / `"fixed_managed"` / `"adaptive_raw"` / `"adaptive_managed"`. |
| `task_category` | Copied from `EvalTask.metadata["category"]`, if the task defines one. |
| `exposed_tools` | The tool names this arm would expose for this task (see §5). |

Everything else on `EvalResult` is exactly what Step 3/5/6 already defined
and is populated identically to a plain `EvaluationRunner.run()` call:
`tool_strategy`, `context_strategy`, `status`, `success`, `error`,
`input_tokens`/`output_tokens`/`total_tokens` (`None` when the provider
didn't report them — never fabricated), `llm_calls`, `tool_calls`, `turns`,
`latency_ms`, `duration_s`, `context_messages`/`context_char_count`,
`context_items_dropped`/`context_items_compressed`/`context_chars_saved`
(Step 6), `checks_run`/`checks_passed`/`checks_detail`, `workspace`,
`trace_path`, `started_at`/`finished_at`. Cost is populated only if the
caller supplies real `TokenPricing` — FORGE ships no price table, exactly
as before.

No new parallel result type was created — this is the same `EvalResult`,
the same `to_dict()`/`to_json()`, the same `read_results_jsonl()` /
`write_results_jsonl()` / `append_result_jsonl()` writer functions Step 3
already shipped.

## 8. Output format

```
<output_dir>/<experiment_id>/
├── metadata.json      # MatrixExperimentSummary — see below
├── results.jsonl       # one EvalResult per line, one per (task, arm)
├── traces/             # one <run_id>.jsonl AgentRuntime trace per execution
└── workspaces/
    └── <task_id>/
        └── <arm_id>/    # that execution's isolated workspace (kept, not deleted)
```

* `output_dir` defaults to `runs/experiments` (CLI: `--output-dir`).
* `experiment_id` defaults to `exp_<UTC timestamp>_<8 hex chars>` (CLI:
  `--experiment-id` to pin one explicitly — e.g. for a reproducible file
  name). The experiment directory is created with `exist_ok=False`: running
  the same `experiment_id` twice against the same `output_dir` raises
  `FileExistsError` rather than silently overwriting the first run's output
  (`tests/test_matrix.py::TestOutput::test_rerunning_the_same_experiment_id_does_not_overwrite`).
* `results.jsonl` is appended to incrementally, one row per completed
  task-arm execution — not written once at the end — so an interrupted
  experiment still leaves every already-completed row on disk, reloadable.
* `metadata.json` is a `MatrixExperimentSummary`: `experiment_id`,
  `started_at`/`finished_at`, `provider`/`model`/`temperature`/
  `max_output_tokens`/`max_llm_turns`/`max_tool_calls` (the held-constant
  configuration — see §3), `arm_ids`, `task_ids`, `output_dir`,
  `results_file`, `label`. Reload with
  `forge.evaluation.matrix.read_matrix_metadata(path)`.
* Reload results with the existing `forge.evaluation.read_results_jsonl(path)`
  — no new reader was written.

This output is what Step 8 will consume; nothing here computes an aggregate
statistic beyond the same `aggregate_results()`/`group_results()` Step 3
already shipped (used only for the CLI's human-readable summary table — see
§9).

## 9. CLI

```bash
# the full baseline suite, all four arms, against the real configured provider
forge experiment

# a quick local check: one task, one arm
forge experiment --task-id create-string-utils --arm fixed_raw --output-dir /tmp/exp

# two arms, a specific output location and experiment id
forge experiment --task-id fix-inclusive-sum --arm fixed_raw --arm adaptive_managed \
                  --output-dir runs/experiments --experiment-id pilot-2026-09 --label pilot
```

`--task-id`/`--arm` are each repeatable; omitting them runs the full suite /
all four arms. `--json` prints `{"summary": ..., "aggregates": {<arm_id>:
<AggregateStats>, ...}}`; the default text output is a small per-arm table
(task count, successes, success rate) plus the paths to `results.jsonl` and
`metadata.json`.

`forge experiment --help` explicitly warns that, with no provider override,
it calls the real configured LLM provider (Gemini by default) once per
task-arm execution and will consume real quota for anything beyond a small
`--task-id`/`--arm` subset — see §10 below for a real run's cost shape.
Nothing in the test suite calls the real provider: every `test_matrix.py`
and CLI `experiment` test injects a scripted `provider_factory` (matrix
tests) or monkeypatches `forge.llm.get_provider` (CLI tests), exactly as the
existing `evaluate` tests already do.

## 10. Running it

### Locally, with a scripted/fake provider (no quota, this is what the tests do)

```python
from forge.evaluation import MatrixRunner, ArtifactCheck, EvalTask
from forge.llm import LLMProvider, LLMResponse, ToolCall

class ScriptedProvider(LLMProvider):
    ...  # see tests/test_matrix.py or tests/test_evaluation.py for a full example

task = EvalTask(
    task_id="demo-calc",
    prompt="Create a new file named calculator.py with add(a, b).",
    checks=(ArtifactCheck(path="calculator.py", must_contain=("def add(a, b)",)),),
)
runner = MatrixRunner(
    settings=my_settings,                      # a ForgeSettings-like object
    provider_factory=lambda: ScriptedProvider([...]),  # fresh instance per run
    output_root="runs/experiments",
)
results, summary = runner.run([task])          # 4 results — one per arm
```

### A real Gemini experiment, later, manually

1. Configure `FORGE_LLM_API_KEY` (or `GEMINI_API_KEY`/`GOOGLE_API_KEY`) as for
   any other FORGE run — never commit it, never put it in a file this repo
   tracks.
2. Start with a small subset to confirm cost/behaviour before a full run:
   `forge experiment --task-id <one id> --arm fixed_raw`.
3. Scale up deliberately: `forge experiment --task-id <a few ids>` (all four
   arms), then eventually the full suite with no `--task-id`/`--arm` filter.
4. A full run costs `4 arms × N tasks` independent agent runs (each
   potentially several LLM turns) — budget real API quota accordingly. This
   step does not attempt to estimate or cap that cost; that is on whoever
   runs it.

This step deliberately stops at "the data is now producible and reloadable"
— no live Gemini 2×2 run was performed to validate Step 7 (see the
implementation report), matching the project instruction to validate
locally first.

## 11. Limitations

* **No wall-clock timeout.** FORGE has turn/tool-call ceilings
  (`max_llm_turns`, `max_tool_calls`) but no wall-clock deadline around a
  single task-arm execution. "Timeout/interruption" as a failure category
  is therefore represented by `limit_reached` (a ceiling hit) plus an
  uncaught `KeyboardInterrupt` aborting the whole experiment cleanly (§6) —
  not a new preemptive per-run timeout, which would require changes to the
  agent loop and/or the Gemini provider adapter that this step's
  instructions explicitly discourage.
* **`exposed_tools` is informational, computed once, not per-turn.** For
  `AdaptiveToolExposure` this is exactly what every turn actually uses
  (Step 5's classifier looks only at the task prompt, never at conversation
  state — see `docs/step-05-adaptive-tool-exposure.md` §3.2), so this is
  exact, not an approximation, for both implemented strategies. A future
  per-turn-varying strategy would need this recomputed per turn.
* **Workspaces are kept, not cleaned up, by default** (`cleanup_workspaces=False`
  on `MatrixRunner.run()`), unlike a bare `EvaluationRunner.run()` call with
  no explicit workspace (which auto-cleans its temp dir). This is
  deliberate — inspecting what each arm actually produced is often the
  point — but means a large experiment leaves a lot of on-disk state under
  `<output_dir>/<experiment_id>/workspaces/`.
* **No statistical analysis in this step.** Step 7 produces comparable rows;
  it does not itself compute significance, confidence intervals, or effect
  sizes over them. `forge.evaluation.analysis` (Step 8) consumes this step's
  output separately and still never computes those three things — see
  `docs/step-08-statistical-analysis.md` §7.
* **No dashboards or charts.** Only JSON/JSONL output and a small CLI text
  table.
* **No WhatsApp/Telegram integration.**
* **No Pi integration.**
* **No SWE-bench benchmark.**
* **No external-agent (Cursor/Copilot/etc.) comparison.**
* **No real Gemini 2×2 run was performed to validate this step** — see §10
  and the implementation report; local scripted validation only, per the
  project instructions (quota is limited).

## 12. Explicitly deferred

* Step 8 (statistical analysis of the `EvalResult` rows this step produces)
  is now implemented — see `forge.evaluation.analysis` and
  `docs/step-08-statistical-analysis.md`. It reads this step's
  `results.jsonl`/`metadata.json` only; nothing in this document changed.
* Dashboards/charts are **not implemented**.
* WhatsApp/Telegram integration is **not implemented**.
* Pi integration is **not implemented**.
* SWE-bench is **not implemented**.
* External-agent comparison is **not implemented**.
