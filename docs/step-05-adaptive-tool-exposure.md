# Step 5 — Adaptive Tool Exposure

## 1. Purpose

Step 5 adds the first real intervention for **research arm A**: instead of
always exposing every registered tool to the LLM (`FixedToolExposure`), the
agent can now expose only the tools a task plausibly needs
(`AdaptiveToolExposure`).

The research question this step starts to answer:

> Can adaptive tool exposure reduce unnecessary tool calls, unnecessary model
> decisions, and token usage — without hurting task success — compared to the
> Fixed + Raw baseline?

Step 5 does **not** run the controlled comparison itself (that is Step 7's
2×2 experiment). It ships the intervention and the measurement hooks needed
to compare it later.

## 2. Scope

**In scope:** a deterministic `AdaptiveToolExposure` strategy, wired into the
existing `ToolExposureStrategy` seam (`forge/tools.py`), the existing
`ExperimentConfig` (`forge/evaluation/experiment.py`), and the existing CLI
(`forge evaluate --tool-strategy adaptive`).

**Out of scope (unchanged from Step 4):**

- Managed context (`ContextStrategy` still only has `RawContextStrategy`;
  `context_strategy="managed"` still raises `NotImplementedError`) — **not
  implemented yet**.
- The controlled 2×2 experiment (Step 7).
- WhatsApp/Telegram, Pi, Cursor/Copilot/Hermes/Antigravity comparisons,
  full SWE-bench, dashboards/UI.

No second agent loop was created. `AgentRuntime.run()` (`forge/agent.py`) is
**byte-for-byte unchanged in its control flow** — it already called
`tool_exposure.select(registry, ctx, task)` since Step 2; Step 5 only adds a
second concrete implementation of that interface. A regression guard
(`tests/test_agent.py::TestFixedToolExposure::test_runtime_uses_the_strategy_abstraction`)
asserts the word "adaptive" never appears in `forge/agent.py`.

## 3. Design

### 3.1 Fixed mode (baseline, unchanged)

`FixedToolExposure.select()` returns `None` (or a pinned static list),
meaning "expose every registered tool, every turn." This is exactly the
Step 2–4 behaviour and is untouched by this step.

### 3.2 Adaptive mode (new)

`AdaptiveToolExposure` (`forge/tools.py`) is a small, fully deterministic,
explainable policy — no ranking model, no embeddings, no LLM call to decide:

1. **Classify** the task's natural-language prompt into one of three fixed
   categories using case-insensitive substring matching against an ordered
   list of keyword rules (`AdaptiveToolExposure.RULES`). The first rule with
   a matching keyword wins.
   - `test`  — the task mentions running code (`pytest`, `run_shell`, "run
     the tests", "confirm the tests pass", ...). Checked **first**, so a task
     that both creates and runs a test file gets the full toolset it needs,
     not the narrower "create" one.
   - `create` — the task is about a brand-new file ("create a new file",
     "create a file named", ...).
   - `edit` — the task modifies existing code ("fix ", "rename", "refactor",
     "extract", "edit ", "modify", "update ", "change ", "add a "/"add an ").
   - no match → category is `None`.
2. **Map** the category to a fixed tuple of tool names
   (`AdaptiveToolExposure.CATEGORY_TOOLS`):

   | Category | Tools exposed |
   |---|---|
   | `create` | `read_file`, `write_file`, `list_directory` |
   | `edit`   | `read_file`, `edit_file`, `list_directory` |
   | `test`   | `read_file`, `write_file`, `edit_file`, `list_directory`, `run_shell` |
   | *(no match)* | **all** registered tools (same as Fixed) |

3. **Filter** the category's tool names against what `registry` actually has
   registered, so a registry that does not carry every built-in tool never
   raises `KeyError` from `ToolRegistry.get_schemas()`.
4. **Fall back** to `None` ("expose everything") whenever step 1 finds no
   category, or step 3 leaves nothing — the *only* way `AdaptiveToolExposure`
   can behave is to either narrow to a sensible subset or match Fixed
   exactly. It never returns an empty tool list and never guesses when it
   isn't confident.

Classification uses only the task's initial prompt text — never `context` —
so the same task string always produces the same subset, on every turn of a
run and across separate runs. This is what "deterministic" means here.

### 3.3 Selecting the mode

Programmatically (`forge/agent.py`, unchanged interface):

```python
from forge.agent import AgentConfig, AgentRuntime
from forge.tools import AdaptiveToolExposure

config = AgentConfig(
    task="Fix the off-by-one bug in ranges.py.",
    tool_exposure=AdaptiveToolExposure(),   # default is FixedToolExposure()
)
AgentRuntime(config).run()
```

Via the evaluation layer:

```python
from forge.evaluation import ExperimentConfig

exp = ExperimentConfig.from_settings(settings, task_id="fix-inclusive-sum",
                                      tool_strategy="adaptive")
tool_strategy, context_strategy = exp.resolve_strategies()
```

Via the CLI:

```bash
forge evaluate --suite-task-id fix-inclusive-sum --tool-strategy adaptive
forge evaluate --suite-task-id fix-inclusive-sum --tool-strategy fixed     # baseline, default
```

`--tool-strategy` now accepts `fixed` (default) or `adaptive`.
`--context-strategy` still only accepts `raw` — passing `managed` exits 2
with a `NotImplementedError` message, exactly as in Step 4.

## 4. Backward compatibility (Fixed + Raw)

- `FixedToolExposure` is untouched: same class, same `select()` contract,
  same default on `AgentConfig.tool_exposure`.
- `AgentRuntime.run()` control flow is unmodified — it already depended only
  on `ToolExposureStrategy.select()`.
- `ExperimentConfig` default (`tool_strategy="fixed"`) and its
  `resolve_strategies()` behaviour for `"fixed"` are unchanged.
- Every Step 2–4 test still passes unmodified (see §7).

## 5. Measurement / observability

No new metrics were invented — Step 5 only makes sure the **existing**
measurement infrastructure carries the extra strategy dimension it was
already designed to carry:

- `RunTracer.record_run_start(..., tool_exposure_strategy=..., tool_subset=...)`
  already recorded these fields (Step 2); they now read `"adaptive"` and the
  classified subset instead of always `"fixed"` / `None`.
- `RunTracer.record_llm_request(..., exposed_tools=...)` already recorded the
  per-turn exposed tool list; with `AdaptiveToolExposure` this list is the
  same narrowed subset on every turn (determinism, see §3.2).
- `EvalResult.tool_strategy` (Step 3) already existed as a field; it now
  takes the value `"adaptive"` for adaptive runs, and `group_results()`
  (Step 3 aggregation) already buckets by `"<tool>+<context>"`, so
  `adaptive+raw` results collect separately from `fixed+raw` with no
  aggregation code changes.
- Tool calls, LLM calls, tokens (input/output/total/cached/reasoning),
  latency, duration, success/failure, error, and trace path all flow through
  unchanged — `AgentRun` / `EvalResult` / `RunTracer` were not modified.

## 6. Files changed

**Modified:**

- `forge/tools.py` — added `AdaptiveToolExposure` (new class only; existing
  `Tool`, `ToolResult`, `ToolRegistry`, `ToolExposureStrategy`,
  `FixedToolExposure` untouched).
- `forge/evaluation/experiment.py` — `TOOL_STRATEGY_ADAPTIVE` moved from
  `FUTURE_TOOL_STRATEGIES` to `IMPLEMENTED_TOOL_STRATEGIES`;
  `resolve_strategies()` now returns `AdaptiveToolExposure()` for
  `tool_strategy="adaptive"`. `CONTEXT_STRATEGY_MANAGED` is unchanged
  (still future/unimplemented).
- `forge/evaluation/__init__.py` — exported `TOOL_STRATEGY_ADAPTIVE`.
- `forge/cli.py` — help text for `--tool-strategy` updated to list
  `adaptive` as implemented.
- `docs/architecture.md` — Arm A status table and module table updated.

**Not modified:** `forge/agent.py`, `forge/context.py`, `forge/builtin_tools.py`,
`forge/observability.py`, `forge/evaluation/runner.py`,
`forge/evaluation/result.py`, `forge/evaluation/aggregate.py`,
`forge/evaluation/task.py`, `forge/evaluation/suite.py`,
`experiments/tasks/*` (the Step 4 baseline suite and its results are
untouched).

## 7. Tests added

- `tests/test_tools.py::TestAdaptiveToolExposureClassification` — keyword
  classification for create/edit/test/ambiguous/empty/`None` prompts,
  case-insensitivity, and the "test beats create" priority rule.
- `tests/test_tools.py::TestAdaptiveToolExposureSelection` — tool subsets per
  category, the "no match → expose everything" fallback, the "nothing
  survives registry filtering → expose everything" fallback, determinism
  across repeated calls, filtering against a partial registry (no crash),
  and a battery of ambiguous/degenerate inputs that must never raise.
- `tests/test_agent.py::TestAdaptiveToolExposureEndToEnd` — the **real**
  `AgentRuntime` completing a create-task and an edit-task end-to-end with
  `AdaptiveToolExposure`, confirming the exposed tool schemas sent to the
  (scripted) provider match the expected narrowed subset every turn, that an
  unrecognised task still exposes every tool, and that the JSONL trace
  records `tool_exposure_strategy="adaptive"` with the expected
  `tool_subset`.
- `tests/test_evaluation.py` — `ExperimentConfig` now resolves `"adaptive"`
  to a real strategy (`test_adaptive_tool_strategy_is_implemented`);
  `"managed"` context remains unimplemented
  (`test_managed_context_strategy_is_not_runnable`,
  `test_unimplemented_strategy_raises_not_implemented` — updated to exercise
  `context_strategy="managed"` instead of the now-implemented adaptive tool
  strategy); a new `EvaluationRunner` test drives a real create-category task
  under `tool_strategy="adaptive"` and asserts both task success and the
  recorded trace metadata.
- `tests/test_cli.py` — `test_evaluate_unimplemented_strategy_exits_2`
  updated to use `--context-strategy managed` (the strategy still
  unimplemented); new `test_evaluate_adaptive_tool_strategy_runs` exercises
  `forge evaluate --tool-strategy adaptive` end-to-end and checks exit code 0
  and `tool_strategy == "adaptive"` in the JSON result.

All existing Step 1–4 tests were re-run unmodified except the three noted
above, whose assertions specifically encoded "adaptive is not implemented
yet" — the premise Step 5 exists to change. No other test's expectations
changed.

## 8. Example behaviour

```text
$ python -c "
from forge.tools import AdaptiveToolExposure
s = AdaptiveToolExposure()
print(s.classify('Create a new file named string_utils.py ...'))   # -> create
print(s.classify('Fix the off-by-one bug in ranges.py.'))          # -> edit
print(s.classify('Rename tally to count_items.'))                  # -> edit
print(s.classify('Run pytest to confirm the tests pass.'))         # -> test
print(s.classify('Please summarise this codebase.'))               # -> None (fallback: all tools)
"
```

Running the Step 4 baseline suite's prompts through the classifier confirms
every task lands in the category its own `metadata.category` implies
(`file_creation`→`create`, `bug_fix`/`file_editing`/`small_feature`/
`refactoring`/`multi_file`→`edit`, `testing`→`test`) — this is a sanity check
of the design, not a benchmark result.

## 9. Limitations

- **Coarse keyword matching.** An unusually-phrased task can land in the
  "no category matched" fallback (safe — full toolset — but no reduction) or,
  in principle, a category that doesn't perfectly fit. There is no
  confidence score or partial match.
- **Static, non-learning policy.** It does not adapt within a run (based on
  what the agent has already tried) or across runs (based on past
  successes/failures). This is deliberate for Step 5 — see the design
  requirements — and is the seam a later, more advanced selector (embedding
  similarity, per-turn re-ranking, a learned model) would replace, with zero
  change to `AgentRuntime` or `ExperimentConfig`.
- **Only the five built-in tools are categorised.** A newly added tool with
  no entry in `CATEGORY_TOOLS` is only ever offered via the safe fallback,
  never via a matched category, until `CATEGORY_TOOLS` is extended.
- **No research conclusion yet.** Step 5 does not claim adaptive exposure
  improves anything — it only makes the comparison possible. Whether it
  actually reduces tokens/latency/tool calls while preserving success is an
  empirical question for the Step 7 controlled 2×2 experiment (across many
  tasks and repeated runs, not the illustrative examples above).
- **Managed Context is not implemented.** Only the `raw` context strategy
  exists; `context_strategy="managed"` still raises `NotImplementedError`
  from `ExperimentConfig.resolve_strategies()`, exactly as before this step.

## 10. How this supports the future 2×2 experiment

`ExperimentConfig.strategy_key` already produces `"fixed+raw"` and
`"adaptive+raw"` labels, and `forge.evaluation.aggregate.group_results()`
already buckets `EvalResult`s by that key. With `adaptive` now a resolvable
strategy, running the existing Step 4 suite once under `tool_strategy=fixed`
and once under `tool_strategy=adaptive` (both `context_strategy=raw`)
already produces two directly comparable result sets — `A = Fixed+Raw` and
`C = Adaptive+Raw` in the eventual 2×2 — with no further plumbing. `B`
(`Fixed+Managed`) and `D` (`Adaptive+Managed`) still require Managed Context
(Step 6) before `context_strategy="managed"` can resolve; Step 7 is where all
four conditions are actually run and compared.
