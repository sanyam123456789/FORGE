# Step 4 — Baseline Evaluation / Controlled Task Suite

## 1. Purpose

Step 4 adds a small, **version-controlled** suite of coding tasks that the
Step 3 evaluation infrastructure can run repeatedly under different FORGE
configurations later.

It is the **evaluation substrate**, not an experiment and not a benchmark:

- no adaptive tool exposure, no managed context, no 2×2 run
- no Pi baseline, no SWE-bench integration
- running the suite once produces **no research conclusions**

The research question is unchanged:

> Can a coding agent achieve comparable software-engineering task performance
> with lower token usage, inference cost, latency, and context consumption
> through adaptive tool exposure and interaction-context management?

## 2. Why a controlled task suite

The eventual comparison is `A = Fixed+Raw`, `B = Fixed+Managed`,
`C = Adaptive+Raw`, `D = Adaptive+Managed`. For those numbers to mean
anything, the *same* task must run under every configuration with only the
tool/context strategy changing.

So a task's **identity is its `task_id` and nothing else** — never the
strategy, provider, model, or run id. `create-string-utils` is one task; it
will later be executed as `create-string-utils / fixed+raw`,
`create-string-utils / adaptive+managed`, … from the *same* definition, with
no duplication.

## 3. Task representation

Step 4 reuses the Step 3 `EvalTask` (`forge/evaluation/task.py`) — no parallel
abstraction. Two small, backward-compatible additions were made to it:

| Addition | What / why |
|---|---|
| `EvalTask.fixtures: tuple[FixtureFile, ...]` | Starter files written into the isolated run workspace **before** the agent starts (needed for editing / bug-fix / refactor / multi-file tasks). `EvalTask.provision(workspace)` does the writing; `EvaluationRunner` calls it. |
| `ArtifactCheck.must_contain_any` / `must_not_contain` | `must_contain` is all-of (AND). `must_contain_any` is at-least-one-of (OR) — for bug fixes that can be written several correct ways. `must_not_contain` asserts a fragment (e.g. the bug) is gone. Still plain substring assertions; still not a test runner. |

Fixture and check paths are validated to be workspace-relative (no `..`, not
absolute).

## 4. Task categories

| Category | Tasks |
|---|---|
| `file_creation` | `create-string-utils`, `create-json-config-io` |
| `file_editing` | `edit-report-summary` |
| `bug_fix` | `fix-inclusive-sum`, `fix-normalize-trim` |
| `small_feature` | `add-greet-excited-flag` |
| `refactoring` | `refactor-round-money-helper`, `rename-tally-to-count-items` |
| `testing` | `add-mathlib-tests` |
| `multi_file` | `multi-file-timeout-setting` |

**10 tasks, 7 categories.** Every task is small, deterministic, offline
(except the LLM call), and verifiable with automated checks.

## 5. Task file format

One `NN-<task-id>.json` per task in `experiments/tasks/`. The `NN` prefix only
orders the listing; identity is the `task_id` field.

```json
{
  "task_id": "fix-inclusive-sum",
  "prompt": "The function inclusive_sum(n) in ranges.py ... Fix the implementation.",
  "fixtures": { "dir": "fixtures/fix-inclusive-sum" },
  "checks": [
    {
      "path": "ranges.py",
      "must_contain": ["def inclusive_sum(n)"],
      "must_contain_any": ["range(n + 1)", "range(1, n + 1)", "n * (n + 1)"],
      "must_not_contain": ["sum(range(n))"]
    }
  ],
  "metadata": { "category": "bug_fix", "difficulty": "medium", "offline": true }
}
```

- `fixtures` may be `{"dir": "<relative path>"}` (real files under
  `experiments/tasks/fixtures/<task-id>/`, kept readable/lintable) **or** an
  inline `[{"path": ..., "content": ...}]` list (the form
  `EvalTask.to_dict()` produces).
- `checks` and `metadata` are optional; a task with no checks succeeds when
  the run reaches `completed`.

## 6. Workspace / fixture approach

`EvaluationRunner` already creates an **isolated workspace** per run (a temp
dir unless one is passed) and cleans up temp dirs it made. Step 4 adds one
line to the runner: right after the workspace is created and before the agent
starts, `task.provision(workspace)` writes the task's fixture files in.

Starter files live only under `experiments/tasks/fixtures/` — never in
`forge/`, never at the repo root, and they are never imported by the test
suite (`pytest` `testpaths = ["tests"]`).

## 7. Success criteria

Machine-checkable only:

- file exists / is absent
- required substrings present (`must_contain`)
- at least one alternative present (`must_contain_any`)
- forbidden substrings absent (`must_not_contain`)

`EvaluationRunner` sets `EvalResult.success = (status == "completed") and
(checks did not fail)`; the per-check detail strings are recorded in
`checks_detail`. There is no subjective "looks correct" check and no
arbitrary code execution during checking — the agent may run `python -m
pytest` on its own work via the existing `run_shell` tool (safety controls
unchanged), but the evaluation check itself stays static.

## 8. Loading and running

Loader: `forge/evaluation/suite.py` (returns ordinary `EvalTask`s).

```python
from forge.evaluation.suite import load_suite, load_task, get_task, list_task_ids

load_suite()                      # all tasks, validated, duplicate-id-checked
get_task("fix-inclusive-sum")     # one task by id
load_task("experiments/tasks/01-create-string-utils.json")
```

CLI:

```bash
forge tasks                                   # list the suite
forge tasks --json                            # machine-readable

forge evaluate --suite-task-id create-string-utils
forge evaluate --suite-task-id fix-inclusive-sum \
               --workspace /tmp/ws --results-file runs/eval_results.jsonl
```

`forge evaluate` still accepts `--task "<inline prompt>"` and
`--task-file <path>`; `--suite-task-id` is the third, mutually-exclusive way
to name what to run. Exit codes are unchanged (`0` success, `1` run did not
succeed, `2` config problem / unknown task / unimplemented strategy).

## 9. Validation

`load_task` / `load_suite` raise `TaskSuiteError` (a `ValueError`) with the
file path for: invalid JSON, non-object JSON, missing/empty `task_id`,
missing/empty `prompt`, `checks` not a list, a malformed check (missing
`path`), non-object `metadata`, an unsafe check/fixture path, a missing or
empty fixture directory, an unreadable fixture file, and **duplicate
`task_id`s across the suite**.

## 10. Current state

**Implemented**

- 10-task baseline suite (`experiments/tasks/`) with fixtures
- `EvalTask` fixtures + `provision()`; `ArtifactCheck.must_contain_any` /
  `must_not_contain`
- `forge.evaluation.suite` loader + validation
- `forge tasks` and `forge evaluate --suite-task-id`
- unit tests + a real plumbing verification

**Current evaluation substrate:** `Fixed + Raw` only.

**Not implemented (later steps)**

- Adaptive tool exposure (Step 5)
- Managed context (Step 6)
- The controlled 2×2 experiment (Step 7)
- Pi external baseline
- Full SWE-bench integration

`ExperimentConfig` still recognises `adaptive` / `managed` as names but
`resolve_strategies()` raises `NotImplementedError` for them.

## 11. How this supports the 2×2 experiment

Each task carries no strategy information, so `load_suite()` × the four
`ExperimentConfig` strategy pairs will later produce 40 runs (10 tasks × 4
configs) whose `EvalResult` rows differ only in `tool_strategy` /
`context_strategy`. `group_results()` already buckets by `"<tool>+<context>"`.
Step 4 changes nothing about the runner's measurement path — it only supplies
the tasks.

## 12. Limitations

- Checks are substring/existence assertions. They confirm the required
  symbols and the removal of a known-bad fragment; they do not prove full
  behavioural correctness. Task prompts are written to make the required
  content unambiguous.
- `experiments/` is not packaged in the `forge` wheel. `DEFAULT_TASKS_DIR`
  resolves relative to the repo root, so the suite is available from a source
  checkout; pass an explicit `tasks_dir` otherwise.
- Task difficulty is deliberately low — the suite is for reliable controlled
  measurement, not for maximising a score.

## 13. Verification vs. research data

The Step 4 real run(s) only confirm the plumbing: tasks load, fixtures are
provisioned into the isolated workspace, `EvaluationRunner` drives the one
real agent loop, checks evaluate, the `EvalResult` carries the stable
`task_id`, and JSONL results + traces are written with no Step 3 regression.

Those runs are **not** benchmark or research data. No success rate, token
saving, latency, or cost claim is made here — those require the controlled
experiment in a later step.
