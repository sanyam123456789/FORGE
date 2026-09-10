# experiments/

Version-controlled research data for FORGE. **Not** shipped in the `forge`
wheel — this tree is for running controlled evaluations from a source checkout.

## `tasks/`

The **baseline evaluation task suite** (Step 4): a small, curated set of
self-contained coding tasks used to validate FORGE's evaluation pipeline and,
later, to compare strategy configurations under controlled conditions.

- One `NN-<task-id>.json` file per task. The `NN` prefix only orders the
  listing; the stable identity is the `task_id` **inside** the file.
- Starter files for a task live under `tasks/fixtures/<task-id>/` and are
  referenced from the task file as `"fixtures": {"dir": "fixtures/<task-id>"}`.
  `EvaluationRunner` writes them into the isolated run workspace before the
  agent starts.
- Loaded by `forge.evaluation.suite` into the ordinary Step 3 `EvalTask` type.

List / run:

```bash
forge tasks
forge evaluate --suite-task-id create-string-utils
```

See `docs/step-04-baseline-tasks.md` for the full description, categories, and
the (deliberately narrow) scope of what Step 4 does and does not do.

Everything here is offline except the LLM call itself. No network downloads,
no external repositories, no credentials.
