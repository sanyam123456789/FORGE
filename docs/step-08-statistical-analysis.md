# Step 8 — Statistical Analysis and Comparison

## 1. Purpose / scope

Step 7's `MatrixRunner` (`forge/evaluation/matrix.py`) produces `results.jsonl`
(+ `metadata.json`): one comparable `EvalResult` row per (task, arm)
execution across the four arms — `fixed_raw`, `fixed_managed`,
`adaptive_raw`, `adaptive_managed`. Step 7 deliberately stopped at "the data
is now producible and reloadable" and computed no comparison beyond a simple
per-arm success-rate table.

Step 8 (`forge/evaluation/analysis.py`) reads that data and produces a
reproducible, machine-readable comparison across the four arms: outcome
counts, per-metric descriptive statistics, paired task comparisons, factor-
level comparisons (tool exposure vs. context strategy), and — where the data
supports it — a real (stdlib-only) statistical test statistic.

It does **not** redesign the agent, tool strategies, context strategies, or
the experiment runner. It contains no agent loop and makes no provider or
network call of any kind — it only reads JSONL/JSON files already on disk.

## 2. Input format

- **Results**: a JSONL file of `EvalResult` rows (see
  `forge/evaluation/result.py`), one JSON object per line — exactly what
  `forge experiment` / `MatrixRunner.run()` writes as `results.jsonl`, or
  what `forge evaluate` appends to. Loaded with the existing
  `forge.evaluation.read_results_jsonl()` — a malformed JSON line raises
  `ValueError` naming the line number; nothing is silently dropped.
- **Metadata** (optional): a `metadata.json` written by `MatrixRunner`
  (`MatrixExperimentSummary`). If `--metadata` is not given, Step 8 looks for
  `metadata.json` next to the results file and uses it if present; its
  absence or being unparsable is not an error (every value it could supply
  beyond free-text fields like `label` is also inferable from the records).

Rows with `arm_id = None` (e.g. from a plain `forge evaluate` run rather than
`forge experiment`) are tolerated: they are counted in an `"unassigned"`
outcome/metric bucket but excluded from every arm-pair/factor-level/
statistical-test computation, which require a real arm id.

## 3. Metrics analyzed and their denominators

### Outcome / success

Every result is bucketed into exactly one of six mutually-exclusive
categories via the existing `forge.evaluation.matrix.outcome_category()`:
`success`, `completed_incorrect`, `runtime_error`, `provider_error`,
`limit_reached`, `runner_error` (plus a defensive `unknown` for a future
status this step doesn't yet recognise).

Per arm (and overall, key `"__all__"`), `OutcomeSummary` reports the raw
count of each category plus two success rates with explicit denominators:

- `success_rate_all = success / total` (`denom_all = total`)
- `success_rate_excl_infra_errors = success / (total - provider_error - runner_error)`
  (`denom_excl_infra_errors`) — excludes runs that never got a fair chance
  because the *infrastructure* (provider call, orchestration layer) failed,
  not the agent's own behaviour.

Both are `None` when their denominator is zero — never a fabricated `0.0`.

### Numeric metrics

Only fields that actually exist on `EvalResult` are analyzed (see
`NUMERIC_METRICS` in `forge/evaluation/analysis.py`): `duration_s`,
`latency_ms`, `llm_calls`, `tool_calls`, `turns`, `input_tokens`,
`output_tokens`, `total_tokens` (provider total if present, else
`input_tokens + output_tokens`, else unavailable — same fallback as
`EvalResult.resolved_total_tokens`), `reasoning_tokens`,
`cached_input_tokens`, `context_messages`, `context_char_count`,
`context_items_dropped`, `context_items_compressed`, `context_chars_saved`
(Step 6), `exposed_tool_count` (`len(exposed_tools)`, Step 7), and `cost_usd`.

Nothing not present in the schema is invented — there is no per-token cost
breakdown, no wall-clock timeout metric (Step 7 has none), and no dashboard
metric of any kind.

## 4. Missing / invalid / not-applicable — never zero

Every value pulled from a record is classified into exactly one bucket
before any statistic is computed:

| Category | Meaning | Example |
|---|---|---|
| `valid` | a real, finite number | `input_tokens = 120` |
| `missing` | the field is `None` — the run/provider never reported it | a provider that doesn't report token counts |
| `invalid` | present but the wrong type or non-finite (`NaN`/`Inf`) | a corrupted JSONL row with `"input_tokens": "banana"` |
| `not_applicable` | deliberately not measured, distinct from "missing" | `cost_usd` when `cost_available` is `False` (no `TokenPricing` was ever supplied — FORGE ships no price table) |

`MetricSummary` reports all four counts plus `mean`/`median`/`minimum`/
`maximum`/`total` computed **only** over `valid` values, and `stdev` only
when there are 2+ valid observations (else `None`). A metric with zero valid
observations across the whole analyzed dataset is listed in the report's
top-level `unavailable_metrics` — never given a misleading `0.0` average.

`validate_results()` additionally raises a `malformed_numeric_value` warning
per metric that has any `invalid` value, naming the affected metric and up
to 10 `run_id`s — the problem is always surfaced, never hidden by silently
excluding the record.

## 5. Paired task comparison

Because the whole point of the 2×2 design is running the *same task* under
each arm, `paired_metric_comparison(results, arm_a, arm_b, metric)` matches
records by `task_id` (using the first record per (task, arm) cell — see
§6 on duplicates) and reports:

- `n_pairs` — tasks with a **valid** value in both arms.
- `per_task_diff` — `value_b - value_a` for each such task.
- `mean_diff` / `median_diff` over those diffs.
- `direction` — `"a_higher"` / `"b_higher"` / `"equal"`, from the sign of `mean_diff`.
- `lower_is_better` / `improved_arm` — a purely *informational* label (see
  `LOWER_IS_BETTER` in `analysis.py`) for metrics with an established
  "smaller is the efficiency win" direction (tokens, calls, cost, duration,
  latency, exposed-tool count). Metrics with no fixed direction (e.g.
  `context_items_dropped`) get `lower_is_better = None` and no improvement
  label — only the raw sign.
- `per_task_pct_change` — `(diff / value_a) * 100`, computed **only** when
  `value_a != 0`; tasks where it is mathematically undefined are listed
  separately in `pct_change_undefined_tasks`, never silently skipped or
  reported as `0%`/`inf%`.
- `tasks_without_record_in_a` / `_b` — tasks present in the *other* arm's
  records but with no record at all in this arm (a missing task-arm cell).
- `tasks_invalid_in_a` / `_b` — a record exists but its value for this metric
  is `invalid` (see §4).

For binary success/correctness, `success_discordance(results, arm_a, arm_b)`
reports the four paired counts (`both_success`, `both_fail`, `a_only`,
`b_only`) plus `n_pairs` — a sign-test-style input, but Step 8 stops at the
counts (see §7).

## 6. Factor-level comparisons

Held-arm-count-aware, per the 2×2 design (`ARM_FIXED_RAW`, `ARM_FIXED_MANAGED`,
`ARM_ADAPTIVE_RAW`, `ARM_ADAPTIVE_MANAGED`):

- **Tool exposure** (`factor_level.tool_exposure`): `fixed_raw` vs.
  `adaptive_raw`, and `fixed_managed` vs. `adaptive_managed` — i.e. the
  context strategy is held constant in each comparison, never collapsed
  across it.
- **Context strategy** (`factor_level.context_strategy`): `fixed_raw` vs.
  `fixed_managed`, and `adaptive_raw` vs. `adaptive_managed`.
- **Interaction** (`factor_level.interaction`): a descriptive
  difference-in-differences per metric, per task —
  `(adaptive_managed - adaptive_raw) - (fixed_managed - fixed_raw)` — over
  tasks with a valid value in **all four** arms. Computed only when all four
  arms are present in the analyzed selection; otherwise `interaction` is
  `null` and an `interaction_skipped` warning explains why. Every
  `InteractionSummary` carries an explicit `note` that this is descriptive
  only and establishes no causal claim, especially for a small or unbalanced
  task set — per the project instruction not to overclaim from limited data.

## 7. Statistical tests — what is and is not computed

Step 8 reports **real, correctly computed test statistics** and **never**
a p-value, confidence interval, effect size, or "significant"/"not
significant" label, anywhere in the output. This is a deliberate,
literal reading of the project instruction — the report gives you the
numbers to reason about; it does not tell you what to conclude.

- **Binary success**: `success_discordance()` (§5) — exact paired counts
  only. No sign-test p-value is derived from them.
- **Numeric paired metrics**: `wilcoxon_paired_metric(results, arm_a, arm_b,
  metric)` computes the standard Wilcoxon signed-rank **W statistic**
  (`min(W+, W-)`, average ranks for ties in `|diff|`) using a small
  stdlib-only implementation (no SciPy/NumPy dependency was added — see
  §9) — but stops there. `WilcoxonResult.note` says explicitly that no
  p-value/CI/significance is computed.

  The test is **skipped** (statistic `None`, `skipped=True`, and a
  human-readable `skip_reason`) when:
  - there are no paired observations at all (`"no paired observations available"`),
  - every paired difference is exactly zero (`"all paired differences are zero (no variation)"`),
  - or fewer than 4 non-zero paired differences remain
    (`"fewer than 4 non-zero paired differences"` — below this the rank-sum
    statistic has no useful resolution).

  A skip is always explicit and recorded — never a silently omitted key.

## 8. Data-quality validation

`validate_results()` and `check_matrix_completeness()` return a list of
`DataQualityIssue{severity, code, message, detail}` — `severity` is
`"error"` (structurally wrong — e.g. an unknown arm id) or `"warning"`
(worth knowing, not fatal). Nothing is ever silently dropped from the
dataset because of one of these findings; every original record is still
counted somewhere in the report.

| Code | Severity | Meaning |
|---|---|---|
| `empty_dataset` | warning | zero records to analyze |
| `missing_task_id` | error | a record's `task_id` is empty/whitespace |
| `missing_arm_id` | warning | `arm_id` is an empty string (not `None`) |
| `unknown_arm_id` | error | `arm_id` set but not one of the 4 canonical arms |
| `inconsistent_strategy_label` | error | `tool_strategy`/`context_strategy` don't match what the record's `arm_id` pins |
| `duplicate_task_arm` | warning | more than one record for the same `(experiment_id, task_id, arm_id)` |
| `mismatched_task_category` | warning | the same `task_id` carries different `task_category` values across records |
| `malformed_numeric_value` | warning | a metric field has a non-numeric/non-finite value on one or more records |
| `incomplete_matrix` | warning | one or more analyzed tasks lack a record for one or more analyzed arms |
| `partial_arm_selection` | warning | fewer than all 4 canonical arms are in scope — some factor-level/interaction comparisons are skipped |
| `interaction_skipped` | warning | the four-arm interaction comparison could not run (see §6) |
| `mixed_experiment_ids` | warning | the analyzed records span more than one `experiment_id` |

Duplicates are kept in every summary/count (never dropped), but paired
comparisons deterministically use only the **first** record encountered for
a given (task, arm) — averaging or guessing "the correct one" would be a
worse kind of silent behaviour than picking one, deterministically, and
flagging the duplication separately.

## 9. Why no p-values, and no new dependency

The project instructions are explicit: *"Do not invent p-values, confidence
intervals, effect sizes, or significance labels,"* and *"Do not add
dependencies merely for cosmetic analysis."* `pyproject.toml` deliberately
keeps FORGE's dependency list to `python-dotenv` + `google-genai`, each
individually justified. Step 8 adds **zero** new runtime dependencies — the
Wilcoxon rank-sum statistic is a ~20-line stdlib-only implementation
(`_wilcoxon_statistic` in `analysis.py`), and everything else is
`statistics`/`math`/`json`/`dataclasses`. This module works exactly the same
whether or not a scientific-computing package happens to be installed in the
environment.

## 10. Output schema

`forge.evaluation.analysis.AnalysisReport.to_dict()` — written as pretty
JSON by `write_analysis_json()`:

```jsonc
{
  "schema_version": 1,
  "generated_at": "2026-...T...+00:00",
  "source_results_path": "runs/experiments/<id>/results.jsonl",
  "source_metadata_path": "runs/experiments/<id>/metadata.json",  // or null
  "experiment_id": "exp_...",           // or null if mixed/unknown
  "arms_analyzed": ["fixed_raw", "fixed_managed", "adaptive_raw", "adaptive_managed"],
  "task_ids_analyzed": ["..."],
  "task_count": 2,
  "record_count": 8,
  "warnings": [ {"severity": "...", "code": "...", "message": "...", "detail": {...}} ],
  "outcome_summary": { "__all__": {...}, "<arm_id>": {...}, "unassigned": {...} },
  "metric_summary": { "__all__": {"<metric>": {...}}, "<arm_id>": {"<metric>": {...}} },
  "paired_comparisons": {
    "<arm_a>__vs__<arm_b>": {
      "success_discordance": {...},
      "metrics": { "<metric>": {...} }
    }
  },
  "factor_level": {
    "tool_exposure": { "fixed_raw__vs__adaptive_raw": {...}, "fixed_managed__vs__adaptive_managed": {...} },
    "context_strategy": { "fixed_raw__vs__fixed_managed": {...}, "adaptive_raw__vs__adaptive_managed": {...} },
    "interaction": { "<metric>": {...} } // or null, see §6
  },
  "statistical_tests": { "<arm_a>__vs__<arm_b>": { "<metric>": {...WilcoxonResult...} } },
  "unavailable_metrics": ["<metric>", ...],
  "metadata": { ...MatrixExperimentSummary... } // or null
}
```

The output is always written to a **separate** file — never the source
JSONL — and an existing output file is never overwritten unless
`overwrite=True` (CLI: `--overwrite`); otherwise `write_analysis_json()`
raises `FileExistsError`.

## 11. CLI

```bash
# Basic: analyze a Step 7 experiment's output; writes <results dir>/analysis.json
forge analyze --results runs/experiments/<experiment_id>/results.jsonl

# Explicit output path, refuse to clobber an existing one unless asked
forge analyze --results runs/experiments/<id>/results.jsonl \
              --output runs/experiments/<id>/analysis.json --overwrite

# Restrict to a subset of arms/tasks already present in the data
forge analyze --results results.jsonl --arm fixed_raw --arm adaptive_managed \
              --task-id create-string-utils

# Print the full JSON to stdout too
forge analyze --results results.jsonl --json
```

`forge analyze --help` documents every option. It performs **no** network
call and **no** provider call of any kind — it only reads local JSON/JSONL
files. Exit codes: `0` analysis computed and written; `2` a configuration
problem (results file not found, malformed JSONL, unknown `--arm` id, or an
output file already exists without `--overwrite`).

## 12. Explicitly NOT implemented (still)

- No dashboard or chart UI.
- No WhatsApp/Telegram integration.
- No Pi integration.
- No SWE-bench benchmark.
- No external-agent (Cursor/Copilot/etc.) comparison.
- No live/automatic Gemini experiment run — this step only ever reads
  JSON/JSONL files already produced by a prior `forge experiment`/
  `forge evaluate` run (or, in tests, by a scripted fake provider); it never
  invokes a provider itself.
- No ML- or LLM-based analysis of any kind.
- No p-values, confidence intervals, effect sizes, or significance labels —
  see §7 and §9 for why, and what is reported instead.
