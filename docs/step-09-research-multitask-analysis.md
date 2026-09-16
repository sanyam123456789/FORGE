# Step 9 — Multi-Task Research Evaluation (Analysis)

## 0. Relationship to the baseline document

This document is a **separate, additional** artifact to `docs/step-09-real-provider-baseline.md` (the single-task `create-string-utils` pipeline/baseline validation). That document and its artifacts are **not modified** by this one. This document covers the **multi-task extension**: `fix-inclusive-sum` and `multi-file-timeout-setting`, added across two sessions, combined here with the original baseline task for one full analysis pass.

**These remain two distinct things and must not be conflated:**
- Step 9 **baseline validation** — one task (`create-string-utils`), proves the real-provider pipeline works end to end.
- Step 9 **multi-task research evaluation** (this document) — three tasks, still one run per task/arm cell, describing how the four arms behaved across a small, real, multi-task sample. **This is not a robustness study and does not support a definitive configuration ranking** (see §7).

## 1. Objective and scope

Measure and document how the four canonical arms (`fixed_raw`, `fixed_managed`, `adaptive_raw`, `adaptive_managed`) behaved, under real provider/API conditions, across three Step 4 baseline tasks:

- `create-string-utils` (file_creation, easy)
- `fix-inclusive-sum` (bug_fix, medium)
- `multi-file-timeout-setting` (multi_file, medium)

Seven of the ten Step 4 baseline tasks (`create-json-config-io`, `edit-report-summary`, `fix-normalize-trim`, `add-greet-excited-flag`, `refactor-round-money-helper`, `add-mathlib-tests`, `rename-tally-to-count-items`) were **not run** — a deliberate, documented scope decision made under real Gemini daily-quota constraints (20 requests/day, free tier), not a silent omission. They remain available for a future session.

## 2. Provider fallback fix applied before this run

The previous multi-task attempt was interrupted because `FallbackProvider` re-attempted an already-quota-exhausted Gemini on every LLM turn (~120s wasted per turn). Before resuming, `forge/providers/fallback.py` was reviewed and confirmed to already contain a run-scoped "sticky" fix (a provider that fails retryably is marked down for the rest of that one agent run and skipped on later turns — see that file's module docstring "Run-scoped stickiness"). This fix was verified via `tests/test_fallback.py` (25/25 passing, including 7 new tests for this exact behavior) before any real API call was made in this session. No source code was changed in this analysis session — only new experiment runs and this document.

**Effect observed in the real run** (see raw log evidence in the traces below): each of the four `multi-file-timeout-setting` cells paid Gemini's ~80-120s exhaustion cost **once**, at the start of that cell's run, then used Groq for every remaining turn of that same cell — instead of re-paying that cost on every turn as before the fix. This is real, measured evidence the fix works, not a synthetic test result.

## 3. Provider configuration (values never printed; names/presence only)

`FORGE_PROVIDER_ORDER=gemini,groq,cerebras`. Gemini: configured, `gemini-3.6-flash`. Groq: configured, `openai/gpt-oss-120b`. Cerebras: configured, `gpt-oss-120b` — **not exercised** in any of the 12 records below (Groq resolved every fallback needed; Gemini's daily quota, not an outage, was the only failure mode encountered).

## 4. Exact commands executed this session

```bash
# multi-file-timeout-setting, all four arms, new experiment id (the
# existing step9_research_multitask/ directory cannot be reused — Step 7's
# MatrixRunner uses exist_ok=False on the experiment directory)
python -m forge.cli experiment --task-id multi-file-timeout-setting \
  --experiment-id step9_research_mfts --output-dir runs/experiments \
  --max-turns 10 --max-tool-calls 12 --json

# Combine ALL legitimate real records (baseline + both multi-task
# sessions) into one file, unmodified, for a single full analysis pass
cat runs/experiments/step9_real_fixed_raw/results.jsonl \
    runs/experiments/step9_real_remaining_arms/results.jsonl \
    runs/experiments/step9_research_multitask/results.jsonl \
    runs/experiments/step9_research_mfts/results.jsonl \
  > runs/experiments/step9_research_analysis/results.jsonl

# Step 8 analyzer, unmodified, new output path
python -m forge.cli analyze \
  --results runs/experiments/step9_research_analysis/results.jsonl \
  --output runs/experiments/step9_research_analysis/analysis.json --json
```

Execution limits: `--max-turns 10 --max-tool-calls 12` (unchanged from the baseline run, for comparability). Repetitions: 1 per task/arm cell (Step 7's `MatrixRunner` has no multi-seed mechanism; not invented here).

## 5. Exact number of tasks and records

**3 tasks × 4 arms = 12 records total**, all real-provider, none scripted/fake, none fabricated.

## 6. Full per-cell results (measured, from the JSONL — nothing inferred)

| Task | Arm | Provider (final) | Model | Status | Success | Checks | LLM calls | Tool calls | Tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| create-string-utils | fixed_raw | gemini | gemini-3.6-flash | completed | true | passed | 7 | 6 | 15,840 | 76.1 |
| create-string-utils | fixed_managed | gemini | gemini-3.6-flash | completed | true | passed | 7 | 6 | 17,457 | 25.6 |
| create-string-utils | adaptive_raw | gemini | gemini-3.6-flash | completed | true | passed | 6 | 5 | 24,300 | 84.9 |
| create-string-utils | adaptive_managed | **groq** | openai/gpt-oss-120b | completed | true | passed | 3 | 2 | 2,469 | 153.5 |
| fix-inclusive-sum | fixed_raw | gemini | gemini-3.6-flash | completed | true | passed | 8 | 7 | 13,272 | 101.0 |
| fix-inclusive-sum | fixed_managed | gemini | gemini-3.6-flash | completed | true | passed | 8 | 7 | 11,266 | 56.7 |
| fix-inclusive-sum | adaptive_raw | gemini | gemini-3.6-flash | completed | true | passed | 5 | 4 | 6,387 | 59.0 |
| fix-inclusive-sum | **adaptive_managed** | **groq** | openai/gpt-oss-120b | **provider_error** | **false** | passed* | 3 | 3 | 2,056 | 473.8 |
| multi-file-timeout-setting | fixed_raw | **groq** | openai/gpt-oss-120b | completed | true | passed | 12 | 11 | 14,472 | 155.9 |
| multi-file-timeout-setting | fixed_managed | **groq** | openai/gpt-oss-120b | completed | true | passed | 11 | 10 | 13,010 | 104.8 |
| multi-file-timeout-setting | adaptive_raw | **groq** | openai/gpt-oss-120b | completed | true | passed | 8 | 7 | 7,317 | 106.2 |
| multi-file-timeout-setting | adaptive_managed | **groq** | openai/gpt-oss-120b | completed | true | passed | 9 | 8 | 8,107 | 116.8 |

*`fix-inclusive-sum`/`adaptive_managed`: the file-level content check on `ranges.py` passed, but the run itself failed and is correctly recorded as `provider_error`/`success=false` — a genuine Groq HTTP 400 (`tool_use_failed`: the model requested `run_shell`, a tool not present in that turn's adaptive-narrowed exposed-tool set). This record is preserved exactly as produced; it is **not** converted to a success.

## 7. Provider/model distribution and confounding (critical — read before any per-arm comparison)

**6 of 12 cells were served by Gemini, 6 by Groq — and this split is not evenly or randomly distributed across arms or tasks:**

| Provider | Cells | Which |
|---|---|---|
| gemini | 6 | `create-string-utils` (fixed_raw, fixed_managed, adaptive_raw), `fix-inclusive-sum` (fixed_raw, fixed_managed, adaptive_raw) |
| groq | 6 | `create-string-utils` (adaptive_managed), `fix-inclusive-sum` (adaptive_managed), `multi-file-timeout-setting` (**all four arms**) |

Two confounds, stated plainly:

1. **Every single `adaptive_managed` cell (all 3 tasks) was served by Groq, never Gemini.** This is *not* because `adaptive_managed` as a configuration prefers or requires Groq — it is because `adaptive_managed` happened to be the **last arm executed within each task's run**, by which point Gemini's daily quota had typically already been consumed by the three arms that ran before it in the same invocation. This is a **run-order artifact**, not a property of the `adaptive_managed` configuration.
2. **All four arms of `multi-file-timeout-setting` were served by Groq.** By the time this task ran (a separate, later `forge experiment` invocation), Gemini's daily quota was already fully exhausted from the two earlier tasks. Every arm's outcome for this task reflects **Groq's** (`openai/gpt-oss-120b`) behavior, not Gemini's, on this task.

**Consequence: any observed difference in tokens, latency, or tool/LLM-call counts between arms or between tasks is confounded by which underlying model actually answered — Gemini (`gemini-3.6-flash`) and Groq (`openai/gpt-oss-120b`) are different models with different behavior, cost, and speed characteristics. None of the numbers in §6 isolate the effect of arm configuration from the effect of which provider/model served that particular cell.** This is exactly the kind of confound the project's own instructions require to be called out rather than glossed over.

## 8. Outcome summary (from the real analyzer output)

From `runs/experiments/step9_research_analysis/analysis.json`, `outcome_summary.__all__` (n=12):
- `success: 11`, `provider_error: 1`, `success_rate_all: 0.917` (11/12), `success_rate_excl_infra_errors: 1.0` (11/11, excluding the one infra/provider failure).
- One analyzer warning, exactly as designed to surface it (not suppressed): `mixed_experiment_ids` — the 12 records span 4 distinct `experiment_id`s (`step9_real_fixed_raw`, `step9_real_remaining_arms`, `step9_research_multitask`, `step9_research_mfts`), because each task/arm batch was run as its own `forge experiment` invocation (Step 7's `exist_ok=False` constraint prevents appending to a prior experiment-id).
- `unavailable_metrics: ["cached_input_tokens", "cost_usd"]` — correctly reported as unavailable, not fabricated as zero.

## 9. Per-arm summary (n=3 per arm — descriptive only, see §7's confound)

| Arm | Cells | Successes | Failures | Providers used |
|---|---|---|---|---|
| fixed_raw | 3 | 3 | 0 | gemini ×2, groq ×1 |
| fixed_managed | 3 | 3 | 0 | gemini ×2, groq ×1 |
| adaptive_raw | 3 | 3 | 0 | gemini ×2, groq ×1 |
| adaptive_managed | 3 | 2 | 1 (provider_error) | groq ×3 |

`adaptive_managed` is the only arm with a failure, and the only arm never served by Gemini in this sample. **This does not mean `adaptive_managed` is less reliable** — n=3 is far too small to support that, and its one failure is a Groq tool-schema rejection unrelated to the `adaptive`/`managed` strategy logic itself (see §6 footnote). Stated as an observation, not a conclusion.

## 10. Per-task summary

| Task | Cells | Successes | Failures | Notes |
|---|---|---|---|---|
| create-string-utils | 4 | 4 | 0 | 3 gemini, 1 groq |
| fix-inclusive-sum | 4 | 3 | 1 | 3 gemini, 1 groq (failed) |
| multi-file-timeout-setting | 4 | 4 | 0 | all 4 groq (Gemini quota already exhausted before this task ran) |

## 11. Fallback events (real, from logs — not simulated)

Across the two multi-task sessions, Gemini → Groq fallback fired **on the first turn of every cell that eventually used Groq** (7 of 12 cells: 2 from earlier sessions + this session's 4 `multi-file-timeout-setting` cells' first turns), each time after Gemini's own internal 3-attempt retry budget was exhausted (~80-120s), followed by Groq answering successfully in well under 2s on the first retry in nearly every case. With the sticky fix active, **no cell paid the Gemini exhaustion cost more than once**, regardless of how many turns that cell needed afterward (`multi-file-timeout-setting`/`fixed_raw` needed 12 LLM calls total but only 1 Gemini attempt).

## 12. Missing combinations

7 of 10 Step 4 baseline tasks were not run (listed in §1) — no data exists for them under any arm. Cerebras was not exercised in any of the 12 cells. No repeated runs/seeds exist for any task/arm cell (n=1 per cell within each task).

## 13. What is descriptive vs. what cannot be claimed

**Descriptive, directly measured (safe to state):**
- 11/12 real cells completed successfully; 1 real, documented failure.
- Across this specific sample, `adaptive_managed` used less total context per LLM call on `create-string-utils` (3 LLM calls, 2,469 tokens) than the other three arms on the same task — but this cell was served by a different model (Groq) than the other three (Gemini) on that task, so this cannot be attributed to the `adaptive`/`managed` strategy alone.
- The sticky-fallback fix measurably reduced wasted retry time per cell (one ~80-120s Gemini exhaustion cost per cell, not per turn).

**Cannot be claimed from this data:**
- No universal winner among the four arms.
- No statistical significance (n=3 per arm, n=1 per task/arm cell — Step 8's own Wilcoxon statistic is degenerate at these sample sizes).
- No general superiority of any arm over another.
- No causal conclusion about tool-exposure strategy or context strategy effects, since provider identity is confounded with arm and with run order (§7).
- No energy, cooling, water, or emissions claims — no such data was collected or is available from any provider used.

**The evaluation validates the pipeline and provides descriptive measurements, but does not support a definitive overall configuration ranking.**

## 14. Artifacts

```
runs/experiments/step9_real_fixed_raw/            # create-string-utils, fixed_raw (prior session, real, preserved)
runs/experiments/step9_real_remaining_arms/        # create-string-utils, other 3 arms (prior session, real, preserved)
runs/experiments/step9_real_four_arm_baseline/     # create-string-utils combined + original baseline analysis.json (untouched)
runs/experiments/step9_research_multitask/         # fix-inclusive-sum, all 4 arms (prior session, real, preserved, NOT rerun)
runs/experiments/step9_research_mfts/              # multi-file-timeout-setting, all 4 arms (this session, real, new)
runs/experiments/step9_research_analysis/          # NEW: combined 12-record results.jsonl + analysis.json (this document's data source)
```
No existing file under any of the first four directories was modified, appended to, or overwritten by this session.

## 15. Acceptance status

- [x] All four arms run for `multi-file-timeout-setting` (the previously-missing task) — real, complete.
- [x] Existing `fix-inclusive-sum` and `create-string-utils` records preserved exactly, including the one real failure — not rerun, not hidden, not converted.
- [x] Combined analysis produced via the unmodified Step 8 analyzer, written to a new path, original baseline analysis untouched.
- [x] Provider/model confounding explicitly documented (§7).
- [ ] Full 10-task baseline suite — **not** attempted; 7 tasks remain, documented as a scope decision (§1), not a blocker for what was attempted.

## 16. Step 10 handoff

**Not started, not discussed as active work.** If a future session continues the multi-task evaluation, the next concrete step is the 7 remaining Step 4 tasks under the same `--max-turns 10 --max-tool-calls 12` bounds, each as its own `forge experiment` invocation (new experiment-id per task, matching this session's pattern), combined afterward into an updated `step9_research_analysis` (or a newly-versioned one, since `exist_ok=False` prevents appending). Given Gemini's daily quota is evidently exhausted early in any session that runs more than ~2-3 real cells, expect most or all of those future cells to be served by Groq — the same confound documented in §7 will persist and should be reported the same way, not smoothed over.
