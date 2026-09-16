# Step 9 — Real-Provider Four-Arm Baseline

## 0. Documentation status

**No Step 9 specification was ever committed to this repository.** `README.md`,
every file in `docs/`, and `experiments/README.md` were searched for "Step 9"
before writing this document — there were zero matches. This document is
therefore written *after the fact*, reconstructed strictly from the artifact
trail left by prior attempts (`runs/experiments/step9_*`, all gitignored and
not previously documented) and the commands actually executed in this
session. It does not redefine or reinterpret those artifacts — it describes
what they already were.

## 1. Objective

Produce a **real** (live-provider, not scripted-fake-provider) run of the
existing Step 7 controlled 2×2 experiment — `fixed_raw`, `fixed_managed`,
`adaptive_raw`, `adaptive_managed` — for the existing Step 4 baseline task
`create-string-utils`, using the unmodified `forge experiment` CLI, then run
the unmodified Step 8 `forge analyze` CLI on the result.

This validates that the full real-provider pipeline (agent loop → Gemini/
fallback provider → tool execution → safety → observability → evaluation →
analysis) works end-to-end under real API constraints, including real
quota exhaustion. **It is a pipeline/baseline validation on one task, one
run per arm — not a robustness study, and no claim of statistical
significance or a "winning" configuration is made.**

## 2. Relationship to Step 8 and earlier attempts

Step 7/8 already proved the harness and analysis pipeline correctness using
a scripted fake provider (see their test suites). What was still unverified
was whether a full real-provider run would even complete, given Gemini's
free-tier quota. Three prior attempts, in earlier sessions, did not produce
usable data:

| Directory | Result |
|---|---|
| `runs/experiments/step9_phase1_pilot` | Interrupted before any record was written — empty directory skeleton only. |
| `runs/experiments/step9_recovery_single_cell` | 1 record, `fixed_raw`, `status=provider_error` — Gemini 429, per-minute rate limit exhausted after internal retry. |
| `runs/experiments/step9_recovery_single_cell_v2` | 1 record, `fixed_raw`, `status=provider_error`, `llm_calls=0` — Gemini's **daily** free-tier quota (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, limit 20/day) was already exhausted; even the retry-hardened Gemini adapter could not succeed. |

These three are preserved untouched under `runs/experiments/` and are
**not** counted as Step 9 data — they demonstrate the blocker that
motivated building the Gemini → Groq → Cerebras provider fallback system
(implemented in the immediately preceding session; see
`forge/providers/{groq,cerebras,fallback}.py`).

## 3. What this session did

### 3.1 Configuration (values never printed; presence/name confirmed only)

- `FORGE_PROVIDER_ORDER=gemini,groq,cerebras` (opt-in fallback, active).
- Gemini: configured, model `gemini-3.6-flash`.
- Groq: configured, model `openai/gpt-oss-120b`.
- Cerebras: configured, model `gpt-oss-120b` (not exercised this run — see §6).

### 3.2 Commands executed (exact, in order)

```bash
# Cell 1 — fixed_raw (bounded, isolated)
python -m forge.cli experiment --task-id create-string-utils --arm fixed_raw \
  --experiment-id step9_real_fixed_raw --output-dir runs/experiments \
  --max-turns 10 --max-tool-calls 12 --json

# Cells 2-4 — remaining three arms, one bounded call, new experiment id
# (fixed_raw is real+valid already; Step 7's MatrixRunner refuses to reuse
# an existing experiment_id — exist_ok=False — so the remaining arms were
# written to a second experiment directory, not appended to the first)
python -m forge.cli experiment --task-id create-string-utils \
  --arm fixed_managed --arm adaptive_raw --arm adaptive_managed \
  --experiment-id step9_real_remaining_arms --output-dir runs/experiments \
  --max-turns 10 --max-tool-calls 12 --json

# Combine the two real results files (raw concatenation, records unmodified)
# into one file so Step 8's analyzer — which takes a single --results file —
# can compare all four arms together.
cat runs/experiments/step9_real_fixed_raw/results.jsonl \
    runs/experiments/step9_real_remaining_arms/results.jsonl \
  > runs/experiments/step9_real_four_arm_baseline/results.jsonl

# Step 8 analysis, unmodified CLI, new output path (Step 8's own artifacts
# are never touched)
python -m forge.cli analyze \
  --results runs/experiments/step9_real_four_arm_baseline/results.jsonl \
  --output runs/experiments/step9_real_four_arm_baseline/analysis.json --json
```

No experiment code, agent loop, provider adapter, or analysis code was
modified to produce this data — only the existing Step 7/8 CLIs were
invoked.

### 3.3 Measured results (real, from the JSONL records — not fabricated)

| Arm | Provider used | Model | Status | Success | Checks | LLM calls | Tool calls | Total tokens | Avg LLM latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `fixed_raw` | gemini | gemini-3.6-flash | completed | true | passed | 7 | 6 | 15,840 | 75,780.6 |
| `fixed_managed` | gemini | gemini-3.6-flash | completed | true | passed | 7 | 6 | 17,457 | 25,264.2 |
| `adaptive_raw` | gemini | gemini-3.6-flash | completed | true | passed | 6 | 5 | 24,300 | 84,713.3 |
| `adaptive_managed` | **groq** (fallback) | openai/gpt-oss-120b | completed | true | passed | 3 | 2 | 2,469 | 153,474.3 |

All four are `checks_passed: true` — the task's own content checks
(`def slugify(`, `def shout(` present, behavior asserted via the agent's own
`run_shell` calls in three of the four runs) passed for real, not assumed.

**`adaptive_managed` is the first real, verified evidence that the
Gemini → Groq fallback actually fires and completes correctly in a genuine
experiment run**, not just a synthetic test. The actual log sequence:

```
WARNING  forge.providers.gemini    Gemini rate limited (429); retrying in 58.0s (attempt 1/3)
WARNING  forge.providers.gemini    Gemini rate limited (429); retrying in 59.0s (attempt 2/3)
WARNING  forge.providers.gemini    Gemini rate limited (429); retrying in 0.0s (attempt 3/3)
WARNING  forge.providers.fallback  Provider 'gemini' failed with a recoverable error after 118523ms;
                                    falling back to the next provider in the order.
                                    reason=Gemini API error: 429 RESOURCE_EXHAUSTED [...]
                                    quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20
INFO     forge.providers.fallback  Fallback succeeded: provider 'groq' served this request in 584ms
                                    after 1 earlier provider(s) failed or were unavailable.
```
Gemini's daily quota was exhausted partway through this session's calls;
`FallbackProvider` correctly classified the error as recoverable, exhausted
Gemini's own bounded internal retry (3 attempts) first, then moved to Groq,
which answered correctly in under a second. **Note the real cost of this
path**: the `adaptive_managed` row's 153s average latency is dominated by
Gemini's own retry backoff (~118s) before the fallback even started — this
is real, measured overhead of falling back only after a provider's own
retry budget is exhausted, not a defect in the data.

A discrepancy versus this session's own starting brief: the brief claimed
an *earlier* smoke test (`logs/run_20260916T192713_411318.jsonl`) already
showed a Gemini-429-to-Groq fallback. Inspecting that trace directly shows
`provider: gemini`, `status: completed`, `error: null`, a single LLM call,
and ~48s latency (consistent with a normal direct Gemini response, not a
429+fallback sequence) — no fallback is evident in that particular trace.
This is noted for the record; it does not affect the real fallback evidence
in §3.3 above, which comes from this session's own experiment run, not that
earlier trace.

### 3.4 Analysis output

`runs/experiments/step9_real_four_arm_baseline/analysis.json` (225,755
bytes, valid JSON, `schema_version: 1`):
- `record_count: 4`, `arms_analyzed`: all four canonical arms, `task_count: 1`.
- `outcome_summary.__all__`: `total: 4, success: 4, success_rate_all: 1.0,
  success_rate_excl_infra_errors: 1.0`.
- One warning, exactly as Step 8's analyzer is designed to surface it (not
  suppressed, not silently merged): `mixed_experiment_ids` — the four
  records come from two `experiment_id`s (`step9_real_fixed_raw` and
  `step9_real_remaining_arms`) because of the `exist_ok=False` constraint
  in §3.2. This is expected and documented, not a data-quality defect.
- `unavailable_metrics: ["cached_input_tokens", "cost_usd"]` — neither
  Gemini nor Groq's usage objects exposed these; correctly reported as
  unavailable rather than fabricated as `0`.

## 4. Reproducibility

- Exact commands: §3.2, runnable as-is from repository root with a
  configured `.env` (see `.env.example` for the variable names — no values
  committed there or here).
- `temperature=0.0` for every arm (Step 7's existing default) — Gemini's
  runs are close to deterministic; Groq's determinism at `temperature=0.0`
  is provider-dependent and not independently verified here.
- Real API responses are inherently not bit-for-bit reproducible run to
  run (model updates, provider-side non-determinism, and — as demonstrated
  here — which provider in the fallback chain actually serves a given call
  depends on live quota state at call time).

## 5. Acceptance status

No formal acceptance criteria were ever documented for Step 9 (see §0). Using
the only criterion inferable from the artifact trail — four valid, real,
`status=completed` records, one per canonical arm, same task, followed by a
successful `forge analyze` run — **that criterion is satisfied**:

- [x] `fixed_raw` — real, complete, checks passed
- [x] `fixed_managed` — real, complete, checks passed
- [x] `adaptive_raw` — real, complete, checks passed
- [x] `adaptive_managed` — real, complete, checks passed (via Groq fallback)
- [x] `forge analyze` run on the combined real data, valid output written

## 6. Limitations

- **One task, one run per arm.** This is a pipeline/baseline validation,
  not a robustness study. No variance, no repeated runs, no statistical
  test result should be treated as meaningful here (Step 8's Wilcoxon
  statistic is computed on n=1 per arm and is not reported as
  significant — see the analysis JSON's own `statistical_tests` values,
  which are degenerate at n=1).
- **Cerebras was never exercised** — Gemini's failures in this session were
  always resolved by Groq (the first fallback), so the Gemini → Groq →
  Cerebras chain's third link has real *adapter-level* test coverage
  (`tests/test_providers_cerebras.py`, all against fakes) but no real-API
  evidence from an actual FORGE run.
- **Not a multi-task study.** `fix-inclusive-sum`, `multi-file-timeout-setting`,
  and the rest of the baseline suite were not run — out of scope for this
  session per its own instructions.
- Gemini's daily free-tier quota (20 requests/day) means a second full
  real-provider run today would very likely exhaust Gemini quickly and lean
  heavily on Groq/Cerebras; this is expected, documented behavior of the
  fallback system, not a bug.
- No overall "winner" is declared between arms or providers — none of this
  session's data supports that claim, and Step 8's analyzer is deliberately
  designed not to produce one (see `docs/step-08-statistical-analysis.md` §9).

## 7. Artifacts

```
runs/experiments/step9_real_fixed_raw/            # fixed_raw only, real, valid
runs/experiments/step9_real_remaining_arms/        # fixed_managed, adaptive_raw, adaptive_managed
runs/experiments/step9_real_four_arm_baseline/     # combined results.jsonl + analysis.json
runs/experiments/step9_phase1_pilot/               # preserved: empty, not usable (prior session)
runs/experiments/step9_recovery_single_cell/       # preserved: failed, not usable (prior session)
runs/experiments/step9_recovery_single_cell_v2/    # preserved: failed, not usable (prior session)
```
None of the three prior (failed) directories were modified, overwritten, or
deleted. None of Step 7/8's own artifacts or documentation were modified.
