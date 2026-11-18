# Step 9 — 2×2 Pilot Results (Managed Context / Adaptive Tool Exposure)

## 0. Documentation status

This is a report on **one real, four-arm, one-repetition-per-arm pilot run**
of the Step 7 formal 2×2 experiment (`forge experiment`), executed on
2026-09-17/18 against the real Gemini API. It follows this repository's
existing `docs/step-09-*.md` naming convention (alongside
`step-09-real-provider-baseline.md` and
`step-09-research-multitask-analysis.md`) rather than inventing a new one.
It is a pilot, not a completed research result — see §13.

## 1. Research question

Can Adaptive Tool Exposure and/or Managed Context reduce token usage,
tool/LLM-call counts, and latency on a real coding task, without reducing
task correctness, compared to the Fixed+Raw baseline?

## 2. Hypothesis

- **H1 (Managed Context):** Fixed+Managed will use fewer input/total tokens
  and less context (chars) than Fixed+Raw, with no loss of correctness.
- **H2 (Adaptive Tool Exposure):** Adaptive+Raw will use fewer tool/LLM
  calls and tokens than Fixed+Raw, with no loss of correctness.
- **H3 (Combined):** Adaptive+Managed will show the largest reduction of
  all four arms, with no loss of correctness.

## 3. Experimental design

A single 2×2 factorial design, one task, one repetition per cell (four
cells total), run in one `forge experiment` invocation so that provider,
model, temperature, and turn/call ceilings were identical across all four
cells by construction.

| | Raw context | Managed context |
|---|---|---|
| **Fixed tools** | Fixed+Raw (baseline) | Fixed+Managed |
| **Adaptive tools** | Adaptive+Raw | Adaptive+Managed |

## 4. The four arms

Confirmed via `forge/evaluation/matrix.py`'s `ARMS` (the exact `--arm`
choices `forge experiment --help` accepts): `fixed_raw`, `fixed_managed`,
`adaptive_raw`, `adaptive_managed`. No new arm names or configuration keys
were invented.

## 5. Controlled variables

Held identical across all four cells by passing them once to the single
`forge experiment` invocation:

| Variable | Value |
|---|---|
| Task | `multi-file-timeout-setting` |
| Provider | `gemini` |
| Model | `gemini-3.6-flash` |
| Temperature | `0.0` |
| Max LLM turns | `10` |
| Max tool calls | `12` |
| Task checker | the task's own `checks` (content-substring checks on `settings_mod.py` and `client.py`) |
| Starting files | `experiments/tasks/fixtures/multi-file-timeout-setting/` (`client.py`, `settings_mod.py`), copied fresh into each arm's own workspace by `EvaluationRunner` |
| Workspace | a fresh, isolated directory per arm under `runs/research_2x2/2x2_pilot_multifile_timeout/workspaces/multi-file-timeout-setting/<arm_id>/`, never reused between arms |
| Dependency environment | the repository's existing `.venv` (unchanged for this pilot) |

## 6. Selected task

**`multi-file-timeout-setting`** (category `multi_file`, difficulty
`medium`, 2 checks, 2 fixture files), chosen after inspecting all 10 tasks
in `experiments/tasks/`:

- Requires real file editing across two files (`settings_mod.py`,
  `client.py`) — not just creation.
- Has a reliable, automatic content-substring checker (2 checks).
- Fully offline, no external websites, no dangerous commands.
- Small fixture files (4 and 6 lines) — fast to run.
- Verified (via `AdaptiveToolExposure().classify()`) to classify into the
  `"edit"` category, exposing only `read_file`, `edit_file`,
  `list_directory` (3 of 5 tools) under Adaptive exposure — a real,
  checkable narrowing, unlike some other tasks in the suite (e.g. the
  testing-category task exposes all 5 tools under Adaptive too, which
  would show no Adaptive effect at all).
- Multi-file editing plausibly produces enough turns for
  `ManagedContextStrategy`'s pruning to have a chance to activate (its
  default `keep_recent_turns=3` protects only the most recent 3 turns).

## 7. Exact commands used

Inspection (safe, read-only, run before any experiment execution):

```powershell
forge --help
forge experiment --help
forge tasks
python -c "from forge.tools import AdaptiveToolExposure; ..."   # verified task -> 'edit' category -> 3 tools
```

The experiment itself (one invocation, all four arms, one repetition each):

```powershell
forge experiment --suite-dir experiments/tasks --task-id multi-file-timeout-setting --output-dir runs/research_2x2 --experiment-id 2x2_pilot_multifile_timeout --label "2x2 pilot - Managed Context / Adaptive Tool Exposure" --max-turns 10 --max-tool-calls 12 --temperature 0.0 --json
```

## 8. Run IDs

| Arm | Run ID | Status |
|---|---|---|
| fixed_raw | `run_20260917T195224_673714` | failed (infrastructure) |
| fixed_managed | `run_20260917T195300_412780` | completed |
| adaptive_raw | `run_20260917T195508_545830` | provider_error |
| adaptive_managed | `run_20260917T195606_677146` | failed (infrastructure) |

Experiment ID: `2x2_pilot_multifile_timeout`. Started
`2026-09-17T19:52:24Z`, finished `2026-09-17T19:56:09Z` (≈3 min 45 s wall
time for all four cells).

## 9. Raw measured metrics

| Metric | Fixed+Raw | Fixed+Managed | Adaptive+Raw | Adaptive+Managed |
| --- | ---: | ---: | ---: | ---: |
| Status | failed | completed | provider_error | failed |
| Success | False | **True** | False | False |
| checks_passed | **True** | **True** | False | False |
| Input tokens | 8,647 | 16,607 | 2,360 | 655 |
| Output tokens | 241 | 355 | 48 | 10 |
| Total tokens | 10,098 | 17,754 | 2,560 | 797 |
| Cached input tokens | N/A | N/A | N/A | N/A |
| Reasoning tokens | 1,210 | 792 | 152 | 132 |
| LLM calls | 6 | 10 | 3 | 1 |
| Tool calls | 6 | 9 | 3 | 1 |
| Context messages (final) | 14 | 21 | 8 | 4 |
| Context chars (final) | 1,841 | 2,671 | 1,686 | 1,488 |
| Context items dropped | 0 | 0 | 0 | 0 |
| Context items compressed | 0 | 0 | 0 | 0 |
| LLM latency (ms) | 20,024.6 | 127,280.6 | 53,704.6 | 2,752.6 |
| Duration (s) | 35.73 | 128.13 | 58.13 | 3.16 |
| Cost estimate | N/A (`cost_available: false`) | N/A | N/A | N/A |
| Exposed tools | 5 (all) | 5 (all) | 3 (read_file, edit_file, list_directory) | 3 (read_file, edit_file, list_directory) |
| Retries observed | 3 (rate-limit, then fallback) | 2 (rate-limit, resolved) | 1 (rate-limit, then non-recoverable 503) | 0 (skipped, quota exhausted) |
| Fallback attempted / succeeded | Yes / **No** | No | No (503 classified non-recoverable) | Yes / **No** |
| Error | `ImportError: groq package required` | None | `Gemini API error: 503 UNAVAILABLE` | `ImportError: groq package required` |

Source: `runs/research_2x2/2x2_pilot_multifile_timeout/results.jsonl` and
its four `traces/*.jsonl` files (unmodified `forge experiment` output).
Full per-run detail, including exact tool-call sequences, is in
`runs/research_2x2/results.json`.

## 10. Pairwise comparisons

Full arithmetic in `runs/research_2x2/comparison.md`. Summary:

**A. Fixed+Raw vs Fixed+Managed (Managed Context effect):** Fixed+Managed
used *more* tokens (+92.1% input, +75.8% total), more calls (+4), and more
latency, the opposite of the hypothesis — **but not attributable to Managed
Context**: trace telemetry shows `context_items_dropped: 0` and
`context_items_compressed: 0` on every turn of the Fixed+Managed run, i.e.
the strategy never actually pruned or compressed anything in this run. The
gap is explained by the model taking a longer path (one failed,
unnecessary `run_shell(pytest)` call plus two redundant re-reads) that
happened not to occur in the Fixed+Raw run.

**B. Fixed+Raw vs Adaptive+Raw (Adaptive Tool Exposure effect):** the
full-run comparison is invalid because Adaptive+Raw was cut off by a 503
outage after 3 tool calls, before any edit. A **matched sub-comparison**
over the identical first three actions both runs performed
(`list_directory`, then reading each file) is valid: Adaptive's smaller
tool-schema list produced **24.5% fewer input tokens** for those same three
actions (2,360 vs 3,127 tokens) — the one number in this pilot that
isolates the Adaptive mechanism cleanly.

**C. Fixed+Raw vs Adaptive+Managed (Combined effect):** not interpretable.
Adaptive+Managed completed exactly one tool call before Gemini's daily
quota was exhausted and its Groq fallback failed. All apparent
"reductions" here (92–93%) reflect an almost-immediate failure, not
efficiency.

**D. Interaction effect:** computed per the requested formula for input
tokens (−9,665), total tokens (−9,419), LLM calls (−6), tool calls (−5),
and latency (−158,208 ms) — all large and negative, but **not
interpretable as a real interaction**, since three of the four cells did
not reach a comparable state of task progress.

## 11. Interaction calculations

See §10.D and `runs/research_2x2/comparison.md` §D for the full worked
arithmetic and sign explanation.

## 12. Correctness results

| Arm | checks_passed | Notes |
|---|---|---|
| Fixed+Raw | **True** | Both files were edited correctly; the run only failed to produce a final confirmation message because its 7th LLM call hit an exhausted per-minute rate limit and a failed Groq fallback. |
| Fixed+Managed | **True** | Completed cleanly. |
| Adaptive+Raw | False | Never reached the edit step (stopped by a 503 outage after 3 calls — the same point Fixed+Raw had also reached after its first 3 calls). |
| Adaptive+Managed | False | Stopped after 1 call (daily quota exhaustion). |

Both arms that reached the edit step produced correct results; both arms
that didn't were stopped by infrastructure failures unrelated to the
strategy being tested. **No correctness difference between Fixed and
Adaptive, or between Raw and Managed, can be attributed to the strategies
themselves from this pilot.**

## 13. Limitations

1. **One repetition per arm** — no statistical inference is possible or
   claimed.
2. **One task, one category (`multi_file`)** — results do not generalize
   to other task types.
3. **Three of four runs were cut short by real infrastructure failures**:
   Gemini's free-tier per-minute rate limit (5 requests/minute) and daily
   quota (20 requests/day) were both hit during this experiment sequence,
   and the configured `FallbackProvider` order (`gemini → groq → cerebras`,
   per `.env`) could not actually fall back because the `groq` package is
   not installed in this environment (`pip install groq` — the identical
   gap already flagged by `pytest -q` in this repository, in
   `tests/test_providers_groq.py`).
4. **Managed Context was not actually exercised** in the one run where it
   could have been (Fixed+Managed) — see §10.A. This pilot cannot support
   any claim about Managed Context's effect, positive or negative.
5. **Latency numbers are confounded** by external per-minute rate-limit
   backoff baked into individual call latencies (e.g. one 62.3 s and one
   46.9 s call in the Fixed+Managed run) — these do not reflect the
   context/tool strategy's own processing cost.
6. **No cost estimate is available** (`cost_available: false` on every
   result) — FORGE's evaluation layer has no pricing table wired up yet.
7. This is a **pilot**, not a validated experiment.

## 14. Honest conclusion

> This four-arm experiment provides an initial controlled indication of the
> effects of Managed Context and Adaptive Tool Exposure. It is not
> sufficient to establish statistical significance or generalize across
> tasks. More repetitions and additional task categories are required.

Concretely, from this pilot:

- **Adaptive Tool Exposure** demonstrably reduced the tool schemas sent to
  the model (3 vs 5 tools) and, for the one valid matched comparison
  available, reduced input tokens by ~24.5% for identical actions. This is
  the one genuinely attributable finding in this pilot.
- **Managed Context** cannot be evaluated from this pilot — its pruning
  mechanism was never triggered in the one run where it ran to completion.
- **No claim of a combined or interaction effect is supported** — three of
  the four cells were terminated early by real provider/dependency
  failures unrelated to the research variables, not by the agent reaching
  a natural stopping point.
- **Task correctness** was preserved in both arms that reached the edit
  step (Fixed+Raw's actual edits, Fixed+Managed), and no correctness
  regression can be attributed to either strategy from the two arms that
  didn't.

## 15. Recommended next experiment

1. Fix the environment gap first: `pip install groq cerebras_cloud_sdk` so
   `FallbackProvider` can actually rescue a run when Gemini is rate-limited,
   instead of failing with `ImportError`.
2. Re-run this same 2×2 (same task, same settings) once Gemini's daily
   quota has reset or with a higher-quota key, to get four cleanly
   completed runs before drawing any conclusion about Managed Context or
   the combined arm.
3. To actually observe Managed Context's effect, choose or construct a
   task/run that exceeds `ManagedContextStrategy`'s default
   `keep_recent_turns=3` window **with genuinely repeated same-tool,
   same-path calls** (e.g. a task requiring the agent to re-read the same
   file several times), since compression only fires on redundant
   successful results for the same tool.
4. Only after a clean single-repetition 2×2 succeeds should repetitions
   (e.g. 3–5 per arm) and additional task categories be added — each of
   those is a materially larger and more expensive experiment than this
   pilot and should be scoped and approved separately.
