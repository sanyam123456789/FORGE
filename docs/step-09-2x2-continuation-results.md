# Step 9 — 2×2 Continuation Results (Groq-primary / Cerebras-fallback, Gemini disabled)

This report documents a continuation of `docs/step-09-2x2-pilot-results.md`
(unchanged, not rewritten). It attempted to complete the three arms that
failed in that pilot (`fixed_raw`, `adaptive_raw`, `adaptive_managed`) by
switching the provider order away from Gemini entirely.

## 1. Why continuation was required

In the original pilot, only `fixed_managed` reached a clean `completed`
status. `fixed_raw`, `adaptive_raw`, and `adaptive_managed` all failed due
to Gemini free-tier rate/quota exhaustion, compounded in two cases by a
missing `groq` package that made `FallbackProvider`'s attempt to fall back
fail with `ImportError` instead of actually rescuing the run.

## 2. Dependency diagnosis

Inspected before installing anything:

- `pyproject.toml` declares `groq>=1.0` and `cerebras_cloud_sdk>=1.0` as
  runtime dependencies (both already listed — this was an installation
  gap, not a missing declaration).
- `forge/providers/groq.py::_import_sdk()` does `import groq`.
- `forge/providers/cerebras.py::_import_sdk()` does
  `import cerebras.cloud.sdk as cerebras_sdk` — the package name is
  `cerebras_cloud_sdk`, but the import path is `cerebras.cloud.sdk`.
- No `requirements.txt` / `requirements-dev.txt` exist in this repository;
  `pyproject.toml` is the sole dependency source.
- `where python` / `where forge` showed a discrepancy: the global `forge`
  command on `PATH` resolves to a different Python install
  (`...Python313\Scripts\forge`), not this project's `.venv`. The `.venv`
  itself has no `forge` console-script installed at all (the package was
  never `pip install -e`'d into it) — but `.venv\Scripts\python -m
  forge.cli` correctly imports the local `forge/` source tree. **All
  commands in this continuation (and the original pilot) were run via the
  `.venv`'s own `python -m forge.cli`, never the bare global `forge`
  command**, to guarantee the installed packages below were actually used.

## 3. Installation results

```text
python -m pip show groq                  -> not found (before)
python -m pip show cerebras_cloud_sdk    -> not found (before)

python -m pip install groq
  -> Successfully installed groq-1.7.0

python -m pip install cerebras_cloud_sdk
  -> Successfully installed cerebras_cloud_sdk-1.91.0 (+ h2, hpack, hyperframe)

python -c "import groq; print('groq import OK')"                    -> OK (groq 1.7.0)
python -c "import cerebras.cloud.sdk as cerebras_sdk"                -> OK

python -m pip check                                                  -> No broken requirements found.
```

Both packages installed cleanly with no dependency conflicts. No unrelated
packages were installed or upgraded.

## 4. Fallback verification results

```text
pytest -q tests/test_fallback.py tests/test_providers_groq.py tests/test_providers_cerebras.py
-> 56 passed in 0.46s   (previously 20 of these failed with ImportError)

pytest -q   (full suite)
-> 527 passed in 292.88s   (previously 507 passed, 20 failed in 29.59s)
```

All 527 tests pass with no code changes — confirming the previous 20
failures were purely a missing-dependency environment gap, not a defect in
`forge/providers/groq.py`, `forge/providers/cerebras.py`, or
`forge/providers/fallback.py`. (The full run is ~5 minutes instead of ~30
seconds now, because those 20 tests previously failed instantly on
`ImportError` before ever reaching their mocked retry-backoff sleeps;
those sleeps now execute for real.)

`FallbackProvider` construction and `preflight()` were also verified
directly (no API call) with the continuation's exact provider order:

```text
provider_order: ('groq', 'cerebras')
Resolved fallback candidate order: ['groq', 'cerebras']
preflight OK for all candidates (keys/models present)
```

## 5. Provider/model configuration

```text
Primary provider:   groq
Primary model:      openai/gpt-oss-120b   (FORGE_GROQ_MODEL)
Fallback provider:  cerebras
Fallback model:     gpt-oss-120b          (FORGE_CEREBRAS_MODEL)
Provider order:     groq,cerebras          (Gemini excluded)
Reason for change:  mandatory for this continuation, to avoid Gemini's
                     exhausted free-tier rate/quota limits from the pilot.
```

Set via `FORGE_PROVIDER_ORDER=groq,cerebras` as a **process environment
variable on the experiment command itself**, not written into `.env`
(`forge/config.py` loads `.env` with `override=False`, so an
already-set process env var takes precedence — this changes the resolved
order for this one invocation without touching the repository's `.env`
file or its committed `gemini,groq,cerebras` default).

API-key presence check (values never printed):

```text
GROQ_API_KEY (FORGE_GROQ_API_KEY): present
CEREBRAS_API_KEY (FORGE_CEREBRAS_API_KEY): present
GEMINI_API_KEY: intentionally unused (provider_order excludes gemini)
```

## 6. The three continuation runs

Command:

```powershell
$env:FORGE_PROVIDER_ORDER = "groq,cerebras"
forge experiment --suite-dir experiments/tasks --task-id multi-file-timeout-setting --arm fixed_raw --arm adaptive_raw --arm adaptive_managed --output-dir runs/research_2x2/continuation --experiment-id 2x2_continuation_multifile_timeout --label "2x2 continuation - Groq primary, Cerebras fallback, Gemini disabled" --max-turns 10 --max-tool-calls 12 --temperature 0.0 --json
```

Exactly 3 task-arm executions ran (`fixed_raw`, `adaptive_raw`,
`adaptive_managed`); `fixed_managed` was not rerun.

| Arm | Run ID | Requested provider | Actual provider | Fallback occurred | Status |
|---|---|---|---|---|---|
| fixed_raw | `run_20260917T201407_005370` | groq | groq (then attempted cerebras) | Yes, failed | provider_error |
| adaptive_raw | `run_20260917T201446_336213` | groq | groq (then attempted cerebras) | Yes, failed | provider_error |
| adaptive_managed | `run_20260917T201550_118188` | groq | **groq** | No (not needed) | **completed** |

**Only `adaptive_managed` completed.** `fixed_raw` and `adaptive_raw` both
hit Groq's per-minute token-rate limit (`8000 TPM` on this account's
`on_demand` tier) repeatedly, exhausted their retry budget, and then
Cerebras rejected the fallback with `402 Payment Required` — the
configured Cerebras account has no active billing, which is an
account-administration issue, not a code defect.

## 7. Merged four-arm table

| Metric | Fixed+Raw | Fixed+Managed | Adaptive+Raw | Adaptive+Managed |
| --- | ---: | ---: | ---: | ---: |
| Provider (best/only run) | groq (failed) | **gemini** | groq (failed) | **groq** |
| Model | openai/gpt-oss-120b | gemini-3.6-flash | openai/gpt-oss-120b | openai/gpt-oss-120b |
| Status | provider_error | completed | provider_error | **completed** |
| Success | False | **True** | False | **True** |
| Checks passed | False | **True** | False | **True** |
| Input tokens | 1,562 | 16,607 | 5,290 | 8,519 |
| Output tokens | 70 | 355 | 620 | 1,186 |
| Total tokens | 1,632 | 17,754 | 5,910 | 9,705 |
| Cached input tokens | N/A | N/A | N/A | N/A |
| Reasoning tokens | N/A (Groq doesn't report) | 792 | N/A | N/A |
| LLM calls | 2 | 10 | 7 | 10 |
| Tool calls | 2 | 9 | 7 | 9 |
| Context messages (final) | 6 | 21 | 16 | 21 |
| Context characters (final) | 1,542 | 2,671 | 1,946 | 2,337 |
| Context items dropped | 0 | 0 | 0 | 0 |
| Context items compressed | 0 | 0 | 0 | **1** |
| Exposed tool count | 5 | 5 | 3 | 3 |
| LLM latency (ms) | 15,593.7 | 127,280.6 | 42,233.3 | 43,361.7 |
| Total duration (s) | 39.30 | 128.13 | 63.72 | 43.52 |
| Retry count | 5 | 2 | 11 | 8 |
| Fallback used | Yes (failed) | No | Yes (failed) | No |
| Cost estimate | N/A | N/A | N/A | N/A |
| Error | Groq 429 TPM, then Cerebras 402 | None | Groq 429 TPM, then Cerebras 402 | None |

**This table is provider-contaminated by construction**: the Fixed+Managed
column is Gemini/`gemini-3.6-flash`; every other column is (an attempt on)
Groq/`openai/gpt-oss-120b`. It must not be read as a same-provider,
fully controlled comparison. The two `provider_error` columns
(Fixed+Raw, Adaptive+Raw) represent the second consecutive failed attempt
at those arms, on a different provider than the first attempt — they are
shown for completeness and diagnosis, not as valid task outcomes.

Full per-attempt detail (both the pilot attempt and this continuation
attempt for each arm) is in `runs/research_2x2/final_merged_results.json`;
a flat version is in `runs/research_2x2/final_merged_results.csv`.

## 8. All raw metrics

See §7 table and `runs/research_2x2/final_merged_results.{json,csv}` for
the complete field-by-field data, including both attempts for `fixed_raw`
and `adaptive_raw`.

## 9. Pairwise comparisons

Full detail in `runs/research_2x2/final_comparison.md`. Summary:

- **A (Managed Context, Fixed+Raw vs Fixed+Managed): not interpretable** —
  Fixed+Raw has no valid completed run on either provider tried.
- **B (Adaptive Tool Exposure, Fixed+Raw vs Adaptive+Raw): not
  interpretable as a full-run comparison**, but a valid **same-provider
  matched sub-comparison** exists: both Groq runs performed identical
  first two actions, and Adaptive's smaller tool-schema list used **20.6%
  fewer input tokens** for those two actions (1,240 vs 1,562). This
  replicates the pilot's Gemini-based matched finding (24.5% over 3
  identical actions) on a completely different provider and model — the
  most reproducible result across both sessions.
- **C (Combined, Fixed+Raw vs Adaptive+Managed): not interpretable** — no
  valid Fixed+Raw baseline exists.
- **D (Interaction effect): not interpretable** — two of the four cells
  lack a valid value; the formula is not computed rather than computed on
  incomplete data.

## 10. Interaction calculations

Not computed. See §9.D and `final_comparison.md` §D for the explicit
reasoning.

## 11. Correctness results

| Arm | Correctness outcome |
|---|---|
| fixed_raw | Never reached an edit in either attempt (pilot: infra cutoff after edits were already correct on a *different* code path from a *later* turn in that run; continuation: infra cutoff before any edit at all) |
| fixed_managed | Correct (Gemini) |
| adaptive_raw | Pilot: infra cutoff before any edit. Continuation: **a genuine, non-infra correctness miss** — the model added the required import to `client.py` but left the old hardcoded `timeout = 10` line in place, which would have failed the checker regardless of the subsequent provider error. |
| adaptive_managed | Correct (Groq) |

## 12. Failed / incomplete runs

- `fixed_raw` — failed twice, on two different provider configurations.
  Both failures are infrastructure/account-level (Gemini rate/quota in the
  pilot; Groq rate limit + unfunded Cerebras account in the continuation).
- `adaptive_raw` — failed twice. The pilot failure was purely
  infrastructure (Gemini 503). The continuation failure combines
  infrastructure (Groq rate limit + unfunded Cerebras) **and** a genuine
  model-correctness miss that was already present before the
  infrastructure error occurred.

No result was fabricated, retried silently beyond the three approved
continuation runs, or edited to make a check pass. Both failed workspaces
and their traces are preserved unmodified.

## 13. Limitations

1. **`fixed_raw` and `adaptive_raw` still have no valid completed run**
   after two attempts each, on two different provider configurations. A
   third attempt was not made without approval, per the run limit for this
   continuation.
2. **Provider/model contamination**: the two arms that did complete used
   different providers and models from each other (Gemini/`gemini-3.6-flash`
   vs Groq/`openai/gpt-oss-120b`), so they cannot be compared as a
   controlled pair even though both are individually valid.
3. **Groq's free/on-demand tier has a very low per-minute token budget**
   (8,000 TPM) for `openai/gpt-oss-120b`, which this task's tool-calling
   conversations repeatedly exceeded — an account-tier constraint, not a
   FORGE defect.
4. **The configured Cerebras account has no active billing** (`402
   Payment Required` on every attempt) — an account-administration issue
   outside FORGE's code; the adapter and fallback wiring are confirmed
   correct (imports cleanly, `preflight()` passes, error correctly
   propagates).
5. **One repetition per arm per attempt** — no statistical inference is
   possible or claimed.
6. **Reasoning tokens and cost estimates are unavailable for Groq/Cerebras**
   runs (`N/A`), unlike Gemini which reports reasoning tokens.
7. This remains an **exploratory pilot**, not a validated experiment.

## 14. Final honest conclusion

> After dependency repair and continuation runs, the four-arm pilot was
> evaluated using the same task and recorded execution metrics. A complete
> controlled comparison was not achieved because of provider or
> environment limitations. Since each arm has only one repetition (per
> attempt), the findings are exploratory and cannot establish statistical
> significance or broad generalization. The one result reasonably well
> supported by this pilot — because it was observed independently on two
> different providers — is that Adaptive Tool Exposure's smaller
> tool-schema list reduces input tokens by roughly 20–25% for identical
> actions.

## 15. Recommended next step

1. **Fund the Cerebras account** (or remove it from the provider order) so
   its fallback slot is actually usable, rather than always failing with
   402.
2. **Space out or pace Groq calls** (e.g. a short delay between arms, or a
   smaller `max_output_tokens`) to stay under its 8,000 TPM budget, or
   request a higher-tier Groq quota, before re-attempting `fixed_raw` and
   `adaptive_raw`.
3. Once either fix is in place, **re-run only `fixed_raw` and
   `adaptive_raw`** (2 runs, not 4) to complete the set — get explicit
   approval first, since this is additional provider spend beyond what was
   authorized for this continuation.
4. Separately, the `adaptive_raw` continuation's incorrect `client.py` edit
   is worth a small, isolated, non-expensive follow-up: rerun just that one
   task-arm cell with a higher `--max-turns` to see whether the model
   self-corrects the leftover `timeout = 10` line given more turns, before
   concluding anything about Adaptive Tool Exposure's effect on
   correctness.
5. Only after all four arms have one clean, same-provider run each should
   repetitions and additional task categories be considered, and each
   should be scoped and approved separately.
