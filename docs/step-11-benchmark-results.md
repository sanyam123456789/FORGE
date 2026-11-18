# Step 11 — Benchmark results (2026-11-18)

**Setup:** `forge bench --experiment-id step11_final`. The 10-task baseline
suite under all six arms: one run per task-arm, 60 runs in total. Provider:
OpenCode Go, model `deepseek-v4.1-flash`, temperature 0, no output cap,
limits of 30 turns and 50 tool calls. Raw results are in
`runs/experiments/step11_final/results.jsonl` (git-ignored); reproduce the
table with `forge report runs/experiments/step11_final/results.jsonl`.

## Per-arm summary

| Arm | Solved | Avg tokens | Avg uncached input | Cache hit | Tokens / solved | Avg LLM calls | Avg time |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_raw (baseline) | 10/10 | 7,144 | 1,655 | 76% | 7,144 | 3.8 | 13.8s |
| fixed_managed | 10/10 | 7,230 | 1,647 | 76% | 7,230 | 3.9 | 12.4s |
| adaptive_raw | 10/10 | 5,594 | 1,436 | 73% | 5,594 | 3.4 | 11.3s |
| adaptive_managed | 10/10 | 5,802 | 1,612 | 71% | 5,802 | 3.5 | 12.3s |
| fixed_stable | 10/10 | 6,988 | 1,671 | 75% | 6,988 | 3.7 | 13.7s |
| adaptive_stable | 10/10 | 7,252 | 1,556 | 77% | 7,252 | 4.2 | 15.8s |

## Per-task total tokens (LLM calls)

| Task | fixed_raw | adaptive_raw | adaptive_stable |
|---|---:|---:|---:|
| create-string-utils | 5,457 (3) | 2,558 (2) | 2,738 (2) |
| create-json-config-io | 5,552 (3) | 2,591 (2) | 8,531 (5) |
| edit-report-summary | 7,385 (4) | 6,369 (4) | 6,433 (4) |
| fix-inclusive-sum | 7,290 (4) | 4,503 (3) | 8,628 (5) |
| fix-normalize-trim | 7,024 (4) | 6,105 (4) | 8,261 (5) |
| add-greet-excited-flag | 7,369 (4) | 4,594 (3) | 6,759 (4) |
| refactor-round-money-helper | 8,418 (4) | 5,020 (3) | 7,039 (4) |
| add-mathlib-tests | 7,801 (4) | 10,408 (5) | 12,791 (6) |
| rename-tally-to-count-items | 7,165 (4) | 6,461 (4) | 6,380 (4) |
| multi-file-timeout-setting | 7,979 (4) | 7,330 (4) | 4,955 (3) |

## What this shows

1. **Correctness is unchanged.** All six arms solved all 10 tasks.
2. **Lean tool exposure saves tokens.** With raw context, adaptive exposure
   averaged 5,594 tokens against 7,144 for fixed: **22% fewer**, and it was
   lower on 9 of 10 tasks.
3. **Prompt caching already covers most input.** 71–77% of input tokens
   were served from cache in every arm. That is OpenCode Go's prefix cache
   at work, helped by the stable session id FORGE now sends.
4. **The context strategies were not exercised.** The largest context in
   any run was about 3,600 characters (roughly 1k tokens). The managed
   budget is 20,000 characters and the stable compaction threshold is
   16,000 estimated tokens, so neither ever pruned anything. Raw, managed
   and stable sent the same requests. Their differences in this table are
   run-to-run variation, not strategy effects.
5. **Run-to-run variation is large.** `adaptive_raw` and `adaptive_stable`
   send identical requests on these tasks, yet a single task differed by up
   to 3.3× (create-json-config-io: 2 calls vs 5). With one run per cell,
   differences smaller than that cannot be attributed to a strategy.

## Limitations and next step

- n = 1 per task-arm, 10 short tasks (3–6 LLM calls each).
- To measure context management, the suite needs longer, multi-step tasks
  whose context grows past the budgets, for example reading several larger
  files, iterating on failing tests, or chat sessions with follow-ups. Each
  cell should run 3–5 times.
- The earlier manual comparison against external agents
  ([benchmark-forge-vs-external-agents.md](benchmark-forge-vs-external-agents.md))
  used other providers and is not directly comparable with these numbers.
