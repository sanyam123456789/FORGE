# Step 12 — Long-task benchmark results (2026-11-19)

**Setup:** `forge bench --suite long --repeat 3 --jobs 4 --experiment-id
step12_long`. Four long tasks on the `ledger` project (~1,500 lines incl.
tests) under all six arms, three repetitions each: 72 runs. Provider
OpenCode Go, model `deepseek-v4.1-flash`, temperature 0, no output cap,
limits of 30 turns and 50 tool calls. Success = the run completed **and**
the task's hidden verify script passed. Raw results:
`runs/experiments/step12_long/results.jsonl` (git-ignored); reproduce the
table with `forge report runs/experiments/step12_long/results.jsonl`.

The stable arms ran with the then-default budget (compact at 16k estimated
tokens, target 8k). See §4 for what changed afterwards.

## 1. Per-arm summary

| Arm | Solved | Avg tokens | vs base | Spread | Uncached input | Cached | Peak input | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_raw (baseline) | **12/12** | 153,707 | base | ±23% | **17,969** | **88%** | 18,286 | 12.2 |
| adaptive_raw | **12/12** | 152,454 | −4% | ±29% | 20,967 | 86% | 18,210 | 11.4 |
| fixed_stable | **12/12** | 149,694 | +4% | ±31% | 29,140 | 80% | 15,395 | 13.7 |
| adaptive_stable | **12/12** | 147,989 | +1% | ±29% | 26,907 | 81% | 15,780 | 13.2 |
| fixed_managed | 5/12 | 174,851 | +25% | ±17% | 94,182 | 45% | 12,268 | 21.2 |
| adaptive_managed | 4/12 | 170,469 | +21% | ±23% | 86,083 | 48% | 12,046 | 20.7 |

*vs base*: per task, the arm's mean tokens over its 3 runs divided by
`fixed_raw`'s, averaged over the 4 tasks. *Spread*: run-to-run coefficient
of variation of total tokens for one task under one arm, averaged over
tasks. *Peak input*: the largest single request, averaged over runs.

## 2. What this shows

1. **The Step 6 managed strategy breaks long tasks.** It solved 9 of 24
   runs; every other arm solved 48 of 48. 14 of its 15 failures hit the
   50-tool-call limit. The traces show why: on `ledger-api-docs`, a
   managed run made 49 `read_file` calls and never wrote the file (raw: 10
   reads, then the file). Its 20,000-character budget compresses earlier
   file contents, the model needs them, reads them again, and they are
   compressed again. Rewriting the history every turn also halves the
   cache hit rate (45–48% vs 88%), so it paid for 5× more uncached input.
2. **Cache-friendly compaction (stable) is safe but did not save tokens.**
   All 24 runs solved, total tokens within ±4% of raw (far inside the
   ±23–31% spread), peak request size 16% smaller. But each compaction
   made the model re-read some files (21 reads instead of 10 in one run),
   so uncached input was ~60% *higher* than raw. On these tasks, with an
   88% cache hit rate, the full history is cheaper than a pruned one.
3. **Lean tool exposure made little difference here** (−4%, inside the
   noise). Three of the four long tasks need running tests, so adaptive
   exposure offered all six tools for them, the same as fixed. Its 22%
   saving on the short suite (Step 11) came from tasks that need fewer
   tools.
4. **Run-to-run variation is large.** The same task under the same arm
   varied by ±17–31% on average, and up to 3× (`ledger-recurring-feature`
   under `adaptive_raw`: 116k, 284k and 361k tokens). Single runs cannot
   separate strategies whose effect is smaller than that.

## 3. Per-task total tokens (mean of 3 runs, solved/3)

| Task | fixed_raw | adaptive_raw | fixed_stable | adaptive_stable | fixed_managed | adaptive_managed |
|---|---:|---:|---:|---:|---:|---:|
| ledger-fix-bugs | 152k (3) | 106k (3) | 70k (3) | 103k (3) | 135k (2) | 154k (2) |
| ledger-recurring-feature | 201k (3) | 254k (3) | 207k (3) | 189k (3) | 223k (3) | 213k (2) |
| ledger-rename-memo | 175k (3) | 176k (3) | 175k (3) | 177k (3) | 161k (0) | 145k (0) |
| ledger-api-docs | 87k (3) | 74k (3) | 146k (3) | 123k (3) | 180k (0) | 169k (0) |

## 4. Changes made because of these results

- **`StableContextStrategy` now compacts at 64k estimated tokens** (target
  32k) instead of 16k (target 8k). The largest request in any run here was
  ~18k tokens, so below the new threshold stable sends exactly the raw
  requests and keeps the full cache benefit; compaction remains as a
  safety net for very long chat sessions. (This new default is not
  benchmarked by these tasks: they never get that large.)
- **The `lean-context` mode now uses `stable`, not `managed`.** No mode
  uses the managed strategy any more; it stays available as a research arm.
- **Duplicate-read collapsing** only points back to an earlier result that
  is still visible in full (fixed before this run; see the Step 12 doc).

## 5. Limitations

- Four tasks on one project, three runs per cell (72 runs).
- The managed strategy was run with its Step 6 defaults; a much larger
  budget might behave better, but would then rarely prune anything on
  tasks like these.
- Token counts are the provider's own; "uncached input" is the part billed
  at the full input price.
