# FORGE vs. External Agents — 3-Prompt Manual Benchmark (2026-09-24/25)

## 0. Status and methodology

This is a **manual, single-repetition (n=1 per cell), three-task benchmark**
run by the project author outside the automated `forge experiment` harness,
comparing FORGE against two external agentic coding tools (Antigravity,
GitHub Copilot) on identical prompts. It is **not** a statistically powered
experiment — it exists to produce real, verifiable, same-task data points
for a mid-viva presentation, following the same no-fabrication standard as
the rest of this repository's `docs/step-09-*.md` results.

For every task, both systems' output was manually inspected (and, for
FORGE, independently verified by reading the produced file and by hand-
tracing/executing the required verification script) for correctness before
any metric was recorded. All three tasks passed correctness on every system
tested.

Where a system's UI does not expose a metric (e.g. tool-call count or
latency for Antigravity/Copilot), it is recorded as **N/A** — never
invented or estimated.

## 1. Task 1 — single-file bug fix + feature add

**Prompt:** create `stats.py` with a buggy `average()` function, fix the
bug, add a `median()` function.

**Systems compared:** FORGE (Groq provider, pinned single-provider mode,
no fallback) vs. Antigravity — **same underlying model**,
`openai/gpt-oss-120b`, confirmed on both sides before running.

| Metric | FORGE | Antigravity |
|---|---:|---:|
| Model | openai/gpt-oss-120b | openai/gpt-oss-120b (same) |
| Correct | Yes | Yes |
| Input tokens | 4,033 | 108,900 |
| Output tokens | 504 | 4,100 |
| Total tokens | 4,537 | ~113,000 |
| LLM / API calls | 4 | 7 |
| Tool calls | 3 | N/A |
| LLM latency | 4,985.9 ms | N/A |
| Cache usage | 0% | 0% |

**Ratios (Antigravity ÷ FORGE):** input 27.0×, output 8.1×, total 24.9×,
calls 1.75×. Both outputs functionally identical, both correct.

## 2. Task 2 — multi-file cross-file wiring

**Prompt:** create `config.py` + `client.py`, remove two hardcoded values
in `client.py`, wire them to `config.py` (including a new
`get_max_retries()` function).

**Systems compared:** FORGE (Groq, `openai/gpt-oss-120b`) vs. GitHub
Copilot — **different model** (Copilot's own default model; not
`gpt-oss-120b`). This pair is **not** a same-model comparison and must not
be read as isolating harness overhead the way Task 1 does.

| Metric | FORGE | Copilot |
|---|---:|---:|
| Model | openai/gpt-oss-120b | Copilot default (unconfirmed, ≠ gpt-oss-120b) |
| Correct | Yes | Yes |
| Input tokens | 9,152 | 79,224 |
| Output tokens | 2,205 | 752 |
| Total tokens | 11,357 | 79,976 |
| LLM calls / model rounds | 8 | 4 |
| Tool calls | 7 | N/A |
| LLM latency | 31,776.8 ms (includes ~18s of Groq rate-limit backoff) | N/A |
| Cache usage | 0% | ~74% (58,980 of 79,224 input tokens reported cached) |

**Notable reversal:** FORGE used *more* output tokens and *more* rounds
than Copilot on this task — the opposite of Task 1's pattern. Copilot's
large raw input-token count is substantially offset by its high cache hit
rate, which FORGE's numbers have no equivalent for (0% cache both times
tested). Both outputs functionally identical, both correct.

## 3. Task 3 — mutable-default-argument bug + edge cases

**Prompt:** create `inventory.py` with a `Cart` class containing a classic
Python mutable-default-argument bug; fix it; add `remove_item()` (no-op if
missing) and `apply_discount()` (rounded).

**Systems compared:** FORGE (Gemini provider, pinned single-provider mode)
vs. Antigravity — **same underlying model**, `gemini-3.6-flash`.

| Metric | FORGE | Antigravity |
|---|---:|---:|
| Model | gemini-3.6-flash | gemini-3.6-flash (same) |
| Correct | Yes | Yes |
| Input tokens | 44,184 | 186,200 |
| Output tokens | 1,370 | 5,500 |
| Total tokens | 49,222 | ~191,700 |
| LLM / API calls | 12 | 9 |
| Tool calls | 11 | N/A |
| LLM latency | 91,281.7 ms (includes ~17s of Gemini rate-limit backoff) | N/A |
| Cache usage | 0% | 52% |

**Ratios (Antigravity ÷ FORGE):** input 4.2×, output 4.0×, total 3.9×.
**Calls: Antigravity used fewer (9) than FORGE (12) on this task** — the
reverse of Task 1.

### A real bug found during this run (not a research variable)

Both FORGE/Gemini attempts on this task hit the identical failure: the
model called `list_directory`/`run_shell` before the workspace directory
existed (FORGE creates it lazily, on first `write_file`, not eagerly at
run start). `list_directory` handles this cleanly (`"directory not found:
."`); `run_shell` does not — it lets a raw `NotADirectoryError:
[WinError 267]` traceback surface as an "Unexpected error in tool
run_shell" log entry, caught one level up so the agent loop itself never
crashes, but inconsistent with how `list_directory` handles the identical
condition. Under `RawContextStrategy`, those extra failed turns get resent
in full on every subsequent turn, which is a real, direct contributor to
this run's higher token count relative to Task 1. This is a genuine,
reproducible FORGE defect (tool-level error-handling inconsistency), not a
result of the research variables (tool exposure / context strategy) being
tested.

## 4. Cross-task summary

| Task | Same model on both sides? | Total-token ratio (external ÷ FORGE) | Calls comparison |
|---|---|---:|---|
| 1 | Yes (gpt-oss-120b) | 24.9× | Antigravity +3 calls |
| 2 | No (Copilot uses a different model) | 7.0× (raw, not cache-adjusted) | Copilot −4 calls |
| 3 | Yes (gemini-3.6-flash) | 3.9× | Antigravity −3 calls |

**Honest conclusion:** across the two valid same-model pairs (Task 1,
Task 3), FORGE used meaningfully fewer tokens than Antigravity for an
identically correct result, but the size of that gap is **not constant**
— it ranged from ~25× down to ~4×, and on Task 3 the gap shrank partly
because of a real FORGE-side bug (above), not because Antigravity became
more efficient. Task 2 is not a controlled pair at all (different model)
and additionally involves cache accounting that FORGE has no equivalent
for. All six system-runs across all three tasks produced correct output —
correctness was never the differentiator in this benchmark; token/call
efficiency was, and it varies by task.

This is a 3-task, n=1-per-cell manual benchmark, not a statistically
powered result — it should be presented with that scope, consistent with
this repository's own `docs/step-09-*.md` results.
