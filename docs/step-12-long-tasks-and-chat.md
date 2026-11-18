# Step 12 — Long-task benchmark, and a chat you can rely on

Step 11 left two gaps:

1. **The context strategies were never measured.** The 10 baseline tasks
   are short (about 1k tokens of context at most), so `managed` and
   `stable` never pruned anything. One run per cell also left differences
   inside run-to-run noise.
2. **The everyday CLI lacked things other coding agents have:** streaming
   output, project rules, resumable chats and approval before changes. It
   also had a bug that could break a chat after Ctrl+C.

## 1. Bug: chat after Ctrl+C or a tool limit

The agent records the assistant message with its tool calls before running
them. Two situations left some of those calls without a result:

- the tool-call limit was reached in the middle of a batch (the loop
  stopped early), and
- Ctrl+C while a tool ran (the exception skipped the rest of the batch).

OpenAI-style APIs reject a conversation in which a tool call has no reply,
so in `forge chat` the next message (and every one after it) would fail
until `/clear`. Also, file changes made before Ctrl+C were not recorded, so
`/undo` could not revert them.

Fixes:

- `AgentRuntime.run()` catches `KeyboardInterrupt` and ends the run with
  status `interrupted` (trace written, counters kept).
- When a run ends, every unanswered tool call of the last assistant turn
  gets a short result: `[FORGE: not run — the user interrupted the run.]`
  (or `the tool-call limit was reached`). Counted in
  `AgentRun.unanswered_calls_closed`.
- `ChatSession.send()` records file changes in a `finally` block, so
  `/diff` and `/undo` cover interrupted tasks.

## 2. Long-task suite (`forge bench --suite long`)

`experiments/long_tasks/` holds four tasks on one fixture project,
`ledger`: a ~1,100-line personal-finance package (money parsing, CSV
import, categorisation rules, reports, budgets, JSON storage, CLI) plus 34
unit tests. Reading most of it takes 15k+ tokens, so the context grows past
the managed and stable budgets.

| Task | What the agent must do | Hidden check |
|---|---|---|
| `ledger-fix-bugs` | Three planted bugs in three modules (accounting negatives, month keys, budgets mixing years); find them from a failing test suite | Tests unchanged (hashes) and passing; each bug checked directly |
| `ledger-recurring-feature` | New module + CLI command + tests | `find_recurring` behaviour on unseen data, exact CLI output, new tests present, suite passes |
| `ledger-rename-memo` | Rename a field used in seven files, keep old files loading | Field renamed, old and v1 files load, CSV headers, rules, CLI, no `.memo` left, suite passes |
| `ledger-api-docs` | Document all 62 public names of 8 modules in `API.md` | A section per module, every public name present, no source changed |

**Hidden verify scripts.** A task may name a `verify` script. It runs
after the agent finishes, as `python -I <script> <workspace>`, from a
temporary folder outside the workspace, so the agent can neither see nor
edit it. Exit code 0 = correct. This checks behaviour, not just text, which
the substring checks of the baseline suite cannot do.

The verify scripts were themselves checked: each fails on the untouched
fixture and passes on a reference solution.

## 3. Repetitions, parallel runs, better report

- `--repeat N` runs every task-arm N times (`EvalResult.repetition`;
  workspaces `<task>/<arm>/rep<n>/`). Repetitions are the outer loop, so
  slow drift in the provider spreads evenly over arms.
- `--jobs N` runs N task-arms at once in threads. Each still has its own
  workspace and provider object; results keep the sequential order.
- `forge report` adds:
  - **vs base** — per task, the arm's mean tokens divided by `fixed_raw`'s
    mean on the same task, averaged over tasks;
  - **spread** — the coefficient of variation of total tokens across
    repetitions, averaged over tasks. A difference smaller than the spread
    is not a reliable result;
  - **peak in** — the largest input of a single call (provider-reported),
    i.e. how large the context got.
- `forge analyze` (the formal 2×2 statistics) still expects one row per
  task-arm; use `forge report` for repeated runs.

## 3a. Results and the changes they led to

Full numbers: [step-12-benchmark-results.md](step-12-benchmark-results.md).
In short: raw and stable context solved 48 of 48 runs, the Step 6 managed
strategy 9 of 24 (it compressed file contents the model still needed, so
the model re-read them until the tool-call limit). Stable compaction at
16k tokens saved no tokens and raised uncached input by ~60%. Therefore:

- `StableContextStrategy` now compacts at 64k estimated tokens (target
  32k), so ordinary tasks send exactly the raw requests;
- the `lean-context` mode uses `stable` instead of `managed`; no mode uses
  `managed` any more (it remains a research arm).

## 4. A token-saver interaction fixed

Duplicate-read collapsing (Step 11) answers a repeated identical read with
a pointer to the earlier result. Under managed or stable context, that
earlier result may already be compressed in what the model sees, so the
pointer referred to text the model no longer had, and it had to read a
third time. The pointer is now used only when the earlier result was sent
in full in the current request (or earlier in the same batch).

## 5. New everyday features

| Feature | How |
|---|---|
| Streaming | Reply text appears as it is generated (`stream: true` with `stream_options.include_usage`, so token counts are unchanged). Tool calls, DeepSeek reasoning text and usage are rebuilt from the stream. A broken stream is retried only if no text was shown yet. `--no-stream` / `FORGE_STREAM=false` turn it off. Providers opt in with `supports_streaming`. |
| Project rules | `AGENTS.md` and `FORGE.md` in the workspace root are added to the system prompt once (cut at 20,000 characters), marked as already included so the model does not read the file again. Being in the system prompt, they are in the cached prefix of every request. `FORGE_PROJECT_RULES=false` turns it off. |
| Saved chats | Each `forge chat` task saves the conversation to `logs/sessions/<id>.json` (messages, mode, totals). `forge chat --resume` continues the newest one for the current folder; `--resume <id>` a specific one; `/sessions` lists them. The /undo history and the provider's cache are not restored. |
| Approval | `--approve`, `/approve on`, or `FORGE_APPROVE=true`: before `write_file`, `edit_file` and `run_shell`, FORGE shows a small diff, the new file, or the command, and asks. `y` runs, `n` (or Enter) skips, `a` allows that tool for the session, other text skips and is passed to the agent as the user's instruction. Reads are never asked. |
| Quieter `forge run` | Routine INFO log lines no longer interleave with the streamed reply; `--verbose` shows them. |

## 6. Verified live

On OpenCode Go (`deepseek-v4.1-flash`):

- streaming returned text in pieces, the same usage numbers, tool calls and
  the reasoning text that DeepSeek needs echoed back;
- a task in a folder with an `AGENTS.md` rule ("start every module
  docstring with ...") produced a file that followed it;
- a chat saved after one task and resumed in a new process answered a
  question about the first task without any tool call.

## 7. Limitations

- The long suite has four tasks on one project; it shows how the
  strategies behave as context grows, not how they behave on every kind of
  codebase.
- Adaptive tool exposure is still a keyword classifier. Step 12 added the
  generic keyword "test suite" (classification of the 10 baseline tasks is
  unchanged); `ledger-api-docs` is classified as an edit task and must turn
  on `write_file` with `enable_tool`, which is part of what the benchmark
  measures.
- Approval is a convenience, not a sandbox: an approved command can still
  do anything the safety layer allows.
