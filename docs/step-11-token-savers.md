# Step 11 — Token savers, cache-friendly context, and the everyday CLI

Step 11 turns FORGE from a research harness into a coding agent you use day
to day, without losing its research focus: **fewer tokens and well-managed
context, at the same output quality.** It was driven by measurements from
real runs against OpenCode Go (`deepseek-v4.1-flash`), recorded below.

## 1. What the first real runs showed

| Observation | Evidence | Consequence |
|---|---|---|
| OpenCode Go rejected every request | HTTP 403 (Cloudflare 1010) for the default Python user agent; then HTTP 400 `MissingSessionID` | Provider now sends `User-Agent: forge-agent/1.0` and a stable `x-opencode-session` id per conversation |
| The provider caches unchanged prompt prefixes | A 3-call run: call 2 had 1,152 of 1,302 input tokens cached, call 3 had 1,280 of 1,363 | Context strategies should keep the prompt prefix stable |
| `ManagedContextStrategy` rewrites old messages every turn | By design (Step 6): a turn that becomes "stale" is compressed in place | Every token after that point misses the cache on the next call → new `StableContextStrategy` |
| Hidden tools were still executed | `_dispatch_tool` looked tools up in the full registry, not the exposed subset | Exposure is now enforced |
| The model never calls a tool that is not in the request | A chat task needing `run_shell` (hidden by adaptive exposure) looped 30 turns, 114k tokens, searching for the text "run_shell" | New `enable_tool` meta-tool + stall detection; the same task then took 4 turns / 6.8k tokens |
| Edits turned LF files into CRLF on Windows | `Path.write_text` translates `\n` → `\r\n` | `write_file` / `edit_file` now write bytes exactly and keep each file's line endings |
| The system prompt listed all tools and said the set never changes | Contradicted adaptive exposure; differed in meaning between arms | Prompt no longer names tools |

## 2. Cache-friendly context (`StableContextStrategy`, context name `stable`)

Providers bill cached input tokens at a fraction of the normal price, and
cache only the longest **unchanged prefix** of a request. Rewriting an early
message throws that away.

`StableContextStrategy` applies the Step 6 managed rules **in batches**:

1. While the estimated size of *(last snapshot + new messages)* is at most
   `compact_at_tokens` (default 16,000), it sends exactly that. The previous
   request is a byte-identical prefix of the new one.
2. Above the threshold it runs the managed rules once over the full raw
   history with a `target_tokens` budget (default 8,000) and freezes the
   result as the new snapshot. Only that call pays a cache miss after the
   system prompt and task.
3. It re-compacts only after the context grew by at least
   `min_growth_tokens` (default half the gap), so a history whose protected
   part alone exceeds the threshold cannot cause a cache miss every turn.

It is deterministic, resets itself for a new conversation, and supports
`request_compaction()` (`/compact` in chat). Budgets are in estimated tokens.

## 3. Token estimate (`forge/tokens.py`)

A dependency-free estimate: words, numbers, punctuation runs and whitespace
are charged roughly what a BPE tokenizer charges. On the first request of a
real run (system prompt + task + tool schemas) it estimated 1,211 tokens
against 1,216 reported by the provider. It is used only to decide what fits
in a budget; every reported metric is still the provider's own count.

## 4. Token savers in the agent loop

All of these apply identically under every strategy, so they never confound a
strategy comparison.

| Saver | Behaviour |
|---|---|
| Exposure enforcement | A call to a hidden tool is refused, never silently run |
| `enable_tool` | When adaptive exposure hides tools, a small meta-tool lists them; calling it offers that tool from the next turn on. A pinned list (`FixedToolExposure(pinned=…)`) never grows |
| Duplicate reads | A read-only call (read_file / list_directory / search_files) returning exactly what the same call returned before gets a one-line pointer instead. Once per call: a further repeat gets the full text, so nothing is lost for good. Managed/stable context never lets the pointer supersede the full result |
| Repeated failures | The 3rd identical failing call gets "try a different approach" |
| Stall detection | After 3 turns in a row with nothing new, the model is told to finish or change approach |
| Cut-off replies | A reply stopped by the length limit is continued (up to 3 times) instead of being taken as the final answer |
| Workspace map | The first message lists workspace files (max 60, secrets and tooling folders skipped); `FORGE_REPO_MAP=false` turns it off |
| `search_files` tool | Regex search returning `path:line: text`, so the agent reads only the lines it needs |
| No output cap | `max_tokens` is not sent unless `FORGE_LLM_MAX_OUTPUT_TOKENS` is set |

## 5. Everyday CLI

```bash
forge "fix the bug in app.py"     # one task (same as forge run "...")
forge chat                         # conversation with /slash commands
forge bench                        # every task under all six arms + report
forge report                       # readable table of the newest results
```

Modes (`--mode`, `/mode`): `baseline` (fixed + raw), `lean-tools`
(adaptive + raw), `lean-context` (fixed + managed), `smart` (adaptive +
stable, the default).

Chat commands: `/help /mode /model /tokens /context /tools /compact /diff
/undo /clear /workspace /exit`. `/diff` and `/undo` use a workspace snapshot
taken before each task (FORGE's own trace folder is excluded). One provider
object, and so one `x-opencode-session` id, is kept per conversation.

## 6. Experiment compatibility

- The formal 2×2 (`forge experiment`, `forge analyze`) is unchanged; two
  extra arms, `fixed_stable` and `adaptive_stable`, run under `forge bench`.
- Changes that affect every arm equally: the system prompt (new hash), the
  sixth tool, the workspace map, the loop savers and the provider headers.
  Results from before Step 11 are therefore not directly comparable with
  results after it; compare arms *within* one run.
- Adaptive classification of the 10 suite tasks is unchanged (checked before
  and after adding the new "run / execute" keywords).

## 7. Limitations

- The token estimate is an estimate; it can be off for unusual text.
- Adaptive exposure still classifies by keywords on the task text;
  `enable_tool` is the safety net, not a better classifier.
- `/undo` restores files from a snapshot taken before each task; files over
  2 MB and workspaces over 5,000 files are not tracked.
- The safety layer is still a heuristic, not a sandbox.
