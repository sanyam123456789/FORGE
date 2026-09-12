# Step 6 — Managed Context Strategy

## 1. Purpose

Step 6 adds the first real intervention for **research arm B**: instead of
always sending the full, unmodified interaction history to the LLM
(`RawContextStrategy`), the agent can now send a pruned/compressed history
(`ManagedContextStrategy`).

The research question this step starts to answer:

> Can managed context reduce context size (messages/characters) turn over
> turn — without hurting task success — compared to the Raw baseline?

Step 6 does **not** run the controlled comparison itself (that is Step 7's
2×2 experiment). It ships the intervention and the measurement hooks needed
to compare it later, exactly as Step 5 did for tool exposure.

## 2. Scope

**In scope:** a deterministic `ManagedContextStrategy`, wired into the
existing `ContextStrategy` seam (`forge/context.py`), the existing
`ExperimentConfig` (`forge/evaluation/experiment.py`), and the existing CLI
(`forge evaluate --context-strategy managed`). All four strategy
combinations (fixed+raw, fixed+managed, adaptive+raw, adaptive+managed) are
now selectable and runnable.

**Out of scope:**

- The controlled 2×2 experiment (many tasks, repeated runs, statistical
  comparison) — Step 7.
- Statistical analysis, dashboards/UI.
- WhatsApp/Telegram, Pi, Cursor/Copilot/Hermes/Antigravity comparisons, full
  SWE-bench.
- An ML/LLM-based summariser — explicitly excluded by design (see §3).

No second agent loop was created, and `AgentRuntime.run()`'s control flow is
**unchanged** — it already called `cfg.context_strategy.prepare(ctx)` since
Step 2; Step 6 adds a second concrete implementation of that interface, plus
a small, additive measurement hook (see §5).

## 3. Design

### 3.1 Raw mode (baseline, unchanged)

`RawContextStrategy.prepare()` still returns `context.get_messages()`
unmodified — byte-for-byte the Step 2 behaviour. The only addition is that
it now also records a trivial `ContextReport` (chars/items before == after,
nothing dropped or compressed) purely so measurement code can treat every
`ContextStrategy` uniformly without an `isinstance` check. This does not
change what is sent to the LLM.

### 3.2 Managed mode (new)

`ManagedContextStrategy` (`forge/context.py`) is a small, fully
deterministic, explainable policy — no summarisation model, no embeddings,
no extra LLM call to decide what to keep:

1. **Always keep** the system message and the original user task message,
   unmodified, first in the output.
2. **Group into turns.** Everything else is split into *turns*: one
   assistant message plus the tool-result message(s) answering its tool
   calls. This mirrors exactly how `AgentRuntime` appends to
   `ConversationContext` (assistant-with-tool-calls, then one `tool`
   message per call, then the next assistant message, ...). A turn is
   always kept or dropped **as a whole** — a tool call and its result are
   never separated, so the prepared history is always structurally valid.
3. **Recent window.** The most recent `keep_recent_turns` turns (default 3)
   are always kept in full.
4. **Errors and test failures.** Any older turn containing a failed tool
   result is always kept in full, never pruned or compressed. Failure is
   detected two ways, both purely textual and deterministic:
   - the tool message's content starts with `"ERROR"` (every
     `ToolResult.from_error(...)` path — safety violations, bad arguments,
     unknown tool, file-not-found, etc. — uses this prefix), or
   - the content contains `"[exit code N]"` with `N != 0` (how
     `RunShellTool` reports a failing command, e.g. a failing `pytest`
     run).
5. **Latest relevant result per tool, older redundant output compressed.**
   Among the remaining (older, non-error) turns, for each tool name the
   single most recent *successful* call's result is kept in full. Every
   other older successful result for that tool — and the assistant text of
   an otherwise-stale turn — is replaced with a short placeholder (e.g.
   `[MANAGED CONTEXT: stale 'read_file' output omitted, 812 chars]`). The
   message itself stays in place (same role, `tool_call_id`, `name`) so
   pairing is never broken; only its content shrinks. Compression never
   makes a message *larger*: if the original content is already shorter than
   its own placeholder marker, it is left unchanged (this mostly matters for
   short synthetic test fixtures — real tool output is almost always longer
   than a ~60-character marker).
6. **Hard budget.** Finally, a deterministic ceiling (`max_messages=40`,
   `max_chars=20_000` by default) is enforced by dropping whole older,
   non-protected turns — oldest first — until the prepared history fits, or
   until no more droppable turns remain. The system message, the task, the
   recent window, and error turns are **never** dropped for budget.

Every step above is a fixed rule over data already in `ConversationContext`
— given the same history, `ManagedContextStrategy` always returns the same
output. That is the whole determinism argument: no sampling, no model call,
no wall-clock- or randomness-dependent behaviour.

### 3.3 What it preserves vs. what it removes

| Kept in full | Compressed (message kept, content shortened) | Dropped entirely (only under budget pressure) |
|---|---|---|
| System message | Assistant text in an older, non-error, non-recent turn | Whole older, non-protected turns, oldest first |
| Original task | Older, redundant *successful* tool results (all but the latest per tool name) | |
| Most recent `keep_recent_turns` turns | | |
| Any turn with a failed tool result | | |
| The latest successful result per tool name | | |

### 3.4 Selecting the mode

Programmatically (`forge/agent.py`, unchanged interface):

```python
from forge.agent import AgentConfig, AgentRuntime
from forge.context import ManagedContextStrategy

config = AgentConfig(
    task="Fix the off-by-one bug in ranges.py.",
    context_strategy=ManagedContextStrategy(),   # default is RawContextStrategy()
)
AgentRuntime(config).run()
```

Via the evaluation layer:

```python
from forge.evaluation import ExperimentConfig

exp = ExperimentConfig.from_settings(settings, task_id="fix-inclusive-sum",
                                      context_strategy="managed")
tool_strategy, context_strategy = exp.resolve_strategies()
```

Via the CLI:

```bash
forge evaluate --suite-task-id fix-inclusive-sum --context-strategy managed
forge evaluate --suite-task-id fix-inclusive-sum --context-strategy raw       # baseline, default
forge evaluate --suite-task-id fix-inclusive-sum --tool-strategy adaptive --context-strategy managed
```

`--context-strategy` now accepts `raw` (default) or `managed`. Combined with
`--tool-strategy` (`fixed` default, or `adaptive`), all four cells of the
eventual 2×2 are runnable today.

## 4. Backward compatibility (Fixed + Raw, Adaptive + Raw)

- `RawContextStrategy.prepare()` returns the exact same list of `Message`
  objects it always did — no filtering, no reordering, no mutation. Its
  only addition (recording a trivial `ContextReport`) has zero effect on
  the return value.
- `AgentRuntime.run()` control flow is unmodified except for one small,
  additive change (see §5) — it already depended only on
  `ContextStrategy.prepare()`.
- `ExperimentConfig` default (`context_strategy="raw"`) and its
  `resolve_strategies()` behaviour for `"raw"` are unchanged.
- `AdaptiveToolExposure` (Step 5) is completely untouched — the two research
  arms are orthogonal seams (`tool_exposure` vs. `context_strategy` on
  `AgentConfig`), so Adaptive + Raw behaves exactly as it did in Step 5.
- Every Step 1–5 test still passes; the handful that specifically encoded
  "managed context is not implemented yet" were updated because that
  premise is exactly what this step changes (see §7, and compare to how
  Step 5 updated the equivalent "adaptive is not implemented" tests).

## 5. The one agent-loop change, and why

`AgentRuntime.run()` gained two lines per turn:

```python
prepared = cfg.context_strategy.prepare(ctx)
context_report = cfg.context_strategy.last_report()
```

and `tracer.record_llm_request(...)` now also receives the report's fields
(before/after item and char counts, dropped/compressed counts) when one is
available. This is the smallest possible way to surface *how much* a
strategy pruned, per turn, without the loop knowing anything about *how* it
decided — `AgentRuntime` still only calls `ContextStrategy.prepare()` and
now also the new optional `last_report()`, never anything strategy-specific.
`last_report()` defaults to `None` on the base class, so a strategy that
does not implement it (any future strategy) costs the loop nothing beyond
one `None` check. At the end of the run, the tracer's cumulative totals are
copied onto `AgentRun` (`context_items_dropped`, `context_items_compressed`,
`context_chars_saved`) the same way token/latency accumulators already were.

No other line of the control flow (tool dispatch, turn/tool-call ceilings,
error handling, status transitions) changed.

## 6. Measurement / observability

- `forge.context.ContextReport` — new dataclass: `items_before/after`,
  `chars_before/after`, `items_dropped`, `items_compressed`, plus
  turn-level detail (`turns_total`, `turns_kept_recent`, `turns_kept_error`,
  `turns_dropped_for_budget`). Returned by `ContextStrategy.last_report()`
  (default `None`; `RawContextStrategy` and `ManagedContextStrategy` both
  implement it).
- `RunTracer.record_llm_request(...)` — extended (additive keys only) with
  `context_items_before/after`, `context_chars_before/after`,
  `context_items_dropped`, `context_items_compressed` on every `llm_call`
  trace event. `RunTracer` also accumulates cumulative
  `context_items_dropped_total` / `context_items_compressed_total` /
  `context_chars_saved_total` across the run (`None`-safe: these stay
  `None` if no turn ever supplied a report), included in `record_run_end`'s
  event data and in `RunTracer.summary()`.
- `AgentRun` — three new fields: `context_items_dropped`,
  `context_items_compressed`, `context_chars_saved` (cumulative across the
  run), populated from the tracer's cumulative counters.
- `EvalResult` — the same three fields, populated by `EvaluationRunner` from
  `AgentRun`. `EvalResult.context_strategy` already existed (Step 3/5); it
  now also takes the value `"managed"`.
- No metric is fabricated: every new field is `None` until a strategy that
  actually reports (`RawContextStrategy` or `ManagedContextStrategy`) has
  run at least one turn.
- Nothing about token counting changed — `input_tokens`/`output_tokens`
  remain whatever the provider reports (`None` when unreported), exactly as
  before. Character counts remain the same character-count proxy
  (`len(message.content)`) used everywhere else in FORGE; no new tokenizer
  was introduced.

## 7. Files changed

**Modified:**

- `forge/context.py` — added `ContextReport`, `ContextStrategy.last_report()`
  (default `None`), `ManagedContextStrategy`, and gave `RawContextStrategy`
  a trivial `last_report()` (its `prepare()` return value is unchanged).
- `forge/agent.py` — two additive lines per turn to capture and forward the
  context report to the tracer (see §5); three new `AgentRun` fields
  populated in the `finally` block.
- `forge/observability.py` — `record_llm_request(...)` gained six optional,
  additive keyword arguments; `RunTracer` gained three cumulative
  properties and included them in `record_run_end` and `summary()`;
  `RunSummary` gained three new optional fields.
- `forge/evaluation/experiment.py` — `CONTEXT_STRATEGY_MANAGED` moved from
  `FUTURE_CONTEXT_STRATEGIES` to `IMPLEMENTED_CONTEXT_STRATEGIES`;
  `resolve_strategies()` now returns `ManagedContextStrategy()` for
  `context_strategy="managed"`.
- `forge/evaluation/__init__.py` — exported `CONTEXT_STRATEGY_MANAGED`;
  updated module docstring.
- `forge/evaluation/result.py` — `EvalResult` gained three new optional
  fields (see §6).
- `forge/evaluation/runner.py` — passes the three new fields from
  `AgentRun` into `EvalResult`; docstring updated.
- `forge/cli.py` — `--context-strategy` help text updated to list `managed`
  as implemented; `--context-strategy`/`--tool-strategy` result fields
  extended with the new context metrics; `ExperimentConfig` construction
  moved inside the existing `try/except (ImportError, NotImplementedError,
  ValueError)` block in `_evaluate_command` so an unrecognised strategy
  *name* (e.g. a typo) now also exits cleanly with code 2 instead of an
  uncaught traceback — previously this path was only reachable via
  `resolve_strategies()` for a recognised-but-unimplemented name, and after
  Step 6 there are no more recognised-but-unimplemented names on either
  axis, so nothing exercised it.
- `docs/architecture.md` — Arm B status table, module table, metrics table,
  and decision log updated.

**Not modified:** `forge/tools.py`, `forge/builtin_tools.py`,
`forge/llm.py`, `forge/providers/gemini.py`, `forge/prompts.py`,
`forge/safety.py`, `forge/config.py`, `forge/evaluation/task.py`,
`forge/evaluation/suite.py`, `forge/evaluation/aggregate.py`,
`experiments/tasks/*` (the Step 4 baseline suite and its results are
untouched). `AgentRuntime.run()`'s control flow (turn/tool-call limits,
tool dispatch, status transitions, error handling) is unchanged beyond the
two additive lines in §5.

## 8. Example behaviour

```text
$ python -c "
from forge.context import ConversationContext, ManagedContextStrategy
from forge.llm import Message, ToolCall

ctx = ConversationContext()
ctx.add_message(Message(role='system', content='You are FORGE.'))
ctx.add_message(Message(role='user', content='Read a.py, then b.py, then edit c.py.'))
for i in range(6):
    ctx.add_message(Message(role='assistant', content='', tool_calls=[
        ToolCall(id=f'c{i}', name='read_file', arguments={'path': 'a.py'})]))
    ctx.add_message(Message(role='tool', content=f'contents of a.py, version {i}', name='read_file', tool_call_id=f'c{i}'))

strategy = ManagedContextStrategy(keep_recent_turns=2)
prepared = strategy.prepare(ctx)
report = strategy.last_report()
print(report)
# -> items_before=14, items_after=... , items_compressed=... (older read_file
#    turns compressed; only the most recent read_file result and the last 2
#    turns kept in full)
"
```

Running the Step 4 baseline suite's scripted (short) task traces through
`ManagedContextStrategy` produces the same prepared history as
`RawContextStrategy` (no compression/dropping triggers on a 2–3 turn run) —
this is expected: the recent window alone covers short runs. The strategies
diverge once a run has more turns than `keep_recent_turns`, which is by
design (see Limitations).

## 9. Limitations

- **Text-convention-based failure detection.** `_is_error_tool_message()`
  recognises exactly two conventions already used by FORGE's own tools
  (`"ERROR"` prefix, `"[exit code N]"` with `N != 0`). A tool that reports
  failure some other way is not recognised as an error turn and may be
  compressed like a successful one. There is no structured "success" flag
  carried on `Message` today — adding one would be a `forge.llm` change
  outside this step's scope.
- **Fixed defaults, not yet CLI-tunable.** `keep_recent_turns`,
  `max_messages`, and `max_chars` are constructor defaults
  (`ManagedContextStrategy()` with no arguments, both from
  `AgentConfig` and from `ExperimentConfig.resolve_strategies()`). Exposing
  them as CLI flags is a natural, small follow-up, deliberately left out to
  keep this step's surface area minimal.
- **Budget vs. protected content.** If the system message, the task, the
  recent window, and any error turns already exceed `max_chars`/
  `max_messages` on their own, the budget is not fully enforced — protecting
  task and error content always wins over the ceiling. This is a
  deliberate trade-off (never silently lose the task or an error), not a
  bug, but it means the budget is a *ceiling under normal conditions*, not
  an absolute guarantee.
- **Compression never grows a message.** If the placeholder marker would
  not actually be shorter than the original content, the message is left
  unchanged rather than "compressed" into something bigger. This mostly
  matters for unusually short tool output; real tool output (file
  contents, directory listings, shell output) is almost always longer than
  a ~60-character marker.
- **Turn granularity for dropping, message granularity for compression.**
  Whole turns are dropped together (to keep tool-call/result pairing
  intact), but individual tool messages within a turn can be compressed
  independently of their sibling assistant message. This asymmetry is
  intentional (see §3) but is a design detail worth restating in a viva.
- **Provider-signature interaction is untested against a real provider.**
  Gemini 3.x attaches an opaque `thought_signature` to some tool-call parts
  and is documented to require it echoed back verbatim on the *immediately
  following* request (see `forge/providers/gemini.py`). Dropping or
  compressing an *older* turn (several turns in the past, its "next
  request" already long satisfied) should not violate that constraint, but
  this has only been verified with scripted/fake providers, per this step's
  explicit instruction not to consume live Gemini quota. Anyone enabling
  Managed Context against a live Gemini run should watch for provider
  errors on the first live run.
- **No research conclusion yet.** Step 6 does not claim Managed Context
  improves anything — it only makes the comparison possible. Whether it
  actually reduces context size/tokens/latency while preserving success is
  an empirical question for the Step 7 controlled 2×2 experiment.
- **The formal 2×2 experiment and statistical analysis are explicitly NOT
  implemented in this step** — only that all four cells are individually
  selectable and runnable through the existing `ExperimentConfig` /
  `EvaluationRunner` / CLI surface.

## 10. Tests added

- `tests/test_context.py::TestManagedContextStrategy` — deterministic
  behaviour (same input → same output, repeated calls), original task/system
  message preservation, recent-window preservation, error/test-failure
  preservation (both the `"ERROR"` prefix and `"[exit code N]"` cases),
  removal/compression of stale redundant successful tool output while
  keeping the latest per tool name, hard budget enforcement (both
  `max_messages` and `max_chars`), empty history, very small history
  (system+task only, or a single turn), long histories (many turns,
  confirms budget dropping and cumulative `ContextReport` counts), and
  `RawContextStrategy`'s unchanged output plus its new trivial
  `last_report()`.
- `tests/test_evaluation.py` — `ExperimentConfig` now resolves `"managed"`
  to a real strategy (replacing the Step 5-era
  `test_managed_context_strategy_is_not_runnable`, whose premise this step
  removes — mirroring how Step 5 updated the equivalent adaptive-tool
  test); new `EvaluationRunner` tests drive the real agent loop under
  `fixed+managed` and `adaptive+managed` and assert both task success and
  the new `EvalResult` context fields.
- `tests/test_agent.py` — an end-to-end scripted-provider run long enough to
  trigger recent-window/compression/budget behaviour under
  `ManagedContextStrategy`, confirming the trace records the new
  `context_items_before/after` etc. fields and that `AgentRun` carries the
  cumulative totals; a regression check that Fixed+Raw and Adaptive+Raw
  traces are unaffected (no new required keys, existing assertions still
  pass unmodified).
- `tests/test_cli.py` — `test_evaluate_unimplemented_strategy_exits_2` was
  retired (its premise — `managed` being recognised-but-unimplemented — no
  longer holds after this step) and replaced with
  `test_evaluate_unknown_strategy_exits_2` (a genuinely unknown strategy
  name, e.g. `--context-strategy banana`, still exits 2) and
  `test_evaluate_managed_context_strategy_runs` (mirrors
  `test_evaluate_adaptive_tool_strategy_runs` for the context axis).
- `tests/test_observability.py` — new assertions that
  `record_llm_request(...)`'s new optional fields round-trip into the event
  data, and that the tracer's cumulative context properties stay `None`
  when no report was ever supplied.

All existing Step 1–5 tests were re-run unmodified except the ones noted
above, whose assertions specifically encoded "managed context is not
implemented yet" — the premise Step 6 exists to change.
