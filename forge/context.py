"""
forge.context — Interaction-context management layer.

In an LLM-based coding agent the "context" is the conversation history
passed to the LLM on every turn.  Context management is the second major
research variable in FORGE:

  Research Arm B: Raw/full interaction history  vs  managed context
                  ↓                                   ↓
          (all messages kept)            (pruning / compression)

Status
------
- ``ConversationContext`` holds the ordered message history for one run.
- ``ContextStrategy`` decides which messages are actually sent to the LLM on
  a given turn.  Step 2 implemented ``RawContextStrategy`` (send the full
  history unchanged).  Step 6 adds ``ManagedContextStrategy``: a
  deterministic, rule-based pruning/compression policy that plugs into the
  same seam without touching the agent loop. See
  ``docs/step-06-managed-context.md`` for the full policy write-up.
- Token counting is deliberately a character-count proxy for now; accurate
  tokeniser-based counting is a later concern.

Design
------
- ``ConversationContext`` holds an ordered list of Messages and exposes
  ``add_message()`` / ``get_messages()`` (the raw, unmodified history).
- The agent runtime holds a ``ContextStrategy`` and calls
  ``strategy.prepare(context)`` to obtain the message list for each LLM call.
  This is the single integration point for the "raw vs managed" research arm.
- A strategy may optionally expose ``last_report()`` — a ``ContextReport``
  describing what the most recent ``prepare()`` call did (sizes before/after,
  items dropped/compressed).  This is purely for measurement; it never
  affects what is sent to the LLM.
"""

from __future__ import annotations

import os
import posixpath
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from forge.llm import Message, ToolCall
from forge.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    """Holds the live conversation history for a single agent run.

    Attributes
    ----------
    messages:
        Ordered list of all messages exchanged so far.
    max_messages:
        Soft ceiling on history length.  When exceeded, the context
        management strategy will be applied (Step 2/3 concern).
    """

    messages: list[Message] = field(default_factory=list)
    max_messages: int = 200

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        """Append *message* to the conversation history.

        When the history exceeds ``max_messages``, a warning is emitted.
        Pruning will be implemented once context strategies are built.
        """
        self.messages.append(message)
        logger.debug(
            "Context: added message role=%s len=%d total_messages=%d",
            message.role,
            len(message.content),
            len(self.messages),
        )
        if len(self.messages) > self.max_messages:
            logger.warning(
                "Context history has exceeded max_messages=%d. "
                "Context pruning is not yet implemented.",
                self.max_messages,
            )

    def get_messages(self) -> list[Message]:
        """Return the full, unmodified message history.

        Any managed-context transformation is applied by a ``ContextStrategy``
        (see below), not here — this method always returns the raw history.
        """
        return list(self.messages)

    def clear(self) -> None:
        """Reset the conversation history (e.g. between agent runs)."""
        self.messages.clear()
        logger.debug("Context cleared.")

    # ------------------------------------------------------------------
    # Introspection helpers (Step 1: very basic)
    # ------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        """Number of messages in the current history."""
        return len(self.messages)

    @property
    def approximate_char_count(self) -> int:
        """Sum of character lengths across all message contents.

        Rough proxy for context size.  Tool-call argument payloads are not
        included.  Accurate token counting is deferred until it is actually
        needed by an experiment.
        """
        return sum(len(m.content) for m in self.messages)


# ---------------------------------------------------------------------------
# Context strategies (research arm B seam)
# ---------------------------------------------------------------------------


class ContextStrategy(ABC):
    """Decides which messages are sent to the LLM for a given turn.

    The agent runtime depends on this abstraction, never on a concrete
    strategy.  Step 2 shipped ``RawContextStrategy``; Step 6 adds
    ``ManagedContextStrategy``.  Both slot in here without changes to the
    agent loop.
    """

    #: Stable identifier recorded in run traces.
    name: str = "abstract"

    @abstractmethod
    def prepare(self, context: "ConversationContext") -> list[Message]:
        """Return the message list to send to the LLM this turn."""

    def last_report(self) -> "ContextReport | None":
        """Stats describing the most recent ``prepare()`` call, if any.

        Default: ``None`` (strategy does not report). This never influences
        what is sent to the LLM — it is measurement only. See
        ``ContextReport``.
        """
        return None


@dataclass(frozen=True)
class ContextReport:
    """Measurement of what one ``ContextStrategy.prepare()`` call did.

    All counts are None-safe by construction (a strategy that ran always
    knows these numbers exactly — this is a deterministic, rule-based
    transformation, never an estimate).

    items_before / items_after:
        Message count in the raw history vs. the list returned to the LLM.
    chars_before / chars_after:
        Character-count proxy for context size (same proxy used elsewhere in
        FORGE — see ``ConversationContext.approximate_char_count``).
    items_dropped:
        Messages removed entirely (whole turns dropped for budget).
    items_compressed:
        Messages kept in the output but with their content replaced by a
        short placeholder (stale/redundant output).
    turns_total / turns_kept_recent / turns_kept_error / turns_dropped_for_budget:
        Turn-level detail for ``ManagedContextStrategy`` (0 for strategies
        that do not use a turn-based policy, e.g. ``RawContextStrategy``).
    """

    items_before: int
    items_after: int
    chars_before: int
    chars_after: int
    items_dropped: int = 0
    items_compressed: int = 0
    turns_total: int = 0
    turns_kept_recent: int = 0
    turns_kept_error: int = 0
    turns_dropped_for_budget: int = 0


class RawContextStrategy(ContextStrategy):
    """Baseline: send the full interaction history, unmodified."""

    name = "raw"

    def __init__(self) -> None:
        self._last_report: ContextReport | None = None

    def prepare(self, context: "ConversationContext") -> list[Message]:
        messages = context.get_messages()
        chars = sum(len(m.content) for m in messages)
        # Trivial pass-through report: nothing is ever dropped or compressed.
        # Recorded purely so run-level measurement code can treat every
        # ContextStrategy uniformly without an isinstance check.
        self._last_report = ContextReport(
            items_before=len(messages),
            items_after=len(messages),
            chars_before=chars,
            chars_after=chars,
        )
        return messages

    def last_report(self) -> ContextReport | None:
        return self._last_report


# ---------------------------------------------------------------------------
# Managed context strategy (Step 6)
# ---------------------------------------------------------------------------

#: A tool-result message produced by ``ToolResult.from_error`` always starts
#: with this prefix (see ``forge.tools.ToolResult.from_error``) — the one
#: place FORGE's own tools mark a hard failure in the text sent to the LLM.
_ERROR_PREFIX = "ERROR"

#: ``RunShellTool`` always appends ``"[exit code N]"`` to its output (see
#: ``forge.builtin_tools.RunShellTool``). A non-zero exit code is FORGE's
#: only other text-only failure signal (there is no separate structured
#: "success" flag carried on ``Message`` — see the Step 6 doc's Limitations).
_EXIT_CODE_RE = re.compile(r"\[exit code (\d+)\]")


def _is_error_tool_message(message: Message) -> bool:
    """True if *message* is a ``role="tool"`` message reporting a failure.

    Deliberately text-only and deterministic: it recognises exactly the two
    failure conventions FORGE's own tools use today. A tool that reports
    failure in some other text shape will not be recognised — see
    Limitations in ``docs/step-06-managed-context.md``.
    """
    if message.role != "tool":
        return False
    content = message.content or ""
    if content.startswith(_ERROR_PREFIX):
        return True
    match = _EXIT_CODE_RE.search(content)
    return bool(match and match.group(1) != "0")


# ---------------------------------------------------------------------------
# Per-resource keys for ManagedContextStrategy
# ---------------------------------------------------------------------------

#: Built-in tools whose results get a resource key. A call to one of these
#: with missing/unparseable arguments falls back to a per-tool-name key (the
#: pre-Fix-3 "latest per tool" behaviour). Any other tool gets a unique key.
_KEYED_TOOLS = frozenset(
    {"read_file", "run_shell", "list_directory", "edit_file", "write_file"}
)
_LABEL_MAX = 80


def _normalize_path(path: str) -> str:
    """Normalize a workspace path so equivalent spellings share one key."""
    norm = posixpath.normpath(path.replace("\\", "/"))
    if norm.startswith("./"):
        norm = norm[2:]
    if os.name == "nt":
        norm = norm.casefold()
    return norm


def _label(text: str) -> str:
    return text if len(text) <= _LABEL_MAX else text[: _LABEL_MAX - 3] + "..."


def _pair_tool_calls(turn: list[Message]) -> dict[int, ToolCall]:
    """Map each tool message's index in *turn* to the call it answers.

    Pairing is done WITHIN the turn only: tool_call_id values are not
    unique across turns (some providers reuse "call_0" every turn). First
    match by id inside this turn's assistant message, then fall back to
    position order (AgentRuntime appends results in call order).
    """
    if not turn or turn[0].role != "assistant":
        return {}
    calls = list(turn[0].tool_calls)
    tool_positions = [j for j, m in enumerate(turn) if m.role == "tool"]
    used: set[int] = set()
    paired: dict[int, ToolCall] = {}
    for j in tool_positions:
        call_id = turn[j].tool_call_id
        if not call_id:
            continue
        for k, call in enumerate(calls):
            if k not in used and call.id == call_id:
                paired[j] = call
                used.add(k)
                break
    remaining = [k for k in range(len(calls)) if k not in used]
    for j in tool_positions:
        if j not in paired and remaining:
            paired[j] = calls[remaining.pop(0)]
    return paired


def _resource_key(
    message: Message, call: ToolCall | None, position: tuple[int, int]
) -> tuple[tuple, str]:
    """Return (key, label) for a tool-result message.

    Results sharing a key supersede each other (only the latest is kept in
    full); the label names the resource in the compression marker.
    """
    name = message.name or (call.name if call else "") or ""
    if name not in _KEYED_TOOLS:
        return ("unique", position), name
    legacy = (("tool", name), name)
    args = call.arguments if call is not None else None
    if not isinstance(args, dict):
        return legacy

    if name == "run_shell":
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return legacy
        collapsed = " ".join(command.split())
        return ("shell", collapsed), _label(collapsed)

    path = args.get("path")
    if name == "list_directory" and path in (None, ""):
        path = "."
    if not isinstance(path, str) or not path:
        return legacy
    kind = {"read_file": "file", "list_directory": "dir"}.get(name, "write")
    return (kind, _normalize_path(path)), _label(path)


class ManagedContextStrategy(ContextStrategy):
    """Deterministic, rule-based context pruning/compression (Step 6).

    No ML summariser, no extra LLM call: every decision is a fixed rule
    evaluated once per ``prepare()`` over the messages already in
    ``ConversationContext``. Given the same history, it always returns the
    same output — this is what makes it safe for a controlled comparison
    against ``RawContextStrategy``.

    Policy (see ``docs/step-06-managed-context.md`` for the full write-up
    and worked examples)
    --------------------
    1. **Always keep** the system message and the original user task
       message, unmodified, first in the output.
    2. The remaining history is split into **turns**: one assistant message
       plus the tool-result message(s) answering its tool calls (this
       mirrors exactly how ``AgentRuntime`` appends messages — see
       ``forge/agent.py``). A turn is always kept or dropped as a whole, so
       a tool call and its result are never separated.
    3. The most recent ``keep_recent_turns`` turns are **always kept in
       full** (the "recent interaction window").
    4. Any older turn containing a failed tool result (see
       ``_is_error_tool_message``) is **always kept in full** — errors and
       test failures are never pruned or compressed.
    5. **Per-resource retention.** Among the remaining (older, non-error)
       turns, each tool result gets a *resource key* from the matching
       tool call in the same turn: ``read_file`` -> the file path,
       ``run_shell`` -> the command (whitespace collapsed),
       ``list_directory`` -> the directory, ``edit_file``/``write_file`` ->
       the written path (a separate namespace, so writes never drop
       reads). Paths are normalized (``\\`` -> ``/``, ``normpath``, no
       leading ``./``, case-folded on Windows). For each key the single most
       recent successful result is kept in full; older results with the
       same key are superseded and replaced with a marker naming the
       resource. Results with different keys are all kept (reading
       ``app.py`` then ``database.py`` keeps both). A built-in tool call
       with missing/unparseable arguments falls back to a per-tool-name key
       (the original Step 6 rule); any other tool gets a unique key and is
       never collapsed. Results are paired with calls *within* each turn
       (by ``tool_call_id``, then by position), because ids may repeat
       across turns. The assistant text of a stale turn is compressed too.
       Compressed messages stay in place (tool-call/result pairing is never
       broken), only their content shrinks. Compression never makes a
       message *larger*: content already shorter than its own placeholder
       marker is left unchanged.
    6. Finally, a hard, deterministic budget (``max_messages`` /
       ``max_chars``) is enforced by dropping whole older, non-protected
       turns (oldest first) until the prepared history fits — or until no
       more droppable turns remain (the system message, the task, the
       recent window and error turns are never dropped).

    Limitations
    -----------
    - Failure detection is text-convention based (see
      ``_is_error_tool_message``); a tool that signals failure some other
      way is not recognised as an error turn.
    - Compression is skipped (message kept as-is) whenever the placeholder
      marker would not actually be shorter than the original content — this
      only matters for unusually short tool/assistant text, since real tool
      output is almost always longer than a ~60-character marker.
    - If the task + system message + protected (recent/error) turns alone
      exceed the budget, the budget is not fully enforced — protecting
      task/error content always wins over the char/message ceiling.
    - This is a placeholder policy, not a learned or embedding-based one;
      see the Step 6 doc for the research framing.
    """

    name = "managed"

    _TOOL_MARKER = "[MANAGED CONTEXT: stale '{name}' output omitted, {chars} chars]"
    _ASSISTANT_MARKER = "[MANAGED CONTEXT: reasoning omitted for stale turn]"

    def __init__(
        self,
        *,
        keep_recent_turns: int = 3,
        max_messages: int = 40,
        max_chars: int = 20_000,
    ) -> None:
        if keep_recent_turns < 0:
            raise ValueError("keep_recent_turns must be >= 0")
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.keep_recent_turns = keep_recent_turns
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._last_report: ContextReport | None = None

    def last_report(self) -> ContextReport | None:
        return self._last_report

    # ------------------------------------------------------------------
    # ContextStrategy interface
    # ------------------------------------------------------------------

    def prepare(self, context: "ConversationContext") -> list[Message]:
        messages = context.get_messages()
        items_before = len(messages)
        chars_before = sum(len(m.content) for m in messages)

        head, turns = self._split(messages)
        n_turns = len(turns)
        recent_start = max(0, n_turns - self.keep_recent_turns)
        recent_idx = set(range(recent_start, n_turns))
        error_idx = {
            i
            for i, turn in enumerate(turns)
            if i not in recent_idx
            and any(_is_error_tool_message(m) for m in turn)
        }
        protected_idx = recent_idx | error_idx

        # Per-resource retention: for each resource key (which file was
        # read, which command was run, ...), the position (turn, index) of
        # the single most recent successful result among non-protected
        # turns is kept in full. Older results with the same key are
        # superseded and compressed; results with different keys never
        # affect each other.
        keys: dict[tuple[int, int], tuple] = {}
        labels: dict[tuple[int, int], str] = {}
        latest_by_key: dict[tuple, tuple[int, int]] = {}
        for i, turn in enumerate(turns):
            if i in protected_idx:
                continue
            calls = _pair_tool_calls(turn)
            for j, m in enumerate(turn):
                if m.role == "tool" and not _is_error_tool_message(m):
                    key, label = _resource_key(m, calls.get(j), (i, j))
                    keys[(i, j)] = key
                    labels[(i, j)] = label
                    latest_by_key[key] = (i, j)
        preserved_positions = set(latest_by_key.values())

        turn_outputs: list[list[Message]] = []
        for i, turn in enumerate(turns):
            if i in protected_idx:
                turn_outputs.append(list(turn))
                continue
            compressed: list[Message] = []
            for j, m in enumerate(turn):
                if m.role == "tool" and (i, j) in keys:
                    if (i, j) in preserved_positions:
                        compressed.append(m)
                    else:
                        compressed.append(
                            self._compress_tool(m, keys[(i, j)], labels[(i, j)])
                        )
                elif m.role == "tool":
                    # Only reachable for an error result, which already
                    # makes the whole turn protected; kept for safety.
                    compressed.append(m)
                elif m.role == "assistant":
                    compressed.append(self._compress_assistant(m))
                else:
                    compressed.append(m)
            turn_outputs.append(compressed)

        # Hard budget: drop oldest non-protected turns until within budget.
        turns_dropped_for_budget = 0
        dropped_idx: set[int] = set()
        for i in range(n_turns):
            items, chars = self._totals(head, turn_outputs)
            if items <= self.max_messages and chars <= self.max_chars:
                break
            if i in protected_idx or not turn_outputs[i]:
                continue
            turn_outputs[i] = []
            dropped_idx.add(i)
            turns_dropped_for_budget += 1

        # Message-level accounting: a message is "compressed" only if its
        # content actually changed (identity check against the original),
        # never inferred from its turn's overall classification — a turn can
        # be partly kept full (the one preserved tool result) and partly
        # compressed (its assistant text) at the same time (see §5 of the
        # Step 6 doc).
        items_compressed = 0
        items_dropped = 0
        for i, (orig_turn, out_turn) in enumerate(zip(turns, turn_outputs)):
            if i in dropped_idx:
                items_dropped += len(orig_turn)
                continue
            for orig_m, out_m in zip(orig_turn, out_turn):
                if out_m is not orig_m:
                    items_compressed += 1

        out = list(head)
        for t in turn_outputs:
            out.extend(t)

        items_after = len(out)
        chars_after = sum(len(m.content) for m in out)

        self._last_report = ContextReport(
            items_before=items_before,
            items_after=items_after,
            chars_before=chars_before,
            chars_after=chars_after,
            items_dropped=items_dropped,
            items_compressed=items_compressed,
            turns_total=n_turns,
            turns_kept_recent=len(recent_idx),
            turns_kept_error=len(error_idx),
            turns_dropped_for_budget=turns_dropped_for_budget,
        )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _totals(
        head: list[Message], turn_outputs: list[list[Message]]
    ) -> tuple[int, int]:
        items = len(head) + sum(len(t) for t in turn_outputs)
        chars = sum(len(m.content) for m in head) + sum(
            len(m.content) for t in turn_outputs for m in t
        )
        return items, chars

    @staticmethod
    def _split(
        messages: list[Message],
    ) -> tuple[list[Message], list[list[Message]]]:
        """Split into (head, turns).

        ``head`` is every system message plus the first user message (the
        original task) — always preserved. Everything after that is grouped
        into turns: each turn starts at an assistant message and includes
        every message up to (not including) the next assistant message,
        matching exactly how ``AgentRuntime`` appends to
        ``ConversationContext`` (assistant-with-tool-calls, then one
        ``tool`` message per call, then the next assistant message, ...).
        """
        head: list[Message] = []
        rest: list[Message] = []
        seen_task = False
        for m in messages:
            if m.role == "system":
                head.append(m)
                continue
            if m.role == "user" and not seen_task:
                head.append(m)
                seen_task = True
                continue
            rest.append(m)

        turns: list[list[Message]] = []
        current: list[Message] | None = None
        for m in rest:
            if m.role == "assistant":
                current = [m]
                turns.append(current)
            elif current is not None:
                current.append(m)
            else:
                # Defensive only: a stray non-assistant message with no open
                # turn should not occur given the agent loop's message
                # order, but is kept (as its own single-message turn) rather
                # than silently dropped.
                turns.append([m])
        return head, turns

    def _compress_tool(self, message: Message, key: tuple, label: str) -> Message:
        chars = len(message.content)
        kind = key[0]
        if kind == "file":
            marker = (
                f"[MANAGED CONTEXT: older read of '{label}' omitted ({chars} chars) "
                "— superseded by a later read; call read_file again if needed]"
            )
        elif kind == "shell":
            marker = (
                f"[MANAGED CONTEXT: older output of '{label}' omitted "
                f"({chars} chars) — a later run exists]"
            )
        elif kind == "dir":
            marker = (
                f"[MANAGED CONTEXT: older listing of '{label}' omitted "
                f"({chars} chars) — a later listing exists]"
            )
        elif kind == "write":
            marker = (
                f"[MANAGED CONTEXT: older {message.name or 'write'} result for "
                f"'{label}' omitted ({chars} chars) — a later write exists]"
            )
        else:  # legacy per-tool-name key
            marker = self._TOOL_MARKER.format(
                name=message.name or "tool", chars=chars
            )
        # Compression must never make content larger — a result already
        # shorter than its own placeholder is left alone (nothing to gain by
        # replacing 8 characters with a 60-character marker).
        if len(marker) >= len(message.content):
            return message
        return Message(
            role=message.role,
            content=marker,
            tool_call_id=message.tool_call_id,
            name=message.name,
            metadata={**message.metadata, "managed_context_compressed": True},
        )

    def _compress_assistant(self, message: Message) -> Message:
        if not message.content or len(self._ASSISTANT_MARKER) >= len(message.content):
            return message
        return Message(
            role=message.role,
            content=self._ASSISTANT_MARKER,
            tool_calls=message.tool_calls,
            metadata={**message.metadata, "managed_context_compressed": True},
        )
