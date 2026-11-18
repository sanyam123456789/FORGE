"""
forge.chat — interactive sessions with slash commands (``forge chat``).

Type a task in plain words; FORGE works on it and keeps the conversation, so
follow-ups ("now add tests") build on what it already knows. Everything else
is a slash command:

    /help               list commands
    /mode [name]        show or switch mode (baseline, lean-tools, lean-context, smart)
    /model [name]       show or switch model
    /tokens             tokens used this session, cache hit rate
    /context            what is in the context now (sizes, compactions)
    /tools              tools offered on the last request
    /compact            compact the context on the next request
    /diff [all]         file changes from the last task (or the whole session)
    /undo               revert the file changes of the last task
    /clear              start a fresh conversation
    /workspace [path]   show or change the folder FORGE works in
    /exit               quit

One provider object is kept for the whole conversation, so every request
carries the same session id — OpenCode Go uses it for prompt caching.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.agent import AgentConfig, AgentRun, AgentRuntime
from forge.builtin_tools import SEARCH_SKIP_DIRS, build_default_registry
from forge.context import ConversationContext, StableContextStrategy
from forge.llm import LLMProvider, get_provider
from forge.modes import DEFAULT_MODE, MODES, Mode, build_strategies, get_mode
from forge.tokens import estimate_messages_tokens

# Workspace snapshots (for /diff and /undo) skip big or numerous files.
_SNAPSHOT_MAX_FILE_BYTES = 2_000_000
_SNAPSHOT_MAX_FILES = 5_000

Snapshot = dict[str, bytes]


def snapshot_workspace(
    root: Path, *, exclude: tuple[Path, ...] = ()
) -> Snapshot | None:
    """Map of workspace-relative path -> bytes, or None if too large to track.

    *exclude* lists folders FORGE itself writes to (run traces), so they are
    never reported as the agent's changes and never touched by /undo.
    """
    files: Snapshot = {}
    if not root.is_dir():
        return files
    excluded = {p.resolve() for p in exclude}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SEARCH_SKIP_DIRS and (Path(dirpath) / d).resolve() not in excluded
        ]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > _SNAPSHOT_MAX_FILE_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            files[path.relative_to(root).as_posix()] = data
            if len(files) > _SNAPSHOT_MAX_FILES:
                return None
    return files


@dataclass
class FileChanges:
    created: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.created or self.modified or self.deleted)

    def summary(self) -> str:
        parts = []
        if self.created:
            parts.append(f"created {', '.join(self.created)}")
        if self.modified:
            parts.append(f"modified {', '.join(self.modified)}")
        if self.deleted:
            parts.append(f"deleted {', '.join(self.deleted)}")
        return "; ".join(parts) if parts else "no file changes"


def compare_snapshots(before: Snapshot, after: Snapshot) -> FileChanges:
    return FileChanges(
        created=sorted(p for p in after if p not in before),
        modified=sorted(p for p in after if p in before and after[p] != before[p]),
        deleted=sorted(p for p in before if p not in after),
    )


def unified_diff(before: Snapshot, after: Snapshot) -> str:
    changes = compare_snapshots(before, after)
    chunks: list[str] = []
    for path in changes.created + changes.modified + changes.deleted:
        old = _text(before.get(path))
        new = _text(after.get(path))
        if old is None or new is None:
            chunks.append(f"Binary or non-UTF-8 file changed: {path}\n")
            continue
        chunks.extend(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{path}" if path in before else "/dev/null",
                tofile=f"b/{path}" if path in after else "/dev/null",
            )
        )
    return "".join(chunks)


def _text(data: bytes | None) -> str | None:
    if data is None:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


@dataclass
class _Totals:
    tasks: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_reported: bool = False
    latency_ms: float = 0.0

    def add(self, run: AgentRun) -> None:
        self.tasks += 1
        self.llm_calls += run.llm_calls
        self.tool_calls += run.tool_calls
        self.input_tokens += run.input_tokens or 0
        self.output_tokens += run.output_tokens or 0
        if run.cached_input_tokens is not None:
            self.cache_reported = True
            self.cached_input_tokens += run.cached_input_tokens
        self.latency_ms += run.llm_latency_ms or 0.0


class ChatSession:
    """One interactive conversation: context, provider, mode, undo history."""

    def __init__(
        self,
        *,
        settings: Any,
        workspace: str | Path | None = None,
        mode: str = DEFAULT_MODE,
        provider: LLMProvider | None = None,
        write_trace: bool = True,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.settings = settings
        self.workspace = Path(workspace or settings.workspace_root).resolve()
        self.mode: Mode = get_mode(mode)
        self._provider = provider
        self._write_trace = write_trace
        self._on_tool_call = on_tool_call
        self.totals = _Totals()
        self.last_run: AgentRun | None = None
        self._history: list[tuple[Snapshot, Snapshot]] = []  # (before, after) per task
        self._session_start: Snapshot | None = None
        self._pending_note: str | None = None
        self._reset_conversation()

    # -- state -------------------------------------------------------------

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider(settings_override=self.settings)
        return self._provider

    @property
    def model(self) -> str:
        return self.provider.model_name

    def _reset_conversation(self) -> None:
        self.context = ConversationContext()
        self.registry = build_default_registry(workspace_root=self.workspace)
        self.tool_exposure, self.context_strategy = build_strategies(self.mode)
        self._history.clear()
        self._session_start = None
        if self._provider is not None and hasattr(self._provider, "new_session"):
            self._provider.new_session()

    def _snapshot(self) -> Snapshot | None:
        return snapshot_workspace(self.workspace, exclude=(Path(self.settings.log_dir),))

    # -- running tasks -----------------------------------------------------

    def send(self, text: str) -> tuple[AgentRun, FileChanges]:
        """Run one user message as an agent task within this conversation."""
        before = self._snapshot()
        if self._session_start is None and before is not None:
            self._session_start = dict(before)
        task = text
        if self._pending_note:
            task = f"{self._pending_note}\n\n{text}"
            self._pending_note = None
        config = AgentConfig(
            task=task,
            tool_exposure=self.tool_exposure,
            context_strategy=self.context_strategy,
        )
        runtime = AgentRuntime(
            config,
            provider=self.provider,
            registry=self.registry,
            workspace_root=self.workspace,
            write_trace=self._write_trace,
            log_dir=self.settings.log_dir,
            context=self.context,
            on_tool_call=self._on_tool_call,
        )
        run = runtime.run()
        self.last_run = run
        self.totals.add(run)
        changes = FileChanges()
        after = self._snapshot()
        if before is not None and after is not None:
            changes = compare_snapshots(before, after)
            if changes.any:
                self._history.append((before, after))
        return run, changes

    # -- slash-command actions ---------------------------------------------

    def set_mode(self, name: str) -> None:
        self.mode = get_mode(name)
        # Same conversation, new strategies (context strategies hold state).
        self.tool_exposure, self.context_strategy = build_strategies(self.mode)

    def set_model(self, name: str) -> None:
        self.settings = dataclasses.replace(self.settings, llm_model=name)
        provider = get_provider(settings_override=self.settings)
        provider.preflight()
        self._provider = provider

    def set_workspace(self, path: str) -> None:
        new = Path(path).expanduser().resolve()
        if not new.is_dir():
            raise ValueError(f"not a directory: {new}")
        self.workspace = new
        self._reset_conversation()

    def clear(self) -> None:
        self._reset_conversation()

    def request_compaction(self) -> bool:
        if isinstance(self.context_strategy, StableContextStrategy):
            self.context_strategy.request_compaction()
            return True
        return False

    def diff(self, *, whole_session: bool = False) -> str:
        if whole_session:
            if self._session_start is None:
                return ""
            now = self._snapshot() or {}
            return unified_diff(self._session_start, now)
        if not self._history:
            return ""
        before, after = self._history[-1]
        return unified_diff(before, after)

    def undo(self) -> FileChanges | None:
        """Restore the workspace to before the last task that changed files."""
        if not self._history:
            return None
        before, after = self._history.pop()
        changes = compare_snapshots(before, after)
        for path in changes.created:
            target = self.workspace / path
            if target.is_file():
                target.unlink()
        for path in changes.modified + changes.deleted:
            target = self.workspace / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(before[path])
        # Tell the model on the next message, so its picture stays accurate.
        self._pending_note = (
            "[Note from the user: your previous file changes were undone "
            f"({changes.summary()}). The files are back to their earlier state.]"
        )
        return changes

    def context_stats(self) -> dict[str, Any]:
        messages = self.context.get_messages()
        report = self.last_run.last_context_report if self.last_run else None
        return {
            "messages": len(messages),
            "history_tokens_est": estimate_messages_tokens(messages),
            "last_sent_tokens_est": report.tokens_after if report else None,
            "last_sent_messages": report.items_after if report else None,
            "compressed": report.items_compressed if report else None,
            "dropped": report.items_dropped if report else None,
            "compactions": getattr(self.context_strategy, "compactions", None),
            "strategy": self.context_strategy.name,
        }


# ---------------------------------------------------------------------------
# Terminal front end
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Type a task in plain words. Slash commands:
  /help               this list
  /mode [name]        show or switch mode: baseline, lean-tools, lean-context, smart
  /model [name]       show or switch model
  /tokens             tokens used this session and cache hit rate
  /context            what is in the context now
  /tools              tools offered on the last request
  /compact            compact the context on the next request (smart mode)
  /diff [all]         file changes from the last task (or the whole session)
  /undo               revert the file changes of the last task
  /clear              start a fresh conversation
  /workspace [path]   show or change the folder FORGE works in
  /exit               quit"""


def format_tokens(n: int | None) -> str:
    if n is None:
        return "n/a"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def result_line(run: AgentRun) -> str:
    """One-line run summary, e.g. '✓ Done · 3 turns · 4.0k tokens · 63% cached · 10.9s'."""
    head = "✓ Done" if run.ok else f"✗ Stopped ({run.status})"
    parts = [head, f"{run.turns} turn{'s' if run.turns != 1 else ''}"]
    if run.total_tokens is not None:
        parts.append(f"{format_tokens(run.total_tokens)} tokens")
    if run.cache_hit_rate is not None:
        parts.append(f"{run.cache_hit_rate:.0%} cached")
    if run.llm_latency_ms:
        parts.append(f"{run.llm_latency_ms / 1000:.1f}s")
    return " · ".join(parts)


def handle_command(session: ChatSession, line: str, out: Callable[[str], None]) -> bool:
    """Run one slash command. Returns False when the session should end."""
    name, _, arg = line[1:].strip().partition(" ")
    name, arg = name.lower(), arg.strip()

    if name in ("exit", "quit", "q"):
        return False
    if name in ("help", "h", "?"):
        out(HELP_TEXT)
    elif name == "mode":
        if not arg:
            for m in MODES.values():
                mark = "*" if m.name == session.mode.name else " "
                out(f" {mark} {m.name:<13} {m.description}")
        else:
            try:
                session.set_mode(arg)
                out(f"mode: {session.mode.name} — {session.mode.description}")
            except ValueError as exc:
                out(f"error: {exc}")
    elif name == "model":
        if not arg:
            out(f"model: {session.model}")
        else:
            try:
                session.set_model(arg)
                out(f"model: {session.model}")
            except (ValueError, ImportError) as exc:
                out(f"error: {exc}")
    elif name == "tokens":
        t = session.totals
        out(f"tasks: {t.tasks}   LLM calls: {t.llm_calls}   tool calls: {t.tool_calls}")
        out(f"input: {format_tokens(t.input_tokens)}   output: {format_tokens(t.output_tokens)}"
            f"   total: {format_tokens(t.input_tokens + t.output_tokens)}")
        if t.cache_reported and t.input_tokens:
            rate = t.cached_input_tokens / t.input_tokens
            out(f"cached input: {format_tokens(t.cached_input_tokens)} ({rate:.0%})"
                f"   uncached input: {format_tokens(t.input_tokens - t.cached_input_tokens)}")
        else:
            out("cached input: not reported by the provider")
    elif name == "context":
        s = session.context_stats()
        out(f"strategy: {s['strategy']}   messages in history: {s['messages']}"
            f"   history size: ~{format_tokens(s['history_tokens_est'])} tokens")
        if s["last_sent_tokens_est"] is not None:
            out(f"last request: {s['last_sent_messages']} messages, "
                f"~{format_tokens(s['last_sent_tokens_est'])} tokens "
                f"({s['compressed']} compressed, {s['dropped']} dropped)")
        if s["compactions"] is not None:
            out(f"compactions so far: {s['compactions']}")
    elif name == "tools":
        last = session.last_run.last_exposed_tools if session.last_run else []
        out(f"offered last request: {', '.join(last) if last else '(no request yet)'}")
        out(f"all tools: {', '.join(session.registry.list_names())}")
    elif name == "compact":
        if session.request_compaction():
            out("context will be compacted on the next request")
        else:
            out("/compact needs a mode with cache-friendly context (smart)")
    elif name == "diff":
        text = session.diff(whole_session=(arg == "all"))
        out(text.rstrip() if text else "no file changes")
    elif name == "undo":
        changes = session.undo()
        out(f"undone: {changes.summary()}" if changes else "nothing to undo")
    elif name == "clear":
        session.clear()
        out("started a fresh conversation")
    elif name == "workspace":
        if not arg:
            out(f"workspace: {session.workspace}")
        else:
            try:
                session.set_workspace(arg)
                out(f"workspace: {session.workspace} (fresh conversation)")
            except ValueError as exc:
                out(f"error: {exc}")
    else:
        out(f"unknown command /{name} — type /help")
    return True


def run_repl(
    session: ChatSession,
    *,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    out(f"FORGE chat · {session.model} · mode: {session.mode.name}")
    out(f"workspace: {session.workspace}")
    out("Type a task, or /help for commands.\n")
    while True:
        try:
            line = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(session, line, out):
                return 0
            continue
        try:
            run, changes = session.send(line)
        except KeyboardInterrupt:
            out("\n(interrupted)")
            continue
        out("")
        out(run.final_answer.strip() if run.final_answer else f"(no answer: {run.error or run.status})")
        if changes.any:
            out(f"files: {changes.summary()}  (/diff to view, /undo to revert)")
        out(result_line(run))
        out("")
