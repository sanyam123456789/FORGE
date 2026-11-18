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
    /approve [on|off]   ask before file edits and shell commands
    /sessions           saved conversations for this workspace
    /clear              start a fresh conversation
    /workspace [path]   show or change the folder FORGE works in
    /exit               quit

One provider object is kept for the whole conversation, so every request
carries the same session id — OpenCode Go uses it for prompt caching.

The conversation is saved after every task (``forge.sessions``), so
``forge chat --resume`` can continue it later. Reply text streams to the
terminal as it arrives.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.agent import AgentConfig, AgentRun, AgentRuntime
from forge.builtin_tools import SEARCH_SKIP_DIRS, build_default_registry
from forge.context import ConversationContext, StableContextStrategy
from forge.llm import LLMProvider, ToolCall, get_provider
from forge.modes import DEFAULT_MODE, MODES, Mode, build_strategies, get_mode
from forge.sessions import (
    list_sessions,
    load_session,
    new_session_id,
    save_session,
    sessions_dir,
)
from forge.logging import get_logger
from forge.tokens import estimate_messages_tokens

logger = get_logger(__name__)

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
        on_text: Callable[[str], None] | None = None,
        approve: Callable[[ToolCall], bool | str] | None = None,
        approval_on: bool = False,
        save_sessions: bool = False,
    ) -> None:
        self.settings = settings
        self.workspace = Path(workspace or settings.workspace_root).resolve()
        self.mode: Mode = get_mode(mode)
        self._provider = provider
        self._write_trace = write_trace
        self._on_tool_call = on_tool_call
        self._on_text = on_text
        # approve: asked before file edits / shell commands while approval_on.
        self.approve = approve
        self.approval_on = approval_on and approve is not None
        self.save_sessions = save_sessions
        self.totals = _Totals()
        self.last_run: AgentRun | None = None
        self._history: list[tuple[Snapshot, Snapshot]] = []  # (before, after) per task
        self._session_start: Snapshot | None = None
        self._pending_note: str | None = None
        self.session_id = new_session_id()
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
        self.totals = _Totals()
        self.last_run = None
        self._pending_note = None
        self.session_id = new_session_id()
        if self._provider is not None and hasattr(self._provider, "new_session"):
            self._provider.new_session()

    # -- saving and resuming -----------------------------------------------

    @property
    def session_path(self) -> Path:
        return sessions_dir(self.settings.log_dir) / f"{self.session_id}.json"

    def save(self) -> Path | None:
        """Write the conversation to its session file (if saving is on)."""
        if not self.save_sessions or not self.context.message_count:
            return None
        try:
            return save_session(
                self.session_path,
                {
                    "session_id": self.session_id,
                    "workspace": str(self.workspace),
                    "mode": self.mode.name,
                    "model": self.model,
                    "totals": dataclasses.asdict(self.totals),
                    "pending_note": self._pending_note,
                },
                self.context.get_messages(),
            )
        except OSError as exc:  # saving must never break the conversation
            logger.warning("could not save the chat session: %s", exc)
            return None

    def resume(self, path: str | Path) -> dict[str, Any]:
        """Load a saved session into this one. Returns the saved fields."""
        fields, messages = load_session(path)
        workspace = Path(fields.get("workspace") or self.workspace)
        if not workspace.is_dir():
            raise ValueError(f"the session's workspace no longer exists: {workspace}")
        self.workspace = workspace.resolve()
        mode = fields.get("mode")
        if mode in MODES:
            self.mode = get_mode(mode)
        self._reset_conversation()
        self.session_id = str(fields.get("session_id") or Path(path).stem)
        for message in messages:
            self.context.add_message(message)
        totals = fields.get("totals")
        if isinstance(totals, dict):
            known = {f.name for f in dataclasses.fields(_Totals)}
            self.totals = _Totals(**{k: v for k, v in totals.items() if k in known})
        note = fields.get("pending_note")
        self._pending_note = note if isinstance(note, str) and note else None
        return fields

    def saved_sessions(self) -> list[dict[str, Any]]:
        return list_sessions(self.settings.log_dir, workspace=self.workspace)

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
            on_text=self._on_text,
            approve=self.approve if self.approval_on else None,
        )
        run: AgentRun | None = None
        changes = FileChanges()
        try:
            run = runtime.run()  # Ctrl+C inside ends the run as "interrupted"
        finally:
            # Record file changes even if the run was cut short, so /diff
            # and /undo always cover what the agent did.
            after = self._snapshot()
            if before is not None and after is not None:
                changes = compare_snapshots(before, after)
                if changes.any:
                    self._history.append((before, after))
            if run is not None:
                self.last_run = run
                self.totals.add(run)
            self.save()
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
        self.save()
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
  /approve [on|off]   ask before file edits and shell commands
  /sessions           saved conversations for this workspace
  /clear              start a fresh conversation
  /workspace [path]   show or change the folder FORGE works in
  /exit               quit
Ctrl+C stops the current task; the conversation stays usable."""


def brief_args(args: dict) -> str:
    """Short one-line view of tool arguments for progress output."""
    for key in ("path", "command", "pattern", "name"):
        if isinstance(args.get(key), str):
            value = args[key].replace("\n", " ")
            return value if len(value) <= 70 else value[:67] + "..."
    return ""


class TerminalOutput:
    """Prints streamed reply text and tool-progress lines without mixing them.

    ``text`` receives streamed chunks; ``tool`` prints a progress line on a
    line of its own. ``streamed`` says whether any reply text was shown
    since the last ``reset`` (then the final answer is already on screen).
    """

    def __init__(self, write: Callable[[str], None] | None = None) -> None:
        self._write = write or _stdout_write
        self.at_line_start = True
        self.streamed = False

    def reset(self) -> None:
        self.streamed = False

    def text(self, chunk: str) -> None:
        if not chunk:
            return
        self._write(chunk)
        self.at_line_start = chunk.endswith("\n")
        self.streamed = True

    def line(self, text: str) -> None:
        if not self.at_line_start:
            self._write("\n")
        self._write(text + "\n")
        self.at_line_start = True

    def tool(self, name: str, args: dict[str, Any]) -> None:
        self.line(f"  · {name} {brief_args(args)}".rstrip())

    def end_reply(self) -> None:
        if not self.at_line_start:
            self._write("\n")
            self.at_line_start = True


def _stdout_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


_PREVIEW_LINES = 20


def describe_action(call: ToolCall) -> str:
    """What an approval prompt shows for a write_file / edit_file / run_shell call."""
    args = call.arguments if isinstance(call.arguments, dict) else {}
    if call.name == "run_shell":
        return f"  ? run_shell: {args.get('command', '')}"
    if call.name == "edit_file":
        old = str(args.get("old_string", "")).splitlines()
        new = str(args.get("new_string", "")).splitlines()
        lines = [f"  ? edit_file {args.get('path', '')}"]
        diff = [d for d in difflib.unified_diff(old, new, lineterm="", n=1)][2:]
        for d in diff[:_PREVIEW_LINES]:
            lines.append(f"    {d}")
        if len(diff) > _PREVIEW_LINES:
            lines.append(f"    ... {len(diff) - _PREVIEW_LINES} more lines")
        return "\n".join(lines)
    if call.name == "write_file":
        content = str(args.get("content", "")).splitlines()
        verb = "overwrite" if args.get("overwrite") else "create"
        lines = [f"  ? write_file ({verb}) {args.get('path', '')} — {len(content)} lines"]
        for d in content[:_PREVIEW_LINES]:
            lines.append(f"    + {d}")
        if len(content) > _PREVIEW_LINES:
            lines.append(f"    ... {len(content) - _PREVIEW_LINES} more lines")
        return "\n".join(lines)
    return f"  ? {call.name} {brief_args(args)}"


class TerminalApprover:
    """Asks in the terminal before file edits and shell commands.

    Answers: y = run it, n = don't, a = always allow this tool for the rest
    of the session, anything else = don't, and pass the text to the model
    as the user's instruction. Enter alone counts as "no".
    """

    PROMPT = "  allow? [y]es / [n]o / [a]lways for this tool / or say what to do instead: "

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] | None = None,
        out: Callable[[str], None] | None = None,
        before_prompt: Callable[[], None] | None = None,
    ) -> None:
        # Looked up at call time, so a replaced builtins.input/print is used.
        self._input = input_fn or (lambda prompt: input(prompt))
        self._out = out or (lambda text: print(text))
        self._before_prompt = before_prompt
        self.always: set[str] = set()

    def __call__(self, call: ToolCall) -> bool | str:
        if call.name in self.always:
            return True
        if self._before_prompt is not None:
            self._before_prompt()
        self._out(describe_action(call))
        try:
            answer = self._input(self.PROMPT).strip()
        except EOFError:
            return False
        low = answer.lower()
        if low in ("y", "yes"):
            return True
        if low in ("a", "always"):
            self.always.add(call.name)
            return True
        if low in ("", "n", "no"):
            return False
        return answer


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
    elif name == "approve":
        if arg in ("on", "off"):
            if arg == "on" and session.approve is None:
                out("approval prompts are not available here")
            else:
                session.approval_on = arg == "on"
        elif arg:
            out("usage: /approve [on|off]")
        out(f"approval before edits and commands: {'on' if session.approval_on else 'off'}")
    elif name == "sessions":
        saved = session.saved_sessions()
        if not saved:
            out("no saved sessions for this workspace")
        for entry in saved[:10]:
            mark = "*" if entry["session_id"] == session.session_id else " "
            tasks = entry["tasks"] if entry["tasks"] is not None else "?"
            out(f" {mark} {entry['session_id']}  {entry['updated_at'][:16].replace('T', ' ')}"
                f"  {tasks} task(s)  {entry['title']}")
        if saved:
            out("resume one with: forge chat --resume <id>")
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
    output: TerminalOutput | None = None,
) -> int:
    out(f"FORGE chat · {session.model} · mode: {session.mode.name}")
    out(f"workspace: {session.workspace}")
    if session.context.message_count:
        out(f"resumed session {session.session_id} ({session.totals.tasks} earlier task(s))")
    if session.approval_on:
        out("approval before edits and commands: on (/approve off to stop asking)")
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
        if output is not None:
            output.reset()
        try:
            run, changes = session.send(line)
        except KeyboardInterrupt:
            out("\n(interrupted)")
            continue
        if output is not None:
            output.end_reply()
        out("")
        if not (output is not None and output.streamed and run.final_answer):
            out(run.final_answer.strip() if run.final_answer
                else f"(no answer: {run.error or run.status})")
            out("")
        if changes.any:
            out(f"files: {changes.summary()}  (/diff to view, /undo to revert)")
        out(result_line(run))
        out("")
