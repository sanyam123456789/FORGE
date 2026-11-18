"""
forge.builtin_tools — the minimal built-in coding toolset.

Five tools, each a small ``Tool`` subclass registered through the standard
``ToolRegistry``:

    read_file       read a UTF-8 text file inside the workspace
    write_file      create / overwrite a text file inside the workspace
    edit_file       replace an exact substring in an existing file
    list_directory  list the entries of a directory inside the workspace
    run_shell       run a non-interactive shell command inside the workspace

Every tool:
- declares a JSON-Schema for its parameters,
- validates its inputs and returns a structured ``ToolResult`` (never raises),
- routes all filesystem / shell access through ``forge.safety`` guards,
- reports useful metadata for observability.

The safety layer is a lightweight heuristic (workspace-boundary checks plus a
command blocklist), **not** a sandbox — see ``forge/safety.py``.
"""

from __future__ import annotations

import itertools
import re
import subprocess
from abc import abstractmethod
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.logging import get_logger
from forge.safety import (
    SafetyError,
    assert_safe_command,
    assert_safe_write,
    assert_within_workspace,
    build_shell_env,
    get_workspace_root,
    is_secret_file,
    redact_secrets,
    secret_file_error,
)
from forge.tools import Tool, ToolRegistry, ToolResult

logger = get_logger(__name__)

_SHELL_TIMEOUT_DEFAULT = 30
_SHELL_TIMEOUT_MAX = 120

# read_file: per-call output caps. The char cap protects against minified
# one-line files that would slip through a line cap.
READ_FILE_MAX_LINES = 400
READ_FILE_MAX_CHARS = 40_000

# run_shell: a stream (stdout or stderr) above either threshold is shortened
# to its head + tail plus error-looking lines from the omitted middle.
SHELL_STREAM_MAX_LINES = 200
SHELL_STREAM_MAX_CHARS = 20_000
SHELL_HEAD_LINES = 40
SHELL_TAIL_LINES = 120
SHELL_MATCH_LINES = 30
# Char budgets that keep a shortened stream near SHELL_STREAM_MAX_CHARS even
# when its lines are very long.
_SHELL_HEAD_CHARS = 4_000
_SHELL_TAIL_CHARS = 12_000
_SHELL_LINE_CHARS = 2_000
_SHELL_MATCH_LINE_CHARS = 200
_SHELL_MATCH_RE = re.compile(
    r"error|exception|traceback|failed|failure|assert", re.IGNORECASE
)

#: Workspace-relative folder for full outputs of shortened shell commands.
FORGE_OUTPUTS_DIR = Path(".forge") / "outputs"
_output_counter = itertools.count(1)


# ---------------------------------------------------------------------------
# Argument helpers (raise ValueError -> converted to a structured ToolResult)
# ---------------------------------------------------------------------------


def _str_arg(kwargs: dict[str, Any], name: str, *, required: bool = True,
             default: str | None = None) -> str | None:
    if name not in kwargs or kwargs[name] is None:
        if required:
            raise ValueError(f"missing required argument '{name}'")
        return default
    value = kwargs[name]
    if not isinstance(value, str):
        raise ValueError(
            f"argument '{name}' must be a string, got {type(value).__name__}"
        )
    return value


def _bool_arg(kwargs: dict[str, Any], name: str, default: bool = False) -> bool:
    if name not in kwargs or kwargs[name] is None:
        return default
    value = kwargs[name]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _int_arg(kwargs: dict[str, Any], name: str, default: int) -> int:
    if name not in kwargs or kwargs[name] is None:
        return default
    value = kwargs[name]
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"argument '{name}' must be an integer")


def _opt_int_arg(kwargs: dict[str, Any], name: str) -> int | None:
    if name not in kwargs or kwargs[name] is None:
        return None
    value = kwargs[name]
    if isinstance(value, bool):
        raise ValueError(f"argument '{name}' must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"argument '{name}' must be an integer")


def _split_lines(text: str) -> list[str]:
    """Split on '\\n' only, keeping line endings, so ''.join() is lossless."""
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def _reject_unknown(kwargs: dict[str, Any], allowed: set[str]) -> None:
    extra = set(kwargs) - allowed
    if extra:
        raise ValueError(f"unexpected argument(s): {sorted(extra)}")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _BuiltinTool(Tool):
    """Shared plumbing: workspace binding + never-raise ``execute()``."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self._workspace_root = Path(workspace_root) if workspace_root else None

    # ``execute`` is the stable public contract; subclasses implement ``_run``.
    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return self._run(**kwargs)
        except SafetyError as exc:
            return ToolResult.from_error(f"safety violation: {exc}")
        except ValueError as exc:
            return ToolResult.from_error(f"invalid arguments: {exc}")
        except Exception as exc:  # noqa: BLE001 - tools must never propagate
            logger.exception("Unexpected error in tool %s", self.name)
            return ToolResult.from_error(f"{type(exc).__name__}: {exc}")

    @abstractmethod
    def _run(self, **kwargs: Any) -> ToolResult: ...

    # -- helpers -----------------------------------------------------------

    def _root(self) -> Path:
        return (self._workspace_root or get_workspace_root()).resolve()

    def _rel(self, resolved: Path) -> str:
        try:
            return str(resolved.relative_to(self._root()))
        except ValueError:
            return str(resolved)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class ReadFileTool(_BuiltinTool):
    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        max_lines: int = READ_FILE_MAX_LINES,
        max_chars: int = READ_FILE_MAX_CHARS,
    ) -> None:
        super().__init__(workspace_root)
        if max_lines < 1 or max_chars < 1:
            raise ValueError("max_lines and max_chars must be >= 1")
        self._max_lines = max_lines
        self._max_chars = max_chars

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a UTF-8 text file from the workspace (path relative to the "
            f"workspace root). Returns at most {self._max_lines} lines / "
            f"{self._max_chars} chars per call; longer files start with a "
            "header saying how to read more. Use start_line/end_line "
            "(1-based, inclusive) to read a specific part."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to the file to read.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to return (1-based, default 1).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to return (inclusive, default end of file).",
                },
            },
            "required": ["path"],
        }

    def _run(self, **kwargs: Any) -> ToolResult:
        _reject_unknown(kwargs, {"path", "start_line", "end_line"})
        path = _str_arg(kwargs, "path")
        start_line = _opt_int_arg(kwargs, "start_line")
        end_line = _opt_int_arg(kwargs, "end_line")
        if start_line is not None and start_line < 1:
            raise ValueError("start_line must be >= 1")
        if end_line is not None and end_line < (start_line or 1):
            raise ValueError("end_line must be >= start_line")

        resolved = assert_within_workspace(
            path, self._workspace_root, operation="read"
        )
        if is_secret_file(resolved):
            return ToolResult.from_error(secret_file_error(resolved))
        if not resolved.exists():
            return ToolResult.from_error(f"file not found: {self._rel(resolved)}")
        if resolved.is_dir():
            return ToolResult.from_error(
                f"path is a directory, not a file: {self._rel(resolved)}"
            )
        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.from_error(
                f"file is not valid UTF-8 text: {self._rel(resolved)}"
            )

        lines = _split_lines(text)
        total = len(lines)
        ranged = start_line is not None or end_line is not None
        if not ranged and total <= self._max_lines and len(text) <= self._max_chars:
            # Whole file fits: output is exactly the file, as before.
            output, first, last, truncated = text, 1, total, False
        else:
            first = start_line or 1
            if first > total:
                return ToolResult.from_error(
                    f"start_line {first} is past the end of "
                    f"{self._rel(resolved)} ({total} lines total)"
                )
            stop = min(end_line or total, total)
            output, last, truncated = self._window(lines, first, stop, total)

        return ToolResult(
            output=redact_secrets(output),
            metadata={
                "path": self._rel(resolved),
                "bytes": len(text.encode("utf-8")),
                "lines": total,
                "total_lines": total,
                "start_line": first,
                "end_line": last,
                "truncated": truncated,
            },
        )

    def _window(
        self, lines: list[str], first: int, stop: int, total: int
    ) -> tuple[str, int, bool]:
        """Return (output, last_line_shown, truncated) for lines first..stop."""
        selected: list[str] = []
        chars = 0
        cut_note = ""
        for number in range(first, stop + 1):
            if len(selected) >= self._max_lines:
                break
            line = lines[number - 1]
            if chars + len(line) > self._max_chars:
                if not selected:
                    # One line longer than the whole cap (minified file).
                    selected.append(line[: self._max_chars])
                    cut_note = (
                        f"; line {number} cut at {self._max_chars} of "
                        f"{len(line)} chars"
                    )
                break
            selected.append(line)
            chars += len(line)

        last = first + len(selected) - 1
        truncated = last < stop or bool(cut_note)
        body = "".join(selected)
        if not truncated:
            return body, last, False
        header = f"[read_file: showing lines {first}-{last} of {total} total{cut_note}."
        if last < total:
            header += f" Call read_file with start_line={last + 1} to read more."
        header += "]"
        return f"{header}\n{body}", last, True


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class WriteFileTool(_BuiltinTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Create a new text file (or overwrite an existing one when "
            "overwrite=true) inside the workspace. Creates parent directories "
            "as needed."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Allow overwriting an existing file (default false).",
                },
            },
            "required": ["path", "content"],
        }

    def _run(self, **kwargs: Any) -> ToolResult:
        _reject_unknown(kwargs, {"path", "content", "overwrite"})
        path = _str_arg(kwargs, "path")
        content = _str_arg(kwargs, "content")
        overwrite = _bool_arg(kwargs, "overwrite", default=False)

        if is_secret_file(path):
            return ToolResult.from_error(secret_file_error(path))
        resolved = assert_safe_write(
            path, allow_overwrite=overwrite, workspace_root=self._workspace_root
        )
        existed = resolved.exists()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult(
            output=(
                f"{'Overwrote' if existed else 'Created'} {self._rel(resolved)} "
                f"({len(content.encode('utf-8'))} bytes)."
            ),
            metadata={
                "path": self._rel(resolved),
                "bytes_written": len(content.encode("utf-8")),
                "created": not existed,
                "overwritten": existed,
            },
        )


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class EditFileTool(_BuiltinTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Replace an exact substring in an existing workspace file. "
            "old_string must match exactly. If it occurs more than once, pass "
            "replace_all=true or extend old_string with surrounding context."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence (default false).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    def _run(self, **kwargs: Any) -> ToolResult:
        _reject_unknown(kwargs, {"path", "old_string", "new_string", "replace_all"})
        path = _str_arg(kwargs, "path")
        old_string = _str_arg(kwargs, "old_string")
        new_string = _str_arg(kwargs, "new_string")
        replace_all = _bool_arg(kwargs, "replace_all", default=False)

        resolved = assert_within_workspace(
            path, self._workspace_root, operation="edit"
        )
        if is_secret_file(resolved):
            return ToolResult.from_error(secret_file_error(resolved))
        if not resolved.exists():
            return ToolResult.from_error(f"file not found: {self._rel(resolved)}")
        if resolved.is_dir():
            return ToolResult.from_error(
                f"path is a directory, not a file: {self._rel(resolved)}"
            )
        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.from_error(
                f"file is not valid UTF-8 text: {self._rel(resolved)}"
            )

        occurrences = text.count(old_string)
        if occurrences == 0:
            return ToolResult.from_error("old_string not found in file")
        if occurrences > 1 and not replace_all:
            return ToolResult.from_error(
                f"old_string is not unique ({occurrences} matches); pass "
                "replace_all=true or add surrounding context"
            )

        count = occurrences if replace_all else 1
        new_text = text.replace(old_string, new_string, -1 if replace_all else 1)
        resolved.write_text(new_text, encoding="utf-8")
        return ToolResult(
            output=(
                f"Edited {self._rel(resolved)}: {count} replacement(s), "
                f"{len(new_text.encode('utf-8'))} bytes."
            ),
            metadata={
                "path": self._rel(resolved),
                "replacements": count,
                "bytes_written": len(new_text.encode("utf-8")),
            },
        )


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


class ListDirectoryTool(_BuiltinTool):
    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "List the immediate entries of a directory inside the workspace. "
            "Directories are suffixed with '/'. Defaults to the workspace root."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory (default '.').",
                }
            },
            "required": [],
        }

    def _run(self, **kwargs: Any) -> ToolResult:
        _reject_unknown(kwargs, {"path"})
        path = _str_arg(kwargs, "path", required=False, default=".") or "."
        resolved = assert_within_workspace(
            path, self._workspace_root, operation="list"
        )
        if not resolved.exists():
            return ToolResult.from_error(f"directory not found: {self._rel(resolved)}")
        if not resolved.is_dir():
            return ToolResult.from_error(
                f"path is not a directory: {self._rel(resolved)}"
            )

        entries = sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines: list[str] = []
        dirs = files = 0
        for entry in entries:
            if entry.is_dir():
                dirs += 1
                lines.append(f"{entry.name}/")
            else:
                files += 1
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{entry.name}\t{size}")
        return ToolResult(
            output="\n".join(lines) if lines else "(empty directory)",
            metadata={
                "path": self._rel(resolved),
                "entries": len(entries),
                "dirs": dirs,
                "files": files,
            },
        )


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------


class RunShellTool(_BuiltinTool):
    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        env_allow: Iterable[str] | None = None,
    ) -> None:
        super().__init__(workspace_root)
        if env_allow is None:
            from forge.config import settings  # local import: avoid cycles

            env_allow = settings.shell_env_allow
        self._env_allow = tuple(env_allow)

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "Run a non-interactive shell command from the workspace root. The "
            "command is split into argv (no shell interpretation). A blocklist "
            "rejects destructive and network commands. Returns stdout, stderr "
            "and the exit code. API keys and other secrets are not available "
            "in the command's environment. Long output is shortened to its "
            "start, end and error lines; the full text is saved to a log file "
            "readable with read_file."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command line to run, e.g. 'python -m pytest -q'.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Seconds before the command is killed "
                        f"(default {_SHELL_TIMEOUT_DEFAULT}, max {_SHELL_TIMEOUT_MAX})."
                    ),
                },
            },
            "required": ["command"],
        }

    def _run(self, **kwargs: Any) -> ToolResult:
        _reject_unknown(kwargs, {"command", "timeout"})
        command = _str_arg(kwargs, "command")
        timeout = _int_arg(kwargs, "timeout", _SHELL_TIMEOUT_DEFAULT)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        timeout = min(timeout, _SHELL_TIMEOUT_MAX)

        # assert_safe_command raises SafetyError for blocked / empty / unparseable
        # commands; that is caught by execute() and returned as a structured error.
        tokens = assert_safe_command(command)

        cwd = self._root()
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, shell=False
                tokens,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                env=build_shell_env(self._env_allow),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.from_error(
                f"command timed out after {timeout}s: {command}"
            )
        except FileNotFoundError:
            return ToolResult.from_error(f"executable not found: {tokens[0]}")

        stdout = redact_secrets(proc.stdout or "")
        stderr = redact_secrets(proc.stderr or "")
        out_text = stdout.rstrip("\n")
        err_text = stderr.rstrip("\n")
        exit_line = f"[exit code {proc.returncode}]"

        truncated = _needs_cap(out_text) or _needs_cap(err_text)
        full_output_path: str | None = None
        omitted_lines = 0
        if truncated:
            full_output_path = self._save_full_output(
                _join_shell_output(out_text, err_text, exit_line)
            )
            if _needs_cap(out_text):
                out_text, n = _shorten_stream(out_text, full_output_path)
                omitted_lines += n
            if _needs_cap(err_text):
                err_text, n = _shorten_stream(err_text, full_output_path)
                omitted_lines += n

        metadata: dict[str, Any] = {
            "command": command,
            "exit_code": proc.returncode,
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
            "truncated": truncated,
            "omitted_lines": omitted_lines,
        }
        if full_output_path is not None:
            metadata["full_output_path"] = full_output_path
        return ToolResult(
            output=_join_shell_output(out_text, err_text, exit_line),
            success=proc.returncode == 0,
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            metadata=metadata,
        )

    def _save_full_output(self, content: str) -> str | None:
        """Write *content* under .forge/outputs/; return its workspace path."""
        root = self._root()
        out_dir = root / FORGE_OUTPUTS_DIR
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"shell_{stamp}_{next(_output_counter)}.log"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            gitignore = root / ".forge" / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text("*\n", encoding="utf-8")
            (out_dir / name).write_text(content, encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not save full shell output: %s", exc)
            return None
        return (FORGE_OUTPUTS_DIR / name).as_posix()


def _join_shell_output(stdout: str, stderr: str, exit_line: str) -> str:
    """Assemble run_shell text; the exit-code line is always last."""
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append("[stderr]\n" + stderr)
    parts.append(exit_line)
    return "\n".join(parts)


def _needs_cap(stream: str) -> bool:
    if not stream:
        return False
    return (
        len(stream) > SHELL_STREAM_MAX_CHARS
        or stream.count("\n") + 1 > SHELL_STREAM_MAX_LINES
    )


def _clip(line: str, limit: int) -> str:
    if len(line) <= limit:
        return line
    return f"{line[:limit]} [... {len(line) - limit} chars cut]"


def _shorten_stream(stream: str, full_output_path: str | None) -> tuple[str, int]:
    """Keep head + tail + error-looking middle lines; return (text, omitted)."""
    lines = stream.split("\n")

    head: list[str] = []
    used = 0
    for line in lines[:SHELL_HEAD_LINES]:
        line = _clip(line, _SHELL_LINE_CHARS)
        if head and used + len(line) > _SHELL_HEAD_CHARS:
            break
        head.append(line)
        used += len(line) + 1

    rest = lines[len(head):]
    tail: list[str] = []
    used = 0
    for line in reversed(rest[-SHELL_TAIL_LINES:] if rest else []):
        line = _clip(line, _SHELL_LINE_CHARS)
        if tail and used + len(line) > _SHELL_TAIL_CHARS:
            break
        tail.append(line)
        used += len(line) + 1
    tail.reverse()

    middle_start = len(head)
    middle = lines[middle_start: len(lines) - len(tail)]
    matches: list[str] = []
    for number, line in enumerate(middle, start=middle_start + 1):
        if len(matches) >= SHELL_MATCH_LINES:
            break
        if _SHELL_MATCH_RE.search(line):
            matches.append(f"L{number}: {_clip(line, _SHELL_MATCH_LINE_CHARS)}")

    omitted = len(middle) - len(matches)
    where = (
        f"Full output saved to {full_output_path} — use read_file on it if needed"
        if full_output_path
        else "Full output could not be saved"
    )
    removed = f"{omitted} lines omitted" if omitted else "long lines shortened"
    marker = f"[... {removed}. {where} ...]"
    return "\n".join(head + [marker] + matches + tail), omitted


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

BUILTIN_TOOL_CLASSES: tuple[type[_BuiltinTool], ...] = (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirectoryTool,
    RunShellTool,
)


def register_builtin_tools(
    registry: ToolRegistry, *, workspace_root: str | Path | None = None
) -> ToolRegistry:
    """Register the five built-in tools on *registry* and return it."""
    for cls in BUILTIN_TOOL_CLASSES:
        registry.register(cls(workspace_root=workspace_root))
    return registry


def build_default_registry(
    *, workspace_root: str | Path | None = None
) -> ToolRegistry:
    """Return a fresh ``ToolRegistry`` populated with the built-in tools."""
    return register_builtin_tools(ToolRegistry(), workspace_root=workspace_root)
