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

import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Any

from forge.logging import get_logger
from forge.safety import (
    SafetyError,
    assert_safe_command,
    assert_safe_write,
    assert_within_workspace,
    get_workspace_root,
)
from forge.tools import Tool, ToolRegistry, ToolResult

logger = get_logger(__name__)

_SHELL_TIMEOUT_DEFAULT = 30
_SHELL_TIMEOUT_MAX = 120


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
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a UTF-8 text file from the workspace and return its full "
            "contents. Path is relative to the workspace root."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to the file to read.",
                }
            },
            "required": ["path"],
        }

    def _run(self, **kwargs: Any) -> ToolResult:
        _reject_unknown(kwargs, {"path"})
        path = _str_arg(kwargs, "path")
        resolved = assert_within_workspace(
            path, self._workspace_root, operation="read"
        )
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
        return ToolResult(
            output=text,
            metadata={
                "path": self._rel(resolved),
                "bytes": len(text.encode("utf-8")),
                "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            },
        )


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
    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "Run a non-interactive shell command from the workspace root. The "
            "command is split into argv (no shell interpretation). A blocklist "
            "rejects destructive and network commands. Returns stdout, stderr "
            "and the exit code."
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
            )
        except subprocess.TimeoutExpired:
            return ToolResult.from_error(
                f"command timed out after {timeout}s: {command}"
            )
        except FileNotFoundError:
            return ToolResult.from_error(f"executable not found: {tokens[0]}")

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        parts = []
        if stdout:
            parts.append(stdout.rstrip("\n"))
        if stderr:
            parts.append("[stderr]\n" + stderr.rstrip("\n"))
        parts.append(f"[exit code {proc.returncode}]")
        return ToolResult(
            output="\n".join(parts),
            success=proc.returncode == 0,
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            metadata={
                "command": command,
                "exit_code": proc.returncode,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stderr_bytes": len(stderr.encode("utf-8")),
            },
        )


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
