"""
forge.safety — Workspace boundary enforcement.

All filesystem operations performed by FORGE tools MUST call the safety
utilities here before touching any path.  This module establishes the
security contract that every future tool implementation must respect.

Design Goals (Step 1)
---------------------
- Define a clear workspace root that constrains all agent file operations.
- Provide simple, composable path-safety predicates and guards.
- Raise explicit, descriptive errors when a path violation is detected.
- Enumerate the categories of operations that are restricted, so future
  tool authors know exactly what checks to apply.

What is NOT implemented here yet
---------------------------------
- Subprocess sandboxing / seccomp / chroot.
- Network-egress restrictions.
- Rate limiting.
- Permission-level enforcement (read-only vs read-write zones).

These will be added incrementally as the tool suite grows.

Safety Contract
---------------
Every function in forge/tools/ that touches the filesystem MUST call
``assert_within_workspace(path)`` before performing the operation.

Every function that executes a shell command MUST call
``assert_safe_command(cmd)`` before executing it.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from forge.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Workspace root resolution
# ---------------------------------------------------------------------------


def get_workspace_root() -> Path:
    """Return the resolved, absolute workspace root.

    Priority:
    1. FORGE_WORKSPACE_ROOT environment variable (if set and non-empty).
    2. Current working directory at import time.

    The returned path is always resolved to an absolute real path, which
    eliminates symlink ambiguity.
    """
    from forge.config import settings  # local import to avoid circular deps

    root = settings.workspace_root.resolve()
    if not root.exists():
        raise SafetyError(
            f"Workspace root does not exist: {root}. "
            "Set FORGE_WORKSPACE_ROOT to an existing directory."
        )
    if not root.is_dir():
        raise SafetyError(
            f"Workspace root is not a directory: {root}."
        )
    return root


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SafetyError(Exception):
    """Raised when an operation would violate FORGE's safety boundaries."""


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def is_within_workspace(path: Path | str, workspace_root: Path | None = None) -> bool:
    """Return True if *path* resolves to a location inside *workspace_root*.

    Parameters
    ----------
    path:
        The path to check (may be relative or absolute).
    workspace_root:
        If None, the FORGE configured workspace root is used.

    Notes
    -----
    - Symlinks are resolved before comparison so that a symlink pointing
      outside the workspace is correctly rejected.
    - Relative paths are resolved relative to the workspace root, not CWD.
    """
    if workspace_root is None:
        workspace_root = get_workspace_root()

    workspace_root = workspace_root.resolve()
    resolved = (workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    try:
        resolved.relative_to(workspace_root)
        return True
    except ValueError:
        return False


def assert_within_workspace(
    path: Path | str,
    workspace_root: Path | None = None,
    *,
    operation: str = "access",
) -> Path:
    """Assert that *path* is inside the workspace root; raise SafetyError otherwise.

    Parameters
    ----------
    path:
        Path to validate.
    workspace_root:
        Workspace root override (uses configured root if None).
    operation:
        Short description of the attempted operation (used in error messages).

    Returns
    -------
    Path
        The resolved absolute path (safe to use directly after this call).
    """
    if workspace_root is None:
        workspace_root = get_workspace_root()

    workspace_root = workspace_root.resolve()

    if not Path(path).is_absolute():
        resolved = (workspace_root / path).resolve()
    else:
        resolved = Path(path).resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        raise SafetyError(
            f"Safety violation: attempted to {operation} path outside workspace.\n"
            f"  Path:      {resolved}\n"
            f"  Workspace: {workspace_root}\n"
            "Only paths inside the workspace root are permitted."
        )

    logger.debug("Path safety check passed: %s (op=%s)", resolved, operation)
    return resolved


# ---------------------------------------------------------------------------
# Destructive-operation guard
# ---------------------------------------------------------------------------

# Operations in this set require an explicit safety acknowledgement
# (currently just a flag parameter — a more rigorous UX can be added later).
_DESTRUCTIVE_EXTENSIONS = {".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h"}


def assert_safe_write(
    path: Path | str,
    *,
    allow_overwrite: bool = False,
    workspace_root: Path | None = None,
) -> Path:
    """Assert that writing to *path* is permitted.

    Rules enforced:
    1. Path must be within workspace.
    2. If the file already exists, *allow_overwrite* must be True.

    Parameters
    ----------
    path:
        Target file path.
    allow_overwrite:
        Set to True to allow overwriting an existing file.
    workspace_root:
        Workspace root override.

    Returns
    -------
    Path
        Resolved safe path.
    """
    resolved = assert_within_workspace(path, workspace_root, operation="write")

    if resolved.exists() and not allow_overwrite:
        raise SafetyError(
            f"Safety violation: attempted to overwrite existing file without "
            f"allow_overwrite=True.\n  File: {resolved}"
        )

    return resolved


# ---------------------------------------------------------------------------
# Command execution guards
# ---------------------------------------------------------------------------

# Shell builtins / commands that are unconditionally blocked.
# This list will grow as the tool suite grows.
_BLOCKED_COMMANDS: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "del",
        "deltree",
        "format",
        "mkfs",
        "dd",
        "shred",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "curl",   # network egress — tool authors should use the LLM provider client
        "wget",
        "nc",
        "ncat",
        "netcat",
        "sudo",
        "su",
    }
)


def split_command(cmd: str) -> list[str]:
    """Split a command string into argv tokens.

    Uses POSIX rules on POSIX platforms.  On Windows, POSIX splitting would
    destroy backslash path separators (``C:\\Python\\python.exe`` ->
    ``C:Pythonpython.exe``), so we split in non-POSIX mode and then strip a
    single layer of matching surrounding quotes from each token.  This is a
    heuristic suitable for the run_shell tool — it is **not** a shell parser.
    """
    if os.name == "nt":
        try:
            raw = shlex.split(cmd, posix=False)
        except ValueError as exc:
            raise SafetyError(f"Could not parse command: {cmd!r}") from exc
        tokens: list[str] = []
        for tok in raw:
            if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
                tok = tok[1:-1]
            tokens.append(tok)
        return tokens
    try:
        return shlex.split(cmd)
    except ValueError as exc:
        raise SafetyError(f"Could not parse command: {cmd!r}") from exc


def assert_safe_command(cmd: str | list[str]) -> list[str]:
    """Check that *cmd* does not invoke a blocked command.

    This is a lightweight heuristic check — NOT a full sandbox.  It parses the
    command, inspects only ``argv[0]`` against a static blocklist, and returns
    the argv tokens.  It does not defend against shell wrappers
    (``sh -c "rm ..."``), interpreter one-liners, output redirection, etc.

    Parameters
    ----------
    cmd:
        Shell command as a string (will be split) or an already-split list.

    Returns
    -------
    list[str]
        The parsed command tokens.

    Raises
    ------
    SafetyError
        If the command is empty/unparseable or ``argv[0]`` is on the blocklist.
    """
    if isinstance(cmd, str):
        tokens = split_command(cmd)
    else:
        tokens = list(cmd)

    if not tokens:
        raise SafetyError("Empty command is not permitted.")

    executable = os.path.basename(tokens[0]).lower()
    # Strip a trailing Windows executable suffix (proper suffix removal, not
    # character-set stripping — ``rstrip('.exe')`` would mangle names like
    # ``npx`` -> ``np`` or ``tox`` -> ``to``).
    for _suffix in (".exe", ".bat", ".cmd", ".com"):
        if executable.endswith(_suffix):
            executable = executable[: -len(_suffix)]
            break
    if executable in _BLOCKED_COMMANDS:
        raise SafetyError(
            f"Safety violation: command '{executable}' is on the blocked list.\n"
            f"  Full command: {tokens}\n"
            "Add explicit tool implementations for destructive operations."
        )

    logger.debug("Command safety check passed: %s", tokens[0])
    return tokens


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------

SAFETY_BOUNDARIES = """
FORGE Safety Boundaries (Step 1)
=================================

1. FILESYSTEM — ALL file read/write/delete operations must be within
   the configured workspace root (FORGE_WORKSPACE_ROOT).

2. OVERWRITE GUARD — Overwriting an existing source file requires
   allow_overwrite=True to be passed explicitly.

3. COMMAND EXECUTION — A static blocklist prevents the most dangerous
   shell commands (rm, curl, sudo, etc.) from being executed directly.
   This is a heuristic, NOT a full sandbox.

4. SECRETS — API keys are never written to disk or logged. All secrets
   must arrive via environment variables.

5. NOT YET ENFORCED:
   - subprocess sandboxing / seccomp
   - network-egress restrictions
   - read-only zones within workspace
   - per-operation rate limiting
"""
