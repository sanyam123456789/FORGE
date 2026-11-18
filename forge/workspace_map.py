"""
forge.workspace_map — a compact list of workspace files for the task prompt.

Exploring a workspace costs whole LLM rounds (list_directory, then another
list_directory, ...), and every round re-sends the conversation. A short
file list in the first message usually answers "what is here?" for a few
hundred tokens instead.

The map is deterministic (sorted, fixed limits) and skips the same tooling,
cache and dependency folders as ``search_files``, plus hidden folders.
Secret files (``.env``, keys) are skipped, so their names never reach the
model.
"""

from __future__ import annotations

import os
from pathlib import Path

from forge.builtin_tools import SEARCH_SKIP_DIRS
from forge.safety import is_secret_file

#: At most this many files are listed; the rest are summarised as a count.
MAP_MAX_FILES = 60


def build_workspace_map(root: str | Path, *, max_files: int = MAP_MAX_FILES) -> str:
    """Return a short file listing of *root*, or "" when it has no files."""
    root = Path(root)
    if not root.is_dir():
        return ""
    files: list[str] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SEARCH_SKIP_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if is_secret_file(path):
                continue
            total += 1
            if len(files) < max_files:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                rel = path.relative_to(root).as_posix()
                files.append(f"  {rel} ({_human(size)})")
    if not total:
        return ""
    lines = [f"Workspace files ({total}):", *files]
    if total > len(files):
        lines.append(f"  ... and {total - len(files)} more (use list_directory / search_files)")
    return "\n".join(lines)


def _human(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"
