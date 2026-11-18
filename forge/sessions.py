"""
forge.sessions — save and resume `forge chat` conversations.

After every task, `forge chat` writes the conversation to
``<log_dir>/sessions/<session_id>.json``. ``forge chat --resume`` picks up
the newest session for the current workspace (or a given id) with the full
message history, mode and token totals, so a conversation can continue
after FORGE was closed.

Not saved: the /undo history (it holds file snapshots) and the provider's
prompt cache, which starts cold on resume.

Session files hold the conversation as the model saw it, including file
contents the agent read, so they live next to the run traces (``logs/``,
git-ignored in this repository).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.llm import Message, ToolCall

SESSION_FORMAT_VERSION = 1


def sessions_dir(log_dir: str | Path) -> Path:
    return Path(log_dir) / "sessions"


def new_session_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# Message (de)serialisation
# ---------------------------------------------------------------------------


def message_to_dict(msg: Message) -> dict[str, Any]:
    data: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "provider_signature": call.provider_signature,
            }
            for call in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        data["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        data["name"] = msg.name
    if msg.metadata:
        data["metadata"] = msg.metadata
    return data


def message_from_dict(data: dict[str, Any]) -> Message:
    if not isinstance(data, dict) or not isinstance(data.get("role"), str):
        raise ValueError("session message must be an object with a 'role'")
    calls = [
        ToolCall(
            id=str(c.get("id") or ""),
            name=str(c.get("name") or ""),
            arguments=c.get("arguments") if c.get("arguments") is not None else {},
            provider_signature=c.get("provider_signature"),
        )
        for c in data.get("tool_calls") or []
        if isinstance(c, dict)
    ]
    metadata = data.get("metadata")
    return Message(
        role=data["role"],
        content=str(data.get("content") or ""),
        tool_calls=calls,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def save_session(path: str | Path, data: dict[str, Any], messages: list[Message]) -> Path:
    """Write one session file atomically. *data* holds the non-message fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first_task = next(
        (m.content for m in messages if m.role == "user"), ""
    )
    payload = {
        "version": SESSION_FORMAT_VERSION,
        **data,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        "title": _title(first_task),
        "messages": [message_to_dict(m) for m in messages],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def load_session(path: str | Path) -> tuple[dict[str, Any], list[Message]]:
    """Read a session file: ``(fields, messages)``. Raises ValueError if invalid."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read session file {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise ValueError(f"{path} is not a FORGE session file")
    version = payload.get("version")
    if version != SESSION_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported session format version {version!r}")
    messages = [message_from_dict(m) for m in payload["messages"]]
    fields = {k: v for k, v in payload.items() if k != "messages"}
    return fields, messages


def list_sessions(
    log_dir: str | Path, *, workspace: str | Path | None = None
) -> list[dict[str, Any]]:
    """Saved sessions, newest first; only those for *workspace* when given."""
    folder = sessions_dir(log_dir)
    if not folder.is_dir():
        return []
    want = _norm(workspace) if workspace is not None else None
    found: list[dict[str, Any]] = []
    for path in folder.glob("*.json"):
        try:
            fields, messages = load_session(path)
        except ValueError:
            continue
        if want is not None and _norm(fields.get("workspace")) != want:
            continue
        found.append(
            {
                "path": path,
                "session_id": fields.get("session_id") or path.stem,
                "updated_at": fields.get("updated_at") or "",
                "workspace": fields.get("workspace"),
                "title": fields.get("title") or "",
                "tasks": (fields.get("totals") or {}).get("tasks"),
            }
        )
    found.sort(key=lambda s: s["updated_at"], reverse=True)
    return found


def find_session(
    log_dir: str | Path, ref: str | None, *, workspace: str | Path
) -> Path | None:
    """Resolve ``--resume [ref]``: newest for *workspace*, or an id (prefix)."""
    if not ref or ref == "latest":
        sessions = list_sessions(log_dir, workspace=workspace)
        return sessions[0]["path"] if sessions else None
    exact = sessions_dir(log_dir) / f"{ref}.json"
    if exact.is_file():
        return exact
    matches = [s["path"] for s in list_sessions(log_dir) if s["session_id"].startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"session id '{ref}' is ambiguous ({len(matches)} matches)")
    return None


def _title(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= 70 else line[:67] + "..."


def _norm(path: Any) -> str | None:
    if not path:
        return None
    try:
        return os.path.normcase(str(Path(str(path)).resolve()))
    except OSError:
        return os.path.normcase(str(path))
