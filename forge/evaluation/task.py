"""
forge.evaluation.task — the evaluation-task representation.

An ``EvalTask`` is a *reusable* description of one piece of work: a stable
``task_id``, the natural-language ``prompt`` handed to the agent, an optional
list of ``ArtifactCheck``s used to decide whether the produced workspace
satisfies the task, and optionally a set of ``FixtureFile``s (starter files
provisioned into the workspace before the agent runs).

The same ``EvalTask`` is meant to be executed many times under different
``ExperimentConfig``s (different tool / context strategies).  Nothing about a
task changes between those runs — task identity is exactly ``task_id`` — so
results stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _is_safe_relative_path(path: str) -> bool:
    """True if *path* is a non-empty, workspace-relative path with no ``..``.

    Rejects absolute paths (POSIX or Windows), drive letters, and any
    ``..`` / empty segment.  Used to keep task-defined check and fixture
    paths inside the run workspace.
    """
    if not isinstance(path, str) or not path.strip():
        return False
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return False
    parts = PurePosixPath(path.replace("\\", "/")).parts
    return bool(parts) and all(p not in ("", "..") for p in parts)


@dataclass(frozen=True)
class ArtifactCheck:
    """A lightweight, deterministic check on a file the task should produce.

    path:
        Workspace-relative path to inspect.
    must_exist:
        When True (default) the file is required to exist.  When False the
        file is required to be *absent* (all substring lists are ignored).
    must_contain:
        Substrings that must **all** be present in the file's text.
    must_contain_any:
        Substrings of which **at least one** must be present.  Useful when a
        correct answer can be written more than one way (e.g. a bug fix).
    must_not_contain:
        Substrings that must **not** appear (e.g. the buggy fragment a fix
        was meant to remove).

    These are intentionally simple string/existence assertions — not a test
    runner.  They let ``EvaluationRunner`` turn "did the agent complete" into
    "did the agent produce the right artefact" with no provider involvement.
    """

    path: str
    must_exist: bool = True
    must_contain: tuple[str, ...] = ()
    must_contain_any: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()

    def evaluate(self, workspace: Path) -> tuple[bool, str]:
        """Return ``(passed, detail)`` for this check against *workspace*."""
        target = Path(workspace) / self.path
        exists = target.is_file()

        if not self.must_exist:
            if exists:
                return False, f"{self.path}: expected absent, but it exists"
            return True, f"{self.path}: absent as expected"

        if not exists:
            return False, f"{self.path}: expected file, not found"

        needs_text = bool(
            self.must_contain or self.must_contain_any or self.must_not_contain
        )
        if not needs_text:
            return True, f"{self.path}: present"

        try:
            text = target.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            return False, f"{self.path}: unreadable ({exc})"

        missing = [s for s in self.must_contain if s not in text]
        if missing:
            return False, f"{self.path}: missing required substring(s) {missing}"

        if self.must_contain_any and not any(s in text for s in self.must_contain_any):
            return (
                False,
                f"{self.path}: none of the alternatives present "
                f"{list(self.must_contain_any)}",
            )

        forbidden = [s for s in self.must_not_contain if s in text]
        if forbidden:
            return False, f"{self.path}: forbidden substring(s) present {forbidden}"

        return True, f"{self.path}: content checks passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "must_exist": self.must_exist,
            "must_contain": list(self.must_contain),
            "must_contain_any": list(self.must_contain_any),
            "must_not_contain": list(self.must_not_contain),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactCheck":
        if "path" not in data:
            raise KeyError("ArtifactCheck requires a 'path'")
        return cls(
            path=data["path"],
            must_exist=bool(data.get("must_exist", True)),
            must_contain=tuple(data.get("must_contain", ()) or ()),
            must_contain_any=tuple(data.get("must_contain_any", ()) or ()),
            must_not_contain=tuple(data.get("must_not_contain", ()) or ()),
        )


@dataclass(frozen=True)
class FixtureFile:
    """A starter file written into the run workspace before the agent starts.

    path:
        Workspace-relative destination (validated: no ``..``, not absolute).
    content:
        Literal UTF-8 text to write.
    """

    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixtureFile":
        if "path" not in data or "content" not in data:
            raise KeyError("FixtureFile requires 'path' and 'content'")
        return cls(path=data["path"], content=data["content"])


@dataclass(frozen=True)
class CheckOutcome:
    """Aggregate result of running every ``ArtifactCheck`` for a task."""

    #: ``None`` when the task defined no checks (so success rests on run status
    #: alone); otherwise True only if *every* check passed.
    passed: bool | None
    details: tuple[str, ...] = ()
    checks_run: int = 0

    @property
    def had_checks(self) -> bool:
        return self.checks_run > 0


@dataclass(frozen=True)
class EvalTask:
    """A single evaluation task, executable repeatedly under any strategy.

    task_id:
        Stable identity.  Two runs of "the same task" share this exactly; it
        is never derived from the strategy, provider, model, or run id.
    prompt:
        The natural-language task description passed straight to the agent as
        its user task.
    checks:
        Optional artefact checks used to decide task success.  Empty means
        success is "the run reached a completed status".
    fixtures:
        Optional starter files provisioned into the workspace before the run.
    metadata:
        Free-form, non-identifying annotations (category, difficulty, ...).
        Recorded with results but never used for control or comparison keys.
    """

    task_id: str
    prompt: str
    checks: tuple[ArtifactCheck, ...] = ()
    fixtures: tuple[FixtureFile, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_id.strip():
            raise ValueError("EvalTask.task_id must be a non-empty string")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("EvalTask.prompt must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("EvalTask.metadata must be an object / dict")
        for chk in self.checks:
            if not _is_safe_relative_path(chk.path):
                raise ValueError(
                    f"EvalTask {self.task_id!r}: unsafe check path {chk.path!r}"
                )
        for fx in self.fixtures:
            if not _is_safe_relative_path(fx.path):
                raise ValueError(
                    f"EvalTask {self.task_id!r}: unsafe fixture path {fx.path!r}"
                )

    # -- execution helpers ---------------------------------------------

    def provision(self, workspace: str | Path) -> list[str]:
        """Write every ``FixtureFile`` into *workspace*; return the paths written."""
        ws = Path(workspace)
        written: list[str] = []
        for fx in self.fixtures:
            if not _is_safe_relative_path(fx.path):  # defensive: never escape ws
                raise ValueError(f"unsafe fixture path {fx.path!r}")
            dest = ws / fx.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(fx.content, encoding="utf-8")
            written.append(fx.path)
        return written

    def run_checks(self, workspace: str | Path) -> CheckOutcome:
        """Evaluate every ``ArtifactCheck`` against *workspace*."""
        if not self.checks:
            return CheckOutcome(passed=None, details=(), checks_run=0)

        ws = Path(workspace)
        results = [check.evaluate(ws) for check in self.checks]
        details = tuple(detail for _, detail in results)
        return CheckOutcome(
            passed=all(ok for ok, _ in results),
            details=details,
            checks_run=len(results),
        )

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "checks": [c.to_dict() for c in self.checks],
            "fixtures": [f.to_dict() for f in self.fixtures],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTask":
        raw_fixtures = data.get("fixtures", []) or []
        if isinstance(raw_fixtures, dict):
            raise ValueError(
                "fixtures given as a mapping (e.g. {'dir': ...}); load the task "
                "via forge.evaluation.suite.load_task, which resolves it from disk"
            )
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            checks=tuple(
                ArtifactCheck.from_dict(c) for c in data.get("checks", []) or []
            ),
            fixtures=tuple(FixtureFile.from_dict(f) for f in raw_fixtures),
            metadata=dict(data.get("metadata", {}) or {}),
        )
