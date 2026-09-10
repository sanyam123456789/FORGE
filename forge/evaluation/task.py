"""
forge.evaluation.task — the evaluation-task representation.

An ``EvalTask`` is a *reusable* description of one piece of work: a stable
``task_id``, the natural-language ``prompt`` handed to the agent, and an
optional list of ``ArtifactCheck``s used to decide whether the produced
workspace actually satisfies the task.

The same ``EvalTask`` is meant to be executed many times under different
``ExperimentConfig``s (different tool / context strategies).  Nothing about a
task changes between those runs — task identity is exactly ``task_id`` — so
results stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactCheck:
    """A lightweight, deterministic check on a file the task should produce.

    path:
        Workspace-relative path to inspect.
    must_exist:
        When True (default) the file is required to exist.  When False the
        file is required to be *absent*.
    must_contain:
        Substrings that must all be present in the file's text (only checked
        when ``must_exist`` is True and the file is readable as UTF-8).

    These checks are intentionally simple string/existence assertions — not a
    test runner.  They exist so ``EvaluationRunner`` can turn "did the agent
    complete" into "did the agent produce the right artefact" without any
    provider involvement.
    """

    path: str
    must_exist: bool = True
    must_contain: tuple[str, ...] = ()

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

        if not self.must_contain:
            return True, f"{self.path}: present"

        try:
            text = target.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            return False, f"{self.path}: unreadable ({exc})"

        missing = [s for s in self.must_contain if s not in text]
        if missing:
            return False, f"{self.path}: missing substring(s) {missing}"
        return True, f"{self.path}: present with {len(self.must_contain)} substring(s)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "must_exist": self.must_exist,
            "must_contain": list(self.must_contain),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactCheck":
        return cls(
            path=data["path"],
            must_exist=bool(data.get("must_exist", True)),
            must_contain=tuple(data.get("must_contain", ()) or ()),
        )


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
        is never derived from the strategy or the run.
    prompt:
        The natural-language task description passed straight to the agent as
        its user task.
    checks:
        Optional artefact checks used to decide task success.  Empty means
        success is "the run reached a completed status".
    metadata:
        Free-form, non-identifying annotations (source, difficulty tag, ...).
        Recorded with results but never used for control or comparison keys.
    """

    task_id: str
    prompt: str
    checks: tuple[ArtifactCheck, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_id.strip():
            raise ValueError("EvalTask.task_id must be a non-empty string")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("EvalTask.prompt must be a non-empty string")

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
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTask":
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            checks=tuple(
                ArtifactCheck.from_dict(c) for c in data.get("checks", []) or []
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )
