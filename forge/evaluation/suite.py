"""
forge.evaluation.suite — the version-controlled baseline task suite + loader.

The task *data* lives in ``experiments/tasks/*.json`` (with any starter files
under ``experiments/tasks/fixtures/<task-id>/``).  This module only loads and
validates it into the Step 3 ``EvalTask`` representation — there is **no** new
task abstraction and **no** second runner.

A task's identity is its ``task_id`` and nothing else.  The same task is run
later under every tool/context strategy without duplicating its definition.

On-disk task file shape::

    {
      "task_id": "create-string-utils",
      "prompt":  "Create a file ...",
      "checks":  [ { "path": "string_utils.py", "must_contain": ["def slugify("] } ],
      "fixtures": { "dir": "fixtures/create-string-utils" },   # optional
      "metadata": { "category": "file_creation", "difficulty": "easy" }
    }

``fixtures`` may also be an inline list of ``{"path": ..., "content": ...}``
objects (the form produced by ``EvalTask.to_dict``); the ``{"dir": ...}`` form
is a convenience that reads real files from ``experiments/tasks/`` so starter
code stays readable and lintable.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge.evaluation.task import ArtifactCheck, EvalTask, FixtureFile

#: Default location of the baseline suite, relative to the repository root.
#: (When FORGE is installed as a wheel the ``experiments/`` tree is not
#: packaged; pass an explicit ``tasks_dir`` in that case.)
DEFAULT_TASKS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "tasks"


class TaskSuiteError(ValueError):
    """Raised when a task definition, fixture, or the suite as a whole is invalid."""


# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------


def _read_json_object(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskSuiteError(f"{path}: cannot read task file ({exc})") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskSuiteError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise TaskSuiteError(f"{path}: task file must contain a JSON object")
    return data


def _resolve_fixtures(path: Path, data: dict) -> tuple[FixtureFile, ...]:
    spec = data.get("fixtures")
    if spec is None:
        return ()

    if isinstance(spec, list):
        try:
            return tuple(FixtureFile.from_dict(f) for f in spec)
        except (KeyError, TypeError) as exc:
            raise TaskSuiteError(f"{path}: malformed inline fixtures ({exc})") from exc

    if isinstance(spec, dict) and "dir" in spec:
        fx_dir = (path.parent / spec["dir"]).resolve()
        if not fx_dir.is_dir():
            raise TaskSuiteError(f"{path}: fixture directory not found: {fx_dir}")
        files = sorted(p for p in fx_dir.rglob("*") if p.is_file())
        if not files:
            raise TaskSuiteError(f"{path}: fixture directory is empty: {fx_dir}")
        out: list[FixtureFile] = []
        for f in files:
            rel = f.relative_to(fx_dir).as_posix()
            try:
                out.append(
                    FixtureFile(path=rel, content=f.read_text(encoding="utf-8"))
                )
            except (UnicodeDecodeError, OSError) as exc:
                raise TaskSuiteError(
                    f"{path}: fixture {rel!r} is not readable UTF-8 text ({exc})"
                ) from exc
        return tuple(out)

    raise TaskSuiteError(
        f"{path}: 'fixtures' must be a list of {{path, content}} objects "
        "or a {'dir': <relative path>} object"
    )


def _build_task(path: Path, data: dict) -> EvalTask:
    if "task_id" not in data:
        raise TaskSuiteError(f"{path}: missing 'task_id'")
    if "prompt" not in data:
        raise TaskSuiteError(f"{path}: missing 'prompt'")

    checks_raw = data.get("checks", [])
    if not isinstance(checks_raw, list):
        raise TaskSuiteError(f"{path}: 'checks' must be a list")
    try:
        checks = tuple(ArtifactCheck.from_dict(c) for c in checks_raw)
    except (KeyError, TypeError, AttributeError) as exc:
        raise TaskSuiteError(f"{path}: malformed check ({exc})") from exc

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TaskSuiteError(f"{path}: 'metadata' must be an object")

    fixtures = _resolve_fixtures(path, data)

    try:
        return EvalTask(
            task_id=data["task_id"],
            prompt=data["prompt"],
            checks=checks,
            fixtures=fixtures,
            metadata=metadata,
        )
    except ValueError as exc:  # EvalTask.__post_init__ validation
        raise TaskSuiteError(f"{path}: {exc}") from exc


def load_task(path: str | Path) -> EvalTask:
    """Load and validate a single task JSON file into an ``EvalTask``."""
    path = Path(path)
    return _build_task(path, _read_json_object(path))


# ---------------------------------------------------------------------------
# Whole-suite loading
# ---------------------------------------------------------------------------


def load_suite(tasks_dir: str | Path = DEFAULT_TASKS_DIR) -> list[EvalTask]:
    """Load every ``*.json`` task in *tasks_dir* (sorted by filename).

    Raises ``TaskSuiteError`` if the directory is missing, contains no task
    files, any file is invalid, or two files declare the same ``task_id``.
    """
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        raise TaskSuiteError(f"task suite directory not found: {tasks_dir}")

    files = sorted(tasks_dir.glob("*.json"))
    if not files:
        raise TaskSuiteError(f"no '*.json' task files found in {tasks_dir}")

    tasks: list[EvalTask] = []
    seen: dict[str, str] = {}
    for f in files:
        task = load_task(f)
        if task.task_id in seen:
            raise TaskSuiteError(
                f"duplicate task_id {task.task_id!r} in {f.name} "
                f"(already defined in {seen[task.task_id]})"
            )
        seen[task.task_id] = f.name
        tasks.append(task)
    return tasks


def load_suite_map(tasks_dir: str | Path = DEFAULT_TASKS_DIR) -> dict[str, EvalTask]:
    """Return the suite as an ordered ``{task_id: EvalTask}`` mapping."""
    return {t.task_id: t for t in load_suite(tasks_dir)}


def list_task_ids(tasks_dir: str | Path = DEFAULT_TASKS_DIR) -> list[str]:
    """Return the ``task_id``s of the suite, in load order."""
    return [t.task_id for t in load_suite(tasks_dir)]


def get_task(
    task_id: str, *, tasks_dir: str | Path = DEFAULT_TASKS_DIR
) -> EvalTask:
    """Return one task from the suite by id, or raise ``TaskSuiteError``."""
    suite = load_suite_map(tasks_dir)
    if task_id not in suite:
        raise TaskSuiteError(
            f"unknown task_id {task_id!r}. Available: {sorted(suite)}"
        )
    return suite[task_id]
