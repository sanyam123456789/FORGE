"""
forge.evaluation.matrix — the formal 2x2 controlled experiment (Step 7).

Step 5 shipped Adaptive tool exposure; Step 6 shipped Managed context. Both
were individually selectable through ``ExperimentConfig``/``EvaluationRunner``
but nothing ran the four combinations together, under the same tasks, and
collected them into one comparable, reloadable output. That is exactly what
this module does — and *only* that:

    A = fixed_raw          B = fixed_managed
    C = adaptive_raw       D = adaptive_managed

``MatrixRunner`` is an orchestration layer *above* the existing
``EvaluationRunner`` (itself a thin wrapper around ``forge.agent.AgentRuntime``
— see ``forge/evaluation/runner.py``). It does not contain a second agent
loop and does not call a provider SDK directly: for every (task, arm) pair it
builds a fresh, isolated workspace and a fresh provider instance, then hands
off to ``EvaluationRunner.run()`` exactly as any other caller would.

Statistical analysis of this module's output is ``forge.evaluation.analysis``
(Step 8) — a separate, read-only consumer of ``results.jsonl``/
``metadata.json`` that does not change anything in this module. Dashboards
remain explicitly NOT implemented anywhere. See
``docs/step-07-controlled-2x2-experiment.md`` and
``docs/step-08-statistical-analysis.md``.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from forge.builtin_tools import build_default_registry
from forge.evaluation.experiment import (
    CONTEXT_STRATEGY_MANAGED,
    CONTEXT_STRATEGY_RAW,
    TOOL_STRATEGY_ADAPTIVE,
    TOOL_STRATEGY_FIXED,
    ExperimentConfig,
)
from forge.evaluation.result import EvalResult, TokenPricing, append_result_jsonl
from forge.evaluation.runner import EvaluationRunner
from forge.evaluation.task import EvalTask
from forge.llm import LLMProvider, get_provider
from forge.logging import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# The four arms — stable, machine-readable identifiers
# ---------------------------------------------------------------------------

ARM_FIXED_RAW = "fixed_raw"
ARM_FIXED_MANAGED = "fixed_managed"
ARM_ADAPTIVE_RAW = "adaptive_raw"
ARM_ADAPTIVE_MANAGED = "adaptive_managed"


@dataclass(frozen=True)
class ExperimentArm:
    """One cell of the 2x2: a stable id plus the two strategy labels it pins."""

    arm_id: str
    tool_strategy: str
    context_strategy: str


#: The four arms, in a fixed, documented, tested order. ``MatrixRunner.run()``
#: always iterates arms in this order (filtered to whatever subset the caller
#: asked for) regardless of the order a caller's ``arms=`` argument lists
#: them in — see ``arms_by_ids``. Do not reorder this tuple without updating
#: ``tests/test_matrix.py::test_stable_arm_ordering`` and the Step 7 doc.
ARMS: tuple[ExperimentArm, ...] = (
    ExperimentArm(ARM_FIXED_RAW, TOOL_STRATEGY_FIXED, CONTEXT_STRATEGY_RAW),
    ExperimentArm(ARM_FIXED_MANAGED, TOOL_STRATEGY_FIXED, CONTEXT_STRATEGY_MANAGED),
    ExperimentArm(ARM_ADAPTIVE_RAW, TOOL_STRATEGY_ADAPTIVE, CONTEXT_STRATEGY_RAW),
    ExperimentArm(ARM_ADAPTIVE_MANAGED, TOOL_STRATEGY_ADAPTIVE, CONTEXT_STRATEGY_MANAGED),
)

ARMS_BY_ID: dict[str, ExperimentArm] = {a.arm_id: a for a in ARMS}


def arms_by_ids(arm_ids: Sequence[str]) -> tuple[ExperimentArm, ...]:
    """Resolve *arm_ids* to ``ExperimentArm``s, in the canonical ``ARMS`` order.

    The canonical order is used regardless of the order *arm_ids* lists them
    in, so arm ordering is always deterministic and independent of caller
    input order. Raises ``ValueError`` naming any unrecognised id.
    """
    wanted = set(arm_ids)
    unknown = wanted - set(ARMS_BY_ID)
    if unknown:
        raise ValueError(
            f"unknown arm id(s) {sorted(unknown)}; known: {sorted(ARMS_BY_ID)}"
        )
    return tuple(a for a in ARMS if a.arm_id in wanted)


# ---------------------------------------------------------------------------
# Failure-handling categories
# ---------------------------------------------------------------------------

#: A task-arm execution that never produced an ``AgentRun`` at all — the
#: orchestration layer itself failed (workspace creation, fixture
#: provisioning, provider construction, ...) before/around the real agent
#: loop. Distinct from ``forge.agent.STATUS_PROVIDER_ERROR`` (the provider
#: call itself failed *inside* a real, otherwise-normal run) and from
#: ``forge.agent.STATUS_FAILED`` (an unexpected exception *inside* the loop,
#: already caught and recorded by ``AgentRuntime.run()``).
STATUS_RUNNER_ERROR = "runner_error"

#: ``AgentRun`` statuses (see ``forge/agent.py``) that represent a ceiling
#: being hit rather than the model choosing to stop — the closest existing
#: FORGE concept to "timeout/interruption" (see Limitations in the Step 7
#: doc: there is no wall-clock per-run timeout).
_LIMIT_STATUSES = frozenset({"max_turns_exceeded", "max_tool_calls_exceeded"})


def outcome_category(result: EvalResult) -> str:
    """Bucket *result* into one of six mutually-exclusive outcome categories.

    - ``"success"``            — completed, and checks (if any) passed.
    - ``"completed_incorrect"``— completed, but checks failed.
    - ``"runtime_error"``      — an unexpected exception inside the agent loop.
    - ``"provider_error"``     — the LLM provider call failed (incl. quota).
    - ``"limit_reached"``      — a turn/tool-call ceiling was hit.
    - ``"runner_error"``       — the orchestration layer itself failed before
      a normal ``AgentRun`` could be produced (see ``STATUS_RUNNER_ERROR``).

    Never collapses these into a single boolean — see ``EvalResult.status``
    and ``EvalResult.success`` for the raw fields this derives from.
    """
    if result.status == STATUS_RUNNER_ERROR:
        return "runner_error"
    if result.status == "provider_error":
        return "provider_error"
    if result.status in _LIMIT_STATUSES:
        return "limit_reached"
    if result.status == "failed":
        return "runtime_error"
    if result.status == "completed":
        return "success" if result.success else "completed_incorrect"
    return "unknown"  # defensive: a future AgentRun status not yet mapped here


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------


def _new_experiment_id() -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    return f"exp_{ts}_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class MatrixExperimentSummary:
    """Metadata describing one full run of the 2x2 matrix.

    Written as ``metadata.json`` alongside ``results.jsonl`` so a later
    (Step 8) analysis knows exactly what configuration produced the results
    without having to re-derive it from the rows.
    """

    experiment_id: str
    started_at: str
    finished_at: str
    provider: str
    model: str
    temperature: float
    max_output_tokens: int
    max_llm_turns: int | None
    max_tool_calls: int | None
    arm_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    output_dir: str
    results_file: str
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_llm_turns": self.max_llm_turns,
            "max_tool_calls": self.max_tool_calls,
            "arm_ids": list(self.arm_ids),
            "task_ids": list(self.task_ids),
            "output_dir": self.output_dir,
            "results_file": self.results_file,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatrixExperimentSummary":
        return cls(
            experiment_id=data["experiment_id"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            provider=data["provider"],
            model=data["model"],
            temperature=data["temperature"],
            max_output_tokens=data["max_output_tokens"],
            max_llm_turns=data.get("max_llm_turns"),
            max_tool_calls=data.get("max_tool_calls"),
            arm_ids=tuple(data.get("arm_ids", ())),
            task_ids=tuple(data.get("task_ids", ())),
            output_dir=data["output_dir"],
            results_file=data["results_file"],
            label=data.get("label"),
        )


def read_matrix_metadata(path: str | Path) -> MatrixExperimentSummary:
    """Reload a ``metadata.json`` file written by ``MatrixRunner.run()``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MatrixExperimentSummary.from_dict(data)


# ---------------------------------------------------------------------------
# The matrix runner
# ---------------------------------------------------------------------------


class MatrixRunner:
    """Runs every given task under every arm of the formal 2x2 experiment.

    Fairness controls (see ``docs/step-07-controlled-2x2-experiment.md`` §3
    for the full list): every arm for a given task gets the exact same
    ``EvalTask`` object (same prompt, same task_id, same fixtures, same
    checks), the same provider/model/temperature/max_output_tokens/
    max_llm_turns/max_tool_calls (all pinned once, from *settings*, for the
    whole experiment), and the same underlying tool registry (the five
    built-ins) — only the *exposed subset* of that registry and the context
    management policy vary, which is exactly the studied variable. Each
    (task, arm) pair gets its own fresh workspace directory and its own
    fresh provider instance, so nothing is inherited between arms or tasks.

    This class contains no agent loop of its own: every single task-arm
    execution is one ``EvaluationRunner.run()`` call, which is itself one
    ``forge.agent.AgentRuntime.run()`` call.
    """

    def __init__(
        self,
        *,
        settings: Any = None,
        provider_factory: Callable[[], LLMProvider] | None = None,
        output_root: str | Path | None = None,
        write_trace: bool = True,
    ) -> None:
        if settings is None:
            from forge.config import settings as _settings

            settings = _settings
        self._settings = settings
        # A *factory*, not a shared instance: each task-arm run gets its own
        # provider object, so nothing (scripted response queues included)
        # leaks between arms. Defaults to the real configured provider.
        self._provider_factory = provider_factory or (
            lambda: get_provider(settings_override=settings)
        )
        self._output_root = (
            Path(output_root) if output_root is not None else Path("runs") / "experiments"
        )
        self._write_trace = write_trace

    # ------------------------------------------------------------------

    def run(
        self,
        tasks: Sequence[EvalTask],
        *,
        arms: Sequence[ExperimentArm] = ARMS,
        experiment_id: str | None = None,
        label: str | None = None,
        pricing: TokenPricing | None = None,
        cleanup_workspaces: bool = False,
    ) -> tuple[list[EvalResult], MatrixExperimentSummary]:
        """Run *tasks* x *arms* and return ``(results, summary)``.

        Deterministic ordering: tasks are visited in the order given (the
        caller controls task selection/order — e.g. the suite loader's
        filename-sorted order); for each task, arms are visited in the fixed
        ``ARMS`` order (see ``arms_by_ids`` for subset selection that
        preserves this canonical order regardless of input order).

        Every (task, arm) pair gets its own workspace directory under
        ``<output_dir>/workspaces/<task_id>/<arm_id>/`` and its own provider
        instance (from ``provider_factory``) — never shared or reused.

        A task-arm execution that raises *any* exception before producing a
        normal ``EvalResult`` (workspace/provider/config problems — not a
        provider or in-loop error, both of which ``EvaluationRunner``
        already turns into a normal, non-raising ``EvalResult``) is caught
        here and recorded as a ``STATUS_RUNNER_ERROR`` result rather than
        aborting the rest of the experiment.
        """
        if not tasks:
            raise ValueError("MatrixRunner.run() requires at least one task")
        if not arms:
            raise ValueError("MatrixRunner.run() requires at least one arm")

        experiment_id = experiment_id or _new_experiment_id()
        exp_dir = self._output_root / experiment_id
        # exist_ok=False: never silently merge into / overwrite a previous
        # experiment's output directory.
        exp_dir.mkdir(parents=True, exist_ok=False)
        ws_root = exp_dir / "workspaces"
        trace_dir = exp_dir / "traces"
        results_file = exp_dir / "results.jsonl"
        metadata_file = exp_dir / "metadata.json"

        started_at = _utc_now_iso()
        results: list[EvalResult] = []

        for task in tasks:
            for arm in arms:
                result = self._run_one(
                    task=task,
                    arm=arm,
                    experiment_id=experiment_id,
                    ws_root=ws_root,
                    trace_dir=trace_dir,
                    pricing=pricing,
                    label=label,
                    cleanup_workspace=cleanup_workspaces,
                )
                results.append(result)
                # Append incrementally: a mid-experiment crash (e.g. Ctrl-C)
                # still leaves every already-completed task-arm result on
                # disk, reloadable, rather than losing the whole batch.
                append_result_jsonl(result, results_file)

        finished_at = _utc_now_iso()
        summary = MatrixExperimentSummary(
            experiment_id=experiment_id,
            started_at=started_at,
            finished_at=finished_at,
            provider=self._settings.llm_provider,
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_output_tokens,
            max_llm_turns=getattr(self._settings, "max_llm_turns", None),
            max_tool_calls=getattr(self._settings, "max_tool_calls", None),
            arm_ids=tuple(a.arm_id for a in arms),
            task_ids=tuple(t.task_id for t in tasks),
            output_dir=str(exp_dir),
            results_file=str(results_file),
            label=label,
        )
        metadata_file.write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Matrix experiment %s done: %d task(s) x %d arm(s) = %d result(s) -> %s",
            experiment_id, len(tasks), len(arms), len(results), exp_dir,
        )
        return results, summary

    # ------------------------------------------------------------------
    # One (task, arm) execution
    # ------------------------------------------------------------------

    def _run_one(
        self,
        *,
        task: EvalTask,
        arm: ExperimentArm,
        experiment_id: str,
        ws_root: Path,
        trace_dir: Path,
        pricing: TokenPricing | None,
        label: str | None,
        cleanup_workspace: bool,
    ) -> EvalResult:
        # Fresh, isolated workspace: <ws_root>/<task_id>/<arm_id>/ — never
        # shared with any other (task, arm) pair, never reused across arms.
        ws_path = ws_root / task.task_id / arm.arm_id
        experiment_cfg = ExperimentConfig.from_settings(
            self._settings,
            task_id=task.task_id,
            tool_strategy=arm.tool_strategy,
            context_strategy=arm.context_strategy,
            label=label,
        )
        try:
            ws_path.mkdir(parents=True, exist_ok=False)
            provider = self._provider_factory()
            exposed_tools = self._exposed_tools(experiment_cfg, task, ws_path)
            eval_runner = EvaluationRunner(
                settings=self._settings,
                provider=provider,
                trace_dir=trace_dir,
                write_trace=self._write_trace,
            )
            result = eval_runner.run(
                task,
                experiment_cfg,
                workspace=ws_path,
                pricing=pricing,
                # We always pass an explicit workspace, so EvaluationRunner's
                # own cleanup (which only ever removes a temp dir *it*
                # created) never applies here — do it ourselves instead.
                cleanup=False,
            )
            result.experiment_id = experiment_id
            result.arm_id = arm.arm_id
            result.task_category = task.metadata.get("category")
            result.exposed_tools = exposed_tools
            if cleanup_workspace:
                shutil.rmtree(ws_path, ignore_errors=True)
            return result
        except Exception as exc:  # noqa: BLE001 - one bad arm must not kill the batch
            logger.exception(
                "Matrix task-arm failed: experiment=%s task=%s arm=%s",
                experiment_id, task.task_id, arm.arm_id,
            )
            return self._error_result(
                task=task,
                arm=arm,
                experiment_cfg=experiment_cfg,
                experiment_id=experiment_id,
                exc=exc,
                ws_path=ws_path,
            )

    @staticmethod
    def _exposed_tools(
        experiment_cfg: ExperimentConfig, task: EvalTask, ws_path: Path
    ) -> list[str] | None:
        """The tool names this arm would expose for *task* — informational.

        Computed independently of the real run (``ToolExposureStrategy.select``
        is a pure function of the registry and the task's prompt text — see
        ``forge/tools.py``), so this never affects, and cannot desync from,
        what the real ``AgentRuntime`` run actually used.
        """
        tool_strategy, _ = experiment_cfg.resolve_strategies()
        registry = build_default_registry(workspace_root=ws_path)
        subset = tool_strategy.select(registry, context=None, task=task.prompt)
        return subset if subset is not None else registry.list_names()

    @staticmethod
    def _error_result(
        *,
        task: EvalTask,
        arm: ExperimentArm,
        experiment_cfg: ExperimentConfig,
        experiment_id: str,
        exc: Exception,
        ws_path: Path,
    ) -> EvalResult:
        run_id = f"errored_{task.task_id}_{arm.arm_id}_{uuid.uuid4().hex[:8]}"
        return EvalResult(
            task_id=task.task_id,
            run_id=run_id,
            provider=experiment_cfg.provider,
            model=experiment_cfg.model,
            tool_strategy=arm.tool_strategy,
            context_strategy=arm.context_strategy,
            status=STATUS_RUNNER_ERROR,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            temperature=experiment_cfg.temperature,
            max_output_tokens=experiment_cfg.max_output_tokens,
            max_llm_turns=experiment_cfg.max_llm_turns,
            max_tool_calls=experiment_cfg.max_tool_calls,
            workspace=str(ws_path),
            label=experiment_cfg.label,
            experiment_id=experiment_id,
            arm_id=arm.arm_id,
            task_category=task.metadata.get("category"),
        )
