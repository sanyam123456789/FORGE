"""
forge.evaluation.runner — the evaluation runner.

``EvaluationRunner`` is a thin measurement wrapper *around* the existing
``forge.agent.AgentRuntime``.  It:

1. resolves the (currently only) ``fixed`` / ``raw`` strategy pair,
2. gives the run an isolated workspace (a temp dir unless one is provided),
3. invokes the one real FORGE agent loop,
4. runs the task's artefact checks against the resulting workspace,
5. folds the ``AgentRun`` counters + the check outcome into an ``EvalResult``.

It contains no agent loop and no provider-SDK calls of its own.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge.agent import STATUS_COMPLETED, AgentConfig, AgentRuntime
from forge.evaluation.experiment import ExperimentConfig
from forge.evaluation.result import EvalResult, TokenPricing
from forge.evaluation.task import EvalTask
from forge.llm import LLMProvider, get_provider
from forge.logging import get_logger

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class EvaluationRunner:
    """Runs one ``EvalTask`` under one ``ExperimentConfig`` and returns metrics.

    Parameters
    ----------
    settings:
        A ``ForgeSettings``-like object.  Used only to build a provider when
        one is not injected; strategy/param values come from the
        ``ExperimentConfig``.  Falls back to ``forge.config.settings``.
    provider:
        Optional pre-built ``LLMProvider`` (tests inject a scripted fake).
        When ``None`` the runner calls ``get_provider(settings_override=...)``.
    trace_dir:
        Where per-run JSONL traces are written (default ``<cwd>/runs/traces``).
    write_trace:
        Forwarded to ``AgentRuntime``; disable for fast unit tests.
    """

    def __init__(
        self,
        *,
        settings: Any = None,
        provider: LLMProvider | None = None,
        trace_dir: str | Path | None = None,
        write_trace: bool = True,
    ) -> None:
        if settings is None:
            from forge.config import settings as _settings

            settings = _settings
        self._settings = settings
        self._provider = provider
        self._trace_dir = Path(trace_dir) if trace_dir is not None else Path("runs") / "traces"
        self._write_trace = write_trace

    # ------------------------------------------------------------------

    def run(
        self,
        task: EvalTask,
        experiment: ExperimentConfig,
        *,
        workspace: str | Path | None = None,
        pricing: TokenPricing | None = None,
        cleanup: bool | None = None,
    ) -> EvalResult:
        """Execute *task* under *experiment* and return an ``EvalResult``.

        Raises ``NotImplementedError`` (via ``experiment.resolve_strategies``)
        if an unimplemented future strategy was requested — the runner never
        silently downgrades.
        """
        tool_strategy, context_strategy = experiment.resolve_strategies()

        created_ws = workspace is None
        if cleanup is None:
            cleanup = created_ws  # only auto-remove dirs we made
        ws_path = (
            Path(tempfile.mkdtemp(prefix="forge_eval_"))
            if created_ws
            else Path(workspace)
        )
        ws_path.mkdir(parents=True, exist_ok=True)

        # Provision any starter/fixture files the task carries, before the
        # agent runs. Tasks with no fixtures make this a no-op.
        provisioned = task.provision(ws_path)
        if provisioned:
            logger.debug(
                "Provisioned %d fixture file(s) into %s", len(provisioned), ws_path
            )

        provider = self._provider or get_provider(settings_override=self._settings)

        agent_config = AgentConfig(
            task=task.prompt,
            tool_exposure=tool_strategy,
            context_strategy=context_strategy,
            temperature=experiment.temperature,
            max_output_tokens=experiment.max_output_tokens,
            max_llm_turns=experiment.max_llm_turns,
            max_tool_calls=experiment.max_tool_calls,
        )
        runtime = AgentRuntime(
            agent_config,
            provider=provider,
            workspace_root=ws_path,
            write_trace=self._write_trace,
            log_dir=self._trace_dir,
        )

        started_at = _utc_now_iso()
        t0 = time.monotonic()
        try:
            agent_run = runtime.run()
        finally:
            duration_s = time.monotonic() - t0
        finished_at = _utc_now_iso()

        check_outcome = task.run_checks(ws_path)

        completed = agent_run.status == STATUS_COMPLETED
        success = completed and (check_outcome.passed is not False)

        cost_usd: float | None = None
        if pricing is not None:
            cost_usd = pricing.cost_for(
                input_tokens=agent_run.input_tokens,
                output_tokens=agent_run.output_tokens,
                cached_input_tokens=agent_run.cached_input_tokens,
            )

        result = EvalResult(
            task_id=task.task_id,
            run_id=agent_run.run_id,
            provider=provider.provider_name,
            model=provider.model_name,
            tool_strategy=experiment.tool_strategy,
            context_strategy=experiment.context_strategy,
            status=agent_run.status,
            success=success,
            stop_reason=agent_run.stop_reason,
            error=agent_run.error,
            input_tokens=agent_run.input_tokens,
            output_tokens=agent_run.output_tokens,
            total_tokens=agent_run.total_tokens,
            latency_ms=agent_run.llm_latency_ms,
            llm_calls=agent_run.llm_calls,
            tool_calls=agent_run.tool_calls,
            turns=agent_run.turns,
            reasoning_tokens=agent_run.reasoning_tokens,
            cached_input_tokens=agent_run.cached_input_tokens,
            context_messages=agent_run.context_messages,
            context_char_count=agent_run.context_char_count,
            cost_available=cost_usd is not None,
            cost_usd=cost_usd,
            cost_source=pricing.source if pricing is not None else None,
            temperature=experiment.temperature,
            max_output_tokens=experiment.max_output_tokens,
            max_llm_turns=experiment.max_llm_turns,
            max_tool_calls=experiment.max_tool_calls,
            workspace=str(ws_path),
            label=experiment.label,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            trace_path=str(agent_run.trace_path) if agent_run.trace_path else None,
            checks_run=check_outcome.checks_run,
            checks_passed=check_outcome.passed,
            checks_detail=list(check_outcome.details),
        )

        if created_ws and cleanup:
            shutil.rmtree(ws_path, ignore_errors=True)
            logger.debug("Removed temp eval workspace %s", ws_path)

        logger.info(
            "Eval done: task=%s run=%s strategy=%s status=%s success=%s",
            task.task_id,
            result.run_id,
            experiment.strategy_key,
            result.status,
            result.success,
        )
        return result
