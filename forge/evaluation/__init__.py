"""
forge.evaluation — measurement infrastructure for controlled FORGE experiments.

This package sits *above* the agent runtime.  It does not contain a second
agent loop and never talks to a provider SDK directly:

    EvalTask                  what to run (stable identity, checks, fixtures)
        │
    ExperimentConfig          the one controlled configuration for a run
        │                     (provider, model, tool + context strategy, params)
        ▼
    EvaluationRunner  ──────►  forge.agent.AgentRuntime  (the only real loop)
        │                          │
        │                          ▼
        │                     JSONL trace + AgentRun
        ▼
    EvalResult                one row of metrics (None-safe; never fabricated)
        │
    aggregate_results / group_results
        ▼
    AggregateStats            success rate, token/latency/cost roll-ups

The Step 4 baseline task suite (``experiments/tasks/*.json``) is loaded by
``forge.evaluation.suite`` into the same ``EvalTask`` type — no parallel
abstraction.

Both the ``fixed``/``adaptive`` tool strategies (Step 5) and the
``raw``/``managed`` context strategies (Step 6) are implemented, so all four
combinations are selectable and runnable. ``forge.evaluation.matrix``
(Step 7) runs the same tasks under all four arms (``fixed_raw``,
``fixed_managed``, ``adaptive_raw``, ``adaptive_managed``) and collects
comparable, reloadable ``EvalResult`` rows — see
``docs/step-07-controlled-2x2-experiment.md``. ``forge.evaluation.analysis``
(Step 8) turns those rows into a reproducible statistical comparison across
the four arms — see ``docs/step-08-statistical-analysis.md``.
"""

from __future__ import annotations

from forge.evaluation.analysis import (
    ANALYSIS_SCHEMA_VERSION,
    NUMERIC_METRICS,
    AnalysisReport,
    DataQualityIssue,
    InteractionSummary,
    MetricSummary,
    OutcomeSummary,
    PairedMetricComparison,
    SuccessDiscordance,
    WilcoxonResult,
    analyze,
    check_matrix_completeness,
    group_by_arm,
    group_by_task,
    interaction_effect,
    paired_metric_comparison,
    read_analysis_json,
    run_analysis,
    success_discordance,
    summarize_metric,
    summarize_outcomes,
    validate_results,
    wilcoxon_paired_metric,
    write_analysis_json,
)

from forge.evaluation.aggregate import (
    AggregateStats,
    aggregate_results,
    group_results,
)
from forge.evaluation.experiment import (
    CONTEXT_STRATEGY_MANAGED,
    CONTEXT_STRATEGY_RAW,
    FUTURE_CONTEXT_STRATEGIES,
    FUTURE_TOOL_STRATEGIES,
    IMPLEMENTED_CONTEXT_STRATEGIES,
    IMPLEMENTED_TOOL_STRATEGIES,
    TOOL_STRATEGY_ADAPTIVE,
    TOOL_STRATEGY_FIXED,
    ExperimentConfig,
)
from forge.evaluation.result import (
    EvalResult,
    TokenPricing,
    append_result_jsonl,
    read_results_jsonl,
    write_results_jsonl,
)
from forge.evaluation.matrix import (
    ARM_ADAPTIVE_MANAGED,
    ARM_ADAPTIVE_RAW,
    ARM_FIXED_MANAGED,
    ARM_FIXED_RAW,
    ARMS,
    ARMS_BY_ID,
    STATUS_RUNNER_ERROR,
    ExperimentArm,
    MatrixExperimentSummary,
    MatrixRunner,
    arms_by_ids,
    outcome_category,
    read_matrix_metadata,
)
from forge.evaluation.runner import EvaluationRunner
from forge.evaluation.suite import (
    DEFAULT_TASKS_DIR,
    TaskSuiteError,
    get_task,
    list_task_ids,
    load_suite,
    load_suite_map,
    load_task,
)
from forge.evaluation.task import (
    ArtifactCheck,
    CheckOutcome,
    EvalTask,
    FixtureFile,
)

__all__ = [
    # task representation
    "EvalTask",
    "ArtifactCheck",
    "FixtureFile",
    "CheckOutcome",
    # baseline task suite + loader
    "DEFAULT_TASKS_DIR",
    "TaskSuiteError",
    "load_task",
    "load_suite",
    "load_suite_map",
    "list_task_ids",
    "get_task",
    # experiment configuration
    "ExperimentConfig",
    "TOOL_STRATEGY_FIXED",
    "TOOL_STRATEGY_ADAPTIVE",
    "CONTEXT_STRATEGY_RAW",
    "CONTEXT_STRATEGY_MANAGED",
    "IMPLEMENTED_TOOL_STRATEGIES",
    "IMPLEMENTED_CONTEXT_STRATEGIES",
    "FUTURE_TOOL_STRATEGIES",
    "FUTURE_CONTEXT_STRATEGIES",
    # runner + result
    "EvaluationRunner",
    "EvalResult",
    "TokenPricing",
    "write_results_jsonl",
    "append_result_jsonl",
    "read_results_jsonl",
    # aggregation
    "AggregateStats",
    "aggregate_results",
    "group_results",
    # formal 2x2 controlled experiment (Step 7)
    "ExperimentArm",
    "ARMS",
    "ARMS_BY_ID",
    "ARM_FIXED_RAW",
    "ARM_FIXED_MANAGED",
    "ARM_ADAPTIVE_RAW",
    "ARM_ADAPTIVE_MANAGED",
    "arms_by_ids",
    "MatrixRunner",
    "MatrixExperimentSummary",
    "read_matrix_metadata",
    "outcome_category",
    "STATUS_RUNNER_ERROR",
    # statistical analysis and comparison (Step 8)
    "ANALYSIS_SCHEMA_VERSION",
    "NUMERIC_METRICS",
    "AnalysisReport",
    "DataQualityIssue",
    "InteractionSummary",
    "MetricSummary",
    "OutcomeSummary",
    "PairedMetricComparison",
    "SuccessDiscordance",
    "WilcoxonResult",
    "analyze",
    "check_matrix_completeness",
    "group_by_arm",
    "group_by_task",
    "interaction_effect",
    "paired_metric_comparison",
    "read_analysis_json",
    "run_analysis",
    "success_discordance",
    "summarize_metric",
    "summarize_outcomes",
    "validate_results",
    "wilcoxon_paired_metric",
    "write_analysis_json",
]
