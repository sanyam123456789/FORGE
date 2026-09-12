"""
forge.evaluation.analysis — Step 8: statistical analysis and comparison.

Consumes the structured ``EvalResult`` rows Step 7's ``MatrixRunner`` (or a
plain ``EvaluationRunner``) already produces and turns them into a
reproducible, machine-readable comparison across the four arms
(``fixed_raw``, ``fixed_managed``, ``adaptive_raw``, ``adaptive_managed`` —
see ``forge.evaluation.matrix``). It contains no agent loop, no provider-SDK
call, and does not redesign any tool/context strategy or the experiment
runner — it only reads ``results.jsonl`` (+ optionally ``metadata.json``)
and computes descriptive statistics over what is already there.

Follows the same rules as the rest of ``forge.evaluation``:

* A metric the run never reported stays ``None`` ("missing") — never
  silently becomes ``0``.
* A value that is present but the wrong type / non-finite is "invalid" —
  counted and reported, never coerced or silently dropped.
* Cost is "not applicable" (not "missing") whenever a run's
  ``cost_available`` is ``False`` — no pricing was ever supplied for it.
* No record is ever silently dropped. Duplicates, unknown arm ids, mismatched
  labels, and incomplete matrices are all surfaced as structured
  ``DataQualityIssue`` warnings/errors, never hidden.

Statistical honesty: this module reports counts, rates, means, medians,
paired differences, and a real (stdlib-only) Wilcoxon signed-rank
*statistic* — but it never computes or prints a p-value, confidence
interval, effect size, or a "significant" / "not significant" label.
Interpreting whether an observed difference is meaningful is left to whoever
reads this report, armed with the sample sizes and skip reasons it always
includes. When a test is skipped (too few observations, all-zero
differences, missing data), the reason is recorded, never silently omitted.

Explicitly NOT implemented here: dashboards/charts, ML- or LLM-based
analysis, SWE-bench/external-agent comparison, WhatsApp/Pi integration, and
no network or provider call of any kind.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from forge.evaluation.matrix import (
    ARM_ADAPTIVE_MANAGED,
    ARM_ADAPTIVE_RAW,
    ARM_FIXED_MANAGED,
    ARM_FIXED_RAW,
    ARMS,
    ARMS_BY_ID,
    MatrixExperimentSummary,
    outcome_category,
    read_matrix_metadata,
)
from forge.evaluation.result import EvalResult, read_results_jsonl
from forge.logging import get_logger

logger = get_logger(__name__)

#: Bump when the on-disk analysis JSON shape changes in a non-additive way.
ANALYSIS_SCHEMA_VERSION = 1

OUTCOME_CATEGORIES: tuple[str, ...] = (
    "success",
    "completed_incorrect",
    "runtime_error",
    "provider_error",
    "limit_reached",
    "runner_error",
    "unknown",
)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Numeric metrics actually present on EvalResult (Step 3/5/6/7 schema)
# ---------------------------------------------------------------------------


def _cost(r: EvalResult) -> float | None:
    return r.cost_usd if r.cost_available else None


def _resolved_total_tokens(r: EvalResult) -> Any:
    """``EvalResult.resolved_total_tokens``, but never raises.

    The property does ``input_tokens + output_tokens`` when ``total_tokens``
    is absent; a malformed record (e.g. a non-numeric ``input_tokens`` from a
    corrupted JSONL row) would otherwise raise ``TypeError`` deep inside
    validation/summarisation. Surface that as NaN instead, which
    ``_classify`` already treats as "invalid" — never crashes, never
    silently becomes a number.
    """
    try:
        return r.resolved_total_tokens
    except TypeError:
        return math.nan


def _exposed_tool_count(r: EvalResult) -> int | None:
    return len(r.exposed_tools) if r.exposed_tools is not None else None


#: metric name -> extractor. Only metrics that actually exist on EvalResult
#: (see forge/evaluation/result.py) are listed here — nothing is invented.
NUMERIC_METRICS: dict[str, Callable[[EvalResult], Any]] = {
    "duration_s": lambda r: r.duration_s,
    "latency_ms": lambda r: r.latency_ms,
    "llm_calls": lambda r: r.llm_calls,
    "tool_calls": lambda r: r.tool_calls,
    "turns": lambda r: r.turns,
    "input_tokens": lambda r: r.input_tokens,
    "output_tokens": lambda r: r.output_tokens,
    "total_tokens": _resolved_total_tokens,
    "reasoning_tokens": lambda r: r.reasoning_tokens,
    "cached_input_tokens": lambda r: r.cached_input_tokens,
    "context_messages": lambda r: r.context_messages,
    "context_char_count": lambda r: r.context_char_count,
    "context_items_dropped": lambda r: r.context_items_dropped,
    "context_items_compressed": lambda r: r.context_items_compressed,
    "context_chars_saved": lambda r: r.context_chars_saved,
    "exposed_tool_count": _exposed_tool_count,
    "cost_usd": _cost,
}

#: Metrics where a *smaller* value is the studied efficiency improvement —
#: informational labelling only (see PairedMetricComparison.improved_arm),
#: never used to compute or imply statistical significance. Metrics not
#: listed here (e.g. context_items_dropped/compressed, context_chars_saved,
#: cached_input_tokens) have no fixed "better" direction on their own — a
#: raw sign is reported for them but no improvement label is attached.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "duration_s",
        "latency_ms",
        "llm_calls",
        "tool_calls",
        "turns",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "exposed_tool_count",
        "cost_usd",
    }
)

FACTOR_TOOL_EXPOSURE_PAIRS: tuple[tuple[str, str], ...] = (
    (ARM_FIXED_RAW, ARM_ADAPTIVE_RAW),
    (ARM_FIXED_MANAGED, ARM_ADAPTIVE_MANAGED),
)
FACTOR_CONTEXT_PAIRS: tuple[tuple[str, str], ...] = (
    (ARM_FIXED_RAW, ARM_FIXED_MANAGED),
    (ARM_ADAPTIVE_RAW, ARM_ADAPTIVE_MANAGED),
)


def _classify(value: Any) -> tuple[str, float | None]:
    """Return ("valid"|"missing"|"invalid", numeric_value_or_None)."""
    if value is None:
        return "missing", None
    if isinstance(value, bool):  # bool is a subclass of int — never a metric
        return "invalid", None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "invalid", None
        return "valid", float(value)
    return "invalid", None


def _classify_metric_value(metric: str, result: EvalResult) -> tuple[str, float | None]:
    if metric == "cost_usd":
        if not result.cost_available:
            return "not_applicable", None
        return _classify(result.cost_usd)
    return _classify(NUMERIC_METRICS[metric](result))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_by_arm(results: Sequence[EvalResult]) -> dict[str | None, list[EvalResult]]:
    """Group *results* by ``arm_id`` (``None`` bucket for non-matrix rows)."""
    out: dict[str | None, list[EvalResult]] = {}
    for r in results:
        out.setdefault(r.arm_id, []).append(r)
    return out


def group_by_task(results: Sequence[EvalResult]) -> dict[str, list[EvalResult]]:
    """Group *results* by ``task_id``."""
    out: dict[str, list[EvalResult]] = {}
    for r in results:
        out.setdefault(r.task_id, []).append(r)
    return out


def _first_by_task(results: Sequence[EvalResult], arm_id: str) -> dict[str, EvalResult]:
    """The first record per ``task_id`` for *arm_id*, in input order.

    Duplicate (task_id, arm_id) records are flagged separately by
    ``validate_results`` (``duplicate_task_arm``) and are never silently
    dropped from the dataset — but a paired comparison needs exactly one
    value per (task, arm) cell, so this picks deterministically (the first
    one encountered) rather than averaging or guessing which is "correct".
    """
    out: dict[str, EvalResult] = {}
    for r in results:
        if r.arm_id == arm_id and r.task_id not in out:
            out[r.task_id] = r
    return out


# ---------------------------------------------------------------------------
# Data-quality validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataQualityIssue:
    """One structured validation finding — never a silently-swallowed problem."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


def validate_results(results: Sequence[EvalResult]) -> list[DataQualityIssue]:
    """Structural / data-quality checks over a raw (unfiltered) record set."""
    issues: list[DataQualityIssue] = []

    if not results:
        issues.append(
            DataQualityIssue("warning", "empty_dataset", "No result records to analyze.")
        )
        return issues

    for r in results:
        if not r.task_id or not str(r.task_id).strip():
            issues.append(
                DataQualityIssue(
                    "error",
                    "missing_task_id",
                    f"Record run_id={r.run_id!r} has an empty task_id.",
                    {"run_id": r.run_id},
                )
            )

    for r in results:
        if r.arm_id is None:
            continue  # not part of a matrix run — not applicable, not an error
        if not str(r.arm_id).strip():
            issues.append(
                DataQualityIssue(
                    "warning",
                    "missing_arm_id",
                    f"Record run_id={r.run_id!r} has an empty (non-None) arm_id.",
                    {"run_id": r.run_id},
                )
            )
        elif r.arm_id not in ARMS_BY_ID:
            issues.append(
                DataQualityIssue(
                    "error",
                    "unknown_arm_id",
                    f"Record run_id={r.run_id!r} has unknown arm_id {r.arm_id!r}.",
                    {"run_id": r.run_id, "arm_id": r.arm_id},
                )
            )

    for r in results:
        arm = ARMS_BY_ID.get(r.arm_id) if r.arm_id else None
        if arm is None:
            continue
        if r.tool_strategy != arm.tool_strategy or r.context_strategy != arm.context_strategy:
            issues.append(
                DataQualityIssue(
                    "error",
                    "inconsistent_strategy_label",
                    f"Record run_id={r.run_id!r} arm_id={r.arm_id!r} carries strategy labels "
                    f"({r.tool_strategy!r}, {r.context_strategy!r}) that do not match the arm's "
                    f"pinned pair ({arm.tool_strategy!r}, {arm.context_strategy!r}).",
                    {"run_id": r.run_id, "arm_id": r.arm_id},
                )
            )

    seen: dict[tuple[Any, str, str], list[str]] = {}
    for r in results:
        if r.arm_id is None:
            continue
        key = (r.experiment_id, r.task_id, r.arm_id)
        seen.setdefault(key, []).append(r.run_id)
    for (exp_id, task_id, arm_id), run_ids in seen.items():
        if len(run_ids) > 1:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "duplicate_task_arm",
                    f"task_id={task_id!r} arm_id={arm_id!r} (experiment_id={exp_id!r}) has "
                    f"{len(run_ids)} records instead of 1; all are kept, but paired comparisons "
                    f"use only the first.",
                    {"task_id": task_id, "arm_id": arm_id, "experiment_id": exp_id, "run_ids": run_ids},
                )
            )

    by_task_cat: dict[str, set[Any]] = {}
    for r in results:
        if r.task_category is None:
            continue
        by_task_cat.setdefault(r.task_id, set()).add(r.task_category)
    for task_id, cats in by_task_cat.items():
        if len(cats) > 1:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "mismatched_task_category",
                    f"task_id={task_id!r} has inconsistent task_category values across "
                    f"records: {sorted(cats)}.",
                    {"task_id": task_id, "categories": sorted(cats)},
                )
            )

    for metric in NUMERIC_METRICS:
        bad_run_ids = [
            r.run_id for r in results if _classify_metric_value(metric, r)[0] == "invalid"
        ]
        if bad_run_ids:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "malformed_numeric_value",
                    f"{len(bad_run_ids)} record(s) have a non-numeric/invalid value for "
                    f"{metric!r}; excluded from that metric's statistics (never treated as "
                    f"missing or zero).",
                    {"metric": metric, "count": len(bad_run_ids), "run_ids": bad_run_ids[:10]},
                )
            )

    return issues


def check_matrix_completeness(
    results: Sequence[EvalResult], arms: Sequence[str], task_ids: Sequence[str]
) -> list[DataQualityIssue]:
    """Warn about a task missing a record for one of the analyzed arms."""
    issues: list[DataQualityIssue] = []
    if not arms or not task_ids:
        return issues

    want_arms = set(arms)
    present: dict[str, set[str]] = {}
    for r in results:
        if r.task_id in task_ids and r.arm_id in want_arms:
            present.setdefault(r.task_id, set()).add(r.arm_id)

    missing_by_task = {
        t: sorted(want_arms - present.get(t, set()))
        for t in task_ids
        if want_arms - present.get(t, set())
    }
    if missing_by_task:
        issues.append(
            DataQualityIssue(
                "warning",
                "incomplete_matrix",
                f"{len(missing_by_task)} of {len(task_ids)} task(s) do not have a record for "
                f"every analyzed arm.",
                {"missing_by_task": missing_by_task},
            )
        )

    if 0 < len(want_arms) < 4:
        issues.append(
            DataQualityIssue(
                "warning",
                "partial_arm_selection",
                f"Only {len(want_arms)} of the 4 canonical arms are included in this analysis "
                f"({sorted(want_arms)}); factor-level/interaction comparisons that need a "
                f"missing arm are skipped.",
                {"arms": sorted(want_arms)},
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Outcome (success/error) summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeSummary:
    key: str  # arm_id, "unassigned", or "__all__"
    total: int
    success: int
    completed_incorrect: int
    runtime_error: int
    provider_error: int
    limit_reached: int
    runner_error: int
    unknown: int
    completed: int  # success + completed_incorrect
    denom_all: int
    denom_excl_infra_errors: int  # total - provider_error - runner_error
    success_rate_all: float | None
    success_rate_excl_infra_errors: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_outcomes(results: Sequence[EvalResult], key: str) -> OutcomeSummary:
    counts = dict.fromkeys(OUTCOME_CATEGORIES, 0)
    for r in results:
        counts[outcome_category(r)] += 1
    total = len(results)
    denom_excl = max(total - counts["provider_error"] - counts["runner_error"], 0)
    return OutcomeSummary(
        key=key,
        total=total,
        success=counts["success"],
        completed_incorrect=counts["completed_incorrect"],
        runtime_error=counts["runtime_error"],
        provider_error=counts["provider_error"],
        limit_reached=counts["limit_reached"],
        runner_error=counts["runner_error"],
        unknown=counts["unknown"],
        completed=counts["success"] + counts["completed_incorrect"],
        denom_all=total,
        denom_excl_infra_errors=denom_excl,
        success_rate_all=(counts["success"] / total) if total else None,
        success_rate_excl_infra_errors=(counts["success"] / denom_excl) if denom_excl else None,
    )


# ---------------------------------------------------------------------------
# Per-metric descriptive statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    valid_count: int
    missing_count: int
    invalid_count: int
    not_applicable_count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    stdev: float | None  # None unless >= 2 valid observations
    total: float | None  # sum of valid observations; None if none are valid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_metric(results: Sequence[EvalResult], metric: str) -> MetricSummary:
    valid: list[float] = []
    missing = invalid = not_applicable = 0
    for r in results:
        status, value = _classify_metric_value(metric, r)
        if status == "valid":
            valid.append(value)  # type: ignore[arg-type]
        elif status == "missing":
            missing += 1
        elif status == "not_applicable":
            not_applicable += 1
        else:
            invalid += 1
    n = len(valid)
    return MetricSummary(
        metric=metric,
        valid_count=n,
        missing_count=missing,
        invalid_count=invalid,
        not_applicable_count=not_applicable,
        mean=(sum(valid) / n) if n else None,
        median=statistics.median(valid) if n else None,
        minimum=min(valid) if n else None,
        maximum=max(valid) if n else None,
        stdev=(statistics.stdev(valid) if n >= 2 else None),
        total=(sum(valid) if n else None),
    )


# ---------------------------------------------------------------------------
# Paired task comparison (same task, two arms)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedMetricComparison:
    metric: str
    arm_a: str
    arm_b: str
    n_pairs: int
    per_task_diff: dict[str, float]  # task_id -> (b - a)
    mean_diff: float | None
    median_diff: float | None
    direction: str | None  # "a_higher" | "b_higher" | "equal"
    lower_is_better: bool | None  # None when the metric has no fixed direction
    improved_arm: str | None  # "arm_a" | "arm_b" | "equal" | None
    per_task_pct_change: dict[str, float]  # task_id -> % change a->b, only where a != 0
    pct_change_undefined_tasks: list[str]  # a == 0: pct change mathematically undefined
    tasks_without_record_in_a: list[str]
    tasks_without_record_in_b: list[str]
    tasks_invalid_in_a: list[str]
    tasks_invalid_in_b: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def paired_metric_comparison(
    results: Sequence[EvalResult],
    arm_a: str,
    arm_b: str,
    metric: str,
    *,
    task_ids: Sequence[str] | None = None,
) -> PairedMetricComparison:
    by_a = _first_by_task(results, arm_a)
    by_b = _first_by_task(results, arm_b)
    universe = sorted(task_ids) if task_ids is not None else sorted(set(by_a) | set(by_b))

    diffs: dict[str, float] = {}
    pct: dict[str, float] = {}
    pct_undef: list[str] = []
    missing_a: list[str] = []
    missing_b: list[str] = []
    invalid_a: list[str] = []
    invalid_b: list[str] = []

    for t in universe:
        ra, rb = by_a.get(t), by_b.get(t)
        if ra is None:
            missing_a.append(t)
        if rb is None:
            missing_b.append(t)
        if ra is None or rb is None:
            continue
        sa, va = _classify_metric_value(metric, ra)
        sb, vb = _classify_metric_value(metric, rb)
        if sa != "valid":
            invalid_a.append(t)
        if sb != "valid":
            invalid_b.append(t)
        if sa != "valid" or sb != "valid":
            continue
        diff = vb - va  # type: ignore[operator]
        diffs[t] = diff
        if va:  # non-zero
            pct[t] = (diff / va) * 100.0
        else:
            pct_undef.append(t)

    n = len(diffs)
    mean_diff = (sum(diffs.values()) / n) if n else None
    median_diff = statistics.median(diffs.values()) if n else None

    direction: str | None = None
    if mean_diff is not None:
        direction = "b_higher" if mean_diff > 0 else "a_higher" if mean_diff < 0 else "equal"

    lower_is_better = metric in LOWER_IS_BETTER if metric in NUMERIC_METRICS else None
    improved_arm: str | None = None
    if mean_diff is not None and lower_is_better is not None:
        if mean_diff == 0:
            improved_arm = "equal"
        elif (mean_diff < 0) == lower_is_better:
            improved_arm = "arm_b"
        else:
            improved_arm = "arm_a"

    return PairedMetricComparison(
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        n_pairs=n,
        per_task_diff=diffs,
        mean_diff=mean_diff,
        median_diff=median_diff,
        direction=direction,
        lower_is_better=lower_is_better,
        improved_arm=improved_arm,
        per_task_pct_change=pct,
        pct_change_undefined_tasks=pct_undef,
        tasks_without_record_in_a=missing_a,
        tasks_without_record_in_b=missing_b,
        tasks_invalid_in_a=invalid_a,
        tasks_invalid_in_b=invalid_b,
    )


@dataclass(frozen=True)
class SuccessDiscordance:
    """Paired success/failure counts between two arms — a sign-test input.

    No p-value is computed (see module docstring); these are exact counts.
    """

    arm_a: str
    arm_b: str
    n_pairs: int
    both_success: int
    both_fail: int
    a_only: int  # a succeeded, b did not
    b_only: int  # b succeeded, a did not
    tasks_without_record_in_a: list[str]
    tasks_without_record_in_b: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def success_discordance(
    results: Sequence[EvalResult],
    arm_a: str,
    arm_b: str,
    *,
    task_ids: Sequence[str] | None = None,
) -> SuccessDiscordance:
    by_a = _first_by_task(results, arm_a)
    by_b = _first_by_task(results, arm_b)
    universe = sorted(task_ids) if task_ids is not None else sorted(set(by_a) | set(by_b))

    both_s = both_f = a_only = b_only = 0
    missing_a: list[str] = []
    missing_b: list[str] = []
    for t in universe:
        ra, rb = by_a.get(t), by_b.get(t)
        if ra is None:
            missing_a.append(t)
        if rb is None:
            missing_b.append(t)
        if ra is None or rb is None:
            continue
        if ra.success and rb.success:
            both_s += 1
        elif not ra.success and not rb.success:
            both_f += 1
        elif ra.success and not rb.success:
            a_only += 1
        else:
            b_only += 1

    return SuccessDiscordance(
        arm_a=arm_a,
        arm_b=arm_b,
        n_pairs=both_s + both_f + a_only + b_only,
        both_success=both_s,
        both_fail=both_f,
        a_only=a_only,
        b_only=b_only,
        tasks_without_record_in_a=missing_a,
        tasks_without_record_in_b=missing_b,
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank *statistic* (stdlib only) — no p-value, ever
# ---------------------------------------------------------------------------

#: Below this many non-zero paired differences, the rank-sum statistic has no
#: useful resolution — the test is skipped rather than computed misleadingly.
_MIN_WILCOXON_N = 4


def _wilcoxon_statistic(
    diffs: Sequence[float],
) -> tuple[int, int, int, float | None, str | None]:
    """Return ``(n_total_pairs, n_ties_zero, n_used, w_statistic, skip_reason)``.

    ``w_statistic`` is ``min(W+, W-)`` — the smaller of the rank-sums of
    positive vs. negative differences, with average ranks for ties in
    ``|diff|`` — the standard Wilcoxon signed-rank test statistic. No
    p-value or significance label is derived from it here.
    """
    n_total = len(diffs)
    nonzero = [d for d in diffs if d != 0]
    n_ties_zero = n_total - len(nonzero)
    n_used = len(nonzero)

    if n_total == 0:
        return n_total, n_ties_zero, n_used, None, "no paired observations available"
    if n_used == 0:
        return n_total, n_ties_zero, n_used, None, "all paired differences are zero (no variation)"
    if n_used < _MIN_WILCOXON_N:
        return (
            n_total,
            n_ties_zero,
            n_used,
            None,
            f"fewer than {_MIN_WILCOXON_N} non-zero paired differences",
        )

    order = sorted(range(n_used), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n_used
    j = 0
    while j < n_used:
        k = j
        while k + 1 < n_used and abs(nonzero[order[k + 1]]) == abs(nonzero[order[j]]):
            k += 1
        avg_rank = (j + 1 + k + 1) / 2.0
        for m in range(j, k + 1):
            ranks[order[m]] = avg_rank
        j = k + 1

    w_pos = sum(ranks[i] for i in range(n_used) if nonzero[i] > 0)
    w_neg = sum(ranks[i] for i in range(n_used) if nonzero[i] < 0)
    return n_total, n_ties_zero, n_used, min(w_pos, w_neg), None


@dataclass(frozen=True)
class WilcoxonResult:
    metric: str
    arm_a: str
    arm_b: str
    n_total_pairs: int
    n_ties_zero: int
    n_used: int
    w_statistic: float | None
    skipped: bool
    skip_reason: str | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_WILCOXON_NOTE = (
    "Descriptive statistic only (Wilcoxon signed-rank W = min(W+, W-) over "
    "|diff| ranks); no p-value, confidence interval, or significance label "
    "is computed."
)


def wilcoxon_paired_metric(
    results: Sequence[EvalResult],
    arm_a: str,
    arm_b: str,
    metric: str,
    *,
    task_ids: Sequence[str] | None = None,
) -> WilcoxonResult:
    comp = paired_metric_comparison(results, arm_a, arm_b, metric, task_ids=task_ids)
    diffs = list(comp.per_task_diff.values())
    n_total, n_ties, n_used, w, reason = _wilcoxon_statistic(diffs)
    return WilcoxonResult(
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        n_total_pairs=n_total,
        n_ties_zero=n_ties,
        n_used=n_used,
        w_statistic=w,
        skipped=reason is not None,
        skip_reason=reason,
        note=_WILCOXON_NOTE,
    )


# ---------------------------------------------------------------------------
# Factor-level comparisons (tool exposure, context strategy, interaction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionSummary:
    """Descriptive difference-in-differences: never a causal claim.

    ``(adaptive_managed - adaptive_raw) - (fixed_managed - fixed_raw)``, per
    task, for tasks with a valid value in all four arms.
    """

    metric: str
    n_tasks: int
    mean_diff_in_diff: float | None
    median_diff_in_diff: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_INTERACTION_NOTE = (
    "Difference-in-differences is descriptive only: it does not establish a "
    "causal interaction effect, especially with a small or unbalanced task "
    "set, and no significance test is applied."
)


def interaction_effect(
    results: Sequence[EvalResult], metric: str, *, task_ids: Sequence[str] | None = None
) -> InteractionSummary:
    by = {arm_id: _first_by_task(results, arm_id) for arm_id in ARMS_BY_ID}
    universe = (
        sorted(task_ids)
        if task_ids is not None
        else sorted(
            set(by[ARM_FIXED_RAW])
            | set(by[ARM_FIXED_MANAGED])
            | set(by[ARM_ADAPTIVE_RAW])
            | set(by[ARM_ADAPTIVE_MANAGED])
        )
    )

    dids: dict[str, float] = {}
    for t in universe:
        recs = {
            arm: by[arm].get(t)
            for arm in (ARM_FIXED_RAW, ARM_FIXED_MANAGED, ARM_ADAPTIVE_RAW, ARM_ADAPTIVE_MANAGED)
        }
        if any(r is None for r in recs.values()):
            continue
        vals: dict[str, float] = {}
        ok = True
        for arm, r in recs.items():
            status, v = _classify_metric_value(metric, r)  # type: ignore[arg-type]
            if status != "valid":
                ok = False
                break
            vals[arm] = v  # type: ignore[assignment]
        if not ok:
            continue
        dids[t] = (vals[ARM_ADAPTIVE_MANAGED] - vals[ARM_ADAPTIVE_RAW]) - (
            vals[ARM_FIXED_MANAGED] - vals[ARM_FIXED_RAW]
        )

    n = len(dids)
    return InteractionSummary(
        metric=metric,
        n_tasks=n,
        mean_diff_in_diff=(sum(dids.values()) / n) if n else None,
        median_diff_in_diff=statistics.median(dids.values()) if n else None,
        note=_INTERACTION_NOTE,
    )


def _pair_section(
    results: Sequence[EvalResult], arm_a: str, arm_b: str, task_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        "success_discordance": success_discordance(results, arm_a, arm_b, task_ids=task_ids).to_dict(),
        "metrics": {
            m: paired_metric_comparison(results, arm_a, arm_b, m, task_ids=task_ids).to_dict()
            for m in NUMERIC_METRICS
        },
    }


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisReport:
    schema_version: int
    generated_at: str
    source_results_path: str | None
    source_metadata_path: str | None
    experiment_id: str | None
    arms_analyzed: list[str]
    task_ids_analyzed: list[str]
    task_count: int
    record_count: int
    warnings: list[dict[str, Any]]
    outcome_summary: dict[str, dict[str, Any]]
    metric_summary: dict[str, dict[str, dict[str, Any]]]
    paired_comparisons: dict[str, dict[str, Any]]
    factor_level: dict[str, Any]
    statistical_tests: dict[str, Any]
    unavailable_metrics: list[str]
    metadata: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "source_results_path": self.source_results_path,
            "source_metadata_path": self.source_metadata_path,
            "experiment_id": self.experiment_id,
            "arms_analyzed": list(self.arms_analyzed),
            "task_ids_analyzed": list(self.task_ids_analyzed),
            "task_count": self.task_count,
            "record_count": self.record_count,
            "warnings": list(self.warnings),
            "outcome_summary": self.outcome_summary,
            "metric_summary": self.metric_summary,
            "paired_comparisons": self.paired_comparisons,
            "factor_level": self.factor_level,
            "statistical_tests": self.statistical_tests,
            "unavailable_metrics": list(self.unavailable_metrics),
            "metadata": self.metadata,
        }


def analyze(
    results: Sequence[EvalResult],
    *,
    metadata: MatrixExperimentSummary | None = None,
    arms: Sequence[str] | None = None,
    task_ids: Sequence[str] | None = None,
    source_results_path: str | None = None,
    source_metadata_path: str | None = None,
) -> AnalysisReport:
    """Compute the full Step 8 comparison over *results*.

    ``arms``/``task_ids`` optionally restrict the analysis to a subset (the
    CLI's ``--arm``/``--task-id``); when omitted, every arm/task present in
    *results* is used. Restricting to arms with zero matching records is not
    an error — that arm's summaries simply show zero observations, and
    ``check_matrix_completeness`` records why.
    """
    all_results = list(results)

    resolved_task_ids = (
        sorted(set(task_ids)) if task_ids is not None else sorted({r.task_id for r in all_results if r.task_id})
    )
    if arms is not None:
        resolved_arms = list(dict.fromkeys(arms))  # de-dup, preserve caller order
    else:
        present = {r.arm_id for r in all_results if r.arm_id in ARMS_BY_ID}
        resolved_arms = [a.arm_id for a in ARMS if a.arm_id in present]

    working = [r for r in all_results if task_ids is None or r.task_id in set(resolved_task_ids)]
    if arms is not None:
        wanted = set(resolved_arms)
        working = [r for r in working if r.arm_id in wanted]

    warnings = validate_results(working)
    warnings += check_matrix_completeness(working, resolved_arms, resolved_task_ids)

    exp_ids = sorted({r.experiment_id for r in working if r.experiment_id})
    if len(exp_ids) == 1:
        experiment_id: str | None = exp_ids[0]
    elif metadata is not None:
        experiment_id = metadata.experiment_id
    else:
        experiment_id = None
    if len(exp_ids) > 1:
        warnings.append(
            DataQualityIssue(
                "warning",
                "mixed_experiment_ids",
                f"Results span {len(exp_ids)} distinct experiment_id values: {exp_ids}.",
                {"experiment_ids": exp_ids},
            )
        )

    by_arm_all = group_by_arm(working)

    outcome_summary: dict[str, dict[str, Any]] = {
        "__all__": summarize_outcomes(working, "__all__").to_dict()
    }
    for arm_id in resolved_arms:
        outcome_summary[arm_id] = summarize_outcomes(by_arm_all.get(arm_id, []), arm_id).to_dict()
    if arms is None and None in by_arm_all:
        outcome_summary["unassigned"] = summarize_outcomes(by_arm_all[None], "unassigned").to_dict()

    metric_summary: dict[str, dict[str, dict[str, Any]]] = {
        "__all__": {m: summarize_metric(working, m).to_dict() for m in NUMERIC_METRICS}
    }
    for arm_id in resolved_arms:
        rows = by_arm_all.get(arm_id, [])
        metric_summary[arm_id] = {m: summarize_metric(rows, m).to_dict() for m in NUMERIC_METRICS}

    unavailable_metrics = sorted(
        m for m in NUMERIC_METRICS if metric_summary["__all__"][m]["valid_count"] == 0
    )

    paired_comparisons: dict[str, dict[str, Any]] = {}
    statistical_tests: dict[str, dict[str, Any]] = {}
    for i, arm_a in enumerate(resolved_arms):
        for arm_b in resolved_arms[i + 1 :]:
            key = f"{arm_a}__vs__{arm_b}"
            paired_comparisons[key] = _pair_section(working, arm_a, arm_b, resolved_task_ids)
            statistical_tests[key] = {
                m: wilcoxon_paired_metric(working, arm_a, arm_b, m, task_ids=resolved_task_ids).to_dict()
                for m in NUMERIC_METRICS
            }

    have = set(resolved_arms)
    factor_level: dict[str, Any] = {"tool_exposure": {}, "context_strategy": {}, "interaction": None}
    for a, b in FACTOR_TOOL_EXPOSURE_PAIRS:
        if a in have and b in have:
            factor_level["tool_exposure"][f"{a}__vs__{b}"] = _pair_section(
                working, a, b, resolved_task_ids
            )
    for a, b in FACTOR_CONTEXT_PAIRS:
        if a in have and b in have:
            factor_level["context_strategy"][f"{a}__vs__{b}"] = _pair_section(
                working, a, b, resolved_task_ids
            )
    if all(a in have for a in (ARM_FIXED_RAW, ARM_FIXED_MANAGED, ARM_ADAPTIVE_RAW, ARM_ADAPTIVE_MANAGED)):
        factor_level["interaction"] = {
            m: interaction_effect(working, m, task_ids=resolved_task_ids).to_dict()
            for m in NUMERIC_METRICS
        }
    else:
        warnings.append(
            DataQualityIssue(
                "warning",
                "interaction_skipped",
                "Interaction (tool exposure x context) comparison requires all four "
                "canonical arms to be present in the analyzed data/selection; skipped.",
                {"arms_present": sorted(have)},
            )
        )

    return AnalysisReport(
        schema_version=ANALYSIS_SCHEMA_VERSION,
        generated_at=_utc_now_iso(),
        source_results_path=source_results_path,
        source_metadata_path=source_metadata_path,
        experiment_id=experiment_id,
        arms_analyzed=resolved_arms,
        task_ids_analyzed=resolved_task_ids,
        task_count=len(resolved_task_ids),
        record_count=len(working),
        warnings=[w.to_dict() for w in warnings],
        outcome_summary=outcome_summary,
        metric_summary=metric_summary,
        paired_comparisons=paired_comparisons,
        factor_level=factor_level,
        statistical_tests=statistical_tests,
        unavailable_metrics=unavailable_metrics,
        metadata=metadata.to_dict() if metadata is not None else None,
    )


# ---------------------------------------------------------------------------
# Path-based orchestration (what the CLI uses)
# ---------------------------------------------------------------------------


def run_analysis(
    results_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    arms: Sequence[str] | None = None,
    task_ids: Sequence[str] | None = None,
) -> AnalysisReport:
    """Load *results_path* (+ optional metadata) and run ``analyze()``.

    When ``metadata_path`` is not given, ``<results_path's dir>/metadata.json``
    is used if it exists (the layout ``MatrixRunner`` writes) — silently
    skipped (not an error) if absent or unreadable, since metadata is purely
    informational here (every value it could supply is also inferable from
    the records themselves, except free-text fields like ``label``).
    """
    results_path = Path(results_path)
    results = read_results_jsonl(results_path)

    metadata: MatrixExperimentSummary | None = None
    resolved_metadata_path: Path | None = None
    if metadata_path is not None:
        resolved_metadata_path = Path(metadata_path)
        metadata = read_matrix_metadata(resolved_metadata_path)
    else:
        candidate = results_path.parent / "metadata.json"
        if candidate.exists():
            try:
                metadata = read_matrix_metadata(candidate)
                resolved_metadata_path = candidate
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                logger.warning("Could not parse metadata.json at %s; continuing without it", candidate)

    return analyze(
        results,
        metadata=metadata,
        arms=arms,
        task_ids=task_ids,
        source_results_path=str(results_path),
        source_metadata_path=str(resolved_metadata_path) if resolved_metadata_path else None,
    )


def write_analysis_json(report: AnalysisReport, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write *report* as pretty-printed JSON. Never overwrites unless asked.

    Never touches the source results JSONL — this always writes to a
    separate path.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True (CLI: --overwrite) to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_analysis_json(path: str | Path) -> dict[str, Any]:
    """Reload an analysis JSON file written by ``write_analysis_json``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
