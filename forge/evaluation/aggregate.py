"""
forge.evaluation.aggregate — None-safe roll-ups over many ``EvalResult``s.

The aggregation rules exist so the future 2x2 comparison

    fixed+raw | fixed+managed | adaptive+raw | adaptive+managed

can be summarised without a metric ever being faked:

- Summing a metric across runs: if *every* run is ``None`` for it, the sum is
  ``None``; otherwise the known values are summed and unknowns contribute
  nothing (they are not silently treated as ``0``).
- Averaging: the mean is taken over the runs that actually reported the
  metric.  If none reported it, the average is ``None``.
- ``success_rate`` is ``None`` when there are no results (zero denominator).
- ``cost_per_successful_task`` is ``None`` when cost is unavailable or no run
  succeeded.

Only ``fixed+raw`` is produced in this step, but ``group_results`` already
keys by strategy pair so more conditions slot in later with no change here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from forge.evaluation.result import EvalResult


def _known(values: Iterable[Any]) -> list[Any]:
    return [v for v in values if v is not None]


def _sum_opt(values: Iterable[Any]) -> Any:
    """Sum, preserving 'all unknown' as ``None``."""
    known = _known(values)
    return sum(known) if known else None


def _avg_opt(values: Iterable[Any]) -> float | None:
    """Mean over the values that are not ``None``; ``None`` if there are none."""
    known = _known(values)
    return (sum(known) / len(known)) if known else None


def _safe_div(numerator: float | None, denominator: float | int | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


@dataclass(frozen=True)
class AggregateStats:
    """Summary statistics for a set of evaluation results."""

    task_count: int
    successful_tasks: int
    success_rate: float | None

    # controlled-condition labels — set only when the group is homogeneous
    provider: str | None = None
    model: str | None = None
    tool_strategy: str | None = None
    context_strategy: str | None = None

    # token roll-ups
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_cached_input_tokens: int | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_total_tokens: float | None = None

    # latency
    avg_latency_ms: float | None = None
    total_latency_ms: float | None = None

    # call counts
    total_llm_calls: int | None = None
    total_tool_calls: int | None = None
    avg_llm_calls: float | None = None
    avg_tool_calls: float | None = None

    # context proxy (character count, NOT tokens)
    avg_context_char_count: float | None = None

    # cost (only when runs carried real pricing)
    cost_available: bool = False
    total_cost_usd: float | None = None
    cost_per_successful_task: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _homogeneous(values: Sequence[Any]) -> Any:
    """Return the single shared value, or ``None`` if they differ / are empty."""
    uniq = set(values)
    return next(iter(uniq)) if len(uniq) == 1 else None


def aggregate_results(results: Sequence[EvalResult]) -> AggregateStats:
    """Roll up *results* into one ``AggregateStats`` (None-safe throughout)."""
    task_count = len(results)
    successful = sum(1 for r in results if r.success)
    success_rate = _safe_div(successful, task_count)

    total_cost = _sum_opt(r.cost_usd for r in results if r.cost_available)
    cost_available = any(r.cost_available for r in results)

    return AggregateStats(
        task_count=task_count,
        successful_tasks=successful,
        success_rate=success_rate,
        provider=_homogeneous([r.provider for r in results]) if results else None,
        model=_homogeneous([r.model for r in results]) if results else None,
        tool_strategy=_homogeneous([r.tool_strategy for r in results]) if results else None,
        context_strategy=(
            _homogeneous([r.context_strategy for r in results]) if results else None
        ),
        total_input_tokens=_sum_opt(r.input_tokens for r in results),
        total_output_tokens=_sum_opt(r.output_tokens for r in results),
        total_tokens=_sum_opt(r.resolved_total_tokens for r in results),
        total_reasoning_tokens=_sum_opt(r.reasoning_tokens for r in results),
        total_cached_input_tokens=_sum_opt(r.cached_input_tokens for r in results),
        avg_input_tokens=_avg_opt(r.input_tokens for r in results),
        avg_output_tokens=_avg_opt(r.output_tokens for r in results),
        avg_total_tokens=_avg_opt(r.resolved_total_tokens for r in results),
        avg_latency_ms=_avg_opt(r.latency_ms for r in results),
        total_latency_ms=_sum_opt(r.latency_ms for r in results),
        total_llm_calls=_sum_opt(r.llm_calls for r in results),
        total_tool_calls=_sum_opt(r.tool_calls for r in results),
        avg_llm_calls=_avg_opt(r.llm_calls for r in results),
        avg_tool_calls=_avg_opt(r.tool_calls for r in results),
        avg_context_char_count=_avg_opt(r.context_char_count for r in results),
        cost_available=cost_available,
        total_cost_usd=total_cost,
        cost_per_successful_task=_safe_div(total_cost, successful),
    )


def group_results(results: Iterable[EvalResult]) -> dict[str, AggregateStats]:
    """Aggregate separately per ``"<tool_strategy>+<context_strategy>"`` key.

    This is the entry point for the eventual 2x2 comparison table; today it
    will normally contain only ``"fixed+raw"``.
    """
    buckets: dict[str, list[EvalResult]] = {}
    for r in results:
        buckets.setdefault(f"{r.tool_strategy}+{r.context_strategy}", []).append(r)
    return {key: aggregate_results(group) for key, group in sorted(buckets.items())}
