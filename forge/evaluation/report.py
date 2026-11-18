"""
forge.evaluation.report — a plain, readable per-arm summary of results.

``forge report`` prints, for every arm in a results JSONL file:

    runs, solved, success rate,
    mean total tokens, mean uncached input tokens, cache hit rate,
    tokens per solved task, mean peak input of one call,
    change vs the baseline arm, run-to-run spread,
    mean LLM calls, mean duration.

"vs base" compares like with like: for each task, the arm's mean tokens over
its repetitions divided by the baseline arm's (``fixed_raw``) mean on the
same task, averaged over tasks. "spread" is the typical run-to-run variation
of one task under one arm: the coefficient of variation (standard deviation
/ mean) of total tokens across repetitions, averaged over tasks. It needs at
least two repetitions (``forge bench --repeat N``). A difference between two
arms smaller than their spread is not a reliable result.

"peak in" is the largest input of a single LLM call, averaged over runs: the
context size that context management is meant to keep bounded.

"Tokens per solved task" (all tokens spent by the arm divided by tasks it
solved) is the headline efficiency number: it charges an arm for the tokens
of its failed attempts too. "Uncached input" is the input the provider had
to process at full price; cached input is billed at a fraction of it.

Only provider-reported numbers are used. A metric no row reported is shown
as "n/a", never estimated. For the formal statistical comparison of the 2x2
arms, use ``forge analyze``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from forge.evaluation.matrix import ALL_ARMS
from forge.evaluation.result import EvalResult, read_results_jsonl


@dataclass(frozen=True)
class ArmReport:
    arm_id: str
    tasks: int
    solved: int
    mean_total_tokens: float | None
    mean_uncached_input: float | None
    cache_hit_rate: float | None
    tokens_per_solved: float | None
    mean_llm_calls: float | None
    mean_duration_s: float | None
    mean_peak_input: float | None = None
    vs_baseline: float | None = None
    spread: float | None = None
    repetitions: int = 1

    @property
    def success_rate(self) -> float | None:
        return self.solved / self.tasks if self.tasks else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


BASELINE_ARM = "fixed_raw"


def _tokens_by_task(rows: list[EvalResult]) -> dict[str, list[int]]:
    by_task: dict[str, list[int]] = {}
    for r in rows:
        if r.resolved_total_tokens is not None:
            by_task.setdefault(r.task_id, []).append(r.resolved_total_tokens)
    return by_task


def _spread(rows: list[EvalResult]) -> float | None:
    cvs = [
        statistics.stdev(v) / statistics.mean(v)
        for v in _tokens_by_task(rows).values()
        if len(v) >= 2 and statistics.mean(v) > 0
    ]
    return _mean(cvs)


def _vs_baseline(rows: list[EvalResult], baseline: list[EvalResult]) -> float | None:
    mine = {t: statistics.mean(v) for t, v in _tokens_by_task(rows).items()}
    base = {t: statistics.mean(v) for t, v in _tokens_by_task(baseline).items()}
    return _mean([mine[t] / base[t] - 1 for t in mine if t in base and base[t] > 0])


def summarize_arm(
    arm_id: str, rows: list[EvalResult], baseline: list[EvalResult] | None = None
) -> ArmReport:
    totals = [r.resolved_total_tokens for r in rows if r.resolved_total_tokens is not None]
    with_cache = [r for r in rows if r.input_tokens and r.cached_input_tokens is not None]
    solved = sum(1 for r in rows if r.success)
    cache_rate = (
        sum(r.cached_input_tokens for r in with_cache) / sum(r.input_tokens for r in with_cache)
        if with_cache
        else None
    )
    return ArmReport(
        arm_id=arm_id,
        tasks=len(rows),
        solved=solved,
        mean_total_tokens=_mean(totals),
        mean_uncached_input=_mean(
            [r.input_tokens - r.cached_input_tokens for r in with_cache]
        ),
        cache_hit_rate=cache_rate,
        tokens_per_solved=(sum(totals) / solved) if solved and totals else None,
        mean_llm_calls=_mean([r.llm_calls for r in rows if r.llm_calls is not None]),
        mean_duration_s=_mean([r.duration_s for r in rows if r.duration_s is not None]),
        mean_peak_input=_mean(
            [r.peak_input_tokens for r in rows if r.peak_input_tokens is not None]
        ),
        vs_baseline=(
            _vs_baseline(rows, baseline)
            if baseline is not None and arm_id != BASELINE_ARM
            else None
        ),
        spread=_spread(rows),
        repetitions=max((r.repetition or 1 for r in rows), default=1),
    )


def build_report(rows: list[EvalResult]) -> list[ArmReport]:
    by_arm: dict[str, list[EvalResult]] = {}
    for r in rows:
        arm = r.arm_id or f"{r.tool_strategy}_{r.context_strategy}"
        by_arm.setdefault(arm, []).append(r)
    order = [a.arm_id for a in ALL_ARMS]
    arms = sorted(by_arm, key=lambda a: (order.index(a) if a in order else len(order), a))
    baseline = by_arm.get(BASELINE_ARM)
    return [summarize_arm(a, by_arm[a], baseline) for a in arms]


def latest_results_file(root: Path = Path("runs")) -> Path | None:
    """Most recently modified results.jsonl under *root*, if any."""
    candidates = [p for p in root.rglob("results.jsonl") if p.is_file()] if root.is_dir() else []
    if not candidates:
        legacy = root / "eval_results.jsonl"
        return legacy if legacy.is_file() else None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _fmt(value: float | None, kind: str = "num") -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "signed":
        return f"{value:+.0%}"
    if kind == "pm":
        return f"±{value:.0%}"
    if kind == "sec":
        return f"{value:.1f}s"
    if kind == "dec":
        return f"{value:.1f}"
    return f"{value:,.0f}"


def format_report(reports: list[ArmReport]) -> str:
    header = (
        f"  {'arm':<18} {'solved':>8} {'avg tokens':>11} {'vs base':>8} {'spread':>7} "
        f"{'uncached in':>12} {'cached':>7} {'peak in':>8} {'tok/solved':>11} "
        f"{'calls':>6} {'time':>7}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for r in reports:
        solved = f"{r.solved}/{r.tasks}"
        vs = "base" if r.arm_id == BASELINE_ARM else _fmt(r.vs_baseline, "signed")
        lines.append(
            f"  {r.arm_id:<18} {solved:>8} {_fmt(r.mean_total_tokens):>11} {vs:>8} "
            f"{_fmt(r.spread, 'pm'):>7} {_fmt(r.mean_uncached_input):>12} "
            f"{_fmt(r.cache_hit_rate, 'pct'):>7} {_fmt(r.mean_peak_input):>8} "
            f"{_fmt(r.tokens_per_solved):>11} {_fmt(r.mean_llm_calls, 'dec'):>6} "
            f"{_fmt(r.mean_duration_s, 'sec'):>7}"
        )
    reps = max((r.repetitions for r in reports), default=1)
    if reps > 1:
        lines.append("")
        lines.append(f"  {reps} repetitions per task and arm")
    return "\n".join(lines)


def report_file(path: str | Path) -> tuple[list[ArmReport], str]:
    rows = read_results_jsonl(path)
    reports = build_report(rows)
    return reports, format_report(reports)
