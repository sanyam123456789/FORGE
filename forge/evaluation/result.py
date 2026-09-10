"""
forge.evaluation.result — the structured result of one evaluation run.

An ``EvalResult`` is one comparable row: what was run, under which controlled
configuration, and every metric the run produced.  It follows the same rule
as ``forge.observability``: a value the provider/runtime did not report is
``None``, never ``0`` and never invented.  Cost is ``None`` unless a caller
supplied real pricing — FORGE ships no price table.

Serialisation is JSONL (one JSON object per line) to match the existing
trace format, so many results collect into a single file for later analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Bump when the on-disk shape changes in a non-additive way.
RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TokenPricing:
    """USD price per 1,000,000 tokens.  Supplied by the caller, never guessed.

    Only used when a caller explicitly passes it; otherwise cost stays
    unavailable.  ``cached_input_per_mtok`` is optional and only applied when
    the run reported ``cached_input_tokens``.
    """

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    source: str | None = None  # free-text note, e.g. "vendor pricing page 2026-09"

    def cost_for(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None = None,
    ) -> float | None:
        """Return total USD cost, or ``None`` if the inputs are insufficient.

        Requires at least one of input/output token counts to be known.  A
        missing count contributes nothing (it is not treated as zero unless
        the other count is present — if *both* are ``None`` the result is
        ``None``).
        """
        if input_tokens is None and output_tokens is None:
            return None
        cost = 0.0
        if input_tokens is not None:
            billable_input = input_tokens
            if (
                cached_input_tokens is not None
                and self.cached_input_per_mtok is not None
            ):
                billable_input = max(input_tokens - cached_input_tokens, 0)
                cost += (cached_input_tokens / 1_000_000) * self.cached_input_per_mtok
            cost += (billable_input / 1_000_000) * self.input_per_mtok
        if output_tokens is not None:
            cost += (output_tokens / 1_000_000) * self.output_per_mtok
        return cost


@dataclass
class EvalResult:
    """Metrics for a single completed (or failed) evaluation run."""

    # -- identity / controlled conditions ------------------------------
    task_id: str
    run_id: str
    provider: str | None
    model: str | None
    tool_strategy: str
    context_strategy: str

    # -- outcome ------------------------------------------------------
    status: str
    success: bool
    stop_reason: str | None = None
    error: str | None = None

    # -- primary research metrics -----------------------------------
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None

    # -- secondary metrics ----------------------------------------------
    llm_calls: int | None = None
    tool_calls: int | None = None
    turns: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None

    # context consumption — CHARACTER PROXY, not a tokenizer count.
    context_messages: int | None = None
    context_char_count: int | None = None

    # -- cost (unavailable unless real pricing was supplied) -----------
    cost_available: bool = False
    cost_usd: float | None = None
    cost_source: str | None = None

    # -- reproducibility metadata -------------------------------------
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_llm_turns: int | None = None
    max_tool_calls: int | None = None
    workspace: str | None = None
    label: str | None = None

    # -- timing / artefacts ----------------------------------------------
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    trace_path: str | None = None

    # -- task checks ---------------------------------------------------
    checks_run: int = 0
    checks_passed: bool | None = None
    checks_detail: list[str] = field(default_factory=list)

    schema_version: int = RESULT_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def resolved_total_tokens(self) -> int | None:
        """Provider total if present, else input+output, else ``None``."""
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# JSONL collection helpers
# ---------------------------------------------------------------------------


def write_results_jsonl(
    results: Iterable[EvalResult],
    path: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write *results* as one JSON object per line.  Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        for result in results:
            fh.write(result.to_json() + "\n")
    return path


def append_result_jsonl(result: EvalResult, path: str | Path) -> Path:
    """Append a single result to a JSONL file (creating it if needed)."""
    return write_results_jsonl([result], path, append=True)


def read_results_jsonl(path: str | Path) -> list[EvalResult]:
    """Read a JSONL results file back into ``EvalResult`` objects.

    Blank lines are skipped.  A malformed line raises ``ValueError`` naming
    the line number — silently dropping data would undermine the point of the
    measurement layer.
    """
    path = Path(path)
    out: list[EvalResult] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(EvalResult.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{lineno}: malformed EvalResult line ({exc})") from exc
    return out
