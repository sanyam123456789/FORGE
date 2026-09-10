"""
forge.evaluation.experiment — the controlled configuration for one eval run.

An ``ExperimentConfig`` is the single object that pins every variable held
constant during the research comparison, plus the two that will vary:

    held constant : provider, model, temperature, max_output_tokens,
                    max_llm_turns, max_tool_calls, the task
    varied later  : tool_strategy  (fixed  -> adaptive)
                    context_strategy (raw  -> managed)

Only ``fixed`` / ``raw`` are implemented in this step.  ``adaptive`` and
``managed`` are recognised names (so a future result file is unambiguous)
but ``resolve_strategies()`` raises ``NotImplementedError`` for them — this
package measures; it does not add agent behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge.context import ContextStrategy, RawContextStrategy
from forge.tools import FixedToolExposure, ToolExposureStrategy

# --- Strategy identifiers -------------------------------------------------

TOOL_STRATEGY_FIXED = "fixed"
TOOL_STRATEGY_ADAPTIVE = "adaptive"

CONTEXT_STRATEGY_RAW = "raw"
CONTEXT_STRATEGY_MANAGED = "managed"

#: Strategies this step can actually execute.
IMPLEMENTED_TOOL_STRATEGIES: frozenset[str] = frozenset({TOOL_STRATEGY_FIXED})
IMPLEMENTED_CONTEXT_STRATEGIES: frozenset[str] = frozenset({CONTEXT_STRATEGY_RAW})

#: Strategies that are planned but NOT implemented — allowed as labels only.
FUTURE_TOOL_STRATEGIES: frozenset[str] = frozenset({TOOL_STRATEGY_ADAPTIVE})
FUTURE_CONTEXT_STRATEGIES: frozenset[str] = frozenset({CONTEXT_STRATEGY_MANAGED})

_KNOWN_TOOL_STRATEGIES = IMPLEMENTED_TOOL_STRATEGIES | FUTURE_TOOL_STRATEGIES
_KNOWN_CONTEXT_STRATEGIES = IMPLEMENTED_CONTEXT_STRATEGIES | FUTURE_CONTEXT_STRATEGIES


@dataclass(frozen=True)
class ExperimentConfig:
    """One fully-specified, comparable run configuration.

    task_id:
        Which ``EvalTask`` this run exercises (identity only — the prompt is
        carried by the task itself).
    provider / model:
        The provider id and model name.  Prefer ``from_settings`` so these
        come from ``forge.config`` rather than being hand-typed.
    tool_strategy / context_strategy:
        The two research variables.  Default to the only implemented pair,
        ``fixed`` / ``raw``.
    temperature / max_output_tokens / max_llm_turns / max_tool_calls:
        Generation and loop parameters held constant across a comparison.
        ``None`` for the two ceilings means "defer to the runtime/global
        settings" and is recorded as such.
    run_id:
        Optional; the runner assigns one when it is ``None``.
    label:
        Optional human tag for a batch (e.g. "pilot-2026-09").  Not a control
        key.
    extra:
        Free-form annotations recorded with the result.
    """

    task_id: str
    provider: str
    model: str
    tool_strategy: str = TOOL_STRATEGY_FIXED
    context_strategy: str = CONTEXT_STRATEGY_RAW
    temperature: float = 0.0
    max_output_tokens: int = 4096
    max_llm_turns: int | None = None
    max_tool_calls: int | None = None
    run_id: str | None = None
    label: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tool_strategy not in _KNOWN_TOOL_STRATEGIES:
            raise ValueError(
                f"unknown tool_strategy {self.tool_strategy!r}; "
                f"known: {sorted(_KNOWN_TOOL_STRATEGIES)}"
            )
        if self.context_strategy not in _KNOWN_CONTEXT_STRATEGIES:
            raise ValueError(
                f"unknown context_strategy {self.context_strategy!r}; "
                f"known: {sorted(_KNOWN_CONTEXT_STRATEGIES)}"
            )
        if not self.task_id or not self.task_id.strip():
            raise ValueError("ExperimentConfig.task_id must be non-empty")

    # -- capability introspection -------------------------------------

    @property
    def is_implemented(self) -> bool:
        """True when both requested strategies can actually run in this step."""
        return (
            self.tool_strategy in IMPLEMENTED_TOOL_STRATEGIES
            and self.context_strategy in IMPLEMENTED_CONTEXT_STRATEGIES
        )

    @property
    def strategy_key(self) -> str:
        """Compact ``"<tool>+<context>"`` key for grouping results."""
        return f"{self.tool_strategy}+{self.context_strategy}"

    def resolve_strategies(self) -> tuple[ToolExposureStrategy, ContextStrategy]:
        """Instantiate the concrete strategy objects for the agent runtime.

        Raises
        ------
        NotImplementedError
            If either requested strategy is a recognised future name that this
            step does not implement.
        """
        if self.tool_strategy not in IMPLEMENTED_TOOL_STRATEGIES:
            raise NotImplementedError(
                f"tool_strategy {self.tool_strategy!r} is not implemented in this "
                f"step. Implemented: {sorted(IMPLEMENTED_TOOL_STRATEGIES)}. "
                f"Planned: {sorted(FUTURE_TOOL_STRATEGIES)}."
            )
        if self.context_strategy not in IMPLEMENTED_CONTEXT_STRATEGIES:
            raise NotImplementedError(
                f"context_strategy {self.context_strategy!r} is not implemented in "
                f"this step. Implemented: {sorted(IMPLEMENTED_CONTEXT_STRATEGIES)}. "
                f"Planned: {sorted(FUTURE_CONTEXT_STRATEGIES)}."
            )
        return FixedToolExposure(), RawContextStrategy()

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "provider": self.provider,
            "model": self.model,
            "tool_strategy": self.tool_strategy,
            "context_strategy": self.context_strategy,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_llm_turns": self.max_llm_turns,
            "max_tool_calls": self.max_tool_calls,
            "run_id": self.run_id,
            "label": self.label,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            task_id=data["task_id"],
            provider=data["provider"],
            model=data["model"],
            tool_strategy=data.get("tool_strategy", TOOL_STRATEGY_FIXED),
            context_strategy=data.get("context_strategy", CONTEXT_STRATEGY_RAW),
            temperature=data.get("temperature", 0.0),
            max_output_tokens=data.get("max_output_tokens", 4096),
            max_llm_turns=data.get("max_llm_turns"),
            max_tool_calls=data.get("max_tool_calls"),
            run_id=data.get("run_id"),
            label=data.get("label"),
            extra=dict(data.get("extra", {}) or {}),
        )

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        task_id: str,
        tool_strategy: str = TOOL_STRATEGY_FIXED,
        context_strategy: str = CONTEXT_STRATEGY_RAW,
        run_id: str | None = None,
        label: str | None = None,
        max_llm_turns: int | None = None,
        max_tool_calls: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "ExperimentConfig":
        """Build a config from a ``ForgeSettings``-like object.

        provider / model / temperature / max_output_tokens are taken from
        *settings* so they are never hard-coded here.  The two ceilings fall
        back to the settings values when not given explicitly.
        """
        return cls(
            task_id=task_id,
            provider=settings.llm_provider,
            model=settings.llm_model,
            tool_strategy=tool_strategy,
            context_strategy=context_strategy,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            max_llm_turns=(
                max_llm_turns
                if max_llm_turns is not None
                else getattr(settings, "max_llm_turns", None)
            ),
            max_tool_calls=(
                max_tool_calls
                if max_tool_calls is not None
                else getattr(settings, "max_tool_calls", None)
            ),
            run_id=run_id,
            label=label,
            extra=dict(extra or {}),
        )
