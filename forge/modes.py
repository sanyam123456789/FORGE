"""
forge.modes — plain-language names for strategy combinations.

Each mode is one (tool exposure, context strategy) pair:

    baseline      all tools, full history every turn            fixed + raw
    lean-tools    only the tools the task needs                 adaptive + raw
    lean-context  managed (pruned/compressed) history           fixed + managed
    smart         lean tools + cache-friendly managed context   adaptive + stable

``smart`` is FORGE's full approach and the default for ``forge`` / ``forge
chat``. The research names (fixed/adaptive, raw/managed/stable) still work
everywhere they did before.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.context import (
    ContextStrategy,
    ManagedContextStrategy,
    RawContextStrategy,
    StableContextStrategy,
)
from forge.tools import AdaptiveToolExposure, FixedToolExposure, ToolExposureStrategy


@dataclass(frozen=True)
class Mode:
    name: str
    tool_strategy: str
    context_strategy: str
    description: str


MODES: dict[str, Mode] = {
    m.name: m
    for m in (
        Mode("baseline", "fixed", "raw", "all tools, full history every turn"),
        Mode("lean-tools", "adaptive", "raw", "only the tools the task needs"),
        Mode("lean-context", "fixed", "managed", "pruned and compressed history"),
        Mode("smart", "adaptive", "stable", "lean tools + cache-friendly managed context"),
    )
}

DEFAULT_MODE = "smart"


def get_mode(name: str) -> Mode:
    try:
        return MODES[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown mode {name!r}; choose one of: {', '.join(MODES)}"
        ) from None


def build_strategies(mode: Mode | str) -> tuple[ToolExposureStrategy, ContextStrategy]:
    """Fresh strategy objects for *mode* (context strategies hold state)."""
    if isinstance(mode, str):
        mode = get_mode(mode)
    tools: ToolExposureStrategy = (
        AdaptiveToolExposure() if mode.tool_strategy == "adaptive" else FixedToolExposure()
    )
    context: ContextStrategy
    if mode.context_strategy == "managed":
        context = ManagedContextStrategy()
    elif mode.context_strategy == "stable":
        context = StableContextStrategy()
    else:
        context = RawContextStrategy()
    return tools, context
