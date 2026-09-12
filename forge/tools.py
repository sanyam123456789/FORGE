"""
forge.tools — Tool registry and base class.

A "tool" in FORGE is a discrete capability that the LLM can invoke during
an agent run.  Examples: read_file, write_file, run_shell_command,
search_codebase.

This module defines:
- The ``Tool`` abstract base class that every tool must implement.
- The ``ToolResult`` dataclass that every tool call returns.
- The ``ToolRegistry`` that maps tool names to Tool instances.
- A global ``registry`` singleton that agent code uses.

Design Principles
-----------------
- Every tool is responsible for its own safety checks (calling safety.py).
- Tools are registered by name; the agent runtime looks them up by name.
- Tools declare a JSON-schema description so the LLM can reason about them.
- The ToolRegistry supports adaptive exposure: a subset of registered tools
  can be selected for a given run (the "adaptive tool exposure" research arm).

Step 1 Status
-------------
- Infrastructure is defined.
- NO concrete tools are implemented yet.
- Concrete tools (read_file, write_file, shell) will be added in Step 2.

Step 5 Status
-------------
- ``AdaptiveToolExposure`` is implemented below: a deterministic,
  keyword-based classifier that narrows the exposed tool set to what a task
  plausibly needs, with a safe "expose everything" fallback. See its
  docstring and ``docs/step-05-adaptive-tool-exposure.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from forge.logging import get_logger

if TYPE_CHECKING:
    from forge.context import ConversationContext

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """The outcome of a single tool invocation.

    output:
        String representation of the tool output (sent back to the LLM).
    success:
        False if the tool raised an error or safety violation.
    error:
        Error message when success=False (None otherwise).
    metadata:
        Arbitrary key-value pairs (e.g. bytes_written, lines_read).
        Collected by the observability layer for research instrumentation.
    """

    output: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, error: str) -> "ToolResult":
        """Convenience constructor for error results."""
        return cls(output=f"ERROR: {error}", success=False, error=error)


# ---------------------------------------------------------------------------
# Abstract Tool
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base class for all FORGE tools.

    Subclasses must implement:
    - ``name`` property
    - ``description`` property
    - ``parameters_schema`` property
    - ``execute()`` method

    Safety contract:
        execute() MUST call the appropriate safety.py guards before
        performing any filesystem or shell operation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (snake_case).  Used as the key in the registry."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the tool does.  Shown to the LLM."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema object describing the tool's input parameters."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments.

        Parameters are validated against ``parameters_schema`` by the
        caller (agent runtime) before this method is invoked.

        Returns
        -------
        ToolResult
            Always returns a ToolResult; never raises (errors are captured).
        """

    def to_schema(self) -> dict[str, Any]:
        """Return the full tool schema in OpenAI function-calling format.

        This is the format passed to the LLM so it knows what tools exist
        and how to call them.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Maps tool names to Tool instances and supports adaptive exposure.

    Usage
    -----
        registry.register(MyTool())
        tool = registry.get("my_tool")
        schemas = registry.get_schemas(["my_tool", "other_tool"])

    Adaptive Tool Exposure
    ----------------------
    The ``get_schemas(subset)`` method is the integration point for the
    adaptive tool exposure research arm.  When ``subset`` is None, all
    registered tools are returned.  When ``subset`` is a list of names,
    only those tools are exposed to the LLM for that turn/run.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a Tool instance.  Raises ValueError if name is already taken."""
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Use a unique name for each tool."
            )
        self._tools[tool.name] = tool
        logger.debug("Tool registered: %s", tool.name)

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name.  Raises KeyError if not found."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                f"Available tools: {sorted(self._tools.keys())}"
            )

    def list_names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def get_schemas(self, subset: list[str] | None = None) -> list[dict[str, Any]]:
        """Return tool schemas for passing to the LLM.

        Parameters
        ----------
        subset:
            If None, return schemas for all registered tools.
            If a list of names, return schemas only for those tools.
            (This is the adaptive tool exposure hook.)
        """
        if subset is None:
            tools = list(self._tools.values())
        else:
            tools = [self.get(name) for name in subset]
        return [t.to_schema() for t in tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Tool exposure strategies (research arm A seam)
# ---------------------------------------------------------------------------


class ToolExposureStrategy(ABC):
    """Decides which registered tools are exposed to the LLM for a turn.

    The agent runtime depends on this abstraction, never on adaptive logic
    directly.  Step 2 ships ``FixedToolExposure`` only.  An ``AdaptiveToolExposure``
    strategy (relevance-based subsetting) is a later research intervention and
    must slot in here without changes to the agent loop.
    """

    #: Stable identifier recorded in run traces.
    name: str = "abstract"

    @abstractmethod
    def select(
        self,
        registry: "ToolRegistry",
        context: "ConversationContext | None" = None,
        task: str | None = None,
    ) -> list[str] | None:
        """Return the tool names to expose, or ``None`` to expose all tools."""


class FixedToolExposure(ToolExposureStrategy):
    """Baseline: expose every registered tool on every turn.

    If ``pinned`` is given, exposes exactly those tools instead — still a
    static, non-adaptive choice (no ranking, no per-turn variation). This is
    a convenience for tests and manual experiments, not adaptive behaviour.
    """

    name = "fixed"

    def __init__(self, pinned: list[str] | None = None) -> None:
        self.pinned = pinned

    def select(
        self,
        registry: "ToolRegistry",
        context: "ConversationContext | None" = None,
        task: str | None = None,
    ) -> list[str] | None:
        return list(self.pinned) if self.pinned is not None else None


class AdaptiveToolExposure(ToolExposureStrategy):
    """Expose only the tools relevant to the current task (Step 5 intervention).

    Deliberately simple and fully deterministic — no ranking, no learned
    model, no per-turn variation:

    1. The task's natural-language prompt is classified into one of a small
       fixed set of *categories* by case-insensitive substring matching
       (``RULES``, checked in order; the first rule with a matching keyword
       wins).
    2. Each category maps to a fixed tuple of tool names (``CATEGORY_TOOLS``)
       — the tools that kind of task plausibly needs.
    3. The category's tools are intersected with what ``registry`` actually
       has, so a registry that does not carry every built-in tool never
       raises ``KeyError`` from ``ToolRegistry.get_schemas``.
    4. If no rule matches the task text, or nothing survives the
       registry intersection, ``select()`` returns ``None`` — the same
       "expose everything" behaviour as ``FixedToolExposure`` — rather than
       guessing. Adaptive exposure can therefore only ever *narrow* what a
       recognised task sees; an unrecognised task is never starved of a tool
       it might need.

    Classification looks only at the initial task text, never at ``context``:
    the same task string always yields the same tool subset, on every turn of
    a run and across separate runs. This is intentionally a placeholder for a
    more advanced policy (embedding similarity, per-turn re-ranking, a
    learned selector, ...) — everything downstream (the agent loop, the
    tracer, ``EvalResult``) depends only on the ``ToolExposureStrategy``
    interface, so a future replacement is a drop-in that changes nothing
    else.

    Limitations (Step 5)
    ---------------------
    - Keyword matching on the raw prompt is coarse: a task phrased
      unusually may fall into the "no category matched" fallback (safe, but
      not narrowed) or, in principle, an unintended category.
    - The policy is static and does not learn from the run's actual tool
      usage or from prior runs.
    - Only the five built-in tools are categorised; a newly added tool with
      no entry in ``CATEGORY_TOOLS`` is simply never offered by a matched
      category (it is still offered whenever the safe fallback applies).
    """

    name = "adaptive"

    #: category -> tool names relevant to that category. Tuples for
    #: immutability; order does not affect exposure (schemas are looked up by
    #: name), only readability.
    CATEGORY_TOOLS: dict[str, tuple[str, ...]] = {
        # A brand-new file: nothing to inspect or run yet, but read_file lets
        # the agent check the workspace state first if it chooses to.
        "create": ("read_file", "write_file", "list_directory"),
        # An existing file needs to be read and changed in place.
        "edit": ("read_file", "edit_file", "list_directory"),
        # The task explicitly involves running code (tests, scripts) and may
        # also need to create/modify files along the way.
        "test": (
            "read_file",
            "write_file",
            "edit_file",
            "list_directory",
            "run_shell",
        ),
    }

    #: Ordered (category, keywords) rules. Matching is a case-insensitive
    #: substring search over the raw task prompt; the first rule with a
    #: matching keyword wins. "test" is checked first: a task that both
    #: creates and *runs* a test file needs the full "test" toolset, not
    #: just "create".
    RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "test",
            (
                "pytest",
                "run_shell",
                "unit test",
                "unit-test",
                "test_",
                "run the tests",
                "run 'python",
                "confirm the tests",
                "tests pass",
            ),
        ),
        (
            "create",
            (
                "create a new file",
                "create a file named",
                "new file named",
                "create the file",
            ),
        ),
        (
            "edit",
            (
                "fix ",
                "rename",
                "refactor",
                "extract",
                "edit ",
                "modify",
                "update ",
                "change ",
                "add a ",
                "add an ",
            ),
        ),
    )

    def classify(self, task: str | None) -> str | None:
        """Return the matched category name, or ``None`` if nothing matched."""
        text = (task or "").lower()
        if not text.strip():
            return None
        for category, keywords in self.RULES:
            if any(keyword in text for keyword in keywords):
                return category
        return None

    def select(
        self,
        registry: "ToolRegistry",
        context: "ConversationContext | None" = None,
        task: str | None = None,
    ) -> list[str] | None:
        category = self.classify(task)
        if category is None:
            return None  # unrecognised task: safe fallback = expose everything
        candidates = self.CATEGORY_TOOLS[category]
        selected = [name for name in candidates if name in registry]
        return selected or None  # nothing survived -> same safe fallback


# ---------------------------------------------------------------------------
# Global registry singleton
# ---------------------------------------------------------------------------

registry = ToolRegistry()
