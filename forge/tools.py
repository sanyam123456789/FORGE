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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from forge.logging import get_logger

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
# Global registry singleton
# ---------------------------------------------------------------------------

registry = ToolRegistry()
