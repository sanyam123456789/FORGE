"""
Tests for forge.tools — ToolRegistry and Tool base class.
"""

from __future__ import annotations

from typing import Any

import pytest

from forge.tools import Tool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Minimal concrete tool for testing
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Returns the input message unchanged."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Text to echo."}
            },
            "required": ["message"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output=kwargs.get("message", ""))


class BrokenTool(Tool):
    @property
    def name(self) -> str:
        return "broken"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.from_error("Something went wrong")


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_defaults(self):
        r = ToolResult(output="done")
        assert r.success is True
        assert r.error is None

    def test_from_error(self):
        r = ToolResult.from_error("oops")
        assert r.success is False
        assert "oops" in r.output
        assert r.error == "oops"

    def test_metadata_defaults_empty(self):
        r = ToolResult(output="ok")
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_register_and_get(self, registry):
        registry.register(EchoTool())
        tool = registry.get("echo")
        assert tool.name == "echo"

    def test_duplicate_registration_raises(self, registry):
        registry.register(EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(EchoTool())

    def test_get_unknown_raises(self, registry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_list_names(self, registry):
        registry.register(EchoTool())
        registry.register(BrokenTool())
        assert registry.list_names() == ["broken", "echo"]

    def test_len(self, registry):
        assert len(registry) == 0
        registry.register(EchoTool())
        assert len(registry) == 1

    def test_contains(self, registry):
        registry.register(EchoTool())
        assert "echo" in registry
        assert "nonexistent" not in registry

    def test_get_schemas_all(self, registry):
        registry.register(EchoTool())
        registry.register(BrokenTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"echo", "broken"}

    def test_get_schemas_subset(self, registry):
        registry.register(EchoTool())
        registry.register(BrokenTool())
        schemas = registry.get_schemas(["echo"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "echo"

    def test_schema_structure(self, registry):
        registry.register(EchoTool())
        schema = registry.get_schemas()[0]
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


# ---------------------------------------------------------------------------
# EchoTool execution
# ---------------------------------------------------------------------------


class TestEchoTool:
    def test_execute_returns_message(self):
        tool = EchoTool()
        result = tool.execute(message="hello world")
        assert result.output == "hello world"
        assert result.success is True

    def test_to_schema(self):
        schema = EchoTool().to_schema()
        assert schema["function"]["name"] == "echo"
