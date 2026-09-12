"""
Tests for forge.tools — ToolRegistry and Tool base class.
"""

from __future__ import annotations

from typing import Any

import pytest

from forge.tools import AdaptiveToolExposure, Tool, ToolRegistry, ToolResult


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


# ---------------------------------------------------------------------------
# AdaptiveToolExposure (Step 5)
# ---------------------------------------------------------------------------


@pytest.fixture
def builtin_registry(tmp_path):
    from forge.builtin_tools import build_default_registry

    return build_default_registry(workspace_root=tmp_path)


class TestAdaptiveToolExposureClassification:
    def test_name_is_adaptive(self):
        assert AdaptiveToolExposure().name == "adaptive"

    def test_create_task_classified_as_create(self):
        strategy = AdaptiveToolExposure()
        assert strategy.classify("Create a new file named foo.py with a hello().") == "create"

    def test_bug_fix_task_classified_as_edit(self):
        strategy = AdaptiveToolExposure()
        assert strategy.classify("Fix the off-by-one bug in ranges.py.") == "edit"

    def test_rename_and_refactor_tasks_classified_as_edit(self):
        strategy = AdaptiveToolExposure()
        assert strategy.classify("Rename the function tally to count_items.") == "edit"
        assert strategy.classify("Extract the duplicated logic into a helper.") == "edit"

    def test_testing_task_classified_as_test(self):
        strategy = AdaptiveToolExposure()
        text = "Write pytest tests and run 'python -m pytest -q' to confirm the tests pass."
        assert strategy.classify(text) == "test"

    def test_test_keywords_take_priority_over_create_keywords(self):
        # A task that both creates a file and mentions pytest must get the
        # full "test" toolset, not the narrower "create" one.
        strategy = AdaptiveToolExposure()
        text = "Create a new file named test_mathlib.py with pytest tests."
        assert strategy.classify(text) == "test"

    def test_unrecognised_task_returns_none(self):
        strategy = AdaptiveToolExposure()
        assert strategy.classify("Please help me understand the moon phases.") is None

    def test_empty_or_missing_task_returns_none(self):
        strategy = AdaptiveToolExposure()
        assert strategy.classify("") is None
        assert strategy.classify("   ") is None
        assert strategy.classify(None) is None

    def test_classification_is_case_insensitive(self):
        strategy = AdaptiveToolExposure()
        assert strategy.classify("FIX THE BUG IN ranges.py") == "edit"


class TestAdaptiveToolExposureSelection:
    def test_create_task_selects_creation_tools(self, builtin_registry):
        strategy = AdaptiveToolExposure()
        selected = strategy.select(
            builtin_registry, task="Create a new file named foo.py with a hello()."
        )
        assert set(selected) == {"read_file", "write_file", "list_directory"}
        assert "edit_file" not in selected
        assert "run_shell" not in selected

    def test_edit_task_selects_editing_tools(self, builtin_registry):
        strategy = AdaptiveToolExposure()
        selected = strategy.select(
            builtin_registry, task="Fix the off-by-one bug in ranges.py."
        )
        assert set(selected) == {"read_file", "edit_file", "list_directory"}
        assert "write_file" not in selected
        assert "run_shell" not in selected

    def test_testing_task_selects_full_toolset(self, builtin_registry):
        strategy = AdaptiveToolExposure()
        text = "Write pytest tests and run 'python -m pytest -q' to confirm the tests pass."
        selected = strategy.select(builtin_registry, task=text)
        assert set(selected) == set(builtin_registry.list_names())

    def test_unrecognised_task_falls_back_to_all_tools(self, builtin_registry):
        strategy = AdaptiveToolExposure()
        selected = strategy.select(
            builtin_registry, task="Please help me understand the moon phases."
        )
        assert selected is None  # same "expose everything" contract as Fixed

    def test_none_task_falls_back_to_all_tools(self, builtin_registry):
        strategy = AdaptiveToolExposure()
        assert strategy.select(builtin_registry, task=None) is None

    def test_selection_is_deterministic(self, builtin_registry):
        strategy = AdaptiveToolExposure()
        task = "Fix the off-by-one bug in ranges.py."
        first = strategy.select(builtin_registry, task=task)
        second = strategy.select(builtin_registry, task=task)
        assert first == second

    def test_candidates_are_filtered_against_a_partial_registry(self, tmp_path):
        from forge.builtin_tools import ReadFileTool, WriteFileTool

        partial = ToolRegistry()
        partial.register(ReadFileTool(workspace_root=tmp_path))
        partial.register(WriteFileTool(workspace_root=tmp_path))

        strategy = AdaptiveToolExposure()
        text = "Write pytest tests and run 'python -m pytest -q' to confirm the tests pass."
        selected = strategy.select(partial, task=text)
        # "test" category wants 5 tools; only 2 are registered -> no crash,
        # and only the registered ones come back.
        assert selected == ["read_file", "write_file"]

    def test_fallback_when_nothing_survives_registry_filtering(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        strategy = AdaptiveToolExposure()
        selected = strategy.select(registry, task="Fix the off-by-one bug in ranges.py.")
        assert selected is None  # none of the "edit" candidates are registered

    def test_ambiguous_task_does_not_crash(self, builtin_registry):
        strategy = AdaptiveToolExposure()
        for weird in ("", "   ", "???", "🚀" * 5, "a" * 5000):
            selected = strategy.select(builtin_registry, task=weird)
            assert selected is None or isinstance(selected, list)
