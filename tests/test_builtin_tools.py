"""
Tests for forge.builtin_tools — the five built-in coding tools.

Every tool is bound to a tmp_path workspace so no test touches the real repo.
"""

from __future__ import annotations

import sys

import pytest

from forge.builtin_tools import (
    BUILTIN_TOOL_CLASSES,
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    RunShellTool,
    WriteFileTool,
    build_default_registry,
    register_builtin_tools,
)
from forge.tools import ToolRegistry, ToolResult


@pytest.fixture
def ws(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_existing_file(self, ws):
        (ws / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
        result = ReadFileTool(workspace_root=ws).execute(path="a.txt")
        assert result.success is True
        assert result.output == "hello\nworld\n"
        assert result.metadata["path"] == "a.txt"
        assert result.metadata["bytes"] == 12

    def test_missing_file_is_structured_error(self, ws):
        result = ReadFileTool(workspace_root=ws).execute(path="nope.txt")
        assert result.success is False
        assert "not found" in result.error

    def test_directory_path_is_error(self, ws):
        (ws / "sub").mkdir()
        result = ReadFileTool(workspace_root=ws).execute(path="sub")
        assert result.success is False
        assert "directory" in result.error

    def test_missing_argument(self, ws):
        result = ReadFileTool(workspace_root=ws).execute()
        assert result.success is False
        assert "path" in result.error

    def test_non_string_argument(self, ws):
        result = ReadFileTool(workspace_root=ws).execute(path=123)
        assert result.success is False
        assert "must be a string" in result.error

    def test_escape_workspace_blocked(self, ws):
        result = ReadFileTool(workspace_root=ws).execute(path="../secret.txt")
        assert result.success is False
        assert "safety violation" in result.error

    def test_unknown_argument_rejected(self, ws):
        (ws / "a.txt").write_text("x", encoding="utf-8")
        result = ReadFileTool(workspace_root=ws).execute(path="a.txt", mode="r")
        assert result.success is False
        assert "unexpected argument" in result.error


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_creates_new_file(self, ws):
        result = WriteFileTool(workspace_root=ws).execute(path="pkg/mod.py", content="x = 1\n")
        assert result.success is True
        assert (ws / "pkg" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
        assert result.metadata["created"] is True
        assert result.metadata["bytes_written"] == 6

    def test_overwrite_requires_flag(self, ws):
        (ws / "a.py").write_text("old", encoding="utf-8")
        result = WriteFileTool(workspace_root=ws).execute(path="a.py", content="new")
        assert result.success is False
        assert "allow_overwrite" in result.error or "overwrite" in result.error
        assert (ws / "a.py").read_text(encoding="utf-8") == "old"

    def test_overwrite_with_flag(self, ws):
        (ws / "a.py").write_text("old", encoding="utf-8")
        result = WriteFileTool(workspace_root=ws).execute(
            path="a.py", content="new", overwrite=True
        )
        assert result.success is True
        assert result.metadata["overwritten"] is True
        assert (ws / "a.py").read_text(encoding="utf-8") == "new"

    def test_missing_content_argument(self, ws):
        result = WriteFileTool(workspace_root=ws).execute(path="a.py")
        assert result.success is False
        assert "content" in result.error

    def test_escape_workspace_blocked(self, ws):
        result = WriteFileTool(workspace_root=ws).execute(path="../evil.py", content="x")
        assert result.success is False
        assert "safety violation" in result.error


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_replaces_unique_substring(self, ws):
        (ws / "m.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        result = EditFileTool(workspace_root=ws).execute(
            path="m.py", old_string="a + b", new_string="a - b"
        )
        assert result.success is True
        assert result.metadata["replacements"] == 1
        assert (ws / "m.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a - b\n"

    def test_missing_old_string_is_error(self, ws):
        (ws / "m.py").write_text("hello", encoding="utf-8")
        result = EditFileTool(workspace_root=ws).execute(
            path="m.py", old_string="absent", new_string="x"
        )
        assert result.success is False
        assert "not found" in result.error

    def test_ambiguous_without_replace_all(self, ws):
        (ws / "m.py").write_text("x\nx\nx\n", encoding="utf-8")
        result = EditFileTool(workspace_root=ws).execute(
            path="m.py", old_string="x", new_string="y"
        )
        assert result.success is False
        assert "not unique" in result.error

    def test_replace_all(self, ws):
        (ws / "m.py").write_text("x\nx\nx\n", encoding="utf-8")
        result = EditFileTool(workspace_root=ws).execute(
            path="m.py", old_string="x", new_string="y", replace_all=True
        )
        assert result.success is True
        assert result.metadata["replacements"] == 3
        assert (ws / "m.py").read_text(encoding="utf-8") == "y\ny\ny\n"

    def test_missing_file(self, ws):
        result = EditFileTool(workspace_root=ws).execute(
            path="nope.py", old_string="a", new_string="b"
        )
        assert result.success is False
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    def test_lists_entries(self, ws):
        (ws / "a.py").write_text("1", encoding="utf-8")
        (ws / "b.py").write_text("22", encoding="utf-8")
        (ws / "sub").mkdir()
        result = ListDirectoryTool(workspace_root=ws).execute()
        assert result.success is True
        assert "sub/" in result.output
        assert "a.py" in result.output
        assert result.metadata["dirs"] == 1
        assert result.metadata["files"] == 2

    def test_default_path_is_workspace_root(self, ws):
        result = ListDirectoryTool(workspace_root=ws).execute()
        assert result.success is True
        assert result.metadata["path"] == "."

    def test_empty_directory(self, ws):
        (ws / "empty").mkdir()
        result = ListDirectoryTool(workspace_root=ws).execute(path="empty")
        assert result.success is True
        assert "empty directory" in result.output

    def test_missing_directory(self, ws):
        result = ListDirectoryTool(workspace_root=ws).execute(path="nope")
        assert result.success is False
        assert "not found" in result.error

    def test_file_path_is_error(self, ws):
        (ws / "a.py").write_text("1", encoding="utf-8")
        result = ListDirectoryTool(workspace_root=ws).execute(path="a.py")
        assert result.success is False
        assert "not a directory" in result.error

    def test_escape_blocked(self, ws):
        result = ListDirectoryTool(workspace_root=ws).execute(path="..")
        assert result.success is False
        assert "safety violation" in result.error


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------


class TestRunShell:
    def test_runs_command_and_captures_output(self, ws):
        result = RunShellTool(workspace_root=ws).execute(
            command=f'{sys.executable} -c "print(2 + 3)"'
        )
        assert result.success is True
        assert "5" in result.output
        assert result.metadata["exit_code"] == 0

    def test_nonzero_exit_is_reported_as_failure(self, ws):
        result = RunShellTool(workspace_root=ws).execute(
            command=f'{sys.executable} -c "import sys; sys.exit(3)"'
        )
        assert result.success is False
        assert result.metadata["exit_code"] == 3
        assert "exit code 3" in result.error

    def test_blocked_command_is_structured_error(self, ws):
        result = RunShellTool(workspace_root=ws).execute(command="rm -rf /")
        assert result.success is False
        assert "safety violation" in result.error
        assert "blocked list" in result.error

    def test_network_command_blocked(self, ws):
        result = RunShellTool(workspace_root=ws).execute(command="curl http://example.com")
        assert result.success is False
        assert "safety violation" in result.error

    def test_missing_command_argument(self, ws):
        result = RunShellTool(workspace_root=ws).execute()
        assert result.success is False
        assert "command" in result.error

    def test_unknown_executable(self, ws):
        result = RunShellTool(workspace_root=ws).execute(
            command="this_executable_does_not_exist_xyz --help"
        )
        assert result.success is False
        assert "not found" in result.error

    def test_runs_in_workspace_cwd(self, ws):
        (ws / "marker.txt").write_text("here", encoding="utf-8")
        result = RunShellTool(workspace_root=ws).execute(
            command=f'{sys.executable} -c "import os; print(os.path.exists(\'marker.txt\'))"'
        )
        assert result.success is True
        assert "True" in result.output


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_build_default_registry_has_five_tools(self, ws):
        registry = build_default_registry(workspace_root=ws)
        assert isinstance(registry, ToolRegistry)
        assert registry.list_names() == [
            "edit_file",
            "list_directory",
            "read_file",
            "run_shell",
            "write_file",
        ]

    def test_register_builtin_tools_on_existing_registry(self, ws):
        registry = ToolRegistry()
        register_builtin_tools(registry, workspace_root=ws)
        assert len(registry) == 5

    def test_all_tools_expose_valid_schema(self, ws):
        registry = build_default_registry(workspace_root=ws)
        for schema in registry.get_schemas():
            assert schema["type"] == "function"
            fn = schema["function"]
            assert isinstance(fn["name"], str) and fn["name"]
            assert isinstance(fn["description"], str) and fn["description"]
            assert fn["parameters"]["type"] == "object"

    def test_every_builtin_returns_toolresult_on_bad_input(self, ws):
        for cls in BUILTIN_TOOL_CLASSES:
            result = cls(workspace_root=ws).execute(bogus_arg=1)
            assert isinstance(result, ToolResult)
            assert result.success is False
