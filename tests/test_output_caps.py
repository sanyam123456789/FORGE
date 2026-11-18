"""
Tests for read_file / run_shell output caps (Fix 2).

Shell output is generated with ``sys.executable`` running a small script in
the tmp workspace, so the tests work on Windows and POSIX alike.
"""

from __future__ import annotations

import re
import sys

import pytest

from forge.builtin_tools import (
    READ_FILE_MAX_CHARS,
    READ_FILE_MAX_LINES,
    ReadFileTool,
    RunShellTool,
)
from forge.context import _is_error_tool_message
from forge.llm import Message

_HEADER_RE = re.compile(
    r"^\[read_file: showing lines (\d+)-(\d+) of (\d+) total\. "
    r"Call read_file with start_line=(\d+) to read more\.\]$"
)


def _numbered(n: int) -> str:
    return "".join(f"line {i}\n" for i in range(1, n + 1))


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFileCaps:
    def test_small_file_identical_to_old_behavior(self, tmp_path):
        content = "alpha\nbeta\n\ngamma"  # no trailing newline
        (tmp_path / "s.txt").write_text(content, encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="s.txt")
        assert result.output == content
        assert result.metadata["truncated"] is False
        assert result.metadata["total_lines"] == 4
        assert result.metadata["lines"] == 4
        assert (result.metadata["start_line"], result.metadata["end_line"]) == (1, 4)

    def test_file_exactly_at_line_cap_is_unchanged(self, tmp_path):
        content = _numbered(READ_FILE_MAX_LINES)
        (tmp_path / "f.txt").write_text(content, encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="f.txt")
        assert result.output == content

    def test_empty_file(self, tmp_path):
        (tmp_path / "e.txt").write_text("", encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="e.txt")
        assert result.success is True
        assert result.output == ""

    def test_large_file_capped_with_header(self, tmp_path):
        (tmp_path / "big.txt").write_text(_numbered(2000), encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="big.txt")
        lines = result.output.split("\n")
        assert lines[0] == (
            "[read_file: showing lines 1-400 of 2000 total. "
            "Call read_file with start_line=401 to read more.]"
        )
        assert lines[1] == "line 1"
        assert lines[400] == "line 400"
        assert result.output.endswith("line 400\n")
        assert result.metadata["truncated"] is True
        assert result.metadata["total_lines"] == 2000
        assert result.metadata["end_line"] == 400

    def test_header_hint_returns_next_chunk(self, tmp_path):
        (tmp_path / "big.txt").write_text(_numbered(2000), encoding="utf-8")
        tool = ReadFileTool(workspace_root=tmp_path)
        first = tool.execute(path="big.txt")
        next_start = int(_HEADER_RE.match(first.output.split("\n")[0]).group(4))
        second = tool.execute(path="big.txt", start_line=next_start)
        assert second.output.split("\n")[1] == "line 401"
        assert second.metadata["start_line"] == 401
        assert second.metadata["end_line"] == 800

    def test_chunked_reading_covers_file_exactly(self, tmp_path):
        content = _numbered(2000)
        (tmp_path / "big.txt").write_text(content, encoding="utf-8")
        tool = ReadFileTool(workspace_root=tmp_path)
        pieces, start = [], 1
        for _ in range(20):
            result = tool.execute(path="big.txt", start_line=start)
            assert result.success, result.output
            if not result.metadata["truncated"]:
                pieces.append(result.output)
                break
            header, body = result.output.split("\n", 1)
            pieces.append(body)
            start = int(_HEADER_RE.match(header).group(4))
        assert "".join(pieces) == content

    def test_explicit_range(self, tmp_path):
        (tmp_path / "big.txt").write_text(_numbered(2000), encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(
            path="big.txt", start_line=10, end_line=12
        )
        assert result.output == "line 10\nline 11\nline 12\n"
        assert result.metadata["truncated"] is False

    def test_end_line_past_eof_is_clamped(self, tmp_path):
        (tmp_path / "f.txt").write_text(_numbered(5), encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(
            path="f.txt", start_line=4, end_line=99
        )
        assert result.output == "line 4\nline 5\n"
        assert result.metadata["end_line"] == 5

    def test_single_huge_line_capped_by_chars(self, tmp_path):
        (tmp_path / "min.js").write_text("x" * 100_000, encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="min.js")
        header, body = result.output.split("\n", 1)
        assert header.startswith("[read_file: showing lines 1-1 of 1 total")
        assert "cut at 40000 of 100000 chars" in header
        assert len(body) == READ_FILE_MAX_CHARS
        assert result.metadata["truncated"] is True

    def test_char_cap_applies_across_lines(self, tmp_path):
        content = ("y" * 999 + "\n") * 100  # 100 lines, 100k chars
        (tmp_path / "wide.txt").write_text(content, encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="wide.txt")
        header, body = result.output.split("\n", 1)
        assert header == (
            "[read_file: showing lines 1-40 of 100 total. "
            "Call read_file with start_line=41 to read more.]"
        )
        assert len(body) == 40_000

    def test_caps_are_overridable(self, tmp_path):
        (tmp_path / "f.txt").write_text(_numbered(10), encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path, max_lines=3).execute(path="f.txt")
        assert result.output.split("\n")[0].startswith("[read_file: showing lines 1-3 of 10")

    @pytest.mark.parametrize(
        "args",
        [
            {"start_line": 0},
            {"start_line": -5},
            {"start_line": 5, "end_line": 4},
            {"end_line": 0},
            {"start_line": "abc"},
            {"start_line": True},
        ],
    )
    def test_invalid_ranges_are_errors(self, tmp_path, args):
        (tmp_path / "f.txt").write_text(_numbered(10), encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="f.txt", **args)
        assert result.success is False
        assert result.output.startswith("ERROR")

    def test_start_past_end_states_total(self, tmp_path):
        (tmp_path / "f.txt").write_text(_numbered(10), encoding="utf-8")
        result = ReadFileTool(workspace_root=tmp_path).execute(path="f.txt", start_line=11)
        assert result.output.startswith("ERROR")
        assert "10 lines total" in result.output

    def test_schema_lists_range_params(self, tmp_path):
        props = ReadFileTool(workspace_root=tmp_path).parameters_schema["properties"]
        assert props["start_line"]["type"] == "integer"
        assert props["end_line"]["type"] == "integer"


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------

_GEN_SCRIPT = """\
import sys
for i in range(1, 1001):
    if i == 500:
        print("Traceback (most recent call last):")
    elif i == 600:
        print("FAILED tests/test_x.py::test_thing - AssertionError")
    else:
        print(f"line {i}")
print("err start", file=sys.stderr)
sys.exit(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
"""


def _run_gen(tmp_path, exit_code: int = 1):
    (tmp_path / "gen.py").write_text(_GEN_SCRIPT, encoding="utf-8")
    return RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
        command=f'"{sys.executable}" gen.py {exit_code}'
    )


class TestRunShellCaps:
    def test_long_output_keeps_head_tail_and_errors(self, tmp_path):
        result = _run_gen(tmp_path, exit_code=1)
        lines = result.output.split("\n")
        assert lines[0] == "line 1"
        assert "line 40" in lines and "line 41" not in lines
        assert "line 1000" in lines and "line 881" in lines
        assert "line 880" not in lines
        assert "line 300" not in lines
        assert "L500: Traceback (most recent call last):" in lines
        assert "L600: FAILED tests/test_x.py::test_thing - AssertionError" in lines
        markers = [l for l in lines if l.startswith("[... ")]
        assert len(markers) == 1
        assert markers[0].startswith("[... 838 lines omitted. Full output saved to .forge/outputs/shell_")
        assert markers[0].endswith(".log — use read_file on it if needed ...]")
        # stderr was short: unchanged, after stdout.
        assert "[stderr]\nerr start\n[exit code 1]" in result.output
        assert lines[-1] == "[exit code 1]"
        assert result.success is False
        assert result.error == "exit code 1"
        assert result.metadata["truncated"] is True
        assert result.metadata["omitted_lines"] == 838

    def test_saved_log_is_complete_and_readable(self, tmp_path):
        result = _run_gen(tmp_path, exit_code=0)
        rel = result.metadata["full_output_path"]
        assert rel.startswith(".forge/outputs/shell_") and rel.endswith(".log")
        saved = (tmp_path / rel).read_text(encoding="utf-8")
        saved_lines = saved.split("\n")
        assert len(saved_lines) == 1000 + 3  # stdout + "[stderr]" + err + exit
        assert saved_lines[299] == "line 300"
        assert saved.endswith("[stderr]\nerr start\n[exit code 0]")
        assert (tmp_path / ".forge" / ".gitignore").read_text(encoding="utf-8").strip() == "*"
        # read_file can open the log (not blocked as a secret file).
        read = ReadFileTool(workspace_root=tmp_path).execute(path=rel)
        assert read.success is True
        assert read.output.split("\n")[1] == "line 1"

    def test_streams_capped_independently(self, tmp_path):
        script = (
            "import sys\n"
            "print('only stdout')\n"
            "for i in range(500):\n"
            "    print(f'err {i}', file=sys.stderr)\n"
        )
        (tmp_path / "e.py").write_text(script, encoding="utf-8")
        result = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
            command=f'"{sys.executable}" e.py'
        )
        out = result.output
        assert out.startswith("only stdout\n[stderr]\nerr 0\n")
        assert "[... 340 lines omitted." in out
        assert out.endswith("err 499\n[exit code 0]")

    def test_long_lines_are_bounded(self, tmp_path):
        (tmp_path / "w.py").write_text("print('z' * 100000)\n", encoding="utf-8")
        result = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
            command=f'"{sys.executable}" w.py'
        )
        assert len(result.output) < 5_000
        assert "[... long lines shortened. Full output saved to" in result.output
        assert result.output.endswith("[exit code 0]")
        saved = (tmp_path / result.metadata["full_output_path"]).read_text(encoding="utf-8")
        assert saved.startswith("z" * 100_000)

    def test_short_output_unchanged_and_nothing_saved(self, tmp_path):
        result = RunShellTool(workspace_root=tmp_path, env_allow=()).execute(
            command=f'"{sys.executable}" -c "print(2 + 3)"'
        )
        assert result.output == "5\n[exit code 0]"
        assert result.metadata["truncated"] is False
        assert result.metadata["omitted_lines"] == 0
        assert "full_output_path" not in result.metadata
        assert not (tmp_path / ".forge").exists()

    def test_managed_context_still_detects_failure(self, tmp_path):
        result = _run_gen(tmp_path, exit_code=2)
        assert result.metadata["truncated"] is True
        msg = Message(role="tool", content=result.output, name="run_shell", tool_call_id="c")
        assert _is_error_tool_message(msg) is True

    def test_managed_context_success_not_error(self, tmp_path):
        result = _run_gen(tmp_path, exit_code=0)
        msg = Message(role="tool", content=result.output, name="run_shell", tool_call_id="c")
        assert _is_error_tool_message(msg) is False
