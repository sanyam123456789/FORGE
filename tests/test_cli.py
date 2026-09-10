"""
Tests for forge.cli — the minimal command-line entry point.

The provider is replaced with a scripted fake via monkeypatch; no network
and no API key are involved.
"""

from __future__ import annotations

import json

import pytest

from forge import cli
from forge.llm import LLMProvider, LLMResponse, ToolCall


class _Scripted(LLMProvider):
    provider_name = "scripted"
    model_name = "fake-model"

    def __init__(self, responses):
        self._responses = list(responses)

    def complete(self, messages, *, tools=None, max_tokens=4096, temperature=0.0):
        return self._responses.pop(0)


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr(
        "forge.llm.get_provider", lambda settings_override=None: provider
    )


def test_no_subcommand_prints_help_and_exits_zero(capsys):
    assert cli.main([]) == 0
    assert "run" in capsys.readouterr().out


def test_help_flag_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["-h"])
    assert exc.value.code == 0


def test_run_completes_and_creates_file(tmp_path, monkeypatch, capsys):
    provider = _Scripted(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "add.py", "content": "def add(a, b):\n    return a + b\n"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="Created add.py with add().", stop_reason="stop"),
        ]
    )
    _patch_provider(monkeypatch, provider)

    code = cli.main(
        [
            "run",
            "--task",
            "create add.py",
            "--workspace",
            str(tmp_path),
            "--no-trace",
        ]
    )
    assert code == 0
    assert (tmp_path / "add.py").read_text(encoding="utf-8").startswith("def add")
    out = capsys.readouterr().out
    assert "Created add.py" in out
    assert "status: completed" in out


def test_run_nonzero_exit_when_limit_hit(tmp_path, monkeypatch):
    loop_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="c", name="list_directory", arguments={})],
        stop_reason="tool_use",
    )

    class _Loop(LLMProvider):
        provider_name = "loop"
        model_name = "fake"

        def complete(self, *a, **k):
            return loop_response

    _patch_provider(monkeypatch, _Loop())
    code = cli.main(
        ["run", "--task", "loop", "--workspace", str(tmp_path), "--no-trace", "--max-turns", "2"]
    )
    assert code == 1


def test_run_json_output(tmp_path, monkeypatch, capsys):
    _patch_provider(monkeypatch, _Scripted([LLMResponse(content="done", stop_reason="stop")]))
    code = cli.main(
        ["run", "--task", "t", "--workspace", str(tmp_path), "--no-trace", "--json"]
    )
    assert code == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["input_tokens"] is None


def test_unimplemented_provider_exits_2(tmp_path, monkeypatch):
    def _boom(settings_override=None):
        raise NotImplementedError("provider 'openai' not implemented")

    monkeypatch.setattr("forge.llm.get_provider", _boom)
    code = cli.main(["run", "--task", "t", "--workspace", str(tmp_path), "--no-trace"])
    assert code == 2


# ---------------------------------------------------------------------------
# `forge evaluate`
# ---------------------------------------------------------------------------


def test_evaluate_writes_result_row(tmp_path, monkeypatch, capsys):
    provider = _Scripted(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(content="done", stop_reason="stop"),
        ]
    )
    _patch_provider(monkeypatch, provider)
    results_file = tmp_path / "eval.jsonl"

    code = cli.main(
        [
            "evaluate",
            "--task", "make calc.py",
            "--task-id", "calc",
            "--workspace", str(tmp_path / "ws"),
            "--results-file", str(results_file),
            "--no-trace",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_id"] == "calc"
    assert payload["tool_strategy"] == "fixed"
    assert payload["context_strategy"] == "raw"
    assert payload["status"] == "completed"
    assert payload["input_tokens"] is None
    # one JSONL row appended
    lines = results_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == payload["run_id"]


def test_evaluate_unimplemented_strategy_exits_2(tmp_path, monkeypatch):
    _patch_provider(monkeypatch, _Scripted([LLMResponse(content="x", stop_reason="stop")]))
    code = cli.main(
        [
            "evaluate",
            "--task", "t",
            "--tool-strategy", "adaptive",
            "--workspace", str(tmp_path / "ws"),
            "--results-file", str(tmp_path / "eval.jsonl"),
            "--no-trace",
        ]
    )
    assert code == 2
