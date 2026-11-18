"""
Tests for the Step 11 token savers and the new user-facing surface:

- forge.tokens estimate
- search_files tool, run_shell on a missing workspace
- agent loop: exposure enforcement, duplicate-read collapsing, repeated
  failure note, cut-off continuation, workspace map, conversation reuse
- managed context: chat follow-ups protected, duplicate pointers never
  supersede the full result
- StableContextStrategy (cache-friendly managed context)
- provider headers (User-Agent, x-opencode-session)
- modes, chat session (/diff, /undo, slash commands), report, CLI shorthand

Every LLM is scripted; nothing touches the network.
"""

from __future__ import annotations

import json

import pytest

from forge import cli
from forge.agent import STATUS_COMPLETED, AgentConfig, AgentRuntime
from forge.builtin_tools import RunShellTool, SearchFilesTool, build_default_registry
from forge.chat import ChatSession, handle_command, result_line, run_repl, snapshot_workspace
from forge.context import (
    DUPLICATE_RESULT_FLAG,
    ConversationContext,
    ManagedContextStrategy,
    StableContextStrategy,
)
from forge.evaluation.report import build_report, format_report
from forge.evaluation.result import EvalResult
from forge.llm import LLMProvider, LLMResponse, Message, ToolCall
from forge.modes import MODES, build_strategies, get_mode
from forge.providers.openai_compat import SESSION_HEADER, USER_AGENT, OpenAICompatibleProvider
from forge.tokens import estimate_messages_tokens, estimate_tokens
from forge.tools import AdaptiveToolExposure, FixedToolExposure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Scripted(LLMProvider):
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    @property
    def provider_name(self):
        return "scripted"

    @property
    def model_name(self):
        return "fake-model"

    def complete(self, messages, *, tools=None, max_tokens=None, temperature=0.0):
        self.calls.append({"messages": list(messages), "tools": tools, "max_tokens": max_tokens})
        return self._responses.pop(0)


def _text(content, stop="stop"):
    return LLMResponse(content=content, stop_reason=stop)


def _tool(name, arguments, call_id="c1", stop="tool_use"):
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        stop_reason=stop,
    )


def _run(tmp_path, provider, task="do it", **config_kw):
    registry = build_default_registry(workspace_root=tmp_path)
    config = AgentConfig(task=task, **config_kw)
    return AgentRuntime(
        config,
        provider=provider,
        registry=registry,
        workspace_root=tmp_path,
        write_trace=False,
    ).run()


def _tool_messages(call):
    return [m for m in call["messages"] if m.role == "tool"]


# ---------------------------------------------------------------------------
# Token estimate
# ---------------------------------------------------------------------------


class TestTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_deterministic_and_monotonic(self):
        text = "def add(a, b):\n    return a + b\n"
        assert estimate_tokens(text) == estimate_tokens(text)
        assert estimate_tokens(text * 2) > estimate_tokens(text)

    def test_roughly_chars_over_four_for_prose(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        est = estimate_tokens(text)
        assert len(text) / 6 < est < len(text) / 2.5

    def test_messages_include_tool_call_args(self):
        plain = Message(role="assistant", content="x")
        with_call = Message(
            role="assistant", content="x",
            tool_calls=[ToolCall(id="1", name="write_file", arguments={"content": "a" * 400})],
        )
        assert estimate_messages_tokens([with_call]) > estimate_messages_tokens([plain])


# ---------------------------------------------------------------------------
# search_files / run_shell
# ---------------------------------------------------------------------------


class TestSearchFiles:
    @pytest.fixture
    def ws(self, tmp_path):
        (tmp_path / "app.py").write_text("def main():\n    return compute(1)\n", encoding="utf-8")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "calc.py").write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("compute later\n", encoding="utf-8")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "junk.py").write_text("compute = 1\n", encoding="utf-8")
        (tmp_path / ".env").write_text("SECRET_compute=1\n", encoding="utf-8")
        return tmp_path

    def test_finds_matches_with_line_numbers(self, ws):
        res = SearchFilesTool(workspace_root=ws).execute(pattern=r"def compute")
        assert res.success
        assert "lib/calc.py:1: def compute(x):" in res.output

    def test_skips_virtualenv_and_secret_files(self, ws):
        res = SearchFilesTool(workspace_root=ws).execute(pattern="compute")
        assert ".venv" not in res.output
        assert ".env" not in res.output
        assert res.metadata["matches"] == 3  # app.py, lib/calc.py, notes.txt

    def test_glob_filter(self, ws):
        res = SearchFilesTool(workspace_root=ws).execute(pattern="compute", glob="*.txt")
        assert res.output.startswith("notes.txt:1:")
        assert res.metadata["matches"] == 1

    def test_invalid_regex_matched_as_text(self, ws):
        res = SearchFilesTool(workspace_root=ws).execute(pattern="compute(")
        assert res.success
        assert "app.py:2:" in res.output
        assert "plain text" in res.output

    def test_max_results_truncates_with_note(self, ws):
        res = SearchFilesTool(workspace_root=ws).execute(pattern="compute", max_results=1)
        assert res.metadata["returned"] == 1
        assert "showing 1 of 3" in res.output

    def test_no_matches(self, ws):
        res = SearchFilesTool(workspace_root=ws).execute(pattern="nothing_here_xyz")
        assert res.success and "No matches" in res.output

    def test_outside_workspace_rejected(self, ws):
        res = SearchFilesTool(workspace_root=ws).execute(pattern="x", path="../")
        assert not res.success


def test_run_shell_missing_workspace_gives_clean_error(tmp_path):
    missing = tmp_path / "not-created-yet"
    res = RunShellTool(workspace_root=missing, env_allow=()).execute(command="python --version")
    assert not res.success
    assert "does not exist yet" in res.output


# ---------------------------------------------------------------------------
# Agent loop token savers
# ---------------------------------------------------------------------------


EDIT_TASK = "Fix the typo in app.py"


class TestExposureEnforcement:
    def test_hidden_tool_refused_then_offered(self, tmp_path):
        provider = Scripted([
            _tool("run_shell", {"command": "python --version"}),
            _text("done"),
        ])
        run = _run(tmp_path, provider, task=EDIT_TASK, tool_exposure=AdaptiveToolExposure())
        assert run.status == STATUS_COMPLETED
        assert run.hidden_tool_calls_refused == 1
        refused = _tool_messages(provider.calls[1])[0].content
        assert "was not offered" in refused
        second_tools = {t["function"]["name"] for t in provider.calls[1]["tools"]}
        assert "run_shell" in second_tools

    def test_enable_tool_offered_and_works(self, tmp_path):
        provider = Scripted([
            _tool("enable_tool", {"name": "run_shell"}),
            _tool("run_shell", {"command": "python --version"}, call_id="c2"),
            _text("done"),
        ])
        run = _run(tmp_path, provider, task=EDIT_TASK, tool_exposure=AdaptiveToolExposure())
        first_tools = {t["function"]["name"] for t in provider.calls[0]["tools"]}
        assert "enable_tool" in first_tools and "run_shell" not in first_tools
        enable = next(t for t in provider.calls[0]["tools"]
                      if t["function"]["name"] == "enable_tool")
        assert "run_shell" in enable["function"]["parameters"]["properties"]["name"]["enum"]
        assert run.tools_enabled_on_request == 1
        assert run.hidden_tool_calls_refused == 0
        shell_result = _tool_messages(provider.calls[2])[1]
        assert "[exit code 0]" in shell_result.content

    def test_enable_tool_rejects_unknown(self, tmp_path):
        provider = Scripted([_tool("enable_tool", {"name": "rm_rf"}), _text("ok")])
        _run(tmp_path, provider, task=EDIT_TASK, tool_exposure=AdaptiveToolExposure())
        assert "cannot" in _tool_messages(provider.calls[1])[0].content or             "not a tool" in _tool_messages(provider.calls[1])[0].content

    def test_pinned_list_never_grows(self, tmp_path):
        provider = Scripted([
            _tool("run_shell", {"command": "python --version"}),
            _text("done"),
        ])
        run = _run(tmp_path, provider, tool_exposure=FixedToolExposure(pinned=["read_file"]))
        assert run.hidden_tool_calls_refused == 1
        for call in provider.calls:
            assert {t["function"]["name"] for t in call["tools"]} == {"read_file"}


def test_stall_nudge_after_repeated_nothing_new(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    calls = [_tool("read_file", {"path": "a.py"}, call_id=f"c{i}") for i in range(4)]
    provider = Scripted(calls + [_text("done")])
    run = _run(tmp_path, provider)
    assert run.stall_nudges == 1
    assert "nothing you had not already seen" in provider.calls[4]["messages"][-1].content


class TestDuplicateReads:
    def test_identical_read_collapsed_once(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n" * 50, encoding="utf-8")
        read = lambda cid: _tool("read_file", {"path": "a.py"}, call_id=cid)  # noqa: E731
        provider = Scripted([read("c1"), read("c2"), read("c3"), _text("done")])
        run = _run(tmp_path, provider)
        assert run.duplicate_reads_collapsed == 1
        tools = _tool_messages(provider.calls[3])
        assert tools[0].content.startswith("x = 1")
        assert tools[1].content.startswith("[FORGE: identical")
        assert tools[1].metadata.get(DUPLICATE_RESULT_FLAG)
        # A further repeat gets the full text again, so nothing is lost for good.
        assert tools[2].content.startswith("x = 1")

    def test_changed_file_not_collapsed(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        provider = Scripted([
            _tool("read_file", {"path": "a.py"}, call_id="c1"),
            _tool("write_file", {"path": "a.py", "content": "x = 2\n", "overwrite": True}, call_id="c2"),
            _tool("read_file", {"path": "a.py"}, call_id="c3"),
            _text("done"),
        ])
        run = _run(tmp_path, provider)
        assert run.duplicate_reads_collapsed == 0
        assert _tool_messages(provider.calls[3])[2].content == "x = 2\n"


def test_repeated_failure_gets_note(tmp_path):
    bad = lambda cid: _tool("read_file", {"path": "missing.py"}, call_id=cid)  # noqa: E731
    provider = Scripted([bad("c1"), bad("c2"), bad("c3"), _text("gave up")])
    _run(tmp_path, provider)
    tools = _tool_messages(provider.calls[3])
    assert "failed 3 times" not in tools[1].content
    assert "failed 3 times" in tools[2].content


class TestCutOffReplies:
    def test_text_cut_off_is_continued(self, tmp_path):
        provider = Scripted([_text("Part one, ", stop="length"), _text("part two.")])
        run = _run(tmp_path, provider)
        assert run.status == STATUS_COMPLETED
        assert run.final_answer == "Part one, part two."
        assert run.length_continuations == 1
        assert "cut off" in provider.calls[1]["messages"][-1].content

    def test_continuations_are_bounded(self, tmp_path):
        provider = Scripted([_text(f"p{i} ", stop="length") for i in range(4)])
        run = _run(tmp_path, provider)
        assert run.status == STATUS_COMPLETED
        assert run.length_continuations == 3
        assert run.llm_calls == 4


class TestWorkspaceMap:
    def test_file_list_added_to_first_message(self, tmp_path):
        (tmp_path / "app.py").write_text("print(1)\n", encoding="utf-8")
        (tmp_path / ".env").write_text("KEY=1\n", encoding="utf-8")
        provider = Scripted([_text("ok")])
        _run(tmp_path, provider, task="fix app", repo_map=True)
        first_user = provider.calls[0]["messages"][1].content
        assert first_user.startswith("fix app")
        assert "app.py" in first_user
        assert ".env" not in first_user

    def test_disabled(self, tmp_path):
        (tmp_path / "app.py").write_text("print(1)\n", encoding="utf-8")
        provider = Scripted([_text("ok")])
        _run(tmp_path, provider, task="fix app", repo_map=False)
        assert provider.calls[0]["messages"][1].content == "fix app"

    def test_empty_workspace_adds_nothing(self, tmp_path):
        provider = Scripted([_text("ok")])
        _run(tmp_path, provider, task="make app", repo_map=True)
        assert provider.calls[0]["messages"][1].content == "make app"


def test_runtime_continues_existing_conversation(tmp_path):
    ctx = ConversationContext()
    registry = build_default_registry(workspace_root=tmp_path)
    provider = Scripted([_text("first answer"), _text("second answer")])
    for task in ("first", "second"):
        AgentRuntime(
            AgentConfig(task=task, repo_map=False), provider=provider, registry=registry,
            workspace_root=tmp_path, write_trace=False, context=ctx,
        ).run()
    roles = [m.role for m in provider.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert provider.calls[1]["messages"][-1].content == "second"


# ---------------------------------------------------------------------------
# Managed context additions
# ---------------------------------------------------------------------------


def _turn(name, args, output, cid):
    return [
        Message(role="assistant", content="thinking " * 20,
                tool_calls=[ToolCall(id=cid, name=name, arguments=args)]),
        Message(role="tool", content=output, tool_call_id=cid, name=name),
    ]


class TestManagedAdditions:
    def test_follow_up_user_message_is_kept(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="system", content="sys"))
        ctx.add_message(Message(role="user", content="task one"))
        for i in range(3):
            for m in _turn("read_file", {"path": f"f{i}.py"}, "x" * 3000, f"a{i}"):
                ctx.add_message(m)
        ctx.add_message(Message(role="assistant", content="done one"))
        ctx.add_message(Message(role="user", content="task two"))
        for i in range(6):
            for m in _turn("read_file", {"path": f"g{i}.py"}, "y" * 3000, f"b{i}"):
                ctx.add_message(m)
        out = ManagedContextStrategy(keep_recent_turns=2, max_chars=8000).prepare(ctx)
        assert any(m.role == "user" and m.content == "task two" for m in out)

    def test_duplicate_pointer_does_not_supersede(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="system", content="sys"))
        ctx.add_message(Message(role="user", content="task"))
        for m in _turn("read_file", {"path": "a.py"}, "FULL CONTENT " * 50, "c1"):
            ctx.add_message(m)
        dup = _turn("read_file", {"path": "a.py"}, "[FORGE: identical ...]", "c2")
        dup[1].metadata[DUPLICATE_RESULT_FLAG] = True
        for m in dup:
            ctx.add_message(m)
        for i in range(4):
            for m in _turn("list_directory", {"path": f"d{i}"}, "z" * 200, f"l{i}"):
                ctx.add_message(m)
        out = ManagedContextStrategy(keep_recent_turns=1).prepare(ctx)
        assert any(m.content.startswith("FULL CONTENT") for m in out)


# ---------------------------------------------------------------------------
# StableContextStrategy
# ---------------------------------------------------------------------------


def _grow(ctx, n, size=2000, start=0):
    for i in range(start, start + n):
        for m in _turn("read_file", {"path": f"f{i}.py"}, "w" * size, f"c{i}"):
            ctx.add_message(m)


def _base_ctx():
    ctx = ConversationContext()
    ctx.add_message(Message(role="system", content="system prompt"))
    ctx.add_message(Message(role="user", content="the task"))
    return ctx


class TestStableContext:
    def test_below_threshold_is_pass_through(self):
        ctx = _base_ctx()
        _grow(ctx, 2, size=100)
        strat = StableContextStrategy(compact_at_tokens=5000, target_tokens=2000)
        assert strat.prepare(ctx) == ctx.get_messages()

    def test_prefix_stays_identical_between_compactions(self):
        ctx = _base_ctx()
        strat = StableContextStrategy(compact_at_tokens=3000, target_tokens=1000)
        sent = []
        for i in range(12):
            _grow(ctx, 1, size=1200, start=i)
            sent.append(strat.prepare(ctx))
        assert strat.compactions >= 1
        compact_turns = [
            i for i in range(1, len(sent))
            if sent[i][: len(sent[i - 1])] != sent[i - 1]
        ]
        # Every request extends the previous one, except the compaction turns.
        assert len(compact_turns) == strat.compactions
        for i in range(1, len(sent)):
            if i not in compact_turns:
                assert sent[i][: len(sent[i - 1])] == sent[i - 1]

    def test_compaction_shrinks_below_threshold(self):
        ctx = _base_ctx()
        _grow(ctx, 20, size=1500)
        strat = StableContextStrategy(compact_at_tokens=4000, target_tokens=2000)
        out = strat.prepare(ctx)
        assert strat.compactions == 1
        assert estimate_messages_tokens(out) < estimate_messages_tokens(ctx.get_messages())
        assert strat.last_report().compacted is True
        assert out[0].content == "system prompt" and out[1].content == "the task"

    def test_no_recompaction_without_growth(self):
        ctx = _base_ctx()
        _grow(ctx, 20, size=1500)
        strat = StableContextStrategy(
            compact_at_tokens=600, target_tokens=300, keep_recent_turns=3
        )
        strat.prepare(ctx)  # protected recent turns alone exceed the threshold
        ctx.add_message(Message(role="assistant", content="small"))
        strat.prepare(ctx)
        assert strat.compactions == 1

    def test_request_compaction(self):
        ctx = _base_ctx()
        _grow(ctx, 6, size=500)
        strat = StableContextStrategy(compact_at_tokens=100_000, target_tokens=50_000)
        strat.prepare(ctx)
        strat.request_compaction()
        strat.prepare(ctx)
        assert strat.compactions == 1

    def test_resets_for_a_new_conversation(self):
        strat = StableContextStrategy(compact_at_tokens=5000, target_tokens=2000)
        a = _base_ctx()
        _grow(a, 3, size=100)
        strat.prepare(a)
        b = _base_ctx()
        assert strat.prepare(b) == b.get_messages()


# ---------------------------------------------------------------------------
# Provider headers
# ---------------------------------------------------------------------------


class _Transport:
    def __init__(self):
        self.headers = []

    def __call__(self, url, headers, payload, timeout):
        self.headers.append(dict(headers))
        return 200, {}, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def test_provider_sends_user_agent_and_stable_session_id():
    transport = _Transport()
    p = OpenAICompatibleProvider(
        base_url="https://x.test/v1", api_key="k", model="m", transport=transport
    )
    p.complete([Message(role="user", content="a")])
    p.complete([Message(role="user", content="b")])
    first, second = transport.headers
    assert first["User-Agent"] == USER_AGENT
    assert first[SESSION_HEADER] and first[SESSION_HEADER] == second[SESSION_HEADER]
    p.new_session()
    p.complete([Message(role="user", content="c")])
    assert transport.headers[2][SESSION_HEADER] != first[SESSION_HEADER]


def test_provider_omits_max_tokens_when_none():
    seen = {}

    def transport(url, headers, payload, timeout):
        seen.update(payload)
        return 200, {}, {"choices": [{"message": {"content": "ok"}}]}

    p = OpenAICompatibleProvider(base_url="https://x.test/v1", api_key="k", model="m",
                                 transport=transport)
    p.complete([Message(role="user", content="a")], max_tokens=None)
    assert "max_tokens" not in seen


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def test_modes_map_to_strategies():
    assert set(MODES) == {"baseline", "lean-tools", "lean-context", "smart"}
    tools, context = build_strategies("smart")
    assert (tools.name, context.name) == ("adaptive", "stable")
    tools, context = build_strategies("baseline")
    assert (tools.name, context.name) == ("fixed", "raw")
    # Step 12: no mode uses the per-turn managed strategy any more.
    tools, context = build_strategies("lean-context")
    assert (tools.name, context.name) == ("fixed", "stable")
    assert context.compact_at_tokens == 64_000
    with pytest.raises(ValueError):
        get_mode("turbo")


# ---------------------------------------------------------------------------
# Chat session
# ---------------------------------------------------------------------------


class _Settings:
    def __init__(self, tmp_path):
        self.workspace_root = tmp_path
        self.log_dir = tmp_path / "logs"
        self.llm_model = "fake-model"


@pytest.fixture
def session_factory(tmp_path):
    def make(responses, mode="smart"):
        return ChatSession(
            settings=_Settings(tmp_path), workspace=tmp_path, mode=mode,
            provider=Scripted(responses), write_trace=False,
        )
    return make


class TestChat:
    def test_send_tracks_changes_diff_and_undo(self, tmp_path, session_factory):
        (tmp_path / "keep.py").write_text("old\n", encoding="utf-8")
        session = session_factory([
            _tool("write_file", {"path": "new.py", "content": "print(1)\n"}, call_id="c1"),
            _tool("write_file", {"path": "keep.py", "content": "new\n", "overwrite": True}, call_id="c2"),
            _text("Made the changes."),
        ])
        run, changes = session.send("make changes")
        assert run.ok
        assert changes.created == ["new.py"] and changes.modified == ["keep.py"]
        diff = session.diff()
        assert "+print(1)" in diff and "-old" in diff and "+new" in diff
        undone = session.undo()
        assert undone is not None
        assert not (tmp_path / "new.py").exists()
        assert (tmp_path / "keep.py").read_text(encoding="utf-8") == "old\n"
        assert session.undo() is None

    def test_follow_up_shares_conversation_and_mentions_undo(self, tmp_path, session_factory):
        session = session_factory([
            _tool("write_file", {"path": "a.py", "content": "1\n"}), _text("one"),
            _text("two"),
        ])
        session.send("first")
        session.undo()
        session.send("second")
        last_call = session.provider.calls[-1]["messages"]
        assert last_call[1].content.startswith("first")
        assert "undone" in last_call[-1].content and last_call[-1].content.endswith("second")

    def test_slash_commands(self, tmp_path, session_factory):
        session = session_factory([_text("hello")])
        session.send("hi")
        out = []
        assert handle_command(session, "/tokens", out.append)
        assert handle_command(session, "/context", out.append)
        assert handle_command(session, "/tools", out.append)
        assert handle_command(session, "/mode baseline", out.append)
        assert session.mode.name == "baseline"
        assert handle_command(session, "/compact", out.append)
        assert handle_command(session, "/nope", out.append)
        assert handle_command(session, "/exit", out.append) is False
        text = "\n".join(out)
        assert "LLM calls: 1" in text and "unknown command" in text

    def test_clear_starts_fresh(self, session_factory):
        session = session_factory([_text("a"), _text("b")])
        session.send("one")
        session.clear()
        session.send("two")
        assert [m.role for m in session.provider.calls[-1]["messages"]] == ["system", "user"]

    def test_repl_runs_task_then_exits(self, tmp_path, session_factory):
        session = session_factory([_text("All set.")])
        inputs = iter(["do the thing", "/exit"])
        out = []
        code = run_repl(session, input_fn=lambda _p: next(inputs), out=out.append)
        assert code == 0
        assert "All set." in out and any(line.startswith("✓ Done") for line in out)

    def test_snapshot_excludes_log_dir(self, tmp_path):
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run.jsonl").write_text("{}", encoding="utf-8")
        (tmp_path / "a.py").write_text("x", encoding="utf-8")
        snap = snapshot_workspace(tmp_path, exclude=(tmp_path / "logs",))
        assert set(snap) == {"a.py"}


def test_result_line():
    from forge.agent import AgentRun

    run = AgentRun(config=AgentConfig(task="t"), run_id="r", status="completed",
                   turns=3, total_tokens=4025, input_tokens=3881,
                   cached_input_tokens=2432, llm_latency_ms=10856.8)
    assert result_line(run) == "✓ Done · 3 turns · 4.0k tokens · 63% cached · 10.9s"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _row(arm, success, total, inp, cached):
    return EvalResult(
        task_id="t", run_id="r", provider="p", model="m",
        tool_strategy="fixed", context_strategy="raw", status="completed",
        success=success, total_tokens=total, input_tokens=inp,
        cached_input_tokens=cached, llm_calls=3, duration_s=2.0, arm_id=arm,
    )


def test_report_numbers():
    rows = [
        _row("fixed_raw", True, 1000, 900, 450),
        _row("fixed_raw", False, 3000, 2900, 1450),
        _row("adaptive_stable", True, 800, 700, 560),
    ]
    reports = {r.arm_id: r for r in build_report(rows)}
    base = reports["fixed_raw"]
    assert (base.tasks, base.solved) == (2, 1)
    assert base.tokens_per_solved == 4000  # failed attempt still counts
    assert base.cache_hit_rate == pytest.approx(0.5)
    assert base.mean_uncached_input == pytest.approx(950)
    assert reports["adaptive_stable"].tokens_per_solved == 800
    table = format_report(list(reports.values()))
    assert "fixed_raw" in table and "4,000" in table


# ---------------------------------------------------------------------------
# CLI shorthand
# ---------------------------------------------------------------------------


def test_cli_task_shorthand(tmp_path, monkeypatch, capsys):
    provider = Scripted([_text("Done it.")])
    fake = lambda settings_override=None: provider  # noqa: E731
    monkeypatch.setattr("forge.llm.get_provider", fake)
    monkeypatch.setattr("forge.agent.get_provider", fake)
    code = cli.main(["say hello", "--workspace", str(tmp_path), "--no-trace", "--mode", "baseline"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Done it." in out and "mode: baseline" in out


def test_cli_run_requires_task(capsys):
    assert cli.main(["run"]) == 2


def test_cli_report_reads_results(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(_row("fixed_raw", True, 1000, 900, 450).to_dict()) + "\n",
        encoding="utf-8",
    )
    assert cli.main(["report", str(path)]) == 0
    assert "fixed_raw" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Line endings (edit_file / write_file used to turn LF files into CRLF on
# Windows, because Path.write_text translates newlines)
# ---------------------------------------------------------------------------


class TestLineEndings:
    def test_edit_keeps_lf(self, tmp_path):
        from forge.builtin_tools import EditFileTool

        f = tmp_path / "a.py"
        f.write_bytes(b"x = 1\ny = 2\n")
        res = EditFileTool(workspace_root=tmp_path).execute(
            path="a.py", old_string="x = 1", new_string="x = 10"
        )
        assert res.success
        assert f.read_bytes() == b"x = 10\ny = 2\n"

    def test_edit_keeps_crlf_and_matches_lf_old_string(self, tmp_path):
        from forge.builtin_tools import EditFileTool

        f = tmp_path / "a.py"
        f.write_bytes(b"def f():\r\n    return 1\r\n")
        res = EditFileTool(workspace_root=tmp_path).execute(
            path="a.py", old_string="def f():\n    return 1", new_string="def f():\n    return 2"
        )
        assert res.success
        assert f.read_bytes() == b"def f():\r\n    return 2\r\n"

    def test_write_new_file_is_lf(self, tmp_path):
        from forge.builtin_tools import WriteFileTool

        WriteFileTool(workspace_root=tmp_path).execute(path="n.py", content="a\nb\n")
        assert (tmp_path / "n.py").read_bytes() == b"a\nb\n"

    def test_overwrite_keeps_existing_crlf(self, tmp_path):
        from forge.builtin_tools import WriteFileTool

        f = tmp_path / "w.txt"
        f.write_bytes(b"old\r\n")
        WriteFileTool(workspace_root=tmp_path).execute(
            path="w.txt", content="new\nline\n", overwrite=True
        )
        assert f.read_bytes() == b"new\r\nline\r\n"
