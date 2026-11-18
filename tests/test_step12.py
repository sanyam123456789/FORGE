"""
Tests for Step 12:

- every tool call gets an answer when a run stops early (tool limit, Ctrl+C),
  so a chat stays usable; /undo covers interrupted tasks
- approval before write_file / edit_file / run_shell
- project rules (AGENTS.md / FORGE.md) in the system prompt
- streaming in the OpenAI-compatible provider
- saved / resumed chat sessions
- terminal helpers (streamed output, approval prompt)
- benchmark: repetitions, parallel jobs, hidden verify scripts, report
  spread and change vs baseline, the long-task suite

Every LLM is scripted; nothing touches the network.
"""

from __future__ import annotations

import json
import pathlib
import threading

import pytest

from forge import cli
from forge.agent import (
    STATUS_INTERRUPTED,
    STATUS_MAX_TOOL_CALLS,
    AgentConfig,
    AgentRuntime,
    _new_run_id,
    load_project_rules,
)
from forge.builtin_tools import build_default_registry
from forge.chat import (
    ChatSession,
    TerminalApprover,
    TerminalOutput,
    describe_action,
    handle_command,
    run_repl,
)
from forge.evaluation.matrix import ALL_ARMS_BY_ID, MatrixRunner
from forge.evaluation.report import build_report, format_report
from forge.evaluation.result import EvalResult
from forge.evaluation.suite import LONG_TASKS_DIR, TaskSuiteError, load_suite, load_task
from forge.evaluation.task import EvalTask, run_verify_script
from forge.llm import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from forge.providers.openai_compat import OpenAICompatibleProvider
from forge.sessions import (
    find_session,
    list_sessions,
    load_session,
    message_from_dict,
    message_to_dict,
    save_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Scripted(LLMProvider):
    """Scripted provider. Items may be LLMResponse or an exception to raise."""

    def __init__(self, responses, *, streaming=False):
        self._responses = list(responses)
        self.calls = []
        self.supports_streaming = streaming

    @property
    def provider_name(self):
        return "scripted"

    @property
    def model_name(self):
        return "fake-model"

    def complete(self, messages, *, tools=None, max_tokens=None, temperature=0.0, **kw):
        self.calls.append({"messages": list(messages), "kw": kw})
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if "on_text" in kw and item.content:
            kw["on_text"](item.content)
        return item


def _text(content):
    return LLMResponse(content=content, stop_reason="stop")


def _tools(*calls):
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=cid, name=name, arguments=args) for cid, name, args in calls],
        stop_reason="tool_use",
    )


def _runtime(tmp_path, provider, task="do it", context=None, **kw):
    config_kw = {k: kw.pop(k) for k in list(kw) if k in ("max_tool_calls", "project_rules")}
    return AgentRuntime(
        AgentConfig(task=task, **config_kw),
        provider=provider,
        registry=build_default_registry(workspace_root=tmp_path),
        workspace_root=tmp_path,
        write_trace=False,
        context=context,
        **kw,
    )


def _assert_all_calls_answered(messages):
    for i, msg in enumerate(messages):
        if msg.role == "assistant" and msg.tool_calls:
            answered = {m.tool_call_id for m in messages[i + 1:] if m.role == "tool"}
            for call in msg.tool_calls:
                assert call.id in answered, f"tool call {call.id} has no result"


# ---------------------------------------------------------------------------
# 1. Unanswered tool calls are closed; Ctrl+C ends a run cleanly
# ---------------------------------------------------------------------------


class TestUnansweredCalls:
    def test_tool_limit_mid_batch_answers_every_call(self, tmp_path):
        provider = Scripted([
            _tools(("a", "list_directory", {"path": "."}),
                   ("b", "list_directory", {"path": "."}),
                   ("c", "list_directory", {"path": "."})),
        ])
        from forge.context import ConversationContext

        ctx = ConversationContext()
        run = _runtime(tmp_path, provider, context=ctx, max_tool_calls=1).run()
        assert run.status == STATUS_MAX_TOOL_CALLS
        assert run.unanswered_calls_closed == 2
        _assert_all_calls_answered(ctx.messages)
        notes = [m.content for m in ctx.messages if m.role == "tool"][1:]
        assert all("tool-call limit" in n for n in notes)

    def test_chat_continues_after_tool_limit(self, tmp_path):
        provider = Scripted([
            _tools(("a", "list_directory", {"path": "."}),
                   ("b", "list_directory", {"path": "."})),
            _text("second answer"),
        ])
        from forge.context import ConversationContext

        ctx = ConversationContext()
        _runtime(tmp_path, provider, context=ctx, max_tool_calls=1).run()
        run2 = _runtime(tmp_path, provider, task="follow up", context=ctx).run()
        assert run2.ok
        # The second request carried a valid history: every call answered.
        _assert_all_calls_answered(provider.calls[-1]["messages"])

    def test_keyboard_interrupt_in_provider(self, tmp_path):
        provider = Scripted([KeyboardInterrupt()])
        run = _runtime(tmp_path, provider).run()
        assert run.status == STATUS_INTERRUPTED
        assert run.error == "interrupted by the user"

    def test_keyboard_interrupt_during_tool(self, tmp_path):
        provider = Scripted([
            _tools(("a", "write_file", {"path": "x.txt", "content": "1"}),
                   ("b", "write_file", {"path": "y.txt", "content": "2"})),
        ])

        def approve(call):  # Ctrl+C while the second action is being decided
            if call.id == "b":
                raise KeyboardInterrupt
            return True

        from forge.context import ConversationContext

        ctx = ConversationContext()
        run = _runtime(tmp_path, provider, context=ctx, approve=approve).run()
        assert run.status == STATUS_INTERRUPTED
        assert (tmp_path / "x.txt").exists() and not (tmp_path / "y.txt").exists()
        _assert_all_calls_answered(ctx.messages)
        assert "interrupted" in ctx.messages[-1].content

    def test_completed_run_adds_nothing(self, tmp_path):
        provider = Scripted([_tools(("a", "list_directory", {"path": "."})), _text("done")])
        run = _runtime(tmp_path, provider).run()
        assert run.ok and run.unanswered_calls_closed == 0

    def test_duplicate_read_not_collapsed_when_earlier_result_is_hidden(self, tmp_path):
        """A context strategy compressed the first read; the repeat must be full."""
        from forge.context import ContextStrategy

        class Compressing(ContextStrategy):
            name = "compressing"

            def prepare(self, ctx):
                return [
                    Message(role=m.role, content="[compressed]", tool_call_id=m.tool_call_id,
                            name=m.name) if m.role == "tool" else m
                    for m in ctx.get_messages()
                ]

        (tmp_path / "a.txt").write_text("the content", encoding="utf-8")
        provider = Scripted([
            _tools(("r1", "read_file", {"path": "a.txt"})),
            _tools(("r2", "read_file", {"path": "a.txt"})),
            _text("done"),
        ])
        run = AgentRuntime(
            AgentConfig(task="t", context_strategy=Compressing()),
            provider=provider, registry=build_default_registry(workspace_root=tmp_path),
            workspace_root=tmp_path, write_trace=False,
        ).run()
        assert run.duplicate_reads_collapsed == 0

    def test_duplicate_read_in_same_batch_still_collapses(self, tmp_path):
        (tmp_path / "a.txt").write_text("the content", encoding="utf-8")
        provider = Scripted([
            _tools(("r1", "read_file", {"path": "a.txt"}), ("r2", "read_file", {"path": "a.txt"})),
            _text("done"),
        ])
        run = _runtime(tmp_path, provider).run()
        assert run.duplicate_reads_collapsed == 1

    def test_run_ids_are_unique(self):
        ids = {_new_run_id() for _ in range(200)}
        assert len(ids) == 200


# ---------------------------------------------------------------------------
# 2. Approval
# ---------------------------------------------------------------------------


class TestApproval:
    def test_declined_action_is_not_run(self, tmp_path):
        provider = Scripted([
            _tools(("a", "write_file", {"path": "x.txt", "content": "1"})),
            _text("ok"),
        ])
        asked = []
        run = _runtime(tmp_path, provider, approve=lambda c: asked.append(c.name) or False).run()
        assert asked == ["write_file"]
        assert not (tmp_path / "x.txt").exists()
        assert run.actions_declined == 1
        tool_msg = [m for m in provider.calls[1]["messages"] if m.role == "tool"][0]
        assert "declined" in tool_msg.content

    def test_user_message_is_passed_on(self, tmp_path):
        provider = Scripted([
            _tools(("a", "run_shell", {"command": "python -V"})),
            _text("ok"),
        ])
        _runtime(tmp_path, provider, approve=lambda c: "use pytest instead").run()
        tool_msg = [m for m in provider.calls[1]["messages"] if m.role == "tool"][0]
        assert "use pytest instead" in tool_msg.content

    def test_approved_action_runs_and_reads_are_not_asked(self, tmp_path):
        (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
        provider = Scripted([
            _tools(("a", "read_file", {"path": "a.txt"}),
                   ("b", "write_file", {"path": "b.txt", "content": "2"})),
            _text("ok"),
        ])
        asked = []
        _runtime(tmp_path, provider, approve=lambda c: asked.append(c.name) or True).run()
        assert asked == ["write_file"]
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "2"


# ---------------------------------------------------------------------------
# 3. Project rules
# ---------------------------------------------------------------------------


class TestProjectRules:
    def test_agents_md_goes_into_system_prompt(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("Use tabs. Never touch vendor/.", encoding="utf-8")
        provider = Scripted([_text("ok")])
        _runtime(tmp_path, provider, project_rules=True).run()
        system = provider.calls[0]["messages"][0]
        assert system.role == "system"
        assert "Project instructions (from AGENTS.md; included here in full" in system.content
        assert "Never touch vendor/." in system.content

    def test_rules_can_be_turned_off(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("secret rule", encoding="utf-8")
        provider = Scripted([_text("ok")])
        _runtime(tmp_path, provider, project_rules=False).run()
        assert "secret rule" not in provider.calls[0]["messages"][0].content

    def test_both_files_and_long_rules_are_cut(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("a" * 30_000, encoding="utf-8")
        (tmp_path / "FORGE.md").write_text("forge rule", encoding="utf-8")
        text = load_project_rules(tmp_path)
        assert "cut at 20000 characters" in text
        assert text.index("AGENTS.md") < text.index("FORGE.md")
        assert load_project_rules(tmp_path / "missing") == ""

    def test_rules_are_not_repeated_in_follow_ups(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("rule-x", encoding="utf-8")
        from forge.context import ConversationContext

        ctx = ConversationContext()
        provider = Scripted([_text("one"), _text("two")])
        _runtime(tmp_path, provider, context=ctx, project_rules=True).run()
        _runtime(tmp_path, provider, task="again", context=ctx, project_rules=True).run()
        joined = "\n".join(m.content for m in provider.calls[1]["messages"])
        assert joined.count("rule-x") == 1


# ---------------------------------------------------------------------------
# 4. Streaming
# ---------------------------------------------------------------------------


def _sse(*chunks):
    return [f"data: {json.dumps(c)}" for c in chunks] + ["", "data: [DONE]"]


class _StreamTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, url, headers, payload, timeout):
        self.payloads.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _provider(stream_transport):
    return OpenAICompatibleProvider(
        base_url="https://x.test/v1", api_key="k", model="m",
        stream_transport=stream_transport, retry_max=1,
    )


class TestStreaming:
    def test_text_is_streamed_and_usage_kept(self):
        lines = _sse(
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                      "total_tokens": 12,
                                      "prompt_tokens_details": {"cached_tokens": 8}}},
        )
        transport = _StreamTransport((200, {}, iter(lines)))
        seen = []
        resp = _provider(transport).complete([Message(role="user", content="hi")],
                                             on_text=seen.append)
        assert seen == ["Hel", "lo"]
        assert resp.content == "Hello" and resp.stop_reason == "stop"
        assert (resp.input_tokens, resp.output_tokens, resp.cached_input_tokens) == (10, 2, 8)
        assert transport.payloads[0]["stream"] is True
        assert transport.payloads[0]["stream_options"] == {"include_usage": True}

    def test_tool_call_deltas_are_joined(self):
        lines = _sse(
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "read_", "arguments": "{\"pa"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "file", "arguments": "th\": \"a.py\"}"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "call_2", "function": {"name": "list_directory",
                                                          "arguments": "{}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        )
        resp = _provider(_StreamTransport((200, {}, iter(lines)))).complete(
            [Message(role="user", content="x")], on_text=lambda t: None
        )
        assert [(c.id, c.name, c.arguments) for c in resp.tool_calls] == [
            ("call_1", "read_file", {"path": "a.py"}),
            ("call_2", "list_directory", {}),
        ]
        assert resp.stop_reason == "tool_use"
        assert resp.tool_calls[0].provider_signature == {"reasoning_content": "think"}
        assert resp.input_tokens is None  # not reported -> never invented

    def test_http_error_while_streaming(self):
        transport = _StreamTransport((401, {}, {"error": {"message": "bad key"}}))
        with pytest.raises(LLMError, match="HTTP 401"):
            _provider(transport).complete([Message(role="user", content="x")],
                                          on_text=lambda t: None)

    def test_error_chunk_raises(self):
        lines = _sse({"error": {"message": "overloaded"}})
        with pytest.raises(LLMError, match="overloaded"):
            _provider(_StreamTransport((200, {}, iter(lines)))).complete(
                [Message(role="user", content="x")], on_text=lambda t: None
            )

    def test_no_retry_after_text_was_shown(self):
        def broken():
            yield 'data: {"choices": [{"delta": {"content": "par"}}]}'
            raise ConnectionError("reset")

        transport = _StreamTransport((200, {}, broken()), (200, {}, iter(_sse())))
        with pytest.raises(LLMError, match="reset"):
            _provider(transport).complete([Message(role="user", content="x")],
                                          on_text=lambda t: None)
        assert len(transport.payloads) == 1  # not retried

    def test_retry_before_any_text(self, monkeypatch):
        monkeypatch.setattr("forge.providers.openai_compat.time.sleep", lambda s: None)
        lines = _sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
        transport = _StreamTransport(ConnectionError("down"), (200, {}, iter(lines)))
        resp = _provider(transport).complete([Message(role="user", content="x")],
                                             on_text=lambda t: None)
        assert resp.content == "ok" and len(transport.payloads) == 2

    def test_agent_streams_only_to_streaming_providers(self, tmp_path):
        seen = []
        plain = Scripted([_text("hello")])
        run = _runtime(tmp_path, plain, on_text=seen.append).run()
        assert run.ok and seen == [] and plain.calls[0]["kw"] == {}
        streaming = Scripted([_text("hello")], streaming=True)
        _runtime(tmp_path, streaming, on_text=seen.append).run()
        assert seen == ["hello"]


# ---------------------------------------------------------------------------
# 5. Sessions
# ---------------------------------------------------------------------------


class _Settings:
    def __init__(self, tmp_path):
        self.workspace_root = tmp_path
        self.log_dir = tmp_path / "logs"
        self.llm_model = "fake-model"


def _session(tmp_path, responses, **kw):
    return ChatSession(
        settings=_Settings(tmp_path), workspace=tmp_path,
        provider=Scripted(responses), write_trace=False, **kw,
    )


class TestSessions:
    def test_message_round_trip(self):
        msg = Message(
            role="assistant", content="x",
            tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "a"},
                                 provider_signature={"reasoning_content": "r"})],
            metadata={"flag": True},
        )
        back = message_from_dict(json.loads(json.dumps(message_to_dict(msg))))
        assert back == msg

    def test_save_list_find_load(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        path = tmp_path / "logs" / "sessions" / "20240101-000000-abcd.json"
        save_session(path, {"session_id": "20240101-000000-abcd", "workspace": str(ws),
                            "totals": {"tasks": 2}},
                     [Message(role="system", content="s"),
                      Message(role="user", content="fix   the\nbug")])
        listed = list_sessions(tmp_path / "logs", workspace=ws)
        assert listed[0]["title"] == "fix the bug" and listed[0]["tasks"] == 2
        assert list_sessions(tmp_path / "logs", workspace=tmp_path) == []
        assert find_session(tmp_path / "logs", None, workspace=ws) == path
        assert find_session(tmp_path / "logs", "20240101", workspace=tmp_path) == path
        assert find_session(tmp_path / "logs", "nope", workspace=ws) is None
        fields, messages = load_session(path)
        assert fields["session_id"] == "20240101-000000-abcd" and len(messages) == 2

    def test_bad_file(self, tmp_path):
        bad = tmp_path / "x.json"
        bad.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_session(bad)

    def test_chat_saves_and_resumes(self, tmp_path):
        first = _session(tmp_path, [_text("first answer")], save_sessions=True)
        first.send("remember the number 7")
        assert first.session_path.is_file()

        second = _session(tmp_path, [_text("it was 7")], save_sessions=True)
        path = find_session(tmp_path / "logs", "latest", workspace=tmp_path)
        second.resume(path)
        assert second.session_id == first.session_id
        assert second.totals.tasks == 1
        second.send("what was the number?")
        sent = second.provider.calls[0]["messages"]
        assert any("remember the number 7" in m.content for m in sent)
        assert any(m.content == "first answer" for m in sent)
        _, saved = load_session(second.session_path)
        assert saved[-1].content == "it was 7"

    def test_clear_starts_a_new_session(self, tmp_path):
        session = _session(tmp_path, [_text("a")], save_sessions=True)
        session.send("hi")
        old = session.session_id
        session.clear()
        assert session.session_id != old and session.totals.tasks == 0

    def test_saving_off_by_default(self, tmp_path):
        session = _session(tmp_path, [_text("a")])
        session.send("hi")
        assert not (tmp_path / "logs" / "sessions").exists()

    def test_sessions_command(self, tmp_path):
        session = _session(tmp_path, [_text("a")], save_sessions=True)
        session.send("first task")
        out = []
        handle_command(session, "/sessions", out.append)
        assert any(session.session_id in line and "first task" in line for line in out)


# ---------------------------------------------------------------------------
# 6. Chat: interrupted tasks, approval toggle, streamed output
# ---------------------------------------------------------------------------


class TestChatAdditions:
    def test_undo_covers_an_interrupted_task(self, tmp_path):
        session = _session(tmp_path, [
            _tools(("a", "write_file", {"path": "made.txt", "content": "1"})),
            KeyboardInterrupt(),
        ])
        run, changes = session.send("make a file")
        assert run.status == STATUS_INTERRUPTED
        assert changes.created == ["made.txt"]
        session.undo()
        assert not (tmp_path / "made.txt").exists()

    def test_approve_command(self, tmp_path):
        session = _session(tmp_path, [], approve=lambda c: True)
        out = []
        handle_command(session, "/approve on", out.append)
        assert session.approval_on and out[-1].endswith("on")
        handle_command(session, "/approve off", out.append)
        assert not session.approval_on
        no_approver = _session(tmp_path, [])
        handle_command(no_approver, "/approve on", out.append)
        assert not no_approver.approval_on

    def test_repl_does_not_repeat_a_streamed_answer(self, tmp_path):
        written = []
        output = TerminalOutput(write=written.append)
        session = ChatSession(
            settings=_Settings(tmp_path), workspace=tmp_path,
            provider=Scripted([_text("streamed answer")], streaming=True),
            write_trace=False, on_text=output.text,
        )
        lines = iter(["do it", "/exit"])
        out = []
        run_repl(session, input_fn=lambda _: next(lines), out=out.append, output=output)
        assert "".join(written).count("streamed answer") == 1
        assert not any("streamed answer" in line for line in out)


class TestTerminalHelpers:
    def test_output_keeps_tool_lines_on_their_own_line(self):
        written = []
        output = TerminalOutput(write=written.append)
        output.text("Let me look")
        output.tool("read_file", {"path": "a.py"})
        output.text("Done")
        output.end_reply()
        assert "".join(written) == "Let me look\n  · read_file a.py\nDone\n"
        assert output.streamed

    def test_approver_answers(self):
        answers = iter(["y", "n", "", "a", "no, run the tests first"])
        shown = []
        approver = TerminalApprover(input_fn=lambda _: next(answers), out=shown.append)
        call = ToolCall(id="1", name="run_shell", arguments={"command": "rm x"})
        assert approver(call) is True
        assert approver(call) is False
        assert approver(call) is False  # Enter alone = no
        assert approver(call) is True  # always
        assert approver(call) is True  # not asked again
        edit = ToolCall(id="2", name="edit_file", arguments={"path": "a.py", "old_string": "x", "new_string": "y"})
        assert approver(edit) == "no, run the tests first"
        assert "rm x" in shown[0]

    def test_approver_eof_means_no(self):
        def eof(_):
            raise EOFError

        assert TerminalApprover(input_fn=eof, out=lambda s: None)(
            ToolCall(id="1", name="write_file", arguments={})) is False

    def test_describe_action(self):
        edit = describe_action(ToolCall(id="1", name="edit_file", arguments={
            "path": "a.py", "old_string": "x = 1\n", "new_string": "x = 2\n"}))
        assert "edit_file a.py" in edit and "-x = 1" in edit and "+x = 2" in edit
        write = describe_action(ToolCall(id="2", name="write_file", arguments={
            "path": "b.py", "content": "\n".join(str(i) for i in range(30))}))
        assert "(create) b.py — 30 lines" in write and "10 more lines" in write


# ---------------------------------------------------------------------------
# 7. Benchmark: verify scripts, repetitions, jobs, report
# ---------------------------------------------------------------------------


class TestVerifyScripts:
    def test_pass_and_fail(self, tmp_path):
        (tmp_path / "f.txt").write_text("good", encoding="utf-8")
        ok = "import sys, pathlib\nsys.exit(0 if (pathlib.Path(sys.argv[1]) / 'f.txt').read_text() == 'good' else 1)\n"
        assert run_verify_script(ok, tmp_path) == (True, "verify script: passed")
        passed, detail = run_verify_script("print('nope'); raise SystemExit(1)", tmp_path)
        assert not passed and "nope" in detail

    def test_relative_workspace_path(self, tmp_path, monkeypatch):
        (tmp_path / "ws").mkdir()
        (tmp_path / "ws" / "f.txt").write_text("good", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        ok = "import sys, pathlib\nsys.exit(0 if (pathlib.Path(sys.argv[1]) / 'f.txt').is_file() else 1)\n"
        assert run_verify_script(ok, pathlib.Path("ws"))[0] is True

    def test_verify_counts_as_a_check(self, tmp_path):
        task = EvalTask(task_id="t", prompt="p", verify_script="raise SystemExit(1)")
        outcome = task.run_checks(tmp_path)
        assert outcome.passed is False and outcome.checks_run == 1
        assert "verify_script" not in EvalTask(task_id="t", prompt="p").to_dict()

    def test_suite_loads_verify_and_skips_pycache(self, tmp_path):
        (tmp_path / "fx" / "__pycache__").mkdir(parents=True)
        (tmp_path / "fx" / "__pycache__" / "m.cpython-313.pyc").write_bytes(b"\x00\xff")
        (tmp_path / "fx" / "m.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "v.py").write_text("print('ok')\n", encoding="utf-8")
        (tmp_path / "t.json").write_text(json.dumps({
            "task_id": "t", "prompt": "p", "fixtures": {"dir": "fx"}, "verify": "v.py",
        }), encoding="utf-8")
        task = load_task(tmp_path / "t.json")
        assert [f.path for f in task.fixtures] == ["m.py"]
        assert task.verify_script == "print('ok')\n"
        (tmp_path / "t.json").write_text(json.dumps({
            "task_id": "t", "prompt": "p", "verify": "missing.py"}), encoding="utf-8")
        with pytest.raises(TaskSuiteError):
            load_task(tmp_path / "t.json")


def test_long_suite_loads_and_unsolved_tasks_fail(tmp_path):
    suite = load_suite(LONG_TASKS_DIR)
    assert [t.task_id for t in suite] == [
        "ledger-fix-bugs", "ledger-recurring-feature", "ledger-rename-memo", "ledger-api-docs",
    ]
    assert all(t.verify_script for t in suite)
    bugs = suite[0]
    bugs.provision(tmp_path)
    outcome = bugs.run_checks(tmp_path)
    assert outcome.passed is False
    assert any("parse_amount" in d for d in outcome.details)


class _Factory:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0

    def __call__(self):
        with self.lock:
            self.count += 1
        return Scripted([_tools(("a", "write_file", {"path": "out.txt", "content": "x"})),
                         LLMResponse(content="done", stop_reason="stop",
                                     input_tokens=100, output_tokens=10, total_tokens=110)])


class _MatrixSettings:
    llm_provider = "scripted"
    llm_model = "fake-model"
    llm_temperature = 0.0
    llm_max_output_tokens = 0
    max_llm_turns = 5
    max_tool_calls = 5


class TestRepetitionsAndJobs:
    @pytest.mark.parametrize("jobs", [1, 3])
    def test_repetitions(self, tmp_path, jobs):
        task = EvalTask(task_id="t1", prompt="write out.txt")
        task2 = EvalTask(task_id="t2", prompt="write out.txt")
        arms = [ALL_ARMS_BY_ID["fixed_raw"], ALL_ARMS_BY_ID["adaptive_stable"]]
        factory = _Factory()
        progress = []
        runner = MatrixRunner(settings=_MatrixSettings(), provider_factory=factory,
                              output_root=tmp_path, write_trace=False)
        results, summary = runner.run(
            [task, task2], arms=arms, experiment_id="e", repetitions=2, jobs=jobs,
            on_result=lambda r, done, total: progress.append((done, total)),
        )
        assert len(results) == 8 and factory.count == 8
        assert [(r.repetition, r.task_id, r.arm_id) for r in results][:4] == [
            (1, "t1", "fixed_raw"), (1, "t1", "adaptive_stable"),
            (1, "t2", "fixed_raw"), (1, "t2", "adaptive_stable"),
        ]
        assert {r.repetition for r in results} == {1, 2}
        assert len({r.workspace for r in results}) == 8
        assert all(r.success for r in results)
        assert sorted(progress) == [(i, 8) for i in range(1, 9)]
        assert summary.repetitions == 2
        lines = (tmp_path / "e" / "results.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 8
        meta = json.loads((tmp_path / "e" / "metadata.json").read_text(encoding="utf-8"))
        assert meta["repetitions"] == 2

    def test_single_run_keeps_old_layout(self, tmp_path):
        runner = MatrixRunner(settings=_MatrixSettings(), provider_factory=_Factory(),
                              output_root=tmp_path, write_trace=False)
        results, _ = runner.run([EvalTask(task_id="t1", prompt="p")],
                                arms=[ALL_ARMS_BY_ID["fixed_raw"]], experiment_id="e")
        assert results[0].repetition is None
        assert results[0].workspace.replace("\\", "/").endswith("e/workspaces/t1/fixed_raw")

    def test_bad_arguments(self, tmp_path):
        runner = MatrixRunner(settings=_MatrixSettings(), provider_factory=_Factory(),
                              output_root=tmp_path, write_trace=False)
        with pytest.raises(ValueError):
            runner.run([EvalTask(task_id="t1", prompt="p")], repetitions=0)
        with pytest.raises(ValueError):
            runner.run([EvalTask(task_id="t1", prompt="p")], jobs=0)

    def test_peak_input_is_recorded(self, tmp_path):
        provider = Scripted([
            LLMResponse(content="", tool_calls=[ToolCall(id="a", name="list_directory",
                                                         arguments={"path": "."})],
                        stop_reason="tool_use", input_tokens=300),
            LLMResponse(content="done", stop_reason="stop", input_tokens=500),
        ])
        run = _runtime(tmp_path, provider).run()
        assert run.peak_input_tokens == 500


def _row(arm, task, total, rep):
    return EvalResult(
        task_id=task, run_id=f"{arm}{task}{rep}", provider="p", model="m",
        tool_strategy="fixed", context_strategy="raw", status="completed",
        success=True, total_tokens=total, input_tokens=total, cached_input_tokens=0,
        peak_input_tokens=total // 2, arm_id=arm, repetition=rep,
    )


def test_report_spread_and_change_vs_baseline():
    rows = [
        _row("fixed_raw", "a", 1000, 1), _row("fixed_raw", "a", 1000, 2),
        _row("fixed_raw", "b", 2000, 1), _row("fixed_raw", "b", 2000, 2),
        _row("fixed_managed", "a", 900, 1), _row("fixed_managed", "a", 1100, 2),
        _row("fixed_managed", "b", 1000, 1), _row("fixed_managed", "b", 1000, 2),
    ]
    reports = {r.arm_id: r for r in build_report(rows)}
    base, managed = reports["fixed_raw"], reports["fixed_managed"]
    assert base.vs_baseline is None and base.spread == pytest.approx(0)
    # task a: 1000 vs 1000 -> 0%; task b: 1000 vs 2000 -> -50%; mean -25%
    assert managed.vs_baseline == pytest.approx(-0.25)
    # task a: stdev(900, 1100) / 1000 = 0.1414; task b: 0
    assert managed.spread == pytest.approx(0.1414 / 2, abs=1e-3)
    assert managed.mean_peak_input == pytest.approx(500)
    table = format_report(list(reports.values()))
    assert "-25%" in table and "base" in table and "2 repetitions" in table


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def test_cli_bench_options_parse():
    args = cli._build_parser().parse_args(
        ["bench", "--suite", "long", "--repeat", "3", "--jobs", "4"]
    )
    assert (args.suite, args.repeat, args.jobs) == ("long", 3, 4)
    args = cli._build_parser().parse_args(["chat", "--resume"])
    assert args.resume == "latest"
    args = cli._build_parser().parse_args(["chat", "--resume", "2024", "--approve"])
    assert args.resume == "2024" and args.approve


def test_cli_bench_rejects_bad_repeat(capsys):
    assert cli.main(["bench", "--repeat", "0"]) == 2


def test_cli_chat_resume_without_sessions(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FORGE_LLM_API_KEY", "k")
    monkeypatch.setenv("FORGE_LOG_DIR", str(tmp_path / "logs"))
    assert cli.main(["chat", "--workspace", str(tmp_path), "--resume"]) == 2
    assert "no saved session" in capsys.readouterr().err


def test_cli_run_with_approve_declines(tmp_path, monkeypatch, capsys):
    provider = Scripted([
        _tools(("a", "write_file", {"path": "x.txt", "content": "1"})),
        _text("I could not write it."),
    ])
    fake = lambda settings_override=None: provider  # noqa: E731
    monkeypatch.setattr("forge.llm.get_provider", fake)
    monkeypatch.setattr("forge.agent.get_provider", fake)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    code = cli.main(["run", "write x", "--workspace", str(tmp_path), "--no-trace", "--approve"])
    assert code == 0
    assert not (tmp_path / "x.txt").exists()
    out = capsys.readouterr().out
    assert "? write_file (create) x.txt" in out and "I could not write it." in out
