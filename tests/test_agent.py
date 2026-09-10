"""
Tests for forge.agent — the FORGE agent loop.

The LLM is a scripted fake; tools are the real built-ins bound to a tmp
workspace.  One mocked end-to-end test drives a two-step coding task.
"""

from __future__ import annotations

import json

import pytest

from forge.agent import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_MAX_TOOL_CALLS,
    STATUS_MAX_TURNS,
    STATUS_PROVIDER_ERROR,
    AgentConfig,
    AgentRuntime,
)
from forge.builtin_tools import build_default_registry
from forge.llm import LLMError, LLMProvider, LLMResponse, ToolCall
from forge.tools import FixedToolExposure


# ---------------------------------------------------------------------------
# Scripted providers
# ---------------------------------------------------------------------------


class ScriptedProvider(LLMProvider):
    """Returns a pre-scripted list of LLMResponses, one per complete() call."""

    def __init__(self, responses, *, model="fake-model"):
        self._responses = list(responses)
        self._model = model
        self.calls = []  # list of {messages, tools}

    @property
    def provider_name(self) -> str:
        return "scripted"

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, messages, *, tools=None, max_tokens=4096, temperature=0.0):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._responses:
            raise AssertionError("ScriptedProvider ran out of responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class AlwaysToolProvider(LLMProvider):
    """Always asks for the same tool call — used to hit loop ceilings."""

    def __init__(self, tool_call):
        self._tool_call = tool_call
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "always-tool"

    @property
    def model_name(self) -> str:
        return "fake"

    def complete(self, messages, *, tools=None, max_tokens=4096, temperature=0.0):
        self.calls += 1
        return LLMResponse(content="", tool_calls=[self._tool_call], stop_reason="tool_use")


def _text(content, **usage):
    return LLMResponse(content=content, stop_reason="stop", **usage)


def _tool(name, arguments, call_id="c1", text=""):
    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        stop_reason="tool_use",
    )


@pytest.fixture
def registry(tmp_path):
    return build_default_registry(workspace_root=tmp_path)


def _runtime(config, provider, registry, tmp_path, **kw):
    return AgentRuntime(
        config,
        provider=provider,
        registry=registry,
        workspace_root=tmp_path,
        log_dir=tmp_path / "logs",
        **kw,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestBasicLoop:
    def test_plain_completion_no_tools(self, registry, tmp_path):
        provider = ScriptedProvider([_text("All done.")])
        run = _runtime(AgentConfig(task="say done"), provider, registry, tmp_path).run()
        assert run.status == STATUS_COMPLETED
        assert run.final_answer == "All done."
        assert run.llm_calls == 1
        assert run.tool_calls == 0
        assert run.turns == 1
        assert run.trace_path is not None and run.trace_path.exists()

    def test_single_tool_call_then_finish(self, registry, tmp_path):
        provider = ScriptedProvider(
            [
                _tool("write_file", {"path": "add.py", "content": "def add(a,b): return a+b\n"}),
                _text("Created add.py."),
            ]
        )
        run = _runtime(AgentConfig(task="make add.py"), provider, registry, tmp_path).run()
        assert run.status == STATUS_COMPLETED
        assert run.llm_calls == 2
        assert run.tool_calls == 1
        assert (tmp_path / "add.py").read_text(encoding="utf-8").startswith("def add")

    def test_end_to_end_two_step_edit(self, registry, tmp_path):
        """LLM -> tool -> result -> LLM -> tool -> result -> LLM -> done."""
        provider = ScriptedProvider(
            [
                _tool("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}, "c1"),
                _tool(
                    "edit_file",
                    {
                        "path": "calc.py",
                        "old_string": "def add(a, b):\n    return a + b\n",
                        "new_string": "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n",
                    },
                    "c2",
                ),
                _text("calc.py now has add() and multiply()."),
            ]
        )
        run = _runtime(AgentConfig(task="add then multiply"), provider, registry, tmp_path).run()
        assert run.status == STATUS_COMPLETED
        assert run.llm_calls == 3
        assert run.tool_calls == 2
        source = (tmp_path / "calc.py").read_text(encoding="utf-8")
        assert "def add(a, b):" in source
        assert "def multiply(a, b):" in source
        # context = system + user + 2*(assistant tool-call + tool result) + final assistant
        assert run.context_messages == 7


# ---------------------------------------------------------------------------
# Failure modes (all recovered, never crash)
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_unknown_tool_is_fed_back_and_run_continues(self, registry, tmp_path):
        provider = ScriptedProvider([_tool("teleport", {"x": 1}), _text("gave up on teleport")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.status == STATUS_COMPLETED
        assert run.tool_calls == 1
        tool_msgs = [m for m in provider.calls[1]["messages"] if m.role == "tool"]
        assert tool_msgs and "unknown tool" in tool_msgs[0].content.lower()

    def test_malformed_tool_call_arguments_not_a_dict(self, registry, tmp_path):
        bad = LLMResponse(
            tool_calls=[ToolCall(id="c1", name="read_file", arguments="path=foo")],
            stop_reason="tool_use",
        )
        provider = ScriptedProvider([bad, _text("recovered")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.status == STATUS_COMPLETED
        tool_msgs = [m for m in provider.calls[1]["messages"] if m.role == "tool"]
        assert "must be a json object" in tool_msgs[0].content.lower()

    def test_invalid_tool_arguments_are_reported(self, registry, tmp_path):
        provider = ScriptedProvider([_tool("read_file", {}), _text("done")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.status == STATUS_COMPLETED
        tool_msgs = [m for m in provider.calls[1]["messages"] if m.role == "tool"]
        assert "invalid arguments" in tool_msgs[0].content.lower()

    def test_tool_failure_is_reported(self, registry, tmp_path):
        provider = ScriptedProvider([_tool("read_file", {"path": "missing.py"}), _text("done")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.status == STATUS_COMPLETED
        tool_msgs = [m for m in provider.calls[1]["messages"] if m.role == "tool"]
        assert "not found" in tool_msgs[0].content.lower()

    def test_provider_error_stops_run_cleanly(self, registry, tmp_path):
        provider = ScriptedProvider([LLMError("gemini exploded")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.status == STATUS_PROVIDER_ERROR
        assert "gemini exploded" in run.error
        assert run.trace_path.exists()

    def test_unexpected_exception_marks_run_failed(self, registry, tmp_path):
        class Boom(LLMProvider):
            provider_name = "boom"
            model_name = "boom"

            def complete(self, *a, **k):
                raise RuntimeError("kaboom")

        run = _runtime(AgentConfig(task="t"), Boom(), registry, tmp_path).run()
        assert run.status == STATUS_FAILED
        assert "RuntimeError" in run.error


# ---------------------------------------------------------------------------
# Loop ceilings
# ---------------------------------------------------------------------------


class TestLimits:
    def test_max_llm_turns_enforced(self, registry, tmp_path):
        provider = AlwaysToolProvider(ToolCall(id="c", name="list_directory", arguments={}))
        run = _runtime(
            AgentConfig(task="loop", max_llm_turns=4), provider, registry, tmp_path
        ).run()
        assert run.status == STATUS_MAX_TURNS
        assert run.llm_calls == 4
        assert provider.calls == 4

    def test_max_tool_calls_enforced(self, registry, tmp_path):
        many = LLMResponse(
            tool_calls=[
                ToolCall(id=f"c{i}", name="list_directory", arguments={}) for i in range(5)
            ],
            stop_reason="tool_use",
        )
        provider = ScriptedProvider([many, _text("unreached")])
        run = _runtime(
            AgentConfig(task="t", max_tool_calls=2), provider, registry, tmp_path
        ).run()
        assert run.status == STATUS_MAX_TOOL_CALLS
        assert run.tool_calls == 2

    def test_max_turns_one_allows_single_completion(self, registry, tmp_path):
        provider = ScriptedProvider([_text("quick answer")])
        run = _runtime(
            AgentConfig(task="t", max_llm_turns=1), provider, registry, tmp_path
        ).run()
        assert run.status == STATUS_COMPLETED


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class TestTokenAccounting:
    def test_usage_is_summed_across_turns(self, registry, tmp_path):
        first = _tool("list_directory", {}, "c1")
        first.input_tokens, first.output_tokens, first.total_tokens = 80, 10, 90
        provider = ScriptedProvider(
            [first, _text("done", input_tokens=100, output_tokens=20, total_tokens=120)]
        )
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.input_tokens == 180
        assert run.output_tokens == 30
        assert run.total_tokens == 210

    def test_missing_usage_stays_none_not_zero(self, registry, tmp_path):
        provider = ScriptedProvider([_text("no usage reported")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.input_tokens is None
        assert run.output_tokens is None
        assert run.total_tokens is None

    def test_partial_usage_tracked_independently(self, registry, tmp_path):
        provider = ScriptedProvider([_text("partial", input_tokens=50)])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        assert run.input_tokens == 50
        assert run.output_tokens is None
        assert run.total_tokens is None


# ---------------------------------------------------------------------------
# Observability wiring
# ---------------------------------------------------------------------------


class TestObservabilityWiring:
    def _events(self, run):
        return [
            json.loads(line)
            for line in run.trace_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_trace_contains_full_event_sequence(self, registry, tmp_path):
        provider = ScriptedProvider([_tool("list_directory", {}), _text("done")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        kinds = [e["event_type"] for e in self._events(run)]
        assert kinds[0] == "run_start"
        assert kinds[-1] == "run_end"
        for expected in ("llm_call", "llm_response", "tool_call", "tool_result"):
            assert expected in kinds

    def test_run_start_records_experiment_metadata(self, registry, tmp_path):
        provider = ScriptedProvider([_text("done")])
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        start = self._events(run)[0]["data"]
        assert start["provider"] == "scripted"
        assert start["model"] == "fake-model"
        assert start["tool_exposure_strategy"] == "fixed"
        assert start["context_strategy"] == "raw"
        assert start["system_prompt_id"].startswith("sha256:")
        assert start["temperature"] is not None
        assert start["max_output_tokens"] is not None

    def test_run_end_carries_token_totals(self, registry, tmp_path):
        provider = ScriptedProvider(
            [_text("done", input_tokens=10, output_tokens=4, total_tokens=14)]
        )
        run = _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        end = self._events(run)[-1]["data"]
        assert end["input_tokens"] == 10
        assert end["total_tokens"] == 14

    def test_no_trace_when_disabled(self, registry, tmp_path):
        provider = ScriptedProvider([_text("done")])
        run = _runtime(
            AgentConfig(task="t"), provider, registry, tmp_path, write_trace=False
        ).run()
        assert run.trace_path is None


# ---------------------------------------------------------------------------
# Fixed tool exposure (research seam A — must stay swappable, not hard-coded)
# ---------------------------------------------------------------------------


class TestFixedToolExposure:
    def test_all_tools_exposed_every_turn_by_default(self, registry, tmp_path):
        provider = ScriptedProvider([_tool("list_directory", {}), _text("done")])
        _runtime(AgentConfig(task="t"), provider, registry, tmp_path).run()
        for call in provider.calls:
            names = {t["function"]["name"] for t in call["tools"]}
            assert names == set(registry.list_names())
            assert len(names) == 5

    def test_pinned_subset_limits_exposure_but_stays_fixed(self, registry, tmp_path):
        provider = ScriptedProvider([_text("done")])
        config = AgentConfig(task="t", tool_exposure=FixedToolExposure(pinned=["read_file"]))
        _runtime(config, provider, registry, tmp_path).run()
        names = {t["function"]["name"] for t in provider.calls[0]["tools"]}
        assert names == {"read_file"}

    def test_tool_subset_convenience_field_is_wired_in(self, registry, tmp_path):
        provider = ScriptedProvider([_text("done")])
        config = AgentConfig(task="t", tool_subset=["read_file", "write_file"])
        run = _runtime(config, provider, registry, tmp_path).run()
        names = {t["function"]["name"] for t in provider.calls[0]["tools"]}
        assert names == {"read_file", "write_file"}
        start = json.loads(run.trace_path.read_text(encoding="utf-8").splitlines()[0])
        assert start["data"]["tool_subset"] == ["read_file", "write_file"]

    def test_runtime_uses_the_strategy_abstraction(self):
        # Guard rail: the loop calls ToolExposureStrategy.select(); it does not
        # branch on a hard-coded adaptive flag.
        import inspect

        from forge import agent

        src = inspect.getsource(agent)
        assert "tool_exposure.select(" in src
        assert "adaptive" not in src.lower()
