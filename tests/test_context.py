"""
Tests for forge.context — ConversationContext container and context strategies.
"""

from __future__ import annotations

import pytest

from forge.context import (
    ContextReport,
    ContextStrategy,
    ConversationContext,
    ManagedContextStrategy,
    RawContextStrategy,
)
from forge.llm import Message, ToolCall


class TestConversationContext:
    def test_add_and_get_messages(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="system", content="sys"))
        ctx.add_message(Message(role="user", content="hello"))
        msgs = ctx.get_messages()
        assert [m.role for m in msgs] == ["system", "user"]

    def test_get_messages_returns_a_copy(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="a"))
        got = ctx.get_messages()
        got.append(Message(role="user", content="b"))
        assert ctx.message_count == 1

    def test_clear(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="a"))
        ctx.clear()
        assert ctx.message_count == 0

    def test_approximate_char_count(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="user", content="12345"))
        ctx.add_message(Message(role="assistant", content="678"))
        assert ctx.approximate_char_count == 8

    def test_max_messages_warning(self):
        import logging

        logger = logging.getLogger("forge.context")
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[assignment]
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            ctx = ConversationContext(max_messages=2)
            for _ in range(3):
                ctx.add_message(Message(role="user", content="x"))
        finally:
            logger.removeHandler(handler)
        assert any("max_messages" in r.getMessage() for r in records)

    def test_tool_call_message_is_stored_intact(self):
        ctx = ConversationContext()
        call = ToolCall(id="c1", name="read_file", arguments={"path": "a"})
        ctx.add_message(Message(role="assistant", content="", tool_calls=[call]))
        stored = ctx.get_messages()[0]
        assert stored.tool_calls[0].name == "read_file"


class TestRawContextStrategy:
    def test_is_a_context_strategy(self):
        assert isinstance(RawContextStrategy(), ContextStrategy)

    def test_name_is_raw(self):
        assert RawContextStrategy().name == "raw"

    def test_prepare_returns_full_history_unmodified(self):
        ctx = ConversationContext()
        for i in range(5):
            ctx.add_message(Message(role="user", content=f"m{i}"))
        prepared = RawContextStrategy().prepare(ctx)
        assert [m.content for m in prepared] == ["m0", "m1", "m2", "m3", "m4"]

    def test_prepare_reflects_later_additions(self):
        ctx = ConversationContext()
        strategy = RawContextStrategy()
        ctx.add_message(Message(role="user", content="first"))
        assert len(strategy.prepare(ctx)) == 1
        ctx.add_message(Message(role="assistant", content="second"))
        assert len(strategy.prepare(ctx)) == 2

    def test_cannot_instantiate_abstract_strategy(self):
        with pytest.raises(TypeError):
            ContextStrategy()  # type: ignore[abstract]

    def test_default_last_report_is_none(self):
        # Base class default: a strategy that never overrides last_report()
        # reports nothing, rather than fabricating a ContextReport.
        class _Bare(ContextStrategy):
            name = "bare"

            def prepare(self, context):
                return context.get_messages()

        assert _Bare().last_report() is None

    def test_last_report_reflects_the_full_unmodified_history(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="system", content="sys"))
        ctx.add_message(Message(role="user", content="task"))
        strategy = RawContextStrategy()
        prepared = strategy.prepare(ctx)
        report = strategy.last_report()
        assert isinstance(report, ContextReport)
        assert report.items_before == report.items_after == len(prepared) == 2
        assert report.chars_before == report.chars_after == len("sys") + len("task")
        assert report.items_dropped == 0
        assert report.items_compressed == 0


# ---------------------------------------------------------------------------
# ManagedContextStrategy (Step 6)
# ---------------------------------------------------------------------------


def _build_ctx(
    n_turns: int,
    *,
    tool_name="read_file",
    content=None,
    task: str = "the original task",
) -> ConversationContext:
    """Build a context with a system + task message, then *n_turns* turns.

    Each turn is (assistant-with-one-tool-call, tool-result) — the exact
    shape ``AgentRuntime`` produces. ``tool_name`` may be a fixed string or a
    callable(i) -> str; ``content`` may be a callable(i) -> str.

    Default assistant/tool content is deliberately padded past the length of
    ManagedContextStrategy's compression markers (~53/~64 chars) so that
    tests exercise real compression, not the "already shorter than its own
    placeholder" no-op guard — realistic tool output (file contents,
    directory listings, shell output) is almost always longer than a short
    marker; short literal test fixtures are the unrealistic case.
    """
    ctx = ConversationContext()
    ctx.add_message(Message(role="system", content="You are FORGE."))
    ctx.add_message(Message(role="user", content=task))
    for i in range(n_turns):
        name = tool_name(i) if callable(tool_name) else tool_name
        text = content(i) if callable(content) else f"result-{i}: " + ("detail " * 12)
        ctx.add_message(
            Message(
                role="assistant",
                content=f"calling {name} (turn {i}): " + ("thinking " * 8),
                tool_calls=[ToolCall(id=f"c{i}", name=name, arguments={})],
            )
        )
        ctx.add_message(
            Message(role="tool", content=text, name=name, tool_call_id=f"c{i}")
        )
    return ctx


class TestManagedContextStrategyBasics:
    def test_is_a_context_strategy_named_managed(self):
        assert isinstance(ManagedContextStrategy(), ContextStrategy)
        assert ManagedContextStrategy().name == "managed"

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError):
            ManagedContextStrategy(keep_recent_turns=-1)
        with pytest.raises(ValueError):
            ManagedContextStrategy(max_messages=0)
        with pytest.raises(ValueError):
            ManagedContextStrategy(max_chars=0)

    def test_empty_history(self):
        ctx = ConversationContext()
        strategy = ManagedContextStrategy()
        prepared = strategy.prepare(ctx)
        assert prepared == []
        report = strategy.last_report()
        assert report.items_before == 0
        assert report.items_after == 0
        assert report.chars_before == 0
        assert report.chars_after == 0
        assert report.turns_total == 0

    def test_very_small_history_system_and_task_only_matches_raw(self):
        ctx = ConversationContext()
        ctx.add_message(Message(role="system", content="sys"))
        ctx.add_message(Message(role="user", content="do the thing"))
        managed = ManagedContextStrategy().prepare(ctx)
        raw = RawContextStrategy().prepare(ctx)
        assert [(m.role, m.content) for m in managed] == [
            (m.role, m.content) for m in raw
        ]

    def test_single_turn_history_kept_in_full(self):
        # One turn is always within the (default) recent window, so nothing
        # is compressed or dropped.
        ctx = _build_ctx(1)
        original = ctx.get_messages()
        strategy = ManagedContextStrategy()
        prepared = strategy.prepare(ctx)
        assert [m.content for m in prepared] == [m.content for m in original]
        assert prepared[0].content == "You are FORGE."
        assert prepared[1].content == "the original task"
        report = strategy.last_report()
        assert report.items_dropped == 0
        assert report.items_compressed == 0

    def test_deterministic_same_input_same_output(self):
        ctx = _build_ctx(8)
        out_a = ManagedContextStrategy(keep_recent_turns=2).prepare(ctx)
        out_b = ManagedContextStrategy(keep_recent_turns=2).prepare(ctx)
        as_tuples = lambda out: [  # noqa: E731
            (m.role, m.content, m.name, m.tool_call_id) for m in out
        ]
        assert as_tuples(out_a) == as_tuples(out_b)

    def test_repeated_prepare_calls_on_same_strategy_are_stable(self):
        ctx = _build_ctx(6)
        strategy = ManagedContextStrategy(keep_recent_turns=2)
        first = strategy.prepare(ctx)
        second = strategy.prepare(ctx)
        assert [(m.role, m.content) for m in first] == [
            (m.role, m.content) for m in second
        ]


class TestManagedContextStrategyPreservation:
    def test_system_and_task_always_preserved_unmodified(self):
        ctx = _build_ctx(10, task="fix the bug in ranges.py")
        strategy = ManagedContextStrategy(keep_recent_turns=1, max_messages=6, max_chars=100)
        prepared = strategy.prepare(ctx)
        assert prepared[0].role == "system"
        assert prepared[0].content == "You are FORGE."
        assert prepared[1].role == "user"
        assert prepared[1].content == "fix the bug in ranges.py"

    def test_recent_window_kept_unmodified(self):
        # Distinct tool names per turn so the "latest per tool" rule cannot
        # coincidentally also preserve an older turn.
        names = ["read_file", "list_directory", "edit_file", "write_file"]
        ctx = _build_ctx(4, tool_name=lambda i: names[i])
        strategy = ManagedContextStrategy(keep_recent_turns=2)
        prepared = strategy.prepare(ctx)
        contents = [m.content for m in prepared]
        # the last two turns' assistant + tool messages are untouched
        assert any(c.startswith("calling edit_file (turn 2)") for c in contents)
        assert any(c.startswith("result-2:") for c in contents)
        assert any(c.startswith("calling write_file (turn 3)") for c in contents)
        assert any(c.startswith("result-3:") for c in contents)

    def test_older_turn_with_error_prefix_is_preserved_in_full(self):
        def content(i):
            return "ERROR: file not found: missing.py" if i == 1 else f"result-{i}"

        ctx = _build_ctx(5, tool_name="read_file", content=content)
        strategy = ManagedContextStrategy(keep_recent_turns=1)
        prepared = strategy.prepare(ctx)
        error_messages = [m for m in prepared if m.content.startswith("ERROR")]
        assert len(error_messages) == 1
        assert error_messages[0].content == "ERROR: file not found: missing.py"
        assert error_messages[0].name == "read_file"

    def test_older_turn_with_nonzero_exit_code_is_preserved_in_full(self):
        def content(i):
            if i == 1:
                return "[stderr]\nFAILED tests/test_x.py\n[exit code 1]"
            return f"[exit code 0] result-{i}"

        ctx = _build_ctx(5, tool_name="run_shell", content=content)
        strategy = ManagedContextStrategy(keep_recent_turns=1)
        prepared = strategy.prepare(ctx)
        preserved = [m for m in prepared if "FAILED tests/test_x.py" in m.content]
        assert len(preserved) == 1
        assert preserved[0].content.endswith("[exit code 1]")

    def test_stale_redundant_successful_output_compressed_latest_kept(self):
        ctx = _build_ctx(5, tool_name="read_file")  # turns 0..4, all read_file
        strategy = ManagedContextStrategy(keep_recent_turns=1)
        prepared = strategy.prepare(ctx)
        tool_msgs = [m for m in prepared if m.role == "tool"]
        # turn 4 is in the recent window (kept); turn 3 is the latest
        # successful older call (kept); turns 0-2 are compressed.
        kept = {m.content for m in tool_msgs if not m.content.startswith("[MANAGED CONTEXT")}
        compressed = [m for m in tool_msgs if m.content.startswith("[MANAGED CONTEXT")]
        assert len(kept) == 2
        assert any(k.startswith("result-3:") for k in kept)
        assert any(k.startswith("result-4:") for k in kept)
        assert len(compressed) == 3
        for m in compressed:
            assert m.name == "read_file"
            assert m.tool_call_id is not None  # pairing preserved
            assert m.metadata.get("managed_context_compressed") is True

        report = strategy.last_report()
        # turns 0-2: both assistant + tool compressed (2 each). turn 3: its
        # tool result is the preserved "latest for read_file" but its
        # assistant text is still stale and gets compressed too.
        assert report.items_compressed == 3 * 2 + 1
        assert report.items_dropped == 0

    def test_tool_call_result_pairing_never_broken(self):
        ctx = _build_ctx(6, tool_name="read_file")
        strategy = ManagedContextStrategy(keep_recent_turns=1, max_messages=4)
        prepared = strategy.prepare(ctx)
        tool_call_ids = {
            c.id for m in prepared if m.role == "assistant" for c in m.tool_calls
        }
        tool_result_ids = {m.tool_call_id for m in prepared if m.role == "tool"}
        assert tool_call_ids == tool_result_ids


class TestManagedContextStrategyBudget:
    def test_budget_drops_oldest_turns_first(self):
        ctx = _build_ctx(10, content=lambda i: "x" * 50)
        strategy = ManagedContextStrategy(
            keep_recent_turns=1, max_messages=6, max_chars=1_000_000
        )
        prepared = strategy.prepare(ctx)
        report = strategy.last_report()
        assert report.turns_dropped_for_budget > 0
        # head (2) + at most max_messages-... budget respected on message count
        assert len(prepared) <= strategy.max_messages
        contents = [m.content for m in prepared]
        # the oldest turn's distinctive content must be gone entirely
        assert not any(c.startswith("calling read_file (turn 0):") for c in contents)
        # the most recent turn always survives
        assert any(c.startswith("calling read_file (turn 9):") for c in contents)

    def test_char_budget_enforced(self):
        ctx = _build_ctx(10, content=lambda i: "y" * 200)
        strategy = ManagedContextStrategy(
            keep_recent_turns=1, max_messages=1000, max_chars=500
        )
        prepared = strategy.prepare(ctx)
        report = strategy.last_report()
        assert report.chars_after <= 500 or report.turns_dropped_for_budget == 9
        assert report.turns_dropped_for_budget > 0

    def test_protected_content_is_never_dropped_even_over_budget(self):
        # An impossible budget: recent window + errors alone cannot fit.
        def content(i):
            return "ERROR: failure " + ("z" * 100) if i % 2 == 0 else f"r{i}"

        ctx = _build_ctx(8, content=content)
        strategy = ManagedContextStrategy(keep_recent_turns=4, max_messages=2, max_chars=10)
        prepared = strategy.prepare(ctx)
        # system + task + at least the protected (recent/error) turns remain
        assert prepared[0].role == "system"
        assert prepared[1].role == "user"
        assert len(prepared) > 2  # protected content always kept despite the tiny budget


class TestManagedContextStrategyLongHistory:
    def test_long_history_is_reduced(self):
        ctx = _build_ctx(50, tool_name="read_file")
        strategy = ManagedContextStrategy()  # defaults
        prepared = strategy.prepare(ctx)
        report = strategy.last_report()
        assert report.items_before == 2 + 50 * 2
        assert report.turns_total == 50
        # with 50 identical-tool turns and defaults, compression and/or
        # budget-dropping must have reduced total character volume.
        assert report.chars_after < report.chars_before
        assert prepared[0].content == "You are FORGE."
        assert prepared[1].content == "the original task"

    def test_long_history_never_raises_and_stays_paired(self):
        ctx = _build_ctx(
            80, tool_name=lambda i: ["read_file", "write_file", "run_shell"][i % 3]
        )
        strategy = ManagedContextStrategy(keep_recent_turns=5, max_messages=20, max_chars=2000)
        prepared = strategy.prepare(ctx)
        tool_call_ids = {
            c.id for m in prepared if m.role == "assistant" for c in m.tool_calls
        }
        tool_result_ids = {m.tool_call_id for m in prepared if m.role == "tool"}
        assert tool_call_ids == tool_result_ids
