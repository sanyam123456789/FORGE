"""
Tests for forge.context — ConversationContext container and context strategies.
"""

from __future__ import annotations

import pytest

from forge.context import ContextStrategy, ConversationContext, RawContextStrategy
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
