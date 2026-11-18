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


# ---------------------------------------------------------------------------
# Per-resource retention (Fix 3)
# ---------------------------------------------------------------------------


def _pad(text: str) -> str:
    """Make content long enough that compression actually shrinks it."""
    return text + " " + ("payload " * 20)


def _turns_ctx(turns) -> ConversationContext:
    """Build a context from explicit turns.

    *turns* is a list of turns; each turn is a list of
    (tool_name, arguments, result_content, call_id) tuples, one per call.
    """
    ctx = ConversationContext()
    ctx.add_message(Message(role="system", content="You are FORGE."))
    ctx.add_message(Message(role="user", content="the task"))
    for t, calls in enumerate(turns):
        ctx.add_message(
            Message(
                role="assistant",
                content=f"turn {t}: " + ("thinking " * 10),
                tool_calls=[
                    ToolCall(id=cid, name=name, arguments=args)
                    for name, args, _content, cid in calls
                ],
            )
        )
        for name, _args, content, cid in calls:
            ctx.add_message(
                Message(role="tool", content=content, name=name, tool_call_id=cid)
            )
    return ctx


def _read(path, content, cid=None, t=0):
    return ("read_file", {"path": path}, _pad(content), cid if cid is not None else f"r{t}")


def _shell(cmd, content, cid, exit_code=0):
    return ("run_shell", {"command": cmd}, _pad(content) + f"\n[exit code {exit_code}]", cid)


def _filler(n, start=0):
    """n distinct shell turns (distinct commands, so none supersede)."""
    return [
        [_shell(f"python step{start + k}.py", f"step {start + k} ok", f"f{start + k}")]
        for k in range(n)
    ]


def _tool_contents(prepared):
    return [m.content for m in prepared if m.role == "tool"]


class TestManagedContextPerResource:
    def test_different_files_both_kept_outside_recent_window(self):
        turns = [[_read("a.py", "A-CONTENT", t=0)], [_read("b.py", "B-CONTENT", t=1)]]
        turns += _filler(4)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert any(c.startswith("A-CONTENT") for c in contents)
        assert any(c.startswith("B-CONTENT") for c in contents)

    def test_same_file_twice_older_compressed_with_named_marker(self):
        turns = [
            [_read("a.py", "A-OLD", t=0)],
            [_read("b.py", "B", t=1)],
            [_read("a.py", "A-NEW", t=2)],
        ]
        turns += _filler(3)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        old_len = len(_pad("A-OLD"))
        assert contents[0] == (
            f"[MANAGED CONTEXT: older read of 'a.py' omitted ({old_len} chars) "
            "— superseded by a later read; call read_file again if needed]"
        )
        assert not any(c.startswith("A-OLD") for c in contents)
        assert contents[1].startswith("B ")
        assert contents[2].startswith("A-NEW")

    def test_shell_commands_keyed_by_command(self):
        turns = [
            [_shell("python -m pytest -q", "first test run", "s0")],
            [_shell("python lint.py", "lint run", "s1")],
            [_shell("python  -m   pytest -q", "second test run", "s2")],
        ]
        turns += _filler(3)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith(
            "[MANAGED CONTEXT: older output of 'python -m pytest -q' omitted ("
        )
        assert contents[0].endswith("chars) — a later run exists]")
        assert contents[1].startswith("lint run")
        assert contents[2].startswith("second test run")

    def test_duplicate_call_ids_across_turns_paired_within_turn(self):
        turns = [
            [_read("a.py", "A-OLD", cid="call_0")],
            [_read("b.py", "B", cid="call_0")],
            [_read("a.py", "A-NEW", cid="call_0")],
        ]
        turns += [
            [_shell(f"python s{k}.py", f"s{k}", "call_0")] for k in range(3)
        ]
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("[MANAGED CONTEXT: older read of 'a.py'")
        assert contents[1].startswith("B ")
        assert contents[2].startswith("A-NEW")

    def test_duplicate_ids_within_one_turn_fall_back_to_position(self):
        turns = [
            [_read("x.py", "X-OLD", cid="call_0"), _read("y.py", "Y", cid="call_0")],
            [_read("x.py", "X-NEW", cid="call_0")],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("[MANAGED CONTEXT: older read of 'x.py'")
        assert contents[1].startswith("Y ")
        assert contents[2].startswith("X-NEW")

    def test_missing_result_ids_paired_by_position(self):
        turns = [
            [_read("a.py", "A-OLD", cid=""), _read("b.py", "B", cid="")],
            [_read("a.py", "A-NEW", cid="")],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("[MANAGED CONTEXT: older read of 'a.py'")
        assert contents[1].startswith("B ")

    def test_path_spellings_share_one_key(self):
        turns = [
            [_read("src\\app.py", "APP-OLD", t=0)],
            [_read("./src/app.py", "APP-NEW", t=1)],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("[MANAGED CONTEXT: older read of 'src\\app.py'")
        assert contents[1].startswith("APP-NEW")

    def test_path_normalization(self):
        from forge.context import _normalize_path

        assert _normalize_path("src\\app.py") == _normalize_path("./src/app.py")
        assert _normalize_path("src/../src/app.py") == _normalize_path("src/app.py")
        assert _normalize_path(".") == "."

    def test_writes_never_supersede_reads(self):
        turns = [
            [_read("a.py", "A-CONTENT", t=0)],
            [("edit_file", {"path": "a.py", "old_string": "x", "new_string": "y"},
              _pad("Edited a.py"), "e1")],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("A-CONTENT")
        assert contents[1].startswith("Edited a.py")

    def test_repeated_writes_supersede_each_other(self):
        turns = [
            [("write_file", {"path": "a.py", "content": "1"}, _pad("Created a.py"), "w0")],
            [("edit_file", {"path": "a.py", "old_string": "1", "new_string": "2"},
              _pad("Edited a.py"), "w1")],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith(
            "[MANAGED CONTEXT: older write_file result for 'a.py'"
        )
        assert contents[1].startswith("Edited a.py")

    def test_list_directory_default_path_is_dot(self):
        turns = [
            [("list_directory", {}, _pad("OLD-LISTING"), "l0")],
            [("list_directory", {"path": "."}, _pad("NEW-LISTING"), "l1")],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("[MANAGED CONTEXT: older listing of '.'")
        assert contents[1].startswith("NEW-LISTING")

    def test_unknown_tools_never_collapsed(self):
        turns = [
            [("grep_tool", {"pattern": "x"}, _pad("GREP-1"), "g0")],
            [("grep_tool", {"pattern": "x"}, _pad("GREP-2"), "g1")],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("GREP-1")
        assert contents[1].startswith("GREP-2")

    def test_error_and_recent_turns_never_compressed(self):
        turns = [
            [_read("a.py", "A-OLD", t=0)],
            [("read_file", {"path": "a.py"}, "ERROR: file not found: a.py", "r1")],
            [_read("a.py", "A-MID", t=2)],
            [_read("a.py", "A-RECENT", t=3)],
        ]
        strategy = ManagedContextStrategy(keep_recent_turns=1)
        prepared = strategy.prepare(_turns_ctx(turns))
        contents = _tool_contents(prepared)
        assert contents[0].startswith("[MANAGED CONTEXT: older read of 'a.py'")
        assert contents[1] == "ERROR: file not found: a.py"  # error turn in full
        assert contents[2].startswith("A-MID")  # latest non-protected read
        assert contents[3].startswith("A-RECENT")  # recent window
        assistants = [m.content for m in prepared if m.role == "assistant"]
        assert assistants[1].startswith("turn 1:")  # error turn text kept
        assert assistants[3].startswith("turn 3:")  # recent turn text kept
        assert strategy.last_report().turns_kept_error == 1

    def test_budget_still_drops_oldest_unprotected_turns(self):
        turns = [[_read(f"f{t}.py", f"F{t}", t=t)] for t in range(10)]
        strategy = ManagedContextStrategy(
            keep_recent_turns=2, max_messages=10, max_chars=1_000_000
        )
        prepared = strategy.prepare(_turns_ctx(turns))
        report = strategy.last_report()
        assert len(prepared) <= 10
        assert report.turns_dropped_for_budget == 6
        assert [c.split(" ")[0] for c in _tool_contents(prepared)] == [
            "F6", "F7", "F8", "F9"
        ]

    def test_tool_call_never_separated_from_result(self):
        turns = []
        for t in range(12):
            turns.append([
                _read(f"m{t % 3}.py", f"M{t}", cid="call_0"),
                _shell("python -m pytest -q", f"run {t}", "call_1"),
            ])
        strategy = ManagedContextStrategy(keep_recent_turns=3, max_messages=15, max_chars=3000)
        prepared = strategy.prepare(_turns_ctx(turns))
        for idx, m in enumerate(prepared):
            if m.role == "assistant":
                results = prepared[idx + 1: idx + 1 + len(m.tool_calls)]
                assert [r.role for r in results] == ["tool"] * len(m.tool_calls)
                assert [r.name for r in results] == [c.name for c in m.tool_calls]
        assert sum(1 for m in prepared if m.role == "tool") == 2 * sum(
            1 for m in prepared if m.role == "assistant"
        )

    def test_per_resource_deterministic(self):
        turns = [[_read(f"m{t % 3}.py", f"M{t}", t=t)] for t in range(9)]
        out_a = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        out_b = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        assert [(m.role, m.content) for m in out_a] == [(m.role, m.content) for m in out_b]

    def test_compression_never_grows_a_message(self):
        turns = [
            [("read_file", {"path": "a.py"}, "tiny", "r0")],
            [("read_file", {"path": "a.py"}, "tiny2", "r1")],
        ]
        turns += _filler(2)
        prepared = ManagedContextStrategy(keep_recent_turns=2).prepare(_turns_ctx(turns))
        assert _tool_contents(prepared)[0] == "tiny"
