"""
Tests for forge.observability — RunTracer event recording, token accounting,
and JSONL flush.
"""

from __future__ import annotations

import json

import pytest

from forge.observability import (
    ERROR,
    LLM_CALL,
    LLM_RESPONSE,
    RUN_END,
    RUN_START,
    TOOL_CALL,
    TOOL_RESULT,
    RunTracer,
    _add,
)


class TestAddHelper:
    """_add preserves 'unknown' (None) semantics for optional counters."""

    def test_none_plus_none_is_none(self):
        assert _add(None, None) is None

    def test_none_plus_value_is_value(self):
        assert _add(None, 5) == 5

    def test_value_plus_none_is_value(self):
        assert _add(5, None) == 5

    def test_value_plus_value_sums(self):
        assert _add(5, 3) == 8


class TestRunTracer:
    @pytest.fixture
    def tracer(self):
        return RunTracer(run_id="test_run_001", task="Fix bug in foo.py")

    def test_initial_state(self, tracer):
        assert tracer.run_id == "test_run_001"
        assert tracer._llm_calls == 0
        assert tracer._tool_calls == 0
        assert tracer._input_tokens is None
        assert tracer._output_tokens is None

    def test_record_run_start_captures_experiment_metadata(self, tracer):
        tracer.record_run_start(
            provider="gemini",
            model="gemini-2.0-flash",
            temperature=0.0,
            max_output_tokens=4096,
            system_prompt_id="sha256:abc123",
            tool_exposure_strategy="fixed",
            tool_subset=None,
            context_strategy="raw",
        )
        assert len(tracer._events) == 1
        event = tracer._events[0]
        assert event.event_type == RUN_START
        assert event.data["provider"] == "gemini"
        assert event.data["model"] == "gemini-2.0-flash"
        assert event.data["system_prompt_id"] == "sha256:abc123"
        assert event.data["tool_exposure_strategy"] == "fixed"
        assert event.data["context_strategy"] == "raw"

    def test_record_llm_request_emits_llm_call_event(self, tracer):
        tracer.record_llm_request(turn=1, message_count=2, exposed_tools=["read_file"])
        event = tracer._events[0]
        assert event.event_type == LLM_CALL
        assert event.data["turn"] == 1
        assert event.data["message_count"] == 2
        assert event.data["exposed_tool_count"] == 1

    def test_record_llm_response_increments_and_accumulates(self, tracer):
        tracer.record_llm_response(input_tokens=100, output_tokens=50, latency_ms=200.0)
        assert tracer._llm_calls == 1
        assert tracer._input_tokens == 100
        assert tracer._output_tokens == 50
        assert tracer._events[0].event_type == LLM_RESPONSE

    def test_multiple_llm_responses_accumulate(self, tracer):
        tracer.record_llm_response(input_tokens=100, output_tokens=50)
        tracer.record_llm_response(input_tokens=200, output_tokens=80)
        assert tracer._llm_calls == 2
        assert tracer._input_tokens == 300
        assert tracer._output_tokens == 130

    def test_input_and_output_availability_are_independent(self, tracer):
        # Provider reports input tokens but never output tokens.
        tracer.record_llm_response(input_tokens=100, output_tokens=None)
        tracer.record_llm_response(input_tokens=120, output_tokens=None)
        assert tracer._input_tokens == 220
        assert tracer._output_tokens is None  # not fabricated as 0

    def test_no_token_data_stays_none(self, tracer):
        tracer.record_llm_response(stop_reason="stop")
        assert tracer._input_tokens is None
        assert tracer._output_tokens is None
        assert tracer.resolved_total_tokens is None

    def test_resolved_total_prefers_provider_total(self, tracer):
        tracer.record_llm_response(input_tokens=10, output_tokens=5, total_tokens=99)
        assert tracer.resolved_total_tokens == 99

    def test_resolved_total_falls_back_to_sum(self, tracer):
        tracer.record_llm_response(input_tokens=10, output_tokens=5)
        assert tracer.resolved_total_tokens == 15

    def test_cached_and_reasoning_tokens_recorded(self, tracer):
        tracer.record_llm_response(
            input_tokens=100, output_tokens=20,
            cached_input_tokens=40, reasoning_tokens=12,
        )
        assert tracer._cached_input_tokens == 40
        assert tracer._reasoning_tokens == 12

    def test_record_tool_call(self, tracer):
        tracer.record_tool_call("read_file", args={"path": "foo.py"})
        assert tracer._tool_calls == 1
        assert "read_file" in tracer._tool_names_used
        event = tracer._events[0]
        assert event.event_type == TOOL_CALL
        assert event.data["tool_name"] == "read_file"
        assert event.data["arg_keys"] == ["path"]

    def test_record_tool_result(self, tracer):
        tracer.record_tool_result("read_file", success=True, metadata={"bytes": 12})
        event = tracer._events[0]
        assert event.event_type == TOOL_RESULT
        assert event.data["success"] is True
        assert event.data["metadata"] == {"bytes": 12}

    def test_record_error(self, tracer):
        tracer.record_error("provider blew up")
        assert tracer._events[0].event_type == ERROR

    def test_record_run_end(self, tracer):
        tracer.record_run_start(provider="gemini", model="m")
        tracer.record_llm_response(input_tokens=10, output_tokens=4)
        tracer.record_run_end(status="completed", final_answer="Done.")
        end = tracer._events[-1]
        assert end.event_type == RUN_END
        assert end.data["status"] == "completed"
        assert end.data["input_tokens"] == 10
        assert end.data["total_tokens"] == 14
        assert end.data["elapsed_ms"] is not None

    def test_summary_reflects_none_when_no_tokens(self, tracer):
        tracer.record_run_start(provider="gemini", model="m")
        summary = tracer.summary(status="completed")
        assert summary.input_tokens is None
        assert summary.output_tokens is None
        assert summary.total_tokens is None
        assert summary.provider == "gemini"

    def test_summary_reflects_tokens_when_present(self, tracer):
        tracer.record_llm_response(input_tokens=500, output_tokens=200)
        summary = tracer.summary(status="completed")
        assert summary.input_tokens == 500
        assert summary.output_tokens == 200
        assert summary.total_tokens == 700

    def test_flush_creates_valid_jsonl(self, tracer, tmp_path):
        tracer.record_run_start(provider="gemini", model="m")
        tracer.record_llm_request(turn=1, message_count=2, exposed_tools=[])
        tracer.record_llm_response(input_tokens=100, output_tokens=40)
        tracer.record_run_end(status="completed")
        path = tracer.flush(log_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4
        for line in lines:
            obj = json.loads(line)
            assert "event_type" in obj
            assert "ts" in obj

    def test_flush_creates_log_dir_if_missing(self, tracer, tmp_path):
        nested = tmp_path / "deep" / "nested"
        tracer.record_run_end(status="completed")
        tracer.flush(log_dir=nested)
        assert nested.exists()
