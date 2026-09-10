"""
Tests for forge.observability — RunTracer event recording and JSONL flush.
"""

from __future__ import annotations

import json

import pytest

from forge.agent import AgentConfig
from forge.observability import (
    LLM_CALL,
    RUN_END,
    RUN_START,
    TOOL_CALL,
    TOOL_RESULT,
    RunTracer,
)


class TestRunTracer:
    @pytest.fixture
    def tracer(self):
        return RunTracer(run_id="test_run_001", task="Fix bug in foo.py")

    @pytest.fixture
    def config(self):
        return AgentConfig(task="Fix bug in foo.py")

    def test_initial_state(self, tracer):
        assert tracer.run_id == "test_run_001"
        assert tracer._llm_calls == 0
        assert tracer._tool_calls == 0

    def test_record_run_start(self, tracer, config):
        tracer.record_run_start(config)
        assert len(tracer._events) == 1
        assert tracer._events[0].event_type == RUN_START

    def test_record_llm_call_increments_counter(self, tracer):
        tracer.record_llm_call(input_tokens=100, output_tokens=50, latency_ms=200.0)
        assert tracer._llm_calls == 1
        assert tracer._input_tokens == 100
        assert tracer._output_tokens == 50

    def test_record_multiple_llm_calls_accumulates(self, tracer):
        tracer.record_llm_call(input_tokens=100, output_tokens=50)
        tracer.record_llm_call(input_tokens=200, output_tokens=80)
        assert tracer._llm_calls == 2
        assert tracer._input_tokens == 300
        assert tracer._output_tokens == 130

    def test_record_tool_call(self, tracer):
        tracer.record_tool_call("read_file", args={"path": "foo.py"})
        assert tracer._tool_calls == 1
        assert "read_file" in tracer._tool_names_used
        event = tracer._events[0]
        assert event.event_type == TOOL_CALL
        assert event.data["tool_name"] == "read_file"

    def test_record_tool_result(self, tracer):
        tracer.record_tool_result("read_file", success=True)
        event = tracer._events[0]
        assert event.event_type == TOOL_RESULT
        assert event.data["success"] is True

    def test_record_run_end(self, tracer, config):
        tracer.record_run_start(config)
        tracer.record_run_end(status="completed", final_answer="Done.")
        end_event = tracer._events[-1]
        assert end_event.event_type == RUN_END
        assert end_event.data["status"] == "completed"

    def test_no_token_data_returns_none_in_summary(self, tracer):
        summary = tracer.summary(status="completed")
        assert summary.total_input_tokens is None
        assert summary.total_output_tokens is None

    def test_token_data_present_in_summary(self, tracer):
        tracer.record_llm_call(input_tokens=500, output_tokens=200)
        summary = tracer.summary(status="completed")
        assert summary.total_input_tokens == 500
        assert summary.total_output_tokens == 200

    def test_flush_creates_jsonl_file(self, tracer, config, tmp_path):
        tracer.record_run_start(config)
        tracer.record_llm_call(input_tokens=100, output_tokens=40)
        tracer.record_run_end(status="completed")
        path = tracer.flush(log_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".jsonl"

    def test_flush_valid_jsonl(self, tracer, config, tmp_path):
        tracer.record_run_start(config)
        tracer.record_llm_call(input_tokens=100, output_tokens=40)
        tracer.record_run_end(status="completed")
        path = tracer.flush(log_dir=tmp_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # run_start, llm_call, run_end
        for line in lines:
            obj = json.loads(line)
            assert "event_type" in obj
            assert "ts" in obj

    def test_flush_creates_log_dir_if_missing(self, tracer, tmp_path):
        nested = tmp_path / "deep" / "nested"
        tracer.record_run_end(status="completed")
        tracer.flush(log_dir=nested)
        assert nested.exists()
