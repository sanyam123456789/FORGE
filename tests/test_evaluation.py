"""
Tests for forge.evaluation — the Step 3 measurement layer.

No real Gemini call: a scripted fake ``LLMProvider`` drives the *real*
``AgentRuntime`` with the real built-in tools in a tmp workspace.  These
tests exercise task representation, experiment configuration, the runner,
result serialisation, and None-safe aggregation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from forge.evaluation import (
    AggregateStats,
    ArtifactCheck,
    EvalResult,
    EvalTask,
    EvaluationRunner,
    ExperimentConfig,
    aggregate_results,
    group_results,
    read_results_jsonl,
    write_results_jsonl,
)
from forge.evaluation.result import TokenPricing, append_result_jsonl
from forge.llm import LLMProvider, LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedProvider(LLMProvider):
    """One pre-scripted LLMResponse per complete() call."""

    def __init__(self, responses, *, model="fake-model", name="scripted"):
        self._responses = list(responses)
        self._model = model
        self._name = name
        self.calls = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, messages, *, tools=None, max_tokens=4096, temperature=0.0):
        self.calls.append({"messages": list(messages), "tools": tools})
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _text(content, **usage):
    return LLMResponse(content=content, stop_reason="stop", **usage)


def _write_calc_then_done(**usage):
    """Script: write calculator.py, then finish."""
    return ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={
                            "path": "calculator.py",
                            "content": "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            _text("Created calculator.py.", **usage),
        ]
    )


_SETTINGS = SimpleNamespace(
    llm_provider="gemini",
    llm_model="gemini-3.6-flash",
    llm_temperature=0.0,
    llm_max_output_tokens=4096,
    max_llm_turns=30,
    max_tool_calls=50,
)


def _calc_task():
    return EvalTask(
        task_id="calc-add-multiply",
        prompt="Create calculator.py with add(a, b) and multiply(a, b).",
        checks=(
            ArtifactCheck(
                path="calculator.py",
                must_contain=("def add(a, b)", "def multiply(a, b)"),
            ),
        ),
    )


def _experiment(**overrides):
    base = dict(task_id="calc-add-multiply", tool_strategy="fixed", context_strategy="raw")
    base.update(overrides)
    return ExperimentConfig.from_settings(_SETTINGS, **base)


# ---------------------------------------------------------------------------
# A. Task representation
# ---------------------------------------------------------------------------


class TestEvalTask:
    def test_identity_and_validation(self):
        t = EvalTask(task_id="t1", prompt="do something")
        assert t.task_id == "t1"
        with pytest.raises(ValueError):
            EvalTask(task_id="", prompt="x")
        with pytest.raises(ValueError):
            EvalTask(task_id="t", prompt="   ")

    def test_to_from_dict_roundtrip(self):
        t = _calc_task()
        again = EvalTask.from_dict(json.loads(json.dumps(t.to_dict())))
        assert again == t
        assert again.checks[0].must_contain == ("def add(a, b)", "def multiply(a, b)")

    def test_run_checks_without_checks_is_none(self, tmp_path):
        outcome = EvalTask(task_id="t", prompt="p").run_checks(tmp_path)
        assert outcome.passed is None
        assert outcome.had_checks is False

    def test_run_checks_pass_and_fail(self, tmp_path):
        (tmp_path / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a*b\n",
            encoding="utf-8",
        )
        good = _calc_task().run_checks(tmp_path)
        assert good.passed is True
        assert good.checks_run == 1

        missing = EvalTask(
            task_id="t",
            prompt="p",
            checks=(ArtifactCheck(path="nope.py"),),
        ).run_checks(tmp_path)
        assert missing.passed is False

    def test_artifact_check_must_be_absent(self, tmp_path):
        ok, _ = ArtifactCheck(path="ghost.py", must_exist=False).evaluate(tmp_path)
        assert ok is True
        (tmp_path / "ghost.py").write_text("x", encoding="utf-8")
        bad, _ = ArtifactCheck(path="ghost.py", must_exist=False).evaluate(tmp_path)
        assert bad is False


# ---------------------------------------------------------------------------
# B. Experiment configuration
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_defaults_resolve_to_fixed_raw(self):
        exp = _experiment()
        assert exp.tool_strategy == "fixed"
        assert exp.context_strategy == "raw"
        assert exp.is_implemented is True
        assert exp.strategy_key == "fixed+raw"
        tool_strat, ctx_strat = exp.resolve_strategies()
        assert tool_strat.name == "fixed"
        assert ctx_strat.name == "raw"

    def test_from_settings_pulls_provider_model_and_params(self):
        exp = _experiment()
        assert exp.provider == "gemini"
        assert exp.model == "gemini-3.6-flash"
        assert exp.temperature == 0.0
        assert exp.max_output_tokens == 4096
        assert exp.max_llm_turns == 30
        assert exp.max_tool_calls == 50

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError):
            _experiment(tool_strategy="banana")
        with pytest.raises(ValueError):
            _experiment(context_strategy="banana")

    def test_adaptive_tool_strategy_is_implemented(self):
        # Step 5: adaptive tool exposure is now a real, runnable strategy.
        adaptive = _experiment(tool_strategy="adaptive")
        assert adaptive.is_implemented is True
        assert adaptive.strategy_key == "adaptive+raw"
        tool_strat, ctx_strat = adaptive.resolve_strategies()
        assert tool_strat.name == "adaptive"
        assert ctx_strat.name == "raw"

    def test_managed_context_strategy_is_implemented(self):
        # Step 6: managed context is now a real, runnable strategy.
        managed = _experiment(context_strategy="managed")
        assert managed.is_implemented is True
        assert managed.strategy_key == "fixed+managed"
        tool_strat, ctx_strat = managed.resolve_strategies()
        assert tool_strat.name == "fixed"
        assert ctx_strat.name == "managed"

    def test_adaptive_and_managed_combine(self):
        # All four cells of the eventual 2x2 are individually resolvable.
        both = _experiment(tool_strategy="adaptive", context_strategy="managed")
        assert both.is_implemented is True
        assert both.strategy_key == "adaptive+managed"
        tool_strat, ctx_strat = both.resolve_strategies()
        assert tool_strat.name == "adaptive"
        assert ctx_strat.name == "managed"

    def test_to_from_dict_roundtrip(self):
        exp = _experiment(label="pilot")
        again = ExperimentConfig.from_dict(json.loads(json.dumps(exp.to_dict())))
        assert again == exp


# ---------------------------------------------------------------------------
# K. Same task, different strategy configs, unchanged identity
# ---------------------------------------------------------------------------


def test_same_task_under_different_strategies_keeps_identity():
    task = _calc_task()
    a = ExperimentConfig.from_settings(_SETTINGS, task_id=task.task_id)
    b = ExperimentConfig.from_settings(
        _SETTINGS, task_id=task.task_id,
        tool_strategy="adaptive", context_strategy="managed",
    )
    assert a.task_id == b.task_id == task.task_id
    assert a.strategy_key != b.strategy_key
    # the task object itself is untouched / reusable
    assert task.to_dict() == _calc_task().to_dict()


# ---------------------------------------------------------------------------
# C + H. Runner produces a populated, strategy-tagged result via the REAL loop
# ---------------------------------------------------------------------------


class TestEvaluationRunner:
    def test_run_produces_result_and_uses_real_runtime(self, tmp_path):
        provider = _write_calc_then_done(
            input_tokens=120, output_tokens=30, total_tokens=150
        )
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        ws = tmp_path / "ws"
        result = runner.run(_calc_task(), _experiment(), workspace=ws)

        assert isinstance(result, EvalResult)
        assert result.status == "completed"
        assert result.success is True
        assert result.tool_strategy == "fixed"
        assert result.context_strategy == "raw"
        assert result.provider == "scripted"
        assert result.model == "fake-model"
        assert result.llm_calls == 2
        assert result.tool_calls == 1
        assert result.input_tokens == 120
        assert result.output_tokens == 30
        assert result.total_tokens == 150
        assert result.checks_run == 1
        assert result.checks_passed is True

        # real runtime side effects: the tool actually wrote the file...
        assert (ws / "calculator.py").read_text(encoding="utf-8").startswith("def add")
        # ...and a real JSONL trace was written by the real observability layer.
        assert result.trace_path is not None
        events = [
            json.loads(line)
            for line in open(result.trace_path, encoding="utf-8").read().splitlines()
        ]
        kinds = [e["event_type"] for e in events]
        assert kinds[0] == "run_start" and kinds[-1] == "run_end"

    def test_run_records_reproducibility_metadata(self, tmp_path):
        provider = _write_calc_then_done()
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        result = runner.run(
            _calc_task(), _experiment(label="batch-1"), workspace=tmp_path / "ws"
        )
        assert result.temperature == 0.0
        assert result.max_output_tokens == 4096
        assert result.max_llm_turns == 30
        assert result.max_tool_calls == 50
        assert result.model == "fake-model"
        assert result.label == "batch-1"
        assert result.workspace == str(tmp_path / "ws")
        assert result.started_at is not None and result.finished_at is not None
        assert result.duration_s is not None and result.duration_s >= 0

    def test_temp_workspace_created_and_cleaned_by_default(self, tmp_path):
        provider = _write_calc_then_done()
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces",
            write_trace=False,
        )
        result = runner.run(_calc_task(), _experiment())  # no workspace -> temp
        from pathlib import Path

        assert result.workspace is not None
        assert not Path(result.workspace).exists()  # auto-removed

    def test_unknown_context_strategy_rejected_before_running(self, tmp_path):
        # An unrecognised strategy name is rejected at ExperimentConfig
        # construction (ValueError), long before the runner would run
        # anything — see TestExperimentConfig.test_unknown_strategy_rejected.
        with pytest.raises(ValueError):
            _experiment(context_strategy="banana")

    def test_run_with_managed_context_strategy(self, tmp_path):
        # Step 6: Fixed + Managed runs the real agent loop end-to-end.
        provider = _write_calc_then_done(
            input_tokens=120, output_tokens=30, total_tokens=150
        )
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        result = runner.run(
            _calc_task(), _experiment(context_strategy="managed"), workspace=tmp_path / "ws"
        )
        assert result.status == "completed"
        assert result.success is True
        assert result.tool_strategy == "fixed"
        assert result.context_strategy == "managed"
        # a short, two-turn run never triggers compression/dropping, but the
        # strategy still reports (None-safe: 0, not fabricated).
        assert result.context_items_dropped == 0
        assert result.context_items_compressed == 0
        assert result.context_chars_saved == 0

    def test_run_with_adaptive_and_managed_strategies(self, tmp_path):
        # Step 6: Adaptive + Managed — the fourth cell of the eventual 2x2 —
        # runs the real agent loop end-to-end.
        task = EvalTask(
            task_id="calc-create",
            prompt=(
                "Create a new file named calculator.py with add(a, b) and "
                "multiply(a, b)."
            ),
            checks=(
                ArtifactCheck(
                    path="calculator.py",
                    must_contain=("def add(a, b)", "def multiply(a, b)"),
                ),
            ),
        )
        provider = _write_calc_then_done()
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        result = runner.run(
            task,
            _experiment(
                task_id=task.task_id, tool_strategy="adaptive", context_strategy="managed"
            ),
            workspace=tmp_path / "ws",
        )
        assert result.status == "completed"
        assert result.success is True
        assert result.tool_strategy == "adaptive"
        assert result.context_strategy == "managed"

    def test_run_with_adaptive_tool_strategy_narrows_exposed_tools(self, tmp_path):
        # A prompt that matches the "create" keyword rule (see
        # AdaptiveToolExposure.RULES) should narrow exposure away from
        # edit_file / run_shell while still letting the real agent loop
        # complete the task end-to-end.
        task = EvalTask(
            task_id="calc-create",
            prompt=(
                "Create a new file named calculator.py with add(a, b) and "
                "multiply(a, b)."
            ),
            checks=(
                ArtifactCheck(
                    path="calculator.py",
                    must_contain=("def add(a, b)", "def multiply(a, b)"),
                ),
            ),
        )
        provider = _write_calc_then_done()
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        result = runner.run(
            task,
            _experiment(task_id=task.task_id, tool_strategy="adaptive"),
            workspace=tmp_path / "ws",
        )

        assert result.tool_strategy == "adaptive"
        assert result.status == "completed"
        assert result.success is True

        events = [
            json.loads(line)
            for line in open(result.trace_path, encoding="utf-8").read().splitlines()
        ]
        start = events[0]["data"]
        assert start["tool_exposure_strategy"] == "adaptive"
        assert set(start["tool_subset"]) == {"read_file", "write_file", "list_directory"}

    # -- D + G + J. None-safe metrics / failed runs ---------------------

    def test_missing_usage_stays_none_not_zero(self, tmp_path):
        provider = _write_calc_then_done()  # no usage kwargs
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        result = runner.run(_calc_task(), _experiment(), workspace=tmp_path / "ws")
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None
        assert result.reasoning_tokens is None
        assert result.cached_input_tokens is None
        assert result.cost_usd is None
        assert result.cost_available is False

    def test_provider_error_run_is_unsuccessful(self, tmp_path):
        from forge.llm import LLMError

        provider = ScriptedProvider([LLMError("boom")])
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        result = runner.run(_calc_task(), _experiment(), workspace=tmp_path / "ws")
        assert result.status == "provider_error"
        assert result.success is False
        assert result.error and "boom" in result.error

    # -- I. cost when pricing supplied --------------------------------

    def test_cost_computed_only_with_pricing(self, tmp_path):
        provider = _write_calc_then_done(
            input_tokens=1_000_000, output_tokens=500_000, total_tokens=1_500_000
        )
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        pricing = TokenPricing(
            input_per_mtok=2.0, output_per_mtok=6.0, source="test pricing"
        )
        result = runner.run(
            _calc_task(), _experiment(), workspace=tmp_path / "ws", pricing=pricing
        )
        # 1.0 * $2  +  0.5 * $6  = $5.00
        assert result.cost_available is True
        assert result.cost_usd == pytest.approx(5.0)
        assert result.cost_source == "test pricing"


# ---------------------------------------------------------------------------
# E. Result serialisation
# ---------------------------------------------------------------------------


def _mk_result(**overrides) -> EvalResult:
    base = dict(
        task_id="t", run_id="run_x", provider="gemini", model="gemini-3.6-flash",
        tool_strategy="fixed", context_strategy="raw", status="completed", success=True,
        input_tokens=100, output_tokens=20, total_tokens=120, latency_ms=1234.5,
        llm_calls=2, tool_calls=1,
    )
    base.update(overrides)
    return EvalResult(**base)


class TestResultSerialisation:
    def test_jsonl_roundtrip(self, tmp_path):
        results = [_mk_result(run_id="r1"), _mk_result(run_id="r2", success=False, status="failed")]
        path = write_results_jsonl(results, tmp_path / "res.jsonl")
        back = read_results_jsonl(path)
        assert [r.run_id for r in back] == ["r1", "r2"]
        assert back[0].to_dict() == results[0].to_dict()

    def test_append_accumulates(self, tmp_path):
        path = tmp_path / "res.jsonl"
        append_result_jsonl(_mk_result(run_id="r1"), path)
        append_result_jsonl(_mk_result(run_id="r2"), path)
        assert len(read_results_jsonl(path)) == 2

    def test_blank_lines_skipped_and_malformed_raises(self, tmp_path):
        path = tmp_path / "res.jsonl"
        path.write_text(
            _mk_result(run_id="r1").to_json() + "\n\n" + "{not json}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="malformed EvalResult"):
            read_results_jsonl(path)

    def test_from_dict_ignores_unknown_keys(self):
        d = _mk_result().to_dict()
        d["some_future_field"] = 123
        r = EvalResult.from_dict(d)
        assert r.run_id == "run_x"


# ---------------------------------------------------------------------------
# F + G + H + I + J. Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_empty_input_is_none_safe(self):
        agg = aggregate_results([])
        assert agg.task_count == 0
        assert agg.successful_tasks == 0
        assert agg.success_rate is None
        assert agg.total_tokens is None
        assert agg.avg_latency_ms is None
        assert agg.cost_per_successful_task is None

    def test_success_rate_and_totals(self):
        results = [
            _mk_result(run_id="r1", success=True, input_tokens=100, output_tokens=10,
                       total_tokens=110, tool_calls=2, llm_calls=3, latency_ms=1000.0),
            _mk_result(run_id="r2", success=False, status="provider_error",
                       input_tokens=50, output_tokens=5, total_tokens=55,
                       tool_calls=1, llm_calls=1, latency_ms=500.0),
        ]
        agg = aggregate_results(results)
        assert agg.task_count == 2
        assert agg.successful_tasks == 1
        assert agg.success_rate == pytest.approx(0.5)
        assert agg.total_input_tokens == 150
        assert agg.total_tokens == 165
        assert agg.avg_total_tokens == pytest.approx(82.5)
        assert agg.total_tool_calls == 3
        assert agg.avg_tool_calls == pytest.approx(1.5)
        assert agg.total_llm_calls == 4
        assert agg.avg_latency_ms == pytest.approx(750.0)

    def test_missing_metrics_not_treated_as_zero(self):
        results = [
            _mk_result(run_id="r1", input_tokens=None, output_tokens=None,
                       total_tokens=None, latency_ms=None, tool_calls=None),
            _mk_result(run_id="r2", input_tokens=None, output_tokens=None,
                       total_tokens=None, latency_ms=None, tool_calls=None),
        ]
        agg = aggregate_results(results)
        assert agg.total_input_tokens is None
        assert agg.total_tokens is None
        assert agg.avg_total_tokens is None
        assert agg.avg_latency_ms is None
        assert agg.total_tool_calls is None

    def test_partial_metrics_use_only_reported_values(self):
        results = [
            _mk_result(run_id="r1", input_tokens=100, latency_ms=200.0),
            _mk_result(run_id="r2", input_tokens=None, latency_ms=None),
        ]
        agg = aggregate_results(results)
        assert agg.total_input_tokens == 100
        assert agg.avg_input_tokens == pytest.approx(100.0)  # mean over the 1 reported
        assert agg.avg_latency_ms == pytest.approx(200.0)

    def test_cost_per_successful_task(self):
        results = [
            _mk_result(run_id="r1", success=True, cost_available=True, cost_usd=4.0),
            _mk_result(run_id="r2", success=True, cost_available=True, cost_usd=6.0),
            _mk_result(run_id="r3", success=False, status="failed",
                       cost_available=True, cost_usd=1.0),
        ]
        agg = aggregate_results(results)
        assert agg.cost_available is True
        assert agg.total_cost_usd == pytest.approx(11.0)
        assert agg.cost_per_successful_task == pytest.approx(5.5)  # 11 / 2 successful

    def test_cost_none_when_no_pricing(self):
        agg = aggregate_results([_mk_result(run_id="r1", success=True)])
        assert agg.cost_available is False
        assert agg.total_cost_usd is None
        assert agg.cost_per_successful_task is None

    def test_cost_per_successful_task_none_when_zero_successful(self):
        agg = aggregate_results(
            [_mk_result(run_id="r1", success=False, status="failed",
                        cost_available=True, cost_usd=3.0)]
        )
        assert agg.total_cost_usd == pytest.approx(3.0)
        assert agg.cost_per_successful_task is None

    def test_strategy_metadata_preserved_and_grouped(self):
        results = [
            _mk_result(run_id="r1", tool_strategy="fixed", context_strategy="raw"),
            _mk_result(run_id="r2", tool_strategy="fixed", context_strategy="raw"),
        ]
        agg = aggregate_results(results)
        assert agg.tool_strategy == "fixed"
        assert agg.context_strategy == "raw"
        assert agg.provider == "gemini"

        grouped = group_results(results)
        assert set(grouped) == {"fixed+raw"}
        assert isinstance(grouped["fixed+raw"], AggregateStats)
        assert grouped["fixed+raw"].task_count == 2

    def test_group_results_separates_future_conditions(self):
        # Mixed strategy labels still bucket cleanly (supports the later 2x2).
        results = [
            _mk_result(run_id="r1", tool_strategy="fixed", context_strategy="raw"),
            _mk_result(run_id="r2", tool_strategy="adaptive", context_strategy="managed"),
        ]
        grouped = group_results(results)
        assert set(grouped) == {"fixed+raw", "adaptive+managed"}
        assert grouped["fixed+raw"].tool_strategy == "fixed"
        assert grouped["adaptive+managed"].context_strategy == "managed"


# ---------------------------------------------------------------------------
# TokenPricing unit behaviour (J — insufficient inputs -> None)
# ---------------------------------------------------------------------------


class TestTokenPricing:
    def test_cost_none_when_both_token_counts_missing(self):
        p = TokenPricing(input_per_mtok=1.0, output_per_mtok=2.0)
        assert p.cost_for(input_tokens=None, output_tokens=None) is None

    def test_cost_uses_available_counts(self):
        p = TokenPricing(input_per_mtok=3.0, output_per_mtok=9.0)
        assert p.cost_for(input_tokens=1_000_000, output_tokens=None) == pytest.approx(3.0)

    def test_cached_input_discounted_when_priced(self):
        p = TokenPricing(
            input_per_mtok=10.0, output_per_mtok=0.0, cached_input_per_mtok=1.0
        )
        # 1M input, 400k cached: 600k @ $10/M + 400k @ $1/M = 6.0 + 0.4
        cost = p.cost_for(
            input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000
        )
        assert cost == pytest.approx(6.4)
