"""
Tests for forge.evaluation.matrix — the Step 7 formal 2x2 controlled experiment.

No real Gemini call: every task-arm execution gets its own scripted fake
``LLMProvider`` (via a ``provider_factory``) driving the *real*
``AgentRuntime``/``EvaluationRunner`` with the real built-in tools in a fresh
tmp workspace per (task, arm) pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.evaluation import (
    ARM_ADAPTIVE_MANAGED,
    ARM_ADAPTIVE_RAW,
    ARM_FIXED_MANAGED,
    ARM_FIXED_RAW,
    ARMS,
    STATUS_RUNNER_ERROR,
    ArtifactCheck,
    EvalTask,
    ExperimentArm,
    MatrixRunner,
    arms_by_ids,
    outcome_category,
    read_matrix_metadata,
    read_results_jsonl,
)
from forge.llm import LLMError, LLMProvider, LLMResponse, ToolCall


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
        if not self._responses:
            raise AssertionError("ScriptedProvider ran out of responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


_SETTINGS = SimpleNamespace(
    llm_provider="gemini",
    llm_model="gemini-3.6-flash",
    llm_temperature=0.0,
    llm_max_output_tokens=4096,
    max_llm_turns=30,
    max_tool_calls=50,
)


def _calc_task(task_id="calc-add", category="file_creation"):
    return EvalTask(
        task_id=task_id,
        prompt="Create calculator.py with add(a, b).",
        checks=(ArtifactCheck(path="calculator.py", must_contain=("def add(a, b)",)),),
        metadata={"category": category},
    )


def _calc_provider() -> ScriptedProvider:
    """A fresh two-turn write_file-then-done script."""
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
                            "content": "def add(a, b):\n    return a + b\n",
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="Created calculator.py.",
                stop_reason="stop",
                input_tokens=80,
                output_tokens=20,
                total_tokens=100,
            ),
        ]
    )


def _runner(tmp_path, provider_factory=_calc_provider, **kw) -> MatrixRunner:
    return MatrixRunner(
        settings=_SETTINGS,
        provider_factory=provider_factory,
        output_root=tmp_path / "experiments",
        write_trace=True,
        **kw,
    )


# ---------------------------------------------------------------------------
# 1-2. Exact four-arm matrix, stable ids
# ---------------------------------------------------------------------------


class TestArms:
    def test_exactly_four_arms(self):
        assert len(ARMS) == 4

    def test_arm_ids_are_stable_strings(self):
        assert [a.arm_id for a in ARMS] == [
            "fixed_raw", "fixed_managed", "adaptive_raw", "adaptive_managed",
        ]
        assert ARM_FIXED_RAW == "fixed_raw"
        assert ARM_FIXED_MANAGED == "fixed_managed"
        assert ARM_ADAPTIVE_RAW == "adaptive_raw"
        assert ARM_ADAPTIVE_MANAGED == "adaptive_managed"

    def test_each_arm_pins_the_right_strategy_pair(self):
        by_id = {a.arm_id: a for a in ARMS}
        assert (by_id["fixed_raw"].tool_strategy, by_id["fixed_raw"].context_strategy) == ("fixed", "raw")
        assert (by_id["fixed_managed"].tool_strategy, by_id["fixed_managed"].context_strategy) == ("fixed", "managed")
        assert (by_id["adaptive_raw"].tool_strategy, by_id["adaptive_raw"].context_strategy) == ("adaptive", "raw")
        assert (by_id["adaptive_managed"].tool_strategy, by_id["adaptive_managed"].context_strategy) == ("adaptive", "managed")

    def test_fixed_raw_is_first_the_default_baseline_position(self):
        assert ARMS[0].arm_id == ARM_FIXED_RAW

    def test_arms_by_ids_preserves_canonical_order_regardless_of_input_order(self):
        picked = arms_by_ids(["adaptive_managed", "fixed_raw"])
        assert [a.arm_id for a in picked] == ["fixed_raw", "adaptive_managed"]

    def test_arms_by_ids_rejects_unknown_id(self):
        with pytest.raises(ValueError):
            arms_by_ids(["banana"])


# ---------------------------------------------------------------------------
# 20. No second agent loop
# ---------------------------------------------------------------------------


def test_matrix_module_introduces_no_second_agent_loop():
    import inspect

    from forge.evaluation import matrix

    src = inspect.getsource(matrix)
    assert "class AgentRuntime" not in src
    assert "EvaluationRunner(" in src  # it delegates, never reimplements
    assert ".complete(" not in src  # never calls a provider directly


# ---------------------------------------------------------------------------
# Core matrix runner behaviour
# ---------------------------------------------------------------------------


class TestMatrixRunnerBasics:
    def test_runs_full_matrix_for_one_task(self, tmp_path):
        runner = _runner(tmp_path)
        results, summary = runner.run([_calc_task()])
        assert len(results) == 4
        assert [r.arm_id for r in results] == [a.arm_id for a in ARMS]
        assert all(r.task_id == "calc-add" for r in results)
        assert all(r.status == "completed" and r.success for r in results)

    def test_result_carries_both_strategy_labels_and_ids(self, tmp_path):
        runner = _runner(tmp_path)
        results, summary = runner.run([_calc_task()])
        by_arm = {r.arm_id: r for r in results}
        for arm in ARMS:
            r = by_arm[arm.arm_id]
            assert r.tool_strategy == arm.tool_strategy
            assert r.context_strategy == arm.context_strategy
            assert r.experiment_id == summary.experiment_id
            assert r.task_id == "calc-add"

    def test_task_category_and_exposed_tools_recorded(self, tmp_path):
        runner = _runner(tmp_path)
        results, _ = runner.run([_calc_task(category="file_creation")])
        for r in results:
            assert r.task_category == "file_creation"
            assert r.exposed_tools  # non-empty for every arm
        by_arm = {r.arm_id: r for r in results}
        # This prompt doesn't match any AdaptiveToolExposure keyword rule
        # (see forge/tools.py), so both Fixed and Adaptive fall back to
        # exposing every registered tool — still a meaningful equality
        # check that exposed_tools is populated consistently either way.
        assert set(by_arm["fixed_raw"].exposed_tools) == set(by_arm["fixed_managed"].exposed_tools)

    def test_correctness_check_result_preserved(self, tmp_path):
        runner = _runner(tmp_path)
        results, _ = runner.run([_calc_task()])
        for r in results:
            assert r.checks_run == 1
            assert r.checks_passed is True


# ---------------------------------------------------------------------------
# 3. Deterministic ordering
# ---------------------------------------------------------------------------


def test_deterministic_task_and_arm_ordering(tmp_path):
    tasks = [_calc_task("t1"), _calc_task("t2")]
    runner = _runner(tmp_path)
    results, _ = runner.run(tasks)
    pairs = [(r.task_id, r.arm_id) for r in results]
    assert pairs == [
        ("t1", "fixed_raw"), ("t1", "fixed_managed"),
        ("t1", "adaptive_raw"), ("t1", "adaptive_managed"),
        ("t2", "fixed_raw"), ("t2", "fixed_managed"),
        ("t2", "adaptive_raw"), ("t2", "adaptive_managed"),
    ]


# ---------------------------------------------------------------------------
# 4. Same task IDs used across arms
# ---------------------------------------------------------------------------


def test_same_task_id_appears_once_per_arm(tmp_path):
    tasks = [_calc_task("t1"), _calc_task("t2")]
    runner = _runner(tmp_path)
    results, _ = runner.run(tasks)
    for task_id in ("t1", "t2"):
        arm_ids = sorted(r.arm_id for r in results if r.task_id == task_id)
        assert arm_ids == sorted(a.arm_id for a in ARMS)


# ---------------------------------------------------------------------------
# 5-6. Fresh workspace per task-arm run, fixture isolation
# ---------------------------------------------------------------------------


class TestWorkspaceIsolation:
    def test_every_task_arm_gets_a_distinct_workspace(self, tmp_path):
        runner = _runner(tmp_path)
        results, _ = runner.run([_calc_task("t1"), _calc_task("t2")])
        workspaces = [r.workspace for r in results]
        assert len(set(workspaces)) == len(workspaces)  # all unique
        for r in results:
            ws = Path(r.workspace)
            assert ws.is_dir()
            assert ws.name == r.arm_id
            assert ws.parent.name == r.task_id

    def test_fixture_is_independently_provisioned_per_arm(self, tmp_path):
        from forge.evaluation import FixtureFile

        task = EvalTask(
            task_id="edit-existing",
            prompt="Edit greeting.py to change hello to hi.",
            fixtures=(FixtureFile(path="greeting.py", content="def hello():\n    return 'hello'\n"),),
        )

        def provider_factory():
            # Every arm just reads then finishes; the point here is fixture
            # provisioning, not a real edit.
            return ScriptedProvider([LLMResponse(content="looked, done", stop_reason="stop")])

        runner = _runner(tmp_path, provider_factory=provider_factory)
        results, _ = runner.run([task], arms=arms_by_ids(["fixed_raw", "adaptive_managed"]))
        for r in results:
            fixture_path = Path(r.workspace) / "greeting.py"
            assert fixture_path.is_file()
            assert "hello" in fixture_path.read_text(encoding="utf-8")

    def test_one_arms_workspace_never_contains_another_arms_files(self, tmp_path):
        # write_file (as scripted) only ever touches calculator.py; confirm
        # each arm's workspace contains exactly that, nothing extra bled in
        # from another arm's run.
        runner = _runner(tmp_path)
        results, _ = runner.run([_calc_task()])
        for r in results:
            entries = {p.name for p in Path(r.workspace).iterdir()}
            assert entries == {"calculator.py"}


# ---------------------------------------------------------------------------
# 7. Correct tool/context strategy passed to each run (via the real trace)
# ---------------------------------------------------------------------------


def test_trace_confirms_the_strategies_actually_used(tmp_path):
    runner = _runner(tmp_path)
    results, _ = runner.run([_calc_task()])
    for r in results:
        arm = next(a for a in ARMS if a.arm_id == r.arm_id)
        events = [
            json.loads(line)
            for line in Path(r.trace_path).read_text(encoding="utf-8").splitlines()
        ]
        start = events[0]["data"]
        assert start["tool_exposure_strategy"] == arm.tool_strategy
        assert start["context_strategy"] == arm.context_strategy


# ---------------------------------------------------------------------------
# 10-11. Structured output reloadable; never overwrites an earlier run
# ---------------------------------------------------------------------------


class TestOutput:
    def test_results_and_metadata_are_reloadable(self, tmp_path):
        runner = _runner(tmp_path)
        results, summary = runner.run([_calc_task()])

        reloaded_results = read_results_jsonl(summary.results_file)
        assert len(reloaded_results) == len(results)
        assert {r.run_id for r in reloaded_results} == {r.run_id for r in results}
        assert {r.arm_id for r in reloaded_results} == {a.arm_id for a in ARMS}

        reloaded_summary = read_matrix_metadata(Path(summary.output_dir) / "metadata.json")
        assert reloaded_summary.experiment_id == summary.experiment_id
        assert reloaded_summary.task_ids == ("calc-add",)
        assert set(reloaded_summary.arm_ids) == {a.arm_id for a in ARMS}

    def test_rerunning_the_same_experiment_id_does_not_overwrite(self, tmp_path):
        runner = _runner(tmp_path)
        results, summary = runner.run([_calc_task()], experiment_id="exp-fixed-id")
        assert Path(summary.results_file).exists()

        with pytest.raises(FileExistsError):
            runner.run([_calc_task()], experiment_id="exp-fixed-id")

        # the first run's output is untouched
        assert len(read_results_jsonl(summary.results_file)) == 4

    def test_auto_generated_experiment_ids_differ_between_runs(self, tmp_path):
        runner = _runner(tmp_path)
        _, summary_a = runner.run([_calc_task()])
        _, summary_b = runner.run([_calc_task()])
        assert summary_a.experiment_id != summary_b.experiment_id
        assert summary_a.output_dir != summary_b.output_dir


# ---------------------------------------------------------------------------
# 12-13. Subset task / arm selection
# ---------------------------------------------------------------------------


def test_subset_task_selection(tmp_path):
    runner = _runner(tmp_path)
    results, _ = runner.run([_calc_task("only-this-one")])
    assert {r.task_id for r in results} == {"only-this-one"}
    assert len(results) == 4  # still all four arms for the one task


def test_subset_arm_selection(tmp_path):
    runner = _runner(tmp_path)
    chosen = arms_by_ids(["fixed_raw", "adaptive_managed"])
    results, summary = runner.run([_calc_task()], arms=chosen)
    assert len(results) == 2
    assert {r.arm_id for r in results} == {"fixed_raw", "adaptive_managed"}
    assert summary.arm_ids == ("fixed_raw", "adaptive_managed")


# ---------------------------------------------------------------------------
# 14-15. Provider/runtime failure preservation + continue-after-failure
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_provider_error_is_preserved_not_hidden(self, tmp_path):
        def provider_factory():
            return ScriptedProvider([LLMError("simulated quota exceeded")])

        runner = _runner(tmp_path, provider_factory=provider_factory)
        results, _ = runner.run([_calc_task()])
        assert len(results) == 4
        for r in results:
            assert r.status == "provider_error"
            assert r.success is False
            assert r.error and "quota exceeded" in r.error
            assert outcome_category(r) == "provider_error"
        # never fabricate tokens/metrics for a run that never got a response
        assert all(r.input_tokens is None for r in results)

    def test_runner_level_error_is_distinguishable_and_does_not_abort_the_batch(self, tmp_path):
        calls = {"n": 0}

        def flaky_factory():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated provider construction failure")
            return _calc_provider()

        runner = _runner(tmp_path, provider_factory=flaky_factory)
        results, _ = runner.run([_calc_task()])  # 4 arms; first factory call fails
        assert len(results) == 4

        errored = [r for r in results if r.status == STATUS_RUNNER_ERROR]
        ok = [r for r in results if r.status == "completed"]
        assert len(errored) == 1
        assert len(ok) == 3
        assert errored[0].success is False
        assert "simulated provider construction failure" in errored[0].error
        assert outcome_category(errored[0]) == "runner_error"
        # the failed arm still carries full identity metadata
        assert errored[0].task_id == "calc-add"
        assert errored[0].arm_id in {a.arm_id for a in ARMS}
        # the other three arms completed normally, proving the batch continued
        assert all(r.success for r in ok)

    def test_incorrect_solution_is_distinct_from_an_error(self, tmp_path):
        def wrong_provider():
            return ScriptedProvider([LLMResponse(content="I did nothing.", stop_reason="stop")])

        runner = _runner(tmp_path, provider_factory=wrong_provider)
        results, _ = runner.run([_calc_task()])
        for r in results:
            assert r.status == "completed"       # the run itself finished fine
            assert r.checks_passed is False        # but the artefact is wrong
            assert r.success is False
            assert r.error is None                 # not an error — a wrong answer
            assert outcome_category(r) == "completed_incorrect"


# ---------------------------------------------------------------------------
# outcome_category coverage
# ---------------------------------------------------------------------------


def test_outcome_category_covers_limit_reached():
    from forge.evaluation import EvalResult

    r = EvalResult(
        task_id="t", run_id="r", provider="p", model="m",
        tool_strategy="fixed", context_strategy="raw",
        status="max_turns_exceeded", success=False,
    )
    assert outcome_category(r) == "limit_reached"


def test_outcome_category_covers_runtime_error():
    from forge.evaluation import EvalResult

    r = EvalResult(
        task_id="t", run_id="r", provider="p", model="m",
        tool_strategy="fixed", context_strategy="raw",
        status="failed", success=False,
    )
    assert outcome_category(r) == "runtime_error"


# ---------------------------------------------------------------------------
# Guard: at least one arm actually resolves to ManagedContextStrategy /
# AdaptiveToolExposure (i.e. this isn't secretly running fixed+raw four times)
# ---------------------------------------------------------------------------


def test_arms_actually_resolve_to_distinct_strategy_objects():
    for arm in ARMS:
        from forge.evaluation.experiment import ExperimentConfig

        cfg = ExperimentConfig(
            task_id="x", provider="gemini", model="m",
            tool_strategy=arm.tool_strategy, context_strategy=arm.context_strategy,
        )
        tool_strat, ctx_strat = cfg.resolve_strategies()
        assert tool_strat.name == arm.tool_strategy
        assert ctx_strat.name == arm.context_strategy
