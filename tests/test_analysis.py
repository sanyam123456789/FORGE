"""
Tests for forge.evaluation.analysis — Step 8 statistical analysis.

All data here is synthetic ``EvalResult``/JSONL test fixture data, built by
hand or via ``MatrixRunner`` with a scripted fake provider (never a real
Gemini call). No network, no API key.
"""

from __future__ import annotations

import json

import pytest

from forge.evaluation import (
    ARM_ADAPTIVE_MANAGED,
    ARM_ADAPTIVE_RAW,
    ARM_FIXED_MANAGED,
    ARM_FIXED_RAW,
    ARMS,
    EvalResult,
    analyze,
    check_matrix_completeness,
    group_by_arm,
    group_by_task,
    interaction_effect,
    paired_metric_comparison,
    read_analysis_json,
    run_analysis,
    success_discordance,
    summarize_metric,
    summarize_outcomes,
    validate_results,
    wilcoxon_paired_metric,
    write_analysis_json,
    write_results_jsonl,
)
from forge.evaluation.analysis import NUMERIC_METRICS


# ---------------------------------------------------------------------------
# Fixture helpers — clearly synthetic test data
# ---------------------------------------------------------------------------


def _mk(
    task_id="t1",
    arm_id=ARM_FIXED_RAW,
    *,
    run_id=None,
    status="completed",
    success=True,
    experiment_id="exp-test",
    task_category="file_creation",
    input_tokens=100,
    output_tokens=50,
    llm_calls=2,
    tool_calls=1,
    duration_s=1.0,
    exposed_tools=("write_file", "read_file"),
    cost_available=False,
    cost_usd=None,
    **extra,
) -> EvalResult:
    from forge.evaluation.matrix import ARMS_BY_ID

    arm = ARMS_BY_ID.get(arm_id) if arm_id else None
    tool_strategy = arm.tool_strategy if arm else "fixed"
    context_strategy = arm.context_strategy if arm else "raw"
    run_id = run_id or f"run-{task_id}-{arm_id}-{id(object())}"
    return EvalResult(
        task_id=task_id,
        run_id=run_id,
        provider="scripted",
        model="fake-model",
        tool_strategy=tool_strategy,
        context_strategy=context_strategy,
        status=status,
        success=success,
        experiment_id=experiment_id,
        arm_id=arm_id,
        task_category=task_category,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        duration_s=duration_s,
        exposed_tools=list(exposed_tools) if exposed_tools is not None else None,
        cost_available=cost_available,
        cost_usd=cost_usd,
        **extra,
    )


def _full_matrix(task_id="t1") -> list[EvalResult]:
    """One record per canonical arm for *task_id*, distinct token counts."""
    return [
        _mk(task_id, ARM_FIXED_RAW, input_tokens=100, output_tokens=50, llm_calls=3),
        _mk(task_id, ARM_FIXED_MANAGED, input_tokens=90, output_tokens=45, llm_calls=3),
        _mk(task_id, ARM_ADAPTIVE_RAW, input_tokens=70, output_tokens=40, llm_calls=2),
        _mk(task_id, ARM_ADAPTIVE_MANAGED, input_tokens=60, output_tokens=35, llm_calls=2),
    ]


# ---------------------------------------------------------------------------
# 1-2. Loading (empty + malformed)
# ---------------------------------------------------------------------------


class TestLoading:
    def test_load_valid_jsonl(self, tmp_path):
        rows = _full_matrix("t1")
        path = write_results_jsonl(rows, tmp_path / "results.jsonl")
        report = run_analysis(path)
        assert report.record_count == 4
        assert report.task_count == 1

    def test_empty_results_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        report = run_analysis(path)
        assert report.record_count == 0
        assert report.task_count == 0
        codes = {w["code"] for w in report.warnings}
        assert "empty_dataset" in codes

    def test_malformed_json_line_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"not": "valid EvalResult"}\nnot even json\n', encoding="utf-8")
        with pytest.raises(ValueError):
            run_analysis(path)


# ---------------------------------------------------------------------------
# 3-4. Validation: duplicates, unknown arm ids
# ---------------------------------------------------------------------------


class TestValidation:
    def test_duplicate_task_arm_flagged_and_kept(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, run_id="r1"),
            _mk("t1", ARM_FIXED_RAW, run_id="r2"),  # duplicate cell
        ]
        issues = validate_results(rows)
        dup = [i for i in issues if i.code == "duplicate_task_arm"]
        assert len(dup) == 1
        assert set(dup[0].detail["run_ids"]) == {"r1", "r2"}

    def test_unknown_arm_id_flagged_as_error(self):
        rows = [_mk("t1", "not_a_real_arm", run_id="r1")]
        issues = validate_results(rows)
        found = [i for i in issues if i.code == "unknown_arm_id"]
        assert len(found) == 1
        assert found[0].severity == "error"

    def test_missing_task_id_flagged(self):
        rows = [_mk("", ARM_FIXED_RAW, run_id="r1")]
        issues = validate_results(rows)
        assert any(i.code == "missing_task_id" for i in issues)

    def test_inconsistent_strategy_label_flagged(self):
        r = _mk("t1", ARM_FIXED_RAW, run_id="r1")
        # Corrupt the label without changing arm_id — simulates bad data.
        import dataclasses

        bad = dataclasses.replace(r, tool_strategy="adaptive")
        issues = validate_results([bad])
        assert any(i.code == "inconsistent_strategy_label" for i in issues)

    def test_mismatched_task_category_flagged(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, run_id="r1", task_category="file_creation"),
            _mk("t1", ARM_FIXED_MANAGED, run_id="r2", task_category="bug_fix"),
        ]
        issues = validate_results(rows)
        assert any(i.code == "mismatched_task_category" for i in issues)

    def test_malformed_numeric_value_flagged_and_excluded(self):
        import dataclasses

        r = _mk("t1", ARM_FIXED_RAW, run_id="r1")
        bad = dataclasses.replace(r, input_tokens="banana")  # malformed
        issues = validate_results([bad])
        found = [i for i in issues if i.code == "malformed_numeric_value"]
        assert any(i.detail["metric"] == "input_tokens" for i in found)
        # excluded from stats, not treated as missing or zero
        summary = summarize_metric([bad], "input_tokens")
        assert summary.valid_count == 0
        assert summary.invalid_count == 1
        assert summary.missing_count == 0
        assert summary.mean is None

    def test_missing_metric_not_treated_as_zero(self):
        r = _mk("t1", ARM_FIXED_RAW, run_id="r1", input_tokens=None)
        summary = summarize_metric([r], "input_tokens")
        assert summary.missing_count == 1
        assert summary.valid_count == 0
        assert summary.mean is None
        assert summary.total is None

    def test_cost_not_available_is_not_applicable_not_missing(self):
        r = _mk("t1", ARM_FIXED_RAW, run_id="r1", cost_available=False, cost_usd=None)
        summary = summarize_metric([r], "cost_usd")
        assert summary.not_applicable_count == 1
        assert summary.missing_count == 0
        assert summary.valid_count == 0


# ---------------------------------------------------------------------------
# 5-6. Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_by_arm(self):
        rows = _full_matrix("t1")
        by_arm = group_by_arm(rows)
        assert set(by_arm) == {a.arm_id for a in ARMS}
        assert all(len(v) == 1 for v in by_arm.values())

    def test_group_by_task(self):
        rows = _full_matrix("t1") + _full_matrix("t2")
        by_task = group_by_task(rows)
        assert set(by_task) == {"t1", "t2"}
        assert len(by_task["t1"]) == 4


# ---------------------------------------------------------------------------
# 7. Correct four-arm grouping / outcome counts
# ---------------------------------------------------------------------------


class TestOutcomeSummary:
    def test_success_and_error_counts(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, status="completed", success=True),
            _mk("t2", ARM_FIXED_RAW, status="completed", success=False),
            _mk("t3", ARM_FIXED_RAW, status="provider_error", success=False, error="boom"),
            _mk("t4", ARM_FIXED_RAW, status="max_turns_exceeded", success=False),
            _mk("t5", ARM_FIXED_RAW, status="failed", success=False),
            _mk("t6", ARM_FIXED_RAW, status="runner_error", success=False, error="boom"),
        ]
        s = summarize_outcomes(rows, "fixed_raw")
        assert s.total == 6
        assert s.success == 1
        assert s.completed_incorrect == 1
        assert s.provider_error == 1
        assert s.limit_reached == 1
        assert s.runtime_error == 1
        assert s.runner_error == 1
        assert s.denom_all == 6
        assert s.denom_excl_infra_errors == 4  # excludes provider_error + runner_error
        assert s.success_rate_all == pytest.approx(1 / 6)
        assert s.success_rate_excl_infra_errors == pytest.approx(1 / 4)

    def test_zero_denominator_is_none_not_zero(self):
        s = summarize_outcomes([], "empty")
        assert s.success_rate_all is None
        assert s.success_rate_excl_infra_errors is None


# ---------------------------------------------------------------------------
# 10. Mean/median/min/max/stdev
# ---------------------------------------------------------------------------


class TestMetricSummary:
    def test_basic_stats(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, input_tokens=10),
            _mk("t2", ARM_FIXED_RAW, input_tokens=20),
            _mk("t3", ARM_FIXED_RAW, input_tokens=30),
        ]
        s = summarize_metric(rows, "input_tokens")
        assert s.valid_count == 3
        assert s.mean == 20
        assert s.median == 20
        assert s.minimum == 10
        assert s.maximum == 30
        assert s.total == 60
        assert s.stdev is not None

    def test_stdev_none_below_two_observations(self):
        rows = [_mk("t1", ARM_FIXED_RAW, input_tokens=10)]
        s = summarize_metric(rows, "input_tokens")
        assert s.valid_count == 1
        assert s.stdev is None

    def test_all_metrics_missing_dataset(self):
        rows = [
            _mk(
                "t1", ARM_FIXED_RAW,
                input_tokens=None, output_tokens=None, llm_calls=None,
                tool_calls=None, duration_s=None, exposed_tools=None,
            )
        ]
        for metric in NUMERIC_METRICS:
            s = summarize_metric(rows, metric)
            assert s.valid_count == 0
            assert s.mean is None


# ---------------------------------------------------------------------------
# 11-12. Paired task comparison + missing pairs
# ---------------------------------------------------------------------------


class TestPairedComparison:
    def test_paired_diffs(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, input_tokens=100),
            _mk("t1", ARM_ADAPTIVE_RAW, input_tokens=60),
            _mk("t2", ARM_FIXED_RAW, input_tokens=200),
            _mk("t2", ARM_ADAPTIVE_RAW, input_tokens=150),
        ]
        comp = paired_metric_comparison(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert comp.n_pairs == 2
        assert comp.per_task_diff == {"t1": -40, "t2": -50}
        assert comp.mean_diff == pytest.approx(-45)
        assert comp.median_diff == pytest.approx(-45)
        # lower_is_better for tokens -> adaptive (arm_b) used fewer -> improved
        assert comp.lower_is_better is True
        assert comp.improved_arm == "arm_b"

    def test_missing_task_in_one_arm(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, input_tokens=100),
            _mk("t1", ARM_ADAPTIVE_RAW, input_tokens=60),
            _mk("t2", ARM_FIXED_RAW, input_tokens=200),  # no adaptive_raw record for t2
        ]
        comp = paired_metric_comparison(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert comp.n_pairs == 1
        assert comp.tasks_without_record_in_b == ["t2"]
        assert comp.tasks_without_record_in_a == []

    def test_invalid_value_excluded_from_pairing(self):
        import dataclasses

        r_bad = dataclasses.replace(
            _mk("t1", ARM_ADAPTIVE_RAW, input_tokens=60), input_tokens="oops"
        )
        rows = [_mk("t1", ARM_FIXED_RAW, input_tokens=100), r_bad]
        comp = paired_metric_comparison(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert comp.n_pairs == 0
        assert comp.tasks_invalid_in_b == ["t1"]


# ---------------------------------------------------------------------------
# 13. Percentage-change edge cases
# ---------------------------------------------------------------------------


class TestPercentageChange:
    def test_normal_pct_change(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, input_tokens=100),
            _mk("t1", ARM_ADAPTIVE_RAW, input_tokens=50),
        ]
        comp = paired_metric_comparison(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert comp.per_task_pct_change["t1"] == pytest.approx(-50.0)

    def test_division_by_zero_marked_undefined(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, tool_calls=0),
            _mk("t1", ARM_ADAPTIVE_RAW, tool_calls=5),
        ]
        comp = paired_metric_comparison(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "tool_calls")
        assert "t1" not in comp.per_task_pct_change
        assert comp.pct_change_undefined_tasks == ["t1"]
        # the raw diff is still reported
        assert comp.per_task_diff["t1"] == 5


# ---------------------------------------------------------------------------
# 14. Factor-level grouping
# ---------------------------------------------------------------------------


class TestFactorLevel:
    def test_factor_level_present_with_full_matrix(self):
        rows = _full_matrix("t1")
        report = analyze(rows)
        assert set(report.factor_level["tool_exposure"]) == {
            f"{ARM_FIXED_RAW}__vs__{ARM_ADAPTIVE_RAW}",
            f"{ARM_FIXED_MANAGED}__vs__{ARM_ADAPTIVE_MANAGED}",
        }
        assert set(report.factor_level["context_strategy"]) == {
            f"{ARM_FIXED_RAW}__vs__{ARM_FIXED_MANAGED}",
            f"{ARM_ADAPTIVE_RAW}__vs__{ARM_ADAPTIVE_MANAGED}",
        }
        assert report.factor_level["interaction"] is not None

    def test_interaction_skipped_with_partial_arms(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, input_tokens=100),
            _mk("t1", ARM_ADAPTIVE_RAW, input_tokens=60),
        ]
        report = analyze(rows)
        assert report.factor_level["interaction"] is None
        assert any(w["code"] == "interaction_skipped" for w in report.warnings)

    def test_interaction_diff_in_diff_value(self):
        rows = _full_matrix("t1")
        result = interaction_effect(rows, "input_tokens")
        assert result.n_tasks == 1
        # (60-70) - (90-100) = -10 - (-10) = 0
        assert result.mean_diff_in_diff == pytest.approx(0.0)
        assert "descriptive" in result.note.lower()


# ---------------------------------------------------------------------------
# 15-17. Statistical test behaviour: small sample / constant diff / skip reasons
# ---------------------------------------------------------------------------


class TestStatisticalTests:
    def test_small_sample_skipped(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, input_tokens=100),
            _mk("t1", ARM_ADAPTIVE_RAW, input_tokens=60),
        ]
        result = wilcoxon_paired_metric(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert result.skipped is True
        assert "fewer than" in result.skip_reason
        assert result.w_statistic is None
        assert "p-value" in result.note

    def test_constant_difference_skipped(self):
        rows = []
        for i in range(6):
            rows.append(_mk(f"t{i}", ARM_FIXED_RAW, input_tokens=100))
            rows.append(_mk(f"t{i}", ARM_ADAPTIVE_RAW, input_tokens=100))  # always equal
        result = wilcoxon_paired_metric(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert result.skipped is True
        assert "zero" in result.skip_reason

    def test_no_pairs_skipped(self):
        result = wilcoxon_paired_metric([], ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert result.skipped is True
        assert "no paired observations" in result.skip_reason

    def test_enough_observations_computes_statistic(self):
        rows = []
        for i, (a, b) in enumerate([(100, 60), (200, 90), (50, 10), (80, 20), (120, 30)]):
            rows.append(_mk(f"t{i}", ARM_FIXED_RAW, input_tokens=a))
            rows.append(_mk(f"t{i}", ARM_ADAPTIVE_RAW, input_tokens=b))
        result = wilcoxon_paired_metric(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        assert result.skipped is False
        assert result.w_statistic is not None
        assert result.n_used == 5
        # never a p-value or significance label anywhere on the result
        assert not hasattr(result, "p_value")
        assert not hasattr(result, "significant")

    def test_never_reports_p_value_in_dict(self):
        rows = []
        for i, (a, b) in enumerate([(100, 60), (200, 90), (50, 10), (80, 20), (120, 30)]):
            rows.append(_mk(f"t{i}", ARM_FIXED_RAW, input_tokens=a))
            rows.append(_mk(f"t{i}", ARM_ADAPTIVE_RAW, input_tokens=b))
        result = wilcoxon_paired_metric(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW, "input_tokens")
        d = result.to_dict()
        for key in d:
            assert "p_value" not in key and "significan" not in key.lower()

    def test_success_discordance_counts(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW, success=True),
            _mk("t1", ARM_ADAPTIVE_RAW, success=False),
            _mk("t2", ARM_FIXED_RAW, success=False),
            _mk("t2", ARM_ADAPTIVE_RAW, success=True),
            _mk("t3", ARM_FIXED_RAW, success=True),
            _mk("t3", ARM_ADAPTIVE_RAW, success=True),
        ]
        d = success_discordance(rows, ARM_FIXED_RAW, ARM_ADAPTIVE_RAW)
        assert d.n_pairs == 3
        assert d.both_success == 1
        assert d.a_only == 1
        assert d.b_only == 1
        assert d.both_fail == 0


# ---------------------------------------------------------------------------
# Missing arms / incomplete matrix
# ---------------------------------------------------------------------------


class TestCompleteness:
    def test_incomplete_matrix_warning(self):
        rows = [
            _mk("t1", ARM_FIXED_RAW),
            _mk("t1", ARM_FIXED_MANAGED),
            # t1 missing adaptive_raw / adaptive_managed
        ]
        issues = check_matrix_completeness(rows, [a.arm_id for a in ARMS], ["t1"])
        found = [i for i in issues if i.code == "incomplete_matrix"]
        assert len(found) == 1
        assert set(found[0].detail["missing_by_task"]["t1"]) == {
            ARM_ADAPTIVE_RAW, ARM_ADAPTIVE_MANAGED,
        }

    def test_no_warning_when_arms_or_tasks_empty(self):
        assert check_matrix_completeness([], [], ["t1"]) == []
        assert check_matrix_completeness([], [ARM_FIXED_RAW], []) == []

    def test_analyze_handles_missing_arm_entirely(self):
        # only two of the four arms ever appear in the data
        rows = [_mk("t1", ARM_FIXED_RAW), _mk("t1", ARM_ADAPTIVE_RAW)]
        report = analyze(rows)
        assert set(report.arms_analyzed) == {ARM_FIXED_RAW, ARM_ADAPTIVE_RAW}
        assert report.factor_level["interaction"] is None


# ---------------------------------------------------------------------------
# One-task dataset / degenerate inputs
# ---------------------------------------------------------------------------


def test_one_task_dataset_does_not_crash():
    rows = _full_matrix("only-task")
    report = analyze(rows)
    assert report.task_count == 1
    assert report.record_count == 4
    # statistical tests all present but skipped (n too small)
    for pair_key, metrics in report.statistical_tests.items():
        for metric, result in metrics.items():
            assert result["skipped"] is True


def test_empty_dataset_analysis_does_not_crash():
    report = analyze([])
    assert report.task_count == 0
    assert report.record_count == 0
    assert report.paired_comparisons == {}
    assert set(report.unavailable_metrics) == set(NUMERIC_METRICS)


# ---------------------------------------------------------------------------
# 18-19. Output JSON round-trip + source JSONL untouched
# ---------------------------------------------------------------------------


class TestOutput:
    def test_write_and_reload_analysis_json(self, tmp_path):
        rows = _full_matrix("t1")
        report = analyze(rows, source_results_path="in-memory")
        out = write_analysis_json(report, tmp_path / "analysis.json")
        reloaded = read_analysis_json(out)
        assert reloaded["record_count"] == 4
        assert reloaded["schema_version"] == report.schema_version
        assert reloaded["arms_analyzed"] == report.arms_analyzed

    def test_refuses_to_overwrite_by_default(self, tmp_path):
        rows = _full_matrix("t1")
        report = analyze(rows)
        out_path = tmp_path / "analysis.json"
        write_analysis_json(report, out_path)
        with pytest.raises(FileExistsError):
            write_analysis_json(report, out_path)
        # succeeds with overwrite=True
        write_analysis_json(report, out_path, overwrite=True)

    def test_source_jsonl_is_unchanged(self, tmp_path):
        rows = _full_matrix("t1")
        results_path = write_results_jsonl(rows, tmp_path / "results.jsonl")
        before = results_path.read_text(encoding="utf-8")
        report = run_analysis(results_path)
        write_analysis_json(report, tmp_path / "analysis.json")
        after = results_path.read_text(encoding="utf-8")
        assert before == after

    def test_analysis_json_is_a_separate_file(self, tmp_path):
        rows = _full_matrix("t1")
        results_path = write_results_jsonl(rows, tmp_path / "results.jsonl")
        report = run_analysis(results_path)
        out = write_analysis_json(report, tmp_path / "analysis.json")
        assert out != results_path
        assert out.exists() and results_path.exists()


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def test_metadata_auto_discovered_next_to_results(tmp_path):
    from forge.evaluation.matrix import MatrixExperimentSummary

    rows = _full_matrix("t1")
    results_path = write_results_jsonl(rows, tmp_path / "results.jsonl")
    summary = MatrixExperimentSummary(
        experiment_id="exp-test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:01:00+00:00",
        provider="scripted",
        model="fake-model",
        temperature=0.0,
        max_output_tokens=4096,
        max_llm_turns=30,
        max_tool_calls=50,
        arm_ids=tuple(a.arm_id for a in ARMS),
        task_ids=("t1",),
        output_dir=str(tmp_path),
        results_file=str(results_path),
    )
    (tmp_path / "metadata.json").write_text(json.dumps(summary.to_dict()), encoding="utf-8")

    report = run_analysis(results_path)
    assert report.metadata is not None
    assert report.metadata["experiment_id"] == "exp-test"
    assert report.source_metadata_path is not None


def test_missing_metadata_is_not_an_error(tmp_path):
    rows = _full_matrix("t1")
    results_path = write_results_jsonl(rows, tmp_path / "results.jsonl")
    report = run_analysis(results_path)  # no metadata.json written
    assert report.metadata is None
    assert report.source_metadata_path is None


# ---------------------------------------------------------------------------
# arm / task filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_arm_filter_restricts_analysis(self):
        rows = _full_matrix("t1")
        report = analyze(rows, arms=[ARM_FIXED_RAW, ARM_ADAPTIVE_RAW])
        assert report.arms_analyzed == [ARM_FIXED_RAW, ARM_ADAPTIVE_RAW]
        assert report.record_count == 2

    def test_task_filter_restricts_analysis(self):
        rows = _full_matrix("t1") + _full_matrix("t2")
        report = analyze(rows, task_ids=["t1"])
        assert report.task_ids_analyzed == ["t1"]
        assert report.record_count == 4

    def test_unmatched_task_filter_yields_empty_not_crash(self):
        rows = _full_matrix("t1")
        report = analyze(rows, task_ids=["no-such-task"])
        assert report.record_count == 0
        assert report.task_count == 1  # the filter list itself, even if unmatched


# ---------------------------------------------------------------------------
# Guard: analysis module introduces no agent loop / no provider calls
# ---------------------------------------------------------------------------


def test_analysis_module_introduces_no_second_agent_loop_or_network():
    import inspect

    from forge.evaluation import analysis

    src = inspect.getsource(analysis)
    assert "class AgentRuntime" not in src
    assert ".complete(" not in src  # never calls a provider directly
    assert "requests." not in src
    assert "urllib" not in src
    assert "socket" not in src


def test_no_p_value_or_significance_anywhere_in_report():
    rows = _full_matrix("t1")
    report = analyze(rows)
    blob = json.dumps(report.to_dict()).lower()
    assert "p_value" not in blob
    assert "pvalue" not in blob
    assert '"significant"' not in blob
    assert '"significance"' not in blob
