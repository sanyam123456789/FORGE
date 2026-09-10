"""
Tests for the Step 4 baseline task suite and its loader
(``forge.evaluation.suite``) plus the small ``EvalTask`` extensions it uses
(fixtures + richer ``ArtifactCheck``).

No real Gemini call: the one end-to-end test drives the real ``AgentRuntime``
with a scripted fake provider and the real built-in tools in a tmp workspace.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from forge.evaluation import (
    ArtifactCheck,
    EvalTask,
    EvaluationRunner,
    ExperimentConfig,
    FixtureFile,
)
from forge.evaluation.suite import (
    DEFAULT_TASKS_DIR,
    TaskSuiteError,
    get_task,
    list_task_ids,
    load_suite,
    load_suite_map,
    load_task,
)
from forge.llm import LLMProvider, LLMResponse, ToolCall

EXPECTED_CATEGORIES = {
    "file_creation",
    "file_editing",
    "bug_fix",
    "small_feature",
    "refactoring",
    "testing",
    "multi_file",
}


# ---------------------------------------------------------------------------
# The real, version-controlled baseline suite
# ---------------------------------------------------------------------------


class TestBaselineSuite:
    def test_suite_loads(self):
        tasks = load_suite()
        assert 8 <= len(tasks) <= 12
        assert all(isinstance(t, EvalTask) for t in tasks)

    def test_task_ids_unique_and_stable(self):
        ids = list_task_ids()
        assert len(ids) == len(set(ids)), "task_ids must be unique"
        # identity is a plain slug — no strategy / provider / model / run id
        for tid in ids:
            for banned in ("fixed", "raw", "adaptive", "managed", "gemini", "run_"):
                assert banned not in tid

    def test_every_task_has_prompt_and_at_least_one_check(self):
        for t in load_suite():
            assert t.prompt.strip()
            assert len(t.checks) >= 1

    def test_categories_and_metadata_present(self):
        seen = set()
        for t in load_suite():
            cat = t.metadata.get("category")
            assert cat, f"{t.task_id} missing category"
            assert t.metadata.get("difficulty") in {"easy", "medium", "hard"}
            assert t.metadata.get("offline") is True
            seen.add(cat)
        assert seen == EXPECTED_CATEGORIES

    def test_fixture_tasks_resolve_starter_files(self):
        by_id = load_suite_map()
        # editing / bug-fix / refactor / multi-file tasks carry starter files
        assert by_id["edit-report-summary"].fixtures
        assert by_id["multi-file-timeout-setting"].fixtures
        for fx in by_id["multi-file-timeout-setting"].fixtures:
            assert fx.path in {"client.py", "settings_mod.py"}
            assert fx.content.strip()
        # pure file-creation tasks carry none
        assert by_id["create-string-utils"].fixtures == ()

    def test_get_task_by_id(self):
        t = get_task("create-string-utils")
        assert t.task_id == "create-string-utils"
        with pytest.raises(TaskSuiteError, match="unknown task_id"):
            get_task("does-not-exist")

    def test_provision_writes_fixture_files(self, tmp_path):
        task = get_task("fix-inclusive-sum")
        written = task.provision(tmp_path)
        assert written == ["ranges.py"]
        assert (tmp_path / "ranges.py").read_text(encoding="utf-8").startswith(
            "def inclusive_sum(n):"
        )

    def test_default_tasks_dir_points_into_repo(self):
        assert DEFAULT_TASKS_DIR.is_dir()
        assert DEFAULT_TASKS_DIR.name == "tasks"
        assert (DEFAULT_TASKS_DIR / "fixtures").is_dir()


# ---------------------------------------------------------------------------
# Loader validation on hand-written task files
# ---------------------------------------------------------------------------


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


class TestLoaderValidation:
    def test_valid_single_task_loads_with_metadata_and_checks(self, tmp_path):
        p = _write(
            tmp_path / "t.json",
            {
                "task_id": "demo-task",
                "prompt": "Create demo.py with a function demo().",
                "checks": [{"path": "demo.py", "must_contain": ["def demo("]}],
                "metadata": {"category": "file_creation", "difficulty": "easy"},
            },
        )
        task = load_task(p)
        assert task.task_id == "demo-task"
        assert task.metadata["category"] == "file_creation"
        assert task.checks[0].must_contain == ("def demo(",)

    def test_missing_task_id_rejected(self, tmp_path):
        p = _write(tmp_path / "t.json", {"prompt": "do a thing"})
        with pytest.raises(TaskSuiteError, match="missing 'task_id'"):
            load_task(p)

    def test_empty_task_id_rejected(self, tmp_path):
        p = _write(tmp_path / "t.json", {"task_id": "  ", "prompt": "do a thing"})
        with pytest.raises(TaskSuiteError, match="task_id must be a non-empty"):
            load_task(p)

    def test_missing_prompt_rejected(self, tmp_path):
        p = _write(tmp_path / "t.json", {"task_id": "x"})
        with pytest.raises(TaskSuiteError, match="missing 'prompt'"):
            load_task(p)

    def test_empty_prompt_rejected(self, tmp_path):
        p = _write(tmp_path / "t.json", {"task_id": "x", "prompt": "   "})
        with pytest.raises(TaskSuiteError, match="prompt must be a non-empty"):
            load_task(p)

    def test_malformed_json_rejected(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(TaskSuiteError, match="invalid JSON"):
            load_task(p)

    def test_non_object_json_rejected(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(TaskSuiteError, match="must contain a JSON object"):
            load_task(p)

    def test_malformed_check_rejected(self, tmp_path):
        p = _write(
            tmp_path / "t.json",
            {"task_id": "x", "prompt": "p", "checks": [{"must_contain": ["z"]}]},
        )
        with pytest.raises(TaskSuiteError, match="malformed check"):
            load_task(p)

    def test_checks_not_a_list_rejected(self, tmp_path):
        p = _write(
            tmp_path / "t.json",
            {"task_id": "x", "prompt": "p", "checks": {"path": "a"}},
        )
        with pytest.raises(TaskSuiteError, match="'checks' must be a list"):
            load_task(p)

    def test_bad_metadata_rejected(self, tmp_path):
        p = _write(
            tmp_path / "t.json",
            {"task_id": "x", "prompt": "p", "metadata": ["not", "an", "object"]},
        )
        with pytest.raises(TaskSuiteError, match="'metadata' must be an object"):
            load_task(p)

    def test_unsafe_check_path_rejected(self, tmp_path):
        p = _write(
            tmp_path / "t.json",
            {
                "task_id": "x",
                "prompt": "p",
                "checks": [{"path": "../escape.py", "must_contain": ["z"]}],
            },
        )
        with pytest.raises(TaskSuiteError, match="unsafe check path"):
            load_task(p)

    def test_missing_fixture_dir_rejected(self, tmp_path):
        p = _write(
            tmp_path / "t.json",
            {"task_id": "x", "prompt": "p", "fixtures": {"dir": "nope"}},
        )
        with pytest.raises(TaskSuiteError, match="fixture directory not found"):
            load_task(p)

    def test_fixture_dir_resolved_from_disk(self, tmp_path):
        fx = tmp_path / "fixtures" / "x"
        fx.mkdir(parents=True)
        (fx / "starter.py").write_text("VALUE = 1\n", encoding="utf-8")
        p = _write(
            tmp_path / "x.json",
            {"task_id": "x", "prompt": "p", "fixtures": {"dir": "fixtures/x"}},
        )
        task = load_task(p)
        assert task.fixtures == (FixtureFile(path="starter.py", content="VALUE = 1\n"),)

    def test_inline_fixture_list_supported(self, tmp_path):
        p = _write(
            tmp_path / "x.json",
            {
                "task_id": "x",
                "prompt": "p",
                "fixtures": [{"path": "a.py", "content": "x = 1\n"}],
            },
        )
        task = load_task(p)
        assert task.fixtures[0].path == "a.py"


class TestSuiteAggregate:
    def test_duplicate_task_ids_rejected(self, tmp_path):
        _write(tmp_path / "a.json", {"task_id": "dup", "prompt": "p1"})
        _write(tmp_path / "b.json", {"task_id": "dup", "prompt": "p2"})
        with pytest.raises(TaskSuiteError, match="duplicate task_id 'dup'"):
            load_suite(tmp_path)

    def test_empty_dir_rejected(self, tmp_path):
        with pytest.raises(TaskSuiteError, match="no '\\*.json' task files"):
            load_suite(tmp_path)

    def test_missing_dir_rejected(self, tmp_path):
        with pytest.raises(TaskSuiteError, match="task suite directory not found"):
            load_suite(tmp_path / "nope")

    def test_multiple_tasks_load_in_filename_order(self, tmp_path):
        _write(tmp_path / "02.json", {"task_id": "second", "prompt": "p"})
        _write(tmp_path / "01.json", {"task_id": "first", "prompt": "p"})
        assert [t.task_id for t in load_suite(tmp_path)] == ["first", "second"]


# ---------------------------------------------------------------------------
# Richer ArtifactCheck behaviour (must_contain_any / must_not_contain)
# ---------------------------------------------------------------------------


class TestArtifactCheckExtensions:
    def test_must_contain_all(self, tmp_path):
        (tmp_path / "f.py").write_text("def a():\n    return 1\n", encoding="utf-8")
        ok, _ = ArtifactCheck(path="f.py", must_contain=("def a(", "return")).evaluate(tmp_path)
        assert ok is True
        bad, detail = ArtifactCheck(path="f.py", must_contain=("def b(",)).evaluate(tmp_path)
        assert bad is False and "missing required substring" in detail

    def test_must_contain_any(self, tmp_path):
        (tmp_path / "f.py").write_text("total = sum(x)\n", encoding="utf-8")
        ok, _ = ArtifactCheck(
            path="f.py", must_contain_any=("sum(", "reduce(")
        ).evaluate(tmp_path)
        assert ok is True
        bad, detail = ArtifactCheck(
            path="f.py", must_contain_any=("reduce(", "numpy")
        ).evaluate(tmp_path)
        assert bad is False and "none of the alternatives" in detail

    def test_must_not_contain(self, tmp_path):
        (tmp_path / "f.py").write_text("return sum(range(n + 1))\n", encoding="utf-8")
        ok, _ = ArtifactCheck(
            path="f.py", must_not_contain=("range(n))",)
        ).evaluate(tmp_path)
        assert ok is True
        bad, detail = ArtifactCheck(
            path="f.py", must_not_contain=("sum(",)
        ).evaluate(tmp_path)
        assert bad is False and "forbidden substring" in detail

    def test_missing_artifact(self, tmp_path):
        bad, detail = ArtifactCheck(path="ghost.py", must_contain=("x",)).evaluate(tmp_path)
        assert bad is False and "not found" in detail

    def test_roundtrip_serialisation_includes_new_fields(self):
        c = ArtifactCheck(
            path="f.py",
            must_contain=("a",),
            must_contain_any=("b", "c"),
            must_not_contain=("d",),
        )
        again = ArtifactCheck.from_dict(json.loads(json.dumps(c.to_dict())))
        assert again == c


# ---------------------------------------------------------------------------
# EvalTask fixture round-trip + safety
# ---------------------------------------------------------------------------


class TestEvalTaskFixtures:
    def test_to_from_dict_roundtrip_with_fixtures(self):
        t = EvalTask(
            task_id="x",
            prompt="p",
            checks=(ArtifactCheck(path="a.py", must_contain=("z",)),),
            fixtures=(FixtureFile(path="starter.py", content="X = 1\n"),),
            metadata={"category": "bug_fix"},
        )
        again = EvalTask.from_dict(json.loads(json.dumps(t.to_dict())))
        assert again == t

    def test_from_dict_rejects_dir_form(self):
        with pytest.raises(ValueError, match="load the task"):
            EvalTask.from_dict({"task_id": "x", "prompt": "p", "fixtures": {"dir": "y"}})

    def test_unsafe_fixture_path_rejected(self):
        with pytest.raises(ValueError, match="unsafe fixture path"):
            EvalTask(
                task_id="x",
                prompt="p",
                fixtures=(FixtureFile(path="/etc/passwd", content="x"),),
            )

    def test_provision_creates_nested_dirs(self, tmp_path):
        t = EvalTask(
            task_id="x",
            prompt="p",
            fixtures=(FixtureFile(path="pkg/mod.py", content="A = 1\n"),),
        )
        t.provision(tmp_path)
        assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == "A = 1\n"


# ---------------------------------------------------------------------------
# Compatibility with the existing (Step 3) EvaluationRunner
# ---------------------------------------------------------------------------


class ScriptedProvider(LLMProvider):
    def __init__(self, responses, *, model="fake-model", name="scripted"):
        self._responses = list(responses)
        self._model, self._name = model, name
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


_SETTINGS = SimpleNamespace(
    llm_provider="gemini",
    llm_model="gemini-3.6-flash",
    llm_temperature=0.0,
    llm_max_output_tokens=4096,
    max_llm_turns=30,
    max_tool_calls=50,
)


class TestRunnerCompatibility:
    def test_loaded_task_runs_through_evaluation_runner(self, tmp_path):
        """fix-inclusive-sum: fixture provisioned, agent edits it, checks pass."""
        task = get_task("fix-inclusive-sum")
        fixed = (
            "def inclusive_sum(n):\n"
            '    """Return the sum of all integers from 1 to n inclusive.\n\n'
            "    For example, inclusive_sum(5) should be 1 + 2 + 3 + 4 + 5 == 15.\n"
            '    """\n'
            "    return sum(range(n + 1))\n"
        )
        provider = ScriptedProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "ranges.py", "content": fixed, "overwrite": True},
                        )
                    ],
                    stop_reason="tool_use",
                ),
                LLMResponse(content="Fixed the off-by-one.", stop_reason="stop"),
            ]
        )
        runner = EvaluationRunner(
            settings=_SETTINGS, provider=provider, trace_dir=tmp_path / "traces"
        )
        experiment = ExperimentConfig.from_settings(_SETTINGS, task_id=task.task_id)
        ws = tmp_path / "ws"
        result = runner.run(task, experiment, workspace=ws)

        # fixture was provisioned before the agent ran
        assert (ws / "ranges.py").exists()
        assert result.task_id == "fix-inclusive-sum"
        assert result.status == "completed"
        assert result.success is True
        assert result.checks_run == 1
        assert result.checks_passed is True
        assert result.tool_strategy == "fixed"
        assert result.context_strategy == "raw"

    def test_task_identity_has_no_strategy_metadata(self):
        task = get_task("create-string-utils")
        d = task.to_dict()
        assert set(d) == {"task_id", "prompt", "checks", "fixtures", "metadata"}
        for banned in ("tool_strategy", "context_strategy", "provider", "model", "run_id"):
            assert banned not in d
            assert banned not in d["metadata"]

    def test_same_task_two_configs_same_identity(self):
        task = get_task("create-string-utils")
        a = ExperimentConfig.from_settings(_SETTINGS, task_id=task.task_id)
        b = ExperimentConfig.from_settings(
            _SETTINGS, task_id=task.task_id,
            tool_strategy="adaptive", context_strategy="managed",
        )
        assert a.task_id == b.task_id == task.task_id
        assert a.strategy_key != b.strategy_key
