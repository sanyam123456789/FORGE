"""
forge.cli — minimal command-line entry point.

    forge run --task "Create add.py with an add(a, b) function"
    forge tasks
    forge evaluate --suite-task-id create-string-utils
    forge experiment --suite-dir experiments/tasks

`run` executes the agent once and prints a summary.  `tasks` lists the
version-controlled baseline task suite (``experiments/tasks/``).  `evaluate`
wraps the same runtime in the measurement layer (forge.evaluation): it runs
one task under one controlled strategy configuration and appends a structured
``EvalResult`` row to a JSONL file for later comparison.  `experiment` runs
the formal 2x2 controlled experiment (Step 7): every selected task under all
four arms (fixed_raw, fixed_managed, adaptive_raw, adaptive_managed), each in
its own isolated workspace, writing one JSONL results file plus a metadata
file per experiment run.

run options:
    --task TEXT              the coding task (required)
    --workspace PATH         workspace root (default: current directory)
    --model NAME             override FORGE_LLM_MODEL
    --max-turns N            override max LLM turns
    --max-tool-calls N       override max tool calls
    --temperature FLOAT      override sampling temperature
    --no-trace              do not write a JSONL run trace
    --json                  print the run summary as JSON
    --quiet                 only print the final answer

tasks options:
    --suite-dir PATH         task suite dir (default: experiments/tasks)
    --json                   print the suite as JSON

evaluate options:
    --task TEXT / --task-file PATH / --suite-task-id ID   (choose one)
    --task-id ID             override the stable task identity
    --suite-dir PATH         suite dir for --suite-task-id
    --tool-strategy NAME     fixed | adaptive (both implemented)
    --context-strategy NAME  raw | managed (both implemented)
    --workspace PATH         reuse a workspace (default: temp dir, cleaned)
    --results-file PATH      JSONL to append the result to
    --model / --max-turns / --max-tool-calls / --temperature / --no-trace / --json

experiment options:
    --suite-dir PATH         task suite dir (default: experiments/tasks)
    --task-id ID             restrict to this task (repeatable; default: full suite)
    --arm ARM_ID             restrict to this arm (repeatable; default: all four —
                             fixed_raw, fixed_managed, adaptive_raw, adaptive_managed)
    --output-dir PATH        where to write <experiment_id>/ (default: runs/experiments)
    --experiment-id ID       override the generated experiment id
    --label TEXT             optional batch label recorded with every result
    --model / --max-turns / --max-tool-calls / --temperature / --no-trace / --json

    NOTE: with no provider override this calls the real configured LLM
    provider (Gemini by default) once per task-arm execution — a full 4-arm
    run against the baseline suite is one real API call sequence per cell
    and will consume real quota. Use a small --task-id/--arm subset for a
    quick check; see docs/step-07-controlled-2x2-experiment.md for how to
    drive it with a scripted/fake provider instead (as the test suite does).

Exit codes:
    0  run completed (for `experiment`: the batch ran to completion — this
       does not mean every individual task-arm succeeded, only that the
       experiment itself did not fail to run; inspect the written results
       for per-task-arm outcomes)
    1  run stopped without completing (limit / provider error / failure) —
       `evaluate` only
    2  configuration problem (e.g. no API key, unimplemented strategy,
       unknown task/arm id)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge", description="FORGE coding-agent harness")
    parser.add_argument("--version", action="version", version=f"forge {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the agent on a task")
    run.add_argument("--task", required=True, help="the coding task, in natural language")
    run.add_argument("--workspace", default=None, help="workspace root (default: cwd)")
    run.add_argument("--model", default=None, help="override the model name")
    run.add_argument("--max-turns", type=int, default=None, help="max LLM turns")
    run.add_argument("--max-tool-calls", type=int, default=None, help="max tool calls")
    run.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    run.add_argument("--no-trace", action="store_true", help="do not write a JSONL trace")
    run.add_argument("--json", action="store_true", help="print the run summary as JSON")
    run.add_argument("--quiet", action="store_true", help="only print the final answer")

    tk = sub.add_parser("tasks", help="list the baseline evaluation task suite")
    tk.add_argument(
        "--suite-dir", default=None,
        help="task suite directory (default: experiments/tasks)",
    )
    tk.add_argument("--json", action="store_true", help="print the suite as JSON")

    ev = sub.add_parser("evaluate", help="run one task under a controlled config and record metrics")
    src = ev.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", default=None, help="inline task prompt")
    src.add_argument("--task-file", default=None, help="path to a task JSON file")
    src.add_argument(
        "--suite-task-id", default=None,
        help="run a task from the baseline suite by its task_id",
    )
    ev.add_argument(
        "--suite-dir", default=None,
        help="task suite directory for --suite-task-id (default: experiments/tasks)",
    )
    ev.add_argument("--task-id", default=None, help="override the stable task id (default: keep as defined)")
    ev.add_argument(
        "--tool-strategy", default="fixed",
        help="tool exposure strategy (implemented: fixed, adaptive)",
    )
    ev.add_argument(
        "--context-strategy", default="raw",
        help="context strategy (implemented: raw, managed)",
    )
    ev.add_argument("--workspace", default=None, help="reuse this workspace (default: temp dir, cleaned)")
    ev.add_argument("--model", default=None, help="override the model name")
    ev.add_argument("--max-turns", type=int, default=None, help="max LLM turns")
    ev.add_argument("--max-tool-calls", type=int, default=None, help="max tool calls")
    ev.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    ev.add_argument(
        "--results-file", default=None,
        help="JSONL file to append the EvalResult to (default: runs/eval_results.jsonl)",
    )
    ev.add_argument("--label", default=None, help="optional batch label recorded with the result")
    ev.add_argument("--no-trace", action="store_true", help="do not write a JSONL run trace")
    ev.add_argument("--json", action="store_true", help="print the EvalResult as JSON")

    exp = sub.add_parser(
        "experiment",
        help="run the formal 2x2 controlled experiment (fixed/adaptive x raw/managed)",
    )
    exp.add_argument(
        "--suite-dir", default=None,
        help="task suite directory (default: experiments/tasks)",
    )
    exp.add_argument(
        "--task-id", dest="task_ids", action="append", default=None,
        help="restrict to this task_id (repeatable; default: the full suite)",
    )
    exp.add_argument(
        "--arm", dest="arm_ids", action="append", default=None,
        choices=["fixed_raw", "fixed_managed", "adaptive_raw", "adaptive_managed"],
        help="restrict to this arm (repeatable; default: all four)",
    )
    exp.add_argument(
        "--output-dir", default=None,
        help="where to write <experiment_id>/ (default: runs/experiments)",
    )
    exp.add_argument("--experiment-id", default=None, help="override the generated experiment id")
    exp.add_argument("--label", default=None, help="optional batch label recorded with every result")
    exp.add_argument("--model", default=None, help="override the model name")
    exp.add_argument("--max-turns", type=int, default=None, help="max LLM turns")
    exp.add_argument("--max-tool-calls", type=int, default=None, help="max tool calls")
    exp.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    exp.add_argument("--no-trace", action="store_true", help="do not write per-run JSONL traces")
    exp.add_argument("--json", action="store_true", help="print the experiment summary as JSON")
    return parser


def _run_command(args: argparse.Namespace) -> int:
    # Imports are deferred so that `forge --help` / `forge --version` do not
    # need the provider SDK or a configured environment.
    import os

    from forge.agent import STATUS_COMPLETED, AgentConfig, AgentRuntime
    from forge.config import load_settings
    from forge.llm import get_provider
    from forge.logging import setup_logging

    if args.model:
        os.environ["FORGE_LLM_MODEL"] = args.model
    if args.workspace:
        os.environ["FORGE_WORKSPACE_ROOT"] = str(Path(args.workspace).resolve())

    settings = load_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    try:
        provider = get_provider(settings_override=settings)
        provider.preflight()
    except (ImportError, NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet and not args.json:
        print(f"FORGE {__version__}  provider={settings.llm_provider} model={settings.llm_model}")
        print(f"workspace: {settings.workspace_root}")
        print(f"task: {args.task}\n")

    config = AgentConfig(
        task=args.task,
        max_llm_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        temperature=args.temperature,
    )
    runtime = AgentRuntime(
        config,
        provider=provider,
        workspace_root=settings.workspace_root,
        write_trace=not args.no_trace,
        log_dir=settings.log_dir,
    )

    run = runtime.run()

    summary = {
        "run_id": run.run_id,
        "status": run.status,
        "llm_calls": run.llm_calls,
        "tool_calls": run.tool_calls,
        "turns": run.turns,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "total_tokens": run.total_tokens,
        "cached_input_tokens": run.cached_input_tokens,
        "reasoning_tokens": run.reasoning_tokens,
        "llm_latency_ms": round(run.llm_latency_ms, 1) if run.llm_latency_ms else None,
        "context_messages": run.context_messages,
        "context_char_count": run.context_char_count,
        "trace_path": str(run.trace_path) if run.trace_path else None,
        "error": run.error,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    elif args.quiet:
        print(run.final_answer)
    else:
        print("--- final answer ---")
        print(run.final_answer or "(no final answer)")
        print("\n--- run summary ---")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    return 0 if run.status == STATUS_COMPLETED else 1


def _load_eval_task(args: argparse.Namespace):
    """Build an ``EvalTask`` from --task, --task-file, or --suite-task-id."""
    import dataclasses

    from forge.evaluation import EvalTask
    from forge.evaluation.suite import DEFAULT_TASKS_DIR, get_task, load_task

    if getattr(args, "suite_task_id", None):
        tasks_dir = Path(args.suite_dir) if args.suite_dir else DEFAULT_TASKS_DIR
        task = get_task(args.suite_task_id, tasks_dir=tasks_dir)
        return dataclasses.replace(task, task_id=args.task_id) if args.task_id else task

    if args.task_file:
        task = load_task(Path(args.task_file))
        return dataclasses.replace(task, task_id=args.task_id) if args.task_id else task

    task_id = args.task_id or "adhoc"
    return EvalTask(task_id=task_id, prompt=args.task)


def _tasks_command(args: argparse.Namespace) -> int:
    from forge.evaluation.suite import DEFAULT_TASKS_DIR, TaskSuiteError, load_suite

    tasks_dir = Path(args.suite_dir) if args.suite_dir else DEFAULT_TASKS_DIR
    try:
        tasks = load_suite(tasks_dir)
    except (OSError, TaskSuiteError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([t.to_dict() for t in tasks], indent=2))
        return 0

    print(f"baseline task suite: {tasks_dir}  ({len(tasks)} tasks)\n")
    print(f"  {'task_id':<30} {'category':<14} {'diff':<7} checks fixtures")
    print(f"  {'-' * 30} {'-' * 14} {'-' * 7} {'-' * 6} {'-' * 8}")
    for t in tasks:
        cat = str(t.metadata.get("category", "-"))
        diff = str(t.metadata.get("difficulty", "-"))
        print(
            f"  {t.task_id:<30} {cat:<14} {diff:<7} "
            f"{len(t.checks):<6} {len(t.fixtures)}"
        )
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    import os

    from forge.config import load_settings
    from forge.evaluation import (
        EvaluationRunner,
        ExperimentConfig,
        append_result_jsonl,
    )
    from forge.llm import get_provider
    from forge.logging import setup_logging

    if args.model:
        os.environ["FORGE_LLM_MODEL"] = args.model

    settings = load_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    try:
        task = _load_eval_task(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: could not load task: {exc}", file=sys.stderr)
        return 2

    # Fail fast (exit 2) on an unknown/unimplemented strategy name or a
    # provider/key problem, exactly like `forge run`. Building the
    # ExperimentConfig is included here (not just resolve_strategies())
    # because an unrecognised --tool-strategy/--context-strategy value
    # raises ValueError at construction time, in __post_init__.
    try:
        experiment = ExperimentConfig.from_settings(
            settings,
            task_id=task.task_id,
            tool_strategy=args.tool_strategy,
            context_strategy=args.context_strategy,
            max_llm_turns=args.max_turns,
            max_tool_calls=args.max_tool_calls,
            label=args.label,
        )
        if args.temperature is not None:
            experiment = ExperimentConfig.from_dict(
                {**experiment.to_dict(), "temperature": args.temperature}
            )
        experiment.resolve_strategies()
        provider = get_provider(settings_override=settings)
        provider.preflight()
    except (ImportError, NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results_file = (
        Path(args.results_file)
        if args.results_file
        else Path("runs") / "eval_results.jsonl"
    )
    # Keep each run's JSONL trace next to the results file it belongs to.
    trace_dir = results_file.parent / "traces"

    if not args.json:
        print(
            f"FORGE {__version__}  evaluate  "
            f"provider={settings.llm_provider} model={settings.llm_model}"
        )
        print(f"task_id: {task.task_id}  strategy: {experiment.strategy_key}")
        print(f"results: {results_file}\n")

    runner = EvaluationRunner(
        settings=settings,
        provider=provider,
        trace_dir=trace_dir,
        write_trace=not args.no_trace,
    )
    result = runner.run(task, experiment, workspace=args.workspace)

    append_result_jsonl(result, results_file)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        fields = [
            "task_id", "run_id", "status", "success", "tool_strategy",
            "context_strategy", "llm_calls", "tool_calls", "input_tokens",
            "output_tokens", "total_tokens", "reasoning_tokens",
            "cached_input_tokens", "latency_ms", "cost_usd", "cost_available",
            "context_messages", "context_char_count", "context_items_dropped",
            "context_items_compressed", "context_chars_saved", "checks_run",
            "checks_passed", "duration_s", "trace_path",
        ]
        print("--- eval result ---")
        for key in fields:
            print(f"  {key}: {getattr(result, key)}")
        if result.checks_detail:
            print("  checks_detail:")
            for line in result.checks_detail:
                print(f"    - {line}")

    return 0 if result.success else 1


def _experiment_command(args: argparse.Namespace) -> int:
    import os

    from forge.config import load_settings
    from forge.evaluation import ARMS, MatrixRunner, aggregate_results, arms_by_ids
    from forge.evaluation.suite import DEFAULT_TASKS_DIR, TaskSuiteError, load_suite
    from forge.llm import get_provider
    from forge.logging import setup_logging

    if args.model:
        os.environ["FORGE_LLM_MODEL"] = args.model

    settings = load_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    tasks_dir = Path(args.suite_dir) if args.suite_dir else DEFAULT_TASKS_DIR
    try:
        suite = load_suite(tasks_dir)
    except (OSError, TaskSuiteError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.task_ids:
        wanted = set(args.task_ids)
        known = {t.task_id for t in suite}
        unknown = wanted - known
        if unknown:
            print(
                f"error: unknown task_id(s) {sorted(unknown)}. "
                f"Available: {sorted(known)}",
                file=sys.stderr,
            )
            return 2
        # Preserve the suite's own (filename-sorted) order, not CLI arg order.
        tasks = [t for t in suite if t.task_id in wanted]
    else:
        tasks = suite

    try:
        arms = arms_by_ids(args.arm_ids) if args.arm_ids else ARMS
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Fail fast (exit 2) on a provider/key problem before running anything,
    # exactly like `forge run` / `forge evaluate`.
    try:
        get_provider(settings_override=settings).preflight()
    except (ImportError, NotImplementedError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir) if args.output_dir else Path("runs") / "experiments"

    if not args.json:
        print(f"FORGE {__version__}  experiment  provider={settings.llm_provider} model={settings.llm_model}")
        print(f"tasks: {[t.task_id for t in tasks]}")
        print(f"arms:  {[a.arm_id for a in arms]}")
        print(f"output: {output_root}\n")

    runner = MatrixRunner(
        settings=settings,
        output_root=output_root,
        write_trace=not args.no_trace,
    )
    results, summary = runner.run(
        tasks,
        arms=arms,
        experiment_id=args.experiment_id,
        label=args.label,
    )

    # Group by arm_id (not the raw "<tool>+<context>" key group_results()
    # uses) for a clearer per-arm table — one arm is exactly one such pair,
    # so this is the same aggregation, just keyed the way this CLI names arms.
    by_arm: dict[str, list] = {a.arm_id: [] for a in arms}
    for r in results:
        by_arm.setdefault(r.arm_id, []).append(r)
    aggregates = {arm_id: aggregate_results(rows) for arm_id, rows in by_arm.items()}

    if args.json:
        print(json.dumps(
            {
                "summary": summary.to_dict(),
                "aggregates": {k: v.to_dict() for k, v in aggregates.items()},
            },
            indent=2,
        ))
    else:
        print(f"experiment_id: {summary.experiment_id}")
        print(f"results file:  {summary.results_file}")
        print(f"{len(results)} task-arm result(s)\n")
        print(f"  {'arm':<20} {'tasks':>5} {'success':>7} {'rate':>6}")
        print(f"  {'-' * 20} {'-' * 5} {'-' * 7} {'-' * 6}")
        for arm in arms:
            agg = aggregates[arm.arm_id]
            rate = f"{agg.success_rate:.0%}" if agg.success_rate is not None else "n/a"
            print(f"  {arm.arm_id:<20} {agg.task_count:>5} {agg.successful_tasks:>7} {rate:>6}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "tasks":
        return _tasks_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "experiment":
        return _experiment_command(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
