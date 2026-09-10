"""
forge.cli — minimal command-line entry point.

    forge run --task "Create add.py with an add(a, b) function"
    forge evaluate --task "Create add.py ..." --task-id add-fn

`run` executes the agent once and prints a summary.  `evaluate` wraps the
same runtime in the measurement layer (forge.evaluation): it runs one task
under one controlled strategy configuration and appends a structured
``EvalResult`` row to a JSONL file for later comparison.

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

evaluate options:
    --task TEXT / --task-file PATH   inline prompt, or a task JSON file
    --task-id ID             stable task identity (default: derived)
    --tool-strategy NAME     fixed (only implemented value)
    --context-strategy NAME  raw   (only implemented value)
    --workspace PATH         reuse a workspace (default: temp dir, cleaned)
    --results-file PATH      JSONL to append the result to
    --model / --max-turns / --max-tool-calls / --temperature / --no-trace / --json

Exit codes:
    0  run completed
    1  run stopped without completing (limit / provider error / failure)
    2  configuration problem (e.g. no API key, unimplemented strategy)
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

    ev = sub.add_parser("evaluate", help="run one task under a controlled config and record metrics")
    src = ev.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", default=None, help="inline task prompt")
    src.add_argument("--task-file", default=None, help="path to a task JSON file")
    ev.add_argument("--task-id", default=None, help="stable task id (default: derived)")
    ev.add_argument(
        "--tool-strategy", default="fixed",
        help="tool exposure strategy (implemented: fixed)",
    )
    ev.add_argument(
        "--context-strategy", default="raw",
        help="context strategy (implemented: raw)",
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
    """Build an ``EvalTask`` from --task or --task-file."""
    from forge.evaluation import EvalTask

    if args.task_file:
        data = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
        if args.task_id:
            data = {**data, "task_id": args.task_id}
        return EvalTask.from_dict(data)

    task_id = args.task_id or "adhoc"
    return EvalTask(task_id=task_id, prompt=args.task)


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
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: could not load task: {exc}", file=sys.stderr)
        return 2

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

    # Fail fast (exit 2) on unimplemented strategy or provider/key problems,
    # exactly like `forge run`.
    try:
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
            "context_messages", "context_char_count", "checks_run",
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
