"""
forge.cli — minimal command-line entry point.

    forge run --task "Create add.py with an add(a, b) function"

Options:
    --task TEXT              the coding task (required)
    --workspace PATH         workspace root (default: current directory)
    --model NAME             override FORGE_LLM_MODEL
    --max-turns N            override max LLM turns
    --max-tool-calls N       override max tool calls
    --temperature FLOAT      override sampling temperature
    --no-trace              do not write a JSONL run trace
    --json                  print the run summary as JSON
    --quiet                 only print the final answer

Exit codes:
    0  run completed
    1  run stopped without completing (limit / provider error / failure)
    2  configuration problem (e.g. no API key)
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
