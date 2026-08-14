"""Run the full agentic pipeline locally from a single command:

    python -m agentic_pipeline.cli "add a min_temperature keyword to SIM settings"

With no argument, prompts interactively for the request. All human-in-the-loop
gates (clarification answers, boundary validation, plan/test confirmation,
release approval) are collected via stdin.
"""

from __future__ import annotations

import argparse
import sys

from agentic_pipeline.orchestrator import WorkflowOrchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GAME/KiMecO agentic delivery pipeline.")
    parser.add_argument("request", nargs="?", help="The change request. If omitted, you will be prompted.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    request = args.request or input("Describe the change you want: ").strip()
    if not request:
        print("No request provided.", file=sys.stderr)
        return 1

    import logging

    orchestrator = WorkflowOrchestrator(log_level=logging.DEBUG if args.verbose else logging.INFO)
    summary = orchestrator.run(request)
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
