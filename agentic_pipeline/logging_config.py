"""Logging setup and per-run debug artifacts.

Every agent invocation writes its raw prompt, tool trace, and final text to
``RUNS_DIR/<run_id>/<agent_name>-<n>.md`` so a stage's behavior can be
inspected after the fact without re-running the pipeline.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from agentic_pipeline.config import RUNS_DIR

logger = logging.getLogger("agentic_pipeline")


def configure_logging(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


class RunRecorder:
    """Writes one debug file per agent invocation for a single pipeline run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = RUNS_DIR / self.run_id
        self._counters: dict[str, count] = {}

    def record(self, agent_name: str, user_message: str, tool_trace: list[dict], final_text: str) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        n = next(self._counters.setdefault(agent_name, count(1)))
        path = self.run_dir / f"{agent_name}-{n}.md"
        lines = [
            f"# {agent_name} (invocation {n})",
            "",
            "## Input",
            "```",
            user_message,
            "```",
            "",
            "## Tool calls",
        ]
        if tool_trace:
            for call in tool_trace:
                lines.append(f"- `{call['name']}({call['input']!r})` -> {call['result_preview']!r}")
        else:
            lines.append("(none)")
        lines += ["", "## Final text", "", final_text]
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("Recorded %s run to %s", agent_name, path)
        return path
