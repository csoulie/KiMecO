"""Environment-driven configuration for the agentic pipeline."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".github" / "agents"
RUNS_DIR = Path(os.environ.get("AGENTIC_PIPELINE_RUNS_DIR", REPO_ROOT / "agentic_pipeline_runs"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Reasoning/coding model used by every agent for its real work.
AGENT_MODEL = os.environ.get("AGENTIC_PIPELINE_MODEL", "claude-opus-5")

# Cheap, tool-free model used only to extract structured fields out of an
# agent's prose output for orchestrator control flow (see claude_client.extract_structured).
EXTRACTION_MODEL = os.environ.get("AGENTIC_PIPELINE_EXTRACTION_MODEL", "claude-haiku-4-5")

# Safety caps to prevent a runaway tool-use loop or clarification loop.
MAX_TOOL_ITERATIONS = int(os.environ.get("AGENTIC_PIPELINE_MAX_TOOL_ITERATIONS", "30"))
MAX_CLARIFICATION_ROUNDS = int(os.environ.get("AGENTIC_PIPELINE_MAX_CLARIFICATION_ROUNDS", "5"))

# Commands the `execute` tool is permitted to run (first token of the command).
ALLOWED_EXECUTE_COMMANDS = {"pytest", "git", "python", "python3"}
