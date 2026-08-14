"""Loads Copilot agent definitions from ``.github/agents/*.agent.md``.

The `.github/agents/` files remain the single source of truth for every
agent's instructions. This module only *reads* them (front-matter + body) --
it never modifies them and never duplicates their text elsewhere in the
Python codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from agentic_pipeline.config import AGENTS_DIR


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    tools: list[str]
    user_invocable: bool
    system_prompt: str  # the markdown body, verbatim, unmodified


def _split_front_matter(raw: str) -> tuple[str, str]:
    """Splits a ``---\\nYAML\\n---\\nbody`` file into (yaml_text, body)."""
    if not raw.startswith("---"):
        raise ValueError("agent file does not start with YAML front matter")
    _, _, rest = raw.partition("---\n")
    yaml_text, sep, body = rest.partition("\n---")
    if not sep:
        raise ValueError("agent file front matter is not terminated")
    return yaml_text, body.lstrip("\n")


@lru_cache(maxsize=None)
def load_agent_definition(name: str) -> AgentDefinition:
    """Parses ``.github/agents/<name>.agent.md`` into an :class:`AgentDefinition`.

    ``name`` is the agent's front-matter ``name`` field, e.g. ``"spec-review"``.
    """
    path = AGENTS_DIR / f"{name}.agent.md"
    if not path.exists():
        raise FileNotFoundError(f"no agent definition at {path}")
    raw = path.read_text(encoding="utf-8")
    yaml_text, body = _split_front_matter(raw)
    meta = yaml.safe_load(yaml_text) or {}
    return AgentDefinition(
        name=meta.get("name", name),
        description=meta.get("description", ""),
        tools=list(meta.get("tools", [])),
        user_invocable=bool(meta.get("user-invocable", False)),
        system_prompt=body.strip("\n"),
    )
