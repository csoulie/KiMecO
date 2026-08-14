"""Base class shared by every pipeline agent.

Loads the agent's real instructions verbatim from ``.github/agents/<name>.agent.md``
and restricts its tool access to exactly what that file's front matter
declares. Subclasses add typed, per-stage methods -- they never touch the
prompt text itself.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from agentic_pipeline.claude_client import AgentRunResult, extract_structured, run_agent
from agentic_pipeline.config import AGENT_MODEL, EXTRACTION_MODEL
from agentic_pipeline.logging_config import RunRecorder, logger
from agentic_pipeline.prompt_loader import AgentDefinition, load_agent_definition
from agentic_pipeline.tools import build_tool_registry

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    name: str  # must match the stem of .github/agents/<name>.agent.md

    def __init__(self, model: str = AGENT_MODEL, recorder: RunRecorder | None = None) -> None:
        if not getattr(self, "name", None):
            raise TypeError(f"{type(self).__name__} must set a class-level `name`")
        self.definition: AgentDefinition = load_agent_definition(self.name)
        self.tool_schemas, self.tool_dispatch = build_tool_registry(self.definition.tools)
        self.model = model
        self.recorder = recorder

    def _run(self, user_message: str) -> AgentRunResult:
        logger.info("dispatching %s (tools=%s)", self.name, self.definition.tools)
        result = run_agent(
            system_prompt=self.definition.system_prompt,
            user_message=user_message,
            tool_schemas=self.tool_schemas,
            tool_dispatch=self.tool_dispatch,
            model=self.model,
        )
        if self.recorder is not None:
            self.recorder.record(self.name, user_message, result.tool_trace, result.text)
        logger.info("%s finished (%d tool calls)", self.name, len(result.tool_trace))
        return result

    def _extract(self, text: str, schema: type[T], instruction: str) -> T:
        return extract_structured(source_text=text, schema=schema, instruction=instruction, model=EXTRACTION_MODEL)
