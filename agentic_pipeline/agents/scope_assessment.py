from __future__ import annotations

from agentic_pipeline.agents.base import BaseAgent


class ScopeAssessmentAgent(BaseAgent):
    """Stage 1: exhaustively maps every system/file/contract a request concerns.

    Read-only (tools: read, search). Output stays plain text -- nothing in
    the pipeline branches on it, later stages just receive it as context.
    """

    name = "scope-assessment"

    def run(self, specification: str) -> str:
        result = self._run(f"Specification:\n{specification}")
        return result.text
