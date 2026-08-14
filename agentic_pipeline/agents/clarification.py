from __future__ import annotations

from agentic_pipeline.agents.base import BaseAgent
from agentic_pipeline.models import ClarificationResult, Gap


class ClarificationAgent(BaseAgent):
    """Stage 3: turns Stage 2's gaps into a minimal set of decision-ready
    questions, grounded strictly in the verbatim user request.
    """

    name = "clarification"

    def run(self, verbatim_request: str, gaps: list[Gap]) -> ClarificationResult:
        gaps_text = "\n".join(f"- [{g.system}] {g.ambiguity} (blocks: {g.blocks_because})" for g in gaps)
        result = self._run(f"Verbatim user request:\n{verbatim_request}\n\nStage 2 gaps:\n{gaps_text}")
        return self._extract(
            result.text,
            ClarificationResult,
            "This is a clarification-questions report. Extract the numbered questions, each with "
            "its concrete options (mark the recommended one if any, or set free_text=true when the "
            "answer is genuinely unbounded), why it matters, and which Stage 2 gap it resolves. If "
            "the report states 'insufficient input', set insufficient_input=true and capture why.",
        )
