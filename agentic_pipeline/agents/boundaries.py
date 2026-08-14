from __future__ import annotations

from agentic_pipeline.agents.base import BaseAgent
from agentic_pipeline.models import BoundariesResult


class BoundariesAgent(BaseAgent):
    """Stage 4: read-only. Defines the modify zone and freeze zone for a
    finalized specification, and the contract edges crossed.
    """

    name = "boundaries"

    def run(self, specification: str, systems_report: str, feedback: str | None = None) -> BoundariesResult:
        message = f"Finalized specification:\n{specification}\n\nStage 1 systems report:\n{systems_report}"
        if feedback:
            message += f"\n\nThe user requested these adjustments to a previous boundaries proposal:\n{feedback}"
        result = self._run(message)
        return self._extract(
            result.text,
            BoundariesResult,
            "This is a safe-boundaries report. Extract the modify zone (each item with a one-line "
            "reason), the freeze zone (each item with a one-line reason), the contract edges (each "
            "marked inside/outside the boundary with any migration note), and the boundary "
            "questions the user must confirm before planning.",
        )
