from __future__ import annotations

from agentic_pipeline.agents.base import BaseAgent
from agentic_pipeline.models import SpecReviewResult


class SpecReviewAgent(BaseAgent):
    """Stage 2: read-only gatekeeper deciding COMPLETE vs NEEDS_INPUT."""

    name = "spec-review"

    def run(self, specification: str, systems_report: str) -> SpecReviewResult:
        result = self._run(f"Specification:\n{specification}\n\nStage 1 systems report:\n{systems_report}")
        return self._extract(
            result.text,
            SpecReviewResult,
            "This is a specification-review report. Extract STATUS (COMPLETE or NEEDS_INPUT), "
            "the numbered list of gaps (each with the concerned system, the specific ambiguity, "
            "and why it blocks implementation) if NEEDS_INPUT, and the one-paragraph settled "
            "interpretation if COMPLETE.",
        )
