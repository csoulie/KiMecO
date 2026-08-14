from __future__ import annotations

from agentic_pipeline.agents.base import BaseAgent
from agentic_pipeline.models import BoundariesResult, ImplementationResult, PlanResult


class PlanningAgent(BaseAgent):
    """Stage 5a/6a: one agent, two phases.

    Phase A (``design_plan``) produces a file-level plan confined to the
    validated modify zone and stops -- no edits. Phase B (``implement``)
    executes the confirmed plan only after the orchestrator relays explicit
    user confirmation.
    """

    name = "planning"

    def design_plan(
        self,
        specification: str,
        systems_report: str,
        boundaries: BoundariesResult,
        feedback: str | None = None,
    ) -> PlanResult:
        message = (
            f"Finalized specification:\n{specification}\n\n"
            f"Stage 1 systems report:\n{systems_report}\n\n"
            f"User-validated boundaries (modify zone / freeze zone / contract edges):\n"
            f"{boundaries.model_dump_json(indent=2)}\n\n"
            "This is Phase A. Produce the plan and stop -- do not edit any file yet."
        )
        if feedback:
            message += f"\n\nThe user requested these changes to a previous plan:\n{feedback}"
        result = self._run(message)
        return self._extract(
            result.text,
            PlanResult,
            "This is an implementation-plan report (Phase A -- no code has been written yet). "
            "Extract the ordered file-level steps (system, change, reason), any contract-change "
            "callouts, and the behaviors Stage 6 tests should cover.",
        )

    def implement(
        self,
        specification: str,
        boundaries: BoundariesResult,
        confirmed_plan: PlanResult,
        decision_context: str,
    ) -> ImplementationResult:
        message = (
            f"Finalized specification:\n{specification}\n\n"
            f"User-validated boundaries:\n{boundaries.model_dump_json(indent=2)}\n\n"
            f"Confirmed plan from Phase A:\n{confirmed_plan.model_dump_json(indent=2)}\n\n"
            f"Carried decision context (approvals already obtained -- do not re-ask): {decision_context}\n\n"
            "This is Phase B. The user has confirmed the plan above. Implement it now, strictly "
            "inside the validated modify zone. After each edit, re-read the file to confirm it "
            "landed before reporting it as done."
        )
        result = self._run(message)
        return self._extract(
            result.text,
            ImplementationResult,
            "This is an implementation report (Phase B). For every file touched, extract: its "
            "path, a one-line summary of the change, whether it is a new file, and whether the "
            "agent confirmed the change persisted on disk by re-reading it. Only when persisted is "
            "false, also extract the exact old_string/new_string the agent intended, so the "
            "orchestrator can apply the edit itself. Also extract the overall summary and any "
            "follow-ups relevant to Stage 6 (tests) and Stage 7 (docs/changelog).",
        )
