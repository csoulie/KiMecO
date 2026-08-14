"""The Python orchestrator: drives the fixed 7-stage KiMecO agentic pipeline,
owns all direct user interaction, and carries decision state forward between
otherwise-stateless agent calls.

Mirrors ``workflow-orchestrator.agent.md``: it does not design plans, tests,
or domain logic itself, and its only writes are the narrow fallback of
applying a specialist's already-confirmed edit when that specialist could
not persist it, plus verifying every write actually landed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from agentic_pipeline.agents import (
    BoundariesAgent,
    CiTestAgent,
    ClarificationAgent,
    PlanningAgent,
    ScopeAssessmentAgent,
    SpecReviewAgent,
    VersionControlAgent,
)
from agentic_pipeline.config import MAX_CLARIFICATION_ROUNDS
from agentic_pipeline.logging_config import RunRecorder, configure_logging, logger
from agentic_pipeline.models import (
    BoundariesResult,
    ImplementationResult,
    PlanResult,
    TestDesignResult,
    TestRunResult,
    VersionControlResult,
)
from agentic_pipeline.tools import ToolError, edit_file, read_file, write_file

STAGES = [
    "1-3: scope assessment / spec review / clarification (looped)",
    "4: safe-boundary definition + validation",
    "5: parallel plan + test design",
    "6: implement + run tests",
    "7: version control",
]


def _default_ask_user(prompt: str) -> str:
    """CLI default for the ask_user callback. Swap for a scripted callable in tests/automation."""
    print("\n" + prompt)
    return input("> ").strip()


@dataclass
class Specification:
    """The running 'specification' object: the original prompt plus every
    clarification gathered so far. Passed to every stage in full, and its
    verbatim_request is passed separately to clarification per the original
    anti-drift rule.
    """

    verbatim_request: str
    clarifications: list[tuple[str, str]] = field(default_factory=list)
    proceed_as_is: bool = False

    def as_text(self) -> str:
        lines = [f"Original request: {self.verbatim_request}"]
        for question, answer in self.clarifications:
            lines.append(f"- Q: {question}\n  A: {answer}")
        if self.proceed_as_is:
            lines.append("(user chose to proceed as-is on remaining open questions -- treat the spec as final)")
        return "\n".join(lines)


class TodoList:
    """Mirrors the six pipeline stages so the user can see progress. Plain
    console output -- not a Claude tool (Copilot's `todo` tool has no
    filesystem/shell effect to gate, so it isn't wired into the tool loop).
    """

    def __init__(self, stages: list[str]) -> None:
        self._stages = stages
        self._current = -1

    def start(self, index: int) -> None:
        self._current = index
        logger.info("=== Stage %d/%d: %s ===", index + 1, len(self._stages), self._stages[index])


class WorkflowOrchestrator:
    def __init__(
        self,
        ask_user: Callable[[str], str] = _default_ask_user,
        recorder: RunRecorder | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        configure_logging(log_level)
        self.ask_user = ask_user
        self.recorder = recorder or RunRecorder()
        self.scope_agent = ScopeAssessmentAgent(recorder=self.recorder)
        self.spec_review_agent = SpecReviewAgent(recorder=self.recorder)
        self.clarification_agent = ClarificationAgent(recorder=self.recorder)
        self.boundaries_agent = BoundariesAgent(recorder=self.recorder)
        self.planning_agent = PlanningAgent(recorder=self.recorder)
        self.ci_test_agent = CiTestAgent(recorder=self.recorder)
        self.version_control_agent = VersionControlAgent(recorder=self.recorder)
        self.todo = TodoList(STAGES)

    # ------------------------------------------------------------------
    # Top-level pipeline -- stages always run in this order.
    # ------------------------------------------------------------------

    def run(self, user_request: str) -> str:
        spec = Specification(verbatim_request=user_request)

        self.todo.start(0)
        systems_report = self._scope_spec_clarify_loop(spec)

        self.todo.start(1)
        boundaries = self._boundaries_stage(spec, systems_report)

        self.todo.start(2)
        plan, test_design = self._plan_and_test_design_stage(spec, systems_report, boundaries)

        self.todo.start(3)
        implementation, test_run = self._implement_and_test_stage(spec, boundaries, plan, test_design)

        self.todo.start(4)
        vc_result = self._version_control_stage(implementation, test_run)

        return self._final_summary(systems_report, implementation, test_run, vc_result)

    # ------------------------------------------------------------------
    # Stages 1-3: scope -> spec-review -> clarify, looped until COMPLETE
    # ------------------------------------------------------------------

    def _scope_spec_clarify_loop(self, spec: Specification) -> str:
        systems_report = ""
        for _ in range(MAX_CLARIFICATION_ROUNDS):
            systems_report = self.scope_agent.run(spec.as_text())
            review = self.spec_review_agent.run(spec.as_text(), systems_report)
            if review.status == "COMPLETE":
                logger.info("spec-review: COMPLETE")
                return systems_report

            logger.info("spec-review: NEEDS_INPUT (%d gaps)", len(review.gaps))
            clar = self.clarification_agent.run(spec.verbatim_request, review.gaps)
            if clar.insufficient_input or not clar.questions:
                logger.warning("clarification returned no questions (%s); proceeding as-is", clar.insufficient_input_reason)
                spec.proceed_as_is = True
                return systems_report

            declined = False
            for question in clar.questions:
                answer = self.ask_user(self._format_question(question))  # HITL gate 1: clarification answers
                if answer.strip().lower() in ("proceed", "proceed as-is", ""):
                    declined = True
                    break
                spec.clarifications.append((question.question, answer))

            if declined:
                spec.proceed_as_is = True
                return systems_report
            # Otherwise: never skip re-assessment after new input -- loop back to Stage 1.

        logger.warning("hit MAX_CLARIFICATION_ROUNDS (%d); proceeding with the specification as accumulated", MAX_CLARIFICATION_ROUNDS)
        spec.proceed_as_is = True
        return systems_report

    @staticmethod
    def _format_question(question) -> str:
        lines = [question.question]
        for opt in question.options:
            marker = " (recommended)" if opt == question.recommended_option else ""
            lines.append(f"  - {opt}{marker}")
        if question.free_text and not question.options:
            lines.append("  (free text)")
        if question.why_it_matters:
            lines.append(f"  why this matters: {question.why_it_matters}")
        lines.append("(type 'proceed' to accept defaults and continue as-is)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 4: boundaries + validation
    # ------------------------------------------------------------------

    def _boundaries_stage(self, spec: Specification, systems_report: str) -> BoundariesResult:
        boundaries = self.boundaries_agent.run(spec.as_text(), systems_report)
        while True:
            prompt = self._format_boundaries(boundaries) + "\n\nApprove these boundaries? (yes / describe changes)"
            answer = self.ask_user(prompt)  # HITL gate 2: boundary validation
            if answer.strip().lower() in ("yes", "y", "approve", "approved", ""):
                return boundaries
            boundaries = self.boundaries_agent.run(spec.as_text(), systems_report, feedback=answer)

    @staticmethod
    def _format_boundaries(b: BoundariesResult) -> str:
        lines = ["Modify zone:"]
        lines += [f"  - {item.target} -- {item.reason}" for item in b.modify_zone] or ["  (none)"]
        lines.append("Freeze zone:")
        lines += [f"  - {item.target} -- {item.reason}" for item in b.freeze_zone] or ["  (none)"]
        if b.contract_edges:
            lines.append("Contract edges:")
            for edge in b.contract_edges:
                side = "inside" if edge.inside_boundary else "outside"
                note = f": {edge.migration_note}" if edge.migration_note else ""
                lines.append(f"  - {edge.contract} ({side}){note}")
        if b.open_questions:
            lines.append("Open boundary questions:")
            lines += [f"  - {q}" for q in b.open_questions]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 5: parallel plan + test design, single confirmation
    # ------------------------------------------------------------------

    def _plan_and_test_design_stage(
        self, spec: Specification, systems_report: str, boundaries: BoundariesResult
    ) -> tuple[PlanResult, TestDesignResult]:
        raw_pref = self.ask_user(
            "Before test design: do you have a particular test or scenario you want covered? (or 'none')"
        )  # HITL: precedes gate 3, feeds ci-test's design phase
        test_pref = None if raw_pref.strip().lower() in ("none", "no", "") else raw_pref.strip()

        plan, test_design = self._design_plan_and_tests(spec, systems_report, boundaries, test_pref)

        while True:
            prompt = self._format_plan_and_tests(plan, test_design) + "\n\nConfirm this plan + test design? (yes / describe changes)"
            answer = self.ask_user(prompt)  # HITL gate 3: plan/test confirmation
            if answer.strip().lower() in ("yes", "y", "confirm", "confirmed", ""):
                return plan, test_design
            plan, test_design = self._design_plan_and_tests(spec, systems_report, boundaries, test_pref, feedback=answer)

    def _design_plan_and_tests(
        self,
        spec: Specification,
        systems_report: str,
        boundaries: BoundariesResult,
        test_pref: str | None,
        feedback: str | None = None,
    ) -> tuple[PlanResult, TestDesignResult]:
        # Dispatched in parallel: both work from the same spec + boundaries and neither
        # consumes the other's output, matching "dispatch planning and ci-test in parallel".
        with ThreadPoolExecutor(max_workers=2) as pool:
            plan_future = pool.submit(self.planning_agent.design_plan, spec.as_text(), systems_report, boundaries, feedback)
            test_future = pool.submit(
                self.ci_test_agent.design_tests, spec.as_text(), systems_report, boundaries, test_pref, feedback
            )
            return plan_future.result(), test_future.result()

    @staticmethod
    def _format_plan_and_tests(plan: PlanResult, test_design: TestDesignResult) -> str:
        lines = ["Plan:"]
        lines += [f"  {i}. [{s.system}] {s.change} -- {s.reason}" for i, s in enumerate(plan.steps, start=1)]
        if plan.contract_changes:
            lines.append("Contract changes:")
            lines += [f"  - {c}" for c in plan.contract_changes]
        lines.append("Proposed tests:")
        lines += [f"  - ({c.category}) {c.name}: {c.checks}" for c in test_design.cases]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage 6: implement, verify persistence, then author + run tests
    # ------------------------------------------------------------------

    def _implement_and_test_stage(
        self,
        spec: Specification,
        boundaries: BoundariesResult,
        plan: PlanResult,
        test_design: TestDesignResult,
    ) -> tuple[ImplementationResult, TestRunResult]:
        decision_context = (
            "User has already approved the plan and boundaries above; do not re-ask. "
            "Ignore unrelated pre-existing git/working-tree changes."
        )
        implementation = self.planning_agent.implement(spec.as_text(), boundaries, plan, decision_context)
        self._verify_persistence(implementation)

        test_run = self.ci_test_agent.author_and_run(implementation, test_design)
        self._verify_test_files_exist(test_run)
        return implementation, test_run

    def _verify_persistence(self, implementation: ImplementationResult) -> None:
        """Re-reads every file `planning` claims to have changed. A subagent's
        "done" is unconfirmed until the change is seen on disk; if it did not
        persist, apply the agent's own exact designed edit here as the
        orchestrator's only write.
        """
        for change in implementation.files_changed:
            if change.persisted:
                try:
                    read_file(change.path)
                except ToolError:
                    logger.warning("planning reported %s as persisted but it is unreadable", change.path)
                continue
            logger.warning("planning could not persist %s; applying its designed edit as the orchestrator fallback", change.path)
            if change.is_new_file and change.new_string is not None:
                write_file(change.path, change.new_string)
                change.persisted = True
            elif change.old_string is not None and change.new_string is not None:
                edit_file(change.path, change.old_string, change.new_string)
                change.persisted = True
            else:
                logger.error("no fallback edit available for unpersisted change to %s", change.path)

    def _verify_test_files_exist(self, test_run: TestRunResult) -> None:
        for path in test_run.files:
            try:
                read_file(path)
            except ToolError:
                logger.warning("ci-test reported %s as written but it is unreadable", path)

    # ------------------------------------------------------------------
    # Stage 7: version control
    # ------------------------------------------------------------------

    def _version_control_stage(self, implementation: ImplementationResult, test_run: TestRunResult) -> VersionControlResult:
        change_summary = (
            f"Implementation summary: {implementation.summary}\n"
            f"Files changed: {[c.path for c in implementation.files_changed]}\n"
            f"Test result: {'passed' if test_run.passed else 'FAILED'} ({test_run.pytest_summary})\n"
            f"Test files: {test_run.files}"
        )
        vc_result = self.version_control_agent.record_change(change_summary)

        release_answer = self.ask_user(
            "Cut a release for this change? Enter a SemVer version (e.g. 1.2.0), or leave blank to skip."
        )
        version = release_answer.strip()
        if version:
            confirm = self.ask_user(f"Confirm release v{version}? (yes/no)")  # HITL gate 4: release approval
            if confirm.strip().lower() in ("yes", "y"):
                vc_result = self.version_control_agent.cut_release(version, approved=True)
        return vc_result

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    @staticmethod
    def _final_summary(
        systems_report: str,
        implementation: ImplementationResult,
        test_run: TestRunResult,
        vc_result: VersionControlResult,
    ) -> str:
        lines = [
            "## Pipeline complete",
            "",
            "### Systems touched",
            systems_report.strip() or "(none)",
            "",
            "### Implemented",
            implementation.summary or "(no summary)",
            *(f"  - {c.path}: {c.summary}" for c in implementation.files_changed),
            "",
            "### Tests",
            f"{'PASSED' if test_run.passed else 'FAILED'}: {test_run.pytest_summary}",
            *(f"  - {f}" for f in test_run.files),
            "",
            "### Docs / changelog",
            vc_result.changelog_entry or "(no entry)",
            *(f"  - {d}" for d in vc_result.docs_updated),
        ]
        if vc_result.release_created:
            lines.append(f"Release cut: v{vc_result.release_version}")
        return "\n".join(lines)
