from __future__ import annotations

from agentic_pipeline.agents.base import BaseAgent
from agentic_pipeline.models import BoundariesResult, ImplementationResult, TestDesignResult, TestRunResult


class CiTestAgent(BaseAgent):
    """Stage 5b/6c: one agent, two phases.

    Phase 1 (``design_tests``) designs standard + edge-case tests from the
    specification and validated boundaries, in parallel with Planning's plan
    design -- neither consumes the other's output. Phase 2
    (``author_and_run``) authors and runs the tests after implementation
    lands, reconciling with the actual diff.
    """

    name = "ci-test"

    def design_tests(
        self,
        specification: str,
        systems_report: str,
        boundaries: BoundariesResult,
        user_test_preference: str | None,
        feedback: str | None = None,
    ) -> TestDesignResult:
        message = (
            f"Finalized specification:\n{specification}\n\n"
            f"Stage 1 systems report:\n{systems_report}\n\n"
            f"User-validated boundaries:\n{boundaries.model_dump_json(indent=2)}\n\n"
            f"User's test preference: {user_test_preference or 'none specified'}\n\n"
            "This is Phase 1 (design). Do not write or run tests yet -- design against the "
            "intended behavior, grounded in the boundaries' modify zone."
        )
        if feedback:
            message += f"\n\nThe user requested these changes to a previous test design:\n{feedback}"
        result = self._run(message)
        return self._extract(
            result.text,
            TestDesignResult,
            "This is a proposed test-design report (Phase 1). Extract the user-requested test if "
            "any, the full list of proposed standard and edge cases (each with what it checks, "
            "why, and its category), and the coverage rationale.",
        )

    def author_and_run(self, implementation: ImplementationResult, confirmed_design: TestDesignResult) -> TestRunResult:
        message = (
            f"Actual implementation diff:\n{implementation.model_dump_json(indent=2)}\n\n"
            f"Confirmed test design from Phase 1:\n{confirmed_design.model_dump_json(indent=2)}\n\n"
            "This is Phase 2 (author + run). Reconcile the designed cases against the actual "
            "diff, author the tests following tests/unit/ conventions (test_<subject>_ci.py, "
            "pytest, independent of the MESS binary), run `pytest tests/unit/ -q`, and iterate "
            "until green. After writing each test file, re-read it to confirm it landed."
        )
        result = self._run(message)
        return self._extract(
            result.text,
            TestRunResult,
            "This is a test-authoring report (Phase 2). Extract the list of new/updated test file "
            "paths, any cases added or dropped after reconciling with the diff, the pytest summary "
            "text, and whether the final pytest run passed.",
        )
