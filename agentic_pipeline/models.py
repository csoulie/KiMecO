"""Pydantic models for the machine-readable state passed between pipeline
stages.

Each agent's *textual* output (matching its original ``## Output Format``
section) is always preserved and shown to the user verbatim. These models
exist only for the small subset of fields the orchestrator must branch on
or forward programmatically (spec-review's COMPLETE/NEEDS_INPUT, boundary
zones, the plan, test cases, ...). They are populated by a dedicated
structured-extraction call (see ``claude_client.extract_structured``) run
against an agent's finished prose -- never by parsing JSON the agent was
asked to embed in its own response.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Stage 2: spec-review
# --------------------------------------------------------------------------


class Gap(BaseModel):
    system: str = Field(description="The concerned system this gap belongs to.")
    ambiguity: str = Field(description="The specific missing decision or ambiguity.")
    blocks_because: str = Field(description="Why this blocks unambiguous implementation.")


class SpecReviewResult(BaseModel):
    status: Literal["COMPLETE", "NEEDS_INPUT"]
    gaps: list[Gap] = Field(default_factory=list)
    settled_interpretation: str = Field(
        default="", description="The one-paragraph confirmation of settled interpretation, when COMPLETE."
    )


# --------------------------------------------------------------------------
# Stage 3: clarification
# --------------------------------------------------------------------------


class ClarificationQuestion(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)
    recommended_option: str | None = None
    free_text: bool = Field(default=False, description="True if the answer space is unbounded.")
    why_it_matters: str = ""
    resolves_gap: str = Field(default="", description="Which Stage 2 gap this question resolves.")


class ClarificationResult(BaseModel):
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    insufficient_input: bool = False
    insufficient_input_reason: str = ""


# --------------------------------------------------------------------------
# Stage 4: boundaries
# --------------------------------------------------------------------------


class BoundaryItem(BaseModel):
    target: str = Field(description="A file, function, class, schema, setting, or GUI section.")
    reason: str


class ContractEdge(BaseModel):
    contract: str
    inside_boundary: bool
    migration_note: str = ""


class BoundariesResult(BaseModel):
    modify_zone: list[BoundaryItem] = Field(default_factory=list)
    freeze_zone: list[BoundaryItem] = Field(default_factory=list)
    contract_edges: list[ContractEdge] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 5a / 6a: planning
# --------------------------------------------------------------------------


class PlanStep(BaseModel):
    system: str
    change: str
    reason: str


class PlanResult(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    contract_changes: list[str] = Field(default_factory=list)
    tests_to_cover: list[str] = Field(default_factory=list)


class FileChange(BaseModel):
    path: str
    summary: str
    persisted: bool = Field(description="Whether the agent confirmed this change landed on disk by re-reading it.")
    is_new_file: bool = False
    # Populated only when persisted=False, so the orchestrator's narrow
    # fallback can apply the exact designed edit itself.
    old_string: str | None = None
    new_string: str | None = None


class ImplementationResult(BaseModel):
    files_changed: list[FileChange] = Field(default_factory=list)
    summary: str = ""
    followups: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 5b / 6c: ci-test
# --------------------------------------------------------------------------


class TestCase(BaseModel):
    name: str
    checks: str
    rationale: str
    category: Literal["standard", "edge", "user_requested"]


class TestDesignResult(BaseModel):
    user_requested_test: str | None = None
    cases: list[TestCase] = Field(default_factory=list)
    coverage_rationale: str = ""


class TestRunResult(BaseModel):
    files: list[str] = Field(default_factory=list)
    added_cases: list[str] = Field(default_factory=list)
    dropped_cases: list[str] = Field(default_factory=list)
    pytest_summary: str = ""
    passed: bool = False


# --------------------------------------------------------------------------
# Stage 7: version-control
# --------------------------------------------------------------------------


class VersionControlResult(BaseModel):
    changelog_entry: str = ""
    docs_updated: list[str] = Field(default_factory=list)
    docs_unaffected_reason: str | None = None
    release_version: str | None = None
    release_created: bool = False
