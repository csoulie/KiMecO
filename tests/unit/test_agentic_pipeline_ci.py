from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_pipeline.claude_client import AgentRunResult, run_agent
from agentic_pipeline.models import (
    BoundaryItem,
    BoundariesResult,
    ClarificationQuestion,
    ClarificationResult,
    FileChange,
    Gap,
    ImplementationResult,
    SpecReviewResult,
)
from agentic_pipeline.orchestrator import WorkflowOrchestrator
from agentic_pipeline.prompt_loader import load_agent_definition
from agentic_pipeline.tools import ToolError, build_tool_registry, edit_file, read_file, run_command, write_file


# --------------------------------------------------------------------------
# prompt_loader: .github/agents/*.md remains the single source of truth
# --------------------------------------------------------------------------


def test_load_agent_definition_parses_front_matter_and_verbatim_body() -> None:
    definition = load_agent_definition("spec-review")

    assert definition.name == "spec-review"
    assert definition.user_invocable is False
    assert definition.tools == ["read", "search"]
    assert "specification gatekeeper" in definition.system_prompt
    # The body must not retain the YAML front-matter delimiters.
    assert not definition.system_prompt.startswith("---")
    assert "STATUS: COMPLETE" in definition.system_prompt


def test_load_agent_definition_orchestrator_declares_full_tool_set() -> None:
    definition = load_agent_definition("workflow-orchestrator")

    assert definition.user_invocable is True
    assert set(definition.tools) == {"read", "search", "edit", "execute", "agent", "todo"}


def test_load_agent_definition_missing_agent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_agent_definition("does-not-exist")


# --------------------------------------------------------------------------
# tools: permission-scoped registry -- the structural enforcement mechanism
# --------------------------------------------------------------------------


def test_build_tool_registry_read_only_agent_gets_no_write_tools() -> None:
    schemas, dispatch = build_tool_registry(["read", "search"])

    names = {schema["name"] for schema in schemas}
    assert names == {"read_file", "search_code"}
    assert set(dispatch) == {"read_file", "search_code"}
    assert "edit_file" not in dispatch
    assert "write_file" not in dispatch
    assert "run_command" not in dispatch


def test_build_tool_registry_writer_agent_gets_edit_and_execute() -> None:
    schemas, dispatch = build_tool_registry(["read", "search", "edit", "execute", "todo"])

    names = {schema["name"] for schema in schemas}
    assert names == {"read_file", "search_code", "edit_file", "write_file", "run_command"}
    assert set(dispatch) == names


def test_build_tool_registry_ignores_todo_and_agent() -> None:
    schemas, dispatch = build_tool_registry(["todo", "agent"])

    assert schemas == []
    assert dispatch == {}


# --------------------------------------------------------------------------
# tools: file primitives (persist-and-verify discipline)
# --------------------------------------------------------------------------


def test_write_file_then_edit_file_roundtrip(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("agentic_pipeline.tools.REPO_ROOT", tmp_path)

    write_file("sub/new_file.py", "value = 1\n")
    assert read_file("sub/new_file.py") == "1\tvalue = 1"

    edit_file("sub/new_file.py", "value = 1", "value = 2")
    assert (tmp_path / "sub" / "new_file.py").read_text() == "value = 2\n"


def test_edit_file_rejects_non_unique_old_string(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("agentic_pipeline.tools.REPO_ROOT", tmp_path)
    (tmp_path / "dup.txt").write_text("x\nx\n")

    with pytest.raises(ToolError):
        edit_file("dup.txt", "x", "y")


def test_read_file_rejects_path_escaping_repo_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("agentic_pipeline.tools.REPO_ROOT", tmp_path)

    with pytest.raises(ToolError):
        read_file("../outside.txt")


def test_run_command_rejects_disallowed_command() -> None:
    with pytest.raises(ToolError):
        run_command("rm -rf /")


def test_run_command_allows_pytest_prefix(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("agentic_pipeline.tools.REPO_ROOT", tmp_path)
    output = run_command("python3 -c \"print('ok')\"")
    assert "ok" in output


# --------------------------------------------------------------------------
# claude_client: the manual tool-use loop, against a fake Anthropic client
# --------------------------------------------------------------------------


def test_run_agent_executes_tool_then_terminates_on_end_turn(monkeypatch) -> None:
    tool_use_block = SimpleNamespace(type="tool_use", name="dummy_tool", id="tu_1", input={"x": "a"})
    first_response = SimpleNamespace(stop_reason="tool_use", content=[tool_use_block])
    final_text_block = SimpleNamespace(type="text", text="done")
    second_response = SimpleNamespace(stop_reason="end_turn", content=[final_text_block])
    responses = iter([first_response, second_response])
    calls: list[dict] = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return next(responses)

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("agentic_pipeline.claude_client.get_client", lambda: FakeClient())

    result: AgentRunResult = run_agent(
        system_prompt="sys",
        user_message="hi",
        tool_schemas=[{"name": "dummy_tool", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        tool_dispatch={"dummy_tool": lambda x: f"handled:{x}"},
        model="fake-model",
    )

    assert result.text == "done"
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["name"] == "dummy_tool"
    assert len(calls) == 2
    # Second call must carry the tool_result back as a user message.
    assert calls[1]["messages"][-1]["content"][0]["content"] == "handled:a"


def test_run_agent_reports_unpermitted_tool_as_error_without_crashing(monkeypatch) -> None:
    tool_use_block = SimpleNamespace(type="tool_use", name="edit_file", id="tu_1", input={})
    first_response = SimpleNamespace(stop_reason="tool_use", content=[tool_use_block])
    second_response = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")])
    responses = iter([first_response, second_response])

    class FakeMessages:
        def create(self, **kwargs):
            return next(responses)

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("agentic_pipeline.claude_client.get_client", lambda: FakeClient())

    # A read-only agent's dispatch dict simply has no "edit_file" handler.
    result = run_agent(
        system_prompt="sys",
        user_message="hi",
        tool_schemas=[],
        tool_dispatch={},
        model="fake-model",
    )

    assert result.tool_trace[0]["result_preview"].startswith("error: tool 'edit_file' is not permitted")


# --------------------------------------------------------------------------
# orchestrator: stage branching and human-in-the-loop gates
# --------------------------------------------------------------------------


def _orchestrator(answers: list[str]) -> WorkflowOrchestrator:
    answer_iter = iter(answers)
    return WorkflowOrchestrator(ask_user=lambda _prompt: next(answer_iter))


def test_scope_spec_clarify_loop_short_circuits_when_complete() -> None:
    orch = _orchestrator([])
    scope_calls = []
    review_calls = []
    orch.scope_agent.run = lambda spec_text: scope_calls.append(spec_text) or "systems report"
    orch.spec_review_agent.run = lambda spec_text, systems: review_calls.append(1) or SpecReviewResult(status="COMPLETE")
    orch.clarification_agent.run = lambda *a, **k: pytest.fail("clarification should not run when spec is COMPLETE")

    from agentic_pipeline.orchestrator import Specification

    systems_report = orch._scope_spec_clarify_loop(Specification(verbatim_request="do X"))

    assert systems_report == "systems report"
    assert len(scope_calls) == 1
    assert len(review_calls) == 1


def test_scope_spec_clarify_loop_reassesses_after_user_answer() -> None:
    orch = _orchestrator(["blue"])  # the user's answer to the one clarifying question
    review_results = iter(
        [
            SpecReviewResult(status="NEEDS_INPUT", gaps=[Gap(system="GUI", ambiguity="which color", blocks_because="unset")]),
            SpecReviewResult(status="COMPLETE"),
        ]
    )
    scope_call_count = 0

    def fake_scope_run(spec_text: str) -> str:
        nonlocal scope_call_count
        scope_call_count += 1
        return "systems report"

    orch.scope_agent.run = fake_scope_run
    orch.spec_review_agent.run = lambda spec_text, systems: next(review_results)
    orch.clarification_agent.run = lambda verbatim, gaps: ClarificationResult(
        questions=[ClarificationQuestion(question="what color?", options=["blue", "red"], why_it_matters="matters")]
    )

    from agentic_pipeline.orchestrator import Specification

    spec = Specification(verbatim_request="add a button")
    orch._scope_spec_clarify_loop(spec)

    # Scope assessment must re-run (Stage 1) after the new clarification answer.
    assert scope_call_count == 2
    assert spec.clarifications == [("what color?", "blue")]


def test_boundaries_stage_loops_until_user_approves() -> None:
    orch = _orchestrator(["make it smaller", "yes"])
    calls: list[str | None] = []

    def fake_run(spec_text, systems, feedback=None):
        calls.append(feedback)
        return BoundariesResult(modify_zone=[BoundaryItem(target="kimeco/foo.py", reason="r")])

    orch.boundaries_agent.run = fake_run

    result = orch._boundaries_stage(spec=SimpleNamespace(as_text=lambda: "spec"), systems_report="systems")

    assert isinstance(result, BoundariesResult)
    assert calls == [None, "make it smaller"]


def test_verify_persistence_applies_fallback_edit_when_agent_could_not_persist(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("agentic_pipeline.tools.REPO_ROOT", tmp_path)
    (tmp_path / "target.py").write_text("old_value = 1\n")

    orch = _orchestrator([])
    implementation = ImplementationResult(
        files_changed=[
            FileChange(
                path="target.py",
                summary="bump value",
                persisted=False,
                old_string="old_value = 1",
                new_string="old_value = 2",
            )
        ],
        summary="bumped",
    )

    orch._verify_persistence(implementation)

    assert (tmp_path / "target.py").read_text() == "old_value = 2\n"
    assert implementation.files_changed[0].persisted is True
