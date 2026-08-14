"""Thin wrapper around the Anthropic SDK: a manual tool-use loop for running
an agent's real work, plus a separate structured-extraction helper for
turning an agent's finished prose into machine-readable state.

No agent framework here on purpose -- this is ~80 lines of explicit control
flow so the whole pipeline stays easy to step through and debug.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import anthropic
from pydantic import BaseModel

from agentic_pipeline.config import ANTHROPIC_API_KEY, MAX_TOOL_ITERATIONS
from agentic_pipeline.tools import ToolError

logger = logging.getLogger("agentic_pipeline.claude_client")

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or run `ant auth login`, before running the pipeline."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


@dataclass
class AgentRunResult:
    text: str
    tool_trace: list[dict] = field(default_factory=list)


def run_agent(
    *,
    system_prompt: str,
    user_message: str,
    tool_schemas: list[dict],
    tool_dispatch: dict[str, Callable[..., str]],
    model: str,
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> AgentRunResult:
    """Runs Claude with ``system_prompt`` (loaded verbatim from the agent's
    ``.github/agents/*.md`` file) against ``user_message``, executing any
    ``tool_use`` blocks strictly against ``tool_dispatch``.

    ``tool_dispatch`` only ever contains handlers for the tools the agent's
    front matter grants (see ``tools.build_tool_registry``) -- a read-only
    agent is given no ``edit_file``/``write_file``/``run_command`` schema at
    all, so it cannot validly request them. The unknown-tool branch below is
    defense in depth, not the primary enforcement mechanism.
    """
    client = get_client()
    messages: list[dict] = [{"role": "user", "content": user_message}]
    tool_trace: list[dict] = []

    for _ in range(max_iterations):
        kwargs: dict = dict(model=model, max_tokens=8000, system=system_prompt, messages=messages)
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        response = client.messages.create(**kwargs)

        if response.stop_reason != "tool_use":
            text = "".join(block.text for block in response.content if block.type == "text")
            return AgentRunResult(text=text, tool_trace=tool_trace)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = tool_dispatch.get(block.name)
            is_error = False
            if handler is None:
                result_text = f"error: tool '{block.name}' is not permitted for this agent"
                is_error = True
            else:
                try:
                    result_text = handler(**block.input)
                except ToolError as exc:
                    result_text, is_error = f"error: {exc}", True
                except Exception as exc:  # tool bug -- surface to the model, don't crash the loop
                    logger.exception("tool %s raised", block.name)
                    result_text, is_error = f"error: {exc}", True
            tool_trace.append({"name": block.name, "input": block.input, "result_preview": result_text[:200]})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result_text, "is_error": is_error}
            )
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"agent exceeded {max_iterations} tool-use iterations without finishing")


T = TypeVar("T", bound=BaseModel)


def extract_structured(*, source_text: str, schema: type[T], instruction: str, model: str) -> T:
    """Extracts a Pydantic model's fields out of an agent's finished prose.

    A separate, tool-free call: the agent's own prompt (loaded verbatim from
    ``.github/agents/``) is never altered to also emit JSON, and the
    orchestrator never regex-parses the agent's human-readable output for
    control flow.
    """
    client = get_client()
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    "Extract the requested fields from the following report. "
                    "If a field is not discussed, use an empty/default value -- do not invent content.\n\n"
                    f"--- report ---\n{source_text}\n--- end report ---"
                ),
            }
        ],
        output_format=schema,
    )
    return response.parsed_output
