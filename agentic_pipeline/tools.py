"""Concrete implementations of the abstract Copilot tools (`read`, `search`,
`edit`, `execute`) plus a permission-scoped registry builder.

Each Copilot agent's front matter declares a subset of ``["read", "search",
"edit", "execute", "todo", "agent"]``. ``build_tool_registry`` turns that
declared subset into the *only* Claude tool schemas/handlers an agent
receives. A read-only agent (``tools: [read, search]``) is simply never
given the ``edit_file``/``write_file``/``run_command`` schemas, so the model
cannot name them in a valid ``tool_use`` block -- permissions are enforced
structurally, not by convention or prompt instruction.

``todo`` and ``agent`` are not exposed as Claude tools: the orchestrator
tracks the todo list itself and performs agent dispatch directly in Python
(see ``orchestrator.py``), matching how this pipeline replaces Copilot's
subagent-dispatch primitive.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable

from agentic_pipeline.config import ALLOWED_EXECUTE_COMMANDS, REPO_ROOT

logger = logging.getLogger("agentic_pipeline.tools")


class ToolError(Exception):
    """Raised when a tool call is invalid or unsafe; surfaced to Claude as an error tool_result."""


def _resolve_within_repo(path: str) -> Path:
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise ToolError(f"path '{path}' escapes the repository root") from None
    return resolved


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------

def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    resolved = _resolve_within_repo(path)
    if not resolved.exists():
        raise ToolError(f"file not found: {path}")
    if not resolved.is_file():
        raise ToolError(f"not a file: {path}")
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    end = len(lines) if limit is None else offset + limit
    numbered = [f"{i + 1}\t{line}" for i, line in enumerate(lines[offset:end], start=offset)]
    return "\n".join(numbered) if numbered else "(empty file)"


READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a file from the repository. Returns line-numbered content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repo-relative file path."},
            "offset": {"type": "integer", "description": "0-indexed line to start from."},
            "limit": {"type": "integer", "description": "Maximum number of lines to return."},
        },
        "required": ["path"],
    },
}


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def search_code(pattern: str, path: str = ".", max_results: int = 200) -> str:
    resolved = _resolve_within_repo(path)
    try:
        proc = subprocess.run(
            ["grep", "-rn", "-I", "--exclude-dir=.git", "-e", pattern, str(resolved)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise ToolError("grep is not available on this system") from None
    if proc.returncode not in (0, 1):
        raise ToolError(f"search failed: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()[:max_results]
    if not lines:
        return "(no matches)"
    root = str(REPO_ROOT) + "/"
    return "\n".join(line.replace(root, "") for line in lines)


SEARCH_CODE_SCHEMA = {
    "name": "search_code",
    "description": "Search the repository for a regex pattern (grep -rn). Returns matching lines with file:line prefixes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "Repo-relative directory or file to search within. Defaults to the whole repo."},
        },
        "required": ["pattern"],
    },
}


# --------------------------------------------------------------------------
# edit
# --------------------------------------------------------------------------

def edit_file(path: str, old_string: str, new_string: str) -> str:
    resolved = _resolve_within_repo(path)
    if not resolved.exists():
        raise ToolError(f"file not found: {path}")
    original = resolved.read_text(encoding="utf-8")
    count = original.count(old_string)
    if count == 0:
        raise ToolError(f"old_string not found in {path}")
    if count > 1:
        raise ToolError(f"old_string is not unique in {path} ({count} occurrences)")
    resolved.write_text(original.replace(old_string, new_string, 1), encoding="utf-8")
    # Persist-and-verify, matching the original agents' "re-read to confirm" discipline.
    persisted = resolved.read_text(encoding="utf-8")
    if new_string not in persisted:
        raise ToolError(f"edit to {path} did not persist as expected")
    return f"edited {path} ({len(old_string)} chars -> {len(new_string)} chars); verified on disk"


EDIT_FILE_SCHEMA = {
    "name": "edit_file",
    "description": "Replace exactly one occurrence of old_string with new_string in an existing file, then verify the write by re-reading the file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repo-relative file path."},
            "old_string": {"type": "string", "description": "Exact text to replace. Must appear exactly once."},
            "new_string": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_string", "new_string"],
    },
}


def write_file(path: str, content: str) -> str:
    resolved = _resolve_within_repo(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    persisted = resolved.read_text(encoding="utf-8")
    if persisted != content:
        raise ToolError(f"write to {path} did not persist as expected")
    return f"wrote {path} ({len(content)} chars); verified on disk"


WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Create a new file, or fully overwrite an existing one, with the given content. Then verify the write by re-reading the file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repo-relative file path."},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["path", "content"],
    },
}


# --------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------

def run_command(command: str, timeout: int = 120) -> str:
    first_token = command.strip().split()[0] if command.strip() else ""
    if first_token not in ALLOWED_EXECUTE_COMMANDS:
        raise ToolError(
            f"command '{first_token}' is not allowed; permitted commands: {sorted(ALLOWED_EXECUTE_COMMANDS)}"
        )
    proc = subprocess.run(
        command,
        shell=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = f"$ {command}\n(exit {proc.returncode})\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    return output[-8000:]


RUN_COMMAND_SCHEMA = {
    "name": "run_command",
    "description": (
        "Run a shell command in the repository root. Only these commands are permitted: "
        + ", ".join(sorted(ALLOWED_EXECUTE_COMMANDS))
        + " (e.g. `pytest tests/unit/ -q`, `git status`)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The full shell command to run."}},
        "required": ["command"],
    },
}


# --------------------------------------------------------------------------
# permission-scoped registry
# --------------------------------------------------------------------------

# Maps a Copilot front-matter tool name to the Claude tool schema(s) + handler(s) it grants.
_TOOL_GROUPS: dict[str, list[tuple[dict, Callable[..., str]]]] = {
    "read": [(READ_FILE_SCHEMA, read_file)],
    "search": [(SEARCH_CODE_SCHEMA, search_code)],
    "edit": [(EDIT_FILE_SCHEMA, edit_file), (WRITE_FILE_SCHEMA, write_file)],
    "execute": [(RUN_COMMAND_SCHEMA, run_command)],
    # "todo" and "agent" intentionally have no Claude-tool equivalent -- see module docstring.
}


def build_tool_registry(copilot_tool_names: list[str]) -> tuple[list[dict], dict[str, Callable[..., str]]]:
    """Given an agent's declared Copilot tool names, return the matching
    Claude tool schemas and a name->handler dispatch dict, restricted to
    exactly those tools. Unknown/unmapped tool names (``todo``, ``agent``)
    are silently skipped -- they have no filesystem/shell effect to gate.
    """
    schemas: list[dict] = []
    dispatch: dict[str, Callable[..., str]] = {}
    for tool_name in copilot_tool_names:
        for schema, handler in _TOOL_GROUPS.get(tool_name, []):
            schemas.append(schema)
            dispatch[schema["name"]] = handler
    return schemas, dispatch
