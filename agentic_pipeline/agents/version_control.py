from __future__ import annotations

from agentic_pipeline.agents.base import BaseAgent
from agentic_pipeline.models import VersionControlResult


class VersionControlAgent(BaseAgent):
    """Stage 7: keeps CHANGELOG.md/wiki/MANUAL.md current, and owns SemVer
    bumps and release cuts -- release cuts only with explicit user approval,
    relayed by the orchestrator.
    """

    name = "version-control"

    def record_change(self, change_summary: str) -> VersionControlResult:
        message = (
            f"Change summary from Stages 5-6:\n{change_summary}\n\n"
            "Record this in CHANGELOG.md under [Unreleased], and update wiki/** and MANUAL.md "
            "only where the change affects documented behavior. Do not cut a release."
        )
        result = self._run(message)
        return self._extract(
            result.text,
            VersionControlResult,
            "This is a docs/changelog report. Extract the changelog entry text added, the list of "
            "wiki/manual pages updated (empty list + docs_unaffected_reason if docs were "
            "unaffected), and leave release_version unset.",
        )

    def cut_release(self, version: str, approved: bool) -> VersionControlResult:
        if not approved:
            raise ValueError("cut_release must only be called after explicit user approval")
        message = (
            f"The user explicitly approved cutting release version {version}. Rename [Unreleased] "
            "to this version with today's date, open a fresh empty [Unreleased] section, update the "
            "compare/link footnotes, bump the version identically in pyproject.toml/setup.py/meta.yaml, "
            "commit on Development, merge into main with --no-ff, and create annotated tag "
            f"v{version} on the merge commit. Leave commits and tags local -- do not push."
        )
        result = self._run(message)
        return self._extract(
            result.text,
            VersionControlResult,
            "This is a release-cut report. Extract the changelog entry, docs updated, the "
            "release_version, and whether release_created is true.",
        )
