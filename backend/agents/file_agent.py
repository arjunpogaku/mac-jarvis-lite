from __future__ import annotations

from backend.agents.models import CommandPlan


def plan_file_command(message: str) -> CommandPlan:
    lowered = message.lower()
    if "summar" in lowered or "read" in lowered:
        return CommandPlan(
            intent="read_or_summarize_file",
            agent="file_agent",
            requires_approval=True,
            safety_level="safe_read_only",
            plan=["Identify the requested approved file.", "Read or summarize it using bounded safe file tools."],
            tools_needed=["read_file", "summarize_file"],
            approval_message="Jarvis wants to read an approved local file. It will not modify files.",
        )
    return CommandPlan(
        intent="search_files",
        agent="file_agent",
        requires_approval=True,
        safety_level="safe_read_only",
        plan=["Search approved workspace folders for matching local files.", "Return matching paths and snippets."],
        tools_needed=["search_files"],
        approval_message="Jarvis wants to search approved folders. It will not modify files.",
    )

