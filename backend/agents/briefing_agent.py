from __future__ import annotations

from backend.agents.models import CommandPlan


def plan_briefing(message: str) -> CommandPlan:
    lowered = message.lower()
    briefing_type = "research" if "research" in lowered else "codebase" if "code" in lowered else "job_application" if "job" in lowered else "workspace"
    return CommandPlan(
        intent="briefing",
        agent="briefing_agent",
        requires_approval=True,
        safety_level="safe_read_only",
        plan=[f"Generate a {briefing_type} briefing from indexed local content.", "Cite sources and include practical next actions."],
        tools_needed=["briefing_generate"],
        approval_message="Jarvis wants to generate a briefing from indexed local content. It will not modify files.",
    )

