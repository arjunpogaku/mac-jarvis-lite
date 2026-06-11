from __future__ import annotations

from backend.agents.models import CommandPlan


def plan_job_help() -> CommandPlan:
    return CommandPlan(
        intent="job_application_help",
        agent="job_agent",
        requires_approval=True,
        safety_level="safe_read_only",
        plan=["Search indexed job application files.", "Generate source-backed resume or application suggestions."],
        tools_needed=["kb_hybrid_search", "briefing_generate"],
        approval_message="Jarvis wants to analyze indexed job files. It will not send email, upload files, or apply anywhere.",
    )

