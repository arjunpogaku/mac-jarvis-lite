from __future__ import annotations

from backend.agents.models import CommandPlan


def plan_research_review() -> CommandPlan:
    return CommandPlan(
        intent="research_review",
        agent="research_agent",
        requires_approval=True,
        safety_level="safe_read_only",
        plan=[
            "Search indexed research workspace content.",
            "Check workspace health for TODOs and weak sections.",
            "Generate source-backed research improvement suggestions.",
        ],
        tools_needed=["kb_hybrid_search", "kb_ask", "workspace_health_check"],
        approval_message="Jarvis wants to analyze indexed research files. It will not modify files.",
    )

