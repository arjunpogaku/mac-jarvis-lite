from __future__ import annotations

from backend.agents.models import CommandPlan


def plan_code_review() -> CommandPlan:
    return CommandPlan(
        intent="codebase_review",
        agent="code_agent",
        requires_approval=True,
        safety_level="safe_read_only",
        plan=["Inspect indexed codebase metadata.", "Look for TODO/FIXME items and risky files.", "Return source-backed codebase suggestions."],
        tools_needed=["workspace_summary", "workspace_health_check", "kb_hybrid_search"],
        approval_message="Jarvis wants to analyze indexed code content. It will not modify files.",
    )

