from __future__ import annotations

from backend.agents.models import CommandPlan


def plan_project_dashboard() -> CommandPlan:
    return CommandPlan(
        intent="project_dashboard",
        agent="project_agent",
        requires_approval=True,
        safety_level="safe_read_only",
        plan=["Load Jarvis project dashboard data.", "Summarize project status, risks, and next actions."],
        tools_needed=["project_dashboard", "project_tasks"],
        approval_message="Jarvis wants to read project metadata from its local SQLite database.",
    )


def plan_extract_tasks() -> CommandPlan:
    return CommandPlan(
        intent="extract_tasks",
        agent="project_agent",
        requires_approval=True,
        safety_level="safe_internal_write",
        plan=["Analyze indexed local content for task evidence.", "Store suggested tasks inside Jarvis SQLite only."],
        tools_needed=["project_extract_tasks"],
        approval_message="Jarvis wants to create internal task records in SQLite. It will not modify source files.",
    )


def plan_list_tasks() -> CommandPlan:
    return CommandPlan(
        intent="list_tasks",
        agent="project_agent",
        requires_approval=False,
        safety_level="safe_internal_read",
        plan=["Read existing Jarvis task records from SQLite.", "Return prioritized open tasks."],
        tools_needed=["project_tasks"],
        approval_message="No approval is needed to read Jarvis internal task records.",
    )

