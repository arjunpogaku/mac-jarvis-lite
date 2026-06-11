from __future__ import annotations

from dataclasses import dataclass, field


SUPPORTED_INTENTS = {
    "chat",
    "search_files",
    "read_or_summarize_file",
    "ask_knowledge_base",
    "workspace_summary",
    "briefing",
    "project_dashboard",
    "extract_tasks",
    "list_tasks",
    "research_review",
    "codebase_review",
    "job_application_help",
    "unknown",
}


@dataclass(frozen=True)
class CommandPlan:
    intent: str
    agent: str
    requires_approval: bool
    safety_level: str
    plan: list[str]
    tools_needed: list[str]
    approval_message: str
    refusal: str | None = None
    plan_id: str | None = None


@dataclass(frozen=True)
class CommandResult:
    answer: str
    sources_used: list[str] = field(default_factory=list)
    actions_performed: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    plan: CommandPlan | None = None

