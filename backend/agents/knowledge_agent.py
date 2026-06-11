from __future__ import annotations

from backend.agents.models import CommandPlan


def plan_kb_question() -> CommandPlan:
    return CommandPlan(
        intent="ask_knowledge_base",
        agent="knowledge_agent",
        requires_approval=True,
        safety_level="safe_read_only",
        plan=["Search the local SQLite knowledge base.", "Use retrieved indexed chunks to answer with sources."],
        tools_needed=["kb_hybrid_search", "kb_ask"],
        approval_message="Jarvis wants to search indexed local workspace content. It will not modify files.",
    )

