from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.agents.briefing_agent import plan_briefing
from backend.agents.code_agent import plan_code_review
from backend.agents.file_agent import plan_file_command
from backend.agents.job_agent import plan_job_help
from backend.agents.knowledge_agent import plan_kb_question
from backend.agents.models import CommandPlan, CommandResult
from backend.agents.project_agent import plan_extract_tasks, plan_list_tasks, plan_project_dashboard
from backend.agents.research_agent import plan_research_review
from backend.agents.safety_agent import blocked_plan, ensure_safe_plan
from backend.config import Settings
from backend.db import DB_PATH, init_db
from backend.llm import OllamaClient
from backend.safety import SafetyError
from backend.tools.briefing import generate_briefing
from backend.tools.indexer import ask_kb
from backend.tools.project_dashboard import extract_tasks, generate_dashboard, list_tasks
from backend.tools.workspace_intelligence import check_workspace_health, generate_workspace_summary


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _log_command(
    *,
    session_id: str | None,
    workspace: str,
    message: str,
    plan: CommandPlan,
    approved: bool,
    executed: bool,
    status: str,
    db_path: Path = DB_PATH,
) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO command_history (
                session_id, workspace, user_message, intent, agent, plan_json,
                approved, executed, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                workspace,
                message[:500],
                plan.intent,
                plan.agent,
                json.dumps(asdict(plan))[:4000],
                1 if approved else 0,
                1 if executed else 0,
                status,
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _fallback_plan(message: str) -> CommandPlan:
    lowered = message.lower()
    if "briefing" in lowered or "brief" in lowered:
        return plan_briefing(message)
    if any(term in lowered for term in ["thesis", "paper", "research", "weak section", "sections need improvement", "review"]):
        return plan_research_review()
    if any(term in lowered for term in ["where did i", "find", "tell me where", "what did i write", "ask"]):
        return plan_kb_question()
    if any(term in lowered for term in ["what should i work on", "today", "tasks", "priorities", "next actions"]):
        if any(term in lowered for term in ["extract", "create tasks", "find tasks"]):
            return plan_extract_tasks()
        return plan_list_tasks()
    if "dashboard" in lowered or "project status" in lowered:
        return plan_project_dashboard()
    if any(term in lowered for term in ["codebase", "module", "todo", "fixme", "dependency"]):
        return plan_code_review()
    if any(term in lowered for term in ["resume", "cover letter", "job description", "job application", "skills"]):
        return plan_job_help()
    if any(term in lowered for term in ["search file", "search files", "read file", "summarize file"]):
        return plan_file_command(message)
    return CommandPlan(
        intent="chat",
        agent="chat_agent",
        requires_approval=False,
        safety_level="no_tool",
        plan=["Answer conversationally without running tools."],
        tools_needed=[],
        approval_message="No tools are needed.",
    )


async def plan_command(
    *,
    message: str,
    workspace: str,
    settings: Settings,
    llm: OllamaClient | None = None,
    session_id: str | None = None,
    db_path: Path = DB_PATH,
) -> CommandPlan:
    if workspace not in settings.workspaces:
        raise SafetyError("Invalid workspace.")
    blocked = blocked_plan(message)
    plan = blocked or _fallback_plan(message)
    plan = ensure_safe_plan(plan)
    plan = CommandPlan(**{**asdict(plan), "plan_id": str(uuid4())})
    _log_command(
        session_id=session_id,
        workspace=workspace,
        message=message,
        plan=plan,
        approved=False,
        executed=False,
        status="rejected" if plan.refusal else "planned",
        db_path=db_path,
    )
    return plan


def _sources_from_kb_answer(answer) -> list[str]:
    return [source.path for source in answer.sources_used]


async def execute_command(
    *,
    message: str,
    workspace: str,
    approved: bool,
    settings: Settings,
    llm: OllamaClient,
    session_id: str | None = None,
    db_path: Path = DB_PATH,
) -> CommandResult:
    plan = await plan_command(message=message, workspace=workspace, settings=settings, llm=llm, session_id=session_id, db_path=db_path)
    if plan.refusal:
        _log_command(session_id=session_id, workspace=workspace, message=message, plan=plan, approved=False, executed=False, status="rejected", db_path=db_path)
        return CommandResult(answer=plan.refusal, limitations=["Unsafe request refused."], plan=plan)
    if plan.requires_approval and not approved:
        raise SafetyError("Command execution requires approval.")

    actions: list[str] = []
    sources: list[str] = []
    limitations: list[str] = []

    if plan.intent == "chat":
        answer = await llm.chat(message)
        actions.append("Answered without tools.")
    elif plan.intent == "ask_knowledge_base":
        kb_answer = await ask_kb(question=message, workspace=workspace, settings=settings, llm=llm, db_path=db_path)
        answer = kb_answer.answer
        sources = _sources_from_kb_answer(kb_answer)
        actions.append(f"Searched local knowledge base with {kb_answer.search_mode_used} mode.")
        if kb_answer.context_limited:
            limitations.append("Context was limited before answering.")
    elif plan.intent == "research_review":
        kb_answer = await ask_kb(question=f"Review this research request and cite weak sections: {message}", workspace=workspace, settings=settings, llm=llm, db_path=db_path)
        try:
            issues = check_workspace_health(settings=settings, workspace=workspace, limit=20, db_path=db_path)
        except SafetyError as exc:
            issues = []
            limitations.append(str(exc))
        issue_text = "\n".join(f"- {issue.path}: {issue.explanation}" for issue in issues[:5])
        answer = kb_answer.answer + ("\n\nWorkspace health signals:\n" + issue_text if issue_text else "")
        sources = _sources_from_kb_answer(kb_answer) + [issue.path for issue in issues[:5]]
        actions.extend(["Searched indexed research content.", "Checked workspace health signals."])
    elif plan.intent == "briefing":
        briefing_type = "research" if "research" in message.lower() else "codebase" if "code" in message.lower() else "job_application" if "job" in message.lower() else "workspace"
        briefing = await generate_briefing(settings=settings, workspace=workspace, briefing_type=briefing_type, llm=llm, db_path=db_path)
        answer = briefing.briefing_text
        sources = briefing.sources_used
        actions.append(f"Generated {briefing_type} briefing from indexed local content.")
        if briefing.context_limited:
            limitations.append("Briefing context was limited.")
    elif plan.intent == "project_dashboard":
        dashboard = await generate_dashboard(settings=settings, workspace=workspace, project_type="general", llm=llm, db_path=db_path)
        answer = (
            f"{dashboard.title}\n\nStatus: {dashboard.status}\n\n{dashboard.overview}\n\n"
            f"Open tasks: {dashboard.open_task_count}\nHigh priority tasks: {dashboard.high_priority_task_count}\n\n"
            "Next actions:\n" + "\n".join(f"- {item}" for item in dashboard.next_actions)
        )
        sources = dashboard.sources_used
        actions.append("Generated project dashboard from Jarvis SQLite metadata.")
    elif plan.intent == "extract_tasks":
        result = await extract_tasks(settings=settings, workspace=workspace, project_type="general", llm=llm, db_path=db_path)
        answer = f"Created {result.tasks_created} task(s), skipped {result.tasks_skipped_duplicates} duplicate(s)."
        sources = result.sources_used
        actions.append("Extracted tasks into Jarvis SQLite only.")
    elif plan.intent == "list_tasks":
        tasks = list_tasks(workspace=workspace, status="open", limit=10, db_path=db_path)
        if tasks:
            answer = "Top open tasks:\n" + "\n".join(f"{idx + 1}. {task.title} ({task.priority})" for idx, task in enumerate(tasks))
        else:
            answer = "No open tasks are currently stored for this workspace."
        actions.append("Read Jarvis internal task records.")
    elif plan.intent in {"codebase_review", "job_application_help", "workspace_summary"}:
        summary = await generate_workspace_summary(settings=settings, workspace=workspace, llm=llm, db_path=db_path)
        answer = summary.summary
        sources = summary.important_files
        actions.append("Generated workspace summary from indexed local content.")
    else:
        answer = "I am not sure how to help with that yet. Try asking me to search, brief, review, or list tasks."
        limitations.append("Unknown intent.")

    _log_command(
        session_id=session_id,
        workspace=workspace,
        message=message,
        plan=plan,
        approved=approved,
        executed=True,
        status="executed",
        db_path=db_path,
    )
    return CommandResult(answer=answer, sources_used=list(dict.fromkeys(sources)), actions_performed=actions, limitations=limitations, plan=plan)
