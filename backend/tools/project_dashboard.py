from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.config import PROJECT_ROOT, Settings
from backend.db import DB_PATH, init_db
from backend.llm import OllamaClient
from backend.safety import SafetyError
from backend.tools.briefing import export_briefing, generate_briefing
from backend.tools.indexer import settings_for_workspace_name
from backend.tools.workspace_intelligence import check_workspace_health, generate_workspace_summary, inventory_workspace


EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
TASK_STATUSES = {"open", "in_progress", "blocked", "done", "dismissed"}
PRIORITIES = {"low", "medium", "high"}
CATEGORIES = {"research", "writing", "coding", "job_application", "data", "config", "cleanup", "unknown"}
PROJECT_TYPES = {"general", "research", "codebase", "job_application"}
DASHBOARD_STATUSES = {"healthy", "needs_attention", "blocked", "unknown"}
MAX_TASK_CONTEXT_CHARS = 14_000


@dataclass(frozen=True)
class ProjectTask:
    id: int
    workspace: str
    title: str
    description: str | None
    source_path: str | None
    source_line_start: int | None
    source_line_end: int | None
    category: str
    priority: str
    status: str
    evidence: str | None
    created_by: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TaskExtractionResult:
    workspace: str
    project_type: str
    tasks_created: int
    tasks_skipped_duplicates: int
    tasks: list[ProjectTask]
    sources_used: list[str]
    context_limited: bool


@dataclass(frozen=True)
class ProjectDashboard:
    id: int
    workspace: str
    project_type: str
    title: str
    overview: str
    status: str
    main_topics: list[str]
    risks: list[str]
    next_actions: list[str]
    open_task_count: int
    high_priority_task_count: int
    sources_used: list[str]
    generated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _dump_list(values: list[str]) -> str:
    return json.dumps([value for value in values if value])


def _load_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(item).strip() for item in data if str(item).strip()] if isinstance(data, list) else []


def _validate_project_type(project_type: str) -> str:
    normalized = project_type.strip().lower().replace(" ", "_")
    if normalized not in PROJECT_TYPES:
        raise SafetyError("Invalid project type.")
    return normalized


def _validate_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized not in TASK_STATUSES:
        raise SafetyError("Invalid task status.")
    return normalized


def _clean_priority(priority: str | None) -> str:
    value = (priority or "low").strip().lower()
    return value if value in PRIORITIES else "low"


def _clean_category(category: str | None) -> str:
    value = (category or "unknown").strip().lower()
    return value if value in CATEGORIES else "unknown"


def _task_from_row(row: sqlite3.Row) -> ProjectTask:
    return ProjectTask(
        id=int(row["id"]),
        workspace=str(row["workspace"]),
        title=str(row["title"]),
        description=row["description"],
        source_path=row["source_path"],
        source_line_start=int(row["source_line_start"]) if row["source_line_start"] is not None else None,
        source_line_end=int(row["source_line_end"]) if row["source_line_end"] is not None else None,
        category=str(row["category"] or "unknown"),
        priority=str(row["priority"] or "low"),
        status=str(row["status"]),
        evidence=row["evidence"],
        created_by=str(row["created_by"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _indexed_task_context(conn: sqlite3.Connection, workspace: str, limit_files: int) -> tuple[str, list[str], bool]:
    rows = conn.execute(
        """
        SELECT files.id AS file_id, files.path, files.file_name, ch.content, ch.start_line, ch.end_line, ch.chunk_index
        FROM indexed_files AS files
        JOIN document_chunks AS ch ON ch.file_id = files.id
        WHERE files.workspace = ?
        ORDER BY files.modified_time DESC, ch.chunk_index ASC
        LIMIT ?
        """,
        (workspace, max(1, limit_files) * 4),
    ).fetchall()
    parts: list[str] = []
    sources: list[str] = []
    total = 0
    limited = False
    todo_re = re.compile(r"\b(TODO|FIXME)\b[:\-\s]*(.*)", re.IGNORECASE)
    for row in rows:
        path = str(row["path"])
        sources.append(path)
        content = str(row["content"])
        matches = todo_re.findall(content)
        excerpt = content[:700]
        if matches:
            excerpt = "\n".join(f"{kind}: {text}".strip() for kind, text in matches)[:900]
        line_info = ""
        if row["start_line"] is not None and row["end_line"] is not None:
            line_info = f" lines {row['start_line']}-{row['end_line']}"
        part = f"Source: {path}{line_info}\nExcerpt:\n{excerpt}"
        if total + len(part) > MAX_TASK_CONTEXT_CHARS:
            limited = True
            break
        parts.append(part)
        total += len(part)
    return "\n\n---\n\n".join(parts), list(dict.fromkeys(sources)), limited


def _summary_task_context(conn: sqlite3.Connection, workspace: str) -> str:
    rows = conn.execute(
        """
        SELECT path, possible_actions, warnings, short_summary
        FROM file_summaries
        WHERE workspace = ?
        ORDER BY path ASC
        LIMIT 50
        """,
        (workspace,),
    ).fetchall()
    parts = []
    for row in rows:
        parts.append(
            f"Source: {row['path']}\nSummary: {row['short_summary'] or ''}\n"
            f"Possible actions: {row['possible_actions'] or '[]'}\nWarnings: {row['warnings'] or '[]'}"
        )
    return "\n\n".join(parts)


def _parse_task_json(text: str) -> list[dict[str, object]]:
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _fallback_tasks(context: str, sources: list[str]) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for line in context.splitlines():
        if "TODO" not in line.upper() and "FIXME" not in line.upper():
            continue
        title = re.sub(r"\b(TODO|FIXME)\b[:\-\s]*", "", line, flags=re.IGNORECASE).strip()[:90]
        if title:
            tasks.append(
                {
                    "title": title,
                    "description": title,
                    "source_path": sources[0] if sources else None,
                    "category": "unknown",
                    "priority": "medium",
                    "evidence": line[:300],
                }
            )
    return tasks[:20]


def _insert_task(conn: sqlite3.Connection, workspace: str, payload: dict[str, object], created_by: str = "jarvis") -> ProjectTask | None:
    title = str(payload.get("title", "")).strip()[:160]
    if not title:
        return None
    source_path = str(payload.get("source_path") or "").strip() or None
    duplicate = conn.execute(
        """
        SELECT id FROM project_tasks
        WHERE workspace = ? AND lower(title) = lower(?) AND COALESCE(source_path, '') = COALESCE(?, '')
        LIMIT 1
        """,
        (workspace, title, source_path),
    ).fetchone()
    if duplicate:
        return None
    timestamp = _now()
    cursor = conn.execute(
        """
        INSERT INTO project_tasks (
            workspace, title, description, source_path, source_line_start, source_line_end,
            category, priority, status, evidence, created_by, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            workspace,
            title,
            str(payload.get("description") or "").strip()[:500] or None,
            source_path,
            payload.get("source_line_start") if isinstance(payload.get("source_line_start"), int) else None,
            payload.get("source_line_end") if isinstance(payload.get("source_line_end"), int) else None,
            _clean_category(str(payload.get("category") or "unknown")),
            _clean_priority(str(payload.get("priority") or "low")),
            _validate_status(str(payload.get("status") or "open")),
            str(payload.get("evidence") or "").strip()[:500] or None,
            created_by,
            timestamp,
            timestamp,
        ),
    )
    row = conn.execute("SELECT * FROM project_tasks WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
    return _task_from_row(row)


async def extract_tasks(
    *,
    settings: Settings,
    workspace: str,
    project_type: str,
    llm: OllamaClient,
    limit_files: int = 30,
    refresh: bool = False,
    db_path: Path = DB_PATH,
) -> TaskExtractionResult:
    settings_for_workspace_name(settings, workspace)
    normalized_type = _validate_project_type(project_type)
    init_db(db_path)
    conn = _connect(db_path)
    try:
        indexed_count = conn.execute("SELECT COUNT(*) FROM indexed_files WHERE workspace = ?", (workspace,)).fetchone()[0]
        if int(indexed_count) == 0:
            raise SafetyError("This workspace has not been indexed yet. Please index it first.")
        context, sources, limited = _indexed_task_context(conn, workspace, limit_files)
        summary_context = _summary_task_context(conn, workspace)
    finally:
        conn.close()
    try:
        issues = check_workspace_health(settings=settings, workspace=workspace, limit=limit_files, db_path=db_path)
        issue_context = "\n".join(f"{issue.path}: {issue.issue_type}: {issue.explanation}" for issue in issues)
    except Exception:
        issue_context = ""
    prompt = (
        "Use only provided local indexed context. Do not invent files. "
        "Extract concrete project tasks as JSON array. Each task must include title, description, "
        "source_path, category, priority, evidence. Keep titles short. Assign priority conservatively; weak evidence is low.\n\n"
        f"Project type: {normalized_type}\n\nIndexed context:\n{context}\n\nFile summaries:\n{summary_context}\n\nHealth issues:\n{issue_context}"
    )
    try:
        response = await llm.chat(prompt)
    except Exception:
        response = ""
    task_payloads = _parse_task_json(str(response)) or _fallback_tasks("\n".join([context, summary_context, issue_context]), sources)
    conn = _connect(db_path)
    created: list[ProjectTask] = []
    skipped = 0
    try:
        for payload in task_payloads:
            task = _insert_task(conn, workspace, payload, created_by="jarvis")
            if task is None:
                skipped += 1
            else:
                created.append(task)
        conn.commit()
    finally:
        conn.close()
    return TaskExtractionResult(
        workspace=workspace,
        project_type=normalized_type,
        tasks_created=len(created),
        tasks_skipped_duplicates=skipped,
        tasks=created,
        sources_used=sources,
        context_limited=limited,
    )


def list_tasks(
    *,
    workspace: str,
    status: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> list[ProjectTask]:
    init_db(db_path)
    query = "SELECT * FROM project_tasks WHERE workspace = ?"
    params: list[object] = [workspace]
    if status:
        query += " AND status = ?"
        params.append(_validate_status(status))
    if category:
        query += " AND category = ?"
        params.append(_clean_category(category))
    if priority:
        query += " AND priority = ?"
        params.append(_clean_priority(priority))
    query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
    params.append(max(1, min(limit, 200)))
    conn = _connect(db_path)
    try:
        return [_task_from_row(row) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def update_task_status(*, task_id: int, status: str, db_path: Path = DB_PATH) -> ProjectTask:
    new_status = _validate_status(status)
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM project_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise SafetyError("Task not found.")
        conn.execute("UPDATE project_tasks SET status = ?, updated_at = ? WHERE id = ?", (new_status, _now(), task_id))
        conn.commit()
        updated = conn.execute("SELECT * FROM project_tasks WHERE id = ?", (task_id,)).fetchone()
        return _task_from_row(updated)
    finally:
        conn.close()


def add_manual_task(
    *,
    workspace: str,
    title: str,
    description: str | None,
    category: str,
    priority: str,
    db_path: Path = DB_PATH,
) -> ProjectTask:
    conn = _connect(db_path)
    try:
        task = _insert_task(
            conn,
            workspace,
            {
                "title": title,
                "description": description,
                "category": category,
                "priority": priority,
                "status": "open",
                "evidence": "Manual task created by user.",
            },
            created_by="user",
        )
        conn.commit()
        if task is None:
            existing = list_tasks(workspace=workspace, limit=1, db_path=db_path)
            if existing:
                return existing[0]
            raise SafetyError("Task could not be created.")
        return task
    finally:
        conn.close()


def _dashboard_from_row(row: sqlite3.Row, open_count: int, high_count: int) -> ProjectDashboard:
    return ProjectDashboard(
        id=int(row["id"]),
        workspace=str(row["workspace"]),
        project_type=str(row["project_type"]),
        title=str(row["title"] or "Project Dashboard"),
        overview=str(row["overview"] or ""),
        status=str(row["status"] or "unknown"),
        main_topics=_load_list(row["main_topics"]),
        risks=_load_list(row["risks"]),
        next_actions=_load_list(row["next_actions"]),
        open_task_count=open_count,
        high_priority_task_count=high_count,
        sources_used=_load_list(row["sources_used"]),
        generated_at=str(row["updated_at"]),
    )


def _parse_dashboard_json(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


async def generate_dashboard(
    *,
    settings: Settings,
    workspace: str,
    project_type: str,
    llm: OllamaClient,
    refresh: bool = False,
    db_path: Path = DB_PATH,
) -> ProjectDashboard:
    settings_for_workspace_name(settings, workspace)
    normalized_type = _validate_project_type(project_type)
    init_db(db_path)
    conn = _connect(db_path)
    try:
        open_count = int(conn.execute("SELECT COUNT(*) FROM project_tasks WHERE workspace = ? AND status != 'done' AND status != 'dismissed'", (workspace,)).fetchone()[0])
        high_count = int(conn.execute("SELECT COUNT(*) FROM project_tasks WHERE workspace = ? AND priority = 'high' AND status != 'done' AND status != 'dismissed'", (workspace,)).fetchone()[0])
        cached = conn.execute(
            "SELECT * FROM project_dashboards WHERE workspace = ? AND project_type = ? ORDER BY id DESC LIMIT 1",
            (workspace, normalized_type),
        ).fetchone()
        if cached and not refresh:
            return _dashboard_from_row(cached, open_count, high_count)
    finally:
        conn.close()
    try:
        summary = await generate_workspace_summary(settings=settings, workspace=workspace, llm=llm, db_path=db_path)
        summary_context = f"Summary: {summary.summary}\nTopics: {summary.main_topics}\nActions: {summary.possible_actions}\nWarnings: {summary.warnings}"
        sources = summary.important_files
    except Exception:
        summary_context = "No workspace summary available."
        sources = []
    tasks = list_tasks(workspace=workspace, limit=100, db_path=db_path)
    task_context = "\n".join(f"{task.priority} {task.status}: {task.title} ({task.source_path or 'manual'}) Evidence: {task.evidence or ''}" for task in tasks[:50])
    prompt = (
        "Use only provided local project context. Return JSON with title, overview, status, main_topics, risks, next_actions. "
        "Status must be healthy, needs_attention, blocked, or unknown. Do not exaggerate; weak evidence means unknown.\n\n"
        f"Workspace: {workspace}\nProject type: {normalized_type}\n\n{summary_context}\n\nTasks:\n{task_context}"
    )
    try:
        response = await llm.chat(prompt)
    except Exception:
        response = ""
    payload = _parse_dashboard_json(str(response)) or {}
    status = str(payload.get("status") or ("needs_attention" if open_count else "unknown")).strip().lower()
    if status not in DASHBOARD_STATUSES:
        status = "unknown"
    title = str(payload.get("title") or f"{workspace} Project Dashboard").strip()
    overview = str(payload.get("overview") or summary_context[:800]).strip()
    main_topics = [str(item) for item in payload.get("main_topics", [])] if isinstance(payload.get("main_topics"), list) else []
    risks = [str(item) for item in payload.get("risks", [])] if isinstance(payload.get("risks"), list) else []
    next_actions = [str(item) for item in payload.get("next_actions", [])] if isinstance(payload.get("next_actions"), list) else [task.title for task in tasks[:5]]
    timestamp = _now()
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO project_dashboards (
                workspace, project_type, title, overview, status, main_topics, risks,
                next_actions, sources_used, model, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace,
                normalized_type,
                title,
                overview,
                status,
                _dump_list(main_topics),
                _dump_list(risks),
                _dump_list(next_actions),
                _dump_list(sources),
                settings.llm.model,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM project_dashboards WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return _dashboard_from_row(row, open_count, high_count)
    finally:
        conn.close()


def export_dashboard(*, workspace: str, dashboard_id: int, approved: bool, db_path: Path = DB_PATH, export_dir: Path = EXPORT_DIR) -> Path:
    if not approved:
        raise SafetyError("Dashboard export requires explicit approval.")
    conn = _connect(db_path)
    try:
        dashboard = conn.execute("SELECT * FROM project_dashboards WHERE id = ? AND workspace = ?", (dashboard_id, workspace)).fetchone()
        if dashboard is None:
            raise SafetyError("Dashboard not found.")
        tasks = [_task_from_row(row) for row in conn.execute("SELECT * FROM project_tasks WHERE workspace = ? ORDER BY status, priority, id", (workspace,)).fetchall()]
    finally:
        conn.close()
    body = [
        f"# {dashboard['title'] or workspace + ' Project Dashboard'}",
        "",
        f"Workspace: {workspace}",
        f"Status: {dashboard['status'] or 'unknown'}",
        "",
        "## Overview",
        str(dashboard["overview"] or ""),
        "",
        "## Main Topics",
        "\n".join(f"- {item}" for item in _load_list(dashboard["main_topics"])) or "- None recorded",
        "",
        "## Risks",
        "\n".join(f"- {item}" for item in _load_list(dashboard["risks"])) or "- None recorded",
        "",
        "## Next Actions",
        "\n".join(f"- {item}" for item in _load_list(dashboard["next_actions"])) or "- None recorded",
        "",
        "## Tasks",
    ]
    for task in tasks:
        body.append(f"- [{task.status}] {task.priority} {task.title} ({task.category})")
    return export_briefing(
        workspace=workspace,
        briefing_type="workspace",
        briefing_text="\n".join(body),
        approved=True,
        export_dir=export_dir,
    )
