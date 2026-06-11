from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.config import PROJECT_ROOT, Settings
from backend.db import DB_PATH, init_db
from backend.llm import OllamaClient
from backend.safety import SafetyError
from backend.tools.indexer import settings_for_workspace_name
from backend.tools.workspace_intelligence import check_workspace_health, explore_topics, inventory_workspace


EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
MAX_BRIEFING_CONTEXT_CHARS = 14_000
BRIEFING_TYPES = {"workspace", "research", "codebase", "job_application"}


@dataclass(frozen=True)
class BriefingResult:
    workspace: str
    briefing_type: str
    briefing_text: str
    sources_used: list[str]
    files_considered: int
    context_limited: bool
    generated_at: str


def _connect(db_path: Path) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_briefing_type(briefing_type: str) -> str:
    normalized = briefing_type.strip().lower().replace(" ", "_")
    if normalized not in BRIEFING_TYPES:
        raise SafetyError("Invalid briefing type.")
    return normalized


def _briefing_instructions(briefing_type: str) -> str:
    if briefing_type == "research":
        return (
            "Produce sections: research topic, problem being addressed, method or system described, "
            "datasets or experiments mentioned, important terms, possible weaknesses, suggested next writing actions, sources used."
        )
    if briefing_type == "codebase":
        return (
            "Produce sections: project purpose, main modules, key files, dependencies mentioned, TODO/FIXME items, "
            "possible risks, suggested next development actions, sources used."
        )
    if briefing_type == "job_application":
        return (
            "Produce sections: roles or job descriptions detected, skills mentioned, resume/cover-letter files detected, "
            "possible application actions, missing information, sources used."
        )
    return (
        "Produce sections: overview, main topics, important files, open issues or TODOs, suggested next actions, sources used."
    )


def _indexed_file_rows(conn: sqlite3.Connection, workspace: str, limit_files: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT files.*, COUNT(ch.id) AS chunk_count
            FROM indexed_files AS files
            LEFT JOIN document_chunks AS ch ON ch.file_id = files.id
            WHERE files.workspace = ?
            GROUP BY files.id
            ORDER BY files.modified_time DESC, files.path ASC
            LIMIT ?
            """,
            (workspace, limit_files),
        ).fetchall()
    )


def _summary_row(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM file_summaries WHERE file_id = ?", (file_id,)).fetchone()


def _chunk_rows(conn: sqlite3.Connection, file_id: int, limit: int = 3) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT content, start_line, end_line, chunk_index
            FROM document_chunks
            WHERE file_id = ?
            ORDER BY chunk_index ASC
            LIMIT ?
            """,
            (file_id, limit),
        ).fetchall()
    )


def _build_context(
    *,
    settings: Settings,
    workspace: str,
    briefing_type: str,
    files: list[sqlite3.Row],
    db_path: Path,
) -> tuple[str, list[str], bool]:
    sources: list[str] = []
    parts: list[str] = []
    total = 0
    context_limited = False

    try:
        inventory = inventory_workspace(settings=settings, workspace=workspace, limit=min(max(len(files), 1), 100), db_path=db_path)
        parts.append(
            "Workspace metadata:\n"
            f"Total indexed files: {inventory['total_indexed_files']}\n"
            f"Total chunks: {inventory['total_chunks']}\n"
            f"Extensions: {inventory['extensions_breakdown']}\n"
            f"Categories: {inventory['category_breakdown']}"
        )
    except Exception:
        pass

    try:
        issues = check_workspace_health(settings=settings, workspace=workspace, limit=30, db_path=db_path)
        if issues:
            parts.append(
                "Workspace health issues:\n"
                + "\n".join(f"{issue.severity}: {issue.path}: {issue.explanation}" for issue in issues[:10])
            )
    except Exception:
        pass

    conn = _connect(db_path)
    try:
        for row in files:
            path = str(row["path"])
            file_id = int(row["id"])
            sources.append(path)
            summary = _summary_row(conn, file_id)
            chunks = _chunk_rows(conn, file_id)
            lines = [
                f"File: {path}",
                f"Name: {row['file_name']}",
                f"Extension: {row['extension']}",
                f"Indexed chunks: {row['chunk_count']}",
            ]
            if summary:
                lines.extend(
                    [
                        f"Cached summary: {summary['short_summary'] or ''}",
                        f"Key points: {summary['key_points'] or '[]'}",
                        f"Topics: {summary['detected_topics'] or '[]'}",
                        f"Actions: {summary['possible_actions'] or '[]'}",
                        f"Warnings: {summary['warnings'] or '[]'}",
                    ]
                )
            if chunks:
                snippet_lines = []
                for chunk in chunks:
                    label = f"Chunk {int(chunk['chunk_index']) + 1}"
                    if chunk["start_line"] is not None and chunk["end_line"] is not None:
                        label += f" lines {chunk['start_line']}-{chunk['end_line']}"
                    snippet_lines.append(f"{label}: {str(chunk['content'])[:900]}")
                lines.append("Indexed excerpts:\n" + "\n".join(snippet_lines))
            part = "\n".join(lines)
            if total + len(part) > MAX_BRIEFING_CONTEXT_CHARS:
                context_limited = True
                break
            parts.append(part)
            total += len(part)
    finally:
        conn.close()

    try:
        topics = []
        # Topic explorer is deterministic if the LLM cannot produce labels; it still uses indexed chunks only.
        # Avoid awaiting here; briefing mode already has enough indexed context, so direct topic explorer is optional.
        _ = topics
    except Exception:
        pass

    context = (
        f"Briefing type: {briefing_type}\n"
        f"Instructions: {_briefing_instructions(briefing_type)}\n\n"
        + "\n\n---\n\n".join(parts)
    )
    return context, list(dict.fromkeys(sources)), context_limited


def _insufficient_briefing(workspace: str, briefing_type: str, reason: str) -> BriefingResult:
    timestamp = _now()
    return BriefingResult(
        workspace=workspace,
        briefing_type=briefing_type,
        briefing_text=(
            f"Insufficient indexed evidence for a {briefing_type.replace('_', ' ')} briefing.\n\n"
            f"Reason: {reason}\n\n"
            "Next action: index this workspace or add more approved local files, then generate the briefing again."
        ),
        sources_used=[],
        files_considered=0,
        context_limited=False,
        generated_at=timestamp,
    )


async def generate_briefing(
    *,
    settings: Settings,
    workspace: str,
    briefing_type: str,
    llm: OllamaClient,
    limit_files: int = 30,
    refresh: bool = False,
    db_path: Path = DB_PATH,
) -> BriefingResult:
    settings_for_workspace_name(settings, workspace)
    normalized_type = _validate_briefing_type(briefing_type)
    bounded_limit = max(1, min(limit_files, 100))
    init_db(db_path)
    conn = _connect(db_path)
    try:
        files = _indexed_file_rows(conn, workspace, bounded_limit)
    finally:
        conn.close()
    if not files:
        return _insufficient_briefing(workspace, normalized_type, "No indexed files were found for this workspace.")
    if sum(int(row["chunk_count"] or 0) for row in files) == 0:
        return _insufficient_briefing(workspace, normalized_type, "Indexed files have no stored text chunks.")

    context, sources, context_limited = _build_context(
        settings=settings,
        workspace=workspace,
        briefing_type=normalized_type,
        files=files,
        db_path=db_path,
    )
    prompt = (
        "Use only the provided local indexed context.\n"
        "Do not invent files or facts.\n"
        "Cite sources using file paths.\n"
        "If evidence is weak, say so clearly.\n"
        "Keep the briefing practical and concise.\n"
        "Give next actions.\n\n"
        f"Workspace: {workspace}\n"
        f"Briefing type: {normalized_type}\n\n"
        f"Local context:\n{context}"
    )
    try:
        briefing_text = await llm.chat(prompt)
    except Exception:
        briefing_text = ""
    if not str(briefing_text).strip():
        briefing_text = (
            f"Briefing for {workspace} based on {len(files)} indexed file(s).\n\n"
            "Evidence was available, but the local model did not produce a briefing. "
            "Review the sources below and try again after confirming Ollama is running."
        )
    return BriefingResult(
        workspace=workspace,
        briefing_type=normalized_type,
        briefing_text=str(briefing_text).strip(),
        sources_used=sources,
        files_considered=len(files),
        context_limited=context_limited,
        generated_at=_now(),
    )


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower()).strip("_")
    return slug or "briefing"


def export_briefing(
    *,
    workspace: str,
    briefing_type: str,
    briefing_text: str,
    approved: bool,
    export_dir: Path = EXPORT_DIR,
) -> Path:
    if not approved:
        raise SafetyError("Briefing export requires explicit approval.")
    normalized_type = _validate_briefing_type(briefing_type)
    export_dir.mkdir(parents=True, exist_ok=True)
    base = export_dir.resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = f"{_safe_slug(workspace)}_{_safe_slug(normalized_type)}_briefing_{timestamp}"
    candidate = (base / f"{stem}.md").resolve()
    counter = 1
    while candidate.exists():
        candidate = (base / f"{stem}_{counter}.md").resolve()
        counter += 1
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SafetyError("Export path escaped the Jarvis exports folder.") from exc
    content = (
        f"# {workspace} {normalized_type.replace('_', ' ').title()} Briefing\n\n"
        f"Generated at: {_now()}\n\n"
        f"{briefing_text.strip()}\n"
    )
    candidate.write_text(content, encoding="utf-8")
    return candidate
