from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings
from backend.db import DB_PATH, init_db
from backend.llm import OllamaClient
from backend.safety import SafetyError, contains_blocked_keyword, is_under_root, max_read_bytes, validate_allowed_path


CHUNK_TARGET_CHARS = 2_000
MAX_CONTEXT_CHARS = 12_000


@dataclass
class IndexSummary:
    workspace: str
    scanned_file_count: int = 0
    indexed_file_count: int = 0
    skipped_unchanged_count: int = 0
    rejected_file_count: int = 0
    total_chunks_created: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KBSearchResult:
    path: str
    file_name: str
    snippet: str
    rank: float | None
    chunk_id: int
    file_id: int
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True)
class KBAnswer:
    answer: str
    sources_used: list[KBSearchResult]
    context_limited: bool


def settings_for_workspace_name(settings: Settings, workspace: str) -> Settings:
    if workspace not in settings.workspaces:
        raise SafetyError("Invalid workspace.")
    return settings.with_allowed_roots(settings.workspaces[workspace].roots)


def _is_hidden_or_blocked(path: Path, root: Path, settings: Settings) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    if any(part.startswith(".") for part in relative.parts):
        return True
    return contains_blocked_keyword(relative, settings.safety.blocked_path_keywords)


def _iter_workspace_files(settings: Settings) -> list[Path]:
    roots = settings.paths.expanded_roots()
    candidates: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                candidates.append(path)
    return candidates


def _validate_index_candidate(path: Path, settings: Settings) -> Path:
    validated = validate_allowed_path(path, settings, require_file=True)
    resolved = validated.resolved
    root = validated.root

    # Indexing is intentionally stricter than manual reads: hidden folders/files
    # and blocked names are never stored in the local knowledge base.
    if _is_hidden_or_blocked(resolved, root, settings):
        raise SafetyError("Path is hidden or blocked.")
    if resolved.suffix.lower() not in settings.safety.allowed_extensions:
        raise SafetyError("File extension is not allowed.")
    if not any(is_under_root(resolved, allowed_root) for allowed_root in settings.paths.expanded_roots()):
        raise SafetyError("Path escapes approved roots.")
    if resolved.stat().st_size > max_read_bytes(settings):
        raise SafetyError("File is larger than the configured indexing limit.")
    return resolved


def _read_index_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise SafetyError("File appears to be binary.")
    return data.decode("utf-8", errors="replace")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, target_chars: int = CHUNK_TARGET_CHARS) -> list[tuple[str, int | None, int | None]]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[tuple[str, int | None, int | None]] = []
    current: list[str] = []
    start_line: int | None = None

    for line_number, line in enumerate(lines, start=1):
        if start_line is None:
            start_line = line_number
        current.append(line)
        current_size = sum(len(item) + 1 for item in current)
        if current_size >= target_chars:
            chunks.append(("\n".join(current), start_line, line_number))
            current = []
            start_line = None

    if current:
        chunks.append(("\n".join(current), start_line, len(lines)))
    return chunks


def _existing_file(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, content_hash FROM indexed_files WHERE path = ?",
        (path,),
    ).fetchone()


def _replace_indexed_file(
    conn: sqlite3.Connection,
    *,
    workspace: str,
    path: Path,
    text: str,
    content_hash: str,
) -> int:
    stat = path.stat()
    now = datetime.now(timezone.utc).isoformat()
    existing = _existing_file(conn, str(path))
    if existing:
        file_id = int(existing["id"])
        conn.execute("DELETE FROM document_chunks_fts WHERE file_id = ?", (str(file_id),))
        conn.execute("DELETE FROM document_chunks WHERE file_id = ?", (file_id,))
        conn.execute(
            """
            UPDATE indexed_files
            SET workspace = ?, file_name = ?, extension = ?, size_bytes = ?,
                modified_time = ?, content_hash = ?, indexed_at = ?, status = ?
            WHERE id = ?
            """,
            (
                workspace,
                path.name,
                path.suffix.lower(),
                stat.st_size,
                stat.st_mtime,
                content_hash,
                now,
                "indexed",
                file_id,
            ),
        )
    else:
        cursor = conn.execute(
            """
            INSERT INTO indexed_files (
                workspace, path, file_name, extension, size_bytes,
                modified_time, content_hash, indexed_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace,
                str(path),
                path.name,
                path.suffix.lower(),
                stat.st_size,
                stat.st_mtime,
                content_hash,
                now,
                "indexed",
            ),
        )
        file_id = int(cursor.lastrowid)

    chunks = chunk_text(text)
    for chunk_index, (content, start_line, end_line) in enumerate(chunks):
        cursor = conn.execute(
            """
            INSERT INTO document_chunks (file_id, chunk_index, content, start_line, end_line)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_id, chunk_index, content, start_line, end_line),
        )
        chunk_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO document_chunks_fts (content, path, workspace, file_id, chunk_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (content, str(path), workspace, str(file_id), str(chunk_id)),
        )

    return len(chunks)


def index_workspace(settings: Settings, workspace: str, db_path: Path = DB_PATH) -> IndexSummary:
    scoped_settings = settings_for_workspace_name(settings, workspace)
    summary = IndexSummary(workspace=workspace)
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for candidate in _iter_workspace_files(scoped_settings):
            summary.scanned_file_count += 1
            try:
                path = _validate_index_candidate(candidate, scoped_settings)
                text = _read_index_text(path)
                content_hash = _content_hash(text)
                existing = _existing_file(conn, str(path))
                if existing and existing["content_hash"] == content_hash:
                    summary.skipped_unchanged_count += 1
                    continue
                chunks_created = _replace_indexed_file(
                    conn,
                    workspace=workspace,
                    path=path,
                    text=text,
                    content_hash=content_hash,
                )
                summary.indexed_file_count += 1
                summary.total_chunks_created += chunks_created
            except (OSError, SafetyError, UnicodeError) as exc:
                summary.rejected_file_count += 1
                summary.errors.append(f"{candidate}: {exc}")
        conn.commit()
    finally:
        conn.close()
    return summary


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w-]+", query)
    if not terms:
        raise SafetyError("Search query is required.")
    return " OR ".join(f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms)


def _fallback_answer(question: str, sources: list[KBSearchResult]) -> str:
    source = sources[0]
    line_text = ""
    if source.start_line is not None and source.end_line is not None:
        line_text = f" lines {source.start_line}-{source.end_line}"
    return (
        f"I found a matching local note in {source.path}{line_text}: "
        f"{source.snippet}"
    )


def _needs_grounded_fallback(answer: str) -> bool:
    lowered = answer.lower()
    return (
        "not found" in lowered
        or "cannot provide" in lowered
        or "official documentation" in lowered
        or "research papers" in lowered
    )


def search_kb(
    *,
    query: str,
    workspace: str,
    settings: Settings,
    limit: int = 10,
    db_path: Path = DB_PATH,
) -> list[KBSearchResult]:
    settings_for_workspace_name(settings, workspace)
    bounded_limit = max(1, min(limit, 50))
    init_db(db_path)
    fts_query = _fts_query(query)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                f.path,
                files.file_name,
                c.content,
                bm25(document_chunks_fts) AS rank,
                c.id AS chunk_id,
                files.id AS file_id,
                c.start_line,
                c.end_line
            FROM document_chunks_fts AS f
            JOIN document_chunks AS c ON c.id = CAST(f.chunk_id AS INTEGER)
            JOIN indexed_files AS files ON files.id = c.file_id
            WHERE document_chunks_fts MATCH ?
              AND f.workspace = ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, workspace, bounded_limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        KBSearchResult(
            path=str(row["path"]),
            file_name=str(row["file_name"]),
            snippet=str(row["content"])[:700],
            rank=float(row["rank"]) if row["rank"] is not None else None,
            chunk_id=int(row["chunk_id"]),
            file_id=int(row["file_id"]),
            start_line=int(row["start_line"]) if row["start_line"] is not None else None,
            end_line=int(row["end_line"]) if row["end_line"] is not None else None,
        )
        for row in rows
    ]


async def ask_kb(
    *,
    question: str,
    workspace: str,
    settings: Settings,
    llm: OllamaClient,
    limit: int = 5,
    db_path: Path = DB_PATH,
) -> KBAnswer:
    results = search_kb(query=question, workspace=workspace, settings=settings, limit=limit, db_path=db_path)
    context_parts: list[str] = []
    used_results: list[KBSearchResult] = []
    total_chars = 0
    context_limited = False

    for result in results:
        source_header = f"Source: {result.path}"
        if result.start_line is not None and result.end_line is not None:
            source_header += f" lines {result.start_line}-{result.end_line}"
        part = f"{source_header}\n{result.snippet}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            context_limited = True
            break
        context_parts.append(part)
        used_results.append(result)
        total_chars += len(part)

    if not context_parts:
        return KBAnswer(
            answer="I could not find an answer in the indexed local knowledge base.",
            sources_used=[],
            context_limited=False,
        )

    prompt = (
        "Use only the provided local context to answer the question.\n"
        "If the answer is not found in the context, say that it was not found.\n"
        "Cite file paths from the retrieved chunks when you answer.\n"
        "Do not pretend to know files that were not retrieved.\n\n"
        "The retrieved chunks below are local search results. If the user asks where something was written, "
        "answer with the matching source path and a brief explanation from the chunk.\n\n"
        f"Question: {question}\n\n"
        "Local context:\n"
        + "\n\n---\n\n".join(context_parts)
    )
    answer = await llm.chat(prompt)
    if used_results and _needs_grounded_fallback(answer):
        answer = _fallback_answer(question, used_results)
    return KBAnswer(answer=answer, sources_used=used_results, context_limited=context_limited)
