from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings
from backend.db import DB_PATH, init_db
from backend.llm import OllamaClient
from backend.safety import SafetyError
from backend.tools.indexer import SemanticSearchOutcome, search_kb, semantic_search_kb, settings_for_workspace_name


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "data",
    "did",
    "do",
    "does",
    "done",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "here",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "like",
    "may",
    "more",
    "most",
    "no",
    "not",
    "note",
    "notes",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "over",
    "project",
    "research",
    "should",
    "so",
    "some",
    "summary",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "under",
    "up",
    "use",
    "used",
    "using",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
}

CODE_EXTENSIONS = {
    ".c",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
}

CONFIG_EXTENSIONS = {".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"}
NOTE_EXTENSIONS = {".md", ".rst", ".tex", ".txt"}
DATA_EXTENSIONS = {".csv", ".tsv"}
LOG_EXTENSIONS = {".log"}


@dataclass(frozen=True)
class InventoryFile:
    path: str
    file_name: str
    extension: str
    size_bytes: int
    modified_time: str
    chunks: int
    category: str


@dataclass(frozen=True)
class FileCard:
    file_id: int
    path: str
    file_name: str
    extension: str
    source_chunk_count: int
    short_summary: str
    key_points: list[str]
    detected_topics: list[str]
    possible_actions: list[str]
    warnings: list[str]
    model: str
    generated_at: str


@dataclass(frozen=True)
class WorkspaceSummary:
    workspace: str
    indexed_file_count: int
    summary: str
    main_topics: list[str]
    important_files: list[str]
    possible_actions: list[str]
    warnings: list[str]
    model: str
    generated_at: str


@dataclass(frozen=True)
class TopicItem:
    topic_label: str
    related_keywords: list[str]
    supporting_files: list[str]
    example_snippets: list[str]


@dataclass(frozen=True)
class HealthIssue:
    path: str
    issue_type: str
    severity: str
    explanation: str
    suggested_next_action: str


@dataclass(frozen=True)
class ParsedFileSummary:
    short_summary: str
    key_points: list[str]
    detected_topics: list[str]
    possible_actions: list[str]
    warnings: list[str]


def _connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [item.strip() for item in value.splitlines() if item.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    return []


def _dump_list(values: list[str]) -> str:
    return json.dumps([value for value in (item.strip() for item in values) if value])


def _hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _category_for_file(path: str, extension: str) -> str:
    lowered = path.lower()
    extension = extension.lower()
    if extension in LOG_EXTENSIONS or "log" in lowered:
        return "logs"
    if extension in CONFIG_EXTENSIONS or "config" in lowered or lowered.endswith(".env"):
        return "config"
    if extension in DATA_EXTENSIONS or any(part in lowered for part in ["data", "dataset", "results", "csv"]):
        return "data"
    if extension in CODE_EXTENSIONS or any(part in lowered for part in ["src", "backend", "frontend", "code", "app", "lib"]):
        return "code"
    if any(part in lowered for part in ["research", "paper", "note", "notes", "docs", "document"]) or extension in NOTE_EXTENSIONS:
        return "notes"
    if any(part in lowered for part in ["research", "paper", "experiment", "analysis"]) or extension in {".pdf"}:
        return "research"
    return "unknown"


def _topic_seed(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
    return [word for word in words if len(word) >= 4 and word not in STOPWORDS]


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 5 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _parse_llm_json(text: str) -> ParsedFileSummary | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    short_summary = str(payload.get("short_summary", "")).strip()
    if not short_summary:
        return None

    def to_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value is None:
            return []
        return _json_list(str(value))

    return ParsedFileSummary(
        short_summary=short_summary,
        key_points=to_list(payload.get("key_points")),
        detected_topics=to_list(payload.get("detected_topics")),
        possible_actions=to_list(payload.get("possible_actions")),
        warnings=to_list(payload.get("warnings")),
    )


def _fallback_file_summary(path: str, chunk_count: int, snippets: list[str]) -> ParsedFileSummary:
    if chunk_count == 0:
        return ParsedFileSummary(
            short_summary="No indexed text was available for this file.",
            key_points=["The file was indexed but no text chunks were stored."],
            detected_topics=["empty or tiny file"],
            possible_actions=["Check whether the file should contain content."],
            warnings=["No indexed chunks were available."],
        )
    snippets_text = " ".join(snippets).strip()
    topic_words = [_stem(word) for word in _topic_seed(snippets_text)]
    top_topics = list(dict.fromkeys(topic_words[:5])) or ["unclear content"]
    return ParsedFileSummary(
        short_summary=f"Indexed content from {chunk_count} chunk(s) was available, but the local model did not return a structured summary.",
        key_points=[f"Indexed chunks were available for {path}."] + ([snippets[0][:120]] if snippets else []),
        detected_topics=top_topics,
        possible_actions=["Regenerate the file card with a stronger local model if needed."],
        warnings=["Summary was generated from a fallback path."],
    )


def _query_indexed_files(conn: sqlite3.Connection, workspace: str, limit: int | None = None) -> list[sqlite3.Row]:
    query = (
        "SELECT files.*, COUNT(ch.id) AS chunk_count "
        "FROM indexed_files AS files "
        "LEFT JOIN document_chunks AS ch ON ch.file_id = files.id "
        "WHERE files.workspace = ? "
        "GROUP BY files.id "
        "ORDER BY files.modified_time DESC, files.path ASC"
    )
    params: list[object] = [workspace]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(query, params).fetchall())


def _query_file_chunks(conn: sqlite3.Connection, file_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT content, start_line, end_line, chunk_index FROM document_chunks WHERE file_id = ? ORDER BY chunk_index ASC",
            (file_id,),
        ).fetchall()
    )


def _query_workspace_chunks(conn: sqlite3.Connection, workspace: str, limit: int | None = None) -> list[sqlite3.Row]:
    query = (
        "SELECT files.path, files.file_name, files.extension, files.id AS file_id, ch.content, ch.chunk_index "
        "FROM document_chunks AS ch "
        "JOIN indexed_files AS files ON files.id = ch.file_id "
        "WHERE files.workspace = ? "
        "ORDER BY files.modified_time DESC, ch.chunk_index ASC"
    )
    params: list[object] = [workspace]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return list(conn.execute(query, params).fetchall())


def _file_card_prompt(path: str, file_name: str, chunks: list[sqlite3.Row]) -> str:
    excerpt_parts: list[str] = []
    total_chars = 0
    for row in chunks:
        header = f"Chunk {row['chunk_index'] + 1}"
        if row["start_line"] is not None and row["end_line"] is not None:
            header += f" lines {row['start_line']}-{row['end_line']}"
        content = str(row["content"])
        part = f"{header}:\n{content}"
        if total_chars + len(part) > 8000:
            break
        excerpt_parts.append(part)
        total_chars += len(part)
    return (
        "Use only the provided indexed chunks. Do not invent details. If the content is unclear, say so. "
        "Keep output concise. Return JSON with keys short_summary, key_points, detected_topics, possible_actions, warnings.\n\n"
        f"File: {path}\n"
        f"Name: {file_name}\n\n"
        "Indexed chunks:\n"
        + "\n\n---\n\n".join(excerpt_parts)
    )


def _store_file_summary(
    conn: sqlite3.Connection,
    *,
    file_id: int,
    workspace: str,
    path: str,
    content_hash: str,
    model: str,
    summary: ParsedFileSummary,
) -> None:
    timestamp = _now()
    conn.execute(
        """
        INSERT INTO file_summaries (
            file_id, workspace, path, content_hash, short_summary, key_points, detected_topics,
            possible_actions, warnings, model, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            workspace=excluded.workspace,
            path=excluded.path,
            content_hash=excluded.content_hash,
            short_summary=excluded.short_summary,
            key_points=excluded.key_points,
            detected_topics=excluded.detected_topics,
            possible_actions=excluded.possible_actions,
            warnings=excluded.warnings,
            model=excluded.model,
            updated_at=excluded.updated_at
        """,
        (
            file_id,
            workspace,
            path,
            content_hash,
            summary.short_summary,
            _dump_list(summary.key_points),
            _dump_list(summary.detected_topics),
            _dump_list(summary.possible_actions),
            _dump_list(summary.warnings),
            model,
            timestamp,
            timestamp,
        ),
    )


def _load_file_summary(conn: sqlite3.Connection, file_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM file_summaries WHERE file_id = ?", (file_id,)).fetchone()


def _parse_file_summary_row(row: sqlite3.Row) -> ParsedFileSummary:
    return ParsedFileSummary(
        short_summary=str(row["short_summary"] or ""),
        key_points=_json_list(str(row["key_points"] or "")),
        detected_topics=_json_list(str(row["detected_topics"] or "")),
        possible_actions=_json_list(str(row["possible_actions"] or "")),
        warnings=_json_list(str(row["warnings"] or "")),
    )


def _summary_from_llm(response: str) -> ParsedFileSummary | None:
    parsed = _parse_llm_json(response)
    if not parsed:
        return None
    return ParsedFileSummary(
        short_summary=parsed.short_summary,
        key_points=parsed.key_points[:5],
        detected_topics=parsed.detected_topics[:5],
        possible_actions=parsed.possible_actions[:5],
        warnings=parsed.warnings[:5],
    )


async def _make_file_summary(
    llm: OllamaClient,
    path: str,
    file_name: str,
    chunks: list[sqlite3.Row],
) -> ParsedFileSummary:
    prompt = _file_card_prompt(path, file_name, chunks)
    try:
        response = await llm.chat(prompt)
    except Exception:
        response = ""
    parsed = _summary_from_llm(str(response))
    if parsed:
        return parsed
    snippets = [str(row["content"])[:180] for row in chunks[:3]]
    return _fallback_file_summary(path, len(chunks), snippets)


async def generate_file_cards(
    *,
    settings: Settings,
    workspace: str,
    llm: OllamaClient,
    limit: int = 20,
    refresh: bool = False,
    db_path: Path = DB_PATH,
) -> list[FileCard]:
    settings_for_workspace_name(settings, workspace)
    bounded_limit = max(1, min(limit, 100))
    init_db(db_path)
    cards: list[FileCard] = []
    conn = _connection(db_path)
    try:
        files = _query_indexed_files(conn, workspace, bounded_limit)
        if not files:
            raise SafetyError("This workspace has not been indexed yet. Please index it from the Knowledge Base page first.")
        for row in files:
            file_id = int(row["id"])
            path = str(row["path"])
            content_hash = str(row["content_hash"])
            cached = _load_file_summary(conn, file_id)
            if cached and not refresh and str(cached["content_hash"]) == content_hash:
                parsed = _parse_file_summary_row(cached)
                model = str(cached["model"])
            else:
                chunks = _query_file_chunks(conn, file_id)
                parsed = await _make_file_summary(llm, path, str(row["file_name"]), chunks)
                model = settings.llm.model
                _store_file_summary(
                    conn,
                    file_id=file_id,
                    workspace=workspace,
                    path=path,
                    content_hash=content_hash,
                    model=model,
                    summary=parsed,
                )
            cards.append(
                FileCard(
                    file_id=file_id,
                    path=path,
                    file_name=str(row["file_name"]),
                    extension=str(row["extension"]),
                    source_chunk_count=int(row["chunk_count"] or 0),
                    short_summary=parsed.short_summary,
                    key_points=parsed.key_points,
                    detected_topics=parsed.detected_topics,
                    possible_actions=parsed.possible_actions,
                    warnings=parsed.warnings,
                    model=model,
                    generated_at=_now(),
                )
            )
        conn.commit()
    finally:
        conn.close()
    return cards


def _workspace_summary_source_hash(rows: list[sqlite3.Row]) -> str:
    payload = "|".join(f"{row['file_id']}:{row['content_hash']}" for row in rows)
    return _hash_source(payload)


def _workspace_summary_prompt(
    workspace: str,
    considered_rows: list[sqlite3.Row],
    total_indexed_files: int,
    total_chunks: int,
) -> str:
    lines = [
        "Use only the provided file summaries and metadata. Do not invent details. Keep output concise.",
        "Return JSON with keys summary, main_topics, important_files, possible_actions, warnings.",
        f"Workspace: {workspace}",
        f"Indexed files considered: {len(considered_rows)} of {total_indexed_files}",
        f"Total chunks in workspace: {total_chunks}",
    ]
    for row in considered_rows:
        lines.append(
            "\n".join(
                [
                    f"File: {row['path']}",
                    f"Summary: {row['short_summary'] or 'No summary available.'}",
                    f"Key points: {row['key_points'] or '[]'}",
                    f"Topics: {row['detected_topics'] or '[]'}",
                    f"Actions: {row['possible_actions'] or '[]'}",
                    f"Warnings: {row['warnings'] or '[]'}",
                ]
            )
        )
    if len(considered_rows) < total_indexed_files:
        lines.append("This workspace summary covers only part of the indexed workspace.")
    return "\n\n".join(lines)


def _parse_workspace_summary(text: str) -> dict[str, list[str] | str] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        return None
    return {
        "summary": summary,
        "main_topics": [str(item).strip() for item in payload.get("main_topics", []) if str(item).strip()],
        "important_files": [str(item).strip() for item in payload.get("important_files", []) if str(item).strip()],
        "possible_actions": [str(item).strip() for item in payload.get("possible_actions", []) if str(item).strip()],
        "warnings": [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()],
    }


def _workspace_summary_fallback(
    workspace: str,
    considered_rows: list[sqlite3.Row],
    total_indexed_files: int,
) -> WorkspaceSummary:
    important_files = [str(row["path"]) for row in considered_rows[:5]]
    warnings = []
    if len(considered_rows) < total_indexed_files:
        warnings.append("Only part of the indexed workspace was considered.")
    main_topics: list[str] = []
    for row in considered_rows:
        if "detected_topics" in row.keys():
            main_topics.extend(_json_list(str(row["detected_topics"] or "")))
    return WorkspaceSummary(
        workspace=workspace,
        indexed_file_count=len(considered_rows),
        summary=f"Workspace intelligence summary is based on {len(considered_rows)} indexed file(s) in {workspace}.",
        main_topics=list(dict.fromkeys(main_topics))[:8],
        important_files=important_files,
        possible_actions=["Regenerate file cards for more context."] if considered_rows else [],
        warnings=warnings,
        model="fallback",
        generated_at=_now(),
    )


async def generate_workspace_summary(
    *,
    settings: Settings,
    workspace: str,
    llm: OllamaClient,
    refresh: bool = False,
    limit_files: int = 30,
    db_path: Path = DB_PATH,
) -> WorkspaceSummary:
    settings_for_workspace_name(settings, workspace)
    bounded_limit = max(1, min(limit_files, 100))
    init_db(db_path)
    conn = _connection(db_path)
    try:
        all_rows = _query_indexed_files(conn, workspace)
        total_indexed_files = len(all_rows)
        if total_indexed_files == 0:
            raise SafetyError("This workspace has not been indexed yet. Please index it from the Knowledge Base page first.")
        considered_rows = all_rows[:bounded_limit]
        for row in considered_rows:
            cached = _load_file_summary(conn, int(row["id"]))
            if cached and str(cached["content_hash"]) == str(row["content_hash"]):
                continue
            chunks = _query_file_chunks(conn, int(row["id"]))
            parsed = await _make_file_summary(llm, str(row["path"]), str(row["file_name"]), chunks)
            _store_file_summary(
                conn,
                file_id=int(row["id"]),
                workspace=workspace,
                path=str(row["path"]),
                content_hash=str(row["content_hash"]),
                model=settings.llm.model,
                summary=parsed,
            )
        conn.commit()
        summary_rows = []
        for row in considered_rows:
            cached = _load_file_summary(conn, int(row["id"]))
            if cached is None:
                continue
            summary_rows.append(cached)
        if not summary_rows:
            return _workspace_summary_fallback(workspace, considered_rows, total_indexed_files)

        source_hash = _workspace_summary_source_hash(summary_rows)
        cached_summary = conn.execute(
            """
            SELECT * FROM workspace_summaries
            WHERE workspace = ? AND indexed_file_count = ? AND limit_files = ? AND source_hash = ?
            ORDER BY id DESC LIMIT 1
            """,
            (workspace, len(considered_rows), bounded_limit, source_hash),
        ).fetchone()
        if cached_summary and not refresh:
            return WorkspaceSummary(
                workspace=workspace,
                indexed_file_count=int(cached_summary["indexed_file_count"]),
                summary=str(cached_summary["summary"]),
                main_topics=_json_list(str(cached_summary["main_topics"] or "")),
                important_files=_json_list(str(cached_summary["important_files"] or "")),
                possible_actions=_json_list(str(cached_summary["possible_actions"] or "")),
                warnings=_json_list(str(cached_summary["warnings"] or "")),
                model=str(cached_summary["model"]),
                generated_at=str(cached_summary["created_at"]),
            )

        prompt = _workspace_summary_prompt(workspace, summary_rows, total_indexed_files, sum(int(row["chunk_count"] or 0) for row in all_rows))
        try:
            response = await llm.chat(prompt)
        except Exception:
            response = ""
        parsed = _parse_workspace_summary(str(response))
        if not parsed:
            fallback = _workspace_summary_fallback(workspace, considered_rows, total_indexed_files)
            conn.execute(
                """
                INSERT INTO workspace_summaries (
                    workspace, indexed_file_count, limit_files, source_hash, summary, main_topics,
                    important_files, possible_actions, warnings, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace,
                    len(considered_rows),
                    bounded_limit,
                    source_hash,
                    fallback.summary,
                    _dump_list(fallback.main_topics),
                    _dump_list(fallback.important_files),
                    _dump_list(fallback.possible_actions),
                    _dump_list(fallback.warnings),
                    fallback.model,
                    fallback.generated_at,
                ),
            )
            conn.commit()
            return fallback

        result = WorkspaceSummary(
            workspace=workspace,
            indexed_file_count=len(considered_rows),
            summary=str(parsed["summary"]),
            main_topics=list(parsed["main_topics"]),
            important_files=list(parsed["important_files"]),
            possible_actions=list(parsed["possible_actions"]),
            warnings=list(parsed["warnings"]),
            model=settings.llm.model,
            generated_at=_now(),
        )
        conn.execute(
            """
            INSERT INTO workspace_summaries (
                workspace, indexed_file_count, limit_files, source_hash, summary, main_topics,
                important_files, possible_actions, warnings, model, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace,
                len(considered_rows),
                bounded_limit,
                source_hash,
                result.summary,
                _dump_list(result.main_topics),
                _dump_list(result.important_files),
                _dump_list(result.possible_actions),
                _dump_list(result.warnings),
                result.model,
                result.generated_at,
            ),
        )
        conn.commit()
        return result
    finally:
        conn.close()


async def _topic_label_from_keywords(llm: OllamaClient | None, label_seed: str, keywords: list[str]) -> str:
    if llm is None:
        return label_seed
    prompt = (
        "Label this topic using only the provided keywords. Do not invent content. Keep the label short.\n"
        f"Keywords: {', '.join(keywords[:8])}\n"
        f"Seed: {label_seed}"
    )
    try:
        response = await llm.chat(prompt)
        if response.strip():
            return response.strip().splitlines()[0][:80]
    except Exception:
        pass
    return label_seed


async def explore_topics(
    *,
    settings: Settings,
    workspace: str,
    limit: int = 20,
    db_path: Path = DB_PATH,
    llm: OllamaClient | None = None,
) -> list[TopicItem]:
    settings_for_workspace_name(settings, workspace)
    bounded_limit = max(1, min(limit, 20))
    init_db(db_path)
    conn = _connection(db_path)
    try:
        rows = _query_workspace_chunks(conn, workspace, 500)
        if not rows:
            raise SafetyError("This workspace has not been indexed yet. Please index it from the Knowledge Base page first.")
        stem_counts: Counter[str] = Counter()
        variants: defaultdict[str, Counter[str]] = defaultdict(Counter)
        supporting_files: defaultdict[str, Counter[str]] = defaultdict(Counter)
        snippets: defaultdict[str, list[str]] = defaultdict(list)
        for row in rows:
            content = str(row["content"])
            words = _topic_seed(content)
            for word in words:
                stem = _stem(word)
                stem_counts[stem] += 1
                variants[stem][word] += 1
                supporting_files[stem][str(row["path"])] += 1
                if len(snippets[stem]) < 3:
                    snippets[stem].append(content[:180])
        topics: list[TopicItem] = []
        for stem, _count in stem_counts.most_common(bounded_limit):
            keyword_list = [item for item, _freq in variants[stem].most_common(6)]
            label = await _topic_label_from_keywords(llm, stem.replace("_", " "), keyword_list)
            if settings.embeddings.enabled and llm is not None:
                outcome: SemanticSearchOutcome = semantic_search_kb(
                    query=label,
                    workspace=workspace,
                    settings=settings,
                    limit=3,
                    db_path=db_path,
                    fallback_to_keyword=True,
                )
                for result in outcome.results:
                    supporting_files[stem][result.path] += 1
                    if len(snippets[stem]) < 5:
                        snippets[stem].append(result.snippet)
            topics.append(
                TopicItem(
                    topic_label=label[:80],
                    related_keywords=keyword_list,
                    supporting_files=[path for path, _freq in supporting_files[stem].most_common(5)],
                    example_snippets=snippets[stem][:3],
                )
            )
        return topics
    finally:
        conn.close()


def _severity_rank(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)


def _classify_health_issue(issue_type: str, severity: str, explanation: str, suggested_next_action: str, path: str) -> HealthIssue:
    return HealthIssue(
        path=path,
        issue_type=issue_type,
        severity=severity,
        explanation=explanation,
        suggested_next_action=suggested_next_action,
    )


def _has_todo(text: str) -> bool:
    lowered = text.lower()
    return "todo" in lowered or "fixme" in lowered


def _is_repeated(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    line_ratio = len(set(lines)) / max(1, len(lines))
    words = _topic_seed(text)
    word_ratio = len(set(words)) / max(1, len(words)) if words else 1.0
    return line_ratio < 0.5 or word_ratio < 0.3


def _looks_like_title(line: str) -> bool:
    lowered = line.strip().lower()
    return bool(lowered) and (
        lowered.startswith("#")
        or lowered.startswith("title:")
        or lowered.startswith("summary:")
        or lowered.startswith("overview:")
        or len(line.strip()) <= 80
    )


def _weak_summary(summary_row: sqlite3.Row | None) -> bool:
    if summary_row is None:
        return True
    summary = str(summary_row["short_summary"] or "").strip().lower()
    warnings = _json_list(str(summary_row["warnings"] or ""))
    if not summary:
        return True
    if any(term in summary for term in ["unclear", "not enough", "fallback"]):
        return True
    return len(warnings) >= 2


def check_workspace_health(
    *,
    settings: Settings,
    workspace: str,
    limit: int = 30,
    db_path: Path = DB_PATH,
) -> list[HealthIssue]:
    settings_for_workspace_name(settings, workspace)
    bounded_limit = max(1, min(limit, 100))
    init_db(db_path)
    conn = _connection(db_path)
    try:
        rows = _query_indexed_files(conn, workspace, bounded_limit)
        if not rows:
            raise SafetyError("This workspace has not been indexed yet. Please index it from the Knowledge Base page first.")
        issues: list[HealthIssue] = []
        for row in rows:
            file_id = int(row["id"])
            path = str(row["path"])
            extension = str(row["extension"])
            size_bytes = int(row["size_bytes"])
            category = _category_for_file(path, extension)
            summary_row = _load_file_summary(conn, file_id)
            chunks = _query_file_chunks(conn, file_id)
            chunk_text = "\n".join(str(chunk["content"]) for chunk in chunks)
            first_non_empty = next((line for line in chunk_text.splitlines() if line.strip()), "")

            if size_bytes == 0 or (size_bytes < 60 and int(row["chunk_count"] or 0) <= 1):
                issues.append(
                    _classify_health_issue(
                        "tiny_or_empty",
                        "high",
                        "The file is empty or so small that it may not carry meaningful indexed content.",
                        "Confirm the file should exist here or add the missing content.",
                        path,
                    )
                )
            if size_bytes >= max(1, int(settings.safety.max_file_read_kb * 1024 * 0.8)):
                issues.append(
                    _classify_health_issue(
                        "very_long_file",
                        "medium",
                        "This file is close to the configured local indexing limit and may be cumbersome to review.",
                        "Consider splitting the file into smaller sections or adding a shorter summary note.",
                        path,
                    )
                )
            if _has_todo(chunk_text):
                severity = "high" if category == "code" else "medium"
                issues.append(
                    _classify_health_issue(
                        "todo_fixme",
                        severity,
                        "The indexed content contains TODO or FIXME markers.",
                        "Resolve the TODO/FIXME items or rewrite them into a concrete next step.",
                        path,
                    )
                )
            if _is_repeated(chunk_text):
                issues.append(
                    _classify_health_issue(
                        "repeated_content",
                        "medium",
                        "The file looks repetitive or has low variation in the indexed text.",
                        "Remove duplicated sections or split repeated material into a smaller note.",
                        path,
                    )
                )
            if category in {"notes", "research"} and first_non_empty and not _looks_like_title(first_non_empty):
                issues.append(
                    _classify_health_issue(
                        "missing_title",
                        "low",
                        "The file does not appear to start with a title-like line.",
                        "Add a short heading or title line at the top of the document.",
                        path,
                    )
                )
            if _weak_summary(summary_row):
                issues.append(
                    _classify_health_issue(
                        "weak_summary",
                        "low",
                        "The cached file summary is missing or too vague.",
                        "Regenerate the file card after clarifying the source content.",
                        path,
                    )
                )
            if category == "config":
                issues.append(
                    _classify_health_issue(
                        "config_file",
                        "low",
                        "This looks like a configuration file, so its purpose may be implicit rather than documented.",
                        "Consider adding a short comment block or nearby note that explains the config purpose.",
                        path,
                    )
                )
        issues.sort(key=lambda item: (_severity_rank(item.severity), item.path, item.issue_type))
        return issues[: max(1, bounded_limit * 2)]
    finally:
        conn.close()


def inventory_workspace(
    *,
    settings: Settings,
    workspace: str,
    limit: int = 100,
    db_path: Path = DB_PATH,
) -> dict[str, object]:
    settings_for_workspace_name(settings, workspace)
    bounded_limit = max(1, min(limit, 500))
    init_db(db_path)
    conn = _connection(db_path)
    try:
        rows = _query_indexed_files(conn, workspace)
        total_files = len(rows)
        total_chunks = int(conn.execute("SELECT COUNT(*) FROM document_chunks AS ch JOIN indexed_files AS files ON files.id = ch.file_id WHERE files.workspace = ?", (workspace,)).fetchone()[0])
        extension_counts = Counter(str(row["extension"]) for row in rows)
        category_counts = Counter(_category_for_file(str(row["path"]), str(row["extension"])) for row in rows)
        inventory_rows = rows[:bounded_limit]
        files = [
            InventoryFile(
                path=str(row["path"]),
                file_name=str(row["file_name"]),
                extension=str(row["extension"]),
                size_bytes=int(row["size_bytes"]),
                modified_time=datetime.fromtimestamp(float(row["modified_time"]), tz=timezone.utc).isoformat(),
                chunks=int(row["chunk_count"] or 0),
                category=_category_for_file(str(row["path"]), str(row["extension"])),
            )
            for row in inventory_rows
        ]
        largest_files = [
            InventoryFile(
                path=str(row["path"]),
                file_name=str(row["file_name"]),
                extension=str(row["extension"]),
                size_bytes=int(row["size_bytes"]),
                modified_time=datetime.fromtimestamp(float(row["modified_time"]), tz=timezone.utc).isoformat(),
                chunks=int(row["chunk_count"] or 0),
                category=_category_for_file(str(row["path"]), str(row["extension"])),
            )
            for row in sorted(rows, key=lambda item: int(item["size_bytes"]), reverse=True)[:5]
        ]
        recent_files = [
            InventoryFile(
                path=str(row["path"]),
                file_name=str(row["file_name"]),
                extension=str(row["extension"]),
                size_bytes=int(row["size_bytes"]),
                modified_time=datetime.fromtimestamp(float(row["modified_time"]), tz=timezone.utc).isoformat(),
                chunks=int(row["chunk_count"] or 0),
                category=_category_for_file(str(row["path"]), str(row["extension"])),
            )
            for row in rows[:5]
        ]
        most_chunks = [
            InventoryFile(
                path=str(row["path"]),
                file_name=str(row["file_name"]),
                extension=str(row["extension"]),
                size_bytes=int(row["size_bytes"]),
                modified_time=datetime.fromtimestamp(float(row["modified_time"]), tz=timezone.utc).isoformat(),
                chunks=int(row["chunk_count"] or 0),
                category=_category_for_file(str(row["path"]), str(row["extension"])),
            )
            for row in sorted(rows, key=lambda item: int(item["chunk_count"] or 0), reverse=True)[:5]
        ]
        last_indexed = conn.execute(
            "SELECT MAX(indexed_at) FROM indexed_files WHERE workspace = ?",
            (workspace,),
        ).fetchone()[0]
        return {
            "workspace": workspace,
            "total_indexed_files": total_files,
            "total_chunks": total_chunks,
            "last_indexed_time": last_indexed,
            "extensions_breakdown": [{"extension": ext, "count": count} for ext, count in extension_counts.most_common()],
            "category_breakdown": [{"category": category, "count": count} for category, count in category_counts.most_common()],
            "files": [item.__dict__ for item in files],
            "largest_files": [item.__dict__ for item in largest_files],
            "recently_modified_files": [item.__dict__ for item in recent_files],
            "files_with_most_chunks": [item.__dict__ for item in most_chunks],
        }
    finally:
        conn.close()
