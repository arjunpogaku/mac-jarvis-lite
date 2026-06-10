from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "jarvis.sqlite"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                user_message TEXT,
                assistant_response TEXT,
                tool_name TEXT,
                tool_status TEXT NOT NULL,
                workspace TEXT
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tool_logs)")}
        if "workspace" not in columns:
            conn.execute("ALTER TABLE tool_logs ADD COLUMN workspace TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                modified_time REAL NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                FOREIGN KEY(file_id) REFERENCES indexed_files(id)
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts
            USING fts5(
                content,
                path UNINDEXED,
                workspace UNINDEXED,
                file_id UNINDEXED,
                chunk_id UNINDEXED
            )
            """
        )


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_interaction(
    *,
    session_id: str | None,
    user_message: str | None,
    assistant_response: str | None,
    tool_name: str | None,
    tool_status: str,
    workspace: str | None = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tool_logs (
                timestamp, session_id, user_message, assistant_response, tool_name, tool_status, workspace
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, session_id, user_message, assistant_response, tool_name, tool_status, workspace),
        )
