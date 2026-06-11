from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
import httpx

from backend.config import AppConfig, EmbeddingsConfig, KnowledgeBaseConfig, LLMConfig, PathsConfig, SafetyConfig, Settings, WorkspaceConfig
from backend.db import init_db
from backend.main import app
from backend.tools.indexer import index_workspace
from backend.tools.workspace_intelligence import (
    check_workspace_health,
    explore_topics,
    generate_file_cards,
    generate_workspace_summary,
    inventory_workspace,
)


def make_settings(tmp_path: Path) -> Settings:
    research = tmp_path / "research"
    jobs = tmp_path / "jobs"
    research.mkdir()
    jobs.mkdir()
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.5.0"),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(enabled=False),
        knowledge_base=KnowledgeBaseConfig(semantic_search_enabled=False, hybrid_search_enabled=False),
        safety=SafetyConfig(
            shell_enabled=False,
            max_file_read_kb=10,
            allowed_extensions=[".txt", ".md", ".py"],
            blocked_path_keywords=[".ssh", ".env", "private", "Passwords", "Library/Keychains"],
        ),
        paths=PathsConfig(allowed_roots=[str(research), str(jobs)]),
        workspaces={
            "research": WorkspaceConfig(description="Research", roots=[str(research)]),
            "jobs": WorkspaceConfig(description="Jobs", roots=[str(jobs)]),
        },
    )


class FakeAsyncLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    async def chat(self, message: str) -> str:
        self.calls.append(message)
        return self.response


class CountingLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def chat(self, message: str) -> str:
        self.calls.append(message)
        payload = {
            "short_summary": f"summary {len(self.calls)}",
            "key_points": ["point"],
            "detected_topics": ["topic"],
            "possible_actions": ["action"],
            "warnings": [],
        }
        return json.dumps(payload)


def index_paths(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [row[0] for row in conn.execute("SELECT path FROM indexed_files ORDER BY path")]


def insert_file_summary(db_path: Path, file_id: int, workspace: str, path: str, content_hash: str, summary: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO file_summaries (
                file_id, workspace, path, content_hash, short_summary, key_points,
                detected_topics, possible_actions, warnings, model, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (
                file_id,
                workspace,
                path,
                content_hash,
                summary,
                json.dumps(["cached point"]),
                json.dumps(["cached topic"]),
                json.dumps(["cached action"]),
                json.dumps([]),
                "test-model",
            ),
        )
        conn.commit()


def test_inventory_only_returns_selected_workspace_data(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    (tmp_path / "research" / "paper.md").write_text("research note", encoding="utf-8")
    (tmp_path / "jobs" / "resume.md").write_text("job note", encoding="utf-8")
    index_workspace(settings, "research", db_path)
    index_workspace(settings, "jobs", db_path)

    inventory = inventory_workspace(settings=settings, workspace="research", db_path=db_path)

    assert inventory["workspace"] == "research"
    assert inventory["total_indexed_files"] == 1
    assert all("research" in item["path"] for item in inventory["files"])


def test_inventory_does_not_include_another_workspace(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    (tmp_path / "research" / "paper.md").write_text("research note", encoding="utf-8")
    (tmp_path / "jobs" / "resume.md").write_text("job note", encoding="utf-8")
    index_workspace(settings, "research", db_path)
    index_workspace(settings, "jobs", db_path)

    inventory = inventory_workspace(settings=settings, workspace="jobs", db_path=db_path)

    assert inventory["workspace"] == "jobs"
    assert inventory["total_indexed_files"] == 1
    assert all("jobs" in item["path"] for item in inventory["files"])
    assert all("research" not in item["path"] for item in inventory["files"])


def test_file_cards_use_cached_summaries_when_content_hash_is_unchanged(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    note = tmp_path / "research" / "paper.md"
    note.write_text("first version of the note", encoding="utf-8")
    index_workspace(settings, "research", db_path)

    llm = CountingLLM()
    first_cards = asyncio.run(generate_file_cards(settings=settings, workspace="research", llm=llm, db_path=db_path))
    calls_after_first = len(llm.calls)
    second_cards = asyncio.run(generate_file_cards(settings=settings, workspace="research", llm=llm, db_path=db_path))

    assert calls_after_first == 1
    assert len(llm.calls) == 1
    assert first_cards[0].short_summary == second_cards[0].short_summary


def test_file_cards_refresh_when_requested(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    note = tmp_path / "research" / "paper.md"
    note.write_text("first version of the note", encoding="utf-8")
    index_workspace(settings, "research", db_path)

    llm = CountingLLM()
    first_cards = asyncio.run(generate_file_cards(settings=settings, workspace="research", llm=llm, db_path=db_path))
    note.write_text("second version of the note", encoding="utf-8")
    index_workspace(settings, "research", db_path)
    refreshed_cards = asyncio.run(generate_file_cards(settings=settings, workspace="research", llm=llm, db_path=db_path, refresh=True))

    assert len(llm.calls) >= 2
    assert first_cards[0].short_summary != refreshed_cards[0].short_summary


def test_workspace_summary_is_generated_from_file_summaries_not_raw_full_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    note = tmp_path / "research" / "paper.md"
    note.write_text("SENTINEL_RAW_CONTENT appears only in the source file", encoding="utf-8")
    index_workspace(settings, "research", db_path)
    with sqlite3.connect(db_path) as conn:
        file_row = conn.execute("SELECT id, path, content_hash FROM indexed_files LIMIT 1").fetchone()
    assert file_row is not None
    insert_file_summary(db_path, int(file_row[0]), "research", str(file_row[1]), str(file_row[2]), "cached summary text")

    llm = FakeAsyncLLM(
        json.dumps(
            {
                "summary": "workspace summary",
                "main_topics": ["topic"],
                "important_files": [str(note.resolve())],
                "possible_actions": ["action"],
                "warnings": [],
            }
        )
    )
    summary = asyncio.run(generate_workspace_summary(settings=settings, workspace="research", llm=llm, db_path=db_path))

    assert summary.summary == "workspace summary"
    assert llm.calls
    assert "cached summary text" in llm.calls[0]
    assert "SENTINEL_RAW_CONTENT" not in llm.calls[0]


def test_topic_explorer_returns_supporting_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    first = tmp_path / "research" / "topic_a.md"
    second = tmp_path / "research" / "topic_b.md"
    first.write_text("alpha beta alpha project notes", encoding="utf-8")
    second.write_text("alpha gamma alpha detailed notes", encoding="utf-8")
    index_workspace(settings, "research", db_path)

    topics = asyncio.run(explore_topics(settings=settings, workspace="research", limit=5, db_path=db_path))

    assert topics
    assert topics[0].supporting_files
    assert any(str(first.resolve()) in topic.supporting_files or str(second.resolve()) in topic.supporting_files for topic in topics)


def test_health_check_detects_todo_fixme(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    todo_file = tmp_path / "research" / "todo.md"
    todo_file.write_text("# Plan\nTODO: finish the summary", encoding="utf-8")
    index_workspace(settings, "research", db_path)

    issues = check_workspace_health(settings=settings, workspace="research", db_path=db_path)

    assert any(issue.issue_type == "todo_fixme" for issue in issues)


def test_health_check_detects_tiny_or_empty_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    tiny_file = tmp_path / "research" / "tiny.md"
    tiny_file.write_text("", encoding="utf-8")
    index_workspace(settings, "research", db_path)

    issues = check_workspace_health(settings=settings, workspace="research", db_path=db_path)

    assert any(issue.issue_type == "tiny_or_empty" for issue in issues)


def test_endpoints_reject_invalid_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    import backend.main as main

    monkeypatch.setattr(main, "settings", settings)

    async def post_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/workspace/inventory", json={"workspace": "missing", "limit": 10})

    response = asyncio.run(post_request())

    assert response.status_code == 400


def test_no_source_files_are_modified(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "kb.sqlite"
    note = tmp_path / "research" / "paper.md"
    note.write_text("source content stays the same", encoding="utf-8")
    before = note.read_text(encoding="utf-8")
    index_workspace(settings, "research", db_path)

    llm = CountingLLM()
    asyncio.run(generate_file_cards(settings=settings, workspace="research", llm=llm, db_path=db_path))
    asyncio.run(generate_workspace_summary(settings=settings, workspace="research", llm=llm, db_path=db_path))
    asyncio.run(explore_topics(settings=settings, workspace="research", db_path=db_path, llm=llm))

    after = note.read_text(encoding="utf-8")

    assert before == after
