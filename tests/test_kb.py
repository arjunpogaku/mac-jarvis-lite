from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from backend.config import (
    AppConfig,
    EmbeddingsConfig,
    KnowledgeBaseConfig,
    LLMConfig,
    PathsConfig,
    SafetyConfig,
    Settings,
    WorkspaceConfig,
)
from backend.db import init_db
from backend.embeddings import EmbeddingError
from backend.safety import SafetyError
from backend.tools.indexer import ask_kb, hybrid_search_kb, index_workspace, search_kb, semantic_search_kb
from backend.tools.safe_shell import validate_shell_request


def make_settings(tmp_path: Path) -> Settings:
    research = tmp_path / "research"
    jobs = tmp_path / "jobs"
    research.mkdir()
    jobs.mkdir()
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.4.0"),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(enabled=False),
        knowledge_base=KnowledgeBaseConfig(semantic_search_enabled=False, hybrid_search_enabled=False),
        safety=SafetyConfig(
            shell_enabled=False,
            max_file_read_kb=10,
            max_search_results=20,
            allowed_extensions=[".txt", ".md", ".py"],
            blocked_path_keywords=[".ssh", ".env", "private", "Passwords", "Library/Keychains"],
        ),
        paths=PathsConfig(allowed_roots=[str(research), str(jobs)]),
        workspaces={
            "research": WorkspaceConfig(description="Research", roots=[str(research)]),
            "jobs": WorkspaceConfig(description="Jobs", roots=[str(jobs)]),
        },
    )


def make_semantic_settings(tmp_path: Path) -> Settings:
    settings = make_settings(tmp_path)
    return settings.model_copy(
        update={
            "embeddings": EmbeddingsConfig(enabled=True, model="test-embed", vector_dimension=2),
            "knowledge_base": KnowledgeBaseConfig(
                semantic_search_enabled=True,
                hybrid_search_enabled=True,
                max_embedding_text_chars=2500,
            ),
        }
    )


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        lowered = text.lower()
        if "candidate" in lowered or "expansion" in lowered or "reducing" in lowered or "pruning" in lowered:
            return [1.0, 0.0]
        if "resume" in lowered or "job" in lowered:
            return [0.0, 1.0]
        return [0.5, 0.5]


class FailingEmbeddingClient:
    def get_embedding(self, text: str) -> list[float]:
        raise EmbeddingError("local embedding model unavailable")


def embedding_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0])


def indexed_paths(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [row[0] for row in conn.execute("SELECT path FROM indexed_files ORDER BY path")]


def test_indexing_only_allowed_workspace_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    research_file = tmp_path / "research" / "paper.md"
    jobs_file = tmp_path / "jobs" / "resume.md"
    research_file.write_text("coverage support research note", encoding="utf-8")
    jobs_file.write_text("coverage support job note", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"

    summary = index_workspace(settings, "research", db_path)

    assert summary.indexed_file_count == 1
    paths = indexed_paths(db_path)
    assert str(research_file.resolve()) in paths
    assert str(jobs_file.resolve()) not in paths


def test_blocked_files_are_not_indexed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    private_dir = tmp_path / "research" / "private"
    private_dir.mkdir()
    blocked = private_dir / "secret.md"
    blocked.write_text("do not index", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"

    summary = index_workspace(settings, "research", db_path)

    assert summary.rejected_file_count == 1
    assert indexed_paths(db_path) == []


def test_env_file_is_not_indexed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    env_file = tmp_path / "research" / ".env"
    env_file.write_text("TOKEN=secret", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"

    summary = index_workspace(settings, "research", db_path)

    assert summary.rejected_file_count == 1
    assert indexed_paths(db_path) == []


def test_symlink_escape_is_not_indexed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside secret", encoding="utf-8")
    link = tmp_path / "research" / "linked.md"
    link.symlink_to(outside)
    db_path = tmp_path / "kb.sqlite"

    summary = index_workspace(settings, "research", db_path)

    assert summary.rejected_file_count == 1
    assert indexed_paths(db_path) == []


def test_unchanged_files_are_skipped_on_second_index(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("first version with pruning", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"

    first = index_workspace(settings, "research", db_path)
    second = index_workspace(settings, "research", db_path)

    assert first.indexed_file_count == 1
    assert second.indexed_file_count == 0
    assert second.skipped_unchanged_count == 1


def test_changed_files_are_reindexed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("first version with pruning", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"

    index_workspace(settings, "research", db_path)
    note.write_text("second version with coverage support", encoding="utf-8")
    second = index_workspace(settings, "research", db_path)
    results = search_kb(query="coverage", workspace="research", settings=settings, db_path=db_path)

    assert second.indexed_file_count == 1
    assert second.skipped_unchanged_count == 0
    assert len(results) == 1
    assert "second version" in results[0].snippet


def test_indexing_stores_embeddings_for_chunks(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("PSC-CPM uses pruning to reduce candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    embedder = FakeEmbeddingClient()

    summary = index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]

    assert summary.embeddings_created == 1
    assert summary.embeddings_skipped == 0
    assert embedding_count(db_path) == 1


def test_unchanged_files_skip_reembedding(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("PSC-CPM uses pruning to reduce candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    embedder = FakeEmbeddingClient()

    index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]
    calls_after_first = len(embedder.calls)
    second = index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]

    assert second.skipped_unchanged_count == 1
    assert len(embedder.calls) == calls_after_first


def test_changed_files_remove_old_embeddings_and_create_new_ones(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("PSC-CPM uses pruning to reduce candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    embedder = FakeEmbeddingClient()

    index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]
    note.write_text("Updated PSC-CPM pruning note about reducing candidate expansion.", encoding="utf-8")
    second = index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]

    assert second.embeddings_created == 1
    assert embedding_count(db_path) == 1


def test_semantic_search_returns_meaningfully_related_result(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("PSC-CPM uses pruning to reduce candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    embedder = FakeEmbeddingClient()
    index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]

    outcome = semantic_search_kb(
        query="reducing candidate expansion",
        workspace="research",
        settings=settings,
        db_path=db_path,
        embedding_client=embedder,  # type: ignore[arg-type]
    )

    assert outcome.search_mode_used == "semantic"
    assert outcome.results
    assert outcome.results[0].semantic_score == pytest.approx(1.0)
    assert outcome.results[0].file_name == "note.md"


def test_semantic_search_respects_workspace_restriction(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    research_note = tmp_path / "research" / "note.md"
    jobs_note = tmp_path / "jobs" / "job.md"
    research_note.write_text("PSC-CPM uses pruning to reduce candidate expansion.", encoding="utf-8")
    jobs_note.write_text("PSC-CPM uses pruning to reduce candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    embedder = FakeEmbeddingClient()
    index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]
    index_workspace(settings, "jobs", db_path, embedding_client=embedder)  # type: ignore[arg-type]

    outcome = semantic_search_kb(
        query="reducing candidate expansion",
        workspace="jobs",
        settings=settings,
        db_path=db_path,
        embedding_client=embedder,  # type: ignore[arg-type]
    )

    assert outcome.results
    assert all("/jobs/" in result.path for result in outcome.results)


def test_hybrid_search_removes_duplicate_chunks(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("PSC-CPM pruning reduces candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    embedder = FakeEmbeddingClient()
    index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]

    outcome = hybrid_search_kb(
        query="PSC-CPM pruning candidate expansion",
        workspace="research",
        settings=settings,
        db_path=db_path,
        embedding_client=embedder,  # type: ignore[arg-type]
    )

    chunk_ids = [result.chunk_id for result in outcome.results]
    assert outcome.search_mode_used == "hybrid"
    assert len(chunk_ids) == len(set(chunk_ids))


def test_kb_ask_uses_hybrid_search_when_enabled(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("PSC-CPM pruning reduces candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    embedder = FakeEmbeddingClient()
    index_workspace(settings, "research", db_path, embedding_client=embedder)  # type: ignore[arg-type]

    answer = asyncio.run(
        ask_kb(
            question="Where did I write about PSC-CPM pruning?",
            workspace="research",
            settings=settings,
            llm=FakeLLM(),  # type: ignore[arg-type]
            db_path=db_path,
            embedding_client=embedder,  # type: ignore[arg-type]
        )
    )

    assert answer.search_mode_used == "hybrid"
    assert answer.sources_used


def test_fallback_to_keyword_search_when_embeddings_fail(tmp_path: Path) -> None:
    settings = make_semantic_settings(tmp_path)
    note = tmp_path / "research" / "note.md"
    note.write_text("PSC-CPM pruning reduces candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path, embedding_client=FailingEmbeddingClient())  # type: ignore[arg-type]

    outcome = semantic_search_kb(
        query="PSC-CPM pruning",
        workspace="research",
        settings=settings,
        db_path=db_path,
        embedding_client=FailingEmbeddingClient(),  # type: ignore[arg-type]
    )

    assert outcome.search_mode_used == "keyword"
    assert outcome.fallback_message is not None
    assert outcome.results


def test_fts_search_returns_expected_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = tmp_path / "research" / "paper.md"
    note.write_text("PSC CPM pruning improves coverage support.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path)

    results = search_kb(query="pruning", workspace="research", settings=settings, db_path=db_path)

    assert len(results) == 1
    assert results[0].file_name == "paper.md"
    assert results[0].start_line == 1


class FakeLLM:
    def __init__(self) -> None:
        self.prompt = ""

    async def chat(self, message: str) -> str:
        self.prompt = message
        return "Found it in the retrieved local context."


class ConfusedLLM:
    async def chat(self, message: str) -> str:
        return "The answer is not found in the provided local context. Check official documentation."


def test_kb_ask_uses_retrieved_context_only(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = tmp_path / "research" / "paper.md"
    note.write_text("PSC CPM pruning appears in this local note.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path)
    llm = FakeLLM()

    answer = asyncio.run(
        ask_kb(
            question="Where did I write about pruning?",
            workspace="research",
            settings=settings,
            llm=llm,  # type: ignore[arg-type]
            db_path=db_path,
        )
    )

    assert answer.sources_used
    assert "Use only the provided local context" in llm.prompt
    assert str(note.resolve()) in llm.prompt
    assert answer.answer == "Found it in the retrieved local context."


def test_kb_ask_falls_back_to_retrieved_source_when_model_contradicts_context(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = tmp_path / "research" / "paper.md"
    note.write_text("PSC-CPM uses pruning to reduce candidate expansion.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path)

    answer = asyncio.run(
        ask_kb(
            question="Where did I write about PSC-CPM pruning?",
            workspace="research",
            settings=settings,
            llm=ConfusedLLM(),  # type: ignore[arg-type]
            db_path=db_path,
        )
    )

    assert str(note.resolve()) in answer.answer
    assert "PSC-CPM uses pruning" in answer.answer


def test_invalid_workspace_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(SafetyError, match="Invalid workspace"):
        index_workspace(settings, "missing", tmp_path / "kb.sqlite")


def test_indexing_requires_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    import backend.main as main

    monkeypatch.setattr(main, "settings", settings)

    async def post_request():
        transport = ASGITransport(app=main.app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/kb/index_workspace", json={"workspace": "research", "approved": False})

    response = asyncio.run(post_request())

    assert response.status_code == 403


def test_shell_remains_disabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(SafetyError, match="disabled"):
        validate_shell_request(["pwd"], tmp_path / "research", settings)


def test_init_db_creates_kb_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "kb.sqlite"
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "indexed_files" in names
    assert "document_chunks" in names
    assert "document_chunks_fts" in names
    assert "chunk_embeddings" in names
    assert "file_summaries" in names
    assert "workspace_summaries" in names
