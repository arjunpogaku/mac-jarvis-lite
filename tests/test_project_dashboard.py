from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from backend.config import AppConfig, EmbeddingsConfig, KnowledgeBaseConfig, LLMConfig, PathsConfig, SafetyConfig, Settings, WorkspaceConfig
from backend.safety import SafetyError
from backend.tools.indexer import index_workspace
from backend.tools.project_dashboard import (
    add_manual_task,
    export_dashboard,
    extract_tasks,
    generate_dashboard,
    list_tasks,
    update_task_status,
)
from backend.tools.safe_shell import validate_shell_request


def make_settings(tmp_path: Path) -> Settings:
    research = tmp_path / "research"
    research.mkdir()
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.7.0"),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(enabled=False),
        knowledge_base=KnowledgeBaseConfig(semantic_search_enabled=False, hybrid_search_enabled=False),
        safety=SafetyConfig(
            shell_enabled=False,
            max_file_read_kb=10,
            allowed_extensions=[".txt", ".md", ".py"],
            blocked_path_keywords=[".ssh", ".env", "private"],
        ),
        paths=PathsConfig(allowed_roots=[str(research)]),
        workspaces={"research": WorkspaceConfig(description="Research", roots=[str(research)])},
    )


class TaskLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def chat(self, message: str) -> str:
        self.calls.append(message)
        return json.dumps(
            [
                {
                    "title": "Finish pruning experiment",
                    "description": "Complete the PSC-CPM pruning experiment writeup.",
                    "source_path": "paper.md",
                    "category": "research",
                    "priority": "medium",
                    "evidence": "TODO: finish pruning experiment",
                }
            ]
        )


class DashboardLLM:
    async def chat(self, message: str) -> str:
        return json.dumps(
            {
                "title": "Research Dashboard",
                "overview": "The project needs attention because open research tasks remain.",
                "status": "needs_attention",
                "main_topics": ["PSC-CPM", "pruning"],
                "risks": ["Experiment writeup is incomplete."],
                "next_actions": ["Finish pruning experiment"],
            }
        )


def seed_workspace(tmp_path: Path, text: str = "TODO: finish pruning experiment") -> tuple[Settings, Path, Path]:
    settings = make_settings(tmp_path)
    source = tmp_path / "research" / "paper.md"
    source.write_text(text, encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path)
    return settings, db_path, source


def test_extract_tasks_rejects_invalid_workspace(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(SafetyError, match="Invalid workspace"):
        asyncio.run(extract_tasks(settings=settings, workspace="missing", project_type="research", llm=TaskLLM(), db_path=tmp_path / "kb.sqlite"))


def test_extract_tasks_requires_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.main as main

    monkeypatch.setattr(main, "settings", make_settings(tmp_path))

    async def post_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/project/extract_tasks",
                json={"workspace": "research", "project_type": "research", "approved": False},
            )

    response = asyncio.run(post_request())
    assert response.status_code == 403


def test_task_extraction_uses_indexed_local_context_only(tmp_path: Path) -> None:
    settings, db_path, source = seed_workspace(tmp_path)
    llm = TaskLLM()

    result = asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=llm, db_path=db_path))

    assert result.tasks_created == 1
    assert llm.calls
    assert str(source.resolve()) in llm.calls[0]
    assert "Use only provided local indexed context" in llm.calls[0]


def test_task_extraction_creates_tasks_with_evidence(tmp_path: Path) -> None:
    settings, db_path, _source = seed_workspace(tmp_path)
    result = asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))

    assert result.tasks[0].evidence
    assert result.tasks[0].priority == "medium"


def test_duplicate_tasks_are_skipped(tmp_path: Path) -> None:
    settings, db_path, _source = seed_workspace(tmp_path)
    first = asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))
    second = asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))

    assert first.tasks_created == 1
    assert second.tasks_created == 0
    assert second.tasks_skipped_duplicates == 1


def test_dashboard_generation_returns_status_and_sources(tmp_path: Path) -> None:
    settings, db_path, source = seed_workspace(tmp_path)
    asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))
    dashboard = asyncio.run(generate_dashboard(settings=settings, workspace="research", project_type="research", llm=DashboardLLM(), db_path=db_path, refresh=True))

    assert dashboard.status == "needs_attention"
    assert dashboard.sources_used
    assert str(source.resolve()) in dashboard.sources_used


def test_task_listing_filters_by_status_category_priority(tmp_path: Path) -> None:
    settings, db_path, _source = seed_workspace(tmp_path)
    asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))

    tasks = list_tasks(workspace="research", status="open", category="research", priority="medium", db_path=db_path)

    assert len(tasks) == 1
    assert tasks[0].title == "Finish pruning experiment"


def test_task_status_update_modifies_only_jarvis_sqlite(tmp_path: Path) -> None:
    settings, db_path, source = seed_workspace(tmp_path)
    before = source.read_text(encoding="utf-8")
    result = asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))

    updated = update_task_status(task_id=result.tasks[0].id, status="in_progress", db_path=db_path)

    assert updated.status == "in_progress"
    assert source.read_text(encoding="utf-8") == before


def test_invalid_task_status_is_rejected(tmp_path: Path) -> None:
    settings, db_path, _source = seed_workspace(tmp_path)
    result = asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))

    with pytest.raises(SafetyError, match="Invalid task status"):
        update_task_status(task_id=result.tasks[0].id, status="deleted", db_path=db_path)


def test_manual_task_add_works(tmp_path: Path) -> None:
    _settings, db_path, _source = seed_workspace(tmp_path)
    task = add_manual_task(
        workspace="research",
        title="Rewrite abstract",
        description="Improve clarity.",
        category="writing",
        priority="high",
        db_path=db_path,
    )

    assert task.created_by == "user"
    assert task.priority == "high"


def test_dashboard_export_requires_approval(tmp_path: Path) -> None:
    settings, db_path, _source = seed_workspace(tmp_path)
    dashboard = asyncio.run(generate_dashboard(settings=settings, workspace="research", project_type="research", llm=DashboardLLM(), db_path=db_path, refresh=True))

    with pytest.raises(SafetyError, match="approval"):
        export_dashboard(workspace="research", dashboard_id=dashboard.id, approved=False, db_path=db_path, export_dir=tmp_path / "exports")


def test_dashboard_export_writes_only_to_data_exports(tmp_path: Path) -> None:
    settings, db_path, _source = seed_workspace(tmp_path)
    dashboard = asyncio.run(generate_dashboard(settings=settings, workspace="research", project_type="research", llm=DashboardLLM(), db_path=db_path, refresh=True))
    export_dir = tmp_path / "data" / "exports"

    exported = export_dashboard(workspace="research", dashboard_id=dashboard.id, approved=True, db_path=db_path, export_dir=export_dir)

    assert exported.exists()
    assert exported.resolve().is_relative_to(export_dir.resolve())


def test_dashboard_export_does_not_overwrite_existing_files(tmp_path: Path) -> None:
    settings, db_path, _source = seed_workspace(tmp_path)
    dashboard = asyncio.run(generate_dashboard(settings=settings, workspace="research", project_type="research", llm=DashboardLLM(), db_path=db_path, refresh=True))
    export_dir = tmp_path / "exports"

    first = export_dashboard(workspace="research", dashboard_id=dashboard.id, approved=True, db_path=db_path, export_dir=export_dir)
    second = export_dashboard(workspace="research", dashboard_id=dashboard.id, approved=True, db_path=db_path, export_dir=export_dir)

    assert first != second


def test_source_workspace_files_are_not_modified(tmp_path: Path) -> None:
    settings, db_path, source = seed_workspace(tmp_path)
    before = source.read_text(encoding="utf-8")
    asyncio.run(extract_tasks(settings=settings, workspace="research", project_type="research", llm=TaskLLM(), db_path=db_path))
    dashboard = asyncio.run(generate_dashboard(settings=settings, workspace="research", project_type="research", llm=DashboardLLM(), db_path=db_path, refresh=True))
    export_dashboard(workspace="research", dashboard_id=dashboard.id, approved=True, db_path=db_path, export_dir=tmp_path / "exports")

    assert source.read_text(encoding="utf-8") == before


def test_shell_remains_disabled_for_project_dashboard(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(SafetyError, match="disabled"):
        validate_shell_request(["pwd"], tmp_path / "research", settings)
