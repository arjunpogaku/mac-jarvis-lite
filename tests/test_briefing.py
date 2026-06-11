from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from backend.config import AppConfig, EmbeddingsConfig, KnowledgeBaseConfig, LLMConfig, PathsConfig, SafetyConfig, Settings, WorkspaceConfig
from backend.safety import SafetyError
from backend.tools.briefing import export_briefing, generate_briefing
from backend.tools.indexer import index_workspace
from backend.tools.safe_shell import validate_shell_request


def make_settings(tmp_path: Path) -> Settings:
    research = tmp_path / "research"
    research.mkdir()
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.6.0"),
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


class CapturingLLM:
    def __init__(self, response: str = "Briefing answer\n\nSources used: source.md") -> None:
        self.response = response
        self.calls: list[str] = []

    async def chat(self, message: str) -> str:
        self.calls.append(message)
        return self.response


def test_self_check_script_passes_in_safe_config() -> None:
    result = subprocess.run([sys.executable, "scripts/self_check.py"], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Result: PASS" in result.stdout


def test_smoke_test_script_passes() -> None:
    result = subprocess.run([sys.executable, "scripts/smoke_test.py"], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Result: PASS" in result.stdout


def test_briefing_generation_rejects_invalid_workspace(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(SafetyError, match="Invalid workspace"):
        asyncio.run(
            generate_briefing(
                settings=settings,
                workspace="missing",
                briefing_type="research",
                llm=CapturingLLM(),  # type: ignore[arg-type]
                db_path=tmp_path / "kb.sqlite",
            )
        )


def test_briefing_generation_uses_indexed_local_context_only(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = tmp_path / "research" / "paper.md"
    note.write_text("Indexed PSC-CPM pruning note.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path)
    llm = CapturingLLM("Briefing from indexed context. Sources used: paper.md")

    result = asyncio.run(
        generate_briefing(
            settings=settings,
            workspace="research",
            briefing_type="research",
            llm=llm,  # type: ignore[arg-type]
            db_path=db_path,
        )
    )

    assert result.sources_used == [str(note.resolve())]
    assert llm.calls
    assert "Use only the provided local indexed context" in llm.calls[0]
    assert str(note.resolve()) in llm.calls[0]
    assert "Indexed PSC-CPM pruning note" in llm.calls[0]


def test_briefing_includes_sources(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    note = tmp_path / "research" / "paper.md"
    note.write_text("Source-backed briefing evidence.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path)

    result = asyncio.run(
        generate_briefing(
            settings=settings,
            workspace="research",
            briefing_type="workspace",
            llm=CapturingLLM(),  # type: ignore[arg-type]
            db_path=db_path,
        )
    )

    assert str(note.resolve()) in result.sources_used
    assert result.files_considered == 1


def test_briefing_handles_insufficient_evidence_clearly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    result = asyncio.run(
        generate_briefing(
            settings=settings,
            workspace="research",
            briefing_type="research",
            llm=CapturingLLM(),  # type: ignore[arg-type]
            db_path=tmp_path / "kb.sqlite",
        )
    )

    assert "Insufficient indexed evidence" in result.briefing_text
    assert result.sources_used == []


def test_briefing_export_requires_approval(tmp_path: Path) -> None:
    with pytest.raises(SafetyError, match="approval"):
        export_briefing(
            workspace="research",
            briefing_type="research",
            briefing_text="hello",
            approved=False,
            export_dir=tmp_path / "exports",
        )


def test_briefing_export_writes_only_to_data_exports(tmp_path: Path) -> None:
    export_dir = tmp_path / "data" / "exports"
    path = export_briefing(
        workspace="research",
        briefing_type="research",
        briefing_text="hello",
        approved=True,
        export_dir=export_dir,
    )

    assert path.exists()
    assert path.resolve().is_relative_to(export_dir.resolve())


def test_briefing_export_does_not_overwrite_existing_files(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    first = export_briefing(
        workspace="research",
        briefing_type="research",
        briefing_text="first",
        approved=True,
        export_dir=export_dir,
    )
    second = export_briefing(
        workspace="research",
        briefing_type="research",
        briefing_text="second",
        approved=True,
        export_dir=export_dir,
    )

    assert first != second
    assert first.read_text(encoding="utf-8") != second.read_text(encoding="utf-8")


def test_path_traversal_in_export_is_impossible(tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    path = export_briefing(
        workspace="../../outside",
        briefing_type="research",
        briefing_text="hello",
        approved=True,
        export_dir=export_dir,
    )

    assert path.resolve().is_relative_to(export_dir.resolve())
    assert ".." not in path.name


def test_briefing_export_does_not_modify_source_workspace_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source = tmp_path / "research" / "paper.md"
    source.write_text("source content", encoding="utf-8")
    before = source.read_text(encoding="utf-8")

    export_briefing(
        workspace="research",
        briefing_type="workspace",
        briefing_text="briefing",
        approved=True,
        export_dir=tmp_path / "exports",
    )

    assert source.read_text(encoding="utf-8") == before
    assert settings.safety.shell_enabled is False


def test_briefing_endpoint_rejects_invalid_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.main as main

    monkeypatch.setattr(main, "settings", make_settings(tmp_path))

    async def post_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/briefing/generate", json={"workspace": "missing", "briefing_type": "research"})

    response = asyncio.run(post_request())
    assert response.status_code == 400


def test_shell_remains_disabled_for_briefing_settings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(SafetyError, match="disabled"):
        validate_shell_request(["pwd"], tmp_path / "research", settings)
