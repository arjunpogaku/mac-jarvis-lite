from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest

from backend.agents.router import execute_command, plan_command
from backend.config import AppConfig, EmbeddingsConfig, KnowledgeBaseConfig, LLMConfig, PathsConfig, SafetyConfig, Settings, WorkspaceConfig
from backend.db import init_db
from backend.safety import SafetyError
from backend.tools.indexer import index_workspace


def make_settings(tmp_path: Path) -> Settings:
    root = tmp_path / "research"
    root.mkdir()
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.8.0"),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(enabled=False),
        knowledge_base=KnowledgeBaseConfig(semantic_search_enabled=False, hybrid_search_enabled=False),
        safety=SafetyConfig(shell_enabled=False, max_file_read_kb=10, allowed_extensions=[".md", ".txt"], blocked_path_keywords=[".env", ".ssh"]),
        paths=PathsConfig(allowed_roots=[str(root)]),
        workspaces={"research": WorkspaceConfig(description="Research", roots=[str(root)])},
    )


class FakeLLM:
    async def chat(self, message: str) -> str:
        if "JSON array" in message:
            return "[]"
        return "Fake local answer from indexed context."


def seed_index(settings: Settings, tmp_path: Path) -> Path:
    note = tmp_path / "research" / "paper.md"
    note.write_text("PSC-CPM pruning reduces candidate expansion. TODO: improve weak sections.", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite"
    index_workspace(settings, "research", db_path)
    return db_path


def test_command_planning_detects_knowledge_base_question(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    plan = asyncio.run(plan_command(message="Where did I write about PSC-CPM pruning?", workspace="research", settings=settings, db_path=tmp_path / "db.sqlite"))
    assert plan.intent == "ask_knowledge_base"
    assert plan.agent == "knowledge_agent"


def test_command_planning_detects_research_review_request(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    plan = asyncio.run(plan_command(message="Find my latest thesis draft and tell me what sections need improvement.", workspace="research", settings=settings, db_path=tmp_path / "db.sqlite"))
    assert plan.intent == "research_review"
    assert plan.agent == "research_agent"


def test_command_planning_detects_project_task_request(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    plan = asyncio.run(plan_command(message="What should I work on today?", workspace="research", settings=settings, db_path=tmp_path / "db.sqlite"))
    assert plan.intent == "list_tasks"
    assert plan.agent == "project_agent"


def test_command_planning_rejects_destructive_file_command(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    plan = asyncio.run(plan_command(message="Delete all old files.", workspace="research", settings=settings, db_path=tmp_path / "db.sqlite"))
    assert plan.safety_level == "refused"
    assert plan.refusal is not None


def test_command_planning_rejects_internet_email_system_control(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    for message in ["Email this resume.", "Download something from the internet.", "Run this terminal command.", "Change system settings."]:
        plan = asyncio.run(plan_command(message=message, workspace="research", settings=settings, db_path=tmp_path / f"{abs(hash(message))}.sqlite"))
        assert plan.safety_level == "refused"


def test_command_execution_requires_approval_for_tool_commands(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    seed_index(settings, tmp_path)
    with pytest.raises(SafetyError, match="approval"):
        asyncio.run(execute_command(message="Where did I write about PSC-CPM pruning?", workspace="research", approved=False, settings=settings, llm=FakeLLM(), db_path=tmp_path / "kb.sqlite"))


def test_simple_chat_does_not_require_approval(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    result = asyncio.run(execute_command(message="Hello Jarvis", workspace="research", approved=False, settings=settings, llm=FakeLLM(), db_path=tmp_path / "db.sqlite"))
    assert "Fake local answer" in result.answer
    assert result.plan is not None
    assert result.plan.requires_approval is False


def test_command_execution_calls_only_allowed_safe_agents(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = seed_index(settings, tmp_path)
    result = asyncio.run(execute_command(message="Create a research briefing for this workspace.", workspace="research", approved=True, settings=settings, llm=FakeLLM(), db_path=db_path))
    assert result.plan is not None
    assert result.plan.agent == "briefing_agent"
    assert all("shell" not in tool for tool in result.plan.tools_needed)


def test_command_execution_returns_sources_for_kb_research_commands(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = seed_index(settings, tmp_path)
    result = asyncio.run(execute_command(message="Where did I write about PSC-CPM pruning?", workspace="research", approved=True, settings=settings, llm=FakeLLM(), db_path=db_path))
    assert result.sources_used
    assert result.actions_performed


def test_research_review_on_unindexed_workspace_returns_limitation_not_error(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "commands.sqlite"

    result = asyncio.run(
        execute_command(
            message="Find my latest research draft and review it.",
            workspace="research",
            approved=True,
            settings=settings,
            llm=FakeLLM(),
            db_path=db_path,
        )
    )

    assert "could not find an answer" in result.answer.lower()
    assert result.limitations
    assert "not been indexed" in result.limitations[0]


def test_command_history_logged_without_full_file_contents(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db_path = seed_index(settings, tmp_path)
    asyncio.run(execute_command(message="Where did I write about PSC-CPM pruning?", workspace="research", approved=True, settings=settings, llm=FakeLLM(), db_path=db_path))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_message, plan_json FROM command_history").fetchall()
    assert rows
    joined = "\n".join(str(item) for row in rows for item in row)
    assert "PSC-CPM pruning reduces candidate expansion. TODO" not in joined


def test_safety_agent_blocks_shell_internet_source_writes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    for message in ["run shell ls", "browse the internet", "move my photos"]:
        plan = asyncio.run(plan_command(message=message, workspace="research", settings=settings, db_path=tmp_path / f"{len(message)}.sqlite"))
        assert plan.agent == "safety_agent"
        assert plan.refusal


def test_streamlit_command_center_does_not_execute_during_planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    db_path = tmp_path / "commands.sqlite"

    asyncio.run(plan_command(message="Where did I write about PSC-CPM pruning?", workspace="research", settings=settings, db_path=db_path))

    with sqlite3.connect(db_path) as conn:
        planned, executed = conn.execute("SELECT COUNT(*), SUM(executed) FROM command_history").fetchone()
    assert planned == 1
    assert executed == 0
