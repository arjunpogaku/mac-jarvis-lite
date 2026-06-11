from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agents.router import execute_command, plan_command
from backend.config import Settings, get_settings
from backend.db import init_db, log_interaction
from backend.embeddings import OllamaEmbeddingClient
from backend.llm import OllamaClient
from backend.safety import SafetyError, check_allowed_path, validate_allowed_path
from backend.tools.file_reader import read_text_file
from backend.tools.file_search import search_files
from backend.tools.briefing import BriefingResult, export_briefing, generate_briefing
from backend.tools.indexer import (
    IndexSummary,
    ask_kb,
    hybrid_search_kb,
    index_workspace,
    search_kb,
    semantic_search_kb,
)
from backend.tools.project_dashboard import (
    ProjectDashboard,
    ProjectTask,
    TaskExtractionResult,
    add_manual_task,
    export_dashboard,
    extract_tasks,
    generate_dashboard,
    list_tasks,
    update_task_status,
)
from backend.tools.workspace_intelligence import (
    check_workspace_health,
    explore_topics,
    generate_file_cards,
    generate_workspace_summary,
    inventory_workspace,
)
from backend.tools.summarizer import summarize_file


settings = get_settings()
llm_client = OllamaClient(settings.llm)
embedding_client = OllamaEmbeddingClient(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title=settings.app.name, version=settings.app.version, lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


class SearchRequest(BaseModel):
    keyword: str = Field(min_length=1)
    session_id: str | None = None
    tool_name: str = "search_files"
    requested_path: str | None = None
    workspace: str | None = "general"
    approved: bool = False
    safety_check_result: str | None = None


class SearchResult(BaseModel):
    path: str
    line_number: int
    line: str


class SearchResponse(BaseModel):
    tool_name: str
    requested_path: str
    approved: bool
    safety_check_result: str
    results: list[SearchResult]


class ReadFileRequest(BaseModel):
    path: str
    session_id: str | None = None
    tool_name: str = "read_file"
    requested_path: str | None = None
    workspace: str | None = "general"
    approved: bool = False
    safety_check_result: str | None = None


class ReadFileResponse(BaseModel):
    tool_name: str
    requested_path: str
    approved: bool
    safety_check_result: str
    path: str
    content: str
    truncated: bool
    size_bytes: int


class SummarizeFileRequest(BaseModel):
    path: str
    session_id: str | None = None
    tool_name: str = "summarize_file"
    requested_path: str | None = None
    workspace: str | None = "general"
    approved: bool = False
    safety_check_result: str | None = None


class SummarizeFileResponse(BaseModel):
    tool_name: str
    requested_path: str
    approved: bool
    safety_check_result: str
    path: str
    summary: str
    truncated: bool


class KBIndexRequest(BaseModel):
    workspace: str
    approved: bool = False


class KBIndexResponse(BaseModel):
    workspace: str
    scanned_file_count: int
    indexed_file_count: int
    skipped_unchanged_count: int
    rejected_file_count: int
    total_chunks_created: int
    embeddings_created: int
    embeddings_skipped: int
    embedding_errors: list[str]
    errors: list[str]


class KBSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    workspace: str
    limit: int = Field(default=10, ge=1, le=50)


class KBSearchResultResponse(BaseModel):
    path: str
    file_name: str
    snippet: str
    rank: float | None
    semantic_score: float | None = None
    combined_score: float | None = None
    chunk_id: int
    file_id: int
    start_line: int | None
    end_line: int | None


class KBSearchResponse(BaseModel):
    workspace: str
    query: str
    search_mode_used: str = "keyword"
    fallback_message: str | None = None
    results: list[KBSearchResultResponse]


class KBAskRequest(BaseModel):
    question: str = Field(min_length=1)
    workspace: str
    limit: int = Field(default=5, ge=1, le=10)


class KBAskResponse(BaseModel):
    answer: str
    sources_used: list[KBSearchResultResponse]
    search_mode_used: str
    context_limited: bool


class WorkspaceInventoryRequest(BaseModel):
    workspace: str
    limit: int = Field(default=100, ge=1, le=500)


class WorkspaceInventoryFileResponse(BaseModel):
    path: str
    file_name: str
    extension: str
    size_bytes: int
    modified_time: str
    chunks: int
    category: str


class WorkspaceInventoryResponse(BaseModel):
    workspace: str
    total_indexed_files: int
    total_chunks: int
    last_indexed_time: str | None
    extensions_breakdown: list[dict[str, object]]
    category_breakdown: list[dict[str, object]]
    files: list[WorkspaceInventoryFileResponse]
    largest_files: list[WorkspaceInventoryFileResponse]
    recently_modified_files: list[WorkspaceInventoryFileResponse]
    files_with_most_chunks: list[WorkspaceInventoryFileResponse]


class WorkspaceFileCardsRequest(BaseModel):
    workspace: str
    limit: int = Field(default=20, ge=1, le=100)
    refresh: bool = False


class WorkspaceFileCardResponse(BaseModel):
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


class WorkspaceFileCardsResponse(BaseModel):
    workspace: str
    cards: list[WorkspaceFileCardResponse]


class WorkspaceSummaryRequest(BaseModel):
    workspace: str
    refresh: bool = False
    limit_files: int = Field(default=30, ge=1, le=100)


class WorkspaceSummaryResponse(BaseModel):
    workspace: str
    indexed_file_count: int
    total_indexed_files: int
    summary: str
    main_topics: list[str]
    important_files: list[str]
    possible_actions: list[str]
    warnings: list[str]
    generated_at: str
    model: str


class WorkspaceTopicsRequest(BaseModel):
    workspace: str
    limit: int = Field(default=20, ge=1, le=20)


class WorkspaceTopicResponse(BaseModel):
    topic_label: str
    related_keywords: list[str]
    supporting_files: list[str]
    example_snippets: list[str]


class WorkspaceTopicsResponse(BaseModel):
    workspace: str
    topics: list[WorkspaceTopicResponse]


class WorkspaceHealthCheckRequest(BaseModel):
    workspace: str
    limit: int = Field(default=30, ge=1, le=100)


class WorkspaceHealthIssueResponse(BaseModel):
    file_path: str
    issue_type: str
    severity: str
    explanation: str
    suggested_next_action: str


class WorkspaceHealthCheckResponse(BaseModel):
    workspace: str
    issues: list[WorkspaceHealthIssueResponse]


class BriefingGenerateRequest(BaseModel):
    workspace: str
    briefing_type: str
    limit_files: int = Field(default=30, ge=1, le=100)
    refresh: bool = False


class BriefingGenerateResponse(BaseModel):
    workspace: str
    briefing_type: str
    briefing_text: str
    sources_used: list[str]
    files_considered: int
    context_limited: bool
    generated_at: str


class BriefingExportRequest(BaseModel):
    workspace: str
    briefing_type: str
    briefing_text: str = Field(min_length=1)
    approved: bool = False


class BriefingExportResponse(BaseModel):
    exported_path: str


class ProjectTaskResponse(BaseModel):
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


class ProjectExtractTasksRequest(BaseModel):
    workspace: str
    project_type: str
    limit_files: int = Field(default=30, ge=1, le=100)
    refresh: bool = False
    approved: bool = False


class ProjectExtractTasksResponse(BaseModel):
    workspace: str
    project_type: str
    tasks_created: int
    tasks_skipped_duplicates: int
    tasks: list[ProjectTaskResponse]
    sources_used: list[str]
    context_limited: bool


class ProjectDashboardRequest(BaseModel):
    workspace: str
    project_type: str
    refresh: bool = False


class ProjectDashboardResponse(BaseModel):
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


class ProjectTasksRequest(BaseModel):
    workspace: str
    status: str | None = None
    category: str | None = None
    priority: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class ProjectTasksResponse(BaseModel):
    workspace: str
    tasks: list[ProjectTaskResponse]


class ProjectUpdateTaskRequest(BaseModel):
    task_id: int
    status: str


class ProjectAddTaskRequest(BaseModel):
    workspace: str
    title: str = Field(min_length=1)
    description: str | None = None
    category: str = "unknown"
    priority: str = "low"


class ProjectExportDashboardRequest(BaseModel):
    workspace: str
    dashboard_id: int
    approved: bool = False


class CommandPlanRequest(BaseModel):
    message: str = Field(min_length=1)
    workspace: str
    session_id: str | None = None


class CommandPlanResponse(BaseModel):
    intent: str
    agent: str
    requires_approval: bool
    safety_level: str
    plan: list[str]
    tools_needed: list[str]
    approval_message: str
    refusal: str | None = None
    plan_id: str | None = None


class CommandExecuteRequest(BaseModel):
    message: str = Field(min_length=1)
    workspace: str
    approved: bool = False
    plan_id: str | None = None
    session_id: str | None = None


class CommandExecuteResponse(BaseModel):
    answer: str
    sources_used: list[str]
    actions_performed: list[str]
    limitations: list[str]
    plan: CommandPlanResponse


def _inventory_file_response(item: dict[str, object]) -> WorkspaceInventoryFileResponse:
    return WorkspaceInventoryFileResponse(**item)


def _file_card_response(item) -> WorkspaceFileCardResponse:
    return WorkspaceFileCardResponse(**item.__dict__)


def _topic_response(item) -> WorkspaceTopicResponse:
    return WorkspaceTopicResponse(**item.__dict__)


def _health_issue_response(item) -> WorkspaceHealthIssueResponse:
    return WorkspaceHealthIssueResponse(**item.__dict__)


def _briefing_response(item: BriefingResult) -> BriefingGenerateResponse:
    return BriefingGenerateResponse(**item.__dict__)


def _project_task_response(item: ProjectTask) -> ProjectTaskResponse:
    return ProjectTaskResponse(**item.__dict__)


def _project_dashboard_response(item: ProjectDashboard) -> ProjectDashboardResponse:
    return ProjectDashboardResponse(**item.__dict__)


def _command_plan_response(item) -> CommandPlanResponse:
    return CommandPlanResponse(**item.__dict__)


def require_tool_approval(approved: bool) -> None:
    if settings.safety.tools_require_approval and not approved:
        raise HTTPException(status_code=403, detail="Explicit tool approval is required.")


def settings_for_workspace(workspace: str | None) -> Settings:
    roots = settings.roots_for_workspace(workspace)
    return settings.with_allowed_roots(roots)


def require_valid_workspace(workspace: str) -> None:
    if workspace not in settings.workspaces:
        raise HTTPException(status_code=400, detail="Invalid workspace.")


def settings_for_requested_search_path(request: SearchRequest) -> tuple[Settings, str, str]:
    workspace_settings = settings_for_workspace(request.workspace)
    if not request.requested_path:
        roots = workspace_settings.paths.allowed_roots
        safety_status = "Allowed: search is limited to the selected approved workspace roots."
        return workspace_settings, ", ".join(roots), safety_status

    validated = validate_allowed_path(request.requested_path, workspace_settings, require_file=False)
    if not validated.resolved.is_dir():
        raise SafetyError("Search target is not a folder.")
    scoped_settings = workspace_settings.with_allowed_roots([str(validated.resolved)])
    return scoped_settings, str(validated.resolved), "Allowed: search target is inside an approved workspace."


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "name": settings.app.name,
        "version": settings.app.version,
        "ollama_ok": await llm_client.health(),
        "model": settings.llm.model,
        "workspaces": {
            name: {"description": workspace.description, "roots": workspace.roots}
            for name, workspace in settings.workspaces.items()
        },
        "embeddings": {
            "enabled": settings.embeddings.enabled,
            "model": settings.embeddings.model,
            "provider": settings.embeddings.provider,
            "available": embedding_client.available() if settings.embeddings.enabled else False,
        },
    }


@app.post("/command/plan", response_model=CommandPlanResponse)
async def command_plan(request: CommandPlanRequest) -> CommandPlanResponse:
    try:
        plan = await plan_command(
            message=request.message,
            workspace=request.workspace,
            settings=settings,
            llm=llm_client,
            session_id=request.session_id,
        )
        return _command_plan_response(plan)
    except SafetyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/command/execute", response_model=CommandExecuteResponse)
async def command_execute(request: CommandExecuteRequest) -> CommandExecuteResponse:
    try:
        result = await execute_command(
            message=request.message,
            workspace=request.workspace,
            approved=request.approved,
            settings=settings,
            llm=llm_client,
            session_id=request.session_id,
        )
        assert result.plan is not None
        return CommandExecuteResponse(
            answer=result.answer,
            sources_used=result.sources_used,
            actions_performed=result.actions_performed,
            limitations=result.limitations,
            plan=_command_plan_response(result.plan),
        )
    except SafetyError as exc:
        raise HTTPException(status_code=403 if "approval" in str(exc).lower() else 400, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    try:
        response = await llm_client.chat(request.message)
        log_interaction(
            session_id=session_id,
            user_message=request.message,
            assistant_response=response,
            tool_name=None,
            tool_status="ok",
        )
        return ChatResponse(session_id=session_id, response=response)
    except Exception as exc:
        log_interaction(
            session_id=session_id,
            user_message=request.message,
            assistant_response=None,
            tool_name=None,
            tool_status="error",
        )
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc


@app.post("/tools/search_files", response_model=SearchResponse)
async def tool_search_files(request: SearchRequest) -> SearchResponse:
    require_tool_approval(request.approved)
    try:
        scoped_settings, requested_path, safety_status = settings_for_requested_search_path(request)
        matches = search_files(request.keyword, scoped_settings)
        log_interaction(
            session_id=request.session_id,
            user_message=f"search:{requested_path}:{request.keyword}",
            assistant_response=f"{len(matches)} results",
            tool_name="search_files",
            tool_status="ok",
        )
        return SearchResponse(
            tool_name="search_files",
            requested_path=requested_path,
            approved=request.approved,
            safety_check_result=safety_status,
            results=[SearchResult(**match.__dict__) for match in matches],
        )
    except SafetyError as exc:
        log_interaction(
            session_id=request.session_id,
            user_message=f"search:{request.keyword}",
            assistant_response=None,
            tool_name="search_files",
            tool_status="rejected",
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tools/read_file", response_model=ReadFileResponse)
async def tool_read_file(request: ReadFileRequest) -> ReadFileResponse:
    require_tool_approval(request.approved)
    scoped_settings = settings_for_workspace(request.workspace)
    requested_path = request.requested_path or request.path
    safety_status = check_allowed_path(requested_path, scoped_settings, require_file=True).message
    try:
        content, truncated, size = read_text_file(requested_path, scoped_settings)
        log_interaction(
            session_id=request.session_id,
            user_message=f"read:{requested_path}",
            assistant_response=f"read {min(len(content.encode('utf-8')), size)} bytes",
            tool_name="read_file",
            tool_status="ok",
        )
        return ReadFileResponse(
            tool_name="read_file",
            requested_path=requested_path,
            approved=request.approved,
            safety_check_result=safety_status,
            path=requested_path,
            content=content,
            truncated=truncated,
            size_bytes=size,
        )
    except SafetyError as exc:
        log_interaction(
            session_id=request.session_id,
            user_message=f"read:{request.path}",
            assistant_response=None,
            tool_name="read_file",
            tool_status="rejected",
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tools/summarize_file", response_model=SummarizeFileResponse)
async def tool_summarize_file(request: SummarizeFileRequest) -> SummarizeFileResponse:
    require_tool_approval(request.approved)
    scoped_settings = settings_for_workspace(request.workspace)
    requested_path = request.requested_path or request.path
    safety_status = check_allowed_path(requested_path, scoped_settings, require_file=True).message
    try:
        summary, truncated = await summarize_file(requested_path, scoped_settings, llm_client)
        log_interaction(
            session_id=request.session_id,
            user_message=f"summarize:{requested_path}",
            assistant_response=summary,
            tool_name="summarize_file",
            tool_status="ok",
        )
        return SummarizeFileResponse(
            tool_name="summarize_file",
            requested_path=requested_path,
            approved=request.approved,
            safety_check_result=safety_status,
            path=requested_path,
            summary=summary,
            truncated=truncated,
        )
    except SafetyError as exc:
        log_interaction(
            session_id=request.session_id,
            user_message=f"summarize:{request.path}",
            assistant_response=None,
            tool_name="summarize_file",
            tool_status="rejected",
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/kb/index_workspace", response_model=KBIndexResponse)
async def kb_index_workspace(request: KBIndexRequest) -> KBIndexResponse:
    require_tool_approval(request.approved)
    require_valid_workspace(request.workspace)
    try:
        summary: IndexSummary = index_workspace(settings, request.workspace, embedding_client=embedding_client)
        log_interaction(
            session_id=None,
            user_message=f"index_workspace:{request.workspace}",
            assistant_response=(
                f"indexed={summary.indexed_file_count}; skipped={summary.skipped_unchanged_count}; "
                f"rejected={summary.rejected_file_count}"
            ),
            tool_name="kb_index_workspace",
            tool_status="ok",
            workspace=request.workspace,
        )
        return KBIndexResponse(**summary.__dict__)
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"index_workspace:{request.workspace}",
            assistant_response=None,
            tool_name="kb_index_workspace",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/kb/search", response_model=KBSearchResponse)
async def kb_search(request: KBSearchRequest) -> KBSearchResponse:
    require_valid_workspace(request.workspace)
    try:
        results = search_kb(
            query=request.query,
            workspace=request.workspace,
            settings=settings,
            limit=request.limit,
        )
        log_interaction(
            session_id=None,
            user_message=f"kb_search:{request.query[:120]}",
            assistant_response=f"{len(results)} results",
            tool_name="kb_search",
            tool_status="ok",
            workspace=request.workspace,
        )
        return KBSearchResponse(
            workspace=request.workspace,
            query=request.query,
            search_mode_used="keyword",
            results=[KBSearchResultResponse(**result.__dict__) for result in results],
        )
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"kb_search:{request.query[:120]}",
            assistant_response=None,
            tool_name="kb_search",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/kb/semantic_search", response_model=KBSearchResponse)
async def kb_semantic_search(request: KBSearchRequest) -> KBSearchResponse:
    require_valid_workspace(request.workspace)
    outcome = semantic_search_kb(
        query=request.query,
        workspace=request.workspace,
        settings=settings,
        limit=request.limit,
        embedding_client=embedding_client,
    )
    log_interaction(
        session_id=None,
        user_message=f"kb_semantic_search:{request.query[:120]}",
        assistant_response=f"{len(outcome.results)} results; mode={outcome.search_mode_used}",
        tool_name="kb_semantic_search",
        tool_status="ok",
        workspace=request.workspace,
    )
    return KBSearchResponse(
        workspace=request.workspace,
        query=request.query,
        search_mode_used=outcome.search_mode_used,
        fallback_message=outcome.fallback_message,
        results=[KBSearchResultResponse(**result.__dict__) for result in outcome.results],
    )


@app.post("/kb/hybrid_search", response_model=KBSearchResponse)
async def kb_hybrid_search(request: KBSearchRequest) -> KBSearchResponse:
    require_valid_workspace(request.workspace)
    outcome = hybrid_search_kb(
        query=request.query,
        workspace=request.workspace,
        settings=settings,
        limit=request.limit,
        embedding_client=embedding_client,
    )
    log_interaction(
        session_id=None,
        user_message=f"kb_hybrid_search:{request.query[:120]}",
        assistant_response=f"{len(outcome.results)} results; mode={outcome.search_mode_used}",
        tool_name="kb_hybrid_search",
        tool_status="ok",
        workspace=request.workspace,
    )
    return KBSearchResponse(
        workspace=request.workspace,
        query=request.query,
        search_mode_used=outcome.search_mode_used,
        fallback_message=outcome.fallback_message,
        results=[KBSearchResultResponse(**result.__dict__) for result in outcome.results],
    )


@app.post("/kb/ask", response_model=KBAskResponse)
async def kb_ask(request: KBAskRequest) -> KBAskResponse:
    require_valid_workspace(request.workspace)
    try:
        answer = await ask_kb(
            question=request.question,
            workspace=request.workspace,
            settings=settings,
            llm=llm_client,
            limit=request.limit,
        )
        log_interaction(
            session_id=None,
            user_message=f"kb_ask:{request.question[:120]}",
            assistant_response=f"sources={len(answer.sources_used)}",
            tool_name="kb_ask",
            tool_status="ok",
            workspace=request.workspace,
        )
        return KBAskResponse(
            answer=answer.answer,
            sources_used=[KBSearchResultResponse(**source.__dict__) for source in answer.sources_used],
            search_mode_used=answer.search_mode_used,
            context_limited=answer.context_limited,
        )
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"kb_ask:{request.question[:120]}",
            assistant_response=None,
            tool_name="kb_ask",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workspace/inventory", response_model=WorkspaceInventoryResponse)
async def workspace_inventory(request: WorkspaceInventoryRequest) -> WorkspaceInventoryResponse:
    require_valid_workspace(request.workspace)
    try:
        payload = inventory_workspace(settings=settings, workspace=request.workspace, limit=request.limit)
        log_interaction(
            session_id=None,
            user_message=f"workspace_inventory:{request.workspace}",
            assistant_response=f"files={payload['total_indexed_files']}; chunks={payload['total_chunks']}",
            tool_name="workspace_inventory",
            tool_status="ok",
            workspace=request.workspace,
        )
        return WorkspaceInventoryResponse(
            workspace=str(payload["workspace"]),
            total_indexed_files=int(payload["total_indexed_files"]),
            total_chunks=int(payload["total_chunks"]),
            last_indexed_time=payload["last_indexed_time"],
            extensions_breakdown=list(payload["extensions_breakdown"]),
            category_breakdown=list(payload["category_breakdown"]),
            files=[_inventory_file_response(item) for item in payload["files"]],
            largest_files=[_inventory_file_response(item) for item in payload["largest_files"]],
            recently_modified_files=[_inventory_file_response(item) for item in payload["recently_modified_files"]],
            files_with_most_chunks=[_inventory_file_response(item) for item in payload["files_with_most_chunks"]],
        )
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"workspace_inventory:{request.workspace}",
            assistant_response=None,
            tool_name="workspace_inventory",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workspace/file_cards", response_model=WorkspaceFileCardsResponse)
async def workspace_file_cards(request: WorkspaceFileCardsRequest) -> WorkspaceFileCardsResponse:
    require_valid_workspace(request.workspace)
    try:
        cards = await generate_file_cards(
            settings=settings,
            workspace=request.workspace,
            llm=llm_client,
            limit=request.limit,
            refresh=request.refresh,
        )
        log_interaction(
            session_id=None,
            user_message=f"workspace_file_cards:{request.workspace}",
            assistant_response=f"cards={len(cards)}; refresh={request.refresh}",
            tool_name="workspace_file_cards",
            tool_status="ok",
            workspace=request.workspace,
        )
        return WorkspaceFileCardsResponse(workspace=request.workspace, cards=[_file_card_response(card) for card in cards])
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"workspace_file_cards:{request.workspace}",
            assistant_response=None,
            tool_name="workspace_file_cards",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workspace/summary", response_model=WorkspaceSummaryResponse)
async def workspace_summary(request: WorkspaceSummaryRequest) -> WorkspaceSummaryResponse:
    require_valid_workspace(request.workspace)
    try:
        summary = await generate_workspace_summary(
            settings=settings,
            workspace=request.workspace,
            llm=llm_client,
            refresh=request.refresh,
            limit_files=request.limit_files,
        )
        inventory = inventory_workspace(settings=settings, workspace=request.workspace, limit=1)
        log_interaction(
            session_id=None,
            user_message=f"workspace_summary:{request.workspace}",
            assistant_response=f"files={summary.indexed_file_count}; total={inventory['total_indexed_files']}",
            tool_name="workspace_summary",
            tool_status="ok",
            workspace=request.workspace,
        )
        return WorkspaceSummaryResponse(
            workspace=summary.workspace,
            indexed_file_count=summary.indexed_file_count,
            total_indexed_files=int(inventory["total_indexed_files"]),
            summary=summary.summary,
            main_topics=summary.main_topics,
            important_files=summary.important_files,
            possible_actions=summary.possible_actions,
            warnings=summary.warnings,
            generated_at=summary.generated_at,
            model=summary.model,
        )
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"workspace_summary:{request.workspace}",
            assistant_response=None,
            tool_name="workspace_summary",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workspace/topics", response_model=WorkspaceTopicsResponse)
async def workspace_topics(request: WorkspaceTopicsRequest) -> WorkspaceTopicsResponse:
    require_valid_workspace(request.workspace)
    try:
        topics = await explore_topics(settings=settings, workspace=request.workspace, limit=request.limit, llm=llm_client)
        log_interaction(
            session_id=None,
            user_message=f"workspace_topics:{request.workspace}",
            assistant_response=f"topics={len(topics)}",
            tool_name="workspace_topics",
            tool_status="ok",
            workspace=request.workspace,
        )
        return WorkspaceTopicsResponse(workspace=request.workspace, topics=[_topic_response(item) for item in topics])
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"workspace_topics:{request.workspace}",
            assistant_response=None,
            tool_name="workspace_topics",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/workspace/health_check", response_model=WorkspaceHealthCheckResponse)
async def workspace_health_check(request: WorkspaceHealthCheckRequest) -> WorkspaceHealthCheckResponse:
    require_valid_workspace(request.workspace)
    try:
        issues = check_workspace_health(settings=settings, workspace=request.workspace, limit=request.limit)
        log_interaction(
            session_id=None,
            user_message=f"workspace_health_check:{request.workspace}",
            assistant_response=f"issues={len(issues)}",
            tool_name="workspace_health_check",
            tool_status="ok",
            workspace=request.workspace,
        )
        return WorkspaceHealthCheckResponse(workspace=request.workspace, issues=[_health_issue_response(item) for item in issues])
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"workspace_health_check:{request.workspace}",
            assistant_response=None,
            tool_name="workspace_health_check",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/briefing/generate", response_model=BriefingGenerateResponse)
async def briefing_generate(request: BriefingGenerateRequest) -> BriefingGenerateResponse:
    require_valid_workspace(request.workspace)
    try:
        result = await generate_briefing(
            settings=settings,
            workspace=request.workspace,
            briefing_type=request.briefing_type,
            llm=llm_client,
            limit_files=request.limit_files,
            refresh=request.refresh,
        )
        log_interaction(
            session_id=None,
            user_message=f"briefing_generate:{request.briefing_type}; files={request.limit_files}",
            assistant_response=f"sources={len(result.sources_used)}; files={result.files_considered}",
            tool_name="briefing_generate",
            tool_status="ok",
            workspace=request.workspace,
        )
        return _briefing_response(result)
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"briefing_generate:{request.briefing_type}",
            assistant_response=None,
            tool_name="briefing_generate",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/briefing/export", response_model=BriefingExportResponse)
async def briefing_export(request: BriefingExportRequest) -> BriefingExportResponse:
    require_valid_workspace(request.workspace)
    try:
        exported = export_briefing(
            workspace=request.workspace,
            briefing_type=request.briefing_type,
            briefing_text=request.briefing_text,
            approved=request.approved,
        )
        log_interaction(
            session_id=None,
            user_message=f"briefing_export:{request.briefing_type}",
            assistant_response=f"export_path={exported}",
            tool_name="briefing_export",
            tool_status="ok",
            workspace=request.workspace,
        )
        return BriefingExportResponse(exported_path=str(exported))
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"briefing_export:{request.briefing_type}",
            assistant_response=None,
            tool_name="briefing_export",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/project/extract_tasks", response_model=ProjectExtractTasksResponse)
async def project_extract_tasks(request: ProjectExtractTasksRequest) -> ProjectExtractTasksResponse:
    require_valid_workspace(request.workspace)
    require_tool_approval(request.approved)
    try:
        result: TaskExtractionResult = await extract_tasks(
            settings=settings,
            workspace=request.workspace,
            project_type=request.project_type,
            llm=llm_client,
            limit_files=request.limit_files,
            refresh=request.refresh,
        )
        log_interaction(
            session_id=None,
            user_message=f"project_extract_tasks:{request.project_type}; files={request.limit_files}",
            assistant_response=f"created={result.tasks_created}; duplicates={result.tasks_skipped_duplicates}",
            tool_name="project_extract_tasks",
            tool_status="ok",
            workspace=request.workspace,
        )
        return ProjectExtractTasksResponse(
            workspace=result.workspace,
            project_type=result.project_type,
            tasks_created=result.tasks_created,
            tasks_skipped_duplicates=result.tasks_skipped_duplicates,
            tasks=[_project_task_response(task) for task in result.tasks],
            sources_used=result.sources_used,
            context_limited=result.context_limited,
        )
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"project_extract_tasks:{request.project_type}",
            assistant_response=None,
            tool_name="project_extract_tasks",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/project/dashboard", response_model=ProjectDashboardResponse)
async def project_dashboard(request: ProjectDashboardRequest) -> ProjectDashboardResponse:
    require_valid_workspace(request.workspace)
    try:
        dashboard = await generate_dashboard(
            settings=settings,
            workspace=request.workspace,
            project_type=request.project_type,
            llm=llm_client,
            refresh=request.refresh,
        )
        log_interaction(
            session_id=None,
            user_message=f"project_dashboard:{request.project_type}; refresh={request.refresh}",
            assistant_response=f"dashboard_id={dashboard.id}; status={dashboard.status}",
            tool_name="project_dashboard",
            tool_status="ok",
            workspace=request.workspace,
        )
        return _project_dashboard_response(dashboard)
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"project_dashboard:{request.project_type}",
            assistant_response=None,
            tool_name="project_dashboard",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/project/tasks", response_model=ProjectTasksResponse)
async def project_tasks(request: ProjectTasksRequest) -> ProjectTasksResponse:
    require_valid_workspace(request.workspace)
    try:
        tasks = list_tasks(
            workspace=request.workspace,
            status=request.status,
            category=request.category,
            priority=request.priority,
            limit=request.limit,
        )
        log_interaction(
            session_id=None,
            user_message=f"project_list_tasks:status={request.status}; category={request.category}; priority={request.priority}",
            assistant_response=f"tasks={len(tasks)}",
            tool_name="project_list_tasks",
            tool_status="ok",
            workspace=request.workspace,
        )
        return ProjectTasksResponse(workspace=request.workspace, tasks=[_project_task_response(task) for task in tasks])
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message="project_list_tasks",
            assistant_response=None,
            tool_name="project_list_tasks",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/project/update_task", response_model=ProjectTaskResponse)
async def project_update_task(request: ProjectUpdateTaskRequest) -> ProjectTaskResponse:
    try:
        task = update_task_status(task_id=request.task_id, status=request.status)
        log_interaction(
            session_id=None,
            user_message=f"project_update_task:{request.task_id}; status={request.status}",
            assistant_response=f"task_id={task.id}; status={task.status}",
            tool_name="project_update_task",
            tool_status="ok",
            workspace=task.workspace,
        )
        return _project_task_response(task)
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"project_update_task:{request.task_id}",
            assistant_response=None,
            tool_name="project_update_task",
            tool_status="rejected",
            workspace=None,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/project/add_task", response_model=ProjectTaskResponse)
async def project_add_task(request: ProjectAddTaskRequest) -> ProjectTaskResponse:
    require_valid_workspace(request.workspace)
    try:
        task = add_manual_task(
            workspace=request.workspace,
            title=request.title,
            description=request.description,
            category=request.category,
            priority=request.priority,
        )
        log_interaction(
            session_id=None,
            user_message=f"project_add_task:{request.title[:120]}",
            assistant_response=f"task_id={task.id}",
            tool_name="project_add_task",
            tool_status="ok",
            workspace=request.workspace,
        )
        return _project_task_response(task)
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"project_add_task:{request.title[:120]}",
            assistant_response=None,
            tool_name="project_add_task",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/project/export_dashboard", response_model=BriefingExportResponse)
async def project_export_dashboard(request: ProjectExportDashboardRequest) -> BriefingExportResponse:
    require_valid_workspace(request.workspace)
    try:
        exported = export_dashboard(workspace=request.workspace, dashboard_id=request.dashboard_id, approved=request.approved)
        log_interaction(
            session_id=None,
            user_message=f"project_export_dashboard:{request.dashboard_id}",
            assistant_response=f"export_path={exported}",
            tool_name="project_export_dashboard",
            tool_status="ok",
            workspace=request.workspace,
        )
        return BriefingExportResponse(exported_path=str(exported))
    except SafetyError as exc:
        log_interaction(
            session_id=None,
            user_message=f"project_export_dashboard:{request.dashboard_id}",
            assistant_response=None,
            tool_name="project_export_dashboard",
            tool_status="rejected",
            workspace=request.workspace,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
