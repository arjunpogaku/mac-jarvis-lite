from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.config import Settings, get_settings
from backend.db import init_db, log_interaction
from backend.llm import OllamaClient
from backend.safety import SafetyError, check_allowed_path, validate_allowed_path
from backend.tools.file_reader import read_text_file
from backend.tools.file_search import search_files
from backend.tools.indexer import IndexSummary, ask_kb, index_workspace, search_kb
from backend.tools.summarizer import summarize_file


settings = get_settings()
llm_client = OllamaClient(settings.llm)


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
    chunk_id: int
    file_id: int
    start_line: int | None
    end_line: int | None


class KBSearchResponse(BaseModel):
    workspace: str
    query: str
    results: list[KBSearchResultResponse]


class KBAskRequest(BaseModel):
    question: str = Field(min_length=1)
    workspace: str
    limit: int = Field(default=5, ge=1, le=10)


class KBAskResponse(BaseModel):
    answer: str
    sources_used: list[KBSearchResultResponse]
    context_limited: bool


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
    }


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
        summary: IndexSummary = index_workspace(settings, request.workspace)
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
