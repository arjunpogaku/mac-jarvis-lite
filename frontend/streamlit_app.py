from __future__ import annotations

import requests
import streamlit as st

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings


settings = get_settings()
BACKEND_URL = f"http://{settings.app.host}:{settings.app.port}"
WORKSPACE_OPTIONS = list(settings.workspaces.keys()) or ["default"]


def workspace_roots(workspace: str) -> list[str]:
    if workspace in settings.workspaces:
        return settings.workspaces[workspace].roots
    return settings.paths.allowed_roots


def describe_safety(target: str, workspace: str, *, is_search: bool) -> str:
    roots = workspace_roots(workspace)
    if is_search and not target:
        return "Allowed after approval: search will be limited to the selected workspace roots."

    expanded_target = Path(target).expanduser()
    try:
        resolved_target = expanded_target.resolve(strict=False)
    except OSError:
        return "Needs backend check: path could not be resolved locally."

    for blocked in settings.safety.blocked_path_keywords:
        if blocked.lower() in str(resolved_target).lower():
            return "Rejected by policy: path contains a blocked sensitive keyword."

    for root in roots:
        resolved_root = Path(root).expanduser().resolve(strict=False)
        try:
            resolved_target.relative_to(resolved_root)
            return "Allowed after approval: target appears inside the selected workspace."
        except ValueError:
            continue
    return "Rejected by policy: target appears outside the selected workspace."


def stage_tool_request(
    tool_name: str,
    target: str,
    reason: str,
    workspace: str,
    payload: dict[str, object],
    safety: str,
    requested_path: str | None = None,
) -> None:
    st.session_state.pending_tool = {
        "tool_name": tool_name,
        "target": target,
        "requested_path": requested_path if requested_path is not None else target,
        "reason": reason,
        "workspace": workspace,
        "payload": payload,
        "safety": safety,
    }


def render_approval() -> None:
    pending = st.session_state.get("pending_tool")
    if not pending:
        return

    st.divider()
    st.subheader("Tool Approval")
    st.write(f"Tool: `{pending['tool_name']}`")
    st.write(f"Target: `{pending['target']}`")
    st.write(f"Reason: {pending['reason']}")
    st.write(f"Safety status: {pending['safety']}")

    approve_col, cancel_col = st.columns(2)
    with approve_col:
        if st.button("Approve", type="primary"):
            run_pending_tool(pending)
            st.session_state.pending_tool = None
            st.rerun()
    with cancel_col:
        if st.button("Cancel"):
            st.session_state.tool_result = {"kind": "cancelled", "message": "Tool request cancelled."}
            st.session_state.pending_tool = None
            st.rerun()


def run_pending_tool(pending: dict[str, object]) -> None:
    tool_name = str(pending["tool_name"])
    endpoint = {
        "search_files": "/tools/search_files",
        "read_file": "/tools/read_file",
        "summarize_file": "/tools/summarize_file",
        "kb_index_workspace": "/kb/index_workspace",
    }[tool_name]
    payload = dict(pending["payload"])
    payload.update(
        {
            "tool_name": tool_name,
            "requested_path": pending["requested_path"],
            "workspace": pending["workspace"],
            "session_id": st.session_state.session_id,
            "approved": True,
            "safety_check_result": pending["safety"],
        }
    )
    try:
        response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=90)
        response.raise_for_status()
        st.session_state.tool_result = {"kind": tool_name, "payload": response.json()}
    except requests.RequestException as exc:
        st.session_state.tool_result = {"kind": "error", "message": f"{tool_name} failed: {exc}"}


def render_tool_result() -> None:
    result = st.session_state.get("tool_result")
    if not result:
        return

    kind = result["kind"]
    if kind == "cancelled":
        st.info(result["message"])
        return
    if kind == "error":
        st.error(result["message"])
        return

    payload = result["payload"]
    st.caption(f"Safety check: {payload.get('safety_check_result', 'not provided')}")
    if kind == "search_files":
        results = payload["results"]
        if not results:
            st.write("No matches.")
        for item in results:
            st.markdown(f"`{item['path']}:{item['line_number']}`")
            st.text(item["line"])
    elif kind == "read_file":
        if payload["truncated"]:
            st.caption("The file was truncated before display.")
        st.text_area("File contents", payload["content"], height=260)
    elif kind == "summarize_file":
        if payload["truncated"]:
            st.caption("The file was truncated before summarizing.")
        st.write(payload["summary"])
    elif kind == "kb_index_workspace":
        st.write(f"Workspace: `{payload['workspace']}`")
        st.write(f"Scanned files: `{payload['scanned_file_count']}`")
        st.write(f"Indexed files: `{payload['indexed_file_count']}`")
        st.write(f"Skipped unchanged: `{payload['skipped_unchanged_count']}`")
        st.write(f"Rejected files: `{payload['rejected_file_count']}`")
        st.write(f"Chunks created: `{payload['total_chunks_created']}`")
        if payload["errors"]:
            with st.expander("Rejected files and errors"):
                for item in payload["errors"]:
                    st.text(item)

st.set_page_config(page_title="Jarvis Lite", page_icon="J", layout="wide")

st.title("Jarvis Lite")
st.warning("Jarvis Lite can only read/search approved folders. It cannot delete or modify files.")

with st.sidebar:
    st.subheader("Status")
    st.write(f"Model: `{settings.llm.model}`")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
        st.write(f"Backend: `ok`")
        st.write(f"Ollama: `{'ok' if health.get('ollama_ok') else 'offline'}`")
    except requests.RequestException:
        st.write("Backend: `offline`")

    st.subheader("Workspace")
    selected_workspace = st.selectbox("Active workspace", WORKSPACE_OPTIONS, index=WORKSPACE_OPTIONS.index("general") if "general" in WORKSPACE_OPTIONS else 0)
    if selected_workspace in settings.workspaces:
        st.caption(settings.workspaces[selected_workspace].description)

    st.subheader("Approved folders")
    for root in workspace_roots(selected_workspace):
        st.code(root)

    st.caption("Shell tools are disabled in v0.3.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None
if "tool_result" not in st.session_state:
    st.session_state.tool_result = None

chat_col, tools_col = st.columns([2, 1])

with chat_col:
    st.subheader("Chat")
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    prompt = st.chat_input("Ask Jarvis Lite")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": prompt, "session_id": st.session_state.session_id},
                    timeout=90,
                )
                response.raise_for_status()
                payload = response.json()
                st.session_state.session_id = payload["session_id"]
                answer = payload["response"]
            except requests.RequestException as exc:
                answer = f"Backend request failed: {exc}"
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tools_col:
    st.subheader("File Search")
    keyword = st.text_input("Keyword")
    search_folder = st.text_input("Optional folder inside workspace", value="")
    if st.button("Review search request", disabled=not keyword):
        target = search_folder.strip() or ", ".join(workspace_roots(selected_workspace))
        stage_tool_request(
            "search_files",
            target,
            f"Search approved local files for {keyword!r}.",
            selected_workspace,
            {"keyword": keyword},
            describe_safety(search_folder.strip(), selected_workspace, is_search=True),
            requested_path=search_folder.strip(),
        )

    st.subheader("Read File")
    file_path = st.text_input("Approved file path")
    if st.button("Review read request", disabled=not file_path):
        stage_tool_request(
            "read_file",
            file_path,
            "Read a bounded excerpt from an approved local text file.",
            selected_workspace,
            {"path": file_path},
            describe_safety(file_path, selected_workspace, is_search=False),
        )

    st.subheader("Summarize File")
    summary_path = st.text_input("File to summarize")
    if st.button("Review summary request", disabled=not summary_path):
        stage_tool_request(
            "summarize_file",
            summary_path,
            "Read a bounded excerpt and ask the local Ollama model for a short summary.",
            selected_workspace,
            {"path": summary_path},
            describe_safety(summary_path, selected_workspace, is_search=False),
        )

st.header("Knowledge Base")
st.info("Jarvis will only read approved folders in this workspace. It cannot modify or delete files.")

kb_col, ask_col = st.columns(2)
with kb_col:
    st.subheader("Index")
    st.write(f"Selected workspace: `{selected_workspace}`")
    if st.button("Index selected workspace"):
        stage_tool_request(
            "kb_index_workspace",
            selected_workspace,
            "Read approved workspace files into the local SQLite knowledge base.",
            selected_workspace,
            {"workspace": selected_workspace},
            "Allowed after approval: indexing is limited to configured roots for this workspace.",
            requested_path=selected_workspace,
        )

    st.subheader("Search")
    kb_query = st.text_input("Knowledge base query")
    kb_limit = st.slider("Search result limit", min_value=1, max_value=20, value=10)
    if st.button("Search knowledge base", disabled=not kb_query):
        try:
            response = requests.post(
                f"{BACKEND_URL}/kb/search",
                json={"query": kb_query, "workspace": selected_workspace, "limit": kb_limit},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.kb_search_result = response.json()
        except requests.RequestException as exc:
            st.session_state.kb_search_result = {"error": f"KB search failed: {exc}"}

    kb_search_result = st.session_state.get("kb_search_result")
    if kb_search_result:
        if "error" in kb_search_result:
            st.error(kb_search_result["error"])
        else:
            for item in kb_search_result["results"]:
                line_range = ""
                if item["start_line"] is not None and item["end_line"] is not None:
                    line_range = f" lines {item['start_line']}-{item['end_line']}"
                st.markdown(f"`{item['path']}`{line_range}")
                st.text(item["snippet"])

with ask_col:
    st.subheader("Ask")
    kb_question = st.text_area("Question over indexed files", height=120)
    ask_limit = st.slider("Context chunks", min_value=1, max_value=10, value=5)
    if st.button("Ask indexed files", disabled=not kb_question):
        try:
            response = requests.post(
                f"{BACKEND_URL}/kb/ask",
                json={"question": kb_question, "workspace": selected_workspace, "limit": ask_limit},
                timeout=90,
            )
            response.raise_for_status()
            st.session_state.kb_ask_result = response.json()
        except requests.RequestException as exc:
            st.session_state.kb_ask_result = {"error": f"KB ask failed: {exc}"}

    kb_ask_result = st.session_state.get("kb_ask_result")
    if kb_ask_result:
        if "error" in kb_ask_result:
            st.error(kb_ask_result["error"])
        else:
            if kb_ask_result["context_limited"]:
                st.caption("Context was limited before sending to Ollama.")
            st.write(kb_ask_result["answer"])
            st.subheader("Sources Used")
            if not kb_ask_result["sources_used"]:
                st.write("No sources retrieved.")
            for source in kb_ask_result["sources_used"]:
                st.markdown(f"`{source['path']}`")

render_approval()
render_tool_result()
