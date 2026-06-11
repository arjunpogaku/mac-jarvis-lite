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
        st.write(f"Embeddings created: `{payload.get('embeddings_created', 0)}`")
        st.write(f"Embeddings skipped: `{payload.get('embeddings_skipped', 0)}`")
        if payload.get("embedding_errors"):
            with st.expander("Embedding errors"):
                for item in payload["embedding_errors"]:
                    st.text(item)
        if payload["errors"]:
            with st.expander("Rejected files and errors"):
                for item in payload["errors"]:
                    st.text(item)


def call_workspace_endpoint(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
    response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def request_error_detail(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        payload = response.json()
        detail = payload.get("detail")
        if detail:
            return str(detail)
    except ValueError:
        pass
    return str(exc)


def render_inventory_table(items: list[dict[str, object]]) -> None:
    if not items:
        st.write("No indexed files were returned.")
        return
    table_rows = [
        {
            "file name": item["file_name"],
            "extension": item["extension"],
            "size": item["size_bytes"],
            "modified": item["modified_time"],
            "chunks": item["chunks"],
            "category": item["category"],
        }
        for item in items
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_workspace_cards(cards: list[dict[str, object]]) -> None:
    if not cards:
        st.write("No file cards were generated.")
        return
    for card in cards:
        with st.expander(f"{card['file_name']} · {card['extension']}"):
            st.caption(f"`{card['path']}`")
            st.write(card["short_summary"])
            if card.get("key_points"):
                st.markdown("**Key points**")
                for item in card["key_points"]:
                    st.write(f"- {item}")
            if card.get("detected_topics"):
                st.markdown("**Topics**")
                st.write(", ".join(card["detected_topics"]))
            if card.get("possible_actions"):
                st.markdown("**Possible actions**")
                for item in card["possible_actions"]:
                    st.write(f"- {item}")
            if card.get("warnings"):
                st.markdown("**Warnings**")
                for item in card["warnings"]:
                    st.write(f"- {item}")


def render_workspace_topics(topics: list[dict[str, object]]) -> None:
    if not topics:
        st.write("No topics were identified yet.")
        return
    for topic in topics:
        with st.expander(topic["topic_label"]):
            if topic.get("related_keywords"):
                st.caption(", ".join(topic.get("related_keywords", [])))
            if topic.get("supporting_files"):
                st.markdown("**Supporting files**")
                for item in topic["supporting_files"]:
                    st.write(f"- {item}")
            if topic.get("example_snippets"):
                st.markdown("**Example snippets**")
                for item in topic["example_snippets"]:
                    st.write(f"- {item}")


def render_workspace_issues(issues: list[dict[str, object]]) -> None:
    if not issues:
        st.write("No health issues were detected.")
        return
    grouped: dict[str, list[dict[str, object]]] = {"high": [], "medium": [], "low": []}
    for issue in issues:
        grouped.setdefault(str(issue.get("severity", "low")), []).append(issue)
    for severity in ["high", "medium", "low"]:
        if not grouped.get(severity):
            continue
        with st.expander(f"{severity.title()} severity ({len(grouped[severity])})", expanded=severity == "high"):
            for issue in grouped[severity]:
                st.markdown(f"**{issue['issue_type']}** — `{issue['file_path']}`")
                st.write(issue["explanation"])
                st.caption(f"Suggested next action: {issue['suggested_next_action']}")


def render_workspace_intelligence(selected_workspace: str) -> None:
    st.header("Workspace Intelligence")
    st.info("Workspace Intelligence only uses indexed content from approved folders. It cannot modify, delete, move, or upload files.")

    try:
        inventory = call_workspace_endpoint("/workspace/inventory", {"workspace": selected_workspace, "limit": 100})
    except requests.RequestException as exc:
        st.error(f"Workspace inventory failed: {exc}")
        return

    if inventory.get("total_indexed_files", 0) == 0:
        st.warning("This workspace has not been indexed yet. Please index it from the Knowledge Base page first.")
        return

    st.subheader("Workspace overview")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Selected workspace", inventory["workspace"])
    overview_cols[1].metric("Indexed files", inventory["total_indexed_files"])
    overview_cols[2].metric("Chunks", inventory["total_chunks"])
    overview_cols[3].metric("Last indexed", inventory.get("last_indexed_time") or "unknown")
    if inventory.get("extensions_breakdown"):
        st.caption(
            "Extension breakdown: "
            + ", ".join(f"{item['extension'] or '(none)'}: {item['count']}" for item in inventory["extensions_breakdown"])
        )

    st.subheader("File inventory")
    render_inventory_table(inventory.get("files", []))

    button_cols = st.columns(4)
    with button_cols[0]:
        if st.button("Generate file cards"):
            try:
                st.session_state.workspace_cards_result = call_workspace_endpoint(
                    "/workspace/file_cards",
                    {"workspace": selected_workspace, "limit": 20, "refresh": False},
                )
            except requests.RequestException as exc:
                st.session_state.workspace_cards_result = {"error": f"File cards failed: {exc}"}
    with button_cols[1]:
        if st.button("Generate workspace summary"):
            try:
                st.session_state.workspace_summary_result = call_workspace_endpoint(
                    "/workspace/summary",
                    {"workspace": selected_workspace, "refresh": False, "limit_files": 30},
                )
            except requests.RequestException as exc:
                st.session_state.workspace_summary_result = {"error": f"Workspace summary failed: {exc}"}
    with button_cols[2]:
        if st.button("Explore topics"):
            try:
                st.session_state.workspace_topics_result = call_workspace_endpoint(
                    "/workspace/topics",
                    {"workspace": selected_workspace, "limit": 20},
                )
            except requests.RequestException as exc:
                st.session_state.workspace_topics_result = {"error": f"Topic explorer failed: {exc}"}
    with button_cols[3]:
        if st.button("Run workspace health check"):
            try:
                st.session_state.workspace_health_result = call_workspace_endpoint(
                    "/workspace/health_check",
                    {"workspace": selected_workspace, "limit": 30},
                )
            except requests.RequestException as exc:
                st.session_state.workspace_health_result = {"error": f"Health check failed: {exc}"}

    st.subheader("Document cards")
    workspace_cards = st.session_state.get("workspace_cards_result")
    if workspace_cards:
        if "error" in workspace_cards:
            st.error(workspace_cards["error"])
        else:
            render_workspace_cards(workspace_cards.get("cards", []))

    st.subheader("Workspace summary")
    workspace_summary = st.session_state.get("workspace_summary_result")
    if workspace_summary:
        if "error" in workspace_summary:
            st.error(workspace_summary["error"])
        else:
            st.write(workspace_summary["summary"])
            st.caption(
                f"Indexed files considered: {workspace_summary['indexed_file_count']} of {workspace_summary['total_indexed_files']}"
            )
            if workspace_summary.get("main_topics"):
                st.markdown("**Main topics**")
                st.write(", ".join(workspace_summary["main_topics"]))
            if workspace_summary.get("important_files"):
                st.markdown("**Important files**")
                for item in workspace_summary["important_files"]:
                    st.write(f"- {item}")
            if workspace_summary.get("possible_actions"):
                st.markdown("**Possible actions**")
                for item in workspace_summary["possible_actions"]:
                    st.write(f"- {item}")
            if workspace_summary.get("warnings"):
                st.markdown("**Warnings**")
                for item in workspace_summary["warnings"]:
                    st.write(f"- {item}")

    st.subheader("Topic explorer")
    workspace_topics = st.session_state.get("workspace_topics_result")
    if workspace_topics:
        if "error" in workspace_topics:
            st.error(workspace_topics["error"])
        else:
            render_workspace_topics(workspace_topics.get("topics", []))

    st.subheader("Health check")
    workspace_health = st.session_state.get("workspace_health_result")
    if workspace_health:
        if "error" in workspace_health:
            st.error(workspace_health["error"])
        else:
            render_workspace_issues(workspace_health.get("issues", []))


def render_command_center(selected_workspace: str) -> None:
    st.header("Command Center")
    st.info(
        "Jarvis can plan and run approved local read-only tools. It cannot delete, edit, move, upload, email, "
        "browse the internet, or control your Mac."
    )
    command = st.text_area("Command", placeholder="Ask Jarvis to search, brief, review, or list tasks...", height=100)
    plan_col, execute_col = st.columns([1, 1])
    with plan_col:
        if st.button("Plan", disabled=not command):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/command/plan",
                    json={"message": command, "workspace": selected_workspace, "session_id": st.session_state.session_id},
                    timeout=60,
                )
                response.raise_for_status()
                st.session_state.command_plan = response.json()
                st.session_state.command_message = command
                st.session_state.command_result = None
            except requests.RequestException as exc:
                st.session_state.command_plan = {"error": f"Planning failed: {request_error_detail(exc)}"}

    plan = st.session_state.get("command_plan")
    if plan:
        if "error" in plan:
            st.error(plan["error"])
        else:
            st.subheader("Jarvis Plan")
            st.write(f"Intent: `{plan['intent']}`")
            st.write(f"Agent: `{plan['agent']}`")
            st.write(f"Safety level: `{plan['safety_level']}`")
            if plan.get("refusal"):
                st.warning(plan["refusal"])
            st.markdown("**Plan**")
            for step in plan["plan"]:
                st.write(f"- {step}")
            st.markdown("**Tools needed**")
            st.write(", ".join(plan["tools_needed"]) if plan["tools_needed"] else "No tools")
            st.caption(plan["approval_message"])

    with execute_col:
        approved = st.checkbox("I approve Jarvis to run this safe read-only plan")
        can_execute = bool(plan and "error" not in plan and st.session_state.get("command_message")) and (
            approved or not plan.get("requires_approval")
        )
        if st.button("Execute", disabled=not can_execute):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/command/execute",
                    json={
                        "message": st.session_state.command_message,
                        "workspace": selected_workspace,
                        "approved": approved,
                        "plan_id": plan.get("plan_id"),
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state.command_result = response.json()
            except requests.RequestException as exc:
                st.session_state.command_result = {"error": f"Execution failed: {request_error_detail(exc)}"}

    result = st.session_state.get("command_result")
    if result:
        st.subheader("Jarvis Response")
        if "error" in result:
            st.error(result["error"])
        else:
            st.write(result["answer"])
            if result.get("actions_performed"):
                st.markdown("**Actions performed**")
                for action in result["actions_performed"]:
                    st.write(f"- {action}")
            if result.get("sources_used"):
                st.markdown("**Sources used**")
                for source in result["sources_used"]:
                    st.write(f"- `{source}`")
            if result.get("limitations"):
                st.markdown("**Limitations**")
                for item in result["limitations"]:
                    st.write(f"- {item}")


def render_briefing_mode(selected_workspace: str) -> None:
    st.header("Briefing Mode")
    st.info(
        "Briefing Mode uses indexed local content only. It cannot modify your original files. "
        "Exporting only writes a new Markdown file inside Jarvis's data/exports folder."
    )
    type_labels = {
        "Workspace": "workspace",
        "Research": "research",
        "Codebase": "codebase",
        "Job Application": "job_application",
    }
    left, right = st.columns([1, 1])
    with left:
        st.write(f"Workspace: `{selected_workspace}`")
        briefing_label = st.selectbox("Briefing type", list(type_labels.keys()))
        limit_files = st.slider("Files to consider", min_value=1, max_value=100, value=30)
        refresh = st.checkbox("Refresh generated context", value=False)
        if st.button("Generate briefing"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/briefing/generate",
                    json={
                        "workspace": selected_workspace,
                        "briefing_type": type_labels[briefing_label],
                        "limit_files": limit_files,
                        "refresh": refresh,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state.briefing_result = response.json()
                st.session_state.briefing_export_result = None
            except requests.RequestException as exc:
                st.session_state.briefing_result = {"error": f"Briefing generation failed: {exc}"}

    with right:
        briefing_result = st.session_state.get("briefing_result")
        if briefing_result:
            if "error" in briefing_result:
                st.error(briefing_result["error"])
            else:
                st.caption(
                    f"{briefing_result['briefing_type']} briefing · files considered: "
                    f"{briefing_result['files_considered']} · generated: {briefing_result['generated_at']}"
                )
                if briefing_result["context_limited"]:
                    st.caption("Context was limited before sending to Ollama.")
                st.text_area("Briefing", briefing_result["briefing_text"], height=360)
                st.subheader("Sources Used")
                if not briefing_result["sources_used"]:
                    st.write("No sources used.")
                for source in briefing_result["sources_used"]:
                    st.markdown(f"`{source}`")

                approved = st.checkbox("I approve exporting this generated briefing to data/exports/")
                if st.button("Export as Markdown", disabled=not approved):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/briefing/export",
                            json={
                                "workspace": briefing_result["workspace"],
                                "briefing_type": briefing_result["briefing_type"],
                                "briefing_text": briefing_result["briefing_text"],
                                "approved": approved,
                            },
                            timeout=30,
                        )
                        response.raise_for_status()
                        st.session_state.briefing_export_result = response.json()
                    except requests.RequestException as exc:
                        st.session_state.briefing_export_result = {"error": f"Briefing export failed: {exc}"}

        export_result = st.session_state.get("briefing_export_result")
        if export_result:
            if "error" in export_result:
                st.error(export_result["error"])
            else:
                st.success(f"Exported to {export_result['exported_path']}")


def render_project_dashboard(selected_workspace: str) -> None:
    st.header("Project Dashboard")
    st.info(
        "Project Dashboard uses indexed local content only. It cannot modify, delete, move, rename, upload, "
        "or edit your original files. Tasks are stored only inside Jarvis's local SQLite database."
    )
    project_types = {
        "General": "general",
        "Research": "research",
        "Codebase": "codebase",
        "Job Application": "job_application",
    }
    status_order = ["open", "in_progress", "blocked", "done", "dismissed"]
    categories = ["research", "writing", "coding", "job_application", "data", "config", "cleanup", "unknown"]
    priorities = ["low", "medium", "high"]

    controls, tasks_col = st.columns([1, 1])
    with controls:
        project_label = st.selectbox("Project type", list(project_types.keys()))
        project_type = project_types[project_label]
        if st.button("Generate project dashboard"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/project/dashboard",
                    json={"workspace": selected_workspace, "project_type": project_type, "refresh": False},
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state.project_dashboard_result = response.json()
                st.session_state.project_export_result = None
            except requests.RequestException as exc:
                st.session_state.project_dashboard_result = {"error": f"Project dashboard failed: {exc}"}

        approve_extract = st.checkbox("I approve extracting tasks from indexed local content into Jarvis SQLite.")
        if st.button("Extract tasks from workspace", disabled=not approve_extract):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/project/extract_tasks",
                    json={
                        "workspace": selected_workspace,
                        "project_type": project_type,
                        "limit_files": 30,
                        "refresh": False,
                        "approved": approve_extract,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                st.session_state.project_extract_result = response.json()
            except requests.RequestException as exc:
                st.session_state.project_extract_result = {"error": f"Task extraction failed: {exc}"}

        st.subheader("Manual task")
        manual_title = st.text_input("Task title")
        manual_description = st.text_area("Task description", height=90)
        manual_category = st.selectbox("Task category", categories, index=categories.index("unknown"))
        manual_priority = st.selectbox("Task priority", priorities, index=priorities.index("medium"))
        if st.button("Add manual task", disabled=not manual_title):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/project/add_task",
                    json={
                        "workspace": selected_workspace,
                        "title": manual_title,
                        "description": manual_description,
                        "category": manual_category,
                        "priority": manual_priority,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                st.session_state.project_manual_task_result = response.json()
            except requests.RequestException as exc:
                st.session_state.project_manual_task_result = {"error": f"Manual task failed: {exc}"}

    with tasks_col:
        dashboard = st.session_state.get("project_dashboard_result")
        if dashboard:
            if "error" in dashboard:
                st.error(dashboard["error"])
            else:
                st.subheader(dashboard["title"])
                st.caption(f"Status: {dashboard['status']} · Generated: {dashboard['generated_at']}")
                st.write(dashboard["overview"])
                st.metric("Open tasks", dashboard["open_task_count"])
                st.metric("High priority tasks", dashboard["high_priority_task_count"])
                if dashboard.get("main_topics"):
                    st.markdown("**Main topics**")
                    st.write(", ".join(dashboard["main_topics"]))
                if dashboard.get("risks"):
                    st.markdown("**Risks**")
                    for item in dashboard["risks"]:
                        st.write(f"- {item}")
                if dashboard.get("next_actions"):
                    st.markdown("**Next actions**")
                    for item in dashboard["next_actions"]:
                        st.write(f"- {item}")
                if dashboard.get("sources_used"):
                    st.markdown("**Sources used**")
                    for source in dashboard["sources_used"]:
                        st.write(f"- `{source}`")
                approve_export = st.checkbox("I approve exporting this dashboard to data/exports/")
                if st.button("Export dashboard as Markdown", disabled=not approve_export):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/project/export_dashboard",
                            json={
                                "workspace": selected_workspace,
                                "dashboard_id": dashboard["id"],
                                "approved": approve_export,
                            },
                            timeout=30,
                        )
                        response.raise_for_status()
                        st.session_state.project_export_result = response.json()
                    except requests.RequestException as exc:
                        st.session_state.project_export_result = {"error": f"Dashboard export failed: {exc}"}
        export_result = st.session_state.get("project_export_result")
        if export_result:
            if "error" in export_result:
                st.error(export_result["error"])
            else:
                st.success(f"Exported to {export_result['exported_path']}")

    extracted = st.session_state.get("project_extract_result")
    if extracted:
        st.subheader("Extracted tasks")
        if "error" in extracted:
            st.error(extracted["error"])
        else:
            st.caption(
                f"Created {extracted['tasks_created']} task(s), skipped {extracted['tasks_skipped_duplicates']} duplicate(s)."
            )
            if extracted["context_limited"]:
                st.caption("Context was limited before task extraction.")
            st.dataframe(extracted["tasks"], use_container_width=True, hide_index=True)

    st.subheader("Task board")
    try:
        response = requests.post(f"{BACKEND_URL}/project/tasks", json={"workspace": selected_workspace, "limit": 200}, timeout=30)
        response.raise_for_status()
        tasks = response.json().get("tasks", [])
    except requests.RequestException as exc:
        st.error(f"Task listing failed: {exc}")
        tasks = []
    for status in status_order:
        grouped = [task for task in tasks if task["status"] == status]
        with st.expander(f"{status} ({len(grouped)})", expanded=status in {"open", "blocked"}):
            if not grouped:
                st.write("No tasks.")
            for task in grouped:
                st.markdown(f"**{task['title']}** · `{task['priority']}` · `{task['category']}`")
                if task.get("description"):
                    st.write(task["description"])
                if task.get("source_path"):
                    st.caption(f"Source: {task['source_path']}")
                new_status = st.selectbox(
                    f"Status for task {task['id']}",
                    status_order,
                    index=status_order.index(task["status"]),
                    key=f"task_status_{task['id']}",
                )
                if st.button("Update status", key=f"update_task_{task['id']}") and new_status != task["status"]:
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/project/update_task",
                            json={"task_id": task["id"], "status": new_status},
                            timeout=30,
                        )
                        response.raise_for_status()
                        st.rerun()
                    except requests.RequestException as exc:
                        st.error(f"Task update failed: {exc}")

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
        embeddings = health.get("embeddings", {})
        st.write(f"Embeddings: `{'enabled' if embeddings.get('enabled') else 'disabled'}`")
        st.write(f"Embedding model: `{embeddings.get('model', settings.embeddings.model)}`")
        st.write(f"Embedding model available: `{'yes' if embeddings.get('available') else 'no'}`")
    except requests.RequestException:
        st.write("Backend: `offline`")

    st.subheader("Workspace")
    selected_workspace = st.selectbox("Active workspace", WORKSPACE_OPTIONS, index=WORKSPACE_OPTIONS.index("general") if "general" in WORKSPACE_OPTIONS else 0)
    if selected_workspace in settings.workspaces:
        st.caption(settings.workspaces[selected_workspace].description)

    st.subheader("Approved folders")
    for root in workspace_roots(selected_workspace):
        st.code(root)

    st.caption("Shell tools are disabled in v0.8.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None
if "tool_result" not in st.session_state:
    st.session_state.tool_result = None
if "command_plan" not in st.session_state:
    st.session_state.command_plan = None
if "command_message" not in st.session_state:
    st.session_state.command_message = None
if "command_result" not in st.session_state:
    st.session_state.command_result = None
if "workspace_cards_result" not in st.session_state:
    st.session_state.workspace_cards_result = None
if "workspace_summary_result" not in st.session_state:
    st.session_state.workspace_summary_result = None
if "workspace_topics_result" not in st.session_state:
    st.session_state.workspace_topics_result = None
if "workspace_health_result" not in st.session_state:
    st.session_state.workspace_health_result = None
if "briefing_result" not in st.session_state:
    st.session_state.briefing_result = None
if "briefing_export_result" not in st.session_state:
    st.session_state.briefing_export_result = None
if "project_dashboard_result" not in st.session_state:
    st.session_state.project_dashboard_result = None
if "project_extract_result" not in st.session_state:
    st.session_state.project_extract_result = None
if "project_manual_task_result" not in st.session_state:
    st.session_state.project_manual_task_result = None
if "project_export_result" not in st.session_state:
    st.session_state.project_export_result = None

render_command_center(selected_workspace)

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
    search_mode = st.selectbox("Search mode", ["Keyword", "Semantic", "Hybrid"], index=2)
    kb_query = st.text_input("Knowledge base query")
    kb_limit = st.slider("Search result limit", min_value=1, max_value=20, value=10)
    if st.button("Search knowledge base", disabled=not kb_query):
        endpoint = {
            "Keyword": "/kb/search",
            "Semantic": "/kb/semantic_search",
            "Hybrid": "/kb/hybrid_search",
        }[search_mode]
        try:
            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
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
            st.caption(f"Search mode used: {kb_search_result.get('search_mode_used', 'keyword')}")
            if kb_search_result.get("fallback_message"):
                st.info(kb_search_result["fallback_message"])
            for item in kb_search_result["results"]:
                line_range = ""
                if item["start_line"] is not None and item["end_line"] is not None:
                    line_range = f" lines {item['start_line']}-{item['end_line']}"
                st.markdown(f"`{item['path']}`{line_range}")
                if item.get("combined_score") is not None:
                    st.caption(f"Combined score: {item['combined_score']:.3f}")
                elif item.get("semantic_score") is not None:
                    st.caption(f"Semantic score: {item['semantic_score']:.3f}")
                elif item.get("rank") is not None:
                    st.caption(f"Keyword rank: {item['rank']:.3f}")
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
            st.caption(f"Search mode used: {kb_ask_result.get('search_mode_used', 'keyword')}")
            if kb_ask_result["context_limited"]:
                st.caption("Context was limited before sending to Ollama.")
            st.write(kb_ask_result["answer"])
            st.subheader("Sources Used")
            if not kb_ask_result["sources_used"]:
                st.write("No sources retrieved.")
            for source in kb_ask_result["sources_used"]:
                st.markdown(f"`{source['path']}`")

render_workspace_intelligence(selected_workspace)
render_briefing_mode(selected_workspace)
render_project_dashboard(selected_workspace)

render_approval()
render_tool_result()
