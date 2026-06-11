# Jarvis Lite

Jarvis Lite is a harmless, local-first Mac assistant. Version 0.8 adds a Command Center and Agent Router that unify the existing safe local tools behind one conversational interface. It can plan, ask for approval, run approved local read-only actions, and return source-backed results. It is not fully autonomous.

## v0.8 Features

- Command Center for natural-language local commands.
- Agent Router that detects intent, selects an internal agent, proposes a safe plan, and asks for approval when needed.
- Safety refusals for destructive file actions, internet/email requests, shell commands, and Mac system control.
- Workspace-scoped local knowledge base.
- Explicit approval before indexing a workspace.
- SQLite storage with FTS5 keyword search.
- Optional local semantic search using Ollama embeddings.
- Hybrid search that merges keyword and semantic results.
- Local vectors stored as JSON in SQLite.
- Local KB search by workspace.
- Ask questions over retrieved KB chunks using Ollama.
- Source paths shown for KB answers.
- Workspace Intelligence with local file cards, workspace summaries, topic explorer, and health checks.
- Cached file summaries and cached workspace summaries in SQLite.
- Self-check and smoke-test scripts for local quality gates.
- Source-backed Briefing Mode over indexed local content.
- Optional approved Markdown export to `data/exports/`.
- Local Project Dashboard with task extraction, dashboard summaries, task board, and manual tasks.
- Project tasks and dashboards stored only in Jarvis SQLite.
- Command history metadata stored in SQLite without full retrieved context.
- No vector database yet, no cloud APIs, and no internet access.

## Safety Model

- Runs locally and talks only to Ollama at `http://localhost:11434`.
- Binds the backend to `127.0.0.1`.
- Reads and searches only folders listed in `config.yaml`.
- Requires explicit tool approval in each tool request.
- Shows the tool name, target, reason, and safety status before frontend tool calls.
- Supports workspace profiles for scoped local folders.
- Indexes only a selected configured workspace after approval.
- Stores KB chunks and vectors locally in SQLite and never writes back to source files.
- Stores file summaries and workspace summaries locally in SQLite and never writes back to source files.
- Briefing export writes only new Markdown files inside `data/exports/` after approval.
- Project Dashboard stores generated tasks internally in SQLite and never writes to source files.
- Command Center plans first and executes only approved safe local actions.
- Does not delete, modify, move, rename, upload, email, or send files.
- Does not index the whole Mac automatically.
- Keeps shell access implemented but disabled by default.
- If shell access is enabled later, it is read-only and allowlisted.

## Setup

```bash
brew install ollama
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
ollama serve
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional small models:

```bash
ollama pull qwen2.5-coder:0.5b
ollama pull llama3.2:1b
ollama pull all-minilm
```

## Run Backend

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 1097 --reload
```

Health check:

```bash
curl http://127.0.0.1:1097/health
```

```bash
curl -X POST http://127.0.0.1:1097/command/plan \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","message":"Where did I write about PSC-CPM pruning?"}'
```

```bash
curl -X POST http://127.0.0.1:1097/command/execute \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","message":"Where did I write about PSC-CPM pruning?","approved":true}'
```

## Run Frontend

```bash
streamlit run frontend/streamlit_app.py
```

## Test

```bash
pytest
```

## Quality Gate Before Every Change

Every future Codex change must run:

```bash
make quality
```

If `make` is not available, run:

```bash
python -m pytest -q
python scripts/self_check.py
python scripts/smoke_test.py
```

In shells where `python` is only available inside the project virtualenv, activate it first:

```bash
source .venv/bin/activate
```

## Configuration

Edit `config.yaml` to choose a model and approved folders. Jarvis Lite will only read or search paths under `paths.allowed_roots`.

Version 0.8 supports workspace profiles:

```yaml
workspaces:
  research:
    description: "Research papers, LaTeX files, experiments, and notes"
    roots:
      - "~/Documents/Research"
      - "~/Desktop/Research"
  jobs:
    description: "Resumes, cover letters, and job descriptions"
    roots:
      - "~/Documents/Jobs"
  general:
    description: "General documents and desktop files"
    roots:
      - "~/Documents"
      - "~/Desktop"
```

The Streamlit sidebar lets you choose the active workspace. File tools are then scoped to that workspace.

## Knowledge Base

Jarvis Lite v0.8 can index approved workspace files into `data/jarvis.sqlite`.
Jarvis Lite v0.8 can also generate local embeddings during indexing when enabled:

```yaml
embeddings:
  provider: "ollama"
  model: "nomic-embed-text"
  base_url: "http://localhost:11434"
  enabled: true
  vector_dimension: 768

knowledge_base:
  semantic_search_enabled: true
  hybrid_search_enabled: true
  max_embedding_text_chars: 2500
```

Indexing rules:

- Select a workspace in the Streamlit sidebar.
- Click `Index selected workspace`.
- Review the approval card.
- Click `Approve`.

The indexer only reads files under that workspace's configured roots. It respects allowed extensions, blocked path keywords, hidden path rejection, symlink escape rejection, binary-file rejection, and the configured max file size.

Search the KB from the `Knowledge Base` section using Keyword, Semantic, or Hybrid mode. Ask questions over indexed files from the ask box. KB answers use retrieved chunks only and show sources.

## Workspace Intelligence

Workspace Intelligence works only on already-indexed SQLite content from the selected workspace.

It can:

- Show a workspace overview from indexed metadata.
- Build file cards from cached or freshly generated local summaries.
- Generate a workspace summary from file summaries only.
- Explore topics with local keyword extraction and optional local Ollama labeling.
- Run health checks for TODO/FIXME markers, tiny files, repeated content, weak summaries, and large files.

The file card workflow is cached by file content hash. If the indexed content has not changed, Jarvis Lite reuses the stored summary unless you refresh it.

The workspace summary workflow uses file summaries, not raw full files. If some files have no cached summaries yet, Jarvis Lite creates them first for the selected subset.

Workspace Intelligence does not modify source files, does not access the internet, does not index automatically, and does not expand beyond approved workspace roots.

## Briefing Mode

Briefing Mode generates concise reports from indexed local SQLite content only. It does not read arbitrary files directly and does not modify original workspace files.

Briefing types:

- Workspace.
- Research.
- Codebase.
- Job Application.

Each briefing asks the local Ollama model to use only provided indexed context, cite source paths, note weak evidence, and suggest next actions.

Exports are optional. Exporting requires explicit approval and writes a new Markdown file only under:

```text
data/exports/
```

## Project Dashboard

Project Dashboard uses indexed local content only. It does not modify source files. It can:

- Extract tasks from TODO/FIXME markers, health checks, cached summaries, warnings, and local briefing-style context.
- Normalize tasks with local Ollama into internal Jarvis task records.
- Show tasks grouped by status: `open`, `in_progress`, `blocked`, `done`, and `dismissed`.
- Add manual tasks to Jarvis SQLite only.
- Generate a project dashboard with status, topics, risks, next actions, task counts, and sources.
- Export a dashboard to Markdown only inside `data/exports/` after explicit approval.

Generated tasks are suggestions. They can be wrong if the local model is weak or the indexed evidence is sparse, so verify them before acting.

## Command Center

Command Center is the main Jarvis-like interface. You enter a natural command, click `Plan`, review Jarvis's detected intent, selected agent, tools needed, safety level, and approval message, then click `Execute` only if you approve.

Planning never runs tools. Tool-using commands require approval. Simple chat can run without approval.

Supported agents:

- `safety_agent`: refuses unsafe or unsupported requests.
- `file_agent`: safe file search, read, and summarization.
- `knowledge_agent`: local KB keyword, semantic, hybrid search, and KB ask.
- `research_agent`: research review, weak sections, TODOs, source-backed suggestions.
- `code_agent`: codebase overview, TODO/FIXME review, module explanation from indexed content.
- `job_agent`: resume/job description help from indexed local content.
- `project_agent`: project dashboard, task listing, task extraction with approval.
- `briefing_agent`: workspace, research, codebase, and job briefings.

Safe command examples:

- `Where did I write about PSC-CPM pruning?`
- `What should I work on today?`
- `Create a research briefing for this workspace.`
- `Find TODOs in this codebase.`
- `Review my thesis draft sections and cite weak spots.`

Refused command examples:

- `Delete all old files.`
- `Move my photos.`
- `Email this resume.`
- `Open Chrome and apply for jobs.`
- `Download something from the internet.`
- `Run this terminal command.`
- `Change system settings.`

## Tool Approval Workflow

For search, read, and summarize actions, the frontend first creates a pending tool request showing:

- Tool name.
- Target folder or file.
- Reason for the tool.
- Safety status.
- Approve and Cancel buttons.

The backend tool endpoint is called only after Approve is clicked.

## API

### `GET /health`

Returns app name, version, Ollama status, and configured model.
It also returns configured workspaces.
It also returns embedding status.

### `POST /command/plan`

Plans a safe local command without executing tools.

```json
{
  "message": "Where did I write about PSC-CPM pruning?",
  "workspace": "research"
}
```

### `POST /command/execute`

Revalidates and executes an approved safe local command.

```json
{
  "message": "Where did I write about PSC-CPM pruning?",
  "workspace": "research",
  "approved": true
}
```

### `POST /chat`

Sends a message to Ollama. Tools do not run automatically.

```json
{
  "message": "Hello",
  "session_id": "optional-session-id"
}
```

### `POST /tools/search_files`

Searches approved folders. Tool approval is required.

```json
{
  "keyword": "TODO",
  "workspace": "general",
  "requested_path": "",
  "session_id": "optional-session-id",
  "approved": true
}
```

### `POST /tools/read_file`

Reads an approved file, up to the configured maximum size.

```json
{
  "path": "~/Documents/note.md",
  "workspace": "general",
  "requested_path": "~/Documents/note.md",
  "session_id": "optional-session-id",
  "approved": true
}
```

### `POST /tools/summarize_file`

Reads a bounded excerpt from an approved file and summarizes it with Ollama.

```json
{
  "path": "~/Documents/note.md",
  "workspace": "general",
  "requested_path": "~/Documents/note.md",
  "session_id": "optional-session-id",
  "approved": true
}
```

### `POST /kb/index_workspace`

Indexes a configured workspace after explicit approval.

```json
{
  "workspace": "research",
  "approved": true
}
```

### `POST /kb/search`

Searches the local SQLite FTS5 index for one workspace.

```json
{
  "query": "coverage support",
  "workspace": "research",
  "limit": 10
}
```

### `POST /kb/semantic_search`

Searches local chunk embeddings for one workspace. If embeddings are unavailable, Jarvis returns a clear fallback message and uses keyword search where possible.

```json
{
  "query": "reducing candidate expansion",
  "workspace": "research",
  "limit": 10
}
```

### `POST /kb/hybrid_search`

Runs keyword and semantic search, removes duplicate chunks, and ranks with a simple combined score.

```json
{
  "query": "PSC-CPM pruning candidate expansion",
  "workspace": "research",
  "limit": 10
}
```

### `POST /kb/ask`

Retrieves matching chunks from the local index and sends a bounded context to Ollama.
When hybrid search is enabled, `/kb/ask` uses hybrid search. Otherwise it uses keyword search.

```json
{
  "question": "Where did I write about PSC-CPM pruning?",
  "workspace": "research",
  "limit": 5
}
```

### `POST /workspace/inventory`

Returns indexed-file metadata only. It does not reread source files.

```json
{
  "workspace": "research",
  "limit": 100
}
```

### `POST /workspace/file_cards`

Builds local file cards from indexed chunks and caches the result by content hash.

```json
{
  "workspace": "research",
  "limit": 20,
  "refresh": false
}
```

### `POST /workspace/summary`

Builds a cached workspace-level summary from file summaries and metadata only.

```json
{
  "workspace": "research",
  "refresh": false,
  "limit_files": 30
}
```

### `POST /workspace/topics`

Extracts simple keyword-based topics from indexed chunks and returns supporting files.

```json
{
  "workspace": "research",
  "limit": 20
}
```

### `POST /workspace/health_check`

Runs local document health checks over indexed metadata and cached summaries.

```json
{
  "workspace": "research",
  "limit": 30
}
```

### `POST /briefing/generate`

Generates a source-backed briefing from indexed local content.

```json
{
  "workspace": "research",
  "briefing_type": "research",
  "limit_files": 30,
  "refresh": false
}
```

### `POST /briefing/export`

Exports generated briefing text to a new Markdown file in `data/exports/`. Approval is required.

```json
{
  "workspace": "research",
  "briefing_type": "research",
  "briefing_text": "...",
  "approved": true
}
```

### `POST /project/extract_tasks`

Extracts suggested tasks from indexed local content and stores them in Jarvis SQLite. Approval is required.

```json
{
  "workspace": "research",
  "project_type": "research",
  "limit_files": 30,
  "refresh": false,
  "approved": true
}
```

### `POST /project/dashboard`

Generates or returns a cached project dashboard.

```json
{
  "workspace": "research",
  "project_type": "research",
  "refresh": false
}
```

### `POST /project/tasks`

Lists internal Jarvis task records.

```json
{
  "workspace": "research",
  "status": "open",
  "category": "research",
  "priority": "high",
  "limit": 50
}
```

### `POST /project/update_task`

Updates task status in Jarvis SQLite only.

```json
{
  "task_id": 1,
  "status": "in_progress"
}
```

### `POST /project/add_task`

Adds a manual task to Jarvis SQLite only.

```json
{
  "workspace": "research",
  "title": "Rewrite abstract",
  "description": "Improve abstract clarity and reduce length.",
  "category": "writing",
  "priority": "high"
}
```

### `POST /project/export_dashboard`

Exports a generated dashboard and task list to Markdown in `data/exports/`. Approval is required.

```json
{
  "workspace": "research",
  "dashboard_id": 1,
  "approved": true
}
```

## Manual Test Examples

```bash
curl http://127.0.0.1:1097/health
```

```bash
curl -X POST http://127.0.0.1:1097/kb/index_workspace \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"general","approved":true}'
```

```bash
curl -X POST http://127.0.0.1:1097/kb/search \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"general","query":"TODO","limit":5}'
```

```bash
curl -X POST http://127.0.0.1:1097/kb/semantic_search \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"general","query":"reducing candidate expansion","limit":5}'
```

```bash
curl -X POST http://127.0.0.1:1097/kb/hybrid_search \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"general","query":"PSC-CPM pruning candidate expansion","limit":5}'
```

```bash
curl -X POST http://127.0.0.1:1097/kb/ask \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"general","question":"What TODOs are in my indexed files?","limit":5}'
```

```bash
curl -X POST http://127.0.0.1:1097/workspace/inventory \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","limit":100}'
```

```bash
curl -X POST http://127.0.0.1:1097/workspace/file_cards \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","limit":20,"refresh":false}'
```

```bash
curl -X POST http://127.0.0.1:1097/workspace/summary \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","refresh":false,"limit_files":30}'
```

```bash
curl -X POST http://127.0.0.1:1097/workspace/topics \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","limit":20}'
```

```bash
curl -X POST http://127.0.0.1:1097/workspace/health_check \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","limit":30}'
```

```bash
curl -X POST http://127.0.0.1:1097/briefing/generate \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","briefing_type":"research","limit_files":30,"refresh":false}'
```

```bash
curl -X POST http://127.0.0.1:1097/briefing/export \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","briefing_type":"research","briefing_text":"# Briefing","approved":true}'
```

```bash
curl -X POST http://127.0.0.1:1097/project/extract_tasks \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","project_type":"research","limit_files":30,"refresh":false,"approved":true}'
```

```bash
curl -X POST http://127.0.0.1:1097/project/dashboard \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","project_type":"research","refresh":false}'
```

```bash
curl -X POST http://127.0.0.1:1097/project/tasks \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","status":"open","limit":50}'
```

```bash
curl -X POST http://127.0.0.1:1097/project/add_task \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"research","title":"Rewrite abstract","description":"Improve clarity.","category":"writing","priority":"high"}'
```

## Safety Checklist

Jarvis Lite v0.8 cannot:

- Delete files.
- Modify files.
- Move or rename files.
- Upload files.
- Send emails or messages.
- Browse the internet.
- Run arbitrary shell commands.
- Use shell tools by default.
- Search outside approved folders.
- Read hidden sensitive folders blocked in `config.yaml`.
- Index the whole Mac automatically.
- Use cloud embedding APIs.
- Modify, delete, move, rename, upload, or email source files from Workspace Intelligence.
- Read unindexed workspace content in the Workspace Intelligence page.
- Export briefings outside `data/exports/`.
- Export briefings or dashboards outside `data/exports/`.
- Overwrite exports silently.
- Modify source files when creating or updating project tasks.
- Control your Mac.
- Execute tools during planning.

## Known Limitations

- Jarvis is not fully autonomous.
- Command routing is intentionally conservative and may ask for approval often.
- Jarvis works best after workspaces are indexed.
- Jarvis cannot control the Mac, browse the internet, email, upload, or modify original files.
- Workspace Intelligence uses local indexed content only, so it can be wrong if the local model is weak or the indexed content is sparse.
- Larger local Ollama models may improve file cards, workspace summaries, and topic labels.
- Workspace intelligence is cached in SQLite and only refreshes when you ask for it.
- Briefing quality depends on indexed content and the local Ollama model.
- Briefings do not read full raw files; index first, then generate.
- Project Dashboard tasks are internal suggestions; verify them before acting.
- Task extraction depends on indexed content and cached local summaries.
- It does not auto-index on startup.

- Semantic search is optional and falls back to keyword search when embeddings are disabled or unavailable.
- Vectors are stored locally as JSON in SQLite; no vector database is used yet.
- Hybrid ranking is intentionally simple.
- Indexing does not happen automatically.
- Large files over the configured read limit are skipped.
- Only text-like allowed extensions are indexed.
- Ollama must be running for chat, summarize, KB ask, and embedding generation.
