# Jarvis Lite

Jarvis Lite is a harmless, local-first Mac assistant. Version 0.3 can chat with a local Ollama model, search approved workspaces, read approved text files, summarize approved files, and build a local SQLite knowledge base for fast keyword search. It does not run tools automatically.

## v0.3 Features

- Workspace-scoped local knowledge base.
- Explicit approval before indexing a workspace.
- SQLite storage with FTS5 keyword search.
- Local KB search by workspace.
- Ask questions over retrieved KB chunks using Ollama.
- Source paths shown for KB answers.
- No embeddings yet, no cloud APIs, and no internet access.

## Safety Model

- Runs locally and talks only to Ollama at `http://localhost:11434`.
- Binds the backend to `127.0.0.1`.
- Reads and searches only folders listed in `config.yaml`.
- Requires explicit tool approval in each tool request.
- Shows the tool name, target, reason, and safety status before frontend tool calls.
- Supports workspace profiles for scoped local folders.
- Indexes only a selected configured workspace after approval.
- Stores KB chunks locally in SQLite and never writes back to source files.
- Does not delete, modify, move, rename, upload, email, or send files.
- Does not index the whole Mac automatically.
- Keeps shell access implemented but disabled by default.
- If shell access is enabled later, it is read-only and allowlisted.

## Setup

```bash
brew install ollama
ollama pull qwen2.5:0.5b
ollama serve
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional small models:

```bash
ollama pull qwen2.5-coder:0.5b
ollama pull llama3.2:1b
```

## Run Backend

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 1097 --reload
```

Health check:

```bash
curl http://127.0.0.1:1097/health
```

## Run Frontend

```bash
streamlit run frontend/streamlit_app.py
```

## Test

```bash
pytest
```

## Configuration

Edit `config.yaml` to choose a model and approved folders. Jarvis Lite will only read or search paths under `paths.allowed_roots`.

Version 0.3 supports workspace profiles:

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

Jarvis Lite v0.3 can index approved workspace files into `data/jarvis.sqlite`.

Indexing rules:

- Select a workspace in the Streamlit sidebar.
- Click `Index selected workspace`.
- Review the approval card.
- Click `Approve`.

The indexer only reads files under that workspace's configured roots. It respects allowed extensions, blocked path keywords, hidden path rejection, symlink escape rejection, binary-file rejection, and the configured max file size.

Search the KB from the `Knowledge Base` section using the search box. Ask questions over indexed files from the ask box. KB answers use retrieved chunks only and show sources.

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

### `POST /kb/ask`

Retrieves matching chunks from the local index and sends a bounded context to Ollama.

```json
{
  "question": "Where did I write about PSC-CPM pruning?",
  "workspace": "research",
  "limit": 5
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
curl -X POST http://127.0.0.1:1097/kb/ask \
  -H 'Content-Type: application/json' \
  -d '{"workspace":"general","question":"What TODOs are in my indexed files?","limit":5}'
```

## Safety Checklist

Jarvis Lite v0.3 cannot:

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

## Known Limitations

- KB search uses SQLite FTS5 keyword search, not embeddings.
- Ranking is lexical and may miss semantic matches.
- Indexing does not happen automatically.
- Large files over the configured read limit are skipped.
- Only text-like allowed extensions are indexed.
- Ollama must be running for chat, summarize, and KB ask.
