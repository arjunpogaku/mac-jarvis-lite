# Jarvis Lite

Jarvis Lite is a harmless, local-first Mac assistant. Version 0.2 can chat with a local Ollama model, search approved workspaces, read approved text files, and summarize approved files. It does not run tools automatically.

## Safety Model

- Runs locally and talks only to Ollama at `http://localhost:11434`.
- Binds the backend to `127.0.0.1`.
- Reads and searches only folders listed in `config.yaml`.
- Requires explicit tool approval in each tool request.
- Shows the tool name, target, reason, and safety status before frontend tool calls.
- Supports workspace profiles for scoped local folders.
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

Version 0.2 also supports workspace profiles:

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

## Safety Checklist

Jarvis Lite v0.1 cannot:

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
