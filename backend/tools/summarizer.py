from __future__ import annotations

from pathlib import Path

from backend.config import Settings
from backend.llm import OllamaClient
from backend.tools.file_reader import read_text_file


async def summarize_file(path: str | Path, settings: Settings, llm: OllamaClient) -> tuple[str, bool]:
    text, truncated, _size = read_text_file(path, settings)
    excerpt = text[: settings.llm.max_tokens * 8]
    prompt = (
        "Summarize this local file briefly. Mention the main topic, important details, "
        "and anything that looks like a TODO or action item. Do not invent details.\n\n"
        f"File excerpt:\n{excerpt}"
    )
    summary = await llm.chat(prompt)
    return summary, truncated or len(excerpt) < len(text)
