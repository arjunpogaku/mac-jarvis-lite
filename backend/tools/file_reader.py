from __future__ import annotations

from pathlib import Path

from backend.config import Settings
from backend.safety import SafetyError, ensure_allowed_extension, max_read_bytes, validate_allowed_path


def read_text_file(path: str | Path, settings: Settings) -> tuple[str, bool, int]:
    validated = validate_allowed_path(path, settings, require_file=True)
    ensure_allowed_extension(validated.resolved, settings.safety)

    limit = max_read_bytes(settings)
    size = validated.resolved.stat().st_size
    if size > limit:
        data = validated.resolved.read_bytes()[:limit]
        truncated = True
    else:
        data = validated.resolved.read_bytes()
        truncated = False

    if b"\x00" in data:
        raise SafetyError("File appears to be binary.")

    text = data.decode("utf-8", errors="replace")
    return text, truncated, size
