from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.config import Settings
from backend.safety import SafetyError, contains_blocked_keyword, ensure_allowed_extension, is_under_root, max_read_bytes


@dataclass(frozen=True)
class SearchMatch:
    path: str
    line_number: int
    line: str


def _iter_candidate_files(settings: Settings) -> list[Path]:
    roots = settings.paths.expanded_roots()
    files: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Block sensitive names only inside the approved root, not in macOS
            # parent path prefixes such as /private/var used for temp folders.
            if contains_blocked_keyword(path.relative_to(root), settings.safety.blocked_path_keywords):
                continue
            if path.suffix.lower() not in settings.safety.allowed_extensions:
                continue
            if not any(is_under_root(path.resolve(), allowed_root) for allowed_root in roots):
                continue
            files.append(path)
    return files


def search_files(keyword: str, settings: Settings) -> list[SearchMatch]:
    query = keyword.strip()
    if not query:
        raise SafetyError("Search keyword is required.")

    results: list[SearchMatch] = []
    limit = max_read_bytes(settings)
    lowered_query = query.lower()

    for path in _iter_candidate_files(settings):
        if len(results) >= settings.safety.max_search_results:
            break
        try:
            ensure_allowed_extension(path, settings.safety)
            if path.stat().st_size > limit:
                continue
            data = path.read_bytes()
            if b"\x00" in data:
                continue
            text = data.decode("utf-8", errors="replace")
        except OSError:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if lowered_query in line.lower():
                results.append(SearchMatch(path=str(path), line_number=line_number, line=line[:500]))
                if len(results) >= settings.safety.max_search_results:
                    break

    return results
