from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.config import SafetyConfig, Settings


class SafetyError(ValueError):
    """Raised when a requested operation violates Jarvis Lite safety rules."""


@dataclass(frozen=True)
class ValidatedPath:
    requested: Path
    resolved: Path
    root: Path


@dataclass(frozen=True)
class SafetyCheckResult:
    ok: bool
    message: str


def is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def contains_blocked_keyword(path: Path, blocked_keywords: list[str]) -> bool:
    normalized = str(path).lower()
    parts = [part.lower() for part in path.parts]
    for keyword in blocked_keywords:
        lowered = keyword.lower()
        if lowered in parts or lowered in normalized:
            return True
    return False


def validate_allowed_path(
    requested_path: str | Path,
    settings: Settings,
    *,
    must_exist: bool = True,
    require_file: bool = False,
) -> ValidatedPath:
    requested = Path(requested_path).expanduser()
    resolved = requested.resolve(strict=must_exist)
    roots = settings.paths.expanded_roots()

    if not roots:
        raise SafetyError("No allowed folders are configured.")

    matching_root = next((root for root in roots if is_under_root(resolved, root)), None)
    if matching_root is None:
        raise SafetyError("Path is outside approved folders.")

    # Block known sensitive folders and secrets relative to the approved root.
    # This avoids false positives from macOS system prefixes like /private/var.
    relative_path = resolved.relative_to(matching_root)
    if contains_blocked_keyword(relative_path, settings.safety.blocked_path_keywords):
        raise SafetyError("Path contains a blocked sensitive folder or filename.")

    if must_exist and not resolved.exists():
        raise SafetyError("Path does not exist.")

    if require_file and not resolved.is_file():
        raise SafetyError("Path is not a file.")

    return ValidatedPath(requested=requested, resolved=resolved, root=matching_root)


def check_allowed_path(
    requested_path: str | Path,
    settings: Settings,
    *,
    must_exist: bool = True,
    require_file: bool = False,
) -> SafetyCheckResult:
    try:
        validate_allowed_path(
            requested_path,
            settings,
            must_exist=must_exist,
            require_file=require_file,
        )
        return SafetyCheckResult(ok=True, message="Allowed: path is inside an approved folder.")
    except SafetyError as exc:
        return SafetyCheckResult(ok=False, message=f"Rejected: {exc}")


def ensure_allowed_extension(path: Path, safety: SafetyConfig) -> None:
    if path.suffix.lower() not in safety.allowed_extensions:
        raise SafetyError("File extension is not allowed.")


def max_read_bytes(settings: Settings) -> int:
    return settings.safety.max_file_read_kb * 1024
