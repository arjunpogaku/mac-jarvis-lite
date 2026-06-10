from pathlib import Path

import pytest

from backend.config import AppConfig, LLMConfig, PathsConfig, SafetyConfig, Settings
from backend.safety import SafetyError, validate_allowed_path
from backend.tools.file_reader import read_text_file
from backend.tools.safe_shell import run_safe_shell, validate_shell_request


def make_settings(tmp_path: Path, *, shell_enabled: bool = False, max_file_read_kb: int = 1) -> Settings:
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.1.0"),
        llm=LLMConfig(),
        safety=SafetyConfig(
            shell_enabled=shell_enabled,
            max_file_read_kb=max_file_read_kb,
            allowed_extensions=[".txt", ".md"],
            blocked_path_keywords=[".ssh", ".env", "private"],
        ),
        paths=PathsConfig(allowed_roots=[str(tmp_path / "allowed")]),
    )


def test_rejects_files_outside_allowed_folders(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(SafetyError, match="outside approved"):
        validate_allowed_path(outside, settings, require_file=True)


def test_rejects_blocked_paths(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    blocked_dir = allowed / ".ssh"
    blocked_dir.mkdir(parents=True)
    secret = blocked_dir / "note.txt"
    secret.write_text("secret", encoding="utf-8")
    settings = make_settings(tmp_path)

    with pytest.raises(SafetyError, match="blocked"):
        validate_allowed_path(secret, settings, require_file=True)


def test_large_files_are_truncated(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    large = allowed / "large.txt"
    large.write_text("a" * 2048, encoding="utf-8")
    settings = make_settings(tmp_path, max_file_read_kb=1)

    content, truncated, size = read_text_file(large, settings)

    assert len(content.encode("utf-8")) == 1024
    assert truncated is True
    assert size == 2048


def test_shell_disabled_by_default(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    settings = make_settings(tmp_path)

    with pytest.raises(SafetyError, match="disabled"):
        validate_shell_request(["pwd"], allowed, settings)


def test_unsafe_shell_commands_are_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    settings = make_settings(tmp_path, shell_enabled=True)

    with pytest.raises(SafetyError):
        validate_shell_request(["rm", "file.txt"], allowed, settings)
    with pytest.raises(SafetyError):
        validate_shell_request(["ls", "|", "cat"], allowed, settings)


def test_safe_shell_can_run_allowlisted_read_only_command_when_enabled(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "note.txt").write_text("hello", encoding="utf-8")
    settings = make_settings(tmp_path, shell_enabled=True)

    output = run_safe_shell(["pwd"], allowed, settings)

    assert str(allowed) in output
