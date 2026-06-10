from pathlib import Path

from backend.config import AppConfig, LLMConfig, PathsConfig, SafetyConfig, Settings
from backend.main import SearchRequest, settings_for_requested_search_path
from backend.tools.file_search import search_files


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.1.0"),
        llm=LLMConfig(),
        safety=SafetyConfig(
            max_file_read_kb=100,
            max_search_results=20,
            allowed_extensions=[".txt", ".md"],
            blocked_path_keywords=[".ssh", ".env", "private"],
        ),
        paths=PathsConfig(allowed_roots=[str(tmp_path / "allowed")]),
    )


def test_file_search_respects_allowed_extensions(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "note.txt").write_text("needle here", encoding="utf-8")
    (allowed / "script.sh").write_text("needle hidden", encoding="utf-8")
    settings = make_settings(tmp_path)

    results = search_files("needle", settings)

    assert len(results) == 1
    assert results[0].path.endswith("note.txt")
    assert results[0].line_number == 1


def test_file_search_skips_blocked_paths(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    private_dir = allowed / "private"
    private_dir.mkdir(parents=True)
    (private_dir / "secret.txt").write_text("needle", encoding="utf-8")
    (allowed / "public.md").write_text("needle", encoding="utf-8")
    settings = make_settings(tmp_path)

    results = search_files("needle", settings)

    assert len(results) == 1
    assert results[0].path.endswith("public.md")


def test_search_request_scopes_to_requested_folder(tmp_path: Path, monkeypatch) -> None:
    allowed = tmp_path / "allowed"
    nested = allowed / "nested"
    nested.mkdir(parents=True)
    settings = make_settings(tmp_path)
    request = SearchRequest(keyword="needle", requested_path=str(nested), workspace=None, approved=True)

    import backend.main as main

    monkeypatch.setattr(main, "settings", settings)
    scoped_settings, requested_path, safety_status = settings_for_requested_search_path(request)

    assert scoped_settings.paths.allowed_roots == [str(nested.resolve())]
    assert requested_path == str(nested.resolve())
    assert safety_status.startswith("Allowed")
