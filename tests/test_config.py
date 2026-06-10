from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import load_settings
from backend.main import app


def test_config_loads() -> None:
    settings = load_settings(Path("config.yaml"))
    assert settings.app.name == "Jarvis Lite"
    assert settings.app.host == "127.0.0.1"
    assert settings.app.version == "0.2.0"
    assert ".md" in settings.safety.allowed_extensions
    assert settings.safety.shell_enabled is False
    assert "research" in settings.workspaces
    assert settings.workspaces["general"].roots == ["~/Documents", "~/Desktop"]


def test_health_works() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Jarvis Lite"
    assert payload["version"] == "0.2.0"
    assert payload["model"] == "qwen2.5:0.5b"
    assert "workspaces" in payload
