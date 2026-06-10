from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class AppConfig(BaseModel):
    name: str
    version: str
    host: str = "127.0.0.1"
    port: int = 1097


class LLMConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:0.5b"
    temperature: float = 0.2
    max_tokens: int = 512


class SafetyConfig(BaseModel):
    tools_require_approval: bool = True
    shell_enabled: bool = False
    max_file_read_kb: int = 100
    max_search_results: int = 20
    allowed_extensions: list[str] = Field(default_factory=list)
    blocked_path_keywords: list[str] = Field(default_factory=list)

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        return [value.lower() if value.startswith(".") else f".{value.lower()}" for value in values]


class PathsConfig(BaseModel):
    allowed_roots: list[str] = Field(default_factory=list)

    def expanded_roots(self) -> list[Path]:
        return [Path(root).expanduser().resolve() for root in self.allowed_roots]


class WorkspaceConfig(BaseModel):
    description: str
    roots: list[str] = Field(default_factory=list)

    def expanded_roots(self) -> list[Path]:
        return [Path(root).expanduser().resolve() for root in self.roots]


class Settings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    safety: SafetyConfig
    paths: PathsConfig
    workspaces: dict[str, WorkspaceConfig] = Field(default_factory=dict)

    def roots_for_workspace(self, workspace: str | None) -> list[str]:
        if workspace and workspace in self.workspaces:
            return self.workspaces[workspace].roots
        return self.paths.allowed_roots

    def with_allowed_roots(self, roots: list[str]) -> "Settings":
        return self.model_copy(update={"paths": PathsConfig(allowed_roots=roots)})


def load_settings(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    path = Path(config_path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings(DEFAULT_CONFIG_PATH)
