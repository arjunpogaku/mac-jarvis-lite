from __future__ import annotations

import httpx
import pytest

from backend.config import AppConfig, EmbeddingsConfig, KnowledgeBaseConfig, LLMConfig, PathsConfig, SafetyConfig, Settings
from backend.embeddings import EmbeddingError, OllamaEmbeddingClient


def make_settings() -> Settings:
    return Settings(
        app=AppConfig(name="Jarvis Lite", version="0.4.0"),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(enabled=True, model="nomic-embed-text"),
        knowledge_base=KnowledgeBaseConfig(max_embedding_text_chars=8),
        safety=SafetyConfig(),
        paths=PathsConfig(allowed_roots=[]),
    )


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost:11434/api/embeddings")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("bad status", request=request, response=response)

    def json(self) -> dict:
        return self.payload


class FakeClient:
    posted_payload: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *args) -> None:
        return None

    def post(self, url: str, json: dict) -> FakeResponse:
        self.__class__.posted_payload = json
        return FakeResponse({"embedding": [0.1, 0.2, 0.3]})


class UnavailableClient(FakeClient):
    def post(self, url: str, json: dict) -> FakeResponse:
        raise httpx.ConnectError("no ollama")


def test_embedding_client_handles_normal_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", FakeClient)
    client = OllamaEmbeddingClient(make_settings())

    vector = client.get_embedding("abcdefghijk")

    assert vector == [0.1, 0.2, 0.3]
    assert FakeClient.posted_payload is not None
    assert FakeClient.posted_payload["prompt"] == "abcdefgh"


def test_embedding_client_handles_ollama_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", UnavailableClient)
    client = OllamaEmbeddingClient(make_settings())

    with pytest.raises(EmbeddingError, match="unavailable"):
        client.get_embedding("hello")
