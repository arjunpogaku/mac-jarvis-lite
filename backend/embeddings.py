from __future__ import annotations

import httpx

from backend.config import Settings


class EmbeddingError(RuntimeError):
    """Raised when local Ollama embeddings are unavailable or invalid."""


class OllamaEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config = settings.embeddings
        self.base_url = self.config.base_url.rstrip("/")

    def get_embedding(self, text: str) -> list[float]:
        if not self.config.enabled:
            raise EmbeddingError("Embeddings are disabled in config.")
        bounded_text = text[: self.settings.knowledge_base.max_embedding_text_chars]
        if not bounded_text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        payload = {"model": self.config.model, "prompt": bounded_text}
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.base_url}/api/embeddings", json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise EmbeddingError("Local Ollama embeddings are unavailable. Is ollama serve running?") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise EmbeddingError(
                f"Ollama embedding request failed for model {self.config.model!r}. "
                f"Try: ollama pull {self.config.model}. Details: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc

        data = response.json()
        vector = data.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise EmbeddingError("Ollama returned no embedding vector.")
        try:
            return [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Ollama returned an invalid embedding vector.") from exc

    def available(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            models = response.json().get("models", [])
            names = {str(model.get("name", "")).split(":")[0] for model in models if isinstance(model, dict)}
            full_names = {str(model.get("name", "")) for model in models if isinstance(model, dict)}
            return self.config.model in names or self.config.model in full_names
        except httpx.HTTPError:
            return False


def get_embedding(text: str, settings: Settings) -> list[float]:
    return OllamaEmbeddingClient(settings).get_embedding(text)
