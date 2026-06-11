from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import AppConfig, EmbeddingsConfig, KnowledgeBaseConfig, LLMConfig, PathsConfig, SafetyConfig, Settings, WorkspaceConfig
from backend.safety import SafetyError
from backend.tools.safe_shell import validate_shell_request


class FakeLLM:
    async def health(self) -> bool:
        return True

    async def chat(self, message: str) -> str:
        return "smoke-test chat response"


def print_result(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return ok


async def main_async() -> int:
    import backend.main as main

    checks: list[bool] = []
    original_llm = main.llm_client
    original_settings = main.settings
    main.llm_client = FakeLLM()  # type: ignore[assignment]

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/health")
        checks.append(print_result("/health", health.status_code == 200, str(health.status_code)))

        chat = await client.post("/chat", json={"message": "hello"})
        checks.append(print_result("/chat mocked", chat.status_code == 200 and "smoke-test" in chat.text, str(chat.status_code)))

        invalid = await client.post("/kb/search", json={"workspace": "__missing__", "query": "anything", "limit": 5})
        checks.append(print_result("Invalid workspace rejected", invalid.status_code == 400, str(invalid.status_code)))

        empty = await client.post("/kb/search", json={"workspace": "kbtest", "query": "unlikely_smoke_query", "limit": 5})
        checks.append(print_result("/kb/search handles empty result", empty.status_code == 200, str(empty.status_code)))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspace"
        root.mkdir()
        source_file = root / "source.md"
        source_file.write_text("Original source content", encoding="utf-8")
        before = source_file.read_text(encoding="utf-8")
        main.settings = Settings(
            app=AppConfig(name="Jarvis Lite", version="0.6.0"),
            llm=LLMConfig(),
            embeddings=EmbeddingsConfig(enabled=False),
            knowledge_base=KnowledgeBaseConfig(semantic_search_enabled=False, hybrid_search_enabled=False),
            safety=SafetyConfig(shell_enabled=False, allowed_extensions=[".md"], blocked_path_keywords=[".env", ".ssh"]),
            paths=PathsConfig(allowed_roots=[str(root)]),
            workspaces={"smoke": WorkspaceConfig(description="Smoke workspace", roots=[str(root)])},
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            await client.post("/kb/search", json={"workspace": "smoke", "query": "Original", "limit": 5})
        after = source_file.read_text(encoding="utf-8")
        checks.append(print_result("No endpoint writes source workspace files", before == after))
        try:
            validate_shell_request(["pwd"], root, main.settings)
            checks.append(print_result("Shell remains disabled", False))
        except SafetyError:
            checks.append(print_result("Shell remains disabled", True))

    main.llm_client = original_llm
    main.settings = original_settings

    passed = all(checks)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> int:
    print("Jarvis Lite smoke test")
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
