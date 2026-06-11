from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.db import init_db
from backend.tools.safe_shell import validate_shell_request
from backend.safety import SafetyError


def report(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return ok


def is_local_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def workspace_roots_valid() -> tuple[bool, str]:
    settings = get_settings()
    home = Path.home().resolve()
    for name, workspace in settings.workspaces.items():
        if not workspace.roots:
            return False, f"{name} has no roots"
        for raw_root in workspace.roots:
            if not raw_root or not raw_root.strip():
                return False, f"{name} has an empty root"
            resolved = Path(raw_root).expanduser().resolve(strict=False)
            if resolved == Path("/"):
                return False, f"{name} resolves to /"
            if resolved == home:
                return False, f"{name} resolves directly to the home directory"
    return True, ""


def main() -> int:
    print("Jarvis Lite self-check")
    checks: list[bool] = []

    checks.append(report("Python version", sys.version_info >= (3, 10), sys.version.split()[0]))

    try:
        settings = get_settings()
        checks.append(report("Config loaded", True))
    except Exception as exc:
        checks.append(report("Config loaded", False, str(exc)))
        print("Result: FAIL")
        return 1

    try:
        import backend.main  # noqa: F401

        checks.append(report("Backend imports", True))
    except Exception as exc:
        checks.append(report("Backend imports", False, str(exc)))

    try:
        init_db()
        checks.append(report("SQLite initialized", True))
    except Exception as exc:
        checks.append(report("SQLite initialized", False, str(exc)))

    required_dirs = [PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend", PROJECT_ROOT / "data", PROJECT_ROOT / "tests"]
    checks.append(report("Required folders exist", all(path.exists() and path.is_dir() for path in required_dirs)))

    checks.append(report("Shell disabled", settings.safety.shell_enabled is False))
    try:
        validate_shell_request(["pwd"], PROJECT_ROOT, settings)
        checks.append(report("Shell tool rejects execution", False, "pwd unexpectedly allowed"))
    except SafetyError:
        checks.append(report("Shell tool rejects execution", True))

    checks.append(report("Backend host is localhost", settings.app.host == "127.0.0.1", settings.app.host))
    checks.append(report("Ollama URL is localhost", is_local_url(settings.llm.base_url), settings.llm.base_url))
    checks.append(report("Embedding URL is localhost", is_local_url(settings.embeddings.base_url), settings.embeddings.base_url))

    roots_ok, roots_detail = workspace_roots_valid()
    checks.append(report("Workspace roots are valid", roots_ok, roots_detail))

    passed = all(checks)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
