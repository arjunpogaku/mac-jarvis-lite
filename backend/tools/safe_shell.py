from __future__ import annotations

import subprocess
from pathlib import Path

from backend.config import Settings
from backend.safety import SafetyError, validate_allowed_path


ALLOWED_COMMANDS = {"pwd", "ls", "find", "rg", "grep", "cat", "head", "tail", "wc", "file", "stat"}
FORBIDDEN_TOKENS = {
    "sudo",
    ">",
    ">>",
    "<",
    "|",
    "&&",
    "||",
    ";",
    "&",
    "rm",
    "mv",
    "cp",
    "chmod",
    "chown",
    "curl",
    "wget",
    "ssh",
    "scp",
    "osascript",
    "open",
    "kill",
    "pkill",
    "killall",
    "brew",
    "pip",
    "npm",
}


def validate_shell_request(command: list[str], cwd: str | Path, settings: Settings) -> Path:
    if not settings.safety.shell_enabled:
        raise SafetyError("Shell access is disabled.")
    if not command:
        raise SafetyError("Command is required.")
    executable = command[0]
    if executable not in ALLOWED_COMMANDS:
        raise SafetyError("Command is not allowlisted.")
    if any(token in FORBIDDEN_TOKENS for token in command):
        raise SafetyError("Command contains a forbidden token.")
    if any(any(marker in token for marker in ("|", ">", "<", ";", "&")) for token in command):
        raise SafetyError("Command chaining and redirection are forbidden.")

    validated_cwd = validate_allowed_path(cwd, settings, require_file=False)
    return validated_cwd.resolved


def run_safe_shell(command: list[str], cwd: str | Path, settings: Settings) -> str:
    safe_cwd = validate_shell_request(command, cwd, settings)
    result = subprocess.run(
        command,
        cwd=safe_cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = result.stdout if result.returncode == 0 else result.stderr
    return output[:20_000]
