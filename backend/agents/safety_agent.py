from __future__ import annotations

from backend.agents.models import CommandPlan


UNSAFE_PATTERNS = {
    "destructive_file": ["delete", "remove", "rm ", "move ", "rename", "trash", "wipe", "clear old files"],
    "internet_or_email": ["email", "send ", "upload", "download", "internet", "browse", "chrome", "safari", "apply for jobs"],
    "system_or_shell": ["terminal", "shell", "sudo", "chmod", "chown", "kill process", "system settings", "osascript"],
}


def refusal_for_message(message: str) -> str | None:
    lowered = message.lower()
    if any(pattern in lowered for pattern in UNSAFE_PATTERNS["destructive_file"]):
        return "I cannot delete, move, rename, or modify files. I can search for candidate files and generate a review list for you to inspect manually."
    if any(pattern in lowered for pattern in UNSAFE_PATTERNS["internet_or_email"]):
        return "I cannot browse the internet, upload files, or send email. I can help draft local text or review indexed local files instead."
    if any(pattern in lowered for pattern in UNSAFE_PATTERNS["system_or_shell"]):
        return "I cannot run shell commands or control Mac system settings. I can use approved local read-only tools inside configured workspaces."
    return None


def blocked_plan(message: str) -> CommandPlan | None:
    refusal = refusal_for_message(message)
    if not refusal:
        return None
    return CommandPlan(
        intent="unknown",
        agent="safety_agent",
        requires_approval=False,
        safety_level="refused",
        plan=["Refuse the unsafe request and offer a safe local alternative."],
        tools_needed=[],
        approval_message="No tools will be run.",
        refusal=refusal,
    )


def ensure_safe_plan(plan: CommandPlan) -> CommandPlan:
    forbidden_tools = {"shell", "internet", "email", "delete", "move", "rename", "upload"}
    if any(tool in forbidden_tools for tool in plan.tools_needed):
        return CommandPlan(
            intent="unknown",
            agent="safety_agent",
            requires_approval=False,
            safety_level="refused",
            plan=["Refuse the unsafe tool plan."],
            tools_needed=[],
            approval_message="No tools will be run.",
            refusal="I cannot run unsafe tools. I can help with approved local read-only analysis instead.",
        )
    return plan

