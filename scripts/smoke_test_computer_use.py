"""Manual smoke test of Garvis's computer-use tools on THIS PC, read-only.

Builds the real agent (real tool registry, Win32 desktop backend, UI
Automation, Tesseract OCR, PowerShell) but replaces the LLM with a fixed
script, so no API credits are used and nothing leaves the PC. Every step is
an inspection: nothing is clicked, typed, opened or closed, and no window
is focused. Prints only success flags and counts -- not screen contents.

Run with: python scripts/smoke_test_computer_use.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.build import build_agent  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.events import AgentEvent, EventType  # noqa: E402
from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall  # noqa: E402

REPO = str(Path(__file__).resolve().parents[1])
STEPS = [
    ("screen_info", {}),
    ("list_windows", {}),
    ("inspect_ui", {"limit": 60}),
    ("read_screen_text", {"region": {"x": 0, "y": 0, "width": 800, "height": 300}}),
    ("list_processes", {"limit": 5}),
    ("system_status", {}),
    ("recent_system_errors", {"hours": 24, "limit": 3}),
    ("list_installed_apps", {"filter": "python", "limit": 3}),
    ("run_powershell", {"command": "git status --short | Measure-Object | Select-Object -ExpandProperty Count",
                        "working_directory": REPO}),
    ("find_files", {"directory": REPO, "pattern": "test_*.py", "max_results": 5}),
    ("read_text_file", {"path": str(Path(REPO) / "README.md"), "max_lines": 3}),
]


class _ScriptedLLM(LLMProvider):
    def __init__(self) -> None:
        self.step = 0

    async def chat(self, messages, tools=None):
        if self.step < len(STEPS):
            name, args = STEPS[self.step]
            self.step += 1
            return LLMResponse(
                message=Message(role=Role.ASSISTANT, tool_calls=[ToolCall(id=f"c{self.step}", name=name, arguments=args)]),
                finish_reason="tool_calls",
            )
        return LLMResponse(message=Message(role=Role.ASSISTANT, content="Inspection complete."))

    async def health_check(self) -> bool:
        return True


def _shape(data) -> str:
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            if isinstance(value, list):
                parts.append(f"{key}=[{len(value)}]")
            elif isinstance(value, (int, float, bool)) or value is None:
                parts.append(f"{key}={value}")
        return ", ".join(parts[:6]) or f"keys={sorted(data)[:6]}"
    if isinstance(data, list):
        return f"{len(data)} items"
    return type(data).__name__


async def main() -> None:
    settings = get_settings().model_copy(
        update={"memory_db_path": Path(tempfile.mkdtemp(prefix="garvis_cu_smoke_")) / "m.db"}
    )
    agent = build_agent(settings)
    agent._llm = _ScriptedLLM()  # no LLM calls, no credits
    agent._max_tool_iterations = len(STEPS) + 2
    results: list[tuple[str, bool, str]] = []
    confirmations: list[str] = []
    pending: dict[str, str] = {}

    def on_event(event: AgentEvent) -> None:
        if event.type == EventType.TOOL_STARTED:
            pending["name"] = event.payload["name"]
        elif event.type == EventType.CONFIRMATION_REQUESTED:
            confirmations.append(event.payload["tool_name"])
            agent.confirmations.resolve(event.payload["id"], False)  # never approve anything here

    agent._events.subscribe(on_event)
    original = agent._execute_tool_call

    async def recording(tool_call):
        result = await original(tool_call)
        results.append((tool_call.name, result.success, _shape(result.data) if result.success else (result.error or "")[:120]))
        return result

    agent._execute_tool_call = recording
    context = await agent._context_provider() if agent._context_provider else None
    print("desktop context line:", "present" if context else "none", f"({len(context or '')} chars)")
    reply = await agent.handle_user_message("Inspect the PC (smoke test).")
    for name, ok, detail in results:
        print(f"{'OK  ' if ok else 'FAIL'} {name:22} {detail}")
    print("confirmations requested:", confirmations or "none")
    print("final reply:", json.dumps(reply))
    if confirmations or not all(ok for _, ok, _ in results):
        raise SystemExit(1)
    if agent.browser_manager is not None:
        await agent.browser_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
