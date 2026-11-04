"""Computer-use tools on a fake desktop (no real input, windows, OCR or UI
Automation): desktop context, window management, reading the screen,
pressing controls, the commit guard, and a multi-step agent task."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import DesktopError
from app.desktop.backend import Rect, WindowInfo, WindowState
from app.desktop.controller import DesktopController
from app.desktop.ocr import OcrEngine, OcrLine, OcrResult
from app.desktop.uia import UiElement, UiInspector, best_match
from app.security.confirmations import ConfirmationManager
from app.security.permissions import PermissionChecker, PermissionDecision
from app.tools.desktop import build_desktop_tools
from app.tools.registry import ToolRegistry
from tests.conftest import final_response, tool_call_response
from tests.desktop_fakes import GARVIS_PID, FakeDesktopBackend


class FakeUia(UiInspector):
    def __init__(self, elements: dict[int, list[UiElement]] | None = None, at: UiElement | None = None):
        self.by_window = elements or {}
        self.at = at
        self.activated: list[tuple[int, str]] = []
        self.patterns: dict[str, str] = {}

    async def elements(self, handle, query=None, limit=120):
        items = self.by_window.get(handle, [])
        if query:
            items = [e for e in items if query.lower() in (e.name + e.automation_id).lower()]
        return items[:limit]

    async def element_at(self, x, y):
        return self.at

    async def activate(self, handle, name, control_type=None):
        element = best_match(self.by_window.get(handle, []), name, control_type)
        if element is None:
            raise DesktopError(f"No control named '{name}'")
        self.activated.append((handle, element.name))
        return {"element": element, "action": self.patterns.get(element.name, "invoked")}


class FakeOcr(OcrEngine):
    def __init__(self, lines: list[OcrLine]):
        self.lines = lines
        self.calls: list[tuple[str, int, int]] = []

    async def recognize(self, image_path, offset_x=0, offset_y=0):
        self.calls.append((Path(image_path).name, offset_x, offset_y))
        shifted = [line.model_copy(update={"left": line.left + offset_x, "top": line.top + offset_y}) for line in self.lines]
        return OcrResult(lines=shifted, engine="fake", elapsed_ms=1, cached=len(self.calls) > 1)


def _el(name, control_type="button", rect=(10, 10, 50, 30), **kw) -> UiElement:
    return UiElement(name=name, control_type=control_type, rect=Rect(left=rect[0], top=rect[1], right=rect[2], bottom=rect[3]), **kw)


@pytest.fixture
def backend():
    b = FakeDesktopBackend()
    b.windows.append(WindowInfo(handle=4, title="Windows PowerShell", process_name="powershell.exe", pid=300,
                                rect=Rect(left=0, top=0, right=800, bottom=600)))
    return b


@pytest.fixture
def controller(backend, tmp_path):
    return DesktopController(backend_factory=lambda: backend, screenshot_dir=tmp_path, own_pid=GARVIS_PID)


def _tools(controller, uia=None, ocr=None, confirm=False):
    return {t.name: t for t in build_desktop_tools(controller, confirm, ocr=ocr, uia=uia)}


async def _decide(tool, args: dict) -> PermissionDecision:
    return (await PermissionChecker().evaluate_call(tool, tool.validate_arguments(args))).decision


# ------------------------------------------------------- desktop context
async def test_active_window_skips_garvis_itself(controller, backend):
    backend.foreground = 3  # the user is typing into Garvis
    window = await controller.resolve_window(None)
    assert window.process_name != "python.exe"


async def test_a_shell_surface_in_the_foreground_is_not_the_active_window(controller, backend, monkeypatch):
    """Clicking the desktop or taskbar makes explorer.exe's shell window the
    foreground; it isn't an app window (list_windows leaves it out), so the
    topmost real application counts as active -- regression from a real run."""
    shell = WindowInfo(handle=77, title="", process_name="explorer.exe", pid=10)
    monkeypatch.setattr(backend, "foreground_window", lambda: shell)
    window = await controller.resolve_window(None)
    assert window.process_name == "Notepad.exe"


async def test_context_line_names_the_active_window_and_other_apps_only(controller, backend):
    line = await controller.describe_context()
    assert line.startswith("active window 'Untitled - Notepad' (Notepad.exe)")
    assert "WhatsApp.Root" in line and "powershell" in line
    assert "Garvis" not in line  # Garvis's own window isn't "the desktop"


# ------------------------------------------------------ window management
async def test_window_actions_are_verified(controller, backend):
    tools = _tools(controller)
    result = await tools["window_action"].execute(tools["window_action"].validate_arguments({"action": "minimize", "window": "notepad"}))
    assert result.success and result.data["minimized"] is True
    assert ("state", 1, WindowState.MINIMIZE) in backend.calls

    result = await tools["window_action"].execute(tools["window_action"].validate_arguments({"action": "close", "window": "whatsapp"}))
    assert result.success and result.data["closed"] is True
    assert backend.window_info(2) is None


async def test_window_action_refuses_to_touch_garvis(controller):
    tool = _tools(controller)["window_action"]
    result = await tool.execute(tool.validate_arguments({"action": "close", "window": "Garvis"}))
    assert not result.success and "Garvis" in result.error


async def test_wait_for_window_verifies_an_app_started(controller, backend):
    tool = _tools(controller)["wait_for_window"]

    async def app_starts_later():
        await asyncio.sleep(0.1)
        backend.windows.append(WindowInfo(handle=9, title="Calculator", process_name="CalculatorApp.exe", pid=900))

    starter = asyncio.create_task(app_starts_later())
    controller_wait = tool.execute(tool.validate_arguments({"window": "calculator", "timeout_seconds": 3}))
    result = await controller_wait
    await starter
    assert result.success and result.data["title"] == "Calculator"

    missing = await tool.execute(tool.validate_arguments({"window": "photoshop", "timeout_seconds": 0.5}))
    assert not missing.success and "within" in missing.error


# ------------------------------------------------------ reading the screen
async def test_read_screen_text_reads_the_active_window_with_screen_coordinates(controller, backend):
    ocr = FakeOcr([OcrLine(text="Error: disk full", confidence=91, left=5, top=5, width=100, height=12),
                   OcrLine(text="API_KEY=sk-abcdefghijklmnopqrstuvwxyz", confidence=90, left=5, top=30, width=200, height=12)])
    tool = _tools(controller, ocr=ocr)["read_screen_text"]
    result = await tool.execute(tool.validate_arguments({"find": "error"}))
    assert result.success
    assert result.data["source"] == "Untitled - Notepad"
    assert ocr.calls[0][1:] == (100, 100)  # the window's top-left offset
    assert result.data["matches"][0]["center"] == [155, 111]
    assert "sk-abcdef" not in result.data["text"] and "[REDACTED]" in result.data["text"]
    again = await tool.execute(tool.validate_arguments({}))
    assert again.data["unchanged_since_last_read"] is True


async def test_read_screen_text_region_and_missing_engine(controller, backend):
    ocr = FakeOcr([])
    tool = _tools(controller, ocr=ocr)["read_screen_text"]
    result = await tool.execute(tool.validate_arguments({"region": {"x": 50, "y": 60, "width": 200, "height": 100}}))
    assert result.success and result.data["source"] == "region"
    assert ocr.calls[0][1:] == (50, 60)
    no_ocr = _tools(controller)["read_screen_text"]
    assert not (await no_ocr.execute(no_ocr.validate_arguments({}))).success


async def test_inspect_ui_lists_controls_without_secrets(controller):
    uia = FakeUia({1: [_el("Save"), _el("", "edit", automation_id="pwd"), _el("Token", "edit", value="ghp_abcdefghijklmnopqrstuvwxyz0123456789")]})
    tool = _tools(controller, uia=uia)["inspect_ui"]
    result = await tool.execute(tool.validate_arguments({}))
    assert result.success and result.data["window"] == "Untitled - Notepad"
    names = [c["name"] for c in result.data["controls"]]
    assert names[0] == "Save" and result.data["controls"][0]["center"] == [30, 20]
    assert "ghp_" not in str(result.data)


# --------------------------------------------------------- pressing controls
async def test_click_ui_element_invokes_without_the_mouse(controller, backend):
    uia = FakeUia({1: [_el("Save"), _el("Cancel")]})
    tool = _tools(controller, uia=uia)["click_ui_element"]
    result = await tool.execute(tool.validate_arguments({"name": "save"}))
    assert result.success and result.data["action"] == "invoked"
    assert uia.activated == [(1, "Save")]
    assert not any(c[0] == "click" for c in backend.calls)


async def test_click_ui_element_falls_back_to_a_real_click(controller, backend):
    uia = FakeUia({1: [_el("Canvas item", rect=(200, 200, 260, 240))]})
    uia.patterns["Canvas item"] = "needs_click"
    tool = _tools(controller, uia=uia)["click_ui_element"]
    result = await tool.execute(tool.validate_arguments({"name": "Canvas item"}))
    assert result.success and result.data["action"] == "clicked"
    assert ("move", 230, 220) in backend.calls and any(c[0] == "click" for c in backend.calls)


# ----------------------------------------------------------- commit guard
async def test_routine_input_runs_without_confirmation(controller):
    uia = FakeUia(at=_el("Open"))
    tools = _tools(controller, uia=uia)
    assert await _decide(tools["mouse_click"], {"x": 20, "y": 20}) == PermissionDecision.ALLOW
    assert await _decide(tools["keyboard_type"], {"text": "hello"}) == PermissionDecision.ALLOW
    assert await _decide(tools["keyboard_press"], {"key": "tab"}) == PermissionDecision.ALLOW
    assert await _decide(tools["keyboard_press"], {"key": "enter"}) == PermissionDecision.ALLOW  # Notepad
    assert await _decide(tools["keyboard_hotkey"], {"keys": "ctrl+s"}) == PermissionDecision.ALLOW
    assert await _decide(tools["click_ui_element"], {"name": "Open"}) == PermissionDecision.ALLOW


async def test_clicking_a_committing_control_needs_confirmation(controller):
    tools = _tools(controller, uia=FakeUia({1: [_el("Send message")]}, at=_el("Send")))
    assert await _decide(tools["mouse_click"], {"x": 20, "y": 20}) == PermissionDecision.REQUIRE_CONFIRMATION
    assert await _decide(tools["mouse_double_click"], {}) == PermissionDecision.REQUIRE_CONFIRMATION
    assert await _decide(tools["mouse_right_click"], {"x": 20, "y": 20}) == PermissionDecision.ALLOW
    # The model said "message", but the control that would be pressed is "Send message".
    assert await _decide(tools["click_ui_element"], {"name": "message"}) == PermissionDecision.REQUIRE_CONFIRMATION


async def test_enter_in_terminals_and_chat_apps_needs_confirmation(controller):
    tools = _tools(controller)
    assert await _decide(tools["keyboard_press"], {"key": "enter", "window": "powershell"}) == PermissionDecision.REQUIRE_CONFIRMATION
    assert await _decide(tools["keyboard_type"], {"text": "rm -rf x\n", "window": "powershell"}) == PermissionDecision.REQUIRE_CONFIRMATION
    assert await _decide(tools["keyboard_type"], {"text": "hi\n", "window": "whatsapp"}) == PermissionDecision.REQUIRE_CONFIRMATION
    assert await _decide(tools["keyboard_type"], {"text": "hi", "window": "whatsapp"}) == PermissionDecision.ALLOW
    assert await _decide(tools["keyboard_type"], {"text": "line\n", "window": "notepad"}) == PermissionDecision.ALLOW


@pytest.mark.parametrize("keys", ["ctrl+enter", "shift+delete", "alt+s"])
async def test_send_and_permanent_delete_hotkeys_always_need_confirmation(controller, keys):
    tool = _tools(controller)["keyboard_hotkey"]
    assert await _decide(tool, {"keys": keys, "window": "notepad"}) == PermissionDecision.REQUIRE_CONFIRMATION


# ---------------------------------------------------------- multi-step task
class _ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        return self.responses.pop(0)

    async def health_check(self):
        return True


async def test_multi_step_task_inspects_acts_and_verifies(controller, backend, tmp_path):
    """Find the open Notepad, read it, press Save (routine, no prompt), then
    press Send in WhatsApp (committing: confirmed first), verifying via
    tool results the whole way."""
    uia = FakeUia({1: [_el("Save"), _el("Text editor", "document", text="draft text")],
                   2: [_el("Send"), _el("Type a message", "edit")]})
    registry = ToolRegistry()
    for tool in build_desktop_tools(controller, ocr=FakeOcr([]), uia=uia):
        registry.register(tool)
    llm = _ScriptedLLM([
        tool_call_response("wait_for_window", {"window": "notepad", "timeout_seconds": 1}),
        tool_call_response("inspect_ui", {"window": "notepad"}),
        tool_call_response("click_ui_element", {"name": "Save", "window": "notepad"}),
        tool_call_response("click_ui_element", {"name": "Send", "window": "whatsapp"}),
        final_response("Saved; sent."),
    ])
    events = EventBus()
    seen: list[AgentEvent] = []
    events.subscribe(seen.append)
    agent = Agent(llm=llm, registry=registry, memory=SQLiteMemoryStore(tmp_path / "m.db"), events=events,
                  confirmations=ConfirmationManager(default_timeout_seconds=3), max_tool_iterations=10)

    async def approve_the_send():
        while not any(e.type == EventType.CONFIRMATION_REQUESTED for e in seen):
            await asyncio.sleep(0.01)
        assert uia.activated == [(1, "Save")]  # Save ran without asking; Send is waiting
        request = next(e for e in seen if e.type == EventType.CONFIRMATION_REQUESTED)
        assert request.payload["tool_name"] == "click_ui_element"
        agent.confirmations.resolve(request.payload["id"], True)

    approver = asyncio.create_task(approve_the_send())
    reply = await agent.handle_user_message("save my note and send it")
    await approver

    assert reply == "Saved; sent."
    assert uia.activated == [(1, "Save"), (2, "Send")]
    requested = [e for e in seen if e.type == EventType.CONFIRMATION_REQUESTED]
    assert len(requested) == 1
    finished = [e.payload for e in seen if e.type == EventType.TOOL_FINISHED]
    assert [f["success"] for f in finished] == [True, True, True, True]
    tool_messages = [m.content for m in llm.calls[-1] if m.role == "tool"]
    assert '"draft text"' in tool_messages[1]  # what inspect_ui saw went back to the model


async def test_step_limit_is_configurable(tmp_path):
    registry = ToolRegistry()
    llm = _ScriptedLLM([tool_call_response("missing_tool", {})] * 3)
    agent = Agent(llm=llm, registry=registry, memory=SQLiteMemoryStore(tmp_path / "m.db"), events=EventBus(),
                  max_tool_iterations=3)
    reply = await agent.handle_user_message("loop")
    assert "step limit" in reply and len(llm.calls) == 3
