"""Desktop targeting: input reaches the intended application window on the
user's own desktop -- or nothing is sent and the blocker is reported.

Regression for the real failure: "open Notepad and type hello" typed into a
Notepad window on another virtual desktop (Garvis's workspace), because
open_application didn't identify/focus the new window and keyboard input
went to whatever had focus, while inspection only saw the current desktop.
"""
from __future__ import annotations

import pytest

from app.core.exceptions import DesktopError
from app.desktop.backend import InputContext, Rect, WindowInfo
from app.desktop.controller import DesktopController
from app.desktop.uia import UiInspector
from app.tools.applications import OpenApplicationTool
from app.tools.desktop import build_desktop_tools
from tests.desktop_fakes import GARVIS_PID, FakeDesktopBackend


class Backend(FakeDesktopBackend):
    """Fake desktop with virtual desktops and an input-context switch."""

    def __init__(self):
        super().__init__()
        self.windows.append(WindowInfo(handle=5, title="hello - Notepad", process_name="Notepad.exe", pid=500,
                                       rect=Rect(left=0, top=0, right=500, bottom=400), on_current_desktop=False))
        self.context = InputContext()
        self.moved: list[int] = []
        self.steal_focus_to: int | None = None

    def list_windows(self, include_other_desktops=False):
        windows = super().list_windows(include_other_desktops)
        return windows if include_other_desktops else [w for w in windows if w.on_current_desktop]

    def input_context(self):
        return self.context

    def focus_window(self, handle):
        ok = super().focus_window(handle)
        if ok and self.steal_focus_to is not None:
            self.foreground = self.steal_focus_to  # e.g. the user clicked elsewhere right after
        return ok

    def move_here(self, handle):
        self.moved.append(handle)
        for index, w in enumerate(self.windows):
            if w.handle == handle:
                self.windows[index] = w.model_copy(update={"on_current_desktop": True})


@pytest.fixture
def backend():
    return Backend()


@pytest.fixture
def controller(backend, tmp_path):
    async def mover(handle):
        backend.move_here(handle)

    return DesktopController(backend_factory=lambda: backend, screenshot_dir=tmp_path, own_pid=GARVIS_PID,
                             window_mover=mover)


def _typed(backend):
    return [c for c in backend.calls if c[0] == "type"]


# ------------------------------------------------------- virtual desktops
async def test_a_target_on_another_virtual_desktop_is_brought_here_before_typing(controller, backend):
    result = await controller.type_text("hello", window="hello - notepad")
    assert backend.moved == [5]
    assert _typed(backend) == [("type", "hello", 5)]
    assert result["handle"] == 5 and result["target_verified"] is True


async def test_without_a_way_to_move_windows_nothing_is_typed_and_the_blocker_is_reported(backend, tmp_path):
    controller = DesktopController(backend_factory=lambda: backend, screenshot_dir=tmp_path, own_pid=GARVIS_PID)
    with pytest.raises(DesktopError, match="another virtual desktop"):
        await controller.type_text("hello", window="hello - notepad")
    assert _typed(backend) == []


async def test_focus_on_a_window_the_user_cant_see_is_not_used_for_typing(controller, backend):
    backend.foreground = 5  # e.g. an app that opened itself on the Garvis workspace
    with pytest.raises(DesktopError, match="isn't on the desktop you're looking at"):
        await controller.type_text("hello")
    assert _typed(backend) == []


async def test_list_windows_shows_every_desktop_and_marks_the_others(controller):
    tool = {t.name: t for t in build_desktop_tools(controller)}["list_windows"]
    rows = (await tool.execute(tool.validate_arguments({}))).data
    notepad = next(r for r in rows if r["title"] == "hello - Notepad")
    assert notepad["desktop"] == "other_desktop"  # no virtual-desktop API attached: just "elsewhere"
    assert next(r for r in rows if r["title"] == "WhatsApp")["desktop"] == "user_desktop"


async def test_screenshots_of_a_window_on_another_desktop_are_refused(controller, backend):
    with pytest.raises(DesktopError, match="another virtual desktop"):
        await controller.capture_screen(window="hello - notepad")
    assert not any(c[0] == "capture" for c in backend.calls)


# -------------------------------------------------- Garvis's own window
async def test_garvis_own_window_is_never_the_target(controller, backend):
    with pytest.raises(DesktopError, match="Garvis's own window doesn't count"):
        await controller.type_text("hi", window="Garvis")
    with pytest.raises(DesktopError):
        await controller.focus_window("python")  # Garvis is python.exe
    assert _typed(backend) == []


async def test_inspection_and_input_resolve_the_same_window(controller, backend):
    inspected = await controller.resolve_window(None)
    typed = await controller.type_text("x")
    assert inspected.handle == typed["handle"] == 1
    backend.foreground = 3  # the user is now typing in Garvis
    inspected = await controller.resolve_window(None)
    typed = await controller.type_text("y")
    assert inspected.handle == typed["handle"] == 1  # both go to the user's app, never Garvis


# ---------------------------------------------------------- verification
async def test_nothing_is_typed_when_windows_refuses_the_focus(controller, backend):
    backend.refuse_focus.add(2)
    with pytest.raises(DesktopError, match="didn't bring 'WhatsApp' to the front.*nothing was sent"):
        await controller.type_text("hi", window="whatsapp")
    assert _typed(backend) == []


async def test_nothing_is_typed_if_focus_moves_before_the_keys_are_sent(controller, backend):
    backend.steal_focus_to = 1
    with pytest.raises(DesktopError, match="didn't bring|Focus moved"):
        await controller.type_text("secret plan", window="whatsapp")
    assert _typed(backend) == []


@pytest.mark.parametrize("context, fragment", [
    (InputContext(session_id=2, console_session_id=1, interactive=False,
                  reason="Garvis runs in Windows session 2, but the screen in front of you is session 1"), "session 2"),
    (InputContext(input_desktop=None, interactive=False, reason="Windows is showing the lock screen"), "lock screen"),
])
async def test_input_is_refused_when_the_desktop_isnt_reachable(controller, backend, context, fragment):
    backend.context = context
    for action in (lambda: controller.type_text("hi"), lambda: controller.click(10, 10),
                   lambda: controller.focus_window("whatsapp"), lambda: controller.press_key("enter")):
        with pytest.raises(DesktopError, match=fragment):
            await action()
    assert not any(c[0] in ("type", "click", "keys", "focus") for c in backend.calls)


# ------------------------------------------------------- open_application
async def test_open_application_identifies_focuses_and_reports_the_new_window(controller, backend):
    def launch():
        backend.windows.append(WindowInfo(handle=9, title="Untitled - Notepad", process_name="Notepad.exe", pid=900))

    result = await controller.launch_and_activate("notepad", launch, timeout_seconds=2)
    assert result["handle"] == 9 and result["new_window"] is True and result["focused"] is True
    assert backend.foreground == 9


async def test_a_new_window_on_another_desktop_is_brought_here(controller, backend):
    def launch():
        backend.windows.append(WindowInfo(handle=9, title="Untitled - Notepad", process_name="Notepad.exe",
                                          pid=900, on_current_desktop=False))

    result = await controller.launch_and_activate("notepad", launch, timeout_seconds=2)
    assert result["moved_from_other_desktop"] is True and backend.moved == [9]


async def test_single_instance_apps_reuse_their_existing_window(controller, backend):
    result = await controller.launch_and_activate("whatsapp", lambda: None, timeout_seconds=2, reuse_after_seconds=0.2)
    assert result["handle"] == 2 and result["new_window"] is False


async def test_no_window_means_failure_not_success(controller):
    with pytest.raises(DesktopError, match="no window of it appeared"):
        await controller.launch_and_activate("ghostapp", lambda: None, timeout_seconds=0.3, reuse_after_seconds=0.1)


async def test_open_application_tool_returns_the_verified_window(controller, backend, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("app.tools.applications.resolve_application", lambda name, allowlist: ("notepad.exe", False))

    def fake_popen(args, shell=False):
        backend.windows.append(WindowInfo(handle=9, title="Untitled - Notepad", process_name="Notepad.exe", pid=900))

    monkeypatch.setattr("app.tools.applications.subprocess.Popen", fake_popen)
    tool = OpenApplicationTool(allowlist={}, desktop=controller)
    result = await tool.execute(tool.validate_arguments({"application": "notepad"}))
    assert result.success and result.data["window_verified"] is True
    assert result.data["window"] == "Untitled - Notepad" and backend.foreground == 9


# ------------------------------------------------- typed-text verification
class _Uia(UiInspector):
    def __init__(self, texts):
        self.texts = list(texts)

    async def elements(self, handle, query=None, limit=120):
        return []

    async def element_at(self, x, y):
        return None

    async def activate(self, handle, name, control_type=None):
        raise DesktopError("unused")

    async def focused_text(self, handle):
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]


@pytest.mark.parametrize("texts, verified", [
    (["", "hello"], True),               # the window's field now shows the text
    (["hello", "hello"], False),         # unchanged: the keys didn't land there
    ([None, None], None),                # unreadable field (canvas, terminal)
])
async def test_keyboard_type_reports_whether_the_window_really_shows_the_text(controller, texts, verified):
    tool = {t.name: t for t in build_desktop_tools(controller, uia=_Uia(texts))}["keyboard_type"]
    result = await tool.execute(tool.validate_arguments({"text": "hello"}))
    assert result.success and result.data["text_verified"] is verified
    assert ("note" in result.data) is (verified is False)


async def test_inspect_ui_explains_content_hidden_on_another_desktop(controller):
    tool = {t.name: t for t in build_desktop_tools(controller, uia=_Uia([None]))}["inspect_ui"]
    result = await tool.execute(tool.validate_arguments({"window": "hello - notepad"}))
    assert result.data["desktop"] == "other_desktop" and "focus_window" in result.data["note"]
