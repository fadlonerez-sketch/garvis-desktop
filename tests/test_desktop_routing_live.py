"""REAL Windows check of desktop routing -- opt-in:

    set GARVIS_LIVE_DESKTOP_TESTS=1 && .venv\\Scripts\\python -m pytest tests/test_desktop_routing_live.py

Uses the production DesktopController (Win32 backend) and WorkspaceManager
(pyvda), and a throwaway window (tests/live_test_window.py) as the
"application". The window's virtual desktop, cloaking and the foreground
window are then read back independently -- straight from pyvda and Win32,
not through Garvis's own code.

Deliberately unintrusive: no keyboard/mouse input is injected, nothing is
activated, the user's view is never switched, and nothing is shown on the
user's screen (the window is minimized whenever it is on their desktop).
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("GARVIS_LIVE_DESKTOP_TESTS") != "1",
    reason="live Windows desktop test; set GARVIS_LIVE_DESKTOP_TESTS=1 to run",
)

_WINDOW_SCRIPT = Path(__file__).with_name("live_test_window.py")
_WM_CLOSE = 0x0010
_SW_SHOWMINNOACTIVE = 7


def _pythonw() -> str:
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate if candidate.exists() else sys.executable)


def _truth(handles: list[int]) -> dict:
    """Ground truth, read on a fresh thread straight from pyvda/Win32."""
    import pyvda

    user32 = ctypes.WinDLL("user32")
    dwmapi = ctypes.WinDLL("dwmapi")

    def cloaked(hwnd: int) -> bool:
        value = ctypes.c_int(0)
        dwmapi.DwmGetWindowAttribute(ctypes.c_void_p(hwnd), 14, ctypes.byref(value), ctypes.sizeof(value))
        return value.value != 0

    desktops = {}
    for handle in handles:
        try:
            desktops[handle] = str(pyvda.AppView(hwnd=handle).desktop_id)
        except Exception:
            desktops[handle] = None
    return {
        "current": str(pyvda.VirtualDesktop.current().id),
        "garvis": next((str(d.id) for d in pyvda.get_virtual_desktops() if d.name == "Garvis"), None),
        "foreground": int(user32.GetForegroundWindow() or 0),
        "desktops": desktops,
        "cloaked": {h: cloaked(h) for h in handles},
    }


async def truth(handles: list[int]) -> dict:
    return await asyncio.to_thread(_truth, handles)


@pytest.fixture
async def stack(tmp_path):
    from app.desktop.build import _pyvda_backend_factory
    from app.desktop.controller import DesktopController
    from app.desktop.win32 import Win32DesktopBackend
    from app.desktop.workspace import WorkspaceManager

    desktop = DesktopController(backend_factory=Win32DesktopBackend, screenshot_dir=tmp_path)
    workspace = WorkspaceManager(backend_factory=_pyvda_backend_factory, name="Garvis")
    desktop.attach_workspace(workspace)
    opened: list[int] = []
    yield desktop, workspace, opened
    user32 = ctypes.WinDLL("user32")
    for handle in opened:
        user32.PostMessageW(ctypes.c_void_p(handle), _WM_CLOSE, 0, 0)
    workspace.shutdown()


async def _launch_into_workspace(desktop, opened):
    from app.tools.applications import make_launcher

    title = f"Garvis routing test {uuid.uuid4().hex[:8]}"
    launch = make_launcher(_pythonw(), False, True, args=(str(_WINDOW_SCRIPT), title))
    result = await desktop.launch_in_workspace(title, launch, timeout_seconds=20, reuse_after_seconds=60)
    opened.append(result["handle"])
    return result


async def test_real_app_launched_into_workspace_lands_there_without_focus_theft(stack):
    desktop, workspace, opened = stack
    others = [w.handle for w in await desktop.list_windows(include_other_desktops=True)]
    before = await truth(others)

    result = await _launch_into_workspace(desktop, opened)
    handle = result["handle"]
    after = await truth([handle, *others])

    print(f"\nlaunch result: {result}\nbefore: current={before['current']} fg={before['foreground']}"
          f"\nafter:  current={after['current']} fg={after['foreground']} window_desktop={after['desktops'][handle]}"
          f" garvis={after['garvis']} cloaked={after['cloaked'][handle]}")
    assert after["garvis"] is not None
    assert after["desktops"][handle] == after["garvis"]  # really on the Garvis virtual desktop
    assert after["cloaked"][handle] is True  # not drawn on the user's screen
    assert after["current"] == before["current"]  # the user's view never switched
    assert after["foreground"] == before["foreground"]  # the user's focus is unchanged
    moved = [h for h in others if before["desktops"].get(h) and after["desktops"].get(h)
             and before["desktops"][h] != after["desktops"][h]]
    assert moved == []  # no unrelated window moved


async def test_real_window_brought_to_users_desktop_is_on_the_current_desktop(stack):
    desktop, workspace, opened = stack
    handle = (await _launch_into_workspace(desktop, opened))["handle"]
    # Minimized first so bringing it over shows nothing on top of the user's work.
    ctypes.WinDLL("user32").ShowWindow(ctypes.c_void_p(handle), _SW_SHOWMINNOACTIVE)
    before = await truth([handle])
    assert before["desktops"][handle] == before["garvis"]

    await workspace.move_window_to_current(handle)
    after = await truth([handle])
    location = await desktop.window_location(handle)

    print(f"\nbefore: window_desktop={before['desktops'][handle]} (garvis)\n"
          f"after:  window_desktop={after['desktops'][handle]} current={after['current']} "
          f"cloaked={after['cloaked'][handle]} garvis_location={location!r} fg={after['foreground']}")
    assert after["desktops"][handle] == after["current"]
    assert after["cloaked"][handle] is False
    assert location == "user"
    assert after["current"] == before["current"] and after["foreground"] == before["foreground"]
