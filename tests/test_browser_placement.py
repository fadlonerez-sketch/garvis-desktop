"""Tests for BrowserWindowPlacer (app/desktop/browser_placement.py): the
page hook that moves Garvis's browser window onto the Garvis desktop."""

from __future__ import annotations

import pytest

from app.core.exceptions import WindowPlacementError
from app.desktop.backend import Rect, WindowInfo
from app.desktop.browser_placement import BrowserWindowPlacer
from app.desktop.controller import DesktopController
from app.desktop.workspace import WorkspaceManager
from tests.desktop_fakes import GARVIS_PID, FakeDesktopBackend
from tests.routing_fakes import garvis_id, make_desktops
from tests.test_workspace import FakeVirtualDesktops

_BROWSER = 50
_OFFSCREEN = Rect(left=-32000, top=-32000, right=-30700, bottom=-31200)


class FakeBrowserPage:
    """A Playwright page whose document.title drives a fake OS window title."""

    def __init__(self, backend: FakeDesktopBackend, title: str = "about:blank") -> None:
        self._backend = backend
        self._title = title
        self.titles: list[str] = []

    async def title(self):
        return self._title

    async def evaluate(self, script, title):
        self._title = title
        self.titles.append(title)
        for window in self._backend.windows:
            if window.handle == _BROWSER:
                window.title = f"{title} - Google Chrome"


def _setup(rect=_OFFSCREEN, workspace_factory=None):
    backend = FakeDesktopBackend()
    backend.windows.append(
        WindowInfo(handle=_BROWSER, title="about:blank - Google Chrome", process_name="chrome.exe", pid=900, rect=rect)
    )
    backend.foreground = _BROWSER  # launching Chrome activated it
    desktop = DesktopController(backend_factory=lambda: backend, screenshot_dir=None, own_pid=GARVIS_PID)
    vd = FakeVirtualDesktops()
    workspace = WorkspaceManager(backend_factory=workspace_factory or (lambda: vd), name="Garvis")
    placer = BrowserWindowPlacer(workspace, desktop, find_timeout_seconds=0.3)
    return backend, vd, workspace, placer


class _MovesWindowAway(FakeVirtualDesktops):
    """Moving a window to another virtual desktop hides it from the
    current desktop's window list, exactly as cloaking does for real."""

    def __init__(self, backend: FakeDesktopBackend) -> None:
        super().__init__()
        self._backend = backend

    def move_window(self, handle, desktop_id):
        super().move_window(handle, desktop_id)
        moved = next(w for w in self._backend.windows if w.handle == handle)
        self._backend.windows.remove(moved)
        self._backend.windows.append(moved)  # bottom of z-order ...
        self._backend.hidden = {handle}  # ... and not listed on this desktop


async def test_new_browser_window_is_moved_placed_and_focus_returned():
    backend, _, _, _ = _setup()
    # Z-order after Chrome's launch: Chrome (focused), then the user's Notepad.
    backend.windows.insert(0, backend.windows.pop())
    vd = _MovesWindowAway(backend)
    original_list = backend.list_windows
    backend.list_windows = lambda include_other_desktops=False: [
        w
        for w in original_list(include_other_desktops)
        if include_other_desktops or w.handle not in getattr(backend, "hidden", set())
    ]
    workspace = WorkspaceManager(backend_factory=lambda: vd, name="Garvis")
    desktop = DesktopController(backend_factory=lambda: backend, screenshot_dir=None, own_pid=GARVIS_PID)
    placer = BrowserWindowPlacer(workspace, desktop, find_timeout_seconds=0.3)
    page = FakeBrowserPage(backend)
    try:
        await placer(page)
        garvis = await workspace.ensure()
        assert vd.window_desktops[_BROWSER] == garvis.id
        placed = [c for c in backend.calls if c[0] == "place"]
        assert placed and placed[0][1] == _BROWSER and placed[0][2].left == 80
        assert backend.foreground != _BROWSER  # focus handed back to the user
        assert backend.foreground == 1
        assert page.titles[-1] == "about:blank"  # title restored after identification
        assert page.titles[0].startswith("garvis-")
    finally:
        workspace.shutdown()


async def test_focus_is_not_touched_if_user_already_regained_it():
    backend, vd, workspace, placer = _setup()
    backend.foreground = 2  # user clicked into WhatsApp before placement ran
    try:
        await placer(FakeBrowserPage(backend))
        assert not [c for c in backend.calls if c[0] == "focus"]
        assert backend.foreground == 2
    finally:
        workspace.shutdown()


async def test_window_already_on_garvis_is_left_alone():
    backend, vd, workspace, placer = _setup(rect=Rect(left=80, top=60, right=1480, bottom=1060))
    try:
        garvis = await workspace.ensure()
        vd.window_desktops[_BROWSER] = garvis.id
        await placer(FakeBrowserPage(backend))
        assert not [c for c in backend.calls if c[0] in ("place", "focus")]
    finally:
        workspace.shutdown()


async def test_workspace_unavailable_is_an_error_not_a_visible_window():
    """Never silently change target: background browsing that can't go to
    the workspace must not end up on the user's screen instead."""

    def unsupported():
        raise NotImplementedError("no virtual desktop API on this build")

    backend, vd, workspace, placer = _setup(workspace_factory=unsupported)
    try:
        with pytest.raises(WindowPlacementError, match="workspace desktop isn't available"):
            await placer(FakeBrowserPage(backend))
        assert not [c for c in backend.calls if c[0] == "place"], "must stay off-screen, not be shown"
        assert vd.window_desktops == {}
    finally:
        workspace.shutdown()


async def test_missing_window_is_logged_not_raised():
    backend, vd, workspace, placer = _setup()
    backend.windows = [w for w in backend.windows if w.handle != _BROWSER]
    try:
        await placer(FakeBrowserPage(backend))  # must not raise
        assert vd.window_desktops == {}
    finally:
        workspace.shutdown()


# ------------------------------------------------------------ user desktop
async def _user_setup(start_on_workspace=False, rect=_OFFSCREEN):
    backend, vd, workspace, desktop = await make_desktops()
    backend.open_window(
        _BROWSER,
        "about:blank - Google Chrome",
        "chrome.exe",
        900,
        desktop_id=garvis_id(vd) if start_on_workspace else None,
    )
    backend.windows[0] = backend.windows[0].model_copy(update={"rect": rect})
    placer = BrowserWindowPlacer(workspace, desktop, find_timeout_seconds=0.3)
    return backend, vd, workspace, placer


async def test_user_target_window_is_shown_focused_and_verified_on_users_desktop():
    backend, vd, workspace, placer = await _user_setup()
    try:
        await placer(FakeBrowserPage(backend), "user_desktop")
        assert vd.window_desktops[_BROWSER] == vd.current_id  # never sent to the workspace
        assert vd.moves == []
        assert [c for c in backend.calls if c[0] == "place"][0][1] == _BROWSER  # brought on-screen
        assert backend.foreground == _BROWSER
        assert placer.last_placement == {
            "target_desktop": "user_desktop",
            "visible": True,
            "focused": True,
            "moved_from_other_desktop": False,
        }
    finally:
        workspace.shutdown()


async def test_user_target_brings_window_over_from_the_workspace():
    backend, vd, workspace, placer = await _user_setup(
        start_on_workspace=True, rect=Rect(left=80, top=60, right=1480, bottom=1060)
    )
    try:
        await placer(FakeBrowserPage(backend), "user_desktop")
        assert vd.window_desktops[_BROWSER] == vd.current_id
        assert vd.current_id != garvis_id(vd)  # the user's view was never switched
        assert vd.switches == []
        assert placer.last_placement["moved_from_other_desktop"] is True
        assert backend.foreground == _BROWSER
    finally:
        workspace.shutdown()


async def test_user_target_that_cant_be_made_visible_is_an_error():
    backend, vd, workspace, placer = await _user_setup(start_on_workspace=True)

    def refuse(handle, desktop_id):
        raise RuntimeError("COM error")

    vd.move_window = refuse
    try:
        with pytest.raises(WindowPlacementError, match="couldn't be moved to yours"):
            await placer(FakeBrowserPage(backend), "user_desktop")
    finally:
        workspace.shutdown()


async def test_workspace_target_moves_only_the_browser_window():
    backend, vd, workspace, placer = await _user_setup()
    backend.foreground = 1  # the user is in Notepad
    try:
        await placer(FakeBrowserPage(backend), "garvis_workspace")
        assert vd.moves == [(_BROWSER, garvis_id(vd))]  # no unrelated window moved
        assert backend.foreground == 1  # focus untouched
        assert placer.last_placement == {"target_desktop": "garvis_workspace", "moved": True}
    finally:
        workspace.shutdown()
