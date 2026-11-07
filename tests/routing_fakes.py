"""A desktop backend and a virtual-desktop backend that agree with each other,
for desktop-routing tests: a window's `on_current_desktop` follows the
virtual desktop it's on, windows on other desktops are only listed with
include_other_desktops, and a window on another desktop can't become the
foreground window (as on real Windows, where it stays cloaked)."""

from __future__ import annotations

from app.desktop.backend import WindowInfo, WindowState
from app.desktop.controller import DesktopController
from app.desktop.workspace import WorkspaceManager
from tests.desktop_fakes import GARVIS_PID, FakeDesktopBackend
from tests.test_workspace import FakeVirtualDesktops


class LinkedVirtualDesktops(FakeVirtualDesktops):
    def __init__(self, names=("",)) -> None:
        super().__init__(names)
        self.moves: list[tuple[int, str]] = []

    def move_window(self, handle, desktop_id):
        super().move_window(handle, desktop_id)
        self.moves.append((handle, desktop_id))


class MultiDesktopBackend(FakeDesktopBackend):
    def __init__(self, vd: LinkedVirtualDesktops, windows: list[WindowInfo] | None = None) -> None:
        super().__init__(windows)
        self.vd = vd
        self.launched: list[str] = []

    def _here(self, handle: int) -> bool:
        return self.vd.window_desktop_id(handle) == self.vd.current_id

    def _window(self, handle: int) -> WindowInfo | None:
        window = super()._window(handle)
        if window is None:
            return None
        return window.model_copy(update={"on_current_desktop": self._here(handle)})

    def list_windows(self, include_other_desktops: bool = False) -> list[WindowInfo]:
        return [
            w for w in super().list_windows(include_other_desktops) if include_other_desktops or w.on_current_desktop
        ]

    def focus_window(self, handle: int) -> bool:
        if not self._here(handle):
            self.calls.append(("focus", handle))
            return False
        return super().focus_window(handle)

    def set_window_rect(self, handle: int, rect) -> None:
        super().set_window_rect(handle, rect)
        for index, w in enumerate(self.windows):
            if w.handle == handle:
                self.windows[index] = w.model_copy(update={"rect": rect})

    def set_window_state(self, handle: int, state: WindowState) -> None:
        super().set_window_state(handle, state)
        if state == WindowState.RESTORE and self._here(handle):
            self.foreground = handle  # SW_RESTORE activates; SHOW_INACTIVE doesn't

    # ---------------------------------------------------- test helpers
    def open_window(
        self,
        handle: int,
        title: str,
        process: str,
        pid: int,
        *,
        minimized: bool = False,
        steal_focus: bool = False,
        desktop_id: str | None = None,
    ) -> None:
        self.windows.insert(
            0, WindowInfo(handle=handle, title=title, process_name=process, pid=pid, is_minimized=minimized)
        )
        self.vd.window_desktops[handle] = desktop_id or self.vd.current_id
        if steal_focus:
            self.foreground = handle


async def make_desktops():
    """(backend, vd, workspace, controller) with the user on desktop 1 and
    the Garvis workspace created as desktop 2."""
    vd = LinkedVirtualDesktops()
    backend = MultiDesktopBackend(vd)
    workspace = WorkspaceManager(backend_factory=lambda: vd, name="Garvis")
    await workspace.ensure()
    desktop = DesktopController(backend_factory=lambda: backend, screenshot_dir=None, own_pid=GARVIS_PID)
    desktop.attach_workspace(workspace)
    return backend, vd, workspace, desktop


def garvis_id(vd: LinkedVirtualDesktops) -> str:
    return next(d.id for d in vd.desktops if d.name == "Garvis")
