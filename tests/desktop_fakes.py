"""Shared in-memory DesktopBackend for desktop-control tests (hermetic, any OS)."""

from __future__ import annotations

from pathlib import Path

from app.desktop.backend import DesktopBackend, MouseButton, Rect, WindowInfo, WindowState

GARVIS_PID = 4242


class FakeDesktopBackend(DesktopBackend):
    def __init__(self, windows: list[WindowInfo] | None = None) -> None:
        self.screen = Rect(left=0, top=0, right=1920, bottom=1080)
        self.cursor = (10, 10)
        self.windows = (
            windows
            if windows is not None
            else [
                WindowInfo(
                    handle=1,
                    title="Untitled - Notepad",
                    process_name="Notepad.exe",
                    pid=100,
                    rect=Rect(left=100, top=100, right=900, bottom=700),
                ),
                WindowInfo(
                    handle=2,
                    title="WhatsApp",
                    process_name="WhatsApp.Root.exe",
                    pid=200,
                    rect=Rect(left=0, top=0, right=800, bottom=600),
                ),
                WindowInfo(
                    handle=3,
                    title="Garvis",
                    process_name="python.exe",
                    pid=GARVIS_PID,
                    rect=Rect(left=0, top=0, right=400, bottom=400),
                ),
            ]
        )
        self.foreground = 1
        self.calls: list[tuple] = []
        self.refuse_focus: set[int] = set()

    def _window(self, handle: int) -> WindowInfo | None:
        for w in self.windows:
            if w.handle == handle:
                return w.model_copy(update={"is_foreground": handle == self.foreground})
        return None

    def virtual_screen(self) -> Rect:
        return self.screen

    def primary_screen_size(self) -> tuple[int, int]:
        return self.screen.width, self.screen.height

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def move_mouse(self, x: int, y: int) -> None:
        self.calls.append(("move", x, y))
        self.cursor = (x, y)

    def click(self, button: MouseButton, count: int) -> None:
        self.calls.append(("click", button, count))

    def scroll(self, clicks: int, horizontal: bool = False) -> None:
        self.calls.append(("scroll", clicks, horizontal))

    def type_text(self, text: str) -> None:
        self.calls.append(("type", text, self.foreground))

    def press_keys(self, vks: list[int], repeat: int = 1) -> None:
        self.calls.append(("keys", list(vks), repeat, self.foreground))

    def list_windows(self, include_other_desktops: bool = False) -> list[WindowInfo]:
        ordered = sorted(self.windows, key=lambda w: w.handle != self.foreground)
        return [self._window(w.handle) for w in ordered]

    def foreground_window(self) -> WindowInfo | None:
        return self._window(self.foreground)

    def focus_window(self, handle: int) -> bool:
        self.calls.append(("focus", handle))
        if handle in self.refuse_focus or self._window(handle) is None:
            return False
        self.foreground = handle
        return True

    def window_info(self, handle: int) -> WindowInfo | None:
        return self._window(handle)

    def set_window_rect(self, handle: int, rect: Rect) -> None:
        self.calls.append(("place", handle, rect))

    def set_window_state(self, handle: int, state: WindowState) -> None:
        self.calls.append(("state", handle, state))
        for index, w in enumerate(self.windows):
            if w.handle == handle:
                self.windows[index] = w.model_copy(update={"is_minimized": state == WindowState.MINIMIZE})
                if state == WindowState.MINIMIZE and self.foreground == handle:
                    self.foreground = next((o.handle for o in self.windows if o.handle != handle), 0)
                return
        raise OSError("That window no longer exists.")

    def close_window(self, handle: int) -> None:
        self.calls.append(("close", handle))
        if self._window(handle) is None:
            raise OSError("That window no longer exists.")
        self.windows = [w for w in self.windows if w.handle != handle]
        if self.foreground == handle:
            self.foreground = self.windows[0].handle if self.windows else 0

    def capture_screen(self, path: Path, region: Rect | None = None) -> tuple[int, int]:
        area = region or self.screen
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG fake")
        self.calls.append(("capture", str(path), region))
        return area.width, area.height
