"""Platform-neutral Desktop Control interface.

`DesktopBackend` is the OS boundary: synchronous, blocking, no validation
or policy -- that all lives in `DesktopController`. Keeping the backend this
thin means a future backend (a UI Automation-based one, a remote machine,
a fake for tests) only has to implement raw primitives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel


class MouseButton(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class WindowState(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    RESTORE = "restore"
    SHOW_INACTIVE = "show_inactive"
    """Restore/show without activating: never takes focus or switches desktop."""


class Rect(BaseModel):
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom


class WindowInfo(BaseModel):
    handle: int
    title: str
    process_name: str = ""
    pid: int = 0
    rect: Rect | None = None
    is_minimized: bool = False
    is_foreground: bool = False
    on_current_desktop: bool = True
    """False when the window lives on another virtual desktop (Windows
    "cloaks" it): it is not on the screen the user sees, screenshots of its
    rectangle show something else, and WinUI apps don't expose their
    content to UI Automation there."""


class InputContext(BaseModel):
    """Where real input would go: Garvis's Windows session and the desktop
    that currently receives input."""

    session_id: int = 0
    console_session_id: int = 0
    input_desktop: str | None = "Default"
    interactive: bool = True
    reason: str | None = None
    """Why input can't be delivered when `interactive` is False."""


class DesktopBackend(ABC):
    """Raw OS primitives. Coordinates are physical screen pixels in the
    virtual-screen coordinate space (the primary monitor's top-left is 0,0;
    other monitors may have negative coordinates)."""

    @abstractmethod
    def virtual_screen(self) -> Rect:
        """Bounding rectangle of all monitors."""

    @abstractmethod
    def primary_screen_size(self) -> tuple[int, int]: ...

    @abstractmethod
    def cursor_position(self) -> tuple[int, int]: ...

    @abstractmethod
    def move_mouse(self, x: int, y: int) -> None: ...

    @abstractmethod
    def click(self, button: MouseButton, count: int) -> None:
        """Click `button` `count` times at the current cursor position."""

    @abstractmethod
    def scroll(self, clicks: int, horizontal: bool = False) -> None:
        """Wheel `clicks` notches at the cursor. Positive = up (or right)."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type literal text (any Unicode), independent of keyboard layout."""

    @abstractmethod
    def press_keys(self, vks: list[int], repeat: int = 1) -> None:
        """Press `vks` down in order, release in reverse order -- a single
        key when len(vks) == 1, a chord/hotkey otherwise."""

    @abstractmethod
    def list_windows(self, include_other_desktops: bool = False) -> list[WindowInfo]:
        """Visible, titled top-level windows in z-order (topmost first) --
        only those on the CURRENT virtual desktop unless
        `include_other_desktops` (windows on other virtual desktops are
        "cloaked": they exist but aren't shown)."""

    @abstractmethod
    def foreground_window(self) -> WindowInfo | None: ...

    @abstractmethod
    def focus_window(self, handle: int) -> bool:
        """Restore (if minimized) and bring `handle` to the foreground.
        Returns whether it actually became the foreground window."""

    @abstractmethod
    def window_info(self, handle: int) -> WindowInfo | None:
        """Info for `handle`, or None if it no longer exists."""

    @abstractmethod
    def set_window_rect(self, handle: int, rect: Rect) -> None:
        """Move/resize a window without activating it."""

    def input_context(self) -> InputContext:
        """Is this process on the interactive desktop the user is looking
        at? Backends that can't tell report it as interactive."""
        return InputContext()

    @abstractmethod
    def set_window_state(self, handle: int, state: WindowState) -> None:
        """Minimize, maximize or restore a window."""

    @abstractmethod
    def close_window(self, handle: int) -> None:
        """Ask a window to close, exactly like its X button (WM_CLOSE): the
        application may still prompt to save unsaved work."""

    @abstractmethod
    def capture_screen(self, path: Path, region: Rect | None = None) -> tuple[int, int]:
        """Save a PNG of `region` (default: the whole virtual screen) to
        `path`. Returns (width, height)."""
