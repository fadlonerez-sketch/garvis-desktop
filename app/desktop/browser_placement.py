"""Puts each controlled-browser window on the desktop it belongs on.

`BrowserWindowPlacer` is handed to BrowserManager as its `page_hook` and
called with every new page and its surface (app/desktop/routing.py):

- "garvis_workspace": the window is moved to the Garvis virtual desktop,
  positioned on-screen there, and focus goes back to whatever the user was
  using. Playwright drives pages over CDP, which needs neither focus nor
  visibility, so everything after that happens without touching the user's
  desktop. If the workspace can't be used, that is an error
  (WindowPlacementError): the page is closed rather than shown to the user.
- "user_desktop": the window is put on the desktop the user is looking at,
  on-screen, and activated -- then checked: on that desktop, not minimized,
  on-screen, in the foreground. `last_placement` records what was verified.

Window identification is exact, not heuristic: the page's title is set to a
unique marker (Chrome's window title is "<page title> - Google Chrome"),
the window carrying that marker is located, then the title is restored.

The browser is launched off-screen (OFFSCREEN_LAUNCH_ARGS) so the moment
between a window's creation and its placement shows nothing on the user's
monitor.
"""
from __future__ import annotations

import asyncio
import uuid

from app.core.exceptions import DesktopError, WindowPlacementError, WorkspaceError
from app.core.logging import get_logger, log_extra
from app.desktop.backend import Rect, WindowInfo
from app.desktop.controller import DesktopController
from app.desktop.workspace import WorkspaceManager

logger = get_logger("desktop.browser_placement")

OFFSCREEN_LAUNCH_ARGS = ["--window-position=-32000,-32000"]
_OFFSCREEN_THRESHOLD = -10000
_FIND_TIMEOUT_SECONDS = 5.0
_POLL_SECONDS = 0.1


class BrowserWindowPlacer:
    def __init__(
        self,
        workspace: WorkspaceManager,
        desktop: DesktopController,
        find_timeout_seconds: float = _FIND_TIMEOUT_SECONDS,
    ) -> None:
        self._workspace = workspace
        self._desktop = desktop
        self._find_timeout = find_timeout_seconds
        self.last_placement: dict | None = None

    async def __call__(self, page, target: str = "garvis_workspace") -> None:
        try:
            if target == "user_desktop":
                await self._place_for_user(page)
            else:
                await self._place_in_workspace(page)
        except WindowPlacementError:
            raise
        except Exception as exc:  # cosmetic failures never break the browser
            log_extra(logger, 30, "browser_placement.failed", target=target, error=str(exc))

    async def _find_window(self, marker: str) -> WindowInfo | None:
        deadline = asyncio.get_running_loop().time() + self._find_timeout
        while True:
            for window in await self._desktop.list_windows(include_other_desktops=True):
                if window.title.startswith(marker):
                    return window
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(_POLL_SECONDS)

    async def _locate(self, page) -> WindowInfo:
        marker = f"garvis-{uuid.uuid4().hex[:12]}"
        original_title = await page.title()
        await page.evaluate("t => { document.title = t }", marker)
        try:
            window = await self._find_window(marker)
        finally:
            await page.evaluate("t => { document.title = t }", original_title)
        if window is None:
            raise DesktopError("Could not find the browser window to place it.")
        return window

    async def _default_rect(self) -> Rect:
        info = await self._desktop.screen_info()
        width, height = info["primary_screen"]["width"], info["primary_screen"]["height"]
        w, h = min(1400, width - 160), min(1000, height - 120)
        return Rect(left=80, top=60, right=80 + w, bottom=60 + h)

    @staticmethod
    def _offscreen(window: WindowInfo) -> bool:
        return window.rect is not None and window.rect.left <= _OFFSCREEN_THRESHOLD

    async def _return_focus(self, browser_pid: int) -> None:
        """By the time a page exists, launching Chrome has already activated
        it. Once it's moved away, give focus to the topmost remaining window
        on the user's desktop -- i.e. whatever they were using before."""
        foreground = await self._desktop.foreground_window()
        if foreground is not None and foreground.pid != browser_pid:
            return
        for window in await self._desktop.list_windows():
            if window.pid != browser_pid and not window.is_minimized:
                await self._desktop.focus_handle(window.handle)
                return

    # ------------------------------------------------------- workspace
    async def _place_in_workspace(self, page) -> None:
        window = await self._locate(page)
        try:
            already_there = await self._workspace.is_window_on_garvis(window.handle)
            if not already_there:
                await self._workspace.move_window_to_garvis(window.handle)
        except WorkspaceError as exc:
            raise WindowPlacementError(
                f"Garvis's workspace desktop isn't available ({exc}), so background browsing was not "
                "started -- it would otherwise appear on your screen. Ask for it on your screen instead, "
                "or check the workspace setup (pyvda)."
            ) from exc
        if self._offscreen(window):
            await self._desktop.place_window(window.handle, await self._default_rect())
        if not already_there:
            await self._return_focus(window.pid)
        self.last_placement = {"target_desktop": "garvis_workspace", "moved": not already_there}
        log_extra(logger, 20, "browser_placement.placed", desktop="garvis", moved=not already_there,
                  was_offscreen=self._offscreen(window))

    # ------------------------------------------------------- user desktop
    async def _place_for_user(self, page) -> None:
        window = await self._locate(page)
        moved = False
        location = await self._workspace.window_location(window.handle)
        if location in ("garvis_workspace", "other"):
            try:
                await self._workspace.move_window_to_current(window.handle)
                moved = True
            except WorkspaceError as exc:
                raise WindowPlacementError(
                    f"The page opened on another virtual desktop and couldn't be moved to yours ({exc})."
                ) from exc
        if self._offscreen(window) or window.is_minimized:
            await self._desktop.place_window(window.handle, await self._default_rect())
        focused = await self._desktop.focus_handle(window.handle)
        await asyncio.sleep(0.1)
        after = await self._desktop.window_info(window.handle)
        if after is None:
            raise WindowPlacementError("The browser window closed before it could be shown.")
        location = await self._workspace.window_location(window.handle)
        visible = (
            location in ("user", "unknown") and after.on_current_desktop
            and not after.is_minimized and not self._offscreen(after)
        )
        if not visible:
            raise WindowPlacementError(
                f"The page's window couldn't be shown on your desktop (desktop: {location}, "
                f"minimized: {after.is_minimized}, off-screen: {self._offscreen(after)})."
            )
        self.last_placement = {"target_desktop": "user_desktop", "visible": True,
                               "focused": bool(focused and after.is_foreground), "moved_from_other_desktop": moved}
        log_extra(logger, 20, "browser_placement.placed", desktop="user", focused=self.last_placement["focused"],
                  moved=moved)
