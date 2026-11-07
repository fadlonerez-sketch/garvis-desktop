"""Open a URL in the user's default browser (no automation/interaction).

This is always for the user: the page opens in THEIR browser, on the
desktop they're looking at. With a DesktopController attached the result
is verified -- the browser window that received the page is found (a new
window, or one whose title changed), brought to the user's desktop if the
browser put it on another virtual desktop, focused, and reported. When it
can't be confirmed, the result says so instead of claiming success.
Background browsing belongs in the controlled browser (browser_navigate
with target_desktop='garvis_workspace'), never in the user's own browser.
"""

from __future__ import annotations

import asyncio
import time
import webbrowser
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.core.exceptions import DesktopError
from app.core.logging import get_logger, log_extra
from app.desktop.routing import DesktopRouting, DesktopTarget
from app.tools.base import RiskLevel, Tool, ToolResult

logger = get_logger("tools.open_website")

_ALLOWED_SCHEMES = {"http", "https"}
BROWSER_PROCESSES = frozenset(
    "chrome.exe msedge.exe brave.exe firefox.exe opera.exe opera_gx.exe vivaldi.exe arc.exe librewolf.exe "
    "waterfox.exe iexplore.exe".split()
)
_VERIFY_SECONDS = 8.0
_POLL_SECONDS = 0.25


class OpenWebsiteInput(BaseModel):
    url: str = Field(description="Full URL to open, e.g. 'https://www.example.com'")
    target_desktop: Literal["user_desktop", "garvis_workspace"] | None = Field(
        default=None,
        description="Always the user's desktop (it's their own browser). For background "
        "browsing use browser_navigate with target_desktop='garvis_workspace' instead.",
    )


def _is_browser(process_name: str) -> bool:
    return process_name.lower() in BROWSER_PROCESSES


class OpenWebsiteTool(Tool):
    name = "open_website"
    description = (
        "Open a URL in the user's own default web browser, on the screen they're looking at "
        "(verified). Garvis can't read or interact with that page afterwards -- prefer "
        "browser_navigate (target_desktop='user_desktop') if the user might ask you to."
    )
    risk_level = RiskLevel.LOW
    presentation = "prominent"  # the user asked to see it
    input_model = OpenWebsiteInput

    def __init__(
        self, desktop=None, routing: DesktopRouting | None = None, verify_seconds: float = _VERIFY_SECONDS
    ) -> None:
        self._desktop = desktop
        self._routing = routing or DesktopRouting()
        self._verify_seconds = verify_seconds

    def egress_url(self, args: OpenWebsiteInput) -> str | None:
        return args.url

    async def _browser_titles(self) -> dict[int, str]:
        windows = await self._desktop.list_windows(include_other_desktops=True)
        return {w.handle: w.title for w in windows if _is_browser(w.process_name)}

    async def _find_opened(self, before: dict[int, str]):
        """The browser window that received the page: a new browser window,
        or one whose title changed (a new tab) -- the foreground one first."""
        deadline = time.monotonic() + self._verify_seconds
        while True:
            windows = await self._desktop.list_windows(include_other_desktops=True)
            changed = [
                w
                for w in windows
                if _is_browser(w.process_name) and (w.handle not in before or before[w.handle] != w.title)
            ]
            if changed:
                return next((w for w in changed if w.is_foreground), changed[0])
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(_POLL_SECONDS)

    async def execute(self, args: OpenWebsiteInput) -> ToolResult:
        parsed = urlparse(args.url)
        if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
            return ToolResult.fail(f"Refusing to open '{args.url}': only http(s) URLs are allowed.")
        resolution = self._routing.resolve("website", args.target_desktop, DesktopTarget.USER)
        if resolution.target == DesktopTarget.WORKSPACE:
            return ToolResult.fail(
                "open_website uses the user's own browser, which always opens on their screen -- nothing was "
                f"opened ({resolution.reason}). For background browsing use browser_navigate with "
                "target_desktop='garvis_workspace'."
            )
        before = None
        if self._desktop is not None:
            try:
                before = await self._browser_titles()
            except DesktopError:
                before = None
        opened = webbrowser.open(args.url)
        if not opened:
            return ToolResult.fail(f"Could not open '{args.url}' in a browser.")
        data: dict = {"opened": args.url, **resolution.as_result()}
        if before is None:
            data["visible_verified"] = False
            log_extra(logger, 20, "open_website.opened", url=args.url, verified=False)
            return ToolResult.ok(data)
        try:
            window = await self._find_opened(before)
            if window is None:
                data["visible_verified"] = False
                data["note"] = (
                    "The browser was asked to open the page, but no browser window showed it within "
                    f"{self._verify_seconds:.0f} s, so it can't be confirmed on the user's screen."
                )
            else:
                shown = await self._desktop.show_window(window.handle)
                data.update(
                    visible_verified=bool(shown["on_user_desktop"]),
                    window=shown["window"],
                    handle=window.handle,
                    browser=shown["app"],
                    moved_from_other_desktop=shown["moved_from_other_desktop"],
                )
        except DesktopError as exc:
            data["visible_verified"] = False
            data["note"] = f"The page was opened, but showing it on the user's screen couldn't be verified: {exc}"
        log_extra(
            logger,
            20,
            "open_website.opened",
            url=args.url,
            verified=data["visible_verified"],
            moved=data.get("moved_from_other_desktop"),
        )
        return ToolResult.ok(data)
