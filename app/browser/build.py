"""Composition helper for the browser automation layer, mirroring
app/voice/build.py: reads config from Settings so nothing about the
browser (headless mode, timeouts) is hardcoded elsewhere.
"""
from __future__ import annotations

from app.browser.manager import BrowserManager, PageHook
from app.core.config import Settings


def build_browser_manager(
    settings: Settings, page_hook: PageHook | None = None, launch_args: list[str] | None = None
) -> BrowserManager:
    return BrowserManager(
        headless=settings.browser_headless,
        nav_timeout_ms=settings.browser_nav_timeout_ms,
        action_timeout_ms=settings.browser_action_timeout_ms,
        channel=settings.browser_channel,
        launch_args=launch_args,
        page_hook=page_hook,
    )
