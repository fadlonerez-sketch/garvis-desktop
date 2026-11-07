"""Tests for the BrowserManager additions: installed-Chrome channel with a
logged fallback, launch args, the page hook (workspace placement), and the
structured extraction helpers used by web_research. Reuses the hermetic
fake Playwright harness from tests/test_browser_manager.py."""

from __future__ import annotations

import logging

import pytest

from app.browser.manager import BrowserManager
from app.core.exceptions import BrowserError
from tests.test_browser_manager import (  # noqa: F401 -- fixture import activates it here
    FakeChromium,
    _FakeHarness,
    _install_fake_playwright,
    fake_public_dns,
)


class ChannelChromium(FakeChromium):
    def __init__(self, browser, fail_channel: bool = False):
        super().__init__(browser)
        self.fail_channel = fail_channel
        self.calls: list[dict] = []

    async def launch(self, headless=True, channel=None, args=None):
        self.calls.append({"headless": headless, "channel": channel, "args": args})
        if channel and self.fail_channel:
            raise RuntimeError("Chromium distribution 'chrome' is not found\n<long playwright trace>")
        return self._browser


def _harness(monkeypatch, fail_channel=False) -> _FakeHarness:
    harness = _FakeHarness()
    harness.chromium = ChannelChromium(harness.browser, fail_channel=fail_channel)
    harness.instance.chromium = harness.chromium
    _install_fake_playwright(monkeypatch, harness)
    return harness


async def test_default_launch_is_unchanged(monkeypatch):
    harness = _harness(monkeypatch)
    await BrowserManager(headless=True).navigate("https://example.com")
    assert harness.chromium.calls == [{"headless": True, "channel": None, "args": None}]


async def test_channel_and_args_are_passed_when_headed(monkeypatch):
    harness = _harness(monkeypatch)
    manager = BrowserManager(headless=False, channel="chrome", launch_args=["--window-position=-32000,-32000"])
    await manager.navigate("https://example.com")
    assert harness.chromium.calls == [
        {"headless": False, "channel": "chrome", "args": ["--window-position=-32000,-32000"]}
    ]


async def test_launch_args_are_ignored_when_headless(monkeypatch):
    harness = _harness(monkeypatch)
    await BrowserManager(headless=True, launch_args=["--x"]).navigate("https://example.com")
    assert harness.chromium.calls[0]["args"] is None


async def test_missing_chrome_falls_back_to_bundled_chromium_and_logs(monkeypatch, caplog):
    harness = _harness(monkeypatch, fail_channel=True)
    logging.getLogger("garvis").propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="garvis.browser"):
            await BrowserManager(headless=False, channel="chrome").navigate("https://example.com")
    finally:
        logging.getLogger("garvis").propagate = False
    assert [c["channel"] for c in harness.chromium.calls] == ["chrome", None]
    assert any("channel_unavailable" in r.getMessage() for r in caplog.records)


async def test_page_hook_runs_for_each_new_page_when_headed(monkeypatch):
    harness = _harness(monkeypatch)
    seen = []

    async def hook(page):
        seen.append(page)

    manager = BrowserManager(headless=False, page_hook=hook)
    await manager.navigate("https://example.com")
    await manager.navigate("https://example.com/2")  # same page: no second hook call
    assert seen == [harness.page]
    await manager.close_page()
    await manager.navigate("https://example.com/3")  # fresh page in existing context
    assert len(seen) == 2


async def test_page_hook_is_skipped_when_headless(monkeypatch):
    _harness(monkeypatch)
    seen = []

    async def hook(page):
        seen.append(page)

    await BrowserManager(headless=True, page_hook=hook).navigate("https://example.com")
    assert seen == []


async def test_failing_page_hook_never_breaks_the_browser(monkeypatch):
    harness = _harness(monkeypatch)

    async def hook(page):
        raise RuntimeError("placement exploded")

    manager = BrowserManager(headless=False, page_hook=hook)
    assert await manager.navigate("https://example.com") == "https://example.com"
    assert harness.page.goto_calls == ["https://example.com"]


async def test_query_elements_passes_fixed_extractor(monkeypatch):
    harness = _harness(monkeypatch)
    captured = {}

    async def eval_on_selector_all(selector, script, arg):
        captured.update(selector=selector, script=script, arg=arg)
        return [{"title": "T", "href": "https://x.test"}]

    harness.page.eval_on_selector_all = eval_on_selector_all
    manager = BrowserManager(headless=True)
    await manager.navigate("https://example.com")
    rows = await manager.query_elements("div.result", {"title": ("a", None), "href": ("a", "href")}, limit=4)
    assert rows == [{"title": "T", "href": "https://x.test"}]
    assert captured["selector"] == "div.result"
    assert captured["arg"] == {"fields": {"title": ["a", None], "href": ["a", "href"]}, "limit": 4}


async def test_query_elements_requires_a_page():
    with pytest.raises(BrowserError, match="No page"):
        await BrowserManager().query_elements("div", {})


async def test_wait_for_settle_swallows_timeouts(monkeypatch):
    harness = _harness(monkeypatch)

    async def wait_for_load_state(state, timeout=None):
        raise TimeoutError("never idle")

    harness.page.wait_for_load_state = wait_for_load_state
    manager = BrowserManager(headless=True)
    await manager.navigate("https://example.com")
    await manager.wait_for_settle(timeout_ms=10)  # must not raise
