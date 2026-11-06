"""Hermetic tests for BrowserManager, using a fake Playwright object graph
injected via sys.modules -- no real browser, no real network, and no
requirement that the `playwright` package even be installed (it mirrors
how faster_whisper/sounddevice/winsound are faked elsewhere in this suite).
"""
from __future__ import annotations

import socket
import sys
import types

import pytest

import app.tools.webpage as webpage_module
from app.core.exceptions import BrowserError
from app.browser.manager import BrowserManager


@pytest.fixture(autouse=True)
def fake_public_dns(monkeypatch):
    """Fakes DNS resolution so these tests never depend on real network
    access. An IP-literal host (e.g. "169.254.169.254") resolves to itself,
    exactly like real getaddrinfo would with no network involved -- so the
    SSRF-blocking tests still see the real address they asked for. Any
    other (name-based) host is resolved to a fixed public IP."""
    import ipaddress

    def fake_getaddrinfo(host, *args, **kwargs):
        try:
            ipaddress.ip_address(host)
            resolved = host
        except ValueError:
            resolved = "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolved, 0))]

    monkeypatch.setattr(webpage_module.socket, "getaddrinfo", fake_getaddrinfo)


# --------------------------------------------------------------- fakes
class FakeRequest:
    def __init__(self, url: str, resource_type: str = "document"):
        self.url = url
        self.resource_type = resource_type


class FakeRoute:
    def __init__(self, request: FakeRequest):
        self.request = request
        self.continued = False
        self.aborted = False

    async def continue_(self):
        self.continued = True

    async def abort(self, *args, **kwargs):
        self.aborted = True


class FakeKeyboard:
    def __init__(self):
        self.pressed: list[str] = []

    async def press(self, key: str):
        self.pressed.append(key)


class FakePage:
    def __init__(self, fail_goto: bool = False, fail_after_route: bool = False, fail_close: bool = False):
        self.url = "about:blank"
        self._route_handler = None
        self.goto_calls: list[str] = []
        self.clicked: list[str] = []
        self.filled: list[tuple[str, str]] = []
        self.keyboard = FakeKeyboard()
        self.fail_goto = fail_goto
        self.fail_after_route = fail_after_route
        self.fail_close = fail_close
        self.closed = False

    async def route(self, pattern, handler):
        self._route_handler = handler

    async def goto(self, url, timeout=None, wait_until=None):
        self.goto_calls.append(url)
        request = FakeRequest(url)
        route = FakeRoute(request)
        await self._route_handler(route)
        if route.aborted:
            raise RuntimeError(f"net::ERR_FAILED at {url}")
        if self.fail_goto:
            raise RuntimeError("simulated navigation failure")
        self.url = url

    async def inner_text(self, selector):
        return "  Some   page   body text  \n with whitespace  "

    async def click(self, selector, timeout=None):
        if self.fail_after_route and selector == "#broken":
            raise RuntimeError("element not found")
        self.clicked.append(selector)

    async def fill(self, selector, text, timeout=None):
        self.filled.append((selector, text))

    async def close(self):
        if self.fail_close:
            raise RuntimeError("simulated close failure")
        self.closed = True


class FakeContext:
    def __init__(self, page: FakePage):
        self._page = page
        self.closed = False

    async def new_page(self):
        return self._page

    async def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, context: FakeContext):
        self._context = context
        self.closed = False

    async def new_context(self):
        return self._context

    async def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, browser: FakeBrowser, fail_launch: bool = False):
        self._browser = browser
        self.fail_launch = fail_launch
        self.launch_kwargs = None

    async def launch(self, headless=True):
        self.launch_kwargs = {"headless": headless}
        if self.fail_launch:
            raise RuntimeError("simulated launch failure")
        return self._browser


class FakePlaywrightInstance:
    def __init__(self, chromium: FakeChromium):
        self.chromium = chromium
        self.stopped = False

    async def stop(self):
        self.stopped = True


class FakeAsyncPlaywrightCM:
    def __init__(self, instance: FakePlaywrightInstance):
        self._instance = instance

    async def start(self):
        return self._instance


class _FakeHarness:
    def __init__(self, fail_launch=False, fail_goto=False, fail_after_route=False, fail_close=False):
        self.page = FakePage(fail_goto=fail_goto, fail_after_route=fail_after_route, fail_close=fail_close)
        self.context = FakeContext(self.page)
        self.browser = FakeBrowser(self.context)
        self.chromium = FakeChromium(self.browser, fail_launch=fail_launch)
        self.instance = FakePlaywrightInstance(self.chromium)
        self.cm = FakeAsyncPlaywrightCM(self.instance)

    def async_api_module(self):
        module = types.ModuleType("playwright.async_api")
        module.async_playwright = lambda: self.cm
        return module


def _install_fake_playwright(monkeypatch: pytest.MonkeyPatch, harness: _FakeHarness) -> None:
    parent = types.ModuleType("playwright")
    monkeypatch.setitem(sys.modules, "playwright", parent)
    monkeypatch.setitem(sys.modules, "playwright.async_api", harness.async_api_module())


# --------------------------------------------------------------------- tests
@pytest.mark.asyncio
async def test_navigate_starts_browser_and_returns_final_url(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)

    manager = BrowserManager(headless=True)
    assert not manager.is_started

    result_url = await manager.navigate("https://example.com/page")

    assert manager.is_started
    assert result_url == "https://example.com/page"
    assert harness.page.goto_calls == ["https://example.com/page"]
    assert harness.chromium.launch_kwargs == {"headless": True}


@pytest.mark.asyncio
async def test_navigate_reuses_already_started_browser(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)

    manager = BrowserManager()
    await manager.navigate("https://example.com/one")
    await manager.navigate("https://example.com/two")

    # only one browser/page was ever launched
    assert harness.page.goto_calls == ["https://example.com/one", "https://example.com/two"]


@pytest.mark.asyncio
async def test_navigate_rejects_disallowed_scheme_before_launching_browser(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)

    manager = BrowserManager()
    with pytest.raises(BrowserError, match="only http"):
        await manager.navigate("file:///etc/passwd")

    assert not manager.is_started  # never even tried to launch


@pytest.mark.asyncio
async def test_navigate_rejects_javascript_scheme(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager()
    with pytest.raises(BrowserError, match="only http"):
        await manager.navigate("javascript:alert(1)")


@pytest.mark.asyncio
async def test_navigate_rejects_private_ip_literal(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)

    manager = BrowserManager()
    with pytest.raises(BrowserError, match="non-public"):
        await manager.navigate("http://169.254.169.254/latest/meta-data/")

    assert not manager.is_started


@pytest.mark.asyncio
async def test_navigate_blocks_redirect_to_private_target(monkeypatch):
    """The route handler must reject a private-network navigation target
    even when it's reached via an in-page redirect, not just the URL the
    caller originally asked for."""
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)

    # Make the fake page's goto() simulate a redirect landing on a private host:
    # the real route handler validates whatever URL goto() is ultimately called
    # with, so here we directly exercise the handler against a private URL to
    # confirm it aborts the route rather than allowing it through.
    manager = BrowserManager()
    await manager.navigate("https://example.com/start")  # starts the browser safely first

    async def goto_redirect(url, timeout=None, wait_until=None):
        request = FakeRequest("http://10.0.0.5/internal")
        route = FakeRoute(request)
        await harness.page._route_handler(route)
        assert route.aborted
        raise RuntimeError("net::ERR_FAILED")

    harness.page.goto = goto_redirect

    with pytest.raises(BrowserError, match="non-public"):
        await manager.navigate("https://example.com/redirector")


@pytest.mark.asyncio
async def test_navigate_wraps_generic_failures(monkeypatch):
    harness = _FakeHarness(fail_goto=True)
    _install_fake_playwright(monkeypatch, harness)

    manager = BrowserManager()
    with pytest.raises(BrowserError, match="Could not navigate"):
        await manager.navigate("https://example.com/")


@pytest.mark.asyncio
async def test_launch_failure_raises_browser_error_and_cleans_up(monkeypatch):
    harness = _FakeHarness(fail_launch=True)
    _install_fake_playwright(monkeypatch, harness)

    manager = BrowserManager()
    with pytest.raises(BrowserError, match="Could not start the browser"):
        await manager.navigate("https://example.com/")
    assert not manager.is_started


@pytest.mark.asyncio
async def test_missing_playwright_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright", None)
    manager = BrowserManager()
    with pytest.raises(BrowserError, match="Playwright is not installed"):
        await manager.navigate("https://example.com/")


@pytest.mark.asyncio
async def test_extract_text_before_navigation_raises():
    manager = BrowserManager()
    with pytest.raises(BrowserError, match="No page has been navigated"):
        await manager.extract_text()


@pytest.mark.asyncio
async def test_extract_text_returns_page_body(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager()
    await manager.navigate("https://example.com/")

    text = await manager.extract_text()
    assert "Some" in text and "page" in text


@pytest.mark.asyncio
async def test_click_and_type_and_press_key(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager()
    await manager.navigate("https://example.com/")

    await manager.click("button.submit")
    await manager.type_text("input[name=q]", "hello")
    await manager.press_key("Enter")

    assert harness.page.clicked == ["button.submit"]
    assert harness.page.filled == [("input[name=q]", "hello")]
    assert harness.page.keyboard.pressed == ["Enter"]


@pytest.mark.asyncio
async def test_click_before_navigation_raises():
    manager = BrowserManager()
    with pytest.raises(BrowserError, match="No page has been navigated"):
        await manager.click("button")


@pytest.mark.asyncio
async def test_click_failure_wrapped_as_browser_error(monkeypatch):
    harness = _FakeHarness(fail_after_route=True)
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager()
    await manager.navigate("https://example.com/")

    with pytest.raises(BrowserError, match="Could not click"):
        await manager.click("#broken")


@pytest.mark.asyncio
async def test_close_page_closes_the_page_and_allows_a_fresh_navigate(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager()
    await manager.navigate("https://example.com/")

    await manager.close_page()

    assert harness.page.closed
    assert not manager.is_started  # no current page until navigated again

    # the browser/context themselves are still alive -- navigating again
    # must not relaunch a whole new browser, just open a fresh page
    await manager.navigate("https://example.org/")
    assert manager.is_started
    assert harness.browser.closed is False
    assert harness.context.closed is False


@pytest.mark.asyncio
async def test_close_page_without_any_page_is_a_noop():
    manager = BrowserManager()
    await manager.close_page()  # must not raise
    assert not manager.is_started


@pytest.mark.asyncio
async def test_close_page_failure_wrapped_as_browser_error(monkeypatch):
    harness = _FakeHarness(fail_close=True)
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager()
    await manager.navigate("https://example.com/")

    with pytest.raises(BrowserError, match="Could not close the page"):
        await manager.close_page()


@pytest.mark.asyncio
async def test_shutdown_closes_everything_and_is_idempotent(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager()
    await manager.navigate("https://example.com/")

    await manager.shutdown()

    assert harness.context.closed
    assert harness.browser.closed
    assert harness.instance.stopped
    assert not manager.is_started

    await manager.shutdown()  # calling shutdown again must not raise


@pytest.mark.asyncio
async def test_shutdown_without_ever_starting_is_a_noop():
    manager = BrowserManager()
    await manager.shutdown()  # must not raise even though nothing was launched
    assert not manager.is_started


@pytest.mark.asyncio
async def test_headless_config_is_forwarded_to_launch(monkeypatch):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager(headless=False)
    await manager.navigate("https://example.com/")
    assert harness.chromium.launch_kwargs == {"headless": False}


# --- DNS rebinding backstop: check the address Chromium connected to ------


class _FakeResponse:
    def __init__(self, ip):
        self._ip = ip

    async def server_addr(self):
        return None if self._ip is None else {"ipAddress": self._ip, "port": 443}


def _goto_returning(harness, ip):
    original = harness.page.goto

    async def goto(url, timeout=None, wait_until=None):
        await original(url, timeout=timeout, wait_until=wait_until)
        return _FakeResponse(ip) if url != "about:blank" else None

    harness.page.goto = goto


@pytest.mark.asyncio
@pytest.mark.parametrize("ip", ["127.0.0.1", "192.168.1.1", "100.100.1.1", "::1", "::ffff:10.0.0.1"])
async def test_navigate_rejects_rebound_document_address(monkeypatch, ip):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager(headless=True)
    await manager._ensure_started()
    _goto_returning(harness, ip)

    with pytest.raises(BrowserError, match="non-public address"):
        await manager.navigate("https://example.com/page")
    assert harness.page.goto_calls[-1] == "about:blank"


@pytest.mark.asyncio
@pytest.mark.parametrize("ip", ["93.184.216.34", None])
async def test_navigate_accepts_public_or_unknown_document_address(monkeypatch, ip):
    harness = _FakeHarness()
    _install_fake_playwright(monkeypatch, harness)
    manager = BrowserManager(headless=True)
    await manager._ensure_started()
    _goto_returning(harness, ip)

    assert await manager.navigate("https://example.com/page") == "https://example.com/page"
