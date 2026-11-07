"""Playwright-isolated browser automation.

This is the ONLY module that imports Playwright. Every browser tool
(app/tools/browser.py) talks to `BrowserManager`, never to Playwright
directly -- the same separation already used for the LLM providers (app/llm) and
faster-whisper/Piper (app/voice): the agent core and tool layer only ever
see plain Python data, never a third-party SDK. Playwright is imported
lazily inside `_ensure_started`, exactly like the lazy `faster_whisper`/
`winsound` imports elsewhere, so importing this module never requires
Playwright to be installed.

One browser is lazily launched on first use and kept alive across tool
calls (cheaper than relaunching per navigation), and torn down explicitly
via `shutdown()` -- callers are responsible for calling it on app exit so
no browser process is left running.

Two independent *surfaces* share that browser, each its own context and
therefore its own OS window: "user_desktop" (a page the user is meant to
see, placed on their visible desktop) and "garvis_workspace" (background
work, placed on Garvis's virtual desktop) -- see app/desktop/routing.py.
Every operation acts on the active surface; `use_surface()` selects it, and
it stays selected until changed, so a multi-step task keeps working on the
window it started in. Which desktop a surface's window lands on is decided
by the page hook (app/desktop/browser_placement.py); this module stays
unaware of desktops.

SSRF/scheme validation reuses `app.tools.webpage`'s validator (the same
logic already used by `fetch_webpage_text`) rather than duplicating it.
Redirects are covered too: navigation goes through Playwright's request
interception (`page.route`), which validates every *document* (top-level
navigation) request -- including ones caused by a server-side redirect --
before it's allowed to proceed. Non-document subresource requests
(images/scripts/styles a page loads on its own) are not re-validated here;
covering those as well would mean a much broader per-request network
policy, out of scope for this tool.

DNS rebinding: Chromium does its own DNS lookup, so the pre-navigation check
can't pin the address the way `fetch_webpage_text` does. As a practical
backstop, `navigate()` checks the address Chromium actually connected to for
the final document (`response.server_addr()`); if it isn't a public address
the page is cleared to about:blank and the navigation fails, so content from
a rebound private host is never handed back to the agent.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from app.core.exceptions import BrowserError, WindowPlacementError
from app.core.logging import get_logger, log_extra
from app.tools.webpage import UnsafeURLError, _validate_url_is_public

logger = get_logger("browser")

PageHook = Callable[..., Awaitable[None]]
"""Called with each newly created Playwright page and its surface name --
used to place the browser window on the right desktop
(app/desktop/browser_placement.py). A WindowPlacementError from it means
the page couldn't be put where it belongs: the page is closed and the
navigation fails. Any other error is logged and ignored."""

SURFACES = ("user_desktop", "garvis_workspace")
DEFAULT_SURFACE = "garvis_workspace"

# Fixed, Garvis-authored extractor (never LLM-supplied): for each element
# matching the container selector, read named fields. "href" is read as the
# resolved property so relative/protocol-relative links come back absolute.
_EXTRACT_JS = """(els, args) => els.slice(0, args.limit).map(el => {
  const out = {};
  for (const [key, spec] of Object.entries(args.fields)) {
    const [sel, attr] = spec;
    const node = sel ? el.querySelector(sel) : el;
    if (!node) { out[key] = null; continue; }
    out[key] = attr === "href" ? node.href : (attr ? node.getAttribute(attr) : node.innerText);
  }
  return out;
})"""

_DEFAULT_NAV_TIMEOUT_MS = 15_000
_DEFAULT_ACTION_TIMEOUT_MS = 10_000


async def _validate_navigation_url(url: str) -> None:
    """Runs the (blocking, DNS-resolving) SSRF validator off the event loop."""
    await asyncio.to_thread(_validate_url_is_public, url)


def _is_local_subresource(url: str) -> bool:
    """True for a subresource (script, XHR/fetch, image...) aimed at this PC or
    the local network by literal address or local name -- what a malicious page
    would use to drive local services (a gateway on 127.0.0.1, the router).
    No DNS lookup: this runs for every request, so names that only *resolve*
    to private addresses aren't caught here (documents still get the full check)."""
    host = (urlsplit(url).hostname or "").rstrip(".").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


async def _non_public_server_address(response) -> str | None:
    """The connected server's IP if it is NOT a public address, else None.
    Unknown (no response, served from cache, fake/older Playwright) -> None."""
    if response is None:
        return None
    server_addr = getattr(response, "server_addr", None)
    if server_addr is None:
        return None
    try:
        info = server_addr()
        if inspect.isawaitable(info):
            info = await info
    except Exception:
        return None
    if not isinstance(info, dict) or not info.get("ipAddress"):
        return None
    raw = str(info["ipAddress"]).strip("[]")
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return None
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return None if ip.is_global else str(ip)


class BrowserManager:
    """Owns a single lazily-launched Playwright browser/context/page."""

    def __init__(
        self,
        headless: bool = False,
        nav_timeout_ms: int = _DEFAULT_NAV_TIMEOUT_MS,
        action_timeout_ms: int = _DEFAULT_ACTION_TIMEOUT_MS,
        channel: str | None = None,
        launch_args: list[str] | None = None,
        page_hook: PageHook | None = None,
    ) -> None:
        self._headless = headless
        self._channel = channel or None
        """Playwright browser channel, e.g. "chrome" for the installed
        Google Chrome (no separate browser download needed). None = the
        bundled Chromium from `playwright install chromium`."""
        self._launch_args = list(launch_args or [])
        self._page_hook = page_hook
        self._nav_timeout_ms = nav_timeout_ms
        self._action_timeout_ms = action_timeout_ms
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        """Context/page of the ACTIVE surface (see `use_surface`)."""
        self._surfaces: dict[str, dict] = {name: {"context": None, "page": None} for name in SURFACES}
        self._active = DEFAULT_SURFACE
        self._lock = asyncio.Lock()
        self._blocked_reason: str | None = None

    @property
    def is_started(self) -> bool:
        return self._page is not None

    @property
    def active_surface(self) -> str:
        return self._active

    @property
    def page_hook(self) -> PageHook | None:
        return self._page_hook

    @property
    def places_windows(self) -> bool:
        """Whether each surface's window is put on its own desktop (a page
        hook is installed and there IS a window, i.e. not headless)."""
        return self._page_hook is not None and not self._headless

    def has_page(self, surface: str) -> bool:
        return self._surfaces.get(surface, {}).get("page") is not None

    async def use_surface(self, surface: str) -> None:
        """Make `surface` ("user_desktop" / "garvis_workspace") the one every
        following operation acts on. Its window is created on first use."""
        if surface not in self._surfaces:
            raise BrowserError(f"Unknown browser surface '{surface}'.")
        async with self._lock:
            self._save_active()
            self._active = surface
            self._context = self._surfaces[surface]["context"]
            self._page = self._surfaces[surface]["page"]

    def _save_active(self) -> None:
        self._surfaces[self._active] = {"context": self._context, "page": self._page}

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._page is not None:
                return
            if self._browser is not None:
                # The browser is alive: open (or reopen, after close_page())
                # this surface's page without relaunching the whole browser.
                if self._context is None:
                    self._context = await self._browser.new_context()
                page = await self._context.new_page()
                await page.route("**/*", self._on_route)
                self._page = page
                self._save_active()
                await self._run_page_hook(page)
                return
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise BrowserError(
                    "Playwright is not installed. Run: pip install playwright && playwright install chromium"
                ) from exc

            try:
                self._playwright = await async_playwright().start()
                self._browser = await self._launch_browser()
                self._context = await self._browser.new_context()
                page = await self._context.new_page()
                await page.route("**/*", self._on_route)
                self._page = page
                self._save_active()
            except Exception as exc:
                await self._teardown_locked()
                raise BrowserError(f"Could not start the browser: {exc}") from exc
            await self._run_page_hook(page)

    async def _launch_browser(self):
        kwargs: dict = {"headless": self._headless}
        if self._launch_args and not self._headless:
            kwargs["args"] = self._launch_args
        if self._channel:
            try:
                return await self._playwright.chromium.launch(channel=self._channel, **kwargs)
            except Exception as exc:
                first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
                log_extra(
                    logger,
                    30,
                    "browser.channel_unavailable_falling_back_to_bundled_chromium",
                    channel=self._channel,
                    error=first_line,
                )
        return await self._playwright.chromium.launch(**kwargs)

    async def _run_page_hook(self, page) -> None:
        if self._page_hook is None or self._headless:
            return
        try:
            if len(inspect.signature(self._page_hook).parameters) >= 2:
                await self._page_hook(page, self._active)
            else:
                await self._page_hook(page)
        except WindowPlacementError as exc:
            # The window can't be where it belongs: don't leave it anywhere.
            self._page = None
            self._save_active()
            try:
                await page.close()
            except Exception as close_exc:
                log_extra(logger, 10, "browser.misplaced_page_close_failed", error=str(close_exc))
            raise BrowserError(str(exc)) from exc
        except Exception as exc:
            log_extra(logger, 30, "browser.page_hook_failed", error=str(exc))

    async def _on_route(self, route) -> None:
        request = route.request
        if request.resource_type != "document":
            if _is_local_subresource(request.url):
                log_extra(logger, 30, "browser.local_subresource_blocked", resource_type=request.resource_type)
                await route.abort()
                return
            await route.continue_()
            return
        try:
            await _validate_navigation_url(request.url)
        except UnsafeURLError as exc:
            self._blocked_reason = str(exc)
            await route.abort()
            return
        await route.continue_()

    async def navigate(self, url: str) -> str:
        """Navigate the controlled browser to `url`. Returns the resulting page URL.

        Raises BrowserError if the URL (or a redirect it leads to) is
        unsafe, or if navigation otherwise fails/times out.
        """
        try:
            await _validate_navigation_url(url)  # fail fast, before even starting the browser
        except UnsafeURLError as exc:
            raise BrowserError(str(exc)) from exc
        await self._ensure_started()

        self._blocked_reason = None
        try:
            response = await self._page.goto(url, timeout=self._nav_timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            if self._blocked_reason:
                reason = self._blocked_reason
                self._blocked_reason = None
                raise BrowserError(reason) from exc
            raise BrowserError(f"Could not navigate to '{url}': {exc}") from exc

        if self._blocked_reason:
            reason = self._blocked_reason
            self._blocked_reason = None
            raise BrowserError(reason)

        bad_ip = await _non_public_server_address(response)
        if bad_ip is not None:
            log_extra(logger, 30, "browser.rebinding_blocked", ip=bad_ip)
            try:
                await self._page.goto("about:blank")
            except Exception as exc:
                log_extra(logger, 10, "browser.blank_after_block_failed", error=str(exc))
            raise BrowserError(f"Refusing to use '{url}': the browser connected to a non-public address ({bad_ip}).")

        return self._page.url

    def _require_page(self):
        if self._page is None:
            raise BrowserError("No page has been navigated to yet. Use browser_navigate first.")
        return self._page

    async def extract_text(self) -> str:
        page = self._require_page()
        try:
            return await page.inner_text("body")
        except Exception as exc:
            raise BrowserError(f"Could not read the page: {exc}") from exc

    async def query_elements(
        self, container_selector: str, fields: dict[str, tuple[str, str | None]], limit: int = 10
    ) -> list[dict]:
        """Structured extraction: for up to `limit` elements matching
        `container_selector`, read each field as (child selector, or "" for
        the element itself; attribute name, or None for its visible text).
        Used by the research runner; selectors come from Garvis's own
        search-engine profiles, never from the LLM."""
        page = self._require_page()
        payload = {"fields": {k: list(v) for k, v in fields.items()}, "limit": limit}
        try:
            return await page.eval_on_selector_all(container_selector, _EXTRACT_JS, payload)
        except Exception as exc:
            raise BrowserError(f"Could not read elements '{container_selector}': {exc}") from exc

    async def wait_for_settle(self, timeout_ms: int = 3000) -> None:
        """Give client-rendered pages a moment to fill in. Best effort --
        a page that never goes network-idle (ads, long-polling) is fine."""
        page = self._require_page()
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception as exc:
            log_extra(logger, 10, "browser.settle_wait_ended", error=str(exc))

    async def page_title(self) -> str:
        page = self._require_page()
        try:
            return await page.title()
        except Exception as exc:
            raise BrowserError(f"Could not read the page title: {exc}") from exc

    async def click(self, selector: str) -> None:
        page = self._require_page()
        try:
            await page.click(selector, timeout=self._action_timeout_ms)
        except Exception as exc:
            raise BrowserError(f"Could not click '{selector}': {exc}") from exc

    async def type_text(self, selector: str, text: str) -> None:
        page = self._require_page()
        try:
            await page.fill(selector, text, timeout=self._action_timeout_ms)
        except Exception as exc:
            raise BrowserError(f"Could not type into '{selector}': {exc}") from exc

    async def describe_element(self, selector: str) -> str:
        """The visible label of the element `selector` points to (text,
        value, aria-label, title, name, id) -- used to decide whether
        clicking it commits something. '' when there's no page/element."""
        if self._page is None:
            return ""
        try:
            label = await self._page.eval_on_selector(
                selector,
                "e => [e.innerText, e.value, e.getAttribute('aria-label'), e.title, e.name, e.id]"
                ".filter(Boolean).join(' | ').slice(0, 300)",
            )
        except Exception:
            return ""
        return str(label or "")

    async def describe_focused_form(self) -> str:
        """Labels of the buttons of the form that holds keyboard focus ('' if
        none) -- pressing Enter there submits that form."""
        if self._page is None:
            return ""
        try:
            labels = await self._page.evaluate(
                "() => { const f = document.activeElement && document.activeElement.form; if (!f) return '';"
                " return [...f.querySelectorAll('button, input[type=submit], input[type=button]')]"
                ".map(b => b.innerText || b.value || b.getAttribute('aria-label') || '').join(' | ').slice(0, 300); }"
            )
        except Exception:
            return ""
        return str(labels or "")

    async def press_key(self, key: str) -> None:
        page = self._require_page()
        try:
            await page.keyboard.press(key)
        except Exception as exc:
            raise BrowserError(f"Could not press '{key}': {exc}") from exc

    async def close_page(self) -> None:
        """Close the current controlled-browser page ("close this tab").
        The browser/context stay alive -- a subsequent navigate() launches
        a fresh page in the same context, same as first use. A no-op (not
        an error) when no page is open, matching shutdown()'s own
        idempotence -- "nothing to close" isn't a failure."""
        page = self._page
        if page is None:
            return
        self._page = None
        self._save_active()
        try:
            await page.close()
        except Exception as exc:
            raise BrowserError(f"Could not close the page: {exc}") from exc

    @property
    def current_url(self) -> str | None:
        return self._page.url if self._page is not None else None

    async def shutdown(self) -> None:
        async with self._lock:
            await self._teardown_locked()

    async def _teardown_locked(self) -> None:
        self._save_active()
        contexts = [surface["context"] for surface in self._surfaces.values() if surface["context"] is not None]
        for obj in (*contexts, self._browser):
            if obj is not None:
                try:
                    await obj.close()
                except Exception as exc:
                    log_extra(logger, 10, "browser.close_failed", error=str(exc))
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                log_extra(logger, 10, "browser.playwright_stop_failed", error=str(exc))
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._surfaces = {name: {"context": None, "page": None} for name in SURFACES}
