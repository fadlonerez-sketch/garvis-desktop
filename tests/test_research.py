"""Tests for browser research (app/browser/research.py) and the
web_research tool -- fake BrowserManager, no network."""
from __future__ import annotations

import base64

import pytest

from app.browser.research import SearchResult, WebResearcher, decode_result_url, relevant_excerpt
from app.core.exceptions import BrowserError
from app.desktop.workspace import WorkspaceManager
from app.tools.research import WebResearchInput, WebResearchTool
from tests.test_workspace import FakeVirtualDesktops


# ------------------------------------------------------- decode_result_url
def test_decode_duckduckgo_redirect():
    href = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.coindesk.com%2Fprice%2Fbitcoin&rut=abc"
    assert decode_result_url(href) == "https://www.coindesk.com/price/bitcoin"


def test_decode_bing_redirect():
    target = "https://www.example.org/a?b=1"
    encoded = "a1" + base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    assert decode_result_url(f"https://www.bing.com/ck/a?!&&p=x&u={encoded}&ntb=1") == target


@pytest.mark.parametrize(
    "href",
    ["", "javascript:void(0)", "https://duckduckgo.com/y.js?ad_provider=x", "https://www.bing.com/images",
     "https://www.bing.com/ck/a?u=a1%%%notbase64", "mailto:x@y.z"],
)
def test_decode_rejects_engine_internal_and_non_http(href):
    assert decode_result_url(href) is None


def test_decode_passes_plain_links_through():
    assert decode_result_url("https://news.example.com/story") == "https://news.example.com/story"


# ------------------------------------------------------- relevant_excerpt
def test_relevant_excerpt_prefers_lines_matching_the_query():
    text = "\n".join([
        "Accept all cookies to continue browsing this website today",
        "Navigation Home Markets News Research Learn About",
        "Bitcoin price today is $64,210.55 USD, up 2.1% in 24 hours",
        "Subscribe to our newsletter for weekly market updates and more",
    ])
    excerpt = relevant_excerpt(text, "current bitcoin price", budget=200)
    assert excerpt.startswith("Bitcoin price today is $64,210.55")
    assert "cookies" not in excerpt


def test_relevant_excerpt_keeps_document_order_and_budget():
    text = "\n".join(f"line {i} mentions widget pricing details here" for i in range(50))
    excerpt = relevant_excerpt(text, "widget pricing", budget=120)
    assert len(excerpt) <= 120
    assert excerpt.splitlines()[0].startswith("line 0")


def test_relevant_excerpt_falls_back_to_page_start():
    text = "A long enough first line about something else entirely.\nAnd another fairly long line of text."
    assert relevant_excerpt(text, "zzz", budget=40) == text.replace("\n", " ")[:40]


# ---------------------------------------------------------------- fakes
class FakeResearchBrowser:
    def __init__(self, rows=None, pages=None, fail_search=False):
        self.rows = rows if rows is not None else [
            {"title": "Ad", "href": "https://duckduckgo.com/y.js?ad=1", "snippet": "", "class": "result result--ad"},
            {"title": "CoinDesk BTC", "href": "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fbtc",
             "snippet": "BTC price", "class": "result"},
            {"title": "Dup", "href": "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fbtc", "snippet": "",
             "class": "result"},
            {"title": "Broken site", "href": "https://b.test/x", "snippet": "", "class": "result"},
            {"title": "Market", "href": "https://c.test/y", "snippet": "live", "class": "result"},
        ]
        self.pages = pages if pages is not None else {
            "https://a.test/btc": "Bitcoin price today is $64,000 according to our live index feed",
            "https://c.test/y": "The bitcoin price moved sharply this week as markets reacted",
        }
        self.fail_search = fail_search
        self.navigated: list[str] = []
        self._current = None

    async def navigate(self, url):
        self.navigated.append(url)
        if "duckduckgo.com/html" in url or "bing.com/search" in url:
            if self.fail_search:
                raise BrowserError("net::ERR_NAME_NOT_RESOLVED")
            self._current = "SERP"
            return url
        if url not in self.pages:
            raise BrowserError("Timeout 15000ms exceeded")
        self._current = url
        return url

    async def query_elements(self, selector, fields, limit=10):
        assert self._current == "SERP"
        return self.rows

    async def wait_for_settle(self, timeout_ms=3000):
        pass

    async def extract_text(self):
        return self.pages[self._current]

    async def page_title(self):
        return f"Title of {self._current}"


def _researcher(browser, workspace=None, fallback=None, **kw):
    return WebResearcher(browser, workspace=workspace, fallback_search=fallback, **kw)


# ------------------------------------------------------------- researcher
async def test_research_happy_path_skips_ads_dupes_and_broken_pages():
    browser = FakeResearchBrowser()
    report = await _researcher(browser, max_sources=2).research("bitcoin price")
    assert [r.url for r in report.results] == ["https://a.test/btc", "https://b.test/x", "https://c.test/y"]
    assert [s.url for s in report.sources] == ["https://a.test/btc", "https://c.test/y"]
    assert "$64,000" in report.sources[0].excerpt
    steps = {p.step: p for p in report.plan}
    assert steps["search"].status == "done"
    assert steps["read_sources"].status == "done" and "skipped 1" in steps["read_sources"].detail
    assert browser.navigated[0] == "https://html.duckduckgo.com/html/?q=bitcoin+price"


async def test_research_falls_back_to_api_search_observably():
    browser = FakeResearchBrowser(rows=[])
    calls = []

    async def fallback(query, n):
        calls.append(query)
        return [SearchResult(title="From API", url="https://a.test/btc")]

    report = await _researcher(browser, fallback=fallback).research("bitcoin price")
    search = next(p for p in report.plan if p.step == "search")
    assert search.status == "fallback" and "ddgs" in search.detail
    assert calls == ["bitcoin price"]
    assert report.sources[0].url == "https://a.test/btc"


async def test_research_search_navigation_error_also_falls_back():
    browser = FakeResearchBrowser(fail_search=True)

    async def fallback(query, n):
        return [SearchResult(title="x", url="https://c.test/y")]

    report = await _researcher(browser, fallback=fallback).research("bitcoin")
    assert next(p for p in report.plan if p.step == "search").status == "fallback"


async def test_research_total_failure_is_reported():
    browser = FakeResearchBrowser(rows=[])

    async def fallback(query, n):
        raise RuntimeError("ddgs down")

    report = await _researcher(browser, fallback=fallback).research("bitcoin")
    assert next(p for p in report.plan if p.step == "search").status == "failed"
    assert report.sources == [] and report.results == []


async def test_research_rejects_empty_query():
    with pytest.raises(BrowserError):
        await _researcher(FakeResearchBrowser()).research("   ")


def test_unknown_engine_is_a_config_error():
    with pytest.raises(ValueError, match="Unknown research search engine"):
        WebResearcher(FakeResearchBrowser(), engine="altavista")


async def test_research_uses_bing_profile():
    browser = FakeResearchBrowser(rows=[{"title": "x", "href": "https://a.test/btc", "snippet": "", "class": ""}])
    await _researcher(browser, engine="bing").research("btc")
    assert browser.navigated[0] == "https://www.bing.com/search?q=btc"


async def test_research_reports_workspace_state():
    vd = FakeVirtualDesktops()
    workspace = WorkspaceManager(backend_factory=lambda: vd, name="Garvis")
    try:
        report = await _researcher(FakeResearchBrowser(), workspace=workspace).research("bitcoin price")
        assert report.workspace == "garvis desktop"
        assert report.plan[0].status == "done" and "Garvis" in report.plan[0].detail
        assert vd.switches == []  # research never switches the user's view
    finally:
        workspace.shutdown()


async def test_research_refuses_to_run_on_the_users_desktop_when_workspace_is_unavailable():
    """Never silently change target: no workspace -> an error, nothing opened."""
    browser = FakeResearchBrowser()
    unavailable = WorkspaceManager(backend_factory=None)
    with pytest.raises(BrowserError, match="workspace desktop isn't available"):
        await _researcher(browser, workspace=unavailable).research("bitcoin price")
    assert browser.navigated == []
    result = await WebResearchTool(_researcher(browser, workspace=unavailable)).execute(
        WebResearchInput(query="bitcoin price"))
    assert not result.success and "wasn't started" in result.error


# ------------------------------------------------------------------ tool
async def test_web_research_tool_success_shape():
    tool = WebResearchTool(_researcher(FakeResearchBrowser()))
    result = await tool.execute(WebResearchInput(query="bitcoin price", max_sources=1))
    assert result.success
    assert set(result.data) == {"query", "engine", "plan", "results", "sources", "workspace", "researched_on"}
    assert len(result.data["sources"]) == 1


async def test_web_research_tool_reports_nothing_found():
    tool = WebResearchTool(_researcher(FakeResearchBrowser(rows=[])))
    result = await tool.execute(WebResearchInput(query="bitcoin price"))
    assert not result.success and "nothing usable" in result.error


def test_web_research_tool_is_low_risk_and_bounded():
    from pydantic import ValidationError

    from app.tools.base import RiskLevel

    assert WebResearchTool.risk_level == RiskLevel.LOW
    with pytest.raises(ValidationError):
        WebResearchInput(query="x", max_sources=9)
