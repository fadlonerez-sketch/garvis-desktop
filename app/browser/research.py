"""Browser research: search -> open top results -> collect relevant text.

`WebResearcher` is deliberately a small, fixed plan executor, not an
autonomous browsing agent: the LLM decides *that* something needs research
(by calling the web_research tool) and summarizes what comes back; the
steps in between are deterministic Garvis code driving the shared
BrowserManager -- always on its "garvis_workspace" surface, whose window
the BrowserWindowPlacer keeps on the Garvis virtual desktop. Research never
runs on the user's desktop: if the workspace can't be used, research fails
with a clear error instead (nothing appears on the user's screen). Showing
a result to the user is a separate, explicit step (app/tools/research.py).

Every run returns its plan with a status per step, so what happened
(including any fallback) is visible to the agent and in the logs:
  1. prepare_workspace  make sure the Garvis desktop exists (never switches view);
                        unavailable -> BrowserError, nothing is opened
  2. search             load the engine's results page, parse result entries
                        -> if that yields nothing (layout change, captcha),
                           fall back to the ddgs API search, marked as such
  3. read_sources       open results in order, keep the most query-relevant
                        text; a page that fails is skipped (next result used)
  4. collect            assemble results + sources for the agent

All page text is untrusted external data -- it only ever reaches the LLM as
a tool result (see app/agent/policies.py's security rule).
"""
from __future__ import annotations

import asyncio
import base64
import re
from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import parse_qs, quote_plus, urlparse

from pydantic import BaseModel

from app.browser.manager import BrowserManager
from app.core.exceptions import BrowserError, DesktopError
from app.core.logging import get_logger, log_extra

logger = get_logger("browser.research")

_MIN_LINE_CHARS = 25
_MAX_CANDIDATES = 8
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    "the a an and or of to in on for is are was what how who when where which with at by from "
    "current latest today now about me find search research summarize".split()
)


class SearchEngineProfile(BaseModel):
    name: str
    url_template: str
    result_selector: str
    title_selector: str
    snippet_selector: str
    skip_selector_class: str = ""
    """Results whose container has this class are ads -- skipped."""


SEARCH_ENGINES: dict[str, SearchEngineProfile] = {
    "duckduckgo": SearchEngineProfile(
        name="duckduckgo",
        url_template="https://html.duckduckgo.com/html/?q={query}",
        result_selector="div.result",
        title_selector="a.result__a",
        snippet_selector=".result__snippet",
        skip_selector_class="result--ad",
    ),
    "bing": SearchEngineProfile(
        name="bing",
        url_template="https://www.bing.com/search?q={query}",
        result_selector="li.b_algo",
        title_selector="h2 a",
        snippet_selector=".b_caption p",
    ),
}

_ENGINE_HOSTS = ("duckduckgo.com", "bing.com")


def decode_result_url(href: str) -> str | None:
    """Unwrap search-engine click-tracking redirects to the real target URL.
    Returns None for links that stay on the engine itself (ads, internal)."""
    if not href:
        return None
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    host = (parsed.hostname or "").lower()
    if host.endswith("duckduckgo.com") and "uddg" in query:
        href = query["uddg"][0]
    elif host.endswith("bing.com") and parsed.path.startswith("/ck/") and "u" in query:
        encoded = query["u"][0]
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        try:
            href = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
    final = urlparse(href)
    final_host = (final.hostname or "").lower()
    if final.scheme not in ("http", "https") or not final_host:
        return None
    if any(final_host == h or final_host.endswith("." + h) for h in _ENGINE_HOSTS):
        return None
    return href


def _query_terms(query: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD_RE.findall(query)) if len(w) > 1 and w not in _STOPWORDS}


def relevant_excerpt(text: str, query: str, budget: int) -> str:
    """Keep the lines of `text` that mention the most query terms (in their
    original order) until `budget` characters -- e.g. the line with the
    price, not the cookie banner. Falls back to the start of the page."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= _MIN_LINE_CHARS]
    terms = _query_terms(query)
    scored = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        score = sum(1 for term in terms if term in lowered)
        if score:
            scored.append((score, index, line))
    chosen: list[tuple[int, str]] = []
    used = 0
    for _score, index, line in sorted(scored, key=lambda item: (-item[0], item[1])):
        if used + len(line) > budget and chosen:
            continue
        chosen.append((index, line[:budget]))
        used += min(len(line), budget) + 1
        if used >= budget:
            break
    if not chosen:
        return " ".join(lines)[:budget]
    return "\n".join(line for _, line in sorted(chosen))[:budget]


class PlanStep(BaseModel):
    step: str
    status: Literal["done", "fallback", "skipped", "failed"]
    detail: str = ""


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class Source(BaseModel):
    title: str
    url: str
    excerpt: str


class ResearchReport(BaseModel):
    query: str
    engine: str
    plan: list[PlanStep]
    results: list[SearchResult]
    sources: list[Source]
    workspace: str
    """Where the browser ran: "garvis desktop" or "current desktop (...)"."""


FallbackSearch = Callable[[str, int], Awaitable[list[SearchResult]]]


async def ddgs_search(query: str, max_results: int) -> list[SearchResult]:
    """API fallback (the same ddgs package web_search already uses)."""
    from ddgs import DDGS

    def _run() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    raw = await asyncio.to_thread(_run)
    return [
        SearchResult(title=r.get("title", ""), url=r.get("href", r.get("link", "")), snippet=r.get("body", ""))
        for r in raw
        if r.get("href") or r.get("link")
    ]


class WebResearcher:
    def __init__(
        self,
        browser: BrowserManager,
        workspace=None,
        engine: str = "duckduckgo",
        max_sources: int = 3,
        source_chars: int = 1500,
        fallback_search: FallbackSearch | None = ddgs_search,
    ) -> None:
        if engine not in SEARCH_ENGINES:
            raise ValueError(f"Unknown research search engine {engine!r}; choose from {sorted(SEARCH_ENGINES)}.")
        self._browser = browser
        self._workspace = workspace
        self._engine = SEARCH_ENGINES[engine]
        self._max_sources = max_sources
        self._source_chars = source_chars
        self._fallback_search = fallback_search
        self._lock = asyncio.Lock()

    async def _prepare_workspace(self, plan: list[PlanStep]) -> str:
        if self._workspace is None:
            plan.append(PlanStep(step="prepare_workspace", status="skipped", detail="workspace disabled in settings"))
            return "current desktop (workspace disabled in settings)"
        try:
            desktop = await self._workspace.ensure()
        except DesktopError as exc:
            log_extra(logger, 30, "research.workspace_unavailable", error=str(exc))
            raise BrowserError(
                f"Garvis's workspace desktop isn't available ({exc}), so the research wasn't started -- it would "
                "otherwise run on the user's screen. Use web_search instead, or browse on the user's desktop "
                "if they want to watch."
            ) from exc
        plan.append(PlanStep(step="prepare_workspace", status="done", detail=f"desktop {desktop.number} '{desktop.name}'"))
        return "garvis desktop"

    async def _search_in_browser(self, query: str) -> list[SearchResult]:
        engine = self._engine
        await self._browser.navigate(engine.url_template.format(query=quote_plus(query)))
        rows = await self._browser.query_elements(
            engine.result_selector,
            {
                "title": (engine.title_selector, None),
                "href": (engine.title_selector, "href"),
                "snippet": (engine.snippet_selector, None),
                "class": ("", "class"),
            },
            limit=_MAX_CANDIDATES * 2,
        )
        results: list[SearchResult] = []
        seen: set[str] = set()
        for row in rows:
            if engine.skip_selector_class and engine.skip_selector_class in (row.get("class") or ""):
                continue
            url = decode_result_url(row.get("href") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                SearchResult(
                    title=" ".join((row.get("title") or "").split()),
                    url=url,
                    snippet=" ".join((row.get("snippet") or "").split()),
                )
            )
        return results[:_MAX_CANDIDATES]

    async def _search(self, query: str, plan: list[PlanStep]) -> list[SearchResult]:
        error = ""
        try:
            results = await self._search_in_browser(query)
        except BrowserError as exc:
            results, error = [], str(exc)
        if results:
            plan.append(PlanStep(step="search", status="done", detail=f"{len(results)} results from {self._engine.name}"))
            return results

        reason = error or "no results could be parsed from the results page (layout change or bot check?)"
        log_extra(logger, 30, "research.search_fallback", engine=self._engine.name, reason=reason)
        if self._fallback_search is None:
            plan.append(PlanStep(step="search", status="failed", detail=reason))
            return []
        try:
            results = (await self._fallback_search(query, _MAX_CANDIDATES))[:_MAX_CANDIDATES]
        except Exception as exc:
            plan.append(PlanStep(step="search", status="failed", detail=f"{reason}; API fallback failed: {exc}"))
            return []
        plan.append(
            PlanStep(step="search", status="fallback", detail=f"{reason}; used ddgs API search ({len(results)} results)")
        )
        return results

    async def _read_sources(self, query: str, results: list[SearchResult], plan: list[PlanStep], wanted: int) -> list[Source]:
        sources: list[Source] = []
        failures: list[str] = []
        for result in results:
            if len(sources) >= wanted:
                break
            try:
                final_url = await self._browser.navigate(result.url)
                await self._browser.wait_for_settle()
                text = await self._browser.extract_text()
                title = await self._browser.page_title() or result.title
            except BrowserError as exc:
                failures.append(f"{result.url}: {str(exc)[:120]}")
                continue
            excerpt = relevant_excerpt(text, query, self._source_chars)
            if not excerpt:
                failures.append(f"{result.url}: no readable text")
                continue
            sources.append(Source(title=title, url=final_url, excerpt=excerpt))
        detail = f"read {len(sources)} of {len(results)} candidates"
        if failures:
            detail += f"; skipped {len(failures)}: " + " | ".join(failures)
        status = "done" if sources else "failed"
        plan.append(PlanStep(step="read_sources", status=status, detail=detail))
        return sources

    async def research(self, query: str, max_sources: int | None = None) -> ResearchReport:
        query = query.strip()
        if not query:
            raise BrowserError("Research query is empty.")
        wanted = max_sources or self._max_sources
        async with self._lock:  # one research task drives the shared browser at a time
            plan: list[PlanStep] = []
            workspace = await self._prepare_workspace(plan)
            # Pin the workspace surface for the whole run, then hand back
            # whichever surface the agent was using (e.g. a page it is
            # showing the user) -- research never navigates that one.
            use_surface = getattr(self._browser, "use_surface", None)
            previous = getattr(self._browser, "active_surface", None)
            if use_surface is not None:
                await use_surface("garvis_workspace")
            try:
                results = await self._search(query, plan)
                sources = await self._read_sources(query, results, plan, wanted) if results else []
            finally:
                if use_surface is not None and previous is not None:
                    await use_surface(previous)
            plan.append(PlanStep(step="collect", status="done" if sources or results else "failed",
                                 detail=f"{len(results)} results, {len(sources)} sources"))
        log_extra(
            logger, 20, "research.completed", engine=self._engine.name, num_results=len(results),
            num_sources=len(sources), workspace=workspace, steps=[f"{p.step}:{p.status}" for p in plan],
        )
        return ResearchReport(
            query=query, engine=self._engine.name, plan=plan, results=results[:_MAX_CANDIDATES],
            sources=sources, workspace=workspace,
        )
