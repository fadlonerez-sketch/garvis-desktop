"""Source metadata for the desktop Sources view, taken from real tool results.

Only the retrieval tools' own result fields are used -- titles and URLs
exactly as the search engine / browser returned them; nothing is inferred
or invented. The agent attaches the list to its TOOL_FINISHED event
(app/agent/agent.py); it never changes what the model receives.

Each entry: {"tool", "kind", "title", "url", "publisher", "snippet"} where
`kind` is "read" (Garvis opened the page and read its text) or
"search_result" (listed by the search engine; only its snippet was seen).
"""
from __future__ import annotations

from urllib.parse import urlparse

_MAX_SNIPPET_CHARS = 320
_MAX_SOURCES = 25
_PAGE_READERS = frozenset({"fetch_webpage_text", "browser_read_page"})


def publisher_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _clip(text: object) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= _MAX_SNIPPET_CHARS else cleaned[: _MAX_SNIPPET_CHARS - 1].rstrip() + "…"


def _entry(tool: str, kind: str, title: object, url: object, snippet: object) -> dict | None:
    url = str(url or "").strip()
    if urlparse(url).scheme not in ("http", "https"):
        return None
    return {
        "tool": tool,
        "kind": kind,
        "title": " ".join(str(title or "").split()),
        "url": url,
        "publisher": publisher_of(url),
        "snippet": _clip(snippet),
    }


def extract_sources(tool_name: str, data: object) -> list[dict]:
    """Sources in a successful tool result, or [] for tools that don't
    retrieve information. Never raises: a malformed result just has none."""
    try:
        entries: list[dict | None] = []
        if tool_name == "web_search" and isinstance(data, list):
            entries = [
                _entry(tool_name, "search_result", r.get("title"), r.get("url"), r.get("snippet"))
                for r in data if isinstance(r, dict)
            ]
        elif tool_name == "web_research" and isinstance(data, dict):
            entries = [
                _entry(tool_name, "read", s.get("title"), s.get("url"), s.get("excerpt"))
                for s in data.get("sources") or [] if isinstance(s, dict)
            ] + [
                _entry(tool_name, "search_result", r.get("title"), r.get("url"), r.get("snippet"))
                for r in data.get("results") or [] if isinstance(r, dict)
            ]
        elif tool_name in _PAGE_READERS and isinstance(data, dict):
            entries = [_entry(tool_name, "read", data.get("title"), data.get("url"), data.get("text"))]
    except Exception:  # noqa: BLE001 -- display metadata must never break a turn
        return []
    sources: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        if entry is None or entry["url"] in seen:
            continue  # first wins: a page that was read outranks its search listing
        seen.add(entry["url"])
        sources.append(entry)
    return sources[:_MAX_SOURCES]
