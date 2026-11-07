"""Which tool results the user should *see*, and what to show them.

A tool opts in explicitly with its `presentation` class attribute
(app/tools/base.py) -- e.g. open_website and open_application are
"prominent": the user asked for something to appear on their screen. Every
other tool defaults to "none" (background work never interrupts). A
protocol step may override the level for its own call.

`describe` builds the card content strictly from the tool's own result data
(the window title the desktop reported, the URL that was opened, the path
that was opened). Nothing is guessed: no title means the card says what was
opened and offers the Activity page instead of a window to show.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse


class Presentation(StrEnum):
    NONE = "none"
    SUBTLE = "subtle"
    """A small card, no animation."""
    PROMINENT = "prominent"
    """The attention animation near the cursor, then the card."""


_KIND_BY_TOOL = {"open_website": "website", "open_application": "application", "open_path": "path"}
_MAX_TEXT = 300


def _clip(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= _MAX_TEXT else text[: _MAX_TEXT - 1] + "…"


def level_for(tool_level: str, data: Any, override: str | None = None) -> Presentation:
    """The level for one successful call. Work placed on Garvis's own
    workspace desktop is background work: never presented, whatever the
    tool or step asked for, so it can't pull the user's attention there."""
    if isinstance(data, dict) and data.get("target_desktop") == "garvis_workspace":
        return Presentation.NONE
    try:
        return Presentation(override or tool_level or Presentation.NONE)
    except ValueError:
        return Presentation.NONE


def describe(
    tool_name: str, data: Any, level: Presentation, *, source: str | None = None, task_id: str | None = None
) -> dict:
    """The PRESENTATION event payload for a successful call."""
    data = data if isinstance(data, dict) else {}
    kind = _KIND_BY_TOOL.get(tool_name, "result")
    url = str(data.get("opened") or "") if kind == "website" else ""
    window = _clip(data.get("window"))
    app = _clip(data.get("browser") or data.get("app"))
    if kind == "website":
        host = urlparse(url).netloc
        title = window or host or "A web page"
        detail = url
    elif kind == "application":
        title = window or _clip(data.get("opened")).title() or "An application"
        detail = app or _clip(data.get("opened"))
    elif kind == "path":
        title = _clip(data.get("name")) or _clip(data.get("path")) or "A file"
        detail = _clip(data.get("path"))
    else:
        title = _clip(data.get("title")) or tool_name.replace("_", " ").capitalize()
        detail = _clip(data.get("detail"))
    handle = data.get("handle")
    return {
        "id": uuid.uuid4().hex,
        "level": level.value,
        "kind": kind,
        "tool": tool_name,
        "title": title,
        "detail": detail,
        "url": url if url.startswith(("http://", "https://")) else "",
        "handle": int(handle) if isinstance(handle, int) and handle > 0 else None,
        "verified": bool(data.get("visible_verified", data.get("window_verified", False))),
        "source": source or "desktop",
        "task_id": task_id,
    }


def notice(
    title: str, message: str, level: Presentation, *, source: str | None = None, task_id: str | None = None
) -> dict:
    """A PRESENTATION payload for an explicit notification (a protocol's
    "Notify me" step, a task summary) rather than a tool result."""
    return {
        "id": uuid.uuid4().hex,
        "level": level.value,
        "kind": "notice",
        "tool": "",
        "title": _clip(title) or "Garvis",
        "detail": _clip(message),
        "url": "",
        "handle": None,
        "verified": False,
        "source": source or "desktop",
        "task_id": task_id,
    }
