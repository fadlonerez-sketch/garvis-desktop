"""Deduplication of recurring notices (e.g. background errors).

The same warning repeated on every turn is noise: a notice with a given
key is reported once, then suppressed while it keeps recurring, and
reported again only after `quiet_seconds` without it (i.e. it went away and
came back) or when `clear()` says the situation changed. Suppressed repeats
are still logged by their callers.
"""
from __future__ import annotations

import time
from collections.abc import Callable


class RecurringNoticeFilter:
    def __init__(self, quiet_seconds: float = 900.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._quiet_seconds = quiet_seconds
        self._clock = clock
        self._last_seen: dict[str, float] = {}

    def should_report(self, key: str) -> bool:
        now = self._clock()
        last = self._last_seen.get(key)
        self._last_seen[key] = now
        return last is None or now - last >= self._quiet_seconds

    def clear(self, key: str | None = None) -> None:
        if key is None:
            self._last_seen.clear()
        else:
            self._last_seen.pop(key, None)
