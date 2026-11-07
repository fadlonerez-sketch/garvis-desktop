"""Garvis's presence: the one state a display shows -- today the desktop orb,
later a face on a tablet (Garvis Face) or a robot.

The contract a display client relies on:

  IDLE          nothing going on (online, waiting)
  LISTENING     Garvis is hearing the user (push-to-talk, a session waiting for speech)
  THINKING      transcribing, an agent turn, a session preparing its reply
  SPEAKING      Garvis's voice is actually playing
  NOTIFICATION  something wants the user's attention while otherwise idle
                (a confirmation to approve, a reminder ringing); `reason` says which
  ERROR         for a few seconds after an error

Each change gets a new `version`, so a client long-polls "anything after
version N?" (GET /api/v1/presence) and never misses or double-counts a change.

This module only *publishes* state. It's fed by the existing view-model
(app/ui/state.py:GarvisState, which derives the mode from real agent and
voice events) and by attention events; it has no idea how a face animates,
and nothing in the conversation logic depends on it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum


class PresenceState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    NOTIFICATION = "notification"
    ERROR = "error"


_FROM_MODE = {
    "idle": PresenceState.IDLE,
    "listening": PresenceState.LISTENING,
    "thinking": PresenceState.THINKING,
    "speaking": PresenceState.SPEAKING,
    "error": PresenceState.ERROR,
}


@dataclass(frozen=True)
class Presence:
    state: PresenceState
    version: int
    since: float
    """Epoch seconds when this state began."""
    reason: str = ""
    """For NOTIFICATION: "confirmation" or "reminder"."""

    def public(self) -> dict:
        data = asdict(self)
        data["state"] = self.state.value
        return data


class PresenceHub:
    """Thread-safe: the view-model publishes from the UI thread, the API waits
    on the assistant's asyncio loop."""

    def __init__(self, *, clock: Callable[[], float] = time.time, notification_seconds: float = 8.0) -> None:
        self._clock = clock
        self._notification_seconds = notification_seconds
        self._lock = threading.Lock()
        self._mode = PresenceState.IDLE
        self._notice: tuple[str, float] | None = None  # (reason, until)
        self._current = Presence(PresenceState.IDLE, 0, clock())
        self._loop: asyncio.AbstractEventLoop | None = None
        self._changed: asyncio.Event | None = None
        self._listeners: list[Callable[[Presence], None]] = []

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """The loop long-polls wait on (the assistant's loop)."""
        self._loop = loop

    def add_listener(self, listener: Callable[[Presence], None]) -> None:
        """An in-process display (called on the publishing thread)."""
        self._listeners.append(listener)

    # ------------------------------------------------------------- inputs
    def set_mode(self, mode: str) -> None:
        """From GarvisState.mode_changed ("idle", "listening", "thinking", "speaking", "error")."""
        state = _FROM_MODE.get(str(mode))
        if state is None:
            return
        with self._lock:
            self._mode = state
        self._refresh()

    def notify(self, reason: str, seconds: float | None = None) -> None:
        """Something wants the user's attention (shown while otherwise idle)."""
        duration = self._notification_seconds if seconds is None else seconds
        with self._lock:
            self._notice = (reason, self._clock() + duration)
        self._refresh()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.call_later, duration + 0.05, self._refresh)
            except RuntimeError:  # loop closed
                pass

    # -------------------------------------------------------------- reads
    def snapshot(self) -> Presence:
        self._refresh()
        with self._lock:
            return self._current

    async def wait(self, after: int, timeout: float) -> Presence:
        """The current presence as soon as its version is past `after` (or at the timeout)."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            current = self.snapshot()
            remaining = deadline - time.monotonic()
            if current.version > after or remaining <= 0:
                return current
            if self._changed is None:
                self._changed = asyncio.Event()
            event = self._changed
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

    # ----------------------------------------------------------- internal
    def _effective(self) -> tuple[PresenceState, str]:
        notice = self._notice
        if notice is not None and notice[1] <= self._clock():
            self._notice = notice = None
        if self._mode == PresenceState.IDLE and notice is not None:
            return PresenceState.NOTIFICATION, notice[0]
        return self._mode, ""

    def _refresh(self) -> None:
        with self._lock:
            state, reason = self._effective()
            if state == self._current.state and reason == self._current.reason:
                return
            self._current = Presence(state, self._current.version + 1, self._clock(), reason)
            current = self._current
        for listener in list(self._listeners):
            try:
                listener(current)
            except Exception:  # noqa: BLE001 -- a display must never break the assistant
                pass
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._wake)
            except RuntimeError:
                pass

    def _wake(self) -> None:
        event, self._changed = self._changed, asyncio.Event()
        if event is not None:
            event.set()
