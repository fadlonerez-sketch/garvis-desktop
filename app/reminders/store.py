"""The one store of reminders and scheduled calls.

Reminders live on the PC (data/reminders/reminders.json) so the assistant,
the desktop and every paired phone see the same list; the phone keeps a
synced copy and arms its own alarms from it, so a reminder still rings when
the PC is off or the phone is offline.

- `version` grows with every change: a phone re-syncs when it differs.
- A client `request_id` makes creation idempotent (a retried request after
  a dropped connection returns the original reminder, never a duplicate).
- `fired(id, occurrence)` is idempotent too: the phone reports each
  occurrence it rang; only the report for the current occurrence advances a
  recurring reminder (or completes a one-off), repeats are ignored.
- Nothing is ever deleted by an edit: cancel/complete keep the record so the
  phone can show it under "Completed" / "Cancelled".
- `spoken` is what Garvis says when it rings (app/reminders/phrasing.py): set
  by the rules whenever the text or repeat rule changes, later replaced by a
  checked model-written sentence (`set_spoken`). Phones get it with the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger, log_extra
from app.reminders.schedule import (
    REPEATS,
    TZDATA_MISSING,
    ScheduleError,
    describe,
    is_valid_zone,
    local_fixed_zone_name,
    next_occurrence,
    normalize_days,
    parse_hhmm,
    zone_database_available,
)

logger = get_logger("reminders.store")

SCHEDULED, COMPLETED, CANCELLED = "scheduled", "completed", "cancelled"
KINDS = ("notify", "call")
MAX_TEXT = 500
MAX_ACTIVE = 200
KEEP_FINISHED = 200
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class ReminderError(Exception):
    """A reminder can't be created or changed. The message is user-facing."""


@dataclass
class Reminder:
    id: str
    text: str
    kind: str = "notify"
    """"notify" (a notification) or "call" (Garvis rings the phone)."""
    status: str = SCHEDULED
    due_at: float | None = None
    """Epoch seconds of the next time it rings; None once finished."""
    tz: str = "UTC"
    repeat: str = "none"
    local_time: str = ""
    """"HH:MM" for repeating reminders."""
    days: list[int] = field(default_factory=list)
    snoozed: bool = False
    """due_at was moved by a snooze (not on the repeat rule)."""
    origin: str = "assistant"
    request_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_fired_at: float | None = None
    fire_count: int = 0
    spoken: str = ""
    """The sentence Garvis says when it rings (see app/reminders/phrasing.py)."""
    spoken_key: str = ""
    """phrasing_key() of the text/rule `spoken` was written for."""
    spoken_by: str = ""
    """"rules" or "model"."""

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("request_id", None)
        data.pop("spoken_key", None)
        data["due_local"] = describe(self.due_at, self.tz) if self.due_at else ""
        return data


def phrasing_key(reminder: Reminder) -> str:
    """Changes when what Garvis should say changes (text or repeat rule) --
    not when a repeating reminder moves on to its next time."""
    basis = "|".join([reminder.text, reminder.repeat, reminder.local_time, ",".join(map(str, reminder.days))])
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        raise ReminderError("The reminder needs some text.")
    if len(text) > MAX_TEXT:
        raise ReminderError(f"The reminder text is too long (limit {MAX_TEXT} characters).")
    return text


class ReminderStore:
    def __init__(
        self, path: Path, *, clock: Callable[[], float] = time.time, phrase: Callable[[Reminder], str] | None = None
    ) -> None:
        """`phrase` (ReminderPhraser-based) writes `spoken` with the rules;
        None leaves it empty (callers then compose it on demand)."""
        self._path = Path(path)
        self._clock = clock
        self._phrase = phrase
        self._lock = threading.Lock()
        self._items: dict[str, Reminder] = {}
        self.version = 0
        self._listeners: list[Callable[[], None]] = []
        self._load()

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    # ---------------------------------------------------------------- reads
    def list(self, *, include_finished: bool = True) -> list[Reminder]:
        with self._lock:
            items = [r for r in self._items.values() if include_finished or r.status == SCHEDULED]
        return sorted(items, key=lambda r: (r.status != SCHEDULED, r.due_at or float("inf"), -r.updated_at))

    def get(self, reminder_id: str) -> Reminder | None:
        with self._lock:
            return self._items.get(reminder_id)

    def find(self, text_or_id: str) -> Reminder | None:
        """By id, or a unique case-insensitive text match among scheduled ones."""
        exact = self.get(text_or_id)
        if exact is not None:
            return exact
        needle = (text_or_id or "").strip().lower()
        if not needle:
            return None
        hits = [r for r in self.list(include_finished=False) if needle in r.text.lower()]
        return hits[0] if len(hits) == 1 else None

    # --------------------------------------------------------------- writes
    def create(
        self,
        text: str,
        *,
        tz: str,
        due_at: float | None = None,
        kind: str = "notify",
        repeat: str = "none",
        local_time: str = "",
        days=None,
        origin: str = "assistant",
        request_id: str | None = None,
    ) -> Reminder:
        if request_id:
            if not _REQUEST_ID.match(request_id):
                raise ReminderError("Invalid request id.")
            with self._lock:
                existing = next((r for r in self._items.values() if r.request_id == request_id), None)
            if existing is not None:
                return existing
        reminder = Reminder(
            id=secrets.token_urlsafe(9), text=_clean_text(text), origin=origin[:20], request_id=request_id or ""
        )
        self._apply_schedule(reminder, tz=tz, due_at=due_at, kind=kind, repeat=repeat, local_time=local_time, days=days)
        with self._lock:
            if sum(1 for r in self._items.values() if r.status == SCHEDULED) >= MAX_ACTIVE:
                raise ReminderError(f"There are already {MAX_ACTIVE} scheduled reminders.")
            self._items[reminder.id] = reminder
        self._changed("created", reminder)
        return reminder

    def update(
        self,
        reminder_id: str,
        *,
        text: str | None = None,
        tz: str | None = None,
        due_at: float | None = None,
        kind: str | None = None,
        repeat: str | None = None,
        local_time: str | None = None,
        days=None,
    ) -> Reminder:
        reminder = self._require(reminder_id)
        if reminder.status == CANCELLED:
            raise ReminderError("That reminder was cancelled; create a new one instead.")
        if text is not None:
            reminder.text = _clean_text(text)
        schedule_changed = any(v is not None for v in (tz, due_at, repeat, local_time, days))
        if schedule_changed or kind is not None:
            # A new instant without a rule makes it one-off; otherwise the rule stays.
            new_repeat = repeat if repeat is not None else ("none" if due_at is not None else reminder.repeat)
            new_due = due_at if due_at is not None else (reminder.due_at if new_repeat == "none" else None)
            self._apply_schedule(
                reminder,
                tz=tz or reminder.tz,
                due_at=new_due,
                kind=kind or reminder.kind,
                repeat=new_repeat,
                local_time=local_time if local_time is not None else reminder.local_time,
                days=days if days is not None else reminder.days,
            )
            reminder.status = SCHEDULED
        self._changed("updated", reminder)
        return reminder

    def complete(self, reminder_id: str) -> Reminder:
        """Done: a one-off finishes; a repeating one skips to its next time."""
        reminder = self._require(reminder_id)
        if reminder.status != SCHEDULED:
            return reminder
        self._advance(reminder, after=max(self._clock(), reminder.due_at or 0))
        self._changed("completed", reminder)
        return reminder

    def cancel(self, reminder_id: str) -> Reminder:
        reminder = self._require(reminder_id)
        if reminder.status == CANCELLED:
            return reminder
        reminder.status, reminder.due_at, reminder.snoozed = CANCELLED, None, False
        self._changed("cancelled", reminder)
        return reminder

    def snooze(self, reminder_id: str, minutes: float) -> Reminder:
        reminder = self._require(reminder_id)
        if reminder.status == CANCELLED:
            raise ReminderError("A cancelled reminder can't be snoozed.")
        if not 1 <= minutes <= 24 * 60:
            raise ReminderError("Snooze for between 1 minute and 24 hours.")
        reminder.status, reminder.due_at, reminder.snoozed = SCHEDULED, self._clock() + minutes * 60, True
        self._changed("snoozed", reminder)
        return reminder

    def set_spoken(self, reminder_id: str, spoken: str, *, key: str, by: str = "model") -> bool:
        """Store a better sentence for the reminder -- only if it was written
        for its current text (`key`); False when the reminder changed since."""
        reminder = self.get(reminder_id)
        if reminder is None or reminder.spoken_key != key or reminder.spoken == spoken:
            return False
        reminder.spoken, reminder.spoken_by = spoken, by
        self._changed("rephrased", reminder)
        return True

    def fired(self, reminder_id: str, occurrence: float) -> Reminder:
        """The phone rang `occurrence`. Only the current occurrence counts."""
        reminder = self._require(reminder_id)
        if reminder.status != SCHEDULED or reminder.due_at is None or abs(reminder.due_at - occurrence) > 1.0:
            return reminder
        reminder.last_fired_at = self._clock()
        reminder.fire_count += 1
        self._advance(reminder, after=max(occurrence, self._clock() - 1))
        self._changed("fired", reminder)
        return reminder

    # -------------------------------------------------------------- helpers
    def _require(self, reminder_id: str) -> Reminder:
        reminder = self.get(reminder_id)
        if reminder is None:
            raise ReminderError("Unknown reminder.")
        return reminder

    def _advance(self, reminder: Reminder, *, after: float) -> None:
        reminder.snoozed = False
        upcoming = next_occurrence(reminder.repeat, reminder.local_time, reminder.tz, reminder.days, after)
        if upcoming is None:
            reminder.status, reminder.due_at = COMPLETED, None
        else:
            reminder.due_at = upcoming

    def _apply_schedule(
        self, reminder: Reminder, *, tz: str, due_at: float | None, kind: str, repeat: str, local_time: str, days
    ) -> None:
        if kind not in KINDS:
            raise ReminderError("kind must be 'notify' or 'call'.")
        if repeat not in REPEATS:
            raise ReminderError(f"repeat must be one of {', '.join(REPEATS)}.")
        if not is_valid_zone(tz):
            if tz and not zone_database_available():
                if repeat != "none":
                    raise ReminderError(TZDATA_MISSING)
                # A one-off's instant is absolute; its zone only labels it. Without the zone
                # database, label it with this PC's offset rather than refuse the reminder.
                log_extra(logger, 30, "reminder.zone_fallback", reason="tzdata_missing")
                tz = local_fixed_zone_name()
            else:
                raise ReminderError(f"Unknown time zone: {tz!r}")
        try:
            day_list = normalize_days(days) if repeat == "weekly" else []
            if repeat == "none":
                if due_at is None:
                    raise ReminderError("A one-time reminder needs a time.")
                if due_at <= self._clock() - 60:
                    raise ReminderError("That time is already in the past.")
                local_time = ""
                next_due = float(due_at)
            else:
                hour, minute = parse_hhmm(local_time)
                local_time = f"{hour:02d}:{minute:02d}"
                if repeat == "weekly" and not day_list:
                    raise ReminderError("A weekly reminder needs at least one weekday.")
                next_due = next_occurrence(repeat, local_time, tz, day_list, self._clock())
        except ScheduleError as exc:
            raise ReminderError(str(exc)) from exc
        reminder.kind, reminder.tz, reminder.repeat = kind, tz, repeat
        reminder.local_time, reminder.days, reminder.due_at, reminder.snoozed = local_time, day_list, next_due, False

    def _refresh_spoken(self, reminder: Reminder) -> None:
        key = phrasing_key(reminder)
        if self._phrase is None or reminder.spoken_key == key:
            return
        try:
            reminder.spoken, reminder.spoken_key, reminder.spoken_by = self._phrase(reminder), key, "rules"
        except Exception:  # noqa: BLE001 -- wording must never stop a reminder from being saved
            logger.exception("reminder.phrasing_failed")

    def _changed(self, action: str, reminder: Reminder) -> None:
        reminder.updated_at = self._clock()
        self._refresh_spoken(reminder)
        with self._lock:
            self.version += 1
            self._trim_locked()
            self._save_locked()
        log_extra(logger, 20, "reminder." + action, reminder=reminder.id, status=reminder.status, kind=reminder.kind)
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:  # noqa: BLE001
                logger.exception("reminder.listener_failed")

    def _trim_locked(self) -> None:
        finished = sorted((r for r in self._items.values() if r.status != SCHEDULED), key=lambda r: r.updated_at)
        for reminder in finished[: max(0, len(finished) - KEEP_FINISHED)]:
            del self._items[reminder.id]

    # ---------------------------------------------------------- persistence
    def _save_locked(self) -> None:
        payload = {"version": self.version, "reminders": [asdict(r) for r in self._items.values()]}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".reminders-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp, self._path)
        except OSError:
            log_extra(logger, 40, "reminder.save_failed")

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except OSError, ValueError:
            log_extra(logger, 30, "reminder.load_failed")
            return
        fields = set(Reminder.__dataclass_fields__)
        for record in data.get("reminders", []) if isinstance(data, dict) else []:
            if not isinstance(record, dict) or not record.get("id"):
                continue
            try:
                reminder = Reminder(**{k: v for k, v in record.items() if k in fields})
            except TypeError:
                continue
            self._refresh_spoken(reminder)  # reminders saved before phrasing existed
            self._items[reminder.id] = reminder
        self.version = int(data.get("version", 0)) if isinstance(data, dict) else 0
