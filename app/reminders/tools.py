"""Assistant tools for reminders and scheduled calls ("remind me in 30
minutes...", "call me at 8 PM...", "every Sunday remind me...").

Times are interpreted in the user's time zone (see ReminderZone): the
phone reports its IANA zone, GARVIS_REMINDERS_TIMEZONE overrides it. The
deterministic parsing lives in schedule.py -- the model passes the words
("tomorrow", "20:00", "sunday"), never a computed timestamp.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from app.reminders.schedule import ScheduleError, describe, get_zone, local_fixed_zone_name, resolve_date, resolve_once
from app.reminders.store import Reminder, ReminderError, ReminderStore
from app.tools.base import RiskLevel, Tool, ToolResult

Repeat = Literal["none", "daily", "weekdays", "weekly"]


class ReminderZone:
    """Which time zone "8 PM" means: the configured one, else the zone the
    paired phone last reported, else this PC's current UTC offset."""

    def __init__(self, configured: str = "") -> None:
        self.configured = configured.strip()
        self.reported = ""

    def report(self, zone: str) -> None:
        from app.reminders.schedule import is_valid_zone

        if zone and is_valid_zone(zone):
            self.reported = zone

    def current(self) -> str:
        return self.configured or self.reported or local_fixed_zone_name()


def _summary(reminder: Reminder) -> dict:
    data = {"id": reminder.id, "text": reminder.text, "kind": reminder.kind, "status": reminder.status,
            "when": describe(reminder.due_at, reminder.tz) if reminder.due_at else "", "repeat": reminder.repeat}
    if reminder.repeat == "weekly":
        data["days"] = reminder.days
    if reminder.repeat != "none":
        data["time"] = reminder.local_time
    return data


class _Base(Tool):
    risk_level = RiskLevel.LOW

    def __init__(self, store: ReminderStore, zone: ReminderZone,
                 phone_ready: Callable[[], bool] = lambda: True, clock: Callable[[], float] = time.time) -> None:
        self._store = store
        self._zone = zone
        self._phone_ready = phone_ready
        self._clock = clock

    def _lookup(self, reminder: str) -> Reminder:
        found = self._store.find(reminder)
        if found is None:
            raise ReminderError(f"No single reminder matches {reminder!r}; call list_reminders for the ids.")
        return found

    def _note(self) -> str:
        return "" if self._phone_ready() else (
            "No phone with the Garvis app is paired, so nothing will ring until one is paired.")

    def _schedule_args(self, *, time_: str | None, date: str | None, in_minutes: float | None, repeat: str,
                       weekdays: list[str] | None) -> dict:
        """-> keyword arguments for ReminderStore.create/update."""
        tz = self._zone.current()
        now = self._clock()
        if repeat == "none":
            return {"tz": tz, "due_at": resolve_once(tz=tz, now=now, time=time_, date=date, in_minutes=in_minutes),
                    "repeat": "none"}
        if in_minutes is not None or not time_:
            raise ScheduleError("A repeating reminder needs a time of day (HH:MM).")
        days = list(weekdays or [])
        if repeat == "weekly" and not days:
            if not date:
                raise ScheduleError("Say which weekday(s) it repeats on.")
            days = [resolve_date(date, get_zone(tz), now).weekday()]
        return {"tz": tz, "repeat": repeat, "local_time": time_, "days": days}


class CreateReminderInput(BaseModel):
    text: str = Field(description="What to remind the user about, phrased for them, e.g. 'Take your book'.")
    time: str | None = Field(default=None, description="Time of day, HH:MM 24-hour (e.g. '20:00', '07:00').")
    date: str | None = Field(default=None, description="'today', 'tomorrow', a weekday name or YYYY-MM-DD. "
                             "Omit for the next time `time` comes round.")
    in_minutes: float | None = Field(default=None, description="Relative delay instead of time/date, e.g. 30.")
    repeat: Repeat = Field(default="none")
    weekdays: list[str] | None = Field(default=None, description="For repeat='weekly': e.g. ['sunday'].")
    call: bool = Field(default=False, description="True when the user asks to be CALLED: the phone rings with "
                       "an incoming Garvis call and Garvis speaks the reminder. False = a notification.")


class CreateReminderTool(_Base):
    name = "create_reminder"
    description = (
        "Create a reminder that rings on the user's phone (Garvis app), in the user's time zone. Examples: "
        "'remind me in 30 minutes to take my book' -> in_minutes=30; 'remind me tomorrow at 7 AM' -> "
        "date='tomorrow', time='07:00'; 'call me at 8 PM about my meeting' -> time='20:00', call=true; "
        "'every Sunday remind me to prepare for school' -> repeat='weekly', weekdays=['sunday'] and a time "
        "(ask for the time if the user gave none). Report the returned 'when' back to the user."
    )
    input_model = CreateReminderInput

    async def execute(self, args: CreateReminderInput) -> ToolResult:
        try:
            schedule = self._schedule_args(time_=args.time, date=args.date, in_minutes=args.in_minutes,
                                           repeat=args.repeat, weekdays=args.weekdays)
            reminder = self._store.create(args.text, kind="call" if args.call else "notify", origin="assistant",
                                          **schedule)
        except (ReminderError, ScheduleError) as exc:
            return ToolResult.fail(str(exc))
        result = _summary(reminder)
        if note := self._note():
            result["note"] = note
        return ToolResult.ok(result)


class ListRemindersInput(BaseModel):
    include_finished: bool = Field(default=False, description="Also list completed and cancelled ones.")


class ListRemindersTool(_Base):
    name = "list_reminders"
    description = "List the user's reminders and scheduled calls (upcoming first) with their ids."
    input_model = ListRemindersInput

    async def execute(self, args: ListRemindersInput) -> ToolResult:
        items = self._store.list(include_finished=args.include_finished)
        return ToolResult.ok({"reminders": [_summary(r) for r in items[:50]], "time_zone": self._zone.current()})


class UpdateReminderInput(BaseModel):
    reminder: str = Field(description="The reminder's id (from list_reminders) or a unique part of its text.")
    text: str | None = None
    time: str | None = Field(default=None, description="New time of day, HH:MM 24-hour.")
    date: str | None = Field(default=None, description="'today', 'tomorrow', a weekday or YYYY-MM-DD.")
    in_minutes: float | None = None
    repeat: Repeat | None = None
    weekdays: list[str] | None = None
    call: bool | None = Field(default=None, description="True = ring as a call, False = a notification.")


class UpdateReminderTool(_Base):
    name = "update_reminder"
    description = "Change a reminder's text, time, repetition or call/notification kind."
    input_model = UpdateReminderInput

    async def execute(self, args: UpdateReminderInput) -> ToolResult:
        try:
            reminder = self._lookup(args.reminder)
            changes: dict = {}
            if args.text is not None:
                changes["text"] = args.text
            if args.call is not None:
                changes["kind"] = "call" if args.call else "notify"
            if any(v is not None for v in (args.time, args.date, args.in_minutes, args.repeat, args.weekdays)):
                repeat = args.repeat or ("none" if args.in_minutes is not None else reminder.repeat)
                time_ = args.time or (reminder.local_time if repeat != "none" else None)
                changes.update(self._schedule_args(time_=time_, date=args.date, in_minutes=args.in_minutes,
                                                   repeat=repeat, weekdays=args.weekdays or reminder.days or None))
            reminder = self._store.update(reminder.id, **changes)
        except (ReminderError, ScheduleError) as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(_summary(reminder))


class ReminderRefInput(BaseModel):
    reminder: str = Field(description="The reminder's id (from list_reminders) or a unique part of its text.")


class CancelReminderTool(_Base):
    name = "cancel_reminder"
    description = "Cancel a reminder or scheduled call (it stays listed as cancelled)."
    input_model = ReminderRefInput

    async def execute(self, args: ReminderRefInput) -> ToolResult:
        try:
            return ToolResult.ok(_summary(self._store.cancel(self._lookup(args.reminder).id)))
        except ReminderError as exc:
            return ToolResult.fail(str(exc))


class CompleteReminderTool(_Base):
    name = "complete_reminder"
    description = "Mark a reminder done. A repeating reminder moves on to its next time."
    input_model = ReminderRefInput

    async def execute(self, args: ReminderRefInput) -> ToolResult:
        try:
            reminder = self._store.complete(self._lookup(args.reminder).id)
        except ReminderError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(_summary(reminder))


class SnoozeReminderInput(ReminderRefInput):
    minutes: float = Field(default=10, description="How long to snooze, 1-1440 minutes.")


class SnoozeReminderTool(_Base):
    name = "snooze_reminder"
    description = "Snooze a reminder: it rings again after the given number of minutes."
    input_model = SnoozeReminderInput

    async def execute(self, args: SnoozeReminderInput) -> ToolResult:
        try:
            reminder = self._store.snooze(self._lookup(args.reminder).id, args.minutes)
        except ReminderError as exc:
            return ToolResult.fail(str(exc))
        return ToolResult.ok(_summary(reminder))


def build_reminder_tools(store: ReminderStore, zone: ReminderZone,
                         phone_ready: Callable[[], bool] = lambda: True) -> list[Tool]:
    return [cls(store, zone, phone_ready) for cls in (CreateReminderTool, ListRemindersTool, UpdateReminderTool,
                                                       CancelReminderTool, CompleteReminderTool, SnoozeReminderTool)]

