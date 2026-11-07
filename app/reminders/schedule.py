"""Time zones, recurrence and the parsing of "when" for reminders.

A companion mobile client (not part of this repository) computes the next
occurrence of a recurring reminder with the same rules so it can re-arm an
alarm while the PC is unreachable; both sides must agree, which is why the
rules are deliberately small:

- repeat "none": one instant.
- repeat "daily" / "weekdays" (Mon-Fri) / "weekly" (on `days`, 0 = Monday)
  at a wall-clock time "HH:MM" in an IANA time zone.

Non-existent wall times (spring-forward gap) resolve with the offset in
force before the gap -- e.g. 02:30 becomes 03:30 -- and repeated wall times
(fall-back overlap) use the earlier instant. java.time's ZonedDateTime.of
does exactly the same, so both sides pick the same instant.
"""
from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

REPEATS = ("none", "daily", "weekdays", "weekly")
WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_FIXED_OFFSET = re.compile(r"^UTC([+-])(\d{2}):(\d{2})$")
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class ScheduleError(ValueError):
    """A time, date or rule can't be understood. The message is user-facing."""


def get_zone(name: str) -> dt.tzinfo:
    """An IANA zone ("Asia/Jerusalem") or a fixed "UTC+03:00" offset."""
    match = _FIXED_OFFSET.match(name or "")
    if match:
        sign = 1 if match.group(1) == "+" else -1
        return dt.timezone(sign * dt.timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ScheduleError(f"Unknown time zone: {name!r}") from exc


def is_valid_zone(name: str) -> bool:
    try:
        get_zone(name)
    except ScheduleError:
        return False
    return bool(name)


def zone_database_available() -> bool:
    """Whether IANA zones can be loaded at all. Windows ships no zone files:
    Python then needs the `tzdata` package, and without it every IANA name
    (the phone always sends one) looks "unknown"."""
    try:
        ZoneInfo("Etc/UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


TZDATA_MISSING = ("Garvis on your PC can't read time zones: the 'tzdata' package is missing from the Python that "
                  "runs Garvis. Install it there (python -m pip install tzdata) and restart Garvis.")


def local_fixed_zone_name() -> str:
    """This PC's current UTC offset as "UTC+HH:MM" -- the last-resort zone
    when neither the settings nor a phone have named an IANA zone."""
    offset = dt.datetime.now().astimezone().utcoffset() or dt.timedelta(0)
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    minutes = abs(minutes)
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def parse_hhmm(text: str) -> tuple[int, int]:
    """"8:05", "08:05", "20:30" -> (hour, minute). Also accepts "8pm"/"8:30 am"."""
    raw = (text or "").strip().lower().replace(".", "")
    ampm = None
    for suffix in ("am", "pm"):
        if raw.endswith(suffix):
            ampm, raw = suffix, raw[: -len(suffix)].strip()
    if ampm and re.fullmatch(r"\d{1,2}", raw):
        raw += ":00"
    match = _HHMM.match(raw)
    if not match:
        raise ScheduleError(f"Couldn't read the time {text!r}; use HH:MM (24-hour).")
    hour, minute = int(match.group(1)), int(match.group(2))
    if ampm:
        if not 1 <= hour <= 12:
            raise ScheduleError(f"Couldn't read the time {text!r}.")
        hour = hour % 12 + (12 if ampm == "pm" else 0)
    return hour, minute


def normalize_days(days) -> list[int]:
    """Weekday numbers (0 = Monday) or names -> sorted unique ints."""
    result: set[int] = set()
    for day in days or []:
        if isinstance(day, int) and 0 <= day <= 6:
            result.add(day)
            continue
        name = str(day).strip().lower()
        matches = [i for i, full in enumerate(WEEKDAY_NAMES) if name and full.startswith(name[:3])]
        if len(name) < 3 or len(matches) != 1:
            raise ScheduleError(f"Unknown weekday: {day!r}")
        result.add(matches[0])
    return sorted(result)


def _instant(date: dt.date, hour: int, minute: int, zone: dt.tzinfo) -> float:
    return dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=zone).timestamp()


def matches_rule(date: dt.date, repeat: str, days: list[int]) -> bool:
    if repeat == "daily":
        return True
    if repeat == "weekdays":
        return date.weekday() < 5
    if repeat == "weekly":
        return date.weekday() in days
    return False


def next_occurrence(repeat: str, local_time: str, tz: str, days: list[int], after: float) -> float | None:
    """The first occurrence strictly after `after` (epoch seconds), or None
    for a one-off rule or a weekly rule without days."""
    if repeat not in ("daily", "weekdays", "weekly") or (repeat == "weekly" and not days):
        return None
    zone = get_zone(tz)
    hour, minute = parse_hhmm(local_time)
    start = dt.datetime.fromtimestamp(after, zone).date()
    for offset in range(0, 9):
        date = start + dt.timedelta(days=offset)
        if matches_rule(date, repeat, days):
            instant = _instant(date, hour, minute, zone)
            if instant > after:
                return instant
    return None


def resolve_date(text: str, zone: dt.tzinfo, now: float) -> dt.date:
    """"today" / "tomorrow" / "YYYY-MM-DD" / a weekday name (the next one,
    today included)."""
    today = dt.datetime.fromtimestamp(now, zone).date()
    word = (text or "").strip().lower()
    if word in ("", "today"):
        return today
    if word == "tomorrow":
        return today + dt.timedelta(days=1)
    try:
        return dt.date.fromisoformat(word)
    except ValueError:
        pass
    try:
        (weekday,) = normalize_days([word])
    except ScheduleError as exc:
        raise ScheduleError(f"Couldn't read the date {text!r}; use today, tomorrow, a weekday or YYYY-MM-DD.") from exc
    return today + dt.timedelta(days=(weekday - today.weekday()) % 7)


def resolve_once(*, tz: str, now: float, time: str | None = None, date: str | None = None,
                 in_minutes: float | None = None) -> float:
    """The instant for a one-off reminder. `in_minutes` wins; otherwise
    `time` (+ optional `date`). A bare time already past today means tomorrow."""
    if in_minutes is not None:
        if not 0 < in_minutes <= 60 * 24 * 366:
            raise ScheduleError("in_minutes must be between 1 minute and a year.")
        return now + in_minutes * 60
    if not time:
        raise ScheduleError("Say when: a time (HH:MM), optionally a date, or a number of minutes from now.")
    zone = get_zone(tz)
    hour, minute = parse_hhmm(time)
    day = resolve_date(date or "", zone, now)
    instant = _instant(day, hour, minute, zone)
    if instant <= now:
        if date and date.strip().lower() not in ("today",):
            raise ScheduleError("That time is already in the past.")
        instant = _instant(day + dt.timedelta(days=1), hour, minute, zone)
    return instant


def describe(instant: float | None, tz: str) -> str:
    """"Sun 26 Oct 2026, 08:00 (Asia/Jerusalem)" -- for tool results and lists."""
    if instant is None:
        return "not scheduled"
    local = dt.datetime.fromtimestamp(instant, get_zone(tz))
    return f"{local.strftime('%a %d %b %Y, %H:%M')} ({tz})"
