"""Reminders: time parsing, recurrence (incl. DST), the store's idempotency
and persistence, and the assistant's reminder tools. No phone, no network."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.reminders.schedule import (
    ScheduleError,
    get_zone,
    next_occurrence,
    normalize_days,
    parse_hhmm,
    resolve_once,
)
from app.reminders.store import CANCELLED, COMPLETED, SCHEDULED, ReminderError, ReminderStore
from app.reminders.tools import ReminderZone, build_reminder_tools

TZ = "Asia/Jerusalem"
ZONE = ZoneInfo(TZ)


def _ts(y, mo, d, h=0, mi=0, zone=ZONE) -> float:
    return dt.datetime(y, mo, d, h, mi, tzinfo=zone).timestamp()


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text,expected", [("8:05", (8, 5)), ("20:30", (20, 30)), ("8pm", (20, 0)),
                                           ("8:30 am", (8, 30)), ("12am", (0, 0)), ("12 pm", (12, 0))])
def test_parse_hhmm(text, expected):
    assert parse_hhmm(text) == expected


@pytest.mark.parametrize("text", ["25:00", "8:61", "noonish", "", "13pm"])
def test_parse_hhmm_rejects_nonsense(text):
    with pytest.raises(ScheduleError):
        parse_hhmm(text)


def test_weekday_names_and_numbers():
    assert normalize_days(["Sunday", "mon", 2]) == [0, 2, 6]
    with pytest.raises(ScheduleError):
        normalize_days(["funday"])
    with pytest.raises(ScheduleError):
        normalize_days(["t"])  # ambiguous


def test_fixed_offset_zone_names_work():
    assert get_zone("UTC+03:00").utcoffset(None) == dt.timedelta(hours=3)
    assert get_zone("UTC-05:30").utcoffset(None) == -dt.timedelta(hours=5, minutes=30)
    with pytest.raises(ScheduleError):
        get_zone("Mars/Olympus")


def test_resolve_once_relative_and_absolute():
    now = _ts(2026, 10, 25, 18, 0)
    assert resolve_once(tz=TZ, now=now, in_minutes=30) == now + 1800
    assert resolve_once(tz=TZ, now=now, time="20:00") == _ts(2026, 10, 25, 20, 0)
    # A bare time already past today means tomorrow.
    assert resolve_once(tz=TZ, now=now, time="07:00") == _ts(2026, 10, 26, 7, 0)
    assert resolve_once(tz=TZ, now=now, time="7am", date="tomorrow") == _ts(2026, 10, 26, 7, 0)
    assert resolve_once(tz=TZ, now=now, time="09:00", date="2026-11-02") == _ts(2026, 11, 2, 9, 0)
    # 2026-10-25 is a Sunday: "friday" is the coming Friday.
    assert resolve_once(tz=TZ, now=now, time="09:00", date="friday") == _ts(2026, 10, 30, 9, 0)
    with pytest.raises(ScheduleError):
        resolve_once(tz=TZ, now=now, time="09:00", date="2026-01-01")
    with pytest.raises(ScheduleError):
        resolve_once(tz=TZ, now=now)


# ---------------------------------------------------------------- recurrence
def test_weekly_sunday_rule():
    saturday_noon = _ts(2026, 10, 24, 12, 0)
    assert next_occurrence("weekly", "09:00", TZ, [6], saturday_noon) == _ts(2026, 10, 25, 9, 0)
    # Exactly at an occurrence -> the next one (strictly after).
    assert next_occurrence("weekly", "09:00", TZ, [6], _ts(2026, 10, 25, 9, 0)) == _ts(2026, 11, 1, 9, 0)


def test_weekdays_rule_skips_the_weekend():
    friday_evening = _ts(2026, 10, 30, 20, 0)
    assert next_occurrence("weekdays", "07:30", TZ, [], friday_evening) == _ts(2026, 11, 2, 7, 30)


def test_daily_rule_across_a_dst_change_keeps_the_wall_clock_time():
    # Israel leaves summer time on Sunday 2026-10-25 02:00 -> 01:00.
    before = _ts(2026, 10, 23, 9, 0)
    first = next_occurrence("daily", "08:00", TZ, [], before)
    second = next_occurrence("daily", "08:00", TZ, [], first)
    assert dt.datetime.fromtimestamp(first, ZONE).hour == 8
    assert dt.datetime.fromtimestamp(second, ZONE).hour == 8
    assert second - first == 25 * 3600  # the day the clocks went back had 25 hours


def test_nonexistent_spring_forward_time_moves_later_like_java_time():
    # US/Eastern 2026-03-08 02:30 doesn't exist; java.time gives 03:30 EDT.
    zone = ZoneInfo("America/New_York")
    occurrence = next_occurrence("daily", "02:30", "America/New_York", [], _ts(2026, 3, 7, 12, 0, zone))
    local = dt.datetime.fromtimestamp(occurrence, zone)
    assert (local.month, local.day, local.hour, local.minute) == (3, 8, 3, 30)


def test_one_off_has_no_next_occurrence():
    assert next_occurrence("none", "", TZ, [], 0) is None
    assert next_occurrence("weekly", "09:00", TZ, [], 0) is None


# --------------------------------------------------------------------- store
def _store(tmp_path, now=None):
    clock = Clock(now or _ts(2026, 10, 25, 18, 0))
    return ReminderStore(tmp_path / "reminders.json", clock=clock), clock


def test_create_is_idempotent_per_request_id(tmp_path):
    store, clock = _store(tmp_path)
    first = store.create("Take the book", tz=TZ, due_at=clock.now + 600, request_id="req-abcdef12")
    again = store.create("Take the book", tz=TZ, due_at=clock.now + 600, request_id="req-abcdef12")
    assert again.id == first.id
    assert len(store.list()) == 1


def test_rejects_past_and_bad_input(tmp_path):
    store, clock = _store(tmp_path)
    with pytest.raises(ReminderError):
        store.create("Late", tz=TZ, due_at=clock.now - 3600)
    with pytest.raises(ReminderError):
        store.create("  ", tz=TZ, due_at=clock.now + 60)
    with pytest.raises(ReminderError):
        store.create("x", tz="Nowhere/Land", due_at=clock.now + 60)
    with pytest.raises(ReminderError):
        store.create("x", tz=TZ, repeat="weekly", local_time="09:00", days=[])
    with pytest.raises(ReminderError):
        store.create("x", tz=TZ, due_at=clock.now + 60, kind="sms")


def test_fired_is_idempotent_and_completes_a_one_off(tmp_path):
    store, clock = _store(tmp_path)
    reminder = store.create("Meeting", tz=TZ, due_at=clock.now + 60, kind="call")
    occurrence = reminder.due_at
    clock.now += 61
    version = store.version
    store.fired(reminder.id, occurrence)
    assert store.get(reminder.id).status == COMPLETED
    assert store.get(reminder.id).fire_count == 1
    store.fired(reminder.id, occurrence)  # a retried report changes nothing
    assert store.get(reminder.id).fire_count == 1
    assert store.version == version + 1


def test_fired_advances_a_repeating_reminder_once(tmp_path):
    store, clock = _store(tmp_path, _ts(2026, 10, 24, 12, 0))
    reminder = store.create("Prepare for school", tz=TZ, repeat="weekly", local_time="09:00", days=["sunday"])
    first = reminder.due_at
    assert first == _ts(2026, 10, 25, 9, 0)
    clock.now = first + 5
    store.fired(reminder.id, first)
    store.fired(reminder.id, first)
    assert store.get(reminder.id).due_at == _ts(2026, 11, 1, 9, 0)
    assert store.get(reminder.id).status == SCHEDULED


def test_snooze_complete_cancel(tmp_path):
    store, clock = _store(tmp_path)
    reminder = store.create("Water plants", tz=TZ, due_at=clock.now + 60)
    store.snooze(reminder.id, 10)
    assert store.get(reminder.id).due_at == clock.now + 600 and store.get(reminder.id).snoozed
    store.complete(reminder.id)
    assert store.get(reminder.id).status == COMPLETED and store.get(reminder.id).due_at is None
    store.snooze(reminder.id, 5)  # a delivered one-off can still be snoozed from its notification
    assert store.get(reminder.id).status == SCHEDULED
    store.cancel(reminder.id)
    assert store.get(reminder.id).status == CANCELLED
    with pytest.raises(ReminderError):
        store.snooze(reminder.id, 5)
    with pytest.raises(ReminderError):
        store.snooze("nope", 5)


def test_update_changes_time_and_kind(tmp_path):
    store, clock = _store(tmp_path)
    reminder = store.create("Call mom", tz=TZ, due_at=clock.now + 60)
    store.update(reminder.id, kind="call")
    assert store.get(reminder.id).kind == "call" and store.get(reminder.id).due_at == clock.now + 60
    store.update(reminder.id, repeat="daily", local_time="19:00")
    assert store.get(reminder.id).repeat == "daily"
    assert store.get(reminder.id).due_at == _ts(2026, 10, 25, 19, 0)
    store.update(reminder.id, due_at=clock.now + 3600)
    assert store.get(reminder.id).repeat == "none"


def test_persists_and_reloads(tmp_path):
    store, clock = _store(tmp_path)
    reminder = store.create("Persist me", tz=TZ, due_at=clock.now + 60, request_id="persist-1234")
    reloaded = ReminderStore(tmp_path / "reminders.json", clock=clock)
    assert reloaded.get(reminder.id).text == "Persist me"
    assert reloaded.version == store.version
    # Dedupe still works after a restart -> no duplicate after a retried request.
    assert reloaded.create("Persist me", tz=TZ, due_at=clock.now + 60, request_id="persist-1234").id == reminder.id
    raw = json.loads((tmp_path / "reminders.json").read_text(encoding="utf-8"))
    assert len(raw["reminders"]) == 1


@pytest.fixture
def no_zone_database(monkeypatch):
    """Garvis's real interpreter on Windows without `tzdata`: no IANA zone loads (fixed offsets still do)."""
    from zoneinfo import ZoneInfoNotFoundError

    import app.reminders.schedule as schedule

    def missing(name):
        raise ZoneInfoNotFoundError(f"No time zone found with key {name}")

    monkeypatch.setattr(schedule, "ZoneInfo", missing)


def test_zone_database_check(no_zone_database):
    from app.reminders.schedule import is_valid_zone, zone_database_available

    assert not zone_database_available()
    assert not is_valid_zone(TZ)
    assert is_valid_zone("UTC+02:00")


def test_phone_one_off_call_is_accepted_without_tzdata(tmp_path, no_zone_database):
    # The regression: the phone always sends its IANA zone; without tzdata every phone reminder was
    # refused ("Unknown time zone"), so a call reminder never reached the PC and the phone dropped its alarm.
    store, clock = _store(tmp_path)
    due = clock.now + 3600
    reminder = store.create("Meeting", tz=TZ, due_at=due, kind="call", origin="phone", request_id="call-abcdef12")
    assert reminder.kind == "call" and reminder.status == SCHEDULED
    assert reminder.due_at == due  # the instant is kept exactly; only the zone label changes
    assert reminder.tz.startswith("UTC")
    assert store.create("Meeting", tz=TZ, due_at=due, kind="call", request_id="call-abcdef12").id == reminder.id
    reloaded = ReminderStore(tmp_path / "reminders.json", clock=clock)
    assert reloaded.get(reminder.id).due_at == due and reloaded.get(reminder.id).kind == "call"
    assert reloaded.get(reminder.id).public()["due_local"]  # still describable after a restart


def test_repeating_reminder_without_tzdata_says_how_to_fix_it(tmp_path, no_zone_database):
    store, _ = _store(tmp_path)
    with pytest.raises(ReminderError, match="tzdata"):
        store.create("Standup", tz=TZ, repeat="daily", local_time="09:00", kind="call")
    assert store.list() == []


def test_unknown_zone_is_still_refused_when_zones_load(tmp_path):
    store, clock = _store(tmp_path)
    with pytest.raises(ReminderError, match="Unknown time zone"):
        store.create("x", tz="Nowhere/Land", due_at=clock.now + 600)


def test_listeners_hear_every_change(tmp_path):
    store, clock = _store(tmp_path)
    calls = []
    store.add_listener(lambda: calls.append(1))
    reminder = store.create("x", tz=TZ, due_at=clock.now + 60)
    store.cancel(reminder.id)
    assert len(calls) == 2


# --------------------------------------------------------------------- tools
async def _run(tool, **args):
    return await tool.execute(tool.validate_arguments(args))


@pytest.fixture
def tools(tmp_path):
    store, clock = _store(tmp_path)
    zone = ReminderZone("")
    zone.report(TZ)
    built = {tool.name: tool for tool in build_reminder_tools(store, zone, phone_ready=lambda: False)}
    for tool in built.values():
        tool._clock = clock
    return built, store, clock


async def test_tool_remind_in_30_minutes(tools):
    built, store, clock = tools
    result = await _run(built["create_reminder"], text="Take my book", in_minutes=30)
    assert result.success
    assert store.get(result.data["id"]).due_at == clock.now + 1800
    assert "No phone" in result.data["note"]


async def test_tool_call_me_at_8pm(tools):
    built, store, _ = tools
    result = await _run(built["create_reminder"], text="Your meeting", time="20:00", call=True)
    reminder = store.get(result.data["id"])
    assert reminder.kind == "call" and reminder.due_at == _ts(2026, 10, 25, 20, 0)
    assert "20:00" in result.data["when"]


async def test_tool_every_sunday(tools):
    built, store, _ = tools
    missing_time = await _run(built["create_reminder"], text="Prepare for school", repeat="weekly",
                              weekdays=["sunday"])
    assert not missing_time.success
    result = await _run(built["create_reminder"], text="Prepare for school", repeat="weekly", weekdays=["sunday"],
                        time="18:30")
    assert result.success and store.get(result.data["id"]).days == [6]


async def test_tool_snooze_cancel_list_by_text(tools):
    built, store, clock = tools
    created = await _run(built["create_reminder"], text="Buy milk", in_minutes=10)
    assert (await _run(built["snooze_reminder"], reminder="milk", minutes=15)).success
    assert store.get(created.data["id"]).due_at == clock.now + 900
    listed = await _run(built["list_reminders"])
    assert listed.data["reminders"][0]["text"] == "Buy milk" and listed.data["time_zone"] == TZ
    assert (await _run(built["cancel_reminder"], reminder=created.data["id"])).success
    assert not (await _run(built["cancel_reminder"], reminder="no such thing")).success


async def test_tool_update_moves_the_time(tools):
    built, store, _ = tools
    created = await _run(built["create_reminder"], text="Dentist", time="20:00")
    result = await _run(built["update_reminder"], reminder=created.data["id"], time="21:15")
    assert result.success and store.get(created.data["id"]).due_at == _ts(2026, 10, 25, 21, 15)


def test_zone_preference_order():
    zone = ReminderZone("")
    assert zone.current().startswith("UTC")
    zone.report("not/a zone")
    assert zone.current().startswith("UTC")
    zone.report("Europe/London")
    assert zone.current() == "Europe/London"
    assert ReminderZone("Asia/Tokyo").current() == "Asia/Tokyo"


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "android").exists(),
                    reason="Garvis Mobile (android/) is not part of this repository")
def test_android_recurrence_fixture_matches_these_rules():
    """The Android app's Recurrence.kt is tested against this fixture; it must
    stay what the PC's rules produce (regenerate: python android/tools/make_recurrence_fixture.py)."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("fixture", root / "android" / "tools" / "make_recurrence_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stored = json.loads((root / "android" / "app" / "src" / "test" / "resources" / "recurrence_cases.json").read_text())
    assert stored == module.cases()
