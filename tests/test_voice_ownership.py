"""The voice ownership authority (app/core/voice_ownership.py): exactly one endpoint may hold a
microphone; the rest sleep. Pure logic with a fake clock -- the API, Desktop and client tests
build on these rules."""

from __future__ import annotations

import asyncio
import random
import threading

import pytest

from app.core.voice_ownership import (
    DESKTOP_ENDPOINT,
    LISTENING_STATES,
    VoiceClaimRefused,
    VoiceOwnershipManager,
    VoiceState,
)

TABLET = "tab12345"
PHONE = "pho12345"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _manager(**kwargs) -> tuple[VoiceOwnershipManager, Clock]:
    clock = Clock()
    manager = VoiceOwnershipManager(clock=clock, lease_seconds=20.0, release_timeout_seconds=5.0, **kwargs)
    manager.register_local(DESKTOP_ENDPOINT, kind="desktop", label="Garvis Desktop")
    return manager, clock


def _tablet(manager, **kwargs) -> dict:
    return manager.heartbeat(TABLET, kind="station", label="Eye Station", **kwargs)


def _phone(manager, **kwargs) -> dict:
    return manager.heartbeat(PHONE, kind="phone", label="Example Phone", **kwargs)


def _ack(manager, endpoint: str) -> dict:
    """The endpoint closed its microphone after seeing the latest version."""
    if endpoint == DESKTOP_ENDPOINT:
        manager.report_mic(endpoint, False, manager.version)
        return manager.view(endpoint)
    view = manager.view(endpoint)
    return manager.heartbeat(
        endpoint,
        kind="station" if endpoint == TABLET else "phone",
        label=endpoint,
        mic_open=False,
        seen_version=view["version"],
        seen_epoch=view["epoch"],
    )


def _listening(manager) -> list[str]:
    return [e for e in (DESKTOP_ENDPOINT, TABLET, PHONE) if manager.state_of(e) in LISTENING_STATES]


def _assert_single_owner(manager) -> None:
    granted = _listening(manager)
    assert len(granted) <= 1, granted
    may_be_open = manager.listening_endpoints()
    assert len(may_be_open) <= 1 or manager.holder() is None, may_be_open


# ------------------------------------------------------------- standby queue
def test_nothing_owns_standby_until_an_endpoint_asks_for_it():
    manager, _ = _manager()
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    assert manager.holder() is None
    assert manager.view(DESKTOP_ENDPOINT)["reason"] == "no_owner"


def test_first_eligible_endpoint_gets_standby_and_the_second_sleeps():
    manager, _ = _manager()
    view = _tablet(manager, want_standby=True)
    assert view["state"] == "standby" and view["listen"] is True
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)  # the user ticks the Desktop's checkbox later
    assert manager.state_of(TABLET) == VoiceState.STANDBY
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    assert manager.view(DESKTOP_ENDPOINT)["reason"] == "in_use"
    _assert_single_owner(manager)


def test_desktop_first_keeps_standby_when_the_tablet_opens():
    manager, _ = _manager()
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    view = _tablet(manager, want_standby=True)
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY
    assert view["state"] == "sleep" and view["listen"] is False
    assert view["standby_owner"]["kind"] == "desktop"


def test_owner_leaving_hands_standby_to_the_next_in_line_after_it_released():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    manager.leave(TABLET)  # the tablet's app closed (mic released first)
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY
    assert manager.state_of(TABLET) == VoiceState.DISCONNECTED


def test_unticking_standby_on_the_owner_passes_it_on_only_after_its_microphone_closed():
    manager, _ = _manager()
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    _tablet(manager, want_standby=True)
    manager.set_wants_standby(DESKTOP_ENDPOINT, False)
    view = manager.view(TABLET)
    assert view["state"] == "sleep" and view["reason"] == "waiting_for_release"
    _ack(manager, DESKTOP_ENDPOINT)
    assert manager.state_of(TABLET) == VoiceState.STANDBY


def test_reconnecting_endpoint_cannot_steal_standby():
    manager, clock = _manager()
    _tablet(manager, want_standby=True)
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    clock.now += 21  # the tablet lost Wi-Fi: its lease ran out
    manager.tick()
    assert manager.state_of(TABLET) == VoiceState.DISCONNECTED
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY
    view = _tablet(manager, want_standby=True, mic_open=False)  # back online
    assert view["state"] == "sleep"
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY


def test_a_second_station_waits_in_line_in_arrival_order():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    manager.heartbeat("tab2", kind="station", label="Kitchen", want_standby=True)
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    manager.leave(TABLET)
    assert manager.state_of("tab2") == VoiceState.STANDBY  # arrived before the Desktop asked
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP


# ---------------------------------------------------------------- conversations
def test_only_the_standby_owner_can_start_a_wake_phrase_conversation():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    with pytest.raises(VoiceClaimRefused) as refused:
        manager.claim(DESKTOP_ENDPOINT, "active", "wake")
    assert refused.value.code == "not_owner"
    claim, view = manager.claim(TABLET, "active", "wake")
    assert view["state"] == "active"
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    _assert_single_owner(manager)


def test_conversation_end_returns_the_owner_to_standby():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    claim, _ = manager.claim(TABLET, "active", "wake")
    assert manager.release(TABLET, claim.id) is True
    assert manager.state_of(TABLET) == VoiceState.STANDBY
    assert manager.release(TABLET, claim.id) is False  # twice is harmless


def test_explicit_desktop_conversation_puts_the_tablet_to_sleep_and_restores_it_afterwards():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    claim, view = manager.claim(DESKTOP_ENDPOINT, "active", "session", explicit=True)
    assert view["state"] == "sleep" and view["reason"] == "waiting_for_release"  # tablet's mic first
    assert manager.state_of(TABLET) == VoiceState.SLEEP
    _ack(manager, TABLET)
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.ACTIVE
    assert manager.holder() == DESKTOP_ENDPOINT
    manager.release(DESKTOP_ENDPOINT, claim.id)
    assert manager.state_of(TABLET) == VoiceState.SLEEP  # Desktop's mic hasn't confirmed closed yet
    _ack(manager, DESKTOP_ENDPOINT)
    assert manager.state_of(TABLET) == VoiceState.STANDBY


def test_explicit_conversation_on_another_device_replaces_a_running_one():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    tablet_claim, _ = manager.claim(TABLET, "active", "wake")
    manager.claim(PHONE, "active", "conversation", explicit=True, kind="phone", label="Example Phone")
    assert manager.view(TABLET)["claims"] == []  # its conversation was ended
    assert manager.release(TABLET, tablet_claim.id) is False
    _ack(manager, TABLET)
    assert manager.state_of(PHONE) == VoiceState.ACTIVE


def test_owner_vanishing_mid_conversation_hands_standby_to_the_next_in_line():
    manager, clock = _manager()
    _tablet(manager, want_standby=True)
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    manager.claim(TABLET, "active", "wake")
    clock.now += 20.5  # the tablet's battery died mid-sentence: no more heartbeats
    manager.tick()
    assert manager.state_of(TABLET) == VoiceState.DISCONNECTED
    assert manager.view(TABLET)["claims"] == []
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.STANDBY


def test_sleeping_reconnect_stays_asleep_and_owner_keeps_standby():
    manager, clock = _manager()
    manager.set_wants_standby(DESKTOP_ENDPOINT, True)
    _tablet(manager, want_standby=True)
    clock.now += 25
    manager.tick()
    view = _tablet(manager, want_standby=True, mic_open=False)
    assert view["state"] == "sleep" and view["standby_owner"]["kind"] == "desktop"


# ---------------------------------------------------------------------- priority
def test_phone_reminder_takes_priority_and_the_tablet_resumes_standby_afterwards():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    claim, view = manager.claim(PHONE, "priority", "reminder", ref="r1", kind="phone", label="Example Phone")
    assert manager.state_of(TABLET) == VoiceState.SLEEP
    _ack(manager, TABLET)
    assert manager.state_of(PHONE) == VoiceState.PRIORITY
    assert manager.view(TABLET)["standby_owner"]["kind"] == "station"  # standby didn't move
    manager.release(PHONE, claim.id)
    _phone(manager, mic_open=False, seen_version=manager.version, seen_epoch=manager.epoch)
    assert manager.state_of(TABLET) == VoiceState.STANDBY
    assert manager.state_of(PHONE) == VoiceState.SLEEP


def test_garvis_call_takes_priority_over_a_running_tablet_conversation():
    manager, _ = _manager()
    _tablet(manager, want_standby=True)
    manager.claim(TABLET, "active", "wake")
    manager.claim(PHONE, "priority", "call", ref="r2", kind="phone", label="Example Phone")
    assert manager.view(TABLET)["claims"] == []  # the conversation was preempted
    _ack(manager, TABLET)
    assert manager.state_of(PHONE) == VoiceState.PRIORITY
    with pytest.raises(VoiceClaimRefused) as refused:
        manager.claim(DESKTOP_ENDPOINT, "active", "session", explicit=True)
    assert refused.value.code == "priority_elsewhere"
    assert "call" in str(refused.value)


def test_priority_from_the_desktop_is_never_needed_but_two_priorities_queue_in_order():
    manager, _ = _manager()
    first, _ = manager.claim(PHONE, "priority", "call", kind="phone", label="Phone")
    manager.claim("pho2", "priority", "reminder", kind="phone", label="Old phone")
    assert manager.state_of(PHONE) == VoiceState.PRIORITY
    assert manager.state_of("pho2") == VoiceState.SLEEP
    manager.release(PHONE, first.id)
    _phone(manager, mic_open=False, seen_version=manager.version, seen_epoch=manager.epoch)
    assert manager.state_of("pho2") == VoiceState.PRIORITY


def test_priority_ending_unexpectedly_expires_and_restores_the_owner():
    manager, clock = _manager()
    _tablet(manager, want_standby=True)
    manager.claim(PHONE, "priority", "reminder", kind="phone", label="Phone", max_seconds=30)
    _ack(manager, TABLET)
    assert manager.state_of(PHONE) == VoiceState.PRIORITY
    # The phone keeps checking in but never releases (a bug): the cap ends it.
    for _ in range(4):
        clock.now += 9
        _phone(manager)
        _tablet(manager)
    assert manager.view(PHONE)["claims"] == []
    _phone(manager, mic_open=False, seen_version=manager.version, seen_epoch=manager.epoch)
    assert manager.state_of(TABLET) == VoiceState.STANDBY


def test_priority_holder_vanishing_never_leaves_the_system_locked():
    manager, clock = _manager()
    _tablet(manager, want_standby=True)
    manager.claim(PHONE, "priority", "call", kind="phone", label="Phone")
    _ack(manager, TABLET)
    for _ in range(3):  # the phone died mid-call; the tablet keeps checking in
        clock.now += 8
        _tablet(manager)
    assert manager.state_of(PHONE) == VoiceState.DISCONNECTED
    assert manager.state_of(TABLET) == VoiceState.STANDBY
    assert manager.holder() == TABLET


def test_priority_with_no_standby_owner_ends_with_no_listener_rather_than_a_stale_one():
    manager, _ = _manager()
    claim, view = manager.claim(PHONE, "priority", "reminder", kind="phone", label="Phone")
    assert view["state"] == "priority"
    manager.release(PHONE, claim.id)
    assert manager.state_of(PHONE) == VoiceState.SLEEP
    assert manager.holder() is None


# --------------------------------------------------------------- hand-over safety
def test_handover_waits_for_a_release_that_has_seen_the_sleep_order():
    manager, _ = _manager()
    view = _tablet(manager, want_standby=True)
    granted_version = view["version"]
    manager.claim(DESKTOP_ENDPOINT, "active", "session", explicit=True)
    # A report sent before the tablet saw the order to sleep proves nothing.
    _tablet(manager, mic_open=False, seen_version=granted_version, seen_epoch=manager.epoch)
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    # From an older server run: proves nothing either.
    _tablet(manager, mic_open=False, seen_version=10_000, seen_epoch="other")
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    _ack(manager, TABLET)
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.ACTIVE


def test_a_listening_owner_reporting_a_closed_mic_is_still_treated_as_listening():
    manager, _ = _manager()
    view = _tablet(manager, want_standby=True)
    _tablet(manager, mic_open=False, seen_version=view["version"], seen_epoch=manager.epoch)
    assert TABLET in manager.listening_endpoints()  # it may open it again at any moment


def test_client_that_never_releases_is_cut_off_by_its_lease():
    manager, clock = _manager()
    _tablet(manager, want_standby=True)
    manager.claim(DESKTOP_ENDPOINT, "active", "session", explicit=True)
    for _ in range(6):  # still heartbeating, still claiming its mic is open
        clock.now += 4
        _tablet(manager, mic_open=True, seen_version=manager.version, seen_epoch=manager.epoch)
        assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.SLEEP
    view = manager.view(TABLET)
    assert view["lease_seconds"] < 20  # quarantined: no longer renewed
    clock.now += view["lease_seconds"] + 0.1
    manager.tick()
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.ACTIVE
    assert manager.state_of(TABLET) == VoiceState.DISCONNECTED


def test_sleeping_endpoint_reporting_an_open_mic_blocks_the_next_handover():
    manager, _ = _manager()
    _tablet(manager, mic_open=True)  # never granted anything, yet its mic is open
    claim, view = manager.claim(DESKTOP_ENDPOINT, "active", "session", explicit=True)
    assert view["reason"] == "waiting_for_release"
    _ack(manager, TABLET)
    assert manager.state_of(DESKTOP_ENDPOINT) == VoiceState.ACTIVE


def test_every_change_bumps_the_version_and_every_view_agrees():
    manager, _ = _manager()
    seen: list[int] = []
    manager.add_listener(seen.append)
    _tablet(manager, want_standby=True)
    _phone(manager)
    manager.claim(PHONE, "priority", "reminder")
    views = [manager.view(e) for e in (DESKTOP_ENDPOINT, TABLET, PHONE)]
    assert len({v["version"] for v in views}) == 1
    assert len({str(v["owner"]) for v in views}) == 1
    assert {v["epoch"] for v in views} == {manager.epoch}
    assert seen == sorted(seen) and len(set(seen)) == len(seen)
    for view in views:
        assert sum(1 for e in view["endpoints"] if e["you"]) == 1


def test_simultaneous_claims_from_many_threads_never_grant_two_owners():
    manager, _ = _manager()
    endpoints = [f"dev{i}" for i in range(8)]
    for e in endpoints:
        manager.heartbeat(e, kind="phone", label=e)
    barrier = threading.Barrier(len(endpoints))
    violations: list[list[str]] = []

    def run(endpoint: str) -> None:
        barrier.wait()
        rng = random.Random(endpoint)
        for _ in range(200):
            try:
                if rng.random() < 0.5:
                    claim, _ = manager.claim(endpoint, rng.choice(["active", "priority"]), "x", explicit=True)
                    manager.release(endpoint, claim.id)
                else:
                    view = manager.view(endpoint)
                    manager.heartbeat(
                        endpoint,
                        kind="phone",
                        label=endpoint,
                        mic_open=False,
                        seen_version=view["version"],
                        seen_epoch=view["epoch"],
                    )
            except VoiceClaimRefused:
                pass
            granted = [e for e in endpoints if manager.state_of(e) in LISTENING_STATES]
            if len(granted) > 1:
                violations.append(granted)

    threads = [threading.Thread(target=run, args=(e,)) for e in endpoints]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert violations == []


def test_random_operation_sequences_keep_at_most_one_granted_microphone():
    rng = random.Random(7)
    for _ in range(40):
        manager, clock = _manager()
        claims: dict[str, list[str]] = {}
        for _ in range(150):
            endpoint = rng.choice([DESKTOP_ENDPOINT, TABLET, PHONE, "tab2"])
            op = rng.random()
            try:
                if op < 0.2:
                    if endpoint == DESKTOP_ENDPOINT:
                        manager.set_wants_standby(endpoint, rng.random() < 0.7)
                    else:
                        manager.heartbeat(endpoint, kind="station", label=endpoint, want_standby=rng.random() < 0.7)
                elif op < 0.4:
                    claim, _ = manager.claim(
                        endpoint,
                        rng.choice(["active", "priority"]),
                        "x",
                        explicit=rng.random() < 0.5,
                        kind=None if endpoint == DESKTOP_ENDPOINT else "phone",
                        label=endpoint,
                    )
                    claims.setdefault(endpoint, []).append(claim.id)
                elif op < 0.55 and claims.get(endpoint):
                    manager.release(endpoint, claims[endpoint].pop())
                elif op < 0.8:
                    _ack(manager, endpoint) if endpoint in (DESKTOP_ENDPOINT, TABLET, PHONE) else None
                elif op < 0.9:
                    clock.now += rng.choice([1, 6, 25])
                    manager.tick()
                elif endpoint != DESKTOP_ENDPOINT:
                    manager.leave(endpoint)
            except VoiceClaimRefused:
                pass
            granted = [e for e in (DESKTOP_ENDPOINT, TABLET, PHONE, "tab2") if manager.state_of(e) in LISTENING_STATES]
            assert len(granted) <= 1, granted
            if granted:
                # The granted one is the only one whose microphone may be open.
                assert set(manager.listening_endpoints()) <= set(granted), manager.listening_endpoints()


async def test_long_poll_wakes_on_change_and_times_out_otherwise():
    manager, _ = _manager()
    manager.bind(asyncio.get_running_loop(), tick_seconds=0.05)
    version = manager.version
    waiter = asyncio.create_task(manager.wait(version, 2.0))
    await asyncio.sleep(0.02)
    assert not waiter.done()
    _tablet(manager, want_standby=True)
    await asyncio.wait_for(waiter, 1.0)
    started = asyncio.get_running_loop().time()
    await manager.wait(manager.version, 0.1)
    assert asyncio.get_running_loop().time() - started >= 0.09


async def test_wait_for_returns_the_view_once_the_grant_arrives():
    manager, _ = _manager()
    manager.bind(asyncio.get_running_loop(), tick_seconds=0.05)
    _tablet(manager, want_standby=True)
    manager.claim(DESKTOP_ENDPOINT, "active", "session", explicit=True)
    waiter = asyncio.create_task(manager.wait_for(DESKTOP_ENDPOINT, lambda v: v["state"] == "active", 2.0))
    await asyncio.sleep(0.05)
    assert not waiter.done()
    _ack(manager, TABLET)
    view = await asyncio.wait_for(waiter, 1.0)
    assert view["state"] == "active"
    assert await manager.wait_for(TABLET, lambda v: v["state"] == "standby", 0.05) is None
