"""Voice ownership: which ONE Garvis endpoint may use a microphone right now.

Garvis has several voice endpoints -- the Desktop's own microphone, Garvis
Mobile (the phone app) and Eye Station (the tablet). Left alone, each would
listen for "Hey Garvis" by itself and they would all answer. Garvis Desktop,
which every client already talks to, is the single authority: this module
decides which endpoint may hold a microphone, and every other endpoint
sleeps -- microphone released, no wake phrase, no ambient sound processing,
no voice replies.

States (per endpoint):

  STANDBY       owns passive wake-phrase listening
  ACTIVE        owns an ongoing voice conversation
  PRIORITY      temporarily owns the voice for a reminder or a Garvis call
  SLEEP         must not listen
  DISCONNECTED  hasn't been heard from within its lease

Rules:

- Standby queue. An endpoint that is open and can listen for the wake phrase
  asks for standby (`want_standby`). The first one gets it; later ones wait in
  arrival order, asleep. The owner keeps standby until it leaves (closes,
  unticks, disconnects); then the next one in line gets it. A reconnecting
  endpoint joins at the back, so it never steals standby. Nothing claims
  standby by merely starting: the Desktop asks only when the user ticks
  "Wake word standby".
- Claims are temporary and never move standby. ACTIVE is a conversation,
  PRIORITY a reminder or a call. When the last claim ends, the standby owner
  (the same one, unless it left meanwhile) listens again. Priority beats a
  conversation; a conversation the user explicitly starts on a device (a
  button, not the wake phrase) beats another device's conversation; a
  wake-phrase conversation can only come from the endpoint that holds the
  microphone (only it was listening).
- Hand-over. The new holder isn't granted the microphone until every other
  endpoint has confirmed its own is closed: it reports `mic_open=False`
  having seen the version that told it to sleep. Lease expiry also counts,
  because a client must stop listening by itself when its lease runs out
  unrenewed. So a vanished device never keeps ownership, and two microphones
  are never granted at once. A client that is told to sleep but still
  reports an open microphone after `release_timeout_seconds` is no longer
  renewed: its lease runs out and the hand-over goes ahead.
- Every change gets a new `version` (with the server's `epoch`). Clients
  long-poll for it and ignore anything older than what they've already seen.

Thread-safe: the Desktop's UI thread, its asyncio loop (the /api/v1 handlers)
and tests all call in.
"""
from __future__ import annotations

import asyncio
import itertools
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.logging import get_logger, log_extra

logger = get_logger("voice.ownership")

DESKTOP_ENDPOINT = "desktop"
"""The Desktop's own microphone (a local endpoint; paired devices use their device id)."""


class VoiceState(StrEnum):
    STANDBY = "standby"
    ACTIVE = "active"
    PRIORITY = "priority"
    SLEEP = "sleep"
    DISCONNECTED = "disconnected"


LISTENING_STATES = frozenset({VoiceState.STANDBY, VoiceState.ACTIVE, VoiceState.PRIORITY})
"""States in which an endpoint may have its microphone open."""

CLAIM_MODES = ("active", "priority")


class VoiceClaimRefused(Exception):
    """A conversation can't start here now. `code`: "priority_elsewhere" (a reminder/call holds
    the voice on another device), "not_owner" (a wake phrase from a device that wasn't the one
    listening) or "invalid"."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class VoiceClaim:
    id: str
    endpoint: str
    mode: str
    reason: str
    ref: str
    seq: int
    expires_at: float | None

    def public(self) -> dict:
        return {"id": self.id, "mode": self.mode, "reason": self.reason, "ref": self.ref}


@dataclass
class _Endpoint:
    id: str
    kind: str
    label: str
    local: bool
    connected: bool = True
    lease_until: float = 0.0
    wants_standby: bool = False
    queue_seq: int | None = None
    state: VoiceState = VoiceState.SLEEP
    reason: str = "no_owner"
    mic_may_be_open: bool = False
    """Pessimistic: True from the moment it may listen until it confirms (having seen the
    order to sleep) that its microphone is closed, or its lease runs out."""
    reported_mic: bool = False
    sleep_version: int = 0
    """The version that last told it to stop listening; only a report that has seen it counts."""
    sleep_since: float | None = None
    quarantined: bool = False
    """Told to sleep, still not released after the release timeout: its lease isn't renewed."""
    extra: dict = field(default_factory=dict)


class VoiceOwnershipManager:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        lease_seconds: float = 20.0,
        release_timeout_seconds: float = 5.0,
        max_priority_seconds: float = 1800.0,
    ) -> None:
        self._clock = clock
        self.lease_seconds = lease_seconds
        self.release_timeout_seconds = release_timeout_seconds
        self.max_priority_seconds = max_priority_seconds
        self.epoch = secrets.token_hex(4)
        self._lock = threading.RLock()
        self._version = 0
        self._endpoints: dict[str, _Endpoint] = {}
        self._claims: dict[str, VoiceClaim] = {}
        self._standby_owner: str | None = None
        self._holder: str | None = None
        """The endpoint currently granted the microphone (None during a hand-over or with no owner)."""
        self._seq = itertools.count(1)
        self._listeners: list[Callable[[int], None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._changed: asyncio.Event | None = None
        self._tick_handle: asyncio.TimerHandle | None = None

    # ------------------------------------------------------------ wiring
    def bind(self, loop: asyncio.AbstractEventLoop, tick_seconds: float = 1.0) -> None:
        """The loop long-polls wait on; leases and timeouts are checked on it every `tick_seconds`."""
        self._loop = loop

        def _tick() -> None:
            self.tick()
            self._tick_handle = loop.call_later(tick_seconds, _tick)

        try:
            loop.call_soon_threadsafe(_tick)
        except RuntimeError:
            pass

    @property
    def bound(self) -> bool:
        return self._loop is not None

    def add_listener(self, listener: Callable[[int], None]) -> None:
        """Called with the new version after every change, on the thread that made it. Must be quick."""
        self._listeners.append(listener)

    @property
    def version(self) -> int:
        return self._version

    # ---------------------------------------------------------- endpoints
    def register_local(self, endpoint_id: str, *, kind: str, label: str) -> None:
        """An in-process endpoint (the Desktop's own microphone): always connected, no lease."""
        with self._lock:
            if endpoint_id not in self._endpoints:
                self._endpoints[endpoint_id] = _Endpoint(endpoint_id, kind, label, local=True)
            changed = self._resolve()
        self._publish(changed)

    def heartbeat(
        self,
        endpoint_id: str,
        *,
        kind: str,
        label: str,
        want_standby: bool | None = None,
        mic_open: bool | None = None,
        seen_version: int = 0,
        seen_epoch: str = "",
    ) -> dict:
        """A remote endpoint checking in: renews its lease, says whether it wants standby and
        whether its microphone is open. Returns its view."""
        with self._lock:
            endpoint = self._touch(endpoint_id, kind, label)
            if want_standby is not None:
                self._set_wants(endpoint, want_standby)
            if mic_open is not None:
                self._report_mic(endpoint, mic_open, seen_version if seen_epoch == self.epoch else 0)
            changed = self._resolve()
            view = self._view(endpoint)
        self._publish(changed)
        return view

    def set_wants_standby(self, endpoint_id: str, wants: bool) -> None:
        with self._lock:
            endpoint = self._endpoints.get(endpoint_id)
            if endpoint is None:
                return
            self._set_wants(endpoint, wants)
            changed = self._resolve()
        self._publish(changed)

    def report_mic(self, endpoint_id: str, mic_open: bool, seen_version: int) -> None:
        """A local endpoint's microphone state, after it acted on `seen_version`."""
        with self._lock:
            endpoint = self._endpoints.get(endpoint_id)
            if endpoint is None:
                return
            self._report_mic(endpoint, mic_open, seen_version)
            changed = self._resolve()
        self._publish(changed)

    def leave(self, endpoint_id: str) -> None:
        """A remote endpoint closed (its microphone already released): it gives up standby and
        its claims at once instead of waiting for its lease to run out."""
        with self._lock:
            endpoint = self._endpoints.get(endpoint_id)
            if endpoint is None or endpoint.local:
                return
            self._drop(endpoint, "left")
            changed = self._resolve()
        self._publish(changed)

    # ------------------------------------------------------------- claims
    def claim(
        self,
        endpoint_id: str,
        mode: str,
        reason: str = "",
        *,
        ref: str = "",
        explicit: bool = False,
        max_seconds: float | None = None,
        kind: str | None = None,
        label: str | None = None,
    ) -> tuple[VoiceClaim, dict]:
        """Ask for the voice: `mode` "active" (a conversation) or "priority" (a reminder or a
        call). Raises VoiceClaimRefused. The claim may still be waiting for the previous holder
        to release its microphone: the endpoint may listen only once its state says so."""
        if mode not in CLAIM_MODES:
            raise VoiceClaimRefused(f"Unknown claim mode {mode!r}.", "invalid")
        try:
            claim, view = self._claim(endpoint_id, mode, reason, ref, explicit, max_seconds, kind, label)
        except VoiceClaimRefused as exc:
            self._publish(getattr(exc, "changed", False))
            raise
        self._publish(True)
        return claim, view

    def _claim(self, endpoint_id, mode, reason, ref, explicit, max_seconds, kind, label) -> tuple[VoiceClaim, dict]:
        with self._lock:
            endpoint = self._endpoints.get(endpoint_id)
            if kind is not None:
                endpoint = self._touch(endpoint_id, kind, label or kind)
            if endpoint is None:
                raise VoiceClaimRefused("Unknown voice endpoint.", "invalid")
            try:
                return self._claim_locked(endpoint, mode, reason, ref, explicit, max_seconds)
            except VoiceClaimRefused as exc:
                exc.changed = True  # its lease was renewed / states re-resolved: let waiters look again
                raise

    def _claim_locked(self, endpoint: _Endpoint, mode, reason, ref, explicit, max_seconds) -> tuple[VoiceClaim, dict]:
        endpoint_id = endpoint.id
        self._resolve()
        now = self._clock()
        if mode == "active":
            foreign_priority = [c for c in self._claims.values()
                                if c.mode == "priority" and c.endpoint != endpoint_id]
            if foreign_priority:
                holder = self._endpoints.get(foreign_priority[0].endpoint)
                raise VoiceClaimRefused(
                    f"Garvis is busy with a {foreign_priority[0].reason or 'priority task'} on "
                    f"{holder.label if holder else 'another device'}.", "priority_elsewhere")
            if not explicit and endpoint.state not in LISTENING_STATES:
                raise VoiceClaimRefused("This device isn't the one listening for Garvis right now.", "not_owner")
            for other in [c for c in self._claims.values() if c.mode == "active"]:
                # One conversation at a time: a new one (here, or explicitly started elsewhere) replaces it.
                log_extra(logger, 20, "voice.claim_replaced", claim=other.id, endpoint=other.endpoint,
                          by=endpoint_id)
                del self._claims[other.id]
            expires_at = None
        else:
            for other in [c for c in self._claims.values() if c.mode == "active" and c.endpoint != endpoint_id]:
                log_extra(logger, 20, "voice.conversation_preempted", claim=other.id, endpoint=other.endpoint,
                          by=endpoint_id, reason=reason)
                del self._claims[other.id]
            limit = self.max_priority_seconds if max_seconds is None else min(max_seconds, self.max_priority_seconds)
            expires_at = now + max(1.0, limit)
        claim = VoiceClaim(secrets.token_hex(8), endpoint_id, mode, reason[:32], ref[:64], next(self._seq),
                           expires_at)
        self._claims[claim.id] = claim
        log_extra(logger, 20, "voice.claimed", endpoint=endpoint_id, kind=endpoint.kind, mode=mode,
                  reason=claim.reason, claim=claim.id, explicit=explicit)
        self._resolve(force=True)
        return claim, self._view(endpoint)

    def release(self, endpoint_id: str, claim_id: str) -> bool:
        """End a claim (only its own endpoint can). False if it was already gone (ended, preempted,
        expired) -- releasing twice is harmless."""
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim is None or claim.endpoint != endpoint_id:
                return False
            del self._claims[claim_id]
            log_extra(logger, 20, "voice.released", endpoint=endpoint_id, mode=claim.mode, reason=claim.reason,
                      claim=claim_id)
            changed = self._resolve(force=True)
        self._publish(changed)
        return True

    # -------------------------------------------------------------- reads
    def view(self, endpoint_id: str) -> dict:
        with self._lock:
            changed = self._resolve()
            endpoint = self._endpoints.get(endpoint_id)
            view = self._view(endpoint) if endpoint is not None else self._unknown_view(endpoint_id)
        self._publish(changed)
        return view

    def state_of(self, endpoint_id: str) -> VoiceState:
        with self._lock:
            endpoint = self._endpoints.get(endpoint_id)
            return endpoint.state if endpoint is not None else VoiceState.DISCONNECTED

    def holder(self) -> str | None:
        """The endpoint granted the microphone right now, if any."""
        with self._lock:
            return self._holder

    def listening_endpoints(self) -> list[str]:
        """Endpoints that may have a microphone open (granted, or not yet confirmed closed)."""
        with self._lock:
            return [e.id for e in self._endpoints.values() if e.mic_may_be_open]

    def tick(self) -> None:
        """Expire leases, release timeouts and priority caps (the bound loop calls this every second)."""
        with self._lock:
            changed = self._resolve()
        self._publish(changed)

    async def wait(self, after: int, timeout: float) -> None:
        """Return as soon as the version is past `after` (or at the timeout). Loop only."""
        deadline = time.monotonic() + max(0.0, timeout)
        while self._version <= after:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._changed is None:
                self._changed = asyncio.Event()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                self.tick()

    async def wait_for(self, endpoint_id: str, predicate: Callable[[dict], bool], timeout: float) -> dict | None:
        """The endpoint's view once `predicate(view)` holds, else None at the timeout. Loop only."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            view = self.view(endpoint_id)
            if predicate(view):
                return view
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await self.wait(view["version"], min(remaining, 1.0))

    # ----------------------------------------------------------- internal
    def _touch(self, endpoint_id: str, kind: str, label: str) -> _Endpoint:
        now = self._clock()
        endpoint = self._endpoints.get(endpoint_id)
        if endpoint is None:
            endpoint = _Endpoint(endpoint_id, kind, label, local=False, lease_until=now + self.lease_seconds)
            self._endpoints[endpoint_id] = endpoint
            log_extra(logger, 20, "voice.endpoint_connected", endpoint=endpoint_id, kind=kind)
            return endpoint
        endpoint.kind, endpoint.label = kind, label
        if endpoint.local:
            return endpoint
        if not endpoint.connected:
            endpoint.connected = True
            log_extra(logger, 20, "voice.endpoint_reconnected", endpoint=endpoint_id, kind=kind)
        if not endpoint.quarantined:
            endpoint.lease_until = now + self.lease_seconds
        return endpoint

    def _set_wants(self, endpoint: _Endpoint, wants: bool) -> None:
        if wants == endpoint.wants_standby:
            return
        endpoint.wants_standby = wants
        endpoint.queue_seq = next(self._seq) if wants else None
        log_extra(logger, 20, "voice.standby_wanted" if wants else "voice.standby_unwanted",
                  endpoint=endpoint.id, kind=endpoint.kind)

    def _report_mic(self, endpoint: _Endpoint, mic_open: bool, seen_version: int) -> None:
        endpoint.reported_mic = mic_open
        if mic_open:
            if endpoint.state not in LISTENING_STATES and not endpoint.mic_may_be_open:
                log_extra(logger, 30, "voice.unexpected_microphone", endpoint=endpoint.id, kind=endpoint.kind,
                          state=endpoint.state.value)
                endpoint.sleep_since = endpoint.sleep_since or self._clock()
            endpoint.mic_may_be_open = True
            return
        if endpoint.state in LISTENING_STATES:
            return  # it may open it again at any moment: still counts as listening
        if seen_version >= endpoint.sleep_version and endpoint.mic_may_be_open:
            endpoint.mic_may_be_open = False
            endpoint.sleep_since = None
            endpoint.quarantined = False
            log_extra(logger, 20, "voice.microphone_released", endpoint=endpoint.id, kind=endpoint.kind)

    def _drop(self, endpoint: _Endpoint, why: str) -> None:
        endpoint.connected = False
        endpoint.wants_standby = False
        endpoint.queue_seq = None
        endpoint.mic_may_be_open = False  # left (mic closed first) or its lease ran out (it stopped by itself)
        endpoint.sleep_since = None
        endpoint.quarantined = False
        for claim_id in [c.id for c in self._claims.values() if c.endpoint == endpoint.id]:
            del self._claims[claim_id]
        log_extra(logger, 20, "voice.endpoint_disconnected", endpoint=endpoint.id, kind=endpoint.kind, why=why)

    def _resolve(self, force: bool = False) -> bool:
        """Recompute every endpoint's state. Caller holds the lock. True if anything changed."""
        now = self._clock()
        for endpoint in self._endpoints.values():
            if not endpoint.local and endpoint.connected and now >= endpoint.lease_until:
                self._drop(endpoint, "lease_expired")
            if endpoint.mic_may_be_open and endpoint.state not in LISTENING_STATES and endpoint.sleep_since is not None:
                waited = now - endpoint.sleep_since
                if not endpoint.local and not endpoint.quarantined and waited >= self.release_timeout_seconds:
                    endpoint.quarantined = True  # no more renewals: its lease runs out, then it must be quiet
                    log_extra(logger, 30, "voice.release_timeout", endpoint=endpoint.id, kind=endpoint.kind)
                if waited >= self.release_timeout_seconds + self.lease_seconds:
                    endpoint.mic_may_be_open = False
                    endpoint.sleep_since = None
                    log_extra(logger, 40, "voice.release_forced", endpoint=endpoint.id, kind=endpoint.kind)
        for claim in [c for c in self._claims.values() if c.expires_at is not None and now >= c.expires_at]:
            del self._claims[claim.id]
            log_extra(logger, 30, "voice.claim_expired", endpoint=claim.endpoint, mode=claim.mode,
                      reason=claim.reason)

        eligible = [e for e in self._endpoints.values() if e.connected and e.wants_standby]
        standby_owner = self._standby_owner
        if standby_owner not in {e.id for e in eligible}:
            standby_owner = min(eligible, key=lambda e: e.queue_seq or 0).id if eligible else None
            if standby_owner != self._standby_owner:
                log_extra(logger, 20, "voice.standby_owner", endpoint=standby_owner or "",
                          previous=self._standby_owner or "")

        priorities = sorted((c for c in self._claims.values() if c.mode == "priority"), key=lambda c: c.seq)
        actives = [c for c in self._claims.values() if c.mode == "active"]
        if priorities:
            target, target_state = priorities[0].endpoint, VoiceState.PRIORITY
        elif actives:
            target, target_state = actives[0].endpoint, VoiceState.ACTIVE
        elif standby_owner is not None:
            target, target_state = standby_owner, VoiceState.STANDBY
        else:
            target, target_state = None, VoiceState.SLEEP
        blocked = target is not None and any(e.mic_may_be_open for e in self._endpoints.values() if e.id != target)

        assigned: dict[str, tuple[VoiceState, str]] = {}
        for endpoint in self._endpoints.values():
            if not endpoint.connected:
                assigned[endpoint.id] = (VoiceState.DISCONNECTED, "")
            elif endpoint.id == target:
                assigned[endpoint.id] = (VoiceState.SLEEP, "waiting_for_release") if blocked else (target_state, "")
            elif target is None:
                assigned[endpoint.id] = (VoiceState.SLEEP, "no_owner")
            else:
                assigned[endpoint.id] = (VoiceState.SLEEP, "in_use")
        holder = None if blocked else target

        changed = force or holder != self._holder or standby_owner != self._standby_owner or any(
            (e.state, e.reason) != assigned[e.id] for e in self._endpoints.values())
        if not changed:
            return False
        self._version += 1
        for endpoint in self._endpoints.values():
            state, reason = assigned[endpoint.id]
            was_listening = endpoint.state in LISTENING_STATES
            if state in LISTENING_STATES:
                endpoint.mic_may_be_open = True
                endpoint.sleep_since = None
            elif was_listening:
                endpoint.sleep_version = self._version
                endpoint.sleep_since = now if endpoint.mic_may_be_open else None
            if state != endpoint.state:
                log_extra(logger, 20, "voice.state", endpoint=endpoint.id, kind=endpoint.kind, state=state.value,
                          previous=endpoint.state.value, version=self._version)
            endpoint.state, endpoint.reason = state, reason
        self._standby_owner = standby_owner
        self._holder = holder
        return True

    def _public(self, endpoint: _Endpoint | None) -> dict | None:
        if endpoint is None:
            return None
        return {"kind": endpoint.kind, "label": endpoint.label, "state": endpoint.state.value}

    def _view(self, endpoint: _Endpoint) -> dict:
        now = self._clock()
        return {
            "epoch": self.epoch,
            "version": self._version,
            "state": endpoint.state.value,
            "reason": endpoint.reason,
            "listen": endpoint.state in LISTENING_STATES,
            "lease_seconds": None if endpoint.local else round(max(0.0, endpoint.lease_until - now), 3),
            "claims": [c.public() for c in self._claims.values() if c.endpoint == endpoint.id],
            "wants_standby": endpoint.wants_standby,
            "owner": self._public(self._endpoints.get(self._holder)) if self._holder else None,
            "standby_owner": self._public(self._endpoints.get(self._standby_owner)) if self._standby_owner else None,
            "endpoints": [{**self._public(e), "you": e is endpoint}
                          for e in self._endpoints.values() if e.connected],
        }

    def _unknown_view(self, endpoint_id: str) -> dict:
        return {"epoch": self.epoch, "version": self._version, "state": VoiceState.DISCONNECTED.value,
                "reason": "", "listen": False, "lease_seconds": 0.0, "claims": [], "wants_standby": False,
                "owner": self._public(self._endpoints.get(self._holder)) if self._holder else None,
                "standby_owner": None, "endpoints": []}

    def _publish(self, changed: bool) -> None:
        if not changed:
            return
        version = self._version
        for listener in list(self._listeners):
            try:
                listener(version)
            except Exception:  # noqa: BLE001 -- a listener must never break ownership
                logger.exception("voice.listener_failed")
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._wake)
            except RuntimeError:  # loop closed
                pass

    def _wake(self) -> None:
        event, self._changed = self._changed, asyncio.Event()
        if event is not None:
            event.set()
