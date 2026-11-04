"""/api/v1 -- the API of the native Garvis Android app.

Mounted on the same HTTPS server as the iPhone web app, behind the same
`_guard` middleware (allowed networks, Host check, security headers). The
differences from the browser API:

- Authentication is `Authorization: Bearer <device token>`. A browser never
  attaches that header by itself, so CSRF is not a concern here; cookies are
  ignored on these routes. The token comes from POST /api/v1/pair (an 8-digit
  code shown on the PC, single use, burnt after 5 wrong tries) and is stored
  on the phone in the Android Keystore; the PC keeps only its SHA-256.
- The phone is the user's own trusted client, but it still only *asks*:
  instructions become TaskManager tasks, protocol runs are re-validated on
  the PC, confirmations go through the ConfirmationManager, phone actions
  through the PhoneActionBroker. No route runs commands, reads files or
  changes settings.
- Every write takes a client `request_id`, so a retry after a dropped
  connection never does anything twice.
- Over Tailscale (GARVIS_MOBILE_REMOTE_ACCESS=tailscale) only these routes
  are reachable (see server.py).
- Garvis Eye Station (the tablet face, a separate app) pairs here too, with
  `"client": "eye_station"`. Its token is a *station* token: it opens only
  the STATION_ROUTES a display/voice client needs (presence, a voice turn
  and its spoken reply, status, token upkeep), and a station is never seen
  as a phone -- the PhoneActionBroker and reminder ringing ignore it.
- /api/v1/voice/* is voice ownership (app/core/voice_ownership.py): the PC
  decides which ONE endpoint may hold a microphone. A client heartbeats
  (long-poll) to keep its lease, says whether it wants wake-phrase standby
  and whether its microphone is open, and claims the voice for a
  conversation (active) or a reminder/call (priority, phones only). It may
  listen only while its returned state says `listen`, and must stop by
  itself when its lease runs out unrenewed.
- Failed authentication (an unknown/expired token, a refused pairing code)
  is logged without the token and counted per client address
  (AuthFailureLimiter): too many in a few minutes and that address gets 429
  until they age out.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web

from app.core.exceptions import SpeechToTextError, TextToSpeechError
from app.core.logging import get_logger, log_extra
from app.core.presence import PresenceState
from app.core.voice_ownership import CLAIM_MODES, VoiceClaimRefused
from app.images.service import ImageError
from app.mobile.companion import Companion
from app.mobile.pairing import DeviceRecord, PairingError, PairingManager
from app.mobile.phone_actions import CAPABILITIES
from app.mobile.service import AudioValidationError, MobileBusyError, MobileService, NoReplyError
from app.reminders.phrasing import language
from app.reminders.schedule import ScheduleError
from app.reminders.store import ReminderError

logger = get_logger("mobile.native_api")

API_VERSION = 1
_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_CONFIRMATION_ID = re.compile(r"^[0-9a-fA-F-]{36}$")
_PROTOCOL_ID = re.compile(r"^[a-f0-9]{12,32}$")
_ACTION_ID = re.compile(r"^[0-9a-f]{16}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_TZ = re.compile(r"^[A-Za-z0-9_+:/-]{1,64}$")
_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}
_AUDIO_ERROR_STATUS = {"empty": 422, "invalid": 400, "unsupported": 415, "too_long": 413}
_MAX_JSON_BYTES = 64 * 1024
LONG_POLL_MAX_SECONDS = 25.0
VOICE_POLL_MAX_SECONDS = 8.0
"""Well inside the voice lease (20 s), so a waiting heartbeat never lets it lapse."""
_CLAIM_REASON = re.compile(r"^[a-z_]{1,32}$")
STATION_CLIENT = "eye_station"
STATION_LABEL = "Eye Station"
AUTH_FAILURE_LIMIT = 10
AUTH_FAILURE_WINDOW_SECONDS = 300.0


class AuthFailureLimiter:
    """Failed sign-ins per client address. After `limit` failures within
    `window` seconds the address is refused (429) until the oldest one ages
    out, so guessing tokens or pairing codes crawls. A phone whose pairing
    was revoked stops after its first 401 and is never affected; a success
    clears the address. Bounded: only the most recent addresses are kept."""

    MAX_ADDRESSES = 1024

    def __init__(self, limit: int = AUTH_FAILURE_LIMIT, window: float = AUTH_FAILURE_WINDOW_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._failures: dict[str, deque[float]] = {}

    def _recent(self, address: str) -> deque[float] | None:
        failures = self._failures.get(address)
        if failures is not None:
            cutoff = self._clock() - self._window
            while failures and failures[0] <= cutoff:
                failures.popleft()
            if not failures:
                del self._failures[address]
                return None
        return failures

    def retry_after(self, address: str) -> float:
        """Seconds until `address` may try again (0: it may now)."""
        failures = self._recent(address)
        if failures is None or len(failures) < self._limit:
            return 0.0
        return max(1.0, failures[0] + self._window - self._clock())

    def failed(self, address: str) -> None:
        failures = self._recent(address)
        if failures is None:
            if len(self._failures) >= self.MAX_ADDRESSES:
                del self._failures[next(iter(self._failures))]
            failures = self._failures[address] = deque(maxlen=self._limit)
        failures.append(self._clock())

    def succeeded(self, address: str) -> None:
        self._failures.pop(address, None)


def _error(status: int, message: str, code: str = "") -> web.Response:
    return web.json_response({"error": message, "code": code or str(status)}, status=status)


def _too_many(wait: float) -> web.Response:
    response = _error(429, "Too many failed sign-in attempts from this device. Try again in a few minutes.",
                      "rate_limited")
    response.headers["Retry-After"] = str(int(wait) + 1)
    return response


def _bearer(request: web.Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if header[:7].lower() != "bearer ":
        return None
    token = header[7:].strip()
    return token or None


def register_native_routes(
    app: web.Application,
    *,
    service: MobileService,
    pairing: PairingManager,
    companion: Companion,
    max_audio_bytes: int,
    is_remote: Callable[[web.Request], bool] = lambda request: False,
    remote_urls: Callable[[web.Request], Awaitable[list[str]]] | None = None,
    limiter: AuthFailureLimiter | None = None,
) -> None:
    tasks = service.tasks
    limiter = limiter or AuthFailureLimiter()

    def auth_failed(request: web.Request, reason: str) -> None:
        """Count and log a refused sign-in: where from and why -- never the token or code."""
        limiter.failed(request.remote or "")
        log_extra(logger, 30, "mobile.native.auth_failed", remote=request.remote, path=request.path,
                  reason=reason, via="tailscale" if is_remote(request) else "lan")
    station_turns: dict[str, deque[str]] = {}
    """Station device id -> the turns (task ids) it started: a station reads only its own."""

    def station_may_see(device: DeviceRecord, task_id: str) -> bool:
        return device.kind != "station" or task_id in station_turns.get(device.id, ())

    def device_of(request: web.Request) -> DeviceRecord | None:
        record = pairing.verify(_bearer(request))
        if record is not None and record.kind == "phone":
            companion.phone.seen(record.id, record.label)
            zone = request.headers.get("X-Garvis-Timezone", "")
            if zone and _TZ.match(zone):
                companion.zone.report(zone)
        return record

    async def read_json(request: web.Request) -> dict[str, Any] | None:
        if request.content_type != "application/json":
            return None
        if (request.content_length or 0) > _MAX_JSON_BYTES:
            raise web.HTTPRequestEntityTooLarge(max_size=_MAX_JSON_BYTES, actual_size=request.content_length)
        try:
            data = await request.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def request_id_of(data: dict[str, Any]) -> str | None:
        value = data.get("request_id")
        if value is None:
            return None
        if not isinstance(value, str) or not _REQUEST_ID.match(value):
            raise web.HTTPBadRequest(reason="Invalid request id.")
        return value

    def authed(handler=None, *, stations: bool = False):
        """Needs a paired device's token; a station token only where `stations`."""
        def decorate(handler):
            async def wrapper(request: web.Request) -> web.StreamResponse:
                if (wait := limiter.retry_after(request.remote or "")) > 0:
                    return _too_many(wait)
                device = device_of(request)
                if device is None:
                    auth_failed(request, "no_token" if _bearer(request) is None else "unknown_token")
                    return _error(401, "This device is not paired with Garvis (or its pairing expired).", "unpaired")
                limiter.succeeded(request.remote or "")
                if device.kind == "station" and not stations:
                    return _error(403, "Eye Station can't use this part of Garvis.", "station_scope")
                return await handler(request, device)
            return wrapper
        return decorate(handler) if handler is not None else decorate

    def garvis_state() -> dict[str, Any]:
        current = tasks.current if tasks is not None else None
        state = "executing" if current is not None else ("busy" if service.is_busy else "online")
        return {
            "state": state,
            "status": service.status_text,
            "active_task": current.summary() if current is not None else None,
            "queued": tasks.queued_count() if tasks is not None else 0,
        }

    # -------------------------------------------------------------- pairing
    async def hello(request: web.Request) -> web.Response:
        # Lets the app check it reached a Garvis PC (TLS already verified)
        # before asking for a pairing code. Says nothing private.
        return web.json_response({"api": API_VERSION, "server": "garvis",
                                  "paired": pairing.verify(_bearer(request)) is not None,
                                  "remote": bool(is_remote(request))})

    async def pair(request: web.Request) -> web.Response:
        if (wait := limiter.retry_after(request.remote or "")) > 0:
            return _too_many(wait)
        data = await read_json(request)
        if data is None or not isinstance(data.get("code"), str):
            return _error(400, "Send the pairing code shown on the PC.")
        name = str(data.get("device_name") or "Android")[:40]
        station = data.get("client") == STATION_CLIENT
        try:
            token = pairing.redeem(data["code"], label=STATION_LABEL if station else "Android",
                                   kind="station" if station else "phone")
        except PairingError as exc:
            auth_failed(request, "pairing_refused")
            return _error(403, str(exc), "pairing_refused")
        limiter.succeeded(request.remote or "")
        record = pairing.verify(token)
        log_extra(logger, 20, "mobile.native.paired", device=record.id, kind=record.kind, name_length=len(name))
        return web.json_response({"token": token, "device_id": record.id, "expires_at": record.expires_at,
                                  "api": API_VERSION, "kind": record.kind}, status=201)

    @authed(stations=True)
    async def refresh(request: web.Request, device: DeviceRecord) -> web.Response:
        rotated = pairing.rotate(_bearer(request))
        if rotated is None:
            return _error(401, "This phone is not paired with Garvis.", "unpaired")
        token, record = rotated
        return web.json_response({"token": token, "device_id": record.id, "expires_at": record.expires_at})

    @authed(stations=True)
    async def unpair(request: web.Request, device: DeviceRecord) -> web.Response:
        pairing.revoke(_bearer(request))
        companion.phone.forget(device.id)
        companion.voice_ownership.leave(device.id)
        return web.json_response({"paired": False})

    # --------------------------------------------------------------- status
    @authed(stations=True)
    async def status(request: web.Request, device: DeviceRecord) -> web.Response:
        return web.json_response({
            "api": API_VERSION, "device_id": device.id, "kind": device.kind, "token_expires_at": device.expires_at,
            "server_time": time.time(), "time_zone": companion.zone.current(),
            "garvis": garvis_state(), "features": companion.features(), "remote": bool(is_remote(request)),
            # The phone saves these as its away-from-home address (no typing a 100.x address).
            "remote_urls": await remote_urls(request) if remote_urls is not None else [],
        })

    @authed
    async def events(request: web.Request, device: DeviceRecord) -> web.Response:
        try:
            after = max(0, int(request.query.get("after", "0")))
            wait = min(LONG_POLL_MAX_SECONDS, max(0.0, float(request.query.get("wait", "0"))))
            known_reminders = int(request.query.get("rv", "-1"))
            known_images = int(request.query.get("iv", "-1"))
            known_tasks = int(request.query.get("tv", "-1"))
        except ValueError:
            return _error(400, "Invalid query parameters.")
        epoch = request.query.get("epoch", "")
        known_actions = {a for a in request.query.get("actions", "").split(",") if a}
        if epoch and epoch != service.epoch:
            after = 0  # Garvis restarted: the client's cursor belongs to the old log

        def has_news() -> bool:
            return (after != service.last_seq or (epoch and epoch != service.epoch)
                    or known_reminders != companion.reminders.version
                    or known_images != companion.images.version
                    or (tasks is not None and known_tasks != tasks.version)
                    or {a["id"] for a in companion.phone.pending_for(device.id)} != known_actions)

        signal = service.change_signal()
        if wait > 0 and not has_news():
            try:
                await asyncio.wait_for(signal.wait(), timeout=wait)
            except TimeoutError:
                pass
        snapshot = service.snapshot(after)
        snapshot["reset"] = bool(snapshot["reset"] or (epoch and epoch != service.epoch))
        snapshot.update(
            reminders_version=companion.reminders.version,
            images_version=companion.images.version,
            phone_actions=companion.phone.pending_for(device.id),
            garvis=garvis_state(),
            server_time=time.time(),
        )
        return web.json_response(snapshot)

    # --------------------------------------------------------- conversation
    @authed
    async def messages(request: web.Request, device: DeviceRecord) -> web.Response:
        data = await read_json(request)
        if data is None or not isinstance(data.get("text"), str):
            return _error(400, "Expected JSON with a 'text' field.")
        try:
            turn = service.submit_text(data["text"], request_id=request_id_of(data), device=device.label)
        except ValueError as exc:
            return _error(400, str(exc))
        except MobileBusyError as exc:
            return _error(409, str(exc), "busy")
        task = tasks.get(turn.id) if tasks is not None else None
        return web.json_response({"turn_id": turn.id, "task": task.summary() if task else None}, status=202)

    @authed(stations=True)
    async def voice(request: web.Request, device: DeviceRecord) -> web.Response:
        if request.content_type not in _AUDIO_TYPES:
            return _error(415, "Unsupported audio format: send audio/wav.")
        if (request.content_length or 0) > max_audio_bytes:
            return _error(413, "The recording is too large.")
        request_id = request.headers.get("X-Request-Id") or None
        if request_id is not None and not _REQUEST_ID.match(request_id):
            return _error(400, "Invalid request id.")
        if request_id and tasks is not None:
            existing = next((t for t in tasks.list(200) if t.request_id == request_id), None)
            if existing is not None and station_may_see(device, existing.id):  # a retried upload: never run it twice
                return web.json_response({"turn_id": existing.id, "transcript": existing.text,
                                          "task": existing.summary()}, status=202)
        audio = await request.read()
        if len(audio) > max_audio_bytes:
            return _error(413, "The recording is too large.")
        try:
            transcript = await service.transcribe_recording(audio)
        except AudioValidationError as exc:
            return _error(_AUDIO_ERROR_STATUS.get(exc.kind, 400), str(exc), "audio_" + exc.kind)
        except MobileBusyError as exc:
            return _error(409, str(exc), "busy")
        except SpeechToTextError as exc:
            log_extra(logger, 30, "mobile.native.transcription_failed", error=str(exc))
            return _error(502, str(exc)[:300], "stt_failed")
        if not transcript:
            return _error(422, "No speech was recognized.", "no_speech")
        try:
            turn = service.submit_text(transcript, kind="voice", request_id=request_id, device=device.label)
        except ValueError as exc:
            return _error(400, str(exc))
        except MobileBusyError as exc:
            return _error(409, str(exc), "busy")
        if device.kind == "station":
            station_turns.setdefault(device.id, deque(maxlen=50)).append(turn.id)
        task = tasks.get(turn.id) if tasks is not None else None
        return web.json_response({"turn_id": turn.id, "transcript": transcript,
                                  "task": task.summary() if task else None}, status=202)

    @authed(stations=True)
    async def task_speech(request: web.Request, device: DeviceRecord) -> web.Response:
        task_id = request.match_info["task_id"]
        if not _ID.match(task_id) or not station_may_see(device, task_id):
            return _error(404, "Unknown task.")
        try:
            audio = await service.synthesize_reply(task_id)
        except NoReplyError as exc:
            return _error(404, str(exc), "no_reply")
        except TextToSpeechError as exc:
            return _error(502, str(exc)[:300], "tts_failed")
        return web.Response(body=audio, content_type="audio/wav")

    @authed
    async def confirm(request: web.Request, device: DeviceRecord) -> web.Response:
        request_id = request.match_info["request_id"]
        data = await read_json(request)
        if not _CONFIRMATION_ID.match(request_id) or data is None or not isinstance(data.get("approved"), bool):
            return _error(400, "Expected JSON with a boolean 'approved'.")
        if not service.resolve_confirmation(request_id, data["approved"]):
            return _error(404, "That request is no longer waiting for an answer.", "gone")
        return web.json_response({"ok": True})

    # ---------------------------------------------------------------- tasks
    @authed
    async def tasks_list(request: web.Request, device: DeviceRecord) -> web.Response:
        return web.json_response(service.list_tasks(limit=100))

    @authed(stations=True)
    async def task_detail(request: web.Request, device: DeviceRecord) -> web.Response:
        task_id = request.match_info["task_id"]
        detail = service.task_detail(task_id) if _ID.match(task_id) and station_may_see(device, task_id) else None
        return web.json_response(detail) if detail is not None else _error(404, "Unknown task.")

    @authed(stations=True)
    async def task_cancel(request: web.Request, device: DeviceRecord) -> web.Response:
        task_id = request.match_info["task_id"]
        if not _ID.match(task_id) or not station_may_see(device, task_id):
            return _error(404, "Unknown task.")
        try:
            return web.json_response(service.cancel_task(task_id), status=202)
        except LookupError as exc:
            return _error(404, str(exc))
        except ValueError as exc:
            return _error(409, str(exc), "not_cancellable")

    @authed
    async def protocols_list(request: web.Request, device: DeviceRecord) -> web.Response:
        return web.json_response({"protocols": service.list_protocols()})

    @authed
    async def protocol_run(request: web.Request, device: DeviceRecord) -> web.Response:
        protocol_id = request.match_info["protocol_id"]
        data = await read_json(request)
        if not _PROTOCOL_ID.match(protocol_id) or data is None:
            return _error(400, "Expected JSON for a known protocol.")
        try:
            return web.json_response(service.run_protocol(protocol_id, request_id_of(data), device.label),
                                     status=202)
        except LookupError as exc:
            return _error(404, str(exc))
        except ValueError as exc:
            return _error(409, str(exc), "refused")

    # ------------------------------------------------------------ reminders
    def reminder_payload() -> dict[str, Any]:
        return {"version": companion.reminders.version, "time_zone": companion.zone.current(),
                "reminders": [r.public() for r in companion.reminders.list()]}

    @authed
    async def reminders_list(request: web.Request, device: DeviceRecord) -> web.Response:
        return web.json_response(reminder_payload())

    def schedule_fields(data: dict[str, Any], *, partial: bool) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for key, kind in (("text", str), ("kind", str), ("repeat", str), ("local_time", str), ("tz", str)):
            if key in data and data[key] is not None:
                if not isinstance(data[key], kind):
                    raise ReminderError(f"'{key}' must be a {kind.__name__}.")
                fields[key] = data[key]
        if data.get("due_at") is not None:
            if not isinstance(data["due_at"], (int, float)):
                raise ReminderError("'due_at' must be epoch seconds.")
            fields["due_at"] = float(data["due_at"])
        if data.get("days") is not None:
            if not isinstance(data["days"], list):
                raise ReminderError("'days' must be a list.")
            fields["days"] = data["days"]
        if not partial:
            fields.setdefault("tz", companion.zone.current())
        return fields

    @authed
    async def reminder_create(request: web.Request, device: DeviceRecord) -> web.Response:
        data = await read_json(request)
        if data is None:
            return _error(400, "Expected JSON.")
        try:
            fields = schedule_fields(data, partial=False)
            text = fields.pop("text", "")
            reminder = companion.reminders.create(text, origin="phone", request_id=request_id_of(data), **fields)
        except (ReminderError, ScheduleError) as exc:
            # The phone shows the reason; the log keeps a trace (never the reminder's text).
            log_extra(logger, 30, "mobile.native.reminder_refused", device=device.id, reason=str(exc)[:200])
            return _error(400, str(exc), "invalid_reminder")
        return web.json_response({"reminder": reminder.public(), "version": companion.reminders.version}, status=201)

    def reminder_op(op: str):
        @authed
        async def handler(request: web.Request, device: DeviceRecord) -> web.Response:
            reminder_id = request.match_info["reminder_id"]
            if not _ID.match(reminder_id):
                return _error(404, "Unknown reminder.")
            data = await read_json(request) or {}
            store = companion.reminders
            try:
                if op == "update":
                    reminder = store.update(reminder_id, **schedule_fields(data, partial=True))
                elif op == "complete":
                    reminder = store.complete(reminder_id)
                elif op == "cancel":
                    reminder = store.cancel(reminder_id)
                elif op == "snooze":
                    minutes = data.get("minutes", 10)
                    if not isinstance(minutes, (int, float)):
                        return _error(400, "'minutes' must be a number.")
                    reminder = store.snooze(reminder_id, float(minutes))
                else:  # fired
                    occurrence = data.get("occurrence")
                    if not isinstance(occurrence, (int, float)):
                        return _error(400, "'occurrence' must be epoch seconds.")
                    rang_before = (store.get(reminder_id).fire_count if store.get(reminder_id) else 0)
                    reminder = store.fired(reminder_id, float(occurrence))
                    if reminder.fire_count > rang_before:
                        companion.presence.notify("reminder")
            except ReminderError as exc:
                log_extra(logger, 30, "mobile.native.reminder_refused", device=device.id, op=op, reason=str(exc)[:200])
                status = 404 if str(exc) == "Unknown reminder." else 400
                return _error(status, str(exc), "invalid_reminder")
            return web.json_response({"reminder": reminder.public(), "version": store.version})
        return handler

    @authed
    async def reminder_speech(request: web.Request, device: DeviceRecord) -> web.Response:
        reminder = companion.reminders.get(request.match_info["reminder_id"])
        if reminder is None:
            return _error(404, "Unknown reminder.")
        spoken = companion.spoken(reminder)
        lang = language(spoken)
        if lang != companion.pc_voice_language:
            # The PC's Piper voice can't say it (a Hebrew reminder, an English voice): the phone's own
            # voice does, from the same sentence -- which it also has from the reminder sync.
            return web.json_response({"error": "Garvis's voice on the PC doesn't speak this language.",
                                      "code": "voice_language", "spoken": spoken, "lang": lang}, status=415)
        try:
            audio = await service.synthesize_text(spoken)
        except TextToSpeechError as exc:
            return _error(502, str(exc)[:300], "tts_failed")
        return web.Response(body=audio, content_type="audio/wav")

    # ------------------------------------------------------------- presence
    @authed(stations=True)
    async def presence(request: web.Request, device: DeviceRecord) -> web.Response:
        """For display clients (Garvis Face): ?after=<version>&wait=<seconds> answers as soon as
        the state is newer than `after`, else at the wait."""
        try:
            after = int(request.query.get("after", "-1"))
            wait = min(LONG_POLL_MAX_SECONDS, max(0.0, float(request.query.get("wait", "0"))))
        except ValueError:
            return _error(400, "Invalid query parameters.")
        current = await companion.presence.wait(after, wait)
        return web.json_response({**current.public(), "states": [s.value for s in PresenceState]})

    # ------------------------------------------------------ voice ownership
    ownership = companion.voice_ownership

    def endpoint_label(device: DeviceRecord) -> str:
        return STATION_LABEL if device.kind == "station" else (device.label or "Phone")

    @authed(stations=True)
    async def voice_heartbeat(request: web.Request, device: DeviceRecord) -> web.Response:
        """{"want": "standby"|"none", "mic": bool, "epoch", "version": last seen, "wait": seconds}.
        Renews the lease, then answers as soon as anything changes after `version` (or at `wait`)."""
        data = await read_json(request)
        if data is None:
            return _error(400, "Expected a JSON body.")
        want, mic = data.get("want"), data.get("mic")
        if want not in (None, "standby", "none") or (mic is not None and not isinstance(mic, bool)):
            return _error(400, "'want' must be 'standby' or 'none'; 'mic' a boolean.")
        try:
            seen = max(0, int(data.get("version", 0)))
            wait = min(VOICE_POLL_MAX_SECONDS, max(0.0, float(data.get("wait", 0))))
        except (TypeError, ValueError):
            return _error(400, "Invalid 'version' or 'wait'.")
        epoch = str(data.get("epoch", ""))[:32]
        if not ownership.bound:  # normally AgentBridge binds it to this same loop at startup
            ownership.bind(asyncio.get_running_loop())
        view = ownership.heartbeat(device.id, kind=device.kind, label=endpoint_label(device),
                                   want_standby=None if want is None else want == "standby",
                                   mic_open=mic, seen_version=seen, seen_epoch=epoch)
        if wait > 0 and epoch == view["epoch"] and view["version"] <= seen:
            # `lease_seconds` stays the lease as renewed on arrival: the client counts it from
            # when it SENT this request, which is never later -- so it always stops first.
            lease = view["lease_seconds"]
            await ownership.wait(seen, wait)
            view = {**ownership.view(device.id), "lease_seconds": lease}
        return web.json_response(view)

    @authed(stations=True)
    async def voice_claim(request: web.Request, device: DeviceRecord) -> web.Response:
        """{"mode": "active"|"priority", "reason", "ref", "explicit", "max_seconds"}. 409 when refused."""
        data = await read_json(request)
        if data is None or data.get("mode") not in CLAIM_MODES:
            return _error(400, "Expected JSON with 'mode': 'active' or 'priority'.")
        reason, ref = data.get("reason", ""), data.get("ref", "")
        explicit, max_seconds = data.get("explicit", False), data.get("max_seconds")
        if (not isinstance(reason, str) or (reason and not _CLAIM_REASON.match(reason))
                or not isinstance(ref, str) or len(ref) > 64 or not isinstance(explicit, bool)
                or (max_seconds is not None and (isinstance(max_seconds, bool)
                                                 or not isinstance(max_seconds, (int, float)) or max_seconds <= 0))):
            return _error(400, "Invalid claim.")
        if data["mode"] == "priority" and device.kind == "station":
            # Reminders and calls ring on the phone; a station only ever converses.
            return _error(403, "Eye Station can't take priority.", "station_scope")
        try:
            claim, view = ownership.claim(device.id, data["mode"], reason, ref=ref, explicit=explicit,
                                          max_seconds=max_seconds, kind=device.kind, label=endpoint_label(device))
        except VoiceClaimRefused as exc:
            return web.json_response({"error": str(exc), "code": exc.code, "voice": ownership.view(device.id)},
                                     status=409)
        return web.json_response({"claim": claim.public(), "voice": view})

    @authed(stations=True)
    async def voice_release(request: web.Request, device: DeviceRecord) -> web.Response:
        data = await read_json(request)
        claim_id = data.get("claim_id") if data else None
        if not isinstance(claim_id, str) or not claim_id:
            return _error(400, "Expected JSON with 'claim_id'.")
        released = ownership.release(device.id, claim_id[:32])
        return web.json_response({"released": released, "voice": ownership.view(device.id)})

    @authed(stations=True)
    async def voice_leave(request: web.Request, device: DeviceRecord) -> web.Response:
        """The client closed (microphone already released): hand its standby on right away."""
        ownership.leave(device.id)
        return web.json_response({"voice": ownership.view(device.id)})

    # --------------------------------------------------------------- images
    @authed
    async def images_list(request: web.Request, device: DeviceRecord) -> web.Response:
        images = companion.images
        return web.json_response({"version": images.version, "available": images.available,
                                  "reason": images.unavailable_reason, "images": [j.public() for j in images.list()]})

    @authed
    async def image_create(request: web.Request, device: DeviceRecord) -> web.Response:
        data = await read_json(request)
        if data is None or not isinstance(data.get("prompt"), str):
            return _error(400, "Expected JSON with a 'prompt'.")
        try:
            job = companion.images.submit(data["prompt"], origin="mobile", request_id=request_id_of(data))
        except ImageError as exc:
            code = "unavailable" if not companion.images.available else "refused"
            return _error(503 if code == "unavailable" else 400, str(exc), code)
        return web.json_response({"image": job.public()}, status=202)

    @authed
    async def image_detail(request: web.Request, device: DeviceRecord) -> web.Response:
        job = companion.images.get(request.match_info["image_id"])
        return web.json_response({"image": job.public()}) if job is not None else _error(404, "Unknown image.")

    @authed
    async def image_file(request: web.Request, device: DeviceRecord) -> web.StreamResponse:
        image_id = request.match_info["image_id"]
        path = companion.images.file_path(image_id) if _ID.match(image_id) else None
        if path is None:
            return _error(404, "That image isn't available.")
        job = companion.images.get(image_id)
        body = await asyncio.to_thread(path.read_bytes)
        return web.Response(body=body, content_type=job.mime or "application/octet-stream")

    def image_op(op: str):
        @authed
        async def handler(request: web.Request, device: DeviceRecord) -> web.Response:
            image_id = request.match_info["image_id"]
            try:
                if op == "cancel":
                    job = companion.images.cancel(image_id)
                    return web.json_response({"image": job.public()})
                companion.images.delete(image_id)
                return web.json_response({"deleted": True})
            except ImageError as exc:
                return _error(404 if "Unknown" in str(exc) else 409, str(exc))
        return handler

    # ---------------------------------------------------------- phone actions
    @authed
    async def phone_capabilities(request: web.Request, device: DeviceRecord) -> web.Response:
        data = await read_json(request)
        if data is None or not isinstance(data.get("capabilities"), list):
            return _error(400, "Expected JSON with a 'capabilities' list.")
        capabilities = [c for c in data["capabilities"] if isinstance(c, str) and c in CAPABILITIES]
        companion.phone.report(device.id, capabilities, paused=bool(data.get("paused")),
                               accessibility=bool(data.get("accessibility")))
        return web.json_response(companion.phone.describe())

    @authed
    async def phone_action_result(request: web.Request, device: DeviceRecord) -> web.Response:
        action_id = request.match_info["action_id"]
        data = await read_json(request)
        if not _ACTION_ID.match(action_id) or data is None or not isinstance(data.get("status"), str):
            return _error(400, "Expected JSON with a 'status'.")
        result_data = data.get("data") if isinstance(data.get("data"), dict) else None
        if result_data is not None and len(str(result_data)) > 20000:
            result_data = {"text": str(result_data.get("text", ""))[:20000]}
        if not companion.phone.complete(device.id, action_id, data["status"], str(data.get("message") or ""),
                                        result_data):
            return _error(404, "That action is no longer waiting for a result.", "gone")
        return web.json_response({"ok": True})

    r = app.router
    r.add_get("/api/v1/hello", hello)
    r.add_post("/api/v1/pair", pair)
    r.add_post("/api/v1/token/refresh", refresh)
    r.add_post("/api/v1/unpair", unpair)
    r.add_get("/api/v1/status", status)
    r.add_get("/api/v1/events", events)
    r.add_post("/api/v1/messages", messages)
    r.add_post("/api/v1/voice", voice)
    r.add_post("/api/v1/confirmations/{request_id}", confirm)
    r.add_get("/api/v1/tasks", tasks_list)
    r.add_get("/api/v1/tasks/{task_id}", task_detail)
    r.add_post("/api/v1/tasks/{task_id}/cancel", task_cancel)
    r.add_get("/api/v1/tasks/{task_id}/speech", task_speech)
    r.add_get("/api/v1/protocols", protocols_list)
    r.add_post("/api/v1/protocols/{protocol_id}/run", protocol_run)
    r.add_get("/api/v1/reminders", reminders_list)
    r.add_post("/api/v1/reminders", reminder_create)
    for op in ("update", "complete", "cancel", "snooze", "fired"):
        path = "/api/v1/reminders/{reminder_id}" + ("" if op == "update" else f"/{op}")
        r.add_post(path, reminder_op(op))
    r.add_get("/api/v1/reminders/{reminder_id}/speech", reminder_speech)
    r.add_get("/api/v1/presence", presence)
    r.add_post("/api/v1/voice/heartbeat", voice_heartbeat)
    r.add_post("/api/v1/voice/claim", voice_claim)
    r.add_post("/api/v1/voice/release", voice_release)
    r.add_post("/api/v1/voice/leave", voice_leave)
    r.add_get("/api/v1/images", images_list)
    r.add_post("/api/v1/images", image_create)
    r.add_get("/api/v1/images/{image_id}", image_detail)
    r.add_get("/api/v1/images/{image_id}/file", image_file)
    r.add_post("/api/v1/images/{image_id}/cancel", image_op("cancel"))
    r.add_post("/api/v1/images/{image_id}/delete", image_op("delete"))
    r.add_post("/api/v1/phone/capabilities", phone_capabilities)
    r.add_post("/api/v1/phone/actions/{action_id}/result", phone_action_result)
