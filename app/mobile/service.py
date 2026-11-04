"""Transport-agnostic logic behind the mobile interface.

Everything here runs on AgentBridge's asyncio loop and uses the objects
the bridge already owns -- the one Agent (and through it the LLM routing,
memory, tools, PermissionChecker and ConfirmationManager), the EventBus and
the VoicePipeline's configured local STT/TTS providers. Nothing here knows
about HTTP; server.py maps these methods and exceptions onto routes.

- Conversation view: an in-memory, sequence-numbered log built from the
  same EventBus events the desktop UI renders, so the phone shows the one
  shared conversation (desktop turns included), not a separate context.
- Turns: at most one mobile turn at a time, and none while the Agent is
  busy with a desktop turn or a hands-free desktop session owns it. The
  Agent's own turn lock is the real guarantee; these checks just turn
  "would have to wait" into a clear "busy" answer for the phone.
- Voice: a WAV uploaded by the phone is validated, written to a private
  temp file, transcribed by the configured STT provider and deleted. The
  reply is synthesized by the configured TTS provider and returned as bytes
  for the phone to play -- nothing is played on the PC.
- Confirmations: the phone may answer a pending ConfirmationManager request
  (first answer wins, desktop or phone). It can't create, skip or widen one.
- Tasks and protocols (when a TaskManager is given): an instruction becomes
  a queued task instead of being refused while Garvis is busy; the phone
  can list tasks, read one task's result, cancel, list protocols and ask
  for a protocol run. The TaskManager -- on the desktop -- accepts,
  validates and executes; the phone only asks.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import secrets
import tempfile
import time
import wave
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.agent import Agent
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import SpeechToTextError, TextToSpeechError
from app.core.logging import get_logger, log_extra, redact
from app.voice.formatting import to_spoken_text
from app.voice.pipeline import VoicePipeline

logger = get_logger("mobile.service")

MOBILE_SOURCE = "mobile"
"""`source` tag on the Agent's USER/ASSISTANT_MESSAGE events for mobile turns."""

_MAX_TURNS_KEPT = 20
_MAX_ARGUMENT_CHARS = 500


class MobileBusyError(Exception):
    """A turn can't start right now. The message is safe to show the user."""


class AudioValidationError(Exception):
    """An uploaded recording was rejected. `kind` is one of "empty",
    "invalid", "unsupported" or "too_long"; the message is user-facing."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class NoReplyError(Exception):
    """The turn has no assistant reply to speak (still running, failed, or unknown)."""


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_rate: int
    duration_seconds: float


@dataclass
class MobileTurn:
    id: str
    kind: str
    """"text" or "voice"."""
    text: str
    done: bool = False
    reply: str | None = None
    """The ASSISTANT_MESSAGE text of this turn; None if the turn failed
    (an LLM error reply is published as an ERROR, not as a reply)."""


def validate_wav(data: bytes, *, max_seconds: float, min_seconds: float = 0.3) -> WavInfo:
    """Accept only uncompressed 16-bit PCM WAV (what the mobile page
    records), fully present, within the duration limits."""
    if not data:
        raise AudioValidationError("The recording is empty.", "empty")
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioValidationError("Unsupported audio format: expected a WAV recording.", "unsupported")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            compression = wav_file.getcomptype()
            pcm = wav_file.readframes(frames)
    except (wave.Error, EOFError, ValueError) as exc:
        raise AudioValidationError("The recording is not a valid WAV file.", "invalid") from exc
    if compression != "NONE" or sample_width != 2 or channels not in (1, 2) or not 8000 <= sample_rate <= 48000:
        raise AudioValidationError(
            "Unsupported WAV format: expected 16-bit PCM, mono or stereo, 8-48 kHz.", "unsupported"
        )
    if len(pcm) < frames * channels * sample_width:
        raise AudioValidationError("The recording is incomplete.", "invalid")
    duration = frames / sample_rate
    if duration > max_seconds:
        raise AudioValidationError(f"The recording is too long (limit {max_seconds:.0f} s).", "too_long")
    if duration < min_seconds:
        raise AudioValidationError("The recording is empty or too short.", "empty")
    return WavInfo(channels=channels, sample_rate=sample_rate, duration_seconds=duration)


def _summarize_arguments(arguments: Any) -> dict[str, str]:
    """Tool arguments made display-safe: secret-looking keys masked (the
    same redaction the logs use), values stringified and truncated."""
    if not isinstance(arguments, dict):
        return {}
    summary: dict[str, str] = {}
    for key, value in list(redact(arguments).items())[:20]:
        text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        summary[str(key)[:60]] = text if len(text) <= _MAX_ARGUMENT_CHARS else text[:_MAX_ARGUMENT_CHARS] + "..."
    return summary


def _unlink_quietly(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        log_extra(logger, 30, "mobile.temp_file_cleanup_failed")


async def _await_with_deadline(awaitable, timeout: float, *, on_late_finish: Callable[[], None] | None = None):
    """Wait up to `timeout` seconds without cancelling the work: STT/TTS run
    in worker threads that can't be interrupted, so a late result is
    discarded and `on_late_finish` (e.g. temp-file cleanup) runs whenever
    the work actually ends. Raises TimeoutError on expiry."""
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if task in done:
        return task.result()

    def _finished(finished: asyncio.Future) -> None:
        if not finished.cancelled():
            finished.exception()  # retrieved, so it is never reported as "never retrieved"
        if on_late_finish is not None:
            on_late_finish()

    task.add_done_callback(_finished)
    raise TimeoutError


class MobileService:
    def __init__(
        self,
        agent: Agent,
        events: EventBus,
        voice: VoicePipeline,
        *,
        is_desktop_session_active: Callable[[], bool] = lambda: False,
        max_message_chars: int = 4000,
        max_audio_seconds: float = 60.0,
        min_audio_seconds: float = 0.3,
        stt_timeout_seconds: float = 120.0,
        tts_timeout_seconds: float = 120.0,
        max_tts_chars: int = 3000,
        history_limit: int = 300,
        tasks=None,
        protocols=None,
        available_tools: Callable[[], set[str] | None] = lambda: None,
    ) -> None:
        self._agent = agent
        self._events = events
        self._voice = voice
        self._is_desktop_session_active = is_desktop_session_active
        self._max_message_chars = max_message_chars
        self._max_audio_seconds = max_audio_seconds
        self._min_audio_seconds = min_audio_seconds
        self._stt_timeout = stt_timeout_seconds
        self._tts_timeout = tts_timeout_seconds
        self._max_tts_chars = max_tts_chars

        self._log: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._seq = 0
        self.epoch = secrets.token_hex(6)
        """Identifies this run of the event log: sequence numbers restart with
        every Garvis start, so a client stores (epoch, seq), never seq alone."""
        self._status = "Idle."
        self._changed = asyncio.Event()
        self._pending_confirmations: dict[str, dict[str, Any]] = {}
        self._turns: OrderedDict[str, MobileTurn] = OrderedDict()
        self._current_turn: MobileTurn | None = None
        self._turn_task: asyncio.Task | None = None
        self._stt_lock = asyncio.Lock()
        self._tts_lock = asyncio.Lock()
        self._unsubscribe: Callable[[], None] | None = None
        self._tasks = tasks
        self._protocols = protocols
        self.last_device_label = ""
        """Type of the paired device ("iPhone", "Android", ...) that sent the
        latest message -- for the desktop's "You (...)" label."""
        self._available_tools = available_tools
        if tasks is not None:
            tasks.add_listener(self._on_task_changed)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._unsubscribe is None:
            self._unsubscribe = self._events.subscribe(self._on_event)

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._notify()  # release any long-poll waiting for news

    # ------------------------------------------------------- event log
    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def change_signal(self) -> asyncio.Event:
        """Set at the next change. Take it BEFORE checking for news, then
        await it, so a change in between is never missed."""
        return self._changed

    def poke(self) -> None:
        """Wake every waiting long-poll: something outside the event log
        (reminders, images, phone actions) changed."""
        self._notify()

    def post_entry(self, kind: str, **fields: Any) -> None:
        """Add a non-Agent entry (e.g. a finished image) to the shared log."""
        self._append(kind, **fields)

    def _append(self, kind: str, **fields: Any) -> None:
        self._seq += 1
        self._log.append({"seq": self._seq, "kind": kind, "time": time.time(), **fields})
        self._notify()

    def _on_event(self, event: AgentEvent) -> None:
        payload = event.payload
        kind = event.type
        if kind == EventType.USER_MESSAGE:
            source = payload.get("source") or "desktop"
            turn_id = self._current_turn.id if source == MOBILE_SOURCE and self._current_turn else None
            self._append("user", text=str(payload.get("text", "")), source=source, turn_id=turn_id)
        elif kind == EventType.ASSISTANT_MESSAGE:
            text = str(payload.get("text", ""))
            source = payload.get("source") or "desktop"
            if source == MOBILE_SOURCE and self._current_turn is not None:
                self._current_turn.reply = text
            turn_id = self._current_turn.id if source == MOBILE_SOURCE and self._current_turn else None
            self._append("assistant", text=text, source=source, turn_id=turn_id)
        elif kind == EventType.TOOL_STARTED:
            self._append(
                "tool_started",
                name=str(payload.get("name", "")),
                arguments=_summarize_arguments(payload.get("arguments")),
            )
        elif kind == EventType.TOOL_FINISHED:
            self._append(
                "tool_finished",
                name=str(payload.get("name", "")),
                success=bool(payload.get("success")),
                error=str(payload.get("error") or ""),
            )
        elif kind == EventType.CONFIRMATION_REQUESTED:
            request = {
                "id": str(payload.get("id", "")),
                "tool_name": str(payload.get("tool_name", "")),
                "arguments": _summarize_arguments(payload.get("arguments")),
                "reason": str(payload.get("reason", "")),
            }
            self._pending_confirmations[request["id"]] = request
            self._append("confirmation_requested", **request)
        elif kind == EventType.CONFIRMATION_RESOLVED:
            request_id = str(payload.get("id", ""))
            self._pending_confirmations.pop(request_id, None)
            self._append("confirmation_resolved", id=request_id, outcome=str(payload.get("outcome", "")))
        elif kind == EventType.ERROR:
            self._append("error", message=str(payload.get("message", "")))
        elif kind == EventType.STATUS:
            self._status = str(payload.get("status", ""))
            self._notify()

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        return {
            "events": [entry for entry in self._log if entry["seq"] > after],
            "last_seq": self._seq,
            # The client's cursor is from an earlier run of the server: it
            # should drop its view and re-read from 0.
            "reset": after > self._seq,
            "status": self._status,
            "busy": self.is_busy,
            "pending_confirmations": list(self._pending_confirmations.values()),
            "tasks_version": self._tasks.version if self._tasks is not None else 0,
            "epoch": self.epoch,
        }

    async def wait_for_updates(self, after: int, wait_seconds: float) -> dict[str, Any]:
        """Long-poll: return as soon as there is anything newer than `after`
        (or any state change), at the latest after `wait_seconds`."""
        if wait_seconds > 0 and after == self._seq:
            changed = self._changed
            try:
                await asyncio.wait_for(changed.wait(), timeout=wait_seconds)
            except TimeoutError:
                pass
        return self.snapshot(after)

    @property
    def last_seq(self) -> int:
        return self._seq

    @property
    def status_text(self) -> str:
        return self._status

    @property
    def tasks(self):
        """The TaskManager, or None when tasks aren't wired in."""
        return self._tasks

    # ----------------------------------------------------------- turns
    @property
    def is_busy(self) -> bool:
        return (self._turn_task is not None and not self._turn_task.done()) or self._agent.is_busy

    def _ensure_can_start_turn(self) -> None:
        if self._tasks is not None:
            return  # queued as a task: it runs when Garvis is free
        if self._turn_task is not None and not self._turn_task.done():
            raise MobileBusyError("Still working on your previous message.")
        if self._is_desktop_session_active():
            raise MobileBusyError("A hands-free voice session is running on the PC. End it there first.")
        if self._agent.is_busy:
            raise MobileBusyError("Garvis is busy with a request from the PC. Try again in a moment.")

    def submit_text(self, text: str, *, kind: str = "text", request_id: str | None = None,
                    device: str = "") -> MobileTurn:
        """Start an Agent turn in the background (or queue it as a task) and
        return immediately. Raises ValueError (bad text) or MobileBusyError."""
        if device:
            self.last_device_label = device
        if self._tasks is not None:
            return self._submit_as_task(text, kind, request_id, device)
        text = (text or "").strip()
        if not text:
            raise ValueError("The message is empty.")
        if len(text) > self._max_message_chars:
            raise ValueError(f"The message is too long (limit {self._max_message_chars} characters).")
        self._ensure_can_start_turn()
        turn = MobileTurn(id=secrets.token_urlsafe(9), kind=kind, text=text)
        self._turns[turn.id] = turn
        while len(self._turns) > _MAX_TURNS_KEPT:
            self._turns.popitem(last=False)
        self._current_turn = turn
        self._turn_task = asyncio.get_running_loop().create_task(self._run_turn(turn))
        log_extra(logger, 20, "mobile.turn_started", turn=turn.id, kind=kind)
        self._notify()
        return turn

    async def _run_turn(self, turn: MobileTurn) -> None:
        try:
            await self._agent.handle_user_message(turn.text, source=MOBILE_SOURCE)
        except Exception:
            # Agent.handle_user_message has its own safety net; this only
            # guards against a turn being left "running" forever.
            logger.exception("mobile.turn_failed")
            self._append("error", message="Something went wrong while processing that. Please try again.")
        finally:
            turn.done = True
            if self._current_turn is turn:
                self._current_turn = None
            self._append("turn_finished", turn_id=turn.id, turn_kind=turn.kind, has_reply=bool(turn.reply))
            log_extra(logger, 20, "mobile.turn_finished", turn=turn.id, has_reply=bool(turn.reply))

    def get_turn(self, turn_id: str) -> MobileTurn | None:
        return self._turns.get(turn_id)

    # ----------------------------------------------------------- tasks
    def _submit_as_task(self, text: str, kind: str, request_id: str | None, device: str = "") -> MobileTurn:
        from app.tasks.manager import TaskError

        try:
            task = self._tasks.submit_instruction(text, origin=MOBILE_SOURCE, request_id=request_id, device=device)
        except TaskError as exc:
            message = str(exc)
            if "waiting" in message or "ready" in message:
                raise MobileBusyError(message) from exc
            raise ValueError(message) from exc
        turn = self._turns.get(task.id)
        if turn is None:
            turn = MobileTurn(id=task.id, kind=kind, text=task.text)
            self._turns[turn.id] = turn
            while len(self._turns) > _MAX_TURNS_KEPT:
                self._turns.popitem(last=False)
            log_extra(logger, 20, "mobile.task_submitted", turn=turn.id, kind=kind)
        self._notify()
        return turn

    def _on_task_changed(self, task) -> None:
        turn = self._turns.get(task.id)
        if turn is not None and task.kind == "instruction":
            if task.status == "running" and self._current_turn is not turn:
                self._current_turn = turn
            elif task.terminal and not turn.done:
                turn.done = True
                if self._current_turn is turn:
                    self._current_turn = None
                self._append("turn_finished", turn_id=turn.id, turn_kind=turn.kind, has_reply=bool(turn.reply),
                             status=task.status)
        self._notify()

    def list_tasks(self, limit: int = 50) -> dict[str, Any]:
        if self._tasks is None:
            return {"tasks": [], "version": 0}
        return {"tasks": [t.summary() for t in self._tasks.list(limit)], "version": self._tasks.version}

    def task_detail(self, task_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(task_id) if self._tasks is not None else None
        return task.detail() if task is not None else None

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Raises LookupError (unknown) or ValueError (can't be cancelled)."""
        from app.tasks.manager import TaskError

        if self._tasks is None or self._tasks.get(task_id) is None:
            raise LookupError("Unknown task.")
        try:
            return self._tasks.cancel(task_id).summary()
        except TaskError as exc:
            raise ValueError(str(exc)) from exc

    def list_protocols(self) -> list[dict[str, Any]]:
        from app.tasks.protocols import public_summary

        if self._protocols is None:
            return []
        tools = self._available_tools()
        return [public_summary(p, tools) for p in self._protocols.protocols()]

    def run_protocol(self, protocol_id: str, request_id: str | None = None, device: str = "") -> dict[str, Any]:
        """Ask the desktop to run a saved protocol. Raises LookupError or
        ValueError (user-facing) -- the TaskManager re-validates it."""
        from app.tasks.manager import TaskError

        if self._tasks is None or self._protocols is None:
            raise LookupError("Protocols aren't available.")
        try:
            task = self._tasks.submit_protocol(protocol_id, origin=MOBILE_SOURCE, request_id=request_id,
                                               device=device)
        except TaskError as exc:
            if "doesn't exist" in str(exc):
                raise LookupError(str(exc)) from exc
            raise ValueError(str(exc)) from exc
        log_extra(logger, 20, "mobile.protocol_requested", task=task.id)
        return task.summary()

    # ----------------------------------------------------------- voice
    async def transcribe_recording(self, data: bytes) -> str:
        """Validate and transcribe one uploaded recording with the configured
        local STT provider. Raises AudioValidationError, MobileBusyError or
        SpeechToTextError. The temp file is always removed."""
        self._ensure_can_start_turn()
        validate_wav(data, max_seconds=self._max_audio_seconds, min_seconds=self._min_audio_seconds)
        if self._stt_lock.locked():
            raise MobileBusyError("Still transcribing the previous recording.")
        async with self._stt_lock:
            fd, path = tempfile.mkstemp(suffix=".wav", prefix="garvis_mobile_")
            cleanup_now = True
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                text = await _await_with_deadline(
                    self._voice.transcribe_file(path),
                    self._stt_timeout,
                    on_late_finish=lambda: _unlink_quietly(path),
                )
            except TimeoutError as exc:
                cleanup_now = False  # still being read by the STT thread; removed when it finishes
                raise SpeechToTextError("Transcription took too long. Please try again.") from exc
            finally:
                if cleanup_now:
                    _unlink_quietly(path)
        return text.strip()

    async def synthesize_reply(self, turn_id: str) -> bytes:
        """Speech (WAV bytes) for a finished mobile turn's reply, from the
        configured local TTS provider. Raises NoReplyError or TextToSpeechError."""
        turn = self._turns.get(turn_id)
        if turn is None or not turn.reply or not turn.reply.strip():
            raise NoReplyError("There is no reply to speak for that message.")
        return await self.synthesize_text(turn.reply)

    async def synthesize_text(self, text: str) -> bytes:
        """Speech (WAV bytes) for `text` from the configured local TTS
        provider -- e.g. the reminder Garvis reads out on a scheduled call."""
        spoken = to_spoken_text(text)
        if len(spoken) > self._max_tts_chars:
            cut = spoken[: self._max_tts_chars]
            spoken = (cut.rsplit(" ", 1)[0] or cut) + "... The rest is on screen."
        async with self._tts_lock:
            try:
                return await _await_with_deadline(self._voice.synthesize_to_bytes(spoken), self._tts_timeout)
            except TimeoutError as exc:
                raise TextToSpeechError("Speech synthesis took too long.") from exc

    # ---------------------------------------------------- confirmations
    def resolve_confirmation(self, request_id: str, approved: bool) -> bool:
        """Answer a pending request through the existing ConfirmationManager.
        False if no such request is pending (already answered, timed out, or
        never existed)."""
        if request_id not in self._pending_confirmations:
            return False
        self._agent.confirmations.resolve(request_id, bool(approved))
        log_extra(logger, 20, "mobile.confirmation_answered", approved=bool(approved))
        return True
