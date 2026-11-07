"""Bridges the asyncio-based Agent (and voice pipeline) to Qt's event loop.

Qt's event loop and asyncio's event loop don't mix directly. We run a
single persistent asyncio loop on a background thread, and use
`asyncio.run_coroutine_threadsafe` to submit work to it. Agent events
(status/tool/confirmation updates) are relayed back to the UI thread as
Qt signals, which Qt guarantees are delivered safely across threads.

Voice input/output rides the same bridge: recording, transcription, and
speech synthesis all run on the background loop, and report status/errors
through the same EventBus the agent already uses, so the UI has one
consistent status/error channel regardless of whether a turn came from
typing or talking.

The optional mobile interface (app/mobile, GARVIS_MOBILE_ENABLED) is served
from this same loop and drives this same Agent/EventBus/VoicePipeline --
it is another way in, not a second assistant.

The Desktop's microphone obeys voice ownership (app/core/voice_ownership.py):
it is one endpoint among the phone and Eye Station, never claims standby
just because Garvis started (only when the user ticks "Wake word standby"),
and every wake listener, session or push-to-talk opens it only once the
ownership manager grants it -- and closes it the moment it says SLEEP.
"""
from __future__ import annotations

import asyncio
import threading

from PySide6.QtCore import QObject, Signal

from app.agent.agent import Agent
from app.agent.build import build_agent
from app.core.config import Settings, get_settings
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import SpeechToTextError, TextToSpeechError
from app.core.logging import content_fingerprint, get_logger, log_extra
from app.core.notices import RecurringNoticeFilter
from app.core.voice_ownership import (
    DESKTOP_ENDPOINT,
    LISTENING_STATES,
    VoiceClaimRefused,
    VoiceOwnershipManager,
    VoiceState,
)
from app.mobile.service import MOBILE_SOURCE
from app.voice.build import build_voice_pipeline
from app.voice.formatting import to_spoken_text
from app.voice.session import ConversationSession, SessionState
from app.voice.wake import (
    GreetingPicker,
    SleepCommandMatcher,
    StandbyListener,
    WakeMatch,
    WakePhraseMatcher,
)

logger = get_logger("ui.agent_bridge")


def _gmail_status(settings: Settings) -> str:
    """From the files GmailManager uses -- no network call, no sign-in."""
    try:
        if settings.gmail_token_path.exists():
            return "signed_in"
        if settings.gmail_client_secret_path.exists():
            return "sign_in_needed"
    except OSError:
        pass
    return "not_configured"


class AgentBridge(QObject):
    status_changed = Signal(str)
    user_message = Signal(str)
    assistant_message = Signal(str)
    tool_started = Signal(str, dict)
    tool_finished = Signal(str, bool, str)
    confirmation_requested = Signal(str, str, dict, str)
    confirmation_resolved = Signal(str, str)
    """(request id, outcome) -- lets the desktop dialog close itself when
    the request was answered elsewhere (the phone) or timed out."""
    remote_user_message = Signal(str, str)
    """(text, source) for a turn typed or spoken on another interface (the
    phone); the desktop already shows its own typed messages itself."""
    error = Signal(str)
    listening_changed = Signal(bool)
    session_state_changed = Signal(str)
    standby_changed = Signal(bool)
    wake_detected = Signal(str)
    """Emitted (from the background loop; Qt queues it to the GUI thread)
    with the canonical wake phrase. The UI decides whether to start a
    session -- it already knows whether a typed/push-to-talk turn is busy."""
    sleep_requested = Signal()
    """The current session heard a standby command / closing statement and
    has finished its acknowledgement. The UI ends the session exactly as
    End Session does, which returns to standby."""
    speaking_changed = Signal(bool)
    """Garvis's voice is playing on the PC's speakers: True when a clip is
    handed to the player, False when it finishes or is stopped -- the real
    playback lifecycle (VoicePipeline.add_playback_listener), which the
    session's "speaking" state (set before synthesis) is not."""
    tool_sources = Signal(str, list)
    """(tool name, sources) for a finished retrieval tool -- the titles/URLs
    its own result returned (app/tools/sources.py)."""
    usage_updated = Signal(dict)
    """Reply to request_usage(): app/ui/usage_view.py:read_usage_summary()."""
    presentation_requested = Signal(dict)
    """A PRESENTATION event (app/agent/presentation.py): something Garvis
    opened or wants the user to see -- drives the attention overlay."""
    task_updated = Signal(dict)
    """A task's summary changed (app/tasks/manager.py)."""
    voice_owner_changed = Signal(str)
    """A one-line description of who holds Garvis's voice now (for the Voice page)."""
    request_started = Signal(str, str)
    """(text, source) for a request that isn't a chat turn (a protocol run)."""
    task_error = Signal(str)
    """A desktop request to run/cancel a task was refused (user-facing)."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        settings = settings or get_settings()
        self._settings = settings

        self._events = EventBus()
        self._events.subscribe(self._on_event)
        self._agent: Agent = build_agent(settings=settings, events=self._events)
        self._voice = build_voice_pipeline(settings)
        add_playback_listener = getattr(self._voice, "add_playback_listener", None)
        if add_playback_listener is not None:
            add_playback_listener(self.speaking_changed.emit)
        self._voice_replies_enabled = False
        self._assistant_message_seq = 0
        """Diagnostic-only counter, logged with each ASSISTANT_MESSAGE
        dispatch (see _on_event) -- lets a log reviewer see whether this
        bridge instance emitted the same content_hash more than once,
        distinguishing a duplicate dispatch here from repeated text
        produced upstream by the LLM itself."""
        self._user_message_seq = 0
        """Companion counter to _assistant_message_seq, logged with each
        USER_MESSAGE dispatch. Agent.handle_user_message() publishes
        exactly one USER_MESSAGE per invocation regardless of caller
        (push-to-talk, typed text, or ConversationSession) -- so this
        counter is a direct, source-agnostic count of Agent invocations,
        and its content_hash lets a log reviewer check a suspected phantom
        utterance against the previous turn's ASSISTANT_MESSAGE hash (a
        match would mean Garvis's own prior reply was transcribed back as
        if the user had said it -- see app/voice/vad.py's self-echo
        handling)."""

        self._session: ConversationSession | None = None
        self._session_active = False
        """True for the whole lifetime of a session (from start_session()
        until end_session()'s cleanup completes) -- used to suppress the
        ASSISTANT_MESSAGE auto-speak path below, since the session already
        speaks each reply itself (interruptibly); without this guard a
        reply would be spoken twice."""
        self._pending_session_stop: "asyncio.Future | None" = None
        """The in-flight end_session() cleanup, if any -- start_session()
        awaits it first so a rapid stop-then-start doesn't race the old
        session's mic teardown against the new session's mic setup (both
        share the same VoicePipeline/recorder)."""

        self._wake_enabled = False
        """The user wants this PC to be the wake-word listener (the Voice page's checkbox). Off at
        every launch -- even with GARVIS_WAKE_WORD_ENABLED=true: the Desktop never claims the
        microphone just because it started (see app/core/voice_ownership.py)."""
        self._sleep_matcher = SleepCommandMatcher(threshold=settings.wake_match_threshold)
        """Stateless; shared by every session and the typed path."""
        self._standby: StandbyListener | None = None
        """Only touched on the background loop. Never active at the same
        time as a session or push-to-talk: all three share one recorder."""
        self._push_to_talk_active = False
        self._greetings = GreetingPicker()
        self._background_errors = RecurringNoticeFilter()
        """Errors raised outside an agent turn (standby/session voice errors,
        the phone server) are shown once, not every time they recur. Errors
        from a turn are always shown: they are that turn's answer."""

        self._loop = asyncio.new_event_loop()
        from app.core.presence import PresenceHub

        self.presence = PresenceHub()
        """What a display shows (idle/listening/thinking/speaking/notification/error); fed by
        GarvisState's mode (app/ui/state.py) and attention events, read by /api/v1/presence."""
        self.presence.bind(self._loop)
        self.voice_ownership = VoiceOwnershipManager()
        """The one authority on which endpoint (this PC, the phone, Eye Station) may use a
        microphone; served to the devices at /api/v1/voice/*."""
        self.voice_ownership.register_local(DESKTOP_ENDPOINT, kind="desktop", label="Garvis Desktop")
        self.voice_ownership.add_listener(self._on_voice_ownership_changed)
        self.voice_ownership.bind(self._loop)
        self._ownership_lock = asyncio.Lock()
        self._session_claim: str | None = None
        self._ptt_claim: str | None = None
        self._ptt_cancelled = False
        self._ptt_recording = False
        """Push-to-talk's own recording is open (standby shares the recorder, so is_listening can't tell)."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        from app.tasks.manager import TaskManager
        from app.tasks.protocols import ProtocolStore

        self._protocols = ProtocolStore(settings.data_dir / "protocols")
        self._mobile = None
        """app.mobile.server.MobileServer when GARVIS_MOBILE_ENABLED, else None."""
        self._companion = None
        """Reminders, images and phone actions for the native Android app
        (app/mobile/companion.py); built with the mobile interface. Its
        tools are registered before the tool catalog is read."""
        if settings.mobile_enabled:
            from app.mobile.build import build_companion, build_companion_tools

            self._companion = build_companion(settings)
            self._companion.presence = self.presence
            self._companion.voice_ownership = self.voice_ownership
            if settings.reminders_model_phrasing:
                from app.reminders.phrasing import ReminderPhrasingWorker
                from app.reminders.schedule import get_zone

                self._companion.phraser.use_model(self._agent.llm)
                ReminderPhrasingWorker(self._companion.reminders, self._companion.phraser, self._loop, get_zone).start()
            self._agent.register_tools(build_companion_tools(
                settings, self._companion,
                phone_ready=lambda: self._mobile is not None and self._mobile.pairing.device_count(kind="phone") > 0,
            ))
        self._tool_names = {entry["name"] for entry in self.tool_catalog()}
        self._tasks = TaskManager(
            self._agent,
            self._events,
            settings.data_dir / "tasks" / "tasks.json",
            self._protocols,
            available_tools=lambda: self._tool_names,
            is_desktop_session_active=lambda: self._session_active,
        )
        """Owns queued/running/finished tasks (phone instructions, protocol
        runs) on the background loop; also records desktop chat turns."""
        asyncio.run_coroutine_threadsafe(self._start_tasks(), self._loop).result(timeout=5)

        # self._mobile: app.mobile.server.MobileServer when GARVIS_MOBILE_ENABLED, else None.
        # Imported lazily so aiohttp/cryptography are only needed when it's on.
        if settings.mobile_enabled:
            from app.mobile.build import build_mobile_server

            self._mobile = build_mobile_server(
                settings,
                agent=self._agent,
                events=self._events,
                voice=self._voice,
                is_desktop_session_active=lambda: self._session_active,
                tasks=self._tasks,
                protocols=self._protocols,
                available_tools=lambda: self._tool_names,
                companion=self._companion,
            )
            asyncio.run_coroutine_threadsafe(self._start_mobile(), self._loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _on_event(self, event: AgentEvent) -> None:
        payload = event.payload
        if event.type == EventType.STATUS:
            self.status_changed.emit(payload.get("status", ""))
        elif event.type == EventType.USER_MESSAGE:
            text = payload.get("text", "")
            self._user_message_seq += 1
            log_extra(
                logger,
                20,
                "agent_bridge.user_message_dispatch",
                seq=self._user_message_seq,
                session_active=self._session_active,
                **content_fingerprint(text),
            )
            self.user_message.emit(text)
            if payload.get("source"):
                self.remote_user_message.emit(text, str(payload["source"]))
        elif event.type == EventType.ASSISTANT_MESSAGE:
            text = payload.get("text", "")
            self._assistant_message_seq += 1
            log_extra(
                logger,
                20,
                "agent_bridge.assistant_message_dispatch",
                seq=self._assistant_message_seq,
                **content_fingerprint(text),
            )
            self.assistant_message.emit(text)
            # Suppressed during an active session: ConversationSession
            # already speaks each reply itself (interruptibly) -- letting
            # this fire too would speak the same reply twice. Mobile turns
            # are spoken on the phone, never through the PC's speakers.
            if (
                self._voice_replies_enabled
                and not self._session_active
                and payload.get("source") != MOBILE_SOURCE
                and text.strip()
                and self._may_speak_aloud()
            ):
                self._loop.create_task(self._safe_speak(text))
        elif event.type == EventType.TOOL_STARTED:
            self.tool_started.emit(payload.get("name", ""), payload.get("arguments", {}))
        elif event.type == EventType.TOOL_FINISHED:
            self.tool_finished.emit(
                payload.get("name", ""), bool(payload.get("success")), payload.get("error") or ""
            )
            if payload.get("sources"):
                self.tool_sources.emit(payload.get("name", ""), list(payload["sources"]))
        elif event.type == EventType.CONFIRMATION_REQUESTED:
            self.presence.notify("confirmation")
            self.confirmation_requested.emit(
                payload.get("id", ""),
                payload.get("tool_name", ""),
                payload.get("arguments", {}),
                payload.get("reason", ""),
            )
        elif event.type == EventType.CONFIRMATION_RESOLVED:
            self.confirmation_resolved.emit(payload.get("id", ""), payload.get("outcome", ""))
        elif event.type == EventType.ERROR:
            message = payload.get("message", "")
            if self._agent.is_busy or self._background_errors.should_report(message):
                self.error.emit(message)
            else:
                log_extra(logger, 20, "agent_bridge.repeated_error_suppressed")
        elif event.type == EventType.SESSION_STATE:
            self.session_state_changed.emit(payload.get("state", ""))
        elif event.type == EventType.PRESENTATION:
            self.presentation_requested.emit(dict(payload))
        elif event.type == EventType.TASK_UPDATED:
            self.task_updated.emit(dict(payload.get("task") or {}))
        elif event.type == EventType.REQUEST_STARTED:
            self.request_started.emit(str(payload.get("text", "")), str(payload.get("source") or "desktop"))

    # --------------------------------------------------------------- text
    def send_message(self, text: str) -> None:
        asyncio.run_coroutine_threadsafe(self._send_message(text), self._loop)

    async def _send_message(self, text: str) -> None:
        # Same rule as a session's transcripts: with wake mode on, a typed
        # standby command / closing statement never reaches the Agent.
        sleep = self._sleep_matcher.match(text) if self._wake_enabled else None
        if sleep is None:
            await self._agent.handle_user_message(text)
            return
        await self._typed_sleep(text, sleep)

    async def _typed_sleep(self, text: str, sleep) -> None:
        """A typed "go to standby": no session holds the mic (typing is
        disabled during one), so standby is already listening -- answer with
        the fixed acknowledgement, through the same events as an Agent turn
        (so it's shown, and spoken when voice replies are on), and make sure
        standby is running. Never ends the app: that is shutdown() only."""
        log_extra(logger, 20, "agent_bridge.typed_sleep_requested", phrase=sleep.phrase)
        self._voice.stop_speaking()  # a previous reply still playing gives way, like a barge-in
        await self._events.publish(AgentEvent(type=EventType.USER_MESSAGE, payload={"text": text}))
        await self._events.publish(
            AgentEvent(type=EventType.ASSISTANT_MESSAGE, payload={"text": sleep.acknowledgement})
        )
        await self._events.publish(AgentEvent(type=EventType.STATUS, payload={"status": "Done."}))
        await self._start_standby()

    def resolve_confirmation(self, request_id: str, approved: bool) -> None:
        self._loop.call_soon_threadsafe(
            self._agent.confirmations.resolve, request_id, approved
        )

    # -------------------------------------------------------------- voice
    def set_voice_replies_enabled(self, enabled: bool) -> None:
        self._voice_replies_enabled = enabled

    def start_listening(self) -> None:
        asyncio.run_coroutine_threadsafe(self._start_listening(), self._loop)

    async def _start_listening(self) -> None:
        # Push-to-talk needs the recorder that standby is holding.
        self._push_to_talk_active = True
        self._ptt_cancelled = False
        try:
            claim, _ = self.voice_ownership.claim(DESKTOP_ENDPOINT, "active", "push_to_talk", explicit=True)
        except VoiceClaimRefused as exc:
            self._push_to_talk_active = False
            self._ptt_cancelled = True
            self.listening_changed.emit(False)
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))
            return
        self._ptt_claim = claim.id
        await self._stop_standby()
        granted = await self._wait_for_microphone(claim.id)
        if self._ptt_cancelled or granted is None:
            # Released before the other device gave up its microphone, or cut off meanwhile.
            self._push_to_talk_active = False
            self._ptt_cancelled = True
            self._end_claim(claim.id)
            if granted is None and self._ptt_claim == claim.id:
                await self._events.publish(AgentEvent(type=EventType.ERROR, payload={
                    "message": "Another Garvis device still has the microphone -- try again in a moment."}))
            return
        try:
            self._voice.start_listening()
            self._ptt_recording = True
        except SpeechToTextError as exc:
            self._push_to_talk_active = False
            self._ptt_cancelled = True
            self._end_claim(claim.id)
            await self._start_standby()
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))
            return
        self.listening_changed.emit(True)
        await self._events.publish(AgentEvent(type=EventType.STATUS, payload={"status": "Listening..."}))

    def stop_listening(self) -> None:
        asyncio.run_coroutine_threadsafe(self._stop_listening(), self._loop)

    async def _stop_listening(self) -> None:
        self.listening_changed.emit(False)
        if self._ptt_cancelled or not self._ptt_recording:
            # Never recorded (refused, still waiting for the microphone) or cut off by a device
            # with priority: nothing to transcribe.
            self._ptt_cancelled = True
            await self._events.publish(AgentEvent(type=EventType.STATUS, payload={"status": "Idle."}))
            return
        await self._events.publish(AgentEvent(type=EventType.STATUS, payload={"status": "Transcribing..."}))
        try:
            self._ptt_recording = False
            text = await self._voice.stop_listening_and_transcribe()
        except SpeechToTextError as exc:
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))
            return
        finally:
            # The mic is released by now; a wake during the agent turn that
            # follows is ignored by the UI's busy gating.
            self._push_to_talk_active = False
            self._end_claim(self._ptt_claim)
            self._ptt_claim = None
            await self._start_standby()

        if not text.strip():
            await self._events.publish(AgentEvent(type=EventType.STATUS, payload={"status": "Idle."}))
            return
        await self._agent.handle_user_message(text)

    async def _safe_speak(self, text: str) -> None:
        # Only the spoken copy is adapted (raw URLs/markdown stripped, lists
        # turned into natural sentences) -- the GUI already has `text`
        # unmodified via assistant_message.emit() above, so no information
        # is lost from the text response.
        try:
            await self._voice.speak(to_spoken_text(text))
        except TextToSpeechError as exc:
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))

    def stop_speaking(self) -> None:
        self._loop.call_soon_threadsafe(self._voice.stop_speaking)

    def voice_level(self) -> float:
        """RMS of Garvis's own voice right now (0.0 when silent), from the
        envelope of the clip being played -- read-only, for the orb's speaking
        animation, which samples it only while speaking_changed says True."""
        try:
            return float(self._voice.reference_energy_now())
        except Exception:  # noqa: BLE001 -- purely cosmetic
            return 0.0

    # ------------------------------------------------------ dashboard info
    def system_info(self) -> dict:
        """Static facts for the dashboard. Never includes secret values --
        only whether a key is configured."""
        from app.llm.registry import REQUIRED_MODEL

        s = self._settings
        return {
            "model": REQUIRED_MODEL,
            "provider": "Anthropic API",
            "api_key_configured": bool(s.anthropic_api_key),
            "stt": f"{s.stt_provider} · {s.stt_model} · {s.stt_device}/{s.stt_compute_type}",
            "stt_language": s.stt_language or "auto",
            "tts": f"{s.tts_provider} · {s.tts_voice}",
            "playback_backend": s.tts_playback_backend,
            "echo_guard": s.session_echo_guard_enabled,
            "wake_word_enabled": s.wake_word_enabled,
            "mobile_enabled": s.mobile_enabled,
            "desktop_control_enabled": s.desktop_control_enabled,
            "workspace_enabled": s.workspace_enabled,
            "research_engine": s.research_search_engine,
            "budget_enabled": s.llm_budget_enabled,
            "daily_budget_usd": s.llm_daily_budget_usd,
            "monthly_budget_usd": s.llm_monthly_budget_usd,
            "max_cost_per_task_usd": s.llm_max_cost_per_task_usd,
            "usage_db": str(s.llm_usage_db_path),
            "global_hotkey": s.global_hotkey,
            "gmail_status": _gmail_status(s),
        }

    def tool_catalog(self) -> list[dict]:
        """The Agent's registered tools (name, description, risk, whether it
        always asks first) for the Tools page -- read from the registry the
        Agent itself uses, so the page can't list a tool that doesn't exist."""
        from app.security.permissions import PermissionChecker, PermissionDecision
        from app.tools.base import RiskLevel, Tool

        registry = getattr(self._agent, "_registry", None)
        if registry is None:
            return []
        # The same checker the Agent asks before every tool call.
        permissions = getattr(self._agent, "_permissions", None) or PermissionChecker()
        catalog = []
        for tool in registry.all_tools():
            risk = RiskLevel(getattr(tool, "risk_level", RiskLevel.LOW))
            decision = permissions.evaluate(tool).decision
            catalog.append({
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "risk": risk.name.lower(),
                "confirm": decision == PermissionDecision.REQUIRE_CONFIRMATION,
                # Decides per call (e.g. PowerShell: anything not read-only asks first).
                "conditional": type(tool).confirmation_reason is not Tool.confirmation_reason,
            })
        return catalog

    def request_usage(self) -> None:
        """Read the local usage ledger off the GUI thread; answers with
        usage_updated(summary)."""
        asyncio.run_coroutine_threadsafe(self._read_usage(), self._loop)

    async def _read_usage(self) -> None:
        from app.ui.usage_view import read_usage_summary

        try:
            summary = await asyncio.to_thread(read_usage_summary, self._settings.llm_usage_db_path)
        except Exception as exc:  # noqa: BLE001 -- shown on the Usage page, never fatal
            logger.exception("agent_bridge.usage_read_failed")
            summary = {"available": False, "reason": f"Could not read the usage ledger: {exc}"}
        self.usage_updated.emit(summary)

    # ------------------------------------------------------- session mode
    def start_session(self, greet: bool = False) -> None:
        """Idempotent: a second call while a session is already active (or
        starting) is a no-op -- see `_session`/`_session_active`, set
        synchronously here on the GUI thread before any awaiting happens.

        `greet=True` is the wake-word path: the session opens with a random
        formal greeting (spoken via TTS, never the LLM) and is seeded with
        standby's learned noise floor. A manual start (default) does
        neither."""
        if self._session is not None:
            return
        try:
            # A wake-phrase session is only possible on the device that was listening; a click is
            # the user's explicit choice and takes the voice from a conversation elsewhere.
            claim, _ = self.voice_ownership.claim(DESKTOP_ENDPOINT, "active", "wake" if greet else "session",
                                                  explicit=not greet)
        except VoiceClaimRefused as exc:
            log_extra(logger, 20, "agent_bridge.session_refused", code=exc.code)
            asyncio.run_coroutine_threadsafe(self._session_refused(str(exc)), self._loop)
            return
        pending_stop = self._pending_session_stop
        self._pending_session_stop = None
        greeting = self._greetings.next() if greet else None
        standby = self._standby
        noise_floor = standby.noise_floor if greet and standby is not None else None

        session = ConversationSession(
            agent=self._agent,
            voice=self._voice,
            events=self._events,
            poll_interval_seconds=self._settings.session_poll_interval_seconds,
            energy_threshold=self._settings.session_energy_threshold,
            min_speech_seconds=self._settings.session_min_speech_seconds,
            min_silence_seconds=self._settings.session_min_silence_seconds,
            barge_in_energy_threshold=self._settings.session_barge_in_energy_threshold,
            max_utterance_seconds=self._settings.session_max_utterance_seconds,
            greeting=greeting,
            initial_noise_floor=noise_floor,
            # Sleep commands only make sense when there is a standby to go
            # back to; without wake mode every utterance reaches the Agent
            # exactly as before.
            sleep_matcher=self._sleep_matcher if self._wake_enabled else None,
            on_sleep=self._on_session_sleep,
        )
        self._session_claim = claim.id  # before _session: the loop must never see a session without its claim
        self._session = session
        self._session_active = True
        asyncio.run_coroutine_threadsafe(self._start_session_async(session, pending_stop, claim.id), self._loop)

    async def _session_refused(self, message: str) -> None:
        await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": message}))
        await self._events.publish(AgentEvent(type=EventType.SESSION_STATE, payload={"state": SessionState.IDLE.value}))

    async def _start_session_async(self, session: ConversationSession, pending_stop, claim_id: str = "") -> None:
        if pending_stop is not None:
            # A prior end_session() may still be releasing the mic (shared
            # VoicePipeline/recorder) -- wait for it before opening a new
            # stream, so the two never race each other.
            try:
                await asyncio.wrap_future(pending_stop)
            except Exception as exc:
                log_extra(logger, 10, "agent_bridge.pending_stop_failed", error=str(exc))
        if session is not self._session:
            return  # superseded by an end_session() that ran before this
        if not claim_id:  # not through start_session(): the session still needs the voice
            try:
                claim_id = self.voice_ownership.claim(DESKTOP_ENDPOINT, "active", "session", explicit=True)[0].id
            except VoiceClaimRefused as exc:
                self._session = None
                self._session_active = False
                await self._session_refused(str(exc))
                return
            self._session_claim = claim_id
        # The microphone is this PC's only once the device that had it has released its own.
        granted = await self._wait_for_microphone(claim_id)
        if session is not self._session:
            return
        if granted is None:
            self._session = None
            self._session_active = False
            self._session_claim = None
            self._end_claim(claim_id)
            await self._session_refused("Another Garvis device still has the microphone, so the session didn't start.")
            return
        # Standby must release the shared mic before the session opens it.
        await self._stop_standby()
        if session is not self._session:
            return
        # Each session is a fresh conversation. Done here, on the loop that
        # owns the Agent, after the previous session's stop() has finished
        # (its cancelled turn may have left partial messages behind) -- and
        # after any in-flight mobile turn, so the reset never lands mid-turn.
        await self._agent.reset_conversation_between_turns()
        if session is not self._session:
            return
        try:
            session.start()
        except SpeechToTextError as exc:
            if self._session is session:
                self._session = None
                self._session_active = False
                self._session_claim = None
            self._end_claim(claim_id)
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))
            await self._events.publish(
                AgentEvent(type=EventType.SESSION_STATE, payload={"state": SessionState.IDLE.value})
            )
            await self._start_standby()

    def end_session(self) -> None:
        """Idempotent: a second call with no active session is a no-op.
        Returns to standby afterwards when wake mode is enabled."""
        if self._session is None:
            return
        session = self._session
        claim_id = self._session_claim
        self._session = None
        self._session_active = False
        self._session_claim = None
        self._pending_session_stop = asyncio.run_coroutine_threadsafe(
            self._end_session_async(session, claim_id), self._loop
        )

    async def _end_session_async(self, session: ConversationSession, claim_id: str | None = None) -> None:
        await session.stop()
        self._end_claim(claim_id)  # the mic is closed: the voice goes back to the standby owner
        await self._start_standby()

    # ------------------------------------------------------- wake standby
    @property
    def standby_enabled(self) -> bool:
        return self._wake_enabled

    @property
    def is_standby_active(self) -> bool:
        return self._standby is not None

    def set_standby_enabled(self, enabled: bool) -> None:
        """Turn wake-word standby on/off at runtime (default comes from
        Settings.wake_word_enabled). Takes effect whenever no session or
        push-to-talk turn holds the mic."""
        self._wake_enabled = enabled
        # Ticking it is the Desktop's explicit request for standby: it joins the queue and
        # listens once (and only while) the ownership manager says it's its turn.
        self.voice_ownership.set_wants_standby(DESKTOP_ENDPOINT, enabled)
        asyncio.run_coroutine_threadsafe(self._apply_voice_ownership(), self._loop)

    async def _start_standby(self) -> None:
        """Loop-only. No-op unless wake mode is on and nothing else holds
        the mic."""
        if (
            not self._wake_enabled
            or self._standby is not None
            or self._session is not None
            or self._push_to_talk_active
            or self.voice_ownership.state_of(DESKTOP_ENDPOINT) != VoiceState.STANDBY
        ):
            return
        settings = self._settings
        standby = StandbyListener(
            voice=self._voice,
            on_wake=self._on_wake,
            matcher=WakePhraseMatcher(threshold=settings.wake_match_threshold),
            poll_interval_seconds=settings.session_poll_interval_seconds,
            energy_threshold=settings.session_energy_threshold,
            min_speech_seconds=settings.session_min_speech_seconds,
            min_silence_seconds=settings.wake_min_silence_seconds,
            min_segment_seconds=settings.wake_min_segment_seconds,
            max_segment_seconds=settings.wake_max_segment_seconds,
            cooldown_seconds=settings.wake_cooldown_seconds,
            playback_tail_seconds=settings.wake_playback_tail_seconds,
            initial_quiet_seconds=settings.wake_playback_tail_seconds,
            warm_up_stt=True,
        )
        try:
            standby.start()
        except SpeechToTextError as exc:
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))
            return
        self._standby = standby
        self.standby_changed.emit(True)

    async def _stop_standby(self) -> None:
        """Loop-only. Releases the mic; safe when standby isn't running."""
        standby = self._standby
        if standby is None:
            return
        self._standby = None
        await standby.stop()
        self.standby_changed.emit(False)

    # ------------------------------------------------------ voice ownership
    def _on_voice_ownership_changed(self, version: int) -> None:
        """Any thread: something about who holds the voice changed -- act on it on the loop."""
        try:
            asyncio.run_coroutine_threadsafe(self._apply_voice_ownership(), self._loop)
        except RuntimeError:  # loop closed (shutting down)
            pass

    async def _apply_voice_ownership(self) -> None:
        """Loop-only. Make this PC's microphone match its ownership state, then tell the
        manager whether it is closed (the hand-over to another device waits for that)."""
        async with self._ownership_lock:
            view = self.voice_ownership.view(DESKTOP_ENDPOINT)
            state = VoiceState(view["state"])
            claims = {c["id"] for c in view["claims"]}
            if state == VoiceState.STANDBY:
                await self._start_standby()
            elif state not in LISTENING_STATES:
                await self._stop_standby()
                if self._session is not None and self._session_claim not in claims:
                    await self._end_preempted_session(view)
                if self._push_to_talk_active and self._ptt_claim not in claims:
                    self._ptt_cancelled = True
                    self._push_to_talk_active = False
                    self._ptt_recording = False
                    self._ptt_claim = None
                    self._voice.stop_continuous_listening()  # releases the recorder, discards the audio
                    self.listening_changed.emit(False)
                    log_extra(logger, 20, "agent_bridge.push_to_talk_preempted")
            holder = self.voice_ownership.holder()
            if holder is not None and holder != DESKTOP_ENDPOINT and self._voice.is_speaking:
                self._voice.stop_speaking()  # another device has the voice: this PC goes quiet
            self.voice_ownership.report_mic(DESKTOP_ENDPOINT, self._voice.is_listening, view["version"])
            self.voice_owner_changed.emit(describe_voice_owner(view))

    async def _end_preempted_session(self, view: dict) -> None:
        session = self._session
        self._session = None
        self._session_active = False
        self._session_claim = None
        owner = (view.get("owner") or {}).get("label") or "another device"
        log_extra(logger, 20, "agent_bridge.session_preempted")
        await session.stop()  # publishes SESSION_STATE idle: the window ends the session view
        await self._events.publish(AgentEvent(type=EventType.STATUS, payload={
            "status": f"Voice moved to {owner} -- this PC stopped listening."}))

    async def _wait_for_microphone(self, claim_id: str) -> dict | None:
        """Wait until this PC's claim is granted (ACTIVE). None if the claim was dropped
        meanwhile, or the other device never let go (it can't take longer than its lease)."""
        manager = self.voice_ownership

        def settled(view: dict) -> bool:
            return view["state"] == VoiceState.ACTIVE.value or claim_id not in {c["id"] for c in view["claims"]}

        view = manager.view(DESKTOP_ENDPOINT)
        if not settled(view):
            owner = next((e["label"] for e in view["endpoints"] if not e["you"] and e["state"] != "disconnected"),
                         "the other device")
            await self._events.publish(AgentEvent(type=EventType.STATUS, payload={
                "status": f"Waiting for {owner} to release its microphone..."}))
        view = await manager.wait_for(DESKTOP_ENDPOINT, settled,
                                      manager.lease_seconds + manager.release_timeout_seconds + 2.0)
        if view is None or view["state"] != VoiceState.ACTIVE.value:
            return None
        return view

    def _end_claim(self, claim_id: str | None) -> None:
        if claim_id:
            self.voice_ownership.report_mic(DESKTOP_ENDPOINT, self._voice.is_listening, self.voice_ownership.version)
            self.voice_ownership.release(DESKTOP_ENDPOINT, claim_id)

    def _may_speak_aloud(self) -> bool:
        """Voice replies to typed messages: not while another device holds Garvis's voice."""
        holder = self.voice_ownership.holder()
        return holder is None or holder == DESKTOP_ENDPOINT

    # ------------------------------------------------------ tasks & protocols
    async def _start_tasks(self) -> None:
        self._tasks.start()

    @property
    def protocol_store(self):
        """app/tasks/protocols.py ProtocolStore (file-based, thread-safe)."""
        return self._protocols

    @property
    def available_tools(self) -> set[str]:
        return set(self._tool_names)

    def run_protocol(self, protocol_id: str) -> None:
        """Ask the TaskManager to run a saved protocol (desktop origin). A
        refusal comes back as task_error."""
        async def _submit() -> None:
            from app.tasks.manager import TaskError

            try:
                self._tasks.submit_protocol(protocol_id, origin="desktop")
            except TaskError as exc:
                self.task_error.emit(str(exc))

        asyncio.run_coroutine_threadsafe(_submit(), self._loop)

    def cancel_task(self, task_id: str) -> None:
        async def _cancel() -> None:
            from app.tasks.manager import TaskError

            try:
                self._tasks.cancel(task_id)
            except TaskError as exc:
                self.task_error.emit(str(exc))

        asyncio.run_coroutine_threadsafe(_cancel(), self._loop)

    def tasks_snapshot(self, limit: int = 50) -> list[dict]:
        """Current task summaries, read on the loop that owns them."""
        async def _read() -> list[dict]:
            return [task.summary() for task in self._tasks.list(limit)]

        try:
            return asyncio.run_coroutine_threadsafe(_read(), self._loop).result(timeout=2)
        except Exception:  # noqa: BLE001 -- the panel shows what the signals deliver instead
            return []

    def task_detail(self, task_id: str) -> dict | None:
        async def _read() -> dict | None:
            task = self._tasks.get(task_id)
            return task.detail() if task is not None else None

        try:
            return asyncio.run_coroutine_threadsafe(_read(), self._loop).result(timeout=2)
        except Exception:  # noqa: BLE001
            return None

    def mobile_device_label(self) -> str:
        """Type of the paired phone that sent the latest message ("iPhone",
        "Android", ...), or "" if unknown."""
        service = getattr(self._mobile, "_service", None) if self._mobile is not None else None
        return getattr(service, "last_device_label", "") or ""

    def reveal_window(self, handle: int) -> None:
        """Bring a window a tool opened to the front -- only ever on the
        user's explicit click (the attention card's Show button)."""
        desktop = getattr(self._agent, "desktop", None)
        if desktop is None or not handle:
            return

        async def _show() -> None:
            try:
                await desktop.show_window(int(handle))
            except Exception as exc:  # noqa: BLE001 -- e.g. the window was closed meanwhile
                self.task_error.emit(f"Couldn't bring that window forward: {exc}")

        asyncio.run_coroutine_threadsafe(_show(), self._loop)

    # ------------------------------------------------------ mobile interface
    @property
    def mobile(self):
        """The MobileServer (pairing codes, URLs, CA fingerprint), or None
        when GARVIS_MOBILE_ENABLED is off."""
        return self._mobile

    async def _start_mobile(self) -> None:
        try:
            await self._mobile.start()
        except Exception as exc:
            logger.exception("agent_bridge.mobile_start_failed")
            message = f"The phone interface could not start: {exc}"
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": message}))

    def _on_session_sleep(self, session: ConversationSession) -> None:
        if session is not self._session:
            return  # already ended/superseded -- never end a newer session
        log_extra(logger, 20, "agent_bridge.sleep_requested")
        self.sleep_requested.emit()

    def _on_wake(self, match: WakeMatch) -> None:
        # v1 drops anything said after the wake phrase in the same breath
        # ("Hey Garvis, open Chrome"): the greeting asks for the command.
        # `match.remainder` is where a future version could pick it up.
        log_extra(
            logger,
            20,
            "agent_bridge.wake_detected",
            phrase=match.phrase,
            remainder_dropped=bool(match.remainder),
        )
        self.wake_detected.emit(match.phrase)

    def shutdown(self) -> None:
        self._wake_enabled = False  # nothing may restart standby from here on
        self.voice_ownership.set_wants_standby(DESKTOP_ENDPOINT, False)
        if self._session is not None:
            session = self._session
            self._session = None
            self._session_active = False
            self._finish_on_loop(session.stop(), "session")
        self._finish_on_loop(self._stop_standby(), "standby")
        # Accepted tasks that haven't finished are recorded as interrupted (never re-run).
        self._finish_on_loop(self._tasks.stop(), "tasks")
        if self._mobile is not None:
            # Close the port and end any mobile turn before the loop stops.
            self._finish_on_loop(self._mobile.stop(), "mobile")
        browser_manager = self._agent.browser_manager
        if browser_manager is not None:
            # Block briefly so the browser process is actually closed before
            # the app exits -- otherwise a launched-but-unclosed browser can
            # be left running as an orphaned process.
            self._finish_on_loop(browser_manager.shutdown(), "browser")
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _finish_on_loop(self, coro, step: str) -> None:
        """Run one shutdown step on the agent loop and wait up to 5 s. A step
        that fails or times out is logged at debug level and skipped, so the
        remaining steps (and the app's exit) still happen."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            future.result(timeout=5)
        except Exception as exc:
            log_extra(logger, 10, "agent_bridge.shutdown_step_failed", step=step, error=str(exc) or type(exc).__name__)


def describe_voice_owner(view: dict) -> str:
    """The Voice page's one line about who holds Garvis's voice (from this PC's ownership view)."""
    state = view.get("state")
    owner = view.get("owner") or {}
    if state == VoiceState.STANDBY.value:
        return "This PC is the listener -- say \u201cHey Garvis\u201d."
    if state == VoiceState.ACTIVE.value:
        return "This PC has Garvis's voice (conversation)."
    if view.get("reason") == "waiting_for_release":
        return "Waiting for another device to release its microphone..."
    if owner:
        doing = {"standby": "listening for \u201cHey Garvis\u201d", "active": "in a conversation",
                 "priority": "delivering a reminder or call"}.get(owner.get("state"), "using the voice")
        queued = " This PC is next in line." if view.get("wants_standby") else ""
        return f"{owner.get('label', 'Another device')} is {doing} -- this PC's microphone is off.{queued}"
    return "No device is listening for \u201cHey Garvis\u201d -- this PC's microphone is off."
