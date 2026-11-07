"""Hands-free conversation session: one "Start Session" click keeps
listening, responding, and speaking in a loop -- no per-turn button press --
until "End Session". See app/ui/agent_bridge.py:AgentBridge for how this
wires into the UI/EventBus the rest of Garvis already uses.

Deliberately NOT a second voice architecture: this sits on top of the
existing `VoicePipeline` (STT/TTS/recorder/player, extended with a small
additive session-mode API -- see its module docstring) and calls
`Agent.handle_user_message()` completely unchanged. It only adds the
orchestration loop (listen -> think -> speak -> repeat) and barge-in
detection that a single push-to-talk button never needed, using the same
`EventBus`/`AgentEvent` the Agent already publishes through (a new
`EventType.SESSION_STATE`, not a competing event system).

Barge-in: while SPEAKING, the microphone is polled concurrently with
playback (VoicePipeline.read_microphone_chunk()); a sustained loud enough
chunk immediately calls VoicePipeline.stop_speaking() -- a true cancellation
(AudioPlayer.stop() purges playback outright), not a "let the sentence
finish" queue. Audio is not discarded while the VAD debounces toward
speech_start: like _capture_utterance's preroll, every chunk read during
that window is kept in a small rolling buffer and spliced in ahead of the
triggering chunk once speech_start fires, so the seed handed to the next
utterance capture (along with the already-active VoiceActivityDetector)
carries the interruption's true onset, not just its last fraction of a
second.

Self-echo rejection: a fixed higher threshold while speaking
(`barge_in_energy_threshold`) alone is not sufficient -- speaker leakage of
Garvis's own TTS output into the microphone can exceed it on real hardware,
causing Garvis to "interrupt itself" repeatedly (see app/voice/vad.py's
module docstring). Each chunk polled during SPEAKING is fed to the VAD
alongside `VoicePipeline.reference_energy_now()`, Garvis's own current
playback loudness, so the effective barge-in threshold tracks it and rises
only for as long as -- and only as much as -- Garvis is actually making
sound, without dulling sensitivity when Garvis is quiet (leading silence,
mid-sentence pauses) or raising the threshold globally.

Thinking-time backlog: the microphone isn't read while the Agent thinks or
TTS synthesizes, so seconds of audio wait in the recorder's queue when
playback starts. That backlog is judged frame by frame at its raw level
(Garvis was silent): only speech still going on as playback begins
interrupts (its whole onset kept); a sound that already ended -- a cough,
a TV, the tail of the user's last words -- is ignored instead of cutting
the reply off and starting a phantom turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum

from app.agent.agent import Agent
from app.core.events import AgentEvent, EventBus, EventType
from app.core.exceptions import SpeechToTextError, TextToSpeechError
from app.core.logging import content_fingerprint, get_logger, log_extra
from app.voice.formatting import to_spoken_text
from app.voice.pipeline import VoicePipeline
from app.voice.vad import VoiceActivityDetector
from app.voice.wake import SleepCommandMatcher

logger = get_logger("voice.session")

_VAD_FRAME_SECONDS = 0.03
"""A long read (the backlog after a pause in polling) is fed to the VAD in
frames of this length, so it sees the audio's real timing rather than one
averaged blob whose whole duration counts as "sustained"."""


class SessionState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    STOPPING = "stopping"


class ConversationSession:
    """One hands-free conversation loop instance.

    Not designed to be restarted after `stop()` -- create a fresh instance
    for each session (see `AgentBridge.start_session()`)."""

    def __init__(
        self,
        agent: Agent,
        voice: VoicePipeline,
        events: EventBus,
        *,
        poll_interval_seconds: float = 0.03,
        energy_threshold: float = 500.0,
        min_speech_seconds: float = 0.15,
        min_silence_seconds: float = 0.7,
        barge_in_energy_threshold: float = 900.0,
        max_utterance_seconds: float = 30.0,
        echo_margin_ratio: float = 1.5,
        max_reference_multiplier: float = 3.0,
        greeting: str | None = None,
        initial_noise_floor: float | None = None,
        sleep_matcher: SleepCommandMatcher | None = None,
        on_sleep: "Callable[[ConversationSession], None] | None" = None,
    ) -> None:
        """`greeting` (wake-word sessions only, see app/voice/wake.py) is
        spoken once through the normal interruptible speak path before the
        first listen; it is never sent to the Agent. `initial_noise_floor`
        seeds the room's ambient level (from standby) so the first VAD skips
        calibration -- which would otherwise overlap the greeting's own
        playback. `sleep_matcher` (only given when wake mode is on) checks
        each transcript before the Agent sees it: a standby command or
        closing statement is never sent to the Agent -- a short fixed
        acknowledgement is spoken, the loop ends, and `on_sleep(self)` asks
        the owner to end the session (see AgentBridge)."""
        self._agent = agent
        self._voice = voice
        self._events = events
        self._poll_interval = poll_interval_seconds
        self._energy_threshold = energy_threshold
        self._min_speech_seconds = min_speech_seconds
        self._min_silence_seconds = min_silence_seconds
        self._barge_in_energy_threshold = barge_in_energy_threshold
        self._max_utterance_seconds = max_utterance_seconds
        self._echo_margin_ratio = echo_margin_ratio
        self._max_reference_multiplier = max_reference_multiplier
        self._greeting = greeting
        self._sleep_matcher = sleep_matcher
        self._on_sleep = on_sleep

        self._state = SessionState.IDLE
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._noise_floor: float | None = initial_noise_floor
        """Carried across every VAD instance for the life of the session
        (listening and speaking phases alike) so only the very first
        utterance pays the one-time noise-floor calibration cost -- see
        app/voice/vad.py's module docstring. None until the first VAD in
        this session reports its learned floor."""
        self._turn_seq = 0
        """Diagnostic-only counter, incremented once per Agent.handle_user_message()
        call this session makes (see _run()) -- lets a log reviewer count
        exactly how many Agent turns a real recorded run produced, and
        correlate each one (via content_fingerprint) against the
        speech_start/barge-in log lines that led to it, to confirm or rule
        out a phantom (self-triggered) turn."""

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Idempotent: a second call while already active is a no-op. Must
        be called from a coroutine running on the loop that should host the
        session's background task (see AgentBridge, which always calls this
        from a coroutine already running on its background loop)."""
        if self.is_active:
            return
        self._stopping = False
        self._voice.start_continuous_listening()
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """Cancel the loop and release the mic/TTS cleanly. Safe to call
        even if the session was never started or is already stopped."""
        self._stopping = True
        await self._set_state(SessionState.STOPPING)
        self._voice.stop_speaking()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                log_extra(logger, 40, "session.run_loop_error_during_stop")
            self._task = None
        self._voice.stop_continuous_listening()
        await self._set_state(SessionState.IDLE)

    async def _set_state(self, state: SessionState) -> None:
        self._state = state
        log_extra(logger, 20, "session.state_changed", state=state.value)
        await self._events.publish(AgentEvent(type=EventType.SESSION_STATE, payload={"state": state.value}))

    async def _run(self) -> None:
        seed_vad: VoiceActivityDetector | None = None
        seed_bytes = b""
        if self._greeting and self._greeting.strip():
            log_extra(logger, 20, "session.greeting_started")
            seed_vad, seed_bytes = await self._speak_interruptibly(self._greeting)
        while not self._stopping:
            followed_barge_in = seed_vad is not None
            text = await self._capture_utterance(seed_vad, seed_bytes)
            seed_vad, seed_bytes = None, b""
            if self._stopping:
                break
            if not text or not text.strip():
                continue
            if self._sleep_matcher is not None:
                sleep = self._sleep_matcher.match(text)
                if sleep is not None:
                    await self._go_to_sleep(sleep)
                    return

            await self._set_state(SessionState.THINKING)
            self._turn_seq += 1
            log_extra(
                logger,
                20,
                "session.agent_call_started",
                turn=self._turn_seq,
                followed_barge_in=followed_barge_in,
                **content_fingerprint(text),
            )
            reply = await self._agent.handle_user_message(text)
            if self._stopping:
                break

            if reply and reply.strip():
                seed_vad, seed_bytes = await self._speak_interruptibly(reply)

    async def _go_to_sleep(self, sleep) -> None:
        log_extra(logger, 20, "session.sleep_requested", phrase=sleep.phrase)
        # Spoken through the same interruptible path as any reply (self-echo
        # guard included); an interruption doesn't cancel going to sleep.
        await self._speak_interruptibly(sleep.acknowledgement)
        if self._on_sleep is not None and not self._stopping:
            self._on_sleep(self)

    async def _capture_utterance(self, seed_vad: VoiceActivityDetector | None, seed_bytes: bytes) -> str | None:
        """Capture one utterance's audio and transcribe it.

        Audio arriving before `speech_start` fires (while the VAD is still
        deciding whether this is really speech) is not simply discarded --
        it's kept in a small, bounded rolling `preroll` buffer and spliced
        in once speech is confirmed, so the first word of the utterance
        isn't clipped by the debounce window (`min_speech_seconds`). Not
        needed for a barge-in continuation (`seed_vad` already active): that
        case starts already-confirmed, with its onset audio in `seed_bytes`.
        """
        await self._set_state(SessionState.LISTENING)
        vad = seed_vad or VoiceActivityDetector(
            sample_rate=self._voice.sample_rate,
            energy_threshold=self._energy_threshold,
            min_speech_seconds=self._min_speech_seconds,
            min_silence_seconds=self._min_silence_seconds,
            initial_noise_floor=self._noise_floor,
        )
        buffer = bytearray(seed_bytes)
        preroll = bytearray()
        max_preroll_bytes = max(1, int(self._min_speech_seconds * 2 * self._voice.sample_rate * 2))
        elapsed = 0.0

        ended = False
        while not self._stopping and not ended:
            # Echo-aware even while listening: the tail of Garvis's last words
            # (device buffers + room reverb) must not start a phantom utterance.
            chunk, residual = self._voice.read_microphone_chunk_with_echo()
            # A residual level describes the whole chunk; raw audio can be judged frame by frame.
            for piece in ([chunk] if residual is not None else self._frames(chunk)) if chunk else ():
                event = vad.process(piece, residual_energy=residual)
                if event == "silence":
                    # Not yet confirmed as speech -- retain it in the
                    # rolling preroll rather than the real buffer.
                    preroll.extend(piece)
                    if len(preroll) > max_preroll_bytes:
                        del preroll[: len(preroll) - max_preroll_bytes]
                elif event == "speech_start":
                    log_extra(
                        logger,
                        20,
                        "session.listening_speech_start",
                        state=self._state.value,
                        seeded_from_barge_in=seed_vad is not None,
                        vad_energy=vad.last_energy,
                        noise_floor=vad.noise_floor,
                        effective_threshold=vad.last_effective_threshold,
                        tts_playing=self._voice.is_speaking,
                    )
                    buffer.extend(preroll)
                    preroll.clear()
                    buffer.extend(piece)
                elif event == "speech":
                    buffer.extend(piece)
                elif event == "speech_end":
                    buffer.extend(piece)
                    ended = True
                    break
            if ended:
                break
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval
            if elapsed > self._max_utterance_seconds:
                break

        self._noise_floor = vad.noise_floor

        if self._stopping or not buffer:
            return None

        try:
            return await self._voice.transcribe_bytes(bytes(buffer))
        except SpeechToTextError as exc:
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))
            return None

    async def _speak_interruptibly(self, text: str) -> tuple[VoiceActivityDetector | None, bytes]:
        await self._set_state(SessionState.SPEAKING)
        try:
            await self._voice.speak(to_spoken_text(text))
        except TextToSpeechError as exc:
            await self._events.publish(AgentEvent(type=EventType.ERROR, payload={"message": str(exc)}))
            return None, b""

        vad = VoiceActivityDetector(
            sample_rate=self._voice.sample_rate,
            energy_threshold=self._barge_in_energy_threshold,
            min_speech_seconds=self._min_speech_seconds,
            min_silence_seconds=self._min_silence_seconds,
            initial_noise_floor=self._noise_floor,
            echo_margin_ratio=self._echo_margin_ratio,
            max_reference_multiplier=self._max_reference_multiplier,
        )
        # Mirrors _capture_utterance's preroll: audio read while the VAD is
        # still debouncing toward speech_start (event == "silence", even
        # though it may already be loud) would otherwise be discarded --
        # only the single chunk that finally crosses min_speech_seconds was
        # kept, clipping up to ~min_speech_seconds of the interruption's
        # true onset before it ever reached STT. See module docstring.
        preroll = bytearray()
        max_preroll_bytes = max(1, int(self._min_speech_seconds * 2 * self._voice.sample_rate * 2))

        # The mic wasn't read while Garvis was thinking and synthesizing, so all
        # of that audio is queued now. It was recorded while Garvis was silent:
        # judge it frame by frame at its raw level. Only speech still going on as
        # playback starts is an interruption; a sound that already ended (a cough,
        # a TV, the tail of the user's last words) must not cut the reply off.
        recent = bytearray()
        for frame in self._frames(self._voice.drain_microphone()):
            event = vad.process(frame)
            recent.extend(frame)
            if event == "speech_end":
                log_extra(
                    logger,
                    20,
                    "session.backlog_speech_ignored",
                    seconds=round(len(recent) / 2 / self._voice.sample_rate, 2),
                )
            if not vad.is_active and len(recent) > max_preroll_bytes:
                del recent[: len(recent) - max_preroll_bytes]
        if vad.is_active:
            return await self._barge_in(vad, bytes(recent), 0.0, source="backlog")
        preroll.extend(recent)

        while self._voice.is_speaking and not self._stopping:
            # With an echo guard, `residual` is the part of the mic signal
            # Garvis's own playback can't explain -- echo never reaches the
            # VAD. Without one (None), fall back to the loudness heuristic.
            chunk, residual = self._voice.read_microphone_chunk_with_echo()
            reference_energy = 0.0 if residual is not None else self._voice.reference_energy_now()
            for piece in ([chunk] if residual is not None else self._frames(chunk)) if chunk else ():
                event = vad.process(piece, reference_energy=reference_energy, residual_energy=residual)
                triggered = event == "speech_start"
                # DEBUG-only: silent at the default INFO log level, so this
                # never spams normal operation -- set GARVIS_LOG_LEVEL=DEBUG
                # for one diagnostic run to capture it (see the real-user
                # barge-in investigation).
                log_extra(
                    logger,
                    10,
                    f"[BARGE-IN] state={self._state.value} energy={vad.last_energy:.0f} "
                    f"noise={vad.noise_floor:.0f} playback={reference_energy:.0f} "
                    f"threshold={vad.last_effective_threshold:.0f} triggered={triggered}",
                )
                if triggered:
                    return await self._barge_in(vad, bytes(preroll) + piece, reference_energy, source="live")
                if event == "silence":
                    preroll.extend(piece)
                    if len(preroll) > max_preroll_bytes:
                        del preroll[: len(preroll) - max_preroll_bytes]
            await asyncio.sleep(self._poll_interval)

        self._noise_floor = vad.noise_floor
        return None, b""

    async def _barge_in(
        self, vad: VoiceActivityDetector, seed_bytes: bytes, reference_energy: float, *, source: str
    ) -> tuple[VoiceActivityDetector, bytes]:
        """Stop Garvis's playback now and hand the already-active VAD plus the
        interruption's onset audio to the next utterance capture."""
        log_extra(
            logger,
            20,
            "session.barge_in_triggered",
            state=self._state.value,
            source=source,
            vad_energy=vad.last_energy,
            noise_floor=vad.noise_floor,
            reference_energy=reference_energy,
            effective_threshold=vad.last_effective_threshold,
            tts_playing=self._voice.is_speaking,
        )
        self._voice.stop_speaking()
        await self._set_state(SessionState.INTERRUPTED)
        self._noise_floor = vad.noise_floor
        return vad, seed_bytes

    def _frames(self, chunk: bytes) -> list[bytes]:
        """`chunk` cut into ~_VAD_FRAME_SECONDS frames when it is long (a
        backlog); a normal short poll is returned whole."""
        size = max(2, int(self._voice.sample_rate * _VAD_FRAME_SECONDS) * 2)
        if len(chunk) <= 2 * size:
            return [chunk] if chunk else []
        return [chunk[i : i + size] for i in range(0, len(chunk), size)]
