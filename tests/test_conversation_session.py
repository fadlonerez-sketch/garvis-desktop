"""Tests for ConversationSession (app/voice/session.py) -- the hands-free
"Start Session" loop sitting on top of VoicePipeline + Agent.

Uses the REAL VoicePipeline and VoiceActivityDetector (both already tested
in isolation elsewhere) wired to fake recorder/STT/TTS/player collaborators,
so these exercise the actual orchestration logic (listen -> think -> speak
-> repeat, barge-in, cancellation) without any real microphone, model, or
speaker -- and without real wall-clock waiting (poll_interval_seconds is
set near-zero).
"""

from __future__ import annotations

import asyncio
import logging
import struct
import wave

import pytest

from app.core.events import AgentEvent, EventBus, EventType
from app.voice.pipeline import VoicePipeline
from app.voice.session import ConversationSession, SessionState
from app.voice.stt.base import TranscriptionResult

_SAMPLE_RATE = 16000


def _tone_chunk(seconds: float, amplitude: int) -> bytes:
    count = int(seconds * _SAMPLE_RATE)
    samples = [amplitude if i % 2 == 0 else -amplitude for i in range(count)]
    return struct.pack(f"<{count}h", *samples)


def _silence_chunk(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * _SAMPLE_RATE)


def _split(total_seconds: float, chunk_seconds: float, make_chunk) -> list[bytes]:
    """Split `total_seconds` of audio into ~`chunk_seconds`-sized pieces --
    mirrors how a real microphone delivers audio incrementally (via
    repeated small read_microphone_chunk() calls), rather than one giant
    chunk in a single call. This matters for the VAD's cold-start
    calibration window (see app/voice/vad.py): calibration consumes a
    bounded amount of *audio time* regardless of how many process() calls
    that spans, so realistically-sized chunks are needed for a test to
    reflect what production actually feeds the VAD."""
    chunks = []
    remaining = total_seconds
    while remaining > 1e-9:
        step = min(chunk_seconds, remaining)
        chunks.append(make_chunk(step))
        remaining -= step
    return chunks


def _speech_burst(seconds: float = 0.3, amplitude: int = 2000) -> list[bytes]:
    return _split(seconds, 0.05, lambda s: _tone_chunk(s, amplitude))


def _utterance_chunks(
    speech_seconds: float = 0.3, silence_seconds: float = 0.4, lead_in_seconds: float = 0.35
) -> list[bytes]:
    """A brief lead-in silence, then loud audio (crosses min_speech_seconds
    used by these tests), then enough silence to cross min_silence_seconds.

    The lead-in matters for a *fresh* (unseeded) VAD's cold-start
    calibration window (see app/voice/vad.py, default 0.3s): in real usage
    there is always some reaction-time gap between the mic opening (or the
    previous reply finishing) and the user actually starting to talk, so
    calibration naturally measures true ambient audio, not the user's own
    voice. Without a lead-in here, this synthetic tone starting at the very
    first sample would itself get absorbed as "the ambient level" during
    calibration -- indistinguishable, at that instant, from a continuous
    hum that was already running. That's a synthetic-test artifact (real
    speech is never perfectly constant-amplitude from sample zero the way
    a test tone is), not a realistic scenario, hence fixing it here rather
    than in the detector.
    """
    return (
        _split(lead_in_seconds, 0.05, _silence_chunk)
        + _speech_burst(speech_seconds)
        + _split(silence_seconds, 0.05, _silence_chunk)
    )


class _ScriptedRecorder:
    """Feeds a pre-queued list of chunks to read_available(); returns b""
    once exhausted (simulating "nothing new yet"), never blocking."""

    def __init__(self, chunks: list[bytes] | None = None, sample_rate: int = _SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._chunks = list(chunks or [])
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def read_available(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def close(self) -> None:
        self.closed = True

    def queue(self, *chunks: bytes) -> None:
        self._chunks.extend(chunks)


class _FakeSTT:
    def __init__(self, text: str = "what is two plus two") -> None:
        self.text = text
        self.transcribed_count = 0

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.transcribed_count += 1
        return TranscriptionResult(text=self.text)


class _FakeTTS:
    def __init__(self) -> None:
        self.synthesize_calls: list[str] = []
        self.stop_calls = 0

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        self.synthesize_calls.append(text)
        return output_path

    def stop(self) -> None:
        self.stop_calls += 1


class _FakePlayer:
    """is_playing counts down on each read (simulating playback finishing
    naturally after N polls) unless finish_after_polls is None, in which
    case it stays "playing" until stop() is called explicitly (for barge-in
    scenarios)."""

    def __init__(self, finish_after_polls: int | None = None) -> None:
        self.play_calls: list[str] = []
        self.stop_calls = 0
        self._playing = False
        self._finish_after_polls = finish_after_polls
        self._on_finished = None

    def play(self, wav_path: str, on_finished=None) -> None:
        self.play_calls.append(wav_path)
        self._playing = True
        self._on_finished = on_finished

    @property
    def is_playing(self) -> bool:
        if self._playing and self._finish_after_polls is not None:
            if self._finish_after_polls <= 0:
                self._finish_locked()
            else:
                self._finish_after_polls -= 1
        return self._playing

    def stop(self) -> None:
        self.stop_calls += 1
        if self._playing:
            self._finish_locked()

    def _finish_locked(self) -> None:
        self._playing = False
        if self._on_finished is not None:
            self._on_finished()
            self._on_finished = None


class _ToneWritingTTS:
    """Unlike _FakeTTS above (which never writes real audio to
    output_path), this writes an actual constant-amplitude WAV -- needed
    to exercise VoicePipeline.reference_energy_now() end to end, since that
    reads back what was actually synthesized (see app/voice/pipeline.py)."""

    def __init__(self, amplitude: int = 5000, seconds: float = 2.0, sample_rate: int = _SAMPLE_RATE) -> None:
        self._amplitude = amplitude
        self._seconds = seconds
        self._sample_rate = sample_rate

    async def synthesize_to_file(self, text: str, output_path: str) -> str:
        pcm = _tone_chunk(self._seconds, self._amplitude)
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(pcm)
        return output_path

    def stop(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[str] = []

    async def handle_user_message(self, text: str) -> str:
        self.calls.append(text)
        if not self._replies:
            return ""
        return self._replies.pop(0)


def _build(agent, recorder, player=None, events=None, **overrides):
    events = events or EventBus()
    player = player or _FakePlayer(finish_after_polls=3)
    voice = VoicePipeline(stt=_FakeSTT(), tts=_FakeTTS(), recorder=recorder, player=player)
    params = dict(
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
    )
    params.update(overrides)
    session = ConversationSession(agent=agent, voice=voice, events=events, **params)
    return session, voice, events, player


async def _run_briefly(session: ConversationSession, iterations: int = 200) -> None:
    """Yield control to the session's background task repeatedly -- stands
    in for real elapsed time since poll_interval_seconds=0."""
    for _ in range(iterations):
        await asyncio.sleep(0)


def _collect_states(events: EventBus) -> list[str]:
    states: list[str] = []

    async def _capture(event: AgentEvent) -> None:
        if event.type == EventType.SESSION_STATE:
            states.append(event.payload["state"])

    events.subscribe(_capture)
    return states


# --------------------------------------------------------------- lifecycle
@pytest.mark.asyncio
async def test_session_starts_and_transitions_to_listening():
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder)
    states = _collect_states(events)

    session.start()
    assert session.is_active
    assert recorder.started
    await _run_briefly(session)

    assert SessionState.LISTENING.value in states
    await session.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent_and_does_not_create_a_second_task():
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder)

    session.start()
    first_task = session._task
    session.start()  # second call while already active
    assert session._task is first_task  # no new task created
    assert recorder.started

    await session.stop()


@pytest.mark.asyncio
async def test_stop_releases_mic_and_returns_to_idle():
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder)

    session.start()
    await _run_briefly(session)
    await session.stop()

    assert recorder.closed
    assert not session.is_active
    assert session.state == SessionState.IDLE


@pytest.mark.asyncio
async def test_stop_without_start_does_not_raise():
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder)

    await session.stop()  # must not raise
    assert session.state == SessionState.IDLE


@pytest.mark.asyncio
async def test_restarting_a_session_after_stop_works_normally():
    agent = _FakeAgent(["four"])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder)

    session.start()
    await _run_briefly(session)
    await session.stop()
    assert not recorder.started or recorder.closed

    recorder.queue(*_utterance_chunks())
    session.start()
    assert session.is_active
    await _run_briefly(session, 500)
    await session.stop()

    assert agent.calls == ["what is two plus two"]


# -------------------------------------------------------------- one full turn
@pytest.mark.asyncio
async def test_full_turn_listens_thinks_speaks_and_returns_to_listening():
    agent = _FakeAgent(["four"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, events, player = _build(agent, recorder, player=_FakePlayer(finish_after_polls=2))
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 500)
    await session.stop()

    assert agent.calls == ["what is two plus two"]
    assert player.play_calls  # TTS actually played
    for expected in (
        SessionState.LISTENING.value,
        SessionState.THINKING.value,
        SessionState.SPEAKING.value,
    ):
        assert expected in states
    # after speaking finished naturally, the loop went back to listening
    # again (at least twice: once for the initial capture, once after)
    assert states.count(SessionState.LISTENING.value) >= 2


@pytest.mark.asyncio
async def test_no_duplicate_agent_invocation_for_one_utterance():
    agent = _FakeAgent(["four"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, events, player = _build(agent, recorder, player=_FakePlayer(finish_after_polls=2))

    session.start()
    await _run_briefly(session, 500)
    await session.stop()

    assert agent.calls == ["what is two plus two"]  # exactly once


@pytest.mark.asyncio
async def test_no_duplicate_tts_invocation_for_one_reply():
    agent = _FakeAgent(["four"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, events, player = _build(agent, recorder, player=_FakePlayer(finish_after_polls=2))

    session.start()
    await _run_briefly(session, 500)
    await session.stop()

    assert player.play_calls == [player.play_calls[0]]  # exactly one play() call
    assert len(player.play_calls) == 1


@pytest.mark.asyncio
async def test_onset_audio_before_speech_start_is_not_clipped():
    """Regression test: audio arriving before the VAD confirms speech_start
    (while it's still debouncing over min_speech_seconds) must not be
    silently dropped -- that would clip the first word of every utterance.
    Feed several short loud chunks that only cross the threshold after a
    few chunks, and confirm all of that audio reaches transcribe_bytes()."""
    agent = _FakeAgent(["ok"])
    recorder = _ScriptedRecorder()
    # A brief lead-in silence first, so the VAD's cold-start noise-floor
    # calibration (see app/voice/vad.py, default 0.3s) resolves against
    # true silence rather than against this test's own onset tone -- see
    # _utterance_chunks()'s docstring above for why that matters.
    recorder.queue(*_split(0.35, 0.05, _silence_chunk))
    # Each chunk is 0.05s -- individually under min_speech_seconds (0.1s),
    # so the VAD needs a couple of them before declaring speech_start.
    onset_chunks = [_tone_chunk(0.05, 2000) for _ in range(4)]
    recorder.queue(*onset_chunks, _silence_chunk(0.4))

    captured_audio_lengths: list[int] = []

    class _RecordingSTT:
        async def transcribe(self, audio_path: str) -> TranscriptionResult:
            import wave

            with wave.open(audio_path, "rb") as wav_file:
                captured_audio_lengths.append(wav_file.getnframes())
            return TranscriptionResult(text="what is two plus two")

    player = _FakePlayer(finish_after_polls=2)
    voice = VoicePipeline(stt=_RecordingSTT(), tts=_FakeTTS(), recorder=recorder, player=player)
    events = EventBus()
    session = ConversationSession(
        agent=agent,
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
    )

    session.start()
    await _run_briefly(session, 500)
    await session.stop()

    assert agent.calls == ["what is two plus two"]
    total_onset_frames = sum(len(c) // 2 for c in onset_chunks)
    # All 4 onset chunks' audio must have reached the transcriber, not just
    # the ones from the moment speech_start happened to fire.
    assert captured_audio_lengths[0] >= total_onset_frames


@pytest.mark.asyncio
async def test_silence_only_never_invokes_agent():
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder([_silence_chunk(0.05) for _ in range(20)])
    session, voice, events, player = _build(agent, recorder, max_utterance_seconds=0.001)

    session.start()
    await _run_briefly(session, 300)
    await session.stop()

    assert agent.calls == []


# ------------------------------------------------------------------ barge-in
@pytest.mark.asyncio
async def test_user_speech_during_playback_stops_tts_immediately():
    agent = _FakeAgent(["some long reply that would normally keep playing"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    # player never finishes on its own -- only a barge-in (stop()) should end it
    player = _FakePlayer(finish_after_polls=None)
    session, voice, events, _ = _build(agent, recorder, player=player)
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 300)  # first utterance -> reply -> starts speaking
    assert player.play_calls  # confirm we reached SPEAKING
    assert player.is_playing

    # user starts talking while Garvis is (still) speaking
    recorder.queue(_tone_chunk(0.3, 2000))
    await _run_briefly(session, 300)

    assert player.stop_calls >= 1  # TTS was actually cancelled, not just ignored
    assert SessionState.INTERRUPTED.value in states

    await session.stop()


@pytest.mark.asyncio
async def test_interrupting_utterance_is_processed_after_barge_in():
    """The words that triggered the barge-in must not be lost -- once the
    user finishes that utterance, the Agent must be called with it."""
    agent = _FakeAgent(["first reply", "second reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)
    stt = _FakeSTT("first thing")
    voice = VoicePipeline(stt=stt, tts=_FakeTTS(), recorder=recorder, player=player)
    events = EventBus()
    session = ConversationSession(
        agent=agent,
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
    )

    session.start()
    await _run_briefly(session, 300)
    assert agent.calls == ["first thing"]
    assert player.is_playing  # now speaking "first reply"

    # user interrupts, then finishes their new utterance
    stt.text = "second thing"
    recorder.queue(_tone_chunk(0.3, 2000), _silence_chunk(0.4))
    await _run_briefly(session, 500)

    assert agent.calls == ["first thing", "second thing"]
    await session.stop()


@pytest.mark.asyncio
async def test_barge_in_onset_audio_before_speech_start_is_not_clipped():
    """Regression test: the barge-in poll loop (_speak_interruptibly) must
    retain audio read while the VAD is still debouncing toward
    speech_start (min_speech_seconds), not just the single final chunk that
    crosses the threshold -- otherwise every barge-in-triggered utterance
    is missing its true onset, a classic Whisper hallucination trigger.
    Companion to test_onset_audio_before_speech_start_is_not_clipped, which
    covers the same preroll behavior for the (already-fixed) LISTENING
    path."""
    agent = _FakeAgent(["a long reply that would normally keep playing", "ok"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)  # only a real barge-in should stop it

    captured_audio_lengths: list[int] = []

    class _RecordingSTT:
        def __init__(self) -> None:
            self.text = "first thing"

        async def transcribe(self, audio_path: str) -> TranscriptionResult:
            with wave.open(audio_path, "rb") as wav_file:
                captured_audio_lengths.append(wav_file.getnframes())
            return TranscriptionResult(text=self.text)

    stt = _RecordingSTT()
    voice = VoicePipeline(stt=stt, tts=_FakeTTS(), recorder=recorder, player=player)
    events = EventBus()
    session = ConversationSession(
        agent=agent,
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
    )

    session.start()
    await _run_briefly(session, 300)  # first utterance -> reply -> now SPEAKING
    assert player.is_playing

    # user interrupts: several short loud chunks, each individually under
    # min_speech_seconds (0.1s), so the VAD needs more than one before
    # declaring speech_start -- mirrors _utterance_chunks' onset shape but
    # during the SPEAKING/barge-in phase instead of LISTENING.
    stt.text = "second thing"
    onset_chunks = [_tone_chunk(0.05, 2000) for _ in range(4)]
    recorder.queue(*onset_chunks, _silence_chunk(0.4))
    await _run_briefly(session, 500)

    assert agent.calls == ["first thing", "second thing"]
    total_onset_frames = sum(len(c) // 2 for c in onset_chunks)
    # The second (post-barge-in) transcription must include all onset
    # audio, not just the fraction captured from the moment speech_start
    # happened to fire.
    assert captured_audio_lengths[1] >= total_onset_frames

    await session.stop()


@pytest.mark.asyncio
async def test_playback_finishing_naturally_is_not_treated_as_interruption():
    agent = _FakeAgent(["a reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    # finishes on its own after a couple of polls, no loud audio queued
    player = _FakePlayer(finish_after_polls=2)
    session, voice, events, _ = _build(agent, recorder, player=player)
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 300)
    stop_calls_before_session_stop = player.stop_calls
    await session.stop()

    assert SessionState.INTERRUPTED.value not in states
    # stop() must not have been called mid-speech (only session.stop()'s own
    # final cleanup, if playback somehow hadn't already finished, may call it)
    assert stop_calls_before_session_stop == 0


@pytest.mark.asyncio
async def test_tts_completion_does_not_trigger_a_second_agent_invocation():
    """Regression test (duplicate-turn investigation): TTS playback ending
    naturally must return control to listening for the *next* utterance --
    it must never, by itself, cause another Agent.handle_user_message()
    call for the reply that was just spoken."""
    agent = _FakeAgent(["a reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=2)
    session, voice, events, _ = _build(agent, recorder, player=player)

    session.start()
    await _run_briefly(session, 300)
    await session.stop()

    assert len(agent.calls) == 1
    assert agent.calls == ["what is two plus two"]


# ------------------------------------------- continuous background noise (AC)
@pytest.mark.asyncio
async def test_continuous_background_noise_from_session_start_does_not_invoke_agent():
    """Reported regression: a continuous ambient sound (e.g. an air
    conditioner) already running when a session starts must not be
    repeatedly mistaken for user speech."""
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder()
    hum_amplitude = 600  # above the default 500 energy_threshold
    recorder.queue(*_split(2.0, 0.05, lambda s: _tone_chunk(s, hum_amplitude)))
    session, voice, events, player = _build(agent, recorder, max_utterance_seconds=1.0)

    session.start()
    await _run_briefly(session, 500)
    await session.stop()

    assert agent.calls == []


@pytest.mark.asyncio
async def test_continuous_background_noise_does_not_interrupt_speaking():
    """The full reported scenario end-to-end: the same continuous hum that
    was correctly rejected while listening must also not trigger a
    barge-in while Garvis is speaking -- the noise floor learned during
    listening carries over rather than resetting for the speaking phase."""
    agent = _FakeAgent(["a reply"])
    hum_amplitude = 600
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)  # only a real barge-in should stop it
    session, voice, events, _ = _build(agent, recorder, player=player)
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 300)
    assert agent.calls == ["what is two plus two"]
    assert player.is_playing  # now speaking the reply

    # the same continuous background noise keeps going while Garvis talks
    recorder.queue(*_split(1.0, 0.05, lambda s: _tone_chunk(s, hum_amplitude)))
    await _run_briefly(session, 300)

    assert SessionState.INTERRUPTED.value not in states
    assert player.is_playing  # never stopped by the background noise

    await session.stop()


@pytest.mark.asyncio
async def test_real_speech_still_interrupts_over_continuous_background_noise():
    """Companion to the test above: real, sufficiently louder speech must
    still interrupt even while the same continuous background noise is
    present -- the fix must reject the specific noise, not desensitize
    barge-in generally."""
    agent = _FakeAgent(["a reply"])
    hum_amplitude = 600
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)
    session, voice, events, _ = _build(agent, recorder, player=player)
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 300)
    assert player.is_playing

    # background noise continues, then the user actually speaks over it
    recorder.queue(*_split(0.5, 0.05, lambda s: _tone_chunk(s, hum_amplitude)))
    recorder.queue(*_speech_burst(0.3, amplitude=3000))
    await _run_briefly(session, 500)

    assert SessionState.INTERRUPTED.value in states
    assert not player.is_playing

    await session.stop()


# --------------------------------------------------- diagnostic logging
@pytest.mark.asyncio
async def test_agent_call_is_logged_with_turn_number_and_fingerprint(caplog):
    """Instrumentation added to prove/disprove the reported self-trigger
    voice loop: every Agent.handle_user_message() call this session makes
    must be logged with an incrementing turn counter and a content
    fingerprint, so a real recorded run's logs can be checked for more
    Agent calls than the user actually made."""
    agent = _FakeAgent(["four"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    session, voice, events, player = _build(agent, recorder, player=_FakePlayer(finish_after_polls=2))

    with caplog.at_level(logging.INFO, logger="garvis.voice.session"):
        session.start()
        await _run_briefly(session, 500)
        await session.stop()

    records = [r for r in caplog.records if r.getMessage() == "session.agent_call_started"]
    assert len(records) == 1
    assert records[0].extra_fields["turn"] == 1
    assert records[0].extra_fields["followed_barge_in"] is False
    assert records[0].extra_fields["content_hash"]


@pytest.mark.asyncio
async def test_barge_in_trigger_is_logged_with_full_diagnostics(caplog):
    """The exact diagnostic bundle requested for the self-trigger
    investigation: VAD energy, noise floor, reference (playback) energy,
    and the effective threshold it was judged against, at the moment a
    barge-in actually fires."""
    agent = _FakeAgent(["a long reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)
    session, voice, events, _ = _build(agent, recorder, player=player)

    with caplog.at_level(logging.INFO, logger="garvis.voice.session"):
        session.start()
        await _run_briefly(session, 300)
        recorder.queue(_tone_chunk(0.3, 2000))
        await _run_briefly(session, 300)
        await session.stop()

    records = [r for r in caplog.records if r.getMessage() == "session.barge_in_triggered"]
    assert len(records) == 1
    fields = records[0].extra_fields
    for key in (
        "state",
        "vad_energy",
        "noise_floor",
        "reference_energy",
        "effective_threshold",
        "tts_playing",
    ):
        assert key in fields


@pytest.mark.asyncio
async def test_per_poll_barge_in_diagnostic_is_silent_at_the_default_info_level(caplog):
    """The requested `[BARGE-IN] ...` per-poll line must not spam normal
    operation at the default INFO log level -- see app/voice/session.py's
    _speak_interruptibly."""
    agent = _FakeAgent(["a reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    recorder.queue(*_split(0.3, 0.03, _silence_chunk))
    player = _FakePlayer(finish_after_polls=None)
    session, voice, events, _ = _build(agent, recorder, player=player)

    with caplog.at_level(logging.INFO, logger="garvis.voice.session"):
        session.start()
        await _run_briefly(session, 300)
        await session.stop()

    assert not [r for r in caplog.records if r.getMessage().startswith("[BARGE-IN]")]


@pytest.mark.asyncio
async def test_per_poll_barge_in_diagnostic_is_available_at_debug_level(caplog):
    """Companion test: with GARVIS_LOG_LEVEL=DEBUG (a deliberate, temporary
    opt-in for a real diagnostic run), every SPEAKING-phase VAD decision
    must be logged with the exact fields requested for the real-user
    barge-in investigation."""
    agent = _FakeAgent(["a reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    # a real continuously-streaming microphone always yields *some* bytes
    # each poll (even near-silence) -- queue quiet chunks so the SPEAKING
    # loop actually has mic data to log a per-poll diagnostic for.
    recorder.queue(*_split(0.3, 0.03, _silence_chunk))
    player = _FakePlayer(finish_after_polls=None)
    session, voice, events, _ = _build(agent, recorder, player=player)

    with caplog.at_level(logging.DEBUG, logger="garvis.voice.session"):
        session.start()
        await _run_briefly(session, 300)
        await session.stop()

    barge_in_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[BARGE-IN]")]
    assert barge_in_lines
    for line in barge_in_lines:
        for field in ("state=", "energy=", "noise=", "playback=", "threshold=", "triggered="):
            assert field in line


# ------------------------------------------- self-echo (TTS speaker bleed)
@pytest.mark.asyncio
async def test_speaker_leakage_of_own_voice_does_not_trigger_barge_in():
    """The confirmed real-runtime root cause: speaker leakage of Garvis's
    own TTS output into the microphone can exceed even the raised
    barge-in threshold on real hardware, causing Garvis to "interrupt
    itself" repeatedly (see app/voice/vad.py). reference_energy_now() must
    let the barge-in VAD tell that leakage apart from a genuine
    interruption -- Case A/B of the required runtime behavior."""
    agent = _FakeAgent(["a long reply that Garvis is speaking aloud"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)  # only a real barge-in should stop it
    tts = _ToneWritingTTS(amplitude=5000, seconds=2.0)
    voice = VoicePipeline(stt=_FakeSTT(), tts=tts, recorder=recorder, player=player)
    events = EventBus()
    session = ConversationSession(
        agent=agent,
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
        echo_margin_ratio=1.5,
    )
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 300)  # first utterance -> reply -> now SPEAKING
    assert player.is_playing

    # speaker leakage of Garvis's own voice: loud enough (2000) to have
    # cleared the old fixed 900 barge-in threshold on real hardware, but
    # under the reference-scaled ceiling (min(5000*1.5, 900*3)=2700 with
    # the default max_reference_multiplier -- see app/voice/vad.py)
    recorder.queue(*_split(0.5, 0.05, lambda s: _tone_chunk(s, 2000)))
    await _run_briefly(session, 300)

    assert SessionState.INTERRUPTED.value not in states
    assert player.is_playing  # never stopped by the leakage
    assert player.stop_calls == 0

    await session.stop()


@pytest.mark.asyncio
async def test_real_speech_still_interrupts_over_own_speaker_leakage():
    """Companion to the above: genuine, louder user speech must still
    interrupt even while Garvis's own voice is loud -- the echo guard must
    reject the specific leakage case, not desensitize barge-in generally
    (Case C of the required runtime behavior)."""
    agent = _FakeAgent(["a long reply", "second reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)
    tts = _ToneWritingTTS(amplitude=5000, seconds=2.0)
    voice = VoicePipeline(stt=_FakeSTT(), tts=tts, recorder=recorder, player=player)
    events = EventBus()
    session = ConversationSession(
        agent=agent,
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
        echo_margin_ratio=1.5,
    )
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 300)
    assert player.is_playing

    # genuine user speech, well above the capped reference ceiling
    # (min(5000*1.5, 900*3)=2700 with the default max_reference_multiplier)
    recorder.queue(*_speech_burst(0.3, amplitude=10000))
    await _run_briefly(session, 300)

    assert SessionState.INTERRUPTED.value in states
    assert not player.is_playing

    await session.stop()


@pytest.mark.asyncio
async def test_realistic_user_speech_interrupts_even_during_a_loud_synthesized_reply():
    """The exact critical bug this task fixes, reproduced end to end:
    real Jarvis-High synthesis measured RMS 4,000-9,000 in ordinary speech
    segments. Uncapped, reference_energy * echo_margin_ratio would demand
    mic energy the user's own microphone could never realistically produce,
    silently defeating barge-in for virtually the whole reply. With
    max_reference_multiplier capping the reference term, realistic
    microphone-domain user speech (here, 3000 -- comfortably above the
    fixed 900 baseline but nowhere near the loud synthesis's own 9000+
    domain) must still interrupt."""
    agent = _FakeAgent(["a long reply", "second reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)
    tts = _ToneWritingTTS(amplitude=9000, seconds=2.0)  # matches measured Jarvis-High peaks
    voice = VoicePipeline(stt=_FakeSTT(), tts=tts, recorder=recorder, player=player)
    events = EventBus()
    session = ConversationSession(
        agent=agent,
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
        echo_margin_ratio=1.5,
        max_reference_multiplier=3.0,
    )
    states = _collect_states(events)

    session.start()
    await _run_briefly(session, 300)
    assert player.is_playing

    # realistic real speech: comfortably above the 900 baseline, but far
    # below the loud synthesis's own 9000+ RMS domain -- exactly the case
    # that was silently swallowed before the reference term was capped.
    recorder.queue(*_speech_burst(0.3, amplitude=3000))
    await _run_briefly(session, 300)

    assert SessionState.INTERRUPTED.value in states
    assert not player.is_playing

    await session.stop()


# --------------------------------------------------------------- end-session
@pytest.mark.asyncio
async def test_end_session_stops_tts_while_speaking():
    agent = _FakeAgent(["a reply"])
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer(finish_after_polls=None)
    session, voice, events, _ = _build(agent, recorder, player=player)

    session.start()
    await _run_briefly(session, 300)
    assert player.is_playing

    await session.stop()
    assert player.stop_calls >= 1
    assert not player._playing


@pytest.mark.asyncio
async def test_end_session_stops_listening():
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder)

    session.start()
    await _run_briefly(session)
    await session.stop()

    assert recorder.closed


@pytest.mark.asyncio
async def test_end_session_prevents_further_agent_invocations():
    """Regression test: audio that arrives (or was already queued) after
    stop() must never reach the Agent -- the loop task is cancelled, not
    merely asked to wind down on its next iteration."""
    agent = _FakeAgent([])
    recorder = _ScriptedRecorder()
    session, voice, events, player = _build(agent, recorder)

    session.start()
    await _run_briefly(session)
    await session.stop()

    # queue a would-be utterance after the session has ended
    recorder.queue(*_utterance_chunks())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert agent.calls == []
    assert not session.is_active


@pytest.mark.asyncio
async def test_end_session_while_thinking_cancels_the_pending_turn():
    """A slow/never-returning Agent call must not block session.stop()."""
    events = EventBus()
    recorder = _ScriptedRecorder(_utterance_chunks())
    player = _FakePlayer()
    voice = VoicePipeline(stt=_FakeSTT(), tts=_FakeTTS(), recorder=recorder, player=player)

    class _HangingAgent:
        async def handle_user_message(self, text: str) -> str:
            await asyncio.sleep(3600)  # never actually waited for
            return "unreachable"

    session = ConversationSession(
        agent=_HangingAgent(),
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=3.0,
    )

    session.start()
    await _run_briefly(session, 300)  # reaches THINKING, stuck awaiting the hang

    await asyncio.wait_for(session.stop(), timeout=2.0)  # must return promptly
    assert session.state == SessionState.IDLE
    assert recorder.closed
