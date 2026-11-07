"""End to end through ConversationSession + VoicePipeline + EchoGuard in a
simulated room: Garvis's reply really comes back through the "microphone"
(delayed, filtered, attenuated) from the moment the player starts it, and
stops when playback stops. No mocked playback-state booleans decide anything
here -- the mic signal does."""

from __future__ import annotations

import asyncio
import wave

import numpy as np

from app.core.events import EventBus, EventType
from app.voice.echo import EchoGuard
from app.voice.pipeline import VoicePipeline
from app.voice.session import ConversationSession
from app.voice.stt.base import TranscriptionResult
from tests.echo_signals import MIC_RATE, TTS_RATE, noise, room_echo, speech_like

BLOCK = 416
DAC_LATENCY = 0.05
ACOUSTIC = 0.04
REPLY = speech_like(3.0, f0=110, seed=1)
USER = speech_like(1.2, f0=215, seed=2, level=5000)


class Room:
    """The microphone's view of the world on a deterministic timeline."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sources: list[list] = []  # [start_time, samples]
        self.recording = False
        self._noise = noise(MIC_RATE * 120, 150.0)
        self._noise_pos = 0

    def add(self, at: float, samples: np.ndarray) -> list:
        source = [at, samples]
        self.sources.append(source)
        return source

    def cut(self, source: list, at: float) -> None:
        keep = max(0, int((at + ACOUSTIC + 0.06 - source[0]) * MIC_RATE))  # what already left the speaker (+ ring-down)
        source[1] = source[1][:keep]

    # MicrophoneRecorder interface ------------------------------------
    sample_rate = MIC_RATE

    @property
    def is_recording(self) -> bool:
        return self.recording

    def start(self) -> None:
        self.recording = True

    def close(self) -> None:
        self.recording = False

    def read_available_timed(self):
        start = self.now
        out = self._noise[self._noise_pos : self._noise_pos + BLOCK].copy()
        self._noise_pos = (self._noise_pos + BLOCK) % (len(self._noise) - BLOCK)
        for at, samples in self.sources:
            offset = int(round((start - at) * MIC_RATE))
            lo, hi = max(offset, 0), min(offset + BLOCK, len(samples))
            if hi > lo:
                out[lo - offset : hi - offset] += samples[lo:hi]
        self.now += BLOCK / MIC_RATE
        return [(start, np.clip(out, -32768, 32767).astype(np.int16).tobytes())]

    def read_available(self) -> bytes:
        return b"".join(pcm for _, pcm in self.read_available_timed())


class RoomPlayer:
    """Plays into the Room: the echo of the clip arrives ACOUSTIC after the
    DAC time it reports; stop() silences the rest."""

    def __init__(self, room: Room) -> None:
        self.room = room
        self.playing = False
        self.plays = 0
        self.stops = 0
        self._source = None
        self._end = 0.0
        self._on_finished = None

    def play(self, wav_path, on_finished=None, on_started=None):
        with wave.open(wav_path, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
            rate = w.getframerate()
        dac = self.room.now + DAC_LATENCY
        self._source = self.room.add(dac, room_echo(pcm, ACOUSTIC, gain=0.6, rate=rate))
        self._end = dac + len(pcm) / rate
        self._on_finished = on_finished
        self.playing = True
        self.plays += 1
        if on_started is not None:
            on_started(dac)

    @property
    def is_playing(self) -> bool:
        if self.playing and self.room.now >= self._end:
            self._finish()
        return self.playing

    def stop(self):
        self.stops += 1
        if self.playing and self._source is not None:
            self.room.cut(self._source, self.room.now)
        self._finish()

    def _finish(self):
        if self.playing:
            self.playing = False
            callback, self._on_finished = self._on_finished, None
            if callback:
                callback()


class ReplyTTS:
    async def synthesize_to_file(self, text, output_path):
        with wave.open(output_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TTS_RATE)
            w.writeframes(np.clip(REPLY, -32768, 32767).astype(np.int16).tobytes())
        return output_path

    def stop(self):
        pass


class CountingSTT:
    def __init__(self):
        self.calls = 0

    async def transcribe(self, audio_path):
        self.calls += 1
        return TranscriptionResult(text=f"user utterance {self.calls}")


class Agent:
    def __init__(self):
        self.turns: list[str] = []

    async def handle_user_message(self, text):
        self.turns.append(text)
        return "Here is a reasonably long answer for you."


def _session(room: Room, guard: EchoGuard | None):
    stt = CountingSTT()
    player = RoomPlayer(room)
    voice = VoicePipeline(
        stt=stt, tts=ReplyTTS(), recorder=room, player=player, echo_guard=guard, clock=lambda: room.now
    )
    events = EventBus()
    states: list[str] = []
    events.subscribe(lambda e: states.append(e.payload["state"]) if e.type == EventType.SESSION_STATE else None)
    agent = Agent()
    session = ConversationSession(
        agent=agent,
        voice=voice,
        events=events,
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.15,
        min_silence_seconds=0.5,
        barge_in_energy_threshold=900.0,
        max_utterance_seconds=5.0,
        initial_noise_floor=150.0,
    )
    return session, agent, stt, player, states


async def _run_until(session, room: Room, until: float) -> None:
    session.start()
    while room.now < until:
        await asyncio.sleep(0)
    await session.stop()


async def test_garvis_hearing_itself_starts_no_turn_no_stt_no_interruption():
    room = Room()
    room.add(100.5, speech_like(0.8, f0=180, seed=5, level=4000))  # the user's question
    session, agent, stt, player, states = _session(room, EchoGuard())
    await _run_until(session, room, 107.0)  # the whole reply plays and its echo tail passes
    assert agent.turns == ["user utterance 1"]
    assert stt.calls == 1
    assert player.plays == 1 and player.stops == 1  # the final stop() is session shutdown, not barge-in
    assert "interrupted" not in states


async def test_without_the_guard_the_same_echo_interrupts_garvis():
    """Proves the scenario reproduces the bug: identical room, guard off."""
    room = Room()
    room.add(100.5, speech_like(0.8, f0=180, seed=5, level=4000))
    session, agent, stt, player, states = _session(room, None)
    await _run_until(session, room, 107.0)
    assert "interrupted" in states and len(agent.turns) > 1


async def test_user_speaking_over_garvis_interrupts_once_and_gets_one_turn():
    room = Room()
    room.add(100.5, speech_like(0.8, f0=180, seed=5, level=4000))
    session, agent, stt, player, states = _session(room, EchoGuard())

    async def interrupt_when_garvis_talks():
        while player.plays == 0:
            await asyncio.sleep(0)
        room.add(room.now + 1.0, USER)  # 1 s into Garvis's reply

    helper = asyncio.ensure_future(interrupt_when_garvis_talks())
    await _run_until(session, room, 111.0)
    await helper
    assert states.count("interrupted") == 1
    assert agent.turns == ["user utterance 1", "user utterance 2"]  # exactly one new turn, no duplicates
    assert stt.calls == 2
    assert player.plays == 2  # Garvis answered the interruption; that reply's echo started nothing


async def test_session_cleanup_leaves_no_reference_behind():
    room = Room()
    room.add(100.5, speech_like(0.8, f0=180, seed=5, level=4000))
    guard = EchoGuard()
    session, agent, stt, player, states = _session(room, guard)
    session.start()
    while player.plays == 0:
        await asyncio.sleep(0)
    while room.now < 103.0:  # stop mid-reply
        await asyncio.sleep(0)
    await session.stop()
    assert not player.is_playing
    assert all(clip.end_time is not None for clip in guard._clips)  # the reference was closed at the stop
