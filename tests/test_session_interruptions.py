"""Barge-in regression tests in a simulated room with real time passing while
Garvis thinks (ConversationSession + VoicePipeline + EchoGuard).

Unlike tests/test_echo_session.py's Room, time here does not stop while the
session isn't reading the microphone: the agent "thinks" for a few seconds,
and everything the microphone heard meanwhile is waiting in the recorder's
queue when playback starts -- exactly as with the real MicrophoneRecorder.
Simulated audio only: no real microphone or speaker is involved.
"""

from __future__ import annotations

import asyncio
import wave

import numpy as np

from app.core.events import EventBus, EventType
from app.voice.echo import EchoGuard
from app.voice.pipeline import VoicePipeline
from app.voice.session import ConversationSession
from app.voice.stt.base import TranscriptionResult
from tests.echo_signals import MIC_RATE, noise, speech_like
from tests.test_echo_session import BLOCK, ReplyTTS, Room, RoomPlayer

THINK_SECONDS = 3.0
QUESTION = speech_like(0.8, f0=180, rate=MIC_RATE, seed=5, level=4000)


class LiveRoom(Room):
    """Room whose clock (`now`) can run ahead of what has been read: reads
    return every block captured since the last read (a backlog), or one new
    block when the reader is caught up."""

    def __init__(self, ambient: float = 150.0) -> None:
        super().__init__()
        self._noise = noise(MIC_RATE * 120, ambient)
        self.read_pos = self.now

    def _render(self, start: float) -> bytes:
        out = self._noise[self._noise_pos : self._noise_pos + BLOCK].copy()
        self._noise_pos = (self._noise_pos + BLOCK) % (len(self._noise) - BLOCK)
        for at, samples in self.sources:
            offset = int(round((start - at) * MIC_RATE))
            lo, hi = max(offset, 0), min(offset + BLOCK, len(samples))
            if hi > lo:
                out[lo - offset : hi - offset] += samples[lo:hi]
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    def read_available_timed(self):
        step = BLOCK / MIC_RATE
        if self.read_pos + step > self.now + 1e-9:
            self.now += step  # caught up: wait for the next block in real time
        blocks = []
        while self.read_pos + step <= self.now + 1e-9:
            blocks.append((self.read_pos, self._render(self.read_pos)))
            self.read_pos += step
        return blocks


class TimedPlayer(RoomPlayer):
    def __init__(self, room: Room) -> None:
        super().__init__(room)
        self.started_at: list[float] = []
        self.stopped_at: list[float] = []

    def play(self, wav_path, on_finished=None, on_started=None):
        super().play(wav_path, on_finished, on_started)
        self.started_at.append(self.room.now)

    def stop(self):
        if self.playing:
            self.stopped_at.append(self.room.now)
        super().stop()


class RecordingSTT:
    def __init__(self) -> None:
        self.durations: list[float] = []

    async def transcribe(self, audio_path):
        with wave.open(audio_path, "rb") as w:
            self.durations.append(w.getnframes() / w.getframerate())
        return TranscriptionResult(text=f"user utterance {len(self.durations)}")


class ThinkingAgent:
    """Takes THINK_SECONDS of room time per turn; `during_thinking(turn, t0)` may add sounds."""

    def __init__(self, room: LiveRoom, during_thinking=None) -> None:
        self.room = room
        self.turns: list[str] = []
        self._during = during_thinking

    async def handle_user_message(self, text):
        self.turns.append(text)
        if self._during is not None:
            self._during(len(self.turns), self.room.now)
        self.room.now += THINK_SECONDS
        return "Here is a reasonably long answer for you."


def _session(room: LiveRoom, during_thinking=None, initial_noise_floor: float | None = 150.0):
    stt = RecordingSTT()
    player = TimedPlayer(room)
    guard = EchoGuard()
    voice = VoicePipeline(
        stt=stt, tts=ReplyTTS(), recorder=room, player=player, echo_guard=guard, clock=lambda: room.now
    )
    events = EventBus()
    states: list[str] = []
    events.subscribe(lambda e: states.append(e.payload["state"]) if e.type == EventType.SESSION_STATE else None)
    agent = ThinkingAgent(room, during_thinking)
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
        initial_noise_floor=initial_noise_floor,
    )
    return session, agent, stt, player, states, guard


async def _run_until(session, room: Room, until: float) -> None:
    session.start()
    while room.now < until:
        await asyncio.sleep(0)
    await session.stop()


def test_drain_returns_the_whole_backlog_and_keeps_the_echo_guard_timeline():
    room = LiveRoom()
    guard = EchoGuard()
    voice = VoicePipeline(
        stt=RecordingSTT(),
        tts=ReplyTTS(),
        recorder=room,
        player=TimedPlayer(room),
        echo_guard=guard,
        clock=lambda: room.now,
    )
    room.now += 2.0  # two seconds pass unread
    pcm = voice.drain_microphone()
    assert len(pcm) // 2 == int(2.0 * MIC_RATE // BLOCK) * BLOCK
    assert (
        guard._pending_time is not None
        and abs(guard._pending_time + len(guard._pending) / MIC_RATE - room.read_pos) < 1e-6
    )


async def test_sound_that_ended_while_garvis_was_thinking_does_not_cut_off_the_reply():
    """The bug: the mic backlog from the thinking time was judged as ONE chunk
    (its average level, its full duration) the instant playback began."""
    room = LiveRoom()
    room.add(100.5, QUESTION)

    def afterthought(turn, t0):
        if turn == 1:  # 1 s of talking (a TV, the user's trailing words) that ends 1.5 s before Garvis speaks
            room.add(t0 + 0.5, speech_like(1.0, f0=200, rate=MIC_RATE, seed=8, level=4000))

    session, agent, stt, player, states, _ = _session(room, afterthought)
    await _run_until(session, room, 110.0)
    assert "interrupted" not in states
    assert agent.turns == ["user utterance 1"]  # no phantom turn, no repeated LLM request
    assert player.plays == 1 and not player.stopped_at  # the reply played out; nothing cut it short


async def test_user_still_talking_when_playback_starts_interrupts_at_once_with_full_onset():
    room = LiveRoom()
    room.add(100.5, QUESTION)

    def keeps_talking(turn, t0):
        if turn == 1:  # starts 0.8 s before playback, ends 0.4 s into it
            room.add(t0 + THINK_SECONDS - 0.8, speech_like(1.2, f0=215, rate=MIC_RATE, seed=2, level=5000))

    session, agent, stt, player, states, _ = _session(room, keeps_talking)
    await _run_until(session, room, 112.0)
    assert states.count("interrupted") == 1
    assert agent.turns == ["user utterance 1", "user utterance 2"]
    assert player.stopped_at[0] - player.started_at[0] < 0.1  # yielded before saying anything
    assert stt.durations[1] >= 1.2  # the words spoken before playback began were kept


async def test_interrupting_mid_reply_stops_playback_promptly_and_ends_the_echo_reference():
    room = LiveRoom()
    room.add(100.5, QUESTION)
    session, agent, stt, player, states, guard = _session(room)
    onset = {}

    async def interrupt():
        while player.plays == 0:
            await asyncio.sleep(0)
        onset["t"] = room.now + 1.0
        room.add(onset["t"], speech_like(1.2, f0=215, rate=MIC_RATE, seed=2, level=5000))

    helper = asyncio.ensure_future(interrupt())
    session.start()
    while not player.stopped_at:
        await asyncio.sleep(0)
    latency = player.stopped_at[0] - onset["t"]
    ended = [clip.end_time for clip in guard._clips]
    while room.now < 112.0:
        await asyncio.sleep(0)
    await session.stop()
    await helper
    assert 0.1 < latency < 0.4  # needs >= min_speech of the user's voice, then stops at once
    assert ended and all(t is not None for t in ended)  # no echo of unplayed audio is expected after the cut
    assert states.count("interrupted") == 1
    assert agent.turns == ["user utterance 1", "user utterance 2"]


async def test_two_interruptions_in_a_row_give_exactly_two_new_turns():
    room = LiveRoom()
    room.add(100.5, QUESTION)
    session, agent, stt, player, states, _ = _session(room)

    async def interrupt_each_reply():
        for reply in (1, 2):
            while player.plays < reply:
                await asyncio.sleep(0)
            # The same voice as the other interruption tests: detection of a burst of very short
            # syllables under loud echo depends on how they line up with Garvis's words (a known limit).
            room.add(room.now + 1.0, speech_like(1.2, f0=215, rate=MIC_RATE, seed=2, level=5000))

    helper = asyncio.ensure_future(interrupt_each_reply())
    await _run_until(session, room, 122.0)
    await helper
    assert states.count("interrupted") == 2
    assert agent.turns == ["user utterance 1", "user utterance 2", "user utterance 3"]
    assert player.plays == 3  # the third reply played to the end with no self-interruption


async def test_steady_fan_noise_and_short_knocks_do_not_interrupt_garvis():
    room = LiveRoom(ambient=600.0)  # louder than the listening threshold; calibrated away
    room.add(100.8, QUESTION)
    session, agent, stt, player, states, _ = _session(room, initial_noise_floor=None)

    async def knocks():
        while player.plays == 0:
            await asyncio.sleep(0)
        start = room.now
        burst = noise(int(0.06 * MIC_RATE), 8000.0, seed=4)  # a 60 ms knock: loud, far too short for speech
        for offset in (0.5, 1.4, 2.3):
            room.add(start + offset, burst)

    helper = asyncio.ensure_future(knocks())
    await _run_until(session, room, 112.0)
    await helper
    assert "interrupted" not in states
    assert agent.turns == ["user utterance 1"]
