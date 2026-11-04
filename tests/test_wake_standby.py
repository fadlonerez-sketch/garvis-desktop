"""Tests for StandbyListener (app/voice/wake.py): local wake-phrase standby
on the shared VoicePipeline microphone.

Same approach as tests/test_conversation_session.py (whose audio helpers
and fakes are reused): the REAL VoicePipeline and VoiceActivityDetector,
wired to a scripted recorder, a fake STT and a fake player, with a
near-zero poll interval and an injectable clock for the time-based guards.
"""
from __future__ import annotations

import wave

import pytest

from app.core.exceptions import SpeechToTextError
from app.voice.pipeline import VoicePipeline
from app.voice.stt.base import TranscriptionResult
from app.voice.wake import StandbyListener, WakeMatch
from tests.test_conversation_session import (
    _FakePlayer,
    _FakeTTS,
    _ScriptedRecorder,
    _silence_chunk,
    _speech_burst,
    _split,
    _tone_chunk,
    _run_briefly,
    _utterance_chunks,
)

_SAMPLE_RATE = 16000


class _RecordingSTT:
    """Returns `text` for every call and records each clip's duration."""

    def __init__(self, text: str = "Hey Garvis") -> None:
        self.text = text
        self.durations: list[float] = []

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        with wave.open(audio_path, "rb") as wav_file:
            self.durations.append(wav_file.getnframes() / wav_file.getframerate())
        return TranscriptionResult(text=self.text)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _wake_utterance(speech_seconds: float = 0.6) -> list[bytes]:
    return _utterance_chunks(speech_seconds=speech_seconds, silence_seconds=0.4)


def _build(chunks=None, stt_text="Hey Garvis", player=None, **overrides):
    recorder = _ScriptedRecorder(chunks)
    stt = _RecordingSTT(stt_text)
    player = player or _FakePlayer(finish_after_polls=None)
    voice = VoicePipeline(stt=stt, tts=_FakeTTS(), recorder=recorder, player=player)
    wakes: list[WakeMatch] = []
    clock = _Clock()
    params = dict(
        poll_interval_seconds=0.0,
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        min_silence_seconds=0.3,
        min_segment_seconds=0.4,
        max_segment_seconds=2.0,
        cooldown_seconds=2.0,
        playback_tail_seconds=0.5,
        clock=clock,
    )
    params.update(overrides)
    standby = StandbyListener(voice, wakes.append, **params)
    return standby, recorder, stt, player, wakes, clock


@pytest.mark.asyncio
async def test_silence_never_invokes_stt():
    standby, recorder, stt, _, wakes, _ = _build(_split(3.0, 0.05, _silence_chunk))
    standby.start()
    await _run_briefly(standby, 300)
    await standby.stop()
    assert stt.durations == []
    assert wakes == []


@pytest.mark.asyncio
async def test_continuous_background_noise_never_invokes_stt():
    """Adaptive noise floor preserved: a hum present from the start is
    calibrated away instead of being transcribed over and over."""
    standby, _, stt, _, wakes, _ = _build(_split(4.0, 0.05, lambda s: _tone_chunk(s, 2000)))
    standby.start()
    await _run_briefly(standby, 300)
    await standby.stop()
    assert stt.durations == []
    assert wakes == []


@pytest.mark.asyncio
async def test_wake_phrase_segment_is_transcribed_once_and_fires_wake_once():
    standby, _, stt, _, wakes, _ = _build(_wake_utterance())
    standby.start()
    await _run_briefly(standby, 300)
    await standby.stop()
    assert len(stt.durations) == 1
    assert [w.phrase for w in wakes] == ["Hey Garvis"]


@pytest.mark.asyncio
async def test_non_wake_speech_is_transcribed_but_does_not_wake():
    standby, _, stt, _, wakes, _ = _build(_wake_utterance(), stt_text="what time is it")
    standby.start()
    await _run_briefly(standby, 300)
    await standby.stop()
    assert len(stt.durations) == 1
    assert wakes == []


@pytest.mark.asyncio
async def test_segment_too_short_for_a_wake_phrase_is_not_transcribed():
    chunks = _split(0.35, 0.05, _silence_chunk) + _speech_burst(0.15) + _split(0.4, 0.05, _silence_chunk)
    standby, _, stt, _, wakes, _ = _build(chunks, min_segment_seconds=0.6)
    standby.start()
    await _run_briefly(standby, 300)
    await standby.stop()
    assert stt.durations == []
    assert wakes == []


@pytest.mark.asyncio
async def test_long_segment_is_capped_and_transcribed_only_once():
    chunks = _split(0.35, 0.05, _silence_chunk) + _speech_burst(6.0) + _split(0.4, 0.05, _silence_chunk)
    standby, _, stt, _, _, _ = _build(chunks, stt_text="blah blah", max_segment_seconds=2.0)
    standby.start()
    await _run_briefly(standby, 400)
    await standby.stop()
    assert len(stt.durations) == 1
    assert stt.durations[0] <= 2.0 + 1e-6


@pytest.mark.asyncio
async def test_wake_in_the_opening_of_a_long_segment_is_still_detected():
    """'Hey Garvis open Chrome ...' in one breath: the capped opening is
    evaluated, and the remainder travels on the WakeMatch (unused in v1)."""
    chunks = _split(0.35, 0.05, _silence_chunk) + _speech_burst(5.0) + _split(0.4, 0.05, _silence_chunk)
    standby, _, stt, _, wakes, _ = _build(chunks, stt_text="Hey Garvis, open Chrome")
    standby.start()
    await _run_briefly(standby, 400)
    await standby.stop()
    assert len(stt.durations) == 1
    assert len(wakes) == 1
    assert wakes[0].remainder == "open chrome"


@pytest.mark.asyncio
async def test_cooldown_prevents_a_double_wake():
    standby, recorder, stt, _, wakes, clock = _build(_wake_utterance())
    standby.start()
    await _run_briefly(standby, 300)
    assert len(wakes) == 1

    recorder.queue(*_wake_utterance())  # clock not advanced: still cooling down
    await _run_briefly(standby, 300)
    assert len(wakes) == 1
    assert len(stt.durations) == 1  # suppressed audio is never transcribed

    clock.now += 2.5
    recorder.queue(*_wake_utterance())
    await _run_briefly(standby, 300)
    await standby.stop()
    assert len(wakes) == 2


@pytest.mark.asyncio
async def test_audio_is_ignored_while_garvis_is_speaking_and_during_the_tail():
    player = _FakePlayer(finish_after_polls=None)
    standby, recorder, stt, _, wakes, clock = _build(player=player)
    standby.start()
    recorder.queue(*_split(0.35, 0.05, _silence_chunk))  # calibrate first
    await _run_briefly(standby, 100)

    player.play("reply.wav")  # Garvis talking, e.g. a typed turn's spoken reply
    recorder.queue(*_wake_utterance())  # its own "...Garvis is standing by." leaking in
    await _run_briefly(standby, 300)
    assert stt.durations == []

    player.stop()
    recorder.queue(*_wake_utterance())  # clock not advanced: inside the tail
    await _run_briefly(standby, 300)
    assert stt.durations == []

    clock.now += 1.0
    recorder.queue(*_wake_utterance())
    await _run_briefly(standby, 300)
    await standby.stop()
    assert len(stt.durations) == 1
    assert len(wakes) == 1


@pytest.mark.asyncio
async def test_speech_cut_off_by_playback_is_discarded_not_evaluated():
    player = _FakePlayer(finish_after_polls=None)
    standby, recorder, stt, _, wakes, clock = _build(player=player)
    standby.start()
    recorder.queue(*_split(0.35, 0.05, _silence_chunk), *_speech_burst(0.5))
    await _run_briefly(standby, 200)
    player.play("reply.wav")
    recorder.queue(*_split(0.4, 0.05, _silence_chunk))
    await _run_briefly(standby, 200)
    player.stop()
    clock.now += 1.0
    recorder.queue(*_split(0.4, 0.05, _silence_chunk))
    await _run_briefly(standby, 200)
    await standby.stop()
    assert stt.durations == []
    assert wakes == []


@pytest.mark.asyncio
async def test_stop_releases_the_microphone():
    standby, recorder, _, _, _, _ = _build()
    standby.start()
    assert recorder.started and standby.is_active
    await standby.stop()
    assert recorder.closed
    assert not standby.is_active


@pytest.mark.asyncio
async def test_stop_without_start_is_safe_and_repeatable():
    standby, _, _, _, _, _ = _build()
    await standby.stop()
    await standby.stop()
    assert not standby.is_active


@pytest.mark.asyncio
async def test_start_is_idempotent():
    standby, _, _, _, _, _ = _build()
    standby.start()
    task = standby._task
    standby.start()
    assert standby._task is task
    await standby.stop()


@pytest.mark.asyncio
async def test_microphone_failure_is_raised_from_start():
    class _BrokenRecorder(_ScriptedRecorder):
        def start(self) -> None:
            raise SpeechToTextError("no microphone")

    voice = VoicePipeline(stt=_RecordingSTT(), tts=_FakeTTS(), recorder=_BrokenRecorder(), player=_FakePlayer())
    standby = StandbyListener(voice, lambda match: None, poll_interval_seconds=0.0)
    with pytest.raises(SpeechToTextError):
        standby.start()
    assert not standby.is_active


@pytest.mark.asyncio
async def test_noise_floor_is_learned_from_the_room():
    standby, recorder, _, _, _, _ = _build(_split(1.0, 0.05, lambda s: _tone_chunk(s, 120)))
    assert standby.noise_floor is None
    standby.start()
    await _run_briefly(standby, 100)
    await standby.stop()
    assert standby.noise_floor is not None
    assert 60 < standby.noise_floor < 200


@pytest.mark.asyncio
async def test_noise_floor_survives_across_segments():
    standby, recorder, _, _, _, _ = _build(_split(1.0, 0.05, lambda s: _tone_chunk(s, 120)))
    standby.start()
    await _run_briefly(standby, 100)
    learned = standby.noise_floor
    recorder.queue(*_speech_burst(0.6), *_split(0.5, 0.05, lambda s: _tone_chunk(s, 120)))
    await _run_briefly(standby, 200)
    await standby.stop()
    assert standby.noise_floor == pytest.approx(learned, rel=0.5)
    assert not standby._vad.is_calibrating  # seeded, never re-calibrated


@pytest.mark.asyncio
async def test_warm_up_loads_stt_without_waking():
    standby, _, stt, _, wakes, _ = _build(warm_up_stt=True)
    standby.start()
    await _run_briefly(standby, 50)
    await standby.stop()
    assert len(stt.durations) == 1  # one silent warm-up clip
    assert wakes == []


@pytest.mark.asyncio
async def test_stt_failure_on_a_candidate_does_not_kill_standby():
    class _FlakySTT(_RecordingSTT):
        def __init__(self) -> None:
            super().__init__("Hey Daddy")
            self.calls = 0

        async def transcribe(self, audio_path: str) -> TranscriptionResult:
            self.calls += 1
            if self.calls == 1:
                raise SpeechToTextError("decoder hiccup")
            return await super().transcribe(audio_path)

    recorder = _ScriptedRecorder(_wake_utterance())
    stt = _FlakySTT()
    voice = VoicePipeline(stt=stt, tts=_FakeTTS(), recorder=recorder, player=_FakePlayer())
    wakes: list[WakeMatch] = []
    standby = StandbyListener(
        voice, wakes.append, poll_interval_seconds=0.0, min_speech_seconds=0.1, min_silence_seconds=0.3
    )
    standby.start()
    await _run_briefly(standby, 300)
    assert wakes == [] and standby.is_active
    recorder.queue(*_wake_utterance())
    await _run_briefly(standby, 300)
    await standby.stop()
    assert [w.phrase for w in wakes] == ["Hey Daddy"]


# ------------------------------------------- return from a session (sleep)
@pytest.mark.asyncio
async def test_initial_quiet_window_ignores_the_tail_of_the_last_reply():
    """Standby restarted right after a sleep acknowledgement must not hear
    the end of "Goodnight, Sir." -- a fresh listener has no playback record,
    so start() opens with a quiet window instead."""
    standby, recorder, stt, _, wakes, clock = _build(initial_quiet_seconds=0.5)
    standby.start()
    recorder.queue(*_wake_utterance())  # clock not advanced: inside the window
    await _run_briefly(standby, 300)
    assert stt.durations == []

    clock.now += 1.0
    recorder.queue(*_wake_utterance())
    await _run_briefly(standby, 300)
    await standby.stop()
    assert len(stt.durations) == 1
    assert len(wakes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("own_words", ["Goodnight, Sir.", "Entering standby, Sir."])
async def test_garvis_acknowledgement_heard_by_standby_does_not_wake_it(own_words):
    standby, _, stt, _, wakes, _ = _build(_wake_utterance(), stt_text=own_words)
    standby.start()
    await _run_briefly(standby, 300)
    await standby.stop()
    assert len(stt.durations) == 1  # even if it slipped through the guards...
    assert wakes == []  # ...it is not a wake phrase
