"""EchoGuard: Garvis's own playback captured by the microphone must not look
like speech, while new speech -- alone or over the echo -- still does.
Deterministic synthetic signals (tests/echo_signals.py), streamed in the
block size and timestamps the real recorder produces."""

from __future__ import annotations

import numpy as np
import pytest

from app.voice.echo import EchoGuard
from app.voice.vad import VoiceActivityDetector
from tests.echo_signals import MIC_RATE, TTS_RATE, blocks, mix, noise, room_echo, speech_like

DAC = 10.0  # when the clip's first sample reached the output device
MIC_START = 9.5  # the recorder has been running for half a second already
ACOUSTIC = 0.04  # true speaker->mic delay beyond the DAC time
REPLY = speech_like(4.0, f0=110, seed=1)  # "Garvis"
USER = speech_like(1.6, f0=215, seed=2, level=5000)  # a different voice


def _timeline(seconds: float) -> int:
    return int(seconds * MIC_RATE)


def _echo_signal(length_s: float = 6.0, gain: float = 0.6, room: float = 150.0, extra=()) -> np.ndarray:
    n = _timeline(length_s)
    echo_at = _timeline(DAC - MIC_START + ACOUSTIC)
    return mix(n, (echo_at, room_echo(REPLY, 0.0, gain)), *extra) + noise(n, room)


def _run(
    signal: np.ndarray,
    guard: EchoGuard | None,
    *,
    end_at: float | None = None,
    block: int = 416,
    threshold: float = 900.0,
    clip_pcm=REPLY,
):
    clip = None
    if guard is not None:
        clip = guard.start_reference(clip_pcm, TTS_RATE, DAC)
        guard.end_reference(clip, end_at if end_at is not None else DAC + len(clip_pcm) / TTS_RATE)
    vad = VoiceActivityDetector(MIC_RATE, threshold, 0.15, 0.7, initial_noise_floor=150.0)
    starts, residuals = [], []
    for t, pcm in blocks(signal, MIC_START, block):
        residual = guard.process(pcm, t) if guard is not None else None
        residuals.append((t, residual))
        if vad.process(pcm, residual_energy=residual) == "speech_start":
            starts.append(round(t, 3))
            vad = VoiceActivityDetector(MIC_RATE, threshold, 0.15, 0.7, initial_noise_floor=vad.noise_floor)
    return starts, residuals


# ----------------------------------------------------------- echo only
def test_garvis_echo_alone_never_counts_as_speech():
    signal = _echo_signal()
    raw_starts, _ = _run(signal, None)
    assert len(raw_starts) >= 2, "the scenario must be hard: a plain VAD hears the echo as speech"
    starts, _ = _run(signal, EchoGuard())
    assert starts == []


def test_echo_is_ignored_on_repeated_replies_after_learning():
    guard = EchoGuard()
    for _ in range(3):  # the learned room profile carries over between replies
        starts, _ = _run(_echo_signal(), guard)
        assert starts == []
        guard.reset()


def test_background_noise_during_playback_causes_no_phantom_utterances():
    starts, _ = _run(_echo_signal(room=450.0), EchoGuard())
    assert starts == []


# ------------------------------------------------------------ real speech
def test_user_speech_without_playback_is_untouched():
    n = _timeline(4.0)
    signal = mix(n, (_timeline(1.0), USER)) + noise(n, 150.0)
    plain, _ = _run(signal, None)
    guard = EchoGuard()
    vad = VoiceActivityDetector(MIC_RATE, 900.0, 0.15, 0.7, initial_noise_floor=150.0)
    starts = []
    for t, pcm in blocks(signal, MIC_START):
        residual = guard.process(pcm, t)
        assert residual is None  # no reference: the raw level is used, unchanged
        if vad.process(pcm, residual_energy=residual) == "speech_start":
            starts.append(round(t, 3))
            vad = VoiceActivityDetector(MIC_RATE, 900.0, 0.15, 0.7, initial_noise_floor=vad.noise_floor)
    assert starts == plain and starts and abs(starts[0] - (MIC_START + 1.05)) < 0.3


@pytest.mark.parametrize("offset", [1.0, 2.2])
def test_user_speaking_over_garvis_triggers_barge_in_promptly(offset):
    onset = DAC + offset
    user_at = _timeline(onset - MIC_START)
    signal = _echo_signal(extra=[(user_at, USER)])
    starts, _ = _run(signal, EchoGuard())
    voice_starts = onset + 0.05 * 1  # speech_like starts its first syllable 50 ms in
    hits = [s for s in starts if voice_starts - 0.05 <= s <= voice_starts + 0.6]
    assert hits, f"barge-in missed (speech_starts: {starts})"
    assert hits[0] - voice_starts <= 0.4  # latency
    assert not [s for s in starts if s < voice_starts - 0.05]  # no false interruption before it


# ------------------------------------------------- playback state changes
def test_reference_is_released_after_playback_and_its_tail():
    guard = EchoGuard()
    end = DAC + len(REPLY) / TTS_RATE
    _, residuals = _run(_echo_signal(length_s=7.0), guard)
    late = [r for t, r in residuals if t > end + 0.6]
    assert late and all(r is None for r in late)  # back to the raw level: no stale reference
    assert guard.active_clip_count() == 0


def test_interrupted_playback_stops_masking_immediately_after_its_tail():
    """Barge-in stops playback at 1.0 s: the rest of the clip was never played,
    so speech after the echo tail must be judged on the raw level."""
    stop = DAC + 1.0
    n = _timeline(5.0)
    played = REPLY[: int(1.0 * TTS_RATE)]
    user_at = _timeline(stop + 0.35 - MIC_START)
    signal = mix(n, (_timeline(DAC - MIC_START + ACOUSTIC), room_echo(played, 0.0, 0.6)), (user_at, USER)) + noise(
        n, 150.0
    )
    starts, residuals = _run(signal, EchoGuard(), end_at=stop)
    assert all(r is None for t, r in residuals if t > stop + 0.45)
    assert [s for s in starts if stop + 0.3 <= s <= stop + 0.9], starts


def test_unknown_start_time_means_no_reference_yet():
    guard = EchoGuard()
    clip = guard.start_reference(REPLY, TTS_RATE, None)  # the player hasn't reported the DAC time yet
    pcm = np.zeros(416, dtype=np.int16).tobytes()
    assert guard.process(pcm, DAC) is None
    guard.set_reference_start(clip, DAC)
    assert guard.process(pcm, DAC + 0.026) is not None


# ------------------------------------------------------ timing robustness
@pytest.mark.parametrize("initial_delay", [-0.08, 0.3])
def test_delay_tracking_converges_from_a_wrong_estimate(initial_delay):
    guard = EchoGuard(initial_delay=initial_delay)
    _run(_echo_signal(length_s=6.5), guard)
    assert abs(guard.delay - ACOUSTIC) <= 0.03


@pytest.mark.parametrize("block", [160, 416, 1024])
def test_block_size_does_not_change_the_decision(block):
    user_at = _timeline(DAC + 1.6 - MIC_START)
    echo_only, _ = _run(_echo_signal(), EchoGuard(), block=block)
    with_user, _ = _run(_echo_signal(extra=[(user_at, USER)]), EchoGuard(), block=block)
    assert echo_only == []
    assert [s for s in with_user if DAC + 1.6 <= s <= DAC + 2.3]


def test_a_gap_or_repeated_block_restarts_framing_without_phantoms():
    guard = EchoGuard()
    clip = guard.start_reference(REPLY, TTS_RATE, DAC)
    guard.end_reference(clip, DAC + len(REPLY) / TTS_RATE)
    vad = VoiceActivityDetector(MIC_RATE, 900.0, 0.15, 0.7, initial_noise_floor=150.0)
    starts = 0
    stream = list(blocks(_echo_signal(), MIC_START))
    replayed = stream[:120] + stream[100:110] + stream[160:]  # a re-delivered stretch, then a dropped one
    for t, pcm in replayed:
        if vad.process(pcm, residual_energy=guard.process(pcm, t)) == "speech_start":
            starts += 1
    assert starts == 0
