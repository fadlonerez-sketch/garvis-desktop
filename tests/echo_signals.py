"""Deterministic synthetic audio for the echo-guard tests.

"Speech" = a harmonic series on a slowly gliding pitch, shaped by a few
formant-like peaks and cut into syllables with pauses -- enough structure
for band/bin analysis to behave as on real voices. The echo is the played
signal resampled to the mic rate, delayed, smeared by a short decaying room
impulse response and attenuated: a delayed, filtered copy, never equal to
the original waveform.
"""
from __future__ import annotations

import numpy as np

MIC_RATE = 16000
TTS_RATE = 22050


def speech_like(seconds: float, f0: float, rate: int = TTS_RATE, seed: int = 0, level: float = 6000.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pitch = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    formants = (rng.uniform(450, 800), rng.uniform(1100, 1800), rng.uniform(2300, 2900))
    signal = np.zeros(n)
    for h in range(1, int(3800 / f0)):
        freq = h * f0
        weight = sum(np.exp(-((freq - f) / 180.0) ** 2) for f in formants) + 0.05
        signal += weight / h ** 0.3 * np.sin(h * phase)
    envelope = np.zeros(n)
    position = int(0.05 * rate)
    while position < n:
        syllable = int(rng.uniform(0.12, 0.25) * rate)
        end = min(n, position + syllable)
        envelope[position:end] = np.hanning(2 * (end - position) + 2)[1:end - position + 1][: end - position] ** 0.3
        position = end + int(rng.uniform(0.05, 0.15) * rate)
    signal *= envelope
    return signal / (np.sqrt(np.mean(signal[envelope > 0.5] ** 2)) + 1e-9) * level


def to_mic_rate(x: np.ndarray, rate: int = TTS_RATE) -> np.ndarray:
    count = int(len(x) * MIC_RATE / rate)
    return np.interp(np.arange(count) * (rate / MIC_RATE), np.arange(len(x)), x)


def room_echo(played: np.ndarray, delay_seconds: float, gain: float = 0.5, rate: int = TTS_RATE) -> np.ndarray:
    """What the microphone hears of `played`, on the mic timeline (sample 0 =
    the moment the first played sample left the DAC)."""
    x = to_mic_rate(played, rate)
    taps = np.zeros(int(0.06 * MIC_RATE))
    taps[0] = 1.0
    taps[int(0.004 * MIC_RATE)] = 0.5
    taps[int(0.011 * MIC_RATE)] = -0.3
    decay = np.exp(-np.arange(len(taps)) / (0.015 * MIC_RATE))
    rng = np.random.default_rng(7)
    taps += 0.08 * rng.standard_normal(len(taps)) * decay
    echo = np.convolve(x, taps)[: len(x)] * gain
    return np.concatenate([np.zeros(int(delay_seconds * MIC_RATE)), echo])


def mix(length: int, *parts: tuple[int, np.ndarray]) -> np.ndarray:
    out = np.zeros(length)
    for offset, part in parts:
        if offset >= length:
            continue
        end = min(length, offset + len(part))
        out[offset:end] += part[: end - offset]
    return out


def noise(length: int, level: float, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(length) * level


def blocks(signal: np.ndarray, start_time: float, block: int = 416):
    """Mic blocks with capture timestamps, as MicrophoneRecorder delivers them."""
    pcm = np.clip(signal, -32768, 32767).astype(np.int16)
    for i in range(0, len(pcm) - block + 1, block):
        yield start_time + i / MIC_RATE, pcm[i:i + block].tobytes()
