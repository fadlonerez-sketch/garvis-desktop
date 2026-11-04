"""Playback-reference-aware echo discrimination for barge-in.

Problem: while Garvis speaks through loudspeakers, the microphone hears
Garvis. An energy-only VAD can't tell that echo from the user talking, so
Garvis "interrupted itself" and transcribed its own words as new turns.

What this does instead -- using the exact PCM Garvis is playing:

  mic(t) = H{ ref(t - delay) } + user(t) + noise(t)

1. Timing. The player reports when the clip's first sample reached the
   output device (PortAudio DAC time, the same clock as time.perf_counter()
   on Windows); mic blocks are timestamped on that clock as they arrive.
   The remaining delay (device + acoustic path, here ~40-60 ms, plus
   recorder jitter) is tracked continuously by correlating the log-energy
   envelopes of mic and reference over the last ~2 s (`_update_delay`).
2. Echo path. H is modelled per FFT bin (31 Hz, 100-4000 Hz) as a power gain G_k
   (speaker + room + mic response), learned as a high percentile of
   mic / reference-window power over recent frames where that band of the
   reference is strong. Echo spreads over neighbouring frames (reverb,
   jitter), so per-frame ratios scatter; echo-only frames make up the bulk
   of that distribution. (Frames are NOT excluded based on the current
   model's verdict: that feedback loop can only ever lower the gain. A
   barge-in lasts ~0.2 s before playback stops -- a tiny share of the 8 s
   buffer -- so it can't move a high percentile.) Only frames clearly above the room's background level count
   (an HDMI/DisplayPort sink waking up can swallow the first ~0.5 s of a
   clip -- the reference "plays" while nothing comes out -- which would
   otherwise teach a near-zero gain), and until enough frames have been
   seen the conservative initial gain (echo assumed as loud as the
   reference) stays in force. The learned path persists across replies.
3. Echo estimate. For each 20 ms mic frame (10 ms hop), expected echo per
   bin = G_k x the loudest reference power over the preceding
   `tail_seconds` (reverberation, speaker ring-down) and `lookahead`
   (timing jitter) at the tracked delay. Bin resolution matters: a user's
   voice under loud echo shows up between Garvis's pitch harmonics, which
   coarse bands average away (measured on real echo: 64/86/98% of a mixed-
   in voice's frames detected at mic levels 1000/2000/3500 RMS with bins,
   versus 42/69/88% with 13 bands).
4. Residual. Energy above `over_subtraction` x expected echo is energy
   the playback cannot explain: the user's voice (different pitch,
   formants, timing), or other new sound. The first `doubt_fraction` of a
   frame's energy is not counted: echo-model error grows with echo level
   (speaker/mic compression), and at 8000 RMS of echo even 5% unexplained
   would otherwise look like speech. Its time-domain RMS equivalent
   is handed to the existing VoiceActivityDetector instead of the raw
   level, so the adaptive noise floor, thresholds and minimum durations
   all apply unchanged -- and Garvis's own echo never even reaches them.

This is echo SUPPRESSION (deciding what is new), not echo CANCELLATION
(subtracting a waveform): on the target hardware (a laptop mic array with
its own DSP, monitor speakers) the waveform correlation between playback
and mic was only ~0.27, too low for a linear adaptive canceller, while the
band-energy model explained 100% of echo-only frames once aligned. The
trade-off: a quiet user voice completely masked by much louder echo in
every band can't be separated -- it's detected in the gaps between Garvis's
words or when it's at least comparably loud. A headset removes echo
entirely. When no reference is active (Garvis silent, tail elapsed) the
raw level passes through untouched.

Pure numpy, streaming, ~0.1 ms per 10 ms frame.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

_MIN_HZ = 100
_MAX_HZ = 4000


def resample_linear(pcm: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """Linear-interpolation resampling -- plenty for band energies below 4.3 kHz."""
    if rate == target_rate or len(pcm) == 0:
        return pcm.astype(np.float64)
    count = int(len(pcm) * target_rate / rate)
    return np.interp(np.arange(count) * (rate / target_rate), np.arange(len(pcm)), pcm.astype(np.float64))


@dataclass
class _Clip:
    """One played clip on the shared clock."""

    bands: np.ndarray  # (frames, bins) power per hop on the clip's own timeline
    envelope: np.ndarray  # (frames,) log energy, for delay tracking
    start_time: float | None
    audible_level: float = 0.0  # total power above which the clip is "sounding" (vs a pause)
    band_levels: np.ndarray | None = None  # per bin: power above which that bin is "strong"
    end_time: float | None = None  # when playback was cut short / finished (samples after it never played)
    hop_seconds: float = 0.01

    def frame_index(self, t: float) -> float:
        return (t - self.start_time) / self.hop_seconds

    def played_frames(self) -> int:
        if self.end_time is None or self.start_time is None:
            return len(self.bands)
        return max(0, min(len(self.bands), int((self.end_time - self.start_time) / self.hop_seconds) + 1))


@dataclass
class EchoFrame:
    """Diagnostics for the most recent analysed frame."""

    time: float
    mic_rms: float
    residual_rms: float
    echo_active: bool
    delay: float


@dataclass
class _History:
    times: deque = field(default_factory=lambda: deque(maxlen=300))
    envelope: deque = field(default_factory=lambda: deque(maxlen=300))


class EchoGuard:
    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        initial_delay: float = 0.05,
        delay_search: tuple[float, float] = (-0.1, 0.6),
        tail_seconds: float = 0.15,
        lookahead_seconds: float = 0.12,
        over_subtraction: float = 2.0,
        doubt_fraction: float = 0.2,
        initial_gain: float = 1.0,
        gain_percentile: float = 90.0,
        clock=time.perf_counter,
    ) -> None:
        self._rate = sample_rate
        self._hop = sample_rate // 100  # 10 ms
        self._window = 2 * self._hop  # 20 ms
        self._nfft = 1 << int(np.ceil(np.log2(self._window)))
        self._hann = np.hanning(self._window)
        freqs = np.fft.rfftfreq(self._nfft, 1 / sample_rate)
        self._bins = np.where((freqs >= _MIN_HZ) & (freqs < _MAX_HZ))[0]
        self._bands = len(self._bins)
        self._delay = initial_delay
        self._delay_search = delay_search
        self._tail_frames = max(0, int(round(tail_seconds * 100)))
        self._look_frames = max(0, int(round(lookahead_seconds * 100)))
        self._over = over_subtraction
        self._doubt = min(max(doubt_fraction, 0.0), 0.95)
        self._gain = np.full(self._bands, float(initial_gain))
        self._ratio_buffer = np.full((800, self._bands), np.nan)
        self._ratio_index = 0
        self._ratio_rows = 0
        self._frames_since_gain_update = 0
        self._min_gain_samples = 120
        self._background_power: float | None = None
        """Slow minimum-follower of the mic level: what the room sounds like
        without Garvis (frames close to it carry no echo-path information)."""
        self._gain_percentile = gain_percentile
        self._clock = clock
        self._lock = threading.Lock()
        self._clips: list[_Clip] = []
        self._pending = np.zeros(0)
        self._pending_time: float | None = None
        self._history = _History()
        self._frames_since_delay_update = 0
        self.last: EchoFrame | None = None

    # ------------------------------------------------------------ analysis
    def _band_powers(self, frame: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(frame * self._hann, self._nfft))[self._bins] ** 2

    def _analyse(self, pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = max(0, (len(pcm) - self._window) // self._hop + 1)
        if frames == 0:
            return np.zeros((0, self._bands)), np.zeros(0)
        index = np.arange(self._window)[None, :] + self._hop * np.arange(frames)[:, None]
        spectra = np.abs(np.fft.rfft(pcm[index] * self._hann, self._nfft, axis=1)) ** 2
        bands = spectra[:, self._bins]
        envelope = np.log(bands.sum(axis=1) + 1e3) if frames else np.zeros(0)
        return bands, envelope

    # ----------------------------------------------------------- reference
    def start_reference(self, pcm: np.ndarray, rate: int, start_time: float | None) -> _Clip:
        """Register a clip about to play. `start_time` = when its first
        sample reaches the output device (None until known: set later via
        `set_reference_start`)."""
        bands, envelope = self._analyse(resample_linear(np.asarray(pcm), rate, self._rate))
        totals = bands.sum(axis=1)
        loud = float(np.percentile(totals, 90)) if len(totals) else 0.0
        band_loud = np.percentile(bands, 90, axis=0) if len(bands) else np.zeros(self._bands)
        clip = _Clip(bands=bands, envelope=envelope, start_time=start_time,
                     audible_level=loud * 1e-3, band_levels=band_loud * 1e-2)
        with self._lock:
            self._clips.append(clip)
        return clip

    @staticmethod
    def set_reference_start(clip: _Clip, start_time: float) -> None:
        clip.start_time = start_time

    def end_reference(self, clip: _Clip, end_time: float | None = None) -> None:
        """Playback of `clip` stopped (finished or interrupted) at `end_time`:
        nothing after it was played, so no echo of it can arrive later than
        end + delay + tail."""
        with self._lock:
            if clip.end_time is None:
                clip.end_time = end_time if end_time is not None else self._clock()
                if clip.start_time is None:
                    clip.start_time = clip.end_time

    def reset(self) -> None:
        """Forget all references and buffered mic audio (the learned echo
        path and delay are kept -- they belong to the hardware)."""
        with self._lock:
            self._clips.clear()
            self._pending = np.zeros(0)
            self._pending_time = None

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def gain_db(self) -> np.ndarray:
        return 10 * np.log10(self._gain + 1e-12)

    def active_clip_count(self) -> int:
        with self._lock:
            return len(self._clips)

    def _prune(self, now: float) -> None:
        horizon = self._delay + (self._tail_frames + 2) / 100 + 0.25
        keep = []
        for clip in self._clips:
            if clip.start_time is None:
                keep.append(clip)
                continue
            end = clip.end_time if clip.end_time is not None else clip.start_time + len(clip.bands) / 100
            if now - end <= horizon:
                keep.append(clip)
        self._clips = keep

    def _expected_echo(self, t: float, delay: float):
        """(window, aligned, clip) for mic time t: the reference band power
        maximised over the tail + lookahead window, the time-aligned frame
        itself (None outside the clip), and the clip -- or None when no
        clip can be heard at t."""
        best = None
        for clip in self._clips:
            if clip.start_time is None:
                continue
            k = clip.frame_index(t - delay)
            lo = int(np.floor(k)) - self._tail_frames
            hi = int(np.ceil(k)) + self._look_frames
            played = clip.played_frames()
            lo, hi = max(lo, 0), min(hi, played - 1)
            if hi < lo:
                continue
            window = clip.bands[lo: hi + 1].max(axis=0)
            aligned_index = int(round(k))
            aligned = clip.bands[aligned_index] if 0 <= aligned_index < played else None
            if best is None or window.sum() > best[0].sum():
                best = (window, aligned, clip)
        return best

    # ------------------------------------------------------------ delay
    def _update_delay(self) -> None:
        """Pick the delay that best lines up mic and reference log-energy
        envelopes over the recent history (only when enough reference
        activity overlaps it). Moves at most halfway per update."""
        times = np.array(self._history.times)
        mic_env = np.array(self._history.envelope)
        clips = [c for c in self._clips if c.start_time is not None and len(c.envelope)]
        if len(times) < 80 or not clips:
            return
        lo, hi = self._delay_search
        candidates = np.arange(lo, hi + 1e-9, 0.01)
        # (candidates, times) reference envelope at mic time t - candidate, NaN where no clip played
        ref_env = np.full((len(candidates), len(times)), np.nan)
        shifted = times[None, :] - candidates[:, None]
        for clip in clips:
            index = np.rint((shifted - clip.start_time) / clip.hop_seconds).astype(int)
            valid = (index >= 0) & (index < clip.played_frames())
            ref_env[valid] = clip.envelope[index[valid]]
        best_delay, best_corr = None, 0.0
        for row, candidate in zip(ref_env, candidates):
            mask = ~np.isnan(row)
            if mask.sum() < 60:
                continue
            r, m = row[mask], mic_env[mask]
            if r.std() < 1e-6 or m.std() < 1e-6:
                continue
            corr = float(np.corrcoef(m, r)[0, 1])
            if corr > best_corr:
                best_delay, best_corr = float(candidate), corr
        if best_delay is not None and best_corr >= 0.5:
            self._delay += 0.5 * (best_delay - self._delay)

    # -------------------------------------------------------------- mic
    def process(self, chunk: bytes, chunk_start_time: float) -> float | None:
        """Feed consecutive mic PCM (16-bit mono at `sample_rate`) whose first
        sample was captured at `chunk_start_time`. Returns the RMS of the
        part the playback can't explain for the frames completed in this
        chunk, or None when no playback reference is active (use the raw
        level then)."""
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return None
        with self._lock:
            # Prune on the microphone's own timeline (not the wall clock), so
            # analysis never depends on when this method happens to run.
            self._prune(chunk_start_time + samples.size / self._rate)
            expected_start = (self._pending_time + len(self._pending) / self._rate) if self._pending_time is not None else None
            if expected_start is None or abs(chunk_start_time - expected_start) > 0.05:
                # first chunk, or a gap/overrun: restart framing on the new timestamp
                self._pending = samples
                self._pending_time = chunk_start_time
            else:
                self._pending = np.concatenate([self._pending, samples])
            if not self._clips:
                self._track_background(float(np.mean(samples ** 2)))
                self._consume_frames(analyse=False)
                return None
            residual_power, frames, any_echo = self._consume_frames(analyse=True)
        if frames == 0:
            return self.last.residual_rms if (self.last and self.last.echo_active) else None
        if not any_echo:
            return None
        return float(np.sqrt(residual_power / frames))

    def _consume_frames(self, analyse: bool) -> tuple[float, int, bool]:
        residual_power = 0.0
        frames = 0
        any_echo = False
        while len(self._pending) >= self._window:
            frame = self._pending[: self._window]
            t = self._pending_time + self._window / (2 * self._rate)  # frame centre
            if analyse:
                power, echo = self._analyse_frame(frame, t)
                residual_power += power
                any_echo = any_echo or echo
                frames += 1
            self._pending = self._pending[self._hop:]
            self._pending_time += self._hop / self._rate
        if analyse and frames:
            self._frames_since_delay_update += frames
            if self._frames_since_delay_update >= 50:
                self._frames_since_delay_update = 0
                self._update_delay()
        return residual_power, frames, any_echo

    def _track_background(self, power: float) -> None:
        if self._background_power is None:
            self._background_power = power
        elif power < self._background_power:
            self._background_power += 0.2 * (power - self._background_power)
        else:
            self._background_power *= 1.002  # creeps up slowly if the room gets louder

    def _analyse_frame(self, frame: np.ndarray, t: float) -> tuple[float, bool]:
        hop_samples = frame[self._hop:]
        mic_power = float(np.mean(hop_samples ** 2))
        self._track_background(mic_power)
        bands = self._band_powers(frame)
        self._history.times.append(t)
        self._history.envelope.append(float(np.log(bands.sum() + 1e3)))
        expected = self._expected_echo(t, self._delay)
        if expected is None:
            self.last = EchoFrame(t, mic_power ** 0.5, mic_power ** 0.5, False, self._delay)
            return mic_power, False
        window, aligned, clip = expected
        echo = self._gain * window * self._over
        total = bands.sum()
        unexplained = np.maximum(bands - echo, 0.0).sum()
        fraction = unexplained / total if total > 0 else 0.0
        clearly_audible = self._background_power is not None and mic_power > 4.0 * self._background_power
        if aligned is not None and aligned.sum() > clip.audible_level and clearly_audible:
            self._learn_gain(bands, window, clip.band_levels)
        # Model error grows with echo level (speaker/mic compression when loud),
        # so the first `doubt_fraction` of unexplained energy is not counted.
        excess = max(0.0, fraction - self._doubt) / (1.0 - self._doubt)
        residual_power = mic_power * excess
        self.last = EchoFrame(t, mic_power ** 0.5, residual_power ** 0.5, True, self._delay)
        return residual_power, True

    def _learn_gain(self, bands: np.ndarray, window: np.ndarray, band_levels: np.ndarray) -> None:
        """Store this frame's mic/reference ratios (for bins where the
        reference carries energy) and refresh the gains every 25 frames."""
        row = np.where(window > band_levels, bands / np.maximum(window, 1e-3), np.nan)
        self._ratio_buffer[self._ratio_index] = row
        self._ratio_index = (self._ratio_index + 1) % len(self._ratio_buffer)
        self._ratio_rows = min(self._ratio_rows + 1, len(self._ratio_buffer))
        self._frames_since_gain_update += 1
        if self._frames_since_gain_update < 25:
            return
        self._frames_since_gain_update = 0
        data = self._ratio_buffer[: self._ratio_rows]
        counts = np.sum(~np.isnan(data), axis=0)
        ready = counts >= self._min_gain_samples
        if ready.any():
            percentile = np.nanpercentile(data[:, ready], self._gain_percentile, axis=0)
            self._gain[ready] = np.clip(percentile, 1e-5, 10.0)
