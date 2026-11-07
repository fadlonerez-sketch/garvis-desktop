"""Plays a synthesized WAV file back to the user, and can stop it early.

Deliberately separate from `TextToSpeechProvider`: that interface only
renders text to an audio file (see piper_provider.py). Actually playing
the file -- and supporting the "stop speaking" button from the product
spec -- is a distinct concern, handled here via the stdlib `winsound`
module. No extra dependency, and Garvis targets Windows only.

Platform guard: rather than checking `sys.platform`, we attempt to
`import winsound` and treat ImportError as "not supported here". This is
functionally identical on real Windows (winsound always imports) and real
non-Windows (winsound never exists, so it always raises) -- it just also
lets tests substitute a fake `winsound` module via `sys.modules` without
needing to actually run on Windows.

Playback-completion limitation: `winsound.PlaySound(..., SND_ASYNC)` has
no completion callback, so there is no exact way to know when playback
actually finishes. We estimate it from the WAV file's own duration (frames
/ frame rate) via a background `threading.Timer`. This is an
approximation: if playback is somehow slower than the file's nominal
duration, `is_playing`/`on_finished` fire a little early. That's an
accepted limitation of using `winsound` -- exact completion detection
would need a different playback API, which is out of scope here.
"""

from __future__ import annotations

import threading
import time
import wave
from collections.abc import Callable

from app.core.exceptions import TextToSpeechError

_MIN_DURATION_SECONDS = 0.05


class AudioPlayer:
    """Asynchronous WAV playback with an explicit `stop()`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._playing = False
        self._timer: threading.Timer | None = None
        self._on_finished: Callable[[], None] | None = None

    def play(self, wav_path: str, on_finished: Callable[[], None] | None = None) -> None:
        """Play `wav_path` asynchronously.

        If `on_finished` is given, it's called (at most once) either when
        estimated playback completes, or when `stop()` is called -- whichever
        happens first. Useful for reliably cleaning up a temp file once
        playback is done with it.
        """
        winsound = self._import_winsound()

        with self._lock:
            self._cancel_timer_locked()
            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except RuntimeError as exc:
                raise TextToSpeechError(f"Could not play '{wav_path}': {exc}") from exc
            self._playing = True
            self._on_finished = on_finished
            duration = self._estimate_duration_seconds(wav_path)
            self._timer = threading.Timer(duration, self._on_timer_elapsed)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> None:
        try:
            winsound = self._import_winsound()
        except TextToSpeechError:
            return  # nothing could have been playing without winsound in the first place

        with self._lock:
            if self._playing:
                winsound.PlaySound(None, winsound.SND_PURGE)
            self._finish_locked()

    @property
    def is_playing(self) -> bool:
        return self._playing

    # ------------------------------------------------------------- internals
    @staticmethod
    def _import_winsound():
        try:
            import winsound
        except ImportError as exc:
            raise TextToSpeechError("Audio playback is only implemented for Windows.") from exc
        return winsound

    def _on_timer_elapsed(self) -> None:
        with self._lock:
            self._finish_locked()

    def _finish_locked(self) -> None:
        """Mark playback as finished and fire the pending callback, if any.
        Idempotent: safe to call from both the timer and an explicit stop()."""
        self._cancel_timer_locked()
        was_playing = self._playing
        self._playing = False
        callback = self._on_finished
        self._on_finished = None
        if was_playing and callback is not None:
            callback()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    @staticmethod
    def _estimate_duration_seconds(wav_path: str) -> float:
        try:
            with wave.open(wav_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate() or 1
                return max(frames / rate, _MIN_DURATION_SECONDS)
        except OSError, wave.Error:
            return _MIN_DURATION_SECONDS


class StreamingAudioPlayer:
    """WAV playback through PortAudio (sounddevice, already used for the
    microphone) that knows WHEN its audio actually reaches the speakers.

    Same interface as AudioPlayer, plus `on_started(dac_time)`: called once
    with the output device's DAC time of the clip's first sample (on
    Windows the same clock as time.perf_counter()). Barge-in echo
    discrimination (app/voice/echo.py) needs that to line the microphone up
    with the exact samples being played; winsound can't provide it.
    `on_finished` fires when the last buffer has really been played (not an
    estimate) or immediately on stop(). If the output stream can't be opened,
    it falls back to AudioPlayer (winsound) with an approximate start time,
    so speech output never breaks over this.
    """

    def __init__(self, sd_module=None, clock=time.perf_counter, fallback: AudioPlayer | None = None) -> None:
        self._sd = sd_module
        self._clock = clock
        self._fallback = fallback or AudioPlayer()
        self._lock = threading.RLock()
        self._stream = None
        self._playing = False
        self._using_fallback = False
        self._on_finished: Callable[[], None] | None = None
        self._generation = 0

    def _import_sounddevice(self):
        if self._sd is None:
            import sounddevice

            self._sd = sounddevice
        return self._sd

    @staticmethod
    def _read_wav(wav_path: str) -> tuple[bytes, int]:
        try:
            with wave.open(wav_path, "rb") as wav_file:
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
        except (OSError, wave.Error) as exc:
            raise TextToSpeechError(f"Could not read '{wav_path}': {exc}") from exc
        if width != 2:
            raise TextToSpeechError(f"'{wav_path}' is not 16-bit PCM.")
        if channels > 1:  # keep the first channel
            frames = b"".join(frames[i : i + 2] for i in range(0, len(frames), 2 * channels))
        return frames, rate

    def play(
        self,
        wav_path: str,
        on_finished: Callable[[], None] | None = None,
        on_started: Callable[[float], None] | None = None,
    ) -> None:
        pcm, rate = self._read_wav(wav_path)
        self.stop()
        with self._lock:
            self._generation += 1
            generation = self._generation
            position = {"byte": 0, "started": False}
            frame_bytes = 2

            def callback(outdata, frames, time_info, status) -> None:
                if not position["started"]:
                    position["started"] = True
                    dac = getattr(time_info, "outputBufferDacTime", 0.0) or 0.0
                    if on_started is not None:
                        on_started(dac if dac > 0 else self._clock())
                start = position["byte"]
                chunk = pcm[start : start + frames * frame_bytes]
                outdata[: len(chunk)] = chunk
                if len(chunk) < frames * frame_bytes:
                    outdata[len(chunk) :] = b"\x00" * (frames * frame_bytes - len(chunk))
                position["byte"] = start + frames * frame_bytes
                if position["byte"] >= len(pcm):
                    raise self._sd.CallbackStop

            def finished() -> None:
                self._finish(generation)

            try:
                sd = self._import_sounddevice()
                stream = sd.RawOutputStream(
                    samplerate=rate, channels=1, dtype="int16", callback=callback, finished_callback=finished
                )
                stream.start()
            except Exception:
                # No usable output stream: winsound, with the start time approximated.
                self._using_fallback = True
                self._playing = True
                self._on_finished = on_finished
                self._fallback.play(wav_path, on_finished=lambda: self._finish(generation))
                if on_started is not None:
                    on_started(self._clock())
                return
            self._using_fallback = False
            self._stream = stream
            self._playing = True
            self._on_finished = on_finished

    def _finish(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or not self._playing:
                return
            self._playing = False
            callback = self._on_finished
            self._on_finished = None
        if callback is not None:
            callback()

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            generation = self._generation
            using_fallback = self._using_fallback
        if using_fallback:
            self._fallback.stop()
        if stream is not None:
            try:
                stream.abort()  # drops buffered audio now; also fires finished_callback
                stream.close()
            except Exception:
                pass
        self._finish(generation)

    @property
    def is_playing(self) -> bool:
        return self._playing
