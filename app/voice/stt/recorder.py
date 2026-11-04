"""Push-to-talk microphone recording, feeding SpeechToTextProvider.

Not part of the `SpeechToTextProvider` interface itself: recording raw
audio from the mic is a distinct concern from transcribing an existing
audio file, and keeping them separate means a future wake-word listener
(continuous capture) can reuse `SpeechToTextProvider` unchanged and just
swap out how the audio file is produced.
"""
from __future__ import annotations

import os
import queue
import tempfile
import time
import wave

from app.core.exceptions import SpeechToTextError

_CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


class MicrophoneRecorder:
    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._stream = None
        self._frames: queue.Queue[tuple[float, bytes]] = queue.Queue()
        """(capture time of the block's first sample, PCM) -- the time lets
        barge-in echo discrimination line the mic up with playback."""
        self._recording = False
        self._input_latency = 0.02

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        if self._recording:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SpeechToTextError(
                "sounddevice is not installed. Run: pip install sounddevice"
            ) from exc

        self._frames = queue.Queue()

        def _callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
            # Windows MME reports no input timestamps, so derive the capture
            # time of the first sample from arrival time (time.perf_counter is
            # also PortAudio's clock on Windows).
            captured = time.perf_counter() - frames / self._sample_rate - self._input_latency
            self._frames.put((captured, bytes(indata)))

        try:
            stream = sd.RawInputStream(
                samplerate=self._sample_rate,
                channels=_CHANNELS,
                dtype="int16",
                callback=_callback,
            )
            stream.start()
        except Exception as exc:
            raise SpeechToTextError(f"Garvis couldn't access the microphone: {exc}") from exc

        self._stream = stream
        self._input_latency = float(getattr(stream, "latency", 0.02) or 0.02)
        self._recording = True

    def stop(self) -> str:
        if not self._recording or self._stream is None:
            raise SpeechToTextError("Recording was not started.")

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False

        chunks: list[bytes] = []
        while not self._frames.empty():
            chunks.append(self._frames.get_nowait()[1])
        audio_bytes = b"".join(chunks)

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="garvis_rec_")
        os.close(fd)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(_CHANNELS)
            wav_file.setsampwidth(_SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio_bytes)
        return path

    # ------------------------------------------------- continuous/session mode
    # The two methods below are additive: push-to-talk (start()/stop() above)
    # is completely unaffected by them. Session mode (see
    # app/voice/session.py:ConversationSession) needs the mic stream to stay
    # open across multiple utterances -- and even while Garvis is speaking,
    # for barge-in detection -- rather than the single start-then-stop cycle
    # push-to-talk uses.

    def read_available(self) -> bytes:
        """Non-blocking drain of whatever raw PCM frames have arrived since
        the last call (or since start()), without stopping the stream.
        Returns b"" if nothing new has arrived yet. Requires start() to
        have been called first."""
        return b"".join(pcm for _, pcm in self.read_available_timed())

    def read_available_timed(self) -> list[tuple[float, bytes]]:
        """Like read_available(), but block by block with each block's
        capture time (time.perf_counter() clock)."""
        if not self._recording:
            raise SpeechToTextError("Recording was not started.")
        blocks: list[tuple[float, bytes]] = []
        while True:
            try:
                blocks.append(self._frames.get_nowait())
            except queue.Empty:
                break
        return blocks

    def close(self) -> None:
        """Stop and release the stream without writing out a WAV -- for
        session mode, where audio is already consumed incrementally via
        read_available(). Safe to call even if never started or already
        closed."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._recording = False
