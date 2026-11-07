"""Lightweight, dependency-free voice activity detection for session mode.

Not a full VAD model (e.g. WebRTC VAD or Silero) -- a deliberately simple
RMS-energy threshold detector with minimum-duration debouncing on both
edges. That's enough to segment utterances in a normal desktop environment
without adding a new dependency (`audioop` is gone as of Python 3.13, and
this project otherwise has no signal-processing library).

Duration-based, not chunk-count-based: `process()` derives how much audio
time each chunk represents from its own byte length, so it stays correct
even when chunks arrive at irregular intervals (e.g. after a synthesis
delay) -- see `ConversationSession`, the only caller, which polls a
microphone queue at a roughly-fixed interval but can't guarantee it.

Noise-floor adaptation: a *fixed* absolute threshold cannot tell a
continuous ambient sound (an air conditioner, a fan, room hum) from real
speech, because the only existing safeguard against false triggers --
requiring `min_speech_seconds` of *sustained* loud audio -- filters brief
spikes, not sustained ones. A continuous hum that happens to sit above the
threshold satisfies "sustained" trivially, for as long as it runs.

The fix is a brief, one-time, cold-start *calibration* window (default
0.3s) rather than continuously chasing the signal in real time: a fast
real-time adaptive floor was tried and rejected during development, because
making it react quickly enough to absorb a hum before
`min_speech_seconds` elapses also makes it react quickly enough to chase
and defeat genuine speech in exactly the same window -- both look
identical to an energy detector for the first fraction of a second, so
there is no way to tell them apart *fast*. Calibrating once, before the
speech/silence state machine starts at all, sidesteps that: whatever
ambient sound (including a running air conditioner) is already present
when a fresh `VoiceActivityDetector` is created gets absorbed into
`noise_floor` unconditionally during that window (no speech_start can fire
yet), and every loudness decision afterward is judged against
`max(energy_threshold, noise_floor * margin_ratio)`. Real speech, well
above the room's ambient level, still clears that bar exactly as before.

Calibration only happens for a genuinely fresh estimate (`initial_noise_floor`
not given). `ConversationSession` seeds each new VAD instance with the
previous one's learned floor, so only the very first utterance of a
session pays the calibration latency -- every subsequent turn starts
already calibrated. A slow, conservative ongoing adaptation (gated on
`not is_loud`, so it can never chase real speech) continues afterward to
track a gradually drifting ambient level.

This is not perfect -- no energy-based VAD can perfectly distinguish
arbitrary noise from speech -- but it robustly rejects the specific
"continuous low-level background noise" failure mode without dulling
sensitivity to real speech.

Known limitations (real, not hidden): with no acoustic echo cancellation,
this cannot distinguish the user's own voice from Garvis's TTS output
bleeding back into the microphone from speakers. A fixed higher threshold
while speaking (below) helps but is not enough on its own: real hardware
was observed where speaker leakage alone cleared even that raised
threshold, causing Garvis to repeatedly "interrupt itself". `process()`'s
optional `reference_energy` parameter addresses that specific case: the
caller (ConversationSession, via VoicePipeline.reference_energy_now())
passes in how loud Garvis's own already-known TTS clip is *right now*,
and the effective barge-in threshold is raised to track it
(`reference_energy * echo_margin_ratio`) for exactly as long as Garvis is
actually making sound. This is still not true AEC -- it cannot tell a
user's voice apart from an echo of comparable loudness to what's playing
at that instant -- but it directly targets the reported failure mode
(leakage energy scaling with whatever Garvis is saying) without dulling
sensitivity when Garvis is silent or between phrases, which is when a
real interruption is most easily told apart. A headset avoids the
problem entirely.

Reference-energy ceiling (`max_reference_multiplier`): `reference_energy` is
measured in a completely different domain from everything else this module
compares it against -- it is the RMS of Garvis's own *synthesized TTS file*
(a clean, studio-loud digital recording), not of anything the microphone
ever captured. Measuring a real Piper clip (Jarvis-High) confirmed this
concretely: ordinary speech segments sit around RMS 4,000-9,000 and peaks
reach ~12,000, while `energy_threshold`/`barge_in_energy_threshold`
(500/900 by default) were calibrated against *microphone-captured* speech
on the same hardware -- an entirely different, much smaller scale, since
real leakage is attenuated by air, distance, and the mic's own gain before
it ever reaches the microphone. Multiplying that studio-domain RMS directly
by `echo_margin_ratio` (with no upper bound) therefore drove the effective
barge-in threshold to 6,000-18,000+ for most of any reply -- many multiples
above what real, close-mic user speech could ever produce on this app's own
calibration -- which silently defeated real-user barge-in for virtually the
whole duration of Garvis speaking, not just the moments with genuine
leakage. `max_reference_multiplier` caps how far the reference-energy term
alone may raise the bar above `energy_threshold`, decoupling the cap from
the synthesis file's arbitrary internal loudness while still leaving it far
above the plain fixed threshold whenever Garvis is actually making sound.
"""

from __future__ import annotations

import struct

_BYTES_PER_SAMPLE = 2  # 16-bit PCM, matches MicrophoneRecorder's format


def _rms(chunk: bytes) -> float:
    """Root-mean-square amplitude of a 16-bit mono PCM chunk. 0.0 for empty
    or odd-length (partial-sample) input rather than raising."""
    count = len(chunk) // _BYTES_PER_SAMPLE
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", chunk[: count * _BYTES_PER_SAMPLE])
    return (sum(s * s for s in samples) / count) ** 0.5


def _chunk_seconds(chunk: bytes, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return (len(chunk) // _BYTES_PER_SAMPLE) / sample_rate


class PlaybackEnvelope:
    """Coarse RMS-over-time energy profile of a single already-synthesized
    audio clip, sampled in fixed-size time frames and queried by elapsed
    wall-clock time since playback of that clip started.

    Built once from the PCM VoicePipeline.speak() already has on disk
    before playback begins, and queried via VoicePipeline.reference_energy_now()
    -- see that method and this module's "Known limitations" section for why:
    it lets barge-in detection tell "the mic is loud because Garvis's own
    voice is currently playing" apart from "the mic is loud for some other
    reason" without full acoustic echo cancellation.

    Not sample-accurate: `winsound.PlaySound(..., SND_ASYNC)` (see
    app/voice/tts/player.py) gives no true playback-position callback, so
    wall-clock elapsed time is the best available proxy for "how far into
    the clip is playback right now" -- the same approximation AudioPlayer
    already relies on for its own completion estimate. Speech envelope
    features (words, pauses) change on the order of 100ms+, so the
    resulting few-ms of clock skew is inconsequential here.
    """

    _TAIL_WINDOW_SECONDS = 0.3
    """How far back from the clip's end to look when a caller reports
    playback is still ongoing past our own estimated duration (see
    `energy_at`'s `held_if_still_playing`) -- the window used to compute a
    representative "still probably this loud" fallback, rather than a
    single last frame that might itself be a trailing near-silence."""

    def __init__(self, pcm: bytes, sample_rate: int, frame_seconds: float = 0.02) -> None:
        self._frame_seconds = frame_seconds if frame_seconds > 0 else 0.02
        frame_bytes = max(_BYTES_PER_SAMPLE, int(self._frame_seconds * sample_rate) * _BYTES_PER_SAMPLE)
        self._frames = [_rms(pcm[i : i + frame_bytes]) for i in range(0, len(pcm), frame_bytes)]
        tail_frame_count = max(1, int(self._TAIL_WINDOW_SECONDS / self._frame_seconds))
        self._tail_energy = max(self._frames[-tail_frame_count:]) if self._frames else 0.0

    def energy_at(self, elapsed_seconds: float, *, held_if_still_playing: bool = False) -> float:
        """Reference RMS at `elapsed_seconds` into the clip. 0.0 before the
        clip starts or for an empty/unreadable clip.

        At/after the clip's own estimated end: 0.0, UNLESS
        `held_if_still_playing` is set (VoicePipeline.reference_energy_now()
        passes this whenever AudioPlayer still reports actual playback in
        progress) -- `winsound`'s own completion timer is itself only an
        estimate (see AudioPlayer's docstring: real playback can run a
        little longer than the WAV's nominal duration), so collapsing to
        0.0 the instant our estimate expires would silently disable the
        self-echo guard for whatever tail of real audio is still playing.
        Held value is the loudest of the last `_TAIL_WINDOW_SECONDS` of the
        clip -- representative of the clip's actual ending loudness, not a
        single possibly-near-silent last frame.
        """
        if elapsed_seconds < 0 or not self._frames:
            return 0.0
        index = int(elapsed_seconds / self._frame_seconds)
        if index >= len(self._frames):
            return self._tail_energy if held_if_still_playing else 0.0
        return self._frames[index]


class VoiceActivityDetector:
    """Stateful speech/silence segmenter. Feed it consecutive audio chunks
    via `process()`; it reports edge-triggered transitions.

    Not thread-safe and not meant to be shared -- one instance per utterance
    (or, for barge-in, one instance handed off from the speaking phase into
    the next listening phase so its "already active" state carries over --
    see ConversationSession._speak_interruptibly / _capture_utterance).
    """

    def __init__(
        self,
        sample_rate: int,
        energy_threshold: float,
        min_speech_seconds: float,
        min_silence_seconds: float,
        *,
        noise_floor_alpha: float = 0.1,
        noise_margin_ratio: float = 1.6,
        min_noise_floor: float = 10.0,
        initial_noise_floor: float | None = None,
        calibration_seconds: float = 0.3,
        echo_margin_ratio: float = 1.5,
        max_reference_multiplier: float = 3.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._threshold = energy_threshold
        self._min_speech_seconds = min_speech_seconds
        self._min_silence_seconds = min_silence_seconds
        self._noise_floor_alpha = noise_floor_alpha
        self._noise_margin_ratio = noise_margin_ratio
        self._min_noise_floor = min_noise_floor
        self._echo_margin_ratio = echo_margin_ratio
        self._max_reference_multiplier = max_reference_multiplier
        self._noise_floor = max(min_noise_floor, initial_noise_floor or min_noise_floor)
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._active = False

        # Only a genuinely fresh (unseeded) estimate needs to calibrate --
        # see module docstring for why this must be a one-time upfront
        # window rather than continuous fast real-time adaptation.
        self._calibrating = initial_noise_floor is None and calibration_seconds > 0
        self._calibration_seconds = calibration_seconds
        self._calibration_elapsed = 0.0
        self._last_energy = 0.0
        self._last_effective_threshold = energy_threshold

    @property
    def last_energy(self) -> float:
        """RMS energy of the most recently processed chunk -- diagnostic
        only, for callers that want to log exactly what `process()` last
        saw (see ConversationSession's barge-in/speech-start logging)."""
        return self._last_energy

    @property
    def last_effective_threshold(self) -> float:
        """The effective threshold the most recent `process()` call judged
        `last_energy` against -- i.e.
        `max(energy_threshold, noise_floor * noise_margin_ratio, reference_energy * echo_margin_ratio)`
        at that moment. Diagnostic only, same purpose as `last_energy`."""
        return self._last_effective_threshold

    @property
    def is_active(self) -> bool:
        """True once a `speech_start` has been reported and no
        `speech_end` has followed it yet."""
        return self._active

    @property
    def noise_floor(self) -> float:
        """Current estimated ambient RMS level -- exposed so a caller can
        seed the next VAD instance with it (see ConversationSession, which
        carries this across listening/speaking phases rather than
        re-learning the room from scratch every time)."""
        return self._noise_floor

    @property
    def is_calibrating(self) -> bool:
        """True during the brief cold-start window where no speech_start
        can fire yet -- see module docstring."""
        return self._calibrating

    def process(self, chunk: bytes, reference_energy: float = 0.0, residual_energy: float | None = None) -> str:
        """Feed one chunk of raw 16-bit mono PCM audio.

        `reference_energy` is an optional, caller-supplied estimate of how
        loud audio Garvis itself is currently emitting is (see
        VoicePipeline.reference_energy_now() / PlaybackEnvelope above);
        0.0 (the default) means "not applicable" -- e.g. while listening,
        or for any caller that doesn't need barge-in-style self-echo
        rejection -- and leaves every existing threshold computation
        unchanged. When non-zero, it raises the effective threshold for
        this chunk only, proportionally, so a moment where Garvis's own
        known output is loud doesn't get mistaken for the user
        interrupting -- see the module docstring's "Known limitations".

        `residual_energy`, when given, is the RMS of the part of this chunk
        that Garvis's own playback can NOT explain (app/voice/echo.py
        EchoGuard): it replaces the raw level for every decision below,
        including noise-floor tracking, so echo never counts as speech nor
        inflates the floor. The reference_energy heuristic is then unused
        (pass 0.0). None = no playback reference active: raw level.

        Returns one of:
          "silence"      -- not (yet) speech, and wasn't already active
          "speech_start"  -- edge: just crossed into sustained speech
          "speech"        -- continuing speech, already active
          "speech_end"    -- edge: just crossed into sustained silence
                              after being active
        """
        duration = _chunk_seconds(chunk, self._sample_rate)
        if duration <= 0:
            return "speech" if self._active else "silence"

        energy = _rms(chunk) if residual_energy is None else residual_energy
        self._last_energy = energy

        if self._calibrating:
            # Unconditionally absorb every chunk -- including one that
            # would otherwise look "loud" -- so a sound already present
            # when this VAD was created (e.g. a running air conditioner)
            # becomes part of the baseline before any speech-duration
            # accumulation starts, rather than racing against it. No
            # speech_start can fire during this window.
            self._noise_floor += 0.3 * (energy - self._noise_floor)
            self._noise_floor = max(self._noise_floor, self._min_noise_floor)
            self._calibration_elapsed += duration
            if self._calibration_elapsed >= self._calibration_seconds:
                self._calibrating = False
            return "silence"

        # Capped independently of the other two terms: noise_floor is
        # already grounded in real microphone-domain measurements, but
        # reference_energy is not (see module docstring) -- without this
        # cap, a loud/clean TTS clip can inflate the bar to many multiples
        # of energy_threshold, far past anything real speech on this same
        # microphone could reach, silently defeating barge-in.
        reference_ceiling = min(
            reference_energy * self._echo_margin_ratio,
            self._threshold * self._max_reference_multiplier,
        )
        effective_threshold = max(
            self._threshold,
            self._noise_floor * self._noise_margin_ratio,
            reference_ceiling,
        )
        self._last_effective_threshold = effective_threshold
        is_loud = energy >= effective_threshold

        if not self._active and not is_loud:
            # Ongoing (post-calibration) drift tracking: gated on "not
            # loud" so it can never chase and defeat real speech the way a
            # fast unconditional adaptation would -- calibration already
            # did the heavy lifting of catching up to whatever ambient
            # sound was already present.
            self._noise_floor += self._noise_floor_alpha * (energy - self._noise_floor)
            self._noise_floor = max(self._noise_floor, self._min_noise_floor)

        if is_loud:
            self._speech_seconds += duration
            self._silence_seconds = 0.0
        else:
            self._silence_seconds += duration
            self._speech_seconds = 0.0

        if not self._active:
            if self._speech_seconds >= self._min_speech_seconds:
                self._active = True
                return "speech_start"
            return "silence"

        if self._silence_seconds >= self._min_silence_seconds:
            self._active = False
            return "speech_end"
        return "speech"
