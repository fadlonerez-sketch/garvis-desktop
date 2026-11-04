"""Tests for the energy-based VoiceActivityDetector (app/voice/vad.py).

Pure deterministic logic -- fed synthetic PCM chunks, no real microphone
or audio hardware involved.
"""
from __future__ import annotations

import struct

import pytest

from app.voice.vad import PlaybackEnvelope, VoiceActivityDetector

_SAMPLE_RATE = 16000


def _tone_chunk(seconds: float, amplitude: int, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """A constant-amplitude 16-bit PCM chunk (alternating +/- amplitude so
    its RMS is exactly `amplitude`, not a sine wave -- simpler and just as
    valid for exercising the RMS-threshold logic)."""
    count = int(seconds * sample_rate)
    samples = [amplitude if i % 2 == 0 else -amplitude for i in range(count)]
    return struct.pack(f"<{count}h", *samples)


def _silence_chunk(seconds: float, sample_rate: int = _SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def _make_vad(**overrides) -> VoiceActivityDetector:
    """Defaults to an already-calibrated VAD (a low seeded noise floor) so
    existing tests exercise the steady-state speech/silence state machine
    directly, without the one-time cold-start calibration window in the
    mix. Tests of calibration itself pass initial_noise_floor=None
    explicitly (see the "noise-floor adaptation" section below)."""
    params = dict(
        sample_rate=_SAMPLE_RATE,
        energy_threshold=500.0,
        min_speech_seconds=0.15,
        min_silence_seconds=0.6,
        initial_noise_floor=10.0,
    )
    params.update(overrides)
    return VoiceActivityDetector(**params)


def test_silence_never_triggers_speech_start():
    vad = _make_vad()
    for _ in range(20):
        assert vad.process(_silence_chunk(0.03)) == "silence"
    assert not vad.is_active


def test_brief_loud_blip_does_not_trigger_speech_start():
    """A short accidental noise (below min_speech_seconds) must not count
    as the start of an utterance."""
    vad = _make_vad(min_speech_seconds=0.15)
    assert vad.process(_tone_chunk(0.03, 2000)) == "silence"
    assert vad.process(_silence_chunk(0.03)) == "silence"
    assert not vad.is_active


def test_sustained_loud_audio_triggers_speech_start_then_speech():
    vad = _make_vad(min_speech_seconds=0.1)
    # under threshold duration
    assert vad.process(_tone_chunk(0.05, 2000)) == "silence"
    # crosses the min_speech_seconds threshold
    assert vad.process(_tone_chunk(0.06, 2000)) == "speech_start"
    assert vad.is_active
    # continues to report "speech" while still active
    assert vad.process(_tone_chunk(0.05, 2000)) == "speech"


def test_brief_pause_during_speech_does_not_end_the_utterance():
    """A short pause between words must not be mistaken for the end of the
    utterance -- only min_silence_seconds of continuous silence should."""
    vad = _make_vad(min_speech_seconds=0.1, min_silence_seconds=0.5)
    vad.process(_tone_chunk(0.15, 2000))
    assert vad.is_active
    assert vad.process(_silence_chunk(0.1)) == "speech"  # brief pause, still active
    assert vad.is_active
    assert vad.process(_tone_chunk(0.1, 2000)) == "speech"  # speech resumes


def test_sustained_silence_after_speech_triggers_speech_end():
    vad = _make_vad(min_speech_seconds=0.1, min_silence_seconds=0.3)
    assert vad.process(_tone_chunk(0.15, 2000)) == "speech_start"
    assert vad.process(_silence_chunk(0.15)) == "speech"  # not yet enough silence
    assert vad.process(_silence_chunk(0.2)) == "speech_end"
    assert not vad.is_active


def test_empty_chunk_is_a_no_op():
    vad = _make_vad()
    assert vad.process(b"") == "silence"
    assert not vad.is_active


def test_higher_threshold_ignores_quieter_audio():
    """Simulates the barge-in use case: a higher energy_threshold (to
    reduce false triggers from TTS speaker bleed) should not react to
    moderate-volume audio that a lower threshold would catch."""
    quiet_vad = _make_vad(energy_threshold=500.0, min_speech_seconds=0.05)
    loud_only_vad = _make_vad(energy_threshold=5000.0, min_speech_seconds=0.05)

    moderate_tone = _tone_chunk(0.1, 1500)
    assert quiet_vad.process(moderate_tone) == "speech_start"
    assert loud_only_vad.process(moderate_tone) == "silence"


def test_state_after_speech_end_can_detect_a_new_utterance():
    vad = _make_vad(min_speech_seconds=0.1, min_silence_seconds=0.2)
    assert vad.process(_tone_chunk(0.15, 2000)) == "speech_start"
    assert vad.process(_silence_chunk(0.25)) == "speech_end"
    assert not vad.is_active
    assert vad.process(_tone_chunk(0.15, 2000)) == "speech_start"


# ------------------------------------------- noise-floor adaptation (AC hum)
def _ambient_hum_chunk(seconds: float, amplitude: int) -> bytes:
    """A continuous, moderate-amplitude "hum" -- stands in for an air
    conditioner or fan: sustained, not a brief spike, and (unlike real-world
    noise) perfectly constant here so the test is deterministic."""
    return _tone_chunk(seconds, amplitude)


def test_continuous_ambient_noise_does_not_trigger_speech_start():
    """The core air-conditioner regression: a sustained hum whose energy
    would, under a fixed threshold, trivially satisfy min_speech_seconds
    (since "sustained" is exactly what a hum is) must not be classified as
    speech, because it's already running (and gets calibrated against) from
    the moment this fresh VAD starts listening -- exactly the reported
    real-world scenario, where the air conditioner was already on when the
    session started."""
    vad = _make_vad(
        energy_threshold=500.0,
        min_speech_seconds=0.15,
        min_silence_seconds=0.5,
        initial_noise_floor=None,  # cold start: the hum is already running
    )
    hum_amplitude = 600  # above the fixed 500 threshold, as an AC hum might be

    results = [vad.process(_ambient_hum_chunk(0.05, hum_amplitude)) for _ in range(40)]  # ~2s

    assert "speech_start" not in results
    assert not vad.is_active


def test_short_noise_spike_does_not_trigger_speech_start():
    vad = _make_vad(min_speech_seconds=0.15)
    assert vad.process(_tone_chunk(0.05, 3000)) != "speech_start"
    assert vad.process(_silence_chunk(0.05)) == "silence"
    assert not vad.is_active


def test_sustained_signal_above_adapted_floor_still_triggers_speech_start():
    """Real speech, well above the learned ambient level, must still be
    detected reliably even after the floor has adapted to a steady hum."""
    vad = _make_vad(
        energy_threshold=500.0,
        min_speech_seconds=0.15,
        min_silence_seconds=0.5,
        initial_noise_floor=None,
    )
    hum_amplitude = 600

    for _ in range(40):  # let the floor adapt to the hum, ~2s
        vad.process(_ambient_hum_chunk(0.05, hum_amplitude))
    assert not vad.is_active

    speech_amplitude = 3000  # clearly above both the fixed threshold and the hum
    results = [vad.process(_tone_chunk(0.05, speech_amplitude)) for _ in range(4)]  # 0.2s

    assert "speech_start" in results
    assert vad.is_active


def test_normal_speech_followed_by_silence_still_ends_correctly_with_floor_adaptation():
    """The existing end-of-utterance behavior must be unchanged by the
    noise-floor addition."""
    vad = _make_vad(min_speech_seconds=0.1, min_silence_seconds=0.3)
    assert vad.process(_tone_chunk(0.15, 2000)) == "speech_start"
    assert vad.process(_silence_chunk(0.15)) == "speech"  # not yet enough silence
    assert vad.process(_silence_chunk(0.2)) == "speech_end"
    assert not vad.is_active


def test_noise_floor_does_not_rise_during_an_active_speech_segment():
    """A brief in-utterance pause (well under min_silence_seconds) must not
    be absorbed into the noise floor -- only genuine ambient audio between
    utterances should shift it."""
    vad = _make_vad(
        energy_threshold=500.0, min_speech_seconds=0.1, min_silence_seconds=0.5
    )
    vad.process(_tone_chunk(0.15, 2000))
    assert vad.is_active
    floor_before = vad.noise_floor
    vad.process(_silence_chunk(0.1))  # brief pause mid-utterance, still active
    assert vad.is_active
    assert vad.noise_floor == floor_before  # unchanged -- not treated as ambient


def test_noise_floor_adapts_upward_only_from_genuinely_quiet_periods():
    vad = _make_vad(energy_threshold=500.0, min_speech_seconds=0.15, min_silence_seconds=0.5)
    initial_floor = vad.noise_floor

    for _ in range(20):
        vad.process(_ambient_hum_chunk(0.05, 400))  # below the fixed threshold on its own

    assert vad.noise_floor > initial_floor
    assert vad.noise_floor < 400 + 1  # converges toward the hum's own level, not beyond it


def test_barge_in_style_high_threshold_still_rejects_continuous_background_noise():
    """Barge-in uses a higher absolute threshold than normal listening, but
    it must also reject a hum loud enough to clear even that higher
    threshold (the reported scenario: the air conditioner triggering
    interruptions while Garvis is speaking)."""
    vad = _make_vad(
        energy_threshold=900.0,
        min_speech_seconds=0.15,
        min_silence_seconds=0.5,
        initial_noise_floor=None,  # cold start: hum already running when speaking begins
    )
    hum_amplitude = 1000  # above the higher barge-in threshold itself

    results = [vad.process(_ambient_hum_chunk(0.05, hum_amplitude)) for _ in range(40)]

    assert "speech_start" not in results
    assert not vad.is_active


def test_barge_in_style_high_threshold_still_detects_real_speech_over_noise():
    vad = _make_vad(
        energy_threshold=900.0,
        min_speech_seconds=0.15,
        min_silence_seconds=0.5,
        initial_noise_floor=None,
    )
    hum_amplitude = 1000
    for _ in range(20):
        vad.process(_ambient_hum_chunk(0.05, hum_amplitude))

    results = [vad.process(_tone_chunk(0.05, 4000)) for _ in range(4)]
    assert "speech_start" in results


def test_calibration_window_never_reports_speech_start():
    """During the one-time cold-start calibration window, even a very loud
    signal must not be reported as speech_start -- it's still being learned
    as the baseline, not yet evaluated as candidate speech."""
    vad = VoiceActivityDetector(
        sample_rate=_SAMPLE_RATE,
        energy_threshold=500.0,
        min_speech_seconds=0.05,
        min_silence_seconds=0.5,
        initial_noise_floor=None,
        calibration_seconds=0.2,
    )
    results = [vad.process(_tone_chunk(0.05, 5000)) for _ in range(4)]  # 0.2s, very loud
    assert "speech_start" not in results
    assert not vad.is_calibrating  # calibration window has elapsed by now


def test_zero_calibration_seconds_skips_calibration_entirely():
    vad = VoiceActivityDetector(
        sample_rate=_SAMPLE_RATE,
        energy_threshold=500.0,
        min_speech_seconds=0.05,
        min_silence_seconds=0.5,
        initial_noise_floor=None,
        calibration_seconds=0.0,
    )
    assert not vad.is_calibrating
    assert vad.process(_tone_chunk(0.1, 3000)) == "speech_start"


def test_initial_noise_floor_can_be_seeded_from_a_prior_estimate():
    """ConversationSession carries the learned floor across
    listening/speaking phases rather than re-learning the room each time --
    verify the constructor param actually seeds it."""
    vad = VoiceActivityDetector(
        sample_rate=_SAMPLE_RATE,
        energy_threshold=500.0,
        min_speech_seconds=0.15,
        min_silence_seconds=0.5,
        initial_noise_floor=650.0,
    )
    assert vad.noise_floor == 650.0


# --------------------------------- self-echo rejection (TTS speaker bleed)
def test_reference_energy_default_does_not_change_existing_behavior():
    """0.0 (the default) must be a true no-op -- every existing caller that
    never passes reference_energy (e.g. the LISTENING-phase VAD) must see
    identical behavior to before this parameter existed."""
    vad = _make_vad(energy_threshold=900.0, min_speech_seconds=0.1)
    assert vad.process(_tone_chunk(0.15, 1000)) == "speech_start"


def test_loud_reference_energy_suppresses_matching_mic_leakage():
    """The originally reported real-world bug: speaker leakage of Garvis's
    own TTS output clears even the raised barge-in threshold on its own.
    When the caller reports that Garvis's own playback is currently this
    loud, mic energy at (or below) a reasonable multiple of it -- but still
    within max_reference_multiplier's cap, see the tests below for what
    happens once it isn't -- must not be treated as the user interrupting."""
    vad = _make_vad(energy_threshold=900.0, min_speech_seconds=0.1, echo_margin_ratio=1.5)
    own_voice_reference = 4000.0  # how loud Garvis's own known clip is right now
    leaked_amplitude = 2000  # bleeds through above the fixed 900 threshold

    results = [
        vad.process(_tone_chunk(0.05, leaked_amplitude), reference_energy=own_voice_reference)
        for _ in range(6)  # ~0.3s, well past min_speech_seconds
    ]

    assert "speech_start" not in results
    assert not vad.is_active


def test_real_speech_still_interrupts_despite_high_reference_energy():
    """Companion to the above: genuine user speech, loud enough to clear
    even the reference-scaled ceiling, must still interrupt -- the
    mechanism must reject the specific echo case, not desensitize barge-in
    generally."""
    vad = _make_vad(energy_threshold=900.0, min_speech_seconds=0.1, echo_margin_ratio=1.5)
    own_voice_reference = 4000.0
    real_speech_amplitude = 8000  # clearly above own_voice_reference * 1.5

    results = [
        vad.process(_tone_chunk(0.05, real_speech_amplitude), reference_energy=own_voice_reference)
        for _ in range(4)
    ]

    assert "speech_start" in results


def test_reference_ceiling_is_capped_so_a_loud_synthesis_clip_cannot_swamp_real_speech():
    """The critical real-runtime bug this task fixes: reference_energy is
    measured from Garvis's own *synthesized TTS file* -- a clean, loud,
    studio-domain recording -- not from anything the microphone ever
    captured. Measuring the actual installed Jarvis-High voice showed
    ordinary speech segments around RMS 4,000-9,000 and peaks near 12,000.
    Multiplied by echo_margin_ratio with no cap, that drove the effective
    threshold to 6,000-18,000+ -- many multiples above energy_threshold/
    barge_in_energy_threshold (500/900), which were calibrated against
    *microphone-captured* speech on the same hardware -- silently
    defeating real-user barge-in for most of any reply. max_reference_multiplier
    (default 3.0, i.e. a cap of energy_threshold * 3) must keep the
    reference term from ever exceeding that, so real speech at a realistic
    microphone-domain level still gets through even while Garvis's own
    synthesis is loud."""
    vad = _make_vad(
        energy_threshold=900.0,
        min_speech_seconds=0.1,
        echo_margin_ratio=1.5,
        max_reference_multiplier=3.0,
    )
    loud_synthesis_reference = 9000.0  # a real loud segment of Jarvis-High, measured
    realistic_real_speech_amplitude = 3000  # realistic mic-domain speech on this hardware

    # Uncapped, this reference would demand energy >= 9000 * 1.5 = 13500 --
    # nowhere near reachable by realistic microphone-captured speech.
    results = [
        vad.process(
            _tone_chunk(0.05, realistic_real_speech_amplitude),
            reference_energy=loud_synthesis_reference,
        )
        for _ in range(4)
    ]

    assert "speech_start" in results
    assert vad.last_effective_threshold == pytest.approx(900.0 * 3.0, rel=0.01)


def test_last_energy_and_effective_threshold_are_introspectable():
    """Diagnostic-only properties used by ConversationSession to log exactly
    what a speech_start/barge-in decision was judged against (see the
    real-runtime self-trigger investigation)."""
    vad = _make_vad(
        energy_threshold=900.0,
        min_speech_seconds=0.1,
        echo_margin_ratio=1.5,
        max_reference_multiplier=3.0,
    )

    vad.process(_tone_chunk(0.05, 2000), reference_energy=1000.0)

    assert vad.last_energy == pytest.approx(2000.0, rel=0.01)
    # reference term: 1000*1.5=1500, below the cap (900*3=2700), so it
    # applies directly and dominates over both the fixed threshold (900)
    # and the seeded noise floor (10*1.6=16).
    assert vad.last_effective_threshold == pytest.approx(1500.0, rel=0.01)


def test_effective_threshold_reflects_the_reference_term_when_it_dominates():
    vad = _make_vad(
        energy_threshold=500.0,
        min_speech_seconds=0.1,
        echo_margin_ratio=1.5,
        max_reference_multiplier=3.0,
    )

    vad.process(_tone_chunk(0.05, 2000), reference_energy=1000.0)

    # reference term: min(1000*1.5, 500*3) = min(1500, 1500) = 1500, above
    # both the fixed threshold (500) and the seeded noise floor (10*1.6=16).
    assert vad.last_effective_threshold == pytest.approx(1500.0, rel=0.01)


def test_reference_energy_has_no_effect_once_garvis_falls_silent():
    """When the caller reports Garvis is momentarily quiet (reference_energy
    near 0, e.g. a mid-sentence pause or the leading TTS silence), a real
    interruption must be detected exactly as if barge-in weren't in play at
    all -- the echo guard must not linger past the moment it's needed."""
    vad = _make_vad(energy_threshold=900.0, min_speech_seconds=0.1, echo_margin_ratio=1.5)

    results = [
        vad.process(_tone_chunk(0.05, 2000), reference_energy=0.0) for _ in range(3)
    ]

    assert "speech_start" in results


# ------------------------------------------------------------- PlaybackEnvelope
def _tone_pcm(seconds: float, amplitude: int, sample_rate: int = _SAMPLE_RATE) -> bytes:
    return _tone_chunk(seconds, amplitude, sample_rate)


def test_playback_envelope_reports_energy_at_elapsed_time():
    pcm = _tone_pcm(1.0, 5000)
    envelope = PlaybackEnvelope(pcm, _SAMPLE_RATE, frame_seconds=0.02)

    assert envelope.energy_at(0.1) == pytest.approx(5000.0, rel=0.01)
    assert envelope.energy_at(0.5) == pytest.approx(5000.0, rel=0.01)


def test_playback_envelope_is_silent_before_and_after_the_clip():
    pcm = _tone_pcm(0.5, 5000)
    envelope = PlaybackEnvelope(pcm, _SAMPLE_RATE, frame_seconds=0.02)

    assert envelope.energy_at(-0.1) == 0.0
    assert envelope.energy_at(5.0) == 0.0


def test_playback_envelope_holds_trailing_loudness_past_its_own_estimate_when_still_playing():
    """Regression test: winsound's own playback-completion estimate (see
    AudioPlayer) can fire a little early relative to real hardware, i.e.
    real audio can still be playing after our elapsed-time estimate has
    passed the clip's nominal duration. If reference_energy_now() dropped
    to 0.0 the instant our own estimate expired, the self-echo guard would
    disable itself right as the trailing end of a reply -- exactly the
    real-world "interrupts itself" failure mode -- was still leaking into
    the mic. held_if_still_playing=True must keep reporting the clip's own
    trailing loudness instead of collapsing to silence."""
    pcm = _tone_pcm(1.0, 5000)
    envelope = PlaybackEnvelope(pcm, _SAMPLE_RATE, frame_seconds=0.02)

    past_the_end = 1.2  # beyond the clip's own 1.0s duration
    assert envelope.energy_at(past_the_end) == 0.0  # default: caller thinks it's done
    assert envelope.energy_at(past_the_end, held_if_still_playing=True) == pytest.approx(
        5000.0, rel=0.01
    )


def test_playback_envelope_tracks_a_quiet_then_loud_clip():
    """A clip that starts with silence (e.g. the leading-silence TTS fix)
    then becomes loud must report low energy early and high energy later --
    this is what lets a real interruption during the silent lead-in still
    go through undampened."""
    pcm = _tone_pcm(0.2, 0) + _tone_pcm(0.2, 6000)
    envelope = PlaybackEnvelope(pcm, _SAMPLE_RATE, frame_seconds=0.02)

    assert envelope.energy_at(0.05) < 100.0
    assert envelope.energy_at(0.3) == pytest.approx(6000.0, rel=0.01)


def test_playback_envelope_handles_empty_pcm():
    envelope = PlaybackEnvelope(b"", _SAMPLE_RATE)
    assert envelope.energy_at(0.0) == 0.0
