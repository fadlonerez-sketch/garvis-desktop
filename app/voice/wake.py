"""Wake-word standby: Garvis waits locally for a wake phrase, then hands off
to the normal hands-free ConversationSession (see app/ui/agent_bridge.py).

Deliberately NOT a second voice pipeline: StandbyListener reads the same
`VoicePipeline` microphone stream (start_continuous_listening /
read_microphone_chunk), segments it with the same `VoiceActivityDetector`
(adaptive noise floor included), and transcribes candidate segments with
the same local STT provider (`VoicePipeline.transcribe_bytes()`, i.e.
faster-whisper). Nothing in standby ever reaches the Agent, the
ConversationContext, or any LLM/API: a transcript is only ever compared
against the wake phrases by `WakePhraseMatcher` and then discarded.

Cost is bounded by gating, not by a dedicated model: the RMS VAD runs on
every chunk (cheap, pure Python), but Whisper only runs on speech segments
of plausible wake-phrase length, at most `max_segment_seconds` of audio per
segment, and never while Garvis itself is speaking (plus a short tail,
since AudioPlayer's completion time is only an estimate) or during the
post-wake cooldown. A dedicated detector (e.g. openWakeWord ONNX models)
could later replace the Whisper step behind the same `on_wake` callback.

Only one consumer may own the microphone at a time: the caller must
`stop()` standby before starting a session or push-to-talk (both share the
same recorder), and may restart it afterwards.

The reverse direction lives here too: `SleepCommandMatcher` recognizes, from
an active session's own transcript, an explicit "go to standby" command or
an unambiguous closing statement ("That's all for now.", "Thanks, Garvis."),
so the session can end and hand back to standby without calling the LLM.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.core.exceptions import SpeechToTextError
from app.core.logging import content_fingerprint, get_logger, log_extra
from app.voice.pipeline import VoicePipeline
from app.voice.vad import VoiceActivityDetector

logger = get_logger("voice.wake")

WAKE_PHRASES: tuple[str, ...] = (
    "Hey Garvis",
    "Hey Daddy",
    "Wake up, Daddy is home",
)

GREETINGS: tuple[str, ...] = (
    "Yes, Sir. Welcome back.",
    "Welcome back, Sir. I'm ready.",
    "Yes, Sir. I'm listening.",
    "At your service, Sir.",
    "Good to have you back, Sir.",
    "I'm here, Sir. How may I assist you?",
    "Certainly, Sir. I'm ready.",
    "Welcome, Sir. Systems are ready.",
    "Yes, Sir. Garvis is standing by.",
)

_TOKEN_VARIANTS: dict[str, tuple[str, ...]] = {
    # Whisper has never seen "Garvis" and spells it inconsistently; the
    # fuzzy ratio below already covers most single-letter slips, these are
    # the common ones it would otherwise miss or score low on.
    "garvis": ("garvis", "jarvis", "garvas", "garvus", "garbis", "carvis", "gervis", "jarvus"),
    "hey": ("hey", "hay", "hei", "heh"),
    "daddy": ("daddy", "daddie", "daddi", "dady", "dadi"),
    "standby": ("standby", "stanby", "standbye"),
}


def _normalize(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"(\w)'s\b", r"\1 is", text)  # "daddy's home" -> "daddy is home"
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


def _windows(words: list[str], start: int, n: int) -> list[tuple[list[str], int]]:
    """Candidate n-word windows starting at `start`, as (words, end index):
    the plain window, plus every n+1 window with one adjacent pair joined
    back together (Whisper splitting "garvis" -> "gar vis", "goodnight" ->
    "good night", "standby" -> "stand by")."""
    candidates: list[tuple[list[str], int]] = []
    if start + n <= len(words):
        candidates.append((words[start : start + n], start + n))
    if start + n + 1 <= len(words):
        window = words[start : start + n + 1]
        for i in range(n):
            merged = window[:i] + [window[i] + window[i + 1]] + window[i + 2 :]
            candidates.append((merged, start + n + 1))
    return candidates


def _score_words(
    candidate: list[str], phrase_words: list[str], threshold: float, exact_max_len: int = 0
) -> float | None:
    """Mean per-word similarity, or None if any word falls below `threshold`.
    Phrase words of at most `exact_max_len` letters must match exactly (or
    a known variant): a fuzzy ratio on short words is too loose ("test" vs
    "rest" is 0.75)."""
    scores = []
    for word, target in zip(candidate, phrase_words, strict=True):
        variants = _TOKEN_VARIANTS.get(target, (target,))
        if len(target) <= exact_max_len:
            ratio = 1.0 if word in variants else 0.0
        else:
            ratio = max(SequenceMatcher(None, word, v).ratio() for v in variants)
        if ratio < threshold:
            return None
        scores.append(ratio)
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class WakeMatch:
    """A detected wake phrase. `remainder` is whatever (normalized) speech
    followed the phrase in the same segment, e.g. "open chrome" from "Hey
    Garvis, open Chrome" -- carried so a future version can optionally use
    it as the first command; v1 callers drop it."""

    phrase: str
    score: float
    remainder: str


class WakePhraseMatcher:
    """Fuzzy, word-by-word match of a transcript against the wake phrases.

    Every phrase word must match a transcript word with a similarity of at
    least `threshold` (difflib ratio, against the word and its known Whisper
    spellings). A phrase word split in two by Whisper ("gar vis") is also
    tried joined back together."""

    def __init__(self, phrases: Sequence[str] = WAKE_PHRASES, threshold: float = 0.75) -> None:
        self._phrases = [(phrase, _normalize(phrase)) for phrase in phrases]
        self._threshold = threshold

    def match(self, transcript: str) -> WakeMatch | None:
        words = _normalize(transcript)
        best: WakeMatch | None = None
        for phrase, phrase_words in self._phrases:
            found = self._find(words, phrase_words)
            if found is None:
                continue
            score, end = found
            if best is None or score > best.score:
                best = WakeMatch(phrase=phrase, score=score, remainder=" ".join(words[end:]))
        return best

    def _find(self, words: list[str], phrase_words: list[str]) -> tuple[float, int] | None:
        n = len(phrase_words)
        best: tuple[float, int] | None = None
        for start in range(len(words)):
            for candidate, end in _windows(words, start, n):
                score = _score_words(candidate, phrase_words, self._threshold)
                if score is not None and (best is None or score > best[0]):
                    best = (score, end)
        return best


# ----------------------------------------------------------- sleep/standby
_STANDBY_VERBS = (
    "Go to",
    "Go into",
    "Go back to",
    "Enter",
    "Switch to",
    "Switch into",
    "Return to",
    "Back to",
    "Put yourself in",
    "Put yourself into",
)
_STANDBY_TARGETS = ("standby", "standby mode", "sleep mode")

STANDBY_COMMANDS: tuple[str, ...] = (
    *(f"{verb} {target}" for verb in _STANDBY_VERBS for target in _STANDBY_TARGETS),
    "Go to sleep",
    "Go back to sleep",
    "Time to sleep",
    "Time for bed",
    "Go to bed",
    "Get some rest",
    "Goodnight, Garvis",
)
"""Explicit "go to standby" commands, complete on their own."""

WEAK_STANDBY_COMMANDS: tuple[str, ...] = (
    "Standby",  # also "stand by" (see _windows)
    "Standby mode",
    "Sleep",
    "Sleep mode",
    "Rest",
)
"""Bare verbs that are only a command when addressed to Garvis or given a
lead-in ("Garvis, stand by.", "You can sleep now.", "Rest for now."): on
their own they are too ambiguous ("Stand by." also means "wait", a lone
"Sleep." is a typical Whisper fragment)."""

CLOSING_PHRASES: tuple[str, ...] = (
    "That's all",
    "That's all for now",
    "That will be all",
    "That's it",
    "I'm done",
    "I am done",
    "We're done",
    "We are done",
    "We're finished",
    "We are finished",
    "Thank you, Garvis",
    "Thanks, Garvis",
    "Goodnight",
    "Goodbye",
)

_SLEEP_FILLERS: tuple[tuple[str, ...], ...] = tuple(
    tuple(f.split())
    for f in (
        "okay",
        "ok",
        "alright",
        "all right",
        "so",
        "well",
        "great",
        "perfect",
        "good",
        "cool",
        "yeah",
        "yes",
        "yep",
        "no",
        "nope",
        "sir",
        "hey",
        "please",
        "just",
        "and",
        "then",
        "thanks",
        "thank you",
        "thank you very much",
        "thank you so much",
        "for today",
        "bye",
        "i think",
        "it is",
    )
)
"""Words allowed around a closing/standby phrase -- and nothing else. They
can never match on their own: "Thanks." or "Okay." alone is not a phrase."""
_SLEEP_LEAD_INS: tuple[tuple[str, ...], ...] = tuple(
    tuple(f.split()) for f in ("now", "for now", "you can", "you may", "go ahead and", "time to", "time for you to")
)
"""Fillers that also make a weak command count: they say "do it now"."""
_POLITE_REQUESTS: tuple[tuple[str, ...], ...] = tuple(
    tuple(f.split()) for f in ("can you", "could you", "would you", "will you")
)
"""Fillers that turn a question into a request ("Could you go to standby?")."""
_NAME = ["garvis"]  # fuzzy, with its Whisper spellings ("Jarvis", "gar vis")
_NEGATIONS = frozenset({"no", "nope"})
_MAX_SLEEP_WORDS = 32
"""Standby commands are short; a longer message is never one (and isn't
worth fuzzy-matching word by word)."""
_EXACT_MAX_LEN = 4

_GOODNIGHT_ACK = "Goodnight, Sir."
_STANDBY_ACK = "Entering standby, Sir."


@dataclass(frozen=True)
class SleepMatch:
    """An utterance recognized as "end this session, back to standby".
    `phrase` is the first canonical phrase found in it."""

    phrase: str
    words: tuple[str, ...]

    @property
    def acknowledgement(self) -> str:
        """Short fixed reply spoken via the normal TTS path (never the LLM)."""
        return _GOODNIGHT_ACK if "goodnight" in self.words else _STANDBY_ACK


@dataclass(frozen=True)
class _Part:
    kind: str  # "command" | "closing" | "weak" | "filler" | "lead_in" | "polite" | "name"
    text: str
    start: int


_PHRASE_KINDS = ("command", "closing", "weak")


def _rank(parts: tuple[_Part, ...]) -> tuple[int, int, int]:
    """Prefer parses with more complete phrases, then more lead-ins/name."""
    return (
        sum(p.kind in ("command", "closing") for p in parts),
        sum(p.kind in ("lead_in", "name") for p in parts),
        sum(p.kind == "weak" for p in parts),
    )


class SleepCommandMatcher:
    """Recognizes an explicit standby command or an unambiguous closing
    statement in a session transcript or a typed message.

    Deliberately strict: the WHOLE utterance must consist of standby/closing
    phrases plus the small filler vocabulary above, with at least one real
    phrase. So "That's all for now, thank you." and "Jarvis, you can go to
    sleep now." match, but "That's it, now open Chrome", "Thanks.", "Don't
    go to sleep." and "How do I make my PC go to sleep?" do not -- a
    mid-conversation remark, a negation or a discussion never ends the
    session. On top of that:
      * a weak command ("stand by", "sleep") needs Garvis's name or a
        lead-in ("you can", "now", "go ahead and") and must not follow a
        bare "no";
      * a question ("Go to sleep?", "That's all?") only counts as a polite
        request with an explicit command ("Could you go to standby?").
    Same normalization and per-word fuzzy matching as WakePhraseMatcher,
    except that short words must match exactly. Shutting Garvis down
    ("Shut down.", "Quit.", "Exit.") is not in this vocabulary at all."""

    def __init__(
        self,
        phrases: Sequence[str] | None = None,
        threshold: float = 0.75,
    ) -> None:
        if phrases is None:
            tagged = (
                [("command", p) for p in STANDBY_COMMANDS]
                + [("closing", p) for p in CLOSING_PHRASES]
                + [("weak", p) for p in WEAK_STANDBY_COMMANDS]
            )
        else:
            tagged = [("command", p) for p in phrases]
        self._phrases = [(kind, phrase, _normalize(phrase)) for kind, phrase in tagged]
        self._fillers = (
            [("filler", f) for f in _SLEEP_FILLERS]
            + [("lead_in", f) for f in _SLEEP_LEAD_INS]
            + [("polite", f) for f in _POLITE_REQUESTS]
        )
        self._threshold = threshold

    def match(self, transcript: str) -> SleepMatch | None:
        words = _normalize(transcript)
        if not words or len(words) > _MAX_SLEEP_WORDS:
            return None
        parts = self._parse(words)
        if parts is None or not self._acceptable(parts, words, transcript.rstrip().endswith("?")):
            return None
        phrases = [p for p in parts if p.kind in _PHRASE_KINDS]
        joined = tuple(w for p in phrases for w in _normalize(p.text))
        return SleepMatch(phrase=phrases[0].text, words=joined)

    def _parse(self, words: list[str]) -> tuple[_Part, ...] | None:
        # covered[i]: the best parse of words[i:], or None if impossible.
        covered: list[tuple[_Part, ...] | None] = [None] * (len(words) + 1)
        covered[len(words)] = ()
        for i in range(len(words) - 1, -1, -1):
            options: list[tuple[_Part, ...]] = []
            for kind, filler in self._fillers:
                end = i + len(filler)
                if tuple(words[i:end]) == filler and covered[end] is not None:
                    options.append((_Part(kind, " ".join(filler), i), *covered[end]))
            for candidate, end in _windows(words, i, 1):
                if covered[end] is not None and self._score(candidate, _NAME):
                    options.append((_Part("name", "Garvis", i), *covered[end]))
            for kind, phrase, phrase_words in self._phrases:
                for candidate, end in _windows(words, i, len(phrase_words)):
                    if covered[end] is not None and self._score(candidate, phrase_words):
                        options.append((_Part(kind, phrase, i), *covered[end]))
            if options:
                covered[i] = max(options, key=_rank)
        return covered[0]

    def _score(self, candidate: list[str], phrase_words: list[str]) -> bool:
        return _score_words(candidate, phrase_words, self._threshold, _EXACT_MAX_LEN) is not None

    @staticmethod
    def _acceptable(parts: tuple[_Part, ...], words: list[str], question: bool) -> bool:
        kinds = {p.kind for p in parts}
        if kinds.isdisjoint(_PHRASE_KINDS):
            return False  # fillers only: "Thanks.", "Okay, Garvis."
        if question:
            return "polite" in kinds and "command" in kinds
        if kinds.isdisjoint({"command", "closing", "lead_in", "name"}):
            return False  # a bare weak command: "Stand by.", "Sleep."
        for part in parts:
            if part.kind == "weak" and part.start > 0 and words[part.start - 1] in _NEGATIONS:
                return False  # "No sleep."
        return True


class GreetingPicker:
    """Random formal greeting, never the same one twice in a row."""

    def __init__(self, greetings: Sequence[str] = GREETINGS, rng: random.Random | None = None) -> None:
        if not greetings:
            raise ValueError("GreetingPicker needs at least one greeting")
        self._greetings = tuple(greetings)
        self._rng = rng or random.Random()
        self._last: str | None = None

    def next(self) -> str:
        choices = [g for g in self._greetings if g != self._last] or list(self._greetings)
        self._last = self._rng.choice(choices)
        return self._last


WakeCallback = Callable[[WakeMatch], None]


class StandbyListener:
    """Local wake-phrase listener on the shared VoicePipeline microphone.

    Like ConversationSession, one instance per standby period: `start()`
    (from a coroutine on the loop that should host it), then `stop()`."""

    def __init__(
        self,
        voice: VoicePipeline,
        on_wake: WakeCallback,
        *,
        matcher: WakePhraseMatcher | None = None,
        poll_interval_seconds: float = 0.03,
        energy_threshold: float = 500.0,
        min_speech_seconds: float = 0.15,
        min_silence_seconds: float = 0.6,
        min_segment_seconds: float = 0.4,
        max_segment_seconds: float = 3.0,
        cooldown_seconds: float = 2.0,
        playback_tail_seconds: float = 0.5,
        warm_up_stt: bool = False,
        initial_quiet_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._voice = voice
        self._on_wake = on_wake
        self._matcher = matcher or WakePhraseMatcher()
        self._poll_interval = poll_interval_seconds
        self._energy_threshold = energy_threshold
        self._min_speech_seconds = min_speech_seconds
        self._min_silence_seconds = min_silence_seconds
        self._min_segment_seconds = min_segment_seconds
        self._max_segment_seconds = max_segment_seconds
        self._cooldown_seconds = cooldown_seconds
        self._playback_tail_seconds = playback_tail_seconds
        self._warm_up_stt = warm_up_stt
        self._initial_quiet_seconds = initial_quiet_seconds
        """Mic ignored this long after start(): standby is restarted right
        after a session ends, possibly while the tail of Garvis's last words
        (e.g. a "Goodnight, Sir." acknowledgement) is still audible -- a
        fresh listener has no record of that playback for the normal
        playback-tail guard to use."""
        self._clock = clock

        self._task: asyncio.Task | None = None
        self._vad: VoiceActivityDetector | None = None
        self._noise_floor: float | None = None
        self._last_playback_at: float | None = None
        self._cooldown_until = 0.0

    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def noise_floor(self) -> float | None:
        """The room's learned ambient level, for seeding the session that
        follows a wake (see ConversationSession's `initial_noise_floor`).
        None until the first calibration has finished."""
        if self._vad is not None and not self._vad.is_calibrating:
            return self._vad.noise_floor
        return self._noise_floor

    def start(self) -> None:
        """Open the mic and begin listening. Raises SpeechToTextError if the
        microphone is unavailable. Idempotent while active."""
        if self.is_active:
            return
        self._voice.start_continuous_listening()
        if self._initial_quiet_seconds > 0:
            self._cooldown_until = self._clock() + self._initial_quiet_seconds
        self._task = asyncio.ensure_future(self._run())
        log_extra(logger, 20, "standby.started")

    async def stop(self) -> None:
        """Cancel listening and release the mic. Safe to call repeatedly or
        without start()."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                log_extra(logger, 40, "standby.run_loop_error_during_stop")
            self._task = None
            log_extra(logger, 20, "standby.stopped")
        self._voice.stop_continuous_listening()

    # ------------------------------------------------------------ internals
    def _new_vad(self) -> VoiceActivityDetector:
        if self._vad is not None and not self._vad.is_calibrating:
            self._noise_floor = self._vad.noise_floor
        self._vad = VoiceActivityDetector(
            sample_rate=self._voice.sample_rate,
            energy_threshold=self._energy_threshold,
            min_speech_seconds=self._min_speech_seconds,
            min_silence_seconds=self._min_silence_seconds,
            initial_noise_floor=self._noise_floor,
        )
        return self._vad

    def _suppressed(self, now: float) -> bool:
        """Garvis's own voice (and the post-wake cooldown) must never be
        evaluated as a wake candidate."""
        if self._voice.is_speaking:
            self._last_playback_at = now
            return True
        if self._last_playback_at is not None and now - self._last_playback_at < self._playback_tail_seconds:
            return True
        return now < self._cooldown_until

    async def _run(self) -> None:
        if self._warm_up_stt:
            await self._warm_up()
        bytes_per_second = self._voice.sample_rate * 2
        max_bytes = max(2, int(self._max_segment_seconds * bytes_per_second))
        max_preroll_bytes = max(1, int(self._min_speech_seconds * 2 * bytes_per_second))

        vad = self._new_vad()
        buffer = bytearray()
        preroll = bytearray()
        segment_done = False  # capped segment already evaluated; skip to its end

        while True:
            chunk = self._voice.read_microphone_chunk()
            if chunk:
                if self._suppressed(self._clock()):
                    if vad.is_active or buffer or segment_done:
                        vad = self._new_vad()
                        buffer.clear()
                        segment_done = False
                    preroll.clear()
                else:
                    event = vad.process(chunk)
                    if event == "silence":
                        preroll.extend(chunk)
                        if len(preroll) > max_preroll_bytes:
                            del preroll[: len(preroll) - max_preroll_bytes]
                    elif event == "speech_start":
                        buffer = bytearray(preroll)
                        buffer.extend(chunk)
                        preroll.clear()
                    elif event == "speech" and not segment_done:
                        buffer.extend(chunk)
                        if len(buffer) >= max_bytes:
                            # Wake phrases are short: evaluate the opening
                            # seconds now (covers "Hey Garvis open Chrome"
                            # said in one breath) and ignore the rest.
                            segment_done = True
                            await self._evaluate(bytes(buffer[:max_bytes]))
                            buffer.clear()
                    elif event == "speech_end":
                        if not segment_done:
                            buffer.extend(chunk)
                            await self._evaluate(bytes(buffer[:max_bytes]))
                        buffer.clear()
                        segment_done = False
                        vad = self._new_vad()
            await asyncio.sleep(self._poll_interval)

    async def _warm_up(self) -> None:
        """Load the STT model up front so the first wake isn't slowed by it."""
        try:
            await self._voice.transcribe_bytes(b"\x00\x00" * self._voice.sample_rate)
        except SpeechToTextError:
            pass

    async def _evaluate(self, segment: bytes) -> None:
        seconds = len(segment) / (self._voice.sample_rate * 2)
        if seconds < self._min_segment_seconds:
            return
        try:
            text = await self._voice.transcribe_bytes(segment)
        except SpeechToTextError as exc:
            log_extra(logger, 30, "standby.transcription_failed", error=str(exc))
            return
        match = self._matcher.match(text or "")
        log_extra(
            logger,
            20,
            "standby.candidate_evaluated",
            segment_seconds=round(seconds, 2),
            matched=match is not None,
            **content_fingerprint(text or ""),
        )
        if match is None:
            return
        self._cooldown_until = self._clock() + self._cooldown_seconds
        log_extra(
            logger,
            20,
            "standby.wake_detected",
            phrase=match.phrase,
            score=round(match.score, 2),
            remainder_words=len(match.remainder.split()),
        )
        self._on_wake(match)
