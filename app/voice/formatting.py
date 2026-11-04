"""Deterministic text-to-speech-friendly text adaptation.

Takes whatever text the Agent produced (the exact same text shown in the
GUI) and adapts it for speech. Used only at the voice boundary -- see
`app/ui/agent_bridge.py:AgentBridge._safe_speak` -- the GUI text itself is
never touched, so nothing here can make the text response less useful.

This is NOT a summarizer, and it makes no judgment about how much detail
an answer should contain -- that's the model's job, guided by the
personality policy's "Voice" guidance (app/personality/policy.py). It only
rewrites things that are fine to *read* in a GUI but awkward, meaningless,
or actively misleading to *hear* spoken aloud:

* raw URLs (nobody wants a query-string-laden link read aloud character by
  character; the GUI text still has it if the user wants it)
* markdown list/heading/emphasis syntax, which would otherwise be spoken
  as literal punctuation ("asterisk asterisk Restaurant A asterisk
  asterisk") -- list items become short natural sentences instead of
  disconnected bulleted lines

Deliberately conservative: it never removes substantive information (an
address, a phone number, a rating) -- only presentation syntax that was
never meant to be spoken. Whether that information should be *included* in
the first place is a content decision left entirely to the model, because
only the model (via the conversation) knows whether the user already asked
for that level of detail.

Deliberately generic: it operates on plain text, not on any tool's output
shape, so it applies equally to web search, Gmail, browser, or any other
tool-derived reply -- not just restaurant results.
"""
from __future__ import annotations

import re

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
# Lone "*"/"_" used for something other than emphasis (e.g. "3 * 4") could in
# principle false-match if there's another lone one later in the same reply
# -- accepted as a rare, low-impact edge case rather than a full markdown
# parser (see module docstring: deliberately small and deterministic).
_EMPHASIS_RE = re.compile(r"\*\*([^*]+)\*\*|\*([^*]+)\*|_([^_]+)_")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

# Natural-delivery fixes verified directly against Piper's actual espeak-ng
# frontend (piper.phonemize_espeak.EspeakPhonemizer), not guessed:
#   - An ASCII dash-as-punctuation ("--", "-", or " - ") used as a stylistic
#     em-dash (as the personality prompt itself sometimes uses) is silently
#     DROPPED with no pause substituted -- the two clauses run together with
#     no break at all. A genuine Unicode em-dash ("--") already phonemizes
#     correctly (as a natural pause), so it's left untouched; only the ASCII
#     substitute needs normalizing. Mapped to a comma, which reliably
#     produces a short natural pause.
#   - Three or more dots ("...") don't just fail to pause -- they drop the
#     word boundary entirely, merging the words on either side into one
#     garbled run ("found it... but" -> "found it-but"). Collapsed to a
#     single period, which correctly ends the sentence with a clean pause.
# Neither is a general prosody/pacing system -- both are narrow,
# deterministic fixes for punctuation Piper's frontend mishandles, not an
# attempt at dynamic rate/pause control (Piper doesn't support that).
_ELLIPSIS_RE = re.compile(r"\.{2,}")
_ASCII_DASH_RE = re.compile(r"\s+-+\s+|-{2,}")

# Proper-noun pronunciation overrides for Piper's English front end. Kept as
# a tiny, explicit lookup -- not a general spelling/phonetics system -- so
# extending it means adding one more `re.compile(r"\bName\b", re.IGNORECASE):
# "Respelling"` entry, nothing more. Each respelling is plain ASCII text
# chosen because it changes what Piper's espeak-ng English frontend produces
# for it -- verify it directly against `piper.phonemize_espeak.EspeakPhonemizer`
# before adding it. Empty by default: add the names you need (for example your
# own name, if Piper mispronounces it).
_PRONUNCIATION_OVERRIDES: dict[re.Pattern[str], str] = {}


def _apply_pronunciation_overrides(text: str) -> str:
    for pattern, replacement in _PRONUNCIATION_OVERRIDES.items():
        text = pattern.sub(replacement, text)
    return text


def to_spoken_text(text: str) -> str:
    """Return a speech-friendly version of `text`.

    Safe to call on any plain string, including one with no formatting at
    all (a no-op beyond whitespace normalization in that case), or one that
    becomes empty once a bare URL is stripped (falls back to the original
    text rather than speaking nothing).
    """
    if not text or not text.strip():
        return text

    is_list = bool(_LIST_MARKER_RE.search(text))

    spoken = _URL_RE.sub("", text)
    spoken = _HEADER_RE.sub("", spoken)
    spoken = _EMPHASIS_RE.sub(lambda m: next(g for g in m.groups() if g is not None), spoken)

    if is_list:
        items = []
        for line in spoken.splitlines():
            item = _LIST_MARKER_RE.sub("", line).strip()
            item = item.rstrip(" -–—:")
            if item:
                items.append(item)
        spoken = ". ".join(item.rstrip(".") for item in items)
    else:
        # Flatten line breaks -- normal prose reads fine as continuous
        # speech; only list-shaped text needs the ". " treatment above.
        spoken = " ".join(spoken.splitlines())

    # Applied after list/prose flattening (not on the raw multi-line text):
    # "\s+-+\s+" would otherwise also match a line-start list marker
    # ("\n- Item"), since "\n" counts as whitespace -- by this point the
    # text is already a single line, so there's no marker left to corrupt.
    spoken = _ELLIPSIS_RE.sub(".", spoken)
    spoken = _ASCII_DASH_RE.sub(", ", spoken)

    spoken = _WHITESPACE_RE.sub(" ", spoken)
    spoken = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", spoken)
    spoken = spoken.strip()
    result = spoken or text.strip()
    return _apply_pronunciation_overrides(result)
