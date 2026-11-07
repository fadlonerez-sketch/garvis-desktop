"""What Garvis says when a reminder rings: one short, natural sentence in
the reminder's own language, fitted to what the reminder is about --
"היי אלכס, רק תזכור להתקשר לאמא שלך." rather than "I'm calling to remind
you: להתקשר לאמא".

Two layers, one result stored on the reminder (`Reminder.spoken`) so every
path says the same thing -- the PC's voice, the phone's own voice when the
PC is away, and every paired phone:

- `ReminderPhraser.compose()`: deterministic rules for Hebrew and English.
  Instant and always available; set the moment a reminder is created or its
  text changes. Recognizes calls, deadlines, events/meetings, shopping,
  medicine and plain tasks; anything else gets a neutral wording (it never
  guesses).
- `ReminderPhraser.refine()`: asks the assistant's existing model (the
  shared LLM router -- no new model, provider or key) for a sentence that
  fits better, then `faithful()` checks it against the reminder: same
  language, the reminder's details kept, no number or name that isn't in it,
  the time still said, short. If anything fails, the rules' sentence stays.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.core.logging import get_logger, log_extra

logger = get_logger("reminders.phrasing")

_HEBREW = re.compile(r"[א-ת]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[\w'׳״\"-]+", re.UNICODE)
_DIGITS = re.compile(r"\d+")
MAX_SPOKEN = 220


def language(text: str) -> str:
    """"he" or "en": the script the reminder starts in ("לקנות iPhone" is
    Hebrew, "Call Dana at דיזנגוף" English)."""
    first = re.search(r"[A-Za-zא-ת]", text or "")
    return "he" if first is not None and _HEBREW.match(first.group(0)) else "en"


# ----------------------------------------------------------------- clock words
_HE_HOURS = {1: "אחת", 2: "שתיים", 3: "שלוש", 4: "ארבע", 5: "חמש", 6: "שש", 7: "שבע", 8: "שמונה", 9: "תשע",
             10: "עשר", 11: "אחת עשרה", 12: "שתים עשרה"}
_HE_MINUTES = {0: "", 5: " וחמישה", 10: " ועשרה", 15: " ורבע", 20: " ועשרים", 30: " וחצי"}
_EN_HOURS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 11: "eleven", 12: "twelve"}


def clock_words(hour: int, minute: int, lang: str) -> str | None:
    """17:00 -> "חמש" / "five"; 17:30 -> "חמש וחצי" / "half past five". None
    for a time people don't say in words (17:10 in English) -- it stays digits."""
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    h12 = hour % 12 or 12
    nxt = h12 % 12 + 1
    if lang == "he":
        if minute == 45:
            return "רבע ל" + _HE_HOURS[nxt]
        return _HE_HOURS[h12] + _HE_MINUTES[minute] if minute in _HE_MINUTES else None
    return {0: _EN_HOURS[h12], 15: "quarter past " + _EN_HOURS[h12], 30: "half past " + _EN_HOURS[h12],
            45: "quarter to " + _EN_HOURS[nxt]}.get(minute)


def _times_in(text: str, lang: str) -> list[tuple[str, int, int]]:
    """(matched text, hour, minute) for the clock times written in a reminder.
    Only unambiguous ones: "17:00", "בשעה 17", "at 5 pm" -- never a bare "5"."""
    found = []
    if lang == "he":
        for m in re.finditer(r"(?:(?<![\w])בשעה\s*|(?<![\w])ב-?)?(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])", text):
            found.append((m.group(0), int(m.group(1)), int(m.group(2))))
        for m in re.finditer(r"(?<![\w])בשעה\s*([01]?\d|2[0-3])(?![\d:])", text):
            found.append((m.group(0), int(m.group(1)), 0))
    else:
        for m in re.finditer(r"\b(?:at\s+)?([01]?\d|2[0-3]):([0-5]\d)\s*(am|pm|a\.m\.|p\.m\.)?(?![\d:])", text, re.I):
            hour = int(m.group(1)) + (12 if (m.group(3) or "").lower().startswith("p") and int(m.group(1)) < 12 else 0)
            found.append((m.group(0), hour, int(m.group(2))))
        for m in re.finditer(r"\b(?:at\s+)?([1-9]|1[0-2])\s*(am|pm|a\.m\.|p\.m\.)(?![\w])", text, re.I):
            hour = int(m.group(1)) % 12 + (12 if m.group(2).lower().startswith("p") else 0)
            found.append((m.group(0), hour, 0))
    return found


def _speak_times(text: str, lang: str) -> str:
    for matched, hour, minute in _times_in(text, lang):
        words = clock_words(hour, minute, lang)
        if words is None:
            continue
        if lang == "he":
            text = text.replace(matched, "ב" + words, 1)
        else:
            text = text.replace(matched, "at " + words, 1)
    return text


# ------------------------------------------------------------ rewording rules
_HE_PREFIX_LETTERS = "ובלהמשכ"
_HE_REQUEST = re.compile(
    r"^(?:בבקשה\s+)?(?:(?:תתקשר|להתקשר|תצלצל)\s+(?:אליי|אלי)\s*(?:בקשר\s+ל|לגבי\s+|על\s+|ב-?|כדי\s+)?"
    r"(?:(?:להזכיר|ולהזכיר|תזכיר)\s+לי\s*)?|תזכיר(?:י)?\s+לי|להזכיר\s+לי|תזכורת)\s*[:,-]?\s*(?:ש(?=\S{3,}))?")
_EN_REQUEST = re.compile(
    r"^(?:please\s+)?(?:call\s+me\s+(?:about|regarding|re:?|to\s+remind\s+me\s+(?:to|about)|to|for)\s+"
    r"|remind\s+me\s+(?:to\s+|that\s+|about\s+)?|reminder\s*[:,-]?\s*|remember\s+to\s+|don'?t\s+forget\s+to\s+)",
    re.I)
_HE_FIRST_PERSON = {"שלי": "שלך", "לי": "לך", "אותי": "אותך", "איתי": "איתך", "אליי": "אליך", "בשבילי": "בשבילך"}
_EN_FIRST_PERSON = {"my": "your", "me": "you", "myself": "yourself", "mine": "yours"}
_HE_FAMILY = {"אמא", "אבא", "סבתא", "סבא"}
_EN_FAMILY = re.compile(r"(?<!your )(?<!my )(?<!his )(?<!her )(?<!their )(?<!our )(?<!the )\b(mom|mum|dad|grandma|grandpa)\b(?!'s)", re.I)

_HE_CALL = {"להתקשר", "לטלפן", "להרים"}
_HE_DEADLINE = {"להגיש", "לשלוח", "לסיים", "למסור", "לשלם", "להחזיר", "לחדש"}
_HE_SHOP = {"לקנות"}
_HE_EVENT = {"פגישה", "ישיבה", "תור", "ראיון", "מבחן", "בוחן", "שיעור", "אימון", "טיסה", "הרצאה", "ארוחה", "ארוחת",
             "מסיבה", "חתונה", "שיחה", "זום", "משמרת", "הופעה", "סדנה"}
_HE_MEDICINE = re.compile(r"^לקחת\s.*(כדור|תרופ|ויטמין|אנטיביוטיקה|משאף)")
_HE_NOT_VERBS = {"לחם", "לילה", "לוח", "לימון", "לקוח", "לקוחה"}
_HE_TIME_WORDS = re.compile(r"(?<![\w])(היום|מחר|מחרתיים|הערב|הבוקר|בערב|בבוקר|בצהריים|הלילה|השבוע|עד|לפני|אחרי|ביום|בשעה)(?![\w])|\d")
_EN_CALL = {"call", "phone", "ring"}
_EN_DEADLINE = {"submit", "send", "finish", "hand", "pay", "return", "renew", "file"}
_EN_SHOP = {"buy", "get", "pick"}
_EN_EVENT = {"meeting", "appointment", "interview", "exam", "test", "class", "lesson", "flight", "dinner", "lunch",
             "training", "practice", "call", "party", "wedding", "shift", "zoom", "lecture"}
_EN_VERBS = _EN_CALL | _EN_DEADLINE | _EN_SHOP | {
    "take", "check", "email", "text", "write", "read", "study", "clean", "book", "cancel", "order", "bring", "feed",
    "water", "walk", "go", "visit", "meet", "prepare", "print", "sign", "charge", "drink", "eat", "start", "stop",
    "turn", "open", "close", "wash", "fix", "move", "put", "message", "reply", "answer", "ask", "tell", "pack",
    "leave", "update", "review", "practice", "exercise", "plan", "schedule", "confirm", "make", "do", "see", "watch"}
_EN_TIME_WORDS = re.compile(r"\b(today|tonight|tomorrow|this (morning|evening|afternoon|week)|by|before|until|at \d|on \w+day)\b|\d", re.I)
_EN_MEDICINE = re.compile(r"^take\b.*\b(pill|pills|medicine|medication|meds|vitamin|vitamins|antibiotic|inhaler)", re.I)
_HE_HOME_WAY = re.compile(r"\s*(?<![\w])בדרך\s+הביתה(?![\w])\s*")
_EN_HOME_WAY = re.compile(r"\s*\bon\s+(?:the|my)\s+way\s+home\b\s*", re.I)


def _he_core(token: str) -> str:
    """A Hebrew word without its attached prefixes (ו, ב, ל, ה, מ, ש, כ)."""
    core = token
    while len(core) > 3 and core[0] in _HE_PREFIX_LETTERS:
        core = core[1:]
    return core


_HE_TRAILING_REQUEST = re.compile(r"\s*[,-]?\s*(?:תתקשר|תצלצל)\s+(?:אליי|אלי)\s*$")
_EN_TRAILING_REQUEST = re.compile(r"\s*[,-]?\s*(?:please\s+)?call\s+me\s*$", re.I)


def _clean(text: str, lang: str) -> str:
    text = " ".join(str(text).split()).strip().rstrip(".!؟?,;:").strip()
    text = (_HE_REQUEST if lang == "he" else _EN_REQUEST).sub("", text).strip()
    # "Dentist, call me" / "רופא, תתקשר אליי": the request to ring, not what the reminder is about.
    text = (_HE_TRAILING_REQUEST if lang == "he" else _EN_TRAILING_REQUEST).sub("", text).strip() or text
    if lang == "en" and text.lower().startswith("to "):
        text = text[3:]
    return text


def _second_person(text: str, lang: str) -> str:
    words = text.split(" ")
    if lang == "he":
        # "להתקשר לאמא שלי" -> "להתקשר לאמא שלך"; "להתקשר לאמא" -> "להתקשר לאמא שלך" (as the user says it).
        words = [_HE_FIRST_PERSON.get(w, w) for w in words]
        out = []
        for i, word in enumerate(words):
            bare = word.rstrip(",.")
            nxt = words[i + 1] if i + 1 < len(words) else ""
            family = bare in _HE_FAMILY or any(bare.startswith(p) and bare[len(p):] in _HE_FAMILY for p in ("ל", "ול", "ו"))
            if family and nxt not in ("שלך", "של", "שלו", "שלה"):
                word = bare + " שלך" + word[len(bare):]
            out.append(word)
        return " ".join(out)
    words = [_EN_FIRST_PERSON.get(w.lower(), w) if w.lower() in _EN_FIRST_PERSON else w for w in words]
    return _EN_FAMILY.sub(lambda m: "your " + m.group(1).lower(), " ".join(words))


def _category(text: str, lang: str) -> str:
    first = text.split(" ", 1)[0].strip(",.:").lower() if text else ""
    if lang == "he":
        core = first
        if core in _HE_CALL:
            return "call"
        if _HE_MEDICINE.search(text):
            return "medicine"
        if core in _HE_DEADLINE:
            return "deadline"
        if core in _HE_SHOP:
            return "shopping"
        if core in _HE_EVENT or (core == "יום" and "הולדת" not in text):
            return "event"
        if core.startswith("ל") and len(core) >= 3 and core not in _HE_NOT_VERBS:
            return "task"
        return "other"
    if first in _EN_CALL and not re.match(r"^call\s+(with|at)\b", text, re.I):
        return "call"
    if _EN_MEDICINE.search(text):
        return "medicine"
    if first in _EN_DEADLINE:
        return "deadline"
    if first in _EN_SHOP:
        return "shopping"
    words = re.sub(r"^(a|an|the|your|my)\s+", "", text, flags=re.I).split()
    noun = words[0].lower() if words else ""
    # "dentist appointment (at 4)": the event word second, then nothing or a preposition.
    compound = (len(words) > 1 and words[1].lower() in _EN_EVENT and first not in _EN_VERBS
                and (len(words) == 2 or words[2].lower() in ("with", "at", "on", "in", "for", "tomorrow", "today")))
    if noun in _EN_EVENT or compound:
        return "event"
    if first in _EN_VERBS:
        return "task"
    return "other"


# Templates: {h} greeting, {x} the reminder reworded; Hebrew verbs follow the user's gender.
_HE = {
    "call": ["{h}, רק {remember} {x}.", "{h}, זה הזמן {x}.", "{h}, אל {forget} {x}."],
    "call_timed": ["{h}, רק {remember} {x}.", "{h}, אל {forget} {x}."],
    "deadline": ["{h}, אל {forget} {x}.", "{h}, רק {remember} {x}.", "{h}, תזכורת קטנה: צריך {x}."],
    "event": ["{h}, רק תזכורת שיש לך {x}.", "{h}, תזכורת קטנה: יש לך {x}.", "{h}, {notice}, יש לך {x}."],
    "shopping": ["{h}, אל {forget} {x}.", "{h}, רק {remember} {x}."],
    "shopping_home": ["{h}, {when_home} הביתה, {remember} לעצור ו{x}.", "{h}, בדרך הביתה, אל {forget} {x}."],
    "medicine": ["{h}, הגיע הזמן {x}.", "{h}, זה הזמן {x}."],
    "medicine_timed": ["{h}, אל {forget} {x}.", "{h}, רק {remember} {x}."],
    "task": ["{h}, רק {remember} {x}.", "{h}, אל {forget} {x}.", "{h}, זה הזמן {x}."],
    "task_timed": ["{h}, רק {remember} {x}.", "{h}, אל {forget} {x}."],
    "other": ["{h}, רק מזכיר לך: {x}.", "{h}, תזכורת בשבילך: {x}."],
}
_HE_GENDER = {
    "male": {"remember": "תזכור", "forget": "תשכח", "notice": "שים לב", "when_home": "כשתחזור"},
    "female": {"remember": "תזכרי", "forget": "תשכחי", "notice": "שימי לב", "when_home": "כשתחזרי"},
}
_EN = {
    "call": ["{h}, just a reminder to {x}.", "{h}, don't forget to {x}.", "{h}, it's time to {x}."],
    "call_timed": ["{h}, just a reminder to {x}.", "{h}, don't forget to {x}."],
    "deadline": ["{h}, don't forget to {x}.", "{h}, just a reminder to {x}."],
    "event": ["{h}, just a heads-up: you have {x}.", "{h}, quick reminder, you have {x}."],
    "shopping": ["{h}, don't forget to {x}.", "{h}, just a reminder to {x}."],
    "shopping_home": ["{h}, on your way home, remember to stop and {x}.", "{h}, on the way home, don't forget to {x}."],
    "medicine": ["{h}, it's time to {x}.", "{h}, time to {x}."],
    "medicine_timed": ["{h}, don't forget to {x}.", "{h}, just a reminder to {x}."],
    "task": ["{h}, just remember to {x}.", "{h}, don't forget to {x}.", "{h}, it's time to {x}."],
    "task_timed": ["{h}, just remember to {x}.", "{h}, don't forget to {x}."],
    "other": ["{h}, a quick reminder: {x}.", "{h}, just reminding you: {x}."],
}


def _decap(text: str) -> str:
    """Lower-case a capitalized first word ("Meeting" -> "meeting"), never an acronym or name-like
    word with more capitals ("E2E", "iPhone")."""
    first = text.split(" ", 1)[0]
    return text[:1].lower() + text[1:] if first[1:] == first[1:].lower() else text


@dataclass(frozen=True)
class Composed:
    text: str
    lang: str
    category: str
    template: str


def compose(text: str, *, name: str = "", gender: str = "male", variant: int = 0,
            avoid: str = "") -> Composed:
    """The rules' sentence for a reminder's text. `variant` picks among the
    category's wordings; `avoid` (a template id) is skipped when possible so
    two reminders in a row don't sound the same."""
    lang = language(text)
    body = _clean(text, lang) or " ".join(str(text).split())
    category = _category(body, lang)
    timed = bool((_HE_TIME_WORDS if lang == "he" else _EN_TIME_WORDS).search(body))
    x = _second_person(_speak_times(body, lang), lang)
    key = category
    if lang == "he":
        if category == "deadline" and not timed:
            first, _, rest = x.partition(" ")
            x = f"{first} היום {rest}".strip()
        if category == "shopping" and _HE_HOME_WAY.search(x):
            x, key = _HE_HOME_WAY.sub(" ", x).strip(), "shopping_home"
        elif category in ("call", "medicine", "task") and timed:
            key = category + "_timed"
    else:
        if category == "deadline" and not timed:
            x = f"{x} today"
        if category == "shopping" and _EN_HOME_WAY.search(x):
            x, key = _EN_HOME_WAY.sub(" ", x).strip(), "shopping_home"
        elif category in ("call", "medicine", "task") and timed:
            key = category + "_timed"
        if category == "event" and not re.match(r"^(a|an|the|your)\b", x, re.I):
            x = ("an " if x[:1].lower() in "aeiou" else "a ") + _decap(x)
        if category in ("call", "deadline", "shopping", "medicine", "task"):
            x = _decap(x)
    templates = (_HE if lang == "he" else _EN)[key]
    index = variant % len(templates)
    if f"{key}:{index}" == avoid and len(templates) > 1:
        index = (index + 1) % len(templates)
    greeting = ("היי" if lang == "he" else "Hey") + (f" {name}" if name else "")
    words = _HE_GENDER.get(gender, _HE_GENDER["male"]) if lang == "he" else {}
    sentence = templates[index].format(h=greeting, x=x, **words)
    sentence = re.sub(r"\s+([,.:])", r"\1", " ".join(sentence.split())).replace("..", ".")
    return Composed(sentence, lang, category, f"{key}:{index}")


# ------------------------------------------------------------- faithfulness
_HE_STOP = {"את", "עם", "של", "על", "אל", "גם", "רק", "או", "לי", "שלי", "לך", "שלך", "אני", "זה", "זאת", "בשעה",
            "יש", "כל", "הזה", "הזאת", "צריך", "צריכה", "בבקשה", "תזכיר", "להזכיר", "תזכורת", "ש", "ו", "ב", "ל", "ה"}
_EN_STOP = {"to", "the", "a", "an", "at", "on", "in", "with", "and", "or", "of", "for", "my", "me", "i", "your",
            "you", "remind", "reminder", "please", "remember", "don't", "dont", "forget", "am", "pm", "is", "be"}


_PHRASING_WORDS = {
    "he": {"היי", "הי", "רק", "אל", "תשכח", "תשכחי", "לשכוח", "תזכור", "תזכרי", "לזכור", "תזכורת", "קטנה", "זה", "הזמן",
           "הגיע", "יש", "לך", "שלך", "היום", "כשתחזור", "כשתחזרי", "הביתה", "לעצור", "שים", "שימי", "לב", "צריך",
           "לא", "מזכיר", "מזכירה", "בשבילך", "עכשיו", "את", "עם", "של", "אז", "בדיוק", "כבר", "פשוט", "שאתה", "שאת",
           "אתה", "חשוב", "בדרך", "תורך", "מחכה", "לבצע", "ואל"},
    "en": {"hey", "hi", "just", "a", "an", "reminder", "don't", "dont", "forget", "remember", "to", "it's", "its", "time",
           "you", "have", "your", "quick", "heads-up", "heads", "up", "today", "on", "way", "home", "stop", "and",
           "the", "at", "now", "about", "that", "is", "so", "got", "coming", "for", "friendly", "reminding"},
}
MAX_ADDED_WORDS = 5
"""Words a model sentence may add beyond the reminder's own, the greeting and
everyday reminder wording. More than this is padding at best, invented
details at worst (a Hebrew name can't be told from any other word)."""


def _tokens(text: str, lang: str) -> list[str]:
    words = [w.strip("'\"-") for w in _WORD.findall(text)]
    if lang == "he":
        return [_he_core(w) for w in words if w]
    return [w.lower() for w in words if w]


def _content_words(text: str, lang: str) -> list[str]:
    body = _clean(text, lang)
    for matched, _, _ in _times_in(body, lang):
        body = body.replace(matched, " ")
    stop = _HE_STOP if lang == "he" else _EN_STOP
    raw = [w for w in _WORD.findall(body) if w and not _DIGITS.fullmatch(w)]
    return [t for w, t in zip(raw, _tokens(" ".join(raw), lang), strict=False) if w.lower() not in stop and t not in stop]


def _has(token: str, pool: set[str]) -> bool:
    return token in pool or any(len(token) >= 3 and len(p) >= 3 and (p.startswith(token) or token.startswith(p))
                                for p in pool)


def faithful(original: str, candidate: str, *, name: str = "") -> str | None:
    """None if `candidate` is a faithful spoken version of the reminder
    `original`, else why not (for the log)."""
    lang = language(original)
    candidate = candidate.strip()
    if not candidate or "\n" in candidate or len(candidate) > MAX_SPOKEN:
        return "length"
    if len(candidate) > 2 * len(original) + 80:
        return "too_long_for_the_reminder"
    if language(candidate) != lang:
        return "language"
    if name and name.lower() not in candidate.lower():
        return "no_name"
    allowed_digits = set(_DIGITS.findall(original))
    if any(d not in allowed_digits for d in _DIGITS.findall(candidate)):
        return "new_number"
    pool = set(_tokens(candidate, lang))
    for _, hour, minute in _times_in(_clean(original, lang), lang):
        words = clock_words(hour, minute, lang)
        said_digits = f"{hour}" in candidate or f"{hour % 12 or 12}" in candidate
        if not said_digits and not (words and all(_has(t, pool) for t in _tokens(words, lang))):
            return "time_dropped"
    content = _content_words(original, lang)
    missing = [w for w in content if not _has(w, pool)]
    if len(missing) > (len(content) - 1) // 3:
        return "details_dropped"
    known = set(_tokens(original, lang)) | set(_tokens(name, lang)) | set(_tokens(" ".join(_PHRASING_WORDS[lang]), lang))
    for _, hour, minute in _times_in(_clean(original, lang), lang):
        known |= set(_tokens(clock_words(hour, minute, lang) or "", lang))
    added = [t for t in _tokens(candidate, lang) if not _DIGITS.fullmatch(t) and not _has(t, known)]
    category = _category(_clean(original, lang), lang)
    # Unclear content ("החבילה של יוסי") is said as it is: no room for a story around it.
    if len(added) > (1 if category == "other" else MAX_ADDED_WORDS):
        return "added_details"
    # No new when: "today" only where the rules say it too (a deadline that rings on the day).
    time_words = _HE_TIME_WORDS if lang == "he" else _EN_TIME_WORDS
    said = {m.group(0).lower() for m in time_words.finditer(candidate)} - {m.group(0).lower() for m in time_words.finditer(original)}
    if said - ({"היום", "today"} if category == "deadline" else set()) - set("0123456789"):
        return "added_time"
    if lang == "en":
        known = {w.lower() for w in _WORD.findall(original)} | {name.lower(), "i", "hey", "hi"}
        sentence_starts = {m.group(1).lower() for m in re.finditer(r"(?:^|[.:!?]\s+)(\w+)", candidate)}
        for word in re.findall(r"\b[A-Z][a-z]+\b", candidate):
            if word.lower() not in known and word.lower() not in sentence_starts:
                return "new_name"
    return None


# ------------------------------------------------------------------ phraser
_SYSTEM = """You write the ONE sentence Garvis says out loud when it phones its user about a reminder. \
Garvis is the user's personal assistant and talks like a warm, relaxed friend -- never like an automated system.

Rules:
- Answer with JSON only: {{"spoken": "<the sentence>"}}
- Same language as the reminder. Hebrew: natural, everyday spoken Israeli Hebrew with correct grammar, \
speaking to a {gender_he}.
- Start with "{greeting},".
- Fit the wording to what the reminder is: a call to make, a task, a deadline, a meeting or event, shopping, \
medicine, studying. Don't use the same pattern for everything.
- Keep every detail of the reminder and its exact meaning. NEVER add anything that isn't in it: no new names, \
times, days, places, amounts, reasons or advice. If the meaning is unclear, use a neutral wording -- don't guess.
- Say written clock times the way people say them aloud (17:00 -> "בחמש" / "at five").
- Talk to the user ("your mom", "אמא שלך"), not about yourself. Never say "I'm calling to remind you".
- One short sentence, at most about 20 words. No emojis, no quotation marks.
{avoid}
Examples (Hebrew, user Alex):
- להתקשר לאמא -> היי אלכס, רק תזכור להתקשר לאמא שלך.
- להגיש את העבודה בהיסטוריה -> היי אלכס, אל תשכח להגיש היום את העבודה בהיסטוריה.
- פגישה עם דניאל בשעה 17:00 -> היי אלכס, רק תזכורת שיש לך פגישה עם דניאל בחמש.
- לקנות חלב בדרך הביתה -> היי אלכס, כשתחזור הביתה, תזכור לעצור ולקנות חלב."""


class ReminderPhraser:
    """Rules now, a model-refined sentence when the model is available."""

    def __init__(self, *, name_en: str = "", name_he: str = "", gender: str = "male",
                 llm=None, timeout: float = 15.0) -> None:
        self.name_en = name_en.strip()
        self.name_he = name_he.strip() or self.name_en
        self.gender = gender if gender in _HE_GENDER else "male"
        self._llm = llm
        self._timeout = timeout
        self._last_template = ""
        self._recent: deque[str] = deque(maxlen=6)

    @property
    def can_refine(self) -> bool:
        return self._llm is not None

    def use_model(self, llm) -> None:
        self._llm = llm

    def name_for(self, lang: str) -> str:
        return self.name_he if lang == "he" else self.name_en

    def compose(self, reminder_id: str, text: str, *, remember: bool = True) -> Composed:
        """The rules' sentence for a reminder. `remember` (a new reminder):
        steer clear of the wording used last and remember this one;
        False: the stable sentence for this reminder, history untouched."""
        lang = language(text)
        variant = int(hashlib.sha1(reminder_id.encode()).hexdigest()[:6], 16)
        composed = compose(text, name=self.name_for(lang), gender=self.gender, variant=variant,
                           avoid=self._last_template if remember else "")
        if remember:
            self._last_template = composed.template
            self._recent.append(composed.text)
        return composed

    async def refine(self, text: str, *, when: str = "", kind: str = "call") -> str | None:
        """A model-written sentence that passed `faithful()`, or None."""
        if self._llm is None:
            return None
        from app.llm.base import Message, Role
        from app.llm.usage import task_meter

        lang = language(text)
        name = self.name_for(lang)
        greeting = ("היי" if lang == "he" else "Hey") + (f" {name}" if name else "")
        hint = compose(text, name=name, gender=self.gender)
        avoid = ("- Don't start like these recent ones: " + " | ".join(self._recent)) if self._recent else ""
        system = _SYSTEM.format(greeting=greeting, avoid=avoid,
                                gender_he="woman (feminine forms)" if self.gender == "female" else "man (masculine forms)")
        user = json.dumps({"reminder": text, "type_guess": hint.category, "rings": when,
                           "delivered_as": "a phone call" if kind == "call" else "a notification"}, ensure_ascii=False)
        try:
            with task_meter() as meter:
                meter.category = "reminder_phrasing"
                response = await asyncio.wait_for(
                    self._llm.chat([Message(role=Role.SYSTEM, content=system), Message(role=Role.USER, content=user)]),
                    timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 -- no key, budget, offline, timeout: the rules' sentence stays
            log_extra(logger, 30, "reminder.phrasing_unavailable", error_type=type(exc).__name__)
            return None
        candidate = _parse(response.message.content)
        problem = "unparseable" if candidate is None else faithful(text, candidate, name=name)
        if problem is not None:
            log_extra(logger, 30, "reminder.phrasing_rejected", reason=problem)
            return None
        self._recent.append(candidate)
        return candidate


def _parse(content: str) -> str | None:
    match = re.search(r"\{.*\}", content or "", re.DOTALL)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0)).get("spoken")
    except (ValueError, AttributeError):
        return None
    return " ".join(value.split()) if isinstance(value, str) else None


def describe_when(due_at: float | None, tz_name: str, repeat: str, local_time: str, days: list[int],
                  zone: Callable[[str], dt.tzinfo]) -> str:
    """How the reminder rings, for the model ("once, Fri 31 Oct 17:00")."""
    if repeat != "none":
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        on = f" on {', '.join(names[d] for d in days if 0 <= d < 7)}" if repeat == "weekly" else ""
        return f"{repeat}{on} at {local_time}"
    if due_at is None:
        return ""
    try:
        local = dt.datetime.fromtimestamp(due_at, zone(tz_name))
    except Exception:  # noqa: BLE001
        return ""
    return "once, " + local.strftime("%a %d %b %H:%M")


def voice_language(voice_name: str) -> str:
    """The language of a Piper voice from its name ("he_IL-..." -> "he").
    Garvis's own voices (Jarvis-High) and en_US-* speak English."""
    prefix = (voice_name or "").split("-", 1)[0].split("_", 1)[0].lower()
    return prefix if len(prefix) == 2 and prefix.isalpha() else "en"


class ReminderPhrasingWorker:
    """Replaces the rules' sentence of scheduled CALL reminders with a
    checked model-written one, in the background on the assistant's loop.
    One attempt per text (a refusal or failure isn't retried until the text
    changes), one at a time; notifications keep the rules' sentence."""

    def __init__(self, store, phraser: ReminderPhraser, loop: asyncio.AbstractEventLoop,
                 zone: Callable[[str], dt.tzinfo]) -> None:
        self._store = store
        self._phraser = phraser
        self._loop = loop
        self._zone = zone
        self._tried: set[str] = set()
        self._running = False

    def start(self) -> None:
        if not self._phraser.can_refine:
            return
        self._store.add_listener(self.kick)
        self.kick()

    def kick(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._ensure_running)
        except RuntimeError:  # loop closed (shutting down)
            pass

    def _ensure_running(self) -> None:
        if not self._running:
            self._running = True
            self._loop.create_task(self._drain())

    def _pending(self) -> list:
        return [r for r in self._store.list(include_finished=False)
                if r.kind == "call" and r.spoken_by == "rules" and r.spoken_key and r.spoken_key not in self._tried]

    async def _drain(self) -> None:
        try:
            while pending := self._pending():
                reminder = pending[0]
                self._tried.add(reminder.spoken_key)
                when = describe_when(reminder.due_at, reminder.tz, reminder.repeat, reminder.local_time,
                                     reminder.days, self._zone)
                spoken = await self._phraser.refine(reminder.text, when=when, kind=reminder.kind)
                if spoken:
                    self._store.set_spoken(reminder.id, spoken, key=reminder.spoken_key)
        finally:
            self._running = False
