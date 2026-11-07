"""What Garvis says when a reminder rings (app/reminders/phrasing.py): the
rules' wording, the checks a model-written sentence must pass, and that the
one stored sentence is what every path speaks. The model is always faked
here -- no real API call."""
from __future__ import annotations

import asyncio
import json
import time
from itertools import pairwise

import pytest

from app.llm.base import LLMResponse, Message, Role
from app.reminders.phrasing import (
    ReminderPhraser,
    ReminderPhrasingWorker,
    clock_words,
    compose,
    faithful,
    language,
    voice_language,
)
from app.reminders.schedule import get_zone
from app.reminders.store import ReminderStore, phrasing_key

ROBOTIC = ("אני מתקשר", "בשביל להזכיר", "להזכיר לך:", "calling to remind", "You asked me")


# ------------------------------------------------------------------ the rules
@pytest.mark.parametrize("text, expected", [
    ("להתקשר לאמא", "היי אלכס, רק תזכור להתקשר לאמא שלך."),
    ("להגיש את העבודה בהיסטוריה", "היי אלכס, אל תשכח להגיש היום את העבודה בהיסטוריה."),
    ("פגישה עם דניאל בשעה 17:00", "היי אלכס, רק תזכורת שיש לך פגישה עם דניאל בחמש."),
    ("לקנות חלב בדרך הביתה", "היי אלכס, כשתחזור הביתה, תזכור לעצור ולקנות חלב."),
])
def test_the_users_own_examples(text, expected):
    assert compose(text, name="אלכס").text == expected


@pytest.mark.parametrize("text, category, must_have", [
    ("תזכיר לי להתקשר לאבא שלי", "call", "להתקשר לאבא שלך"),
    ("להתקשר לרופא בשעה 9:30", "call", "בתשע וחצי"),
    ("לקחת כדור", "medicine", "לקחת כדור"),
    ("ללמוד למבחן במתמטיקה", "task", "ללמוד למבחן במתמטיקה"),
    ("מבחן בפיזיקה מחר בשעה 8:45", "event", "ברבע לתשע"),
    ("לשלם חשבון חשמל עד יום חמישי", "deadline", "לשלם חשבון חשמל עד יום חמישי"),
    ("לשלוח לי את הקובץ", "deadline", "לשלוח היום לך את הקובץ"),
    ("תור לרופא שיניים בשעה 16:15", "event", "יש לך תור לרופא שיניים בארבע ורבע"),
    ("לקנות מתנה לנועה", "shopping", "לקנות מתנה לנועה"),
    ("לחם וחלב", "other", "לחם וחלב"),
])
def test_hebrew_reminders_of_every_kind(text, category, must_have):
    composed = compose(text, name="אלכס")
    assert composed.lang == "he" and composed.category == category
    assert composed.text.startswith("היי אלכס, ") and must_have in composed.text, composed.text
    assert not any(phrase in composed.text for phrase in ROBOTIC)
    assert faithful(text, composed.text, name="אלכס") is None  # the rules pass their own checks


def test_unknown_content_gets_a_neutral_wording_and_nothing_is_guessed():
    composed = compose("החבילה של יוסי", name="אלכס")
    assert composed.category == "other"
    assert composed.text in ("היי אלכס, רק מזכיר לך: החבילה של יוסי.", "היי אלכס, תזכורת בשבילך: החבילה של יוסי.")


def test_a_time_in_the_text_never_becomes_its_time_to():
    # "זה הזמן" would be false when the reminder rings before the 20:00 it mentions.
    for variant in range(4):
        assert "זה הזמן" not in compose("להתקשר לדני בשעה 20:00", name="אלכס", variant=variant).text


def test_feminine_hebrew_when_set():
    assert compose("להתקשר לאמא", name="נועה", gender="female").text == "היי נועה, רק תזכרי להתקשר לאמא שלך."
    assert "כשתחזרי הביתה, תזכרי" in compose("לקנות חלב בדרך הביתה", gender="female").text


def test_no_name_configured():
    assert compose("להתקשר לאמא").text == "היי, רק תזכור להתקשר לאמא שלך."


@pytest.mark.parametrize("text, expected", [
    ("remind me to call mom", "Hey Alex, just a reminder to call your mom."),
    ("meeting with Daniel at 17:00", "Hey Alex, just a heads-up: you have a meeting with Daniel at five."),
    ("buy milk on the way home", "Hey Alex, on your way home, remember to stop and buy milk."),
    ("submit the history paper", "Hey Alex, don't forget to submit the history paper today."),
    ("take my pills", "Hey Alex, it's time to take your pills."),
])
def test_english_reminders(text, expected):
    assert compose(text, name="Alex").text == expected


def test_call_me_about_is_the_request_not_the_content():
    assert "call you" not in compose("Call me about the meeting").text
    assert compose("E2E call me", name="Alex").text in ("Hey Alex, a quick reminder: E2E.", "Hey Alex, just reminding you: E2E.")
    assert compose("dentist appointment at 16:15").category == "event"
    assert compose("Dentist, call me").text.endswith("Dentist.")
    assert compose("רופא שיניים, תתקשר אליי", name="אלכס").text.endswith("רופא שיניים.")
    assert compose("Call me to take my pills").category == "medicine"
    assert compose("תתקשר אליי להזכיר לי לקחת כדור", name="אלכס").text == "היי אלכס, הגיע הזמן לקחת כדור."


def test_consecutive_reminders_dont_sound_the_same():
    phraser = ReminderPhraser(name_he="אלכס")
    texts = ["להתקשר לאמא", "להתקשר לאבא", "להתקשר לסבתא", "להתקשר לדני"]
    sentences = [phraser.compose(f"id{i}", t).text for i, t in enumerate(texts)]
    openings = [s.split(" ", 3)[2] for s in sentences]
    assert all(a != b for a, b in pairwise(openings)), sentences
    # The stable form (what a phone asks for again later) is the same every time.
    assert phraser.compose("id0", texts[0], remember=False).text == phraser.compose("id0", texts[0], remember=False).text


def test_clock_words():
    assert clock_words(17, 0, "he") == "חמש" and clock_words(5, 30, "he") == "חמש וחצי"
    assert clock_words(8, 45, "he") == "רבע לתשע" and clock_words(12, 0, "he") == "שתים עשרה"
    assert clock_words(17, 10, "he") == "חמש ועשרה" and clock_words(17, 7, "he") is None
    assert clock_words(17, 0, "en") == "five" and clock_words(16, 15, "en") == "quarter past four"
    assert clock_words(17, 7, "en") is None


def test_language_and_voice_language():
    assert language("להתקשר לאמא") == "he" and language("call mom") == "en" and language("לקנות iPhone") == "he"
    assert voice_language("Jarvis-High") == "en" and voice_language("en_US-lessac-medium") == "en"
    assert voice_language("he_IL-model-medium") == "he"


# --------------------------------------------------------- faithfulness checks
@pytest.mark.parametrize("candidate, problem", [
    ("היי אלכס, רק לא לשכוח: בחמש יש לך פגישה עם דניאל.", None),
    ("היי אלכס, רק תזכורת שיש לך פגישה עם דניאל ויוסי בחמש במשרד בתל אביב ליד הים.", "added_details"),
    ("היי אלכס, רק תזכורת שיש לך פגישה עם דניאל בשש.", "time_dropped"),
    ("היי אלכס, רק תזכורת שיש לך פגישה עם דניאל ב-18:00.", "new_number"),
    ("היי אלכס, יש לך משהו בחמש.", "details_dropped"),
    ("Hey Alex, you have a meeting with Daniel at five.", "language"),
    ("היי, רק תזכורת שיש לך פגישה עם דניאל בחמש.", "no_name"),
    ("היי אלכס, " + "פגישה עם דניאל בחמש " * 12, "length"),
])
def test_faithful_hebrew(candidate, problem):
    assert faithful("פגישה עם דניאל בשעה 17:00", candidate, name="אלכס") == problem


def test_faithful_refuses_a_story_around_unclear_content():
    # Seen live from the model: the package "arrives today at noon" -- nothing in the reminder says so.
    assert faithful("החבילה של יוסי", "היי אלכס, רק תזכורת שהחבילה של יוסי מגיעה אליך היום בצהריים.", name="אלכס") in (
        "added_details", "added_time")
    assert faithful("החבילה של יוסי", "היי אלכס, רק תזכורת לגבי החבילה של יוסי.", name="אלכס") is None
    assert faithful("להתקשר לאמא", "היי אלכס, תזכור להתקשר לאמא שלך הערב.", name="אלכס") == "added_time"
    assert faithful("להגיש את העבודה", "היי אלכס, אל תשכח להגיש היום את העבודה.", name="אלכס") is None


def test_faithful_english_refuses_invented_names_and_times():
    original = "meeting with Daniel at 17:00"
    assert faithful(original, "Hey Alex, heads-up: you have a meeting with Daniel at five.", name="Alex") is None
    assert faithful(original, "Hey Alex, you have a meeting with Daniel and Sarah at five.", name="Alex") == "new_name"
    assert faithful(original, "Hey Alex, you have a meeting with Daniel at six.", name="Alex") == "time_dropped"
    assert faithful(original, "Hey Alex, you have a meeting at 5:30.", name="Alex") == "new_number"


# --------------------------------------------------------------- model layer
class FakeModel:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.requests: list[list[Message]] = []

    async def chat(self, messages, tools=None):
        self.requests.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(message=Message(role=Role.ASSISTANT, content=self.reply))

    async def health_check(self):
        return True


def _reply(sentence: str) -> str:
    return json.dumps({"spoken": sentence}, ensure_ascii=False)


async def test_a_model_sentence_that_keeps_the_reminder_is_used():
    model = FakeModel(_reply("היי אלכס, רק לא לשכוח: בחמש יש לך פגישה עם דניאל."))
    phraser = ReminderPhraser(name_he="אלכס", llm=model)
    phraser.compose("a", "להתקשר לאבא")  # a recent sentence the model is asked not to echo
    assert await phraser.refine("פגישה עם דניאל בשעה 17:00", when="once, Fri 31 Oct 16:30") == \
        "היי אלכס, רק לא לשכוח: בחמש יש לך פגישה עם דניאל."
    system, user = model.requests[0]
    assert system.role == Role.SYSTEM and "NEVER add anything" in system.content and "היי אלכס" in system.content
    assert "להתקשר לאבא שלך" in system.content  # don't repeat recent wordings
    assert json.loads(user.content) == {"reminder": "פגישה עם דניאל בשעה 17:00", "type_guess": "event",
                                        "rings": "once, Fri 31 Oct 16:30", "delivered_as": "a phone call"}


@pytest.mark.parametrize("reply", [
    _reply("היי אלכס, יש לך פגישה עם דניאל ועם שרה בשש."),   # invented time
    _reply("Hey Alex, meeting with Daniel at five."),        # wrong language
    "Sure! Here's a friendly reminder.",                      # not JSON
    RuntimeError("budget exceeded"),                          # no model right now
])
async def test_a_model_sentence_that_fails_the_checks_is_dropped(reply):
    phraser = ReminderPhraser(name_he="אלכס", llm=FakeModel(reply))
    assert await phraser.refine("פגישה עם דניאל בשעה 17:00") is None


async def test_no_model_means_rules_only():
    assert await ReminderPhraser().refine("להתקשר לאמא") is None


# ------------------------------------------------------------ store + worker
def _store(tmp_path, phraser=None):
    phraser = phraser or ReminderPhraser(name_he="אלכס")
    return ReminderStore(tmp_path / "reminders.json",
                         phrase=lambda r: phraser.compose(r.id, r.text).text), phraser


def test_the_store_keeps_the_sentence_in_step_with_the_text(tmp_path):
    store, _ = _store(tmp_path)
    reminder = store.create("להתקשר לאמא", tz="UTC", repeat="daily", local_time="09:00", kind="call")
    assert reminder.spoken.endswith("להתקשר לאמא שלך.") and reminder.spoken_by == "rules"
    assert reminder.public()["spoken"] == reminder.spoken and "spoken_key" not in reminder.public()
    first = reminder.spoken
    store.fired(reminder.id, reminder.due_at)  # moves on to tomorrow: same words
    assert store.get(reminder.id).spoken == first
    store.update(reminder.id, text="להתקשר לאבא")
    assert "לאבא שלך" in store.get(reminder.id).spoken


def test_a_model_sentence_for_old_text_is_refused(tmp_path):
    store, _ = _store(tmp_path)
    reminder = store.create("להתקשר לאמא", tz="UTC", due_at=time.time() + 3600, kind="call")
    old_key = reminder.spoken_key
    store.update(reminder.id, text="להתקשר לאבא")
    assert store.set_spoken(reminder.id, "היי אלכס, תתקשר לאמא שלך.", key=old_key) is False
    assert store.set_spoken(reminder.id, "היי אלכס, מה דעתך להתקשר עכשיו לאבא שלך?", key=phrasing_key(reminder)) is True
    reloaded = ReminderStore(tmp_path / "reminders.json", phrase=lambda r: "rules")
    assert reloaded.get(reminder.id).spoken == "היי אלכס, מה דעתך להתקשר עכשיו לאבא שלך?"  # survives a restart
    assert reloaded.get(reminder.id).spoken_by == "model"


def test_reminders_saved_before_phrasing_get_a_sentence(tmp_path):
    path = tmp_path / "reminders.json"
    path.write_text(json.dumps({"version": 3, "reminders": [
        {"id": "old1", "text": "להתקשר לאמא", "kind": "call", "due_at": time.time() + 600, "tz": "UTC"}]}),
        encoding="utf-8")
    store = ReminderStore(path, phrase=lambda r: compose(r.text, name="אלכס").text)
    assert store.get("old1").spoken == "היי אלכס, רק תזכור להתקשר לאמא שלך."


async def test_the_worker_refines_call_reminders_once(tmp_path):
    model = FakeModel(_reply("היי אלכס, זה הזמן להתקשר לאמא שלך."))
    phraser = ReminderPhraser(name_he="אלכס", llm=model)
    store, _ = _store(tmp_path, phraser)
    ReminderPhrasingWorker(store, phraser, asyncio.get_running_loop(), get_zone).start()
    call = store.create("להתקשר לאמא", tz="UTC", due_at=time.time() + 3600, kind="call")
    note = store.create("לקנות חלב", tz="UTC", due_at=time.time() + 3600, kind="notify")
    for _ in range(50):
        await asyncio.sleep(0.01)
        if store.get(call.id).spoken_by == "model":
            break
    assert store.get(call.id).spoken == "היי אלכס, זה הזמן להתקשר לאמא שלך."
    assert store.get(note.id).spoken_by == "rules"  # notifications aren't spoken: no model call
    assert len(model.requests) == 1
    store.snooze(call.id, 10)  # same text: not asked again
    await asyncio.sleep(0.05)
    assert len(model.requests) == 1


async def test_the_worker_keeps_the_rules_when_the_model_fails(tmp_path):
    model = FakeModel(RuntimeError("offline"))
    phraser = ReminderPhraser(name_he="אלכס", llm=model)
    store, _ = _store(tmp_path, phraser)
    ReminderPhrasingWorker(store, phraser, asyncio.get_running_loop(), get_zone).start()
    call = store.create("להתקשר לאמא", tz="UTC", due_at=time.time() + 3600, kind="call")
    await asyncio.sleep(0.05)
    assert store.get(call.id).spoken == "היי אלכס, רק תזכור להתקשר לאמא שלך." or \
        store.get(call.id).spoken_by == "rules"
    store.create("להתקשר לאבא", tz="UTC", due_at=time.time() + 3600, kind="call")  # the next change: no retry storm
    await asyncio.sleep(0.05)
    assert len(model.requests) == 2
