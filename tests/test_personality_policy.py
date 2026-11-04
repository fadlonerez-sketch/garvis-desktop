"""Tests for the personality/behavior policy (app/personality/policy.py) and
its integration into the system prompt (app/agent/policies.py).

These test policy *construction and integration*, not LLM creativity: no
test here depends on a model producing any particular sentence. Providers
are never called -- ScriptedLLMProvider only records what it was sent.
"""
from __future__ import annotations

import pytest

from app.agent.agent import Agent
from app.agent.memory import SQLiteMemoryStore, MemoryItem
from app.agent.policies import _BASE_SYSTEM_PROMPT, build_system_prompt
from app.core.events import EventBus
from app.llm.base import Role
from app.llm.registry import ModelTier
from app.llm.router import TieredLLMRouter
from app.personality.policy import PERSONALITY_POLICY
from app.tools.registry import ToolRegistry

from tests.conftest import final_response


@pytest.fixture
def memory_store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "test.db")


# ------------------------------------------------------- policy content
def test_english_is_declared_the_default_language():
    lowered = PERSONALITY_POLICY.lower()
    assert "english is the default language" in lowered


def test_explicit_user_language_request_overrides_default():
    lowered = PERSONALITY_POLICY.lower()
    assert "explicitly ask" in lowered or "explicit request" in lowered
    assert "never force english" in lowered


def test_no_hebrew_default_instruction_present_anywhere():
    """Migration guard: no default-language instruction anywhere in the
    combined system prompt may prefer Hebrew (or any language) over
    English -- see the migration note in app/personality/policy.py."""
    combined = build_system_prompt().lower()
    assert "hebrew" not in combined


def test_sir_is_the_declared_form_of_address():
    lowered = PERSONALITY_POLICY.lower()
    assert '"sir" is part of who you are' in lowered
    assert "not a verbal tic" in lowered
    assert "never force it into every sentence" in lowered  # bounded, not a verbal tic


def test_concision_and_directness_rules_present():
    lowered = PERSONALITY_POLICY.lower()
    assert "answer the actual question first" in lowered
    assert "minimum text" in lowered


def test_filler_phrases_are_explicitly_discouraged():
    lowered = PERSONALITY_POLICY.lower()
    for filler in ("of course!", "absolutely!", "i'd be happy to help!"):
        assert filler in lowered  # named as an example to avoid


def test_unnecessary_questions_are_discouraged():
    lowered = PERSONALITY_POLICY.lower()
    assert "clarifying question" in lowered
    assert "shortest useful question" in lowered


def test_humor_is_present_but_explicitly_bounded():
    lowered = PERSONALITY_POLICY.lower()
    assert "humor" in lowered
    assert "occasional" in lowered
    assert "drop it entirely for serious" in lowered


def test_humor_is_never_directed_at_the_user():
    lowered = PERSONALITY_POLICY.lower()
    assert "never at the user" in lowered
    assert "never mock or belittle the user" in lowered


def test_action_oriented_computer_control_style_present():
    lowered = PERSONALITY_POLICY.lower()
    assert "opening chrome" in lowered
    assert "i will now attempt to" in lowered  # named as the style to avoid


def test_narration_of_internal_process_is_discouraged():
    """Goal 1 (natural conversation): Garvis must not narrate what it's
    about to do or is currently doing -- only the concrete result."""
    lowered = PERSONALITY_POLICY.lower()
    assert "i am currently attempting to" in lowered
    assert "please allow me to" in lowered
    assert "never announce every tiny step" in lowered


def test_extreme_concision_examples_present():
    """Goal 1: the policy must give concrete good/bad examples of extreme
    concision, not just the abstract instruction -- these are the exact
    examples specified for the natural-conversation behavior change."""
    lowered = PERSONALITY_POLICY.lower()
    assert "408" in lowered  # "what is 17 times 24?" -> "408."
    assert "24c and clear" in lowered  # "what's the weather?" -> "it's 24C and clear."


def test_natural_spoken_phrasing_examples_present():
    lowered = PERSONALITY_POLICY.lower()
    for phrase in ("yep, that's fixed", "already open", "give me a second"):
        assert phrase in lowered


def test_varied_phrasing_and_rhythm_instructed():
    """Goal 1: responses should not all sound identical in rhythm/shape."""
    lowered = PERSONALITY_POLICY.lower()
    assert "vary your phrasing and rhythm" in lowered


def test_done_sir_must_not_be_appended_after_a_substantive_answer():
    """Regression guard: a reply that already states the result (e.g.
    "Acknowledged, Sir. Your name is Alex.") must not additionally tack on
    a bare "Done, Sir." -- that combination was the reported bug. The
    policy must still allow "Done, Sir." on its own when there is nothing
    else to report."""
    lowered = PERSONALITY_POLICY.lower()
    assert "one confirmation per action" in lowered
    assert "never tack it onto a reply that has already confirmed the outcome" in lowered
    assert "never append it after a reply that already states the result" in lowered
    # the example phrasing itself must still be present -- not removed
    assert '"done, sir."' in lowered


def test_voice_guidance_present():
    """No dynamic text/voice mode switch exists in Agent (it has no notion
    of whether a reply will be spoken), so voice guidance is baked into the
    one policy that applies to every reply -- verified here at the content
    level rather than via a runtime toggle."""
    lowered = PERSONALITY_POLICY.lower()
    assert "voice" in lowered
    assert "read aloud" in lowered or "spoken" in lowered


def test_priority_order_places_correctness_above_personality():
    lowered = PERSONALITY_POLICY.lower()
    assert "personality never overrides correctness" in lowered
    assert "humor never overrides usefulness" in lowered


def test_never_fabricate_or_bluff_present():
    lowered = PERSONALITY_POLICY.lower()
    assert "never fabricate" in lowered
    assert "i'm not certain, sir" in lowered


# ------------------------------------------------------- prompt composition
def test_build_system_prompt_combines_base_and_personality_once():
    prompt = build_system_prompt()
    assert _BASE_SYSTEM_PROMPT in prompt
    assert PERSONALITY_POLICY in prompt
    # base operational content still intact (regression guard)
    assert "data" in prompt.lower()
    assert "must not" in prompt.lower() or "not obey" in prompt.lower()


def test_build_system_prompt_appends_memory_after_personality():
    items = [MemoryItem(id=1, content="User prefers dark mode.", created_at="2026-01-01T00:00:00")]
    prompt = build_system_prompt(items)
    assert prompt.index(PERSONALITY_POLICY) < prompt.index("User prefers dark mode.")


def test_build_system_prompt_has_no_duplicated_personality_text():
    prompt = build_system_prompt()
    assert prompt.count(PERSONALITY_POLICY) == 1


# --------------------------------------------------------- Agent integration
@pytest.mark.asyncio
async def test_agent_sends_exactly_one_system_message_including_personality(
    make_scripted_llm, memory_store
):
    """One clear source of truth: the Agent must never send multiple/
    competing system messages -- personality is folded into the same single
    system message as the operational rules."""
    llm = make_scripted_llm([final_response("Hello, Sir.")])
    agent = Agent(llm=llm, registry=ToolRegistry(), memory=memory_store, events=EventBus())

    await agent.handle_user_message("hi")

    system_messages = [m for m in llm.calls[0] if m.role == Role.SYSTEM]
    assert len(system_messages) == 1
    assert "sir" in system_messages[0].content.lower()
    assert "english is the default language" in system_messages[0].content.lower()


@pytest.mark.asyncio
async def test_personality_reaches_the_llm_transparently_through_the_router(memory_store, make_scripted_llm):
    """The personality policy must reach the model through the router --
    proving the layering (Agent -> personality in system prompt ->
    TieredLLMRouter -> provider) doesn't drop or duplicate it."""
    simple = make_scripted_llm([final_response("Hello, Sir.")])
    capable = make_scripted_llm([])
    router = TieredLLMRouter({ModelTier.SMALL: [("small", simple)], ModelTier.MEDIUM: [("medium", capable)],
                              ModelTier.LARGE: [("large", make_scripted_llm([]))]})
    agent = Agent(llm=router, registry=ToolRegistry(), memory=memory_store, events=EventBus())

    await agent.handle_user_message("hi")

    system_messages = [m for m in simple.calls[0] if m.role == Role.SYSTEM]
    assert len(system_messages) == 1
    assert PERSONALITY_POLICY in system_messages[0].content
    assert capable.calls == []
