"""Garvis's personality / behavior policy: how it communicates.

See app/personality/policy.py for the actual policy text and
app/agent/policies.py:build_system_prompt() for how it's combined with the
operational/security system prompt into the single system message the
Agent sends to whichever LLMProvider is active.
"""

from __future__ import annotations
