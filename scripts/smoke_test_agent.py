"""Manual smoke test: exercises the real agent loop against the configured
external LLM provider (GARVIS_LLM_PROVIDER, default 'tiered').

Not part of the automated pytest suite (network + provider-key dependent).
Run with: python scripts/smoke_test_agent.py
"""

from __future__ import annotations

import asyncio

from app.agent.build import build_agent
from app.core.config import get_settings
from app.core.events import AgentEvent
from app.core.logging import configure_logging


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir)
    agent = build_agent(settings)

    async def print_event(event: AgentEvent) -> None:
        print(f"[event] {event.type}: {event.payload}")

    agent._events.subscribe(print_event)  # noqa: SLF001 (manual smoke test only)

    for prompt in [
        "What is 12 * 7?",
        "What's today's date?",
    ]:
        print(f"\n=== USER: {prompt}")
        reply = await agent.handle_user_message(prompt)
        print(f"=== GARVIS: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
