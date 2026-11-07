"""Set up and verify Garvis's primary model: Claude Haiku 5.5 via the direct Anthropic API.

    python scripts/check_anthropic.py --set-key     # paste your Claude API key (hidden input) into .env
    python scripts/check_anthropic.py               # key present? accepted? model available? (no tokens used)
    python scripts/check_anthropic.py --live-test   # + one real tool round trip (2 requests, well under $0.01)

The key is stored as ANTHROPIC_API_KEY in Garvis's gitignored .env and is
never printed. --live-test sends exactly what Garvis sends (the registry's
MEDIUM entry: thinking off, effort, prompt caching, Garvis's real system
prompt) and prints each request's served model, token usage -- including
cache writes/reads -- and estimated cost, from the API's own usage fields.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import anthropic  # noqa: E402

from app.agent.policies import build_system_prompt_parts  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.exceptions import LLMError  # noqa: E402
from app.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from app.llm.base import Message, Role  # noqa: E402
from app.llm.registry import ModelTier, load_model_registry  # noqa: E402
from app.tools.calculator import CalculatorTool  # noqa: E402
from app.tools.registry import ToolRegistry  # noqa: E402
from env_file import ENV_PATH, set_env_value  # noqa: E402


def _primary_entry(settings: Settings):
    registry = load_model_registry(settings.llm_model_registry_path)
    return next((e for e in registry.tier(ModelTier.MEDIUM) if e.provider == "anthropic"), None)


def _provider(settings: Settings, entry) -> AnthropicProvider:
    return AnthropicProvider(
        api_key=settings.anthropic_api_key,
        model=entry.model,
        base_url=settings.anthropic_base_url,
        timeout_seconds=settings.anthropic_request_timeout_seconds,
        reasoning_effort=entry.reasoning_effort,
        thinking=entry.thinking,
        max_output_tokens=entry.max_output_tokens,
        prompt_cache=entry.prompt_cache,
        verify_served_model=entry.verify_served_model,
        pricing=entry.pricing,
    )


def _usage_line(label: str, response, pricing) -> str:
    u = response.usage
    if u is None:
        return f"  {label}: served by {response.served_model}, no usage reported"
    cost = pricing.cost(u) if pricing else None
    return (
        f"  {label}: served by {response.served_model} ({response.request_id}) | input {u.input_tokens}, "
        f"output {u.output_tokens}, cache read {u.cache_read_tokens}, cache write 5m {u.cache_write_5m_tokens} "
        f"/ 1h {u.cache_write_1h_tokens}" + (f" | ~${cost:.6f}" if cost is not None else "")
    )


async def verify(settings: Settings, live_test: bool) -> bool:
    if not settings.anthropic_api_key:
        print(f"Key: NOT SET -- add ANTHROPIC_API_KEY to {ENV_PATH} with: python scripts/check_anthropic.py --set-key")
        print("Until then every Garvis request fails with an error asking for the key (no other model is used).")
        return False
    print("Key: present (not shown).")
    entry = _primary_entry(settings)
    if entry is None:
        print(
            "Registry: no enabled 'anthropic' entry in the MEDIUM tier of config/models.json -- nothing uses the key."
        )
        return False
    print(f"Registry: primary = {entry.label} (thinking {entry.thinking}, prompt cache {entry.prompt_cache})")

    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, base_url=settings.anthropic_base_url, max_retries=0, timeout=30.0
    )
    try:
        model = await client.models.retrieve(entry.model)
        print(f"API: key accepted; model {model.id} is available to this account (no tokens used).")
    except anthropic.AuthenticationError:
        print("API: the key was REJECTED (401). Paste it again with --set-key.")
        return False
    except anthropic.APIStatusError as exc:
        print(f"API: error {exc.status_code}: {str(getattr(exc, 'message', exc))[:200]}")
        return False
    except anthropic.APIConnectionError:
        print("API: can't reach api.anthropic.com -- check the internet connection.")
        return False

    if not live_test:
        print("RESULT: PASS (run with --live-test for one real tool round trip and a cache check)")
        return True

    provider = _provider(settings, entry)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    stable, per_turn = build_system_prompt_parts(None, "active window 'Garvis check' (python.exe).")
    system = Message(role=Role.SYSTEM, content=f"{stable}\n\n{per_turn}", cacheable_prefix_chars=len(stable))
    history = [system, Message(role=Role.USER, content="What is 17 times 24? Use the calculator tool.")]
    try:
        first = await provider.chat(history, tools=registry.tool_specs())
        print(_usage_line("request 1", first, provider.pricing))
        if not first.message.tool_calls:
            print(f"Tool call: NOT MADE -- the model answered directly: {first.message.content[:80]!r}")
            return False
        call = first.message.tool_calls[0]
        history += [
            first.message,
            Message(role=Role.TOOL, content='{"success":true,"data":408}', tool_call_id=call.id, name=call.name),
        ]
        second = await provider.chat(history, tools=registry.tool_specs())
        print(_usage_line("request 2", second, provider.pricing))
    except LLMError as exc:
        print(f"Live test: FAILED -- {exc}")
        return False
    print(f"Tool call: OK ({call.name} {call.arguments}) -> answer {second.message.content.strip()[:60]!r}")
    if entry.prompt_cache != "off":
        read = second.usage.cache_read_tokens if second.usage else 0
        print(
            "Prompt cache: "
            + (
                f"CONFIRMED ({read} tokens read from cache on request 2)"
                if read
                else "NOT confirmed -- request 2 read nothing from the cache"
            )
        )
    print("RESULT: PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-key", action="store_true", help="prompt for the Claude API key and save it to .env")
    parser.add_argument("--live-test", action="store_true", help="also send one real tool round trip (2 requests)")
    args = parser.parse_args()

    if args.set_key:
        key = getpass.getpass("Paste your Claude API key (input hidden): ").strip()
        if not key:
            print("No key entered; nothing changed.")
            return 1
        if not key.startswith("sk-ant-"):
            print("That doesn't look like a Claude API key (they start with 'sk-ant-'); nothing changed.")
            return 1
        set_env_value("ANTHROPIC_API_KEY", key)
        print(f"Saved ANTHROPIC_API_KEY to {ENV_PATH} (gitignored). Restart Garvis to use it.")

    return 0 if asyncio.run(verify(Settings(), args.live_test)) else 1


if __name__ == "__main__":
    sys.exit(main())
