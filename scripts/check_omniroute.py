"""Check the security posture of the local OmniRoute gateway, if it still runs.

    python scripts/check_omniroute.py --set-key     # paste OmniRoute's client API key (hidden input) into .env
    python scripts/check_omniroute.py               # verify loopback URL, key enforcement and CORS

Garvis no longer sends model requests through OmniRoute: it uses only Claude
Haiku 5.5 via the Anthropic API (single-model policy, app/llm/router.py).
This script sends NO prompts to any model -- it lists the catalog and probes
auth/CORS, so an OmniRoute left running on this PC stays locked down. The
former --tool-test (a prompt to every registry model) was removed.

The key is OmniRoute's *client* API key (Dashboard -> API Keys). It is only
ever written to the gitignored .env; this script never prints it.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.exceptions import ConfigurationError  # noqa: E402
from app.llm.omniroute import validate_base_url  # noqa: E402
from env_file import ENV_PATH, set_env_value  # noqa: E402

KEYLESS_PREFIXES = (
    "ddgw/",
    "duckduckgo-web/",
    "felo/",
    "felo-web/",
    "tllm/",
    "theoldllm/",
    "pepper/",
    "chipotle/",
    "cfp/",
    "cloudflare-playground/",
    "veo-free/",
    "veoaifree-web/",
    "dva/",
    "devin-cli-agentic/",
    "aug/",
    "auggie/",
    "zc/",
    "zcode/",
    "cxa/",
    "codex-app-server/",
    "unc/",
    "uncloseai/",
    "horde/",
    "aihorde/",
    "oc/",
    "opencode/",
)


async def verify(settings: Settings) -> bool:
    ok = True
    try:
        base_url = validate_base_url(settings.omniroute_base_url)
    except ConfigurationError as exc:
        print(f"Base URL: REFUSED -- {exc}")
        return False
    print(f"Base URL: {base_url} (loopback)")
    if not settings.omniroute_api_key:
        print("Key: NOT CONFIGURED -- no GARVIS_OMNIROUTE_API_KEY. Run: python scripts/check_omniroute.py --set-key")
        return False
    auth = {"Authorization": f"Bearer {settings.omniroute_api_key}"}

    async with httpx.AsyncClient(base_url=base_url, timeout=15) as client:
        try:
            anonymous = await client.get("/models")
        except httpx.ConnectError:
            print(
                f"Gateway: NOT REACHABLE at {base_url}. Start it with OMNIROUTE_SERVER_HOST=127.0.0.1 omniroute serve"
            )
            return False
        print(
            f"Auth: /models without a key -> HTTP {anonymous.status_code}"
            + (" (enforced)" if anonymous.status_code == 401 else " -- WARNING: API is open without a key")
        )
        ok = ok and anonymous.status_code == 401

        listed = await client.get("/models", headers=auth)
        if listed.status_code != 200:
            print(f"Key: REJECTED -- HTTP {listed.status_code}")
            return False
        ids = {m.get("id", "") for m in listed.json().get("data", [])}
        print(f"Key: accepted, {len(ids)} catalog entries (static catalog -- not proof a model is usable or free).")

        keyless = sorted(i for i in ids if i.startswith(KEYLESS_PREFIXES))
        if keyless:
            print(
                f"Keyless providers: WARNING -- {len(keyless)} still listed (e.g. {keyless[:3]}). "
                "Block them in OmniRoute (see README)."
            )
        else:
            print("Keyless providers: none listed.")

        # What matters is whether a web page can make an *unauthenticated* call.
        # OmniRoute echoes the Origin on /v1 preflights by design (token-auth
        # routes); with a key required the real request is still refused. The
        # model id below doesn't exist, so even an open gateway couldn't run it.
        cross_site = await client.post(
            "/chat/completions",
            headers={"Origin": "https://example.com"},
            json={"model": "garvis-probe-nonexistent/x", "messages": [{"role": "user", "content": "probe"}]},
        )
        allow = cross_site.headers.get("access-control-allow-origin")
        if cross_site.status_code != 401 or allow == "*":
            print(
                f"CORS: WARNING -- a website origin got HTTP {cross_site.status_code} without a key "
                f"(allow-origin={allow!r}). Enable REQUIRE_API_KEY in OmniRoute."
            )
            ok = False
        else:
            preflight = await client.options(
                "/chat/completions",
                headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
            )
            echoed = preflight.headers.get("access-control-allow-origin")
            print(
                f"CORS: website origins can't call without a key (HTTP 401). "
                f"Preflight allow-origin={echoed!r} (OmniRoute echoes it on /v1 by design; harmless while a key is required)."
            )

    print(
        "Models: Garvis sends no model requests through OmniRoute (Claude Haiku 5.5 only, via the "
        "Anthropic API); no prompt was sent."
    )
    print("RESULT:", "PASS" if ok else "FAIL -- see above.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--set-key", action="store_true", help="prompt for the OmniRoute client key and save it to .env"
    )
    args = parser.parse_args()

    if args.set_key:
        key = getpass.getpass("Paste your OmniRoute client API key (input hidden): ").strip()
        if not key:
            print("No key entered; nothing changed.")
            return 1
        set_env_value("GARVIS_OMNIROUTE_API_KEY", key)
        print(f"Saved GARVIS_OMNIROUTE_API_KEY to {ENV_PATH} (gitignored).")

    settings = Settings()  # re-read .env
    return 0 if asyncio.run(verify(settings)) else 1


if __name__ == "__main__":
    sys.exit(main())
