"""Image-generation providers.

Image generation is a separate service from the LLM: it never touches
the Anthropic routing, and no provider is ever enabled implicitly. With
GARVIS_IMAGE_PROVIDER unset the feature reports itself as unavailable
(and why) instead of producing anything.

Supported: "cloudflare" -- Cloudflare Workers AI, model
@cf/black-forest-labs/flux-1-schnell by default. Needs a Cloudflare account
id and an API token with the "Workers AI" permission. On Cloudflare's Free
plan requests stop at the daily free allocation (they are never billed);
on a paid Workers plan usage beyond it is billed by Cloudflare.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.core.logging import get_logger, log_extra

logger = get_logger("images.providers")


class ImageProviderError(Exception):
    """Generation failed; the message is user-facing and holds no secrets."""


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime: str


def sniff_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class ImageProvider:
    name = "base"
    max_prompt_chars = 2048

    async def generate(self, prompt: str) -> GeneratedImage:  # pragma: no cover - interface
        raise NotImplementedError


class CloudflareImageProvider(ImageProvider):
    name = "cloudflare"

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        model: str,
        steps: int = 4,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        self._token = api_token
        self._steps = max(1, min(8, steps))
        self._timeout = timeout
        self._transport = transport
        self.model = model

    async def generate(self, prompt: str) -> GeneratedImage:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    self._url,
                    json={"prompt": prompt, "steps": self._steps},
                    headers={"Authorization": f"Bearer {self._token}"},
                )
        except httpx.TimeoutException as exc:
            raise ImageProviderError("The image service took too long to answer.") from exc
        except httpx.HTTPError as exc:
            raise ImageProviderError("Couldn't reach the image service (Cloudflare Workers AI).") from exc
        if response.status_code in (401, 403):
            raise ImageProviderError(
                "Cloudflare refused the API token (check GARVIS_CLOUDFLARE_API_TOKEN and its Workers AI permission)."
            )
        if response.status_code == 429:
            raise ImageProviderError(
                "The image service's limit is reached (the daily free allocation may be used up). Try again later."
            )
        if response.status_code >= 400:
            detail = ""
            try:
                errors = response.json().get("errors") or []
                detail = "; ".join(str(e.get("message", "")) for e in errors if isinstance(e, dict))[:200]
            except ValueError:
                pass
            log_extra(logger, 30, "images.provider_error", status=response.status_code)
            raise ImageProviderError(
                f"The image service refused the request (HTTP {response.status_code})"
                + (f": {detail}" if detail else ".")
            )
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("image/"):
            data = response.content
        else:
            try:
                body = response.json()
            except ValueError as exc:
                raise ImageProviderError("The image service sent an unreadable answer.") from exc
            encoded = (body.get("result") or {}).get("image") if isinstance(body.get("result"), dict) else None
            encoded = encoded or body.get("image")
            if not isinstance(encoded, str) or not encoded:
                raise ImageProviderError(
                    "The image service answered without an image (the prompt may have been refused)."
                )
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ImageProviderError("The image service sent a damaged image.") from exc
        mime = sniff_mime(data)
        if mime is None:
            raise ImageProviderError("The image service sent something that isn't an image.")
        return GeneratedImage(data=data, mime=mime)


def build_image_provider(settings: Settings) -> tuple[ImageProvider | None, str]:
    """-> (provider, "") or (None, what is missing -- user-facing, no secrets)."""
    choice = (settings.image_provider or "").strip().lower()
    if not choice:
        return None, (
            "Image generation isn't set up. To turn it on, set GARVIS_IMAGE_PROVIDER=cloudflare, "
            "GARVIS_CLOUDFLARE_ACCOUNT_ID and GARVIS_CLOUDFLARE_API_TOKEN in Garvis's .env and "
            "restart Garvis."
        )
    if choice != "cloudflare":
        return None, f"Unknown GARVIS_IMAGE_PROVIDER {choice!r} (supported: cloudflare)."
    if not settings.cloudflare_account_id or not settings.cloudflare_api_token:
        return None, (
            "GARVIS_IMAGE_PROVIDER=cloudflare needs GARVIS_CLOUDFLARE_ACCOUNT_ID and "
            "GARVIS_CLOUDFLARE_API_TOKEN in Garvis's .env."
        )
    return CloudflareImageProvider(
        settings.cloudflare_account_id,
        settings.cloudflare_api_token,
        model=settings.image_model,
        steps=settings.image_steps,
        timeout=settings.image_timeout_seconds,
    ), ""
