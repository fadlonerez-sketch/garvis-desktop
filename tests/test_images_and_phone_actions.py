"""Image jobs (with a fake provider and the Cloudflare adapter on a mocked
transport -- no network) and the phone-action broker/tool."""
from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from app.core.config import Settings
from app.images.providers import (
    CloudflareImageProvider,
    GeneratedImage,
    ImageProvider,
    ImageProviderError,
    build_image_provider,
)
from app.images.service import CANCELLED, COMPLETED, FAILED, ImageError, ImageService
from app.images.tools import GenerateImageTool
from app.mobile.phone_actions import PhoneActionBroker, PhoneActionError, PhoneActionTool, validate_params
from tests.mobile_fakes import wait_until

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class FakeProvider(ImageProvider):
    name = "fake"

    def __init__(self, data: bytes = JPEG, error: str | None = None, gate: asyncio.Event | None = None) -> None:
        self.data, self.error, self.gate = data, error, gate
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> GeneratedImage:
        self.prompts.append(prompt)
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise ImageProviderError(self.error)
        return GeneratedImage(self.data, "image/jpeg")


# --------------------------------------------------------------- providers
def test_no_provider_unless_configured():
    provider, reason = build_image_provider(Settings(_env_file=None, image_provider=""))
    assert provider is None and "GARVIS_IMAGE_PROVIDER" in reason
    provider, reason = build_image_provider(Settings(_env_file=None, image_provider="cloudflare"))
    assert provider is None and "ACCOUNT_ID" in reason
    provider, reason = build_image_provider(Settings(_env_file=None, image_provider="openai",
                                                     cloudflare_account_id="a", cloudflare_api_token="t"))
    assert provider is None and "Unknown" in reason
    provider, _ = build_image_provider(Settings(_env_file=None, image_provider="cloudflare",
                                                cloudflare_account_id="acct", cloudflare_api_token="secret"))
    assert isinstance(provider, CloudflareImageProvider)


def _cloudflare(handler) -> CloudflareImageProvider:
    return CloudflareImageProvider("acct", "secret-token", model="@cf/black-forest-labs/flux-1-schnell",
                                   transport=httpx.MockTransport(handler))


async def test_cloudflare_request_and_json_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"success": True, "result": {"image": base64.b64encode(JPEG).decode()}})

    image = await _cloudflare(handler).generate("an orange sphere")
    assert image.data == JPEG and image.mime == "image/jpeg"
    assert seen["url"].endswith("/accounts/acct/ai/run/@cf/black-forest-labs/flux-1-schnell")
    assert seen["auth"] == "Bearer secret-token"
    assert seen["body"] == {"prompt": "an orange sphere", "steps": 4}


async def test_cloudflare_raw_image_response():
    image = await _cloudflare(lambda r: httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
                              ).generate("x")
    assert image.mime == "image/png"


@pytest.mark.parametrize("response,fragment", [
    (httpx.Response(401, json={"errors": []}), "token"),
    (httpx.Response(429, json={}), "limit"),
    (httpx.Response(400, json={"errors": [{"message": "NSFW"}]}), "NSFW"),
    (httpx.Response(200, json={"result": {}}), "without an image"),
    (httpx.Response(200, json={"result": {"image": base64.b64encode(b"not an image").decode()}}), "isn't an image"),
])
async def test_cloudflare_failures_are_reported_without_secrets(response, fragment):
    with pytest.raises(ImageProviderError) as info:
        await _cloudflare(lambda r: response).generate("x")
    assert fragment in str(info.value) and "secret-token" not in str(info.value)


# ------------------------------------------------------------------ service
async def test_unavailable_service_refuses_and_says_why(tmp_path):
    service = ImageService(tmp_path, None, "Image generation isn't set up.")
    assert not service.available
    with pytest.raises(ImageError, match="isn't set up"):
        service.submit("a cat")
    result = await GenerateImageTool(service).execute(GenerateImageTool.input_model(prompt="a cat"))
    assert not result.success and "isn't set up" in result.error


async def test_job_completes_with_a_real_file_and_dedupes(tmp_path):
    provider = FakeProvider()
    service = ImageService(tmp_path, provider)
    changes = []
    service.add_listener(lambda job: changes.append(job.status))
    job = service.submit("  futuristic   city ", request_id="img-request-1")
    assert service.submit("futuristic city", request_id="img-request-1").id == job.id
    await service.wait(job.id, 5)
    assert job.status == COMPLETED and job.prompt == "futuristic city"
    assert service.file_path(job.id).read_bytes() == JPEG
    assert changes[-1] == COMPLETED
    assert provider.prompts == ["futuristic city"]
    reloaded = ImageService(tmp_path, provider)
    assert reloaded.get(job.id).status == COMPLETED and reloaded.file_path(job.id) is not None


async def test_failure_and_cancel(tmp_path):
    failing = ImageService(tmp_path / "a", FakeProvider(error="The image service refused the request."))
    job = failing.submit("x")
    await failing.wait(job.id, 5)
    assert job.status == FAILED and "refused" in job.error and failing.file_path(job.id) is None

    gate = asyncio.Event()
    slow = ImageService(tmp_path / "b", FakeProvider(gate=gate))
    job = slow.submit("y")
    await wait_until(lambda: job.status == "running")
    slow.cancel(job.id)
    assert job.status == CANCELLED
    gate.set()
    await asyncio.sleep(0.05)
    assert job.status == CANCELLED and slow.file_path(job.id) is None
    with pytest.raises(ImageError):
        slow.cancel(job.id)


async def test_interrupted_jobs_are_failed_on_restart(tmp_path):
    gate = asyncio.Event()
    service = ImageService(tmp_path, FakeProvider(gate=gate))
    job = service.submit("z")
    await wait_until(lambda: job.status == "running")
    reloaded = ImageService(tmp_path, FakeProvider())
    assert reloaded.get(job.id).status == FAILED and "not retried" in reloaded.get(job.id).error
    gate.set()


async def test_generate_image_tool_returns_the_id(tmp_path):
    service = ImageService(tmp_path, FakeProvider())
    result = await GenerateImageTool(service).execute(GenerateImageTool.input_model(prompt="orange glowing sphere"))
    assert result.success and service.get(result.data["image_id"]).status == COMPLETED


# ------------------------------------------------------------ phone actions
def test_validation_rules():
    assert validate_params("open_app", {"app": "Spotify", "url": None}) == {"app": "Spotify"}
    with pytest.raises(PhoneActionError):
        validate_params("open_app", {})
    with pytest.raises(PhoneActionError):
        validate_params("open_url", {"url": "javascript:alert(1)"})
    with pytest.raises(PhoneActionError):
        validate_params("dial", {"number": "call my boss"})
    with pytest.raises(PhoneActionError):
        validate_params("open_settings", {"page": "developer_options"})
    with pytest.raises(PhoneActionError):
        validate_params("format_phone", {})


async def test_no_phone_connected_fails_fast():
    with pytest.raises(PhoneActionError, match="isn't connected"):
        await PhoneActionBroker().request("open_app", {"app": "Maps"}, timeout=1)


async def test_capability_off_and_paused():
    broker = PhoneActionBroker()
    broker.report("dev1", ["apps"], paused=False, accessibility=False)
    with pytest.raises(PhoneActionError, match="'communication' capability is switched off"):
        await broker.request("dial", {"number": "+972501234567"}, timeout=1)
    broker.report("dev1", ["apps"], paused=True, accessibility=False)
    with pytest.raises(PhoneActionError, match="stopped"):
        await broker.request("open_app", {"app": "Maps"}, timeout=1)


async def test_round_trip_with_confirmation_flag():
    broker = PhoneActionBroker()
    broker.report("dev1", ["communication"], paused=False, accessibility=False)
    request = asyncio.ensure_future(broker.request("dial", {"number": "+972501234567"}, timeout=5))
    await wait_until(lambda: broker.pending_for("dev1"))
    (pending,) = broker.pending_for("dev1")
    assert pending["confirm"] is True and pending["params"] == {"number": "+972501234567"}
    assert not broker.complete("other-device", pending["id"], "done")  # only the target phone may answer
    assert broker.complete("dev1", pending["id"], "declined", "User said no")
    assert (await request) == {"status": "declined", "message": "User said no", "data": {}}
    assert broker.pending_for("dev1") == []


async def test_timeout_and_pause_resolve_pending_requests():
    broker = PhoneActionBroker()
    broker.report("dev1", ["apps"], paused=False, accessibility=False)
    assert (await broker.request("open_app", {"app": "Maps"}, timeout=0.05))["status"] == "failed"
    request = asyncio.ensure_future(broker.request("open_app", {"app": "Maps"}, timeout=5))
    await wait_until(lambda: broker.pending_for("dev1"))
    broker.report("dev1", ["apps"], paused=True, accessibility=False)
    assert (await request)["status"] == "declined"


async def test_phone_action_tool_reports_what_the_phone_did():
    broker = PhoneActionBroker()
    broker.report("dev1", ["apps", "screen_reading"], paused=False, accessibility=True)
    tool = PhoneActionTool(broker, timeout=5)

    async def phone():
        await wait_until(lambda: broker.pending_for("dev1"))
        (pending,) = broker.pending_for("dev1")
        broker.complete("dev1", pending["id"], "done", "Opened Spotify")

    asyncio.ensure_future(phone())
    result = await tool.execute(tool.validate_arguments({"action": "open_app", "app": "Spotify"}))
    assert result.success and result.data["message"] == "Opened Spotify"

    async def failing_phone():
        await wait_until(lambda: broker.pending_for("dev1"))
        (pending,) = broker.pending_for("dev1")
        broker.complete("dev1", pending["id"], "unsupported", "No app called Zork")

    asyncio.ensure_future(failing_phone())
    result = await tool.execute(tool.validate_arguments({"action": "open_app", "app": "Zork"}))
    assert not result.success and "can't do that" in result.error and "Zork" in result.error
