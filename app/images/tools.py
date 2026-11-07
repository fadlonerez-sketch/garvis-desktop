"""The assistant's image tool: "generate an image of ..." from chat or voice."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.images.service import COMPLETED, ImageError, ImageService
from app.tools.base import RiskLevel, Tool, ToolResult


class GenerateImageInput(BaseModel):
    prompt: str = Field(description="A detailed visual description of the image to create, in English.")


class GenerateImageTool(Tool):
    name = "generate_image"
    description = (
        "Create an image from a description with the configured image service (not the LLM). The image appears "
        "in the Garvis phone app's gallery and conversation. To put it in the phone's photo gallery as well, "
        "follow up with phone_action action='save_image' and the returned image_id. Fails with the reason when "
        "no image service is configured -- report that, never pretend an image exists."
    )
    risk_level = RiskLevel.LOW
    input_model = GenerateImageInput

    def __init__(self, images: ImageService, timeout: float = 150.0) -> None:
        self._images = images
        self._timeout = timeout

    async def execute(self, args: GenerateImageInput) -> ToolResult:
        try:
            job = self._images.submit(args.prompt, origin="assistant")
        except ImageError as exc:
            return ToolResult.fail(str(exc))
        job = await self._images.wait(job.id, self._timeout)
        if job.status == COMPLETED:
            return ToolResult.ok(
                {
                    "image_id": job.id,
                    "status": job.status,
                    "mime": job.mime,
                    "note": "Shown in the phone app's Images gallery.",
                }
            )
        if job.status in ("queued", "running"):
            return ToolResult.ok(
                {
                    "image_id": job.id,
                    "status": job.status,
                    "note": "Still being made; it will appear in the phone app's gallery when done.",
                }
            )
        return ToolResult.fail(job.error or f"The image was not made ({job.status}).")
