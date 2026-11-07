"""Image jobs: queued, generated one at a time, stored on the PC.

Files go to data/images/<id>.<ext> with a small index.json. A job is
"completed" only when the provider returned a real image and it was
written to disk; a failure keeps the provider's (secret-free) reason.
Jobs that were running when Garvis closed are marked failed on restart,
never retried silently. A client `request_id` makes submission idempotent.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger, log_extra
from app.images.providers import ImageProvider, ImageProviderError

logger = get_logger("images.service")

QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED = "queued", "running", "completed", "failed", "cancelled"
ACTIVE = frozenset({QUEUED, RUNNING})
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
MAX_QUEUED = 5
KEEP = 200


class ImageError(Exception):
    """A job can't be accepted or changed; the message is user-facing."""


@dataclass
class ImageJob:
    id: str
    prompt: str
    status: str = QUEUED
    origin: str = "mobile"
    request_id: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str = ""
    file: str = ""
    mime: str = ""
    size: int = 0

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("request_id")
        data.pop("file")
        data["has_image"] = self.status == COMPLETED and bool(self.file)
        return data


class ImageService:
    def __init__(self, directory: Path, provider: ImageProvider | None, unavailable_reason: str = "") -> None:
        self._dir = Path(directory)
        self._provider = provider
        self.unavailable_reason = "" if provider is not None else (unavailable_reason or "Image generation is off.")
        self._jobs: dict[str, ImageJob] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[ImageJob], None]] = []
        self.version = 0
        self._load()

    @property
    def available(self) -> bool:
        return self._provider is not None

    @property
    def provider_name(self) -> str:
        return getattr(self._provider, "name", "") if self._provider is not None else ""

    def add_listener(self, listener: Callable[[ImageJob], None]) -> None:
        self._listeners.append(listener)

    # ---------------------------------------------------------------- reads
    def list(self, limit: int = 100) -> list[ImageJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def get(self, job_id: str) -> ImageJob | None:
        return self._jobs.get(job_id)

    def file_path(self, job_id: str) -> Path | None:
        job = self._jobs.get(job_id)
        if job is None or job.status != COMPLETED or not job.file:
            return None
        path = self._dir / job.file
        return path if path.is_file() else None

    # --------------------------------------------------------------- writes
    def submit(self, prompt: str, *, origin: str = "mobile", request_id: str | None = None) -> ImageJob:
        if request_id:
            if not _REQUEST_ID.match(request_id):
                raise ImageError("Invalid request id.")
            existing = next((j for j in self._jobs.values() if j.request_id == request_id), None)
            if existing is not None:
                return existing
        if self._provider is None:
            raise ImageError(self.unavailable_reason)
        prompt = " ".join(str(prompt or "").split())
        if not prompt:
            raise ImageError("Describe the image you want.")
        if len(prompt) > self._provider.max_prompt_chars:
            raise ImageError(f"The description is too long (limit {self._provider.max_prompt_chars} characters).")
        if sum(1 for j in self._jobs.values() if j.status in ACTIVE) >= MAX_QUEUED:
            raise ImageError(f"{MAX_QUEUED} images are already being made. Wait for one to finish.")
        job = ImageJob(id=secrets.token_urlsafe(9), prompt=prompt, origin=origin[:20], request_id=request_id or "")
        self._jobs[job.id] = job
        self._changed(job)
        self._running[job.id] = asyncio.get_running_loop().create_task(self._run(job))
        log_extra(logger, 20, "images.submitted", job=job.id, origin=job.origin)
        return job

    def cancel(self, job_id: str) -> ImageJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise ImageError("Unknown image.")
        if job.status not in ACTIVE:
            raise ImageError("That image has already finished.")
        task = self._running.get(job_id)
        if task is not None:
            task.cancel()  # only an HTTP request is in flight; nothing on the PC is half-done
        self._finish(job, CANCELLED, error="Cancelled.")
        return job

    def delete(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            raise ImageError("Unknown image.")
        if job.status in ACTIVE:
            raise ImageError("Cancel it first.")
        if job.file:
            try:
                (self._dir / job.file).unlink(missing_ok=True)
            except OSError:
                log_extra(logger, 30, "images.delete_failed", job=job.id)
        del self._jobs[job.id]
        self._changed(job)

    async def wait(self, job_id: str, timeout: float) -> ImageJob:
        task = self._running.get(job_id)
        if task is not None:
            await asyncio.wait({task}, timeout=timeout)
        return self._jobs[job_id]

    # --------------------------------------------------------------- worker
    async def _run(self, job: ImageJob) -> None:
        try:
            async with self._lock:  # one generation at a time
                if job.status != QUEUED:
                    return
                job.status = RUNNING
                self._changed(job)
                result = await self._provider.generate(job.prompt)
                extension = _EXTENSIONS.get(result.mime, "img")
                self._dir.mkdir(parents=True, exist_ok=True)
                name = f"{job.id}.{extension}"
                await asyncio.to_thread((self._dir / name).write_bytes, result.data)
                if job.status != RUNNING:  # cancelled while writing
                    (self._dir / name).unlink(missing_ok=True)
                    return
                job.file, job.mime, job.size = name, result.mime, len(result.data)
                self._finish(job, COMPLETED)
        except asyncio.CancelledError:
            if job.status in ACTIVE:
                self._finish(job, CANCELLED, error="Cancelled.")
            raise
        except ImageProviderError as exc:
            self._finish(job, FAILED, error=str(exc))
        except OSError:
            logger.exception("images.write_failed")
            self._finish(job, FAILED, error="The image couldn't be saved on the PC.")
        except Exception:  # noqa: BLE001 -- never leave a job "running" forever
            logger.exception("images.crashed")
            self._finish(job, FAILED, error="Something went wrong while making the image.")
        finally:
            self._running.pop(job.id, None)

    # -------------------------------------------------------------- helpers
    def _finish(self, job: ImageJob, status: str, *, error: str = "") -> None:
        if job.status not in ACTIVE:
            return
        job.status, job.finished_at, job.error = status, time.time(), error[:500]
        log_extra(logger, 20, "images.finished", job=job.id, status=status)
        self._changed(job)

    def _changed(self, job: ImageJob) -> None:
        self.version += 1
        self._trim()
        self._save()
        for listener in list(self._listeners):
            try:
                listener(job)
            except Exception:  # noqa: BLE001
                logger.exception("images.listener_failed")

    def _trim(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.status not in ACTIVE), key=lambda j: j.created_at)
        for job in finished[: max(0, len(finished) - KEEP)]:
            if job.file:
                try:
                    (self._dir / job.file).unlink(missing_ok=True)
                except OSError:
                    pass
            del self._jobs[job.id]

    def _save(self) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=".index-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {"version": self.version, "images": [asdict(j) for j in self._jobs.values()]},
                    handle,
                    ensure_ascii=False,
                )
            os.replace(tmp, self._dir / "index.json")
        except OSError:
            log_extra(logger, 40, "images.index_save_failed")

    def _load(self) -> None:
        try:
            data = json.loads((self._dir / "index.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except OSError, ValueError:
            log_extra(logger, 30, "images.index_unreadable")
            return
        fields = set(ImageJob.__dataclass_fields__)
        for record in data.get("images", []) if isinstance(data, dict) else []:
            try:
                job = ImageJob(**{k: v for k, v in record.items() if k in fields})
            except TypeError:
                continue
            if job.status in ACTIVE:
                job.status, job.finished_at = FAILED, job.finished_at or time.time()
                job.error = "Garvis was closed before this image was finished. It was not retried."
            self._jobs[job.id] = job
        self.version = int(data.get("version", 0)) if isinstance(data, dict) else 0
