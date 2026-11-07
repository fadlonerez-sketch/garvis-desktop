"""Offline OCR of screenshots, so a text-only model can read what is on the
screen (dialogs, error messages, canvases, remote-desktop windows -- things
UI Automation can't see).

Uses the Tesseract executable, called directly (no Python binding, no new
dependency). It is found via GARVIS_OCR_TESSERACT_PATH, then PATH and the
standard installer locations.
Nothing is uploaded anywhere: recognition runs on this PC.

Word boxes come back in screenshot pixels and are shifted by the capture's
offset, so every line has real screen coordinates to click on.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from app.core.exceptions import DesktopError

CREATE_NO_WINDOW = 0x08000000
_MIN_LINE_CONFIDENCE = 35.0


class OcrLine(BaseModel):
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)


class OcrResult(BaseModel):
    lines: list[OcrLine]
    engine: str
    elapsed_ms: int
    cached: bool = False
    """True when the image was identical to the previous one: the earlier
    result was reused instead of running OCR again."""

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class OcrEngine(ABC):
    @abstractmethod
    async def recognize(self, image_path: Path, offset_x: int = 0, offset_y: int = 0) -> OcrResult:
        """Text lines of a PNG, with boxes shifted by the offset. Raises DesktopError."""


def candidate_tesseract_paths(configured: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if configured:
        configured_path = Path(os.path.expandvars(os.path.expanduser(configured)))
        candidates.append(configured_path / "tesseract.exe" if configured_path.is_dir() else configured_path)
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(Path(on_path))
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates.append(Path(base) / "Tesseract-OCR" / "tesseract.exe")
            candidates.append(Path(base) / "Programs" / "Tesseract-OCR" / "tesseract.exe")
    return candidates


def parse_tsv(tsv: str, offset_x: int = 0, offset_y: int = 0) -> list[OcrLine]:
    """Group Tesseract's word rows (level 5) into lines (block/par/line)."""
    groups: dict[tuple[str, str, str, str], list[tuple[str, float, int, int, int, int]]] = {}
    for row in tsv.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        word = "\t".join(parts[11:]).strip()
        if not word:
            continue
        try:
            conf = float(parts[10])
            left, top, width, height = (int(v) for v in parts[6:10])
        except ValueError:
            continue
        groups.setdefault((parts[1], parts[2], parts[3], parts[4]), []).append((word, conf, left, top, width, height))
    lines: list[OcrLine] = []
    for words in groups.values():
        confidence = sum(w[1] for w in words) / len(words)
        if confidence < _MIN_LINE_CONFIDENCE:
            continue
        left = min(w[2] for w in words)
        top = min(w[3] for w in words)
        right = max(w[2] + w[4] for w in words)
        bottom = max(w[3] + w[5] for w in words)
        lines.append(
            OcrLine(
                text=" ".join(w[0] for w in words),
                confidence=round(confidence, 1),
                left=left + offset_x,
                top=top + offset_y,
                width=right - left,
                height=bottom - top,
            )
        )
    lines.sort(key=lambda line: (line.top // 8, line.left))
    return lines


class TesseractOcr(OcrEngine):
    def __init__(
        self,
        configured_path: str | None = None,
        languages: str = "eng+heb",
        model: str = "fast",
        timeout_seconds: float = 45.0,
    ) -> None:
        self._configured_path = configured_path
        self._languages = languages
        self._model = model
        self._timeout = timeout_seconds
        self._resolved: tuple[Path, Path, str] | None = None
        self._last: tuple[str, OcrResult] | None = None

    def _resolve(self) -> tuple[Path, Path, str]:
        if self._resolved is not None:
            return self._resolved
        exe = next((p for p in candidate_tesseract_paths(self._configured_path) if p.is_file()), None)
        if exe is None:
            raise DesktopError(
                "No OCR engine found. Install Tesseract or set GARVIS_OCR_TESSERACT_PATH "
                "to tesseract.exe. inspect_ui still works without it."
            )
        tessdata = exe.parent / "tessdata" / self._model
        if not tessdata.is_dir():
            tessdata = exe.parent / "tessdata"
        available = {p.stem for p in tessdata.glob("*.traineddata")}
        wanted = [lang for lang in self._languages.split("+") if lang]
        languages = [lang for lang in wanted if lang in available] or sorted(available)[:2]
        if not languages:
            raise DesktopError(f"Tesseract at {exe} has no language data in {tessdata}.")
        self._resolved = (exe, tessdata, "+".join(languages))
        return self._resolved

    def _run(self, image_path: Path, offset_x: int, offset_y: int) -> OcrResult:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest() + f"@{offset_x},{offset_y}"
        if self._last is not None and self._last[0] == digest:
            return self._last[1].model_copy(update={"cached": True, "elapsed_ms": 0})
        exe, tessdata, languages = self._resolve()
        command = [
            str(exe),
            str(image_path),
            "stdout",
            "-l",
            languages,
            "--psm",
            "3",
            "--tessdata-dir",
            str(tessdata),
            "-c",
            "tessedit_create_tsv=1",
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=self._timeout,
                cwd=str(exe.parent),
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise DesktopError(f"OCR took longer than {self._timeout:.0f} s; try a smaller window or region.") from exc
        except OSError as exc:
            raise DesktopError(f"Tesseract could not be started: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
            raise DesktopError(
                f"OCR failed (exit {completed.returncode})" + (f": {detail[-1][:200]}" if detail else "")
            )
        tsv = completed.stdout.decode("utf-8", "replace")
        result = OcrResult(
            lines=parse_tsv(tsv, offset_x, offset_y),
            engine=f"tesseract ({languages})",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        self._last = (digest, result)
        return result

    async def recognize(self, image_path: Path, offset_x: int = 0, offset_y: int = 0) -> OcrResult:
        return await asyncio.to_thread(self._run, image_path, offset_x, offset_y)
