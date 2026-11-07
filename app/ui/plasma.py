"""Renders the approved Garvis plasma orb (plasma_orb.frag) into QImages.

The orb is one fragment shader. It runs in an offscreen OpenGL context
(QOffscreenSurface + a framebuffer object, from PySide6's own QtOpenGL -- no
new dependency) and comes back as a premultiplied QImage, which ordinary
QPainter widgets paint: the dashboard core, the translucent floating orb.

Quality adapts to the display size without changing the design:

- full detail at 220 px and up, compact detail (fewer layers, no shards or
  sparks) below -- the approved size ladder;
- a frame is rendered at most MAX_RENDER_PX square and scaled up beyond that;
- on a software rasterizer (no GPU driver, e.g. a remote session) frames are
  rendered at SOFTWARE_MAX_PX with compact detail -- the approved CPU fallback --
  and paced at SOFTWARE_FPS;
- with no OpenGL at all (Qt's "offscreen" platform in tests) `renderer()` is
  None and callers draw `still_frame()`, a pre-rendered frame of the orb.

One renderer is shared by every orb in the process; it lives on the GUI thread.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QGuiApplication, QImage, QOffscreenSurface, QOpenGLContext, QSurfaceFormat, QVector2D

from app.core.logging import get_logger, log_extra

logger = get_logger("ui.plasma")

SHADER_PATH = Path(__file__).with_name("plasma_orb.frag")
STILL_PATH = Path(__file__).with_name("orb_still.png")
"""Idle frame of the orb, 512 px (scripts/render_orb_still.py writes it from the shader)."""

FULL_DETAIL_MIN_PX = 220
MAX_RENDER_PX = 900
SOFTWARE_MAX_PX = 160
SOFTWARE_FPS = 10

UNIFORMS = ("u_time", "u_drift", "u_swirl", "u_glow", "u_heat", "u_plasma", "u_swell", "u_alert", "u_standby",
            "u_detail")

_GL_TRIANGLES = 0x0004
_GL_FLOAT = 0x1406
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_RENDERER = 0x1F01
_SOFTWARE_NAMES = ("llvmpipe", "softpipe", "software", "gdi generic", "swiftshader", "basic render")


def render_size(physical_px: float, software: bool) -> int:
    """The square frame size actually rendered for an orb `physical_px` device pixels across."""
    limit = SOFTWARE_MAX_PX if software else MAX_RENDER_PX
    return max(16, min(int(round(physical_px)), limit))


def detail_for(physical_px: float, software: bool, compact: bool = False) -> float:
    return 0.0 if compact or software or physical_px < FULL_DETAIL_MIN_PX else 1.0


class PlasmaRenderer:
    def __init__(self, shader_path: Path = SHADER_PATH) -> None:
        # Imported here: QtOpenGL loads the GL DLLs, which a process without an orb never needs.
        from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram

        self._context = QOpenGLContext()
        self._context.setFormat(QSurfaceFormat.defaultFormat())
        if not self._context.create():
            raise RuntimeError("no OpenGL context")
        self._surface = QOffscreenSurface()
        self._surface.setFormat(self._context.format())
        self._surface.create()
        if not self._context.makeCurrent(self._surface):
            raise RuntimeError("cannot make the OpenGL context current")
        self._gl = self._context.functions()
        header = "#version 100\nprecision highp float;\n" if self._context.isOpenGLES() else "#version 120\n"
        program = QOpenGLShaderProgram()
        vertex = header + "attribute vec2 a_pos;\nvoid main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n"
        fragment = header + shader_path.read_text(encoding="utf-8")
        if not (program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex)
                and program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment)):
            raise RuntimeError("shader compile failed: " + program.log())
        program.bindAttributeLocation("a_pos", 0)
        if not program.link():
            raise RuntimeError("shader link failed: " + program.log())
        self._program = program
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)  # one triangle covering the frame
        self._vbo.create()
        self._vbo.bind()
        data = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype=np.float32).tobytes()
        self._vbo.allocate(data, len(data))
        self._vbo.release()
        self._fbos: dict[int, object] = {}
        self.name = str(self._gl.glGetString(_GL_RENDERER) or "")
        self.software = any(word in self.name.lower() for word in _SOFTWARE_NAMES)
        self._context.doneCurrent()

    def render(self, size: int, uniforms: dict[str, float]) -> QImage:
        """One size x size frame, premultiplied, top row first."""
        from PySide6.QtOpenGL import QOpenGLFramebufferObject

        self._context.makeCurrent(self._surface)
        try:
            fbo = self._fbos.get(size)
            if fbo is None:
                if len(self._fbos) >= 4:  # a resized window must not pile up framebuffers
                    self._fbos.pop(next(iter(self._fbos)))
                fbo = self._fbos[size] = QOpenGLFramebufferObject(QSize(size, size))
            fbo.bind()
            gl = self._gl
            gl.glViewport(0, 0, size, size)
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClear(_GL_COLOR_BUFFER_BIT)
            self._program.bind()
            self._program.setUniformValue("u_res", QVector2D(float(size), float(size)))
            for name in UNIFORMS:
                self._program.setUniformValue1f(name, float(uniforms.get(name, 0.0)))
            self._vbo.bind()
            self._program.enableAttributeArray(0)
            self._program.setAttributeBuffer(0, _GL_FLOAT, 0, 2, 0)
            gl.glDrawArrays(_GL_TRIANGLES, 0, 3)
            self._vbo.release()
            self._program.release()
            image = fbo.toImage()
            fbo.release()
        finally:
            self._context.doneCurrent()
        return image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


_renderer: PlasmaRenderer | None = None
_tried = False


def renderer() -> PlasmaRenderer | None:
    """The shared renderer, created on first use; None where OpenGL isn't available."""
    global _renderer, _tried
    if not _tried:
        _tried = True
        if QGuiApplication.platformName() in ("offscreen", "minimal"):
            return None
        try:
            _renderer = PlasmaRenderer()
            log_extra(logger, logging.INFO, "plasma.renderer", renderer=_renderer.name, software=_renderer.software)
        except Exception:  # any GL trouble: the orb falls back to its still frame, Garvis keeps working
            logger.warning("plasma.renderer_unavailable", exc_info=True)
            _renderer = None
    return _renderer


@lru_cache(maxsize=1)
def still_frame() -> QImage:
    """The pre-rendered orb (idle), for when there's no OpenGL. Empty if the file is missing."""
    image = QImage(str(STILL_PATH))
    return image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied) if not image.isNull() else image
