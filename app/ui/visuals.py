"""The Garvis orb: the approved blue plasma sphere (2026-10-31), shared by the
dashboard's centre and the floating orb.

How it looks is entirely app/ui/plasma_orb.frag (rendered by app/ui/plasma.py);
a companion mobile client (not part of this repository) runs a byte-identical
copy of that shader. Its appearance is an approved design -- don't change it
without the owner's approval. The logo (title bar, window and tray icons) is NOT this
orb: it is the original amber orb, frozen in app/ui/logo.py.

This module owns the orb's *behaviour*: `Dynamics` smooths every assistant
state into the shader's inputs. States only change targets, which are
approached smoothly, never jumped to: idle breathes quietly; listening
brightens and gathers; thinking stirs the plasma faster and lets the core
pulse; speaking follows the real playback loudness (AgentBridge.voice_level,
sampled only while speaking); leaving speech -- an interruption included --
briefly settles the glow before the next state takes over; standby dims and
slows; an error shifts toward red.

Reduced motion (Windows' "Show animations" off): the plasma holds still and
nothing breathes; brightness still follows the state and the voice level.

CPU: the timer runs only while the widget is visible and its window isn't
minimized -- 30 fps when active, 15 fps idle, 10 fps in standby (and at most
plasma.SOFTWARE_FPS on a software rasterizer), and not at all when idle with
reduced motion. A frame is one GPU pass plus a read-back: about 3 ms at
560 px on the integrated GPU it was measured on.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from PySide6.QtCore import QElapsedTimer, QRectF, Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.ui import plasma
from app.ui.state import Mode
from app.ui.theme import Accent

ACTIVE_INTERVAL_MS = 33
IDLE_INTERVAL_MS = 66
STANDBY_INTERVAL_MS = 100

_BREATH_PERIOD = 5.5
_PULSE_PERIOD = 1.8

# Per-state targets, each 0..1: halo glow, heart heat, plasma flow speed and
# opacity, breathing depth, thinking stir, alert (red shift).
_TARGETS = {
    Mode.IDLE: dict(glow=0.55, heat=0.35, flow=0.12, plasma=0.45, breathe=1.0, stir=0.0, alert=0.0),
    Mode.LISTENING: dict(glow=0.78, heat=0.55, flow=0.20, plasma=0.60, breathe=0.4, stir=0.0, alert=0.0),
    Mode.THINKING: dict(glow=0.66, heat=0.50, flow=0.45, plasma=0.72, breathe=0.0, stir=1.0, alert=0.0),
    Mode.SPEAKING: dict(glow=0.62, heat=0.45, flow=0.26, plasma=0.58, breathe=0.0, stir=0.0, alert=0.0),
    Mode.ERROR: dict(glow=0.45, heat=0.28, flow=0.08, plasma=0.40, breathe=0.6, stir=0.0, alert=1.0),
}
_STANDBY = dict(glow=0.30, heat=0.20, flow=0.05, plasma=0.30, breathe=0.6, stir=0.0, alert=0.0)


def _approach(current: float, target: float, rate: float, dt: float) -> float:
    return target + (current - target) * math.exp(-rate * dt)


def system_prefers_reduced_motion() -> bool:
    """Windows' "Show animations in Windows" setting is off."""
    try:
        import ctypes

        value = ctypes.c_int(1)
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(value), 0):  # SPI_GETCLIENTAREAANIMATION
            return not value.value
    except AttributeError, OSError:
        pass
    return False


class Dynamics:
    """Smoothed animation state, advanced once per frame."""

    def __init__(self, reduced_motion: bool = False) -> None:
        self.reduced_motion = reduced_motion
        self.time = 0.0
        self.drift = 0.0
        """How far the plasma has flowed (advances with the `flow` value)."""
        self.swirl = 0.0
        """Plasma rotation in radians (advances while thinking)."""
        self.values = dict(_TARGETS[Mode.IDLE])
        self.values["standby"] = 0.0
        self.level = 0.0
        self.glow_level = 0.0
        self.peak = 0.0
        self.flash = 0.0
        self.hush = 0.0
        """Settling after speech stops: peaks after ~0.15 s, gone in ~1 s."""
        self._hush_goal = 0.0
        self._last_mode = Mode.IDLE

    def step(self, dt: float, mode: Mode, raw_level: float = 0.0, standby: bool = False) -> None:
        dt = max(0.0, min(dt, 0.25))  # a stalled frame must not teleport the animation
        self.time += dt
        resting = standby and mode == Mode.IDLE
        for key, target in (_STANDBY if resting else _TARGETS[mode]).items():
            self.values[key] = _approach(self.values[key], target, 3.5, dt)
        self.values["standby"] = _approach(self.values["standby"], 1.0 if resting else 0.0, 2.5, dt)
        if self._last_mode == Mode.SPEAKING and mode != Mode.SPEAKING:
            self._hush_goal = 1.0  # the voice stopped -- finished or interrupted
        self._last_mode = mode
        self.hush = _approach(self.hush, self._hush_goal, 9.0, dt)
        self._hush_goal = _approach(self._hush_goal, 0.0, 3.0, dt)
        if not self.reduced_motion:
            self.drift += self.values["flow"] * dt
            self.swirl += self.values["stir"] * 0.5 * dt
        if mode == Mode.SPEAKING and raw_level > 0:
            self.peak = max(raw_level, self.peak * math.exp(-0.35 * dt))
            target_level = min(1.0, raw_level / max(self.peak, 700.0))
        else:
            self.peak *= math.exp(-0.35 * dt)
            target_level = 0.0
        rate = 22.0 if target_level > self.level else 7.0  # quick attack, soft release
        self.level = _approach(self.level, target_level, rate, dt)
        # What the orb shows: follows syllables, but never flickers.
        self.glow_level = _approach(self.glow_level, self.level, 8.0, dt)
        self.flash = _approach(self.flash, 0.0, 3.2, dt)

    def appearance(self) -> dict[str, float]:
        """This frame's drawing intensities, derived from the smoothed values."""
        v = self.values
        still = self.reduced_motion
        breath = 0.0 if still else math.sin(2 * math.pi * self.time / _BREATH_PERIOD)
        pulse = 0.0 if still else math.sin(2 * math.pi * self.time / _PULSE_PERIOD)
        level, hush = self.glow_level, self.hush
        return dict(
            glow=max(
                0.0, min(1.0, v["glow"] + 0.07 * v["breathe"] * breath + 0.38 * level + 0.45 * self.flash - 0.28 * hush)
            ),
            heat=max(
                0.0, min(1.0, v["heat"] + 0.10 * v["stir"] * pulse + 0.38 * level + 0.35 * self.flash - 0.22 * hush)
            ),
            plasma=max(0.0, min(1.0, v["plasma"] + 0.22 * level - 0.15 * hush)),
            swell=1.0
            + (0.0 if still else 0.012 * v["breathe"] * breath + 0.035 * level)
            + 0.03 * self.flash
            - 0.02 * hush,
            alert=v["alert"],
            standby=v["standby"],
        )

    @property
    def settled(self) -> bool:
        return (
            self.level < 0.01
            and self.glow_level < 0.01
            and self.flash < 0.01
            and self.hush < 0.01
            and self._hush_goal < 0.01
        )


def shader_uniforms(dyn: Dynamics, detail: float = 1.0) -> dict[str, float]:
    """This frame's inputs for plasma_orb.frag."""
    look = dyn.appearance()
    return dict(
        u_time=0.0 if dyn.reduced_motion else dyn.time,  # reduced motion: the sparks hold still too
        u_drift=dyn.drift,
        u_swirl=dyn.swirl,
        u_glow=look["glow"],
        u_heat=look["heat"],
        u_plasma=look["plasma"],
        u_swell=look["swell"],
        u_alert=look["alert"],
        u_standby=look["standby"],
        u_detail=detail,
    )


def paint_orb(painter: QPainter, rect: QRectF, dyn: Dynamics, compact: bool = False) -> None:
    """Draw the orb filling the square `rect` (its glow and tendrils stay inside it).
    `compact` forces the small-size detail level (the floating orb)."""
    side = min(rect.width(), rect.height())
    if side <= 0:
        return
    target = QRectF(rect.center().x() - side / 2, rect.center().y() - side / 2, side, side)
    device = painter.device()
    physical = side * (device.devicePixelRatioF() if device is not None else 1.0)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    gl = plasma.renderer()
    if gl is not None:
        size = plasma.render_size(physical, gl.software)
        frame = gl.render(size, shader_uniforms(dyn, plasma.detail_for(physical, gl.software, compact)))
        painter.drawImage(target, frame)
    else:
        # No OpenGL (tests, headless): a pre-rendered frame of the same orb, dimmed with the state.
        look = dyn.appearance()
        still = plasma.still_frame()
        if not still.isNull():
            painter.setOpacity(max(0.0, min(1.0, (0.55 + 0.6 * look["glow"]) * (1.0 - 0.45 * look["standby"]))))
            painter.drawImage(target, still)
    painter.restore()


class CoreVisual(QWidget):
    """A self-animating orb filling the widget's square. `set_mode` comes from
    GarvisState and `set_standby` from its standby flag; `level_source`
    (bridge.voice_level) is sampled only while speaking. `accent` is accepted
    for the callers' theme plumbing; the orb's colours are its own."""

    def __init__(
        self,
        accent: Accent,
        compact: bool = False,
        level_source: Callable[[], float] | None = None,
        parent: QWidget | None = None,
        *,
        reduced_motion: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self._accent = accent
        self._compact = compact
        self._level_source = level_source
        self._mode = Mode.IDLE
        self._standby = False
        self._dyn = Dynamics(system_prefers_reduced_motion() if reduced_motion is None else reduced_motion)
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def standby(self) -> bool:
        return self._standby

    @property
    def dynamics(self) -> Dynamics:
        return self._dyn

    @property
    def is_animating(self) -> bool:
        return self._timer.isActive()

    @property
    def timer_interval(self) -> int:
        return self._timer.interval()

    def set_mode(self, mode: str | Mode) -> None:
        self._mode = Mode(mode)
        self._update_timer()

    def set_standby(self, standby: bool) -> None:
        self._standby = bool(standby)
        self._update_timer()

    def pulse(self) -> None:
        """A brief flare (e.g. the orb landing after a collapse)."""
        self._dyn.flash = 1.0
        self._update_timer()

    def _update_timer(self) -> None:
        if not self.isVisible():
            self._timer.stop()
            return
        active = self._mode != Mode.IDLE or not self._dyn.settled
        if not active and self._dyn.reduced_motion:
            # Nothing moves at rest: draw the settled frame once and stop.
            self._timer.stop()
            self.update()
            return
        interval = ACTIVE_INTERVAL_MS if active else STANDBY_INTERVAL_MS if self._standby else IDLE_INTERVAL_MS
        gl = plasma.renderer()
        if gl is not None and gl.software:
            interval = max(interval, 1000 // plasma.SOFTWARE_FPS)  # a CPU-rendered frame costs ~25 ms
        if not self._timer.isActive() or self._timer.interval() != interval:
            self._timer.start(interval)
            if not self._clock.isValid():
                self._clock.start()

    def _tick(self) -> None:
        window = self.window()
        if window is not None and window.isMinimized():
            self._clock.restart()
            return
        dt = self._clock.restart() / 1000.0 if self._clock.isValid() else 0.0
        level = 0.0
        if self._mode == Mode.SPEAKING and self._level_source is not None:
            level = self._level_source()
        self._dyn.step(dt, self._mode, level, self._standby)
        if self._mode == Mode.IDLE and self._dyn.settled and self._timer.interval() == ACTIVE_INTERVAL_MS:
            self._update_timer()
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._clock.start()
        self._update_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        paint_orb(painter, QRectF(self.rect()), self._dyn, compact=self._compact)
        painter.end()
