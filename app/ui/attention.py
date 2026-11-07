"""The desktop attention system: when Garvis opens or presents something the
user should see, an orbital pulse appears at the mouse cursor and resolves
into a small card naming exactly what was opened, with a way to bring it
into view.

Driven only by PRESENTATION events (app/agent/presentation.py) -- the same
event for a chat request, a phone task or a protocol step -- so it shows
real results: the window title the desktop reported, the URL or path that
was opened. Nothing here guesses or fetches content.

Windows: both surfaces are frameless, always-on-top tool windows that never
take keyboard focus (WindowDoesNotAcceptFocus / ShowWithoutActivating), so
typing in another app is never interrupted. The pulse is fully
click-through; the card takes clicks (its buttons) but not focus. Positions
are Qt logical coordinates from QCursor/QScreen, so they are right on every
monitor and at any display scaling; the card is kept inside the available
area of the cursor's screen.

Pacing: one card at a time -- later items during a burst (a protocol opening
five pages) stack into it -- and at most one pulse per PULSE_COOLDOWN.
Everything is torn down by dismiss()/shutdown(); timers are children of
their windows.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable

from PySide6.QtCore import QEasingCurve, QObject, QPoint, QPointF, QRect, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFontMetrics,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.ui import theme

PULSE_SIZE = 280
PULSE_MS = 820
CARD_DELAY_MS = 240
PULSE_COOLDOWN = 1.6
DISMISS_MS = {"prominent": 9000, "subtle": 6000}
MAX_ROWS = 4
CARD_WIDTH = 360
_FLAGS = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
          | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.NoDropShadowWindowHint)
_KIND_LABEL = {"website": "WEB PAGE", "application": "APPLICATION", "path": "FILE · FOLDER", "notice": "GARVIS",
               "result": "RESULT"}


def _floating(widget: QWidget, click_through: bool) -> None:
    widget.setWindowFlags(_FLAGS | (Qt.WindowType.WindowTransparentForInput if click_through else Qt.WindowType.Widget))
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    if click_through:
        widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


def _ease(t: float) -> float:
    return 1 - (1 - t) ** 3


class PulseOverlay(QWidget):
    """The orbital pulse: rings opening out of a point of light, cooling
    from the orb's orange to the command center's cyan. Click-through."""

    def __init__(self) -> None:
        super().__init__(None)
        _floating(self, click_through=True)
        self.setWindowTitle("Garvis attention")
        self.setFixedSize(PULSE_SIZE, PULSE_SIZE)
        self._t = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(PULSE_MS)
        self._anim.valueChanged.connect(self._set_t)
        self._anim.finished.connect(self.hide)

    @property
    def running(self) -> bool:
        return self._anim.state() == QVariantAnimation.State.Running

    def play(self, center: QPoint, screen_rect: QRect | None = None) -> None:
        half = PULSE_SIZE // 2
        top_left = QPoint(center.x() - half, center.y() - half)
        self.move(top_left)
        self._t = 0.0
        self.show()
        self._anim.stop()
        self._anim.start()

    def stop(self) -> None:
        self._anim.stop()
        self.hide()

    def _set_t(self, value) -> None:
        self._t = float(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        t = self._t
        e = _ease(t)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        c = QPointF(self.width() / 2, self.height() / 2)
        fade = (1 - t) ** 1.3
        # A soft dark backing so the light reads on white pages as well as dark desktops
        backing = QRadialGradient(c, 135)
        backing.setColorAt(0.0, QColor(2, 8, 20, int(120 * fade)))
        backing.setColorAt(0.6, QColor(2, 8, 20, int(55 * fade)))
        backing.setColorAt(1.0, QColor(2, 8, 20, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(backing))
        painter.drawEllipse(c, 135, 135)
        # Glow: orange cooling to cyan
        glow_r = 26 + 70 * e
        glow = QRadialGradient(c, glow_r)
        warm = QColor.fromRgbF(1.0 - 0.75 * e, 0.55 + 0.28 * e, 0.12 + 0.88 * e, 0.42 * fade)
        glow.setColorAt(0.0, warm)
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(c, glow_r, glow_r)
        # Two inclined orbits opening out and turning
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for tilt, angle, width, lag in ((0.34, -28.0, 1.6, 0.0), (0.58, 34.0, 1.2, 0.08)):
            local = max(0.0, min(1.0, (t - lag) / (1 - lag)))
            r = 14 + 112 * _ease(local)
            painter.save()
            painter.translate(c)
            painter.rotate(angle + 55 * e)
            painter.setPen(QPen(QColor(110, 215, 255, int(245 * (1 - local) ** 1.1)), width + 0.6))
            painter.drawEllipse(QPointF(0, 0), r, r * tilt)
            # a spark riding the outer orbit
            if tilt < 0.4:
                a = math.radians(300 * e)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(235, 250, 255, int(240 * (1 - local))))
                painter.drawEllipse(QPointF(r * math.cos(a), r * tilt * math.sin(a)), 2.4, 2.4)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.restore()
        # Expanding hairline ring
        ring_r = 10 + 120 * e
        painter.setPen(QPen(QColor(190, 240, 255, int(200 * fade)), 1.5))
        painter.drawEllipse(c, ring_r, ring_r)
        # The point of light at the cursor
        core_r = 7 * (1 - e) + 2.5
        core = QRadialGradient(c, core_r * 2.4)
        core.setColorAt(0.0, QColor(255, 244, 225, int(255 * fade)))
        core.setColorAt(0.45, QColor.fromRgbF(1.0 - 0.7 * e, 0.62 + 0.25 * e, 0.25 + 0.75 * e, 0.8 * fade))
        core.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(core))
        painter.drawEllipse(c, core_r * 2.4, core_r * 2.4)
        painter.end()


class _Row(QWidget):
    def __init__(self, item: dict, on_show: Callable[[int], None], compact: bool, parent: QWidget) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        text = QVBoxLayout()
        text.setSpacing(1)
        width = CARD_WIDTH - (130 if item.get("handle") else 60)
        title = QLabel(QFontMetrics(theme.display_font(10.5)).elidedText(item.get("title", ""), Qt.TextElideMode.ElideRight, width))
        title.setStyleSheet(f"color: {theme.TEXT_1}; font-family: '{theme.display_family()}'; "
                            f"font-size: {13 if compact else 14}px; font-weight: 600;")
        title.setToolTip(item.get("title", ""))
        text.addWidget(title)
        detail = item.get("detail") or ""
        if detail:
            sub = QLabel(QFontMetrics(theme.mono_font(8)).elidedText(detail, Qt.TextElideMode.ElideMiddle, width))
            sub.setStyleSheet(f"color: {theme.TEXT_DIM}; font-family: '{theme.mono_family()}'; font-size: 8pt;")
            sub.setToolTip(detail)
            text.addWidget(sub)
        layout.addLayout(text, 1)
        self.show_button: QPushButton | None = None
        if item.get("handle"):
            self.show_button = QPushButton("Show")
            self.show_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.show_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.show_button.setToolTip("Bring this window to the front")
            self.show_button.clicked.connect(lambda: on_show(int(item["handle"])))
            layout.addWidget(self.show_button, 0, Qt.AlignmentFlag.AlignVCenter)


class AttentionCard(QWidget):
    """The card that names what was opened. Clickable, never focus-taking."""

    show_requested = Signal(int)
    activity_requested = Signal()
    dismissed = Signal()

    def __init__(self) -> None:
        super().__init__(None)
        _floating(self, click_through=False)
        self.setWindowTitle("Garvis — opened")
        self.setFixedWidth(CARD_WIDTH)
        self.setStyleSheet(theme.stylesheet())
        self._items: list[dict] = []
        self._level = "subtle"
        self._hover = False
        self._progress = 1.0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 12, 14, 12)
        outer.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.kicker = QLabel("")
        self.kicker.setStyleSheet(f"color: {theme.ACCENT_TEXT}; font-family: '{theme.display_family()}'; "
                                  "font-size: 11px; font-weight: 600;")
        theme.tracked(self.kicker, 1.2)
        head.addWidget(self.kicker)
        head.addStretch(1)
        self.close_button = QPushButton("×")
        self.close_button.setProperty("flat", True)
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.setToolTip("Dismiss")
        self.close_button.clicked.connect(self.dismiss)
        head.addWidget(self.close_button)
        outer.addLayout(head)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(4)
        outer.addLayout(self.rows)
        foot = QHBoxLayout()
        self.more_label = QLabel("")
        self.more_label.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 12px;")
        foot.addWidget(self.more_label)
        foot.addStretch(1)
        self.activity_button = QPushButton("Activity")
        self.activity_button.setProperty("flat", True)
        self.activity_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.activity_button.setToolTip("See this in Garvis's Activity page")
        self.activity_button.clicked.connect(self._on_activity)
        foot.addWidget(self.activity_button)
        outer.addLayout(foot)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)
        self._enter = QVariantAnimation(self)
        self._enter.setStartValue(0.0)
        self._enter.setEndValue(1.0)
        self._enter.setDuration(260)
        self._enter.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._enter.valueChanged.connect(self._set_progress)
        self._anchor = QPoint()

    @property
    def items(self) -> list[dict]:
        return list(self._items)

    @property
    def dismiss_interval(self) -> int:
        return self._timer.interval()

    def present(self, items: list[dict], level: str, anchor: QPoint, area: QRect) -> None:
        self._items = list(items)
        self._level = level
        self._rebuild()
        self.adjustSize()
        self.move(self._position(anchor, area))
        self._anchor = self.pos()
        self._progress = 0.0
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self._enter.stop()
        self._enter.start()
        self._restart_timer()

    def add(self, item: dict, level: str) -> None:
        self._items.append(item)
        if level == "prominent":
            self._level = "prominent"
        self._rebuild()
        self.adjustSize()
        self._restart_timer()

    def _rebuild(self) -> None:
        while self.rows.count():
            widget = self.rows.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        latest = self._items[-MAX_ROWS:]
        kinds = {item.get("kind") for item in latest}
        if len(self._items) == 1:
            self.kicker.setText(("OPENED · " if latest[0].get("kind") != "notice" else "")
                                + _KIND_LABEL.get(latest[0].get("kind"), "RESULT"))
        else:
            self.kicker.setText(f"OPENED · {len(self._items)} ITEMS" if kinds != {"notice"} else "GARVIS")
        for item in reversed(latest):
            self.rows.addWidget(_Row(item, self.show_requested.emit, len(self._items) > 1, self))
        hidden = len(self._items) - len(latest)
        self.more_label.setText(f"+ {hidden} more in Activity" if hidden else "")

    def _position(self, anchor: QPoint, area: QRect) -> QPoint:
        width, height = self.width(), max(self.height(), self.sizeHint().height())
        x, y = anchor.x() + 26, anchor.y() + 26
        if x + width > area.right() - 8:
            x = anchor.x() - 26 - width
        if y + height > area.bottom() - 8:
            y = anchor.y() - 26 - height
        x = max(area.left() + 8, min(x, area.right() - 8 - width))
        y = max(area.top() + 8, min(y, area.bottom() - 8 - height))
        return QPoint(x, y)

    def _restart_timer(self) -> None:
        if not self._hover:
            self._timer.start(DISMISS_MS.get(self._level, 6000) + 1500 * max(0, len(self._items) - 1))

    def _set_progress(self, value) -> None:
        self._progress = float(value)
        self.setWindowOpacity(self._progress)
        self.move(self._anchor.x(), self._anchor.y() + int(round(8 * (1 - self._progress))))
        self.update()

    def _on_activity(self) -> None:
        self.activity_requested.emit()
        self.dismiss()

    def dismiss(self) -> None:
        self._timer.stop()
        self._enter.stop()
        was_visible = self.isVisible()
        self.hide()
        self._items = []
        if was_visible:
            self.dismissed.emit()

    def enterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover = True
        self._timer.stop()  # reading it: stay
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover = False
        self._timer.start(3000)
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, 16, 16)
        # The approved raised surface (Midnight Blue gradient, hairline border)
        surface = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        surface.setColorAt(0.0, QColor(19, 48, 90, 246))
        surface.setColorAt(1.0, QColor(14, 36, 66, 246))
        painter.fillPath(path, surface)
        painter.setPen(QPen(QColor(160, 200, 245, 34), 1))
        painter.drawPath(path)
        # Accent on the left edge: amber for a prominent result, Electric Blue otherwise
        accent = QColor(theme.ORANGE.primary if self._level == "prominent" else theme.ELECTRIC_HI)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(QRectF(rect.left() + 6, rect.top() + 14, 3, rect.height() - 28), 1.5, 1.5)
        painter.end()


class AttentionController(QObject):
    """Turns PRESENTATION payloads into pulses and cards. Thread: GUI only
    (the bridge's signal is queued here from the agent loop)."""

    reveal_requested = Signal(int)
    activity_requested = Signal()

    def __init__(self, cursor_pos: Callable[[], QPoint] = QCursor.pos, clock: Callable[[], float] = time.monotonic,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cursor_pos = cursor_pos
        self._clock = clock
        self.animations = True
        self.pulse = PulseOverlay()
        self.card = AttentionCard()
        self.card.show_requested.connect(self._on_show)
        self.card.activity_requested.connect(self.activity_requested)
        self._seen: list[str] = []
        self._recent: dict[tuple, float] = {}
        self._last_pulse = -1e9
        self._pending: list[tuple[dict, str]] = []
        self._card_timer = QTimer(self)
        self._card_timer.setSingleShot(True)
        self._card_timer.timeout.connect(self._show_pending)
        self.shown_count = 0
        """Presentations actually displayed (each once)."""

    def present(self, payload: dict) -> bool:
        """Show one presentation; False if it's a duplicate or needs no attention."""
        level = payload.get("level")
        if level not in ("subtle", "prominent"):
            return False
        event_id = str(payload.get("id") or "")
        key = (payload.get("kind"), payload.get("title"), payload.get("detail"))
        now = self._clock()
        if (event_id and event_id in self._seen) or now - self._recent.get(key, -1e9) < 2.0:
            return False
        if event_id:
            self._seen = (self._seen + [event_id])[-200:]
        self._recent = {k: v for k, v in self._recent.items() if now - v < 10.0}
        self._recent[key] = now
        self.shown_count += 1
        item = dict(payload)
        if self.card.isVisible() and not self._card_timer.isActive():
            self.card.add(item, level)
            return True
        self._pending.append((item, level))
        if self._card_timer.isActive():
            return True
        anchor, area = self._anchor()
        pulse = level == "prominent" and self.animations and now - self._last_pulse >= PULSE_COOLDOWN
        if pulse:
            self._last_pulse = now
            self.pulse.play(anchor, area)
        self._card_timer.start(CARD_DELAY_MS if pulse else 0)
        return True

    def _anchor(self) -> tuple[QPoint, QRect]:
        anchor = self._cursor_pos()
        screen = QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else QRect(0, 0, 1920, 1080)
        if not area.contains(anchor):  # cursor unknown or off-screen: use the screen's centre
            anchor = area.center()
        return anchor, area

    def _show_pending(self) -> None:
        if not self._pending:
            return
        items = [item for item, _ in self._pending]
        level = "prominent" if any(lvl == "prominent" for _, lvl in self._pending) else "subtle"
        self._pending = []
        anchor, area = self._anchor()
        if self.card.isVisible():
            for item in items:
                self.card.add(item, level)
        else:
            self.card.present(items, level, anchor, area)

    def _on_show(self, handle: int) -> None:
        self.reveal_requested.emit(handle)
        self.card.dismiss()

    def dismiss(self) -> None:
        self._card_timer.stop()
        self._pending = []
        self.pulse.stop()
        self.card.dismiss()

    def shutdown(self) -> None:
        self.dismiss()
        self.pulse.deleteLater()
        self.card.deleteLater()
