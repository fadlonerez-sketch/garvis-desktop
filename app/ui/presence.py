"""Garvis's presence on the desktop: the floating orb, the orb <-> dashboard
transition, and the tray icon.

The orb is another view of the same Garvis (it reads the shared GarvisState
and calls back into MainWindow); it never talks to the agent itself.

Orb window: frameless, translucent, always on top, a tool window (no
taskbar / Alt+Tab entry) that never takes keyboard focus
(Qt.WindowDoesNotAcceptFocus -> WS_EX_NOACTIVATE on Windows). Its fully
transparent corners let clicks through to the window beneath.

Transition: the dashboard is never animated itself (re-laying out a full
widget tree every frame would stutter). Instead a snapshot of it is drawn on
a click-through overlay at the dashboard's exact geometry, revealed by a
light ring growing out of the orb (expand) or drawn back into it (collapse).
The real dashboard is shown only at the end, under the identical last
frame, so nothing jumps or flickers. One progress value 0..1 drives it, so
an interrupted transition simply reverses from where it is. When animations
are off (Settings), the switch is immediate.
"""

from __future__ import annotations

import math

from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget

from app.ui import theme
from app.ui.logo import render_logo_pixmap
from app.ui.placement import (
    UiPrefs,
    UiStateStore,
    corner_position,
    default_dashboard_rect,
    fit_rect,
    restore_position,
)
from app.ui.state import MODE_LABELS, GarvisState, Mode
from app.ui.visuals import CoreVisual

ORB_SIZE = 124
EXPAND_MS = 430
COLLAPSE_MS = 340
_FLOATING_FLAGS = (
    Qt.WindowType.Tool
    | Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.WindowDoesNotAcceptFocus
    | Qt.WindowType.NoDropShadowWindowHint
)


def _make_floating(widget: QWidget) -> None:
    widget.setWindowFlags(_FLOATING_FLAGS)
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)


class OrbWindow(QWidget):
    expand_requested = Signal()
    sources_requested = Signal(QPoint)
    menu_requested = Signal(QPoint)
    moved = Signal(QPoint)

    def __init__(self, state: GarvisState, level_source=None) -> None:
        super().__init__(None)
        _make_floating(self)
        self.setWindowTitle("Garvis orb")
        self.setFixedSize(ORB_SIZE, ORB_SIZE)
        self._state = state
        self._visual = CoreVisual(theme.ORANGE, compact=True, level_source=level_source, parent=self)
        self._visual.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._visual.setGeometry(0, 0, ORB_SIZE, ORB_SIZE)
        self._decor = _OrbDecor(self)
        self._decor.setGeometry(0, 0, ORB_SIZE, ORB_SIZE)
        self._decor.raise_()
        self._badge_count = 0
        self._press_global: QPoint | None = None
        self._press_pos = QPoint()
        self._dragging = False
        self._hover = False
        state.mode_changed.connect(self._on_mode_changed)
        state.voice_changed.connect(lambda: self._visual.set_standby(state.standby))
        state.status_changed.connect(lambda _status: self._refresh_tooltip())
        state.sources_changed.connect(self._refresh_badge)
        state.activity_changed.connect(self._refresh_badge)
        self._visual.set_mode(state.mode)
        self._visual.set_standby(state.standby)
        self._refresh_tooltip()

    @property
    def visual(self) -> CoreVisual:
        return self._visual

    @property
    def badge_count(self) -> int:
        return self._badge_count

    def pulse(self) -> None:
        self._visual.pulse()

    def _on_mode_changed(self, mode: str) -> None:
        self._visual.set_mode(mode)
        self._refresh_tooltip()

    def _refresh_badge(self) -> None:
        turn = self._state.latest_turn_with_sources
        count = len(turn.sources) if turn is not None else 0
        if count != self._badge_count:
            self._badge_count = count
            self._decor.update()

    def _refresh_tooltip(self) -> None:
        label = MODE_LABELS[self._state.mode]
        if self._state.mode == Mode.IDLE and self._state.standby:
            label = "Standby · wake word armed"
        self.setToolTip(f"Garvis — {label}\nClick to open · drag to move · right-click for more")

    @property
    def hovered(self) -> bool:
        return self._hover

    @staticmethod
    def badge_rect() -> QRectF:
        return QRectF(ORB_SIZE * 0.66, ORB_SIZE * 0.12, 22, 22)

    # --------------------------------------------------------------- mouse
    def enterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover = True
        self._decor.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover = False
        self._decor.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_global = event.globalPosition().toPoint()
            self._press_pos = self.pos()
            self._dragging = False
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._press_global is None:
            return
        delta = event.globalPosition().toPoint() - self._press_global
        if not self._dragging and delta.manhattanLength() >= QApplication.startDragDistance():
            self._dragging = True
        if self._dragging:
            self.move(self._press_pos + delta)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() != Qt.MouseButton.LeftButton or self._press_global is None:
            return
        self._press_global = None
        if self._dragging:
            self._dragging = False
            self.moved.emit(self.pos())
        elif self._badge_count and self.badge_rect().contains(event.position()):
            self.sources_requested.emit(event.globalPosition().toPoint())
        else:
            self.expand_requested.emit()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.menu_requested.emit(event.globalPos())


class _OrbDecor(QWidget):
    """Hover ring and sources badge, drawn above the orb's core."""

    def __init__(self, orb: OrbWindow) -> None:
        super().__init__(orb)
        self._orb = orb
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        orb = self._orb
        if not orb.badge_count and not orb.hovered:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center = QPointF(ORB_SIZE / 2, ORB_SIZE / 2)
        if orb.hovered:
            ring = QRadialGradient(center, ORB_SIZE * 0.47)
            ring.setColorAt(0.80, QColor(22, 119, 210, 0))
            ring.setColorAt(0.93, QColor(58, 146, 230, 80))
            ring.setColorAt(1.0, QColor(58, 146, 230, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(ring))
            painter.drawEllipse(center, ORB_SIZE * 0.47, ORB_SIZE * 0.47)
        if orb.badge_count:
            rect = orb.badge_rect()
            painter.setPen(QPen(QColor(theme.CYAN), 1.2))
            painter.setBrush(QColor(4, 16, 34, 235))
            painter.drawEllipse(rect)
            painter.setPen(QColor("#EAF6FF"))
            painter.setFont(theme.display_font(7.5, spacing=0))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(min(orb.badge_count, 99)))
        painter.end()


class TransitionOverlay(QWidget):
    """Click-through, focus-less layer that draws the orb <-> dashboard morph."""

    def __init__(self) -> None:
        super().__init__(None)
        _make_floating(self)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWindowTitle("Garvis transition")
        self._snapshot = QPixmap()
        self._origin = QPointF()
        self._progress = 0.0

    def set_scene(self, snapshot: QPixmap, origin: QPointF) -> None:
        self._snapshot = snapshot
        self._origin = origin

    @property
    def progress(self) -> float:
        return self._progress

    def set_progress(self, value: float) -> None:
        self._progress = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        p = self._progress
        w, h = self.width(), self.height()
        origin = self._origin
        orb_r = ORB_SIZE * 0.25
        far = max(math.hypot(origin.x() - x, origin.y() - y) for x, y in ((0, 0), (w, 0), (0, h), (w, h)))
        radius = orb_r + (far - orb_r) * p
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if not self._snapshot.isNull() and p > 0.0:
            clip = QPainterPath()
            clip.addEllipse(origin, radius, radius)
            painter.save()
            painter.setClipPath(clip)
            scale = 0.92 + 0.08 * p
            painter.translate(origin)
            painter.scale(scale, scale)
            painter.translate(-origin)
            painter.setOpacity(min(1.0, 0.2 + 1.1 * p))
            painter.drawPixmap(QRectF(0, 0, w, h), self._snapshot, QRectF(self._snapshot.rect()))
            painter.setOpacity(1.0)
            wash = QRadialGradient(origin, max(radius, 1.0))
            wash.setColorAt(0.0, QColor(47, 200, 255, int(70 * (1 - p))))
            wash.setColorAt(1.0, QColor(10, 60, 140, int(150 * (1 - p) ** 1.5)))
            painter.fillRect(QRectF(0, 0, w, h), QBrush(wash))
            painter.restore()

        fade = (1.0 - p) ** 0.6
        if 0.0 < p < 1.0:
            # The core's inclined orbits sweeping outward: the orb opening into the command center.
            sweep = math.sin(math.pi * p)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for tilt, angle, width in ((0.30, -24.0, 1.4), (0.52, 38.0, 1.1)):
                r = radius * (0.9 + 0.1 * p)
                painter.save()
                painter.translate(origin)
                painter.rotate(angle + 40.0 * p)
                painter.setPen(QPen(QColor(110, 215, 255, int(150 * sweep)), width))
                painter.drawEllipse(QPointF(0, 0), r, r * tilt)
                painter.restore()
        if fade > 0.01:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(47, 200, 255, int(60 * fade)), 18))
            painter.drawEllipse(origin, radius * 0.985, radius * 0.985)
            painter.setPen(QPen(QColor(200, 245, 255, int(230 * fade)), 1.6 + 3.0 * (1 - p)))
            painter.drawEllipse(origin, radius, radius)
            # The orb's own glow: its icy core spreading into the dashboard's Electric Blue.
            glow = QRadialGradient(origin, orb_r * (1.6 + 2.0 * p))
            glow.setColorAt(0.0, QColor.fromRgbF(0.80 - 0.15 * p, 0.93 - 0.08 * p, 1.0, 0.95 * fade))
            glow.setColorAt(0.45, QColor.fromRgbF(0.09 + 0.05 * p, 0.47 + 0.10 * p, 0.82 + 0.10 * p, 0.55 * fade))
            glow.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(origin, orb_r * (1.6 + 2.0 * p), orb_r * (1.6 + 2.0 * p))
        painter.end()


class PresenceController(QObject):
    """Owns the orb, the transition and the tray; drives `dashboard`
    (MainWindow) through show/hide/geometry only."""

    view_changed = Signal(str)  # "expanded" | "collapsed"
    menu_requested = Signal(QPoint)
    sources_requested = Signal(QPoint)
    quit_requested = Signal()

    def __init__(
        self,
        dashboard: QWidget,
        state: GarvisState,
        store: UiStateStore,
        level_source=None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._dashboard = dashboard
        self._state = state
        self._store = store
        self.prefs: UiPrefs = store.load()
        self._orb = OrbWindow(state, level_source)
        self._orb.expand_requested.connect(lambda: self.expand())
        self._orb.moved.connect(self._on_orb_moved)
        self._orb.menu_requested.connect(self.menu_requested)
        self._orb.sources_requested.connect(self.sources_requested)
        self._overlay = TransitionOverlay()
        self._anim = QVariantAnimation(self)
        self._anim.valueChanged.connect(self._overlay.set_progress)
        self._anim.finished.connect(self._on_anim_finished)
        self._target = "expanded"  # Garvis starts on the dashboard
        self._tray: QSystemTrayIcon | None = None

    # -------------------------------------------------------------- reads
    @property
    def orb(self) -> OrbWindow:
        return self._orb

    @property
    def overlay(self) -> TransitionOverlay:
        return self._overlay

    @property
    def view(self) -> str:
        return self._target

    @property
    def animating(self) -> bool:
        return self._anim.state() == QVariantAnimation.State.Running

    def _animations_on(self) -> bool:
        return self.prefs.animations and QGuiApplication.platformName() not in ("offscreen", "minimal")

    # ---------------------------------------------------------- geometry
    @staticmethod
    def _screens() -> list[QRect]:
        return [screen.availableGeometry() for screen in QGuiApplication.screens()]

    @staticmethod
    def _primary_available() -> QRect:
        screen = QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 800)

    def orb_position(self) -> QPoint:
        size = QSize(ORB_SIZE, ORB_SIZE)
        saved = tuple(self.prefs.orb_pos) if self.prefs.orb_pos else None
        restored = restore_position(saved, size, self._screens())
        return (
            restored
            if restored is not None
            else corner_position(self._primary_available(), size, self.prefs.orb_corner)
        )

    def dashboard_rect(self) -> QRect:
        if self.prefs.dashboard_rect:
            rect = QRect(*self.prefs.dashboard_rect)
            screen = QGuiApplication.screenAt(rect.center())
            if screen is not None:
                return fit_rect(rect, screen.availableGeometry())
        return default_dashboard_rect(self._primary_available())

    def remember_dashboard_rect(self, rect: QRect) -> None:
        self.prefs.dashboard_rect = [rect.x(), rect.y(), rect.width(), rect.height()]

    def _orb_center_in(self, rect: QRect) -> QPointF:
        """The orb's centre in `rect`'s coordinates, kept on the dashboard's
        own screen (the overlay only covers that screen)."""
        center = self.orb_position() + QPoint(ORB_SIZE // 2, ORB_SIZE // 2)
        local = QPointF(center - rect.topLeft())
        margin = ORB_SIZE * 0.25
        return QPointF(
            min(max(local.x(), -margin), rect.width() + margin), min(max(local.y(), -margin), rect.height() + margin)
        )

    # -------------------------------------------------------- transitions
    def expand(self, animated: bool = True) -> None:
        if self._target == "expanded" and not self.animating:
            self._dashboard_up()
            return
        if self._target == "expanded":
            return  # already on its way
        self._target = "expanded"
        if not animated or not self._animations_on():
            self._anim.stop()
            self._finish_expanded()
            return
        if not self.animating:  # starting from the orb: build the scene
            rect = self._dashboard.target_rect(self)
            self._dashboard.setGeometry(rect)
            self._prepare_overlay(rect, self._dashboard.snapshot(), 0.0)
        self._orb.hide()
        self._run(1.0, EXPAND_MS, QEasingCurve.Type.OutCubic)

    def collapse(self, animated: bool = True) -> None:
        if not self.prefs.orb_enabled:
            self._dashboard.showMinimized()  # no orb to go to: plain minimize
            return
        if self._target == "collapsed":
            return
        self._target = "collapsed"
        if self._dashboard.isVisible():
            self._dashboard.remember_geometry(self)
        if not animated or not self._animations_on():
            self._anim.stop()
            self._finish_collapsed()
            return
        if not self.animating:
            rect = self._dashboard.geometry()
            self._prepare_overlay(rect, self._dashboard.snapshot(), 1.0)
            self._overlay.repaint()
            QTimer.singleShot(16, self._hide_dashboard_if_collapsing)
        self._run(0.0, COLLAPSE_MS, QEasingCurve.Type.InOutCubic)

    def toggle(self) -> None:
        if self._target == "expanded":
            self.collapse()
        else:
            self.expand()

    def _prepare_overlay(self, rect: QRect, snapshot: QPixmap, progress: float) -> None:
        self._overlay.setGeometry(rect)
        self._overlay.set_scene(snapshot, self._orb_center_in(rect))
        self._overlay.set_progress(progress)
        self._overlay.show()

    def _run(self, end: float, full_ms: int, curve: QEasingCurve.Type) -> None:
        start = self._overlay.progress
        self._anim.stop()
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.setDuration(max(60, int(full_ms * abs(end - start))))
        self._anim.setEasingCurve(curve)
        self._anim.start()

    def _hide_dashboard_if_collapsing(self) -> None:
        if self._target == "collapsed":
            self._dashboard.hide()

    def _on_anim_finished(self) -> None:
        if self._target == "expanded":
            self._finish_expanded()
        else:
            self._finish_collapsed()

    def _finish_expanded(self) -> None:
        self._orb.hide()
        self._dashboard_up()
        # Hidden only after the real window has painted under the last frame.
        QTimer.singleShot(45, self._hide_overlay_if_settled)
        self.view_changed.emit("expanded")

    def _finish_collapsed(self) -> None:
        self._dashboard.hide()
        self._overlay.hide()
        self._overlay.set_scene(QPixmap(), QPointF())
        self.show_orb()
        self._orb.pulse()
        self.save()
        self.view_changed.emit("collapsed")

    def _hide_overlay_if_settled(self) -> None:
        if self._target == "expanded" and not self.animating:
            self._overlay.hide()
            self._overlay.set_scene(QPixmap(), QPointF())

    def _dashboard_up(self) -> None:
        self._dashboard.present(self)  # shows, restores, raises and focuses (an explicit user action)

    # ---------------------------------------------------------------- orb
    def show_orb(self) -> None:
        if not self.prefs.orb_enabled:
            return
        self._orb.move(self.orb_position())
        self._orb.show()
        self._orb.raise_()

    def hide_orb(self) -> None:
        self._orb.hide()

    def set_orb_enabled(self, enabled: bool) -> None:
        self.prefs.orb_enabled = enabled
        self.save()
        if not enabled:
            self._orb.hide()
            if self._target == "collapsed":
                self.expand(animated=False)  # never leave Garvis with no window at all
        elif self._target == "collapsed":
            self.show_orb()

    def set_orb_corner(self, corner: str) -> None:
        self.prefs.orb_corner = corner
        self.prefs.orb_pos = None  # the corner wins over a dragged position
        self.save()
        if self._orb.isVisible():
            self._orb.move(self.orb_position())

    def set_animations(self, enabled: bool) -> None:
        self.prefs.animations = enabled
        self.save()

    def _on_orb_moved(self, pos: QPoint) -> None:
        self.prefs.orb_pos = [pos.x(), pos.y()]
        self.save()

    def save(self) -> None:
        self._store.save(self.prefs)

    # --------------------------------------------------------------- tray
    def install_tray(self, menu: QMenu) -> QSystemTrayIcon | None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return None
        tray = QSystemTrayIcon(QIcon(render_logo_pixmap(64, theme.ORANGE)), self)
        tray.setToolTip("Garvis")
        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        self._tray = tray
        return tray

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.expand() if self._target == "collapsed" else self._dashboard_up()

    def shutdown(self) -> None:
        self._anim.stop()
        if self._dashboard.isVisible():
            self._dashboard.remember_geometry(self)
        self.save()
        self._orb.close()
        self._overlay.close()
        if self._tray is not None:
            self._tray.hide()
