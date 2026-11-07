"""Restrained motion for the redesigned Desktop UI.

Hover tints fade in and out over ~160 ms (ease-out) and pressing settles a
control; when Windows' "Show animations" is off (reduced motion) nothing
animates -- tints and toggles jump straight to their end state.

Style sheets can't animate, so push buttons get their hover tint from a
small transparent child (`_Tint`) faded by `HoverTints`, an event filter
installed once on the application. Custom-painted controls (navigation,
title-bar buttons, switches) use `Fader` directly.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from app.ui import theme

_reduced: bool | None = None


def reduced_motion() -> bool:
    """Windows' "Show animations in Windows" is off (read once per run)."""
    global _reduced
    if _reduced is None:
        from app.ui.visuals import system_prefers_reduced_motion

        _reduced = system_prefers_reduced_motion()
    return _reduced


def set_reduced_motion(value: bool | None) -> None:
    """Override (tests); None re-reads the system setting."""
    global _reduced
    _reduced = value


class Fader(QObject):
    """A 0..1 level that eases toward a target in MOTION_MS (or jumps there
    under reduced motion), calling `on_change` with each value."""

    def __init__(
        self, on_change: Callable[[float], None], duration: int = theme.MOTION_MS, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.value = 0.0
        self._target = 0.0
        self._on_change = on_change
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(duration)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._set)

    @property
    def target(self) -> float:
        return self._target

    def to(self, target: float) -> None:
        if target == self._target and (
            self._animation.state() == QVariantAnimation.State.Running or self.value == target
        ):
            return
        self._target = target
        self._animation.stop()
        if reduced_motion():
            self._set(target)
            return
        self._animation.setStartValue(float(self.value))
        self._animation.setEndValue(float(target))
        self._animation.start()

    def _set(self, value) -> None:
        self.value = float(value)
        self._on_change(self.value)


class _Tint(QWidget):
    """The hover wash painted over a push button (mouse-transparent)."""

    def __init__(self, button: QPushButton) -> None:
        super().__init__(button)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.level = 0.0
        self.fader = Fader(self._on_level, parent=self)
        self.setGeometry(button.rect())
        self.hide()

    def _on_level(self, value: float) -> None:
        self.level = value
        self.setVisible(value > 0.001)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        button = self.parentWidget()
        if button is None or self.level <= 0.001:
            return
        if button.property("accent"):
            color = QColor(255, 255, 255, int(20 * self.level))
        elif button.property("flat") or button.property("ghost"):
            color = QColor(58, 146, 230, int(30 * self.level))
        else:
            color = QColor(58, 146, 230, int(26 * self.level))
        radius = 8.0 if button.property("flat") else 15.0 if button.property("chip") else 12.0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = min(radius, rect.height() / 2)
        path.addRoundedRect(rect, radius, radius)
        painter.fillPath(path, color)
        painter.end()


class HoverTints(QObject):
    """Application event filter: a fading hover tint on every enabled push
    button (opt out with the dynamic property noTint=True)."""

    _ENTER = (QEvent.Type.Enter, QEvent.Type.HoverEnter)
    _LEAVE = (QEvent.Type.Leave, QEvent.Type.HoverLeave, QEvent.Type.Hide, QEvent.Type.EnabledChange)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        kind = event.type()
        if kind in self._ENTER or kind in self._LEAVE or kind == QEvent.Type.Resize:
            if isinstance(obj, QPushButton) and not obj.property("noTint"):
                self._handle(obj, kind)
        return False

    @staticmethod
    def _handle(button: QPushButton, kind) -> None:
        tint = button.findChild(_Tint, options=Qt.FindChildOption.FindDirectChildrenOnly)
        if kind == QEvent.Type.Resize:
            if tint is not None:
                tint.setGeometry(button.rect())
            return
        if kind in HoverTints._ENTER:
            if not button.isEnabled():
                return
            if tint is None:
                tint = _Tint(button)
            tint.setGeometry(button.rect())
            tint.raise_()
            tint.fader.to(1.0)
        elif tint is not None:
            tint.fader.to(0.0)


_installed: HoverTints | None = None


def install(app: QApplication | None = None) -> HoverTints | None:
    """Install the hover tints once per application (idempotent)."""
    global _installed
    app = app or QApplication.instance()
    if app is None:
        return None
    if _installed is None or _installed.parent() is not app:
        _installed = HoverTints(app)
        app.installEventFilter(_installed)
    return _installed
