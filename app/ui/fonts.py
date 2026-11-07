"""The bundled typefaces (app/ui/fonts) and the Material Symbols icon font.

Approved design (2026-10-31): Manrope for headings and numbers, Inter for all
other text, Material Symbols Rounded for icons -- outlined normally, filled
when selected. The files ship with Garvis (OFL / Apache licences beside them)
and are registered with Qt once, the first time anything asks for a family.

Nothing here may break the app: a font that fails to load is logged and the
UI falls back to Windows' own faces (Segoe UI) -- and, for icons, to the
painted glyphs in app/ui/dashboard.py or to text-only controls.

Icons are looked up by name in material_symbols_rounded.codepoints and drawn
by codepoint (never by ligature), so a missing name gives an empty string
instead of the name spelled out in the icon font.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QIcon, QPainter, QPixmap

FONT_DIR = Path(__file__).resolve().with_name("fonts")
CODEPOINTS_FILE = FONT_DIR / "material_symbols_rounded.codepoints"

DISPLAY = "Manrope"
TEXT = "Inter"
ICONS = "Material Symbols Rounded"
ICONS_FILLED = "Material Symbols Rounded Filled"

FONT_FILES: dict[str, tuple[str, ...]] = {
    DISPLAY: ("manrope_medium.ttf", "manrope_semibold.ttf", "manrope_bold.ttf", "manrope_extrabold.ttf"),
    TEXT: ("inter_regular.ttf", "inter_medium.ttf", "inter_semibold.ttf", "inter_bold.ttf"),
    ICONS: ("material_symbols_rounded.ttf",),
    ICONS_FILLED: ("material_symbols_rounded_filled.ttf",),
}

_loaded: dict[str, bool] | None = None


def _log(level: int, event: str, **fields) -> None:
    try:
        from app.core.logging import get_logger, log_extra

        log_extra(get_logger("ui.fonts"), level, event, **fields)
    except Exception:  # noqa: BLE001 -- logging must never stop the UI
        pass


def ensure_loaded() -> dict[str, bool]:
    """Register every bundled font with Qt (once; needs a Q(Gui)Application).
    Returns {family: available} -- available means Qt reports the family
    after registration, whether from our file or already installed."""
    global _loaded
    if _loaded is not None:
        return _loaded
    if QGuiApplication.instance() is None:
        return {family: False for family in FONT_FILES}  # too early: try again later
    result: dict[str, bool] = {}
    for family, files in FONT_FILES.items():
        registered = False
        for name in files:
            path = FONT_DIR / name
            font_id = QFontDatabase.addApplicationFont(str(path)) if path.exists() else -1
            if font_id < 0:
                _log(30, "fonts.load_failed", file=name, exists=path.exists())
                continue
            if family in QFontDatabase.applicationFontFamilies(font_id):
                registered = True
            else:
                _log(
                    30,
                    "fonts.unexpected_family",
                    file=name,
                    families=", ".join(QFontDatabase.applicationFontFamilies(font_id)),
                )
        result[family] = registered or family in QFontDatabase.families()
        if not result[family]:
            _log(30, "fonts.family_unavailable", family=family)
    _loaded = result
    _log(10, "fonts.loaded", **{k.replace(" ", "_"): v for k, v in result.items()})
    return result


def available(family: str) -> bool:
    return ensure_loaded().get(family, family in QFontDatabase.families())


def icons_available(filled: bool = False) -> bool:
    return available(ICONS_FILLED if filled else ICONS) and bool(codepoints())


@lru_cache(maxsize=1)
def codepoints() -> dict[str, int]:
    """Icon name -> codepoint, from the .codepoints file shipped with the font."""
    table: dict[str, int] = {}
    try:
        for line in CODEPOINTS_FILE.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    table.setdefault(parts[0], int(parts[1], 16))
                except ValueError:
                    continue
    except OSError as exc:
        _log(30, "fonts.codepoints_unreadable", error=str(exc))
    return table


def icon_char(name: str) -> str:
    """The icon's character, or "" when the name is unknown."""
    point = codepoints().get(name)
    return chr(point) if point is not None else ""


def icon_font(pixel_size: float, filled: bool = False) -> QFont:
    font = QFont(ICONS_FILLED if filled else ICONS)
    font.setPixelSize(max(1, int(round(pixel_size))))
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def draw_icon(
    painter: QPainter, rect: QRectF, name: str, color: QColor | str, pixel_size: float, filled: bool = False
) -> bool:
    """Draw icon `name` centred in `rect`. False (nothing drawn) when the
    icon font or the name is unavailable -- callers fall back."""
    char = icon_char(name)
    if not char or not icons_available(filled):
        return False
    painter.save()
    painter.setFont(icon_font(pixel_size, filled))
    painter.setPen(QColor(color))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, char)
    painter.restore()
    return True


def icon_pixmap(name: str, color: QColor | str, size: int = 20, filled: bool = False, ratio: float = 2.0) -> QPixmap:
    pixmap = QPixmap(int(size * ratio), int(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    draw_icon(painter, QRectF(0, 0, size, size), name, color, size, filled)
    painter.end()
    return pixmap


def icon(
    name: str, color: QColor | str, size: int = 20, filled: bool = False, disabled_color: QColor | str | None = None
) -> QIcon:
    """A QIcon of the glyph (empty when icons are unavailable, so a button
    keeps just its text)."""
    if not icon_char(name) or not icons_available(filled):
        return QIcon()
    result = QIcon(icon_pixmap(name, color, size, filled))
    if disabled_color is not None:
        result.addPixmap(icon_pixmap(name, disabled_color, size, filled), QIcon.Mode.Disabled)
    return result


def icon_html(name: str, color: str, pixel_size: int = 16, filled: bool = False) -> str:
    """The icon as a rich-text span (QLabel), or "" when unavailable."""
    char = icon_char(name)
    if not char or not icons_available(filled):
        return ""
    family = ICONS_FILLED if filled else ICONS
    return f"<span style=\"font-family:'{family}'; font-size:{pixel_size}px; color:{color};\">&#x{ord(char):x};</span>"
