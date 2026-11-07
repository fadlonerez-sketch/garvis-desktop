"""Garvis visual identity: colour tokens, fonts and the dashboard stylesheet.

The approved Desktop redesign (2026-10-31):
Midnight Blue raised surfaces, Electric Blue for primary actions, the active
navigation item and focus, rounded cards on a deep navy background. Manrope
for headings and numbers, Inter for all other text, Material Symbols Rounded
for icons -- all bundled in app/ui/fonts and registered by app/ui/fonts.py;
if one fails to load, Windows' own faces take over.

The title-bar logo lockup is NOT part of this palette: the amber logo is
frozen in app/ui/logo.py and the wordmark keeps its original Bahnschrift
type and colours (roles "wordmark" / "tagline").
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication

from app.ui import fonts


@dataclass(frozen=True)
class Accent:
    """The colours one luminous visual is drawn with."""

    core: str  # hottest centre
    primary: str  # main glow
    deep: str  # outer, darkest tone
    trace: str  # fine lines / ticks


BLUE = Accent(core="#E8FBFF", primary="#2FC8FF", deep="#0B4F9E", trace="#5FD7FF")
ORANGE = Accent(core="#FFF1D6", primary="#FF8A1F", deep="#9E2E06", trace="#FFB25C")
ALERT = Accent(core="#FFE3E6", primary="#FF4D5E", deep="#7A0E1C", trace="#FF8A95")

# ---- Approved design tokens -------------------------------------------------
BG = "#050B16"
MIDNIGHT = "#102B4C"
"""Midnight Blue: raised surfaces and selected states."""
ELECTRIC = "#1677D2"
"""Electric Blue: primary actions, the active navigation item, focus."""
ELECTRIC_HI = "#3A92E6"
ELECTRIC_GRADIENT_END = "#2A86DE"
ELECTRIC_SOFT = "rgba(22, 119, 210, 0.16)"
SURFACE_1 = "#0B1A2F"
SURFACE_2 = "#0F2440"
SURFACE_3 = "#15325A"
CARD_TOP = "#102440"
CARD_BOTTOM = "#0B1A2F"
TEXT_1 = "#EAF2FB"
TEXT_2 = "#A3B6CE"
TEXT_3 = "#6C84A2"
ACCENT_TEXT = "#7DB9F2"
"""Card-title icons and in-card links."""
NAV_ACTIVE = "#8CC4F5"
LINE = "rgba(160, 200, 245, 0.08)"
LINE_STRONG = "rgba(58, 146, 230, 0.45)"
OK = "#3DD6A0"
WARN = "#F2B652"
WARN_TEXT = "#F5C77A"
ERROR = "#F06A6A"
MOTION_MS = 160
"""Hover tints and toggles: 160 ms, ease-out (nothing moves with reduced motion)."""

# ---- Names the rest of the UI already uses, mapped onto the tokens ----------
BG_TOP = BG
BG_BOTTOM = BG
PANEL_BG = "rgba(16, 36, 64, 0.78)"
TEXT = TEXT_1
TEXT_DIM = TEXT_2
TEXT_FAINT = TEXT_3
CYAN = ELECTRIC_HI
"""Generic accent (running marks, chart lines, highlights)."""
# Telemetry series: one hue per resource, all within the blue family.
CHART_CPU = ELECTRIC_HI
CHART_MEMORY = "#7D9BFF"
CHART_GPU = "#46D6C2"
CHART_PROCESS = "#8FD3FF"
CHART_DISK_READ = "#5FB4F5"
CHART_DISK_WRITE = "#8FA8FF"

# The original logo lockup (unchanged by the redesign).
WORDMARK = "#EAF6FF"
TAGLINE = "#4C6685"


def qcolor(value: str, alpha: float | None = None) -> QColor:
    color = QColor(value)
    if alpha is not None:
        color.setAlphaF(max(0.0, min(1.0, alpha)))
    return color


_family_cache: frozenset[str] | None = None


def _families() -> frozenset[str]:
    global _family_cache
    if _family_cache is not None:
        return _family_cache
    fonts.ensure_loaded()
    families = frozenset(QFontDatabase.families())
    if QGuiApplication.instance() is not None:
        _family_cache = families  # only final once the bundled fonts could register
    return families


def _pick(*candidates: str) -> str:
    families = _families()
    return next((name for name in candidates if name in families), candidates[-1])


def display_family() -> str:
    """Headings and numbers."""
    return _pick(fonts.DISPLAY, "Segoe UI Variable Display", "Segoe UI", "Sans Serif")


def text_family() -> str:
    """All other text."""
    return _pick(fonts.TEXT, "Segoe UI Variable Text", "Segoe UI", "Sans Serif")


def brand_family() -> str:
    """The wordmark's original face (the logo lockup is unchanged)."""
    return _pick("Bahnschrift", "Segoe UI Variable Display", "Segoe UI", "Sans Serif")


def mono_family() -> str:
    return _pick("Cascadia Mono", "Cascadia Code", "Consolas", "Monospace")


def display_font(point_size: float, weight: QFont.Weight = QFont.Weight.DemiBold, spacing: float = 0.0) -> QFont:
    font = QFont(display_family())
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def text_font(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(text_family())
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    return font


def mono_font(point_size: float) -> QFont:
    font = QFont(mono_family())
    font.setPointSizeF(point_size)
    return font


def tracked(widget, spacing: float):
    """Letter-spacing for a widget's text (style sheets can't set it; their
    font-family/size still apply on top). Returns the widget."""
    font = widget.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    widget.setFont(font)
    return widget


_ICON_DIR = Path(tempfile.gettempdir()) / "garvis-ui-icons"


def icon_file(name: str, color: str, size: int = 16) -> str:
    """A PNG of a Material Symbols glyph for style-sheet `image:` rules
    (combo-box arrows, check marks). "" when it can't be made."""
    try:
        path = _ICON_DIR / f"{name}-{color.strip('#')}-{size}.png"
        if not path.exists():
            if not fonts.icons_available():
                return ""
            _ICON_DIR.mkdir(parents=True, exist_ok=True)
            if not fonts.icon_pixmap(name, color, size, ratio=2.0).save(str(path)):
                return ""
        return path.as_posix()
    except Exception:  # noqa: BLE001 -- purely cosmetic
        return ""


def _image(name: str, color: str, size: int = 16) -> str:
    path = icon_file(name, color, size)
    return f'image: url("{path}");' if path else "image: none;"


def stylesheet() -> str:
    display, text, mono, brand = display_family(), text_family(), mono_family(), brand_family()
    return f"""
QWidget {{
    color: {TEXT_1};
    font-family: "{text}";
    font-size: 13px;
}}
QLabel {{ background: transparent; }}
QLabel[role="panelTitle"] {{ color: {TEXT_1}; font-family: "{display}"; font-size: 14px; font-weight: 600; }}
QLabel[role="cardIcon"] {{ color: {ACCENT_TEXT}; }}
QLabel[role="panelIndex"] {{ color: {TEXT_3}; font-size: 11px; font-weight: 500; }}
QLabel[role="dim"] {{ color: {TEXT_2}; }}
QLabel[role="faint"] {{ color: {TEXT_3}; font-size: 12px; }}
QLabel[role="key"] {{ color: {TEXT_3}; font-size: 13px; }}
QLabel[role="value"] {{ color: {TEXT_1}; font-size: 13px; }}
QLabel[role="stat"] {{ color: {TEXT_1}; font-family: "{display}"; font-size: 19px; font-weight: 700; }}
QLabel[role="statLabel"] {{ color: {TEXT_3}; font-size: 12px; }}
QLabel[role="metric"] {{ color: {TEXT_1}; font-family: "{display}"; font-size: 20px; font-weight: 700; }}
QLabel[role="metricSmall"] {{ color: {TEXT_1}; font-family: "{display}"; font-size: 13px; font-weight: 600; }}
QLabel[role="section"] {{ color: {TEXT_3}; font-size: 11px; font-weight: 600; }}
QLabel[role="pageTitle"] {{ color: {TEXT_1}; font-family: "{display}"; font-size: 22px; font-weight: 700; }}
QLabel[role="pageSub"] {{ color: {TEXT_2}; font-size: 13px; }}
QLabel[role="stateTitle"] {{ color: {TEXT_1}; font-family: "{display}"; font-size: 28px; font-weight: 700; }}
QLabel[role="stateHint"] {{ color: {TEXT_2}; font-size: 14px; }}
QLabel[role="meterLabel"] {{ color: {TEXT_2}; font-size: 13px; }}
QLabel[role="clock"] {{ color: {TEXT_2}; font-family: "{display}"; font-size: 13px; font-weight: 600; }}
QLabel[role="titleStatus"] {{ color: {TEXT_2}; font-size: 13px; }}
QLabel[role="wordmark"] {{ color: {WORDMARK}; font-family: "{brand}"; font-size: 14pt; font-weight: 400; }}
QLabel[role="tagline"] {{ color: {TAGLINE}; font-family: "{brand}"; font-size: 7.5pt; font-weight: 400; }}
QLabel[role="chip"] {{
    color: {TEXT_2}; font-size: 12px; font-weight: 500;
    border: 1px solid {LINE}; border-radius: 10px; padding: 2px 9px; background: rgba(16, 43, 76, 0.7);
}}
QLabel[role="chip"][tone="ok"] {{ color: {OK}; background: rgba(61, 214, 160, 0.08); border-color: rgba(61, 214, 160, 0.25); }}
QLabel[role="chip"][tone="warn"] {{ color: {WARN_TEXT}; background: rgba(242, 182, 82, 0.08);
    border-color: rgba(242, 182, 82, 0.25); }}
QLabel[role="chip"][tone="error"] {{ color: #F59A9A; background: rgba(240, 106, 106, 0.10);
    border-color: rgba(240, 106, 106, 0.32); }}
QLabel[role="chip"][tone="live"] {{ color: {NAV_ACTIVE}; background: {ELECTRIC_SOFT}; border-color: rgba(58, 146, 230, 0.30); }}
QLabel[role="titleChip"] {{
    color: {TEXT_2}; font-size: 13px; font-weight: 500; min-height: 26px; max-height: 26px;
    border: 1px solid {LINE}; border-radius: 14px; padding: 0px 12px; background: rgba(16, 43, 76, 0.7);
}}
QLabel[role="titleChip"][tone="warn"] {{ color: {WARN_TEXT}; background: rgba(242, 182, 82, 0.08);
    border-color: rgba(242, 182, 82, 0.25); }}
QLabel[role="titleChip"][tone="error"] {{ color: #F59A9A; background: rgba(240, 106, 106, 0.10);
    border-color: rgba(240, 106, 106, 0.32); }}
QLabel[role="titleChip"][tone="live"] {{ color: {TEXT_1}; background: rgba(22, 119, 210, 0.18);
    border-color: rgba(58, 146, 230, 0.32); }}
QLabel[role="empty"] {{ color: {TEXT_3}; font-size: 13px; }}
QLabel[role="link"] {{ color: {ACCENT_TEXT}; font-weight: 600; }}
QLabel[role="url"] {{ color: {TEXT_3}; font-family: "{mono}"; font-size: 11px; }}

QPushButton {{
    color: {TEXT_1}; background: {SURFACE_3};
    border: 1px solid rgba(160, 200, 245, 0.10); border-radius: 12px;
    padding: 7px 16px; font-family: "{display}"; font-size: 13px; font-weight: 600;
}}
QPushButton:hover {{ border-color: rgba(58, 146, 230, 0.35); }}
QPushButton:pressed {{ background: #11294A; }}
QPushButton:focus {{ border-color: rgba(58, 146, 230, 0.55); }}
QPushButton:disabled {{ color: rgba(163, 182, 206, 0.40); background: rgba(21, 50, 90, 0.45);
    border-color: rgba(160, 200, 245, 0.05); }}
QPushButton[accent="true"] {{ color: #FFFFFF; border-color: transparent;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {ELECTRIC}, stop:1 {ELECTRIC_GRADIENT_END}); }}
QPushButton[accent="true"]:pressed {{ background: #1268B8; }}
QPushButton[accent="true"]:focus {{ border-color: rgba(255, 255, 255, 0.35); }}
QPushButton[accent="true"]:disabled {{ color: rgba(255, 255, 255, 0.45); background: rgba(22, 119, 210, 0.32); }}
QPushButton[ghost="true"] {{ background: transparent; border-color: transparent; color: {TEXT_3}; }}
QPushButton[ghost="true"]:enabled {{ color: {TEXT_2}; }}
QPushButton[ghost="true"]:disabled {{ background: transparent; color: rgba(108, 132, 162, 0.55); }}
QPushButton[recording="true"] {{ color: #FFE1E1; background: rgba(240, 106, 106, 0.22);
    border-color: rgba(240, 106, 106, 0.55); }}
QPushButton[flat="true"] {{ background: transparent; border: 1px solid transparent; border-radius: 8px;
    padding: 3px 8px; color: {ACCENT_TEXT}; font-family: "{text}"; font-size: 12px; font-weight: 500; }}
QPushButton[flat="true"]:hover {{ border-color: transparent; }}
QPushButton[flat="true"]:pressed {{ background: rgba(58, 146, 230, 0.16); }}
QPushButton[flat="true"]:disabled {{ color: {TEXT_3}; background: transparent; }}
QPushButton[danger="true"]:hover {{ color: {ERROR}; }}
QPushButton[chip="true"] {{ background: rgba(16, 43, 76, 0.7); border: 1px solid {LINE}; border-radius: 13px;
    padding: 5px 14px; color: {TEXT_2}; font-family: "{text}"; font-size: 13px; font-weight: 500; }}
QPushButton[chip="true"]:hover {{ color: {TEXT_1}; border-color: rgba(58, 146, 230, 0.30); }}

QFrame#CommandBar QPushButton {{ min-height: 40px; max-height: 40px; padding: 0px 16px; }}
QFrame#CommandBar QPushButton[ghost="true"] {{ padding: 0px 12px; }}

QLineEdit, QAbstractSpinBox {{
    color: {TEXT_1}; background: rgba(5, 11, 22, 0.6); selection-background-color: {ELECTRIC};
    border: 1px solid rgba(160, 200, 245, 0.12); border-radius: 10px; padding: 6px 12px; font-size: 13px;
}}
QLineEdit:hover, QAbstractSpinBox:hover {{ border-color: rgba(58, 146, 230, 0.35); }}
QLineEdit:focus, QAbstractSpinBox:focus {{ border-color: {ELECTRIC}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled {{ color: {TEXT_3}; background: rgba(5, 11, 22, 0.35); }}
QLineEdit#CommandInput {{ border-radius: 14px; padding: 0px 16px; min-height: 44px; max-height: 44px; font-size: 14px; }}
QLineEdit#CommandInput:focus {{ border: 2px solid {ELECTRIC}; padding: 0px 15px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ border: none; background: transparent; width: 18px; }}
QAbstractSpinBox::up-arrow {{ {_image("expand_less", TEXT_2, 14)} width: 12px; height: 12px; }}
QAbstractSpinBox::down-arrow {{ {_image("expand_more", TEXT_2, 14)} width: 12px; height: 12px; }}
QPlainTextEdit {{ color: {TEXT_1}; background: rgba(5, 11, 22, 0.6); border: 1px solid rgba(160, 200, 245, 0.12);
    border-radius: 10px; padding: 4px; selection-background-color: {ELECTRIC}; }}
QPlainTextEdit:focus {{ border-color: {ELECTRIC}; }}
QTextEdit {{ background: transparent; border: none; color: {TEXT_1}; selection-background-color: {ELECTRIC}; }}
QListWidget {{ background: transparent; border: none; outline: 0; }}
QListWidget::item {{ padding: 6px 8px; border-radius: 8px; color: {TEXT_1}; border-bottom: 1px solid rgba(160, 200, 245, 0.05); }}
QListWidget::item:selected {{ background: rgba(22, 119, 210, 0.22); color: {TEXT_1}; }}
QListWidget::item:hover {{ background: rgba(58, 146, 230, 0.08); }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: rgba(160, 200, 245, 0.16); border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(160, 200, 245, 0.28); }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; height: 0; width: 0;
}}
QScrollBar:horizontal {{ height: 0; }}
QCheckBox {{ spacing: 8px; color: {TEXT_2}; font-size: 13px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid rgba(160, 200, 245, 0.30);
    border-radius: 5px; background: rgba(5, 11, 22, 0.6); }}
QCheckBox::indicator:hover {{ border-color: rgba(58, 146, 230, 0.6); }}
QCheckBox::indicator:checked {{ background: {ELECTRIC}; border-color: {ELECTRIC}; {_image("check", "#FFFFFF", 14)} }}
QCheckBox:disabled {{ color: {TEXT_3}; }}
QCheckBox::indicator:disabled {{ border-color: rgba(160, 200, 245, 0.12); }}
QCheckBox::indicator:checked:disabled {{ background: rgba(22, 119, 210, 0.35); border-color: rgba(22, 119, 210, 0.35); }}
QComboBox {{ color: {TEXT_1}; background: rgba(5, 11, 22, 0.6); border: 1px solid rgba(160, 200, 245, 0.12);
    border-radius: 10px; padding: 5px 12px; }}
QComboBox:hover {{ border-color: rgba(58, 146, 230, 0.35); }}
QComboBox:focus {{ border-color: {ELECTRIC}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ {_image("expand_more", TEXT_2, 16)} width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; color: {TEXT_1}; selection-background-color: rgba(22, 119, 210, 0.35);
    border: 1px solid rgba(160, 200, 245, 0.14); outline: 0; padding: 4px; }}
QToolTip {{ color: {TEXT_1}; background: #13305A; border: 1px solid rgba(160, 200, 245, 0.16); padding: 5px 8px; }}
QMessageBox, QDialog {{ background: #0E2442; }}
QMessageBox QLabel, QDialog QLabel {{ color: {TEXT_1}; }}
"""


def menu_stylesheet(accent: str = ACCENT_TEXT) -> str:
    return f"""
QMenu {{ background: {SURFACE_2}; color: {TEXT_1}; border: 1px solid rgba(160, 200, 245, 0.14); padding: 6px 0;
    font-family: "{text_family()}"; font-size: 13px; }}
QMenu::item {{ padding: 7px 24px 7px 16px; }}
QMenu::item:selected {{ background: rgba(22, 119, 210, 0.28); color: {TEXT_1}; }}
QMenu::item:disabled {{ color: {TEXT_3}; }}
QMenu::separator {{ height: 1px; background: rgba(160, 200, 245, 0.10); margin: 5px 10px; }}
QMenu::section {{ color: {accent}; padding: 5px 16px; font-family: "{display_family()}"; font-size: 12px;
    font-weight: 600; }}
"""
