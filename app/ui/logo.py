"""The Garvis logo: the original warm amber orb, frozen as a still image.

This is the logo the owner keeps unchanged -- the dashboard's title-bar mark,
the window icon and the tray icon. It is NOT the live orb (app/ui/visuals.py,
the approved blue plasma); the two were one drawing until 2026-10-31 and were
separated so the live orb could change while the logo stays exactly as it was.
The code below is the pre-2026-10-31 orb painter, moved here verbatim, and
tests/test_command_center.py pins its output pixel for pixel.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPixmap, QRadialGradient, QTransform

from app.ui.theme import Accent
from app.ui.visuals import Dynamics

# Palette.
WHITE_HOT = "#FFF4D6"
GOLD = "#FFC857"
AMBER = "#FF9A2E"
ORANGE = "#F2700F"
EMBER = "#B8410A"
DEEP = "#5A1A04"
GLOW = "#FF8A2A"
ALERT_RED = "#D8381C"

HALO_REACH = 1.75
"""How far the glow reaches, in sphere radii."""
TEXTURE_SIZE = 128
TEXTURE_SEED = 7


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
        a.alphaF() + (b.alphaF() - a.alphaF()) * t,
    )


def _alpha(color: QColor, alpha: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(max(0.0, min(1.0, alpha)))
    return c


# --------------------------------------------------------------------------
# Plasma texture
# --------------------------------------------------------------------------
def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Lattice value in [0, 1): 32-bit integer hash, wrap-around arithmetic."""
    seeded = np.uint32((seed * 1274126177) & 0xFFFFFFFF)
    h = ix.astype(np.uint32) * np.uint32(374761393) + iy.astype(np.uint32) * np.uint32(668265263) + seeded
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    h = h ^ (h >> np.uint32(16))
    return h.astype(np.float64) / 4294967296.0


def _value_noise(n: int, period: int, seed: int) -> np.ndarray:
    """Smooth value noise over an n x n tile with `period` cells per side;
    wraps seamlessly at the tile's edges."""
    coords = np.arange(n) * period / n
    cell = np.floor(coords).astype(np.int64)
    f = coords - cell
    u = f * f * (3.0 - 2.0 * f)
    i0, i1 = cell % period, (cell + 1) % period
    x0, y0 = np.meshgrid(i0, i0)
    x1, y1 = np.meshgrid(i1, i1)
    ux, uy = np.meshgrid(u, u)
    top = _hash(x0, y0, seed) + (_hash(x1, y0, seed) - _hash(x0, y0, seed)) * ux
    bottom = _hash(x0, y1, seed) + (_hash(x1, y1, seed) - _hash(x0, y1, seed)) * ux
    return top + (bottom - top) * uy


def plasma_intensity(n: int = TEXTURE_SIZE, seed: int = TEXTURE_SEED) -> np.ndarray:
    """(n, n) intensities in [0, 1]: soft clouds plus thin bright ridges."""
    total = np.zeros((n, n))
    weight = 0.0
    for octave, period in enumerate((4, 8, 16, 32)):
        amplitude = 0.5 ** octave
        total += amplitude * _value_noise(n, period, seed + octave)
        weight += amplitude
    fbm = total / weight
    fbm = (fbm - fbm.min()) / max(fbm.max() - fbm.min(), 1e-9)
    ridge = 1.0 - np.abs(2.0 * fbm - 1.0)
    return np.clip(0.45 * ridge ** 3 + 0.6 * fbm ** 2, 0.0, 1.0)


@lru_cache(maxsize=1)
def _plasma_texture() -> QImage:
    # A QImage, not a QPixmap: this cache may outlive the QApplication.
    intensity = plasma_intensity()
    # Orange to gold: added onto the body, it warms and brightens without washing toward yellow-green.
    low, high = np.array(QColor(ORANGE).getRgbF()[:3]), np.array(QColor(GOLD).getRgbF()[:3])
    rgb = low[None, None, :] + (high - low)[None, None, :] * intensity[..., None]
    alpha = intensity * 0.85
    premultiplied = rgb * alpha[..., None]
    bgra = np.empty((*intensity.shape, 4), dtype=np.uint8)
    bgra[..., 0] = np.round(premultiplied[..., 2] * 255)
    bgra[..., 1] = np.round(premultiplied[..., 1] * 255)
    bgra[..., 2] = np.round(premultiplied[..., 0] * 255)
    bgra[..., 3] = np.round(alpha * 255)
    size = intensity.shape[0]
    return QImage(bgra.tobytes(), size, size, 4 * size, QImage.Format.Format_ARGB32_Premultiplied).copy()


# --------------------------------------------------------------------------
# Painting
# --------------------------------------------------------------------------
class _Palette:
    """The warm palette, shifted toward red for an alert and dimmed in standby."""

    def __init__(self, alert: float, standby: float) -> None:
        red = QColor(ALERT_RED)
        dark = QColor(DEEP)

        def tone(hex_color: str, alert_share: float = 0.45) -> QColor:
            return _mix(_mix(QColor(hex_color), red, alert * alert_share), dark, standby * 0.25)

        self.white_hot = tone(WHITE_HOT, 0.1)
        self.gold = tone(GOLD)
        self.amber = tone(AMBER)
        self.orange = tone(ORANGE)
        self.ember = tone(EMBER, 0.3)
        self.deep = QColor(DEEP)
        self.glow = tone(GLOW)


def paint_logo(painter: QPainter, center: QPointF, radius: float, dyn: Dynamics, accent: Accent | None = None,
               compact: bool = False) -> None:
    """Draw the logo at `center`; `radius` is the sphere's base radius (its
    glow reaches HALO_REACH times that). `accent` is accepted for the callers'
    theme plumbing but the orb is always warm: it is Garvis's identity, not a
    theme colour. `compact` (icons, the floating orb) uses one plasma layer."""
    look = dyn.appearance()
    pal = _Palette(look["alert"], look["standby"])
    r = radius * look["swell"]
    glow, heat = look["glow"], look["heat"]
    cx, cy = center.x(), center.y()

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.setPen(Qt.PenStyle.NoPen)

    # 1. The glow, fading into the dark.
    halo_r = r * HALO_REACH * (1.0 + 0.05 * glow)
    edge = r / halo_r
    halo = QRadialGradient(center, halo_r)
    # Roughly Gaussian falloff beyond the sphere's edge: soft, never a ring.
    halo.setColorAt(0.0, _alpha(pal.glow, 0.45 * glow))
    halo.setColorAt(edge * 0.97, _alpha(pal.glow, 0.36 * glow))
    # The last stop eases the tail in, so no edge shows where the glow meets black.
    for share, strength in ((0.12, 0.26), (0.28, 0.15), (0.48, 0.065), (0.72, 0.02), (0.87, 0.005)):
        halo.setColorAt(edge + (1 - edge) * share, _alpha(pal.glow, strength * glow))
    halo.setColorAt(1.0, _alpha(pal.glow, 0.0))
    painter.setBrush(QBrush(halo))
    painter.drawEllipse(center, halo_r, halo_r)

    # 2. The body: lit from inside and above, deepening toward the rim.
    body = QRadialGradient(center, r, QPointF(cx - 0.18 * r, cy - 0.30 * r))
    body.setColorAt(0.00, _mix(pal.gold, pal.white_hot, 0.35 + 0.55 * heat))
    body.setColorAt(0.13, pal.gold)
    body.setColorAt(0.36, pal.amber)
    body.setColorAt(0.60, pal.orange)
    body.setColorAt(0.84, pal.ember)
    body.setColorAt(1.00, _mix(pal.ember, pal.deep, 0.6))
    painter.setBrush(QBrush(body))
    painter.drawEllipse(center, r, r)

    sphere = QPainterPath()
    sphere.addEllipse(center, r, r)
    painter.setClipPath(sphere)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

    # 3. Molten light flowing inside: the plasma texture, drifting.
    texture = _plasma_texture()
    d = dyn.drift
    sway = math.degrees(dyn.swirl) + (0.0 if dyn.reduced_motion else 6.0 * math.sin(dyn.time * 0.07))
    layers = [(1.6, 40.0 * d, -25.0 * d, sway, 0.70)]
    if not compact:
        layers.append((1.05, -55.0 * d, 31.0 * d, -0.6 * sway + 40.0, 0.45))
    for span, ox, oy, angle, share in layers:
        k = 2.0 * span * r / TEXTURE_SIZE
        transform = QTransform()
        transform.translate(cx, cy)
        transform.rotate(angle)
        transform.scale(k, k)
        transform.translate(ox, oy)
        brush = QBrush(texture)
        brush.setTransform(transform)
        painter.setBrush(brush)
        painter.setOpacity(look["plasma"] * share)
        painter.drawEllipse(center, r, r)
    painter.setOpacity(1.0)

    # 4. The heart: the hottest light, a little above centre.
    heart_c = QPointF(cx, cy - 0.08 * r)
    heart = QRadialGradient(heart_c, 0.62 * r)
    heart.setColorAt(0.0, _alpha(pal.white_hot, 0.45 * heat))
    heart.setColorAt(0.45, _alpha(pal.gold, 0.18 * heat))
    heart.setColorAt(1.0, _alpha(pal.gold, 0.0))
    painter.setBrush(QBrush(heart))
    painter.drawEllipse(heart_c, 0.62 * r, 0.62 * r)

    # 5. Limb darkening: rich, dark edges give the sphere its depth.
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    limb = QRadialGradient(center, r)
    limb.setColorAt(0.0, _alpha(pal.deep, 0.0))
    limb.setColorAt(0.50, _alpha(pal.deep, 0.0))
    limb.setColorAt(0.72, _alpha(pal.deep, 0.14))
    limb.setColorAt(0.87, _alpha(pal.deep, 0.36))
    limb.setColorAt(0.95, _alpha(pal.deep, 0.52))
    limb.setColorAt(1.0, _alpha(pal.deep, 0.30))  # eases off at the very edge so the glow takes over, no outline
    painter.setBrush(QBrush(limb))
    painter.drawEllipse(center, r, r)

    # 6. A faint warm rim, as light wraps round the edge.
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    rim = QRadialGradient(center, r)
    rim.setColorAt(0.86, _alpha(pal.amber, 0.0))
    rim.setColorAt(0.96, _alpha(pal.amber, 0.05 + 0.10 * glow))
    rim.setColorAt(1.0, _alpha(pal.amber, 0.0))
    painter.setBrush(QBrush(rim))
    painter.drawEllipse(center, r, r)

    # 7. A soft highlight up and to the left.
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    spot_c = QPointF(cx - 0.28 * r, cy - 0.42 * r)
    spot = QRadialGradient(spot_c, 0.34 * r)
    spot.setColorAt(0.0, _alpha(pal.white_hot, 0.30 + 0.10 * heat))
    spot.setColorAt(1.0, _alpha(pal.white_hot, 0.0))
    painter.setBrush(QBrush(spot))
    painter.drawEllipse(spot_c, 0.34 * r, 0.34 * r)

    painter.restore()


def render_logo_pixmap(size: int, accent: Accent | None = None, device_pixel_ratio: float = 1.0) -> QPixmap:
    """The logo at `size` px (window/tray icons, title bar)."""
    pixmap = QPixmap(int(size * device_pixel_ratio), int(size * device_pixel_ratio))
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    dyn = Dynamics(reduced_motion=True)
    dyn.values["glow"] = 0.75  # icons sit on bright taskbars too: a little more presence
    paint_logo(painter, QPointF(size / 2.0, size / 2.0), size * 0.27, dyn, accent, compact=True)
    painter.end()
    return pixmap
