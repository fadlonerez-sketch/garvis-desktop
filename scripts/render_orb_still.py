"""Renders the orb's still frame from the shader: the picture shown where
OpenGL isn't available (app/ui/orb_still.png).

Run after any approved change to app/ui/plasma_orb.frag (needs a real GPU or
Qt's software GL -- not the "offscreen" platform):

    python scripts/render_orb_still.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QGuiApplication  # noqa: E402


def main() -> int:
    app = QGuiApplication(sys.argv)  # noqa: F841 (keeps Qt alive)
    from app.ui import plasma
    from app.ui.state import Mode
    from app.ui.visuals import Dynamics, shader_uniforms

    gl = plasma.renderer()
    if gl is None:
        print("No OpenGL here; run this on a normal Windows desktop session.")
        return 1
    dyn = Dynamics()
    dyn.time = 7.0
    for _ in range(120):  # settle into idle
        dyn.step(1 / 30, Mode.IDLE)
    image = plasma.PlasmaRenderer().render(512, shader_uniforms(dyn, 1.0))
    targets = [ROOT / "app/ui/orb_still.png"]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(target))
        print("wrote", target.relative_to(ROOT), f"({gl.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
