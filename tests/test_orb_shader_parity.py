"""Garvis Desktop and Garvis Mobile draw the same approved orb: the phone ships a
byte-identical copy of app/ui/plasma_orb.frag (it swaps only the GLSL entry point
for AGSL at run time). The two apps stay independent -- nothing is shared at
build time -- so this test is what keeps the copies from drifting apart."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "app" / "ui" / "plasma_orb.frag"
MOBILE = ROOT / "android" / "app" / "src" / "main" / "assets" / "plasma_orb.frag"
MARKER = "// ---- GLSL entry point"


@pytest.mark.skipif(not MOBILE.exists(), reason="Garvis Mobile (android/) is not part of this repository")
def test_mobile_ships_the_desktop_shader_byte_for_byte():
    assert MOBILE.read_bytes() == DESKTOP.read_bytes(), (
        "Copy app/ui/plasma_orb.frag to android/app/src/main/assets/ after an approved change."
    )


def test_the_entry_point_is_the_only_platform_specific_part():
    source = DESKTOP.read_text(encoding="utf-8")
    design, entry = source.split(MARKER)
    assert "gl_Frag" not in design and "vec4 plasma(vec2 fragCoord)" in design
    assert "gl_FragColor = plasma(gl_FragCoord.xy)" in entry
