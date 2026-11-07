"""The command-center redesign: the Garvis orb (app/ui/visuals.py, the
approved blue plasma since 2026-10-31) and the logo (app/ui/logo.py), the
reorganized pages (app/ui/dashboard.py), telemetry demand in MainWindow, the
Tools catalog (AgentBridge.tool_catalog) and the usage trend.

Offscreen Qt against the fake bridge from tests/test_ui_dashboard.py and a
fake telemetry sampler: no Agent, LLM, microphone or real input. What the
core looks like and how the window behaves on a real desktop is in the
manual checklist, not here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from pydantic import BaseModel
from PySide6.QtCore import QEvent, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import app.ui.main_window as main_window
from app.llm.usage import UsageLedger
from app.security.permissions import PermissionChecker
from app.system.telemetry import Disk, GpuAdapter, Snapshot, StaticInfo
from app.tools.base import RiskLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry
from app.ui import dashboard, plasma, theme
from app.ui.agent_bridge import AgentBridge, _gmail_status
from app.ui.logo import HALO_REACH, paint_logo, plasma_intensity, render_logo_pixmap
from app.ui.state import Mode
from app.ui.telemetry import DETAIL, OFF, SUMMARY, GiB, TelemetryMonitor
from app.ui.usage_view import read_usage_summary
from app.ui.visuals import (
    ACTIVE_INTERVAL_MS,
    IDLE_INTERVAL_MS,
    STANDBY_INTERVAL_MS,
    CoreVisual,
    Dynamics,
    paint_orb,
    shader_uniforms,
)
from tests.test_ui_dashboard import _FakeBridge, _research_turn

CATALOG = [
    {"name": "web_research", "description": "Research a question on the web. Opens pages.", "risk": "low",
     "confirm": False, "conditional": False},
    {"name": "gmail_send", "description": "Send an email.", "risk": "high", "confirm": True, "conditional": False},
    {"name": "gmail_search", "description": "Search Gmail.", "risk": "low", "confirm": False, "conditional": False},
    {"name": "run_powershell", "description": "Run PowerShell.", "risk": "medium", "confirm": False,
     "conditional": True},
    {"name": "calculator", "description": "Do arithmetic.", "risk": "low", "confirm": False, "conditional": False},
    {"name": "brand_new_tool", "description": "Something new.", "risk": "low", "confirm": False,
     "conditional": False},
]


class _CatalogBridge(_FakeBridge):
    def tool_catalog(self) -> list[dict]:
        return CATALOG


class _StillSampler:
    """Never touches the OS: the window tests only check demand switching."""

    def static_info(self) -> StaticInfo:
        return StaticInfo()

    def sample(self, detail: bool = False) -> Snapshot:
        return Snapshot(at=time.time(), detail=detail)

    def close(self) -> None:
        pass


class _QuietMonitor(TelemetryMonitor):
    def __init__(self, parent=None) -> None:
        super().__init__(sampler_factory=_StillSampler, parent=parent)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "AgentBridge", _CatalogBridge)
    monkeypatch.setattr(main_window, "TelemetryMonitor", _QuietMonitor)
    win = main_window.MainWindow()
    yield win
    win.close()
    win.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _snapshot(detail: bool = True, **overrides) -> Snapshot:
    values = dict(at=time.time(), detail=detail, cpu_percent=37.0, ram_total=32 * GiB, ram_available=8 * GiB,
                  process_cpu_percent=1.5, process_memory=300 * 1024 ** 2, process_private=280 * 1024 ** 2,
                  uptime_seconds=93_784.0)
    if detail:
        values.update(per_core=(10.0, 95.0, 30.0, 40.0), cpu_mhz=4200.0,
                      disks=(Disk("C:", total=1000 * GiB, free=250 * GiB),), disk_read_bps=2.5 * 1024 ** 2,
                      disk_write_bps=0.0, gpu_percent=12.0, gpu_memory_used=0, gpu_shared_used=GiB)
    values.update(overrides)
    return Snapshot(**values)


def _feed(window, *snaps, static: StaticInfo | None = None) -> None:
    monitor = window.telemetry
    if static is not None:
        monitor._on_static(static)
    for snap in snaps:
        monitor._on_sampled(snap)


# --------------------------------------------------------------- the logo
# The logo is the original warm orb, frozen (app/ui/logo.py). It must never change:
# these tests pin it, including pixel for pixel against images saved before the
# live orb was replaced (tests/golden/).
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).with_name("golden")


def _render_logo(dyn: Dynamics, size: int = 420, ratio: float = 0.27, compact: bool = False) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    paint_logo(painter, QPointF(size / 2, size / 2), size * ratio, dyn, theme.BLUE, compact=compact)
    painter.end()
    return image


def _settled(mode: Mode, level: float = 0.0, standby: bool = False, frames: int = 90, **kw) -> Dynamics:
    dyn = Dynamics(**kw)
    for _ in range(frames):
        dyn.step(0.033, mode, level, standby)
    return dyn


def _luma(color: QColor) -> float:
    return 0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()


@pytest.mark.parametrize(("size", "ratio", "golden"), [(64, 1.0, "logo_64.png"), (34, 2.0, "logo_34_dpr2.png")])
def test_the_logo_is_pixel_identical_to_the_original(qapp, size, ratio, golden):
    image = render_logo_pixmap(size, theme.ORANGE, ratio).toImage()
    expected = QImage(str(GOLDEN / golden)).convertToFormat(image.format())
    assert image == expected


def test_the_logo_is_a_warm_sphere_whatever_the_theme_accent(qapp):
    image = _render_logo(_settled(Mode.IDLE))  # painted with the dashboard's BLUE accent
    for x, y in ((210, 210), (210, 170), (250, 240)):
        c = image.pixelColor(x, y)
        assert c.red() > 200 and c.red() > c.green() > c.blue()  # amber/gold, never blue


def test_the_logo_glow_fades_smoothly_outward_with_no_rings(qapp):
    size, ratio = 420, 0.27
    image = _render_logo(_settled(Mode.IDLE), size, ratio)
    r = size * ratio
    alphas = [image.pixelColor(int(size / 2 + r * k), size // 2).alpha() for k in
              [1.04 + 0.02 * i for i in range(int((HALO_REACH - 1.04) / 0.02))]]
    assert alphas[0] > 0
    assert all(b <= a + 1 for a, b in pairwise(alphas))  # monotonic: no rings, ticks or bands
    assert image.pixelColor(2, 2).alpha() == 0 and image.pixelColor(size // 2, 1).alpha() == 0


def test_the_logo_is_lit_from_above_with_darker_richer_edges(qapp):
    size, r = 420, 420 * 0.27
    image = _render_logo(_settled(Mode.IDLE), size)
    c = size / 2
    upper = _luma(image.pixelColor(int(c - 0.15 * r), int(c - 0.3 * r)))
    lower_edge = _luma(image.pixelColor(int(c), int(c + 0.9 * r)))
    side_edge = _luma(image.pixelColor(int(c + 0.92 * r), int(c)))
    assert upper > lower_edge + 0.15 and upper > side_edge + 0.15


def test_logo_icon_is_rendered(qapp):
    pixmap = render_logo_pixmap(64, theme.ORANGE)
    assert not pixmap.isNull() and pixmap.toImage().pixelColor(32, 32).alpha() > 200


def test_logo_texture_is_deterministic_and_tiles_seamlessly():
    tex = plasma_intensity()
    assert tex.shape == (128, 128) and 0.0 <= tex.min() and tex.max() <= 1.0
    assert (tex == plasma_intensity()).all()
    inner = max(abs(tex[:, 1:] - tex[:, :-1]).max(), abs(tex[1:, :] - tex[:-1, :]).max())
    assert abs(tex[:, 0] - tex[:, -1]).max() <= inner + 1e-9  # the wrap is no bigger a step than any other
    assert abs(tex[0, :] - tex[-1, :]).max() <= inner + 1e-9


def test_the_title_bar_window_and_tray_use_the_logo_not_the_live_orb():
    for module in ("main_window", "presence", "dashboard"):
        source = (ROOT / "app" / "ui" / f"{module}.py").read_text(encoding="utf-8")
        assert "render_logo_pixmap" in source and "render_orb_pixmap" not in source


# --------------------------------------------------------------- the orb
# How the plasma orb looks is plasma_orb.frag, approved by the owner and checked on a
# real GPU below (in a child process) and by eye. Offscreen Qt has no OpenGL, so the
# in-process tests exercise the still-frame path and the state model.
def _render_orb(dyn: Dynamics, size: int = 300, compact: bool = False) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    paint_orb(painter, QRectF(0, 0, size, size), dyn, compact=compact)
    painter.end()
    return image


def test_without_opengl_the_orb_draws_its_still_frame(qapp):
    assert plasma.renderer() is None  # the offscreen test platform has no OpenGL
    still = plasma.still_frame()
    assert not still.isNull() and still.width() == still.height() == 512
    image = _render_orb(_settled(Mode.IDLE))
    centre = image.pixelColor(150, 150)
    assert centre.alpha() > 200 and centre.blue() > centre.red()  # the blue plasma, not the amber logo
    assert image.pixelColor(1, 1).alpha() <= 2  # (near-)transparent corners: it composites onto the UI


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("compact", [False, True])
def test_every_state_renders(qapp, mode, compact):
    size = 124 if compact else 420
    image = _render_orb(_settled(mode, 2500.0, standby=mode == Mode.IDLE, frames=40), size, compact)
    assert image.pixelColor(size // 2, size // 2).alpha() > 60  # the orb is drawn (dimmed with the state)
    assert image.pixelColor(1, 1).alpha() <= 2  # and fades out inside its widget


def test_quality_adapts_to_the_display_size():
    assert plasma.detail_for(560, software=False) == 1.0 and plasma.detail_for(256, software=False) == 1.0
    assert plasma.detail_for(160, software=False) == 0.0  # the approved compact level for small orbs
    assert plasma.detail_for(560, software=False, compact=True) == 0.0
    assert plasma.render_size(560, software=False) == 560
    assert plasma.render_size(2400, software=False) == plasma.MAX_RENDER_PX  # scaled up past the cap
    # No GPU driver: the approved CPU fallback, 160 px compact.
    assert plasma.render_size(560, software=True) == 160 and plasma.detail_for(560, software=True) == 0.0


def test_shader_inputs_follow_the_state_model():
    idle, thinking = shader_uniforms(_settled(Mode.IDLE)), shader_uniforms(_settled(Mode.THINKING))
    assert set(idle) == set(plasma.UNIFORMS)
    assert thinking["u_drift"] > idle["u_drift"] and thinking["u_swirl"] > 0 == idle["u_swirl"]
    assert shader_uniforms(_settled(Mode.ERROR))["u_alert"] > 0.9
    assert shader_uniforms(_settled(Mode.IDLE, standby=True))["u_standby"] > 0.9
    assert shader_uniforms(_settled(Mode.SPEAKING, 3000.0))["u_heat"] > idle["u_heat"]
    still = shader_uniforms(_settled(Mode.THINKING, reduced_motion=True))
    assert still["u_time"] == 0 and still["u_drift"] == 0 and still["u_swirl"] == 0  # reduced motion holds still


_GL_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
from PySide6.QtGui import QGuiApplication
app = QGuiApplication([])
from app.ui import plasma
from app.ui.state import Mode
from app.ui.visuals import Dynamics, shader_uniforms
gl = plasma.renderer()
if gl is None:
    print("NOGL")
    raise SystemExit(0)
dyn = Dynamics()
for _ in range(60):
    dyn.step(1 / 30, Mode.LISTENING)
full = gl.render(256, shader_uniforms(dyn, 1.0))
small = gl.render(96, shader_uniforms(dyn, 0.0))
c, corner = full.pixelColor(128, 128), full.pixelColor(1, 1)
print("OK", c.red(), c.green(), c.blue(), c.alpha(), corner.alpha(), small.pixelColor(48, 48).alpha())
"""


@pytest.mark.skipif(os.name != "nt", reason="Windows GL probe")
def test_the_shader_compiles_and_renders_on_real_opengl():
    """A child process on Qt's normal Windows platform (an offscreen GL surface; no
    window is ever shown): the real shader compiles, links and draws the orb."""
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    result = subprocess.run([sys.executable, "-c", _GL_PROBE, str(ROOT)], capture_output=True, text=True,
                            timeout=120, env=env)
    lines = result.stdout.strip().splitlines()
    out = lines[-1] if lines else ""
    if out == "NOGL":
        pytest.skip("no OpenGL on this machine")
    assert result.returncode == 0 and out.startswith("OK"), result.stderr[-2000:]
    r, g, b, a, corner, small = map(int, out.split()[1:])
    assert a > 200 and b >= g > r  # a bright core in the Electric Blue -> cyan -> white family
    assert corner <= 2 and small > 150  # the Midnight haze fades to (almost) nothing at the corners


def test_states_differ_in_brightness_without_jumps():
    idle, listening = _settled(Mode.IDLE).appearance(), _settled(Mode.LISTENING).appearance()
    standby = _settled(Mode.IDLE, standby=True).appearance()
    speaking = _settled(Mode.SPEAKING, 3000.0).appearance()
    assert standby["glow"] < idle["glow"] < listening["glow"] < speaking["glow"]
    dyn = Dynamics()
    previous = dyn.appearance()
    for mode, frames in ((Mode.LISTENING, 30), (Mode.THINKING, 60), (Mode.SPEAKING, 60), (Mode.LISTENING, 40),
                         (Mode.ERROR, 40), (Mode.IDLE, 60)):
        for _ in range(frames):
            dyn.step(0.033, mode, 3000.0)
            look = dyn.appearance()
            assert abs(look["glow"] - previous["glow"]) < 0.1  # no flicker or sudden flare
            assert abs(look["swell"] - previous["swell"]) < 0.01
            previous = look


def test_standby_is_a_calmer_slower_idle_and_only_while_idle():
    dyn = _settled(Mode.IDLE, standby=True, frames=120)
    idle = _settled(Mode.IDLE, frames=120)
    assert dyn.values["standby"] > 0.9
    assert dyn.values["flow"] < idle.values["flow"] and dyn.appearance()["glow"] < idle.appearance()["glow"]
    for _ in range(120):
        dyn.step(0.033, Mode.THINKING, standby=True)  # working wins over standby
    assert dyn.values["standby"] < 0.1


def test_thinking_stirs_the_plasma_faster_than_idle():
    idle, thinking = _settled(Mode.IDLE), _settled(Mode.THINKING)
    assert thinking.drift > 2 * idle.drift and thinking.swirl > 0 == idle.swirl


def test_speech_stopping_settles_the_glow_then_listening_takes_over():
    dyn = _settled(Mode.SPEAKING, 3000.0, frames=30)
    speaking = dyn.appearance()["glow"]
    dyn.step(0.033, Mode.LISTENING)  # interrupted: playback stopped, the mic is live
    for _ in range(4):
        dyn.step(0.033, Mode.LISTENING)
    assert dyn.hush > 0.5 and dyn.appearance()["glow"] < speaking - 0.2  # a clear, quick settle
    for _ in range(60):
        dyn.step(0.033, Mode.LISTENING)
    assert dyn.hush < 0.05 and dyn.appearance()["glow"] == pytest.approx(_settled(Mode.LISTENING).appearance()["glow"],
                                                                        abs=0.03)


def test_reduced_motion_holds_the_plasma_still_but_keeps_the_state_visible():
    still = _settled(Mode.THINKING, reduced_motion=True)
    assert still.drift == 0 and still.swirl == 0
    assert _settled(Mode.IDLE, reduced_motion=True).appearance()["swell"] == pytest.approx(1.0)
    loud = _settled(Mode.SPEAKING, 3000.0, reduced_motion=True).appearance()
    assert loud["glow"] > _settled(Mode.IDLE, reduced_motion=True).appearance()["glow"]


def test_core_stops_its_timer_at_rest_with_reduced_motion(qapp):
    core = CoreVisual(theme.BLUE, reduced_motion=True)
    core.resize(300, 300)
    core.show()
    QTest.qWait(10)
    assert not core.is_animating
    core.set_mode(Mode.THINKING)
    assert core.is_animating
    core.hide()
    core.deleteLater()


def test_core_paces_its_timer_by_state(qapp):
    core = CoreVisual(theme.BLUE, reduced_motion=False)
    core.resize(300, 300)
    core.show()
    QTest.qWait(10)
    assert core.timer_interval == IDLE_INTERVAL_MS
    core.set_standby(True)
    assert core.timer_interval == STANDBY_INTERVAL_MS
    core.set_mode(Mode.SPEAKING)
    assert core.timer_interval == ACTIVE_INTERVAL_MS
    core.hide()
    assert not core.is_animating
    core.deleteLater()


def test_dashboard_and_orb_follow_standby(window):
    window._bridge.standby_changed.emit(True)
    assert window._overview.core.standby
    window._overview.flush()
    assert window._overview.mode_label.text() == "Standby"  # approved redesign: sentence case, no all-caps
    assert "Hey Garvis" in window._overview.mode_detail.text()
    window._bridge.standby_changed.emit(False)
    assert not window._overview.core.standby


# ------------------------------------------------------------ structure
def test_pages_and_navigation(window):
    assert dashboard.PAGES == ("overview", "activity", "protocols", "system", "sources", "voice", "usage", "tools",
                               "settings")
    assert set(window._nav.buttons) == set(dashboard.PAGES)
    for key in dashboard.PAGES:
        window.show_page(key)
        assert window.current_page == key


def test_tool_log_lives_on_activity_not_on_the_overview(window):
    assert not window._overview.isAncestorOf(window._activity_list)
    assert window._activity_page.isAncestorOf(window._activity_list)
    assert window._overview.isAncestorOf(window._conversation_view)


def test_overview_links_open_their_pages(window):
    window.show()
    for key in ("activity", "sources"):
        window._overview.quick_links[key].click()
        assert window.current_page == key
    window.show_page("overview")
    window._overview.system_link.click()
    assert window.current_page == "system"
    window.show_page("overview")
    window._overview.usage_link.click()
    assert window.current_page == "usage"


def test_activity_page_summarizes_and_numbers_the_steps(window):
    _research_turn(window._bridge)
    window._bridge.user_message.emit("second")
    window._bridge.tool_started.emit("calculator", {"expression": "1/0"})
    window._bridge.tool_finished.emit("calculator", False, "division by zero")
    window._bridge.status_changed.emit("Done.")
    window.show()
    window.show_page("activity")
    page = window._activity_page
    page.flush()
    assert page.stat_requests.value_label.text() == "2"
    assert page.stat_steps.value_label.text() == "2"
    assert page.stat_failed.value_label.text() == "1"
    assert page.stat_sources.value_label.text() == "2"


# ------------------------------------------------------------ telemetry
def test_sampling_follows_what_is_on_screen(window):
    monitor = window.telemetry
    assert monitor.demand == OFF  # never shown: nothing sampled
    window.show()
    assert monitor.demand == SUMMARY
    window.show_page("system")
    assert monitor.demand == DETAIL
    window.show_page("sources")
    assert monitor.demand == SUMMARY
    window.hide()  # collapsed to the orb
    assert monitor.demand == OFF
    window.show()
    window.close()
    assert not monitor.running


def test_overview_shows_real_resources_and_says_so_when_waiting(window):
    window.show()
    overview = window._overview
    overview._render_resources()
    assert overview.cpu_line.value_label.text() == "—"
    assert "Waiting" in overview.process_label.text()
    _feed(window, _snapshot(detail=False))
    assert overview.cpu_line.value_label.text() == "37%"
    assert overview.ram_line.value_label.text() == "75%"
    assert overview.ram_line.detail_label.text() == "24.0 GB / 32.0 GB"
    assert "300 MB" in overview.process_label.text()


def test_system_page_shows_measurements_and_marks_the_unavailable(window):
    window.show()
    window.show_page("system")
    page = window._system_page
    page.flush()
    assert page.cpu_gauge.value is None
    assert page.gpu_info.value("Adapter") == "Unavailable"
    static = StaticInfo(cpu_model="Test CPU 9000", logical_cpus=4, base_mhz=3000, total_ram=32 * GiB,
                        gpus=(GpuAdapter("Test GPU", 2 * GiB),), os_name="Windows 11 Pro (build 26200)")
    _feed(window, _snapshot(), static=static)
    page.flush()
    assert page.cpu_gauge.value == pytest.approx(37.0)
    assert page.ram_gauge.value == pytest.approx(75.0)
    assert page.core_grid.loads == (10.0, 95.0, 30.0, 40.0)
    assert page.cpu_info.value("Model") == "Test CPU 9000"
    assert "estimate" in page.cpu_info.value("Speed")  # derived figures say so
    assert page.ram_info.value("In use") == "24.0 GB"
    assert "C:" in page._disk_widgets
    assert page.disk_read.value_label.text() == "2.5 MB/s"
    assert page.gpu_info.value("Shared memory") == "1.0 GB in use"
    assert page.gpu_info.value("Temperature").startswith("Not exposed")
    assert page.machine_info.value("Uptime") == "1 d 2 h"
    assert page.live_chip.text().startswith("● Live")  # approved redesign: sentence-case chips


def test_resource_warning_reaches_the_overview_alerts(window):
    window.show()
    now = time.time()
    _feed(window, *[_snapshot(detail=False, at=now - 3 + i, cpu_percent=98.0) for i in range(4)])
    overview = window._overview
    overview.flush()
    texts = [overview.alerts_body.itemAt(i).layout().itemAt(1).widget().text()
             for i in range(overview.alerts_body.count()) if overview.alerts_body.itemAt(i).layout()]
    assert any("CPU has stayed above 90%" in text for text in texts)


# ------------------------------------------------------------------ tools
def test_tools_page_groups_the_real_catalog_and_filters(window):
    window.show()
    window.show_page("tools")
    page = window._tools_page
    page.flush()
    assert sorted(page.tool_names) == sorted(t["name"] for t in CATALOG)
    assert dashboard.tool_group("gmail_send") == "mail"
    assert dashboard.tool_group("brand_new_tool") == "other"  # nothing is dropped
    page.search.setText("gmail")
    visible = [row.tool_name for _p, rows in page._rows for row in rows
               if isinstance(row, dashboard.ToolRow) and not row.isHidden()]
    assert sorted(visible) == ["gmail_search", "gmail_send"]
    page.search.setText("no such capability")
    assert not page.no_match.isHidden()
    page.search.setText("")
    assert page.no_match.isHidden()


def test_tools_page_without_a_catalog_says_so(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "AgentBridge", _FakeBridge)
    monkeypatch.setattr(main_window, "TelemetryMonitor", _QuietMonitor)
    win = main_window.MainWindow()
    win._tools_page.flush()
    assert win._tools_page.tool_names == []
    win.close()


def test_integration_status_comes_only_from_the_configuration():
    rows = {r["name"]: r for r in dashboard.integration_rows({
        "model": "claude-haiku-5-5", "api_key_configured": False, "desktop_control_enabled": False,
        "mobile_enabled": True, "gmail_status": "not_configured", "budget_enabled": True,
        "daily_budget_usd": 0.5, "max_cost_per_task_usd": 0.05,
    })}
    assert rows["claude-haiku-5-5 · Anthropic API"]["status"] == "API key missing"
    assert rows["Desktop control"]["status"] == "Disabled" and rows["Desktop control"]["action"] is None
    assert rows["iPhone interface"]["action"] == ("Pair iPhone…", "pair")
    assert rows["Gmail"]["status"] == "Not configured"
    assert "Local speech" not in rows  # no info about it: not claimed
    signed_in = dashboard.integration_rows({"gmail_status": "signed_in"})
    assert next(r for r in signed_in if r["name"] == "Gmail")["status"] == "Signed in"


def test_gmail_status_reads_only_the_files(tmp_path):
    settings = SimpleNamespace(gmail_token_path=tmp_path / "token.json",
                               gmail_client_secret_path=tmp_path / "client.json")
    assert _gmail_status(settings) == "not_configured"
    settings.gmail_client_secret_path.write_text("{}")
    assert _gmail_status(settings) == "sign_in_needed"
    settings.gmail_token_path.write_text("{}")
    assert _gmail_status(settings) == "signed_in"


class _Args(BaseModel):
    pass


class _LowTool(Tool):
    name = "low_tool"
    description = "Reads something. More detail here."
    args_model = _Args

    async def execute(self, args) -> ToolResult:
        return ToolResult.ok()


class _HighTool(_LowTool):
    name = "high_tool"
    risk_level = RiskLevel.HIGH


class _MediumConfirmTool(_LowTool):
    name = "medium_confirm_tool"
    risk_level = RiskLevel.MEDIUM
    requires_confirmation = True


class _ConditionalTool(_LowTool):
    name = "conditional_tool"
    risk_level = RiskLevel.MEDIUM

    async def confirmation_reason(self, args) -> str | None:
        return None


def test_tool_catalog_reflects_the_agents_own_permission_policy():
    registry = ToolRegistry()
    for tool in (_LowTool(), _HighTool(), _MediumConfirmTool(), _ConditionalTool()):
        registry.register(tool)
    fake = SimpleNamespace(_agent=SimpleNamespace(_registry=registry, _permissions=PermissionChecker()))
    catalog = {entry["name"]: entry for entry in AgentBridge.tool_catalog(fake)}
    assert catalog["low_tool"]["confirm"] is False and catalog["low_tool"]["conditional"] is False
    assert catalog["high_tool"]["confirm"] is True and catalog["high_tool"]["risk"] == "high"
    assert catalog["medium_confirm_tool"]["confirm"] is True
    assert catalog["conditional_tool"]["confirm"] is False and catalog["conditional_tool"]["conditional"] is True
    assert dashboard.first_sentence(catalog["low_tool"]["description"]) == "Reads something."


# ------------------------------------------------------------------ usage
def test_usage_summary_has_a_daily_series_with_real_zeros(tmp_path):
    db = tmp_path / "usage.db"
    ledger = UsageLedger(db)
    now = datetime(2026, 10, 23, 15, 0, 0)
    ledger.record(status="ok", model_requested="claude-haiku-5-5", cost_usd=0.02, now=now)
    ledger.record(status="ok", model_requested="claude-haiku-5-5", cost_usd=0.03, now=datetime(2026, 10, 20, 9))
    ledger.record(status="ok", model_requested="claude-haiku-5-5", cost_usd=9.0, now=datetime(2026, 9, 1, 9))
    daily = read_usage_summary(db, now=now)["daily"]
    assert len(daily) == 14 and daily[0]["day"] == "2026-10-10" and daily[-1]["day"] == "2026-10-23"
    by_day = {d["day"]: d for d in daily}
    assert by_day["2026-10-23"]["cost_usd"] == pytest.approx(0.02)
    assert by_day["2026-10-20"]["calls"] == 1
    assert by_day["2026-10-21"] == {"day": "2026-10-21", "calls": 0, "cost_usd": 0.0}
    assert sum(d["cost_usd"] for d in daily) == pytest.approx(0.05)  # older history stays out


def test_usage_page_draws_the_trend_and_overview_the_budget(window):
    days = [{"day": f"2026-10-{d:02d}", "calls": 1, "cost_usd": 0.01} for d in range(10, 24)]
    window._bridge.usage_updated.emit({
        "available": True, "daily": days, "recent": [],
        "today": {"calls": 3, "ok": 3, "errors": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                  "cache_write_tokens": 0, "cost_usd": 0.25, "tasks": 1},
        "month": {"calls": 9, "ok": 9, "errors": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                  "cache_write_tokens": 0, "cost_usd": 1.0, "tasks": 2},
    })
    window._usage_page.flush()
    assert len(window._usage_page.trend.days) == 14
    assert "$0.1400 over 14 days" in window._usage_page.trend_caption.text()
    window._overview.flush()
    assert window._overview.tile_month.value_label.text() == "$1.00"
    assert window._overview.budget_bar.ratio == pytest.approx(0.5)
    assert "50% of the $0.5000 daily limit" in window._overview.budget_caption.text()
