"""Design-review screenshots of the real Desktop UI, with the real GPU orb.

The window is built against the test suite's fake bridge (no Agent, LLM,
microphone or network) and rendered with Qt::WA_DontShowOnScreen: nothing ever
appears on the desktop and no input is touched. Output: PNGs in OUT_DIR.

    python scripts/render_ui_review.py OUT_DIR [--pages overview,voice,...] [--modes idle,speaking]
                                               [--size 1360x860] [--demo] [--dialog]

--demo fills the window with sample content shaped like the approved mockup
(a short conversation, a tool step, usage figures and one telemetry sample),
fed through the same signals and slots the real bridge drives. --dialog also
renders the approval dialog over the window (desktop_dialog.png).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ["GARVIS_MOBILE_ENABLED"] = "false"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["GARVIS_DATA_DIR"] = tempfile.mkdtemp(prefix="garvis-review-")

from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

PAGES = "overview,activity,protocols,system,sources,voice,usage,tools,settings"


def wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def demo_content(win) -> None:
    """Sample content through the window's real entry points (no Agent)."""
    from app.system.telemetry import Disk, GpuAdapter, Snapshot, StaticInfo

    gib = 1024 ** 3
    bridge = win._bridge
    request = "Open the Garvis project folder and summarize what changed today"
    win._append_conversation("You", request)
    bridge.user_message.emit(request)
    bridge.status_changed.emit("Thinking...")
    bridge.tool_started.emit("open_path", {"path": r"C:\Users\me\Projects\garvis"})
    bridge.tool_finished.emit("open_path", True, "")
    reply = (r"Opened <b>C:\Users\me\Projects\garvis</b>. Today's changes: the plasma orb, a separate "
             "logo module, and the phone's audio routing.")
    bridge.assistant_message.emit(reply)
    bridge.status_changed.emit("Done.")
    second = "Thanks. Remind me at 6 to call mom"
    win._append_conversation("You", second)
    bridge.user_message.emit(second)
    bridge.assistant_message.emit("Done — I'll call your phone at 18:00.")
    bridge.status_changed.emit("Done.")
    research = "Latest news on the Artemis program?"
    win._append_conversation("You", research)
    bridge.user_message.emit(research)
    bridge.tool_started.emit("web_research", {"query": "Artemis program news"})
    bridge.tool_finished.emit("web_research", True, "")
    bridge.tool_sources.emit("web_research", [
        {"tool": "web_research", "kind": "read", "title": "Artemis II crew completes launch rehearsal",
         "url": "https://news.example/artemis-ii-rehearsal", "publisher": "news.example",
         "snippet": "The four astronauts completed a full countdown rehearsal at Kennedy Space Center."},
        {"tool": "web_research", "kind": "search_result", "title": "Artemis program overview",
         "url": "https://space.example/artemis", "publisher": "space.example", "snippet": ""}])
    bridge.assistant_message.emit("The Artemis II crew finished its launch rehearsal this week.")
    bridge.status_changed.emit("Done.")
    bridge.remote_user_message.emit("What's on my calendar tomorrow?", "mobile")
    bridge.assistant_message.emit("Nothing booked — your day is free.")
    bridge.status_changed.emit("Idle.")
    days = [{"day": f"2026-10-{18 + i:02d}", "cost_usd": c, "calls": n}
            for i, (c, n) in enumerate([(0.02, 4), (0.0, 0), (0.06, 9), (0.11, 14), (0.03, 5), (0.0, 0), (0.08, 10),
                                        (0.05, 7), (0.09, 12), (0.02, 3), (0.12, 16), (0.07, 9), (0.01, 2),
                                        (0.04, 12)])]
    today = {"cost_usd": 0.04, "calls": 12, "ok": 12, "errors": 0, "tasks": 5, "input_tokens": 48_210,
             "output_tokens": 6_120, "cache_read_tokens": 120_400, "cache_write_tokens": 9_800}
    recent = [{"ts": "2026-10-31T14:52:10", "tier": "main", "status": "ok", "error_type": None,
               "input_tokens": 4100, "output_tokens": 380, "cache_read_tokens": 11_000, "cost_usd": 0.0031,
               "latency_ms": 1800, "model": "claude-haiku-5-5"}] * 4
    win._on_usage_updated({"available": True, "today": today, "month": {"cost_usd": 1.18, "calls": 212},
                           "daily": days, "recent": recent})
    monitor = win.telemetry
    monitor._on_static(StaticInfo(cpu_model="Example 16-thread CPU", logical_cpus=16, base_mhz=1400,
                                  total_ram=32 * gib, gpus=(GpuAdapter("Example Graphics", 0),),
                                  os_name="Windows 11"))
    now = time.time()
    for i in range(30):
        monitor._on_sampled(Snapshot(
            at=now - 30 + i, detail=True, cpu_percent=18 + (i * 7) % 11, ram_total=32 * gib,
            ram_available=32 * gib * 0.39, process_cpu_percent=1.2, process_memory=310 * 1024 ** 2,
            process_private=290 * 1024 ** 2, uptime_seconds=93_784.0,
            per_core=tuple(float((i * 13 + k * 17) % 60) for k in range(16)), cpu_mhz=3100.0,
            disks=(Disk("C:", total=1000 * gib, free=412 * gib),), disk_read_bps=1.2 * 1024 ** 2,
            disk_write_bps=0.4 * 1024 ** 2, gpu_percent=9.0, gpu_memory_used=0, gpu_shared_used=gib))


def demo_protocol() -> None:
    """One saved protocol in the review's throw-away data folder."""
    from app.core.config import get_settings
    from app.tasks.protocols import Protocol, ProtocolStore, Step

    store = ProtocolStore(get_settings().data_dir / "protocols")
    store.save(Protocol(name="Programming workspace", description="Open the project and get ready to work.",
                        steps=[Step(type="open_path", params={"path": str(ROOT)}),
                               Step(type="wait", params={"seconds": 2}),
                               Step(type="notify", params={"title": "Workspace ready", "message": ""})]))


def render_dialog(win, out: Path, suffix: str) -> Path:
    from app.ui.approval import ApprovalDialog

    dialog = ApprovalDialog("gmail_send", {"to": "dana@example.com", "subject": "Monthly report — October",
                                           "attach": "report-october.pdf"},
                            "Sending email always asks first.", win,
                            title="Garvis wants to send an email")
    dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    dialog.show()
    wait(400)
    path = out / f"desktop_dialog{suffix}.png"
    dialog.grab().save(str(path))
    dialog.reject()
    dialog.deleteLater()
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--pages", default="overview", help=f"comma-separated, or 'all' ({PAGES})")
    ap.add_argument("--modes", default="idle")
    ap.add_argument("--size", default="1360x860")
    ap.add_argument("--demo", action="store_true", help="fill the window with mockup-like sample content")
    ap.add_argument("--dialog", action="store_true", help="also render the approval dialog")
    ap.add_argument("--suffix", default="", help="appended to every file name")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)  # noqa: F841

    from app.ui import main_window
    from app.ui.state import Mode
    from tests import test_command_center as fakes

    main_window.AgentBridge = fakes._CatalogBridge
    main_window.TelemetryMonitor = fakes._QuietMonitor
    if args.demo:
        demo_protocol()
        class _DemoMonitor(fakes._QuietMonitor):
            """No sampling thread: the demo's samples stay on screen."""

            def set_demand(self, demand: str) -> None:
                self._demand = demand

        main_window.TelemetryMonitor = _DemoMonitor
    win = main_window.MainWindow()
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    w, h = (int(v) for v in args.size.split("x"))
    win.resize(w, h)
    win.show()
    if args.demo:
        demo_content(win)
    wait(600)
    pages = PAGES if args.pages == "all" else args.pages
    for page in pages.split(","):
        win.show_page(page)
        for mode in args.modes.split(","):
            win._state.mode_changed.emit(Mode(mode).value)  # what the bridge-fed state would announce
            wait(1200)
            path = args.out / f"desktop_{page}_{mode}{args.suffix}.png"
            win.grab().save(str(path))
            print("wrote", path)
    if args.dialog:
        win.show_page("overview")
        wait(800)
        print("wrote", render_dialog(win, args.out, args.suffix))
    win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
