"""Background sampling of app/system/telemetry.py for the dashboard.

One daemon thread owns the TelemetrySampler; the Qt main thread never makes a
telemetry call. How much is sampled follows what is on screen (`set_demand`):

  "off"      the dashboard is hidden or minimized (e.g. collapsed to the orb):
             the thread sleeps on a condition variable -- no sampling at all
  "summary"  the dashboard is visible: CPU, memory and Garvis's own process
             every 2 s (well under a millisecond per sample)
  "detail"   the System page is visible: everything, every second

Samples cross to the GUI thread as a queued Qt signal; the history buffers
(bounded deques of real samples with their timestamps) live on the GUI side
only. A pause leaves a real gap in the history -- charts draw it as a gap
rather than inventing the missing points.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal, Slot

from app.core.logging import get_logger, log_extra
from app.system.telemetry import Snapshot, StaticInfo, TelemetrySampler

logger = get_logger("ui.telemetry")

OFF, SUMMARY, DETAIL = "off", "summary", "detail"
INTERVALS = {SUMMARY: 2.0, DETAIL: 1.0}
HISTORY_SECONDS = 180.0
_MAX_POINTS = 240

GiB = 1024**3


class Series:
    """(timestamp, value) pairs, oldest first; None values are not stored."""

    def __init__(self, maxlen: int = _MAX_POINTS) -> None:
        self.points: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def add(self, at: float, value: float | None) -> None:
        if value is not None:
            self.points.append((at, float(value)))

    @property
    def latest(self) -> float | None:
        return self.points[-1][1] if self.points else None

    def since(self, cutoff: float) -> list[tuple[float, float]]:
        return [p for p in self.points if p[0] >= cutoff]

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class ResourceAlert:
    level: str  # "warn" | "error"
    text: str


class TelemetryHistory:
    KEYS = ("cpu", "ram", "process_cpu", "process_memory", "gpu", "disk_read", "disk_write")

    def __init__(self) -> None:
        self.series = {key: Series() for key in self.KEYS}
        self.latest: Snapshot | None = None
        self.latest_detail: Snapshot | None = None

    def add(self, snap: Snapshot) -> None:
        self.latest = snap
        if snap.detail:
            self.latest_detail = snap
        s = self.series
        s["cpu"].add(snap.at, snap.cpu_percent)
        s["ram"].add(snap.at, snap.ram_percent)
        s["process_cpu"].add(snap.at, snap.process_cpu_percent)
        s["process_memory"].add(snap.at, snap.process_memory)
        if snap.detail:
            s["gpu"].add(snap.at, snap.gpu_percent)
            s["disk_read"].add(snap.at, snap.disk_read_bps)
            s["disk_write"].add(snap.at, snap.disk_write_bps)


def resource_alerts(history: TelemetryHistory, now: float | None = None) -> list[ResourceAlert]:
    """Warnings only for thresholds that are genuinely exceeded right now."""
    snap = history.latest
    if snap is None:
        return []
    now = now or time.time()
    alerts: list[ResourceAlert] = []
    recent = [v for _, v in history.series["cpu"].since(now - 12.0)]
    if len(recent) >= 3 and min(recent) >= 90.0:
        alerts.append(
            ResourceAlert("warn", f"CPU has stayed above 90% for {len(recent)} samples (now {recent[-1]:.0f}%).")
        )
    ram = snap.ram_percent
    if ram is not None and ram >= 90.0:
        alerts.append(
            ResourceAlert(
                "error" if ram >= 96.0 else "warn",
                f"Memory is {ram:.0f}% used — {snap.ram_available / GiB:.1f} GB left.",
            )
        )
    detail = history.latest_detail
    for disk in (detail.disks or ()) if detail is not None else ():
        if disk.total and disk.free / disk.total < 0.10:
            alerts.append(
                ResourceAlert(
                    "warn",
                    f"Drive {disk.root} has only {disk.free / GiB:.1f} GB free ({100 * disk.free / disk.total:.0f}%).",
                )
            )
    if snap.process_memory is not None and snap.process_memory >= 2 * GiB:
        alerts.append(ResourceAlert("warn", f"Garvis itself is using {snap.process_memory / GiB:.1f} GB of memory."))
    return alerts


class TelemetryMonitor(QObject):
    """Qt-facing owner of the sampling thread. `updated` fires on the GUI
    thread after `history` took in a new sample; `static_ready` once with
    the hardware description."""

    updated = Signal()
    static_ready = Signal(object)
    _sampled = Signal(object)
    _static = Signal(object)

    def __init__(
        self, sampler_factory=TelemetrySampler, intervals: dict[str, float] | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._factory = sampler_factory
        self._intervals = dict(intervals or INTERVALS)
        self.history = TelemetryHistory()
        self.static: StaticInfo | None = None
        self._demand = OFF
        self._stopping = False
        self._wake = threading.Condition()
        self._thread: threading.Thread | None = None
        self._sampled.connect(self._on_sampled)
        self._static.connect(self._on_static)

    @property
    def demand(self) -> str:
        return self._demand

    @property
    def interval(self) -> float | None:
        return self._intervals.get(self._demand)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_demand(self, demand: str) -> None:
        if demand not in (OFF, SUMMARY, DETAIL):
            raise ValueError(demand)
        if self._stopping:
            return
        with self._wake:
            changed = demand != self._demand
            self._demand = demand
            self._wake.notify_all()
        if demand != OFF and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="garvis-telemetry", daemon=True)
            self._thread.start()
        if changed:
            log_extra(logger, 10, "telemetry.demand", demand=demand)

    def stop(self, timeout: float = 2.0) -> None:
        """Ends the thread (it finishes at most the sample in progress)."""
        with self._wake:
            self._stopping = True
            self._demand = OFF
            self._wake.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)

    # -------------------------------------------------------------- thread
    def _run(self) -> None:
        sampler = None
        try:
            sampler = self._factory()
            self._emit(self._static, sampler.static_info())
            while True:
                with self._wake:
                    while self._demand == OFF and not self._stopping:
                        self._wake.wait()
                    if self._stopping:
                        return
                    demand = self._demand
                snap = sampler.sample(detail=demand == DETAIL)
                if not self._emit(self._sampled, snap):
                    return
                with self._wake:
                    if self._stopping:
                        return
                    if self._demand == demand:  # a demand change wakes us early
                        self._wake.wait(self._intervals.get(demand, 1.0))
        except Exception:  # noqa: BLE001 -- telemetry must never take Garvis down
            logger.exception("telemetry.thread_failed")
        finally:
            if sampler is not None:
                try:
                    sampler.close()
                except Exception:  # noqa: BLE001
                    pass

    def _emit(self, signal, value) -> bool:
        try:
            signal.emit(value)
            return True
        except RuntimeError:  # the QObject is already gone (shutdown)
            return False

    # ----------------------------------------------------------- GUI side
    @Slot(object)
    def _on_sampled(self, snap: Snapshot) -> None:
        if self._stopping:
            return
        self.history.add(snap)
        self.updated.emit()

    @Slot(object)
    def _on_static(self, info: StaticInfo) -> None:
        self.static = info
        self.static_ready.emit(info)
