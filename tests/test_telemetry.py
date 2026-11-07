"""Live computer telemetry (app/system/telemetry.py) and its background
monitor (app/ui/telemetry.py).

The sampler is exercised against this machine's real Windows interfaces
(plausibility only -- the values are whatever the PC is doing); the monitor's
threading, demand and history logic run against a fake sampler. Waits use a
real QEventLoop: QTest.qWait keeps the GIL, which would stall the sampling
thread's cross-thread signal.
"""

from __future__ import annotations

import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from app.system.telemetry import Disk, Snapshot, StaticInfo, TelemetrySampler, busiest_gpu_percent
from app.ui.telemetry import (
    DETAIL,
    OFF,
    SUMMARY,
    GiB,
    Series,
    TelemetryHistory,
    TelemetryMonitor,
    resource_alerts,
)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="telemetry is implemented for Windows")


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _spin_until(predicate, timeout_ms: int = 3000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        if predicate():
            return True
        _spin(20)
    return predicate()


# ------------------------------------------------------------ pure logic
def test_gpu_figure_sums_processes_per_engine_and_takes_the_busiest():
    instances = {
        "pid_10_luid_0x0_0x1_phys_0_eng_0_engtype_3D": 20.0,
        "pid_11_luid_0x0_0x1_phys_0_eng_0_engtype_3D": 15.0,
        "pid_10_luid_0x0_0x1_phys_0_eng_3_engtype_VideoDecode": 30.0,
        "garbage": 99.0,
    }
    assert busiest_gpu_percent(instances) == pytest.approx(35.0)
    assert busiest_gpu_percent({}) is None
    assert busiest_gpu_percent({"pid_1_luid_0x0_0x1_phys_0_eng_0_engtype_3D": 180.0}) == 100.0


def test_snapshot_memory_figures_reconcile():
    snap = Snapshot(at=0.0, detail=False, ram_total=32 * GiB, ram_available=8 * GiB)
    assert snap.ram_used == 24 * GiB
    assert snap.ram_percent == pytest.approx(75.0)
    assert Snapshot(at=0.0, detail=False).ram_percent is None
    disk = Disk("C:", total=1000, free=250)
    assert disk.used == 750 and disk.percent == pytest.approx(75.0)


def test_series_keeps_real_samples_only():
    series = Series(maxlen=3)
    series.add(1.0, 10.0)
    series.add(2.0, None)  # an unavailable metric is a gap, not a zero
    series.add(3.0, 30.0)
    series.add(4.0, 40.0)
    series.add(5.0, 50.0)
    assert list(series.points) == [(3.0, 30.0), (4.0, 40.0), (5.0, 50.0)]
    assert series.latest == 50.0
    assert series.since(4.0) == [(4.0, 40.0), (5.0, 50.0)]


def test_resource_alerts_only_for_genuinely_exceeded_thresholds():
    history = TelemetryHistory()
    now = 1000.0
    assert resource_alerts(history, now) == []
    for i in range(4):
        history.add(
            Snapshot(
                at=now - 3 + i,
                detail=False,
                cpu_percent=50.0,
                ram_total=16 * GiB,
                ram_available=8 * GiB,
                process_memory=300 * 1024**2,
            )
        )
    assert resource_alerts(history, now) == []
    history = TelemetryHistory()
    for i in range(4):
        history.add(
            Snapshot(
                at=now - 3 + i,
                detail=True,
                cpu_percent=97.0,
                ram_total=16 * GiB,
                ram_available=GiB // 2,
                process_memory=3 * GiB,
                disks=(Disk("C:", total=100 * GiB, free=5 * GiB), Disk("D:", 100 * GiB, 60 * GiB)),
            )
        )
    texts = " ".join(alert.text for alert in resource_alerts(history, now))
    assert "CPU" in texts and "Memory" in texts and "Drive C:" in texts and "Garvis itself" in texts
    assert "Drive D:" not in texts


def test_a_short_cpu_spike_is_not_an_alert():
    history = TelemetryHistory()
    history.add(Snapshot(at=99.0, detail=False, cpu_percent=20.0))
    history.add(Snapshot(at=100.0, detail=False, cpu_percent=99.0))
    assert resource_alerts(history, 100.0) == []


# ------------------------------------------------ the real sampler (Windows)
@windows_only
def test_sampler_reads_plausible_values_from_this_machine():
    sampler = TelemetrySampler()
    static = sampler.static_info()
    assert static.logical_cpus and static.logical_cpus >= 1
    assert static.total_ram and static.total_ram > 512 * 1024**2
    first = sampler.sample(detail=True)
    assert first.cpu_percent is None  # a rate needs two samples: never invented
    time.sleep(0.3)
    snap = sampler.sample(detail=True)
    sampler.close()
    assert snap.errors == ()
    assert 0.0 <= snap.cpu_percent <= 100.0
    assert snap.per_core is not None and all(0.0 <= v <= 100.0 for v in snap.per_core)
    assert 0 < snap.ram_available <= snap.ram_total
    assert 0.0 <= snap.ram_percent <= 100.0
    assert snap.process_memory and snap.process_memory < snap.ram_total
    assert snap.uptime_seconds and snap.uptime_seconds > 0
    assert snap.disks and all(d.total >= d.free >= 0 for d in snap.disks)


@windows_only
def test_summary_samples_skip_the_detail_metrics():
    sampler = TelemetrySampler()
    sampler.sample()
    snap = sampler.sample(detail=False)
    sampler.close()
    assert snap.detail is False
    assert snap.per_core is None and snap.disks is None and snap.gpu_percent is None
    assert snap.ram_total is not None


def test_unsupported_platforms_report_unavailable(monkeypatch):
    import app.system.telemetry as telemetry

    monkeypatch.setattr(telemetry, "_IS_WINDOWS", False)
    sampler = telemetry.TelemetrySampler()
    snap = sampler.sample(detail=True)
    assert snap.cpu_percent is None and snap.ram_total is None and snap.errors
    assert sampler.static_info() == StaticInfo()


# ---------------------------------------------------- the background monitor
class _FakeSampler:
    instances: list["_FakeSampler"] = []

    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.closed = False
        self.thread = None
        _FakeSampler.instances.append(self)

    def static_info(self) -> StaticInfo:
        return StaticInfo(cpu_model="Test CPU", logical_cpus=4, total_ram=8 * GiB)

    def sample(self, detail: bool = False) -> Snapshot:
        self.thread = threading.current_thread()
        self.calls.append(detail)
        return Snapshot(
            at=time.time(),
            detail=detail,
            cpu_percent=12.0,
            ram_total=8 * GiB,
            ram_available=4 * GiB,
            per_core=(10.0, 20.0) if detail else None,
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def monitor(qapp):
    _FakeSampler.instances.clear()
    mon = TelemetryMonitor(sampler_factory=_FakeSampler, intervals={SUMMARY: 0.05, DETAIL: 0.03})
    yield mon
    mon.stop()


def test_no_thread_and_no_sampling_until_something_is_on_screen(monitor):
    _spin(100)
    assert not monitor.running and _FakeSampler.instances == []
    monitor.set_demand(OFF)
    assert not monitor.running


def test_samples_off_the_gui_thread_and_fills_the_history(monitor):
    monitor.set_demand(SUMMARY)
    assert _spin_until(lambda: len(monitor.history.series["cpu"]) >= 3)
    sampler = _FakeSampler.instances[0]
    assert sampler.thread is not threading.main_thread()
    assert set(sampler.calls) == {False}
    assert monitor.static is not None and monitor.static.cpu_model == "Test CPU"
    assert monitor.history.latest_detail is None


def test_detail_only_while_the_system_page_wants_it(monitor):
    monitor.set_demand(DETAIL)
    assert _spin_until(lambda: monitor.history.latest_detail is not None)
    assert monitor.history.latest_detail.per_core == (10.0, 20.0)
    monitor.set_demand(SUMMARY)
    sampler = _FakeSampler.instances[0]
    _spin(80)
    count = len(sampler.calls)
    assert _spin_until(lambda: len(sampler.calls) > count + 1)
    assert sampler.calls[-1] is False


def test_off_pauses_sampling_and_stop_ends_the_thread(monitor):
    monitor.set_demand(SUMMARY)
    assert _spin_until(lambda: len(monitor.history.series["cpu"]) >= 1)
    monitor.set_demand(OFF)
    _spin(80)
    sampler = _FakeSampler.instances[0]
    paused_at = len(sampler.calls)
    _spin(200)
    assert len(sampler.calls) <= paused_at + 1  # at most the sample that was already in flight
    assert monitor.running  # paused, not ended: resuming needs no new thread
    monitor.stop()
    assert not monitor.running and sampler.closed
    monitor.set_demand(SUMMARY)  # after stop(): ignored, never restarts
    assert not monitor.running


def test_a_failing_sampler_never_raises_into_the_gui(qapp):
    class Broken(_FakeSampler):
        def sample(self, detail: bool = False) -> Snapshot:
            raise OSError("counter vanished")

    mon = TelemetryMonitor(sampler_factory=Broken, intervals={SUMMARY: 0.02, DETAIL: 0.02})
    mon.set_demand(SUMMARY)
    _spin(150)
    assert mon.history.latest is None
    mon.stop()
    assert not mon.running
