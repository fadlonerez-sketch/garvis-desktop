"""Live computer telemetry for the dashboard (System page and Overview).

Real measurements only, from Windows' own interfaces through ctypes -- no
extra dependency, no subprocess, nothing launched per sample:

  CPU total          GetSystemTimes (kernel32)
  CPU per core       NtQuerySystemInformation(SystemProcessorPerformanceInformation)
  CPU speed          PDH "% Processor Performance" x the registry's base clock
                     -- the same estimate Task Manager shows, labelled as such
  Memory             GlobalMemoryStatusEx
  Garvis process     GetProcessTimes + K32GetProcessMemoryInfo
  Disks              GetLogicalDriveStringsW / GetDriveTypeW / GetDiskFreeSpaceExW
  Disk throughput    PDH PhysicalDisk(_Total) read/write bytes per second
  GPU                PDH "GPU Engine" / "GPU Adapter Memory" counters; names and
                     dedicated memory from the display-adapter registry keys
  Uptime             GetTickCount64

Anything unavailable (another OS, a counter Windows doesn't expose, a
permission problem) is None -- callers show "unavailable", never a guess.
GPU temperature is deliberately not read: Windows has no reliable,
unprivileged interface for it.

`TelemetrySampler` is not thread-safe; one background thread owns it (see
app/ui/telemetry.py). Rate counters (CPU %, throughput) need two samples, so
the first sample reports them as None.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

_IS_WINDOWS = sys.platform == "win32"


@dataclass(frozen=True)
class GpuAdapter:
    name: str
    dedicated_bytes: int | None


@dataclass(frozen=True)
class StaticInfo:
    cpu_model: str | None = None
    logical_cpus: int | None = None
    base_mhz: int | None = None
    total_ram: int | None = None
    gpus: tuple[GpuAdapter, ...] = ()
    os_name: str | None = None


@dataclass(frozen=True)
class Disk:
    root: str
    total: int
    free: int

    @property
    def used(self) -> int:
        return max(0, self.total - self.free)

    @property
    def percent(self) -> float:
        return 100.0 * self.used / self.total if self.total else 0.0


@dataclass(frozen=True)
class Snapshot:
    at: float
    detail: bool
    cpu_percent: float | None = None
    per_core: tuple[float, ...] | None = None
    cpu_mhz: float | None = None
    ram_total: int | None = None
    ram_available: int | None = None
    process_cpu_percent: float | None = None
    process_memory: int | None = None
    """Working set: the physical memory Garvis's own process holds now."""
    process_private: int | None = None
    uptime_seconds: float | None = None
    disks: tuple[Disk, ...] | None = None
    disk_read_bps: float | None = None
    disk_write_bps: float | None = None
    gpu_percent: float | None = None
    gpu_memory_used: int | None = None
    """Dedicated (on-board) GPU memory in use, busiest adapter."""
    gpu_shared_used: int | None = None
    """System memory the GPU is using (integrated GPUs live here)."""
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ram_used(self) -> int | None:
        if self.ram_total is None or self.ram_available is None:
            return None
        return max(0, self.ram_total - self.ram_available)

    @property
    def ram_percent(self) -> float | None:
        used = self.ram_used
        return None if used is None or not self.ram_total else 100.0 * used / self.ram_total


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def busiest_gpu_percent(instances: dict[str, float]) -> float | None:
    """Task Manager's GPU figure from "GPU Engine(*)" instances
    ("pid_12_luid_0x0_0x1_phys_0_eng_3_engtype_3D" -> value): sum every
    process's share per (adapter, engine), then take the busiest engine."""
    if not instances:
        return None
    totals: dict[tuple[str, str], float] = {}
    for name, value in instances.items():
        parts = name.split("_")
        try:
            luid = "_".join(parts[parts.index("luid") + 1: parts.index("luid") + 3])
            engine = parts[parts.index("eng") + 1]
        except (ValueError, IndexError):
            continue
        key = (luid, engine)
        totals[key] = totals.get(key, 0.0) + max(0.0, value)
    return _clamp_percent(max(totals.values())) if totals else None


# --------------------------------------------------------------------------
# Windows implementation
# --------------------------------------------------------------------------
if _IS_WINDOWS:
    import ctypes
    import winreg
    from ctypes import wintypes

    class _FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        @property
        def value(self) -> int:
            return (self.high << 32) | self.low

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _PROCESSOR_PERFORMANCE(ctypes.Structure):
        _fields_ = [
            ("IdleTime", ctypes.c_int64),
            ("KernelTime", ctypes.c_int64),
            ("UserTime", ctypes.c_int64),
            ("DpcTime", ctypes.c_int64),
            ("InterruptTime", ctypes.c_int64),
            ("InterruptCount", ctypes.c_ulong),
        ]

    class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
        _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

    class _PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
        _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _PDH_FMT_COUNTERVALUE)]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetTickCount64.restype = ctypes.c_uint64
    _kernel32.GetLogicalDriveStringsW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    _kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    _kernel32.GetDiskFreeSpaceExW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_uint64),
                                              ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    _kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(_FILETIME)] * 4
    _kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.GetActiveProcessorCount.argtypes = [wintypes.WORD]
    _kernel32.GetActiveProcessorCount.restype = wintypes.DWORD
    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtQuerySystemInformation.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                                                ctypes.POINTER(ctypes.c_ulong)]
    _ntdll.NtQuerySystemInformation.restype = ctypes.c_long

    _DRIVE_FIXED = 3
    _PDH_FMT_DOUBLE = 0x00000200
    _PDH_FMT_NOCAP100 = 0x00008000
    _PDH_MORE_DATA = 0x800007D2
    _PDH_VALID = (0, 1)  # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA
    _GPU_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"

    try:
        _pdh = ctypes.WinDLL("pdh")
        _pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]
        _pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_void_p,
                                               ctypes.POINTER(wintypes.HANDLE)]
        _pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        _pdh.PdhGetFormattedCounterValue.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                     ctypes.POINTER(wintypes.DWORD),
                                                     ctypes.POINTER(_PDH_FMT_COUNTERVALUE)]
        _pdh.PdhGetFormattedCounterArrayW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                      ctypes.POINTER(wintypes.DWORD),
                                                      ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        _pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        for _fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
                    "PdhGetFormattedCounterValue", "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(_pdh, _fn).restype = wintypes.DWORD
    except OSError:  # pragma: no cover - pdh.dll ships with every supported Windows
        _pdh = None

    def _system_times() -> tuple[int, int, int] | None:
        idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
        if not _kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return idle.value, kernel.value, user.value

    def _core_times() -> list[tuple[int, int]] | None:
        """(idle, busy+idle) 100-ns totals per logical processor (current group)."""
        count = max(1, int(_kernel32.GetActiveProcessorCount(0)))
        buffer = (_PROCESSOR_PERFORMANCE * count)()
        returned = ctypes.c_ulong(0)
        status = _ntdll.NtQuerySystemInformation(8, buffer, ctypes.sizeof(buffer), ctypes.byref(returned))
        if status != 0:
            return None
        n = returned.value // ctypes.sizeof(_PROCESSOR_PERFORMANCE)
        # KernelTime includes IdleTime.
        return [(buffer[i].IdleTime, buffer[i].KernelTime + buffer[i].UserTime) for i in range(n)]

    def _memory() -> tuple[int, int] | None:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)

    def _process_times() -> int | None:
        created, exited, kernel, user = _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
        if not _kernel32.GetProcessTimes(_kernel32.GetCurrentProcess(), ctypes.byref(created),
                                         ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        return kernel.value + user.value

    def _process_memory() -> tuple[int, int] | None:
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not _kernel32.K32GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                                 counters.cb):
            return None
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)

    def _fixed_disks() -> tuple[Disk, ...]:
        buffer = ctypes.create_unicode_buffer(512)
        length = _kernel32.GetLogicalDriveStringsW(len(buffer), buffer)
        roots = [r for r in buffer[:length].split("\x00") if r]
        disks = []
        for root in roots:
            if _kernel32.GetDriveTypeW(root) != _DRIVE_FIXED:
                continue
            available, total, free = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
            if _kernel32.GetDiskFreeSpaceExW(root, ctypes.byref(available), ctypes.byref(total),
                                             ctypes.byref(free)) and total.value:
                disks.append(Disk(root=root.rstrip("\\"), total=int(total.value), free=int(free.value)))
        return tuple(disks)

    def _registry_value(root, path: str, name: str):
        try:
            with winreg.OpenKey(root, path) as key:
                return winreg.QueryValueEx(key, name)[0]
        except OSError:
            return None

    def _gpu_adapters() -> tuple[GpuAdapter, ...]:
        adapters: list[GpuAdapter] = []
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _GPU_CLASS) as cls:
                index = 0
                while True:
                    try:
                        sub = winreg.EnumKey(cls, index)
                    except OSError:
                        break
                    index += 1
                    if not sub.isdigit():
                        continue
                    path = f"{_GPU_CLASS}\\{sub}"
                    name = _registry_value(winreg.HKEY_LOCAL_MACHINE, path, "DriverDesc")
                    if not name or "Microsoft" in str(name):  # Basic Display / Remote Display adapters
                        continue
                    size = _registry_value(winreg.HKEY_LOCAL_MACHINE, path, "HardwareInformation.qwMemorySize")
                    if size is None:
                        size = _registry_value(winreg.HKEY_LOCAL_MACHINE, path, "HardwareInformation.MemorySize")
                    if isinstance(size, bytes):
                        size = int.from_bytes(size[:8], "little") if size else None
                    dedicated = int(size) if isinstance(size, int) and size > 0 else None
                    if all(a.name != name for a in adapters):
                        adapters.append(GpuAdapter(name=str(name).strip(), dedicated_bytes=dedicated))
        except OSError:
            return ()
        return tuple(adapters)

    class _PdhQuery:
        """One PDH query; a counter that can't be added is simply absent."""

        def __init__(self, paths: dict[str, str]) -> None:
            self._query = wintypes.HANDLE()
            self._counters: dict[str, wintypes.HANDLE] = {}
            if _pdh is None or _pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)) != 0:
                self._query = None
                return
            for key, path in paths.items():
                counter = wintypes.HANDLE()
                if _pdh.PdhAddEnglishCounterW(self._query, path, None, ctypes.byref(counter)) == 0:
                    self._counters[key] = counter

        @property
        def keys(self) -> set[str]:
            return set(self._counters)

        def collect(self) -> bool:
            return self._query is not None and _pdh.PdhCollectQueryData(self._query) == 0

        def value(self, key: str, flags: int = 0) -> float | None:
            counter = self._counters.get(key)
            if counter is None:
                return None
            result = _PDH_FMT_COUNTERVALUE()
            status = _pdh.PdhGetFormattedCounterValue(counter, _PDH_FMT_DOUBLE | flags, None, ctypes.byref(result))
            if status != 0 or result.CStatus not in _PDH_VALID:
                return None
            return float(result.doubleValue)

        def values(self, key: str, flags: int = 0) -> dict[str, float] | None:
            counter = self._counters.get(key)
            if counter is None:
                return None
            size, count = wintypes.DWORD(0), wintypes.DWORD(0)
            status = _pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_DOUBLE | flags, ctypes.byref(size),
                                                       ctypes.byref(count), None)
            if status != _PDH_MORE_DATA or not size.value:
                return {} if status == 0 else None
            buffer = ctypes.create_string_buffer(size.value)
            status = _pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_DOUBLE | flags, ctypes.byref(size),
                                                       ctypes.byref(count), buffer)
            if status != 0:
                return None
            items = ctypes.cast(buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM))
            return {items[i].szName: float(items[i].FmtValue.doubleValue)
                    for i in range(count.value) if items[i].FmtValue.CStatus in _PDH_VALID}

        def close(self) -> None:
            if self._query is not None:
                _pdh.PdhCloseQuery(self._query)
                self._query = None


_DETAIL_COUNTERS = {
    "cpu_perf": r"\Processor Information(_Total)\% Processor Performance",
    "disk_read": r"\PhysicalDisk(_Total)\Disk Read Bytes/sec",
    "disk_write": r"\PhysicalDisk(_Total)\Disk Write Bytes/sec",
    "gpu_engine": r"\GPU Engine(*)\Utilization Percentage",
    "gpu_memory": r"\GPU Adapter Memory(*)\Dedicated Usage",
    "gpu_shared": r"\GPU Adapter Memory(*)\Shared Usage",
}


class TelemetrySampler:
    """Takes one Snapshot per call. `detail=False` is the cheap Overview set
    (CPU, memory, Garvis's process); `detail=True` adds per-core load, CPU
    speed, disks and the PDH counters (System page only)."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._prev_system: tuple[int, int, int] | None = None
        self._prev_cores: list[tuple[int, int]] | None = None
        self._prev_process: tuple[float, int] | None = None
        self._pdh = None
        self._static: StaticInfo | None = None

    @property
    def supported(self) -> bool:
        return _IS_WINDOWS

    def static_info(self) -> StaticInfo:
        if self._static is None:
            self._static = self._read_static()
        return self._static

    def _read_static(self) -> StaticInfo:
        if not _IS_WINDOWS:
            return StaticInfo()
        cpu_key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        model = _registry_value(winreg.HKEY_LOCAL_MACHINE, cpu_key, "ProcessorNameString")
        mhz = _registry_value(winreg.HKEY_LOCAL_MACHINE, cpu_key, "~MHz")
        product = _registry_value(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                                  "ProductName")
        build = _registry_value(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                                "CurrentBuildNumber")
        if product and build and build.isdigit() and int(build) >= 22000:
            product = str(product).replace("Windows 10", "Windows 11")  # the registry still says 10 on 11
        memory = _memory()
        logical = int(_kernel32.GetActiveProcessorCount(0xFFFF)) or None
        return StaticInfo(
            cpu_model=" ".join(str(model).split()) if model else None,
            logical_cpus=logical,
            base_mhz=int(mhz) if isinstance(mhz, int) and mhz > 0 else None,
            total_ram=memory[0] if memory else None,
            gpus=_gpu_adapters(),
            os_name=f"{product} (build {build})" if product and build else (str(product) if product else None),
        )

    def sample(self, detail: bool = False) -> Snapshot:
        now = self._clock()
        if not _IS_WINDOWS:
            return Snapshot(at=time.time(), detail=detail, errors=("Live telemetry is only implemented on Windows.",))
        errors: list[str] = []
        values: dict = {}

        def attempt(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 -- one failing metric must not lose the others
                errors.append(f"{label}: {exc}")

        attempt("cpu", lambda: values.update(cpu_percent=self._cpu_percent()))
        attempt("memory", lambda: values.update(self._memory_values()))
        attempt("process", lambda: values.update(self._process_values(now)))
        attempt("uptime", lambda: values.update(uptime_seconds=_kernel32.GetTickCount64() / 1000.0))
        if detail:
            attempt("cores", lambda: values.update(per_core=self._per_core()))
            attempt("disks", lambda: values.update(disks=_fixed_disks()))
            attempt("counters", lambda: values.update(self._counter_values()))
        else:
            self._prev_cores = None  # a later detail sample must not average over the gap
            self._close_pdh()
        return Snapshot(at=time.time(), detail=detail, errors=tuple(errors), **values)

    def close(self) -> None:
        self._close_pdh()

    # ------------------------------------------------------------ helpers
    def _cpu_percent(self) -> float | None:
        current = _system_times()
        previous, self._prev_system = self._prev_system, current
        if current is None or previous is None:
            return None
        idle = current[0] - previous[0]
        total = (current[1] - previous[1]) + (current[2] - previous[2])
        return None if total <= 0 else _clamp_percent(100.0 * (1.0 - idle / total))

    def _per_core(self) -> tuple[float, ...] | None:
        current = _core_times()
        previous, self._prev_cores = self._prev_cores, current
        if current is None or previous is None or len(previous) != len(current):
            return None
        loads = []
        for (idle_now, total_now), (idle_before, total_before) in zip(current, previous, strict=True):
            total = total_now - total_before
            loads.append(0.0 if total <= 0 else _clamp_percent(100.0 * (1.0 - (idle_now - idle_before) / total)))
        return tuple(loads)

    @staticmethod
    def _memory_values() -> dict:
        memory = _memory()
        return {} if memory is None else {"ram_total": memory[0], "ram_available": memory[1]}

    def _process_values(self, now: float) -> dict:
        values: dict = {}
        memory = _process_memory()
        if memory is not None:
            values["process_memory"], values["process_private"] = memory
        cpu_time = _process_times()
        previous, self._prev_process = self._prev_process, (now, cpu_time) if cpu_time is not None else None
        if cpu_time is not None and previous is not None and now > previous[0]:
            cpus = self.static_info().logical_cpus or 1
            wall = (now - previous[0]) * 10_000_000  # 100-ns units
            values["process_cpu_percent"] = _clamp_percent(100.0 * (cpu_time - previous[1]) / (wall * cpus))
        return values

    def _counter_values(self) -> dict:
        if self._pdh is None:
            self._pdh = _PdhQuery(_DETAIL_COUNTERS)
        if not self._pdh.collect():
            return {}
        values: dict = {}
        perf = self._pdh.value("cpu_perf", _PDH_FMT_NOCAP100)
        base = self.static_info().base_mhz
        if perf is not None and base:
            values["cpu_mhz"] = base * perf / 100.0
        values["disk_read_bps"] = self._pdh.value("disk_read")
        values["disk_write_bps"] = self._pdh.value("disk_write")
        engines = self._pdh.values("gpu_engine", _PDH_FMT_NOCAP100)
        if engines is not None:
            values["gpu_percent"] = busiest_gpu_percent(engines) if engines else 0.0
        memory = self._pdh.values("gpu_memory")
        if memory:
            values["gpu_memory_used"] = int(max(memory.values()))
        shared = self._pdh.values("gpu_shared")
        if shared:
            values["gpu_shared_used"] = int(max(shared.values()))
        return values

    def _close_pdh(self) -> None:
        if self._pdh is not None:
            self._pdh.close()
            self._pdh = None
