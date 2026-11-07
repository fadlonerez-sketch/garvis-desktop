"""Read-only system queries: processes, overall status, recent error events
and installed applications. Fixed scripts with base64-embedded parameters;
the LLM never supplies script text here."""

from __future__ import annotations

from collections.abc import Callable

from app.system.powershell import PowerShellError, PowerShellRunner, ps_string

_PROCESSES_SCRIPT = r"""
$cores = [Environment]::ProcessorCount
$first = @{}
foreach ($p in Get-Process) { try { $first[$p.Id] = $p.TotalProcessorTime.TotalSeconds } catch {} }
Start-Sleep -Milliseconds 500
$rows = foreach ($p in Get-Process) {
    $cpu = $null
    try { if ($first.ContainsKey($p.Id)) { $cpu = [math]::Round((($p.TotalProcessorTime.TotalSeconds - $first[$p.Id]) / 0.5 / $cores) * 100, 1) } } catch {}
    $responding = $true; try { $responding = $p.Responding } catch {}
    [pscustomobject]@{ name = $p.ProcessName; pid = $p.Id; cpu_percent = $cpu;
        memory_mb = [math]::Round($p.WorkingSet64 / 1MB, 1); responding = $responding; window_title = $p.MainWindowTitle }
}
@($rows) | ConvertTo-Json -Compress
"""

_STATUS_SCRIPT = r"""
$os = Get-CimInstance Win32_OperatingSystem
$cpu = @(Get-CimInstance Win32_Processor)
$disks = @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3")
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
[pscustomobject]@{
    os = "$($os.Caption) ($($os.Version))"
    computer = $env:COMPUTERNAME
    uptime_hours = [math]::Round(((Get-Date) - $os.LastBootUpTime).TotalHours, 1)
    cpu = [pscustomobject]@{ model = $cpu[0].Name.Trim(); load_percent = [math]::Round(($cpu | Measure-Object LoadPercentage -Average).Average, 0); logical_cores = [Environment]::ProcessorCount }
    memory = [pscustomobject]@{ total_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1); free_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 1) }
    disks = @($disks | ForEach-Object { [pscustomobject]@{ drive = $_.DeviceID; label = $_.VolumeName; total_gb = [math]::Round($_.Size / 1GB, 1); free_gb = [math]::Round($_.FreeSpace / 1GB, 1) } })
    battery = $(if ($battery) { [pscustomobject]@{ percent = $battery.EstimatedChargeRemaining; status = $battery.BatteryStatus } } else { $null })
} | ConvertTo-Json -Depth 4 -Compress
"""

_ERRORS_SCRIPT = r"""
$filter = __FILTER__
$since = (Get-Date).AddHours(-__HOURS__)
$events = Get-WinEvent -FilterHashtable @{ LogName = 'Application', 'System'; Level = 1, 2; StartTime = $since } -MaxEvents 400 -ErrorAction SilentlyContinue
$rows = foreach ($e in $events) {
    $message = ($e.Message -replace '\s+', ' ')
    if ($filter -and -not ($e.ProviderName -like "*$filter*" -or $message -like "*$filter*")) { continue }
    [pscustomobject]@{ time = $e.TimeCreated.ToString('s'); log = $e.LogName; source = $e.ProviderName; id = $e.Id;
        level = $e.LevelDisplayName; message = $(if ($message.Length -gt 400) { $message.Substring(0, 400) + '...' } else { $message }) }
}
@($rows | Select-Object -First __LIMIT__) | ConvertTo-Json -Compress
"""


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class SystemInfo:
    def __init__(
        self,
        runner: PowerShellRunner | None = None,
        registry_apps: Callable[[], list[dict]] | None = None,
        start_apps: Callable[[], list[tuple[str, str]]] | None = None,
    ) -> None:
        self._runner = runner or PowerShellRunner()
        self._registry_apps = registry_apps or installed_apps_from_registry
        self._start_apps = start_apps

    async def processes(self) -> list[dict]:
        return _as_list(await self._runner.run_json(_PROCESSES_SCRIPT, timeout_seconds=30))

    async def status(self) -> dict:
        data = await self._runner.run_json(_STATUS_SCRIPT, timeout_seconds=30)
        if not isinstance(data, dict):
            raise PowerShellError("No system information was returned.")
        return data

    async def recent_errors(self, hours: float, filter_text: str | None, limit: int) -> list[dict]:
        script = (
            _ERRORS_SCRIPT.replace("__FILTER__", ps_string(filter_text or ""))
            .replace("__HOURS__", str(float(hours)))
            .replace("__LIMIT__", str(int(limit)))
        )
        return _as_list(await self._runner.run_json(script, timeout_seconds=60))

    def installed_apps(self) -> list[dict]:
        apps = {a["name"].lower(): a for a in self._registry_apps()}
        if self._start_apps is not None:
            for name, _app_id in self._start_apps():
                apps.setdefault(name.lower(), {"name": name, "source": "start menu"})
        return sorted(apps.values(), key=lambda a: a["name"].lower())


def installed_apps_from_registry() -> list[dict]:
    """DisplayName/Version/Publisher of everything in the Uninstall keys."""
    try:
        import winreg
    except ImportError:
        return []
    apps: list[dict] = []
    roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for hive, path in roots:
        try:
            root = winreg.OpenKey(hive, path)
        except OSError:
            continue
        with root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                try:
                    with winreg.OpenKey(root, winreg.EnumKey(root, index)) as key:
                        values = {}
                        for field in ("DisplayName", "DisplayVersion", "Publisher", "SystemComponent"):
                            try:
                                values[field] = winreg.QueryValueEx(key, field)[0]
                            except OSError:
                                pass
                except OSError:
                    continue
                if not values.get("DisplayName") or values.get("SystemComponent") == 1:
                    continue
                apps.append(
                    {
                        "name": str(values["DisplayName"]),
                        "version": str(values.get("DisplayVersion") or ""),
                        "publisher": str(values.get("Publisher") or ""),
                    }
                )
    return apps
