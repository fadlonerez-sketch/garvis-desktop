"""The dedicated Garvis workspace: a Windows virtual desktop named "Garvis".

Why a virtual desktop (and not a hidden Win32 desktop via CreateDesktop):
windows on another *virtual* desktop keep rendering and remain fully
drivable by Playwright/CDP, and the user can peek at them any time
(Win+Ctrl+Left/Right or Task View) -- while their own primary desktop is
never covered or switched. A CreateDesktop-style hidden desktop receives no
input unless it becomes the *input* desktop, which would take over the
user's screen, and Chrome/GPU rendering there is unreliable.

Windows has no public API to create/switch virtual desktops or move another
process's window between them (IVirtualDesktopManager only covers the
calling process's own windows), so this uses `pyvda`, which wraps the
undocumented IVirtualDesktopManagerInternal COM interfaces per Windows
build. That is a real fragility: a future Windows update can break it. The
design isolates it accordingly --
  - `VirtualDesktopBackend` is the only thing that knows pyvda exists, and
    it's imported lazily;
  - every call runs on ONE dedicated worker thread (pyvda's COM objects are
    thread-local/apartment-bound, and Garvis's agent runs on a background
    asyncio thread);
  - any failure marks the workspace unavailable and is logged; work that
    targets the workspace then fails with a clear error (it is never moved
    onto the user's desktop instead -- app/desktop/routing.py), while
    everything aimed at the user's desktop keeps working.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel

from app.core.exceptions import WorkspaceError
from app.core.logging import get_logger, log_extra

logger = get_logger("desktop.workspace")


class DesktopRef(BaseModel):
    id: str
    number: int
    name: str = ""


class VirtualDesktopBackend(ABC):
    """Synchronous virtual-desktop primitives. Always called on the
    WorkspaceManager's single worker thread."""

    @abstractmethod
    def list_desktops(self) -> list[DesktopRef]: ...

    @abstractmethod
    def current_desktop(self) -> DesktopRef: ...

    @abstractmethod
    def create_desktop(self, name: str) -> DesktopRef: ...

    @abstractmethod
    def switch_to(self, desktop_id: str) -> None: ...

    @abstractmethod
    def move_window(self, handle: int, desktop_id: str) -> None: ...

    @abstractmethod
    def window_desktop_id(self, handle: int) -> str | None: ...


class PyvdaVirtualDesktopBackend(VirtualDesktopBackend):
    def __init__(self) -> None:
        import pyvda  # lazy: only ever imported on this worker thread

        self._pyvda = pyvda

    @staticmethod
    def _ref(desktop) -> DesktopRef:
        return DesktopRef(id=str(desktop.id), number=desktop.number, name=desktop.name or "")

    def _find(self, desktop_id: str):
        for desktop in self._pyvda.get_virtual_desktops():
            if str(desktop.id) == desktop_id:
                return desktop
        raise WorkspaceError(f"Virtual desktop {desktop_id} no longer exists.")

    def list_desktops(self) -> list[DesktopRef]:
        return [self._ref(d) for d in self._pyvda.get_virtual_desktops()]

    def current_desktop(self) -> DesktopRef:
        return self._ref(self._pyvda.VirtualDesktop.current())

    def create_desktop(self, name: str) -> DesktopRef:
        desktop = self._pyvda.VirtualDesktop.create()
        desktop.rename(name)
        return self._ref(desktop)

    def switch_to(self, desktop_id: str) -> None:
        self._find(desktop_id).go()

    def move_window(self, handle: int, desktop_id: str) -> None:
        self._pyvda.AppView(hwnd=handle).move(self._find(desktop_id))

    def window_desktop_id(self, handle: int) -> str | None:
        try:
            return str(self._pyvda.AppView(hwnd=handle).desktop_id)
        except Exception:
            return None


class WorkspaceManager:
    """Async facade over the Garvis virtual desktop. Safe to construct
    anywhere; nothing touches the OS until first use."""

    def __init__(
        self,
        backend_factory: Callable[[], VirtualDesktopBackend] | None,
        name: str = "Garvis",
    ) -> None:
        self._backend_factory = backend_factory
        self._backend: VirtualDesktopBackend | None = None
        self._name = name
        self._garvis_id: str | None = None
        self._return_desktop_id: str | None = None
        """The desktop the user was on when Garvis last switched the view to
        its own workspace -- where show_primary() goes back to."""
        self._unavailable_reason: str | None = None if backend_factory else "not supported on this platform"
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    # ---------------------------------------------------------- plumbing
    async def _call(self, func: Callable[[VirtualDesktopBackend], object]):
        if self._unavailable_reason is not None:
            raise WorkspaceError(f"The Garvis workspace is unavailable: {self._unavailable_reason}.")
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="garvis-workspace"
            )

        def _run():
            if self._backend is None:
                self._backend = self._backend_factory()
            return func(self._backend)

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, _run)
        except WorkspaceError:
            raise
        except (ImportError, NotImplementedError) as exc:
            # pyvda missing, or its COM interface IDs don't match this
            # Windows build -- permanent for this process.
            self._unavailable_reason = str(exc) or type(exc).__name__
            log_extra(logger, 30, "workspace.unavailable", reason=self._unavailable_reason)
            raise WorkspaceError(f"The Garvis workspace is unavailable: {self._unavailable_reason}.") from exc
        except Exception as exc:  # COMError etc. -- e.g. Explorer restarted
            log_extra(logger, 40, "workspace.call_failed", error=repr(exc))
            raise WorkspaceError(f"Virtual desktop operation failed: {exc}") from exc

    async def is_available(self) -> bool:
        try:
            await self.ensure()
        except WorkspaceError:
            return False
        return True

    # ----------------------------------------------------------- desktops
    async def ensure(self) -> DesktopRef:
        """Return the Garvis desktop, finding it by name (it survives Garvis
        restarts) or creating it. Never switches the user's view."""
        async with self._lock:
            def _ensure(backend: VirtualDesktopBackend) -> tuple[DesktopRef, bool]:
                desktops = backend.list_desktops()
                if self._garvis_id:
                    for desktop in desktops:
                        if desktop.id == self._garvis_id:
                            return desktop, False
                for desktop in desktops:
                    if desktop.name == self._name:
                        return desktop, False
                return backend.create_desktop(self._name), True

            desktop, created = await self._call(_ensure)
            self._garvis_id = desktop.id
        if created:
            log_extra(logger, 20, "workspace.created", name=self._name, number=desktop.number)
        return desktop

    async def status(self) -> dict:
        if self._unavailable_reason is not None:
            return {"available": False, "reason": self._unavailable_reason}
        try:
            garvis = await self.ensure()
            current = await self._call(lambda b: b.current_desktop())
        except WorkspaceError as exc:
            return {"available": False, "reason": str(exc)}
        return {
            "available": True,
            "garvis_desktop": {"name": garvis.name, "number": garvis.number},
            "current_desktop": {"name": current.name, "number": current.number},
            "viewing_garvis_desktop": current.id == garvis.id,
        }

    async def show_garvis(self) -> DesktopRef:
        """Switch the user's VIEW to the Garvis desktop (only on explicit request)."""
        garvis = await self.ensure()
        current = await self._call(lambda b: b.current_desktop())
        if current.id != garvis.id:
            self._return_desktop_id = current.id
            await self._call(lambda b: b.switch_to(garvis.id))
            log_extra(logger, 20, "workspace.view_switched", to="garvis", from_number=current.number)
        return garvis

    async def show_primary(self) -> DesktopRef:
        """Switch the view back to where the user was (or the first
        non-Garvis desktop if Garvis never switched it)."""
        garvis = await self.ensure()

        def _target(backend: VirtualDesktopBackend) -> DesktopRef:
            desktops = backend.list_desktops()
            for desktop in desktops:
                if desktop.id == self._return_desktop_id:
                    return desktop
            for desktop in desktops:
                if desktop.id != garvis.id:
                    return desktop
            raise WorkspaceError("There is no primary desktop to switch back to.")

        target = await self._call(_target)
        current = await self._call(lambda b: b.current_desktop())
        if current.id != target.id:
            await self._call(lambda b: b.switch_to(target.id))
            log_extra(logger, 20, "workspace.view_switched", to="primary", number=target.number)
        self._return_desktop_id = None
        return target

    # ------------------------------------------------------------ windows
    async def move_window_to_garvis(self, handle: int) -> DesktopRef:
        garvis = await self.ensure()
        await self._call(lambda b: b.move_window(handle, garvis.id))
        log_extra(logger, 20, "workspace.window_moved", handle=handle, number=garvis.number)
        return garvis

    async def move_window_to_current(self, handle: int) -> DesktopRef:
        """Bring a window to the desktop the user is looking at (e.g. an app
        the user asked Garvis to type into that opened on another virtual
        desktop). Doesn't switch the user's view."""
        current = await self._call(lambda b: b.current_desktop())
        if await self._call(lambda b: b.window_desktop_id(handle)) != current.id:
            await self._call(lambda b: b.move_window(handle, current.id))
            log_extra(logger, 20, "workspace.window_moved_to_current", handle=handle, number=current.number)
        return current

    async def window_location(self, handle: int) -> str:
        """"user" (the desktop being viewed), "garvis_workspace", "other"
        (another of the user's desktops) or "unknown" (no virtual-desktop
        API). Viewing the Garvis desktop itself counts as "user": what the
        user is looking at is the user's desktop."""
        try:
            garvis = await self.ensure()
            current = await self._call(lambda b: b.current_desktop())
            where = await self._call(lambda b: b.window_desktop_id(handle))
        except WorkspaceError:
            return "unknown"
        if where is None or where == current.id:
            return "user"
        return "garvis_workspace" if where == garvis.id else "other"

    async def locate_windows(self, handles: list[int]) -> dict[int, str]:
        """window_location() for many windows in one round trip."""
        try:
            garvis = await self.ensure()

            def _locate(backend: VirtualDesktopBackend) -> dict[int, str]:
                current = backend.current_desktop().id
                found = {}
                for handle in handles:
                    where = backend.window_desktop_id(handle)
                    found[handle] = ("user" if where is None or where == current
                                     else "garvis_workspace" if where == garvis.id else "other")
                return found

            return await self._call(_locate)
        except WorkspaceError:
            return {handle: "unknown" for handle in handles}

    async def is_window_on_garvis(self, handle: int) -> bool:
        garvis = await self.ensure()
        return await self._call(lambda b: b.window_desktop_id(handle)) == garvis.id

    def shutdown(self) -> None:
        """The Garvis desktop itself is deliberately left in place (it's
        reused next launch and may hold windows the user wants to see)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
