"""Tests for the Garvis workspace (app/desktop/workspace.py) and the
switch_workspace tool, against an in-memory virtual-desktop backend."""

from __future__ import annotations

import threading
import uuid

import pytest

from app.core.exceptions import WorkspaceError
from app.desktop.workspace import DesktopRef, VirtualDesktopBackend, WorkspaceManager
from app.tools.workspace import SwitchWorkspaceInput, SwitchWorkspaceTool


class FakeVirtualDesktops(VirtualDesktopBackend):
    def __init__(self, names=("",)) -> None:
        self.desktops = [DesktopRef(id=str(uuid.uuid4()), number=i + 1, name=n) for i, n in enumerate(names)]
        self.current_id = self.desktops[0].id
        self.window_desktops: dict[int, str] = {}
        self.threads: set[int] = set()
        self.switches: list[int] = []

    def _touch(self):
        self.threads.add(threading.get_ident())

    def list_desktops(self):
        self._touch()
        return list(self.desktops)

    def current_desktop(self):
        self._touch()
        return next(d for d in self.desktops if d.id == self.current_id)

    def create_desktop(self, name):
        self._touch()
        ref = DesktopRef(id=str(uuid.uuid4()), number=len(self.desktops) + 1, name=name)
        self.desktops.append(ref)
        return ref

    def switch_to(self, desktop_id):
        self._touch()
        self.current_id = desktop_id
        self.switches.append(next(d.number for d in self.desktops if d.id == desktop_id))

    def move_window(self, handle, desktop_id):
        self._touch()
        self.window_desktops[handle] = desktop_id

    def window_desktop_id(self, handle):
        self._touch()
        return self.window_desktops.get(handle, self.desktops[0].id)


@pytest.fixture
def vd():
    return FakeVirtualDesktops()


@pytest.fixture
def workspace(vd):
    manager = WorkspaceManager(backend_factory=lambda: vd, name="Garvis")
    yield manager
    manager.shutdown()


async def test_ensure_creates_named_desktop_without_switching_view(workspace, vd):
    garvis = await workspace.ensure()
    assert garvis.name == "Garvis" and garvis.number == 2
    assert vd.current_id == vd.desktops[0].id and vd.switches == []
    assert (await workspace.ensure()).id == garvis.id  # idempotent
    assert len(vd.desktops) == 2


async def test_existing_garvis_desktop_is_reused_across_restarts():
    vd = FakeVirtualDesktops(names=("", "Work", "Garvis"))
    manager = WorkspaceManager(backend_factory=lambda: vd, name="Garvis")
    try:
        assert (await manager.ensure()).number == 3
        assert len(vd.desktops) == 3
    finally:
        manager.shutdown()


async def test_all_backend_calls_run_on_one_dedicated_thread(workspace, vd):
    await workspace.ensure()
    await workspace.status()
    await workspace.move_window_to_garvis(7)
    assert len(vd.threads) == 1
    assert threading.get_ident() not in vd.threads


async def test_show_garvis_then_primary_returns_to_where_the_user_was():
    vd = FakeVirtualDesktops(names=("", "Work"))
    vd.current_id = vd.desktops[1].id  # user is on desktop 2
    manager = WorkspaceManager(backend_factory=lambda: vd, name="Garvis")
    try:
        await manager.show_garvis()
        assert (await manager.status())["viewing_garvis_desktop"] is True
        await manager.show_primary()
        assert vd.current_id == vd.desktops[1].id
        assert vd.switches == [3, 2]
    finally:
        manager.shutdown()


async def test_show_primary_without_prior_switch_goes_to_first_non_garvis(workspace, vd):
    await workspace.show_primary()  # already on desktop 1: nothing to do
    assert vd.switches == []


async def test_move_window_and_membership(workspace):
    assert not await workspace.is_window_on_garvis(42)
    await workspace.move_window_to_garvis(42)
    assert await workspace.is_window_on_garvis(42)


async def test_unsupported_interface_marks_workspace_unavailable_permanently():
    calls = []

    def factory():
        calls.append(1)
        raise NotImplementedError(
            "Interface IVirtualDesktopManagerInternal not supported for windows version 10.0.99999"
        )

    manager = WorkspaceManager(backend_factory=factory)
    try:
        assert await manager.is_available() is False
        status = await manager.status()
        assert status["available"] is False and "not supported" in status["reason"]
        with pytest.raises(WorkspaceError, match="unavailable"):
            await manager.ensure()
        assert calls == [1]  # no retry storm
    finally:
        manager.shutdown()


async def test_non_windows_workspace_is_unavailable():
    manager = WorkspaceManager(backend_factory=None)
    assert (await manager.status()) == {"available": False, "reason": "not supported on this platform"}


async def test_transient_backend_error_is_wrapped(workspace, vd):
    def boom(*_):
        raise OSError("COM call failed: explorer restarted")

    vd.create_desktop = boom
    with pytest.raises(WorkspaceError, match="explorer restarted"):
        await workspace.ensure()


# ------------------------------------------------------------------- tool
async def test_switch_workspace_tool_round_trip(workspace, vd):
    tool = SwitchWorkspaceTool(workspace)
    shown = await tool.execute(SwitchWorkspaceInput(target="garvis"))
    assert shown.success and shown.data["viewing_garvis_desktop"] is True
    back = await tool.execute(SwitchWorkspaceInput(target="primary"))
    assert back.success and back.data["viewing_garvis_desktop"] is False
    status = await tool.execute(SwitchWorkspaceInput(target="status"))
    assert status.data["garvis_desktop"] == {"name": "Garvis", "number": 2}


async def test_switch_workspace_tool_reports_unavailable():
    tool = SwitchWorkspaceTool(WorkspaceManager(backend_factory=None))
    result = await tool.execute(SwitchWorkspaceInput(target="garvis"))
    assert not result.success and "unavailable" in result.error
