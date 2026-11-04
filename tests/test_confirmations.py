import asyncio

import pytest

from app.core.exceptions import ConfirmationDeniedError, ConfirmationTimeoutError
from app.security.confirmations import ConfirmationManager


@pytest.mark.asyncio
async def test_confirmation_approved():
    manager = ConfirmationManager(default_timeout_seconds=5)
    request = manager.create_request("send_email", {"to": "a@b.com"}, "high risk")

    async def approve_soon():
        await asyncio.sleep(0.05)
        manager.resolve(request.id, True)

    asyncio.get_event_loop().create_task(approve_soon())
    approved = await manager.wait_for_decision(request.id)
    assert approved is True


@pytest.mark.asyncio
async def test_confirmation_denied_raises():
    manager = ConfirmationManager(default_timeout_seconds=5)
    request = manager.create_request("delete_file", {}, "high risk")

    async def deny_soon():
        await asyncio.sleep(0.05)
        manager.resolve(request.id, False)

    asyncio.get_event_loop().create_task(deny_soon())
    with pytest.raises(ConfirmationDeniedError):
        await manager.wait_for_decision(request.id)


@pytest.mark.asyncio
async def test_confirmation_timeout_raises():
    manager = ConfirmationManager(default_timeout_seconds=0.05)
    request = manager.create_request("delete_file", {}, "high risk")
    with pytest.raises(ConfirmationTimeoutError):
        await manager.wait_for_decision(request.id)
