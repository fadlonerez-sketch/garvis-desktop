import pytest

from app.agent.memory import SQLiteMemoryStore


@pytest.mark.asyncio
async def test_remember_and_recall(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.remember("The user's favorite color is blue")
    await store.remember("The user lives in Tel Aviv")

    all_items = await store.recall()
    assert len(all_items) == 2

    filtered = await store.recall(query="blue")
    assert len(filtered) == 1
    assert "blue" in filtered[0].content


@pytest.mark.asyncio
async def test_forget_removes_item(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    item = await store.remember("temporary fact")
    removed = await store.forget(item.id)
    assert removed is True
    assert await store.recall() == []
