"""Persistent memory: facts the user has explicitly asked Garvis to remember.

Deliberately simple for v0.1 -- a SQLite table with substring search, no
vector database. `MemoryStore` is an abstraction so the backend can be
swapped later if a concrete need for semantic recall shows up.
"""

from __future__ import annotations

import asyncio
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


class MemoryItem(BaseModel):
    id: int
    content: str
    created_at: str


class MemoryStore(ABC):
    @abstractmethod
    async def remember(self, content: str) -> MemoryItem: ...

    @abstractmethod
    async def recall(self, query: str | None = None, limit: int = 20) -> list[MemoryItem]: ...

    @abstractmethod
    async def forget(self, item_id: int) -> bool: ...


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    async def remember(self, content: str) -> MemoryItem:
        return await asyncio.to_thread(self._remember_sync, content)

    def _remember_sync(self, content: str) -> MemoryItem:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute("INSERT INTO memory (content, created_at) VALUES (?, ?)", (content, created_at))
            item_id = cursor.lastrowid
        return MemoryItem(id=item_id, content=content, created_at=created_at)

    async def recall(self, query: str | None = None, limit: int = 20) -> list[MemoryItem]:
        return await asyncio.to_thread(self._recall_sync, query, limit)

    def _recall_sync(self, query: str | None, limit: int) -> list[MemoryItem]:
        with self._connect() as conn:
            if query:
                rows = conn.execute(
                    "SELECT id, content, created_at FROM memory WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, content, created_at FROM memory ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [MemoryItem(id=r[0], content=r[1], created_at=r[2]) for r in rows]

    async def forget(self, item_id: int) -> bool:
        return await asyncio.to_thread(self._forget_sync, item_id)

    def _forget_sync(self, item_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memory WHERE id = ?", (item_id,))
            return cursor.rowcount > 0
