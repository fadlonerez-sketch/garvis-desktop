"""Read-only summary of the local LLM usage ledger for the dashboard.

The ledger (app/llm/usage.py:UsageLedger, data/usage.db) records every model
request with provider-reported token counts and Garvis's own cost *estimate*
(tokens x config/models.json prices). This only reads it -- opened in
SQLite read-only mode, so a missing ledger is reported, never created -- and
nothing here is a bill: the Anthropic Console is authoritative.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

_RECENT_LIMIT = 12
_TREND_DAYS = 14


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def read_usage_summary(db_path: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    if not Path(db_path).exists():
        return {"available": False, "reason": "No model requests recorded yet (the usage ledger doesn't exist)."}
    day = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
    month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
    with closing(_connect_read_only(Path(db_path))) as conn:
        def totals(since: str) -> dict:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(status = 'ok'), 0), COALESCE(SUM(status != 'ok'), 0), "
                "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
                "COALESCE(SUM(cache_read_tokens), 0), "
                "COALESCE(SUM(COALESCE(cache_write_5m_tokens, 0) + COALESCE(cache_write_1h_tokens, 0) "
                "+ COALESCE(cache_write_unknown_ttl_tokens, 0)), 0), "
                "COALESCE(SUM(cost_usd), 0), COUNT(DISTINCT task_id) FROM llm_calls WHERE ts >= ?",
                (since,),
            ).fetchone()
            keys = ("calls", "ok", "errors", "input_tokens", "output_tokens", "cache_read_tokens",
                    "cache_write_tokens", "cost_usd", "tasks")
            return dict(zip(keys, row, strict=True))

        recent = [
            dict(zip(("ts", "tier", "model", "status", "error_type", "input_tokens", "output_tokens",
                      "cache_read_tokens", "cost_usd", "latency_ms"), row, strict=True))
            for row in conn.execute(
                "SELECT ts, tier, COALESCE(model_served, model_requested), status, error_type, input_tokens, "
                "output_tokens, cache_read_tokens, cost_usd, latency_ms FROM llm_calls ORDER BY ts DESC, id DESC LIMIT ?",
                (_RECENT_LIMIT,),
            )
        ]
        first_day = (now - timedelta(days=_TREND_DAYS - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        per_day = {
            row[0]: (row[1], row[2])
            for row in conn.execute(
                "SELECT substr(ts, 1, 10), COUNT(*), COALESCE(SUM(cost_usd), 0) FROM llm_calls "
                "WHERE ts >= ? GROUP BY substr(ts, 1, 10)",
                (first_day.isoformat(timespec="seconds"),),
            )
        }
        daily = []
        for offset in range(_TREND_DAYS):  # days without requests are real zeros, not gaps
            key = (first_day + timedelta(days=offset)).strftime("%Y-%m-%d")
            calls, cost = per_day.get(key, (0, 0.0))
            daily.append({"day": key, "calls": calls, "cost_usd": cost})
        return {"available": True, "today": totals(day), "month": totals(month), "recent": recent,
                "daily": daily, "read_at": now.isoformat(timespec="seconds")}
