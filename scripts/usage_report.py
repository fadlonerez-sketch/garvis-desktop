"""Garvis model usage and estimated spend, from the local ledger (data/usage.db).

    python scripts/usage_report.py              # today, this month, last 14 days, top tasks
    python scripts/usage_report.py --days 31 --top 20

Costs are Garvis's estimates: provider-reported token counts x the prices in
config/models.json. Cache writes that OmniRoute reports without a 5m/1h
split are priced at the 1-hour rate, so estimates err high. Your provider's
console (Anthropic Console > Usage / Cost) is the billing source of truth.
No prompt or reply text is stored, so none is shown.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.llm.usage import UsageLedger  # noqa: E402

_TOTALS = (
    "COUNT(*), COUNT(DISTINCT task_id), SUM(status = 'ok'), SUM(status != 'ok'), "
    "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cache_read_tokens), 0), "
    "COALESCE(SUM(COALESCE(cache_write_5m_tokens, 0) + COALESCE(cache_write_1h_tokens, 0) "
    "+ COALESCE(cache_write_unknown_ttl_tokens, 0)), 0), COALESCE(SUM(cost_usd), 0), "
    "SUM(status = 'ok' AND cost_usd IS NULL)"
)


def _line(label: str, row: tuple) -> str:
    calls, tasks, ok, failed, inp, out, read, write, cost, unpriced = (v or 0 for v in row)
    prompt = inp + read + write
    hit = f"{read / prompt:5.1%}" if prompt else "  n/a"
    per_task = f"{calls / tasks:4.1f}" if tasks else " n/a"
    extra = f"  ({failed} failed)" if failed else ""
    extra += f"  ({unpriced} without usage/price)" if unpriced else ""
    return (
        f"{label:12} ${cost:9.4f}  calls {calls:5} tasks {tasks:4} calls/task {per_task}  "
        f"in {inp:>9,} out {out:>8,} cache-read {read:>9,} ({hit}) cache-write {write:>8,}{extra}"
    )


def report(ledger: UsageLedger, days: int, top: int, now: datetime | None = None) -> str:
    now = now or datetime.now()
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month0 = day0.replace(day=1)
    lines = [f"Garvis model usage -- ledger {ledger.path}", ""]
    for label, since in (("Today", day0), ("This month", month0)):
        lines.append(
            _line(
                label,
                ledger.query(f"SELECT {_TOTALS} FROM llm_calls WHERE ts >= ?", (since.isoformat(timespec="seconds"),))[
                    0
                ],
            )
        )
    lines += ["", f"Last {days} days:"]
    since = (day0 - timedelta(days=days - 1)).isoformat(timespec="seconds")
    for day, *row in ledger.query(
        f"SELECT substr(ts, 1, 10) AS d, {_TOTALS} FROM llm_calls WHERE ts >= ? GROUP BY d ORDER BY d", (since,)
    ):
        lines.append(_line(day, tuple(row)))

    lines += ["", "By category (tier:reason), this month:"]
    month = month0.isoformat(timespec="seconds")
    for category, calls, tasks, cost in ledger.query(
        "SELECT COALESCE(category, '?'), COUNT(*), COUNT(DISTINCT task_id), COALESCE(SUM(cost_usd), 0) "
        "FROM llm_calls WHERE ts >= ? GROUP BY 1 ORDER BY 4 DESC",
        (month,),
    ):
        lines.append(f"  {category:32} ${cost:8.4f}  {tasks:4} tasks  {calls:5} calls")

    lines += ["", f"Most expensive tasks this month (top {top}):"]
    for task_id, started, category, calls, cost, inp, tools in ledger.query(
        "SELECT task_id, MIN(ts), MAX(category), COUNT(*), COALESCE(SUM(cost_usd), 0), "
        "COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(cache_read_tokens), 0), GROUP_CONCAT(tools_requested) "
        "FROM llm_calls WHERE ts >= ? AND task_id IS NOT NULL GROUP BY task_id ORDER BY 5 DESC LIMIT ?",
        (month, top),
    ):
        used = sorted({t for t in (tools or "").split(",") if t})
        lines.append(
            f"  {started}  ${cost:8.5f}  {calls:2} calls  prompt {inp:>8,}  {category or '?':24} "
            f"tools: {', '.join(used) or '-'}  [{task_id}]"
        )

    mismatched = ledger.query("SELECT COUNT(*) FROM llm_calls WHERE status = 'model_mismatch'")[0][0]
    served = ledger.query(
        "SELECT model_requested, COALESCE(model_served, '?'), COUNT(*) FROM llm_calls "
        "WHERE status = 'ok' GROUP BY 1, 2 ORDER BY 3 DESC"
    )
    lines += ["", "Requested -> served model (successful calls):"]
    lines += [f"  {req} -> {srv}: {n}" for req, srv, n in served] or ["  (none yet)"]
    if mismatched:
        lines.append(f"  WARNING: {mismatched} replies came from a different model and were discarded.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    path = Settings().llm_usage_db_path
    if not path.exists():
        print(f"No usage recorded yet ({path} doesn't exist).")
        return 0
    print(report(UsageLedger(path), args.days, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
