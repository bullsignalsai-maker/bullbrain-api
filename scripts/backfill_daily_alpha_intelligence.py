# scripts/backfill_daily_alpha_intelligence.py

from collections import defaultdict

from backend.market_memory_sheet import get_alpha_opportunities
from backend.daily_alpha_intelligence import save_daily_alpha_session


def main():
    rows = get_alpha_opportunities()

    grouped = defaultdict(list)

    for row in rows:
        market_day = str(row.get("market_day") or "").strip()
        session_type = str(row.get("session_type") or "").strip().upper()

        if not market_day or not session_type:
            continue

        grouped[(market_day, session_type)].append(row)

    print(f"[backfill] total rows={len(rows)} groups={len(grouped)}", flush=True)

    total_saved = 0

    for (market_day, session_type), items in sorted(grouped.items()):
        result = save_daily_alpha_session(
            market_day=market_day,
            session_type=session_type,
            items=items,
            source="sheet_backfill_alpha_opportunities",
        )

        print(f"[backfill] {market_day} {session_type} -> {result}", flush=True)

        if result.get("ok"):
            total_saved += result.get("saved", 0)

    print(f"[backfill] done total_saved={total_saved}", flush=True)


if __name__ == "__main__":
    main()