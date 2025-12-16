# backend/stockdetail_cron.py

"""
BullSignalsAI — StockDetail Cron (Firestore Precompute)

Purpose:
- Precompute /stockdetail payloads
- Store into Firestore so the mobile UI loads instantly
"""

import os
import time
import traceback
from typing import List

from backend.firestore_paths import get_db, stockdetail_doc_ref
from backend.stockdetail_builder import build_stockdetail_payload

DEFAULT_LIMIT_CANDLES = int(os.getenv("STOCKDETAIL_LIMIT_CANDLES", "180"))
MAX_SYMBOLS_PER_RUN = int(os.getenv("STOCKDETAIL_MAX_SYMBOLS", "120"))

# Example:
# STOCKDETAIL_UNIVERSE="AAPL,TSLA,NVDA,MSFT"
STOCKDETAIL_UNIVERSE = os.getenv("STOCKDETAIL_UNIVERSE", "").strip()


def get_universe() -> List[str]:
    if not STOCKDETAIL_UNIVERSE:
        return []
    syms = [s.strip().upper() for s in STOCKDETAIL_UNIVERSE.split(",") if s.strip()]
    return list(dict.fromkeys(syms))[:MAX_SYMBOLS_PER_RUN]


def should_skip(existing: dict, force: bool) -> bool:
    if force or not existing:
        return False
    try:
        return existing.get("expiresAt", 0) > int(time.time())
    except Exception:
        return False


def run(force: bool = False, force_grok: bool = False):
    db = get_db()
    symbols = get_universe()

    if not symbols:
        print("⚠️ No symbols to process (STOCKDETAIL_UNIVERSE empty)")
        return

    print(f"🚀 StockDetail cron | symbols={len(symbols)} | force={force} | force_grok={force_grok}")

    for i, symbol in enumerate(symbols, 1):
        t0 = time.time()
        try:
            ref = stockdetail_doc_ref(symbol, db=db)
            snap = ref.get()
            existing = snap.to_dict() if snap.exists else None

            if should_skip(existing, force):
                print(f"⏭️ [{i}/{len(symbols)}] {symbol} skip (fresh)")
                continue

            payload = build_stockdetail_payload(
                symbol=symbol,
                force_grok=force_grok,
                limit_candles=DEFAULT_LIMIT_CANDLES,
            )
            ref.set(payload, merge=True)

            ms = int((time.time() - t0) * 1000)
            print(f"✅ [{i}/{len(symbols)}] {symbol} updated ({ms}ms)")

        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            print(f"❌ [{i}/{len(symbols)}] {symbol} failed ({ms}ms): {e}")
            traceback.print_exc()

    print("✅ StockDetail cron finished")


if __name__ == "__main__":
    force = os.getenv("STOCKDETAIL_FORCE", "false").lower() == "true"
    force_grok = os.getenv("STOCKDETAIL_FORCE_GROK", "false").lower() == "true"
    run(force=force, force_grok=force_grok)
