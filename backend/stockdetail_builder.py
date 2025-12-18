# backend/stockdetail_cron.py

"""
BullSignalsAI — StockDetail Cron

Purpose:
- Precompute StockDetail payloads
- Store into Firestore
- Make StockDetail screen load instantly
"""

import os
import time
import traceback
from typing import List

from backend.firestore_utils import get_db
from backend.firestore_paths import stockdetail_doc_ref
from backend.stockdetail_builder import build_stockdetail_payload


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEFAULT_LIMIT_CANDLES = int(os.getenv("STOCKDETAIL_LIMIT_CANDLES", "180"))
DEFAULT_TTL_MINUTES = int(os.getenv("STOCKDETAIL_TTL_MINUTES", "15"))
MAX_SYMBOLS_PER_RUN = int(os.getenv("STOCKDETAIL_MAX_SYMBOLS", "120"))

# Example:
# STOCKDETAIL_UNIVERSE="AAPL,TSLA,NVDA,MSFT"
STOCKDETAIL_UNIVERSE = os.getenv("STOCKDETAIL_UNIVERSE", "").strip()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_universe() -> List[str]:
    """
    Returns list of symbols to compute.
    (Later we can expand to watchlists / recently viewed)
    """
    if not STOCKDETAIL_UNIVERSE:
        return []

    symbols = [
        s.strip().upper()
        for s in STOCKDETAIL_UNIVERSE.split(",")
        if s.strip()
    ]

    # de-dupe, preserve order, cap size
    return list(dict.fromkeys(symbols))[:MAX_SYMBOLS_PER_RUN]


def should_skip(existing: dict, force: bool) -> bool:
    """
    Skip if document exists and TTL is still valid.
    """
    if force or not existing:
        return False

    try:
        return existing.get("expiresAt", 0) > int(time.time())
    except Exception:
        return False


# --------------------------------------------------
# MAIN RUNNER
# --------------------------------------------------

def run(force: bool = False, force_grok: bool = False):
    db = get_db()
    symbols = get_universe()

    if not symbols:
        print("⚠️ StockDetail cron: no symbols found (STOCKDETAIL_UNIVERSE empty)")
        return

    print(
        f"🚀 StockDetail cron started | symbols={len(symbols)} "
        f"| force={force} | force_grok={force_grok}"
    )

    ok = 0
    skipped = 0
    failed = 0

    for idx, symbol in enumerate(symbols, start=1):
        t0 = time.time()

        try:
            ref = stockdetail_doc_ref(symbol, db=db)
            snap = ref.get()
            existing = snap.to_dict() if snap.exists else None

            if should_skip(existing, force):
                skipped += 1
                print(f"⏭️ [{idx}/{len(symbols)}] {symbol} skipped (fresh)")
                continue

            payload = build_stockdetail_payload(
                symbol=symbol,
                limit_candles=DEFAULT_LIMIT_CANDLES,
                ttl_minutes=DEFAULT_TTL_MINUTES,
                force_grok=force_grok,
            )

            ref.set(payload, merge=True)

            ok += 1
            ms = int((time.time() - t0) * 1000)
            print(f"✅ [{idx}/{len(symbols)}] {symbol} updated ({ms}ms)")

        except Exception as e:
            failed += 1
            ms = int((time.time() - t0) * 1000)
            print(f"❌ [{idx}/{len(symbols)}] {symbol} failed ({ms}ms): {e}")
            traceback.print_exc()

    print(
        f"✅ StockDetail cron finished | ok={ok} skipped={skipped} failed={failed}"
    )


# --------------------------------------------------
# ENTRYPOINT (Render Cron)
# --------------------------------------------------

if __name__ == "__main__":
    force = os.getenv("STOCKDETAIL_FORCE", "false").lower() == "true"
    force_grok = os.getenv("STOCKDETAIL_FORCE_GROK", "false").lower() == "true"

    run(force=force, force_grok=force_grok)
