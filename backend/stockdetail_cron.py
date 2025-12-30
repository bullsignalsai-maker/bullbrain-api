# backend/stockdetail_cron.py
"""
BullSignalsAI — StockDetail Cron (Firestore Precompute)

Purpose:
- Precompute /stockdetail payloads
- Store into Firestore so the mobile UI loads instantly

Render Cron:
  Command: python -m backend.stockdetail_cron
  Schedule: */15 * * * 1-5   (or whatever you prefer)
"""

import os
import time
import traceback
from typing import List

from backend.firestore_paths import get_db, stockdetail_doc_ref
from backend.stockdetail_builder import build_stockdetail_payload


# ----------------------------
# Config
# ----------------------------
MAX_SYMBOLS_PER_RUN = int(os.getenv("STOCKDETAIL_MAX_SYMBOLS", "120"))

# Example:
# STOCKDETAIL_UNIVERSE="AAPL,TSLA,NVDA,MSFT"
STOCKDETAIL_UNIVERSE = os.getenv("STOCKDETAIL_UNIVERSE", "").strip()


# ----------------------------
# Helpers
# ----------------------------
def get_universe() -> List[str]:
    if not STOCKDETAIL_UNIVERSE:
        return []
    syms = [s.strip().upper() for s in STOCKDETAIL_UNIVERSE.split(",") if s.strip()]
    return list(dict.fromkeys(syms))[:MAX_SYMBOLS_PER_RUN]


import datetime

def should_skip(existing: dict, force: bool) -> bool:
    if force or not existing:
        return False

    expires_at = existing.get("expiresAt")
    if not expires_at:
        return False

    try:
        exp = datetime.datetime.fromisoformat(
            expires_at.replace("Z", "+00:00")
        )
        return exp > datetime.datetime.utcnow()
    except Exception:
        return False


# ----------------------------
# Main Cron Runner
# ----------------------------
def run(force: bool = False, force_grok: bool = False):
    db = get_db()
    symbols = get_universe()

    if not symbols:
        print("⚠️ No symbols to process (STOCKDETAIL_UNIVERSE empty)")
        return

    print(
        f"🚀 StockDetail cron | symbols={len(symbols)} "
        f"| force={force} | force_grok={force_grok}"
    )

    ok = skipped = failed = 0

    for i, symbol in enumerate(symbols, 1):
        t0 = time.time()
        try:
            ref = stockdetail_doc_ref(db, symbol)
            snap = ref.get()
            existing = snap.to_dict() if snap.exists else None

            if should_skip(existing, force):
                skipped += 1
                print(f"⏭️ [{i}/{len(symbols)}] {symbol} skip (fresh)")
                continue

            payload = build_stockdetail_payload(
                symbol=symbol,
                force_grok=force_grok,
            )

            ref.set(payload, merge=True)

            ok += 1
            ms = int((time.time() - t0) * 1000)
            print(f"✅ [{i}/{len(symbols)}] {symbol} updated ({ms}ms)")

        except Exception as e:
            failed += 1
            ms = int((time.time() - t0) * 1000)
            print(f"❌ [{i}/{len(symbols)}] {symbol} failed ({ms}ms): {e}")
            traceback.print_exc()

    print(f"✅ StockDetail cron finished | ok={ok} skipped={skipped} failed={failed}")


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    force = os.getenv("STOCKDETAIL_FORCE", "false").lower() == "true"
    force_grok = os.getenv("STOCKDETAIL_FORCE_GROK", "false").lower() == "true"
    run(force=force, force_grok=force_grok)
