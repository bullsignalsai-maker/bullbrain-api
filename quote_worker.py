# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher
# Runs every 30 seconds (Render background worker)
# ---------------------------------------------------------

import os
import json
import time
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, Any, Set

from quote_provider import (
    fetch_equity_quote,
    fetch_crypto_snapshot,
)

# ---------------------------------------------------------
# Firebase Init
# ---------------------------------------------------------
def init_firebase():
    if firebase_admin._apps:
        return

    raw = os.getenv("FIREBASE_ADMIN_JSON")
    if not raw:
        raise RuntimeError("FIREBASE_ADMIN_JSON missing")

    cred = credentials.Certificate(json.loads(raw))
    firebase_admin.initialize_app(cred)
    print("[quote-worker] 🔥 Firebase initialized")


init_firebase()
db = firestore.client()
print("[quote-worker] ✅ Firestore client ready")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def log(msg: str):
    print(f"[quote-worker] {msg}", flush=True)


# ---------------------------------------------------------
# Collect all tickers to refresh
# ---------------------------------------------------------
def collect_tickers(db) -> Set[str]:
    out: Set[str] = set()

    snap = db.collection("bullsignals_ai").document("homescreen_snapshot").get()
    if snap.exists:
        d = snap.to_dict() or {}

        for m in d.get("mag7", []):
            if isinstance(m, dict) and m.get("symbol"):
                out.add(m["symbol"])

        for card in d.get("carousel", []):
            for it in card.get("items", []):
                label = it.get("label", "")
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip()
                    out.add(sym)

    for doc in ["market_hotlist", "market_bearwatch"]:
        s = db.collection("bullsignals_ai").document(doc).get()
        if s.exists:
            for row in s.to_dict().get(doc.split("_")[-1], []):
                if row.get("symbol"):
                    out.add(row["symbol"])

    return out


# ---------------------------------------------------------
# Update Firestore Quotes (NO BREAKING CHANGES)
# ---------------------------------------------------------
def update_quotes(db, quotes: Dict[str, Dict[str, Any]]):
    now = utc_now_iso()

    # Home screen
    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if snap.exists:
        d = snap.to_dict() or {}

        for m in d.get("mag7", []):
            sym = m.get("symbol")
            if sym in quotes:
                m["price"] = quotes[sym].get("price")
                m["changePct"] = quotes[sym].get("changePct")
                m["quote_updated_at"] = now

        for card in d.get("carousel", []):
            for it in card.get("items", []):
                label = it.get("label", "")
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip()
                    q = quotes.get(sym, {})
                    chg = q.get("changePct")
                    it["value"] = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "--"
                    it["quote_updated_at"] = now

        ref.set(
            {
                "mag7": d.get("mag7", []),
                "carousel": d.get("carousel", []),
                "quote_refreshed_at": now,
            },
            merge=True,
        )

    # Hotlist / Bearwatch
    for name in ["market_hotlist", "market_bearwatch"]:
        r = db.collection("bullsignals_ai").document(name)
        s = r.get()
        if s.exists:
            key = "hotlist" if "hot" in name else "bearwatch"
            rows = s.to_dict().get(key, [])
            for row in rows:
                sym = row.get("symbol")
                if sym in quotes:
                    row["price"] = quotes[sym].get("price")
                    row["changePct"] = quotes[sym].get("changePct")
                    row["quote_updated_at"] = now
            r.set({key: rows}, merge=True)


# ---------------------------------------------------------
# Market overview update
# ---------------------------------------------------------
def update_market_overview(db):
    crypto = fetch_crypto_snapshot()
    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if not snap.exists:
        return

    d = snap.to_dict() or {}
    for card in d.get("carousel", []):
        if card.get("id") == "crypto":
            for it in card.get("items", []):
                sym = it.get("label")
                chg = crypto.get(sym)
                it["value"] = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "--"
                it["quote_updated_at"] = utc_now_iso()

    ref.set({"carousel": d.get("carousel", [])}, merge=True)


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
def main():
    log("🚀 Quote worker started (30s loop)")

    while True:
        try:
            tickers = collect_tickers(db)
            log(f"Refreshing quotes for {len(tickers)} tickers")

            quotes: Dict[str, Dict[str, Any]] = {}

            for sym in sorted(tickers):
                quotes[sym] = fetch_equity_quote(sym)
                time.sleep(0.15)

            update_quotes(db, quotes)
            update_market_overview(db)

            log("✅ Quote refresh cycle completed")

        except Exception as e:
            log(f"❌ Quote worker error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
