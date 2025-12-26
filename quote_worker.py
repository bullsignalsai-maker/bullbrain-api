# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher (30s loop)
# ---------------------------------------------------------

import os
import json
import time
import datetime
from typing import Dict, Any, Set

import firebase_admin
from firebase_admin import credentials, firestore

from quote_provider import (
    fetch_equity_quote,
    fetch_index_snapshot,
    fetch_crypto_snapshot,
    fetch_sector_snapshot,
)

# ---------------------------------------------------------
# Firebase Init (ONCE)
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


def log(msg: str) -> None:
    print(f"[quote-worker] {msg}", flush=True)


# ---------------------------------------------------------
# Collect ALL tickers needing quotes
# ---------------------------------------------------------
def collect_tickers() -> Set[str]:
    tickers: Set[str] = set()

    # Home snapshot
    snap = db.collection("bullsignals_ai").document("homescreen_snapshot").get()
    if snap.exists:
        d = snap.to_dict() or {}

        for m in d.get("mag7", []):
            if m.get("symbol"):
                tickers.add(m["symbol"])

        for card in d.get("carousel", []):
            for it in card.get("items", []):
                label = it.get("label", "")
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip()
                    tickers.add(sym)

    # Hotlist
    hot = db.collection("bullsignals_ai").document("market_hotlist").get()
    if hot.exists:
        for h in hot.to_dict().get("hotlist", []):
            if h.get("symbol"):
                tickers.add(h["symbol"])

    # Bearwatch
    bear = db.collection("bullsignals_ai").document("market_bearwatch").get()
    if bear.exists:
        for b in bear.to_dict().get("bearwatch", []):
            if b.get("symbol"):
                tickers.add(b["symbol"])

    return tickers


# ---------------------------------------------------------
# Update Firestore
# ---------------------------------------------------------
def update_quotes(quotes: Dict[str, Dict[str, Any]]) -> None:
    now = utc_now_iso()

    # Home screen
    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if snap.exists:
        data = snap.to_dict() or {}

        for m in data.get("mag7", []):
            sym = m.get("symbol")
            if sym in quotes:
                m["price"] = quotes[sym]["price"]
                m["changePct"] = quotes[sym]["changePct"]
                m["quote_updated_at"] = now

        for card in data.get("carousel", []):
            for it in card.get("items", []):
                label = it.get("label", "")
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip()
                    if sym in quotes and quotes[sym]["changePct"] is not None:
                        it["value"] = f"{quotes[sym]['changePct']:+.2f}%"

        ref.set(
            {"mag7": data.get("mag7", []), "carousel": data.get("carousel", [])},
            merge=True,
        )

    # Hotlist
    hot_ref = db.collection("bullsignals_ai").document("market_hotlist")
    hot = hot_ref.get()
    if hot.exists:
        d = hot.to_dict()
        for h in d.get("hotlist", []):
            sym = h.get("symbol")
            if sym in quotes:
                h["price"] = quotes[sym]["price"]
                h["changePct"] = quotes[sym]["changePct"]
                h["quote_updated_at"] = now
        hot_ref.set({"hotlist": d.get("hotlist", [])}, merge=True)

    # Bearwatch
    bear_ref = db.collection("bullsignals_ai").document("market_bearwatch")
    bear = bear_ref.get()
    if bear.exists:
        d = bear.to_dict()
        for b in d.get("bearwatch", []):
            sym = b.get("symbol")
            if sym in quotes:
                b["price"] = quotes[sym]["price"]
                b["changePct"] = quotes[sym]["changePct"]
                b["quote_updated_at"] = now
        bear_ref.set({"bearwatch": d.get("bearwatch", [])}, merge=True)


# ---------------------------------------------------------
# Market Overview Update
# ---------------------------------------------------------
def update_market_overview() -> None:
    overview = fetch_index_snapshot()
    overview["crypto"] = fetch_crypto_snapshot()
    overview["sectors"] = fetch_sector_snapshot()
    overview["updated_at"] = utc_now_iso()

    db.collection("bullsignals_ai").document("market_overview_live").set(
        overview, merge=True
    )


# ---------------------------------------------------------
# MAIN LOOP (30s)
# ---------------------------------------------------------
def main():
    log("🚀 Quote worker started (30s loop)")

    while True:
        try:
            tickers = collect_tickers()
            log(f"Refreshing quotes for {len(tickers)} tickers")

            quotes: Dict[str, Dict[str, Any]] = {}

            for sym in sorted(tickers):
                quotes[sym] = fetch_equity_quote(sym)
                time.sleep(0.15)

            update_quotes(quotes)
            update_market_overview()

            log("✅ Quote refresh cycle completed")

        except Exception as e:
            log(f"❌ Worker error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
