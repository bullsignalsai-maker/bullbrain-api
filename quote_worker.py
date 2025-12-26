# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher (30s loop)
# ---------------------------------------------------------

import os
import sys
import json
import time
import datetime
from typing import Dict, Any, Set

# ✅ Ensure this script's folder is on sys.path (prevents import issues on Render)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import firebase_admin
from firebase_admin import credentials, firestore

# ✅ Import from sibling file quote_provider.py
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

    # IMPORTANT: FIREBASE_ADMIN_JSON must be valid JSON with double quotes
    cred = credentials.Certificate(json.loads(raw))
    firebase_admin.initialize_app(cred)
    print("[quote-worker] 🔥 Firebase initialized", flush=True)


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
def collect_tickers(db) -> Set[str]:
    tickers: Set[str] = set()

    # Home snapshot
    snap = db.collection("bullsignals_ai").document("homescreen_snapshot").get()
    if snap.exists:
        d = snap.to_dict() or {}

        # MAG7
        for m in d.get("mag7", []):
            if isinstance(m, dict) and m.get("symbol"):
                tickers.add(m["symbol"])

        # Carousel proxies like "S&P 500 (SPY)"
        for card in d.get("carousel", []):
            if not isinstance(card, dict):
                continue
            for it in card.get("items", []):
                if not isinstance(it, dict):
                    continue
                label = it.get("label", "")
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip()
                    if sym:
                        tickers.add(sym)

    # Hotlist
    hot = db.collection("bullsignals_ai").document("market_hotlist").get()
    if hot.exists:
        for h in (hot.to_dict() or {}).get("hotlist", []):
            if isinstance(h, dict) and h.get("symbol"):
                tickers.add(h["symbol"])

    # Bearwatch
    bear = db.collection("bullsignals_ai").document("market_bearwatch").get()
    if bear.exists:
        for b in (bear.to_dict() or {}).get("bearwatch", []):
            if isinstance(b, dict) and b.get("symbol"):
                tickers.add(b["symbol"])

    return tickers


# ---------------------------------------------------------
# Update Firestore
# ---------------------------------------------------------
def update_quotes(db, quotes: Dict[str, Dict[str, Any]]) -> None:
    now = utc_now_iso()

    # Home screen
    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if snap.exists:
        data = snap.to_dict() or {}

        # MAG7 update
        mag7_list = data.get("mag7", [])
        if isinstance(mag7_list, list):
            for m in mag7_list:
                if not isinstance(m, dict):
                    continue
                sym = m.get("symbol")
                if sym in quotes:
                    m["price"] = quotes[sym].get("price")
                    m["changePct"] = quotes[sym].get("changePct")
                    m["quote_updated_at"] = now

        # Carousel update (value is a string like +1.23%)
        carousel_list = data.get("carousel", [])
        if isinstance(carousel_list, list):
            for card in carousel_list:
                if not isinstance(card, dict):
                    continue
                items = card.get("items", [])
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    label = it.get("label", "")
                    if "(" in label and ")" in label:
                        sym = label.split("(")[-1].replace(")", "").strip()
                        q = quotes.get(sym) or {}
                        chg = q.get("changePct")
                        it["value"] = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "--"

        ref.set({"mag7": mag7_list, "carousel": carousel_list}, merge=True)

    # Hotlist
    hot_ref = db.collection("bullsignals_ai").document("market_hotlist")
    hot = hot_ref.get()
    if hot.exists:
        d = hot.to_dict() or {}
        hotlist = d.get("hotlist", [])
        if isinstance(hotlist, list):
            for h in hotlist:
                if not isinstance(h, dict):
                    continue
                sym = h.get("symbol")
                if sym in quotes:
                    h["price"] = quotes[sym].get("price")
                    h["changePct"] = quotes[sym].get("changePct")
                    h["quote_updated_at"] = now
        hot_ref.set({"hotlist": hotlist}, merge=True)

    # Bearwatch
    bear_ref = db.collection("bullsignals_ai").document("market_bearwatch")
    bear = bear_ref.get()
    if bear.exists:
        d = bear.to_dict() or {}
        bearwatch = d.get("bearwatch", [])
        if isinstance(bearwatch, list):
            for b in bearwatch:
                if not isinstance(b, dict):
                    continue
                sym = b.get("symbol")
                if sym in quotes:
                    b["price"] = quotes[sym].get("price")
                    b["changePct"] = quotes[sym].get("changePct")
                    b["quote_updated_at"] = now
        bear_ref.set({"bearwatch": bearwatch}, merge=True)


# ---------------------------------------------------------
# Market Overview Update
# ---------------------------------------------------------
def update_market_overview(db) -> None:
    overview = fetch_index_snapshot()
    crypto = fetch_crypto_snapshot()
    sectors = fetch_sector_snapshot()

    doc = {
        **overview,
        "crypto": crypto,
        "sectors": sectors,
        "updated_at": utc_now_iso(),
    }

    db.collection("bullsignals_ai").document("market_overview_live").set(doc, merge=True)


# ---------------------------------------------------------
# MAIN LOOP (30s)
# ---------------------------------------------------------
def main():
    init_firebase()
    db = firestore.client()
    log("✅ quote_worker Firestore client ready")
    log("🚀 Quote worker started (30s loop)")

    while True:
        cycle = utc_now_iso()
        try:
            tickers = collect_tickers(db)
            log(f"Refreshing quotes for {len(tickers)} tickers | cycle={cycle}")

            quotes: Dict[str, Dict[str, Any]] = {}
            for sym in sorted(tickers):
                quotes[sym] = fetch_equity_quote(sym)
                time.sleep(0.15)  # gentle throttling

            update_quotes(db, quotes)
            update_market_overview(db)

            log(f"✅ Quote refresh cycle completed | tickers={len(tickers)}")

        except Exception as e:
            log(f"❌ Worker error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
