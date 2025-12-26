# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher (30s loop)
#
# Runs as a LONG-RUNNING background worker (NOT cron)
# Refreshes quotes for:
#   - Home screen carousel
#   - MAG7
#   - Hotlist
#   - Bearwatch
#
# Frontend reads Firestore only.
# ---------------------------------------------------------

import time
import datetime
import firebase_admin
from firebase_admin import firestore  # type: ignore
from typing import Dict, Any, Set

import main as backend  # reuse existing quote fetcher


# ---------------------------------------------------------
# Init Firestore (safe)
# ---------------------------------------------------------
def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


# ---------------------------------------------------------
# Time helper
# ---------------------------------------------------------
def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
def log(msg: str) -> None:
    try:
        backend.log(f"[quote-worker] {msg}")
    except Exception:
        pass
    print(f"[quote-worker] {msg}", flush=True)


# ---------------------------------------------------------
# Safe quote fetch (NO THROW)
# ---------------------------------------------------------
def fetch_quote_safe(symbol: str) -> Dict[str, Any]:
    try:
        q = backend.backend_fetch_quote(symbol)
        if isinstance(q, dict):
            return q
    except Exception as e:
        log(f"Quote fetch failed for {symbol}: {e}")
    return {}


def normalize_change_pct(v):
    try:
        v = float(v)
        if abs(v) <= 1.5:  # 0.0086 → 0.86%
            return v * 100.0
        return v
    except Exception:
        return None


# ---------------------------------------------------------
# Collect ALL tickers that need quotes
# ---------------------------------------------------------
def collect_quote_tickers(db) -> Set[str]:
    tickers: Set[str] = set()

    # ---------------------------
    # 1️⃣ Home screen snapshot
    # ---------------------------
    snap = (
        db.collection("bullsignals_ai")
        .document("homescreen_snapshot")
        .get()
    )

    if snap.exists:
        data = snap.to_dict() or {}

        # MAG7
        for item in data.get("mag7", []):
            if isinstance(item, dict) and item.get("symbol"):
                tickers.add(item["symbol"])

        # Carousel proxies
        for card in data.get("carousel", []):
            for it in card.get("items", []):
                label = it.get("label", "")
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip()
                    if sym.isalpha():
                        tickers.add(sym)

    # ---------------------------
    # 2️⃣ Hotlist
    # ---------------------------
    hot = (
        db.collection("bullsignals_ai")
        .document("market_hotlist")
        .get()
    )
    if hot.exists:
        for h in hot.to_dict().get("hotlist", []):
            if h.get("symbol"):
                tickers.add(h["symbol"])

    # ---------------------------
    # 3️⃣ Bearwatch
    # ---------------------------
    bear = (
        db.collection("bullsignals_ai")
        .document("market_bearwatch")
        .get()
    )
    if bear.exists:
        for b in bear.to_dict().get("bearwatch", []):
            if b.get("symbol"):
                tickers.add(b["symbol"])

    return tickers


# ---------------------------------------------------------
# Update Firestore docs with quotes (MERGE SAFE)
# ---------------------------------------------------------
def update_quotes(db, quotes: Dict[str, Dict[str, Any]]) -> None:
    now = utc_now_iso()

    # ---------------------------
    # Home screen snapshot
    # ---------------------------
    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()

    if snap.exists:
        data = snap.to_dict() or {}

        # MAG7 update
        for item in data.get("mag7", []):
            sym = item.get("symbol")
            if sym in quotes:
                item["price"] = quotes[sym].get("price")
                item["changePct"] = quotes[sym].get("changePct")
                item["quote_updated_at"] = now

        # Carousel update
        for card in data.get("carousel", []):
            for it in card.get("items", []):
                label = it.get("label", "")
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip()
                    if sym in quotes:
                        it["value"] = f"{quotes[sym]['changePct']:+.2f}%" if quotes[sym].get("changePct") is not None else "--"

        ref.set(
            {
                "mag7": data.get("mag7", []),
                "carousel": data.get("carousel", []),
            },
            merge=True,
        )

    # ---------------------------
    # Hotlist
    # ---------------------------
    hot_ref = db.collection("bullsignals_ai").document("market_hotlist")
    hot = hot_ref.get()
    if hot.exists:
        data = hot.to_dict() or {}
        for h in data.get("hotlist", []):
            sym = h.get("symbol")
            if sym in quotes:
                h["price"] = quotes[sym].get("price")
                h["changePct"] = quotes[sym].get("changePct")
                h["quote_updated_at"] = now

        hot_ref.set({"hotlist": data.get("hotlist", [])}, merge=True)

    # ---------------------------
    # Bearwatch
    # ---------------------------
    bear_ref = db.collection("bullsignals_ai").document("market_bearwatch")
    bear = bear_ref.get()
    if bear.exists:
        data = bear.to_dict() or {}
        for b in data.get("bearwatch", []):
            sym = b.get("symbol")
            if sym in quotes:
                b["price"] = quotes[sym].get("price")
                b["changePct"] = quotes[sym].get("changePct")
                b["quote_updated_at"] = now

        bear_ref.set({"bearwatch": data.get("bearwatch", [])}, merge=True)


# ---------------------------------------------------------
# MAIN LOOP — every 30 seconds
# ---------------------------------------------------------
def main():
    log("Quote worker started")
    db = get_db()

    while True:
        try:
            tickers = collect_quote_tickers(db)
            log(f"Refreshing quotes for {len(tickers)} tickers")

            quotes: Dict[str, Dict[str, Any]] = {}

            for sym in sorted(tickers):
                q = fetch_quote_safe(sym)
                price = q.get("price") or q.get("close")
                chg = normalize_change_pct(q.get("changePct"))

                quotes[sym] = {
                    "price": price,
                    "changePct": chg,
                }

                time.sleep(0.1)  # gentle throttling

            update_quotes(db, quotes)
            log("Quote refresh cycle completed")

        except Exception as e:
            log(f"Quote worker error: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()
