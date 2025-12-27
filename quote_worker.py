# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher
# Runs as a Render BACKGROUND WORKER (long-running process)
#
# What it updates:
#   - homescreen_snapshot.mag7 (price/changePct/quote_updated_at)
#   - homescreen_snapshot.carousel (US market + commodities + crypto + sentiment)
#   - homescreen_snapshot.market_overview.top_sectors (new field, safe)
#   - market_hotlist + market_bearwatch (price/changePct/quote_updated_at)
#
# Market-aware cadence:
#   - Market open (Mon-Fri 9:30-16:00 ET): 30 sec
#   - After hours (16:00-20:00 ET): 15 min
#   - Night (20:00-9:30 ET weekdays): 60 min
#   - Weekend: 6 hours
# ---------------------------------------------------------

import os
import json
import time
import datetime
from zoneinfo import ZoneInfo
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, Any, Set, Optional

from quote_provider import (
    fetch_equity_quote,
    fetch_crypto_snapshot,
    fetch_sector_snapshot,
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
    print("[quote-worker] 🔥 Firebase initialized", flush=True)


init_firebase()
db = firestore.client()
print("[quote-worker] ✅ Firestore client ready", flush=True)

# ---------------------------------------------------------
# Time helpers
# ---------------------------------------------------------
NY = ZoneInfo("America/New_York")
MARKET_OPEN = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)
AFTER_HOURS_CLOSE = datetime.time(20, 0)


def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def now_et() -> datetime.datetime:
    return datetime.datetime.now(NY)


def is_weekday(dt: datetime.datetime) -> bool:
    return dt.weekday() < 5  # Mon–Fri


def is_market_open(dt: datetime.datetime) -> bool:
    return is_weekday(dt) and MARKET_OPEN <= dt.time() < MARKET_CLOSE


def is_after_hours(dt: datetime.datetime) -> bool:
    return is_weekday(dt) and MARKET_CLOSE <= dt.time() < AFTER_HOURS_CLOSE


def compute_cadence_seconds(dt: datetime.datetime) -> int:
    """
    Market open: 30s
    After hours: 15min
    Night: 60min
    Weekend: 6h
    """
    if is_market_open(dt):
        return 30
    if is_after_hours(dt):
        return 15 * 60
    if is_weekday(dt):
        return 60 * 60
    return 6 * 60 * 60


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
def log(msg: str):
    print(f"[quote-worker] {msg}", flush=True)


def percent_str(x: Optional[float], digits: int = 2) -> str:
    try:
        if x is None:
            return "--"
        return f"{float(x):+.{digits}f}%"
    except Exception:
        return "--"


# ---------------------------------------------------------
# Collect all tickers to refresh (stocks/ETFs only)
# ---------------------------------------------------------
def collect_tickers(db) -> Set[str]:
    out: Set[str] = set()

    snap = db.collection("bullsignals_ai").document("homescreen_snapshot").get()
    if snap.exists:
        d = snap.to_dict() or {}

        # MAG7 symbols
        for m in d.get("mag7", []):
            if isinstance(m, dict) and m.get("symbol"):
                out.add(str(m["symbol"]).strip().upper())

        # Carousel items that look like "S&P 500 (SPY)"
        for card in d.get("carousel", []):
            items = card.get("items", [])
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                label = str(it.get("label", ""))
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip().upper()
                    if sym:
                        out.add(sym)

    # Hotlist and Bearwatch symbols
    for doc in ["market_hotlist", "market_bearwatch"]:
        s = db.collection("bullsignals_ai").document(doc).get()
        if s.exists:
            key = "hotlist" if "hot" in doc else "bearwatch"
            rows = (s.to_dict() or {}).get(key, [])
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("symbol"):
                        out.add(str(row["symbol"]).strip().upper())

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

        # MAG7 update
        for m in d.get("mag7", []):
            if not isinstance(m, dict):
                continue
            sym = (m.get("symbol") or "").strip().upper()
            if sym in quotes:
                m["price"] = quotes[sym].get("price")
                m["changePct"] = quotes[sym].get("changePct")
                m["quote_updated_at"] = now

        # Carousel update for items with "(TICKER)" format
        for card in d.get("carousel", []):
            if not isinstance(card, dict):
                continue
            items = card.get("items", [])
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                label = str(it.get("label", ""))
                if "(" in label and ")" in label:
                    sym = label.split("(")[-1].replace(")", "").strip().upper()
                    q = quotes.get(sym, {})
                    chg = q.get("changePct")
                    it["value"] = percent_str(chg)
                    it["quote_updated_at"] = now

        ref.set(
            {
                "mag7": d.get("mag7", []),
                "carousel": d.get("carousel", []),
                "quote_refreshed_at": now,
            },
            merge=True,
        )

    # Hotlist / Bearwatch update
    for name in ["market_hotlist", "market_bearwatch"]:
        r = db.collection("bullsignals_ai").document(name)
        s = r.get()
        if s.exists:
            key = "hotlist" if "hot" in name else "bearwatch"
            rows = (s.to_dict() or {}).get(key, [])
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    sym = (row.get("symbol") or "").strip().upper()
                    if sym in quotes:
                        row["price"] = quotes[sym].get("price")
                        row["changePct"] = quotes[sym].get("changePct")
                        row["quote_updated_at"] = now
            r.set({key: rows}, merge=True)


# ---------------------------------------------------------
# Market overview update (Crypto + Sentiment sync + Sectors)
# ---------------------------------------------------------
def update_market_overview(db):
    """
    Updates ONLY dynamic quote-driven parts of the homescreen:
      - Crypto carousel (BTC, ETH, SOL, XRP, DOGE) via CoinGecko
      - Keeps sentiment card aligned with market_overview.fearGreed
      - Adds/updates market_overview.top_sectors (ETF proxy % changes)

    This function:
      ✔ DOES NOT create the entire carousel from scratch
      ✔ DOES NOT touch MAG7 signals
      ✔ DOES NOT conflict with market_cron
    """

    now = utc_now_iso()

    # Fetch crypto + sectors
    crypto = fetch_crypto_snapshot()
    sectors = fetch_sector_snapshot()

    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if not snap.exists:
        return

    d = snap.to_dict() or {}
    carousel = d.get("carousel", [])
    if not isinstance(carousel, list):
        carousel = []

    # ----------------------------
    # 1) Update / rebuild CRYPTO card
    # ----------------------------
    for card in carousel:
        if isinstance(card, dict) and card.get("id") == "crypto":
            card["items"] = []
            for sym in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
                chg = crypto.get(sym)
                card["items"].append(
                    {
                        "label": sym,
                        "value": percent_str(chg),
                        "quote_updated_at": now,
                    }
                )
            card["updated_at"] = now

    # ----------------------------
    # 2) Keep SENTIMENT card aligned to overview.fearGreed
    # ----------------------------
    overview = d.get("market_overview", {})
    if not isinstance(overview, dict):
        overview = {}

    fg = overview.get("fearGreed")
    if isinstance(fg, dict):
        label = fg.get("label", "Neutral")
        value = fg.get("value", 50)
        for card in carousel:
            if isinstance(card, dict) and card.get("id") == "sentiment":
                card["items"] = [{"label": "Mood", "value": f"{label} ({value})"}]
                card["updated_at"] = now

    # ----------------------------
    # 3) Attach sector snapshot to market_overview (SAFE)
    #    This does NOT change UI unless UI reads it later.
    # ----------------------------
    overview["top_sectors"] = {
        name: (v if isinstance(v, (int, float)) else None)
        for name, v in (sectors or {}).items()
    }
    overview["sectors_updated_at"] = now

    # Persist minimal changes (NO schema break)
    ref.set(
        {
            "carousel": carousel,
            "market_overview": overview,
            "quote_refreshed_at": now,
        },
        merge=True,
    )


# ---------------------------------------------------------
# Main loop (market-aware cadence)
# ---------------------------------------------------------
def main():
    log("🚀 Quote worker started (market-aware cadence)")

    while True:
        dt = now_et()
        cadence = compute_cadence_seconds(dt)

        mode = (
            "MARKET_OPEN" if is_market_open(dt)
            else "AFTER_HOURS" if is_after_hours(dt)
            else "NIGHT" if is_weekday(dt)
            else "WEEKEND"
        )

        try:
            log(f"⏱ Mode={mode} | cadence={cadence}s | ET={dt.isoformat()}")

            # Stocks/ETFs refresh only when useful
            if mode in ("MARKET_OPEN", "AFTER_HOURS"):
                tickers = collect_tickers(db)
                log(f"Refreshing STOCK/ETF quotes for {len(tickers)} tickers")

                quotes: Dict[str, Dict[str, Any]] = {}
                for sym in sorted(tickers):
                    quotes[sym] = fetch_equity_quote(sym)
                    time.sleep(0.15)

                update_quotes(db, quotes)
            else:
                log("Skipping stock refresh (market closed)")

            # Crypto + sentiment sync + sectors (lightweight)
            update_market_overview(db)

            log("✅ Quote refresh cycle completed")

        except Exception as e:
            log(f"❌ Quote worker error: {e}")

        time.sleep(cadence)


if __name__ == "__main__":
    main()
