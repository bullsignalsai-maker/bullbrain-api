# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher (Production)
# Render background worker (long-running)
#
# Refreshes quote-backed fields for Firestore:
#  - MAG7 quotes
#  - Hotlist quotes
#  - Bearwatch quotes
#  - Carousel: proxy-based quotes (SPY/QQQ/GLD/USO/SLV etc.),
#             Crypto card (BTC/ETH/SOL/XRP/DOGE),
#             Sentiment card sync (market_overview.fearGreed),
#             Top Sectors (5th carousel card)
#
# Enhancements:
#  ✅ Market-hours aware throttling (30s open, 5m after-hours)
#  ✅ Night + Weekend throttling
#  ✅ US market holiday pause
#  ✅ "Market Closed" badge on us_market after-hours/weekends/holidays
#
# IMPORTANT:
#  - Safe Firestore merges
#  - No schema breaking
#  - No UI changes required
# ---------------------------------------------------------

import os
import json
import time
import datetime
from typing import Dict, Any, Set, Optional, List

import firebase_admin
from firebase_admin import credentials, firestore

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore

from quote_provider import (
    fetch_equity_quote,
    fetch_crypto_snapshot,
    fetch_sector_snapshot,
)

# ---------------------------------------------------------
# Firebase Init
# ---------------------------------------------------------
def init_firebase() -> None:
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
def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def log(msg: str) -> None:
    print(f"[quote-worker] {msg}", flush=True)


def _get_et_tz():
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    return datetime.timezone(datetime.timedelta(hours=-5))


ET = _get_et_tz()

# ---------------------------------------------------------
# US Market holiday handling (lightweight)
# ---------------------------------------------------------
US_MARKET_HOLIDAYS = {
    # 2025
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
    # 2026
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",  # observed
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
}

_extra = os.getenv("EXTRA_MARKET_HOLIDAYS", "").strip()
if _extra:
    for d in _extra.split(","):
        d = d.strip()
        if d:
            US_MARKET_HOLIDAYS.add(d)


def is_us_market_holiday(now_utc: datetime.datetime) -> bool:
    et = now_utc.astimezone(ET)
    return et.date().isoformat() in US_MARKET_HOLIDAYS


def is_weekend(now_utc: datetime.datetime) -> bool:
    et = now_utc.astimezone(ET)
    return et.weekday() >= 5


def is_market_open(now_utc: datetime.datetime) -> bool:
    """
    Regular NYSE hours only:
      Mon-Fri 9:30am-4:00pm ET
    Holiday/weekends -> closed.
    """
    if is_weekend(now_utc) or is_us_market_holiday(now_utc):
        return False

    et = now_utc.astimezone(ET)
    open_t = et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= et <= close_t


def is_night_et(now_utc: datetime.datetime) -> bool:
    """
    Night throttling window (ET): 8:00pm - 7:00am
    """
    et = now_utc.astimezone(ET)
    h = et.hour
    return (h >= 20) or (h < 7)


def choose_sleep_seconds(now_utc: datetime.datetime) -> int:
    """
    Cost control policy (what you asked):
      - Market open: 30s
      - After hours (weekday, not night): 5 min
      - Night: 30 min
      - Weekend: 2 hours
      - Holiday: 6 hours
    """
    if is_us_market_holiday(now_utc):
        return 6 * 3600
    if is_weekend(now_utc):
        return 2 * 3600
    if is_market_open(now_utc):
        return 30
    if is_night_et(now_utc):
        return 30 * 60
    return 5 * 60


# ---------------------------------------------------------
# Collect tickers (equities/ETFs only)
# ---------------------------------------------------------
def collect_tickers(db) -> Set[str]:
    out: Set[str] = set()

    snap = db.collection("bullsignals_ai").document("homescreen_snapshot").get()
    if snap.exists:
        d = snap.to_dict() or {}

        for m in d.get("mag7", []):
            if isinstance(m, dict) and m.get("symbol"):
                out.add(str(m["symbol"]).upper())

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
                    if sym:
                        out.add(sym)

    for doc in ["market_hotlist", "market_bearwatch"]:
        s = db.collection("bullsignals_ai").document(doc).get()
        if s.exists:
            payload = s.to_dict() or {}
            key = doc.split("_")[-1]  # hotlist / bearwatch
            rows = payload.get(key, [])
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("symbol"):
                        out.add(str(row["symbol"]).upper())

    # Ensure sector ETF proxies are always refreshed (so sector card never shows stale "--")
    for etf in ["XLK", "XLF", "XLE", "XLV", "XLY"]:
        out.add(etf)

    return out


# ---------------------------------------------------------
# Update Firestore Quotes (NO BREAKING CHANGES)
# ---------------------------------------------------------
def update_quotes(db, quotes: Dict[str, Dict[str, Any]]) -> None:
    now = utc_now_iso()

    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if snap.exists:
        d = snap.to_dict() or {}

        mag7_list = d.get("mag7", [])
        if isinstance(mag7_list, list):
            for m in mag7_list:
                if not isinstance(m, dict):
                    continue
                sym = str(m.get("symbol") or "").upper()
                if sym and sym in quotes:
                    m["price"] = quotes[sym].get("price")
                    m["changePct"] = quotes[sym].get("changePct")
                    m["quote_updated_at"] = now

        carousel_list = d.get("carousel", [])
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
                    label = str(it.get("label") or "")
                    if "(" in label and ")" in label:
                        sym = label.split("(")[-1].replace(")", "").strip().upper()
                        q = quotes.get(sym) or {}
                        chg = q.get("changePct")
                        it["value"] = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "--"
                        it["quote_updated_at"] = now

        ref.set(
            {
                "mag7": mag7_list,
                "carousel": carousel_list,
                "quote_refreshed_at": now,
            },
            merge=True,
        )

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
                    sym = str(row.get("symbol") or "").upper()
                    if sym and sym in quotes:
                        row["price"] = quotes[sym].get("price")
                        row["changePct"] = quotes[sym].get("changePct")
                        row["quote_updated_at"] = now
            r.set({key: rows}, merge=True)


# ---------------------------------------------------------
# Market overview update (Crypto + Sentiment + Top Sectors + Market Closed badge)
# ---------------------------------------------------------
def _ensure_card(carousel: List[Dict[str, Any]], card_id: str, default_card: Dict[str, Any]) -> None:
    if any(isinstance(c, dict) and c.get("id") == card_id for c in carousel):
        return
    carousel.append(default_card)


def update_market_overview(db) -> None:
    now_iso = utc_now_iso()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    market_open = is_market_open(now_utc)

    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if not snap.exists:
        return

    d = snap.to_dict() or {}
    carousel = d.get("carousel", [])
    if not isinstance(carousel, list):
        carousel = []

    # Ensure cards exist (no schema break)
    _ensure_card(
        carousel,
        "crypto",
        {"id": "crypto", "title": "Crypto Movers", "subtitle": "24h change", "items": [], "updated_at": now_iso},
    )
    _ensure_card(
        carousel,
        "sentiment",
        {
            "id": "sentiment",
            "title": "Market Sentiment",
            "subtitle": "Fear & Greed (crypto proxy)",
            "items": [{"label": "Mood", "value": "--"}],
            "updated_at": now_iso,
        },
    )
    _ensure_card(
        carousel,
        "sectors",
        {"id": "sectors", "title": "Top Sectors", "subtitle": "ETF performance", "items": [], "updated_at": now_iso},
    )

    # 1) Crypto card ✅
    crypto = fetch_crypto_snapshot(symbols=["BTC", "ETH", "SOL", "XRP", "DOGE"], per_page=10)
    for card in carousel:
        if isinstance(card, dict) and card.get("id") == "crypto":
            items = []
            for sym in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
                chg = crypto.get(sym)
                items.append(
                    {
                        "label": sym,
                        "value": f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "--",
                        "quote_updated_at": now_iso,
                    }
                )
            card["items"] = items
            card["updated_at"] = now_iso

    # 2) Sentiment sync (match market_overview.fearGreed)
    overview = d.get("market_overview", {}) if isinstance(d.get("market_overview"), dict) else {}
    fg = overview.get("fearGreed") if isinstance(overview.get("fearGreed"), dict) else None
    if fg:
        label = fg.get("label", "Neutral")
        value = fg.get("value", 50)
        for card in carousel:
            if isinstance(card, dict) and card.get("id") == "sentiment":
                card["items"] = [{"label": "Mood", "value": f"{label} ({value})"}]
                card["updated_at"] = now_iso

    # 3) Top Sectors (5th card)
    sectors = fetch_sector_snapshot()
    for card in carousel:
        if isinstance(card, dict) and card.get("id") == "sectors":
            items = []
            for name in ["Technology", "Financials", "Energy", "Healthcare", "Consumer"]:
                chg = sectors.get(name)
                items.append(
                    {
                        "label": name,
                        "value": f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "--",
                        "quote_updated_at": now_iso,
                    }
                )
            card["items"] = items
            card["updated_at"] = now_iso

    # 4) Market Closed badge on us_market
    for card in carousel:
        if isinstance(card, dict) and card.get("id") == "us_market":
            if market_open:
                card.pop("badge", None)
            else:
                card["badge"] = "Market Closed"

    ref.set(
        {
            "carousel": carousel,
            "quote_refreshed_at": now_iso,
        },
        merge=True,
    )


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
def main() -> None:
    log("🚀 Quote worker started")

    while True:
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        # Holiday "auto-pause"
        if is_us_market_holiday(now_utc):
            sleep_s = choose_sleep_seconds(now_utc)
            log(f"🎌 US Market Holiday — pausing for {sleep_s}s")
            time.sleep(sleep_s)
            continue

        try:
            sleep_s = choose_sleep_seconds(now_utc)

            tickers = collect_tickers(db)
            log(f"Refreshing quotes for {len(tickers)} tickers | next_sleep={sleep_s}s")

            quotes: Dict[str, Dict[str, Any]] = {}

            # Gentle throttling (avoid API spikes)
            per_symbol_delay = 0.15 if is_market_open(now_utc) else 0.25

            for sym in sorted(tickers):
                quotes[sym] = fetch_equity_quote(sym)
                time.sleep(per_symbol_delay)

            update_quotes(db, quotes)
            update_market_overview(db)

            log("✅ Quote refresh cycle completed")
            time.sleep(sleep_s)

        except Exception as e:
            log(f"❌ Quote worker error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
