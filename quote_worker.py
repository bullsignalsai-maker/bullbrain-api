# quote_worker.py
# ---------------------------------------------------------
# BullSignalsAI — Central Quote Refresher (Production)
# Render background worker (long-running)
#
# Refreshes quote-backed fields for Firestore:
#  - MAG7 quotes
#  - Hotlist quotes
#  - Bearwatch quotes
#  - Carousel: US market, commodities (from existing values w/ parentheses),
#             Crypto card (BTC/ETH/SOL/XRP/DOGE),
#             Sentiment card sync (from market_overview.fearGreed),
#             Top Sectors card (5th carousel card)
#
# Enhancements:
#  ✅ Market-hours aware throttling (30s open, 5m after-hours)
#  ✅ Weekend + Night throttling
#  ✅ US market holiday pause
#  ✅ "Market Closed" badge on us_market card after-hours/weekends/holidays
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
from typing import Dict, Any, Set, List

import firebase_admin
from firebase_admin import credentials, firestore

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore

from quote_provider import (
    fetch_equity_quote,
    fetch_crypto_simple_snapshot,
    fetch_sector_snapshot,
)
from backend.quote_repo import (
    get_pending_quotes,
    save_quote,
    clear_needs_refresh,
    mark_needs_refresh,   # ✅ ADD THIS
)


# -----------------------------
# Refresh policies
# -----------------------------
CRYPTO_MIN_REFRESH_SECONDS = 1800   # 30 minutes
SECTOR_MIN_REFRESH_SECONDS = 300   # 5 minutes (market hours only)


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

def _seconds_since(ts_iso: str | None) -> int | None:
    try:
        if not ts_iso:
            return None
        ts = datetime.datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return int((datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds())
    except Exception:
        return None

# ---------------------------------------------------------
# US Market holiday handling
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
    # 2026 (common NYSE holidays)
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
    return (et.hour >= 20) or (et.hour < 7)


def choose_sleep_seconds(now_utc: datetime.datetime) -> int:
    """
    Cost control policy:
      - Market open: 30s
      - After hours (weekday): 5 min   ✅ (your request)
      - Night (weekday): 1 hour
      - Weekend: 6 hours
      - Holiday: 6 hours
    """
    if is_us_market_holiday(now_utc):
        return 6 * 3600
    if is_weekend(now_utc):
        return 6 * 3600
    if is_market_open(now_utc):
        return 30
    if is_night_et(now_utc):
        return 3600
    return 5 * 60


# ---------------------------------------------------------
# Collect tickers (equities/ETFs only)
# ---------------------------------------------------------
def collect_tickers(db) -> Set[str]:
    out: Set[str] = set()

    snap = db.collection("bullsignals_ai").document("homescreen_snapshot").get()
    if snap.exists:
        d = snap.to_dict() or {}

        # MAG7 list contains symbols directly
        for m in d.get("mag7", []):
            if isinstance(m, dict) and m.get("symbol"):
                out.add(str(m["symbol"]).upper())

        # Carousel proxies (only labels like "S&P 500 (SPY)" etc.)
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

    # Hotlist / Bearwatch tickers
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

    return out

# ---------------------------------------------------------
# Collect on-demand quote refresh requests
# ---------------------------------------------------------
def collect_on_demand_quotes(db) -> Set[str]:
    """
    Returns symbols explicitly requested via quote_demand.ensure_quote()
    BUT filters out non-equity symbols (BTC/ETH/etc) to avoid Finnhub empty quotes.
    """
    try:
        pending = get_pending_quotes()
        out: Set[str] = set()

        for sym in pending:
            if not sym:
                continue

            s = str(sym).upper().strip()

            # ✅ equities/ETFs are usually A-Z only (OPEN, NBIS, SPY, QQQ)
            # ❌ filter out crypto / weird ids (BTC, ETH, SOL, XRP, DOGE)
            if s in {"BTC", "ETH", "SOL", "XRP", "DOGE"}:
                continue

            # optional extra guard: reject anything with non letters
            if not s.replace(".", "").isalpha():
                continue

            out.add(s)

        return out

    except Exception as e:
        log(f"⚠️ Failed to read pending quotes: {e}")
        return set()


# ---------------------------------------------------------
# Update Firestore Quotes (NO BREAKING CHANGES)
# ---------------------------------------------------------
def update_quotes(db, quotes: Dict[str, Dict[str, Any]]) -> None:
    now = utc_now_iso()

    # -----------------------------------------------------
    # HOME SCREEN
    # -----------------------------------------------------
    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()

    if snap.exists:
        d = snap.to_dict() or {}

        # -----------------------------
        # MAG7 quote update (SAFE)
        # -----------------------------
        mag7_list = d.get("mag7", [])
        if isinstance(mag7_list, list):
            for m in mag7_list:
                if not isinstance(m, dict):
                    continue

                sym = str(m.get("symbol") or "").upper()
                q = quotes.get(sym)

                if not q:
                    continue

                # ✅ overwrite ONLY if valid
                if isinstance(q.get("price"), (int, float)):
                    m["price"] = q["price"]

                if isinstance(q.get("changePct"), (int, float)):
                    m["changePct"] = q["changePct"]

                m["quote_updated_at"] = now

        # -------------------------------------------------
        # CAROUSEL (SPY / QQQ / GLD / USO / SLV)
        # ❌ NEVER overwrite with "--"
        # -------------------------------------------------
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
                    if "(" not in label or ")" not in label:
                        continue

                    sym = (
                        label.split("(")[-1]
                        .replace(")", "")
                        .strip()
                        .upper()
                    )

                    q = quotes.get(sym)
                    if not q:
                        continue

                    chg = q.get("changePct")
                    price = q.get("price")

                    chg_ok = isinstance(chg, (int, float))
                    price_ok = isinstance(price, (int, float))

                    # ✅ Update % only when available
                    if chg_ok:
                        it["value"] = f"{chg:+.2f}%"

                    # ✅ Attach full quote if ANY useful data exists
                    if chg_ok or price_ok:
                        it["quote"] = {
                            "price": price,
                            "change": q.get("change"),
                            "changePct": chg,
                            "open": q.get("open"),
                            "high": q.get("high"),
                            "low": q.get("low"),
                            "prevClose": q.get("prevClose"),
                            "timestamp": q.get("timestamp"),
                            "source": q.get("source"),
                        }

                        it["quote_updated_at"] = now


                    # ❌ else → preserve existing value

        ref.set(
            {
                "mag7": mag7_list,
                "carousel": carousel_list,
                "quote_refreshed_at": now,
            },
            merge=True,
        )

    # -----------------------------------------------------
    # HOTLIST / BEARWATCH (SAFE UPDATE)
    # -----------------------------------------------------
    for name in ["market_hotlist", "market_bearwatch"]:
        r = db.collection("bullsignals_ai").document(name)
        s = r.get()

        if not s.exists:
            continue

        key = "hotlist" if "hot" in name else "bearwatch"
        rows = (s.to_dict() or {}).get(key, [])

        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue

            sym = str(row.get("symbol") or "").upper()
            q = quotes.get(sym)

            if not q:
                continue

            if isinstance(q.get("price"), (int, float)):
                row["price"] = q["price"]

            if isinstance(q.get("changePct"), (int, float)):
                row["changePct"] = q["changePct"]

            row["quote_updated_at"] = now

        r.set({key: rows}, merge=True)


# ---------------------------------------------------------
# Market overview update (Crypto + Sentiment sync + Top Sectors + Market Closed badge)
# ---------------------------------------------------------
def _ensure_card_after(
    carousel: List[Dict[str, Any]],
    card_id: str,
    default_card: Dict[str, Any],
    after_id: str,
) -> None:
    """
    Ensures card exists.
    If missing, insert right after `after_id` card if present, else append.
    This guarantees 'Top Sectors' becomes the 5th card for your current structure.
    """
    if any(isinstance(c, dict) and c.get("id") == card_id for c in carousel):
        return

    insert_at = None
    for i, c in enumerate(carousel):
        if isinstance(c, dict) and c.get("id") == after_id:
            insert_at = i + 1
            break

    if insert_at is None:
        carousel.append(default_card)
    else:
        carousel.insert(insert_at, default_card)

def ensure_homescreen_base(d: Dict[str, Any], now_iso: str) -> Dict[str, Any]:
    """
    Ensures REQUIRED homescreen schema always exists.
    Never overwrites populated data.
    """

    # -----------------------------
    # Market overview (header data)
    # -----------------------------
    if not isinstance(d.get("market_overview"), dict):
        d["market_overview"] = {
            "marketStatus": "Unknown",
            "marketMood": "Unknown",
            "risk_level": "Unknown",
            "fearGreed": None,
            "updated_at": now_iso,
        }

    # -----------------------------
    # Carousel base
    # -----------------------------
    carousel = d.get("carousel")
    if not isinstance(carousel, list):
        carousel = []
        d["carousel"] = carousel

    def has(card_id: str) -> bool:
        return any(isinstance(c, dict) and c.get("id") == card_id for c in carousel)

    # 1️⃣ US Market
    if not has("us_market"):
        carousel.insert(
            0,
            {
                "id": "us_market",
                "title": "US Market",
                "subtitle": "S&P 500 / Nasdaq",
                "items": [],
                "updated_at": now_iso,
            },
        )

    # 2️⃣ Commodities
    if not has("commodities"):
        carousel.insert(
            1,
            {
                "id": "commodities",
                "title": "Commodities",
                "subtitle": "Gold, Oil, Silver",
                "items": [],
                "updated_at": now_iso,
            },
        )

    # 3️⃣ Crypto
    if not has("crypto"):
        carousel.append(
            {
                "id": "crypto",
                "title": "Crypto Movers",
                "subtitle": "24h change",
                "items": [],
                "updated_at": now_iso,
            }
        )

    # 4️⃣ Sentiment
    if not has("sentiment"):
        carousel.append(
            {
                "id": "sentiment",
                "title": "Market Sentiment",
                "subtitle": "Fear & Greed (crypto proxy)",
                "items": [{"label": "Mood", "value": "--"}],
                "updated_at": now_iso,
            }
        )

    # 5️⃣ Sectors
    if not has("sectors"):
        carousel.append(
            {
                "id": "sectors",
                "title": "Top Sectors",
                "subtitle": "ETF performance",
                "items": [],
                "updated_at": now_iso,
            }
        )

    return d

# -------------------------------------------------
# Ensure crypto quote docs always exist (SEED)
# -------------------------------------------------
CRYPTO_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

for sym in CRYPTO_SYMBOLS:
    ref = (
        db.collection("bullsignals_ai")
          .document("quotes")
          .collection("symbols")
          .document(sym)
    )
    if not ref.get().exists:
        save_quote(
            sym,
            {
                "symbol": sym,
                "source": "crypto",
                "needs_refresh": True,
            },
        )

def update_market_overview(db) -> None:
    now_iso = utc_now_iso()
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    ref = db.collection("bullsignals_ai").document("homescreen_snapshot")
    snap = ref.get()
    if not snap.exists:
        return

    d = snap.to_dict() or {}
    d = ensure_homescreen_base(d, now_iso)
    carousel = d["carousel"]
    if not isinstance(carousel, list):
        carousel = []

    # Ensure crypto card exists (do NOT overwrite if already exists)
    if not any(isinstance(c, dict) and c.get("id") == "crypto" for c in carousel):
        carousel.append(
            {
                "id": "crypto",
                "title": "Crypto Movers",
                "subtitle": "24h change",
                "items": [],
                "updated_at": now_iso,
            }
        )

    # Ensure sentiment card exists
    if not any(isinstance(c, dict) and c.get("id") == "sentiment" for c in carousel):
        carousel.append(
            {
                "id": "sentiment",
                "title": "Market Sentiment",
                "subtitle": "Fear & Greed (crypto proxy)",
                "items": [{"label": "Mood", "value": "--"}],
                "updated_at": now_iso,
            }
        )

    # ✅ Ensure sectors card exists and is 5th card (insert after commodities)
    _ensure_card_after(
        carousel,
        "sectors",
        {
            "id": "sectors",
            "title": "Top Sectors",
            "subtitle": "ETF performance",
            "items": [],
            "updated_at": now_iso,
        },
        after_id="commodities",
    )
    
    # -----------------------------
    # 1) Crypto update (RATE-LIMIT SAFE)
    # -----------------------------

    
    from backend.quote_repo import get_quote_safe

    btc = get_quote_safe("BTC") or {}
    age = _seconds_since(btc.get("updated_at"))

    # ✅ throttle crypto refresh
    if age is not None and age < CRYPTO_MIN_REFRESH_SECONDS:
        crypto = {}
    else:
        crypto = fetch_crypto_simple_snapshot()

    # ✅ Persist crypto quotes to Firestore
    for sym, data in crypto.items():
        if not isinstance(data, dict):
            continue

        payload = {
            "symbol": sym,
            "source": "crypto",
        }

        if isinstance(data.get("price"), (int, float)):
            payload["price"] = data["price"]

        if isinstance(data.get("changePct"), (int, float)):
            payload["changePct"] = data["changePct"]

        # 🔒 Only save if at least ONE numeric value exists
        if len(payload) > 2:
            save_quote(sym, payload)



    symbols = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

    for card in carousel:
        if isinstance(card, dict) and card.get("id") == "crypto":
            items = []

            for sym in symbols:
                q = get_quote_safe(sym) or {}
                # ✅ Prefer freshly fetched value (same run), fallback to Firestore
                fresh = crypto.get(sym) if isinstance(crypto.get(sym), dict) else {}
                chg = fresh.get("changePct")

                if not isinstance(chg, (int, float)):
                    chg = q.get("changePct")

                items.append(
                    {
                        "label": sym,
                        "value": f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "--",
                        "quote": q,
                        "quote_updated_at": q.get("updated_at"),
                    }
                )

            card["items"] = items
            card["updated_at"] = now_iso

    # -----------------------------
    # 2) Sentiment sync (from market_overview.fearGreed)
    # -----------------------------
    overview = d.get("market_overview", {}) if isinstance(d.get("market_overview"), dict) else {}
    fg = overview.get("fearGreed") if isinstance(overview.get("fearGreed"), dict) else None

    if fg:
        label = fg.get("label", "Neutral")
        value = fg.get("value", 50)

        for card in carousel:
            if isinstance(card, dict) and card.get("id") == "sentiment":
                card.setdefault("title", "Market Sentiment")
                card.setdefault("subtitle", "Fear & Greed (crypto proxy)")
                card["items"] = [{"label": "Mood", "value": f"{label} ({value})"}]
                card["updated_at"] = now_iso

    # -----------------------------
    # 3) Top Sectors update (MARKET HOURS ONLY)
    # -----------------------------
    if not is_market_open(now_utc):
        log("⏸️ Market closed — skipping sector refresh")
    else:
        sectors = fetch_sector_snapshot()

        has_valid_sectors = any(
            isinstance(v, (int, float)) for v in sectors.values()
        )

        if not has_valid_sectors:
            log("⚠️ Sector snapshot empty — preserving existing carousel values")
        else:
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

    # -----------------------------
    # 4) Market Closed badge on us_market card
    # -----------------------------
    market_open = is_market_open(now_utc)

    for card in carousel:
        if isinstance(card, dict) and card.get("id") == "us_market":
            if market_open:
                card.pop("badge", None)
            else:
                card["badge"] = "Market Closed"

    # Persist (safe merge)
    ref.set(
        {
            "carousel": carousel,
            "quote_refreshed_at": now_iso,
            "market_overview": d.get("market_overview"),
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
        sleep_seconds = choose_sleep_seconds(now_utc)

        try:
            # -------------------------------------------------
            # LIGHT MODE — holidays / weekends
            # -------------------------------------------------
            if is_us_market_holiday(now_utc) or is_weekend(now_utc):
                reason = "holiday" if is_us_market_holiday(now_utc) else "weekend"
                log(f"⏸️ Market closed ({reason}) — light refresh then sleep={sleep_seconds}s")

                # still safe to run (badge + crypto + sectors)
                update_market_overview(db)
                time.sleep(sleep_seconds)
                continue

            # -------------------------------------------------
            # Collect symbols (THIS is the fix)
            #   1) homescreen tickers (MAG7 + SPY/QQQ/GLD/USO/SLV + hotlist/bearwatch)
            #   2) on-demand pending (needs_refresh == True)
            #   3) active symbols (recent watchlist adds)
            # -------------------------------------------------
            tickers = collect_tickers(db)
            on_demand = collect_on_demand_quotes(db)

            # OPTIONAL but strongly recommended if you have this collection:
            # - active_symbols keeps system scalable and guarantees new adds get picked up
            active = set()
            try:
                s = db.collection("bullsignals_ai").document("active_symbols").get()
                if s.exists:
                    d = s.to_dict() or {}
                    # expect: { "symbols": ["OPEN","NBIS",...], "updated_at": ... }
                    syms = d.get("symbols", [])
                    if isinstance(syms, list):
                        active = {str(x).upper() for x in syms if x}
            except Exception as e:
                log(f"⚠️ Failed to read active_symbols: {e}")
                active = set()

            all_tickers = set()
            all_tickers |= tickers
            all_tickers |= on_demand
            all_tickers |= active

            log(
                f"Refreshing quotes | "
                f"homescreen={len(tickers)} "
                f"pending={len(on_demand)} "
                f"active={len(active)} "
                f"total={len(all_tickers)} | "
                f"next_sleep={sleep_seconds}s"
            )

            quotes: Dict[str, Dict[str, Any]] = {}
            per_symbol_delay = 0.15 if is_market_open(now_utc) else 0.25

            # -------------------------------------------------
            # SAFE QUOTE FETCH LOOP (NO POISONING)
            # -------------------------------------------------
            CRYPTO_SYMBOLS = {"BTC", "ETH", "SOL", "XRP", "DOGE"}

            for sym in sorted(all_tickers):
                try:
                    # ❌ NEVER fetch crypto via Finnhub
                    if sym in CRYPTO_SYMBOLS:
                        continue

                    q = fetch_equity_quote(sym)


                    price = q.get("price") if isinstance(q, dict) else None
                    chg = q.get("changePct") if isinstance(q, dict) else None

                    # Empty quote → keep needs_refresh true (retry later)
                    if price is None and chg is None:
                        log(f"⚠️ Empty quote for {sym} — will retry")
                        mark_needs_refresh(sym)
                        time.sleep(per_symbol_delay)
                        continue

                    # Valid quote → persist
                    quotes[sym] = q
                    save_quote(sym, q)

                    # clear refresh flag ONLY on success
                    if sym in on_demand:
                        clear_needs_refresh(sym)

                except Exception as e:
                    log(f"⚠️ Quote fetch failed for {sym}: {e}")
                    mark_needs_refresh(sym)

                time.sleep(per_symbol_delay)

            # Apply UI updates
            update_quotes(db, quotes)
            update_market_overview(db)

            log("✅ Quote refresh cycle completed")
            time.sleep(sleep_seconds)

        except Exception as e:
            log(f"❌ Quote worker error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
