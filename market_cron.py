# =========================================================
# BullSignalsAI — Market Intelligence Cron
#
# Schedule:
#   */15 * * * 1-5
#
# Responsibilities:
#   1) Refresh global stock intelligence (/stocks/{SYMBOL})
#   2) Use user relevance (active_symbols)
#   3) Rotate SP500 discovery (discovery_cursor)
#   4) Always include MAG-7
#   5) Build Hotlist + BearWatch
#   6) Update Homescreen snapshot (MAG-7 only)  ✅ (unchanged contract)
#
# ✅ ADDITION (RESTORE OLD BEHAVIOR, NO UI CHANGES):
#   7) Ensure Homescreen carousel baseline cards exist:
#        - us_market (SPY/QQQ)
#        - commodities (GLD/USO/SLV)
#        - crypto, sentiment, sectors (kept if already there)
#   8) Populate market_overview fields so UI header is NOT "unknown":
#        - marketStatus
#        - marketMood
#        - fearGreed (simple proxy)
#
# DOES NOT:
#   - Fetch news
#   - Do per-article sentiment
#   - Change UI contract
# =========================================================

import datetime
import math
import random
import time
from typing import Dict, Any, List, Optional

import firebase_admin
from firebase_admin import firestore

import main as backend
from symbols_clean import REAL_TICKERS, COMPANY_NAMES

from backend.candle_store import get_candles
from backend.bull_insights import generate_bull_insights

# ✅ Reuse your central quote provider (Finnhub)
# (safe: if FINNHUB_KEY missing, it returns {})
from quote_provider import fetch_equity_quote

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore


# =========================================================
# CONSTANTS
# =========================================================

MAG7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

ACTIVE_SYMBOL_LIMIT = 60
DISCOVERY_LIMIT = 50
TOTAL_SCAN_LIMIT = 120

DISCOVERY_SHARDS = 8

COL_ROOT = "bullsignals_ai"
COL_STOCKS = "stocks"
DOC_ACTIVE = "active_symbols"
DOC_DISCOVERY = "discovery_cursor"
DOC_HOTLIST = "market_hotlist"
DOC_BEARWATCH = "market_bearwatch"
DOC_HOMESCREEN = "homescreen_snapshot"


# =========================================================
# FIRESTORE / TIME HELPERS
# =========================================================

def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# =========================================================
# MARKET HOURS HELPERS (for header)
# =========================================================

def _get_et_tz():
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    return datetime.timezone(datetime.timedelta(hours=-5))


ET = _get_et_tz()

# Keep same list as quote_worker (safe + deterministic)
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


def is_us_market_holiday(now_utc: datetime.datetime) -> bool:
    et = now_utc.astimezone(ET)
    return et.date().isoformat() in US_MARKET_HOLIDAYS


def is_market_open(now_utc: datetime.datetime) -> bool:
    """
    Regular NYSE hours only:
      Mon-Fri 9:30am-4:00pm ET
    """
    et = now_utc.astimezone(ET)
    if et.weekday() >= 5:
        return False
    if is_us_market_holiday(now_utc):
        return False

    open_t = et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= et <= close_t


# =========================================================
# MODEL LOADER
# =========================================================

def ensure_bullbrain_loaded():
    if backend.bullbrain_model is None:
        backend.bullbrain_model = backend.load_bullbrain_model()


# =========================================================
# ACTIVE SYMBOLS (READ-ONLY)
# =========================================================

def load_active_symbols() -> Dict[str, Dict[str, Any]]:
    db = get_db()
    snap = db.collection(COL_ROOT).document(DOC_ACTIVE).get()
    if not snap.exists:
        return {}
    return (snap.to_dict() or {}).get("symbols", {})


def rank_active_symbols(active: Dict[str, Dict[str, Any]]) -> List[str]:
    now = datetime.datetime.now(datetime.timezone.utc)

    def score(meta: Dict[str, Any]) -> float:
        base = float(meta.get("count", 0))
        last_seen = meta.get("last_seen")
        boost = 0.0
        try:
            if last_seen:
                ts = datetime.datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                mins = (now - ts).total_seconds() / 60.0
                if mins < 60:
                    boost = (60 - mins) / 60.0
        except Exception:
            pass
        return base + boost

    ranked = sorted(
        active.items(),
        key=lambda kv: score(kv[1]),
        reverse=True,
    )

    return [sym for sym, _ in ranked[:ACTIVE_SYMBOL_LIMIT]]


# =========================================================
# DISCOVERY CURSOR (CRON-OWNED)
# =========================================================

def get_discovery_symbols() -> List[str]:
    db = get_db()
    ref = db.collection(COL_ROOT).document(DOC_DISCOVERY)

    snap = ref.get()
    data = snap.to_dict() or {}
    shard_index = int(data.get("shard_index", 0))

    total = len(REAL_TICKERS)
    shard_size = math.ceil(total / DISCOVERY_SHARDS)

    start = shard_index * shard_size
    end = min(start + shard_size, total)

    shard = REAL_TICKERS[start:end][:DISCOVERY_LIMIT]

    next_index = (shard_index + 1) % DISCOVERY_SHARDS
    ref.set(
        {
            "shard_index": next_index,
            "total_shards": DISCOVERY_SHARDS,
            "updated_at": utc_now_iso(),
        },
        merge=True,
    )

    return shard


# =========================================================
# SCAN UNIVERSE
# =========================================================

def build_scan_universe() -> List[str]:
    active = rank_active_symbols(load_active_symbols())
    discovery = get_discovery_symbols()

    universe = list(dict.fromkeys(MAG7 + active + discovery))
    return universe[:TOTAL_SCAN_LIMIT]


# =========================================================
# PER-SYMBOL COMPUTE
# =========================================================
def compute_symbol(symbol: str) -> Dict[str, Any] | None:
    candles = get_candles(symbol, min_points=120)
    if not candles:
        return None

    feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles)
    if feats_vec is None:
        return None

    infer = backend.bullbrain_infer(feats_vec)
    if infer is None:
        return None

    prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
    prob_down = float(infer.get("probability_down") or (1.0 - prob_up))

    if prob_up >= 0.58:
        signal = "BUY"
    elif prob_down >= 0.58:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(max(prob_up, prob_down) * 100.0, 2)

    # -------------------------
    # ✅ TECHNICAL SNAPSHOT
    # -------------------------
    technical = build_technical_snapshot(
        symbol=symbol,
        feat=feat_dict,
        last_close=feat_dict.get("close"),
    )

    # -------------------------
    # ✅ SMART PATTERN
    # -------------------------
    smart_pattern = None
    pattern_stats = None

    try:
        quote_for_pattern = {
            "price": feat_dict.get("close"),
            "changePct": feat_dict.get("return_1d"),
        }

        sp = detect_smart_pattern(
            feat_dict,
            quote_for_pattern,
            technical,
        )

        if sp and sp.get("pattern") != "NO CLEAR PATTERN":
            hist = scan_smart_pattern_history(symbol, candles) or {}
            smart_pattern = sp
            pattern_stats = hist

    except Exception:
        pass

    # -------------------------
    # INSIGHTS (unchanged)
    # -------------------------
    insights = generate_bull_insights(
        symbol=symbol,
        features=feat_dict,
        bullbrain={
            "signal": signal,
            "confidence": confidence,
            "prob_up": prob_up,
            "prob_down": prob_down,
        },
        technical=technical,
        seed_key=f"{symbol}:{utc_now_iso()}",
    )

    doc = {
        "symbol": symbol,
        "company_name": COMPANY_NAMES.get(symbol, symbol),

        # CORE
        "bullbrain": {
            "signal": signal,
            "confidence": confidence,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
        },
        "features_meta": feat_dict,
        "insights": insights,

        # ✅ NEW (critical)
        "technical": technical,
        "smartPattern": smart_pattern,
        "patternStats": pattern_stats,

        # META
        "computed_at": utc_now_iso(),
        "schema_version": "v1",
    }

    db = get_db()
    db.collection(COL_ROOT).document(COL_STOCKS) \
        .collection("symbols").document(symbol) \
        .set(doc, merge=True)

    return doc

# =========================================================
# PHASE 2 — HOTLIST + BEARWATCH
# =========================================================

def build_hotlist_bearwatch(results: List[Dict[str, Any]]):
    buys = []
    sells = []

    for r in results:
        bb = r.get("bullbrain", {})
        if bb.get("signal") == "BUY":
            buys.append(r)
        elif bb.get("signal") == "SELL":
            sells.append(r)

    buys.sort(key=lambda x: x["bullbrain"]["confidence"], reverse=True)
    sells.sort(key=lambda x: x["bullbrain"]["confidence"], reverse=True)

    return buys[:5], sells[:5]


def save_market_lists(hotlist, bearwatch):
    db = get_db()

    db.collection(COL_ROOT).document(DOC_HOTLIST).set(
        {
            "count": len(hotlist),
            "hotlist": hotlist,
            "updated_at": utc_now_iso(),
        },
        merge=True,
    )

    db.collection(COL_ROOT).document(DOC_BEARWATCH).set(
        {
            "count": len(bearwatch),
            "bearwatch": bearwatch,
            "updated_at": utc_now_iso(),
        },
        merge=True,
    )


# =========================================================
# ✅ RESTORE OLD BEHAVIOR: Homescreen Market Overview + Baseline Carousel Cards
# =========================================================

def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):+.2f}%"
    except Exception:
        return "--"


def _get_quote_change_pct(symbol: str) -> Optional[float]:
    """
    Safe wrapper around Finnhub quote_provider.
    Returns changePct (float) or None.
    """
    try:
        q = fetch_equity_quote(symbol)
        chg = q.get("changePct") if isinstance(q, dict) else None
        return float(chg) if isinstance(chg, (int, float)) else None
    except Exception:
        return None


def ensure_market_overview_and_baseline_carousel():
    """
    - Ensures 'us_market' and 'commodities' cards exist WITH items (labels with parentheses)
      so quote_worker can refresh them continuously.
    - Writes market_overview so UI header is NOT unknown.
    - DOES NOT remove/overwrite your existing carousel cards.
    """
    db = get_db()
    now_iso = utc_now_iso()
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    ref = db.collection(COL_ROOT).document(DOC_HOMESCREEN)
    snap = ref.get()
    data = snap.to_dict() or {}

    carousel = data.get("carousel", [])
    if not isinstance(carousel, list):
        carousel = []

    # ---- Build baseline items (with labels containing "(SYMBOL)")
    # These labels are IMPORTANT because quote_worker extracts symbol from parentheses.
    spy = _get_quote_change_pct("SPY")
    qqq = _get_quote_change_pct("QQQ")
    gld = _get_quote_change_pct("GLD")
    uso = _get_quote_change_pct("USO")
    slv = _get_quote_change_pct("SLV")

    us_market_items = [
        {"label": "S&P 500 (SPY)", "value": _fmt_pct(spy), "quote_updated_at": now_iso},
        {"label": "Nasdaq (QQQ)", "value": _fmt_pct(qqq), "quote_updated_at": now_iso},
    ]
    commodities_items = [
        {"label": "Gold (GLD)", "value": _fmt_pct(gld), "quote_updated_at": now_iso},
        {"label": "Oil (USO)", "value": _fmt_pct(uso), "quote_updated_at": now_iso},
        {"label": "Silver (SLV)", "value": _fmt_pct(slv), "quote_updated_at": now_iso},
    ]

    # ---- Ensure card exists or update items if it exists
    def upsert_card(card_id: str, title: str, subtitle: str, items: List[Dict[str, Any]]):
        nonlocal carousel
        found = False
        for c in carousel:
            if isinstance(c, dict) and c.get("id") == card_id:
                # Keep title/subtitle if already present, but ensure items exist
                c.setdefault("title", title)
                c.setdefault("subtitle", subtitle)
                # Update items if empty or missing
                if not isinstance(c.get("items"), list) or len(c.get("items")) == 0:
                    c["items"] = items
                c["updated_at"] = now_iso
                found = True
                break
        if not found:
            carousel.append(
                {
                    "id": card_id,
                    "title": title,
                    "subtitle": subtitle,
                    "items": items,
                    "updated_at": now_iso,
                }
            )

    upsert_card("us_market", "US Market", "S&P 500 / Nasdaq", us_market_items)
    upsert_card("commodities", "Commodities", "Gold, Oil, Silver", commodities_items)

    # ---- Market Overview for UI header (NOT unknown)
    market_open = is_market_open(now_utc)
    market_status = "Market Open" if market_open else "Market Closed"

    # Simple, stable mood proxy (based on SPY/QQQ average)
    vals = [v for v in [spy, qqq] if isinstance(v, (int, float))]
    avg = sum(vals) / len(vals) if vals else None

    if avg is None:
        mood = "Unknown"
        fg_label, fg_val = "Neutral", 50
    else:
        if avg >= 0.50:
            mood = "Bullish"
            fg_label, fg_val = "Greed", 65
        elif avg <= -0.50:
            mood = "Risk-Off"
            fg_label, fg_val = "Fear", 35
        else:
            mood = "Neutral"
            fg_label, fg_val = "Neutral", 50

    # Write market_overview (quote_worker reads fearGreed)
    market_overview = {
        "marketStatus": market_status,
        "marketMood": mood,
        "fearGreed": {
            "label": fg_label,
            "value": fg_val,
        },
        "updated_at": now_iso,
    }

    # Persist safely
    ref.set(
        {
            "carousel": carousel,
            "market_overview": market_overview,
            "updated_at": data.get("updated_at", now_iso),
            "version": data.get("version", "v1"),
        },
        merge=True,
    )


# =========================================================
# MAG-7 HOMESCREEN SNAPSHOT (UNCHANGED CONTRACT)
# =========================================================

def build_mag7_snapshot():
    db = get_db()
    items = []

    for sym in MAG7:
        snap = (
            db.collection(COL_ROOT)
              .document(COL_STOCKS)
              .collection("symbols")
              .document(sym)
              .get()
        )
        if snap.exists:
            items.append(snap.to_dict())

    return items


def save_homescreen_snapshot():
    db = get_db()
    db.collection(COL_ROOT).document(DOC_HOMESCREEN).set(
        {
            "mag7": build_mag7_snapshot(),
            "updated_at": utc_now_iso(),
            "version": "v1",
        },
        merge=True,
    )


# =========================================================
# ENTRYPOINT
# =========================================================

def main():
    ensure_bullbrain_loaded()

    scan_symbols = build_scan_universe()
    results = []

    for sym in scan_symbols:
        try:
            r = compute_symbol(sym)
            if r:
                results.append(r)
        except Exception:
            pass

        time.sleep(random.uniform(0.15, 0.25))

    hotlist, bearwatch = build_hotlist_bearwatch(results)
    save_market_lists(hotlist, bearwatch)

    # ✅ Keep your existing homescreen MAG7 snapshot behavior
    save_homescreen_snapshot()

    # ✅ Restore old behavior: market header + US Market + Commodities baseline
    # (safe merge; does not remove anything)
    try:
        ensure_market_overview_and_baseline_carousel()
    except Exception:
        pass


if __name__ == "__main__":
    main()
