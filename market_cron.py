# market_cron.py
# =========================================================
# BullSignalsAI — Market Intelligence Cron
#
# Schedule:
#   */15 * * * 1-5
#
# Responsibilities:
#   1) Refresh stock intelligence for MAG7 + active symbols
#   2) Include market movers (Phase-0 gainers/losers)
#   3) Use user relevance (active_symbols)
#   4) Persist intelligence ONLY in stocks collection
#   5) Maintain homescreen market_overview + baseline carousel
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
# HARDENING (THIS PATCH):
#   ✅ Add FULL logging visibility (no silent failures)
#   ✅ Guarantee stock docs repopulate after accidental deletion
#   ✅ Write an index marker doc under bullsignals_ai/stocks (optional but useful)
#   ✅ Defensive checks for NaN/invalid feature vectors
# =========================================================

import datetime
import math
import random
import time
import traceback
import os
import requests
from typing import Dict, Any, List, Optional, Tuple

import firebase_admin
from firebase_admin import firestore

import main as backend
from symbols_clean import COMPANY_NAMES, REAL_TICKERS
from backend.market_momentum import save_market_momentum_screen
from backend.candle_store import get_candles
from backend.explain.indicator_states import compute_indicator_states
from backend.explain.narrative_engine import build_full_narrative_bundle
from backend.push_alerts import (
    run_watchlist_push_alerts,
    run_watchlist_price_alerts,
    run_portfolio_push_alerts,
    run_crypto_market_alerts,
)

# ✅ Reuse your central quote provider (Finnhub)
# (safe: if FINNHUB_KEY missing, it returns {})
from quote_provider import fetch_equity_quote
from backend.quote_repo import get_quote_safe, is_quote_fresh, save_quote
from backend.news_repo import fetch_symbol_news
# ✅ These are referenced in your compute_symbol() but were missing in your pasted code.
# If they already exist elsewhere, keep these imports here (no breaking).
from backend.technicals import build_technical_snapshot
from backend.market_awareness import build_market_awareness
from backend.alpha_watch_logic import persist_alpha_watch
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore
from backend.firestore_utils import utc_now_iso

# =========================================================
# CONSTANTS
# =========================================================


CORE_UNIVERSE = [
    "NVDA",   # NVIDIA
    "GOOGL",  # Alphabet (Google)
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "AVGO",   # Broadcom
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "LLY",    # Eli Lilly
    "WMT",    # Walmart
    "BRK.B",  # Berkshire Hathaway
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "MA",     # Mastercard
    "XOM",    # Exxon Mobil
    "UNH",    # UnitedHealth Group
    "PG",     # Procter & Gamble
    "HD",     # Home Depot
    "COST",   # Costco
    "MRK"     # Merck
]

ACTIVE_SYMBOL_LIMIT = 60
TOTAL_SCAN_LIMIT = 80
COL_ROOT = "bullsignals_ai"
COL_STOCKS = "stocks"  # document id under bullsignals_ai
DOC_ACTIVE = "active_symbols"
DOC_HOMESCREEN = "homescreen_snapshot"

SP500_DISCOVERY_LIMIT = 540
DAILY_MOVER_GAINERS_LIMIT = 25
DAILY_MOVER_LOSERS_LIMIT = 25

DISCOVERY_FULL_SCAN_MAX_AGE_SECONDS = 60
DISCOVERY_FETCH_DELAY_SECONDS = 0.15

# Logging / behavior toggles (safe defaults)
LOG_EVERY_N = int((__import__("os").getenv("CRON_LOG_EVERY_N") or "1").strip() or "1")
FAIL_FAST = ((__import__("os").getenv("CRON_FAIL_FAST") or "0").strip() == "1")


# =========================================================
# LOGGING HELPERS
# =========================================================

def log(msg: str) -> None:
    print(f"[market-cron] {msg}", flush=True)


def log_exc(prefix: str, e: BaseException) -> None:
    tb = traceback.format_exc()
    log(f"❌ {prefix}: {e}\n{tb}")


# =========================================================
# FIRESTORE / TIME HELPERS
# =========================================================

def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
        log("🔥 Firebase initialized")
    return firestore.client()


# =========================================================
# PHASE 0 — MARKET GAINERS / LOSERS (INTERNAL)
# =========================================================

def get_internal_market_movers(limit: int = 20) -> List[str]:
    """
    Computes market movers from existing stock docs.
    Uses intraday % change from quotes.
    """
    db = get_db()

    snaps = (
        db.collection(COL_ROOT)
          .document(COL_STOCKS)
          .collection("symbols")
          .stream()
    )

    movers = []

    for doc in snaps:
        data = doc.to_dict() or {}
        quote = data.get("quote", {})
        chg = quote.get("changePct")
        updated_at = quote.get("updated_at")

        if (
            isinstance(chg, (int, float)) and
            isinstance(updated_at, str)
        ):
            movers.append((doc.id, abs(chg)))


    movers.sort(key=lambda x: x[1], reverse=True)

    return [sym for sym, _ in movers[:limit]]

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

def get_canonical_quote(symbol: str, fallback_price: float | None = None) -> Dict[str, Any]:
    """
    Return the best available normalized quote.

    Safe behavior:
    - Fresh cached quote -> use it
    - Stale/missing quote -> try live Finnhub
    - Live success -> save to quote repo and return it
    - Live fail -> return stale/fallback safely, marked needs_refresh
    """
    symbol = symbol.upper().strip()
    cached = get_quote_safe(symbol)

    # ✅ Use cached quote only when fresh
    if (
        isinstance(cached, dict)
        and cached.get("price") is not None
        and is_quote_fresh(cached)
    ):
        return dict(cached)

    # ✅ Cached is stale/missing, so try live quote
    live = fetch_equity_quote(symbol) or {}

    live_has_value = (
        isinstance(live.get("price"), (int, float))
        or isinstance(live.get("changePct"), (int, float))
    )

    if live_has_value:
        live_payload = {
            "symbol": symbol,
            "price": live.get("price"),
            "change": live.get("change"),
            "changePct": live.get("changePct"),
            "open": live.get("open"),
            "high": live.get("high"),
            "low": live.get("low"),
            "prevClose": live.get("prevClose"),
            "timestamp": live.get("timestamp"),
            "source": live.get("source", "finnhub"),
            "needs_refresh": False,
            "schema_version": "v2",
        }

        # ✅ update central quote repo also
        save_quote(symbol, live_payload)

        return {
            **live_payload,
            "updated_at": utc_now_iso(),
        }

    # ⚠️ Live failed: keep stale values if available, but clearly mark stale
    if isinstance(cached, dict):
        return {
            **cached,
            "symbol": symbol,
            "needs_refresh": True,
            "source": cached.get("source", "stale_fallback"),
            "schema_version": "v2",
        }

    return {
        "symbol": symbol,
        "price": fallback_price,
        "change": None,
        "changePct": None,
        "source": "fallback",
        "needs_refresh": True,
        "schema_version": "v2",
        "updated_at": utc_now_iso(),
    }
def _get_best_quote(symbol: str) -> Dict[str, Any]:
    """
    Prefer cached quote if fresh.
    Fall back to live Finnhub quote if missing or stale.
    Never writes to quote_repo.
    """
    cached = get_quote_safe(symbol)

    if isinstance(cached, dict) and is_quote_fresh(cached):
        # cached quote repo doc contains top-level fields
        log(f"📦 {symbol} using cached quote")
        return dict(cached)

    # fallback: live fetch
    log(f"🌐 {symbol} fetching live quote")
    live = fetch_equity_quote(symbol)
    return live if isinstance(live, dict) else {}

# =========================================================
# MODEL LOADER
# =========================================================

def ensure_bullbrain_loaded():
    if backend.bullbrain_model is None:
        backend.bullbrain_model = backend.load_bullbrain_model()
        try:
            log(f"🧠 BullBrain loaded | model_features={len(backend.bullbrain_model.feature_names)}")
        except Exception:
            log("🧠 BullBrain loaded")


# =========================================================
# ACTIVE SYMBOLS (READ-ONLY)
# =========================================================

def load_active_symbols() -> Dict[str, Dict[str, Any]]:
    db = get_db()
    snap = db.collection(COL_ROOT).document(DOC_ACTIVE).get()
    if not snap.exists:
        log("ℹ️ active_symbols doc not found (yet) → using empty active list")
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

    ranked = sorted(active.items(), key=lambda kv: score(kv[1]), reverse=True)
    top = [sym for sym, _ in ranked[:ACTIVE_SYMBOL_LIMIT]]
    return [s.upper() for s in top if s]

# =========================================================
# SCAN UNIVERSE
# =========================================================

def load_persisted_market_movers(limit: int = 20) -> List[str]:
    db = get_db()
    snap = db.collection(COL_ROOT).document("market_movers").get()
    if not snap.exists:
        return []
    movers = snap.to_dict().get("movers", [])
    return [m["symbol"] for m in movers[:limit] if "symbol" in m]

def _today_et() -> str:
    return datetime.datetime.now(ET).date().isoformat()


def _daily_movers_doc_ref(db, date_str: Optional[str] = None):
    return (
        db.collection(COL_ROOT)
          .document("market_memory")
          .collection("daily_movers")
          .document(date_str or _today_et())
    )

def _seconds_since_iso(ts_iso: Optional[str]) -> Optional[int]:
    try:
        if not ts_iso:
            return None
        ts = datetime.datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return int((datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds())
    except Exception:
        return None


def load_daily_mover_symbols(limit: int = 40) -> List[str]:
    """
    Reads today's lightweight-discovered movers.
    Used by market_cron full intelligence universe.
    """
    db = get_db()
    snap = _daily_movers_doc_ref(db).get()

    if not snap.exists:
        return []

    data = snap.to_dict() or {}
    symbols = data.get("symbols", [])

    if not isinstance(symbols, list):
        return []

    out = []
    for sym in symbols:
        s = str(sym).upper().strip()
        if s and s not in out:
            out.append(s)

    return out[:limit]

def load_spreadsheet_discovery_metadata(limit: int = 80) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Reads latest spreadsheet symbols + catalyst metadata as discovery seeds.
    No quote enrichment.
    No Firestore write.
    Cron discovery validates real movement later.
    """
    try:
        db = get_db()
        snap = db.collection(COL_ROOT).document("grok_market_memory").get()

        if not snap.exists:
            return [], {}

        data = snap.to_dict() or {}
        latest = data.get("premarket_latest") or {}

        buckets = [
            latest.get("alpha_opportunities") or [],
            latest.get("premarket_gainers") or [],
            latest.get("premarket_losers") or [],
        ]

        symbols = []
        metadata_by_symbol: Dict[str, Dict[str, Any]] = {}

        for bucket in buckets:
            for item in bucket:
                if not isinstance(item, dict):
                    continue

                sym = str(item.get("symbol") or "").upper().strip()
                if not sym:
                    continue

                if sym not in symbols:
                    symbols.append(sym)

                metadata_by_symbol[sym] = {
                    "spreadsheetSource": item.get("source"),
                    "spreadsheetSector": item.get("sector"),
                    "primaryCatalysts": item.get("primary_catalysts"),
                    "reason": item.get("reason"),
                    "moverQuality": item.get("mover_quality"),
                    "riskLevel": item.get("risk_level"),
                    "marketDay": item.get("market_day"),
                    "sessionType": item.get("session_type"),
                    "generatedAt": item.get("generated_at"),
                }

        symbols = symbols[:limit]
        metadata_by_symbol = {
            sym: metadata_by_symbol[sym]
            for sym in symbols
            if sym in metadata_by_symbol
        }

        return symbols, metadata_by_symbol

    except Exception as e:
        log(f"⚠️ spreadsheet discovery metadata unavailable: {e}")
        return [], {}
      
def get_discovery_phase() -> Optional[str]:
    """
    Returns the full-scan phase name if current ET time is inside a scan window.
    """
    now_et = datetime.datetime.now(ET)
    hhmm = now_et.hour * 100 + now_et.minute

    # 9:20–9:29 warm cache before open
    if 915 <= hhmm <= 929:
        return "premarket_warm"

    # 9:32–9:44 true opening discovery
    if 932 <= hhmm <= 944:
        return "morning_discovery"

    # 12:00–12:14 midday correction
    if 1200 <= hhmm <= 1214:
        return "midday_correction"

    # 3:30–3:44 closing correction
    if 1530 <= hhmm <= 1544:
        return "closing_correction"

    return None


def should_refresh_daily_movers() -> Tuple[bool, Optional[str]]:
    """
    Full 506-symbol discovery runs only once per defined phase.
    """
    phase = get_discovery_phase()

    if not phase:
        return False, None

    db = get_db()
    snap = _daily_movers_doc_ref(db).get()

    if not snap.exists:
        return True, phase

    data = snap.to_dict() or {}
    discovery = data.get("discovery", {}) if isinstance(data.get("discovery"), dict) else {}

    done_key = f"{phase}_done"

    if discovery.get(done_key) is True:
        return False, phase

    return True, phase

def refresh_daily_movers_from_sp500() -> List[str]:
    """
    Lightweight discovery:
    - fetch quotes only for REAL_TICKERS
    - rank top 20 gainers and top 20 losers
    - save /bullsignals_ai/daily_movers_YYYY-MM-DD
    - save quote repo entries so quote_worker has fresh baseline
    """
    should_run, phase = should_refresh_daily_movers()

    if not should_run:
        existing = load_daily_mover_symbols(
            limit=DAILY_MOVER_GAINERS_LIMIT + DAILY_MOVER_LOSERS_LIMIT
        )
        if existing:
            log(f"📌 daily movers existing — using saved list | count={len(existing)}")
        return existing

    log(f"🚦 daily movers full scan phase={phase}")

    db = get_db()
    today = _today_et()
    doc_ref = _daily_movers_doc_ref(db, today)
    now_iso = utc_now_iso()

    spreadsheet_symbols, spreadsheet_metadata = load_spreadsheet_discovery_metadata()

    symbols = []
    for sym in [*spreadsheet_symbols, *REAL_TICKERS[:SP500_DISCOVERY_LIMIT]]:
        s = str(sym).upper().strip()
        if s and s not in symbols:
            symbols.append(s)

    log(
        f"📄 spreadsheet discovery seeds added | "
        f"spreadsheet={len(spreadsheet_symbols)} total_universe={len(symbols)}"
    )
    rows: List[Dict[str, Any]] = []

    log(f"🔎 daily movers discovery start | universe={len(symbols)}")

    for i, sym in enumerate(symbols, start=1):
        try:
            cached = get_quote_safe(sym)

            use_cache = False
            if isinstance(cached, dict):
                cached_price = cached.get("price")
                cached_chg = cached.get("changePct")
                cached_age = _seconds_since_iso(cached.get("updated_at"))

                if (
                    isinstance(cached_price, (int, float))
                    and isinstance(cached_chg, (int, float))
                    and cached_age is not None
                    and cached_age <= DISCOVERY_FULL_SCAN_MAX_AGE_SECONDS
                ):
                    use_cache = True

            if use_cache:
                q = cached or {}
            else:
                q = fetch_equity_quote(sym) or {}
                time.sleep(DISCOVERY_FETCH_DELAY_SECONDS)

            price = q.get("price")
            chg = q.get("changePct")

            if not isinstance(price, (int, float)) or not isinstance(chg, (int, float)):
                continue

            payload = {
                "symbol": sym,
                "price": price,
                "change": q.get("change"),
                "changePct": chg,
                "open": q.get("open"),
                "high": q.get("high"),
                "low": q.get("low"),
                "prevClose": q.get("prevClose"),
                "timestamp": q.get("timestamp"),
                "source": q.get("source", "finnhub"),
                "needs_refresh": False,
                "schema_version": "v2",
            }

            save_quote(sym, payload)

            rows.append({
                "symbol": sym,
                "company_name": COMPANY_NAMES.get(sym, sym),
                "price": round(float(price), 2),
                "change": q.get("change"),
                "changePct": round(float(chg), 4),
                "direction": "up" if chg >= 0 else "down",
                "quote_updated_at": now_iso,
                "spreadsheet": spreadsheet_metadata.get(sym),
            })

        except Exception as e:
            log(f"⚠️ daily mover quote failed for {sym}: {e}")

        if i % 100 == 0:
            log(f"… daily movers quote scan progress {i}/{len(symbols)} | valid={len(rows)}")

    gainers = sorted(
        [r for r in rows if isinstance(r.get("changePct"), (int, float)) and r["changePct"] >= 0],
        key=lambda r: r["changePct"],
        reverse=True,
    )[:DAILY_MOVER_GAINERS_LIMIT]

    losers = sorted(
        [r for r in rows if isinstance(r.get("changePct"), (int, float)) and r["changePct"] < 0],
        key=lambda r: r["changePct"],
    )[:DAILY_MOVER_LOSERS_LIMIT]

    mover_symbols = list(dict.fromkeys(
    [r["symbol"] for r in gainers] + [r["symbol"] for r in losers]
    ))

    # ---------------------------------------------------------
    # Preserve previous discovery phase flags before merge write
    # ---------------------------------------------------------
    existing_discovery = {}

    existing_snap = doc_ref.get()
    if existing_snap.exists:
        existing_data = existing_snap.to_dict() or {}
        if isinstance(existing_data.get("discovery"), dict):
            existing_discovery = existing_data["discovery"]

    phase_key = f"{phase}_done" if phase else "unknown_done"

    existing_discovery[phase_key] = True
    existing_discovery["last_phase"] = phase
    existing_discovery["last_full_scan_at"] = now_iso
    existing_discovery["full_scan_max_age_seconds"] = DISCOVERY_FULL_SCAN_MAX_AGE_SECONDS
    existing_discovery["fetch_delay_seconds"] = DISCOVERY_FETCH_DELAY_SECONDS
    existing_discovery["last_universe_count"] = len(symbols)
    existing_discovery["last_valid_quote_count"] = len(rows)

    doc_ref.set(
        {
            "date": today,
            "source": "sp500_light_quote_scan",
            "universe": "REAL_TICKERS",
            "universe_count": len(symbols),
            "valid_quote_count": len(rows),
            "gainers": gainers,
            "losers": losers,
            "symbols": mover_symbols,
            "updated_at": now_iso,
            "expires_at": (
                datetime.datetime.now(ET).date() + datetime.timedelta(days=1)
            ).isoformat(),
            "discovery": existing_discovery,
            "schema_version": "v1",
            "spreadsheet_metadata": spreadsheet_metadata,
        },
        merge=True,
    )

    log(
        f"✅ daily movers saved | doc=market_memory/daily_movers/{today} "
        f"gainers={len(gainers)} losers={len(losers)} symbols={len(mover_symbols)}"
    )

    return mover_symbols

def build_scan_universe() -> Tuple[List[str], Dict[str, Any]]:
    # Existing internal movers from previous stock docs
    phase0 = load_persisted_market_movers(limit=20)

    # New daily movers from lightweight S&P/REAL_TICKERS discovery
    daily_movers = refresh_daily_movers_from_sp500()

    active_raw = load_active_symbols()
    active = rank_active_symbols(active_raw)

    universe = list(
        dict.fromkeys(
            [*daily_movers, *phase0, *CORE_UNIVERSE, *active]
        )
    )

    meta = {
        "daily_movers": len(daily_movers),
        "market_movers": len(phase0),
        "core_universe": len(CORE_UNIVERSE),
        "active_ranked": len(active),
        "universe": len(universe),
    }

    return universe[:TOTAL_SCAN_LIMIT], meta

def build_homescreen_universe() -> List[str]:
    """
    Curated Top 20 high-signal universe for Homescreen.
    Then we rank + boost momentum inside persist_homescreen_core_signals()
    """
    return [
       "NVDA",   # NVIDIA
    "GOOGL",  # Alphabet (Google)
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "AVGO",   # Broadcom
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "LLY",    # Eli Lilly
    "WMT",    # Walmart
    "BRK.B",  # Berkshire Hathaway
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "MA",     # Mastercard
    "XOM",    # Exxon Mobil
    "UNH",    # UnitedHealth Group
    "PG",     # Procter & Gamble
    "HD",     # Home Depot
    "COST",   # Costco
    "MRK"     # Merck
    ]
# =========================================================
# VALIDATION HELPERS
# =========================================================

def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
        return not (math.isnan(v) or math.isinf(v))
    except Exception:
        return False


def _validate_feature_dict(symbol: str, feat_dict: Dict[str, Any]) -> bool:
    # Minimal sanity checks to avoid NaN poisoning.
    critical = ["close", "rsi14", "macd", "macd_signal", "atr14", "volatility_20d"]
    missing = [k for k in critical if k not in feat_dict]
    if missing:
        log(f"⚠️ {symbol} features missing keys: {missing}")
        # do not hard-fail; model may not need them all depending on implementation
    # Check a sample of numeric values for finiteness
    bad = []
    for k in critical:
        if k in feat_dict and feat_dict[k] is not None and not _is_finite_number(feat_dict[k]):
            bad.append(k)
    if bad:
        log(f"❌ {symbol} features contain non-finite values: {bad}")
        return False
    return True



def _build_chart_summary(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight 1Y chart summary for StockDetail.
    FullChart still uses /candles/{symbol}.
    """
    valid = [c for c in candles if isinstance(c, dict)][-252:]

    closes = []
    highs = []
    lows = []

    for c in valid:
        try:
            if c.get("close") is not None:
                closes.append(float(c["close"]))
            if c.get("high") is not None:
                highs.append(float(c["high"]))
            if c.get("low") is not None:
                lows.append(float(c["low"]))
        except Exception:
            pass

    first_close = closes[0] if closes else None
    last_close = closes[-1] if closes else None

    return_pct = None
    if first_close not in (None, 0) and last_close is not None:
        return_pct = ((last_close - first_close) / first_close) * 100

    return {
        "range": "1Y",
        "basis": "close",
        "source": "candle_store",
        "points": 30,
        "closeLow": round(min(closes), 2) if closes else None,
        "closeHigh": round(max(closes), 2) if closes else None,
        "intradayLow": round(min(lows), 2) if lows else None,
        "intradayHigh": round(max(highs), 2) if highs else None,
        "firstClose": round(first_close, 2) if first_close is not None else None,
        "lastClose": round(last_close, 2) if last_close is not None else None,
        "returnPct": round(return_pct, 2) if return_pct is not None else None,
        "candleCount": len(valid),
        "updated_at": utc_now_iso(),
    }
# =========================================================
# PER-SYMBOL COMPUTE
# =========================================================

def compute_symbol(symbol: str) -> Dict[str, Any] | None:
    t0 = time.time()
    symbol = symbol.upper()
    log(f"▶ {symbol} compute start")

    # ---------------------------------------------------------
    # Local helpers (self-contained, prevents NameError issues)
    # ---------------------------------------------------------
    def _normalize_to_list(raw: Any) -> List[Dict[str, Any]]:
        """
        Returns list[dict] candles no matter what get_candles() returns.
        Supports:
          - list[dict]
          - dict wrapper {"candles": [...]}
          - dict-of-arrays {open/high/low/close/volume/timestamp}
        """
        if raw is None:
            return []

        if isinstance(raw, list):
            return [c for c in raw if isinstance(c, dict)]

        if isinstance(raw, dict):
            # wrapper
            if isinstance(raw.get("candles"), list):
                return [c for c in raw["candles"] if isinstance(c, dict)]

            # dict-of-arrays
            if isinstance(raw.get("close"), list) and all(k in raw for k in ["open", "high", "low", "close"]):
                o = raw.get("open") or []
                h = raw.get("high") or []
                l = raw.get("low") or []
                cl = raw.get("close") or []
                v = raw.get("volume") or []
                ts = raw.get("timestamp") or []

                n = min(len(o), len(h), len(l), len(cl))
                out: List[Dict[str, Any]] = []
                for i in range(n):
                    out.append({
                        "t": ts[i] if i < len(ts) else None,
                        "open": o[i],
                        "high": h[i],
                        "low": l[i],
                        "close": cl[i],
                        "volume": v[i] if i < len(v) else 0.0,
                    })
                return out

        return []

    def _list_to_arrays(lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert list candles to dict-of-arrays (what ML + scan_smart_pattern_history need).
        """
        out = {"open": [], "high": [], "low": [], "close": [], "volume": [], "timestamp": []}
        for c in lst:
            if not isinstance(c, dict):
                continue
            o = c.get("open")
            h = c.get("high")
            l = c.get("low")
            cl = c.get("close")
            if o is None or h is None or l is None or cl is None:
                continue

            t = c.get("t") or c.get("timestamp") or c.get("time")
            out["open"].append(float(o))
            out["high"].append(float(h))
            out["low"].append(float(l))
            out["close"].append(float(cl))
            out["volume"].append(float(c.get("volume") or 0.0))

            # normalize to ms epoch if numeric; else None (scanner can still run)
            try:
                if isinstance(t, (int, float)):
                    t_ms = int(t * 1000) if t < 10_000_000_000 else int(t)
                    out["timestamp"].append(t_ms)
                else:
                    out["timestamp"].append(None)
            except Exception:
                out["timestamp"].append(None)

        return out

    def _build_sparkline(lst: List[Dict[str, Any]], points: int = 30) -> List[float]:
        closes: List[float] = []
        # sample a bit more than needed to tolerate missing closes
        for c in lst[-max(points * 3, points):]:
            if isinstance(c, dict) and c.get("close") is not None:
                try:
                    closes.append(float(c["close"]))
                except Exception:
                    pass
        return closes[-points:]

    # ---------------------------------------------------------
    # 1) Fetch candles (raw)
    # ---------------------------------------------------------
    try:
        raw_candles = get_candles(symbol, min_points=120)
    except Exception as e:
        log_exc(f"{symbol} get_candles failed", e)
        return None

    if not raw_candles:
        log(f"⛔ {symbol} no candles returned → skip")
        return None

    # ---------------------------------------------------------
    # 2) Build BOTH representations (this prevents all your crashes)
    # ---------------------------------------------------------
    # candles_arrays = what compute_bullbrain_features() expects (dict-of-arrays)
    if isinstance(raw_candles, dict) and isinstance(raw_candles.get("close"), list):
        candles_arrays = raw_candles
        candles_list = _normalize_to_list(raw_candles)
    else:
        candles_list = _normalize_to_list(raw_candles)
        candles_arrays = _list_to_arrays(candles_list)

    if not candles_arrays or not isinstance(candles_arrays, dict) or not isinstance(candles_arrays.get("close"), list):
        log(f"⛔ {symbol} candles_arrays invalid → skip")
        return None

    n = len(candles_arrays.get("close") or [])
    log(f"✅ {symbol} candles={n}")

    # UI-safe persistence fields
    
    sparkline = _build_sparkline(candles_list, points=30)
    chart_summary = _build_chart_summary(candles_list)

    # ---------------------------------------------------------
    # 3) Features (MUST use candles_arrays)
    # ---------------------------------------------------------
    try:
        feats_vec, feat_dict, _ = backend.compute_bullbrain_features(candles_arrays)
    except Exception as e:
        log_exc(f"{symbol} compute_bullbrain_features threw", e)
        return None

    if feats_vec is None or not isinstance(feat_dict, dict):
        log(f"⛔ {symbol} features invalid → skip")
        return None

    if not _validate_feature_dict(symbol, feat_dict):
        log(f"⛔ {symbol} feature validation failed → skip")
        return None
    # ---------------------------------------------------------
    # 4) Authoritative BullBrain Decision (patterns + ladder)
    # ---------------------------------------------------------
    try:
        core = backend.run_bullbrain_from_inputs(
            symbol=symbol,
            candles_arrays=candles_arrays,
            feat_dict=feat_dict,
        )

        bull = core["bullbrain"]
        decision = core["decision"]

        final_signal = decision["finalSignal"]
        

        prob_up = bull.get("raw", {}).get("prob_up")
        

    except Exception as e:
        log_exc(f"{symbol} run_bullbrain_from_inputs crashed", e)
        return None

    # ---------------------------------------------------------
    # 4.1) Compute Indicator States (DETERMINISTIC)
    # ---------------------------------------------------------
    indicator_state_bundle = compute_indicator_states({
        "features_meta": feat_dict,
        "bullbrain": bull,
        "decision": decision,
        "patternHistory": core.get("patternHistory"),
        "content": {
            "ui": {
                "hybridProbUp": prob_up
            }
        }
    })

    # ---------------------------------------------------------
    # 4.2) Build Institutional Narratives (ONCE)
    # ---------------------------------------------------------
    narratives = build_full_narrative_bundle(
        indicator_states=indicator_state_bundle["states"],
        seed=hash(symbol)
    )

    # ---------------------------------------------------------
    # 5) Quote (Firestore-only StockDetail needs this!)
    #    This does NOT write elsewhere; we persist inside the symbol doc.
    # ---------------------------------------------------------
    try:
        quote = get_canonical_quote(
            symbol=symbol,
            fallback_price=feat_dict.get("close"),
        )
    except Exception as e:
        log(f"⚠️ {symbol} canonical quote failed: {e}")
        quote = {
            "symbol": symbol,
            "price": feat_dict.get("close"),
            "needs_refresh": True,
            "schema_version": "v2",
            "updated_at": utc_now_iso(),
        }

    # ---------------------------------------------------------
    # 6) Technical snapshot
    # ---------------------------------------------------------
    try:
        technical = build_technical_snapshot(
            symbol=symbol,
            features=feat_dict,
            last_close=feat_dict.get("close"),
        )
    except Exception as e:
        log_exc(f"{symbol} build_technical_snapshot failed", e)
        technical = None

    # ---------------------------------------------------------
    # 7) Market Awareness / Why It's Moving
    # ---------------------------------------------------------
    try:
        news_items = fetch_symbol_news(
            symbol=symbol,
            company_name=COMPANY_NAMES.get(symbol, symbol),
            limit=5,
        )
    except Exception as e:
        log(f"⚠️ {symbol} catalyst news fetch failed: {e}")
        news_items = []

    market_awareness = build_market_awareness(
        symbol=symbol,
        company_name=COMPANY_NAMES.get(symbol, symbol),
        quote=quote,
        technical=technical,
        pattern=core.get("pattern"),
        bullbrain=core.get("bullbrain"),
        decision=core.get("decision"),
        news_items=news_items,
    )

    # ---------------------------------------------------------
    # 8) Preserve existing profile/logo metadata
    # ---------------------------------------------------------
    try:
        existing_snap = (
            get_db()
            .collection(COL_ROOT)
            .document(COL_STOCKS)
            .collection("symbols")
            .document(symbol)
            .get()
        )
        existing_data = existing_snap.to_dict() or {}
        profile = existing_data.get("profile") or {}
    except Exception as e:
        log(f"⚠️ {symbol} profile/logo read failed: {e}")
        profile = {}

    # ---------------------------------------------------------
    # 9) Build doc (everything your Firestore-only stockdetail needs)
    # ---------------------------------------------------------
    doc = {
        "symbol": symbol,
        "company_name": COMPANY_NAMES.get(symbol, symbol),
        "profile": profile,
        "quote": quote,
        "features_meta": feat_dict,
        "technical": technical,
        "bullbrain": core["bullbrain"],
        "decision": core["decision"],
        "pattern": core.get("pattern"),
        "patternBias": core.get("patternBias"),
        "patternHistory": core.get("patternHistory"),
        "sparkline": sparkline,
        "chart": chart_summary,
        "indicator_states": indicator_state_bundle["states"],
        "narratives": narratives,   
        "marketAwareness": market_awareness,
        "computed_at": utc_now_iso(),
        "schema_version": "v2",
    }

    # ---------------------------------------------------------
    # 10) Firestore write
    # ---------------------------------------------------------
    try:
        db = get_db()

        # index marker doc
        db.collection(COL_ROOT).document(COL_STOCKS).set(
            {
                "schema_version": "v1",
                "updated_at": utc_now_iso(),
                "note": "stocks document is an index marker; symbol docs live in stocks/symbols/*",
            },
            merge=True,
        )

        db.collection(COL_ROOT).document(COL_STOCKS) \
            .collection("symbols").document(symbol) \
            .set(doc, merge=True)

        dt = time.time() - t0
        final_signal = core["decision"]["finalSignal"]
        conf = core["bullbrain"].get("confidence")
        # 🔐 Defensive alias (backward compatibility)
    
        log(
            f"✅ {symbol} wrote stocks/symbols/{symbol} | "
            f"final={final_signal} conf={conf}% | {dt:.2f}s"
        )

        return doc

    except Exception as e:
        log_exc(f"{symbol} Firestore write failed", e)
        return None

def load_today_spreadsheet_mover_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Reads metadata saved during today's discovery scan.
    Used only to attach catalyst/reason to validated internal movers.
    """
    try:
        db = get_db()
        snap = _daily_movers_doc_ref(db).get()

        if not snap.exists:
            return {}

        data = snap.to_dict() or {}
        meta = data.get("spreadsheet_metadata") or {}

        return meta if isinstance(meta, dict) else {}

    except Exception as e:
        log(f"⚠️ daily spreadsheet mover metadata unavailable: {e}")
        return {}
    
def persist_internal_market_movers():
    db = get_db()

    snaps = (
        db.collection(COL_ROOT)
          .document(COL_STOCKS)
          .collection("symbols")
          .stream()
    )

    movers = []
    spreadsheet_metadata = load_today_spreadsheet_mover_metadata()

    for doc in snaps:
        sym = str(doc.id or "").upper().strip()

        if sym not in REAL_TICKERS:
            continue

        data = doc.to_dict() or {}
        quote = data.get("quote", {})
        chg = quote.get("changePct")
        price = quote.get("price")
        profile = data.get("profile") or {}
        logo_url = profile.get("logoUrl")

        try:
            px = float(price)
        except Exception:
            continue

        if px < 1.0:
            continue

        updated_at = quote.get("updated_at")

        if not updated_at:
            continue

        try:
            ts = datetime.datetime.fromisoformat(
                updated_at.replace("Z", "+00:00")
            )
            age_seconds = (
                datetime.datetime.now(datetime.timezone.utc) - ts
            ).total_seconds()
        except Exception:
            continue

        # Skip very stale quotes from movers
        # 20 minutes is safe because cron runs every 15 minutes.
        if age_seconds > 20 * 60:
            continue
        sheet_meta = spreadsheet_metadata.get(sym) or {}    
        if isinstance(chg, (int, float)):
            movers.append({
                "symbol": sym,
                "company": data.get("company_name") or COMPANY_NAMES.get(sym, sym),
                "price": round(px, 2),
                "change": quote.get("change"),
                "changePct": round(chg, 2),
                "direction": "up" if chg >= 0 else "down",
                "quote_updated_at": updated_at,
                "logoUrl": logo_url or None,

                # Optional spreadsheet catalyst metadata.
                # Present only when this symbol came from the spreadsheet
                # and later became a real validated internal mover.
                "primaryCatalysts": sheet_meta.get("primaryCatalysts"),
                "reason": sheet_meta.get("reason"),
                "moverQuality": sheet_meta.get("moverQuality"),
                "riskLevel": sheet_meta.get("riskLevel"),
                "spreadsheetSource": sheet_meta.get("spreadsheetSource"),
                "spreadsheetSector": sheet_meta.get("spreadsheetSector"),
            })

    # ✅ Separate gainers and losers
    gainers = [m for m in movers if m.get("direction") == "up"]
    losers = [m for m in movers if m.get("direction") == "down"]

    # ✅ Top gainers = highest positive %
    gainers.sort(key=lambda x: x.get("changePct", 0), reverse=True)

    # ✅ Top losers = most negative %
    losers.sort(key=lambda x: x.get("changePct", 0))

    # ✅ Prepare cached views once during cron
    home_rising = gainers[:5]
    preview_rising = gainers[:4]
    preview_falling = losers[:4]
    final_movers = gainers[:25] + losers[:25]

    db.collection(COL_ROOT).document("market_movers").set(
        {
            "as_of": datetime.datetime.now(ET).date().isoformat(),
            "updated_at": utc_now_iso(),
            "source": "internal_quote_change",
            "limit": {
                "home_rising": 5,
                "preview_rising": 4,
                "preview_falling": 4,
                "gainers": 25,
                "losers": 25,
                "total": 50,
            },
            "home_rising": home_rising,
            "preview_rising": preview_rising,
            "preview_falling": preview_falling,
            "gainers_count": len(gainers[:25]),
            "losers_count": len(losers[:25]),
            "movers": final_movers,
            "schema_version": "market_movers_v4",
        },
        merge=True,
    )
    log(
        f"🔥 market_movers updated | "
        f"gainers={len(gainers[:25])} "
        f"losers={len(losers[:25])} "
        f"total={len(final_movers)}"
    )

def persist_homescreen_core_signals():
    """
    Writes stable curated AI-ranked HomeScreen signals.
    """
    db = get_db()
    now_iso = utc_now_iso()

    symbols = build_homescreen_universe()   # Top 20
    ranked: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            snap = (
                db.collection(COL_ROOT)
                  .document(COL_STOCKS)
                  .collection("symbols")
                  .document(sym)
                  .get()
            )
            if not snap.exists:
                continue

            data = snap.to_dict() or {}
            quote = data.get("quote") or {}
            bullbrain = data.get("bullbrain") or {}
            decision = data.get("decision") or {}
            pattern = data.get("pattern") or {}
            market_awareness = data.get("marketAwareness") or {}
            raw_signal = decision.get("finalSignal") or bullbrain.get("signal", "HOLD")
            change_pct = quote.get("changePct")
            profile = data.get("profile") or {}
            logo_url = profile.get("logoUrl")

            display_signal = raw_signal

            try:
                cp = float(change_pct or 0)

                # HomeScreen is daily-move focused.
                # Avoid showing Bullish on hard red days or Bearish on strong green days.
                if cp <= -2.0 and raw_signal == "BUY":
                    display_signal = "HOLD"
                elif cp >= 2.0 and raw_signal == "SELL":
                    display_signal = "HOLD"
            except Exception:
                pass
            ranked.append({
                "symbol": sym,
                "companyName": data.get("company_name") or COMPANY_NAMES.get(sym, sym),
                "price": quote.get("price"),
                "change": quote.get("change"),
                "changePct": quote.get("changePct"),
                "lastUpdated": quote.get("updated_at") or data.get("computed_at"),
                "signal": display_signal,
                "rawSignal": raw_signal,
                "confidence": bullbrain.get("confidence", 0),
                "pattern": pattern.get("pattern") or pattern.get("patternLabel"),
                "logoUrl": logo_url or None,   
                "patternWinRate": (
                    ((data.get("patternHistory") or {})
                     .get("forwardReturns") or {})
                     .get("days5", {})
                     .get("winRate")
                ),
                "summary": (
                    market_awareness.get("oneLiner")
                    or market_awareness.get("summary")
                ),
            })

        except Exception as e:
            log(f"⚠️ homescreen core signal failed for {sym}: {e}")

    # === FINAL SORT (Momentum First) ===
    ranked.sort(
        key=lambda x: (
            abs(float(x.get("changePct") or 0)) * 1.4,
            float(x.get("confidence") or 0) * 0.8,
            25 if abs(float(x.get("changePct") or 0)) > 5.0 else 0,
        ),
        reverse=True,
    )

    db.collection(COL_ROOT).document(DOC_HOMESCREEN).set(
        {
            "core_universe": symbols,
            "core_signals": ranked[:10],
            "core_universe_count": len(symbols),
            "core_signals_updated_at": now_iso,
            "updated_at": now_iso,
            "schema_version": "home_v2",
        },
        merge=True,
    )

    top5 = [item["symbol"] for item in ranked[:5]]
    log(f"🏠 homescreen updated | count={len(ranked[:10])} | top5={top5}")
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
    try:
        quote = _get_best_quote(symbol)
        chg = quote.get("changePct") if isinstance(quote, dict) else None
        return float(chg) if isinstance(chg, (int, float)) else None
    except Exception as e:
        log(f"⚠️ quote change pct failed for {symbol}: {e}")
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

    def upsert_card(card_id: str, title: str, subtitle: str, items: List[Dict[str, Any]]):
        nonlocal carousel
        for c in carousel:
            if isinstance(c, dict) and c.get("id") == card_id:
                c.setdefault("title", title)
                c.setdefault("subtitle", subtitle)
                if not isinstance(c.get("items"), list) or len(c.get("items")) == 0:
                    c["items"] = items
                c["updated_at"] = now_iso
                return
        carousel.append(
            {"id": card_id, "title": title, "subtitle": subtitle, "items": items, "updated_at": now_iso}
        )

    upsert_card("us_market", "US Market", "S&P 500 / Nasdaq", us_market_items)
    upsert_card("commodities", "Commodities", "Gold, Oil, Silver", commodities_items)

    market_open = is_market_open(now_utc)
    market_status = "Market Open" if market_open else "Market Closed"

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

    market_overview = {
        "marketStatus": market_status,
        "marketMood": mood,
        "fearGreed": {"label": fg_label, "value": fg_val},
        "updated_at": now_iso,
    }

    ref.set(
        {
            "carousel": carousel,
            "market_overview": market_overview,
            "updated_at": data.get("updated_at", now_iso),
            "version": data.get("version", "v1"),
        },
        merge=True,
    )

    log(f"✅ homescreen market_overview set | status={market_status} mood={mood}")

# =========================================================
# ENTRYPOINT
# =========================================================
def get_cron_mode() -> str:
    if os.getenv("FORCE_MARKET_CRON") == "1":
        return "market_open_intelligence"

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_et = now_utc.astimezone(ET)
    hhmm = now_et.hour * 100 + now_et.minute

    if now_et.weekday() >= 5 or is_us_market_holiday(now_utc):
        return "skip"

    if 915 <= hhmm <= 929:
        return "quote_discovery_only"

    if 930 <= hhmm <= 944:
        return "quote_discovery_plus_intelligence"

    if 945 <= hhmm <= 1600:
        return "market_open_intelligence"

    if 1601 <= hhmm <= 1630:
        return "final_close_intelligence"

    return "skip"

def main():
    run_id = utc_now_iso()
    log("🧬 market_cron VERSION = 2026-01-CRON | NO HOTLIST | NO BEARWATCH | NO MAG7 SNAPSHOT")
    log(f"🚀 cron start | run_id={run_id}")

    mode = get_cron_mode()
    log(f"🧭 cron mode={mode}")

    if mode == "skip":
        log("⏸️ outside market intelligence window — skipping heavy cron")
        return

    # For quote warm phase, do not load BullBrain yet
    if mode != "quote_discovery_only":
        ensure_bullbrain_loaded()

    # ---------------------------------------------------------
    # 0️⃣ HOMESCREEN BASELINE FIRST (SPY / QQQ / Commodities)
    # ---------------------------------------------------------
    try:
        ensure_market_overview_and_baseline_carousel()
        log("🏠 homescreen baseline ensured (us_market + commodities)")
    except Exception as e:
        log_exc("homescreen baseline failed", e)

    # 9:15 warm scan: quote-only, no BullBrain
    if mode == "quote_discovery_only":
        refresh_daily_movers_from_sp500()

        # Build and save Momentum Movers screen cache after pre-market discovery
        try:
            momentum_screen = save_market_momentum_screen(lookback_snapshots=12)
            log(
                f"📈 market momentum screen updated after quote discovery | "
                f"repeated={momentum_screen.get('pulse', {}).get('repeatedMovers')} "
                f"positive={len(momentum_screen.get('momentumMovers', []))} "
                f"pullbacks={len(momentum_screen.get('pullbackWatch', []))} "
                f"ai={len(momentum_screen.get('aiWatchlistMomentum', []))}"
            )
        except Exception as e:
            log_exc("market momentum screen persistence failed after quote discovery", e)

        log("✅ quote discovery warm scan completed — skipping BullBrain")
        return
    scan_symbols, scan_meta = build_scan_universe()

    log(
    f"📦 scan universe built | "
    f"total={len(scan_symbols)} | "
    f"daily_movers={scan_meta['daily_movers']} "
    f"market_movers={scan_meta['market_movers']} "
    f"core={scan_meta['core_universe']} "
    f"active={scan_meta['active_ranked']}"
    )


    # ---------------------------------------------------------
    # 🔁 COMPUTE LOOP
    # ---------------------------------------------------------
    results: List[Dict[str, Any]] = []
    success = 0
    skipped = 0
    failed = 0

    for i, sym in enumerate(scan_symbols, start=1):
        try:
            r = compute_symbol(sym)
            if r:
                results.append(r)
                success += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            log_exc(f"{sym} unhandled exception", e)
            if FAIL_FAST:
                raise

        if LOG_EVERY_N > 0 and (i % LOG_EVERY_N == 0):
            log(
                f"… progress {i}/{len(scan_symbols)} | "
                f"ok={success} skip={skipped} fail={failed}"
            )

        time.sleep(random.uniform(0.15, 0.25))

    log(
        f"✅ compute complete | "
        f"ok={success} skip={skipped} fail={failed} | results={len(results)}"
    )
    persist_internal_market_movers()
    persist_homescreen_core_signals()

    # ---------------------------------------------------------
    # GROK ENHANCEMENT LAYER (SAFE)
    # ---------------------------------------------------------
    # Spreadsheet/Grok UI pipeline disabled.
    # Spreadsheet symbols should only be used as lightweight discovery seeds.
    # No quote enrichment. No verified-alpha Firestore write.
    # No separate UI pipeline.
    try:
        log("📄 Grok/verified-alpha pipeline disabled — using internal movers only")
    except Exception as e:
        log_exc("grok enhancement layer disabled block failed", e)
    try:
        alpha_watch = persist_alpha_watch(get_db(), results)
        log(
            f"🎯 alpha_watch updated | "
            f"count={alpha_watch.get('count', 0)} "
            f"regime={alpha_watch.get('market_regime')} "
            f"updated_at={alpha_watch.get('updated_at')}"
        )

    except Exception as e:
        log_exc("alpha_watch persistence failed", e)

    # ---------------------------------------------------------
    # MARKET MOMENTUM SCREEN CACHE
    # Builds Firestore-first UI-ready data for Momentum Movers screen
    # ---------------------------------------------------------
    try:
        momentum_screen = save_market_momentum_screen(lookback_snapshots=12)
        log(
            f"📈 market momentum screen updated | "
            f"repeated={momentum_screen.get('pulse', {}).get('repeatedMovers')} "
            f"positive={len(momentum_screen.get('momentumMovers', []))} "
            f"pullbacks={len(momentum_screen.get('pullbackWatch', []))} "
            f"ai={len(momentum_screen.get('aiWatchlistMomentum', []))}"
        )
    except Exception as e:
        log_exc("market momentum screen persistence failed", e)
            
    try:
        alert_result = run_watchlist_push_alerts()
        log(f"🔔 watchlist push alerts checked | {alert_result}")
    except Exception as e:
        log_exc("watchlist signal alerts failed", e)
    try:
        price_alert_result = run_watchlist_price_alerts()
        log(f"🎯 watchlist price alerts checked | {price_alert_result}")
    except Exception as e:
        log_exc("watchlist price alerts failed", e)
    try:
        portfolio_alert_result = run_portfolio_push_alerts()
        log(f"💼 portfolio push alerts checked | {portfolio_alert_result}")
    except Exception as e:
        log_exc("portfolio push alerts failed", e)
    try:
        crypto_alert_result = run_crypto_market_alerts()
        log(f"🪙 crypto market alerts checked | {crypto_alert_result}")
    except Exception as e:
        log_exc("crypto market alerts failed", e)
    log("🏁 cron done")

# =========================================================
# RUNNER (DO NOT REMOVE)
# =========================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exc("FATAL: cron crashed", e)
        raise
