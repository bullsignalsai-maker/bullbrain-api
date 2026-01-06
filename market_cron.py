# market_cron.py
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
from typing import Dict, Any, List, Optional, Tuple

import firebase_admin
from firebase_admin import firestore

import main as backend
from symbols_clean import REAL_TICKERS, COMPANY_NAMES

from backend.candle_store import get_candles
from backend.bull_insights import generate_bull_insights

# ✅ Reuse your central quote provider (Finnhub)
# (safe: if FINNHUB_KEY missing, it returns {})
from quote_provider import fetch_equity_quote

# ✅ These are referenced in your compute_symbol() but were missing in your pasted code.
# If they already exist elsewhere, keep these imports here (no breaking).
from backend.technicals import build_technical_snapshot
from backend.smart_patterns import detect_smart_pattern, scan_smart_pattern_history

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
COL_STOCKS = "stocks"  # document id under bullsignals_ai
DOC_ACTIVE = "active_symbols"
DOC_DISCOVERY = "discovery_cursor"
DOC_HOTLIST = "market_hotlist"
DOC_BEARWATCH = "market_bearwatch"
DOC_HOMESCREEN = "homescreen_snapshot"

# Logging / behavior toggles (safe defaults)
LOG_EVERY_N = int((__import__("os").getenv("CRON_LOG_EVERY_N") or "1").strip() or "1")
DEBUG_FEATURES_SAMPLE = int((__import__("os").getenv("CRON_DEBUG_FEATURES_SAMPLE") or "0").strip() or "0")
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
        try:
            nf = getattr(backend, "BULLBRAIN_NUM_FEATURES", None)
            log(f"🧠 BullBrain loaded | num_features={nf}")
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

    return [s.upper() for s in shard if s]


# =========================================================
# SCAN UNIVERSE
# =========================================================

def build_scan_universe() -> Tuple[List[str], Dict[str, Any]]:
    active_raw = load_active_symbols()
    active = rank_active_symbols(active_raw)
    discovery = get_discovery_symbols()

    universe = list(dict.fromkeys([s.upper() for s in (MAG7 + active + discovery) if s]))
    meta = {
        "mag7": len(MAG7),
        "active_ranked": len(active),
        "discovery": len(discovery),
        "universe": len(universe),
    }
    return universe[:TOTAL_SCAN_LIMIT], meta


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


def _normalize_candles_list(raw: Any) -> List[Dict[str, Any]]:
    """
    get_candles() may return:
      A) list[dict] candles  ✅
      B) dict with key 'candles' holding list[dict]  ✅
      C) dict-of-arrays (open/high/low/close/volume/timestamp)  ✅ (convert)
    This function returns list[dict] always.
    """
    if raw is None:
        return []

    # A) already a list of candle dicts
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]

    # B) dict wrapper: {"candles": [...]}
    if isinstance(raw, dict):
        if isinstance(raw.get("candles"), list):
            return [c for c in raw["candles"] if isinstance(c, dict)]

        # C) dict-of-arrays
        keys = ["open", "high", "low", "close"]
        if all(k in raw for k in keys) and isinstance(raw.get("close"), list):
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


def _candles_list_to_arrays(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert get_candles() output (list of candle dicts) into the dict-of-arrays
    format expected by scan_smart_pattern_history().
    """
    out = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "timestamp": [],  # ms epoch expected by your scanner
    }

    for c in candles:
        if not isinstance(c, dict):
            continue

        # tolerate multiple key names
        t = c.get("t") or c.get("timestamp") or c.get("time")
        o = c.get("open")
        h = c.get("high")
        l = c.get("low")
        cl = c.get("close")
        v = c.get("volume")

        # require the core OHLC
        if o is None or h is None or l is None or cl is None:
            continue

        out["open"].append(float(o))
        out["high"].append(float(h))
        out["low"].append(float(l))
        out["close"].append(float(cl))
        out["volume"].append(float(v) if v is not None else 0.0)

        # normalize timestamp to ms epoch if possible
        try:
            if isinstance(t, (int, float)):
                # if it's seconds, convert to ms
                t_ms = int(t * 1000) if t < 10_000_000_000 else int(t)
                out["timestamp"].append(t_ms)
            else:
                out["timestamp"].append(None)
        except Exception:
            out["timestamp"].append(None)

    return out


def _build_candle_summary(candles: List[Dict[str, Any]], cap: int = 120) -> Dict[str, Any]:
    """
    Persist minimal candle data in stocks doc (UI-safe and cost-safe).
    """
    tail = [c for c in candles if isinstance(c, dict)][-cap:]
    return {
        "count": len(candles),
        "candles": [
            {
                "t": c.get("t") or c.get("timestamp") or c.get("time"),
                "close": c.get("close"),
            }
            for c in tail
        ],
    }


def _build_sparkline(candles: List[Dict[str, Any]], points: int = 30) -> List[float]:
    """
    Small sparkline array for UI (last N closes).
    """
    closes: List[float] = []
    for c in candles[-max(points * 2, points):]:
        if isinstance(c, dict) and c.get("close") is not None:
            try:
                closes.append(float(c["close"]))
            except Exception:
                pass
    return closes[-points:]


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

    def _build_candle_summary(lst: List[Dict[str, Any]], cap: int = 120) -> Dict[str, Any]:
        tail = [c for c in lst if isinstance(c, dict)][-cap:]
        return {
            "count": len(lst),
            "candles": [{"t": (c.get("t") or c.get("timestamp") or c.get("time")), "close": c.get("close")} for c in tail],
        }

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
    candle_summary = _build_candle_summary(candles_list, cap=120)
    sparkline = _build_sparkline(candles_list, points=30)

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
    # 4) Inference
    # ---------------------------------------------------------
    try:
        infer = backend.bullbrain_infer(feats_vec)
    except Exception as e:
        log_exc(f"{symbol} bullbrain_infer threw", e)
        return None

    if not isinstance(infer, dict):
        log(f"⛔ {symbol} infer invalid → skip")
        return None

    prob_up = float(infer.get("probability_up") or infer.get("raw_output") or 0.5)
    prob_down = float(infer.get("probability_down") or (1.0 - prob_up))

    if not _is_finite_number(prob_up) or not _is_finite_number(prob_down):
        log(f"⛔ {symbol} inference non-finite probs → skip")
        return None

    if prob_up >= 0.58:
        signal = "BUY"
    elif prob_down >= 0.58:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(max(prob_up, prob_down) * 100.0, 2)

    # ---------------------------------------------------------
    # 5) Quote (Firestore-only StockDetail needs this!)
    #    This does NOT write elsewhere; we persist inside the symbol doc.
    # ---------------------------------------------------------
    try:
        quote = fetch_equity_quote(symbol) or {}
        if not isinstance(quote, dict):
            quote = {}
    except Exception as e:
        log(f"⚠️ {symbol} fetch_equity_quote failed: {e}")
        quote = {}

    # Always keep at least a price if quote missing
    quote.setdefault("price", feat_dict.get("close"))
    quote["updated_at"] = utc_now_iso()

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
    # 7) Smart patterns (NEVER NULL)
    # ---------------------------------------------------------
    try:
        quote_for_pattern = {
            "price": feat_dict.get("close"),
            "changePct": feat_dict.get("return_1d"),  # fallback if quote changePct missing
        }
        smart_pattern = detect_smart_pattern(feat_dict, quote_for_pattern, technical)

        # history scanner MUST receive dict-of-arrays
        pattern_stats = scan_smart_pattern_history(symbol, candles_arrays) or {}

        # force non-null objects
        if not isinstance(smart_pattern, dict):
            smart_pattern = {
                "pattern": "NO CLEAR PATTERN",
                "winRate": None,
                "explanation": "Pattern engine produced no result.",
            }
        if not isinstance(pattern_stats, dict):
            pattern_stats = {
                "currentPattern": None,
                "historyForCurrent": None,
                "allPatterns": [],
                "note": "Pattern stats unavailable.",
            }
    except Exception as e:
        log_exc(f"{symbol} smart pattern failed", e)
        smart_pattern = {
            "pattern": "NO CLEAR PATTERN",
            "winRate": None,
            "explanation": "Pattern engine failed this run.",
        }
        pattern_stats = {
            "currentPattern": None,
            "historyForCurrent": None,
            "allPatterns": [],
            "note": "Pattern stats unavailable for this run.",
        }

    # ---------------------------------------------------------
    # 8) Insights (must never block)
    # ---------------------------------------------------------
    try:
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
    except Exception as e:
        log_exc(f"{symbol} generate_bull_insights failed", e)
        insights = {
            "oneLiner": "Insights unavailable.",
            "summaryLine": "Insights unavailable.",
            "trendSummary": "Insights unavailable.",
            "momentumSummary": "Insights unavailable.",
            "volumeSummary": "Insights unavailable.",
            "volatilitySummary": "Insights unavailable.",
            "combinedTechnicalSummary": "Insights unavailable.",
        }

    # ---------------------------------------------------------
    # 9) Build doc (everything your Firestore-only stockdetail needs)
    # ---------------------------------------------------------
    doc = {
        "symbol": symbol,
        "company_name": COMPANY_NAMES.get(symbol, symbol),

        "quote": quote,

        "bullbrain": {
            "signal": signal,
            "confidence": confidence,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
        },

        "features_meta": feat_dict,
        "technical": technical,

        "smartPattern": smart_pattern,
        "patternStats": pattern_stats,

        # UI helpers
        "candles": candle_summary,
        "sparkline": sparkline,

        "insights": insights,

        "computed_at": utc_now_iso(),
        "schema_version": "v1",
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
        log(f"✅ {symbol} wrote stocks/symbols/{symbol} | signal={signal} conf={confidence}% | {dt:.2f}s")
        return doc

    except Exception as e:
        log_exc(f"{symbol} Firestore write failed", e)
        return None


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

    log(f"✅ saved hotlist={len(hotlist)} bearwatch={len(bearwatch)}")


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
    log("✅ homescreen_snapshot mag7 updated")


# =========================================================
# ENTRYPOINT
# =========================================================
def main():
    run_id = utc_now_iso()
    log(f"🚀 cron start | run_id={run_id}")

    ensure_bullbrain_loaded()

    # ---------------------------------------------------------
    # 0️⃣ HOMESCREEN BASELINE FIRST (SPY / QQQ / Commodities)
    # ---------------------------------------------------------
    try:
        ensure_market_overview_and_baseline_carousel()
        log("🏠 homescreen baseline ensured (us_market + commodities)")
    except Exception as e:
        log_exc("homescreen baseline failed", e)

    # ---------------------------------------------------------
    # 1️⃣ MAG-7 FIRST (ALWAYS)
    # ---------------------------------------------------------
    mag7 = [s.upper() for s in MAG7 if s]
    log(f"🧲 PHASE 1 — MAG7 | count={len(mag7)} | symbols={mag7}")

    # ---------------------------------------------------------
    # 2️⃣ ACTIVE SYMBOLS (USER INTENT)
    # ---------------------------------------------------------
    active_raw = load_active_symbols()
    active_ranked = rank_active_symbols(active_raw)
    log(
        f"⭐ PHASE 2 — Active symbols | "
        f"count={len(active_ranked)} | symbols={active_ranked[:10]}"
    )

    # ---------------------------------------------------------
    # 3️⃣ DISCOVERY SHARD (SP500 ROTATION)
    # ---------------------------------------------------------
    discovery = get_discovery_symbols()
    log(
        f"🔍 PHASE 3 — Discovery shard | "
        f"count={len(discovery)} | symbols={discovery[:10]}"
    )

    # ---------------------------------------------------------
    # 🔀 MERGE UNIVERSE (PRIORITY-SAFE, NO DUPLICATES)
    # ---------------------------------------------------------
    scan_symbols = list(
        dict.fromkeys(
            [*mag7, *active_ranked, *discovery]
        )
    )[:TOTAL_SCAN_LIMIT]

    log(
        f"📦 scan universe built | total={len(scan_symbols)} | "
        f"mag7={len(mag7)} active={len(active_ranked)} discovery={len(discovery)}"
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

    # ---------------------------------------------------------
    # 📊 HOTLIST + BEARWATCH
    # ---------------------------------------------------------
    hotlist, bearwatch = build_hotlist_bearwatch(results)
    save_market_lists(hotlist, bearwatch)

    # ---------------------------------------------------------
    # 🏠 HOMESCREEN MAG-7 SNAPSHOT (UNCHANGED CONTRACT)
    # ---------------------------------------------------------
    save_homescreen_snapshot()

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
