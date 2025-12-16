# backend/stockdetail_cron.py
"""
BullSignalsAI — StockDetail Cron (Firestore Precompute)

Purpose:
- Precompute /stockdetail payloads (quote, candles, bullbrain v2, technical,
  news, grok, hybrid, smartPattern)
- Store into Firestore so the mobile UI loads instantly
"""

import os
import time
import traceback
import datetime
from typing import Any, Dict, List, Optional

from google.cloud import firestore

# ----------------------------
# Schema & Firestore helpers
# ----------------------------
from backend.schema_versions import STOCKDETAIL_SCHEMA_VERSION
from backend.firestore_paths import stockdetail_doc_ref

# ----------------------------
# Reuse existing backend logic
# ----------------------------
from main import (
    backend_fetch_quote,
    fetch_daily_candles,
    compute_bullbrain_features,
    bullbrain_infer,
    build_technical_snapshot,
    _class_probs_from_prob_up,
    get_symbol_news,
    get_stockdetail_grok,
    _hybrid_from_probs,
    scan_smart_pattern_history,
    bullbrain_model,
    BULLBRAIN_VERSION,
)

# ----------------------------
# Config
# ----------------------------
DEFAULT_LIMIT_CANDLES = int(os.getenv("STOCKDETAIL_LIMIT_CANDLES", "180"))
DEFAULT_TTL_MINUTES = int(os.getenv("STOCKDETAIL_TTL_MINUTES", "15"))
MAX_SYMBOLS_PER_RUN = int(os.getenv("STOCKDETAIL_MAX_SYMBOLS", "120"))

# Example:
# STOCKDETAIL_UNIVERSE="AAPL,TSLA,NVDA,MSFT"
STOCKDETAIL_UNIVERSE = os.getenv("STOCKDETAIL_UNIVERSE", "").strip()


# ----------------------------
# Time helpers
# ----------------------------
def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


def iso(dt: datetime.datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"


def compute_expires_at_ts(now: datetime.datetime) -> int:
    return int(now.timestamp()) + DEFAULT_TTL_MINUTES * 60


def safe_float(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


# ----------------------------
# Builders
# ----------------------------
def build_candles_payload(symbol: str, candles: dict) -> Optional[dict]:
    if not candles:
        return None

    closes = candles.get("close") or []
    highs = candles.get("high") or []
    lows = candles.get("low") or []
    opens = candles.get("open") or closes
    vols = candles.get("volume") or []
    ts_list = candles.get("timestamp") or []

    n = len(closes)
    if n == 0:
        return None

    use_n = min(DEFAULT_LIMIT_CANDLES, n)
    start = n - use_n

    items = []
    for i in range(start, n):
        if i < len(ts_list) and ts_list[i]:
            dt = datetime.datetime.utcfromtimestamp(ts_list[i] / 1000.0)
        else:
            dt = utcnow() - datetime.timedelta(days=(n - 1 - i))

        items.append(
            {
                "t": iso(dt),
                "open": safe_float(opens[i]),
                "high": safe_float(highs[i]),
                "low": safe_float(lows[i]),
                "close": safe_float(closes[i]),
                "volume": safe_float(vols[i]),
            }
        )

    return {
        "symbol": symbol,
        "source": candles.get("source", "polygon"),
        "candles": items,
    }


def build_bullbrain_block(candles: dict):
    if not candles or bullbrain_model is None:
        return None, None, None, None

    features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
    inference = bullbrain_infer(features_vec)

    prob_up = float(inference.get("probability_up") or inference.get("raw_output") or 0.5)
    class_probs = _class_probs_from_prob_up(prob_up)

    bullbrain = {
        "version": BULLBRAIN_VERSION,
        "signal": inference.get("signal"),
        "confidence": inference.get("confidence"),
        "probabilities": class_probs,
        "raw": {"prob_up": prob_up, "prob_down": 1.0 - prob_up},
    }

    return bullbrain, feature_dict, float(last_close), prob_up


def build_smart_pattern_safe(symbol: str, candles: dict) -> Dict[str, Any]:
    try:
        raw = scan_smart_pattern_history(symbol, candles)
    except Exception:
        raw = None

    safe = {
        "pattern": None,
        "headline": None,
        "winRate": None,
        "occurrences": 0,
        "samples": [],
        "forwardReturns": {},
    }
    dates = []

    if raw:
        cp = raw.get("currentPattern")
        hist = raw.get("historyForCurrent")
        if cp and cp.get("pattern"):
            safe = {
                "pattern": cp.get("pattern"),
                "headline": cp.get("headline"),
                "winRate": cp.get("winRate"),
                "occurrences": hist.get("occurrences") if hist else 0,
                "samples": hist.get("samples") if hist else [],
                "forwardReturns": hist.get("forwardReturns") if hist else {},
            }
            if hist and hist.get("samples"):
                dates = hist["samples"][:5]

    return {
        "smartPattern": safe,
        "patternDates": dates,
        "patternStats": raw,
    }


def build_stockdetail_payload(symbol: str, force_grok: bool = False) -> Dict[str, Any]:
    now = utcnow()
    symbol = symbol.upper()

    quote = backend_fetch_quote(symbol)
    candles = fetch_daily_candles(symbol)

    bullbrain, feature_dict, last_close, bull_prob_up = build_bullbrain_block(candles)

    if last_close is None and quote:
        last_close = safe_float(quote.get("current")) or 0.0

    technical = None
    if feature_dict is not None and last_close is not None:
        technical = build_technical_snapshot(symbol, feature_dict, last_close)

    candles_payload = build_candles_payload(symbol, candles)

    news = get_symbol_news(symbol, limit=8)
    grok = get_stockdetail_grok(symbol, quote, technical, force=force_grok)
    grok_prob_up = grok.get("prob_up")

    hybrid_p, hybrid_signal, hybrid_conf = _hybrid_from_probs(
        bull_prob_up, grok_prob_up
    )

    sp = build_smart_pattern_safe(symbol, candles)

    return {
        "symbol": symbol,
        "schemaVersion": STOCKDETAIL_SCHEMA_VERSION,
        "asOf": iso(now),
        "computedAt": iso(now),
        "expiresAt": compute_expires_at_ts(now),

        "quote": quote,
        "price": last_close,
        "bullbrain": bullbrain,
        "features": feature_dict,
        "technical": technical,
        "candles": candles_payload,
        "news": news,
        "grok": grok,
        "hybridProbUp": hybrid_p,
        "hybridSignal": hybrid_signal,
        "hybridScore": hybrid_conf,

        "smartPattern": sp["smartPattern"],
        "patternDates": sp["patternDates"],
        "patternStats": sp["patternStats"],
    }


# ----------------------------
# Cron runner
# ----------------------------
def should_skip(existing: dict, force: bool) -> bool:
    if force or not existing:
        return False
    try:
        return existing.get("expiresAt", 0) > int(time.time())
    except Exception:
        return False


def get_universe() -> List[str]:
    if not STOCKDETAIL_UNIVERSE:
        return []
    syms = [s.strip().upper() for s in STOCKDETAIL_UNIVERSE.split(",") if s.strip()]
    return list(dict.fromkeys(syms))[:MAX_SYMBOLS_PER_RUN]


def run(force: bool = False, force_grok: bool = False):
    db = firestore.Client()
    symbols = get_universe()

    if not symbols:
        print("⚠️ No symbols to process (STOCKDETAIL_UNIVERSE empty)")
        return

    print(f"🚀 StockDetail cron | symbols={len(symbols)} ttl={DEFAULT_TTL_MINUTES}m")

    for i, symbol in enumerate(symbols, 1):
        t0 = time.time()
        try:
            ref = stockdetail_doc_ref(symbol)
            snap = ref.get()
            existing = snap.to_dict() if snap.exists else None

            if should_skip(existing, force):
                print(f"⏭️ [{i}/{len(symbols)}] {symbol} skip (fresh)")
                continue

            payload = build_stockdetail_payload(symbol, force_grok)
            ref.set(payload, merge=True)

            ms = int((time.time() - t0) * 1000)
            print(f"✅ [{i}/{len(symbols)}] {symbol} updated ({ms}ms)")

        except Exception as e:
            print(f"❌ [{i}/{len(symbols)}] {symbol} failed: {e}")
            traceback.print_exc()

    print("✅ StockDetail cron finished")


if __name__ == "__main__":
    force = os.getenv("STOCKDETAIL_FORCE", "false").lower() == "true"
    force_grok = os.getenv("STOCKDETAIL_FORCE_GROK", "false").lower() == "true"
    run(force=force, force_grok=force_grok)
