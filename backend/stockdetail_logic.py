# backend/stockdetail_logic.py
"""
StockDetail core business logic (PURE MODULE)

Rules:
- MUST NOT import FastAPI
- MUST NOT import firestore helpers
- MUST NOT import stockdetail_builder / cron / main
- Safe to be imported by BOTH cron + API
"""

import datetime
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# ------------------------------------------------------------------
# External dependencies already used in main.py
# ------------------------------------------------------------------
from main import (
    backend_fetch_quote,
    fetch_daily_candles,
    bullbrain_infer,
    build_technical_snapshot,
    _class_probs_from_prob_up,
    get_symbol_news,
    get_stockdetail_grok,
    _hybrid_from_probs,
    bullbrain_model,
    BULLBRAIN_VERSION,
)

# Smart pattern + feature computation
from main import (
    scan_smart_pattern_history,
    compute_bullbrain_features,
)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def utc_iso() -> str:
    return (
        datetime.datetime.utcnow()
        .replace(microsecond=0)
        .isoformat()
        + "Z"
    )


def safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


# ------------------------------------------------------------------
# Candle payload builder (shared by API + cron)
# ------------------------------------------------------------------
def build_candles_payload(
    symbol: str,
    candles: dict,
    limit_candles: int = 180,
) -> Optional[dict]:

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

    use_n = min(limit_candles, n)
    start = n - use_n

    items = []
    for i in range(start, n):
        ts = ts_list[i] if i < len(ts_list) else None
        if ts:
            t_iso = (
                datetime.datetime.utcfromtimestamp(ts / 1000)
                .replace(microsecond=0)
                .isoformat()
                + "Z"
            )
        else:
            t_iso = utc_iso()

        items.append(
            {
                "t": t_iso,
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


# ------------------------------------------------------------------
# BullBrain block (v2 – 48 features)
# ------------------------------------------------------------------
def build_bullbrain_block(
    candles: dict,
) -> tuple[Optional[dict], Optional[dict], Optional[float], Optional[float]]:

    if not candles or bullbrain_model is None:
        return None, None, None, None

    features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
    inference = bullbrain_infer(features_vec)

    prob_up = float(
        inference.get("probability_up")
        or inference.get("raw_output")
        or 0.5
    )

    class_probs = _class_probs_from_prob_up(prob_up)

    bullbrain_block = {
        "version": BULLBRAIN_VERSION,
        "signal": inference.get("signal"),
        "confidence": inference.get("confidence"),
        "probabilities": class_probs,
        "raw": {
            "prob_up": prob_up,
            "prob_down": 1.0 - prob_up,
        },
    }

    return bullbrain_block, feature_dict, float(last_close), prob_up


# ------------------------------------------------------------------
# Smart Pattern (UI-safe wrapper)
# ------------------------------------------------------------------
def build_smart_pattern_safe(
    symbol: str,
    candles: dict,
) -> Dict[str, Any]:

    raw = None
    try:
        raw = scan_smart_pattern_history(symbol, candles)
    except Exception:
        raw = None

    safe_pattern = {
        "pattern": None,
        "headline": None,
        "winRate": None,
        "occurrences": 0,
        "samples": [],
        "forwardReturns": {},
    }

    pattern_dates = []
    pattern_stats = raw

    if raw:
        cp = raw.get("currentPattern")
        hist = raw.get("historyForCurrent")

        if cp and cp.get("pattern"):
            safe_pattern = {
                "pattern": cp.get("pattern"),
                "headline": cp.get("headline"),
                "winRate": cp.get("winRate"),
                "occurrences": hist.get("occurrences", 0) if hist else 0,
                "samples": hist.get("samples", []) if hist else [],
                "forwardReturns": hist.get("forwardReturns", {}) if hist else {},
            }

            if hist and hist.get("samples"):
                pattern_dates = hist["samples"][:5]

    return {
        "smartPattern": safe_pattern,
        "patternDates": pattern_dates,
        "patternStats": pattern_stats,
    }


# ------------------------------------------------------------------
# MASTER BUILDER (used by API + cron)
# ------------------------------------------------------------------
def build_stockdetail_core(
    symbol: str,
    limit_candles: int = 180,
    force_grok: bool = False,
) -> Dict[str, Any]:

    symbol = symbol.upper()

    quote = backend_fetch_quote(symbol)
    candles = fetch_daily_candles(symbol)

    # BullBrain
    bullbrain_block, feature_dict, last_close, bull_prob_up = (
        build_bullbrain_block(candles)
    )

    if last_close is None and quote:
        last_close = safe_float(quote.get("current")) or 0.0

    # Technical snapshot
    technical = None
    if feature_dict is not None and last_close is not None:
        technical = build_technical_snapshot(
            symbol, feature_dict, last_close
        )

    # Candles
    candles_payload = build_candles_payload(
        symbol, candles, limit_candles
    )

    # News + Grok
    news = get_symbol_news(symbol, limit=8)
    grok_pack = get_stockdetail_grok(
        symbol, quote, technical, force=force_grok
    )

    grok_prob_up = grok_pack.get("prob_up")

    # Hybrid
    hybrid_p, hybrid_signal, hybrid_conf = _hybrid_from_probs(
        bull_prob_up, grok_prob_up
    )

    # Smart Pattern
    sp = build_smart_pattern_safe(symbol, candles)

    return {
        "symbol": symbol,
        "asOf": utc_iso(),
        "quote": quote,
        "price": last_close,
        "bullbrain": bullbrain_block,
        "features": feature_dict,
        "technical": technical,
        "candles": candles_payload,
        "news": news,
        "grok": grok_pack,
        "hybridProbUp": hybrid_p,
        "hybridSignal": hybrid_signal,
        "hybridScore": hybrid_conf,
        "smartPattern": sp["smartPattern"],
        "patternDates": sp["patternDates"],
        "patternStats": sp["patternStats"],
    }
