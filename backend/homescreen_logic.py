# backend/homescreen_logic.py
# ------------------------------------------------------------
# HomeScreen — Pure Logic Layer
# Builds global MAG7 snapshot blocks using:
#   - Latest quote
#   - BullBrain v2 (48 features)
#   - Optional Grok one-liner
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

from backend.market_data import (
    fetch_quote,
    fetch_daily_candles,
)

from backend.bullbrain import (
    ensure_bullbrain_loaded,
    compute_bullbrain_features,
    bullbrain_infer,
)

from backend.technicals import build_technical_snapshot
from backend.grok_ai import get_ticker_summary_grok

from symbols_clean import COMPANY_NAMES


# ------------------------------------------------------------
# Default universe
# ------------------------------------------------------------
DEFAULT_MAG7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


# ------------------------------------------------------------
# Market mood (derived from MAG7 BullBrain probabilities)
# ------------------------------------------------------------
def compute_market_mood_from_mag7(
    mag7_blocks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not mag7_blocks:
        return {
            "label": "Neutral",
            "score": 50,
            "explanation": "No MAG7 data available",
        }

    probs: List[float] = []

    for t in mag7_blocks:
        p = t.get("bullbrain", {}).get("probability_up")
        probs.append(_safe_float(p, 0.5))

    avg_prob = sum(probs) / max(len(probs), 1)
    score = int(round(avg_prob * 100))

    if score >= 58:
        label = "Bullish"
    elif score <= 42:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "label": label,
        "score": score,
        "explanation": "Based on MAG7 BullBrain probability average",
    }


# ------------------------------------------------------------
# Single MAG7 ticker block
# ------------------------------------------------------------
def build_mag7_ticker_block(
    symbol: str,
    include_grok: bool = True,
) -> Optional[Dict[str, Any]]:
    sym = (symbol or "").upper().strip()
    if not sym:
        return None

    company_name = COMPANY_NAMES.get(sym, f"{sym} Corp.")

    # ----------------------------
    # Quote
    # ----------------------------
    quote = fetch_quote(sym) or {}
    price = quote.get("price")
    change_pct = quote.get("changePct")
    price_timestamp = quote.get("timestamp")  # epoch or ISO, passthrough

    if price is None:
        return None

    # ----------------------------
    # Candles + BullBrain
    # ----------------------------
    candles = fetch_daily_candles(sym)
    if not candles:
        # Quote-only fallback
        return {
            "symbol": sym,
            "company_name": company_name,
            "price": price,
            "price_timestamp": price_timestamp,
            "change_pct": change_pct,
            "bullbrain": {
                "version": "v2-48f",
                "signal": "HOLD",
                "confidence": 50,
                "probability_up": 0.5,
            },
            "summary": "Insufficient historical data for model inference.",
            "technical_hint": "",
        }

    ensure_bullbrain_loaded()

    feats_vec, feat_dict, last_close = compute_bullbrain_features(candles)
    infer = bullbrain_infer(feats_vec) or {}

    prob_up = infer.get("probability_up") or infer.get("raw_output") or 0.5
    prob_up = _safe_float(prob_up, 0.5)

    signal = (infer.get("signal") or "HOLD").upper()

    confidence = infer.get("confidence")
    if confidence is None:
        confidence = int(round(50 + (prob_up - 0.5) * 100))
    confidence = max(1, min(99, int(confidence)))

    # ----------------------------
    # Lightweight technical hint
    # ----------------------------
    technical_hint = ""
    try:
        tech = build_technical_snapshot(sym, feat_dict or {}, last_close)
        technical_hint = (
            tech.get("headline")
            or tech.get("summary")
            or ""
        )[:80]
    except Exception:
        technical_hint = ""

    # ----------------------------
    # Grok one-liner (cron-safe)
    # ----------------------------
    summary = ""
    if include_grok:
        try:
            summary = get_ticker_summary_grok(
                symbol=sym,
                name=company_name,
                price=price,
                changePct=change_pct,
                signal=signal,
                confidence=confidence,
                probability_up=prob_up,
            ) or ""
        except Exception:
            summary = ""

    if not summary:
        summary = f"{signal} bias with {confidence}% BullBrain confidence."

    # ----------------------------
    # Final block
    # ----------------------------
    return {
        "symbol": sym,
        "company_name": company_name,
        "price": price,
        "price_timestamp": price_timestamp,
        "change_pct": change_pct,
        "bullbrain": {
            "version": "v2-48f",
            "signal": signal,
            "confidence": confidence,
            "probability_up": prob_up,
        },
        "summary": summary,
        "technical_hint": technical_hint,
    }


# ------------------------------------------------------------
# HomeScreen raw payload (PURE LOGIC)
# ------------------------------------------------------------
def build_homescreen_raw(
    universe: Optional[List[str]] = None,
    include_grok: bool = True,
    include_carousel: bool = False,
) -> Dict[str, Any]:
    symbols = universe or DEFAULT_MAG7

    mag7_blocks: List[Dict[str, Any]] = []

    for sym in symbols:
        block = build_mag7_ticker_block(sym, include_grok=include_grok)
        if block:
            mag7_blocks.append(block)

    market_mood = compute_market_mood_from_mag7(mag7_blocks)

    payload: Dict[str, Any] = {
        "mag7": {
            "universe": [s.upper() for s in symbols],
            "count": len(mag7_blocks),
            "tickers": mag7_blocks,
        },
        "market_mood": market_mood,
    }

    # Carousel intentionally left empty here.
    # It will be composed in cron from market snapshots.
    if include_carousel:
        payload["carousel"] = {}

    return payload
