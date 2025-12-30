# backend/market_bearwatch_logic.py
# ---------------------------------------------------------
# BullSignalsAI — Market BearWatch Logic (SELL / HOLD side)
# ---------------------------------------------------------
# Pure backend logic:
# - No Firestore
# - No FastAPI
# - No cron
# - Safe to reuse from market_cron.py or anywhere else
# ---------------------------------------------------------

import math
from typing import List, Dict, Any, Tuple

from symbols_clean import COMPANY_NAMES


# ---------------------------------------------------------
# Safe feature getter
# ---------------------------------------------------------
def _safe_feat(feat_dict: dict, key: str):
    try:
        v = float(feat_dict.get(key, float("nan")))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# ---------------------------------------------------------
# SELL / HOLD explanation builder (verbatim behavior)
# ---------------------------------------------------------
def build_bear_explanations(
    symbol: str,
    prob_up: float,
    prob_down: float,
    kind: str,
    feat_dict: dict,
) -> Tuple[str, str]:
    up = prob_up * 100.0
    down = prob_down * 100.0

    rsi = _safe_feat(feat_dict, "rsi14")
    ret10 = _safe_feat(feat_dict, "return_10d")
    price_vs_20 = _safe_feat(feat_dict, "price_vs_sma20_pct")
    vol_vs_ma20 = _safe_feat(feat_dict, "volume_vs_ma20_pct")
    trend = _safe_feat(feat_dict, "trend_strength_20")

    if kind == "STRONG_SELL":
        label = "strong SELL"
        short = (
            "Bearish pressure dominates — about "
            f"{down:.1f}% chance of downside vs {up:.1f}% upside."
        )
    elif kind == "SELL":
        label = "SELL"
        short = (
            "Bearish bias — about "
            f"{down:.1f}% chance of downside vs {up:.1f}% upside."
        )
    else:
        label = "HOLD"
        short = (
            "No clear edge — model sees roughly "
            f"{up:.1f}% up vs {down:.1f}% down."
        )

    parts = []

    if ret10 is not None:
        parts.append(f"~{ret10:+.1f}% move over the last 10 sessions")

    if price_vs_20 is not None:
        side = "below" if price_vs_20 <= 0 else "above"
        parts.append(f"price is {abs(price_vs_20):.1f}% {side} its 20-day average")

    if rsi is not None:
        if rsi <= 40:
            parts.append(f"RSI ≈ {rsi:.0f} (weak momentum, below 40)")
        elif rsi <= 50:
            parts.append(f"RSI ≈ {rsi:.0f} (neutral / slightly weak)")
        else:
            parts.append(f"RSI ≈ {rsi:.0f} (still not deeply oversold)")

    if vol_vs_ma20 is not None and abs(vol_vs_ma20) >= 15:
        side = "higher" if vol_vs_ma20 > 0 else "lighter"
        parts.append(f"volume is {abs(vol_vs_ma20):.0f}% {side} than 20-day normal")

    if trend is not None:
        if trend <= -0.6:
            parts.append("downtrend looks strong on the 1-month window")
        elif trend >= 0.6:
            parts.append("trend is mixed — some strength despite this short-term risk")

    tech_sentence = ""
    if parts:
        tech_sentence = " | ".join(parts[:3]) + ". "

    if label in ("strong SELL", "SELL"):
        risk_sentence = (
            "Signal: SELL zone. Many traders use this type of setup to reduce exposure "
            "or tighten stops instead of adding fresh risk."
        )
    else:
        risk_sentence = (
            "Signal: HOLD. Price action looks more sideways or uncertain — waiting for a "
            "clearer edge can often be safer."
        )

    return short, tech_sentence + risk_sentence


# ---------------------------------------------------------
# Build BearWatch (Top 5 SELL / HOLD)
# ---------------------------------------------------------
def build_market_bearwatch(
    all_symbols: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Input:
      all_symbols: output from BullBrain scan
        [
          {
            symbol,
            prob_up_raw,
            prob_down_raw,
            confidence,
            kind,
            feat_dict
          }
        ]

    Output:
      {
        count,
        bearwatch,
        updated_at (added by caller)
      }
    """

    bear_candidates = []

    for item in all_symbols:
        kind = item["kind"]
        if kind not in ("STRONG_SELL", "SELL", "HOLD"):
            continue

        sym = item["symbol"]
        prob_up = item["prob_up_raw"]
        prob_down = item["prob_down_raw"]
        feat_dict = item["feat_dict"]

        short, risk = build_bear_explanations(
            sym, prob_up, prob_down, kind, feat_dict
        )

        signal_label = "SELL" if kind in ("STRONG_SELL", "SELL") else "HOLD"

        bear_candidates.append(
            {
                "symbol": sym,
                "company_name": COMPANY_NAMES.get(sym, sym),
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "confidence": item["confidence"],
                "signal": signal_label,
                "kind": kind,
                "explanation_short": short,
                "explanation_risk": risk,
            }
        )

    # Sort by downside probability
    bear_candidates.sort(key=lambda x: x["prob_down"], reverse=True)

    bearwatch = bear_candidates[:5]

    return {
        "count": len(bearwatch),
        "bearwatch": bearwatch,
    }
