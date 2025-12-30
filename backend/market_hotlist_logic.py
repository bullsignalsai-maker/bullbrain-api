# backend/market_hotlist_logic.py
# ---------------------------------------------------------
# BullSignalsAI — Market Hotlist Logic (BUY side only)
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
# BUY explanation builder (verbatim behavior)
# ---------------------------------------------------------
def build_buy_explanations(
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

    if kind == "STRONG_BUY":
        label = "strong BUY"
    elif kind == "BUY":
        label = "BUY"
    else:
        label = "watchlist BUY"

    short = (
        f"BullBrain flags a {label} setup — about "
        f"{up:.1f}% chance of upside vs {down:.1f}% downside."
    )

    parts = []

    if ret10 is not None:
        parts.append(f"~{ret10:+.1f}% move over the last 10 sessions")

    if price_vs_20 is not None:
        side = "above" if price_vs_20 >= 0 else "below"
        parts.append(f"price is {abs(price_vs_20):.1f}% {side} its 20-day average")

    if rsi is not None:
        if rsi >= 60:
            parts.append(f"RSI ≈ {rsi:.0f} (strong momentum zone)")
        elif rsi >= 50:
            parts.append(f"RSI ≈ {rsi:.0f} (slightly bullish, just above 50)")
        else:
            parts.append(f"RSI ≈ {rsi:.0f} (still not overbought)")

    if vol_vs_ma20 is not None and abs(vol_vs_ma20) >= 15:
        side = "higher" if vol_vs_ma20 > 0 else "lighter"
        parts.append(f"volume is {abs(vol_vs_ma20):.0f}% {side} than 20-day normal")

    if trend is not None:
        if trend >= 0.6:
            parts.append("trend strength looks solid on the 1-month window")
        elif trend <= -0.6:
            parts.append("trend is still shaky, so be extra careful")

    tech_sentence = ""
    if parts:
        tech_sentence = " | ".join(parts[:3]) + ". "

    risk_sentence = (
        "This is not financial advice — consider position sizing, a clear stop-loss, "
        "and your own research before acting."
    )

    return short, tech_sentence + risk_sentence


# ---------------------------------------------------------
# Build Hotlist (Top 5 BUY)
# ---------------------------------------------------------
def build_market_hotlist(
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
        hotlist,
        updated_at (added by caller)
      }
    """

    buy_candidates = []

    for item in all_symbols:
        kind = item["kind"]
        if kind not in ("STRONG_BUY", "BUY"):
            continue

        sym = item["symbol"]
        prob_up = item["prob_up_raw"]
        prob_down = item["prob_down_raw"]
        feat_dict = item["feat_dict"]

        short, risk = build_buy_explanations(
            sym, prob_up, prob_down, kind, feat_dict
        )

        buy_candidates.append(
            {
                "symbol": sym,
                "company_name": COMPANY_NAMES.get(sym, sym),
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "confidence": item["confidence"],
                "signal": "BUY",
                "kind": kind,
                "explanation_short": short,
                "explanation_risk": risk,
            }
        )

    # Sort strongest first
    buy_candidates.sort(key=lambda x: x["prob_up"], reverse=True)
    hotlist = buy_candidates[:5]

    # -----------------------------------------------------
    # Fallback fill (WATCHLIST_BUY)
    # -----------------------------------------------------
    if len(hotlist) < 5:
        already = {h["symbol"] for h in hotlist}
        extras = [x for x in all_symbols if x["symbol"] not in already]
        extras.sort(key=lambda x: x["prob_up_raw"], reverse=True)

        needed = 5 - len(hotlist)
        for extra in extras[:needed]:
            sym = extra["symbol"]
            prob_up = extra["prob_up_raw"]
            prob_down = extra["prob_down_raw"]
            feat_dict = extra["feat_dict"]

            short, risk = build_buy_explanations(
                sym, prob_up, prob_down, "WATCHLIST_BUY", feat_dict
            )

            hotlist.append(
                {
                    "symbol": sym,
                    "company_name": COMPANY_NAMES.get(sym, sym),
                    "prob_up": round(prob_up, 4),
                    "prob_down": round(prob_down, 4),
                    "confidence": extra["confidence"],
                    "signal": "BUY",
                    "kind": "WATCHLIST_BUY",
                    "explanation_short": short,
                    "explanation_risk": risk,
                }
            )

    return {
        "count": len(hotlist),
        "hotlist": hotlist[:5],
    }
