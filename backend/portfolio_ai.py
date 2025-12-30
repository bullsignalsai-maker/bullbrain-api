# backend/portfolio_ai.py

from typing import Dict, Any
import time

from backend.market_data import fetch_daily_candles
from backend.bullbrain import compute_bullbrain_features, bullbrain_infer


# ---------------------------------------------------------------
# AI INSIGHT (DYNAMIC) — BullBrain v2 + Rebalancing + 5-Day Trend
# ---------------------------------------------------------------

# 15-min cache (900 sec)
_AI_CACHE: Dict[Any, Dict[str, Any]] = {}


def _set_cache(key, data):
    _AI_CACHE[key] = {
        "data": data,
        "ts": time.time(),
    }


def _get_cache(key):
    item = _AI_CACHE.get(key)
    if not item:
        return None
    if time.time() - item["ts"] > 900:
        return None
    return item["data"]


def portfolio_ai_insight(
    symbol: str,
    allocation_pct: float = 0.0,
    gain_pct: float = 0.0,
    position_value: float = 0.0,
    portfolio_total_value: float = 0.0,
):
    """
    Dynamic BullBrain v2 insight + 5-day trend probability + rebalancing suggestions.
    Backend-only logic. No FastAPI.
    """

    symbol = symbol.upper()

    cache_key = (
        symbol,
        round(allocation_pct, 2),
        round(gain_pct, 2),
        round(position_value, 2),
        round(portfolio_total_value, 2),
    )

    cached = _get_cache(cache_key)
    if cached:
        return cached

    try:
        # 1) Fetch candles
        candles = fetch_daily_candles(symbol)
        if not candles:
            return {"error": "Insufficient candle data"}

        # 2) Compute features
        features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
        if features_vec is None:
            return {"error": "Feature computation failed"}

        # 3) Model inference
        out = bullbrain_infer(features_vec)
        prob_up = float(out.get("probability_up") or 0.5)
        signal = out.get("signal") or "NEUTRAL"

        # Trend
        trend = (
            "Bullish" if signal == "BUY"
            else "Bearish" if signal == "SELL"
            else "Neutral"
        )

        # Expected move
        vol = feature_dict.get("volatility_5d", 0.02)
        expected_move = vol * (prob_up * 2 - 1)
        expected_move_pct = f"{expected_move * 100:+.2f}%"

        confidence_pct = f"{prob_up * 100:.0f}%"

        # Risk
        if vol < 0.015:
            risk = "Low"
        elif vol < 0.035:
            risk = "Medium"
        else:
            risk = "High"

        # Pattern
        sma5 = feature_dict.get("sma5", 0)
        sma20 = feature_dict.get("sma20", 0)
        if sma5 > sma20:
            pattern = "Short-term Momentum"
        elif sma5 < sma20:
            pattern = "Reversal Risk"
        else:
            pattern = "Sideways Consolidation"

        five_day_prob = f"{int(prob_up * 100)}% Bullish"

        # Rebalancing
        suggestion = "No rebalancing needed."
        if portfolio_total_value > 0 and last_close > 0:
            diff = (allocation_pct / 100) - prob_up
            dollar_diff = abs(diff) * portfolio_total_value
            shares_diff = round(dollar_diff / last_close)

            if diff > 0.02:
                suggestion = (
                    f"Trim ~{shares_diff} shares (≈${dollar_diff:,.0f}). "
                    f"This reduces {symbol} to an optimal allocation."
                )
            elif diff < -0.02:
                suggestion = (
                    f"Add ~{shares_diff} shares (≈${dollar_diff:,.0f}). "
                    f"{symbol} shows improving momentum."
                )

        message = (
            f"AI View Today:\n"
            f"{symbol} trend: {trend}\n"
            f"Expected move: {expected_move_pct}\n"
            f"Risk: {risk}\n"
            f"Confidence: {confidence_pct}\n"
            f"Pattern: {pattern}\n"
            f"5-Day Bullish Probability: {five_day_prob}\n"
            f"(BullBrain v2)"
        )

        result = {
            "symbol": symbol,
            "trend": trend,
            "expected_move": expected_move_pct,
            "risk": risk,
            "confidence": confidence_pct,
            "pattern": pattern,
            "five_day_prob": five_day_prob,
            "rebalancing": suggestion,
            "last_price": last_close,
            "message": message,
        }

        _set_cache(cache_key, result)
        return result

    except Exception as e:
        print("Portfolio AI insight error:", e)
        return {"error": "AI insight unavailable"}
