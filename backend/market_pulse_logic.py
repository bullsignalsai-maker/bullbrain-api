# backend/market_pulse_logic.py
# ------------------------------------------------------------
# Market Pulse Intelligence (Overview, Hotlist, Bearwatch)
# ------------------------------------------------------------

from typing import Dict, Any, List
import statistics
import time

from backend.market_data import (
    fetch_quote,
    fetch_daily_candles,
    fetch_market_news,
)

from backend.bullbrain import (
    compute_bullbrain_features,
    bullbrain_infer,
)

from backend.smart_patterns import detect_smart_pattern


# ============================================================
# MARKET OVERVIEW
# ============================================================

def build_market_overview(symbols: List[str]) -> Dict[str, Any]:
    """
    Builds a lightweight market-wide overview used by Market screen header.
    """

    adv = 0
    dec = 0
    unchanged = 0
    pct_moves = []

    details = []

    for sym in symbols:
        q = fetch_quote(sym)
        if not q:
            continue

        chg = q.get("changePct") or 0

        if chg > 0.2:
            adv += 1
        elif chg < -0.2:
            dec += 1
        else:
            unchanged += 1

        pct_moves.append(chg)
        details.append(q)

    breadth = {
        "advancers": adv,
        "decliners": dec,
        "unchanged": unchanged,
    }

    avg_move = statistics.mean(pct_moves) if pct_moves else 0

    sentiment = (
        "Bullish" if avg_move > 0.5 else
        "Bearish" if avg_move < -0.5 else
        "Neutral"
    )

    return {
        "breadth": breadth,
        "avgMovePct": round(avg_move, 2),
        "sentiment": sentiment,
        "timestamp": int(time.time()),
        "symbols": details,
    }


# ============================================================
# HOTLIST (Bullish Movers with AI Confirmation)
# ============================================================

def build_hotlist(symbols: List[str], limit: int = 8) -> List[Dict[str, Any]]:
    """
    Hotlist = strong upside movers with BullBrain confirmation.
    """

    hot = []

    for sym in symbols:
        quote = fetch_quote(sym)
        if not quote or not quote.get("changePct"):
            continue

        if quote["changePct"] < 1.5:
            continue

        candles = fetch_daily_candles(sym)
        if not candles:
            continue

        features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
        if features_vec is None:
            continue

        out = bullbrain_infer(features_vec)
        prob_up = float(out.get("probability_up") or 0.5)

        if prob_up < 0.6:
            continue

        pattern = detect_smart_pattern(
            features=feature_dict,
            quote=quote,
            technical={},
        )

        hot.append(
            {
                "symbol": sym,
                "price": last_close,
                "changePct": round(quote["changePct"], 2),
                "bullProb": round(prob_up * 100, 1),
                "signal": out.get("signal"),
                "pattern": pattern,
            }
        )

    hot.sort(key=lambda x: x["bullProb"], reverse=True)
    return hot[:limit]


# ============================================================
# BEARWATCH (Downside Risk Radar)
# ============================================================

def build_bearwatch(symbols: List[str], limit: int = 8) -> List[Dict[str, Any]]:
    """
    Bearwatch = downside movers with elevated risk signals.
    """

    bears = []

    for sym in symbols:
        quote = fetch_quote(sym)
        if not quote or not quote.get("changePct"):
            continue

        if quote["changePct"] > -1.0:
            continue

        candles = fetch_daily_candles(sym)
        if not candles:
            continue

        features_vec, feature_dict, last_close = compute_bullbrain_features(candles)
        if features_vec is None:
            continue

        out = bullbrain_infer(features_vec)
        prob_up = float(out.get("probability_up") or 0.5)

        if prob_up > 0.45:
            continue

        pattern = detect_smart_pattern(
            features=feature_dict,
            quote=quote,
            technical={},
        )

        bears.append(
            {
                "symbol": sym,
                "price": last_close,
                "changePct": round(quote["changePct"], 2),
                "bullProb": round(prob_up * 100, 1),
                "signal": out.get("signal"),
                "pattern": pattern,
            }
        )

    bears.sort(key=lambda x: x["bullProb"])
    return bears[:limit]


# ============================================================
# MARKET PULSE (Combined Payload)
# ============================================================

def build_market_pulse(symbols: List[str]) -> Dict[str, Any]:
    """
    Full Market Pulse payload used by:
    - Market tab
    - market_cron.py (Firestore precompute)
    """

    overview = build_market_overview(symbols)
    hotlist = build_hotlist(symbols)
    bearwatch = build_bearwatch(symbols)

    news = fetch_market_news(query="stock market", limit=10)

    return {
        "overview": overview,
        "hotlist": hotlist,
        "bearwatch": bearwatch,
        "news": news,
        "updatedAt": int(time.time()),
    }
