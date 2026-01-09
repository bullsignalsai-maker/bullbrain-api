# backend/ui_stock_builder.py
# -------------------------------------------------
# UI Enhancement Layer (READ-ONLY, ADDITIVE ONLY)
#
# Purpose:
# - Convert Firestore stockdetail JSON into UI-friendly helpers
# - NO background jobs
# - NO Firestore writes
# - NO schema changes
# -------------------------------------------------

from typing import Dict, Any, List
from datetime import datetime, timezone


# -------------------------------------------------
# Sparkline (FALLBACK-ONLY)
# -------------------------------------------------
def build_sparkline(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candles:
        return {}

    closes = [c.get("close") for c in candles if c.get("close") is not None]
    if len(closes) < 2:
        return {}

    min_p = min(closes)
    max_p = max(closes)
    span = max_p - min_p or 1

    points = []
    for i, price in enumerate(closes):
        x = round(i * 100 / (len(closes) - 1), 1)
        y = round((max_p - price) * 30 / span, 1)
        points.append(f"{x},{y}")

    return {
        "path": "M " + " L ".join(points),
        "min": round(min_p, 2),
        "max": round(max_p, 2),
        "direction": "up" if closes[-1] >= closes[0] else "down",
    }


# -------------------------------------------------
# UI badges derived from Firestore data
# -------------------------------------------------
def build_ui_badges(stockdetail: Dict[str, Any]) -> List[Dict[str, str]]:
    badges: List[Dict[str, str]] = []

    bullbrain = stockdetail.get("bullbrain") or {}
    signal = bullbrain.get("signal")
    if signal:
        badges.append({"type": "signal", "label": signal.replace("_", " ")})

    technical = stockdetail.get("technical") or {}
    trend_label = technical.get("trend", {}).get("label")
    if trend_label:
        badges.append({"type": "trend", "label": trend_label})

    sp = stockdetail.get("smartPattern") or {}
    if sp.get("pattern"):
        badges.append({"type": "pattern", "label": sp.get("pattern")})

    # Also badge the "currentPattern" if available (from patternStats)
    current = (stockdetail.get("patternStats") or {}).get("currentPattern") or {}
    if current.get("pattern"):
        badges.append({"type": "patternDetected", "label": current.get("pattern")})

    return badges


# -------------------------------------------------
# UI sentiment summary (from existing insights)
# -------------------------------------------------
def build_ui_sentiment(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    insights = stockdetail.get("insights") or {}
    summary = insights.get("summaryLine")
    if not summary:
        return {}

    s = summary.lower()
    tone = "bearish" if "bear" in s else "bullish" if "bull" in s else "neutral"

    return {"headline": summary, "tone": tone}


# -------------------------------------------------
# Confidence tier (human-readable)
# -------------------------------------------------
def build_confidence_tier(confidence: float | None) -> Dict[str, Any]:
    if confidence is None:
        return {}

    if confidence >= 75:
        tier = "Very High"
    elif confidence >= 65:
        tier = "High"
    elif confidence >= 55:
        tier = "Moderate"
    else:
        tier = "Low"

    return {"value": round(confidence, 2), "tier": tier}


# -------------------------------------------------
# Signal strength label
# -------------------------------------------------
def build_signal_strength(bullbrain: Dict[str, Any]) -> Dict[str, str]:
    if not bullbrain:
        return {}

    signal = bullbrain.get("signal")
    confidence = bullbrain.get("confidence", 0)

    if signal == "BUY":
        label = "Strong Buy" if confidence >= 70 else "Buy"
    elif signal == "SELL":
        label = "Strong Sell" if confidence >= 70 else "Sell"
    else:
        label = "Neutral"

    return {"signal": signal, "label": label}


# -------------------------------------------------
# Risk meter (derived from technicals)
# -------------------------------------------------
def build_risk_meter(technical: Dict[str, Any]) -> Dict[str, str]:
    if not technical:
        return {}

    vol = technical.get("volatility") or {}
    volatility_value = None

    # Handle both numeric and object forms safely
    if isinstance(vol, (int, float)):
        volatility_value = vol
    elif isinstance(vol, dict):
        volatility_value = vol.get("volatility_20d")

    atr = technical.get("atr") or technical.get("atr14") or 0

    if volatility_value is None:
        return {}

    if volatility_value > 3 or atr > 20:
        level = "High"
    elif volatility_value > 1.5:
        level = "Medium"
    else:
        level = "Low"

    return {"level": level}


# -------------------------------------------------
# Trend alignment (BullBrain vs Technicals)
# -------------------------------------------------
def build_trend_alignment(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    bullbrain = stockdetail.get("bullbrain") or {}
    technical = stockdetail.get("technical") or {}

    signal = bullbrain.get("signal")
    # your technical.trend doesn't include "direction" in pasted JSON,
    # so alignment may be unavailable. Keep safe.
    direction = (technical.get("trend") or {}).get("direction")

    if not signal or not direction:
        return {}

    aligned = (
        (signal == "BUY" and direction == "up") or
        (signal == "SELL" and direction == "down")
    )

    return {"aligned": aligned, "label": "Aligned" if aligned else "Diverging"}


# -------------------------------------------------
# Freshness indicator
# -------------------------------------------------
def build_freshness(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    ts = stockdetail.get("computed_at")
    if not ts:
        return {}

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        minutes = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
    except Exception:
        return {}

    if minutes < 5:
        label = "Just now"
    elif minutes < 30:
        label = "Recent"
    else:
        label = "Stale"

    return {"minutesAgo": minutes, "label": label}


# -------------------------------------------------
# Smart Pattern Insight (UI projection)
# - Uses the REAL fields you pasted:
#   stockdetail.smartPattern
#   stockdetail.patternStats.currentPattern
#   stockdetail.patternStats.historyForCurrent
# -------------------------------------------------
def build_ui_pattern_insight(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    pattern_stats = stockdetail.get("patternStats") or {}
    sp = stockdetail.get("smartPattern") or {}  # simple pattern object
    current = pattern_stats.get("currentPattern") or {}
    history = pattern_stats.get("historyForCurrent") or {}

    # If nothing exists, return empty
    if not sp and not current and not history:
        return {}

    # Prefer "currentPattern" if present (richer context)
    chosen_pattern = current.get("pattern") or sp.get("pattern")
    chosen_winrate = current.get("winRate")
    if chosen_winrate is None:
        chosen_winrate = sp.get("winRate")

    confidence_pct = None
    if chosen_winrate is not None:
        try:
            confidence_pct = int(round(float(chosen_winrate) * 100))
        except Exception:
            confidence_pct = None

    fr = history.get("forwardReturns") or {}
    d5 = fr.get("days5") or {}
    d10 = fr.get("days10") or {}

    avg10 = d10.get("avg")
    if avg10 is not None:
        edge_label = "Historically Strong" if avg10 >= 3 else "Moderate Edge" if avg10 >= 1 else "Weak / Neutral"
    else:
        edge_label = "Unknown"

    # “What happened?” UI: show last few occurrences (samples)
    samples = history.get("samples") or []
    recent_samples = []
    for s in samples[:5]:
        recent_samples.append({
            "date": s.get("date"),
            "headline": s.get("headline"),
            "bias": s.get("bias"),
            "changePct": s.get("changePct"),
            "fwd5d": s.get("fwd_5d"),
            "fwd10d": s.get("fwd_10d"),
        })

    return {
        "pattern": chosen_pattern,
        "confidencePct": confidence_pct,
        "label": edge_label,

        # From smartPattern object (simple human explanation)
        "explanation": sp.get("explanation"),

        # From currentPattern (current detection)
        "current": {
            "date": current.get("date"),
            "headline": current.get("headline"),
            "bias": current.get("bias"),
            "changePct": current.get("changePct"),
        } if current else {},

        # From historyForCurrent (stats + occurrences)
        "history": {
            "occurrences": history.get("occurrences"),
            "forwardReturns": {
                "days5": {
                    "avg": d5.get("avg"),
                    "median": d5.get("median"),
                    "best": d5.get("best"),
                    "worst": d5.get("worst"),
                    "count": d5.get("count"),
                } if d5 else None,
                "days10": {
                    "avg": d10.get("avg"),
                    "median": d10.get("median"),
                    "best": d10.get("best"),
                    "worst": d10.get("worst"),
                    "count": d10.get("count"),
                } if d10 else None,
            } if fr else None,
            "recentSamples": recent_samples if recent_samples else None,
        } if history else None,
    }


# -------------------------------------------------
# Probability Cone (expected range UI)
# - Uses patternStats.historyForCurrent.forwardReturns (best/avg/worst)
# - Anchors from technical.lastClose
# -------------------------------------------------
def build_probability_cone(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    pattern_stats = stockdetail.get("patternStats") or {}

    stats = (
        pattern_stats.get("historyForCurrent")
        or pattern_stats.get("history")
        or pattern_stats
        or {}
    )

    technical = stockdetail.get("technical") or {}
    quote = stockdetail.get("quote") or {}

    last_price = (
        technical.get("lastClose")
        or quote.get("price")
        or quote.get("current")
    )

    if not stats or last_price is None:
        return {}

    fr = (
        stats.get("forwardReturns")
        or stats.get("forward_returns")
        or {}
    )

    d5 = fr.get("days5") or {}
    d10 = fr.get("days10") or {}

    if d5.get("avg") is None and d10.get("avg") is None:
        return {}

    def price_from_return_pct(ret_pct):
        if ret_pct is None:
            return None
        try:
            return round(float(last_price) * (1 + float(ret_pct) / 100.0), 2)
        except Exception:
            return None

    ranges = {
        "days5": {
            "low": price_from_return_pct(d5.get("worst")),
            "mid": price_from_return_pct(d5.get("avg")),
            "high": price_from_return_pct(d5.get("best")),
        } if d5 else None,
        "days10": {
            "low": price_from_return_pct(d10.get("worst")),
            "mid": price_from_return_pct(d10.get("avg")),
            "high": price_from_return_pct(d10.get("best")),
        } if d10 else None,
    }

    return {
        "type": "expected-range",
        "anchorPrice": round(last_price, 2),
        "pattern": stats.get("pattern"),
        "occurrences": stats.get("occurrences"),
        "ranges": ranges,
        "note": "Historical price range based on past occurrences of this pattern. Not a prediction.",
    }


# -------------------------------------------------
# Master UI Enhancer (SAFE ENTRY POINT)
# -------------------------------------------------
def build_ui_enhancements(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPORTANT GUARANTEES:
    - Uses Firestore data only
    - Adds UI helpers only
    - NEVER overwrites stockdetail fields
    - Safe for real-time endpoints
    """
    ui: Dict[str, Any] = {}

    # Sparkline: fallback-only (if missing OR invalid)
    existing_spark = stockdetail.get("sparkline")
    if not (isinstance(existing_spark, dict) and existing_spark.get("path")):
        candles = (stockdetail.get("candles") or {}).get("candles", [])
        if candles:
            ui["sparkline"] = build_sparkline(candles)

    # Badges
    ui["badges"] = build_ui_badges(stockdetail)

    # Sentiment
    sentiment = build_ui_sentiment(stockdetail)
    if sentiment:
        ui["sentiment"] = sentiment

    # Confidence + signal strength
    bullbrain = stockdetail.get("bullbrain") or {}
    if bullbrain:
        ui["confidence"] = build_confidence_tier(bullbrain.get("confidence"))
        ui["signalStrength"] = build_signal_strength(bullbrain)

    # Risk + alignment
    technical = stockdetail.get("technical") or {}
    if technical:
        ui["risk"] = build_risk_meter(technical)
        align = build_trend_alignment(stockdetail)
        if align:
            ui["trendAlignment"] = align

    # Freshness
    ui["freshness"] = build_freshness(stockdetail)

    # Smart patterns UI insight
    pattern_ui = build_ui_pattern_insight(stockdetail)
    if pattern_ui:
        ui["patternInsight"] = pattern_ui

    # Probability cone
    cone = build_probability_cone(stockdetail)
    if cone:
        ui["probabilityCone"] = cone

    return ui
