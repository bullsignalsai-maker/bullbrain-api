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
# UI badges
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

    smart = stockdetail.get("smartPattern")
    if smart and smart.get("pattern"):
        badges.append({"type": "pattern", "label": smart.get("pattern")})

    return badges


# -------------------------------------------------
# Sentiment summary
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
# Confidence tier
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
# Signal strength
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
# Risk meter
# -------------------------------------------------
def build_risk_meter(technical: Dict[str, Any]) -> Dict[str, str]:
    if not technical:
        return {}

    volatility = technical.get("volatility") or 0
    atr = technical.get("atr") or 0

    if volatility > 0.05 or atr > 5:
        level = "High"
    elif volatility > 0.025:
        level = "Medium"
    else:
        level = "Low"

    return {"level": level}


# -------------------------------------------------
# Trend alignment
# -------------------------------------------------
def build_trend_alignment(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    bullbrain = stockdetail.get("bullbrain") or {}
    technical = stockdetail.get("technical") or {}

    signal = bullbrain.get("signal")
    direction = technical.get("trend", {}).get("direction")

    if not signal or not direction:
        return {}

    aligned = (
        (signal == "BUY" and direction == "up") or
        (signal == "SELL" and direction == "down")
    )

    return {"aligned": aligned, "label": "Aligned" if aligned else "Diverging"}


# -------------------------------------------------
# Freshness
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

    label = "Just now" if minutes < 5 else "Recent" if minutes < 30 else "Stale"
    return {"minutesAgo": minutes, "label": label}


# -------------------------------------------------
# Smart Pattern Insight (UI)
# -------------------------------------------------
def build_ui_pattern_insight(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    sp = stockdetail.get("smartPattern")
    stats = stockdetail.get("patternStats", {}).get("historyForCurrent")
    current = stockdetail.get("currentPattern")

    if not sp:
        return {}

    win_rate = sp.get("winRate", 0)
    confidence_pct = round(win_rate * 100)

    avg10 = None
    if stats:
        avg10 = stats.get("forwardReturns", {}).get("days10", {}).get("avg")

    if avg10 is not None:
        edge_label = "Historically Strong" if avg10 >= 3 else "Moderate Edge" if avg10 >= 1 else "Weak / Neutral"
    else:
        edge_label = "Unknown"

    return {
        "pattern": sp.get("pattern"),
        "confidencePct": confidence_pct,
        "label": edge_label,
        "explanation": sp.get("explanation"),
        "history": {
            "occurrences": stats.get("occurrences") if stats else None,
            "avgReturn5d": stats.get("forwardReturns", {}).get("days5", {}).get("avg") if stats else None,
            "avgReturn10d": avg10,
        },
        "recent": {
            "date": current.get("date") if current else None,
            "headline": current.get("headline") if current else None,
            "bias": current.get("bias") if current else None,
        }
    }


# -------------------------------------------------
# Probability Cone (EXPECTED RANGE UI)
# -------------------------------------------------
def build_probability_cone(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    stats = stockdetail.get("patternStats", {}).get("historyForCurrent")
    last_price = stockdetail.get("technical", {}).get("lastClose")

    if not stats or not last_price:
        return {}

    fr = stats.get("forwardReturns", {})
    d5 = fr.get("days5", {})
    d10 = fr.get("days10", {})

    def band(ret):
        if ret is None:
            return None
        return round(last_price * (1 + ret / 100), 2)

    cone = {
        "days5": {
            "low": band(d5.get("worst")),
            "mid": band(d5.get("avg")),
            "high": band(d5.get("best")),
        },
        "days10": {
            "low": band(d10.get("worst")),
            "mid": band(d10.get("avg")),
            "high": band(d10.get("best")),
        }
    }

    return {
        "type": "expected-range",
        "anchorPrice": last_price,
        "ranges": cone,
        "note": "Historical probability range, not a prediction",
    }


# -------------------------------------------------
# MASTER UI ENHANCER
# -------------------------------------------------
def build_ui_enhancements(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    ui: Dict[str, Any] = {}

    # Sparkline (fallback-only)
    if "sparkline" not in stockdetail:
        candles = stockdetail.get("candles", {}).get("candles", [])
        if candles:
            ui["sparkline"] = build_sparkline(candles)

    ui["badges"] = build_ui_badges(stockdetail)

    sentiment = build_ui_sentiment(stockdetail)
    if sentiment:
        ui["sentiment"] = sentiment

    bullbrain = stockdetail.get("bullbrain") or {}
    if bullbrain:
        ui["confidence"] = build_confidence_tier(bullbrain.get("confidence"))
        ui["signalStrength"] = build_signal_strength(bullbrain)

    technical = stockdetail.get("technical") or {}
    if technical:
        ui["risk"] = build_risk_meter(technical)
        ui["trendAlignment"] = build_trend_alignment(stockdetail)

    ui["freshness"] = build_freshness(stockdetail)

    pattern_ui = build_ui_pattern_insight(stockdetail)
    if pattern_ui:
        ui["patternInsight"] = pattern_ui

    cone = build_probability_cone(stockdetail)
    if cone:
        ui["probabilityCone"] = cone

    return ui
