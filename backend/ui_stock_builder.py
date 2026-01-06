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
# Sparkline (visual-only)
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
        badges.append({
            "type": "signal",
            "label": signal.replace("_", " "),
        })

    technical = stockdetail.get("technical") or {}
    trend_label = technical.get("trend", {}).get("label")
    if trend_label:
        badges.append({
            "type": "trend",
            "label": trend_label,
        })

    smart = stockdetail.get("smartPattern")
    if smart and smart.get("pattern"):
        badges.append({
            "type": "pattern",
            "label": smart.get("pattern"),
        })

    return badges


# -------------------------------------------------
# UI sentiment summary (from existing insights)
# -------------------------------------------------
def build_ui_sentiment(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    insights = stockdetail.get("insights") or {}
    summary = insights.get("summaryLine")
    if not summary:
        return {}

    summary_l = summary.lower()

    tone = (
        "bearish" if "bear" in summary_l
        else "bullish" if "bull" in summary_l
        else "neutral"
    )

    return {
        "headline": summary,
        "tone": tone,
    }


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

    return {
        "value": round(confidence, 2),
        "tier": tier,
    }


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

    return {
        "signal": signal,
        "label": label,
    }


# -------------------------------------------------
# Risk meter (derived from technicals)
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

    return {
        "level": level,
    }


# -------------------------------------------------
# Trend alignment (BullBrain vs Technicals)
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

    return {
        "aligned": aligned,
        "label": "Aligned" if aligned else "Diverging",
    }


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

    return {
        "minutesAgo": minutes,
        "label": label,
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

    # Sparkline
    candles = stockdetail.get("candles", {}).get("candles", [])
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
        ui["trendAlignment"] = build_trend_alignment(stockdetail)

    # Freshness
    ui["freshness"] = build_freshness(stockdetail)

    return ui
