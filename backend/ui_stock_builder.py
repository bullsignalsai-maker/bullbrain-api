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

from backend.technical_explanations import (
    build_technical_explanations
)

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

    # 1) Final decision signal (preferred) → fallback to bullbrain.signal
    decision = stockdetail.get("decision") or {}
    bullbrain = stockdetail.get("bullbrain") or {}
    final_signal = decision.get("finalSignal") or bullbrain.get("signal")

    if final_signal:
        badges.append({"type": "signal", "label": str(final_signal).replace("_", " ")})

    # 2) Trend label
    technical = stockdetail.get("technical") or {}
    trend_label = (technical.get("trend") or {}).get("label")
    if trend_label:
        badges.append({"type": "trend", "label": str(trend_label)})

    # 3) Pattern label (new schema)
    pattern = stockdetail.get("pattern") or {}
    pat_name = pattern.get("pattern") or pattern.get("patternLabel")
    if pat_name:
        badges.append({"type": "pattern", "label": str(pat_name)})

    return badges


# -------------------------------------------------
# UI sentiment summary (from existing insights)
# -------------------------------------------------
def build_ui_sentiment(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    decision = stockdetail.get("decision") or {}
    insights = stockdetail.get("insights") or {}

    headline = (
        insights.get("oneLiner")
        or insights.get("summaryLine")
    )
    if not headline:
        return {}

    final_signal = decision.get("finalSignal", "HOLD")

    tone = (
        "bullish" if final_signal == "BUY"
        else "bearish" if final_signal == "SELL"
        else "neutral"
    )

    return {
        "headline": headline,
        "tone": tone,
        "signal": final_signal
        "why": insights.get("whySignal")
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

    return {"value": round(confidence, 2), "tier": tier}


# -------------------------------------------------
# Signal strength label
# -------------------------------------------------
def build_signal_strength(stockdetail: Dict[str, Any]) -> Dict[str, str]:
    bullbrain = stockdetail.get("bullbrain") or {}
    decision = stockdetail.get("decision") or {}

    signal = decision.get("finalSignal", bullbrain.get("signal", "HOLD"))
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
    decision = stockdetail.get("decision") or {}
    bullbrain = stockdetail.get("bullbrain") or {}
    technical = stockdetail.get("technical") or {}

    signal = decision.get("finalSignal") or bullbrain.get("signal")
    direction = (technical.get("trend") or {}).get("direction")

    if not signal or not direction:
        return {}

    aligned = (
        (signal == "BUY" and direction == "up") or
        (signal == "SELL" and direction == "down")
    )

    return {
        "aligned": aligned,
        "label": "Aligned" if aligned else "Diverging"
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

    return {"minutesAgo": minutes, "label": label}


# -------------------------------------------------
# Smart Pattern Insight (UI projection)
# - Uses the REAL fields you pasted:
#   stockdetail.smartPattern
#   stockdetail.patternStats.currentPattern
#   stockdetail.patternStats.historyForCurrent
# -------------------------------------------------
def build_ui_pattern_insight(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    pattern = stockdetail.get("pattern") or {}
    history = stockdetail.get("patternHistory") or {}

    if not pattern:
        return {}

    fwd = history.get("forwardReturns") or {}
    d5 = fwd.get("days5") or {}
    d10 = fwd.get("days10") or {}

    win_rate = (
        pattern.get("winRate")
        or history.get("winRate")
    )

    confidence_pct = (
        int(round(win_rate * 100))
        if isinstance(win_rate, (int, float))
        else None
    )

    samples = history.get("samples") or []
    recent = samples[:5]

    return {
        "pattern": pattern.get("pattern"),
        "bias": pattern.get("bias"),
        "confidencePct": confidence_pct,
        "historicalEdge": {
            "days5": d5,
            "days10": d10,
        },
        "recentOccurrences": [
            {
                "date": s.get("date"),
                "headline": s.get("headline"),
                "bias": s.get("bias"),
                "changePct": s.get("changePct"),
                "fwd5d": s.get("fwd_5d"),
                "fwd10d": s.get("fwd_10d"),
            }
            for s in recent
        ] if recent else [],
        "note": "Pattern statistics are based on historical occurrences, not predictions.",
    }


# -------------------------------------------------
# Probability Cone (expected range UI)
# - Uses patternStats.historyForCurrent.forwardReturns (best/avg/worst)
# - Anchors from technical.lastClose
# -------------------------------------------------
def build_probability_cone(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    history = stockdetail.get("patternHistory") or {}

    technical = stockdetail.get("technical") or {}
    quote = stockdetail.get("quote") or {}

    last_price = (
        technical.get("lastClose")
        or quote.get("price")
        or quote.get("current")
    )

    if not history or last_price is None:
        return {}

    fr = history.get("forwardReturns") or {}
    d5 = fr.get("days5") or {}
    d10 = fr.get("days10") or {}

    if d5.get("avg") is None and d10.get("avg") is None:
        return {}

    def price_from_return_pct(ret_pct):
        try:
            return round(float(last_price) * (1 + float(ret_pct) / 100.0), 2)
        except Exception:
            return None

    return {
        "type": "expected-range",
        "anchorPrice": round(last_price, 2),
        "pattern": history.get("pattern"),
        "occurrences": history.get("occurrences"),
        "ranges": {
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
        },
        "note": "Historical price range based on past occurrences. Not a prediction.",
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
        ui["signalStrength"] = build_signal_strength(stockdetail)

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
        
    # -------------------------------------------------
    # Deep Technical Explanations (OPTION A)
    # -------------------------------------------------
    features_meta = stockdetail.get("features_meta") or {}
    technical = stockdetail.get("technical") or {}

    if features_meta and technical:
        ui["explanations"] = build_technical_explanations(stockdetail)

    
    return ui
