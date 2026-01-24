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

from backend.technical_explanations import build_technical_explanations


# =================================================
# Executive Summary Condenser (2–3 lines)
# =================================================
def condense_executive_summary(text: str, max_lines: int = 3) -> str:
    if not text:
        return ""

    sentences = [
        s.strip()
        for s in text.replace("\n", " ").split(". ")
        if len(s.strip()) > 30
    ]

    if len(sentences) <= max_lines:
        return ". ".join(sentences).rstrip(".") + "."

    priority = (
        "trend", "momentum", "volume", "pattern",
        "confidence", "wait", "risk", "pressure"
    )

    scored = []
    for s in sentences:
        score = sum(1 for k in priority if k in s.lower())
        scored.append((score, s))

    top = sorted(scored, key=lambda x: -x[0])[:max_lines]
    selected = [s for _, s in sorted(top, key=lambda x: sentences.index(x[1]))]

    return ". ".join(selected).rstrip(".") + "."


# =================================================
# Sparkline (fallback-only)
# =================================================
def build_sparkline(candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candles:
        return {}

    closes = [c.get("close") for c in candles if c.get("close") is not None]
    if len(closes) < 2:
        return {}

    min_p, max_p = min(closes), max(closes)
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


# =================================================
# Badges
# =================================================
def build_ui_badges(stockdetail: Dict[str, Any]) -> List[Dict[str, str]]:
    badges: List[Dict[str, str]] = []

    decision = stockdetail.get("decision") or {}
    bullbrain = stockdetail.get("bullbrain") or {}
    technical = stockdetail.get("technical") or {}
    pattern = stockdetail.get("pattern") or {}

    final_signal = decision.get("finalSignal") or bullbrain.get("signal")
    if final_signal:
        badges.append({"type": "signal", "label": final_signal.replace("_", " ")})

    trend_label = (technical.get("trend") or {}).get("label")
    if trend_label:
        badges.append({"type": "trend", "label": trend_label})

    pat = pattern.get("pattern") or pattern.get("patternLabel")
    if pat:
        badges.append({"type": "pattern", "label": pat})

    return badges


# =================================================
# Signal One-Liner (WHY signal)
# =================================================
def build_signal_one_liner(stockdetail: Dict[str, Any]) -> str:
    insights = stockdetail.get("insights") or {}
    return insights.get("whySignal") or ""


# =================================================
# Pattern One-Liner (WHY pattern)
# =================================================
def build_pattern_one_liner(stockdetail: Dict[str, Any]) -> str:
    pattern = stockdetail.get("pattern") or {}
    headline = pattern.get("headline")

    if headline and len(headline) > 20:
        return headline.rstrip(".")

    insights = stockdetail.get("insights") or {}
    trend = insights.get("trendSummary")
    momentum = insights.get("momentumSummary")

    if trend and momentum:
        return f"{trend} {momentum}".strip()

    return ""


# =================================================
# Sentiment Summary
# =================================================
def build_ui_sentiment(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    decision = stockdetail.get("decision") or {}
    insights = stockdetail.get("insights") or {}

    headline = insights.get("oneLiner") or insights.get("summaryLine")
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
        "signal": final_signal,
        "why": insights.get("whySignal"),
    }


# =================================================
# Confidence Tier
# =================================================
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


# =================================================
# Signal Strength
# =================================================
def build_signal_strength(stockdetail: Dict[str, Any]) -> Dict[str, str]:
    bullbrain = stockdetail.get("bullbrain") or {}
    decision = stockdetail.get("decision") or {}

    signal = decision.get("finalSignal") or bullbrain.get("signal", "HOLD")
    confidence = bullbrain.get("confidence", 0)

    if signal == "BUY":
        label = "Strong Buy" if confidence >= 70 else "Buy"
    elif signal == "SELL":
        label = "Strong Sell" if confidence >= 70 else "Sell"
    else:
        label = "Neutral"

    return {"signal": signal, "label": label}


# =================================================
# Risk Meter
# =================================================
def build_risk_meter(technical: Dict[str, Any]) -> Dict[str, str]:
    if not technical:
        return {}

    vol = technical.get("volatility") or {}
    volatility_value = vol if isinstance(vol, (int, float)) else vol.get("volatility_20d")
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


# =================================================
# Trend Alignment
# =================================================
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

    return {"aligned": aligned, "label": "Aligned" if aligned else "Diverging"}


# =================================================
# Freshness
# =================================================
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


# =================================================
# Risks & Opportunities
# =================================================
def build_risks_opportunities(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    features = stockdetail.get("features_meta") or {}
    insights = stockdetail.get("insights") or {}
    decision = stockdetail.get("decision") or {}
    history = stockdetail.get("patternHistory") or {}

    risks, opportunities = [], []

    if features.get("price_vs_sma20_pct", 0) < -3:
        risks.append("Price remains below short-term averages, making rallies fragile.")

    if features.get("volume_zscore_20", 0) < -1:
        risks.append("Below-normal volume reduces signal reliability.")

    if decision.get("quality", {}).get("liquidity") == "POOR":
        risks.append("Poor liquidity can increase volatility and slippage.")

    if insights.get("momentumSummary"):
        risks.append(insights["momentumSummary"])

    rsi = features.get("rsi14")
    if isinstance(rsi, (int, float)) and rsi < 30:
        opportunities.append("Deep oversold momentum could support a short-term bounce.")

    days5 = (history.get("forwardReturns") or {}).get("days5") or {}
    if isinstance(days5.get("best"), (int, float)) and days5["best"] > 3:
        opportunities.append("Historical pattern samples show upside spikes in some cases.")

    if insights.get("volatilitySummary"):
        opportunities.append(insights["volatilitySummary"])

    if not risks and not opportunities:
        return {}

    return {
        "short": insights.get("summaryLine"),
        "medium": insights.get("combinedTechnicalSummary"),
        "risks": risks[:4],
        "opportunities": opportunities[:4],
    }


# =================================================
# Trade Idea (Educational, Non-Prescriptive)
# =================================================
def build_trade_idea(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    decision = stockdetail.get("decision") or {}
    bullbrain = stockdetail.get("bullbrain") or {}
    insights = stockdetail.get("insights") or {}
    features = stockdetail.get("features_meta") or {}

    stance = decision.get("finalSignal") or bullbrain.get("signal") or "HOLD"
    summary = insights.get("whySignal") or "No clean directional edge is present."

    atr = features.get("atr14")
    note = (
        f"Typical daily movement is around ±{round(atr, 2)} points."
        if isinstance(atr, (int, float))
        else None
    )

    return {"stance": stance.title(), "summary": summary, "note": note}

def build_sparkline_from_prices(prices: list[float]) -> dict[str, Any]:
    if not prices or len(prices) < 2:
        return {}

    min_p, max_p = min(prices), max(prices)
    span = max_p - min_p or 1

    points = []
    for i, price in enumerate(prices):
        x = round(i * 100 / (len(prices) - 1), 1)
        y = round((max_p - price) * 30 / span, 1)
        points.append(f"{x},{y}")

    return {
        "path": "M " + " L ".join(points),
        "min": round(min_p, 2),
        "max": round(max_p, 2),
        "direction": "up" if prices[-1] >= prices[0] else "down",
    }

# =================================================
# Master UI Enhancer
# =================================================
def build_ui_enhancements(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    ui: Dict[str, Any] = {}

    # -------------------------------------------------
    # Sparkline (fallback-only, schema-safe)
    # -------------------------------------------------
    existing = stockdetail.get("sparkline")

    # Case 1: Already render-ready (rare)
    if isinstance(existing, dict) and existing.get("path"):
        ui["sparkline"] = existing

    # Case 2: Firestore raw sparkline array (THIS IS YOUR CASE)
    elif isinstance(existing, list) and len(existing) >= 2:
        ui["sparkline"] = build_sparkline_from_prices(existing)

    # Case 3: Candle-based fallback (future-safe)
    else:
        candles_block = stockdetail.get("candles") or []

        if isinstance(candles_block, dict):
            candles = candles_block.get("candles", [])
        elif isinstance(candles_block, list):
            candles = candles_block
        else:
            candles = []

        if candles and len(candles) >= 2:
            ui["sparkline"] = build_sparkline(candles)

    ui["badges"] = build_ui_badges(stockdetail)

    sentiment = build_ui_sentiment(stockdetail)
    if sentiment:
        ui["sentiment"] = sentiment

    bullbrain = stockdetail.get("bullbrain") or {}
    if bullbrain:
        ui["confidence"] = build_confidence_tier(bullbrain.get("confidence"))
        ui["signalStrength"] = build_signal_strength(stockdetail)

    technical = stockdetail.get("technical") or {}
    if technical:
        ui["risk"] = build_risk_meter(technical)
        align = build_trend_alignment(stockdetail)
        if align:
            ui["trendAlignment"] = align

    ui["freshness"] = build_freshness(stockdetail)

    insights = stockdetail.get("insights") or {}
    full_summary = (
        insights.get("combinedTechnicalSummary")
        or insights.get("summaryLine")
        or insights.get("oneLiner")
    )
    if full_summary:
        ui["executiveSummaryShort"] = condense_executive_summary(full_summary)

    signal_line = build_signal_one_liner(stockdetail)
    if signal_line:
        ui["signalOneLiner"] = signal_line

    pattern_why = build_pattern_one_liner(stockdetail)
    if pattern_why:
        ui["patternWhy"] = pattern_why

    rop = build_risks_opportunities(stockdetail)
    if rop:
        ui["risksOpportunities"] = rop

    trade = build_trade_idea(stockdetail)
    if trade:
        ui["tradeIdea"] = trade

    if stockdetail.get("features_meta") and stockdetail.get("technical"):
        ui["explanations"] = build_technical_explanations(stockdetail)

    # -------------------------------------------------
    # Decision Intelligence (EXPLICIT, UI-SAFE)
    # -------------------------------------------------
    decision = stockdetail.get("decision") or {}
    bullbrain = stockdetail.get("bullbrain") or {}
    raw = bullbrain.get("raw") or {}

    decision_ui = {}

    # Probabilities
    if isinstance(raw.get("prob_up"), (int, float)) and isinstance(raw.get("prob_down"), (int, float)):
        decision_ui["probability"] = {
            "up": round(raw["prob_up"] * 100, 1),
            "down": round(raw["prob_down"] * 100, 1),
        }

    # Bias label (UI-ready)
    if "probability" in decision_ui:
        up = decision_ui["probability"]["up"]
        down = decision_ui["probability"]["down"]

        if abs(up - down) < 5:
            bias = "Neutral"
        elif up > down:
            bias = "Bullish"
        else:
            bias = "Bearish"

        decision_ui["bias"] = {
            "label": bias,
            "strength": abs(up - down)  # percentage gap
        }

    # Decision reasons
    reasons = decision.get("decisionReasons")
    if isinstance(reasons, list) and reasons:
        decision_ui["reasons"] = reasons

    # Market regime / quality
    regime = (decision.get("quality") or {}).get("regime")
    if regime:
        decision_ui["regime"] = regime

    if decision_ui:
        ui["decision"] = decision_ui
    

    return ui
