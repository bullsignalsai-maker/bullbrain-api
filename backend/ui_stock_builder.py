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
# Sparkline builders
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


def build_sparkline_from_prices(prices: List[float]) -> Dict[str, Any]:
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
# Confidence / Strength / Risk
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


def build_risk_meter(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    quality = stockdetail.get("decision_quality") or {}
    regime = quality.get("regime")
    liquidity = quality.get("liquidity")

    if not regime and not liquidity:
        return {}

    if regime == "HIGH_VOL" or liquidity in ("THIN", "POOR"):
        level = "High"
    elif regime == "NORMAL":
        level = "Medium"
    else:
        level = "Low"

    return {
        "level": level,
        "regime": regime,
        "liquidity": liquidity,
    }


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
    history = stockdetail.get("patternHistory") or {}

    risks, opportunities = [], []

    if features.get("price_vs_sma20_pct", 0) < -3:
        risks.append("Price remains below short-term averages, making rallies fragile.")

    if features.get("volume_zscore_20", 0) < -1:
        risks.append("Below-normal volume reduces signal reliability.")

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
# Trade Idea
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


# =================================================
# Master UI Enhancer
# =================================================
def build_ui_enhancements(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    ui: Dict[str, Any] = {}

    # 🔒 UI contract version
    ui["ui_version"] = "v2"

    # ---- Sparkline ----
    existing = stockdetail.get("sparkline")
    if isinstance(existing, dict) and existing.get("path"):
        ui["sparkline"] = existing
    elif isinstance(existing, list) and len(existing) >= 2:
        ui["sparkline"] = build_sparkline_from_prices(existing)
    else:
        candles_block = stockdetail.get("candles") or {}
        candles = candles_block.get("candles", []) if isinstance(candles_block, dict) else candles_block
        if isinstance(candles, list) and len(candles) >= 2:
            ui["sparkline"] = build_sparkline(candles)

    # ---- Core helpers ----
    ui["badges"] = build_ui_badges(stockdetail)

    if stockdetail.get("bullbrain"):
        ui["confidence"] = build_confidence_tier(stockdetail["bullbrain"].get("confidence"))
        ui["signalStrength"] = build_signal_strength(stockdetail)
        ui["hybridScore"] = round(stockdetail["bullbrain"].get("confidence", 0), 1)

    ui["risk"] = build_risk_meter(stockdetail)
    ui["freshness"] = build_freshness(stockdetail)

    insights = stockdetail.get("insights") or {}
    summary = insights.get("combinedTechnicalSummary") or insights.get("summaryLine") or insights.get("oneLiner")
    if summary:
        ui["executiveSummaryShort"] = condense_executive_summary(summary)

    if insights.get("whySignal"):
        ui["signalOneLiner"] = insights["whySignal"]

    rop = build_risks_opportunities(stockdetail)
    if rop:
        ui["risksOpportunities"] = rop

    trade = build_trade_idea(stockdetail)
    if trade:
        ui["tradeIdea"] = trade

    # ---- Explanations ----
    narratives = stockdetail.get("narratives") or {}
    sections = narratives.get("sections") or {}

    if sections:
        ui["explanations"] = {
            "version": "tech_explain_v2",
            "groups": {
                "technical_outlook": {
                    "short": narratives.get("summary"),
                    "medium": narratives.get("tradeIdea"),
                    "long": (
                        sections.get("trend", []) +
                        sections.get("momentum", []) +
                        sections.get("volatility", []) +
                        sections.get("volume", [])
                    ),
                },
                "risks_opportunities": {
                    "risks": sections.get("risk", []) or [],
                    "opportunities": sections.get("opportunity", []) or [],
                },
                "final_recommendation": {
                    "signal": (stockdetail.get("decision") or {}).get("finalSignal"),
                    "confidence": (stockdetail.get("bullbrain") or {}).get("confidence"),
                    "text": narratives.get("summary"),
                },
            },
        }

    # ---- Decision Intelligence ----
    probs = stockdetail.get("probabilities") or {}
    indicator_states = stockdetail.get("indicator_states") or {}

    if isinstance(probs.get("up"), (int, float)) and isinstance(probs.get("down"), (int, float)):
        diff = abs(probs["up"] - probs["down"])
        ui["hybridProbUp"] = round(probs["up"], 4)
        ui["decision"] = {
            "probability": {
                "up": round(probs["up"], 4),
                "down": round(probs["down"], 4),
            },
            "bias": {
                "label": (
                    "Neutral"
                    if diff < 0.05
                    else "Bullish"
                    if probs["up"] > probs["down"]
                    else "Bearish"
                ),
                "strength": round(diff * 100, 1),
                "state": indicator_states.get("probability_composite"),
            },
        }

    return ui
