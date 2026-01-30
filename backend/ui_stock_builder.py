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

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


# =================================================
# Small helpers
# =================================================
def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


def _first_nonempty(*vals: Any) -> Optional[str]:
    for v in vals:
        if _is_nonempty_str(v):
            return v.strip()
    return None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not _is_nonempty_str(x):
            continue
        k = x.strip()
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _prune(obj: Any) -> Any:
    """
    Remove None, empty strings, empty lists/dicts recursively.
    (Highlights should NOT be pruned; explanations can be.)
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        return s if s else None
    if isinstance(obj, list):
        cleaned = []
        for v in obj:
            pv = _prune(v)
            if pv is None:
                continue
            cleaned.append(pv)
        return cleaned if cleaned else None
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            pv = _prune(v)
            if pv is None:
                continue
            cleaned[k] = pv
        return cleaned if cleaned else None
    return obj


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
        badges.append({"type": "signal", "label": str(final_signal).replace("_", " ")})

    trend_label = (technical.get("trend") or {}).get("label")
    if trend_label:
        badges.append({"type": "trend", "label": str(trend_label)})

    pat = pattern.get("pattern") or pattern.get("patternLabel")
    if pat:
        badges.append({"type": "pattern", "label": str(pat)})

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

    return {"value": round(float(confidence), 2), "tier": tier}


def build_signal_strength(stockdetail: Dict[str, Any]) -> Dict[str, str]:
    bullbrain = stockdetail.get("bullbrain") or {}
    decision = stockdetail.get("decision") or {}

    signal = decision.get("finalSignal") or bullbrain.get("signal") or "HOLD"
    confidence = bullbrain.get("confidence", 0) or 0

    if signal == "BUY":
        label = "Strong Buy" if confidence >= 70 else "Buy"
    elif signal == "SELL":
        label = "Strong Sell" if confidence >= 70 else "Sell"
    else:
        label = "Neutral"

    return {"signal": str(signal), "label": label}


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

    return {"level": level, "regime": regime, "liquidity": liquidity}


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
# Risks & Opportunities (UI block)
# =================================================
def build_risks_opportunities(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    features = stockdetail.get("features_meta") or {}
    insights = stockdetail.get("insights") or {}
    history = stockdetail.get("patternHistory") or {}

    risks, opportunities = [], []

    if isinstance(features.get("price_vs_sma20_pct"), (int, float)) and features.get("price_vs_sma20_pct") < -3:
        risks.append("Price remains below short-term averages, making rallies fragile.")

    if isinstance(features.get("volume_zscore_20"), (int, float)) and features.get("volume_zscore_20") < -1:
        risks.append("Below-normal volume reduces signal reliability.")

    if _is_nonempty_str(insights.get("momentumSummary")):
        risks.append(insights["momentumSummary"])

    rsi = features.get("rsi14")
    if isinstance(rsi, (int, float)) and rsi < 30:
        opportunities.append("Deep oversold momentum could support a short-term bounce.")

    days5 = (history.get("forwardReturns") or {}).get("days5") or {}
    if isinstance(days5.get("best"), (int, float)) and days5["best"] > 3:
        opportunities.append("Historical pattern samples show upside spikes in some cases.")

    if _is_nonempty_str(insights.get("volatilitySummary")):
        opportunities.append(insights["volatilitySummary"])

    # ✅ guarantee at least 1 item each when possible (prevents empty arrays in UI)
    risks = _dedupe_keep_order(risks)[:4]
    opportunities = _dedupe_keep_order(opportunities)[:4]

    if not risks and not opportunities:
        return {}

    return {
        "short": _first_nonempty(insights.get("summaryLine"), insights.get("oneLiner")),
        "medium": _first_nonempty(insights.get("combinedTechnicalSummary")),
        "risks": risks,
        "opportunities": opportunities,
    }


# =================================================
# Trade Idea (UI block)
# =================================================
def build_trade_idea(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    decision = stockdetail.get("decision") or {}
    bullbrain = stockdetail.get("bullbrain") or {}
    insights = stockdetail.get("insights") or {}
    features = stockdetail.get("features_meta") or {}

    stance = decision.get("finalSignal") or bullbrain.get("signal") or "HOLD"
    summary = _first_nonempty(insights.get("whySignal")) or "No clean directional edge is present."

    atr = features.get("atr14")
    note = (
        f"Typical daily movement is around ±{round(float(atr), 2)} points."
        if isinstance(atr, (int, float))
        else None
    )

    return {"stance": str(stance).title(), "summary": summary, "note": note}


# =================================================
# NEW: Highlights (StockDetailScreen hero)
# =================================================
def build_highlights(stockdetail: Dict[str, Any], ui: Dict[str, Any]) -> Dict[str, Any]:
    bullbrain = stockdetail.get("bullbrain") or {}
    decision = stockdetail.get("decision") or {}
    narratives = stockdetail.get("narratives") or {}
    insights = stockdetail.get("insights") or {}
    technical = stockdetail.get("technical") or {}

    signal = decision.get("finalSignal") or bullbrain.get("signal") or "HOLD"
    confidence = bullbrain.get("confidence")

    conf_tier = (ui.get("confidence") or {}).get("tier") or (
        build_confidence_tier(confidence).get("tier") if isinstance(confidence, (int, float)) else "Low"
    )

    risk_level = (ui.get("risk") or {}).get("level") or "Medium"

    headline = _first_nonempty(
        narratives.get("summary"),
        (ui.get("explanations") or {}).get("groups", {}).get("technical_outlook", {}).get("short"),
        insights.get("combinedTechnicalSummary"),
        insights.get("oneLiner"),
    ) or "No strong directional edge is present right now."

    subline = _first_nonempty(
        narratives.get("tradeIdea"),
        (ui.get("explanations") or {}).get("groups", {}).get("technical_outlook", {}).get("medium"),
        insights.get("whySignal"),
    ) or "Wait for clearer confirmation before taking aggressive positions."

    # timeframe label
    trend_label = (technical.get("trend") or {}).get("label")
    momentum_label = (technical.get("momentum") or {}).get("label")
    timeframe = None
    if _is_nonempty_str(trend_label) and _is_nonempty_str(momentum_label):
        timeframe = f"{trend_label} trend · {momentum_label} momentum"
    elif _is_nonempty_str(trend_label):
        timeframe = f"{trend_label} trend"
    elif _is_nonempty_str(momentum_label):
        timeframe = f"{momentum_label} momentum"
    else:
        timeframe = "Short-term view"

    return {
        "headline": headline,
        "subline": subline,
        "signal": signal,
        "confidence": round(float(confidence), 2) if isinstance(confidence, (int, float)) else None,
        "confidenceTier": conf_tier,
        "riskLevel": risk_level,
        "timeframe": timeframe,
    }


# =================================================
# NEW: Explanations (sectioned narrative)
# =================================================
def build_explanations(stock: dict, ui: dict | None = None) -> dict:
    """
    UI-FIRST explanations builder
    Guarantees:
    - No null strings
    - No empty arrays
    - All groups expected by StockDetailScreen exist
    """

    ui = ui or {}
    narratives = stock.get("narratives") or {}
    sections = narratives.get("sections") or {}

    decision = stock.get("decision") or {}
    signal = decision.get("final") or ui.get("signalStrength", {}).get("signal") or "HOLD"
    confidence = decision.get("confidence") or ui.get("confidence", {}).get("value")

    # ---------------------------------------------------------
    # TECHNICAL OUTLOOK
    # ---------------------------------------------------------
    tech_short = (
        narratives.get("summary")
        or sections.get("trend", [None])[0]
        or "Momentum signals are mixed, limiting directional conviction."
    )

    tech_medium = (
        narratives.get("tradeIdea")
        or sections.get("momentum", [None])[0]
        or "Price action suggests equilibrium with no dominant edge."
    )

    tech_long = (
        sections.get("trend", [])
        + sections.get("momentum", [])
        + sections.get("volatility", [])
        + sections.get("volume", [])
    )

    if not tech_long:
        tech_long = [
            "Price is oscillating without a sustained directional trend.",
            "Momentum indicators are neutral, offering limited confirmation.",
            "Volatility remains elevated, increasing outcome dispersion.",
        ]

    # ---------------------------------------------------------
    # RISKS & OPPORTUNITIES
    # ---------------------------------------------------------
    risks = sections.get("risks") or []
    opportunities = sections.get("opportunities") or []

    # 🔒 HARD GUARANTEES (UI MUST NEVER SEE EMPTY)
    if not risks:
        risks = [
            "Sideways conditions increase the risk of false breakouts and whipsaws."
        ]

    if not opportunities:
        opportunities = [
            "A confirmed breakout with volume could improve reward-to-risk."
        ]

    # ---------------------------------------------------------
    # TRADE IDEA
    # ---------------------------------------------------------
    trade = ui.get("tradeIdea") or {}

    trade_stance = trade.get("stance") or signal.capitalize()
    trade_summary = (
        trade.get("summary")
        or narratives.get("tradeIdea")
        or "No clean directional edge is present at current levels."
    )

    trade_note = trade.get("note")

    # ---------------------------------------------------------
    # FINAL RECOMMENDATION
    # ---------------------------------------------------------
    final_text = (
        narratives.get("summary")
        or tech_short
    )

    # ---------------------------------------------------------
    # BUILD FINAL OBJECT (UI CONTRACT)
    # ---------------------------------------------------------
    return {
        "groups": {
            "technical_outlook": {
                "short": tech_short,
                "medium": tech_medium,
                "long": tech_long,
            },

            "risks_opportunities": {
                "risks": risks,
                "opportunities": opportunities,
            },

            "trade_idea": {
                "stance": trade_stance,
                "summary": trade_summary,
                "note": trade_note,
            },

            "final_recommendation": {
                "signal": signal,
                "confidence": confidence,
                "text": final_text,
            },
        }
    }

# =================================================
# Master UI Enhancer
# =================================================
def build_ui_enhancements(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    ui: Dict[str, Any] = {}

    # 🔒 UI contract version
    ui["ui_version"] = "v2"

    # ---- Sparkline (schema-safe) ----
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

    # ---- Core UI helpers ----
    ui["badges"] = build_ui_badges(stockdetail)

    if stockdetail.get("bullbrain"):
        ui["confidence"] = build_confidence_tier((stockdetail["bullbrain"] or {}).get("confidence"))
        ui["signalStrength"] = build_signal_strength(stockdetail)
        ui["hybridScore"] = round(float((stockdetail["bullbrain"] or {}).get("confidence", 0) or 0), 1)

    ui["freshness"] = build_freshness(stockdetail)

    # Risk uses decision_quality (regime/liquidity)
    ui["risk"] = build_risk_meter(stockdetail)

    # executive summary
    insights = stockdetail.get("insights") or {}
    summary = _first_nonempty(insights.get("combinedTechnicalSummary"), insights.get("summaryLine"), insights.get("oneLiner"))
    if summary:
        ui["executiveSummaryShort"] = condense_executive_summary(summary)

    if _is_nonempty_str(insights.get("whySignal")):
        ui["signalOneLiner"] = insights["whySignal"]

    rop = build_risks_opportunities(stockdetail)
    if rop:
        ui["risksOpportunities"] = rop

    trade = build_trade_idea(stockdetail)
    if trade:
        ui["tradeIdea"] = trade

    # ---- Narrative explanations groups (UI block) ----
    narratives = stockdetail.get("narratives") or {}
    sections = narratives.get("sections") or {}
    if sections:
        # ensure final recommendation block exists
        ui["explanations"] = {
            "version": "tech_explain_v2",
            "groups": {
                "technical_outlook": {
                    "short": narratives.get("summary"),
                    "medium": narratives.get("tradeIdea"),
                    "long": _dedupe_keep_order(
                        (sections.get("trend", []) or []) +
                        (sections.get("momentum", []) or []) +
                        (sections.get("volatility", []) or []) +
                        (sections.get("volume", []) or [])
                    )[:8],
                },
                "risks_opportunities": {
                    "risks": _dedupe_keep_order(sections.get("risk", []) or [])[:6],
                    "opportunities": _dedupe_keep_order(sections.get("opportunity", []) or [])[:6],
                },
                "final_recommendation": {
                    "signal": (stockdetail.get("decision") or {}).get("finalSignal"),
                    "confidence": (stockdetail.get("bullbrain") or {}).get("confidence"),
                    "text": narratives.get("summary"),
                },
            },
        }

    # ---- Decision Intelligence (probability bar) ----
    decision_ui = {}

    probs = stockdetail.get("probabilities") or {}
    indicator_states = stockdetail.get("indicator_states") or {}

    if isinstance(probs.get("up"), (int, float)) and isinstance(probs.get("down"), (int, float)):
        decision_ui["probability"] = {
            "up": round(float(probs["up"]), 4),
            "down": round(float(probs["down"]), 4),
        }
        ui["hybridProbUp"] = decision_ui["probability"]["up"]

        diff = abs(float(probs["up"]) - float(probs["down"]))
        decision_ui["bias"] = {
            "label": (
                "Neutral" if diff < 0.05
                else "Bullish" if probs["up"] > probs["down"]
                else "Bearish"
            ),
            "strength": round(diff * 100, 1),
            "state": indicator_states.get("probability_composite"),
        }

    if decision_ui:
        ui["decision"] = decision_ui

    return ui
