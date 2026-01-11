# backend/narratives/technical_explanations.py
# ------------------------------------------------------------
# Deterministic Technical Explainability Engine
#
# Purpose:
# - Generate human-readable narratives from technicals + features_meta
# - NO LLMs
# - NO Firestore writes
# - SAFE for /stockdetail
# ------------------------------------------------------------

from typing import Dict, Any, List


# ============================================================
# Helpers
# ============================================================

def _safe(v):
    return v is not None


def _fmt_pct(v, digits=1):
    try:
        return f"{float(v):.{digits}f}%"
    except Exception:
        return None


# ============================================================
# TECHNICAL OUTLOOK
# ============================================================

def build_technical_outlook(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    tech = stockdetail.get("technical", {})
    feats = stockdetail.get("features_meta", {})

    bullets: List[str] = []

    p_vs_sma20 = feats.get("price_vs_sma20_pct")
    rsi = feats.get("rsi14")
    macd_label = tech.get("macd", {}).get("label")
    trend_label = tech.get("trend", {}).get("label")

    if _safe(p_vs_sma20):
        bullets.append(f"Price is {_fmt_pct(abs(p_vs_sma20))} "
                       f"{'above' if p_vs_sma20 > 0 else 'below'} the 20-day average.")

    if _safe(rsi):
        bullets.append(
            "RSI indicates overbought momentum."
            if rsi >= 70 else
            "RSI indicates oversold momentum."
            if rsi <= 30 else
            "RSI is in a neutral momentum range."
        )

    if macd_label:
        bullets.append(f"MACD remains {macd_label.lower()}.")

    if trend_label:
        bullets.append(f"Trend regime is {trend_label.lower()}.")

    return {
        "short": "Momentum is strong but conditions are stretched."
        if rsi and rsi >= 70 else
        "Momentum is mixed with no strong directional edge.",

        "medium": (
            "Price remains supported above averages, but momentum indicators "
            "show overextension risk."
            if p_vs_sma20 and p_vs_sma20 > 0 and rsi and rsi >= 70 else
            "Technical indicators show balanced conditions without strong extremes."
        ),

        "long": [
            "The stock is trading relative to key moving averages, defining short-term trend bias.",
            "Momentum indicators such as RSI and MACD help assess exhaustion versus continuation risk.",
            "Trend structure determines whether momentum signals favor continuation or consolidation.",
            "Overall conditions favor disciplined positioning rather than aggressive chasing."
        ],

        "bullets": bullets,
    }


# ============================================================
# RISKS & OPPORTUNITIES
# ============================================================

def build_risks_opportunities(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    tech = stockdetail.get("technical", {})
    feats = stockdetail.get("features_meta", {})

    risks: List[str] = []
    opportunities: List[str] = []

    # ---- RSI ----
    rsi = feats.get("rsi14")
    if _safe(rsi):
        if rsi >= 75:
            risks.append(
                "RSI is deeply overbought, increasing the probability of short-term pullbacks."
            )
        elif rsi <= 25:
            opportunities.append(
                "RSI is deeply oversold, increasing the probability of a rebound."
            )

    # ---- Trend regime ----
    trend_label = tech.get("trend", {}).get("label")
    if trend_label == "Sideways":
        risks.append(
            "Sideways trend regimes increase whipsaw risk and reduce directional conviction."
        )
    elif trend_label == "Uptrend":
        opportunities.append(
            "An established uptrend supports trend-following and continuation strategies."
        )
    elif trend_label == "Downtrend":
        risks.append(
            "A downtrend structure increases downside continuation risk."
        )

    # ---- Price extension ----
    p_vs_sma20 = feats.get("price_vs_sma20_pct")
    if _safe(p_vs_sma20):
        if p_vs_sma20 > 6:
            risks.append(
                "Price is extended above short-term averages, reducing margin of safety."
            )
        elif p_vs_sma20 > 0:
            opportunities.append(
                "Price holding above the 20-day average supports bullish structure."
            )

    # ---- MACD ----
    macd_label = tech.get("macd", {}).get("label")
    if macd_label == "Bullish":
        opportunities.append(
            "MACD remains bullish, confirming underlying trend momentum."
        )
    elif macd_label == "Bearish":
        risks.append(
            "MACD is bearish, signaling weakening momentum."
        )

    return {
        "short": "Momentum strength is balanced by extension risk.",
        "medium": (
            "Overbought conditions raise pullback risk, while trend structure "
            "still allows for upside continuation."
        ),
        "risks": risks,
        "opportunities": opportunities,
    }


# ============================================================
# TRADE IDEA (NON-LLM, GUIDANCE ONLY)
# ============================================================

def build_trade_idea(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    tech = stockdetail.get("technical", {})
    feats = stockdetail.get("features_meta", {})

    trend = tech.get("trend", {}).get("label")
    rsi = feats.get("rsi14")
    p_vs_sma20 = feats.get("price_vs_sma20_pct")

    if trend == "Uptrend" and rsi and rsi < 70:
        stance = "Bullish continuation"
        idea = (
            "Trend-following setups may be favored, with entries on pullbacks "
            "toward short-term moving averages."
        )
    elif trend == "Sideways":
        stance = "Range-bound"
        idea = (
            "Mean-reversion or range-trading strategies may be more effective "
            "until a clear breakout occurs."
        )
    elif trend == "Downtrend":
        stance = "Defensive / bearish"
        idea = (
            "Rallies into resistance may present better risk–reward than chasing downside moves."
        )
    else:
        stance = "Neutral"
        idea = "No clear technical edge is present."

    return {
        "stance": stance,
        "summary": idea,
        "note": "This is not financial advice. Position sizing and risk management remain essential.",
    }


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

def build_final_recommendation(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    bullbrain = stockdetail.get("bullbrain", {})
    tech = stockdetail.get("technical", {})

    signal = bullbrain.get("signal")
    confidence = bullbrain.get("confidence")
    trend = tech.get("trend", {}).get("label")

    if signal == "BUY":
        rec = (
            "Technical structure and AI signals lean bullish, but entries should "
            "respect momentum exhaustion and volatility."
        )
    elif signal == "SELL":
        rec = (
            "Bearish technicals and AI signals suggest caution, defensive positioning, "
            "or active risk management."
        )
    else:
        rec = (
            "Signals are mixed, suggesting patience and selective positioning "
            "rather than aggressive exposure."
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "trend": trend,
        "text": rec,
    }


# ============================================================
# FEATURE-LEVEL EXPLANATIONS (by_feature)
# ============================================================

def build_feature_explanations(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    feats = stockdetail.get("features_meta", {})

    out: Dict[str, Any] = {}

    # RSI
    rsi = feats.get("rsi14")
    if _safe(rsi):
        out["rsi14"] = {
            "value": round(rsi, 2),
            "label": "Overbought" if rsi >= 70 else "Oversold" if rsi <= 30 else "Neutral",
            "bias": "bearish" if rsi >= 70 else "bullish" if rsi <= 30 else "neutral",
            "severity": 4 if rsi >= 75 or rsi <= 25 else 2,
            "short": f"RSI is {'overbought' if rsi >= 70 else 'oversold' if rsi <= 30 else 'neutral'}.",
            "medium": (
                "RSI above 70 indicates stretched momentum and rising pullback risk."
                if rsi >= 70 else
                "RSI below 30 indicates selling pressure may be exhausted."
                if rsi <= 30 else
                "RSI indicates balanced momentum."
            ),
            "long": [
                "RSI measures the speed and persistence of recent price movements.",
                "Extreme values increase the probability of consolidation or reversal.",
                "RSI should be interpreted alongside trend and volume."
            ],
            "why_it_matters": "RSI helps assess momentum extremes and short-term risk."
        }

    return out


# ============================================================
# MASTER ENTRY POINT
# ============================================================

def build_technical_explanations(stockdetail: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": "tech_explain_v1",

        "groups": {
            "technical_outlook": build_technical_outlook(stockdetail),
            "risks_opportunities": build_risks_opportunities(stockdetail),
            "trade_idea": build_trade_idea(stockdetail),
            "final_recommendation": build_final_recommendation(stockdetail),
        },

        "by_feature": build_feature_explanations(stockdetail),
    }
