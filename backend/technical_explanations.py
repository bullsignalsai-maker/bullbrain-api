# backend/narratives/technical_explanations.py
# -------------------------------------------------
# Deep Technical Explanations (Deterministic)
#
# Option A:
# - Every metric: short + medium
# - Important metrics: long[]
# - Grouped narratives for UI sections
# -------------------------------------------------

from typing import Dict, Any, List
import math


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _fmt(v, d=2):
    try:
        return round(float(v), d)
    except Exception:
        return None


def _bias_from_label(label: str | None):
    if not label:
        return "neutral"
    l = label.lower()
    if "bull" in l or "up" in l:
        return "bullish"
    if "bear" in l or "down" in l or "overbought" in l:
        return "bearish"
    return "neutral"


# -------------------------------------------------
# Feature-level explanations
# -------------------------------------------------

def explain_rsi(rsi: float | None) -> Dict[str, Any]:
    if rsi is None:
        return {}

    label = (
        "Overbought" if rsi >= 70 else
        "Oversold" if rsi <= 30 else
        "Neutral"
    )

    return {
        "value": _fmt(rsi),
        "label": label,
        "bias": _bias_from_label(label),
        "severity": 4 if label == "Overbought" else 3 if label == "Oversold" else 1,
        "short": f"RSI is {label.lower()}.",
        "medium": (
            "RSI above 70 indicates stretched momentum and rising pullback risk."
            if label == "Overbought"
            else "RSI below 30 suggests selling pressure may be exhausted."
            if label == "Oversold"
            else "RSI is balanced with no extreme momentum."
        ),
        "long": [
            "RSI (Relative Strength Index) measures the speed and persistence of recent price movements.",
            "Values above 70 typically indicate aggressive buying pressure and stretched momentum.",
            "Overbought conditions do not guarantee an immediate reversal but increase the probability of consolidation or pullbacks.",
            "Sustained trends can remain overbought for extended periods, so RSI should be interpreted alongside trend and volume."
        ],
        "why_it_matters": "RSI helps assess momentum extremes and short-term risk conditions."
    }


def explain_macd(macd: float | None, signal: float | None, hist: float | None) -> Dict[str, Any]:
    if macd is None or signal is None:
        return {}

    label = "Bullish" if macd > signal else "Bearish" if macd < signal else "Neutral"

    return {
        "value": _fmt(macd),
        "signal": _fmt(signal),
        "histogram": _fmt(hist),
        "label": label,
        "bias": _bias_from_label(label),
        "short": f"MACD is {label.lower()}.",
        "medium": (
            "MACD above its signal line confirms positive trend momentum."
            if label == "Bullish"
            else "MACD below its signal line suggests weakening momentum."
            if label == "Bearish"
            else "MACD is flat with no clear momentum signal."
        ),
        "long": [
            "MACD compares short-term and long-term exponential moving averages to gauge trend momentum.",
            "When MACD is above its signal line, bullish momentum is dominant.",
            "The histogram reflects the strength and acceleration of momentum.",
            "Divergences between MACD and price can signal early trend shifts."
        ],
        "why_it_matters": "MACD helps confirm trend direction and momentum strength."
    }


def explain_price_vs_sma(pct: float | None) -> Dict[str, Any]:
    if pct is None:
        return {}

    label = "Above Trend" if pct > 0 else "Below Trend"

    return {
        "value": _fmt(pct),
        "label": label,
        "bias": "bullish" if pct > 0 else "bearish",
        "short": f"Price is {abs(_fmt(pct))}% {'above' if pct > 0 else 'below'} its 20-day average.",
        "medium": (
            "Trading above the 20-day average supports short-term trend strength."
            if pct > 0 else
            "Trading below the 20-day average suggests short-term weakness."
        ),
        "long": [
            "The 20-day moving average reflects short-term trend direction.",
            "Prices above this level indicate buyers are maintaining control.",
            "Large deviations can signal trend strength or overextension."
        ],
        "why_it_matters": "Relative position to key averages defines trend bias."
    }


def explain_volume(vol_vs_ma20: float | None, z: float | None) -> Dict[str, Any]:
    if vol_vs_ma20 is None:
        return {}

    label = (
        "High" if vol_vs_ma20 > 20 else
        "Low" if vol_vs_ma20 < -20 else
        "Normal"
    )

    return {
        "value": _fmt(vol_vs_ma20),
        "label": label,
        "bias": "bullish" if label == "High" else "neutral",
        "short": f"Volume is {label.lower()} relative to average.",
        "medium": (
            "Elevated volume confirms stronger participation."
            if label == "High"
            else "Light volume suggests weaker conviction."
            if label == "Low"
            else "Volume is in line with recent activity."
        ),
        "long": [
            "Volume measures participation and conviction behind price moves.",
            "Moves supported by strong volume tend to be more durable.",
            "Low volume increases the risk of false breakouts or reversals."
        ],
        "why_it_matters": "Volume confirms whether price moves are supported."
    }


def explain_volatility(vol20: float | None, atr: float | None) -> Dict[str, Any]:
    if vol20 is None:
        return {}

    label = (
        "High" if vol20 > 3 else
        "Low" if vol20 < 1.5 else
        "Moderate"
    )

    return {
        "value": _fmt(vol20),
        "atr": _fmt(atr),
        "label": label,
        "bias": "bearish" if label == "High" else "neutral",
        "short": f"Volatility is {label.lower()}.",
        "medium": (
            "High volatility increases risk and position sizing importance."
            if label == "High"
            else "Low volatility reflects stable price behavior."
            if label == "Low"
            else "Volatility is within a normal range."
        ),
        "long": [
            "Volatility measures the magnitude of price fluctuations.",
            "Higher volatility increases both opportunity and risk.",
            "ATR helps quantify expected price movement ranges."
        ],
        "why_it_matters": "Volatility defines risk and trade sizing."
    }


# -------------------------------------------------
# Grouped narratives (UI-ready)
# -------------------------------------------------

def build_group_technical_outlook(features: Dict[str, Any], technical: Dict[str, Any]) -> Dict[str, Any]:
    rsi = features.get("rsi14")
    macd = features.get("macd")
    macd_sig = features.get("macd_signal")
    trend_label = (technical.get("trend") or {}).get("label")
    price_vs = features.get("price_vs_sma20_pct")

    bullets = []

    if price_vs is not None:
        bullets.append(
            f"Price is {abs(_fmt(price_vs))}% {'above' if price_vs > 0 else 'below'} the 20-day average."
        )

    if rsi is not None:
        bullets.append(
            "RSI indicates overbought momentum." if rsi >= 70 else
            "RSI indicates oversold momentum." if rsi <= 30 else
            "RSI shows balanced momentum."
        )

    if macd is not None and macd_sig is not None:
        bullets.append(
            "MACD remains bullish." if macd > macd_sig else
            "MACD momentum is weakening."
        )

    if trend_label:
        bullets.append(f"Trend regime is {trend_label.lower()}.")

    return {
        "short": "Momentum is strong but conditions are stretched.",
        "medium": "Price remains supported above averages, but momentum indicators show overextension risk.",
        "long": [
            "The stock is trading above key short-term averages, supporting a constructive trend bias.",
            "RSI readings suggest momentum is stretched, increasing the probability of near-term consolidation.",
            "MACD remains positive, helping offset some overbought risk.",
            "Overall conditions favor caution rather than aggressive positioning."
        ],
        "bullets": bullets
    }


# -------------------------------------------------
# Master builder (entry point)
# -------------------------------------------------

def build_technical_explanations(
    symbol: str,
    features_meta: Dict[str, Any],
    technical: Dict[str, Any],
) -> Dict[str, Any]:

    explanations: Dict[str, Any] = {
        "version": "tech_explain_v1",
        "groups": {},
        "by_feature": {}
    }

    # ---- Feature explanations ----
    explanations["by_feature"]["rsi14"] = explain_rsi(features_meta.get("rsi14"))
    explanations["by_feature"]["macd"] = explain_macd(
        features_meta.get("macd"),
        features_meta.get("macd_signal"),
        features_meta.get("macd_hist"),
    )
    explanations["by_feature"]["price_vs_sma20_pct"] = explain_price_vs_sma(
        features_meta.get("price_vs_sma20_pct")
    )
    explanations["by_feature"]["volume"] = explain_volume(
        features_meta.get("volume_vs_ma20_pct"),
        features_meta.get("volume_zscore_20"),
    )
    explanations["by_feature"]["volatility"] = explain_volatility(
        features_meta.get("volatility_20d"),
        features_meta.get("atr14"),
    )

    # ---- Group narratives ----
    explanations["groups"]["technical_outlook"] = build_group_technical_outlook(
        features_meta,
        technical,
    )

    return explanations
