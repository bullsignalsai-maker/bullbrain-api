# backend/technical_explainer.py

from typing import Dict, Any, List


def explain_technical(stock: Dict[str, Any]) -> Dict[str, Any]:
    technical = stock.get("technical", {}) or {}
    features = stock.get("features_meta", {}) or {}

    # -----------------------------
    # High-level blocks (already humanized)
    # -----------------------------
    overview = {
        "trend": technical.get("trend"),
        "momentum": technical.get("rsi"),
        "volume": technical.get("volume"),
        "volatility": technical.get("volatility"),
        "pricePosition": technical.get("pricePosition"),
    }

    # -----------------------------
    # Indicator groups (UI sections)
    # -----------------------------
    trend_indicators = [
        {
            "name": "SMA (20)",
            "value": features.get("sma20"),
            "comment": f"Price is {features.get('price_vs_sma20_pct', 0):.1f}% vs 20-day average.",
            "bias": "bearish" if features.get("price_vs_sma20_pct", 0) < 0 else "bullish",
        },
        {
            "name": "Trend Strength",
            "value": features.get("trend_strength_20"),
            "comment": "Measures consistency of directional movement.",
            "bias": "neutral",
        },
    ]

    momentum_indicators = [
        {
            "name": "RSI (14)",
            "value": features.get("rsi14"),
            "comment": "Below 30 = oversold, above 70 = overbought.",
            "bias": (
                "bearish" if features.get("rsi14", 50) < 40
                else "bullish" if features.get("rsi14", 50) > 60
                else "neutral"
            ),
        },
        {
            "name": "MACD",
            "value": features.get("macd"),
            "comment": "Negative MACD suggests weakening momentum.",
            "bias": "bearish" if features.get("macd", 0) < 0 else "bullish",
        },
    ]

    volume_indicators = [
        {
            "name": "Volume vs MA20",
            "value": features.get("volume_vs_ma20_pct"),
            "comment": "Confirms or weakens price moves.",
            "bias": "bullish" if features.get("volume_vs_ma20_pct", 0) > 20 else "neutral",
        }
    ]

    volatility_indicators = [
        {
            "name": "ATR (14)",
            "value": features.get("atr14"),
            "comment": "Measures daily price range risk.",
            "bias": "high-risk" if features.get("atr14", 0) > 6 else "normal",
        }
    ]

    # -----------------------------
    # Evidence buckets (very useful for UI)
    # -----------------------------
    bullish = []
    bearish = []
    neutral = []

    def bucket(item):
        if item["bias"] == "bullish":
            bullish.append(item)
        elif item["bias"] == "bearish":
            bearish.append(item)
        else:
            neutral.append(item)

    for grp in (
        trend_indicators
        + momentum_indicators
        + volume_indicators
        + volatility_indicators
    ):
        bucket(grp)

    # -----------------------------
    # Summary (human-first)
    # -----------------------------
    summary = {
        "headline": "Technical signals are mixed.",
        "whatItMeans": (
            "Momentum is weakening while trend remains sideways. "
            "This favors patience or short-term tactical trades."
        ),
        "riskNote": "Sideways regimes increase whipsaw risk.",
    }

    return {
        "technicalOverview": overview,
        "indicatorGroups": {
            "trendIndicators": trend_indicators,
            "momentumIndicators": momentum_indicators,
            "volumeIndicators": volume_indicators,
            "volatilityIndicators": volatility_indicators,
        },
        "featureEvidence": {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
        },
        "summary": summary,
    }
