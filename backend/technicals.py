# backend/technicals.py
# ------------------------------------------------------------
# Technical Indicators & Snapshot Builder
# ------------------------------------------------------------

from typing import Dict, Any, Optional
import math


# ------------------------------------------------------------
# Interpretation helpers
# ------------------------------------------------------------

def _interpret_rsi(rsi: Optional[float]) -> Dict[str, Any]:
    if rsi is None or math.isnan(rsi):
        return {"label": "Unknown", "comment": ""}

    if rsi >= 70:
        return {
            "label": "Overbought",
            "comment": "Momentum is strong but the stock may be stretched to the upside.",
        }
    elif rsi <= 30:
        return {
            "label": "Oversold",
            "comment": "Selling pressure looks exhausted and a bounce is possible.",
        }
    elif rsi >= 55:
        return {
            "label": "Bullish",
            "comment": "Momentum favors buyers with healthy upside pressure.",
        }
    elif rsi <= 45:
        return {
            "label": "Bearish",
            "comment": "Momentum favors sellers and downside risk remains.",
        }

    return {
        "label": "Neutral",
        "comment": "Momentum is balanced with no strong directional bias.",
    }


def _interpret_macd(macd: Optional[float], signal: Optional[float]) -> Dict[str, Any]:
    if macd is None or signal is None:
        return {"label": "Unknown", "comment": ""}

    if macd > signal:
        return {
            "label": "Bullish",
            "comment": "Trend momentum is positive and buyers remain in control.",
        }
    elif macd < signal:
        return {
            "label": "Bearish",
            "comment": "Momentum is weakening and sellers may dominate.",
        }

    return {
        "label": "Neutral",
        "comment": "Momentum is flat with no clear trend direction.",
    }


def _interpret_volume(vol_vs_ma20: Optional[float]) -> Dict[str, Any]:
    if vol_vs_ma20 is None or math.isnan(vol_vs_ma20):
        return {"label": "Unknown", "comment": ""}

    if vol_vs_ma20 > 20:
        return {
            "label": "High",
            "comment": "Trading activity is well above average, showing strong interest.",
        }
    elif vol_vs_ma20 < -20:
        return {
            "label": "Low",
            "comment": "Volume is light, so price moves may lack conviction.",
        }

    return {
        "label": "Normal",
        "comment": "Volume is in line with recent trading activity.",
    }


def _interpret_trend(trend_strength: Optional[float]) -> Dict[str, Any]:
    if trend_strength is None or math.isnan(trend_strength):
        return {"label": "Unknown", "comment": ""}

    if trend_strength > 15:
        return {
            "label": "Strong Uptrend",
            "comment": "Price is rising consistently with strong trend momentum.",
        }
    elif trend_strength > 5:
        return {
            "label": "Uptrend",
            "comment": "The stock is trending higher with moderate strength.",
        }
    elif trend_strength < -15:
        return {
            "label": "Strong Downtrend",
            "comment": "Price is falling sharply with strong selling pressure.",
        }
    elif trend_strength < -5:
        return {
            "label": "Downtrend",
            "comment": "The stock is trending lower with persistent weakness.",
        }

    return {
        "label": "Sideways",
        "comment": "Price action is range-bound without a clear trend.",
    }


def _interpret_volatility(volatility_20d: Optional[float]) -> Dict[str, Any]:
    if volatility_20d is None or math.isnan(volatility_20d):
        return {"label": "Unknown", "comment": ""}

    if volatility_20d > 4:
        return {
            "label": "High",
            "comment": "Large price swings indicate elevated risk.",
        }
    elif volatility_20d < 1.5:
        return {
            "label": "Low",
            "comment": "Price movement is calm and stable.",
        }

    return {
        "label": "Moderate",
        "comment": "Volatility is within a normal range.",
    }


# ------------------------------------------------------------
# Public API — Technical Snapshot
# ------------------------------------------------------------

def build_technical_snapshot(
    symbol: str,
    features: Dict[str, Any],
    last_close: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a compact, explainable technical snapshot for UI & AI.

    Input:
      - symbol
      - BullBrain feature_dict (48 features)
      - last_close (optional)

    Output:
      - Human-readable technical analysis
    """

    rsi = features.get("rsi14")
    macd = features.get("macd")
    macd_signal = features.get("macd_signal")
    vol_vs_ma20 = features.get("volume_vs_ma20_pct")
    trend_strength = features.get("trend_strength_20")
    volatility_20d = features.get("volatility_20d")
    price_vs_sma20 = features.get("price_vs_sma20_pct")

    rsi_block = _interpret_rsi(rsi)
    macd_block = _interpret_macd(macd, macd_signal)
    volume_block = _interpret_volume(vol_vs_ma20)
    trend_block = _interpret_trend(trend_strength)
    volatility_block = _interpret_volatility(volatility_20d)

    summary_parts = []

    if rsi_block["label"] != "Unknown":
        summary_parts.append(f"RSI is {rsi_block['label'].lower()}.")

    if trend_block["label"] != "Unknown":
        summary_parts.append(f"Trend is {trend_block['label'].lower()}.")

    if volume_block["label"] == "High":
        summary_parts.append("Volume confirms the move.")

    if volatility_block["label"] == "High":
        summary_parts.append("Risk is elevated due to volatility.")

    summary = " ".join(summary_parts)

    return {
        "symbol": symbol.upper(),
        "lastClose": last_close,
        "rsi": {
            "value": rsi,
            **rsi_block,
        },
        "macd": {
            "value": macd,
            "signal": macd_signal,
            **macd_block,
        },
        "volume": {
            "volume_vs_ma20_pct": vol_vs_ma20,
            **volume_block,
        },
        "trend": {
            "trend_strength_20": trend_strength,
            **trend_block,
        },
        "volatility": {
            "volatility_20d": volatility_20d,
            **volatility_block,
        },
        "pricePosition": {
            "price_vs_sma20_pct": price_vs_sma20,
            "label": (
                "Above Trend"
                if price_vs_sma20 is not None and price_vs_sma20 > 0
                else "Below Trend"
            ),
        },
        "summary": summary,
    }
