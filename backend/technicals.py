# backend/technicals.py

import numpy as np
import pandas as pd
from typing import Dict, Any


# ------------------------------------------------------------
# Core Technical Snapshot (UI + AI Friendly)
# ------------------------------------------------------------

def build_technical_snapshot(
    symbol: str,
    feature_dict: Dict[str, float],
    last_close: float,
) -> Dict[str, Any]:
    """
    Builds a clean, UI-ready technical snapshot from feature dict.
    """

    rsi = feature_dict.get("rsi14")
    macd = feature_dict.get("macd")
    macd_signal = feature_dict.get("macd_signal")
    macd_hist = feature_dict.get("macd_hist")

    sma20 = feature_dict.get("sma20")
    sma50 = feature_dict.get("sma50")
    sma200 = feature_dict.get("sma200")

    price_vs_sma20 = feature_dict.get("price_vs_sma20_pct")

    trend_strength = feature_dict.get("trend_strength_20")
    volatility = feature_dict.get("volatility_20d")

    # ----------------------------
    # RSI interpretation
    # ----------------------------
    if rsi is None:
        rsi_signal = None
    elif rsi >= 70:
        rsi_signal = "Overbought"
    elif rsi <= 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"

    # ----------------------------
    # MACD interpretation
    # ----------------------------
    if macd is None or macd_signal is None:
        macd_trend = None
    elif macd > macd_signal:
        macd_trend = "Bullish"
    else:
        macd_trend = "Bearish"

    # ----------------------------
    # Trend interpretation
    # ----------------------------
    if trend_strength is None:
        trend_label = None
    elif trend_strength > 0:
        trend_label = "Uptrend"
    else:
        trend_label = "Downtrend"

    # ----------------------------
    # Volatility bucket
    # ----------------------------
    if volatility is None:
        vol_bucket = None
    elif volatility < 1.2:
        vol_bucket = "Low"
    elif volatility < 2.5:
        vol_bucket = "Moderate"
    else:
        vol_bucket = "High"

    # ----------------------------
    # Moving-average structure
    # ----------------------------
    ma_structure = None
    if sma20 and sma50 and sma200:
        if sma20 > sma50 > sma200:
            ma_structure = "Strong Bullish"
        elif sma20 < sma50 < sma200:
            ma_structure = "Strong Bearish"
        else:
            ma_structure = "Mixed"

    # ----------------------------
    # Assemble snapshot
    # ----------------------------
    return {
        "symbol": symbol,
        "price": round(last_close, 2) if last_close else None,

        "rsi": {
            "value": round(rsi, 2) if rsi is not None else None,
            "signal": rsi_signal,
        },

        "macd": {
            "macd": round(macd, 4) if macd is not None else None,
            "signal": round(macd_signal, 4) if macd_signal is not None else None,
            "histogram": round(macd_hist, 4) if macd_hist is not None else None,
            "trend": macd_trend,
        },

        "movingAverages": {
            "sma20": round(sma20, 2) if sma20 else None,
            "sma50": round(sma50, 2) if sma50 else None,
            "sma200": round(sma200, 2) if sma200 else None,
            "structure": ma_structure,
            "priceVsSMA20Pct": round(price_vs_sma20, 2)
            if price_vs_sma20 is not None
            else None,
        },

        "trend": {
            "direction": trend_label,
            "strength": round(trend_strength, 4)
            if trend_strength is not None
            else None,
        },

        "volatility": {
            "bucket": vol_bucket,
            "value": round(volatility, 2)
            if volatility is not None
            else None,
        },
    }
