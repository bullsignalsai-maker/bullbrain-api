# backend/bull_insights.py
# ------------------------------------------------------
# BullBrain Insights Generator (Backend)
# - Port of services/generateBullInsights.js (concept parity)
# - Pure function: (features, bullbrain, technical) -> insights dict
# - Frontend becomes dumb render (insights come from API/Firestore)
# ------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, List
import math
import random
import hashlib


def _num(v: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
            return float(v)
    except Exception:
        pass
    return fallback


def _seeded_pick(options: List[str], seed_key: str) -> str:
    """
    Deterministic pick per (symbol + day) or any seed_key you pass.
    Prevents UI text from changing every refresh.
    """
    if not options:
        return ""
    h = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    # use first 8 hex chars as deterministic int
    idx = int(h[:8], 16) % len(options)
    return options[idx]


def generate_bull_insights(
    *,
    symbol: str,
    features: Dict[str, Any],
    bullbrain: Dict[str, Any],
    technical: Optional[Dict[str, Any]] = None,
    seed_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        oneLiner,
        summaryLine,
        trendSummary,
        momentumSummary,
        volumeSummary,
        volatilitySummary,
        combinedTechnicalSummary,
      }
    """

    technical = technical or {}

    # -----------------------------
    # Extract important numbers
    # -----------------------------
    close = _num(features.get("close"))
    sma5 = _num(features.get("sma5"))
    sma20 = _num(features.get("sma20"))
    rsi14 = _num(features.get("rsi14"))
    macd = _num(features.get("macd"))
    macd_signal = _num(features.get("macd_signal"))
    macd_hist = _num(features.get("macd_hist"))

    highlowrange_pct = _num(features.get("intraday_range_pct")) or _num(
        (technical.get("candle") or {}).get("intraday_range_pct")
    )

    vol_z = _num(features.get("volume_zscore_20"))
    vol20 = _num(features.get("volatility_20d"))
    trend_strength_20 = _num(features.get("trend_strength_20"))

    signal = (
        (bullbrain.get("signal"))
        or ((bullbrain.get("bullbrain") or {}).get("signal"))
        or "HOLD"
    )
    signal = str(signal).upper()

    # -----------------------------
    # One-liner library (kept)
    # -----------------------------
    ONE_LINERS = {
        "BUY": [
            "Buying strength is building with improving trend — may favor a bullish stance.",
            "Momentum looks constructive as buyers step in — upside continuation is possible.",
            "Price action leans positive, with buyers gradually taking control.",
            "Trend is tilting upward and buyers are active — conditions support a bullish bias.",
            "Bulls are gaining traction as price stabilizes and begins to push higher.",
        ],
        "SELL": [
            "Stock is losing strength as sellers press the trend downward.",
            "Downward pressure is building — caution is warranted for long positions.",
            "Bearish momentum is forming below key trend levels.",
            "Selling activity is elevated — risk of further downside remains.",
            "Weak structure and downward drift suggest protecting capital or trimming risk.",
        ],
        "HOLD": [
            "Stock is consolidating with no clear directional edge.",
            "Momentum is neutral and price is in a wait-and-see zone.",
            "Market appears indecisive with balanced buying and selling.",
            "Trend is calm and stable — no strong entry or exit signal yet.",
            "Price is in a consolidation phase — monitoring for a breakout makes sense.",
        ],
        "VOLATILITY": [
            "Large intraday swings — risk management becomes critical.",
            "Volatility is elevated — entries and exits should be handled carefully.",
            "Uncertain sharp moves suggest smaller position sizing.",
            "Volatile conditions — waiting for clearer structure may help.",
            "Fast, choppy price action — consider a more defensive approach.",
        ],
        "REVERSAL_UP": [
            "Price may be turning upward from a weaker phase — early opportunity for patient buyers.",
            "Momentum is starting to shift upward, hinting at a potential bullish reversal.",
            "Downtrend is losing strength as buyers begin to absorb selling.",
            "Recovery signals are emerging from recent lows.",
            "Pressure from the downside is easing, opening room for a bounce.",
        ],
        "REVERSAL_DOWN": [
            "Uptrend is losing steam — locking in profits or tightening stops can be wise.",
            "Momentum is cooling from elevated levels, hinting at a possible pullback.",
            "Bullish phase appears to be fading — short-term caution is reasonable.",
            "Signs of a potential top are emerging after a strong run.",
            "Recent strength is softening, suggesting a possible near-term correction.",
        ],
    }

    # Deterministic seed: symbol + (technical updated_at or computed_at) preferred
    seed_key = seed_key or f"{symbol}:{(technical.get('updated_at') or features.get('asOf') or '')}"

    # -----------------------------
    # 1-Liner selection logic
    # -----------------------------
    one_liner = ""

    is_volatile = (highlowrange_pct is not None) and (highlowrange_pct > 4)

    # macd_hist thresholds mirrored from JS idea
    mh = macd_hist if macd_hist is not None else 0.0

    if is_volatile:
        one_liner = _seeded_pick(ONE_LINERS["VOLATILITY"], seed_key)
    elif mh > 0.5 and signal == "BUY":
        one_liner = _seeded_pick(ONE_LINERS["REVERSAL_UP"], seed_key)
    elif mh < -0.5 and signal == "SELL":
        one_liner = _seeded_pick(ONE_LINERS["REVERSAL_DOWN"], seed_key)
    elif signal == "BUY":
        one_liner = _seeded_pick(ONE_LINERS["BUY"], seed_key)
    elif signal == "SELL":
        one_liner = _seeded_pick(ONE_LINERS["SELL"], seed_key)
    else:
        one_liner = _seeded_pick(ONE_LINERS["HOLD"], seed_key)

    # -----------------------------
    # Trend summary
    # -----------------------------
    trend_summary = ""
    if sma5 is not None and sma20 is not None:
        if sma5 > sma20:
            trend_summary = "Short-term trend leans bullish with prices holding above key averages."
        elif sma5 < sma20:
            trend_summary = "Short-term trend leans bearish with prices below the mid-term average."
        else:
            trend_summary = "Trend is neutral with aligned short- and mid-term averages."

    # /technical override (better quality if present)
    tech_trend = (technical.get("trend") or {}).get("summary")
    if isinstance(tech_trend, str) and tech_trend.strip():
        trend_summary = tech_trend.strip()

    # -----------------------------
    # Momentum summary
    # -----------------------------
    momentum_summary = "Momentum mixed with no dominant direction."
    if rsi14 is not None:
        if rsi14 < 30:
            momentum_summary = "Momentum oversold; potential rebound zone."
        elif rsi14 > 70:
            momentum_summary = "Momentum overbought; risk of pullback."
        else:
            # MACD relationship
            if macd is not None and macd_signal is not None and mh is not None:
                if macd > macd_signal and mh > 0:
                    momentum_summary = "Momentum strengthening via positive MACD crossover."
                elif macd < macd_signal and mh < 0:
                    momentum_summary = "Momentum weakening via negative MACD crossover."

    tech_mom = (technical.get("momentum") or {}).get("summary_rsi")
    if isinstance(tech_mom, str) and tech_mom.strip():
        momentum_summary = tech_mom.strip().rstrip(".") + "."

    # -----------------------------
    # Volume summary
    # -----------------------------
    volume_summary = "Volume sits near typical levels."
    if vol_z is not None:
        if vol_z > 2:
            volume_summary = "Strong volume spike confirms high participation."
        elif vol_z > 1:
            volume_summary = "Volume elevated above 20-day average."
        elif vol_z < -1:
            volume_summary = "Volume unusually low."

    tech_vol = (technical.get("volume") or {}).get("summary")
    if isinstance(tech_vol, str) and tech_vol.strip():
        volume_summary = tech_vol.strip()

    # -----------------------------
    # Volatility summary
    # -----------------------------
    volatility_summary = "Volatility stable within normal range."
    if highlowrange_pct is not None:
        if highlowrange_pct > 4:
            volatility_summary = "High intraday volatility with wide price swings."
        elif highlowrange_pct > 2:
            volatility_summary = "Moderate intraday volatility."

    tech_vola = (technical.get("volatility") or {}).get("summary")
    if isinstance(tech_vola, str) and tech_vola.strip():
        volatility_summary = tech_vola.strip()

    # -----------------------------
    # Option-C smart technical summary line
    # -----------------------------
    summary_line = "Trend and momentum balanced; watching for next directional move."

    if trend_strength_20 is not None and mh is not None:
        if trend_strength_20 > 0.3 and mh > 0:
            summary_line = "Uptrend gaining strength with improving momentum and supportive volume."
        elif trend_strength_20 < -0.3 and mh < 0:
            summary_line = "Downtrend firming with weakening momentum and elevated volatility risk."
        elif vol20 is not None and vol20 > 3:
            summary_line = "Market tone volatile; trend signals mixed — caution recommended."

    # Strong /technical override
    if isinstance(tech_trend, str) and tech_trend.strip():
        summary_line = tech_trend.strip()

    combined = "\n".join(
        [s for s in [trend_summary, momentum_summary, volume_summary, volatility_summary] if s]
    )

    return {
        "oneLiner": one_liner,
        "summaryLine": summary_line,
        "trendSummary": trend_summary,
        "momentumSummary": momentum_summary,
        "volumeSummary": volume_summary,
        "volatilitySummary": volatility_summary,
        "combinedTechnicalSummary": combined,
    }
