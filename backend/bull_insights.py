# backend/bull_insights.py
# ------------------------------------------------------
# BullBrain Insights Generator (Backend)
# ------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Optional, List
import math
import hashlib


# ======================================================
# Utilities
# ======================================================

def _num(v: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
            return float(v)
    except Exception:
        pass
    return fallback


def _seeded_pick(options: List[str], seed_key: str) -> str:
    if not options:
        return ""
    h = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(options)
    return options[idx]


def _conf_tier(conf: float) -> str:
    if conf >= 80:
        return "high"
    if conf >= 65:
        return "moderate"
    return "low"


def _humanize_reason(code: str) -> str:
    mapping = {
        "PatternQualityFailed": "the detected pattern lacks a strong historical edge",
        "WeakFeatureConsensus": "technical signals are not aligned strongly enough",
        "VolumeGateFailed": "volume confirmation is missing",
        "SignalTooFragile": "price action is unstable and prone to whipsaws",
        "MomentumExhausted": "the recent move appears stretched",
        "NegativeEV": "historical outcomes do not favor this setup",
        "Liquidity=POOR": "liquidity conditions reduce signal reliability",
    }
    return mapping.get(code, code.lower())


# ======================================================
# Narrative Builders
# ======================================================

def _build_market_state_sentence(
    *,
    sma5: Optional[float],
    sma20: Optional[float],
    rsi14: Optional[float],
    macd: Optional[float],
    macd_signal: Optional[float],
    vol_z: Optional[float],
    intraday_range_pct: Optional[float],
    seed_key: str,
) -> str:
    parts = []

    # Price / trend
    if sma5 is not None and sma20 is not None:
        if sma5 > sma20:
            parts.append("price is holding above its short-term average")
        elif sma5 < sma20:
            parts.append("price remains below its short-term average")

    # Momentum
    if rsi14 is not None:
        if rsi14 < 30:
            parts.append("momentum looks oversold and stretched")
        elif rsi14 > 70:
            parts.append("momentum appears overheated after a strong run")
        else:
            if macd is not None and macd_signal is not None:
                if macd < macd_signal:
                    parts.append("momentum continues to weaken")
                elif macd > macd_signal:
                    parts.append("momentum is gradually improving")

    # Volume
    if vol_z is not None:
        if vol_z > 2:
            parts.append("trading activity is unusually strong")
        elif vol_z < -1:
            parts.append("trading participation remains thin")

    # Volatility
    if intraday_range_pct is not None:
        if intraday_range_pct > 4:
            parts.append("price swings remain elevated")
        elif intraday_range_pct > 2:
            parts.append("intraday price movement is moderately volatile")

    if not parts:
        return "Market conditions remain mixed with no dominant technical driver."

    connectors = ["while", "and", "with", "as"]
    joiner = _seeded_pick(connectors, seed_key)
    return f"{parts[0]} {joiner} {', '.join(parts[1:])}."


def _build_signal_sentence(
    *,
    signal: str,
    confidence: float,
    reasons: List[str],
    seed_key: str,
) -> str:
    conf_label = _conf_tier(confidence)

    if signal == "BUY":
        base = "Conditions favor bullish continuation"
    elif signal == "SELL":
        base = "Downside risk remains dominant"
    else:
        base = "Clear directional edge is not yet established"

    if reasons:
        reason_text = _seeded_pick(reasons, seed_key)
        return f"{base}, but {reason_text}."

    return f"{base}, suggesting a {conf_label}-confidence environment."


def _build_pattern_sentence(
    *,
    patt_name: Optional[str],
    patt_bias: Optional[str],
    patt_stats_line: Optional[str],
) -> Optional[str]:
    if not patt_name:
        return None

    bias_part = f" with a {patt_bias} bias" if patt_bias else ""
    if patt_stats_line:
        return f"The current pattern ({patt_name}){bias_part}, showing {patt_stats_line.lower()}."
    return f"The current pattern ({patt_name}){bias_part} is influencing price behavior."


# ======================================================
# Main Generator
# ======================================================

def generate_bull_insights(
    *,
    symbol: str,
    features: Dict[str, Any],
    bullbrain: Dict[str, Any],
    technical: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
    pattern: Optional[Dict[str, Any]] = None,
    pattern_history: Optional[Dict[str, Any]] = None,
    seed_key: Optional[str] = None,
) -> Dict[str, Any]:

    technical = technical or {}
    decision = decision or {}
    pattern = pattern or {}
    pattern_history = pattern_history or {}

    signal = str(
        decision.get("finalSignal") or bullbrain.get("signal") or "HOLD"
    ).upper()

    confidence = float(_num(bullbrain.get("confidence"), 0.0) or 0.0)
    reasons_raw = decision.get("decisionReasons") or decision.get("reasons") or []
    reasons = [_humanize_reason(str(r)) for r in reasons_raw if r]

    seed_key = seed_key or f"{symbol}:{technical.get('updated_at') or ''}"

    # --- Extract features
    sma5 = _num(features.get("sma5"))
    sma20 = _num(features.get("sma20"))
    rsi14 = _num(features.get("rsi14"))
    macd = _num(features.get("macd"))
    macd_signal = _num(features.get("macd_signal"))
    vol_z = _num(features.get("volume_zscore_20"))
    intraday_range_pct = _num(features.get("intraday_range_pct"))

    # --- Pattern context
    patt_name = pattern.get("pattern") or pattern.get("patternLabel")
    patt_bias = pattern.get("bias") or pattern.get("patternBias")

    patt_stats_line = None
    days5 = (pattern_history.get("forwardReturns") or {}).get("days5") or {}
    if days5.get("winRate") is not None and days5.get("avg") is not None:
        patt_stats_line = (
            f"a {days5['winRate']:.0%} win rate and "
            f"{days5['avg']:+.2f}% average return over 5 days"
        )

    # ==================================================
    # Build Narratives
    # ==================================================

    market_sentence = _build_market_state_sentence(
        sma5=sma5,
        sma20=sma20,
        rsi14=rsi14,
        macd=macd,
        macd_signal=macd_signal,
        vol_z=vol_z,
        intraday_range_pct=intraday_range_pct,
        seed_key=seed_key,
    )

    signal_sentence = _build_signal_sentence(
        signal=signal,
        confidence=confidence,
        reasons=reasons,
        seed_key=seed_key,
    )

    pattern_sentence = _build_pattern_sentence(
        patt_name=patt_name,
        patt_bias=patt_bias,
        patt_stats_line=patt_stats_line,
    )

    # ==================================================
    # Final Assembly (UI-facing)
    # ==================================================

    one_liner = market_sentence

    summary_parts = [signal_sentence]
    if pattern_sentence:
        summary_parts.append(pattern_sentence)

    summary_line = " ".join(summary_parts)

    combined = " ".join(
        p for p in [market_sentence, signal_sentence, pattern_sentence] if p
    )

    return {
        "oneLiner": one_liner,
        "whySignal": signal_sentence,
        "summaryLine": summary_line,
        "trendSummary": None,
        "momentumSummary": None,
        "volumeSummary": None,
        "volatilitySummary": None,
        "combinedTechnicalSummary": combined,
    }
