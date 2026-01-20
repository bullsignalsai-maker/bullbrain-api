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

def _soften_if_hold(text: str, signal: str) -> str:
    if signal != "HOLD":
        return text
    if not text:
        return text
    return (
        text
        .replace("potential", "possible")
        .replace("may be", "could be")
        .replace("is", "appears")
    )

def generate_bull_insights(
    *,
    symbol: str,
    features: Dict[str, Any],
    bullbrain: Dict[str, Any],
    technical: Optional[Dict[str, Any]] = None,
    seed_key: Optional[str] = None,
    decision: Optional[Dict[str, Any]] = None,
    pattern: Optional[Dict[str, Any]] = None,
    pattern_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns a UI-friendly insights dict aligned with:
      - signal + confidence
      - decision reasons + quality
      - pattern + patternHistory forward stats
      - features_meta + technical summaries
    """

    technical = technical or {}
    decision = decision or {}
    reasons = decision.get("decisionReasons") or decision.get("reasons") or []
    quality = decision.get("quality") or {}

    # -----------------------------
    # Helpers
    # -----------------------------
    def _as_list(x):
        return x if isinstance(x, list) else []

    def _pct(v: Optional[float]) -> str:
        if v is None:
            return "--"
        try:
            return f"{float(v):+.2f}%"
        except Exception:
            return "--"

    def _conf_tier(conf: float) -> str:
        if conf >= 80:
            return "High"
        if conf >= 65:
            return "Medium"
        return "Low"

    def _reason_explain(code: str) -> str:
        # Decision ladder reason → human explanation
        if not isinstance(code, str):
            return ""
        if code.startswith("Liquidity="):
            val = code.split("=", 1)[1]
            return f"Liquidity looks {val.lower()} (thin volume / unreliable moves), so the system avoids aggressive calls."
        if code == "PatternQualityFailed":
            return "This pattern has weak historical edge (win rate / average return not strong enough), so signal is blocked."
        if code == "TimeframeMisalignment":
            return "Short-, mid-, and 10-day returns don’t agree yet, so the setup isn’t consistent across timeframes."
        if code == "VolumeGateFailed":
            return "Volume confirmation is missing, so the move may not have strong participation."
        if code == "WeakFeatureConsensus":
            return "Trend, momentum, and volume signals are mixed, so there isn’t enough agreement for a directional call."
        if code == "NoUpsidePressure":
            return "Indicators don’t show enough upside pressure to justify a BUY despite some positive signals."
        if code == "NoDownsidePressure":
            return "Indicators don’t show enough downside pressure to justify a SELL despite some negative signals."
        if code == "SignalTooFragile":
            return "Price action is unstable (wide swings / indecision), so risk is elevated and signal is blocked."
        if code == "MomentumExhausted":
            return "The move looks stretched (exhaustion risk), so the system avoids chasing."
        if code == "NegativeEV":
            return "Expected value is negative based on historical outcomes + risk penalties, so signal is blocked."
        if code == "SignalPatternConflict":
            return "Model direction conflicts with pattern bias, so the system avoids a contradictory trade."
        if code == "ALL_GATES_PASSED":
            return "All quality gates passed: the signal is considered actionable."
        return code

    def _pick_top_reasons(reason_list: List[Any], max_n: int = 2) -> List[str]:
        out = []
        for r in reason_list:
            txt = _reason_explain(str(r))
            if txt and txt not in out:
                out.append(txt)
            if len(out) >= max_n:
                break
        return out

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
    vol_z = _num(features.get("volume_zscore_20"))
    vol20 = _num(features.get("volatility_20d"))
    trend_strength_20 = _num(features.get("trend_strength_20"))
    intraday_range_pct = _num(features.get("intraday_range_pct"))

    signal = (
        (bullbrain.get("signal"))
        or ((bullbrain.get("bullbrain") or {}).get("signal"))
        or "HOLD"
    )
    signal = str(signal).upper()

    confidence = float(_num(bullbrain.get("confidence"), 0.0) or 0.0)
    conf_label = _conf_tier(confidence)

    prob_up = _num(bullbrain.get("prob_up"))
    prob_down = _num(bullbrain.get("prob_down"))

    # Deterministic seed
    seed_key = seed_key or f"{symbol}:{(technical.get('updated_at') or '')}"

    # -----------------------------
    # Pattern context (if present)
    # -----------------------------
    patt_name = None
    patt_bias = None
    if isinstance(pattern, dict):
        patt_name = pattern.get("pattern") or pattern.get("patternLabel")
        patt_bias = pattern.get("bias") or pattern.get("patternBias")

    patt_stats_line = ""
    if isinstance(pattern_history, dict):
        days5 = (pattern_history.get("forwardReturns") or {}).get("days5") or {}
        wr = days5.get("winRate")
        avg = days5.get("avg")
        cnt = days5.get("count")
        if wr is not None and avg is not None and cnt:
            patt_stats_line = f"Pattern edge (5D): winRate={wr:.0%}, avg={avg:+.2f}%, samples={cnt}"

    # -----------------------------
    # Trend / Momentum / Volume / Volatility (base)
    # -----------------------------
    trend_summary = ""
    if sma5 is not None and sma20 is not None:
        if sma5 > sma20:
            trend_summary = "Trend: short-term strength (price above key averages)."
        elif sma5 < sma20:
            trend_summary = "Trend: short-term weakness (price below key averages)."
        else:
            trend_summary = "Trend: neutral (averages aligned)."

    tech_trend = (technical.get("trend") or {}).get("summary")
    if isinstance(tech_trend, str) and tech_trend.strip():
        trend_summary = tech_trend.strip().rstrip(".") + "."

    momentum_summary = "Momentum: mixed."
    mh = macd_hist if macd_hist is not None else 0.0
    if rsi14 is not None:
        if rsi14 < 30:
            momentum_summary = "Momentum: oversold (bounce possible, needs confirmation)."
        elif rsi14 > 70:
            momentum_summary = "Momentum: overbought (pullback risk)."
        else:
            if macd is not None and macd_signal is not None:
                if macd > macd_signal and mh > 0:
                    momentum_summary = "Momentum: strengthening (positive MACD structure)."
                elif macd < macd_signal and mh < 0:
                    momentum_summary = "Momentum: weakening (negative MACD structure)."

    tech_mom = (technical.get("momentum") or {}).get("summary_rsi")
    if isinstance(tech_mom, str) and tech_mom.strip():
        momentum_summary = tech_mom.strip().rstrip(".") + "."

    volume_summary = "Volume: near normal."
    if vol_z is not None:
        if vol_z > 2:
            volume_summary = "Volume: strong spike (high participation)."
        elif vol_z > 1:
            volume_summary = "Volume: elevated (above average)."
        elif vol_z < -1:
            volume_summary = "Volume: unusually low (thin participation)."

    tech_vol = (technical.get("volume") or {}).get("summary")
    if isinstance(tech_vol, str) and tech_vol.strip():
        volume_summary = tech_vol.strip().rstrip(".") + "."

    volatility_summary = "Volatility: normal."
    if intraday_range_pct is not None:
        if intraday_range_pct > 4:
            volatility_summary = "Volatility: high (wide intraday swings)."
        elif intraday_range_pct > 2:
            volatility_summary = "Volatility: moderate."

    tech_vola = (technical.get("volatility") or {}).get("summary")
    if isinstance(tech_vola, str) and tech_vola.strip():
        volatility_summary = tech_vola.strip().rstrip(".") + "."

    # -----------------------------
    # Signal-aligned, confidence-aware OneLiner + Summary
    # -----------------------------
    # Key idea:
    # - BUY: “Actionable bullish bias” (with confidence + risk cue)
    # - SELL: “Defensive / downside bias”
    # - HOLD: “Blocked by X” OR “No edge yet”
    top_reason_lines = _pick_top_reasons(_as_list(reasons), max_n=2)

    if signal == "BUY":
        one_liner = f"Bullish setup ({conf_label} confidence) — conditions support a long bias with risk controls."
        summary_line = "BUY signal: trend/momentum/volume align well enough to be actionable."
    elif signal == "SELL":
        one_liner = f"Bearish setup ({conf_label} confidence) — downside risk remains, consider defensive positioning."
        summary_line = "SELL signal: weakness/pressure dominates with enough confirmation to act."
    else:
        # HOLD: be explicit why, if we have reasons
        if top_reason_lines:
            one_liner = "HOLD: setup is blocked by quality filters — waiting avoids low-edge trades."
            summary_line = " | ".join(top_reason_lines[:1])
        else:
            one_liner = "HOLD: no clean edge yet — wait for confirmation."
            summary_line = "Trend/momentum are not aligned enough for a directional call."

    # -----------------------------
    # Add pattern context into summary if available
    # -----------------------------
    if patt_name:
        patt_part = f"Pattern: {patt_name}"
        if patt_bias:
            patt_part += f" ({patt_bias})"
        if patt_stats_line:
            summary_line = f"{summary_line} • {patt_part} • {patt_stats_line}"
        else:
            summary_line = f"{summary_line} • {patt_part}"

    # -----------------------------
    # Combined technical summary (kept, but cleaner)
    # -----------------------------
    combined = " ".join(
        [s for s in [trend_summary, momentum_summary, volume_summary, volatility_summary] if s]
    ).strip()

    # -----------------------------
    # NEW: Structured explanations (still inside insights map)
    # -----------------------------
    explain = {
        "signal": signal,
        "confidence": round(confidence, 2),
        "confidenceLabel": conf_label,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "decisionReasons": [str(x) for x in _as_list(reasons)],
        "decisionWhy": top_reason_lines,          # human readable
        "quality": quality,                       # already in Firestore; just mirrored for UI
        "pattern": {
            "name": patt_name,
            "bias": patt_bias,
            "statsLine": patt_stats_line or None,
        },
    }

    return {
        # original fields (UI-safe)
        "oneLiner": one_liner,
        "summaryLine": summary_line,
        "trendSummary": trend_summary,
        "momentumSummary": momentum_summary,
        "volumeSummary": volume_summary,
        "volatilitySummary": volatility_summary,
        "combinedTechnicalSummary": combined,

        # new fields (optional UI)
        "explain": explain,
    }
